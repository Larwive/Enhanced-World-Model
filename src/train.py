from collections.abc import Callable
from pathlib import Path
from typing import Any
import logging
from datetime import datetime

import numpy as np
import torch
from torch.nn import MSELoss
from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard.summary import hparams

from WorldModel import WorldModel, render_first_env

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler("train.log", mode="w"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class HyperSummaryWriter(SummaryWriter):
    """
    Add possiblity to store hyperparameters.
    """

    def add_hparams(self, hparam_dict: dict, metric_dict: dict) -> None:  # pyright: ignore[reportIncompatibleMethodOverride]
        torch._C._log_api_usage_once("tensorboard.logging.add_hparams")
        if type(hparam_dict) is not dict or type(metric_dict) is not dict:
            raise TypeError("hparam_dict and metric_dict should be dictionary.")
        exp, ssi, sei = hparams(hparam_dict, metric_dict)

        logdir = self._get_file_writer().get_logdir()

        with SummaryWriter(log_dir=logdir) as w_hp:
            assert w_hp.file_writer is not None
            w_hp.file_writer.add_summary(exp)
            w_hp.file_writer.add_summary(ssi)
            w_hp.file_writer.add_summary(sei)
            for k, v in metric_dict.items():
                w_hp.add_scalar(k, v)


def state_transform(state: np.ndarray, is_image_based: bool, device: torch.device) -> torch.Tensor:
    if is_image_based:
        # Transpose state from (H, W, C) to (C, H, W) for PyTorch
        if state.ndim == 3:
            state = state[None]
        state_transposed = np.transpose(state, (0, 3, 1, 2))
        state_tensor = torch.from_numpy(state_transposed).float().to(device)
        # Normalize image data to [0, 1]
        return state_tensor / 255.0
    else:
        if state.ndim == 1:
            state = state[None]
        return torch.from_numpy(state).float().to(device)


def step(
    model: WorldModel,
    state: np.ndarray,
    envs: Any,
    optimizer: torch.optim.Optimizer,
    policy_optimizer: torch.optim.Optimizer,
    device: torch.device,
    is_image_based: bool,
    action_space: Any,
    cumulated_reward: torch.Tensor,
    discounted_return: torch.Tensor,
    gamma: float,
    iter_num: int = 0,
    tensorboard_writer: HyperSummaryWriter | None = None,
    loss_instance: Callable | None = None,
) -> tuple[torch.Tensor, torch.Tensor, np.ndarray, torch.Tensor]:
    if loss_instance is None:
        loss_instance = MSELoss()  # Same as below.

    state_tensor = state_transform(state, is_image_based=is_image_based, device=device)

    output_dict = model(
        state_tensor,
        action_space=action_space,
        is_image_based=is_image_based,
        return_losses=True,
        last_reward=cumulated_reward,
    )
    total_loss = output_dict["total_loss"]

    actions = output_dict["action"].cpu().detach().numpy()

    new_state, step_reward, terminated, truncated, info = envs.step(actions)
    dones = torch.from_numpy(terminated | truncated).to(device, dtype=torch.bool)

    step_reward_t = torch.tensor(step_reward, dtype=torch.float32, device=device)

    cumulated_reward += step_reward_t
    # cumulated_reward[dones] = step_reward_t[dones]
    # cumulated_reward[~dones] += step_reward_t[~dones]

    reward_prediction_loss = None
    if model.reward_predictor is not None:
        predicted_reward = output_dict["predicted_reward"].squeeze(-1)
        reward_prediction_loss = loss_instance(predicted_reward, step_reward_t).mean()
        total_loss = total_loss + reward_prediction_loss

    vision_encoded = model.vision.encode(
        state_transform(new_state, is_image_based=is_image_based, device=device),
        is_image_based=is_image_based,
    )

    if is_image_based:
        vision_encoded = vision_encoded.mean(dim=(2, 3))

    memory_loss = torch.nn.functional.mse_loss(vision_encoded, output_dict["memory_prediction"])
    total_loss = total_loss + memory_loss
    # total_loss = total_loss - (output_dict["log_probs"].squeeze(-1) * step_reward_t).mean()

    optimizer.zero_grad(set_to_none=True)
    total_loss.backward()
    optimizer.step()

    # Recomputing the graph for a separate backward.
    h_t_det = output_dict["memory_hidden"].detach()
    z_e = model.vision.encode(state_tensor, is_image_based=is_image_based)
    if is_image_based:
        z_t = z_e.mean(dim=(2, 3))
    else:
        z_t = z_e
    z_t_det = z_t.detach()

    action_for_policy, log_probs_det, value_det, entropy_det = model.controller(z_t_det, h_t_det)
    # log_probs_det (B,1); value_det (B,1)
    log_probs = log_probs_det.view(-1)
    value = value_det.view(-1)
    entropy = entropy_det.view(-1)

    """discounted_return = gamma * discounted_return + step_reward_t
    discounted_return[dones] = 0.0
    advantage = (discounted_return - value).detach()"""

    td_target = step_reward_t + gamma * value.detach() * (~dones)  # shape (B,)
    advantage = td_target - value
    # Normalization across batch for stability.
    adv_mean = advantage.mean()
    adv_std = advantage.std(unbiased=False) + 1e-8
    advantage = (advantage - adv_mean) / adv_std

    policy_loss = -(log_probs * advantage).mean()
    value_loss = loss_instance(value, discounted_return)
    # print("memory_loss", memory_loss.detach().item())
    # print("policy_loss", policy_loss.detach().item())
    # print("total_loss", total_loss.detach().item())

    entropy_coeff = 0.01  # TODO: To be determined in [0.0, 0.1].
    entropy_loss = -entropy_coeff * entropy.mean()
    policy_loss = policy_loss + entropy_loss

    policy_optimizer.zero_grad(set_to_none=True)
    (policy_loss + value_loss).backward()
    policy_optimizer.step()

    if tensorboard_writer is not None:
        tensorboard_writer.add_scalar("train/loss", total_loss.mean().item(), iter_num)
        tensorboard_writer.add_scalar("train/policy_loss", policy_loss.item(), iter_num)
        tensorboard_writer.add_scalar("train/value_loss", value_loss.item(), iter_num)
        # Component-specific losses for training observability
        tensorboard_writer.add_scalar(
            "train/recon_loss", output_dict["recon_loss"].mean().item(), iter_num
        )
        tensorboard_writer.add_scalar(
            "train/vq_loss", output_dict["vq_loss"].mean().item(), iter_num
        )
        tensorboard_writer.add_scalar("train/memory_loss", memory_loss.item(), iter_num)
        tensorboard_writer.add_scalar("train/entropy_loss", entropy_loss.item(), iter_num)
        if reward_prediction_loss is not None:
            tensorboard_writer.add_scalar(
                "train/reward_loss", reward_prediction_loss.item(), iter_num
            )
        for name, param in model.named_parameters():
            if param.grad is not None:
                tensorboard_writer.add_scalar(
                    f"gradients/{name}", param.grad.norm().item(), iter_num
                )

    return cumulated_reward, total_loss.detach(), new_state, dones


# TODO: Remove render_env arg when rendering of the first env is not done through cv2 anymore.
def train(
    model: WorldModel,
    envs: Any,
    max_iter: int = 10000,
    device: torch.device = torch.device("cpu"),
    use_tensorboard: bool = True,
    learning_rate: float = 0.01,
    loss_func: Callable = MSELoss,
    save_path: Path = Path("./"),
    render_mode: str = "",
) -> None:
    world_params = list(model.vision.parameters()) + list(model.memory.parameters())
    if model.reward_predictor is not None:
        world_params += list(model.reward_predictor.parameters())
    policy_params = list(model.controller.parameters())

    optimizer = torch.optim.AdamW(world_params, lr=learning_rate)
    policy_optimizer = torch.optim.AdamW(policy_params, lr=learning_rate)
    loss_func = loss_func()  # Add potential args here.
    is_image_based = len(envs.single_observation_space.shape) == 3
    action_space = envs.single_action_space

    writer = HyperSummaryWriter() if use_tensorboard else None

    cumulated_reward = torch.zeros(
        envs.num_envs, device=device, dtype=torch.float32
    )  # , dtype=torch.get_default_dtype()

    best_loss = torch.inf
    last_save = model.iter_num
    nb_experiments = 0
    state, info = envs.reset()
    local_iter_num = torch.zeros(envs.num_envs)
    total_episode_loss = torch.zeros(envs.num_envs)

    # For the reinforce-style loss.
    discounted_return = torch.zeros(envs.num_envs, device=device)
    gamma = 0.99

    while nb_experiments < max_iter:
        cumulated_reward, loss, state, dones = step(
            model,
            state,
            envs,
            optimizer,
            policy_optimizer,
            device,
            is_image_based,
            action_space,
            cumulated_reward,
            discounted_return,
            gamma,
            iter_num=model.iter_num,
            tensorboard_writer=writer,
            loss_instance=loss_func,
        )

        total_episode_loss += loss

        if render_mode == "human":
            render_first_env(envs)

        model.iter_num += envs.num_envs
        local_iter_num += 1

        if loss < best_loss and model.iter_num > last_save + 5 * envs.num_envs:
            best_loss = loss
            last_save = model.iter_num
            model.save(
                Path(
                    f"{save_path}{envs.spec.id}_{datetime.now().isoformat(timespec='minutes')}.pt"
                ),
                envs.single_observation_space,
                envs.single_action_space,
            )

        # logger.info(f"Experiment {experiment_index}\nIteration {local_iter_num + 1}, Mean Loss: {total_episode_loss/(local_iter_num + 1):.4f}")
        cumulated_reward[dones] = 0
        finished_envs = torch.where(dones)[0]
        for env_id in finished_envs:
            print(
                f"Experiment {model.nb_experiments + 1}\nEnded at iteration {local_iter_num[env_id]}, Mean Loss: {total_episode_loss[env_id] / local_iter_num[env_id]:.4f}"
            )
            model.nb_experiments += 1
            local_iter_num[env_id] = 0
            total_episode_loss[env_id] = 0
            nb_experiments += 1
            model.reset_env_memory(env_id)
