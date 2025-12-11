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

    def add_hparams(self, hparam_dict, metric_dict):
        torch._C._log_api_usage_once("tensorboard.logging.add_hparams")
        if type(hparam_dict) is not dict or type(metric_dict) is not dict:
            raise TypeError("hparam_dict and metric_dict should be dictionary.")
        exp, ssi, sei = hparams(hparam_dict, metric_dict)

        logdir = self._get_file_writer().get_logdir()

        with SummaryWriter(log_dir=logdir) as w_hp:
            w_hp.file_writer.add_summary(exp)
            w_hp.file_writer.add_summary(ssi)
            w_hp.file_writer.add_summary(sei)
            for k, v in metric_dict.items():
                w_hp.add_scalar(k, v)


class RolloutBuffer:
    """
    Buffer to store rollout experiences for PPO-style training.
    """

    def __init__(self, buffer_size: int, num_envs: int, device: torch.device):
        self.buffer_size = buffer_size
        self.num_envs = num_envs
        self.device = device
        self.reset()

    def reset(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []
        self.z_ts = []  # Latent states
        self.h_ts = []  # Hidden states
        self.ptr = 0

    def add(
        self,
        state: np.ndarray,
        action: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        log_prob: torch.Tensor,
        value: torch.Tensor,
        z_t: torch.Tensor,
        h_t: torch.Tensor,
    ):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.z_ts.append(z_t)
        self.h_ts.append(h_t)
        self.ptr += 1

    def is_full(self) -> bool:
        return self.ptr >= self.buffer_size

    def compute_returns_and_advantages(
        self, last_value: torch.Tensor, gamma: float = 0.99, gae_lambda: float = 0.95
    ):
        """
        Compute GAE (Generalized Advantage Estimation) returns and advantages.
        """
        # Stack tensors: shape (buffer_size, num_envs)
        rewards = torch.stack(self.rewards)  # (T, num_envs)
        values = torch.stack(self.values)  # (T, num_envs)
        dones = torch.stack(self.dones)  # (T, num_envs)

        T = len(self.rewards)
        advantages = torch.zeros(T, self.num_envs, device=self.device)
        last_gae = torch.zeros(self.num_envs, device=self.device)

        # Compute GAE backwards
        for t in reversed(range(T)):
            if t == T - 1:
                next_value = last_value
            else:
                next_value = values[t + 1]

            next_non_terminal = ~dones[t]
            delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae

        returns = advantages + values
        return returns, advantages

    def get_batches(self, batch_size: int, returns: torch.Tensor, advantages: torch.Tensor):
        """
        Generate random mini-batches for training.
        """
        # Flatten everything: (T * num_envs,)
        total_size = self.ptr * self.num_envs
        indices = np.random.permutation(total_size)

        # Stack and flatten
        actions = torch.stack(self.actions).view(-1)  # (T * num_envs,)
        log_probs = torch.stack(self.log_probs).view(-1)  # (T * num_envs,)
        values = torch.stack(self.values).view(-1)  # (T * num_envs,)
        z_ts = torch.stack(self.z_ts).view(-1, self.z_ts[0].shape[-1])  # (T * num_envs, z_dim)
        h_ts = torch.stack(self.h_ts).view(-1, self.h_ts[0].shape[-1])  # (T * num_envs, h_dim)
        returns_flat = returns.view(-1)
        advantages_flat = advantages.view(-1)

        # Generate mini-batches
        for start in range(0, total_size, batch_size):
            end = min(start + batch_size, total_size)
            batch_indices = indices[start:end]

            yield {
                "actions": actions[batch_indices],
                "old_log_probs": log_probs[batch_indices],
                "old_values": values[batch_indices],
                "z_ts": z_ts[batch_indices],
                "h_ts": h_ts[batch_indices],
                "returns": returns_flat[batch_indices],
                "advantages": advantages_flat[batch_indices],
            }


def state_transform(state, is_image_based, device):
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


def collect_rollout_step(
    model: WorldModel,
    state: np.ndarray,
    envs,
    device: torch.device,
    is_image_based: bool,
    action_space,
    cumulated_reward: torch.Tensor,
):
    """
    Collect a single step of experience without gradient computation.
    Returns experience data for the rollout buffer.
    """
    state_tensor = state_transform(state, is_image_based=is_image_based, device=device)

    with torch.no_grad():
        output_dict = model(
            state_tensor,
            action_space=action_space,
            is_image_based=is_image_based,
            return_losses=True,
            last_reward=cumulated_reward,
        )

    actions = output_dict["action"]
    log_probs = output_dict["log_probs"].squeeze(-1)  # (num_envs,)
    values = output_dict["value"].squeeze(-1)  # (num_envs,)
    z_t = output_dict["memory_prediction"]  # Latent state
    h_t = output_dict["memory_hidden"]  # Hidden state

    # Step environment
    actions_np = actions.cpu().detach().numpy()
    new_state, step_reward, terminated, truncated, info = envs.step(actions_np)
    dones = torch.from_numpy(terminated | truncated).to(device, dtype=torch.bool)
    step_reward_t = torch.tensor(step_reward, dtype=torch.float32, device=device)

    return {
        "state": state,
        "action": actions,
        "reward": step_reward_t,
        "done": dones,
        "log_prob": log_probs,
        "value": values,
        "z_t": z_t,
        "h_t": h_t,
        "new_state": new_state,
        "output_dict": output_dict,
    }


def update_world_model(
    model: WorldModel,
    states: list,
    new_states: list,
    rewards: list,
    actions: list,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    is_image_based: bool,
    action_space,
    loss_func,
):
    """
    Update vision, memory, and reward predictor on collected rollout data.
    Recomputes forward passes with gradients enabled.
    """
    total_loss = torch.tensor(0.0, device=device, requires_grad=True)
    recon_losses = []
    vq_losses = []
    memory_losses = []
    reward_losses = []

    # Sample a subset of transitions for efficiency (don't train on all 128 steps)
    num_samples = min(32, len(states))
    indices = np.random.choice(len(states), num_samples, replace=False)

    import gymnasium as gym

    # Determine action encoding
    if isinstance(action_space, gym.spaces.Discrete):
        action_dim = action_space.n
        is_discrete = True
    else:
        action_dim = action_space.shape[0]
        is_discrete = False

    for idx in indices:
        state = states[idx]
        new_state = new_states[idx]
        reward = rewards[idx]
        action = actions[idx]

        state_tensor = state_transform(state, is_image_based=is_image_based, device=device)
        new_state_tensor = state_transform(new_state, is_image_based=is_image_based, device=device)

        # Convert action to proper format for memory model
        if is_discrete:
            action_encoded = torch.nn.functional.one_hot(
                action.long(), num_classes=action_dim
            ).float()
        else:
            action_encoded = action

        # Full forward pass with gradients
        recon, vq_loss = model.vision(state_tensor)
        z_e = model.vision.encode(state_tensor, is_image_based=is_image_based)

        if is_image_based:
            z_t = z_e.mean(dim=(2, 3))
        else:
            z_t = z_e

        # Vision reconstruction loss
        recon_loss = torch.nn.functional.mse_loss(recon, state_tensor)

        # Get memory prediction (with gradients)
        h_t = model.memory.update_memory(z_t, action_encoded)
        memory_pred = model.memory.predict_next(z_t, action_encoded, h_t)

        # Target: encoded next state (no gradients needed for target)
        with torch.no_grad():
            vision_encoded_next = model.vision.encode(
                new_state_tensor, is_image_based=is_image_based
            )
            if is_image_based:
                vision_encoded_next = vision_encoded_next.mean(dim=(2, 3))

        memory_loss = torch.nn.functional.mse_loss(memory_pred, vision_encoded_next)

        step_loss = recon_loss + vq_loss.mean() + memory_loss

        # Reward prediction loss
        if model.reward_predictor is not None:
            predicted_reward = model.reward_predictor(z_t, h_t, reward).squeeze(-1)
            reward_loss = loss_func(predicted_reward, reward).mean()
            step_loss = step_loss + reward_loss
            reward_losses.append(reward_loss.item())

        total_loss = total_loss + step_loss
        recon_losses.append(recon_loss.item())
        vq_losses.append(vq_loss.mean().item())
        memory_losses.append(memory_loss.item())

    # Average loss over sampled steps
    total_loss = total_loss / num_samples

    optimizer.zero_grad(set_to_none=True)
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.vision.parameters(), max_norm=1.0)
    torch.nn.utils.clip_grad_norm_(model.memory.parameters(), max_norm=1.0)
    if model.reward_predictor is not None:
        torch.nn.utils.clip_grad_norm_(model.reward_predictor.parameters(), max_norm=1.0)
    optimizer.step()

    return {
        "total_loss": total_loss.item(),
        "recon_loss": np.mean(recon_losses),
        "vq_loss": np.mean(vq_losses),
        "memory_loss": np.mean(memory_losses),
        "reward_loss": np.mean(reward_losses) if reward_losses else 0.0,
    }


def update_policy_ppo(
    model: WorldModel,
    rollout_buffer: RolloutBuffer,
    returns: torch.Tensor,
    advantages: torch.Tensor,
    policy_optimizer: torch.optim.Optimizer,
    num_epochs: int = 4,
    batch_size: int = 64,
    clip_epsilon: float = 0.2,
    entropy_coeff: float = 0.01,
    value_coeff: float = 0.5,
):
    """
    Update policy using PPO with clipped objective.
    """
    # Normalize advantages
    adv_mean = advantages.mean()
    adv_std = advantages.std() + 1e-8
    advantages = (advantages - adv_mean) / adv_std

    policy_losses = []
    value_losses = []
    entropy_losses = []
    clip_fractions = []

    for _ in range(num_epochs):
        for batch in rollout_buffer.get_batches(batch_size, returns, advantages):
            z_ts = batch["z_ts"]
            h_ts = batch["h_ts"]
            old_log_probs = batch["old_log_probs"]
            old_values = batch["old_values"]
            batch_returns = batch["returns"]
            batch_advantages = batch["advantages"]
            actions = batch["actions"]

            # Evaluate actions using the controller's evaluate_actions method
            new_log_probs, new_values, entropy = model.controller.evaluate_actions(
                z_ts, h_ts, actions
            )

            # PPO clipped objective
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * batch_advantages
            surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * batch_advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss (clipped)
            value_pred_clipped = old_values + torch.clamp(
                new_values - old_values, -clip_epsilon, clip_epsilon
            )
            value_loss1 = (new_values - batch_returns) ** 2
            value_loss2 = (value_pred_clipped - batch_returns) ** 2
            value_loss = 0.5 * torch.max(value_loss1, value_loss2).mean()

            # Entropy bonus
            entropy_loss = -entropy_coeff * entropy.mean()

            # Total loss
            loss = policy_loss + value_coeff * value_loss + entropy_loss

            policy_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.controller.parameters(), max_norm=0.5)
            policy_optimizer.step()

            # Track metrics
            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())
            entropy_losses.append(entropy_loss.item())

            # Clip fraction
            clip_fraction = ((ratio - 1).abs() > clip_epsilon).float().mean().item()
            clip_fractions.append(clip_fraction)

    return {
        "policy_loss": np.mean(policy_losses),
        "value_loss": np.mean(value_losses),
        "entropy_loss": np.mean(entropy_losses),
        "clip_fraction": np.mean(clip_fractions),
    }


def train(
    model: WorldModel,
    envs,
    max_iter=10000,
    device: torch.device = torch.device("cpu"),
    use_tensorboard: bool = True,
    learning_rate: float = 0.01,
    loss_func: callable = MSELoss,
    save_path="./",
    render_mode: str = "",
    # PPO hyperparameters
    rollout_steps: int = 128,
    num_epochs: int = 4,
    batch_size: int = 64,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_epsilon: float = 0.2,
    entropy_coeff: float = 0.01,
):
    """
    PPO-style training loop with experience buffer and multiple update epochs.
    """
    world_params = list(model.vision.parameters()) + list(model.memory.parameters())
    if model.reward_predictor is not None:
        world_params += list(model.reward_predictor.parameters())
    policy_params = list(model.controller.parameters())

    optimizer = torch.optim.AdamW(world_params, lr=learning_rate)
    policy_optimizer = torch.optim.AdamW(
        policy_params, lr=learning_rate * 3
    )  # Often higher LR for policy
    loss_func_instance = loss_func()

    is_image_based = len(envs.single_observation_space.shape) == 3
    action_space = envs.single_action_space

    writer = HyperSummaryWriter() if use_tensorboard else None

    # Initialize rollout buffer
    rollout_buffer = RolloutBuffer(rollout_steps, envs.num_envs, device)

    cumulated_reward = torch.zeros(envs.num_envs, device=device, dtype=torch.float32)
    episode_rewards = torch.zeros(envs.num_envs, device=device, dtype=torch.float32)
    episode_lengths = torch.zeros(envs.num_envs, device=device, dtype=torch.float32)

    best_avg_reward = -float("inf")
    nb_experiments = 0
    state, info = envs.reset()

    # Store states for world model update
    collected_states = []
    collected_new_states = []
    collected_actions = []
    collected_rewards = []

    logger.info(f"Starting PPO training with {envs.num_envs} parallel environments")
    logger.info(f"Rollout steps: {rollout_steps}, Epochs: {num_epochs}, Batch size: {batch_size}")

    while nb_experiments < max_iter:
        # === COLLECT ROLLOUT ===
        rollout_buffer.reset()
        collected_states.clear()
        collected_new_states.clear()
        collected_actions.clear()
        collected_rewards.clear()

        for _ in range(rollout_steps):
            # Collect one step
            step_data = collect_rollout_step(
                model, state, envs, device, is_image_based, action_space, cumulated_reward
            )

            # Store in buffer
            rollout_buffer.add(
                state=step_data["state"],
                action=step_data["action"],
                reward=step_data["reward"],
                done=step_data["done"],
                log_prob=step_data["log_prob"],
                value=step_data["value"],
                z_t=step_data["z_t"],
                h_t=step_data["h_t"],
            )

            # Store for world model update
            collected_states.append(step_data["state"])
            collected_new_states.append(step_data["new_state"])
            collected_actions.append(step_data["action"])
            collected_rewards.append(step_data["reward"])

            # Update state
            state = step_data["new_state"]
            dones = step_data["done"]

            # Track episode stats
            episode_rewards += step_data["reward"]
            episode_lengths += 1
            cumulated_reward += step_data["reward"]

            if render_mode == "human":
                render_first_env(envs)

            # Handle episode ends
            finished_envs = torch.where(dones)[0]
            for env_id in finished_envs:
                print(
                    f"Experiment {model.nb_experiments + 1} | "
                    f"Length: {int(episode_lengths[env_id])} | "
                    f"Reward: {episode_rewards[env_id]:.1f}"
                )
                model.nb_experiments += 1
                nb_experiments += 1
                episode_rewards[env_id] = 0
                episode_lengths[env_id] = 0
                cumulated_reward[env_id] = 0
                model.reset_env_memory(env_id)

        # === COMPUTE ADVANTAGES ===
        with torch.no_grad():
            state_tensor = state_transform(state, is_image_based=is_image_based, device=device)
            output_dict = model(
                state_tensor,
                action_space=action_space,
                is_image_based=is_image_based,
                return_losses=True,
                last_reward=cumulated_reward,
            )
            last_value = output_dict["value"].squeeze(-1)

        returns, advantages = rollout_buffer.compute_returns_and_advantages(
            last_value, gamma=gamma, gae_lambda=gae_lambda
        )

        # === UPDATE WORLD MODEL ===
        world_metrics = update_world_model(
            model,
            collected_states,
            collected_new_states,
            collected_rewards,
            collected_actions,
            optimizer,
            device,
            is_image_based,
            action_space,
            loss_func_instance,
        )

        # === UPDATE POLICY (PPO) ===
        policy_metrics = update_policy_ppo(
            model,
            rollout_buffer,
            returns,
            advantages,
            policy_optimizer,
            num_epochs=num_epochs,
            batch_size=batch_size,
            clip_epsilon=clip_epsilon,
            entropy_coeff=entropy_coeff,
        )

        model.iter_num += rollout_steps * envs.num_envs

        # === LOGGING ===
        if writer is not None:
            writer.add_scalar("train/world_loss", world_metrics["total_loss"], model.iter_num)
            writer.add_scalar("train/recon_loss", world_metrics["recon_loss"], model.iter_num)
            writer.add_scalar("train/vq_loss", world_metrics["vq_loss"], model.iter_num)
            writer.add_scalar("train/memory_loss", world_metrics["memory_loss"], model.iter_num)
            writer.add_scalar("train/policy_loss", policy_metrics["policy_loss"], model.iter_num)
            writer.add_scalar("train/value_loss", policy_metrics["value_loss"], model.iter_num)
            writer.add_scalar("train/entropy_loss", policy_metrics["entropy_loss"], model.iter_num)
            writer.add_scalar(
                "train/clip_fraction", policy_metrics["clip_fraction"], model.iter_num
            )

        # === SAVE BEST MODEL ===
        avg_reward = episode_rewards.mean().item()
        if avg_reward > best_avg_reward and model.iter_num > 1000:
            best_avg_reward = avg_reward
            model.save(
                f"{save_path}{envs.spec.id}_{datetime.now().isoformat(timespec='minutes')}.pt",
                envs.single_observation_space,
                envs.single_action_space,
            )

    logger.info(f"Training complete. Total experiments: {nb_experiments}")
