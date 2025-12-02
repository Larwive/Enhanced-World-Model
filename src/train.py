# src/train.py

import torch
import numpy as np
import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard.summary import hparams

from WorldModel import WorldModel


class SummaryWriter(SummaryWriter):
    """
    Add possiblity to store hyperparameters.
    """

    def add_hparams(self, hparam_dict, metric_dict):
        torch._C._log_api_usage_once("tensorboard.logging.add_hparams")
        if type(hparam_dict) is not dict or type(metric_dict) is not dict:
            raise TypeError('hparam_dict and metric_dict should be dictionary.')
        exp, ssi, sei = hparams(hparam_dict, metric_dict)

        logdir = self._get_file_writer().get_logdir()

        with SummaryWriter(log_dir=logdir) as w_hp:
            w_hp.file_writer.add_summary(exp)
            w_hp.file_writer.add_summary(ssi)
            w_hp.file_writer.add_summary(sei)
            for k, v in metric_dict.items():
                w_hp.add_scalar(k, v)


def step(model,
         state,
         optimizer,
         device,
         is_image_based,
         action_space,
         reward,
         iter_num: int = 0,
         tensorboard_writer=None):
    optimizer.zero_grad()
    if is_image_based:
        # Transpose state from (B, H, W, C) to (B, C, H, W) for PyTorch
        state_transposed = np.transpose(state, (0, 3, 1, 2))
        state_tensor = torch.from_numpy(state_transposed).float().to(device)
        # Normalize image data to [0, 1]
        state_tensor = state_tensor / 255.0
    else:
        # For vector data, just move to device (already has batch dimension)
        state_tensor = torch.from_numpy(state).float().to(device)

    output_dict = model(state_tensor, action_space=action_space, is_image_based=is_image_based, return_losses=True)
    # print(output_dict)
    total_loss = torch.sum(output_dict["total_loss"]) - (reward * output_dict["log_probs"]).mean()

    total_loss.backward()

    if tensorboard_writer is not None:
        tensorboard_writer.add_scalar("train/loss", total_loss.item(), iter_num)
        for name, param in model.named_parameters():
            assert param is not None, "Parameter {} is None".format(name)
            #log the gradient that all
            if param.grad is not None:
                tensorboard_writer.add_scalar(f"gradients/{name}", param.grad.norm().item(), iter_num)

    optimizer.step()

    return output_dict["action"], total_loss.item()


def train(model: WorldModel, envs, max_iter=10000, device='cpu', use_tensorboard: bool = True, learning_rate:float = 0.01):
    """
    Legacy training function (simple policy gradient).
    NOTE: This has known issues. Use train_a2c.py for better results.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    is_image_based = len(envs.single_observation_space.shape) == 3
    action_space = envs.single_action_space
    num_envs = envs.num_envs

    writer = SummaryWriter() if use_tensorboard else None

    last_rewards = np.zeros(num_envs)

    for iter_num in range(max_iter):
        state, info = envs.reset()
        dones = np.zeros(num_envs, dtype=bool)
        total_episode_loss = 0
        steps = 0

        while not dones.all():
            action_tensor, loss = step(model,
                                       state,
                                       optimizer,
                                       device,
                                       is_image_based,
                                       action_space,
                                       last_rewards.mean(),  # Use mean reward
                                       iter_num=iter_num,
                                       tensorboard_writer=writer)

            # Handle different action spaces
            if isinstance(action_space, gym.spaces.Discrete):
                action_np = action_tensor.cpu().numpy()
            else:  # Continuous action space
                action_np = action_tensor.cpu().detach().numpy()

            next_state, rewards, terminated, truncated, info = envs.step(action_np)
            dones = terminated | truncated
            state = next_state
            last_rewards = rewards
            total_episode_loss += loss
            steps += 1

            # Reset memories for done environments
            for env_idx in np.where(dones)[0]:
                model.reset_env_memory(env_idx)

        print(f"Iteration {iter_num + 1}/{max_iter}, Steps: {steps}, Total Loss: {total_episode_loss:.4f}, Mean Reward: {last_rewards.mean():.2f}")
