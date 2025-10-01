# src/train.py

import torch
import numpy as np
import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard.summary import hparams

from interface.interface import GymEnvInterface
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
        # Transpose state from (H, W, C) to (C, H, W) for PyTorch
        state_transposed = np.transpose(state, (2, 0, 1))
        state_tensor = torch.from_numpy(state_transposed).float().unsqueeze(0).to(device)
        # Normalize image data to [0, 1]
        state_tensor = state_tensor / 255.0
    else:
        # For vector data, just add batch dimension and move to device
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)

    output_dict = model(state_tensor, return_losses=True, action_space=action_space)
    print(output_dict)
    total_loss = torch.sum(output_dict["total_loss"]) - (reward * output_dict["log_probs"]).mean()

    if tensorboard_writer is not None:
        tensorboard_writer.add_scalar("train/loss", total_loss.item(), iter_num)
        for name, param in model.named_parameters():
            if param.grad is not None:
                tensorboard_writer.add_scalar(f"gradients/{name}", param.grad.norm().item(), iter_num)
            else:
                print("ERROR IN GRADIENTS ! {} is None".format(name))

    total_loss.backward()
    optimizer.step()

    return output_dict["action"], total_loss.item()


def train(model: WorldModel, interface: GymEnvInterface, max_iter=10000, device='cpu', use_tensorboard: bool = True, learning_rate:float = 0.01):
    print(model.parameters)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    is_image_based = len(interface.env.observation_space.shape) == 3
    action_space = interface.env.action_space

    writer = SummaryWriter() if use_tensorboard else None

    last_reward = 0

    for iter_num in range(max_iter):
        state, info = interface.reset()
        done = False
        total_episode_loss = 0

        while not done:
            action_tensor, loss = step(model,
                                       state,
                                       optimizer,
                                       device,
                                       is_image_based,
                                       action_space,
                                       last_reward,
                                       iter_num=iter_num,
                                       tensorboard_writer=writer)

            # Handle different action spaces
            if isinstance(action_space, gym.spaces.Discrete):
                action_np = action_tensor.argmax(dim=1).cpu().numpy()
            else:  # Continuous action space
                action_np = action_tensor.squeeze(0).cpu().detach().numpy()

            next_state, last_reward, done, info = interface.step(action_np)
            state = next_state
            total_episode_loss += loss
            if interface.env.render_mode == 'human':
                interface.render()

        print(f"Iteration {iter_num + 1}/{max_iter}, Total Loss: {total_episode_loss:.4f}")
