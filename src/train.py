# src/train.py

import torch
import numpy as np
import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter

from interface.interface import GymEnvInterface
from WorldModel import WorldModel


# SummaryWriter class remains the same...

def train(model: WorldModel, interface: GymEnvInterface, max_iter=10000, device='cpu', use_tensorboard: bool = True, learning_rate:float = 0.01):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    is_image_based = len(interface.env.observation_space.shape) == 3
    action_space = interface.env.action_space

    writer = SummaryWriter() if use_tensorboard else None

    for iter_num in range(max_iter):
        state, info = interface.reset()
        done = False
        total_episode_loss = 0
        episode_reward = 0

        while not done:
            optimizer.zero_grad()

            if is_image_based:
                state_transposed = np.transpose(state, (2, 0, 1))
                state_tensor = torch.from_numpy(state_transposed).float().unsqueeze(0).to(device)
                state_tensor = state_tensor / 255.0
            else:
                state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
            
            # This is a shared forward pass.
            output_dict = model(state_tensor, return_losses=True, action_space=action_space)
            
            action_tensor = output_dict["action"]
            
            if isinstance(action_space, gym.spaces.Discrete):
                action_np = action_tensor.argmax(dim=1).cpu().detach().numpy()
            else:
                action_np = action_tensor.squeeze(0).cpu().detach().numpy()

            next_state, reward, done, info = interface.step(action_np)

            # Separate losses for clarity and stable backpropagation
            vision_loss = output_dict["recon_loss"] + output_dict["vq_loss"]
            policy_loss = -(reward * output_dict["log_probs"] * 2).mean()
            
            # Combine losses
            total_loss = vision_loss + policy_loss

            # Backpropagate
            total_loss.backward()

            if writer is not None:
                writer.add_scalar("train/total_loss", total_loss.item(), iter_num)
                writer.add_scalar("train/vision_loss", vision_loss.item(), iter_num)
                writer.add_scalar("train/policy_loss", policy_loss.item(), iter_num)
                for name, param in model.named_parameters():
                    if param.grad is not None and param.grad.norm().item() > 0:
                        writer.add_scalar(f"gradients/{name}", param.grad.norm().item(), iter_num)
                    else:
                        print(f"Warning: Gradient for {name} is None or zero.")

            optimizer.step()

            state = next_state
            total_episode_loss += total_loss.item()
            episode_reward += reward
            if interface.env.render_mode == 'human':
                interface.render()

        if writer is not None:
            writer.add_scalar("train/episode_reward", episode_reward, iter_num)

        print(f"Iteration {iter_num + 1}/{max_iter}, Total Loss: {total_episode_loss:.4f}, Episode Reward: {episode_reward:.2f}")