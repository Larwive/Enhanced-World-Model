# src/train_a2c.py

import torch
import numpy as np
import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter
from collections import deque

from WorldModel import WorldModel


def compute_gae(rewards, values, next_values, dones, gamma=0.99, gae_lambda=0.95):
    """
    Compute Generalized Advantage Estimation (GAE).

    Args:
        rewards: (T, B) - rewards at each timestep
        values: (T, B) - value estimates at each timestep
        next_values: (T, B) - value estimates at next timestep
        dones: (T, B) - done flags
        gamma: discount factor
        gae_lambda: GAE lambda parameter

    Returns:
        advantages: (T, B) - advantage estimates
        returns: (T, B) - discounted returns
    """
    T, B = rewards.shape
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros(B, device=rewards.device)

    # Backward pass to compute GAE
    for t in reversed(range(T)):
        if t == T - 1:
            next_value = next_values[t]
        else:
            next_value = values[t + 1]

        # TD error: δ_t = r_t + γ * V(s_{t+1}) * (1 - done) - V(s_t)
        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]

        # GAE: A_t = δ_t + γ * λ * (1 - done) * A_{t+1}
        gae = delta + gamma * gae_lambda * (1 - dones[t]) * gae
        advantages[t] = gae

    # Returns = Advantages + Values
    returns = advantages + values

    return advantages, returns


def state_transform(state, is_image_based, device):
    """Transform state to tensor."""
    if is_image_based:
        if state.ndim == 3:
            state = state[None]
        state_transposed = np.transpose(state, (0, 3, 1, 2))
        state_tensor = torch.from_numpy(state_transposed).float().to(device)
        return state_tensor / 255.0
    else:
        if state.ndim == 1:
            state = state[None]
        return torch.from_numpy(state).float().to(device)


def train_a2c(model: WorldModel,
              envs,
              max_epochs=200,
              n_steps=128,
              device='cpu',
              learning_rate=3e-4,
              gamma=0.99,
              gae_lambda=0.95,
              value_coef=0.5,
              entropy_coef=0.01,
              memory_coef=0.1,
              max_grad_norm=0.5,
              use_tensorboard=True,
              save_path="./",
              save_prefix="a2c"):
    """
    Train world model using A2C (Advantage Actor-Critic).

    Args:
        model: WorldModel instance
        envs: Vectorized gymnasium environments
        max_epochs: Number of epochs to train
        n_steps: Number of steps to collect before update
        device: Device to train on
        learning_rate: Learning rate
        gamma: Discount factor
        gae_lambda: GAE lambda
        value_coef: Value loss coefficient
        entropy_coef: Entropy bonus coefficient
        memory_coef: Memory prediction loss coefficient
        max_grad_norm: Max gradient norm for clipping
        use_tensorboard: Whether to log to tensorboard
        save_path: Path to save checkpoints
        save_prefix: Prefix for checkpoint filenames
    """
    model.to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    is_image_based = len(envs.single_observation_space.shape) == 3
    action_space = envs.single_action_space
    num_envs = envs.num_envs

    writer = SummaryWriter() if use_tensorboard else None

    # Storage for rollout data
    states = []
    actions = []
    log_probs = []
    values = []
    rewards = []
    dones = []
    z_predictions = []

    # Initialize environment
    state, info = envs.reset()
    episode_rewards = deque(maxlen=100)
    episode_lengths = deque(maxlen=100)
    current_episode_rewards = np.zeros(num_envs)
    current_episode_lengths = np.zeros(num_envs)

    global_step = 0
    best_mean_reward = -float('inf')

    print(f"Starting A2C training on {envs.spec.id} with {num_envs} parallel environments")
    print(f"Controller: {model.controller.__class__.__name__}")
    print(f"Planning enabled: {getattr(model.controller, 'use_planning', False)}")

    for epoch in range(max_epochs):
        # Collect n_steps of experience
        for step in range(n_steps):
            state_tensor = state_transform(state, is_image_based, device)

            with torch.no_grad():
                output = model(state_tensor, action_space=action_space,
                             is_image_based=is_image_based, return_losses=True)

            # Store data
            states.append(state_tensor)
            actions.append(output["action"])
            log_probs.append(output["log_probs"])
            values.append(output["value"])
            z_predictions.append(output["memory_prediction"])

            # Execute action in environment
            if isinstance(action_space, gym.spaces.Discrete):
                action_np = output["action"].cpu().numpy()
            else:
                action_np = output["action"].cpu().numpy()

            next_state, reward, terminated, truncated, info = envs.step(action_np)
            done = terminated | truncated

            rewards.append(torch.from_numpy(reward).float().to(device))
            dones.append(torch.from_numpy(done).float().to(device))

            # Track episode statistics
            current_episode_rewards += reward
            current_episode_lengths += 1

            for env_idx in range(num_envs):
                if done[env_idx]:
                    episode_rewards.append(current_episode_rewards[env_idx])
                    episode_lengths.append(current_episode_lengths[env_idx])
                    current_episode_rewards[env_idx] = 0
                    current_episode_lengths[env_idx] = 0
                    model.reset_env_memory(env_idx)

            state = next_state
            global_step += num_envs

        # Get value of final state for bootstrapping
        with torch.no_grad():
            next_state_tensor = state_transform(state, is_image_based, device)
            final_output = model(next_state_tensor, action_space=action_space,
                               is_image_based=is_image_based, return_losses=True)
            next_values = final_output["value"]

        # Convert lists to tensors
        states_tensor = torch.cat(states, dim=0)  # (T*B, C, H, W) or (T*B, D)
        actions_tensor = torch.stack(actions, dim=0)  # (T, B, ...)
        log_probs_old = torch.cat(log_probs, dim=0).view(n_steps, num_envs)  # (T, B)
        values_tensor = torch.cat(values, dim=0).view(n_steps, num_envs)  # (T, B)
        rewards_tensor = torch.stack(rewards, dim=0)  # (T, B)
        dones_tensor = torch.stack(dones, dim=0)  # (T, B)
        z_pred_tensor = torch.stack(z_predictions[:-1], dim=0)  # (T-1, B, latent_dim)

        # Compute advantages and returns
        next_values_tensor = torch.cat([values_tensor[1:], next_values.unsqueeze(0)], dim=0)
        advantages, returns = compute_gae(
            rewards_tensor, values_tensor, next_values_tensor,
            dones_tensor, gamma, gae_lambda
        )

        # Normalize advantages (improves stability)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # === COMPUTE LOSSES ===
        optimizer.zero_grad()

        # Forward pass through model for all collected states
        # Reshape for batch processing
        batch_size = n_steps * num_envs
        states_batch = states_tensor.view(batch_size, *states_tensor.shape[2:])

        # Split into chunks to avoid memory issues
        chunk_size = 32
        all_outputs = []
        for i in range(0, batch_size, chunk_size):
            chunk = states_batch[i:i+chunk_size]
            with torch.set_grad_enabled(True):
                chunk_output = model(chunk, action_space=action_space,
                                   is_image_based=is_image_based, return_losses=True)
            all_outputs.append(chunk_output)

        # Combine chunk outputs
        combined_output = {
            "total_loss": torch.cat([o["total_loss"] for o in all_outputs]),
            "recon_loss": torch.cat([o["recon_loss"] for o in all_outputs]),
            "vq_loss": torch.cat([o["vq_loss"] for o in all_outputs]),
            "log_probs": torch.cat([o["log_probs"] for o in all_outputs]),
            "value": torch.cat([o["value"] for o in all_outputs]),
            "memory_prediction": torch.cat([o["memory_prediction"] for o in all_outputs]),
        }

        # Reshape outputs
        log_probs_new = combined_output["log_probs"].view(n_steps, num_envs)
        values_new = combined_output["value"].view(n_steps, num_envs)
        z_pred_new = combined_output["memory_prediction"].view(n_steps, num_envs, -1)

        # Policy loss (A2C actor loss)
        policy_loss = -(log_probs_new * advantages.detach()).mean()

        # Value loss (A2C critic loss)
        value_loss = torch.nn.functional.mse_loss(values_new, returns.detach())

        # Entropy bonus (encourages exploration)
        # Note: entropy is approximated from log_probs for simplicity
        entropy = -log_probs_new.mean()

        # Vision losses (reconstruction + VQ)
        vision_loss = combined_output["total_loss"].mean()

        # Memory prediction loss (predict next latent state)
        # Get actual next latents
        with torch.no_grad():
            next_states = torch.cat([states_tensor[1:], next_state_tensor.unsqueeze(0)], dim=0)
            next_latents = []
            for i in range(0, next_states.shape[0], chunk_size):
                chunk = next_states[i:i+chunk_size]
                z_e = model.vision.encode(chunk, is_image_based=is_image_based)
                if is_image_based:
                    z_e = z_e.mean(dim=(2, 3))
                next_latents.append(z_e)
            next_latents = torch.cat(next_latents, dim=0).view(n_steps, num_envs, -1)

        memory_loss = torch.nn.functional.mse_loss(
            z_pred_new, next_latents.detach()
        )

        # Total loss
        total_loss = (policy_loss +
                     value_coef * value_loss +
                     vision_loss +
                     memory_coef * memory_loss -
                     entropy_coef * entropy)

        # Backward pass
        total_loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer.step()

        # === LOGGING ===
        if writer is not None and global_step % 100 == 0:
            writer.add_scalar("train/policy_loss", policy_loss.item(), global_step)
            writer.add_scalar("train/value_loss", value_loss.item(), global_step)
            writer.add_scalar("train/entropy", entropy.item(), global_step)
            writer.add_scalar("train/vision_loss", vision_loss.item(), global_step)
            writer.add_scalar("train/memory_loss", memory_loss.item(), global_step)
            writer.add_scalar("train/total_loss", total_loss.item(), global_step)

            if len(episode_rewards) > 0:
                mean_reward = np.mean(episode_rewards)
                mean_length = np.mean(episode_lengths)
                writer.add_scalar("train/mean_episode_reward", mean_reward, global_step)
                writer.add_scalar("train/mean_episode_length", mean_length, global_step)

        # Clear rollout buffers
        states.clear()
        actions.clear()
        log_probs.clear()
        values.clear()
        rewards.clear()
        dones.clear()
        z_predictions.clear()

        # Print progress
        if epoch % 10 == 0:
            if len(episode_rewards) > 0:
                mean_reward = np.mean(episode_rewards)
                print(f"Epoch {epoch}/{max_epochs} | Step {global_step} | "
                      f"Mean Reward: {mean_reward:.2f} | "
                      f"Policy Loss: {policy_loss.item():.4f} | "
                      f"Value Loss: {value_loss.item():.4f} | "
                      f"Memory Loss: {memory_loss.item():.4f}")

                # Save best model
                if mean_reward > best_mean_reward:
                    best_mean_reward = mean_reward
                    save_file = f"{save_path}{save_prefix}_{envs.spec.id}_best.pt"
                    model.save(save_file, envs.single_observation_space, envs.single_action_space)
                    print(f"  → Saved new best model with reward {mean_reward:.2f}")
            else:
                print(f"Epoch {epoch}/{max_epochs} | Step {global_step} | "
                      f"Policy Loss: {policy_loss.item():.4f} | "
                      f"Value Loss: {value_loss.item():.4f}")

        # Periodic checkpoint
        if epoch % 50 == 0 and epoch > 0:
            save_file = f"{save_path}{save_prefix}_{envs.spec.id}_epoch{epoch}.pt"
            model.save(save_file, envs.single_observation_space, envs.single_action_space)
            print(f"  → Saved checkpoint at epoch {epoch}")

    # Final save
    save_file = f"{save_path}{save_prefix}_{envs.spec.id}_final.pt"
    model.save(save_file, envs.single_observation_space, envs.single_action_space)
    print(f"\nTraining complete! Final model saved to {save_file}")

    if writer is not None:
        writer.close()
