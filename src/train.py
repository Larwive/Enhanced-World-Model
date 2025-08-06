# src/train.py

import torch
import numpy as np
import gymnasium as gym
from interface.interface import GymEnvInterface
from WorldModel import WorldModel

def step(model, state, optimizer, device, is_image_based):
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
    
    output_dict = model(state_tensor, return_losses=True)
    total_loss = torch.sum(output_dict["total_loss"])
    total_loss.backward()
    optimizer.step()
    
    return output_dict["action"], total_loss.item()

def train(model: WorldModel, interface: GymEnvInterface, max_iter=10000, device='cpu'):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    is_image_based = len(interface.env.observation_space.shape) == 3
    action_space = interface.env.action_space
    
    for iter_num in range(max_iter):
        state, info = interface.reset()
        done = False
        total_episode_loss = 0
        
        while not done:
            action_tensor, loss = step(model, state, optimizer, device, is_image_based)
            
            # Handle different action spaces
            if isinstance(action_space, gym.spaces.Discrete):
                action_np = action_tensor.argmax(dim=1).cpu().numpy()
            else: # Continuous action space
                action_np = action_tensor.squeeze(0).cpu().detach().numpy()

            next_state, reward, done, info = interface.step(action_np)
            state = next_state
            total_episode_loss += loss
            if interface.env.render_mode == 'human':
                interface.render()
            
        print(f"Iteration {iter_num + 1}/{max_iter}, Total Loss: {total_episode_loss:.4f}")
