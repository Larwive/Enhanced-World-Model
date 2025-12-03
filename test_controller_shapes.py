#!/usr/bin/env python3
"""
Quick test to identify controller shape mismatches
"""
import torch
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.getcwd(), 'src'))

import gymnasium as gym
from WorldModel import WorldModel
from vision.Identity import Identity
from vision.VQ_VAE import VQ_VAE
from memory.TemporalTransformer import TemporalTransformer
from controller.ImprovedDiscreteController import ImprovedDiscreteController
from controller.ImprovedContinuousController import ImprovedContinuousController

def test_cartpole():
    print("\n" + "="*60)
    print("Testing CartPole-v1 (Discrete)")
    print("="*60)

    # Create environment
    env = gym.make("CartPole-v1")
    obs_space = env.observation_space
    action_space = env.action_space

    print(f"Observation space: {obs_space}")
    print(f"Action space: {action_space}")

    # Configure world model
    input_shape = obs_space.shape
    vision_args = {"embed_dim": 4}  # Small for CartPole
    memory_args = {"d_model": 128, "latent_dim": 4, "action_dim": action_space.n, "nhead": 8}
    controller_args = {"action_dim": action_space.n, "use_planning": True, "planning_horizon": 5}

    # Create model
    model = WorldModel(
        vision_model=Identity,
        memory_model=TemporalTransformer,
        controller_model=ImprovedDiscreteController,
        input_shape=input_shape,
        vision_args=vision_args,
        memory_args=memory_args,
        controller_args=controller_args,
    )

    print(f"\nModel created successfully")
    print(f"Vision embed_dim: {model.vision.embed_dim}")
    print(f"Memory d_model: {model.memory_d_model}")
    print(f"Controller z_dim: {model.controller.z_dim}")
    print(f"Controller h_dim: {model.controller.h_dim}")

    # Test with different batch sizes
    for batch_size in [1, 4]:
        print(f"\n--- Testing with batch_size={batch_size} ---")

        # Create fake observation
        obs = torch.randn(batch_size, *obs_space.shape)

        print(f"Input shape: {obs.shape}")

        # Forward pass
        try:
            output = model(obs, action_space=action_space, is_image_based=False, return_losses=True)
            print(f"✅ Forward pass successful!")
            print(f"   Action shape: {output['action'].shape}")
            print(f"   Value shape: {output['value'].shape}")
            print(f"   Log probs shape: {output['log_probs'].shape}")
        except Exception as e:
            print(f"❌ Forward pass failed!")
            print(f"   Error: {e}")
            import traceback
            traceback.print_exc()
            return False

    return True

def test_continuous():
    print("\n" + "="*60)
    print("Testing Continuous Action Space")
    print("="*60)

    # Create a simple continuous environment
    env = gym.make("Pendulum-v1")
    obs_space = env.observation_space
    action_space = env.action_space

    print(f"Observation space: {obs_space}")
    print(f"Action space: {action_space}")

    # Configure world model
    input_shape = obs_space.shape
    action_dim = action_space.shape[0]
    # For Identity model, embed_dim will be obs_space.shape[0]
    embed_dim = obs_space.shape[0]
    vision_args = {"embed_dim": embed_dim}
    memory_args = {"d_model": 128, "latent_dim": embed_dim, "action_dim": action_dim, "nhead": 8}
    controller_args = {"action_dim": action_dim, "use_planning": True, "planning_horizon": 5}

    # Create model
    model = WorldModel(
        vision_model=Identity,
        memory_model=TemporalTransformer,
        controller_model=ImprovedContinuousController,
        input_shape=input_shape,
        vision_args=vision_args,
        memory_args=memory_args,
        controller_args=controller_args,
    )

    print(f"\nModel created successfully")
    print(f"Vision embed_dim: {model.vision.embed_dim}")
    print(f"Memory d_model: {model.memory_d_model}")
    print(f"Controller z_dim: {model.controller.z_dim}")
    print(f"Controller h_dim: {model.controller.h_dim}")

    # Test with different batch sizes
    for batch_size in [1, 4]:
        print(f"\n--- Testing with batch_size={batch_size} ---")

        # Create fake observation
        obs = torch.randn(batch_size, *obs_space.shape)

        print(f"Input shape: {obs.shape}")

        # Forward pass
        try:
            output = model(obs, action_space=action_space, is_image_based=False, return_losses=True)
            print(f"✅ Forward pass successful!")
            print(f"   Action shape: {output['action'].shape}")
            print(f"   Value shape: {output['value'].shape}")
            print(f"   Log probs shape: {output['log_probs'].shape}")
        except Exception as e:
            print(f"❌ Forward pass failed!")
            print(f"   Error: {e}")
            import traceback
            traceback.print_exc()
            return False

    return True

if __name__ == "__main__":
    print("Testing controller shapes...\n")

    success = True
    success = test_cartpole() and success
    success = test_continuous() and success

    if success:
        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("❌ SOME TESTS FAILED")
        print("="*60)
        sys.exit(1)
