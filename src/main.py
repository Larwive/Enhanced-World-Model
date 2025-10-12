# src/main.py


import argparse
import torch
import numpy as np
import gymnasium as gym
import logging


from interface.interface import GymEnvInterface
from vision.VQ_VAE import VQ_VAE
from vision.Identity import Identity
from memory.TemporalTransformerXCPC import TemporalTransformer
from controller.ModelPredictiveController import ModelPredictiveController
from controller.StochasticController import StochasticController
from WorldModel import WorldModel
from train import train
from reward_predictor.LinearPredictor import LinearPredictorModel
from reward_predictor.DensePredictor import DensePredictorModel


device = torch.device("mps") if torch.backends.mps.is_available() else torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")

torch.set_default_device(device)
torch.set_default_dtype(torch.float32)

# Setup logging to file and console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.FileHandler("train.log", mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--ui', action='store_true', help='Launch the Gradio interface instead of training directly.')
    parser.add_argument('--env-name', type=str, default='CarRacing-v3',
                        help='The Gym environment to use.')  # CartPole-v1
    parser.add_argument('--random-seed', type=int, default=42)
    parser.add_argument('--max-epoch', type=int, default=200)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--save-path', default='./saved_models/')
    parser.add_argument('--load-path', default='TOFILL')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--render-mode', type=str, default='human')
    args = parser.parse_args()

    if args.ui:
        import subprocess
        import sys
        # Launch the Gradio app and exit
        subprocess.run([sys.executable, 'src/interface/app.py'])
        return

    logger.info(str(args))
    logger.info(f"Using device: {device}")

    try:
        # Initialize the environment interface
        interface = GymEnvInterface(args.env_name, render_mode=args.render_mode)
        obs_space = interface.env.observation_space

        is_image_based = len(obs_space.shape) == 3

        if is_image_based:
            logger.info("Detected image-based environment.")
            # (H, W, C) -> (C, H, W)
            obs_shape = obs_space.shape
            input_shape = (obs_shape[2], obs_shape[0], obs_shape[1])
            vision_model = VQ_VAE
            vision_args = {"output_dim": input_shape[0], "embed_dim": 64}
        else:
            logger.info("Detected vector-based environment.")
            input_shape = obs_space.shape
            vision_model = Identity
            vision_args = {}

        # Configure memory and controller based on environment
        action_space = interface.env.action_space
        if isinstance(action_space, gym.spaces.Discrete):
            action_dim = 1  # action_space.n is actually the number of possible values
        else:  # Box, etc.
            action_dim = action_space.shape[0]

        memory_args = {"d_model": 128, "nhead": 8}
        controller_args = {"action_dim": action_dim}
        logger.info(f"Vision model: {vision_model}")
        # Initialize the World Model
        world_model = WorldModel(
            vision_model=vision_model,
            memory_model=TemporalTransformer,
            controller_model=StochasticController,  #ModelPredictiveController,
            input_shape=input_shape,
            vision_args=vision_args,
            memory_args=memory_args,
            controller_args=controller_args,
        ).to(device)

        reward_predictor = world_model.get_reward_predictor(LinearPredictorModel)

        # Start training
        train(world_model, interface, max_iter=args.max_epoch, device=device, learning_rate=args.learning_rate, reward_predictor_model=reward_predictor)

        # Save the trained model
        world_model.save(f"{args.save_path}{args.env_name}_world_model.pt")

        logger.info(f"Model saved to {args.save_path}{args.env_name}_world_model.pt")

        # Close the environment
        interface.close()
        logger.info("Environment closed.")
    except Exception as e:
        logger.exception(f"Exception during training: {e}")

if __name__ == "__main__":
    main()
