import argparse
import torch
import numpy as np
import gymnasium as gym
import logging


from vision.VQ_VAE import VQ_VAE
from vision.Identity import Identity
from memory.TemporalTransformer import TemporalTransformer
from controller.DiscreteModelPredictiveController import DiscreteModelPredictiveController
from controller.ContinuousModelPredictiveController import ContinuousModelPredictiveController
from controller.StochasticController import StochasticController
from WorldModel import WorldModel
from train import train
from pretrain import pretrain
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
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--save-path', default='./saved_models/')
    parser.add_argument('--load-path', default='')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--render-mode', type=str, default='rgb_array')
    parser.add_argument('--env-batch-number', type=str, default='auto')

    # Pretraining args
    parser.add_argument('--manual-mode-delay', type=float, default=0.05, help='Delay between each step during manual training.')
    parser.add_argument('--pretrain-mode', type=str, default='random', choices=['manual', 'random'])
    parser.add_argument('--pretrain-vision', action='store_true')
    parser.add_argument('--pretrain-memory', action='store_true')


    args = parser.parse_args()

    env_batch_size = int(args.env_batch_number) if args.env_batch_number.isdigit() else "auto"
    if env_batch_size == "auto":
        # TODO: Automatically determines the maximum size of the batch.
        env_batch_size = 2

    logger.info(f"Running with {env_batch_size} parallel environments.")

    if args.ui:
        import subprocess
        import sys
        # Launch the Gradio app and exit
        subprocess.run([sys.executable, 'src/interface/app.py'])
        return

    logger.info(str(args))
    logger.info(f"Using device: {device}")

    try:
        if args.pretrain_vision and args.pretrain_mode == "manual":
            args.render_mode = "rgb_array" #"human"
        
        real_render_mode = args.render_mode
        if args.render_mode == "human":  # Temporary `if` as long as the rendering of the first env is done through cv2.
            real_render_mode = "rgb_array"

        envs = gym.make_vec(args.env_name, num_envs=env_batch_size, render_mode=real_render_mode) #args.render_mode)
        obs_space = envs.single_observation_space
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
            vision_args = {"embed_dim": obs_space.shape[0]}

        # Configure memory and controller based on environment
        action_space = envs.single_action_space
        if isinstance(action_space, gym.spaces.Discrete):
            action_dim = action_space.n  # action_space.n is actually the number of possible values
            controller_model = DiscreteModelPredictiveController
        else:  # Box, etc.
            action_dim = action_space.shape[0]
            controller_model = ContinuousModelPredictiveController

        memory_args = {"d_model": 128, "latent_dim": vision_args["embed_dim"], "action_dim": action_dim, "nhead": 8}
        controller_args = {"action_dim": action_dim}
        logger.info(f"Vision model: {vision_model}")
        # Initialize the World Model
        world_model = WorldModel(
            vision_model=vision_model,
            memory_model=TemporalTransformer,
            controller_model=controller_model,#StochasticController,  #ModelPredictiveController,
            input_shape=input_shape,
            vision_args=vision_args,
            memory_args=memory_args,
            controller_args=controller_args,
        ).to(device)

        world_model.set_reward_predictor(LinearPredictorModel)

        if args.load_path:
            print(f"Loading model from {args.load_path}")
            world_model.load(args.load_path, obs_space=obs_space, action_space=action_space)


        if args.pretrain_vision or args.pretrain_memory:
            if not args.pretrain_vision:
                for param in world_model.vision.parameters():
                    param.requires_grad = False
            
            if not args.pretrain_memory:
                for param in world_model.memory.parameters():
                    param.requires_grad = False

            for param in world_model.controller.parameters():
                param.requires_grad = False

            if world_model.reward_predictor is not None:
                for param in world_model.reward_predictor.parameters():
                    param.requires_grad = False

            save_prefix = "" + ("V" if args.pretrain_vision else "") + ("M" if args.pretrain_memory else "")
            pretrain(world_model, envs, max_iter=args.max_epoch, device=device, learning_rate=args.learning_rate, mode=args.pretrain_mode, delay=args.manual_mode_delay, save_path=args.save_path, save_prefix=save_prefix, pretrain_vision=args.pretrain_vision, pretrain_memory=args.pretrain_memory, render_mode = args.render_mode)
        else:
            if args.load_path:
                for param in world_model.parameters():
                    param.requires_grad = True
                world_model.train()
            train(world_model, envs, max_iter=args.max_epoch, device=device, learning_rate=args.learning_rate, render_mode = args.render_mode)
            world_model.save(f"{args.save_path}{args.env_name}_world_model.pt", obs_space=obs_space, action_space=action_space)

            logger.info(f"Model saved to {args.save_path}{args.env_name}_world_model.pt")

        envs.close()
        logger.info("Environment closed.")
    except Exception as e:
        logger.exception(f"Exception during training: {e}")

if __name__ == "__main__":
    main()
