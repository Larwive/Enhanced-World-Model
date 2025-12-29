from pathlib import Path
import argparse
import logging
from datetime import datetime

import gymnasium as gym
import torch

import vision
import memory
import controller
import reward_predictor
from pretrain import pretrain
from train import train
from utils.registry import discover_modules
from WorldModel import WorldModel

VISION_REGISTRY: dict = discover_modules(vision)
MEMORY_REGISTRY: dict = discover_modules(memory)
CONTROLLER_REGISTRY: dict = discover_modules(controller)
REWARD_PREDICTOR_REGISTRY: dict = discover_modules(reward_predictor)

torch.autograd.set_detect_anomaly(True)

device = (
    torch.device("mps")
    if torch.backends.mps.is_available()
    else torch.device("cuda")
    if torch.cuda.is_available()
    else torch.device("cpu")
)

torch.set_default_device(device)
torch.set_default_dtype(torch.float32)

# Setup logging to file and console
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler("train.log", mode="w"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ui",
        action="store_true",
        help="Launch the Gradio interface instead of training directly.",
    )
    parser.add_argument(
        "--env-name",
        type=str,
        default="CarRacing-v3",
        help="The Gym environment to use.",
    )  # CartPole-v1
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--max-epoch", type=int, default=200)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--save-path", default="./saved_models/")
    parser.add_argument("--load-path", default="")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--render-mode", type=str, default="rgb_array")
    parser.add_argument("--env-batch-number", type=str, default="auto")
    parser.add_argument("--vision", type=str, default="Identity")
    parser.add_argument("--memory", type=str, default="TemporalTransformer")
    parser.add_argument("--controller", type=str, default="DiscreteModelPredictiveController")
    parser.add_argument("--reward-predictor", type=str, default="LinearPredictor")

    # Pretraining args
    parser.add_argument(
        "--manual-mode-delay",
        type=float,
        default=0.05,
        help="Delay between each step during manual training.",
    )
    parser.add_argument("--pretrain-mode", type=str, default="random", choices=["manual", "random"])
    parser.add_argument("--pretrain-vision", action="store_true")
    parser.add_argument("--pretrain-memory", action="store_true")

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
        subprocess.run([sys.executable, "src/interface/app.py"])
        return

    logger.info(str(args))
    logger.info(f"Using device: {device}")

    try:
        if args.pretrain_vision and args.pretrain_mode == "manual":
            args.render_mode = "rgb_array"  # "human"

        real_render_mode = args.render_mode
        if (
            args.render_mode == "human"
        ):  # Temporary `if` as long as the rendering of the first env is done through cv2.
            real_render_mode = "rgb_array"

        envs = gym.make_vec(
            args.env_name, num_envs=env_batch_size, render_mode=real_render_mode
        )  # args.render_mode)
        obs_space = envs.single_observation_space

        obs_shape = obs_space.shape
        assert obs_shape is not None
        is_image_based = len(obs_shape) == 3

        vision_model = VISION_REGISTRY.get(args.vision, None)
        if vision_model is None:
            raise Exception(
                f"Vision model {args.vision} is not available.\nAvailable models: {list(VISION_REGISTRY.keys())}"
            )

        if is_image_based:
            logger.info("Detected image-based environment.")
            # (H, W, C) -> (C, H, W)
            input_shape = (obs_shape[2], obs_shape[0], obs_shape[1])
            vision_args = {"output_dim": input_shape[0], "embed_dim": 64}
            if "image_based" not in vision_model.tags:
                logger.warning(f"Vision model {args.vision} is not image-based.")
        else:
            logger.info("Detected vector-based environment.")
            input_shape = obs_space.shape
            vision_args = {"embed_dim": obs_space.shape[0]}
            if "vector_based" not in vision_model.tags:
                logger.warning(f"Vision model {args.vision} is not vector-based.")

        memory_model = MEMORY_REGISTRY.get(args.memory, None)
        if memory_model is None:
            raise Exception(
                f"Memory model {args.memory} is not available.\nAvailable models: {list(MEMORY_REGISTRY.keys())}"
            )

        # Configure memory and controller based on environment
        controller_model = CONTROLLER_REGISTRY.get(args.controller, None)
        if controller_model is None:
            raise Exception(
                f"Controller model {args.controller} is not available.\nAvailable models: {list(CONTROLLER_REGISTRY.keys())}"
            )
        action_space = envs.single_action_space
        if isinstance(action_space, gym.spaces.Discrete):
            action_dim = action_space.n  # action_space.n is actually the number of possible values
            if "discrete" not in controller_model.tags:
                logger.warning(
                    f"Controller model {args.controller} is not suitable for discrete action space."
                )
        else:  # Box, etc.
            action_dim = action_space.shape[0]
            if "continuous" not in controller_model.tags:
                logger.warning(
                    f"Controller model {args.controller} is not suitable for continuous action space."
                )

        memory_args = {
            "d_model": 128,
            "latent_dim": vision_args["embed_dim"],
            "action_dim": action_dim,
            "nhead": 8,
        }
        controller_args = {"action_dim": action_dim}
        logger.info(f"Vision model: {vision_model}")
        logger.info(f"Memory model: {memory_model}")
        logger.info(f"Controller model: {controller_model}")

        world_model = WorldModel(
            vision_model=vision_model,
            memory_model=memory_model,
            controller_model=controller_model,  # StochasticController,  #ModelPredictiveController,
            input_shape=input_shape,
            vision_args=vision_args,
            memory_args=memory_args,
            controller_args=controller_args,
        ).to(device)

        reward_predictor_model = REWARD_PREDICTOR_REGISTRY.get(args.reward_predictor, None)
        if reward_predictor_model is None:
            raise Exception(
                f"Reward predictor model {args.reward_predictor} is not available.\nAvailable models: {list(REWARD_PREDICTOR_REGISTRY.keys())}"
            )

        world_model.set_reward_predictor(reward_predictor_model)

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

            save_prefix = (
                "" + ("V" if args.pretrain_vision else "") + ("M" if args.pretrain_memory else "")
            )
            pretrain(
                world_model,
                envs,
                max_iter=args.max_epoch,
                device=device,
                learning_rate=args.learning_rate,
                mode=args.pretrain_mode,
                delay=args.manual_mode_delay,
                save_path=args.save_path,
                save_prefix=save_prefix,
                pretrain_vision=args.pretrain_vision,
                pretrain_memory=args.pretrain_memory,
                render_mode=args.render_mode,
            )
        else:
            if args.load_path:
                for param in world_model.parameters():
                    param.requires_grad = True
                world_model.train()
            train(
                world_model,
                envs,
                max_iter=args.max_epoch,
                device=device,
                learning_rate=args.learning_rate,
                render_mode=args.render_mode,
            )

            save_name = Path(
                f"{args.save_path}{args.env_name}_{datetime.now().isoformat(timespec='minutes')}.pt"
            )
            world_model.save(
                save_name,
                obs_space=obs_space,
                action_space=action_space,
            )

            logger.info(f"Model saved to {save_name}")

        envs.close()
        logger.info("Environment closed.")
    except Exception as e:
        logger.exception(f"Exception during training: {e}")


if __name__ == "__main__":
    main()
