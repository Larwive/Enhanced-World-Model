import argparse
import logging
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import torch

import controller
import memory
import vision
from inference import evaluate
from pretrain import pretrain
from train import train
from utils.cli import CLI
from utils.registry import discover_modules
from utils.model import create_world_model

VISION_REGISTRY: dict = discover_modules(vision)
MEMORY_REGISTRY: dict = discover_modules(memory)
CONTROLLER_REGISTRY: dict = discover_modules(controller)

torch.autograd.set_detect_anomaly(True)

device: torch.device = (
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
    interface_group = parser.add_mutually_exclusive_group()
    interface_group.add_argument(
        "--ui",
        action="store_true",
        help="Launch the Gradio interface instead of training directly.",
    )
    interface_group.add_argument(
        "--cli",
        action="store_true",
        help="Runs the command line interface.",
    )
    parser.add_argument(
        "--env",
        type=str,
        default="CartPole-v1",  # "CarRacing-v3",
        help="The Gym environment to use.",
    )  # CartPole-v1
    parser.add_argument("--vision", type=str, default="Identity")
    parser.add_argument("--memory", type=str, default="TemporalTransformer")
    parser.add_argument("--controller", type=str, default="DeepDiscreteController")

    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--patience", type=int, default=5)  # Unused yet, not in CLI.
    parser.add_argument("--batch-size", type=str, default="auto")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--render-mode", type=str, default="rgb_array")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-path", default="./saved_models/")
    parser.add_argument("--load-path", default="")
    parser.add_argument(
        "--save-freq", type=int, default=10, help="Frequency of saving model checkpoints."
    )
    parser.add_argument(
        "--log-freq", type=int, default=10, help="Frequency of logging training progress."
    )
    parser.add_argument("--tensorboard", action="store_true", help="Enable tensorboard logging.")

    # Pretraining args
    parser.add_argument("--pretrain-vision", action="store_true")
    parser.add_argument("--pretrain-memory", action="store_true")
    parser.add_argument("--pretrain-mode", type=str, default="random", choices=["manual", "random"])
    parser.add_argument(
        "--manual-mode-delay",
        type=float,
        default=0.05,
        help="Delay between each step during manual training.",
    )

    # PPO arguments
    parser.add_argument("--rollout-steps", type=int, default=128, help="Number of rollout steps.")
    parser.add_argument(
        "--ppo-epochs", type=int, default=4, help="Number of epochs for PPO training."
    )
    parser.add_argument(
        "--ppo-lr", type=float, default=3e-4, help="Learning rate for PPO training."
    )
    parser.add_argument(
        "--ppo-batch-size", type=int, default=64, help="Batch size for PPO training."
    )
    parser.add_argument(
        "--ppo-clip-range", type=float, default=0.2, help="Clipping parameter for PPO training."
    )
    parser.add_argument(
        "--ppo-range-vf", type=float, default=None, help="Value function for PPO training."
    )
    parser.add_argument(
        "--gamma", type=float, default=0.99, help="Gamma parameter for GAE in PPO training."
    )
    parser.add_argument(
        "--gae-lambda", type=float, default=0.95, help="Lambda parameter for GAE in PPO training."
    )
    parser.add_argument(
        "--value-coef", type=float, default=0.5, help="Value loss coefficient in PPO training."
    )
    parser.add_argument(
        "--entropy-coef", type=float, default=0.01, help="Entropy coefficient in PPO training."
    )
    parser.add_argument(
        "--max-grad-norm", type=float, default=0.5, help="Maximum gradient norm in PPO training."
    )
    parser.add_argument(
        "--no-train-world-model", action="store_true", help="Train the world model."
    )
    parser.add_argument(
        "--world-model-epochs",
        type=int,
        default=1,
        help="Number of epochs for world model training.",
    )

    # Inference arguments
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to run.")
    parser.add_argument("--infer", action="store_true", help="Enable inference mode.")

    args = parser.parse_args()
    if args.cli:
        CLI(args, VISION_REGISTRY, MEMORY_REGISTRY, CONTROLLER_REGISTRY)
    env_batch_size = int(args.batch_size) if args.batch_size.isdigit() else "auto"
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
            args.env, num_envs=env_batch_size, render_mode=real_render_mode
        )  # args.render_mode)

        try:
            log_messages: dict[str, list[str]] = {"info": [], "warning": [], "error": []}
            world_model, obs_space, action_space, log_messages = create_world_model(
                args, VISION_REGISTRY, MEMORY_REGISTRY, CONTROLLER_REGISTRY, device, log_messages
            )
        except Exception as e:
            raise e
        finally:
            for info_message in log_messages["info"]:
                logger.info(info_message)

            for warning_message in log_messages["warning"]:
                logger.warning(warning_message)

            for error_message in log_messages["error"]:
                logger.error(error_message)

        logger.info(f"Vision model: {world_model.vision.__class__.__name__}")
        logger.info(f"Memory model: {world_model.memory.__class__.__name__}")
        logger.info(f"Controller model: {world_model.controller.__class__.__name__}")

        if args.load_path:
            print(f"Loading model from {args.load_path}")
            world_model.load(
                args.load_path, obs_space=obs_space, action_space=action_space, device=device
            )

        if args.infer:
            world_model.eval()
            evaluate(
                world_model, args.env, num_episodes=args.episodes, render_mode=args.render_mode
            )
        elif args.pretrain_vision or args.pretrain_memory:
            if not args.pretrain_vision:
                for param in world_model.vision.parameters():
                    param.requires_grad = False

            if not args.pretrain_memory:
                for param in world_model.memory.parameters():
                    param.requires_grad = False

            for param in world_model.controller.parameters():
                param.requires_grad = False

            save_prefix = (
                "" + ("V" if args.pretrain_vision else "") + ("M" if args.pretrain_memory else "")
            )
            pretrain(
                world_model,
                envs,
                max_iter=args.epochs,
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
                max_iter=args.epochs,
                device=device,
                rollout_steps=args.rollout_steps,
                num_ppo_epochs=args.ppo_epochs,
                batch_size=args.ppo_batch_size,
                clip_range=args.ppo_clip_range,
                clip_range_vf=args.ppo_range_vf,
                gamma=args.gamma,
                gae_lambda=args.gae_lambda,
                learning_rate=args.lr,
                policy_lr=args.ppo_lr,
                value_coef=args.value_coef,
                entropy_coef=args.entropy_coef,
                max_grad_norm=args.max_grad_norm,
                train_world_model=not args.no_train_world_model,
                world_model_epochs=args.world_model_epochs,
                use_tensorboard=args.tensorboard,
                save_path=Path(args.save_path),
                save_freq=args.save_freq,
                log_freq=args.log_freq,
                render_mode=args.render_mode,
            )

            save_name = Path(
                f"{args.save_path}{args.env}_{datetime.now().isoformat(timespec='minutes')}.pt"
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
