import json
import os
from argparse import Namespace
from collections.abc import Callable

from utils.gym_tools import action_space_is_discrete, get_all_gym_envs, gym_is_image_based
from utils.logger import Logger, Style

cli_printer = Logger()

bool_state = ["Disabled", "Enabled"]


def save_args(args: Namespace) -> None:
    with open("cli.json", "w") as f:
        json.dump(vars(args), f, indent=4)
    cli_printer.log("Arguments saved successfully.", style=Style.CYAN + Style.INVERT)


def load_args(args: Namespace) -> None:
    if not os.path.exists("cli.json"):
        cli_printer.error("No saved arguments found. Save args before loading.")
        return
    with open("cli.json") as f:
        args_dict = json.load(f)
    args.__dict__.update(args_dict)
    cli_printer.log("Arguments loaded successfully.", style=Style.CYAN + Style.INVERT)


def print_separator() -> None:
    cli_printer.log("------------------------------------------------", style=Style.YELLOW)


def print_main_args(
    args: Namespace, vision_registry: dict, _memory_registry: dict, controller_registry: dict
) -> None:
    print_separator()
    cli_printer.log("Main arguments:", style=Style.CYAN + Style.UNDERLINE)

    is_image_based = gym_is_image_based(args.env)
    if is_image_based and "image_based" not in vision_registry[args.vision].tags:
        vision_warning = (
            f"{Style.RED} Warning: The selected vision model is not image-based.{Style.RESET}"
        )
    elif not is_image_based and "image_based" in vision_registry[args.vision].tags:
        vision_warning = (
            f"{Style.RED} Warning: The selected vision model is image-based.{Style.RESET}"
        )
    else:
        vision_warning = ""

    is_discrete = action_space_is_discrete(args.env)
    if is_discrete and "continuous" in controller_registry[args.controller].tags:
        controller_warning = (
            f"{Style.RED} Warning: The selected controller is continuous.{Style.RESET}"
        )
    elif not is_discrete and "discrete" in controller_registry[args.controller].tags:
        controller_warning = (
            f"{Style.RED} Warning: The selected controller is discrete.{Style.RESET}"
        )
    else:
        controller_warning = ""

    main_dict = {
        "  Env": str(args.env),
        "  - Vision    ": str(args.vision) + vision_warning,
        "  - Memory    ": str(args.memory),
        "  - Controller": str(args.controller) + controller_warning,
        "  - Epochs": str(args.epochs),
        "  - Patience": str(args.patience),
        "  - Batch size": str(args.batch_size),
        "  - Learning rate": str(args.lr),
        "  - Dropout": str(args.dropout),
        "  - Render mode": str(args.render_mode),
    }

    cli_printer.dict_log(main_dict)

    if args.load_path:
        cli_printer.warn("\nModel choices will be overriden by model to load !")


def edit_main_args(
    args: Namespace, vision_registry: dict, memory_registry: dict, controller_registry: dict
) -> None:
    print_separator()

    while True:
        cli_printer.log("\nEditing main configuration:", style=Style.CYAN + Style.INVERT)
        main_edit_dict = {
            "  - 0: Environment": str(args.env),
            "  - 1: Vision": str(args.vision),
            "  - 2: Memory": str(args.memory),
            "  - 3: Controller": str(args.controller),
            "  - 4: Epochs": str(args.epochs),
            "  - 5: Patience": str(args.patience),
            "  - 6: Batch size": str(args.batch_size),
            "  - 7: Learning rate": str(args.lr),
            "  - 8: Dropout": str(args.dropout),
            "  - 9: Render mode": str(args.render_mode),
            "  - x: : Back ": "",
        }
        cli_printer.dict_log(main_edit_dict)

        command = cli_printer.input("What to edit ? ")

        match command:
            case "0":
                env_names = get_all_gym_envs()
                env_listing = {f"{i}": name for i, name in enumerate(env_names)}

                cli_printer.log("Available envs:")
                cli_printer.dict_log(env_listing, value_style=Style.MAGENTA)
                env_id = cli_printer.input("Enter environment id: ")
                if env_id.isdigit() and 0 <= int(env_id) < len(env_names):
                    args.env = env_names[int(env_id)]
                else:
                    cli_printer.error("Invalid environment choice.")
                    continue
            case "1" | "2" | "3":
                edit_choice = int(command)
                registries = [vision_registry, memory_registry, controller_registry][
                    edit_choice - 1
                ]

                models = {
                    f"{i}": f"{name}{Style.RESET} [{Style.BLUE}{', '.join(cls.tags)}{Style.RESET}]"
                    for i, (name, cls) in enumerate(registries.items())
                }
                cli_printer.dict_log(models, value_style=Style.MAGENTA)
                model_choice = cli_printer.input("Model number: ")

                if not model_choice.isdigit():
                    cli_printer.error("Invalid model choice.")
                    continue
                model_id_choice = int(model_choice)
                if 0 <= model_id_choice < len(registries):
                    fields = ["vision", "memory", "controller"]

                    setattr(args, fields[edit_choice - 1], list(registries.keys())[model_id_choice])
                else:
                    cli_printer.error(
                        f"Invalid model choice. Enter a number between 0 and {len(registries) - 1}."
                    )
                    continue
            case "4" | "5" | "6" | "7" | "8":
                edit_choice = int(command)
                value = cli_printer.input(
                    f"Enter new value for {['number of epochs', 'patience', 'batch size', 'learning rate', 'dropout'][edit_choice - 4]}: "
                )
                if value == "auto" and command == "6":
                    args.batch_size = "auto"
                    continue
                fields = ["epochs", "patience", "batch_size", "lr", "dropout"]
                try:

                    def str_int(x: str) -> str:
                        return str(int(x))

                    casters: list[Callable[[str], object]] = [int, int, str_int, float, float]
                    value_object = casters[edit_choice - 4](value)
                    if (
                        isinstance(value_object, int) or isinstance(value_object, float)
                    ) and value_object < 0:
                        cli_printer.error("Negative value.")
                        continue
                    setattr(args, fields[edit_choice - 4], value_object)
                except ValueError:
                    cli_printer.error("Invalid value.")
                    continue
            case "9":
                render_modes = {
                    "0": ("rgb_array", "RGB array (no render)"),
                    "1": ("human", "Human"),
                }
                for key, (_, description) in render_modes.items():
                    cli_printer.log(f"  - {key}: {Style.YELLOW}{description}")

                value = cli_printer.input("Choose new render mode: ")

                if value in render_modes:
                    args.render_mode = render_modes[value][0]
                else:
                    cli_printer.error("Invalid render choice.")
                    continue
            case "x":
                break
            case _:
                cli_printer.error("Invalid command.")


def print_advanced_args(args: Namespace) -> None:
    cli_printer.log("\nAdvanced arguments:", style=Style.CYAN + Style.UNDERLINE)

    advanced_dict = {
        "  - Random seed": str(args.seed),
        "  - Save path": str(args.save_path),
        "  - Load path": str(args.load_path),
        "  - Save frequency": str(args.save_freq),
        "  - Log frequency": str(args.log_freq),
        "  - Tensorboard": bool_state[args.tensorboard],
        "  - Pretrain vision": bool_state[args.pretrain_vision],
        "  - Pretrain memory": bool_state[args.pretrain_memory],
        "  - Pretraining mode": str(args.pretrain_mode),
    }

    cli_printer.dict_log(advanced_dict)

    pretraining = args.pretrain_vision or args.pretrain_memory
    state = (
        "Training"
        if not pretraining
        else "Pretraining "
        + " and ".join(["Vision", "Memory"][1 - args.pretrain_vision : 1 + args.pretrain_memory])
    )
    cli_printer.log(f"\n  State: {Style.MAGENTA}{state}", style=Style.CYAN)


def edit_advanced_args(args: Namespace) -> None:
    print_separator()

    while True:
        cli_printer.log("\nEditing advanced arguments:", style=Style.CYAN + Style.INVERT)
        advanced_edit_dict = {
            "  - 0: Seed": str(args.seed),
            "  - 1: Save path": str(args.save_path),
            "  - 2: Load path": str(args.load_path),
            "  - 3: Save frequency": str(args.save_freq),
            "  - 4: Log frequency": str(args.log_freq),
            "  - 5: Tensorboard": bool_state[args.tensorboard],
            "  - 6: Pretrain vision": bool_state[args.pretrain_vision],
            "  - 7: Pretrain memory": bool_state[args.pretrain_memory],
            "  - 8: Pretraining mode": str(args.pretrain_mode),
            "  - x: : Back ": "",
        }
        cli_printer.dict_log(advanced_edit_dict)

        command = cli_printer.input("What to edit ? ")

        match command:
            case "0":
                new_seed = cli_printer.input("Enter new seed: ")
                if new_seed.isdigit():
                    args.random_seed = int(new_seed)
                else:
                    cli_printer.error("Invalid seed value.")
                    continue

            # No path autocompletion to not add dependencies for now. Only checking if paths exist.
            case "1":
                save_path = cli_printer.input("Enter path to save model: ")
                if save_path and os.path.exists(save_path):
                    args.save_path = save_path
                else:
                    cli_printer.error("Invalid path.")
                    continue
            case "2":
                load_path = cli_printer.input("Enter path to load model: ")
                if load_path and os.path.exists(load_path):
                    args.load_path = load_path
                else:
                    cli_printer.error("Invalid path.")
                    continue
            case "3":
                save_freq = cli_printer.input("Enter save frequency: ")
                if save_freq.isdigit() and int(save_freq) >= 0:
                    args.save_freq = int(save_freq)
                else:
                    cli_printer.error("Invalid value.")
                    continue
            case "4":
                log_freq = cli_printer.input("Enter log frequency: ")
                if log_freq.isdigit() and int(log_freq) >= 0:
                    args.log_freq = int(log_freq)
                else:
                    cli_printer.error("Invalid value.")
                    continue
            case "5":
                use_tensorboard = cli_printer.input("Use tensorboard? (y/n): ")
                if use_tensorboard.lower() == "y":
                    args.tensorboard = True
                elif use_tensorboard.lower() == "n":
                    args.tensorboard = False
                else:
                    cli_printer.error("Invalid input.")
                    continue
            case "6":
                pretrain_vision = cli_printer.input("Pretrain vision model? (y/n): ")
                if pretrain_vision.lower() == "y":
                    args.pretrain_vision = True
                elif pretrain_vision.lower() == "n":
                    args.pretrain_vision = False
                else:
                    cli_printer.error("Invalid input.")
                    continue
            case "7":
                pretrain_memory = cli_printer.input("Pretrain memory model? (y/n): ")
                if pretrain_memory.lower() == "y":
                    args.pretrain_memory = True
                elif pretrain_memory.lower() == "n":
                    args.pretrain_memory = False
                else:
                    cli_printer.error("Invalid input.")
                    continue
            case "8":
                pretrain_dict = {"  - 0: ": "Random", "  - 1: ": "Manual"}
                cli_printer.dict_log(pretrain_dict)
                value = cli_printer.input("Choose new pretraining mode: ")
                modes = ["random", "manual"]
                if value.isdigit() and 0 <= int(value) < len(modes):
                    args.pretrain_mode = modes[int(value)]
                else:
                    cli_printer.error("Invalid mode choice.")
                    continue
            case "x":
                break
            case _:
                cli_printer.error("Invalid command.")


def edit_ppo_args(args: Namespace) -> None:
    print_separator()

    while True:
        cli_printer.log("\nEditing PPO arguments:", style=Style.CYAN + Style.INVERT)
        cli_printer.log("\nAvailable commands:", style=Style.CYAN + Style.UNDERLINE)
        ppo_edit_dict = {
            "  - 0: Rollout steps": str(args.rollout_steps),
            "  - 1: Epochs": str(args.ppo_epochs),
            "  - 2: Learning rate": str(args.ppo_lr),
            "  - 3: Batch size": str(args.ppo_batch_size),
            "  - 4: Clip range": str(args.ppo_clip_range),
            "  - 5: Clip range for value function": str(args.ppo_range_vf),
            "  - 6: Gamma": str(args.gamma),
            "  - 7: GAE lambda": str(args.gae_lambda),
            "  - 8: Value loss coefficient": str(args.value_coef),
            "  - 9: Entropy coefficient": str(args.entropy_coef),
            "  - 10: Maximum gradient norm": str(args.max_grad_norm),
            "  - 11: Train world model (vision/memory)": bool_state[not args.no_train_world_model],
            "  - 12: World model epochs": str(args.world_model_epochs),
            "  - x: : Back ": "",
        }
        cli_printer.dict_log(ppo_edit_dict)
        command = cli_printer.input("What to edit ? ")

        match command:
            case "0" | "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9" | "10":
                edit_choice = int(command)
                value_str = cli_printer.input(
                    f"Enter new value for {['number of rollout steps', 'epochs', 'learning rate', 'batch size', 'clip range', 'clip range for value function', 'gamma', 'GAE lambda', 'value loss coefficient', 'entropy coefficient', 'maximum gradient norm'][edit_choice]}: "
                )
                fields = [
                    "rollout_steps",
                    "ppo_epochs",
                    "ppo_lr",
                    "ppo_batch_size",
                    "ppo_clip_range",
                    "ppo_range_vf",
                    "gamma",
                    "gae_lambda",
                    "value_coef",
                    "entropy_coef",
                    "max_grad_norm",
                ]
                try:
                    casters: list[Callable[[str], int | float]] = [
                        int,
                        int,
                        float,
                        int,
                        float,
                        float,
                        float,
                        float,
                        float,
                        float,
                        float,
                    ]
                    value = casters[edit_choice](value_str)
                    if value < 0:
                        cli_printer.error("Negative value.")
                        continue
                    setattr(args, fields[edit_choice], value)
                except ValueError:
                    cli_printer.error("Invalid value.")
                    continue
            case "11":
                train_world_model = cli_printer.input(
                    "Train world model (vision and memory)? (y/n): "
                )
                if train_world_model.lower() == "y":
                    args.no_train_world_model = True
                elif train_world_model.lower() == "n":
                    args.no_train_world_model = False
                else:
                    cli_printer.error("Invalid input.")
                    continue
            case "12":
                world_epoch = cli_printer.input("Enter number of epochs for world model: ")
                if world_epoch.isdigit() and int(world_epoch) >= 0:
                    args.world_model_epochs = int(world_epoch)
                else:
                    cli_printer.error("Invalid value.")
                    continue
            case "exit" | "x":
                break
            case _:
                cli_printer.error("Invalid command.")


def CLI(
    args: Namespace, vision_registry: dict, memory_registry: dict, controller_registry: dict
) -> None:
    cli_printer.log("Welcome to the Enhanced World Model CLI!", style=Style.GREEN + Style.BOLD)
    while True:
        print_separator()
        print_main_args(args, vision_registry, memory_registry, controller_registry)
        print_advanced_args(args)

        cli_printer.log("\nAvailable commands:", style=Style.CYAN + Style.UNDERLINE)
        command_dict = {
            "  - 0": "Run",
            "  - 1": "Edit main arguments",
            "  - 2": "Edit advanced arguments",
            "  - 3": "See PPO arguments.",
            "  - 7": "Save args",
            "  - 8": "Load saved args",
            "  - exit": "Exit",
        }
        cli_printer.dict_log(command_dict, value_style=Style.MAGENTA)

        command = cli_printer.input("Enter command: ")
        match command:
            case "0":
                break
            case "1":
                edit_main_args(args, vision_registry, memory_registry, controller_registry)
            case "2":
                edit_advanced_args(args)
            case "3":
                edit_ppo_args(args)
            case "7":
                save_args(args)
            case "8":
                load_args(args)
            case "exit" | "x":
                exit(0)
            case _:
                cli_printer.error("Invalid command.")
