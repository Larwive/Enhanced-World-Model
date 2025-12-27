from collections.abc import Callable

from argparse import Namespace
import os
import json

from utils.colors import Color
from utils.gym_tools import get_all_gym_envs, gym_is_image_based, action_space_is_discrete


def save_args(args: Namespace) -> None:
    with open("cli.json", "w") as f:
        json.dump(vars(args), f, indent=4)
    print(f"{Color.GREEN}Arguments saved successfully.{Color.RESET}")


def load_args(args: Namespace) -> None:
    if not os.path.exists("cli.json"):
        print(f"{Color.RED}No saved arguments found. Save args before loading.{Color.RESET}")
        return
    with open("cli.json") as f:
        args_dict = json.load(f)
    args.__dict__.update(args_dict)
    print(f"{Color.GREEN}Arguments loaded successfully.{Color.RESET}")


def print_separator() -> None:
    print(f"{Color.YELLOW}------------------------------------------------{Color.RESET}")


def print_main_args(
    args: Namespace,
    vision_registry: dict,
    _memory_registry: dict,
    controller_registry: dict,
    _reward_predictor_registry: dict,
) -> None:
    print_separator()
    print(f"{Color.YELLOW}Main arguments:{Color.RESET}")
    print(f"{Color.CYAN}  Env: {Color.BLUE}{args.env_name}{Color.RESET}")

    is_image_based = gym_is_image_based(args.env_name)
    if is_image_based and "image_based" not in vision_registry[args.vision].tags:
        vision_warning = (
            f"{Color.RED} Warning: The selected vision model is not image-based.{Color.RESET}"
        )
    elif not is_image_based and "image_based" in vision_registry[args.vision].tags:
        vision_warning = (
            f"{Color.RED} Warning: The selected vision model is image-based.{Color.RESET}"
        )
    else:
        vision_warning = ""

    print(f"{Color.CYAN}  - Vision    : {Color.YELLOW}{args.vision}{Color.RESET}{vision_warning}")
    print(f"{Color.CYAN}  - Memory    : {Color.YELLOW}{args.memory}{Color.RESET}")

    is_discrete = action_space_is_discrete(args.env_name)
    if is_discrete and "continuous" in controller_registry[args.controller].tags:
        controller_warning = (
            f"{Color.RED} Warning: The selected controller is continuous.{Color.RESET}"
        )
    elif not is_discrete and "discrete" in controller_registry[args.controller].tags:
        controller_warning = (
            f"{Color.RED} Warning: The selected controller is discrete.{Color.RESET}"
        )
    else:
        controller_warning = ""

    print(
        f"{Color.CYAN}  - Controller: {Color.YELLOW}{args.controller}{Color.RESET}{controller_warning}"
    )
    print(f"{Color.CYAN}  - Reward predictor: {Color.YELLOW}{args.reward_predictor}{Color.RESET}")
    print(f"{Color.CYAN}  - Learning rate: {Color.YELLOW}{args.learning_rate}{Color.RESET}")
    print(f"{Color.CYAN}  - Batch size: {Color.YELLOW}{args.env_batch_number}{Color.RESET}")
    print(f"{Color.CYAN}  - Epochs: {Color.YELLOW}{args.max_epoch}{Color.RESET}")
    print(f"{Color.CYAN}  - Render mode: {Color.YELLOW}{args.render_mode}{Color.RESET}")

    if args.load_path:
        print(f"{Color.RED} Model choices will be overriden by model to load !{Color.RESET}")


def edit_main_args(
    args: Namespace,
    vision_registry: dict,
    memory_registry: dict,
    controller_registry: dict,
    reward_predictor_registry: dict,
) -> None:
    print_separator()

    while True:
        print(f"{Color.YELLOW}Editing main configuration:{Color.RESET}")
        print(f"{Color.CYAN}  - 0: Environment: {Color.YELLOW}{args.env_name}{Color.RESET}")
        print(f"{Color.CYAN}  - 1: Vision: {Color.YELLOW}{args.vision}{Color.RESET}")
        print(f"{Color.CYAN}  - 2: Memory: {Color.YELLOW}{args.memory}{Color.RESET}")
        print(f"{Color.CYAN}  - 3: Controller: {Color.YELLOW}{args.controller}{Color.RESET}")
        print(
            f"{Color.CYAN}  - 4: Reward predictor: {Color.YELLOW}{args.reward_predictor}{Color.RESET}"
        )
        print(f"{Color.CYAN}  - 5: Learning rate: {Color.YELLOW}{args.learning_rate}{Color.RESET}")
        print(f"{Color.CYAN}  - 6: Batch size: {Color.YELLOW}{args.env_batch_number}{Color.RESET}")
        print(f"{Color.CYAN}  - 7: Epochs: {Color.YELLOW}{args.max_epoch}{Color.RESET}")
        print(f"{Color.CYAN}  - 8: Render mode: {Color.YELLOW}{args.render_mode}{Color.RESET}")
        print(f"{Color.CYAN}  - 9: Back{Color.RESET}")

        command = input(f"{Color.GREEN}What to edit ? {Color.RESET} ")
        if not command.isdigit():
            print(f"{Color.RED}Invalid command{Color.RESET}")
            continue
        edit_choice = int(command)

        if edit_choice == 0:
            env_names = get_all_gym_envs()
            env_listing = "\n".join(
                f"{Color.CYAN}{i}: {Color.MAGENTA}{name}{Color.RESET}"
                for i, name in enumerate(env_names)
            )
            print(f"{Color.CYAN}  Available envs: {Color.YELLOW}{env_listing}{Color.RESET}")
            env_id = input(f"{Color.GREEN}Enter environment id: {Color.RESET}")
            if env_id.isdigit() and 0 <= int(env_id) < len(env_names):
                args.env_name = env_names[int(env_id)]
            else:
                print(f"{Color.RED}Invalid environment choice.{Color.RESET}")
                continue

        elif 0 < edit_choice < 5:
            registry = [
                vision_registry,
                memory_registry,
                controller_registry,
                reward_predictor_registry,
            ][edit_choice - 1]
            models = "\n".join(
                f"{Color.CYAN}{i}: {Color.MAGENTA}{name}{Color.RESET} ([{Color.BLUE}{', '.join(cls.tags)}{Color.RESET}])"
                for i, (name, cls) in enumerate(registry.items())
            )
            model_choice = input(
                f"Choose the model.\n{models}\n{Color.GREEN}Model number: {Color.RESET}"
            )
            if not model_choice.isdigit():
                print(f"{Color.RED}Invalid model choice.{Color.RESET}")
                continue
            model_id_choice = int(model_choice)
            if 0 <= model_id_choice < len(registry):
                fields = ["vision", "memory", "controller", "reward_predictor"]

                setattr(args, fields[edit_choice - 1], list(registry.keys())[model_id_choice])
            else:
                print(
                    f"{Color.RED}Invalid model choice. Enter a number between 0 and {len(registry) - 1}.{Color.RESET}"
                )
                continue

        elif 4 < edit_choice < 8:
            value = input(
                f"{Color.GREEN}Enter new value for {['learning rate', 'batch size', 'number of epochs'][edit_choice - 5]}: {Color.RESET}"
            )
            if value == "auto" and edit_choice == 6:
                args.env_batch_number = "auto"
                continue
            fields = ["learning_rate", "env_batch_number", "max_epoch"]
            try:

                def str_int(x: str) -> str:
                    return str(int(x))

                casters: list[Callable[[str], object]] = [float, str_int, int]
                setattr(args, fields[edit_choice - 5], casters[edit_choice - 5](value))
            except ValueError:
                print(f"{Color.RED}Invalid value{Color.RESET}")
                continue
        elif edit_choice == 8:
            print(f"{Color.CYAN}  - 0: {Color.YELLOW}RGB array (no render){Color.RESET}")
            print(f"{Color.CYAN}  - 1: {Color.YELLOW}Human{Color.RESET}")
            value = input(f"{Color.GREEN}Choose new render mode: {Color.RESET}")
            render_choices = ["rgb_array", "human"]
            if value.isdigit() and 0 <= int(value) < len(render_choices):
                args.render_mode = render_choices[int(value)]
            else:
                print(f"{Color.RED}Invalid render choice.{Color.RESET}")
                continue

        elif edit_choice == 9:
            break
        else:
            print(f"{Color.RED}Invalid command{Color.RESET}")


def print_advanced_args(args: Namespace) -> None:
    print(f"{Color.YELLOW}Advanced arguments:{Color.RESET}")
    print(f"{Color.CYAN}  - Random seed: {Color.YELLOW}{args.random_seed}{Color.RESET}")
    print(f"{Color.CYAN}  - Save path: {Color.YELLOW}{args.save_path}{Color.RESET}")
    print(f"{Color.CYAN}  - Load path: {Color.YELLOW}{args.load_path}{Color.RESET}")
    print(f"{Color.CYAN}  - GPU: {Color.YELLOW}{args.gpu}{Color.RESET}")
    pretraining = args.pretrain_vision or args.pretrain_memory
    state = (
        "Training"
        if not pretraining
        else "Pretraining "
        + "and ".join(["Vision", "Memory"][1 - args.pretrain_vision : 1 + args.pretrain_memory])
    )
    print(f"\n{Color.CYAN}  State: {Color.MAGENTA}{state}{Color.RESET}")


def edit_advanced_args(args: Namespace) -> None:
    print_separator()

    while True:
        print(f"{Color.YELLOW}Editing advanced arguments:{Color.RESET}")
        print(f"{Color.CYAN}  - 0: Random seed: {Color.YELLOW}{args.random_seed}{Color.RESET}")
        print(f"{Color.CYAN}  - 1: Save path: {Color.YELLOW}{args.save_path}{Color.RESET}")
        print(f"{Color.CYAN}  - 2: Load path: {Color.YELLOW}{args.load_path}{Color.RESET}")
        print(f"{Color.CYAN}  - 3: GPU: {Color.YELLOW}{args.gpu}{Color.RESET}")
        print(
            f"{Color.CYAN}  - 4: Pretrain vision: {Color.YELLOW}{args.pretrain_vision}{Color.RESET}"
        )
        print(
            f"{Color.CYAN}  - 5: Pretrain memory: {Color.YELLOW}{args.pretrain_memory}{Color.RESET}"
        )
        print(
            f"{Color.CYAN}  - 6: Pretraining mode: {Color.YELLOW}{args.pretrain_mode}{Color.RESET}"
        )
        print(f"{Color.CYAN}  - 9: Back{Color.RESET}")

        command = input(f"{Color.GREEN}What to edit ? {Color.RESET}")
        if not command.isdigit():
            print(f"{Color.RED}Invalid command{Color.RESET}")
            continue
        edit_choice = int(command)

        if edit_choice == 0:
            new_seed = input(f"{Color.GREEN}Enter new random seed: {Color.RESET}")
            if new_seed.isdigit():
                args.random_seed = int(new_seed)
            else:
                print(f"{Color.RED}Invalid seed{Color.RESET}")
                continue

        # No path autocompletion to not add dependencies for now. Only checking if paths exist.
        elif edit_choice == 1:
            save_path = input(f"{Color.GREEN}Enter path to save model: {Color.RESET}")
            if save_path and os.path.exists(save_path):
                args.save_path = save_path
            else:
                print(f"{Color.RED}Invalid path{Color.RESET}")
                continue
        elif edit_choice == 2:
            load_path = input(f"{Color.GREEN}Enter path to load model: {Color.RESET}")
            if load_path and os.path.exists(load_path):
                args.load_path = load_path
            else:
                print(f"{Color.RED}Invalid path{Color.RESET}")
                continue
        elif edit_choice == 3:
            gpu_id = input(f"{Color.GREEN}Enter GPU ID: {Color.RESET}")
            if gpu_id.isdigit() and int(gpu_id) >= 0:
                args.gpu = int(gpu_id)
            else:
                print(f"{Color.RED}Invalid GPU ID.{Color.RESET}")
                continue
        elif edit_choice == 4:
            pretrain_vision = input(f"{Color.GREEN}Pretrain vision model? (y/n): {Color.RESET}")
            if pretrain_vision.lower() == "y":
                args.pretrain_vision = True
            elif pretrain_vision.lower() == "n":
                args.pretrain_vision = False
            else:
                print(f"{Color.RED}Invalid input.{Color.RESET}")
                continue
        elif edit_choice == 5:
            pretrain_memory = input(f"{Color.GREEN}Pretrain memory model? (y/n): {Color.RESET}")
            if pretrain_memory.lower() == "y":
                args.pretrain_memory = True
            elif pretrain_memory.lower() == "n":
                args.pretrain_memory = False
            else:
                print(f"{Color.RED}Invalid input.{Color.RESET}")
                continue
        elif edit_choice == 6:
            print(f"{Color.CYAN}  - 0: {Color.YELLOW}Random{Color.RESET}")
            print(f"{Color.CYAN}  - 1: {Color.YELLOW}Manual{Color.RESET}")
            value = input(f"{Color.GREEN}Choose new pretraining mode: {Color.RESET}")
            modes = ["random", "manual"]
            if value.isdigit() and 0 <= int(value) < len(modes):
                args.pretrain_mode = modes[int(value)]
            else:
                print(f"{Color.RED}Invalid mode choice.{Color.RESET}")
                continue
        elif edit_choice == 9:
            break
        else:
            print(f"{Color.RED}Invalid command{Color.RESET}")


def CLI(
    args: Namespace,
    vision_registry: dict,
    memory_registry: dict,
    controller_registry: dict,
    reward_predictor_registry: dict,
) -> None:
    print(f"{Color.GREEN}Welcome to the Enhanced World Model CLI!{Color.RESET}")
    while True:
        print_separator()
        print_main_args(
            args, vision_registry, memory_registry, controller_registry, reward_predictor_registry
        )
        print_advanced_args(args)

        print(f"\n{Color.YELLOW}Available commands:{Color.RESET}")
        print(f"{Color.CYAN}  - 0: Run{Color.RESET}")
        print(f"{Color.CYAN}  - 1: Edit main arguments{Color.RESET}")
        print(f"{Color.CYAN}  - 2: Edit advanced arguments{Color.RESET}")
        print(f"{Color.CYAN}  - 7: Save args{Color.RESET}")
        print(f"{Color.CYAN}  - 8: Load saved args{Color.RESET}")
        print(f"{Color.CYAN}  - 9: Exit{Color.RESET}")
        command = input(f"{Color.GREEN}Enter command:{Color.RESET} ")
        if command == "0":
            break
        if command == "1":
            edit_main_args(
                args,
                vision_registry,
                memory_registry,
                controller_registry,
                reward_predictor_registry,
            )
        elif command == "2":
            edit_advanced_args(args)
        elif command == "7":
            save_args(args)
        elif command == "8":
            load_args(args)
        elif command == "9":
            exit(0)
        else:
            print(f"{Color.RED}Invalid command{Color.RESET}")
