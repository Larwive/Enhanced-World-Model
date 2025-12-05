import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard.summary import hparams
from torch.nn import MSELoss
from time import sleep

from WorldModel import WorldModel, render_first_env
from manual_control import register_input

class HyperSummaryWriter(SummaryWriter):
    """
    Add possiblity to store hyperparameters.
    """

    def add_hparams(self, hparam_dict, metric_dict):
        torch._C._log_api_usage_once("tensorboard.logging.add_hparams")
        if type(hparam_dict) is not dict or type(metric_dict) is not dict:
            raise TypeError('hparam_dict and metric_dict should be dictionary.')
        exp, ssi, sei = hparams(hparam_dict, metric_dict)

        logdir = self._get_file_writer().get_logdir()

        with SummaryWriter(log_dir=logdir) as w_hp:
            w_hp.file_writer.add_summary(exp)
            w_hp.file_writer.add_summary(ssi)
            w_hp.file_writer.add_summary(sei)
            for k, v in metric_dict.items():
                w_hp.add_scalar(k, v)

def state_transform(state, is_image_based, device):
    if is_image_based:
        # Transpose state from (H, W, C) to (C, H, W) for PyTorch
        if state.ndim == 3:
            state = state[None]
        state_transposed = np.transpose(state, (0, 3, 1, 2))
        state_tensor = torch.from_numpy(state_transposed).float().to(device)
        # Normalize image data to [0, 1]
        return state_tensor / 255.0
    else:
        if state.ndim == 1:
            state = state[None]
        return torch.from_numpy(state).float().to(device)

def step(model,
         state,
         envs,
         optimizer,
         device,
         is_image_based,
         action_space,
         iter_num: int = 0,
         tensorboard_writer=None,
         loss_instance=None,
         mode:str="random",
         delay:float=0.2,
         pretrain_vision: bool=False,
         pretrain_memory: bool=False):

    if loss_instance is None:
        loss_instance = MSELoss()  # Same as below.
    optimizer.zero_grad()

    state_tensor = state_transform(state, is_image_based=is_image_based, device=device)

    output_dict = model(state_tensor, action_space=action_space, is_image_based=is_image_based, return_losses=True)
    total_loss = output_dict["total_loss"]

    if mode == "random" or envs.num_envs > 1: # Forbidding manual mode if multiple environments. TODO: Allow manual mode in first env.
        actions = np.stack([envs.single_action_space.sample() for _ in range(envs.num_envs)])
    elif mode == "manual":
        sleep(delay)
        actions, restart, quit = register_input(envs)

    new_state, _, terminated, truncated, info = envs.step(actions)
    dones = torch.from_numpy(terminated | truncated).to(device=device, dtype=torch.bool)

    if pretrain_memory:
       vision_encoded = model.vision.encode(state_transform(new_state, is_image_based=is_image_based, device=device), is_image_based=is_image_based)
       if is_image_based:
            vision_encoded = vision_encoded.mean(dim=(2, 3))
       total_loss = total_loss + torch.nn.functional.mse_loss(vision_encoded, output_dict["memory_prediction"])

    if pretrain_vision or pretrain_memory: #if pretrain_vision or (iter_num - 1): # Testing if only memory is being pretrained.
        total_loss.mean().backward()

    if tensorboard_writer is not None:
        tensorboard_writer.add_scalar("train/loss_mean", total_loss.mean().item(), iter_num)
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                tensorboard_writer.add_scalar(f"gradients/{name}", param.grad.norm().item(), iter_num)
                

    optimizer.step()

    return total_loss.detach(), new_state, dones

# TODO: Remove render_env arg when rendering of the first env is not done through cv2 anymore.
def pretrain(model: WorldModel, envs, max_iter=10000, device:torch.device=torch.device('cpu'), use_tensorboard: bool = True, learning_rate: float = 0.01, loss_func: callable=MSELoss, mode:str="random", delay:float=0.2, save_path="./", save_prefix="", pretrain_vision: bool=False, pretrain_memory: bool=False, render_mode:str=""):
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    loss_func = loss_func()  # Add potential args here.
    is_image_based = len(envs.single_observation_space.shape) == 3
    action_space = envs.single_action_space

    writer = HyperSummaryWriter() if use_tensorboard else None

    model.iter_num += 1

    best_loss = torch.inf
    last_save = model.iter_num
    nb_experiments = 0
    state, info = envs.reset()
    local_iter_num = torch.zeros(envs.num_envs)
    total_episode_loss = torch.zeros(envs.num_envs)
    while nb_experiments < max_iter:
        loss, state, dones = step(model,
                                    state,
                                    envs,
                                    optimizer,
                                    device,
                                    is_image_based,
                                    action_space,
                                    iter_num=model.iter_num,
                                    tensorboard_writer=writer,
                                    loss_instance=loss_func,
                                    mode=mode,
                                    delay=delay,
                                    pretrain_vision=pretrain_vision,
                                    pretrain_memory=pretrain_memory)
        # print(loss)
        total_episode_loss += loss
        if render_mode  == 'human':
            render_first_env(envs)

        model.iter_num += envs.num_envs
        local_iter_num += 1

        if loss < best_loss and model.iter_num > last_save + 5:
            best_loss = loss
            last_save = model.iter_num
            model.save(f"{save_path}pretrained_{save_prefix}_{envs.spec.id}.pt", envs.single_observation_space, envs.single_action_space)
        
        finished_envs = torch.where(dones)[0]
        for env_id in finished_envs:
            print(f"Experiment {model.nb_experiments + 1}\nEnded at iteration {local_iter_num[env_id]}, Mean Loss: {total_episode_loss[env_id]/local_iter_num[env_id]:.4f}")
            model.nb_experiments += 1
            local_iter_num[env_id] = 0
            total_episode_loss[env_id] = 0
            nb_experiments += 1
            model.reset_env_memory(env_id)

