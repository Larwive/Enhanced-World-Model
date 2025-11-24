import torch
import numpy as np
import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard.summary import hparams
from torch.nn import MSELoss
from time import sleep

from WorldModel import WorldModel
from Model import Model
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
        state_transposed = np.transpose(state, (2, 0, 1))
        state_tensor = torch.from_numpy(state_transposed).float().unsqueeze(0).to(device)
        # Normalize image data to [0, 1]
        return state_tensor / 255.0
    else:
        # For vector data, just add batch dimension and move to device
        return torch.from_numpy(state).float().unsqueeze(0).to(device)


def step(model,
         state,
         env,
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

    output_dict = model(state_tensor, return_losses=True, action_space=action_space)
    total_loss = torch.sum(output_dict["total_loss"])

    if mode == "random":
        action = env.action_space.sample()
    elif mode == "manual":
        sleep(delay)
        action, restart, quit = register_input(env)

    new_state, _, terminated, truncated, info = env.step(action)
    done = terminated or truncated

    if pretrain_memory:
       total_loss = total_loss + torch.abs(model.vision.encode(state_transform(new_state, is_image_based=is_image_based, device=device)).detach() - output_dict["memory_prediction"]).mean()

    if pretrain_vision or (iter_num - 1): # Testing if only memory is being pretrained.
        total_loss.backward()

    if tensorboard_writer is not None:
        tensorboard_writer.add_scalar("train/loss", total_loss.item(), iter_num)
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                tensorboard_writer.add_scalar(f"gradients/{name}", param.grad.norm().item(), iter_num)

    optimizer.step()

    return total_loss.item(), new_state, done


def pretrain(model: WorldModel, env, max_iter=10000, device:torch.device=torch.device('cpu'), use_tensorboard: bool = True, learning_rate: float = 0.01, loss_func: callable=MSELoss, mode:str="random", delay:float=0.2, save_path="./", save_prefix="", pretrain_vision: bool=False, pretrain_memory: bool=False):
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    loss_func = loss_func()  # Add potential args here.
    is_image_based = len(env.observation_space.shape) == 3
    action_space = env.action_space

    writer = HyperSummaryWriter() if use_tensorboard else None

    model.iter_num += 1

    best_loss = torch.inf
    last_save = model.iter_num
    for experiment_index in range(model.nb_experiments + 1, model.nb_experiments + max_iter + 1):
        state, info = env.reset()
        done = False
        total_episode_loss = 0

        local_iter_num = 0
        while not done:
            loss, state, done = step(model,
                                       state,
                                       env,
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

            total_episode_loss += loss
            if env.render_mode == 'human':
                env.render()

            model.iter_num += 1
            model.nb_experiments += 1
            local_iter_num += 1

            if loss < best_loss and model.iter_num > last_save + 5:
                best_loss = loss
                last_save = model.iter_num
                model.save(f"{save_path}pretrained_{save_prefix}_{env.spec.id}.pt", env.observation_space, env.action_space)
        
        print(f"Experiment {experiment_index}\nIteration {local_iter_num + 1}, Mean Loss: {total_episode_loss/(local_iter_num + 1):.4f}")
