import torch
import numpy as np
import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard.summary import hparams
from torch.nn import MSELoss
from time import sleep

from interface.interface import GymEnvInterface
from WorldModel import WorldModel
from Model import Model
from manual_control import register_input

class SummaryWriter(SummaryWriter):
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


def step(model,
         state,
         interface,
         optimizer,
         device,
         is_image_based,
         action_space,
         cumulated_reward,
         iter_num: int = 0,
         tensorboard_writer=None,
         loss_instance=None,
         mode:str="random",
         delay:float=0.2):
    if loss_instance is None:
        loss_instance = MSELoss()  # Same as below.
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

    last_reward = torch.tensor([cumulated_reward], dtype=torch.get_default_dtype())
    output_dict = model(state_tensor, return_losses=True, action_space=action_space, last_reward=cumulated_reward)
    total_loss = torch.sum(output_dict["total_loss"])


    # Handle different action spaces
    if mode == "random":
        action = interface.env.action_space.sample()
    elif mode == "manual":
        sleep(delay)
        action, restart, quit = register_input(interface)
        
    if isinstance(action_space, gym.spaces.Discrete):
        raw_action_np = np.int64(np.round(output_dict["action"].detach().cpu().numpy()))
        action_np = np.clip(raw_action_np, 0, action_space.n - 1)  # Not using `describe_action_space`
    else:  # Continuous action space
        action_np = output_dict["action"].squeeze(0).cpu().detach().numpy()

    new_state, step_reward, done, info = interface.step(action)

    if done:
        cumulated_reward = torch.tensor([step_reward], dtype=torch.get_default_dtype())
    else:
        cumulated_reward += step_reward


    total_loss = total_loss
    #total_loss = torch.abs(total_loss)
    total_loss.backward()

    if tensorboard_writer is not None:
        tensorboard_writer.add_scalar("train/loss", total_loss.item(), iter_num)
        for name, param in model.named_parameters():
            #if iter_num and torch.isclose(torch.zeros_like(param.grad.norm()), param.grad.norm()): # Will ideally be removed in the future.
            #    print("{}'s gradient is low ! ({})".format(name, param.grad.norm().item()))
            if param.requires_grad:
                tensorboard_writer.add_scalar(f"gradients/{name}", param.grad.norm().item(), iter_num)

    optimizer.step()

    return cumulated_reward, total_loss.item(), new_state, done


def pretrain(model: WorldModel, interface: GymEnvInterface, max_iter=10000, device='cpu', use_tensorboard: bool = True, learning_rate: float = 0.01, loss_func: callable=MSELoss, mode:str="random", delay:float=0.2, save_path="./"):
    # print(model.parameters)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    loss_func = loss_func()  # Add potential args here.
    is_image_based = len(interface.env.observation_space.shape) == 3
    action_space = interface.env.action_space

    writer = SummaryWriter() if use_tensorboard else None

    cumulated_reward = torch.tensor([0], dtype=torch.get_default_dtype())
    model.iter_num += 1

    best_loss = torch.inf
    last_save = model.iter_num
    for experiment_index in range(model.nb_experiments + 1, model.nb_experiments + max_iter + 1):
        state, info = interface.reset()
        done = False
        total_episode_loss = 0

        local_iter_num = 0
        while not done:
            cumulated_reward, loss, state, done = step(model,
                                       state,
                                       interface,
                                       optimizer,
                                       device,
                                       is_image_based,
                                       action_space,
                                       cumulated_reward,
                                       iter_num=model.iter_num,
                                       tensorboard_writer=writer,
                                       loss_instance=loss_func,
                                       mode=mode,
                                       delay=delay)

            total_episode_loss += loss
            if interface.env.render_mode == 'human':
                interface.render()
            model.iter_num += 1
            local_iter_num += 1
            if loss < best_loss and model.iter_num > last_save + 5:
                best_loss = loss
                last_save = model.iter_num
                model.save(f"{save_path}pretrained_V_{interface.env.spec.id}.pt", interface.env.observation_space, interface.env.action_space)
        
        print(f"Experiment {experiment_index}\nIteration {local_iter_num + 1}, Mean Loss: {total_episode_loss/(local_iter_num + 1):.4f}")
    model.nb_experiments = experiment_index
