# Enhanced-World-Model
An extensible and modular reimplementation of Ha & Schmidhuber’s
*Recurrent World Models Facilitate Policy Evolution (2018)*.

This project provides a flexible framework for experimenting with
modern world-model architectures in reinforcement learning. It enables
researchers to easily swap vision, memory, and controller components
while preserving a unified training pipeline.

## Architecture Overview

The enhanced world model is composed of three interchangeable modules:

1. **Vision Model** – Encodes observations into latent representations
2. **Memory Model** – Models temporal dynamics in latent space
3. **Controller** – Produces actions based on latent states

Each component can be independently selected and trained, enabling
rapid experimentation with different architectures.

## Features

- **Modular design** allowing independent vision, memory and controller components
- **Flexible training pipeline** supporting multiple environments and
  model architectures
- **Reproducible experiments** through explicit configuration and
  seeding

This modular design enables rapid experimentation with different world-model configurations.

## Environments

The framework targets compatibility with environments provided by
**Gymnasium**.
https://gymnasium.farama.org

Currently tested environments:
- CartPole-v1
- CarRacing-v3

## Requirements

- Python **3.11 – 3.12.11**
- NVIDIA GPU recommended for vision-based environments

## Installation
This project uses **uv** for dependency management.

- Clone the repository
  ```$ git clone https://github.com/Larwive/Enhanced-World-Model/```
- Install dependencies
  ```$ uv sync```

## Training

To train or pretrain models, run `src/main.py`.

The accepted arguments are:
### Interface
- `--ui` to use the Gradio web interface. Not compatible with `--cli`.
- `--cli` to use the CLI. Not compatible with `--ui`.

### Environment & Training
- `--env` to set the environment to use.
- `--seed` to set the seed.
- `--epochs` to set the number of epochs to run.
- `--patience` to set the number of iterations without noticeable improvement before early stopping.
- `--batch-size` to set the number of environments to run in parallel. Can be set automatically with `auto`.
- `--lr` the learning rate for the model, except the controller model.
- `--dropout` to control the dropout rate.
- `--render-mode` to set render mode between `human` and `rgb_array` (no render).
- `--save-path` for the path to save the model.
- `--load-path` to load an existing model.
- `--vision` to set the vision model to use. Loading an existing model will overwrite this argument.
- `--memory` to set the memory model to use. Loading an existing model will overwrite this argument.
- `--controller` to set the controller model to use. Loading an existing model will overwrite this argument.
- `--save-freq` the number of epochs between each save.
- `--log-freq` the number of epochs between each log.
- `--tensorboard` whether to log gradients and losses into TensorBoard.

## Pretraining

Pretraining allows the vision and memory models to be trained
independently before full reinforcement learning.

During pretraining:
- The controller is bypassed
- Actions are either random or manually provided
- The resulting model can later be loaded for full training

Pretraining-specific arguments:
- `--pretrain-vision`
- `--pretrain-memory`
- `--pretrain-mode` (`random` or `manual`)

## PPO Configuration

The controller is trained using Proximal Policy Optimization (PPO).
The following arguments control the PPO training process:
- `--rollout-steps` the number of rollout steps.
- `--ppo-lr` for the PPO learning rate.
- `--ppo-epochs` the number of PPO epochs.
- `--ppo-batch-size` the batch size for PPO updates by batch.
- `--ppo-clip-range` the interval for PPO gradient clipping.
- `--ppo-range-vf` the value function for PPO gradient clipping.
- `--gamma` the discount factor.
- `--gae-lambda` the GAE lambda parameter.
- `--value-coef` the coefficient for the value loss.
- `--entropy-coef` the coefficient for the entropy loss.
- `--max-grad-norm` the value to clip gradients at.
- `--train-world-model` whether to train vision and memory too.
- `--world-model-epochs` the number of epochs per rollout to train vision and memory.

For example, if you want to train a model on CartPole-v3, run:
```$ python src/main.py --env CartPole-v3```

To use specific models, run (as an example):
```$ python src/main.py --env CartPole-v3 --vision VQ_VAE --memory LSTMMemory --controller DeepContinuousController```

## Inference

To infer on pretrained models, run `src/inference.py`.

The accepted arguments are:
- `--load-path` the path to the pretrained model.
- `--episodes` the number of episodes to infer on.
- `--render-mode` to set render mode between `human` and `rgb_array` (no render).

## What you can do
You can train your own models, using the implemented vision, memory and controller sub models.

Alternatively, you can use the pretrained models provided and run them
on their respective environments.

## Quick Start

```bash
uv sync
python src/main.py --env CartPole-v1 --render-mode human
```
