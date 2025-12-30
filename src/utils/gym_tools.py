import gymnasium as gym
from gymnasium.envs.registration import registry


def get_all_gym_envs() -> list:
    """Returns a list of all registered gym environments."""
    all_envs = list(registry.keys())
    return [env for env in all_envs if not env.startswith("_")]


def action_space_is_discrete(env_name: str) -> bool:
    """Returns True if the action space of the environment is discrete."""
    env = gym.make(env_name)
    is_discrete = isinstance(env.action_space, gym.spaces.Discrete)
    env.close()
    return is_discrete


def gym_is_image_based(env_name: str) -> bool:
    """Returns True if the observation space of the environment is image-based."""
    env = gym.make(env_name)
    assert env.observation_space.shape is not None
    is_image_based = len(env.observation_space.shape) == 3
    env.close()
    return is_image_based
