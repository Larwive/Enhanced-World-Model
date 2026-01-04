import gymnasium as gym
from gymnasium.envs.registration import registry
from gymnasium.spaces import Space


def get_all_gym_envs() -> list:
    """Returns a list of all registered gym environments."""
    all_envs = list(registry.keys())
    return [env for env in all_envs if not env.startswith("_")]


def get_env_info(env_name: str) -> tuple[Space, Space, bool, bool]:
    """Returns information about the environment."""
    env = gym.make(env_name)
    observation_space = env.observation_space
    action_space = env.action_space
    assert observation_space.shape is not None
    is_image_based = len(observation_space.shape) == 3
    is_discrete = isinstance(action_space, gym.spaces.Discrete)
    env.close()
    return observation_space, action_space, is_image_based, is_discrete
