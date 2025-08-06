# src/interface/interface.py

import gymnasium as gym

class GymEnvInterface:
    def __init__(self, env_name, render_mode=None):
        self.env = gym.make(env_name, render_mode=render_mode)
        # The initial reset is done here to set up the instance attributes
        self.state, self.info = self.env.reset()
        self.done = False

    def reset(self):
        """Resets the environment and returns the initial state and info."""
        self.state, self.info = self.env.reset()
        self.done = False
        # Return both the state and the info dictionary
        return self.state, self.info

    def step(self, action):
        """Takes a step in the environment."""
        next_state, reward, terminated, truncated, info = self.env.step(action)
        self.done = terminated or truncated
        self.state = next_state
        return next_state, reward, self.done, info

    def render(self):
        """Renders the environment."""
        if self.env.render_mode == 'human':
            self.env.render()

    def close(self):
        """Closes the environment."""
        self.env.close()
