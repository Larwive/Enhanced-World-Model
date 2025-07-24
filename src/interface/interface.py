import gymnasium as gym

class GymEnvInterface:
    def __init__(self, env_name, render_mode=None):
        self.env = gym.make(env_name, render_mode=render_mode)
        self.state, self.info = self.env.reset()
        self.done = False

    def reset(self):
        self.state, self.info = self.env.reset()
        self.done = False
        return self.state

    def step(self, action):
        next_state, reward, terminated, truncated, info = self.env.step(action)
        self.done = terminated or truncated
        self.state = next_state
        return next_state, reward, self.done, info

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()
