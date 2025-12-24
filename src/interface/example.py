import gymnasium as gym

# Example usage with CartPole-v1
env = gym.make("CarRacing-v3", render_mode="human")  # CartPole-v1
state = env.reset()
done = False

while not done:
    action = env.action_space.sample()  # Replace with your model's action
    next_state, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    env.render()

env.close()
