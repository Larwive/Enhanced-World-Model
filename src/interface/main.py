from interface import GymEnvInterface 


# Example usage with CartPole-v1
interface = GymEnvInterface("CartPole-v1", render_mode="human")
state = interface.reset()

while not interface.done:
    action = interface.env.action_space.sample()  # Replace with your model's action
    next_state, reward, done, info = interface.step(action)
    interface.render()

interface.close()
