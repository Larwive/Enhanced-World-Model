import pytest


# test_vision.py
@pytest.fixture
def vision_input_args():
    return {"input_shape": (3, 32, 32)}


# test_memory.py
@pytest.fixture
def memory_input_args():
    return {}


# test_controller.py
@pytest.fixture
def controller_input_args():
    return {"z_dim": 1, "h_dim": 1, "action_dim": 1}


# test_main.py
@pytest.fixture
def main_input_args():
    return {
        "correct": [
            # TODO: Uncomment after implementing the `evaluate_actions` method.
            [
                "prog",
                "--env",
                "CartPole-v1",
                "--vision",
                "Identity",
                "--memory",
                "TemporalTransformer",
                "--controller",
                "DeepDiscreteController",
            ],
            # ["prog", "--env", "CartPole-v1", "--vision", "Identity", "--memory", "LSTMMemory", "--controller", "DiscreteModelPredictiveController"],
            [
                "prog",
                "--env",
                "CarRacing-v3",
                "--vision",
                "VQ_VAE",
                "--memory",
                "LSTMMemory",
                "--controller",
                "DeepContinuousController",
            ],
            # ["prog", "--env", "CarRacing-v3", "--vision", "JEPA", "--memory", "LSTMMemory", "--controller", "ContinuousModelPredictiveController"],
            # ["prog", "--env", "CarRacing-v3", "--vision", "JEPA", "--memory", "LSTMMemory", "--controller", "StochasticController"],
        ],
        "error": [
            [
                "prog",
                "--env",
                "CartPole-v1",
                "--vision",
                "VQ_VAE",
                "--memory",
                "TemporalTransformer",
                "--controller",
                "DeepDiscreteController",
            ],
            [
                "prog",
                "--env",
                "CartPole-v1",
                "--vision",
                "Identity",
                "--memory",
                "LSTMMemory",
                "--controller",
                "DeepContinuousController",
            ],
            [
                "prog",
                "--env",
                "CarRacing-v3",
                "--vision",
                "Identity",
                "--memory",
                "LSTMMemory",
                "--controller",
                "DeepContinuousController",
            ],
            [
                "prog",
                "--env",
                "CarRacing-v3",
                "--vision",
                "VQ_VAE",
                "--memory",
                "LSTMMemory",
                "--controller",
                "DeepDiscreteController",
            ],
        ],
    }
