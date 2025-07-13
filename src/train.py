import torch

from vision import VQ_VAE
from memory import TemporalTransformerXCPC
from controller import ModelPredictiveControl
from WorldModel import WorldModel


def step(model, input_tensor, optimizer):
    optimizer.zero_grad()
    output_dict = model(input_tensor)
    total_loss = torch.sum(output_dict["total_loss"])
    total_loss.backward()
    optimizer.step()


def train(model: WorldModel, data, max_iter=10000):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for iter in range(max_iter):
        input_tensor = None  # See with interface.
        step(model, input_tensor, optimizer)


if __name__ == "__main__":
    vision_model = VQ_VAE
    memory_model = TemporalTransformerXCPC
    controller_model = ModelPredictiveControl
    data = None  # See with interface
    input_shape = data.shape[0].shape[1:]  # Not definitive. Only a prediction.
    world_model = WorldModel(vision_model, memory_model, controller_model, input_shape, None, None, None)

    train(world_model, data)
    world_model.save("./test_world_model.pt")
