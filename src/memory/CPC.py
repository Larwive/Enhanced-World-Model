import torch
import torchvision

from Model import Model


class CPC(Model):

    def __init__(self, latent_dim, context_dim: int = 128, prediction_steps: int = 12):
        super().__init__()
        self.gru = torch.nn.GRU(latent_dim, context_dim, batch_first=True)
        self.predictors = torch.nn.ModuleList(
            [torch.nn.Linear(context_dim, latent_dim) for _ in range(prediction_steps)])

    def forward(self, z_seq: torch.Tensor):
        # z_seq: (B, T, D)
        context, _ = self.gru(z_seq)
        predictions = [W(context) for W in self.predictors]
        return predictions


def info_nce_loss(z_seq, predictions, k_steps=12):
    B, T, D = z_seq.shape
    total_loss = 0
    for k in range(1, k_steps + 1):
        if T - k <= 0:
            break
        z_target = z_seq[:, k:, :]  # future
        z_pred = predictions[k - 1][:, :-k, :]  # predicted

        # Dot product similarity
        logits = torch.bmm(z_pred, z_target.transpose(1, 2))  # (B, T-k, T-k)
        labels = torch.arange(z_pred.size(1)).to(z_pred.device)
        labels = labels.unsqueeze(0).expand(B, -1)

        loss = torch.nn.functional.cross_entropy(logits, labels)
        total_loss += loss
    return total_loss / k_steps
