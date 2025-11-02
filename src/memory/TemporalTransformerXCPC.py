import torch

from Model import Model
from .MemoryModel import flatten_vision_latents

class TemporalTransformer(Model):
    def __init__(self, input_dim, latent_dim=4, d_model=128, nhead=8, num_layers=4):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        
        self.input_proj = torch.nn.Linear(input_dim, d_model)
        
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            batch_first=True
        )
        self.transformer = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.output_proj = torch.nn.Linear(d_model, latent_dim)

    
    def forward(self, z_current, action=None):
        """
        Args:
            z_current: (B, latent_dim) - représentation latente actuelle
            action: (B, action_dim) - action prise (optionnel)
        Returns:
            z_next_pred: (B, latent_dim, 1, 1) - prédiction du prochain état latent
            hidden: (B, d_model) - état caché pour usage futur
        """

        if action is not None:
            x = torch.cat([z_current, action], dim=-1)
            if not hasattr(self, 'action_proj'):
                action_dim = action.shape[-1]
                self.action_proj = torch.nn.Linear(
                    self.input_dim + action_dim, 
                    self.d_model
                ).to(z_current.device)
            projected = self.action_proj(x)
        else:
            projected = self.input_proj(z_current)
    
        projected = projected.unsqueeze(1)  # (B, 1, d_model)
        
        memory = self.transformer(projected)  # (B, 1, d_model)
        hidden = memory.squeeze(1)  # (B, d_model)
        
        z_next_pred = self.output_proj(hidden)  # (B, latent_dim)
        
        z_next_pred = z_next_pred.unsqueeze(-1).unsqueeze(-1)  # (B, latent_dim, 1, 1)
        
        return z_next_pred, hidden

    def export_hyperparams(self):
        return {
            "input_dim": self.input_dim,
            "latent_dim": self.latent_dim,
            "d_model": self.d_model,
            "nhead": self.nhead,
            "num_layers": self.num_layers
        }

    def save_state(self):
        return self.state_dict()

    def load(self, state_dict):
        self.load_state_dict(state_dict)
