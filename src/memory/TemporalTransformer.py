import torch

from Model import Model
from .MemoryModel import flatten_vision_latents

class TemporalTransformer(Model):
    def __init__(self, latent_dim=4, action_dim=3, d_model=128, nhead=8, num_layers=4, max_len=32):
        super().__init__()
        
        assert d_model % nhead == 0
        
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.input_dim = latent_dim + action_dim
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.max_len = max_len

        self.memory_input_proj = torch.nn.Linear(self.input_dim, d_model)

        self.prior_proj = torch.nn.Linear(latent_dim + action_dim, d_model)

        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            batch_first=True
        )
        self.transformer = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_proj = torch.nn.Linear(d_model, latent_dim)

        self.seq_buffer = None
        self.seq_lengths = None

    def update_memory(self, z_t, a_prev):
        """
        z_t:   (B, latent_dim)
        a_prev:(B, action_dim)

        IMPROVED: Allows gradient flow through current timestep while maintaining
        buffer stability by detaching historical entries.
        """

        B = z_t.size(0)
        device = z_t.device

        if self.seq_buffer is None:
            self.seq_buffer = torch.zeros(B, self.max_len, self.d_model, device=device)
            self.seq_lengths = torch.zeros(B, dtype=torch.long, device=device)

        # FIXED: Don't detach z_t and a_prev - allow gradients to flow
        x = torch.cat([z_t, a_prev], dim=-1)
        x = self.memory_input_proj(x).unsqueeze(1)

        # Update buffer with detached version (for memory persistence)
        self.seq_buffer = torch.roll(self.seq_buffer, shifts=-1, dims=1)
        self.seq_buffer[:, -1] = x.squeeze(1).detach()
        self.seq_lengths = torch.clamp(self.seq_lengths + 1, max=self.max_len)

        # For transformer forward: use detached history but gradient-connected current
        memory_in = self.seq_buffer.clone()  # Already detached from buffer update
        memory_in[:, -1] = x.squeeze(1)       # Replace last with grad-connected tensor

        mask = torch.arange(self.max_len, device=device).unsqueeze(0) >= self.seq_lengths.unsqueeze(1)

        memory_out = self.transformer(memory_in, src_key_padding_mask=mask)

        last_indices = self.seq_lengths - 1
        h_t = memory_out[torch.arange(B, device=device), last_indices]

        return h_t

    def predict_next(self, z_t, a_t, h_t):
        """
        Prédit z_{t+1} à partir de (z_t, a_t, h_t)
        """

        x = torch.cat([z_t, a_t], dim=-1)     # (B, latent+action)
        x = self.prior_proj(x)                # (B, d_model)

        x = x + h_t

        z_next = self.output_proj(x)          # (B, latent_dim)
        return z_next

    def forward(self, z_t, a_prev, a_t):
        """
        Full step:
          h_t = update_memory(z_t, a_prev)
          z_next_pred = predict_next(z_t, a_t, h_t)
        """

        h_t = self.update_memory(z_t, a_prev)
        z_next_pred = self.predict_next(z_t, a_t, h_t)
        return z_next_pred, h_t

    def reset_env_memory(self, env_idx):
        env_idx = int(env_idx)
        self.seq_buffer[env_idx].zero_()
        self.seq_lengths[env_idx] = 0

    def export_hyperparams(self):
        return {
            "latent_dim": self.latent_dim,
            "action_dim": self.action_dim,
            "d_model": self.d_model,
            "nhead": self.nhead,
            "num_layers": self.num_layers,
            "max_len": self.max_len
        }

    def save_state(self):
        return self.state_dict()

    def load(self, state_dict):
        self.load_state_dict(state_dict)
