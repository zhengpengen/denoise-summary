import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionalEmbedding(nn.Module):
    """Helper module for creating sinusoidal timestep embeddings."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class SummarizationDenoiser(nn.Module):
    """
    A Transformer-based model to denoise a summary mask.
    It predicts the original mask (x_0) from a noisy version (x_t).
    """
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config

        # Destructure config for clarity
        model_dim = config.model.dim
        num_heads = config.model.num_heads
        num_layers = config.model.num_layers
        dropout = config.model.dropout
        sentence_embedding_dim = config.summarization.sentence_embedding_dim
        num_mask_classes = 2  # 0 for not-in-summary, 1 for in-summary
        
        # For embedding the timestep t
        self.time_embed = nn.Sequential(
            SinusoidalPositionalEmbedding(model_dim),
            nn.Linear(model_dim, model_dim * 4),
            nn.GELU(),
            nn.Linear(model_dim * 4, model_dim)
        )

        # For embedding the noisy mask labels (0 for not-in-summary, 1 for in-summary)
        self.mask_embed = nn.Embedding(num_mask_classes, model_dim)

        # Input projection to combine sentence embeddings and mask embeddings
        self.input_proj = nn.Linear(sentence_embedding_dim + model_dim, model_dim)

        # The core Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=model_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # Output layer to predict the logits for each sentence being in the summary
        self.output_proj = nn.Linear(model_dim, num_mask_classes)

    def forward(self,
                sentence_embeddings: torch.Tensor,
                noisy_mask: torch.Tensor,
                timesteps: torch.Tensor,
                attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sentence_embeddings (Tensor): Embeddings of each sentence. Shape: (B, L, E_dim).
            noisy_mask (Tensor): The noisy summary mask at step t. Shape: (B, L).
            timesteps (Tensor): The current timestep for each item in the batch. Shape: (B,).
            attention_mask (Tensor): Mask for padding in sentence_embeddings. Shape: (B, L).

        Returns:
            Tensor: Logits for each class (0 or 1) for each sentence. Shape: (B, L, 2).
        """
        # 1. Embed timestep and noisy mask
        time_emb = self.time_embed(timesteps) # (B, model_dim)
        mask_emb = self.mask_embed(noisy_mask) # (B, L, model_dim)

        # 2. Combine embeddings
        # Project sentence embeddings to match model_dim and add mask embeddings
        combined_input = self.input_proj(torch.cat([sentence_embeddings, mask_emb], dim=-1))

        # 3. Add timestep embedding to each position
        x = combined_input + time_emb.unsqueeze(1) # (B, L, model_dim)

        # 4. Pass through Transformer
        # The transformer expects a src_key_padding_mask where `True` indicates a padded value
        src_key_padding_mask = (attention_mask == 0)
        transformer_out = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)

        # 5. Get output logits
        logits = self.output_proj(transformer_out) # (B, L, 2)
        return logits