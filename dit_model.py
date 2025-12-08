import math
import torch
import torch.nn as nn
from omegaconf import DictConfig

class SinusoidalPositionalEmbedding(nn.Module):
    """Standard sinusoidal embedding for timesteps."""
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

def modulate(x, shift, scale):
    """
    Adaptive Layer Norm modulation.
    x: (B, L, D)
    shift, scale: (B, D)
    """
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class DiTBlock(nn.Module):
    """
    A Transformer block that uses Adaptive Layer Norm (AdaLN) 
    conditioned on the timestep.
    """
    def __init__(self, hidden_size, num_heads, dropout=0.1, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, hidden_size),
            nn.Dropout(dropout)
        )
        
        # adaLN modulation: regresses shift/scale parameters from the timestep embedding
        # We produce 6 parameters: shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c, src_key_padding_mask=None):
        """
        x: Input tokens (B, L, D)
        c: Timestep embedding (B, D)
        src_key_padding_mask: (B, L)
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        
        # 1. Self-Attention with modulation
        x_norm = modulate(self.norm1(x), shift_msa, scale_msa)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, key_padding_mask=src_key_padding_mask)
        x = x + gate_msa.unsqueeze(1) * attn_out
        
        # 2. MLP with modulation
        x_norm = modulate(self.norm2(x), shift_mlp, scale_mlp)
        mlp_out = self.mlp(x_norm)
        x = x + gate_mlp.unsqueeze(1) * mlp_out
        
        return x

class SummarizationDenoiser(nn.Module):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        
        model_dim = config.model.dim
        num_heads = config.model.num_heads
        num_layers = config.model.num_layers
        dropout = config.model.dropout
        sent_emb_dim = config.summarization.sentence_embedding_dim
        
        # 1. Timestep Embedder
        self.time_embed = nn.Sequential(
            SinusoidalPositionalEmbedding(model_dim),
            nn.Linear(model_dim, model_dim),
            nn.SiLU(),
            nn.Linear(model_dim, model_dim),
        )

        # 2. Input Projection
        # We project sentence embedding + class embedding (for the mask) into model_dim
        self.mask_embed = nn.Embedding(2, model_dim)
        self.input_proj = nn.Linear(sent_emb_dim + model_dim, model_dim)

        # 3. The Backbone: Stack of DiT Blocks
        self.blocks = nn.ModuleList([
            DiTBlock(model_dim, num_heads, dropout=dropout) for _ in range(num_layers)
        ])
        
        # 4. Final Layer Norm & Output
        self.final_norm = nn.LayerNorm(model_dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_final = nn.Sequential(
             nn.SiLU(),
             nn.Linear(model_dim, 2 * model_dim, bias=True)
        )
        self.output_proj = nn.Linear(model_dim, 2) # 2 classes: (not_summary, summary)

        # Initialize weights (Crucial for DiT stability)
        self.apply(self._init_weights)
        
        # Zero-out the final gates in DiT blocks to make the block identity at init
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, sentence_embeddings, noisy_mask, timesteps, attention_mask):
        # Embed time
        t_emb = self.time_embed(timesteps) # (B, model_dim)
        
        # Embed mask and combine with sentences
        mask_emb = self.mask_embed(noisy_mask)
        x = self.input_proj(torch.cat([sentence_embeddings, mask_emb], dim=-1))
        
        # Pass through DiT blocks
        # Pytorch MHA expects mask as True = Ignore, False = Keep. 
        # Your dataloader likely gives 1=Keep, 0=Pad. So we invert it.
        key_padding_mask = (attention_mask == 0) 
        
        for block in self.blocks:
            x = block(x, t_emb, src_key_padding_mask=key_padding_mask)
            
        # Final Norm
        shift, scale = self.adaLN_final(t_emb).chunk(2, dim=1)
        x = modulate(self.final_norm(x), shift, scale)
        
        logits = self.output_proj(x)
        return logits