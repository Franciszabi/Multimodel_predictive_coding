"""
Per-frame semantic Autoencoder for ablation (no temporal modeling).

Trainers can flatten (B, L, K) -> (B*L, K) and run this module so that time is only
stacked into the batch dimension, analogous to the visual Autoencoder path in Trainer.

Flow:
  token_ids (B*, K), token_mask (B*, K)
    -> Embedding -> (B*, K, D)
    -> masked mean over K -> (B*, D)
    -> Encoder MLP -> bottleneck (B*, Z)
    -> Decoder MLP -> logits (B*, num_classes)

Loss (outside this module): e.g. BCEWithLogits(logits, target_multi_hot) on the *same* frame.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn


class SemanticFrameAutoencoder(nn.Module):
    """
    Frame-wise token AE: reconstruct current-frame multi-label semantics from pooled token embedding.

    Inputs are 2D (already flattened batch over sequences): B* = B×L typical.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_classes: int,
        bottleneck_dim: int,
        hidden_dim: Optional[int] = None,
        padding_idx: int = 0,
        dropout: float = 0.1,
        embedding_matrix: Optional[np.ndarray] = None,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_classes = num_classes
        self.bottleneck_dim = bottleneck_dim
        self.padding_idx = padding_idx
        h = hidden_dim if hidden_dim is not None else max(d_model, bottleneck_dim)

        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        if embedding_matrix is not None:
            emb = torch.from_numpy(embedding_matrix).float()
            if emb.shape != self.embed.weight.data.shape:
                raise ValueError(
                    f"embedding_matrix shape {tuple(emb.shape)} must match "
                    f"nn.Embedding shape {tuple(self.embed.weight.data.shape)}"
                )
            with torch.no_grad():
                self.embed.weight.copy_(emb)

        self.dropout = nn.Dropout(dropout)
        self.ln_pooled = nn.LayerNorm(d_model)

        self.encoder = nn.Sequential(
            nn.Linear(d_model, h),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(h, bottleneck_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, h),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(h, num_classes),
        )

    def pool_tokens(self, token_ids: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
        """token_ids: (B*, K), token_mask: (B*, K) True = valid -> pooled (B*, D)."""
        e = self.embed(token_ids)
        m = token_mask.unsqueeze(-1).float()
        denom = m.sum(dim=1).clamp_min(1.0)
        pooled = (e * m).sum(dim=1) / denom
        return pooled

    def forward(
        self,
        token_ids: torch.Tensor,
        token_mask: torch.Tensor,
        return_latents: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
        """
        token_ids: (B*, K) long
        token_mask: (B*, K) bool, True where token is valid (non-padding)

        Returns:
          logits: (B*, num_classes)
        """
        pooled = self.pool_tokens(token_ids, token_mask)
        pooled = self.ln_pooled(pooled)
        pooled = self.dropout(pooled)
        z = self.encoder(pooled)
        logits = self.decoder(z)

        if return_latents:
            latents: Dict[str, Any] = {
                "pooled": pooled.detach(),
                "bottleneck": z.detach(),
            }
            return logits, latents
        return logits
