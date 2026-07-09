"""
Pure semantic sequence prediction model (GPT-like).

Design:
- Input: token_ids (B, L, K), token_mask (B, L, K). L=32 frames, K tokens per frame.
- Frame×word flattened to sequence length N = L*K; causal self-attention.
- Position encoding: sinusoidal, frame-only (same encoding for all K tokens in a frame).
- Output: (B, L, V) logits for per-frame multi-label prediction.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_frame_encoding(L_max: int, D: int, device: torch.device) -> torch.Tensor:
    """
    Sinusoidal position encoding for frame indices only. Shape (L_max, D).
    Same as original Transformer PE but over L positions (one per frame).
    """
    pe = torch.zeros(L_max, D, device=device)
    position = torch.arange(0, L_max, dtype=torch.float32, device=device).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, D, 2, dtype=torch.float32, device=device) * (-math.log(10000.0) / D)
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    if D % 2 != 0:
        pe[:, -1] = 0.0
    return pe  # (L_max, D)


class CausalSelfAttentionBlock(nn.Module):
    """Single transformer decoder block: causal self-attention + FFN."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        assert self.d_head * num_heads == d_model

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # x: (B, N, D). key_padding_mask: (B, N), True = ignore (padding)
        B, N, D = x.shape
        residual = x
        x = self.ln1(x)
        q = self.q_proj(x).view(B, N, self.num_heads, self.d_head).transpose(1, 2)   # (B, H, N, d_head)
        k = self.k_proj(x).view(B, N, self.num_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.num_heads, self.d_head).transpose(1, 2)
        # Causal + key_padding: attn_mask (B, 1, N, N), True = ignore (PyTorch convention)
        if key_padding_mask is not None:
            causal = torch.triu(torch.ones(N, N, device=x.device, dtype=torch.bool), diagonal=1)  # j>i
            pad = key_padding_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, N)
            attn_mask = causal.unsqueeze(0).unsqueeze(0) | pad  # (B, 1, N, N)
            use_causal = False
        else:
            attn_mask = None
            use_causal = True
        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=use_causal,
            scale=1.0 / math.sqrt(self.d_head),
        )  # (B, H, N, d_head)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, N, D)
        attn_out = self.out_proj(attn_out)
        if key_padding_mask is not None:
            attn_out = attn_out.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
        x = residual + self.dropout(attn_out)

        residual = x
        x = self.ln2(x)
        x = residual + self.ffn(x)
        return x


class SemanticGPT(nn.Module):
    """
    Pure semantic sequence model: frame×word flattened, causal self-attention,
    frame-only sinusoidal position encoding, per-frame prediction (B, L, V).
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        L: int = 32,
        K: int = 16,
        num_layers: int = 4,
        num_heads: int = 4,
        d_ff: Optional[int] = None,
        num_classes: int = 25,
        padding_idx: int = 0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.L = L
        self.K = K
        self.N = L * K
        self.d_model = d_model
        self.padding_idx = padding_idx
        d_ff = d_ff or d_model * 4

        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        self.register_buffer(
            "pos_frame",
            sinusoidal_frame_encoding(L, d_model, device=torch.device("cpu")),
        )  # (L, D); will move with model

        self.blocks = nn.ModuleList([
            CausalSelfAttentionBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.ln_final = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        token_ids: torch.Tensor,
        token_mask: torch.Tensor,
        return_latents: Union[bool, str] = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
        """
        token_ids: (B, L, K) long
        token_mask: (B, L, K) bool, True = valid token
        return_latents: False | True | "full"
          - False: return only logits (B, L, num_classes)
          - True: return (logits, latents_dict) with latents_dict['final'] (B, L, D), ['final_seq'] (B, N, D)
          - "full": also include 'before_attn_layers' and 'after_attn_layers' (list of (B, N, D) per layer)
        Returns: logits or (logits, latents_dict)
        """
        B, L, K = token_ids.shape
        assert L == self.L and K == self.K
        device = token_ids.device
        if self.pos_frame.device != device:
            self.pos_frame = self.pos_frame.to(device)

        # (B, L, K, D)
        x = self.embed(token_ids)
        # Frame-only position: (L, D) -> (1, L, 1, D) and add to (B, L, K, D)
        x = x + self.pos_frame.unsqueeze(0).unsqueeze(2)
        x = self.dropout(x)

        # Flatten to (B, N, D)
        N = L * K
        x = x.view(B, N, self.d_model)
        # Padding mask for attention: True = ignore. Shape (B, N)
        key_padding_mask = ~token_mask.view(B, N)

        save_full = return_latents == "full"
        before_attn_layers: list = []
        after_attn_layers: list = []

        for block in self.blocks:
            if save_full:
                before_attn_layers.append(x.detach().clone())
            x = block(x, key_padding_mask=key_padding_mask)
            if save_full:
                after_attn_layers.append(x.detach().clone())

        x = self.ln_final(x)
        # (B, N, D) -> (B, L, K, D)
        x_view = x.view(B, L, K, self.d_model)
        # Masked mean over K: (B, L, D)
        token_mask_f = token_mask.unsqueeze(-1).float()
        x_pooled = (x_view * token_mask_f).sum(dim=2) / (token_mask_f.sum(dim=2).clamp(min=1e-6))
        logits = self.head(x_pooled)  # (B, L, num_classes)

        if return_latents:
            latents: Dict[str, Any] = {
                "final": x_pooled.detach(),      # (B, L, D) per-frame, before head
                "final_seq": x.detach(),         # (B, N, D) before pool
            }
            if save_full:
                latents["before_attn_layers"] = before_attn_layers
                latents["after_attn_layers"] = after_attn_layers
            return logits, latents
        return logits
