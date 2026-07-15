"""Frame-causal Transformer for future semantic observation prediction."""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


ATTENTION_MASK_MODES = ("frame_causal", "token_causal_legacy")


def sinusoidal_frame_encoding(length: int, d_model: int, device: torch.device) -> torch.Tensor:
    """Return sinusoidal encodings over frame indices, shape ``(length, d_model)``."""

    encoding = torch.zeros(length, d_model, device=device)
    position = torch.arange(length, dtype=torch.float32, device=device).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float32, device=device)
        * (-math.log(10000.0) / d_model)
    )
    encoding[:, 0::2] = torch.sin(position * div_term)
    encoding[:, 1::2] = torch.cos(position * div_term)
    if d_model % 2:
        encoding[:, -1] = 0.0
    return encoding


def build_frame_causal_attention_mask(
    sequence_length: int,
    tokens_per_frame: int,
    token_mask: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build an additive frame-causal mask with shape ``(B, 1, N, N)``.

    A query token in frame ``t`` may attend to every valid token in frames
    ``<= t``. All tokens in the same frame are mutually visible. Padding keys
    and keys in future frames receive ``-inf``.
    """

    batch_size, length, width = token_mask.shape
    if (length, width) != (sequence_length, tokens_per_frame):
        raise ValueError(
            f"token_mask shape {tuple(token_mask.shape)} does not match "
            f"L={sequence_length}, K={tokens_per_frame}"
        )
    frame_ids = torch.arange(sequence_length, device=token_mask.device).repeat_interleave(
        tokens_per_frame
    )
    frame_allowed = frame_ids.unsqueeze(0) <= frame_ids.unsqueeze(1)
    valid_keys = token_mask.reshape(batch_size, -1).unsqueeze(1).unsqueeze(2)
    allowed = frame_allowed.unsqueeze(0).unsqueeze(0) & valid_keys
    additive_mask = torch.zeros(allowed.shape, device=token_mask.device, dtype=dtype)
    return additive_mask.masked_fill(~allowed, float("-inf"))


def build_token_causal_attention_mask(
    token_mask: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build the old flattened-token causal mask for checkpoint comparisons."""

    batch_size = token_mask.shape[0]
    token_count = token_mask.shape[1] * token_mask.shape[2]
    positions = torch.arange(token_count, device=token_mask.device)
    causal_allowed = positions.unsqueeze(0) <= positions.unsqueeze(1)
    valid_keys = token_mask.reshape(batch_size, token_count).unsqueeze(1).unsqueeze(2)
    allowed = causal_allowed.unsqueeze(0).unsqueeze(0) & valid_keys
    additive_mask = torch.zeros(allowed.shape, device=token_mask.device, dtype=dtype)
    return additive_mask.masked_fill(~allowed, float("-inf"))


class CausalSelfAttentionBlock(nn.Module):
    """Pre-norm self-attention and feed-forward block."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if d_model % num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.d_head = self.d_model // self.num_heads

        self.q_proj = nn.Linear(self.d_model, self.d_model)
        self.k_proj = nn.Linear(self.d_model, self.d_model)
        self.v_proj = nn.Linear(self.d_model, self.d_model)
        self.out_proj = nn.Linear(self.d_model, self.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(self.d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, self.d_model),
            nn.Dropout(dropout),
        )
        self.ln1 = nn.LayerNorm(self.d_model)
        self.ln2 = nn.LayerNorm(self.d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor,
        query_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, token_count, d_model = x.shape
        residual = x
        normalized = self.ln1(x)
        q = self.q_proj(normalized).view(
            batch_size, token_count, self.num_heads, self.d_head
        ).transpose(1, 2)
        k = self.k_proj(normalized).view(
            batch_size, token_count, self.num_heads, self.d_head
        ).transpose(1, 2)
        v = self.v_proj(normalized).view(
            batch_size, token_count, self.num_heads, self.d_head
        ).transpose(1, 2)

        attended = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_mask,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=False,
        )
        attended = torch.nan_to_num(attended)
        attended = attended.transpose(1, 2).contiguous().view(
            batch_size, token_count, d_model
        )
        attended = self.out_proj(attended)
        x = residual + self.dropout(attended)
        x = x.masked_fill(query_padding_mask.unsqueeze(-1), 0.0)

        residual = x
        x = residual + self.ffn(self.ln2(x))
        return x.masked_fill(query_padding_mask.unsqueeze(-1), 0.0)


class SemanticGPT(nn.Module):
    """Predict frame ``t+h`` from semantic observations available through frame ``t``."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        L: int = 25,
        K: int = 16,
        num_layers: int = 4,
        num_heads: int = 4,
        d_ff: Optional[int] = None,
        num_classes: int = 25,
        padding_idx: int = 0,
        dropout: float = 0.1,
        attention_mask_mode: str = "frame_causal",
    ) -> None:
        super().__init__()
        if attention_mask_mode not in ATTENTION_MASK_MODES:
            raise ValueError(
                f"attention_mask_mode must be one of {ATTENTION_MASK_MODES}, "
                f"got {attention_mask_mode!r}"
            )
        self.L = int(L)
        self.K = int(K)
        self.N = self.L * self.K
        self.d_model = int(d_model)
        self.padding_idx = int(padding_idx)
        self.attention_mask_mode = attention_mask_mode
        d_ff = int(d_ff or self.d_model * 4)

        self.embed = nn.Embedding(vocab_size, self.d_model, padding_idx=self.padding_idx)
        self.register_buffer(
            "pos_frame",
            sinusoidal_frame_encoding(self.L, self.d_model, torch.device("cpu")),
        )
        self.blocks = nn.ModuleList(
            CausalSelfAttentionBlock(self.d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        )
        self.ln_final = nn.LayerNorm(self.d_model)
        self.head = nn.Linear(self.d_model, num_classes)
        self.dropout = nn.Dropout(dropout)

    def _attention_mask(self, token_mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        if self.attention_mask_mode == "frame_causal":
            return build_frame_causal_attention_mask(self.L, self.K, token_mask, dtype)
        return build_token_causal_attention_mask(token_mask, dtype)

    def forward(
        self,
        token_ids: torch.Tensor,
        token_mask: torch.Tensor,
        return_latents: Union[bool, str] = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
        """Return future-semantic logits with shape ``(B, L, num_classes)``.

        Under ``frame_causal`` masking, the latent at frame ``t`` is computed
        only from frames ``<= t``. The training target aligned to that latent is
        frame ``t+h``; input tokens from later frames cannot leak into it.
        """

        if token_ids.ndim != 3 or token_mask.shape != token_ids.shape:
            raise ValueError("token_ids and token_mask must both have shape (B, L, K)")
        batch_size, length, width = token_ids.shape
        if (length, width) != (self.L, self.K):
            raise ValueError(
                f"Input shape {tuple(token_ids.shape)} does not match model L={self.L}, K={self.K}"
            )
        token_mask = token_mask.bool()

        x = self.embed(token_ids)
        x = x + self.pos_frame.unsqueeze(0).unsqueeze(2)
        x = self.dropout(x)
        x = x.reshape(batch_size, self.N, self.d_model)
        query_padding_mask = ~token_mask.reshape(batch_size, self.N)
        x = x.masked_fill(query_padding_mask.unsqueeze(-1), 0.0)
        attention_mask = self._attention_mask(token_mask, x.dtype)

        save_full = return_latents == "full"
        before_attn_layers: list[torch.Tensor] = []
        after_attn_layers: list[torch.Tensor] = []
        for block in self.blocks:
            if save_full:
                before_attn_layers.append(x.detach().clone())
            x = block(x, attention_mask, query_padding_mask)
            if save_full:
                after_attn_layers.append(x.detach().clone())

        x = self.ln_final(x)
        x = x.masked_fill(query_padding_mask.unsqueeze(-1), 0.0)
        x_view = x.reshape(batch_size, self.L, self.K, self.d_model)
        mask_float = token_mask.unsqueeze(-1).to(x_view.dtype)
        pooled = (x_view * mask_float).sum(dim=2) / mask_float.sum(dim=2).clamp_min(1.0)
        logits = self.head(pooled)

        if not return_latents:
            return logits
        latents: Dict[str, Any] = {
            "final": pooled.detach(),
            "final_seq": x.detach(),
        }
        if save_full:
            latents["before_attn_layers"] = before_attn_layers
            latents["after_attn_layers"] = after_attn_layers
        return logits, latents
