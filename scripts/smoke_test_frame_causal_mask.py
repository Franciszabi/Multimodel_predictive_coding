"""Verify that changing future input frames cannot change earlier logits."""

from __future__ import annotations

import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.models.semantic_gpt import SemanticGPT


def main() -> None:
    torch.manual_seed(7)
    length, width, vocab_size = 5, 3, 11
    model = SemanticGPT(
        vocab_size=vocab_size,
        d_model=24,
        L=length,
        K=width,
        num_layers=2,
        num_heads=4,
        num_classes=6,
        dropout=0.0,
        attention_mask_mode="frame_causal",
    ).eval()
    original = torch.randint(2, vocab_size, (2, length, width))
    token_mask = torch.ones_like(original, dtype=torch.bool)

    with torch.no_grad():
        for frame in (0, 1, 3):
            changed = original.clone()
            changed[:, frame + 1 :] = ((changed[:, frame + 1 :] - 2 + 3) % (vocab_size - 2)) + 2
            logits_original = model(original, token_mask)
            logits_changed = model(changed, token_mask)
            assert torch.allclose(
                logits_original[:, frame], logits_changed[:, frame], atol=1e-6, rtol=1e-6
            ), f"future leakage detected at frame {frame}"
    print("PASS: future input frames do not affect earlier frame logits")


if __name__ == "__main__":
    main()
