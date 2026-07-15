"""Run one tiny CPU AE epoch and verify same-frame latent output."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.smoke_utils import write_tiny_trail
from train_semantic_ae import main as train_main


def main() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        data_root = root / "trail_ae_dry_run"
        out_dir = root / "out"
        latent_path = root / "ae_latents.npz"
        cycle = ["A", "B", "C", "D"]
        tokens = [[cycle[index % len(cycle)]] for index in range(36)]
        write_tiny_trail(data_root, tokens, np.repeat(np.arange(3), 12))
        train_main(
            [
                "--data_root",
                str(data_root),
                "--out_dir",
                str(out_dir),
                "--sequence_length",
                "3",
                "--max_tokens_per_frame",
                "2",
                "--batch_size",
                "4",
                "--epochs",
                "1",
                "--d_model",
                "16",
                "--hidden_dim",
                "16",
                "--bottleneck_dim",
                "8",
                "--dropout",
                "0",
                "--num_workers",
                "0",
                "--device",
                "cpu",
                "--limit_train_samples",
                "8",
                "--limit_val_samples",
                "4",
                "--extract_data_root",
                str(data_root),
                "--extract_out_npz",
                str(latent_path),
                "--limit_extract_samples",
                "3",
            ]
        )
        with np.load(latent_path) as cache:
            assert cache["z"].shape == (3, 3, 16, 1, 1)
            assert cache["semantics"].shape == (3, 3, 5)
            assert cache["semantic_logits"].shape == cache["semantics"].shape
            assert cache["positions"].shape == (3, 3, 3)
        assert (out_dir / "last.ckpt").exists()
    print("PASS: tiny same-frame AE training and latent export produced valid artifacts")


if __name__ == "__main__":
    main()
