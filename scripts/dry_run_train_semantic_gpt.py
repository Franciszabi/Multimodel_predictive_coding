"""Run one tiny CPU training epoch and verify checkpoint/config output."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.smoke_utils import write_tiny_trail
from semantic_gpt_latent import main as latent_main
from train_semantic_gpt import main as train_main


def main() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        data_root = root / "trail_dry_run"
        out_dir = root / "out"
        token_cycle = ["A", "B", "C", "D"]
        tokens = [[token_cycle[index % len(token_cycle)]] for index in range(36)]
        episodes = np.repeat(np.arange(3), 12)
        write_tiny_trail(data_root, tokens, episodes)
        train_main(
            [
                "--data_root",
                str(data_root),
                "--out_dir",
                str(out_dir),
                "--sequence_length",
                "3",
                "--horizon",
                "1",
                "--max_tokens_per_frame",
                "2",
                "--batch_size",
                "4",
                "--epochs",
                "1",
                "--d_model",
                "16",
                "--num_layers",
                "1",
                "--num_heads",
                "4",
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
            ]
        )
        assert (out_dir / "config.json").exists()
        assert (out_dir / "last.ckpt").exists()
        assert (out_dir / "semantic_vocab.json").exists()
        latent_path = root / "latents.npz"
        latent_main(
            [
                "--data_root",
                str(data_root),
                "--ckpt",
                str(out_dir / "best.ckpt"),
                "--out_npz",
                str(latent_path),
                "--device",
                "cpu",
                "--limit_samples",
                "3",
            ]
        )
        with np.load(latent_path) as cache:
            assert cache["z"].shape == (3, 3, 16, 1, 1)
            assert cache["semantics_input"].shape == cache["semantics_target"].shape
            assert np.all(
                cache["frame_indices_target"] == cache["frame_indices_input"] + 1
            )
            assert cache["image_paths_input"].shape == (3, 3)
            assert cache["image_paths_target"].shape == (3, 3)
    print("PASS: tiny training and shifted latent export produced valid artifacts")


if __name__ == "__main__":
    main()
