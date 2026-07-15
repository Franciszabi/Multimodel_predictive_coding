"""Verify shifted semantic windows and episode boundaries."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.smoke_utils import write_tiny_trail
from src.data.semantic_tokenizer import AtomicSemanticTokenizer
from src.data.trail_semantic_dataset import TrailSemanticSequenceDataset


def decoded_frames(dataset, token_ids, token_mask) -> list[str]:
    return [
        dataset.tokenizer.decode_token_id(int(token_ids[t][token_mask[t]][0]))
        for t in range(token_ids.shape[0])
    ]


def target_frames(tokenizer, targets) -> list[str]:
    return [tokenizer.tokens[int(frame.argmax())] for frame in targets.numpy()]


def main() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir) / "trail_smoke"
        frame_tokens = [[token] for token in ["A", "B", "C", "D", "A", "B", "C", "D"]]
        write_tiny_trail(root, frame_tokens, np.asarray([0] * 4 + [1] * 4))
        tokenizer = AtomicSemanticTokenizer(["A", "B", "C", "D"])

        h1 = TrailSemanticSequenceDataset(
            root,
            tokenizer,
            sequence_length=3,
            horizon=1,
            max_tokens_per_frame=2,
            stride=1,
        )
        input_ids, input_mask, targets = h1[0]
        assert decoded_frames(h1, input_ids, input_mask) == ["A", "B", "C"]
        assert target_frames(tokenizer, targets) == ["B", "C", "D"]
        assert len(h1) == 2

        h2 = TrailSemanticSequenceDataset(
            root,
            tokenizer,
            sequence_length=2,
            horizon=2,
            max_tokens_per_frame=2,
            stride=1,
        )
        input_ids, input_mask, targets = h2[0]
        assert decoded_frames(h2, input_ids, input_mask) == ["A", "B"]
        assert target_frames(tokenizer, targets) == ["C", "D"]
        assert all(
            h2.episode_codes[start] == h2.episode_codes[start + 3]
            for start in h2.sequence_map
        )
    print("PASS: horizon shifts are correct and windows do not cross episodes")


if __name__ == "__main__":
    main()
