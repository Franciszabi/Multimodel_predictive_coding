"""Print raw semantic strings and their atomic tokenizer representation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.semantic_tokenizer import build_atomic_tokenizer
from src.data.trail_semantic_dataset import load_semantic_jsonl


def resolve_path(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect raw trail semantics before and after atomic tokenization."
    )
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--vocab_path", default="")
    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument("--num_frames", type=int, default=5)
    parser.add_argument("--max_tokens_per_frame", type=int, default=16)
    parser.add_argument("--sequence_length", type=int, default=3)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--show_vocab_limit", type=int, default=50)
    parser.add_argument(
        "--include_empty_token",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    data_root = resolve_path(args.data_root)
    vocab_path = resolve_path(args.vocab_path) if args.vocab_path else None
    tokenizer, source = build_atomic_tokenizer(
        data_root,
        vocab_path=vocab_path,
        include_empty_token=args.include_empty_token,
    )
    frames = load_semantic_jsonl(
        data_root / "semantics.jsonl",
        include_empty_token=args.include_empty_token,
    )
    if args.start_frame < 0 or args.start_frame >= len(frames):
        raise ValueError(
            f"start_frame must be in [0, {len(frames) - 1}], got {args.start_frame}"
        )

    print(f"data_root: {data_root}")
    print(f"tokenizer_type: {tokenizer.tokenizer_type}")
    print(f"vocabulary_source: {source}")
    print(
        f"semantic_classes: {tokenizer.num_classes} | input_vocab_size: "
        f"{tokenizer.vocab_size} (PAD=0, UNK=1)"
    )
    print("\nclass_index -> input_token_id -> atomic_token")
    limit = min(max(args.show_vocab_limit, 0), tokenizer.num_classes)
    for class_index, token in enumerate(tokenizer.tokens[:limit]):
        print(f"{class_index:4d} -> {class_index + 2:4d} -> {token}")
    if limit < tokenizer.num_classes:
        print(f"... {tokenizer.num_classes - limit} more classes")

    stop = min(len(frames), args.start_frame + max(args.num_frames, 0))
    for frame_index in range(args.start_frame, stop):
        raw_tokens = frames[frame_index]
        token_ids, token_mask = tokenizer.encode_frame(
            raw_tokens, args.max_tokens_per_frame
        )
        valid_ids = token_ids[token_mask]
        decoded = [tokenizer.decode_token_id(int(token_id)) for token_id in valid_ids]
        multihot = tokenizer.frame_to_multihot(raw_tokens)
        active_classes = np.flatnonzero(multihot).tolist()
        active_labels = [tokenizer.tokens[index] for index in active_classes]
        print(f"\nframe {frame_index}")
        print(f"  raw_tokens:          {raw_tokens}")
        print(f"  token_ids_valid:     {valid_ids.tolist()}")
        print(f"  token_ids_padded:    {token_ids.tolist()}")
        print(f"  token_mask:          {token_mask.astype(np.int8).tolist()}")
        print(f"  decoded_valid_ids:   {decoded}")
        print(f"  multihot_classes:    {active_classes}")
        print(f"  multihot_labels:     {active_labels}")

    input_start = args.start_frame
    input_stop = input_start + args.sequence_length
    target_start = input_start + args.horizon
    target_stop = target_start + args.sequence_length
    if target_stop <= len(frames):
        print("\nwindow alignment")
        print(f"  input frame indices:  {list(range(input_start, input_stop))}")
        print(f"  target frame indices: {list(range(target_start, target_stop))}")
        print(
            "  relation: latent at each input frame t predicts target frame t+h; "
            "frame-causal attention blocks access to frames after t."
        )
    else:
        print("\nwindow alignment: not enough remaining frames for requested L+h")


if __name__ == "__main__":
    main()
