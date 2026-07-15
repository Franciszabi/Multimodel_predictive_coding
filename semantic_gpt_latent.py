"""Export frame-causal SemanticGPT latents on a Unity trail dataset."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from src.data.semantic_tokenizer import AtomicSemanticTokenizer, build_atomic_tokenizer
from src.data.trail_semantic_dataset import TrailSemanticSequenceDataset
from src.models.semantic_gpt import SemanticGPT
from train_semantic_gpt import REPO_ROOT, resolve_device, resolve_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export SemanticGPT trail latents.")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out_npz", required=True)
    parser.add_argument("--vocab_path", default="")
    parser.add_argument("--sequence_length", type=int, default=None)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--max_tokens_per_frame", type=int, default=None)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit_samples", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    return parser.parse_args(argv)


def load_checkpoint(path: Path, device: torch.device) -> tuple[dict, dict, dict | None]:
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Unsupported checkpoint object in {path}")
    if "model_state_dict" in checkpoint:
        return (
            checkpoint["model_state_dict"],
            checkpoint.get("config", {}),
            checkpoint.get("semantic_vocab"),
        )
    return checkpoint, {}, None


def tokenizer_for_checkpoint(
    args: argparse.Namespace,
    data_root: Path,
    ckpt_path: Path,
    checkpoint_vocab: dict | None,
) -> AtomicSemanticTokenizer:
    if args.vocab_path:
        return AtomicSemanticTokenizer.load(resolve_path(args.vocab_path))
    saved_vocab = ckpt_path.parent / "semantic_vocab.json"
    if saved_vocab.exists():
        return AtomicSemanticTokenizer.load(saved_vocab)
    if checkpoint_vocab is not None:
        tokens = checkpoint_vocab.get("tokens")
        if isinstance(tokens, list):
            return AtomicSemanticTokenizer(tokens)
    tokenizer, _ = build_atomic_tokenizer(data_root)
    return tokenizer


def infer_layer_count(state_dict: dict[str, torch.Tensor]) -> int:
    layer_ids = {
        int(key.split(".")[1])
        for key in state_dict
        if key.startswith("blocks.") and key.split(".")[1].isdigit()
    }
    return max(layer_ids) + 1 if layer_ids else 0


def collated_paths_to_batch(value: Any, batch_size: int) -> np.ndarray:
    array = np.asarray(value, dtype=str)
    if array.ndim == 2 and array.shape[1] == batch_size:
        array = array.T
    return array


@torch.no_grad()
def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    data_root = resolve_path(args.data_root)
    ckpt_path = resolve_path(args.ckpt)
    out_npz = resolve_path(args.out_npz)
    if not data_root.exists():
        raise FileNotFoundError(f"data_root not found: {data_root}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

    device = resolve_device(args.device)
    state_dict, config, checkpoint_vocab = load_checkpoint(ckpt_path, device)
    tokenizer = tokenizer_for_checkpoint(
        args, data_root, ckpt_path, checkpoint_vocab
    )

    sequence_length = args.sequence_length or int(config.get("sequence_length", 25))
    horizon = args.horizon or int(config.get("horizon", 1))
    max_tokens = args.max_tokens_per_frame or int(
        config.get("max_tokens_per_frame", 16)
    )
    d_model = int(config.get("d_model", state_dict["embed.weight"].shape[1]))
    num_layers = int(config.get("num_layers", infer_layer_count(state_dict)))
    num_heads = int(config.get("num_heads", 4))
    dropout = float(config.get("dropout", 0.1))
    attention_mask_mode = str(config.get("attention_mask_mode", "frame_causal"))

    expected_vocab_size = int(state_dict["embed.weight"].shape[0])
    expected_classes = int(state_dict["head.weight"].shape[0])
    if tokenizer.vocab_size != expected_vocab_size or tokenizer.num_classes != expected_classes:
        raise ValueError(
            "Tokenizer/checkpoint mismatch: "
            f"tokenizer vocab={tokenizer.vocab_size}, classes={tokenizer.num_classes}; "
            f"checkpoint vocab={expected_vocab_size}, classes={expected_classes}. "
            "Use the semantic_vocab.json saved beside the checkpoint."
        )

    dataset = TrailSemanticSequenceDataset(
        data_root=data_root,
        tokenizer=tokenizer,
        sequence_length=sequence_length,
        horizon=horizon,
        max_tokens_per_frame=max_tokens,
        stride=args.stride,
        split_policy="all",
        include_actions=True,
        return_metadata=True,
    )
    dataset_for_loader = dataset
    if 0 < args.limit_samples < len(dataset):
        dataset_for_loader = Subset(dataset, range(args.limit_samples))
    loader = DataLoader(
        dataset_for_loader,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = SemanticGPT(
        vocab_size=tokenizer.vocab_size,
        d_model=d_model,
        L=sequence_length,
        K=max_tokens,
        num_layers=num_layers,
        num_heads=num_heads,
        num_classes=tokenizer.num_classes,
        padding_idx=tokenizer.pad_id,
        dropout=dropout,
        attention_mask_mode=attention_mask_mode,
    ).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    arrays: dict[str, list[np.ndarray]] = {
        "z": [],
        "semantics_input": [],
        "semantics_target": [],
        "semantic_logits": [],
        "frame_indices_input": [],
        "frame_indices_target": [],
        "episode_ids": [],
    }
    optional_keys = (
        "state_input",
        "state_target",
        "actions_input",
        "actions_target",
        "unity_frames_input",
        "unity_frames_target",
        "image_paths_input",
        "image_paths_target",
    )

    for token_ids, token_mask, targets, metadata in loader:
        token_ids = token_ids.to(device, non_blocking=True)
        token_mask = token_mask.to(device, non_blocking=True)
        logits, latents = model(token_ids, token_mask, return_latents=True)
        z = latents["final"].cpu().numpy()[:, :, :, None, None]
        arrays["z"].append(z)
        arrays["semantics_input"].append(metadata["semantics_input"].numpy())
        arrays["semantics_target"].append(targets.numpy())
        arrays["semantic_logits"].append(logits.cpu().numpy())
        arrays["frame_indices_input"].append(metadata["frame_indices_input"].numpy())
        arrays["frame_indices_target"].append(metadata["frame_indices_target"].numpy())
        arrays["episode_ids"].append(metadata["episode_id"].numpy())

        batch_size = token_ids.shape[0]
        for key in optional_keys:
            if key not in metadata:
                continue
            arrays.setdefault(key, [])
            value = metadata[key]
            if torch.is_tensor(value):
                arrays[key].append(value.numpy())
            else:
                arrays[key].append(collated_paths_to_batch(value, batch_size))

    saved = {key: np.concatenate(values, axis=0) for key, values in arrays.items() if values}
    saved["semantics"] = saved["semantics_input"]
    if "state_input" in saved:
        saved["positions"] = saved["state_input"]
        saved["state_columns"] = np.asarray(["x", "z", "yaw"])
    saved["horizon"] = np.asarray(horizon, dtype=np.int64)
    saved["tokenizer_type"] = np.asarray("atomic")
    saved["semantic_tokens"] = np.asarray(tokenizer.tokens)

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, **saved)
    print(f"[model] checkpoint={ckpt_path}")
    print(
        f"[model] L={sequence_length} horizon={horizon} mask={attention_mask_mode} "
        f"tokenizer=atomic"
    )
    print(f"[save] {out_npz}")
    print(
        f"[save] z={saved['z'].shape} input={saved['semantics_input'].shape} "
        f"target={saved['semantics_target'].shape}"
    )


if __name__ == "__main__":
    main()
