"""Train the non-temporal, same-frame semantic autoencoder baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from src.data.semantic_tokenizer import build_atomic_tokenizer
from src.data.trail_semantic_dataset import TrailSemanticSequenceDataset
from src.models.semantic_autoencoder import SemanticFrameAutoencoder
from src.training_progress import EpochTimer, batch_progress, timing_summary
from train_semantic_gpt import limited_dataset, resolve_device, resolve_path, set_seed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a frame-wise current-semantic reconstruction baseline."
    )
    parser.add_argument("--data_root", default="data/trail_latest")
    parser.add_argument("--val_data_root", default="")
    parser.add_argument("--out_dir", default="experiments/semantic_ae")
    parser.add_argument("--vocab_path", default="")
    parser.add_argument("--sequence_length", type=int, default=25)
    parser.add_argument("--max_tokens_per_frame", type=int, default=16)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--bottleneck_dim", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no_progress", action="store_true", help="Hide batch progress bars; keep epoch summaries")
    parser.add_argument("--early_stopping_patience", type=int, default=3)
    parser.add_argument("--limit_train_samples", type=int, default=0)
    parser.add_argument("--limit_val_samples", type=int, default=0)
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument(
        "--include_empty_token",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--extract_data_root", default="")
    parser.add_argument("--extract_out_npz", default="")
    parser.add_argument("--extract_batch_size", type=int, default=64)
    parser.add_argument("--limit_extract_samples", type=int, default=0)
    parser.add_argument("--latent_source", choices=("pooled", "bottleneck"), default="pooled")
    return parser.parse_args(argv)


def flatten_frames(
    token_ids: torch.Tensor,
    token_mask: torch.Tensor,
    current_targets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Flatten time into batch; this baseline has no temporal information."""

    batch_size, length, width = token_ids.shape
    classes = current_targets.shape[-1]
    return (
        token_ids.reshape(batch_size * length, width),
        token_mask.reshape(batch_size * length, width),
        current_targets.reshape(batch_size * length, classes),
    )


def make_datasets(
    args: argparse.Namespace,
    tokenizer,
    data_root: Path,
    val_data_root: Path | None,
) -> tuple[TrailSemanticSequenceDataset, TrailSemanticSequenceDataset]:
    # horizon=0 is intentional: AE reconstructs only the current frame.
    common = dict(
        tokenizer=tokenizer,
        sequence_length=args.sequence_length,
        horizon=0,
        max_tokens_per_frame=args.max_tokens_per_frame,
        stride=args.stride,
        include_actions=False,
        return_metadata=False,
        include_empty_token=args.include_empty_token,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )
    if val_data_root is not None:
        return (
            TrailSemanticSequenceDataset(data_root, split_policy="all", **common),
            TrailSemanticSequenceDataset(val_data_root, split_policy="all", **common),
        )
    return (
        TrailSemanticSequenceDataset(data_root, split_policy="train", **common),
        TrailSemanticSequenceDataset(data_root, split_policy="val", **common),
    )


def run_epoch(
    model: SemanticFrameAutoencoder,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    *, progress: bool = True, description: str | None = None,
) -> float:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_frames = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context, batch_progress(loader, desc=description or ("Train" if training else "Val"), enabled=progress) as batches:
        for token_ids, token_mask, current_targets in batches:
            token_ids = token_ids.to(device, non_blocking=True)
            token_mask = token_mask.to(device, non_blocking=True)
            current_targets = current_targets.to(device, non_blocking=True)
            token_ids, token_mask, current_targets = flatten_frames(
                token_ids, token_mask, current_targets
            )
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            logits = model(token_ids, token_mask)
            loss = F.binary_cross_entropy_with_logits(logits, current_targets)
            if optimizer is not None:
                loss.backward()
                optimizer.step()
            total_loss += float(loss.item()) * token_ids.shape[0]
            total_frames += token_ids.shape[0]
            batches.set_postfix(loss=f"{total_loss / total_frames:.6f}", refresh=False)
    return total_loss / max(total_frames, 1)


def checkpoint_payload(
    model: SemanticFrameAutoencoder,
    optimizer: torch.optim.Optimizer,
    config: dict,
    tokenizer,
    epoch: int,
    best_val_loss: float,
) -> dict:
    return {
        "format_version": 2,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
        "semantic_vocab": tokenizer.to_dict(),
        "epoch": int(epoch),
        "best_val_loss": float(best_val_loss),
    }


@torch.no_grad()
def extract_latents(
    model: SemanticFrameAutoencoder,
    tokenizer,
    args: argparse.Namespace,
    device: torch.device,
    data_root: Path,
    out_npz: Path,
) -> None:
    dataset = TrailSemanticSequenceDataset(
        data_root=data_root,
        tokenizer=tokenizer,
        sequence_length=args.sequence_length,
        horizon=0,
        max_tokens_per_frame=args.max_tokens_per_frame,
        stride=args.stride,
        split_policy="all",
        include_actions=True,
        return_metadata=True,
        include_empty_token=args.include_empty_token,
    )
    dataset_for_loader: Dataset = dataset
    if 0 < args.limit_extract_samples < len(dataset):
        dataset_for_loader = Subset(dataset, range(args.limit_extract_samples))
    loader = DataLoader(
        dataset_for_loader,
        batch_size=args.extract_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    outputs: dict[str, list[np.ndarray]] = {
        "z": [],
        "semantics": [],
        "semantic_logits": [],
        "frame_indices_input": [],
        "episode_ids": [],
    }
    model.eval()
    for token_ids, token_mask, current_targets, metadata in loader:
        batch_size, length, width = token_ids.shape
        ids_flat = token_ids.to(device).reshape(batch_size * length, width)
        mask_flat = token_mask.to(device).reshape(batch_size * length, width)
        logits_flat, latents = model(ids_flat, mask_flat, return_latents=True)
        feature = latents[args.latent_source]
        channels = feature.shape[-1]
        outputs["z"].append(
            feature.reshape(batch_size, length, channels).cpu().numpy()[:, :, :, None, None]
        )
        outputs["semantics"].append(current_targets.numpy())
        outputs["semantic_logits"].append(
            logits_flat.reshape(batch_size, length, -1).cpu().numpy()
        )
        outputs["frame_indices_input"].append(metadata["frame_indices_input"].numpy())
        outputs["episode_ids"].append(metadata["episode_id"].numpy())
        if "state_input" in metadata:
            outputs.setdefault("state_input", []).append(metadata["state_input"].numpy())

    saved = {key: np.concatenate(value, axis=0) for key, value in outputs.items()}
    if "state_input" in saved:
        saved["positions"] = saved["state_input"]
        saved["state_columns"] = np.asarray(["x", "z", "yaw"])
    saved["latent_source"] = np.asarray(args.latent_source)
    saved["tokenizer_type"] = np.asarray("atomic")
    saved["semantic_tokens"] = np.asarray(tokenizer.tokens)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, **saved)
    print(f"[extract] saved={out_npz} z={saved['z'].shape}")


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    set_seed(args.seed)
    data_root = resolve_path(args.data_root)
    val_data_root = resolve_path(args.val_data_root) if args.val_data_root else None
    out_dir = resolve_path(args.out_dir)
    vocab_path = resolve_path(args.vocab_path) if args.vocab_path else None
    if not data_root.exists():
        raise FileNotFoundError(f"data_root not found: {data_root}")
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer, vocab_source = build_atomic_tokenizer(
        data_root,
        vocab_path=vocab_path,
        include_empty_token=args.include_empty_token,
    )
    tokenizer.save(out_dir / "semantic_vocab.json")
    train_raw, val_raw = make_datasets(args, tokenizer, data_root, val_data_root)
    train_dataset = limited_dataset(train_raw, args.limit_train_samples)
    val_dataset = limited_dataset(val_raw, args.limit_val_samples)
    device = resolve_device(args.device)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    model = SemanticFrameAutoencoder(
        vocab_size=tokenizer.vocab_size,
        d_model=args.d_model,
        num_classes=tokenizer.num_classes,
        bottleneck_dim=args.bottleneck_dim,
        hidden_dim=args.hidden_dim,
        padding_idx=tokenizer.pad_id,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    config = {
        **vars(args),
        "data_root": str(data_root),
        "val_data_root": str(val_data_root) if val_data_root else "",
        "out_dir": str(out_dir),
        "vocab_path": str(vocab_path) if vocab_path else "",
        "vocab_source": vocab_source,
        "vocab_size": tokenizer.vocab_size,
        "num_classes": tokenizer.num_classes,
        "tokenizer_type": "atomic",
        "horizon": 0,
        "objective": "same_frame_reconstruction",
        "state_used_as_input": False,
        "actions_used_as_input": False,
    }
    with (out_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    print(f"[setup] data_root={data_root} out_dir={out_dir}")
    print(
        f"[setup] frames={train_raw.num_frames} episodes={train_raw.num_episodes} "
        f"train_samples={len(train_dataset)} val_samples={len(val_dataset)}"
    )
    print(
        "[setup] objective=same_frame_reconstruction temporal_input=false "
        "state_input=false actions_input=false tokenizer=atomic"
    )

    train_history: list[float] = []
    val_history: list[float] = []
    best_val_loss = float("inf")
    stale_epochs = 0
    timer = EpochTimer(args.epochs)
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, device, optimizer,
                              progress=not args.no_progress, description=f"Train {epoch}/{args.epochs}")
        val_loss = run_epoch(model, val_loader, device, None,
                            progress=not args.no_progress, description=f"Val {epoch}/{args.epochs}")
        train_history.append(train_loss)
        val_history.append(val_loss)
        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            stale_epochs = 0
        else:
            stale_epochs += 1
        payload = checkpoint_payload(
            model, optimizer, config, tokenizer, epoch, best_val_loss
        )
        torch.save(payload, out_dir / "last.ckpt")
        if improved:
            torch.save(payload, out_dir / "best.ckpt")
        np.save(out_dir / "train_loss.npy", np.asarray(train_history, dtype=np.float32))
        np.save(out_dir / "val_loss.npy", np.asarray(val_history, dtype=np.float32))
        timing = timer.finish_epoch(epoch)
        print(
            f"[epoch {epoch:03d}] train_loss={train_loss:.6f} "
            f"val_loss={val_loss:.6f} {timing_summary(timing)}",
            flush=True,
        )
        if args.early_stopping_patience > 0 and stale_epochs >= args.early_stopping_patience:
            print(f"[early_stop] no validation improvement for {stale_epochs} epochs")
            break

    if args.extract_data_root:
        extract_root = resolve_path(args.extract_data_root)
        extract_out = (
            resolve_path(args.extract_out_npz)
            if args.extract_out_npz
            else out_dir / "semantic_ae_latents.npz"
        )
        try:
            checkpoint = torch.load(
                out_dir / "best.ckpt", map_location=device, weights_only=False
            )
        except TypeError:
            checkpoint = torch.load(out_dir / "best.ckpt", map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        extract_latents(model, tokenizer, args, device, extract_root, extract_out)
    print(f"[done] best_val_loss={best_val_loss:.6f} checkpoint={out_dir / 'best.ckpt'}")


if __name__ == "__main__":
    main()
