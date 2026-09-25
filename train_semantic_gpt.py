"""Train SemanticGPT as a causal future-semantic prediction model."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from src.data.semantic_tokenizer import build_atomic_tokenizer
from src.data.trail_semantic_dataset import TrailSemanticSequenceDataset
from src.models.semantic_gpt import ATTENTION_MASK_MODES, SemanticGPT
from src.training_progress import EpochTimer, batch_progress, timing_summary


REPO_ROOT = Path(__file__).resolve().parent


def resolve_path(value: str | Path, base: Path = REPO_ROOT) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def resolve_device(requested: str) -> torch.device:
    requested = requested.strip().lower()
    if requested in {"", "auto"}:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false.")
    return device


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def limited_dataset(dataset: Dataset, limit: int) -> Dataset:
    if limit <= 0 or limit >= len(dataset):
        return dataset
    return Subset(dataset, range(limit))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train frame-causal SemanticGPT to predict future semantic frames."
    )
    parser.add_argument("--data_root", default="data/trail_latest")
    parser.add_argument("--val_data_root", default="")
    parser.add_argument("--out_dir", default="experiments/semantic_gpt")
    parser.add_argument("--vocab_path", default="")
    parser.add_argument("--sequence_length", type=int, default=25)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--max_tokens_per_frame", type=int, default=16)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no_progress", action="store_true", help="Hide batch progress bars; keep epoch summaries")
    parser.add_argument("--early_stopping_patience", type=int, default=3)
    parser.add_argument("--limit_train_samples", type=int, default=0)
    parser.add_argument("--limit_val_samples", type=int, default=0)
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument(
        "--attention_mask_mode",
        choices=ATTENTION_MASK_MODES,
        default="frame_causal",
    )
    parser.add_argument(
        "--include_empty_token",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args(argv)


def make_datasets(
    args: argparse.Namespace,
    tokenizer,
    data_root: Path,
    val_data_root: Path | None,
) -> tuple[TrailSemanticSequenceDataset, TrailSemanticSequenceDataset]:
    common = dict(
        tokenizer=tokenizer,
        sequence_length=args.sequence_length,
        horizon=args.horizon,
        max_tokens_per_frame=args.max_tokens_per_frame,
        stride=args.stride,
        include_actions=False,
        return_metadata=False,
        include_empty_token=args.include_empty_token,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )
    if val_data_root is not None:
        train_dataset = TrailSemanticSequenceDataset(
            data_root=data_root,
            split_policy="all",
            **common,
        )
        val_dataset = TrailSemanticSequenceDataset(
            data_root=val_data_root,
            split_policy="all",
            **common,
        )
    else:
        train_dataset = TrailSemanticSequenceDataset(
            data_root=data_root,
            split_policy="train",
            **common,
        )
        val_dataset = TrailSemanticSequenceDataset(
            data_root=data_root,
            split_policy="val",
            **common,
        )
    return train_dataset, val_dataset


def train_one_epoch(
    model: SemanticGPT,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *, progress: bool = True, description: str = "Train",
) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0
    with batch_progress(loader, desc=description, enabled=progress) as batches:
        for token_ids, token_mask, future_targets in batches:
            token_ids = token_ids.to(device, non_blocking=True)
            token_mask = token_mask.to(device, non_blocking=True)
            future_targets = future_targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(token_ids, token_mask)
            loss = F.binary_cross_entropy_with_logits(logits, future_targets)
            loss.backward()
            optimizer.step()

            batch_size = token_ids.shape[0]
            total_loss += float(loss.item()) * batch_size
            total_samples += batch_size
            batches.set_postfix(loss=f"{total_loss / total_samples:.6f}", refresh=False)
    return total_loss / max(total_samples, 1)


@torch.no_grad()
def evaluate(
    model: SemanticGPT, loader: DataLoader, device: torch.device,
    *, progress: bool = True, description: str = "Val",
) -> float:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    with batch_progress(loader, desc=description, enabled=progress) as batches:
        for token_ids, token_mask, future_targets in batches:
            token_ids = token_ids.to(device, non_blocking=True)
            token_mask = token_mask.to(device, non_blocking=True)
            future_targets = future_targets.to(device, non_blocking=True)
            logits = model(token_ids, token_mask)
            loss = F.binary_cross_entropy_with_logits(logits, future_targets)
            batch_size = token_ids.shape[0]
            total_loss += float(loss.item()) * batch_size
            total_samples += batch_size
            batches.set_postfix(loss=f"{total_loss / total_samples:.6f}", refresh=False)
    return total_loss / max(total_samples, 1)


def checkpoint_payload(
    model: SemanticGPT,
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


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.horizon < 1:
        raise ValueError("SemanticGPT is future-prediction only: --horizon must be >= 1.")
    if args.d_model % args.num_heads:
        raise ValueError("--d_model must be divisible by --num_heads")

    set_seed(args.seed)
    data_root = resolve_path(args.data_root)
    val_data_root = resolve_path(args.val_data_root) if args.val_data_root else None
    out_dir = resolve_path(args.out_dir)
    vocab_path = resolve_path(args.vocab_path) if args.vocab_path else None
    if not data_root.exists():
        raise FileNotFoundError(f"data_root not found: {data_root}")
    if val_data_root is not None and not val_data_root.exists():
        raise FileNotFoundError(f"val_data_root not found: {val_data_root}")
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer, vocab_source = build_atomic_tokenizer(
        data_root,
        vocab_path=vocab_path,
        include_empty_token=args.include_empty_token,
    )
    tokenizer.save(out_dir / "semantic_vocab.json")
    train_dataset_raw, val_dataset_raw = make_datasets(
        args, tokenizer, data_root, val_data_root
    )
    train_dataset = limited_dataset(train_dataset_raw, args.limit_train_samples)
    val_dataset = limited_dataset(val_dataset_raw, args.limit_val_samples)

    device = resolve_device(args.device)
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
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
        "tokenizer_type": tokenizer.tokenizer_type,
        "state_used_as_input": False,
        "actions_used_as_input": False,
        "target_relation": "input[t] -> semantics[t+h]",
    }
    with (out_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    model = SemanticGPT(
        vocab_size=tokenizer.vocab_size,
        d_model=args.d_model,
        L=args.sequence_length,
        K=args.max_tokens_per_frame,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        num_classes=tokenizer.num_classes,
        padding_idx=tokenizer.pad_id,
        dropout=args.dropout,
        attention_mask_mode=args.attention_mask_mode,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    print(f"[setup] data_root={data_root}")
    print(f"[setup] out_dir={out_dir}")
    print(
        f"[setup] frames={train_dataset_raw.num_frames} "
        f"episodes={train_dataset_raw.num_episodes}"
    )
    print(
        f"[setup] L={args.sequence_length} horizon={args.horizon} stride={args.stride} "
        f"train_samples={len(train_dataset)} val_samples={len(val_dataset)}"
    )
    print(
        f"[setup] semantic_vocab={tokenizer.num_classes} tokenizer=atomic "
        f"source={vocab_source}"
    )
    print(
        f"[setup] attention_mask={args.attention_mask_mode} device={device} "
        "state_input=false actions_input=false"
    )

    train_history: list[float] = []
    val_history: list[float] = []
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    log_path = out_dir / "train_log.jsonl"
    log_path.write_text("", encoding="utf-8")

    timer = EpochTimer(args.epochs)
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device,
                                     progress=not args.no_progress, description=f"Train {epoch}/{args.epochs}")
        val_loss = evaluate(model, val_loader, device,
                            progress=not args.no_progress, description=f"Val {epoch}/{args.epochs}")
        train_history.append(train_loss)
        val_history.append(val_loss)

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        payload = checkpoint_payload(
            model, optimizer, config, tokenizer, epoch, best_val_loss
        )
        torch.save(payload, out_dir / "last.ckpt")
        if improved:
            torch.save(payload, out_dir / "best.ckpt")

        np.save(out_dir / "train_loss.npy", np.asarray(train_history, dtype=np.float32))
        np.save(out_dir / "val_loss.npy", np.asarray(val_history, dtype=np.float32))
        timing = timer.finish_epoch(epoch)
        log_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "best_val_loss": best_val_loss,
            **timing,
        }
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(log_record) + "\n")
        print(
            f"[epoch {epoch:03d}] train_loss={train_loss:.6f} "
            f"val_loss={val_loss:.6f} {timing_summary(timing)}",
            flush=True,
        )

        if (
            args.early_stopping_patience > 0
            and epochs_without_improvement >= args.early_stopping_patience
        ):
            print(
                "[early_stop] validation loss did not improve for "
                f"{args.early_stopping_patience} epochs"
            )
            break

    print(f"[done] best_val_loss={best_val_loss:.6f} checkpoint={out_dir / 'best.ckpt'}")


if __name__ == "__main__":
    main()
