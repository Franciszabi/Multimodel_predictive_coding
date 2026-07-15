"""Evaluate an atomic-token semantic autoencoder on same-frame reconstruction."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from src.data.semantic_tokenizer import AtomicSemanticTokenizer, build_atomic_tokenizer
from src.data.trail_semantic_dataset import TrailSemanticSequenceDataset
from src.models.semantic_autoencoder import SemanticFrameAutoencoder
from train_semantic_ae import flatten_frames
from train_semantic_gpt import resolve_device, resolve_path


def safe_div(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator + 1e-12)


@torch.no_grad()
def evaluate(model, loader, device, threshold: float) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    frame_count = 0
    tp = fp = fn = 0.0
    exact = hamming_correct = total_labels = 0.0
    tp_class = fp_class = fn_class = None

    for token_ids, token_mask, targets in loader:
        token_ids, token_mask, targets = flatten_frames(
            token_ids.to(device), token_mask.to(device), targets.to(device)
        )
        logits = model(token_ids, token_mask)
        predictions = torch.sigmoid(logits) >= threshold
        truth = targets >= 0.5
        loss = F.binary_cross_entropy_with_logits(logits, targets)
        total_loss += float(loss.item()) * len(targets)
        frame_count += len(targets)

        tp += float((predictions & truth).sum().item())
        fp += float((predictions & ~truth).sum().item())
        fn += float((~predictions & truth).sum().item())
        batch_tp = (predictions & truth).sum(0).cpu().numpy().astype(np.float64)
        batch_fp = (predictions & ~truth).sum(0).cpu().numpy().astype(np.float64)
        batch_fn = (~predictions & truth).sum(0).cpu().numpy().astype(np.float64)
        if tp_class is None:
            tp_class, fp_class, fn_class = batch_tp, batch_fp, batch_fn
        else:
            tp_class += batch_tp
            fp_class += batch_fp
            fn_class += batch_fn
        exact += float((predictions == truth).all(1).sum().item())
        hamming_correct += float((predictions == truth).sum().item())
        total_labels += float(predictions.numel())

    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    class_precision = tp_class / (tp_class + fp_class + 1e-12)
    class_recall = tp_class / (tp_class + fn_class + 1e-12)
    class_f1 = 2 * class_precision * class_recall / (
        class_precision + class_recall + 1e-12
    )
    return {
        "bce": safe_div(total_loss, frame_count),
        "precision_micro": precision,
        "recall_micro": recall,
        "f1_micro": safe_div(2 * precision * recall, precision + recall),
        "f1_macro": float(np.mean(class_f1)),
        "exact_match": safe_div(exact, frame_count),
        "hamming_accuracy": safe_div(hamming_correct, total_labels),
        "n_frames": float(frame_count),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--vocab_path", default="")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--sequence_length", type=int, default=None)
    parser.add_argument("--max_tokens_per_frame", type=int, default=None)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--limit_samples", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    data_root = resolve_path(args.data_root)
    ckpt_path = resolve_path(args.ckpt)
    device = resolve_device(args.device)
    try:
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    config = checkpoint.get("config", {})

    if args.vocab_path:
        tokenizer = AtomicSemanticTokenizer.load(resolve_path(args.vocab_path))
    elif (ckpt_path.parent / "semantic_vocab.json").exists():
        tokenizer = AtomicSemanticTokenizer.load(ckpt_path.parent / "semantic_vocab.json")
    elif isinstance(checkpoint.get("semantic_vocab"), dict):
        tokenizer = AtomicSemanticTokenizer(checkpoint["semantic_vocab"]["tokens"])
    else:
        tokenizer, _ = build_atomic_tokenizer(data_root)

    sequence_length = args.sequence_length or int(config.get("sequence_length", 25))
    max_tokens = args.max_tokens_per_frame or int(
        config.get("max_tokens_per_frame", 16)
    )
    dataset = TrailSemanticSequenceDataset(
        data_root=data_root,
        tokenizer=tokenizer,
        sequence_length=sequence_length,
        horizon=0,
        max_tokens_per_frame=max_tokens,
        stride=args.stride,
        split_policy="all",
    )
    dataset_for_loader = dataset
    if 0 < args.limit_samples < len(dataset):
        dataset_for_loader = Subset(dataset, range(args.limit_samples))
    loader = DataLoader(
        dataset_for_loader,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    model = SemanticFrameAutoencoder(
        vocab_size=tokenizer.vocab_size,
        d_model=int(config.get("d_model", state_dict["embed.weight"].shape[1])),
        num_classes=tokenizer.num_classes,
        bottleneck_dim=int(
            config.get("bottleneck_dim", state_dict["encoder.3.weight"].shape[0])
        ),
        hidden_dim=int(config.get("hidden_dim", state_dict["encoder.0.weight"].shape[0])),
        padding_idx=tokenizer.pad_id,
        dropout=float(config.get("dropout", 0.1)),
    ).to(device)
    model.load_state_dict(state_dict, strict=True)
    metrics = evaluate(model, loader, device, args.threshold)
    print(f"[eval] checkpoint={ckpt_path} data_root={data_root}")
    for name, value in metrics.items():
        print(f"{name}: {value:.6f}")


if __name__ == "__main__":
    main()
