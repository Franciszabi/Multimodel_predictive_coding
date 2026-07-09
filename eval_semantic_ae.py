"""
Evaluate frame-wise Semantic Autoencoder checkpoint on multi-label metrics.

Usage example:
  python eval_semantic_ae.py \
    --ckpt experiments/semantic_ae/best.ckpt \
    --data_root /home/ubuntu/project/data/data_11272025_twinmansion/data_11272025_50000samples \
    --threshold 0.5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from train_semantic_ae import (
    BOTTLENECK_DIM,
    D_MODEL,
    DROPOUT,
    HIDDEN_DIM,
    MAX_TOKENS_PER_FRAME,
    NUM_WORKERS,
    PAD_TOKEN_ID,
    PredefinedPathSemanticDatasetWithPos,
    SEQUENCE_LENGTH,
)
from train_semantic_gpt import (
    SemanticSequenceDatasetFromPaths,
    SubwordTokenizer,
    load_vocabulary,
)
from src.models.semantic_autoencoder import SemanticFrameAutoencoder


def flatten_frames(
    token_ids: torch.Tensor,
    token_mask: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    b, l, k = token_ids.shape
    v = targets.shape[-1]
    return (
        token_ids.reshape(b * l, k),
        token_mask.reshape(b * l, k),
        targets.reshape(b * l, v),
    )


def safe_div(numerator: float, denominator: float, eps: float = 1e-12) -> float:
    return float(numerator) / float(denominator + eps)


@torch.no_grad()
def evaluate(
    model: SemanticFrameAutoencoder,
    loader: DataLoader,
    device: torch.device,
    threshold: float,
) -> dict[str, float]:
    model.eval()
    thr = float(threshold)

    total_loss = 0.0
    n_frames = 0

    # Micro counters
    tp_micro = 0.0
    fp_micro = 0.0
    fn_micro = 0.0
    tn_micro = 0.0

    # Macro counters by class
    tp_c = None
    fp_c = None
    fn_c = None

    exact_match = 0.0
    hamming_correct = 0.0
    total_labels = 0.0

    for batch in loader:
        # path-mode dataset yields (token_ids, token_mask, targets)
        # grid/predefined dataset yields (token_ids, token_mask, targets, positions)
        if len(batch) == 3:
            token_ids, token_mask, targets = batch
        elif len(batch) == 4:
            token_ids, token_mask, targets, _ = batch
        else:
            raise ValueError(f"Unexpected batch size: len(batch)={len(batch)}")
        token_ids = token_ids.to(device)
        token_mask = token_mask.to(device)
        targets = targets.to(device)

        token_ids_f, token_mask_f, targets_f = flatten_frames(token_ids, token_mask, targets)
        logits = model(token_ids_f, token_mask_f)
        probs = torch.sigmoid(logits)
        preds = probs >= thr
        gt = targets_f >= 0.5

        loss = F.binary_cross_entropy_with_logits(logits, targets_f, reduction="mean")
        total_loss += loss.item() * token_ids_f.size(0)
        n_frames += token_ids_f.size(0)

        # Per-element confusion
        tp = (preds & gt).sum().item()
        fp = (preds & (~gt)).sum().item()
        fn = ((~preds) & gt).sum().item()
        tn = ((~preds) & (~gt)).sum().item()
        tp_micro += tp
        fp_micro += fp
        fn_micro += fn
        tn_micro += tn

        # Per-class confusion
        tp_batch_c = (preds & gt).sum(dim=0).cpu().numpy()
        fp_batch_c = (preds & (~gt)).sum(dim=0).cpu().numpy()
        fn_batch_c = ((~preds) & gt).sum(dim=0).cpu().numpy()
        if tp_c is None:
            tp_c = tp_batch_c.astype(np.float64)
            fp_c = fp_batch_c.astype(np.float64)
            fn_c = fn_batch_c.astype(np.float64)
        else:
            tp_c += tp_batch_c
            fp_c += fp_batch_c
            fn_c += fn_batch_c

        # Exact-match accuracy (frame-level all labels correct)
        exact_match += (preds == gt).all(dim=1).sum().item()

        # Hamming accuracy (label-level)
        hamming_correct += (preds == gt).sum().item()
        total_labels += float(preds.numel())

    val_loss = safe_div(total_loss, n_frames)

    precision_micro = safe_div(tp_micro, tp_micro + fp_micro)
    recall_micro = safe_div(tp_micro, tp_micro + fn_micro)
    f1_micro = safe_div(2.0 * precision_micro * recall_micro, precision_micro + recall_micro)

    precision_c = tp_c / (tp_c + fp_c + 1e-12)
    recall_c = tp_c / (tp_c + fn_c + 1e-12)
    f1_c = 2.0 * precision_c * recall_c / (precision_c + recall_c + 1e-12)

    return {
        "val_loss_bce": val_loss,
        "precision_micro": float(precision_micro),
        "recall_micro": float(recall_micro),
        "f1_micro": float(f1_micro),
        "precision_macro": float(np.mean(precision_c)),
        "recall_macro": float(np.mean(recall_c)),
        "f1_macro": float(np.mean(f1_c)),
        "exact_match_acc": safe_div(exact_match, n_frames),
        "hamming_acc": safe_div(hamming_correct, total_labels),
        "hamming_loss": 1.0 - safe_div(hamming_correct, total_labels),
        "n_frames": float(n_frames),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate SemanticFrameAutoencoder checkpoint.")
    ap.add_argument("--ckpt", type=str, required=True, help="Path to model checkpoint (.ckpt)")
    ap.add_argument("--data_root", type=str, required=True, help="Dataset root containing objects_map.npy")
    ap.add_argument("--batch_size", type=int, default=64, help="Eval batch size")
    ap.add_argument("--threshold", type=float, default=0.5, help="Sigmoid threshold for binary labels")
    ap.add_argument("--device", type=str, default="", help="cuda or cpu; empty=auto")
    ap.add_argument("--sequence_length", type=int, default=SEQUENCE_LENGTH, help="Sequence length L")
    ap.add_argument("--max_tokens_per_frame", type=int, default=MAX_TOKENS_PER_FRAME, help="Token cap K")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    ckpt = Path(args.ckpt)
    if not data_root.exists():
        raise FileNotFoundError(f"data_root not found: {data_root}")
    if not ckpt.exists():
        raise FileNotFoundError(f"ckpt not found: {ckpt}")

    if args.device.strip():
        device = torch.device(args.device.strip())
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    num_classes, vocabulary = load_vocabulary(data_root)
    tokenizer = SubwordTokenizer(vocabulary)
    vocab_size = tokenizer.vocab_size

    # Auto-detect dataset layout:
    # - grid/predefined mode: data_root contains objects_map.npy + positions.npy
    # - path mode: data_root contains path*/objects_map.npy
    has_objmap = (data_root / "objects_map.npy").exists()
    has_positions = (data_root / "positions.npy").exists()
    if has_objmap and has_positions:
        dataset_mode = "grid_predefined"
        ds = PredefinedPathSemanticDatasetWithPos(
            root=data_root,
            vocabulary=vocabulary,
            subword_tokenizer=tokenizer,
            sequence_length=int(args.sequence_length),
            max_tokens_per_frame=int(args.max_tokens_per_frame),
            pad_token_id=PAD_TOKEN_ID,
            thr=0.5,
        )
    else:
        dataset_mode = "paths"
        ds = SemanticSequenceDatasetFromPaths(
            root=data_root,
            vocabulary=vocabulary,
            subword_tokenizer=tokenizer,
            sequence_length=int(args.sequence_length),
            max_tokens_per_frame=int(args.max_tokens_per_frame),
            pad_token_id=PAD_TOKEN_ID,
            objmap_name="objects_map.npy",
        )
    loader = DataLoader(
        ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    model = SemanticFrameAutoencoder(
        vocab_size=vocab_size,
        d_model=D_MODEL,
        num_classes=num_classes,
        bottleneck_dim=BOTTLENECK_DIM,
        hidden_dim=HIDDEN_DIM,
        padding_idx=PAD_TOKEN_ID,
        dropout=DROPOUT,
    ).to(device)

    state = torch.load(ckpt, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=True)

    metrics = evaluate(model, loader, device, threshold=float(args.threshold))
    print(f"[eval] ckpt={ckpt}")
    print(f"[eval] data_root={data_root}")
    print(f"[eval] dataset_mode={dataset_mode}")
    print(f"[eval] threshold={args.threshold}")
    for k in [
        "val_loss_bce",
        "precision_micro",
        "recall_micro",
        "f1_micro",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "exact_match_acc",
        "hamming_acc",
        "hamming_loss",
        "n_frames",
    ]:
        print(f"{k}: {metrics[k]:.6f}")


if __name__ == "__main__":
    main()

