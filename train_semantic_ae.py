"""
Train frame-wise Semantic Autoencoder (ablation: no temporal modeling).

This script reuses the existing semantic data layout and tokenizer from
`a97_train_semantic_gpt.py`, but changes training target to same-frame
reconstruction:
  - input:  token_ids, token_mask from current frame
  - target: current-frame multi-hot semantics

Crucially, time dimension is flattened into batch:
  (B, L, K) -> (B*L, K)
  (B, L, V) -> (B*L, V)

Usage:
  python train_semantic_ae.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from train_semantic_gpt import (
    SubwordTokenizer,
    SemanticSequenceDatasetFromPaths,
    load_vocabulary,
)
from src.models.semantic_autoencoder import SemanticFrameAutoencoder


# --------------- Script config (edit here) ---------------
DATA_ROOT = Path("/home/ubuntu/project/data/data_11272025_twinmansion/data_11272025_100000samples")  # or str
VAL_DATA_ROOT: Optional[Path] = Path("/home/ubuntu/project/data/data_11272025_twinmansion/data_11272025_50000samples") # None = split from DATA_ROOT 90/10
OUT_DIR = Path("experiments/semantic_ae")

SEQUENCE_LENGTH = 25
MAX_TOKENS_PER_FRAME = 16
PAD_TOKEN_ID = 0

BATCH_SIZE = 32
EPOCHS = 80
LR = 1e-3
WEIGHT_DECAY = 0.0

# model dims
D_MODEL = 256
BOTTLENECK_DIM = 64
HIDDEN_DIM = 256
DROPOUT = 0.1

EARLY_STOPPING_PATIENCE = 3
NUM_WORKERS = 0
TRAIN_VAL_SPLIT = 0.9

# latent extraction for downstream placefield / vocab_select
EXTRACT_LATENTS_AFTER_TRAIN = True
EXTRACT_DATA_ROOT = Path("/home/ubuntu/project/data/data_11272025_twinmansion/pre_defined_path_samples")
EXTRACT_OUT_NPZ = Path("/home/ubuntu/project/analysis_out/semantic_ae_predefined_latents/latents_on_grid_ae.npz")
EXTRACT_BATCH_SIZE = 64
LATENT_SOURCE = "pooled"  # "pooled" (recommended for fair compare) or "bottleneck"
# ---------------------------------------------------------


def flatten_frames(
    token_ids: torch.Tensor,    # (B,L,K)
    token_mask: torch.Tensor,   # (B,L,K)
    targets: torch.Tensor,      # (B,L,V)
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    b, l, k = token_ids.shape
    v = targets.shape[-1]
    token_ids_f = token_ids.reshape(b * l, k)
    token_mask_f = token_mask.reshape(b * l, k)
    targets_f = targets.reshape(b * l, v)
    return token_ids_f, token_mask_f, targets_f


def train_one_epoch(
    model: SemanticFrameAutoencoder,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    n_frames = 0

    for token_ids, token_mask, targets in loader:
        token_ids = token_ids.to(device)
        token_mask = token_mask.to(device)
        targets = targets.to(device)

        token_ids_f, token_mask_f, targets_f = flatten_frames(token_ids, token_mask, targets)

        optimizer.zero_grad(set_to_none=True)
        logits = model(token_ids_f, token_mask_f)  # (B*L, V)
        loss = F.binary_cross_entropy_with_logits(logits, targets_f, reduction="mean")
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * token_ids_f.size(0)
        n_frames += token_ids_f.size(0)

    return total_loss / max(n_frames, 1)


@torch.no_grad()
def eval_loss(
    model: SemanticFrameAutoencoder,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    n_frames = 0

    for token_ids, token_mask, targets in loader:
        token_ids = token_ids.to(device)
        token_mask = token_mask.to(device)
        targets = targets.to(device)

        token_ids_f, token_mask_f, targets_f = flatten_frames(token_ids, token_mask, targets)
        logits = model(token_ids_f, token_mask_f)
        loss = F.binary_cross_entropy_with_logits(logits, targets_f, reduction="mean")

        total_loss += loss.item() * token_ids_f.size(0)
        n_frames += token_ids_f.size(0)

    return total_loss / max(n_frames, 1)


def build_dataloaders() -> Tuple[DataLoader, DataLoader, int, int]:
    data_root = Path(DATA_ROOT)
    if not data_root.exists():
        raise FileNotFoundError(f"DATA_ROOT not found: {data_root}")

    v, vocabulary = load_vocabulary(data_root)
    tokenizer = SubwordTokenizer(vocabulary)
    vocab_size = tokenizer.vocab_size

    train_dataset = SemanticSequenceDatasetFromPaths(
        root=data_root,
        vocabulary=vocabulary,
        subword_tokenizer=tokenizer,
        sequence_length=SEQUENCE_LENGTH,
        max_tokens_per_frame=MAX_TOKENS_PER_FRAME,
        pad_token_id=PAD_TOKEN_ID,
        objmap_name="objects_map.npy",
    )

    if VAL_DATA_ROOT is not None:
        val_dataset: Dataset = SemanticSequenceDatasetFromPaths(
            root=Path(VAL_DATA_ROOT),
            vocabulary=vocabulary,
            subword_tokenizer=tokenizer,
            sequence_length=SEQUENCE_LENGTH,
            max_tokens_per_frame=MAX_TOKENS_PER_FRAME,
            pad_token_id=PAD_TOKEN_ID,
            objmap_name="objects_map.npy",
        )
    else:
        n_total = len(train_dataset)
        n_train = int(n_total * TRAIN_VAL_SPLIT)
        n_val = n_total - n_train
        g = torch.Generator().manual_seed(42)
        train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [n_train, n_val], generator=g)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    return train_loader, val_loader, vocab_size, v


def reshape_to_seq(arr: np.ndarray, seq_len: int) -> np.ndarray:
    """
    Normalize arrays to (N, seq_len, D) for either:
      - (N, L, D): if L != seq_len, flatten and re-chunk
      - (N*L, D): directly reshape
    """
    seq_len = int(seq_len)
    if arr.ndim == 3:
        n, l, d = arr.shape
        if l == seq_len:
            return arr
        t = n * l
        if t % seq_len != 0:
            raise ValueError(f"Total length T={t} not divisible by seq_len={seq_len}")
        return arr.reshape(t // seq_len, seq_len, d)
    if arr.ndim == 2:
        t, d = arr.shape
        if t % seq_len != 0:
            raise ValueError(f"Total length T={t} not divisible by seq_len={seq_len}")
        return arr.reshape(t // seq_len, seq_len, d)
    raise ValueError(f"Expected ndim 2 or 3, got shape={arr.shape}")


class PredefinedPathSemanticDatasetWithPos(Dataset):
    """
    Load from a single directory containing:
      - objects_map.npy: (N,L,V) or (N*L,V)
      - positions.npy:   (N,L,3) or (N*L,3)
    Build token_ids/token_mask and return aligned positions.
    """

    def __init__(
        self,
        root: Path,
        vocabulary: np.ndarray,
        subword_tokenizer: SubwordTokenizer,
        sequence_length: int,
        max_tokens_per_frame: int = MAX_TOKENS_PER_FRAME,
        pad_token_id: int = PAD_TOKEN_ID,
        thr: float = 0.5,
    ):
        self.root = Path(root)
        self.vocabulary = np.atleast_1d(vocabulary)
        self.tokenizer = subword_tokenizer
        self.sequence_length = int(sequence_length)
        self.max_tokens_per_frame = int(max_tokens_per_frame)
        self.pad_token_id = int(pad_token_id)
        self.thr = float(thr)

        objmap = np.load(self.root / "objects_map.npy", allow_pickle=True)
        pos = np.load(self.root / "positions.npy", allow_pickle=True)
        objmap = reshape_to_seq(objmap, self.sequence_length)  # (N,L,V)
        pos = reshape_to_seq(pos, self.sequence_length)        # (N,L,3)
        if objmap.shape[:2] != pos.shape[:2]:
            raise ValueError(f"After reshape mismatch: objects_map={objmap.shape}, positions={pos.shape}")
        self.objects_map = objmap.astype(np.float32)
        self.positions = pos.astype(np.float32)
        self.N, self.L, self.V = self.objects_map.shape

    def __len__(self) -> int:
        return int(self.N)

    def __getitem__(self, idx: int):
        semantic = self.objects_map[idx]   # (L,V)
        pos = self.positions[idx]          # (L,3)
        l, _ = semantic.shape
        k = self.max_tokens_per_frame

        token_ids = np.full((l, k), self.pad_token_id, dtype=np.int64)
        token_mask = np.zeros((l, k), dtype=np.bool_)

        for t in range(l):
            active = np.where(semantic[t] > self.thr)[0]
            if active.size == 0:
                continue
            ids = self.tokenizer.encode_frame(active, self.vocabulary)[:k]
            n_valid = len(ids)
            token_ids[t, :n_valid] = np.array(ids, dtype=np.int64)
            token_mask[t, :n_valid] = True

        return (
            torch.from_numpy(token_ids),    # (L,K)
            torch.from_numpy(token_mask),   # (L,K)
            torch.from_numpy(semantic),     # (L,V)
            torch.from_numpy(pos),          # (L,3)
        )


@torch.no_grad()
def extract_and_cache_latents(
    model: SemanticFrameAutoencoder,
    device: torch.device,
    vocabulary: np.ndarray,
    tokenizer: SubwordTokenizer,
) -> None:
    if LATENT_SOURCE not in {"pooled", "bottleneck"}:
        raise ValueError(f"LATENT_SOURCE must be 'pooled' or 'bottleneck', got {LATENT_SOURCE}")

    ds = PredefinedPathSemanticDatasetWithPos(
        root=Path(EXTRACT_DATA_ROOT),
        vocabulary=vocabulary,
        subword_tokenizer=tokenizer,
        sequence_length=SEQUENCE_LENGTH,
        max_tokens_per_frame=MAX_TOKENS_PER_FRAME,
        pad_token_id=PAD_TOKEN_ID,
    )
    loader = DataLoader(ds, batch_size=EXTRACT_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    z_list = []
    pos_list = []
    sem_list = []
    logits_list = []

    model.eval()
    for token_ids, token_mask, targets, pos in loader:
        token_ids = token_ids.to(device)      # (B,L,K)
        token_mask = token_mask.to(device)    # (B,L,K)
        targets = targets.to(device)          # (B,L,V)

        b, l, k = token_ids.shape
        v = targets.shape[-1]
        token_ids_f = token_ids.reshape(b * l, k)
        token_mask_f = token_mask.reshape(b * l, k)

        logits_f, lat = model(token_ids_f, token_mask_f, return_latents=True)
        feat_f = lat[LATENT_SOURCE]                  # (B*L, D_or_Z)
        c = feat_f.shape[-1]
        feat = feat_f.view(b, l, c).cpu().numpy()   # (B,L,C)
        z = feat[:, :, :, None, None]                # (B,L,C,1,1)

        logits = logits_f.view(b, l, v).cpu().numpy()    # (B,L,V)
        z_list.append(z)
        pos_list.append(pos.numpy())
        sem_list.append(targets.cpu().numpy())
        logits_list.append(logits)

    z_all = np.concatenate(z_list, axis=0)
    pos_all = np.concatenate(pos_list, axis=0)
    sem_all = np.concatenate(sem_list, axis=0)
    logits_all = np.concatenate(logits_list, axis=0)

    out_npz = Path(EXTRACT_OUT_NPZ)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz,
        z=z_all,
        positions=pos_all,
        semantics=sem_all,
        semantic_logits=logits_all,
        latent_source=np.array([LATENT_SOURCE]),
    )
    print(f"[extract] saved: {out_npz}")
    print(f"[extract] z={z_all.shape} positions={pos_all.shape} semantics={sem_all.shape}")


def main() -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, vocab_size, num_classes = build_dataloaders()
    _, vocabulary = load_vocabulary(Path(DATA_ROOT))
    tokenizer = SubwordTokenizer(vocabulary)

    model = SemanticFrameAutoencoder(
        vocab_size=vocab_size,
        d_model=D_MODEL,
        num_classes=num_classes,
        bottleneck_dim=BOTTLENECK_DIM,
        hidden_dim=HIDDEN_DIM,
        padding_idx=PAD_TOKEN_ID,
        dropout=DROPOUT,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    history_train = []
    history_val = []
    best_val = float("inf")
    patience = 0

    print(f"[setup] device={device} vocab_size={vocab_size} num_classes={num_classes}")
    print(f"[setup] train_batches={len(train_loader)} val_batches={len(val_loader)}")

    for epoch in range(1, EPOCHS + 1):
        tr = train_one_epoch(model, train_loader, optimizer, device)
        va = eval_loss(model, val_loader, device)
        history_train.append(tr)
        history_val.append(va)
        print(f"[epoch {epoch:03d}] train_loss={tr:.6f} val_loss={va:.6f}")

        torch.save(model.state_dict(), out_dir / "last.ckpt")
        if va < best_val:
            best_val = va
            patience = 0
            torch.save(model.state_dict(), out_dir / "best.ckpt")
        else:
            patience += 1
            if EARLY_STOPPING_PATIENCE > 0 and patience >= EARLY_STOPPING_PATIENCE:
                print(f"[early_stop] no val improvement for {patience} epochs")
                break

    np.save(out_dir / "train_loss.npy", np.array(history_train, dtype=np.float32))
    np.save(out_dir / "val_loss.npy", np.array(history_val, dtype=np.float32))
    if EXTRACT_LATENTS_AFTER_TRAIN:
        best_ckpt = out_dir / "best.ckpt"
        if best_ckpt.exists():
            state = torch.load(best_ckpt, map_location=device, weights_only=True)
            model.load_state_dict(state, strict=True)
        extract_and_cache_latents(model, device, vocabulary=vocabulary, tokenizer=tokenizer)
    print(f"[done] saved to {out_dir}")


if __name__ == "__main__":
    main()

