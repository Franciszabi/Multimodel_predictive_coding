"""
Train SemanticGPT (pure semantic sequence model).

Data: set DATA_ROOT (and optionally VAL_DATA_ROOT) in the config below. Same structure as a02/a95:
  DATA_ROOT/path0/objects_map.npy, positions.npy, *.png
  DATA_ROOT/path1/...
  vocabulary.npy at DATA_ROOT.parent / "vocabulary.npy" (object names, order = columns of objects_map).
  objects_map.npy shape (num_frames, V). Token_ids/token_mask are built with subword tokenization:
  each frame's active objects -> object names -> split by '_' (e.g. prop_bed_08 -> prop, bed, 08) -> subword ids.

Usage:
  python a97_train_semantic_gpt.py

Latent saving: set SAVE_LATENTS_EVERY in config. Saves to out_dir/latents_epoch{N}.npz.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from src.models.semantic_gpt import SemanticGPT


# --------------- Script config (edit here) ---------------
DATA_ROOT = Path("/home/ubuntu/project/data/data_11272025_twinmansion/data_11272025_100000samples")  # or str
VAL_DATA_ROOT: Optional[Path] = Path("/home/ubuntu/project/data/data_11272025_twinmansion/data_11272025_50000samples") # None = split from DATA_ROOT 90/10
VOCABULARY_NPY: Optional[Path] = None  # None = DATA_ROOT.parent / "vocabulary.npy"
OUT_DIR = Path("experiments/semantic_gpt")
SEQUENCE_LENGTH = 25
MAX_TOKENS_PER_FRAME = 16
PAD_TOKEN_ID = 0

BATCH_SIZE = 32
EPOCHS = 80
LR = 1e-3
D_MODEL = 256
NUM_LAYERS = 4
NUM_HEADS = 4
# VOCAB_SIZE and NUM_CLASSES are inferred from data (objects_map shape + subword vocab built from vocabulary.npy)
SAVE_LATENTS_EVERY = 0   # save latents every N epochs (0 = never)
SAVE_LATENTS_BATCHES = 10
EARLY_STOPPING_PATIENCE = 3   # stop if val loss does not improve for this many epochs; 0 = disabled
# ---------------


def load_vocabulary(data_root: Path) -> Tuple[int, np.ndarray]:
    """Load vocabulary.npy (object names); V = len(vocabulary). Path: VOCABULARY_NPY or DATA_ROOT.parent / 'vocabulary.npy'."""
    vocab_path = Path(VOCABULARY_NPY) if VOCABULARY_NPY is not None else Path(data_root).parent / "vocabulary.npy"
    if not vocab_path.exists():
        raise FileNotFoundError(f"Vocabulary not found at {vocab_path}. Set VOCABULARY_NPY or place vocabulary.npy in DATA_ROOT.parent.")
    vocabulary = np.load(vocab_path, allow_pickle=True)
    V = len(vocabulary)
    return V, vocabulary


class SubwordTokenizer:
    """
    Subword tokenizer from object names: split by '_' (e.g. prop_bed_08 -> ['prop','bed','08']).
    Id 0 = PAD, 1 = UNK. Vocab = [PAD, UNK] + sorted(unique subwords).
    """

    def __init__(self, object_names: np.ndarray):
        object_names = np.atleast_1d(object_names)
        names = [str(n).strip() for n in object_names]
        subwords = set()
        for n in names:
            subwords.update(n.split("_"))
        subwords.discard("")
        self._subword_list = ["<PAD>", "<UNK>"] + sorted(subwords)
        self._stoi = {s: i for i, s in enumerate(self._subword_list)}
        self.pad_id = 0
        self.unk_id = 1

    @property
    def vocab_size(self) -> int:
        return len(self._subword_list)

    def encode_name(self, name: str) -> List[int]:
        parts = str(name).strip().split("_")
        return [self._stoi.get(p, self.unk_id) for p in parts if p]

    def encode_frame(self, active_indices: np.ndarray, vocabulary: np.ndarray) -> List[int]:
        """
        Encode one frame to a single list of subword ids.
        Order: 按物体在 vocabulary 中的下标顺序，依次把每个物体名拆成 subword 再拼接。
        即 [物体 active_indices[0] 的 subwords, 物体 active_indices[1] 的 subwords, ...]，
        没有做排列组合，同一帧内多个物体的 subword 序列按下标顺序首尾相接。
        """
        ids = []
        for i in active_indices:
            name = vocabulary[int(i)]
            ids.extend(self.encode_name(name))
        return ids


class SemanticSequenceDatasetFromPaths(Dataset):
    """
    Load from existing data layout: DATA_ROOT/path0/, path1/ ... with objects_map.npy per path.
    Builds token_ids (L, K), token_mask (L, K), targets (L, V):
    - Subword tokenization: each frame's active objects -> object names (vocabulary) -> split by '_' -> subword ids.
    - targets: multi-hot (L, V) unchanged.
    """

    def __init__(
        self,
        root: Path,
        vocabulary: np.ndarray,
        subword_tokenizer: SubwordTokenizer,
        sequence_length: int,
        max_tokens_per_frame: int = 16,
        pad_token_id: int = 0,
        objmap_name: str = "objects_map.npy",
    ):
        self.sequence_length = sequence_length
        self.max_tokens_per_frame = max_tokens_per_frame
        self.pad_token_id = pad_token_id
        self.objmap_name = objmap_name
        self.vocabulary = np.atleast_1d(vocabulary)
        self.tokenizer = subword_tokenizer
        self.episodes: list = []
        self.sequence_map: list = []

        episode_paths = sorted(
            [p for p in Path(root).iterdir() if p.is_dir()],
            key=lambda x: int(x.name.replace("path", "")) if x.name.replace("path", "").isdigit() else 0,
        )
        if not episode_paths:
            raise FileNotFoundError(f"No episode directories found in {root}")
        print(f"[SemanticSequenceDatasetFromPaths] Found {len(episode_paths)} episodes at {root}")
        for episode_idx, path in enumerate(episode_paths):
            semantics_data = np.load(path / objmap_name, allow_pickle=True)
            if semantics_data.ndim != 2:
                raise ValueError(f"Expected {objmap_name}.npy shape (num_frames, V), got ndim={semantics_data.ndim}")
            num_frames = len(semantics_data)
            if num_frames < sequence_length:
                continue
            self.episodes.append({objmap_name: semantics_data})
            num_sequences_in_episode = num_frames // sequence_length
            for i in range(num_sequences_in_episode):
                self.sequence_map.append((episode_idx, i * sequence_length))
        if not self.episodes:
            raise ValueError(f"No valid episodes in {root}")
        print(f"[SemanticSequenceDatasetFromPaths] Indexing complete. {len(self.sequence_map)} sequences.")

    def __len__(self) -> int:
        return len(self.sequence_map)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        episode_idx, start_index = self.sequence_map[idx]
        L = self.sequence_length
        semantic = self.episodes[episode_idx][self.objmap_name][start_index : start_index + L]
        semantic = np.asarray(semantic, dtype=np.float32)
        V = semantic.shape[1]
        K = self.max_tokens_per_frame
        token_ids = np.full((L, K), self.pad_token_id, dtype=np.int64)
        token_mask = np.zeros((L, K), dtype=np.bool_)
        for t in range(L):
            active = np.where(semantic[t] > 0.5)[0]
            if len(active) == 0:
                continue
            subword_ids = self.tokenizer.encode_frame(active, self.vocabulary)
            subword_ids = subword_ids[:K]
            n_valid = len(subword_ids)
            token_ids[t, :n_valid] = subword_ids
            token_mask[t, :n_valid] = True
        targets = torch.from_numpy(semantic)
        return torch.from_numpy(token_ids), torch.from_numpy(token_mask), targets


class SyntheticSemanticDataset(Dataset):
    """Synthetic (token_ids, token_mask, targets) for testing. No real tokenization."""

    def __init__(
        self,
        num_samples: int,
        L: int,
        K: int,
        vocab_size: int,
        num_classes: int,
        pad_idx: int = 0,
        seed: Optional[int] = None,
    ):
        self.L = L
        self.K = K
        self.vocab_size = vocab_size
        self.num_classes = num_classes
        self.pad_idx = pad_idx
        self.num_samples = num_samples
        if seed is not None:
            np.random.seed(seed)
        # Pre-generate random data (could be replaced by loading from disk)
        self._token_ids = np.random.randint(1, vocab_size, (num_samples, L, K), dtype=np.int64)
        self._token_ids[:, :, -2:] = pad_idx  # simulate padding: last 2 slots often pad
        n_valid = np.random.randint(K // 2, K + 1, (num_samples, L))
        for b in range(num_samples):
            for t in range(L):
                self._token_ids[b, t, n_valid[b, t] :] = pad_idx
        self._targets = (np.random.rand(num_samples, L, num_classes) > 0.7).astype(np.float32)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        token_ids = torch.from_numpy(self._token_ids[idx].copy())
        token_mask = token_ids != self.pad_idx
        targets = torch.from_numpy(self._targets[idx].copy())
        return token_ids, token_mask, targets


def train_one_epoch(
    model: SemanticGPT,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    n = 0
    for token_ids, token_mask, targets in loader:
        token_ids = token_ids.to(device)
        token_mask = token_mask.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        logits = model(token_ids, token_mask)
        loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="mean")
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * token_ids.size(0)
        n += token_ids.size(0)
    return total_loss / max(n, 1)


@torch.no_grad()
def eval_loss(model: SemanticGPT, loader: DataLoader, device: torch.device) -> float:
    """Compute mean binary cross-entropy loss on the given loader (e.g. val_loader)."""
    model.eval()
    total_loss = 0.0
    n = 0
    for token_ids, token_mask, targets in loader:
        token_ids = token_ids.to(device)
        token_mask = token_mask.to(device)
        targets = targets.to(device)
        logits = model(token_ids, token_mask)
        loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="mean")
        total_loss += loss.item() * token_ids.size(0)
        n += token_ids.size(0)
    return total_loss / max(n, 1)


@torch.no_grad()
def eval_and_save_latents(
    model: SemanticGPT,
    loader: DataLoader,
    device: torch.device,
    max_batches: int,
    out_path: Path,
) -> None:
    model.eval()
    latents_list = []
    for bi, (token_ids, token_mask, _) in enumerate(loader):
        if bi >= max_batches:
            break
        token_ids = token_ids.to(device)
        token_mask = token_mask.to(device)
        _, latents = model(token_ids, token_mask, return_latents=True)
        latents_list.append(latents["final"].cpu().numpy())
    if not latents_list:
        return
    final = np.concatenate(latents_list, axis=0)  # (T, L, D)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, final=final)
    print(f"  Saved latents to {out_path} shape {final.shape}")


def main() -> None:
    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"
    )
    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    # V = len(vocabulary); vocabulary from vocabulary.npy
    V, vocabulary = load_vocabulary(Path(DATA_ROOT))
    NUM_CLASSES = V
    subword_tokenizer = SubwordTokenizer(vocabulary)
    VOCAB_SIZE = subword_tokenizer.vocab_size
    print(f"[config] NUM_CLASSES (V)={NUM_CLASSES} from objects_map; VOCAB_SIZE (subword)={VOCAB_SIZE}")

    full_ds = SemanticSequenceDatasetFromPaths(
        root=Path(DATA_ROOT),
        vocabulary=vocabulary,
        subword_tokenizer=subword_tokenizer,
        sequence_length=SEQUENCE_LENGTH,
        max_tokens_per_frame=MAX_TOKENS_PER_FRAME,
        pad_token_id=PAD_TOKEN_ID,
    )
    n = len(full_ds)
    if VAL_DATA_ROOT is not None:
        val_ds = SemanticSequenceDatasetFromPaths(
            root=Path(VAL_DATA_ROOT),
            vocabulary=vocabulary,
            subword_tokenizer=subword_tokenizer,
            sequence_length=SEQUENCE_LENGTH,
            max_tokens_per_frame=MAX_TOKENS_PER_FRAME,
            pad_token_id=PAD_TOKEN_ID,
        )
        train_ds = full_ds
    else:
        train_n = int(0.9 * n)
        train_ds = torch.utils.data.Subset(full_ds, range(0, train_n))
        val_ds = torch.utils.data.Subset(full_ds, range(train_n, n))
    print(f"Train sequences: {len(train_ds)}, Val sequences: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model = SemanticGPT(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        L=SEQUENCE_LENGTH,
        K=MAX_TOKENS_PER_FRAME,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        num_classes=NUM_CLASSES,
        padding_idx=PAD_TOKEN_ID,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss = eval_loss(model, val_loader, device)
        print(f"Epoch {epoch} train_loss={train_loss:.4f} val_loss={val_loss:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), out_dir / "best.ckpt")
        else:
            epochs_without_improvement += 1
        if SAVE_LATENTS_EVERY and (epoch % SAVE_LATENTS_EVERY == 0 or epoch == EPOCHS):
            eval_and_save_latents(
                model,
                val_loader,
                device,
                SAVE_LATENTS_BATCHES,
                out_dir / f"latents_epoch{epoch}.npz",
            )
        if EARLY_STOPPING_PATIENCE and epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping: val loss did not improve for {EARLY_STOPPING_PATIENCE} epochs.")
            break
    print(f"Done. Best val_loss {best_val_loss:.4f}. Checkpoint: {out_dir / 'best.ckpt'}")


if __name__ == "__main__":
    main()
