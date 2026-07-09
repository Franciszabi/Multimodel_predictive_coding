from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from src.models.semantic_gpt import SemanticGPT
from train_semantic_gpt import SubwordTokenizer, MAX_TOKENS_PER_FRAME, PAD_TOKEN_ID, D_MODEL, NUM_LAYERS, NUM_HEADS


# ============== 你需要改的参数 ==============
TEST_DIR = Path("/home/ubuntu/project/data/data_11272025_twinmansion/pre_defined_path_samples")

# 你的 semantic_gpt checkpoint（训练 L=多少就要填多少的 L）
CKPT_PATH = Path("/home/ubuntu/project/experiments/semantic_gpt/best.ckpt")

# 关键：必须等于你训练时的 SEQUENCE_LENGTH（例如你现在训练是 32，则填 32）
TEST_SEQUENCE_LENGTH = 25

OUT_NPZ = Path("/home/ubuntu/project/analysis_out/semantic_gpt_predefined_latents/latents_on_grid.npz")

# 词表（一般在 TEST_DIR.parent / "vocabulary.npy"）
VOCABULARY_NPY: Optional[Path] = None
# ===========================================


def _load_positions(root: Path, sequence_length: int) -> np.ndarray:
    pos = np.load(root / "positions.npy", allow_pickle=True)
    # 支持 (N,L,3) 或 (N*L,3)
    if pos.ndim == 3:
        return pos
    if pos.ndim == 2:
        n_total, d = pos.shape
        if n_total % sequence_length != 0:
            raise ValueError(f"positions length {n_total} not divisible by sequence_length {sequence_length}")
        n = n_total // sequence_length
        return pos.reshape(n, sequence_length, d)
    raise ValueError(f"Unexpected positions.npy ndim={pos.ndim}, shape={pos.shape}")

def reshape_to_seq(arr: np.ndarray, seq_len: int) -> np.ndarray:
    """
    将 positions/objects_map 统一 reshape 成 (N, seq_len, D)
    支持输入:
      - (N, L, D) 但 L 不等于 seq_len：会 flatten 后重切
      - (N*L, D) ：直接 reshape
    """
    seq_len = int(seq_len)
    if arr.ndim == 3:
        n, l, d = arr.shape
        if l == seq_len:
            return arr
        t = n * l
        if t % seq_len != 0:
            raise ValueError(f"Total length T={t} not divisible by seq_len={seq_len}")
        n_new = t // seq_len
        return arr.reshape(n_new, seq_len, d)
    elif arr.ndim == 2:
        t, d = arr.shape
        if t % seq_len != 0:
            raise ValueError(f"Total length T={t} not divisible by seq_len={seq_len}")
        n_new = t // seq_len
        return arr.reshape(n_new, seq_len, d)
    else:
        raise ValueError(f"Expected ndim 2 or 3, got ndim={arr.ndim} shape={arr.shape}")


class PredefinedPathSemanticDatasetWithPos(Dataset):
    """
    从 pre_defined_path_samples 目录提取 token_ids/token_mask/targets，并附带 positions。
    objects_map.npy: (N*L, V) 或 (N, L, V)
    positions.npy: (N, L, 3) 或 (N*L, 3)
    """
    def __init__(
        self,
        root: Path,
        vocabulary: np.ndarray,
        subword_tokenizer: SubwordTokenizer,
        sequence_length: int,
        max_tokens_per_frame: int = MAX_TOKENS_PER_FRAME,
        pad_token_id: int = PAD_TOKEN_ID,
        objects_map_fn: str = "objects_map.npy",
        positions_fn: str = "positions.npy",
        thr: float = 0.5,
    ):
        self.root = Path(root)
        self.vocabulary = np.atleast_1d(vocabulary)
        self.tokenizer = subword_tokenizer
        self.sequence_length = int(sequence_length)
        self.max_tokens_per_frame = int(max_tokens_per_frame)
        self.pad_token_id = int(pad_token_id)
        self.thr = float(thr)

        # objmap = np.load(self.root / objects_map_fn, allow_pickle=True)
        # if objmap.ndim == 3:
        #     self.objects_map = objmap
        # elif objmap.ndim == 2:
        #     n_total, v = objmap.shape
        #     if n_total % self.sequence_length != 0:
        #         raise ValueError(f"objects_map length {n_total} not divisible by sequence_length {self.sequence_length}")
        #     n = n_total // self.sequence_length
        #     self.objects_map = objmap.reshape(n, self.sequence_length, v)
        # else:
        #     raise ValueError(f"Expected objects_map (N,L,V) or (N*L,V), got shape={objmap.shape}")

        # self.positions = _load_positions(self.root, self.sequence_length)

        # if self.positions.shape[0] != self.objects_map.shape[0] or self.positions.shape[1] != self.objects_map.shape[1]:
        #     raise ValueError(f"positions shape {self.positions.shape} does not match objects_map {self.objects_map.shape}")

        # self.N, self.L, self.V = self.objects_map.shape

        objmap = np.load(self.root / objects_map_fn, allow_pickle=True)
        pos = np.load(self.root / positions_fn, allow_pickle=True)
        objmap = reshape_to_seq(objmap, sequence_length)  # -> (N, L, V)
        pos = reshape_to_seq(pos, sequence_length)         # -> (N, L, 3)
        if objmap.shape[0] != pos.shape[0] or objmap.shape[1] != pos.shape[1]:
            raise ValueError(f"After reshape, positions {pos.shape} != objects_map {objmap.shape}")
        self.objects_map = objmap
        self.positions = pos
        self.N, self.L, self.V = self.objects_map.shape
        # positions: (N,L,3) typically
        if self.positions.shape[-1] < 2:
            raise ValueError(f"positions last dim should be >=2, got {self.positions.shape[-1]}")

        print(f"[dataset] N={self.N}, L={self.L}, V={self.V}, positions={tuple(self.positions.shape)}")

    def __len__(self) -> int:
        return int(self.N)

    def __getitem__(self, idx: int):
        semantic = np.asarray(self.objects_map[idx], dtype=np.float32)  # (L,V)
        pos = np.asarray(self.positions[idx], dtype=np.float32)  # (L,3) usually

        L, V = semantic.shape
        K = self.max_tokens_per_frame

        token_ids = np.full((L, K), self.pad_token_id, dtype=np.int64)
        token_mask = np.zeros((L, K), dtype=np.bool_)

        for t in range(L):
            active = np.where(semantic[t] > self.thr)[0]
            if active.size == 0:
                continue

            # active -> object names -> split by '_' -> subword ids
            subword_ids = self.tokenizer.encode_frame(active, self.vocabulary)
            subword_ids = subword_ids[:K]
            n_valid = len(subword_ids)

            token_ids[t, :n_valid] = np.array(subword_ids, dtype=np.int64)
            token_mask[t, :n_valid] = True

        targets = torch.from_numpy(semantic)  # (L,V)
        return (
            torch.from_numpy(token_ids),        # (L,K)
            torch.from_numpy(token_mask),       # (L,K)
            targets,                            # (L,V)
            torch.from_numpy(pos),             # (L,3)
        )


@torch.no_grad()
def extract_and_cache():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    vocab_path = VOCABULARY_NPY if VOCABULARY_NPY is not None else TEST_DIR.parent / "vocabulary.npy"
    if not Path(vocab_path).exists():
        raise FileNotFoundError(f"vocabulary.npy not found at: {vocab_path}")

    vocabulary = np.load(vocab_path, allow_pickle=True)
    V = len(vocabulary)
    tokenizer = SubwordTokenizer(vocabulary)
    VOCAB_SIZE = tokenizer.vocab_size

    model = SemanticGPT(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        L=TEST_SEQUENCE_LENGTH,
        K=MAX_TOKENS_PER_FRAME,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        num_classes=V,
        padding_idx=PAD_TOKEN_ID,
    ).to(device)

    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=True)
    model.load_state_dict(ckpt, strict=True)
    model.eval()
    print(f"[model] loaded {CKPT_PATH}")

    ds = PredefinedPathSemanticDatasetWithPos(
        root=TEST_DIR,
        vocabulary=vocabulary,
        subword_tokenizer=tokenizer,
        sequence_length=TEST_SEQUENCE_LENGTH,
        max_tokens_per_frame=MAX_TOKENS_PER_FRAME,
        pad_token_id=PAD_TOKEN_ID,
    )
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)

    z_list = []
    pos_list = []

    # 可选：存 semantics / logits 方便后续你做 next-step 语义指标
    sem_list = []
    logits_list = []

    for token_ids, token_mask, targets, pos in loader:
        token_ids = token_ids.to(device)
        token_mask = token_mask.to(device)
        targets = targets.to(device)
        pos_np = pos.cpu().numpy()

        logits, latents = model(token_ids, token_mask, return_latents=True)
        # latents["final"]: (B,L,D)
        z = latents["final"].cpu().numpy()  # (B,L,D)
        # 扩成 (B,L,D,1,1) 以复用 (N,L,C,H,W) 的 placefield 代码接口
        z = z[:, :, :, None, None]         # (B,L,D,1,1)

        z_list.append(z)
        pos_list.append(pos_np)

        sem_list.append(targets.cpu().numpy())
        logits_list.append(logits.cpu().numpy())

    z_all = np.concatenate(z_list, axis=0)          # (N,L,D,1,1)
    positions_all = np.concatenate(pos_list, axis=0) # (N,L,3)
    semantics_all = np.concatenate(sem_list, axis=0) # (N,L,V)
    semantic_logits_all = np.concatenate(logits_list, axis=0) # (N,L,V)

    OUT_NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT_NPZ,
        z=z_all,
        positions=positions_all,
        semantics=semantics_all,
        semantic_logits=semantic_logits_all,
    )
    print(f"[save] {OUT_NPZ}")
    print(f"        z={z_all.shape} positions={positions_all.shape} semantics={semantics_all.shape}")


if __name__ == "__main__":
    extract_and_cache()
