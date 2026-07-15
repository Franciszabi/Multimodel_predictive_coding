"""Dataset support for top-level Unity trail semantic trajectories."""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .semantic_tokenizer import AtomicSemanticTokenizer, DEFAULT_EMPTY_TOKEN


TOKEN_LIST_FIELDS = (
    "modelTokens",
    "model_tokens",
    "tokens",
    "visibleTokens",
    "visible_tokens",
    "semantic_tokens",
)
OBJECT_LIST_FIELDS = ("objects", "visibleObjects", "visible_objects")
OBJECT_TOKEN_FIELDS = (
    "modelToken",
    "model_token",
    "token",
    "semanticLabel",
    "semantic_label",
)


def _normalize_token_values(value: Any, line_number: int, field_name: str) -> list[str]:
    if isinstance(value, str):
        values: Sequence[Any] = [value]
    elif isinstance(value, list):
        values = value
    else:
        raise ValueError(
            f"semantics.jsonl line {line_number}: field '{field_name}' must be a list "
            f"of atomic token strings, got {type(value).__name__}."
        )

    tokens: list[str] = []
    for item in values:
        if isinstance(item, str):
            token = item.strip()
        elif isinstance(item, dict):
            nested = next((item.get(key) for key in OBJECT_TOKEN_FIELDS if key in item), None)
            if nested is None:
                raise ValueError(
                    f"semantics.jsonl line {line_number}: item in '{field_name}' has keys "
                    f"{sorted(item.keys())}, but no supported token field."
                )
            token = str(nested).strip()
        else:
            token = str(item).strip()
        if token:
            tokens.append(token)
    return list(dict.fromkeys(tokens))


def parse_semantic_record(record: dict[str, Any], line_number: int) -> list[str]:
    for field in TOKEN_LIST_FIELDS:
        if field in record:
            return _normalize_token_values(record[field], line_number, field)

    for field in OBJECT_LIST_FIELDS:
        if field not in record:
            continue
        objects = record[field]
        if not isinstance(objects, list):
            raise ValueError(
                f"semantics.jsonl line {line_number}: field '{field}' must be a list."
            )
        tokens: list[str] = []
        for object_index, item in enumerate(objects):
            if isinstance(item, str):
                token = item.strip()
            elif isinstance(item, dict):
                selected = next((item[key] for key in OBJECT_TOKEN_FIELDS if key in item), None)
                if selected is None:
                    raise ValueError(
                        f"semantics.jsonl line {line_number}: object {object_index} in "
                        f"'{field}' has keys {sorted(item.keys())}, but no supported token field."
                    )
                token = str(selected).strip()
            else:
                raise ValueError(
                    f"semantics.jsonl line {line_number}: object {object_index} in "
                    f"'{field}' must be a string or object."
                )
            if token:
                tokens.append(token)
        return list(dict.fromkeys(tokens))

    expected = [*TOKEN_LIST_FIELDS, *OBJECT_LIST_FIELDS]
    raise ValueError(
        f"semantics.jsonl line {line_number}: no recognized semantic token list. "
        f"Available keys: {sorted(record.keys())}. Expected one of: {expected}."
    )


def load_semantic_jsonl(
    path: str | Path,
    include_empty_token: bool = True,
    empty_token: str = DEFAULT_EMPTY_TOKEN,
) -> list[list[str]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Primary semantic source not found: {path}")
    frames: list[list[str]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                raise ValueError(f"semantics.jsonl line {line_number} is blank.")
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"semantics.jsonl line {line_number} is invalid JSON: {error.msg}"
                ) from error
            if not isinstance(record, dict):
                raise ValueError(
                    f"semantics.jsonl line {line_number} must contain a JSON object."
                )
            tokens = parse_semantic_record(record, line_number)
            if not tokens and include_empty_token:
                tokens = [empty_token]
            frames.append(tokens)
    if not frames:
        raise ValueError(f"No frames found in {path}")
    return frames


def _load_aligned_array(path: Path, frame_count: int, required: bool = False) -> np.ndarray | None:
    if not path.exists():
        if required:
            raise FileNotFoundError(path)
        return None
    array = np.load(path, allow_pickle=True)
    if array.ndim == 0 or len(array) != frame_count:
        raise ValueError(
            f"Alignment mismatch: {path.name} has shape {array.shape}, but semantics has "
            f"{frame_count} frames."
        )
    return array


def _load_image_paths(path: Path, frame_count: int) -> list[str] | None:
    if not path.exists():
        return None
    image_paths: list[str] = []
    path_fields = (
        "image_path",
        "imagePath",
        "relative_path",
        "path",
        "image",
        "filename",
        "file",
    )
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            record = json.loads(raw_line)
            if not isinstance(record, dict):
                raise ValueError(f"frame_index.jsonl line {line_number} must be an object.")
            folder = record.get("path") or record.get("folder") or record.get("path_dir")
            filename = record.get("filename") or record.get("image_filename")
            if folder and filename:
                image_path = str(Path(str(folder)) / str(filename))
            else:
                image_path = next((record[key] for key in path_fields if key in record), "")
            image_paths.append(str(image_path))
    if len(image_paths) != frame_count:
        raise ValueError(
            f"Alignment mismatch: frame_index.jsonl has {len(image_paths)} records, "
            f"but semantics has {frame_count} frames."
        )
    return image_paths


class TrailSemanticSequenceDataset(Dataset):
    """Sliding semantic windows that remain inside contiguous episode segments.

    For a start index ``s``, model inputs are frames ``s:s+L`` and targets are
    ``s+h:s+h+L``. With the frame-causal model mask, the latent at input frame
    ``t`` can only use frames up to ``t`` while predicting frame ``t+h``.
    """

    def __init__(
        self,
        data_root: str | Path,
        tokenizer: AtomicSemanticTokenizer,
        sequence_length: int,
        horizon: int = 1,
        max_tokens_per_frame: int = 16,
        stride: int = 1,
        split_policy: str = "all",
        include_actions: bool = False,
        return_metadata: bool = False,
        include_empty_token: bool = True,
        empty_token: str = DEFAULT_EMPTY_TOKEN,
        val_fraction: float = 0.1,
        seed: int = 42,
    ) -> None:
        self.data_root = Path(data_root).expanduser().resolve()
        self.tokenizer = tokenizer
        self.sequence_length = int(sequence_length)
        self.horizon = int(horizon)
        self.max_tokens_per_frame = int(max_tokens_per_frame)
        self.stride = int(stride)
        self.split_policy = str(split_policy)
        self.include_actions = bool(include_actions)
        self.return_metadata = bool(return_metadata)
        self.include_empty_token = bool(include_empty_token)
        self.empty_token = str(empty_token)
        self.val_fraction = float(val_fraction)
        self.seed = int(seed)

        if self.sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        if self.horizon < 0:
            raise ValueError("horizon must be >= 0")
        if self.stride <= 0:
            raise ValueError("stride must be positive")
        if self.split_policy not in {"all", "train", "val"}:
            raise ValueError("split_policy must be one of: all, train, val")
        if not 0.0 < self.val_fraction < 1.0:
            raise ValueError("val_fraction must be between 0 and 1")

        self.frame_tokens = load_semantic_jsonl(
            self.data_root / "semantics.jsonl",
            include_empty_token=self.include_empty_token,
            empty_token=self.empty_token,
        )
        self.num_frames = len(self.frame_tokens)

        episodes = _load_aligned_array(self.data_root / "episodes.npy", self.num_frames)
        if episodes is None:
            warnings.warn(
                "episodes.npy is missing; treating the entire trail as one episode. "
                "Train/validation splitting will fall back to sequence-level splitting.",
                stacklevel=2,
            )
            episodes = np.zeros(self.num_frames, dtype=np.int64)
            self.has_episode_file = False
        else:
            episodes = np.asarray(episodes).reshape(self.num_frames, -1)[:, 0]
            self.has_episode_file = True
        self.episode_values = episodes
        _, self.episode_codes = np.unique(episodes.astype(str), return_inverse=True)
        self.episode_codes = self.episode_codes.astype(np.int64)
        self.num_episodes = int(np.unique(self.episode_codes).size)

        self.state = _load_aligned_array(self.data_root / "state.npy", self.num_frames)
        self.actions = _load_aligned_array(self.data_root / "actions.npy", self.num_frames)
        self.unity_frames = _load_aligned_array(
            self.data_root / "unity_frames.npy", self.num_frames
        )
        self.image_paths = _load_image_paths(
            self.data_root / "frame_index.jsonl", self.num_frames
        )

        self.token_ids = np.full(
            (self.num_frames, self.max_tokens_per_frame),
            self.tokenizer.pad_id,
            dtype=np.int64,
        )
        self.token_mask = np.zeros_like(self.token_ids, dtype=np.bool_)
        self.semantics = np.zeros(
            (self.num_frames, self.tokenizer.num_classes), dtype=np.float32
        )
        unknown_tokens: set[str] = set()
        for frame_index, tokens in enumerate(self.frame_tokens):
            ids, mask = self.tokenizer.encode_frame(tokens, self.max_tokens_per_frame)
            self.token_ids[frame_index] = ids
            self.token_mask[frame_index] = mask
            self.semantics[frame_index] = self.tokenizer.frame_to_multihot(tokens)
            unknown_tokens.update(
                token for token in tokens if token not in self.tokenizer.token_to_class_index
            )
        if unknown_tokens:
            preview = sorted(unknown_tokens)[:10]
            warnings.warn(
                f"{len(unknown_tokens)} semantic tokens are absent from the vocabulary and "
                f"will use UNK as input with no target class. Examples: {preview}",
                stacklevel=2,
            )

        all_sequences = self._build_sequence_map()
        self.sequence_map = self._apply_split(all_sequences)
        if not self.sequence_map:
            raise ValueError(
                f"No valid {self.split_policy} sequences in {self.data_root}. Need at least "
                f"L+h={self.sequence_length + self.horizon} contiguous frames per episode."
            )
        self.sample_episode_ids = np.asarray(
            [self.episode_codes[start] for start in self.sequence_map], dtype=np.int64
        )

    def _build_sequence_map(self) -> list[int]:
        starts: list[int] = []
        run_start = 0
        for index in range(1, self.num_frames + 1):
            run_ended = index == self.num_frames
            if not run_ended:
                run_ended = self.episode_codes[index] != self.episode_codes[index - 1]
            if not run_ended:
                continue
            run_end = index
            last_start = run_end - (self.sequence_length + self.horizon)
            if last_start >= run_start:
                starts.extend(range(run_start, last_start + 1, self.stride))
            run_start = index
        return starts

    def _apply_split(self, starts: list[int]) -> list[int]:
        if self.split_policy == "all":
            return starts
        rng = np.random.default_rng(self.seed)
        episode_ids = np.unique([self.episode_codes[start] for start in starts])
        if len(episode_ids) > 1:
            shuffled = rng.permutation(episode_ids)
            n_val = min(len(shuffled) - 1, max(1, int(round(len(shuffled) * self.val_fraction))))
            val_ids = set(int(value) for value in shuffled[:n_val])
            want_val = self.split_policy == "val"
            return [
                start
                for start in starts
                if (int(self.episode_codes[start]) in val_ids) == want_val
            ]

        warnings.warn(
            "Only one episode is available; using a random sequence-level train/val split. "
            "Adjacent windows can leak similar observations across the split.",
            stacklevel=2,
        )
        order = rng.permutation(len(starts))
        n_val = min(len(starts) - 1, max(1, int(round(len(starts) * self.val_fraction))))
        val_indices = set(int(value) for value in order[:n_val])
        want_val = self.split_policy == "val"
        return [start for i, start in enumerate(starts) if (i in val_indices) == want_val]

    def __len__(self) -> int:
        return len(self.sequence_map)

    def __getitem__(self, index: int):
        start = self.sequence_map[int(index)]
        input_slice = slice(start, start + self.sequence_length)
        target_slice = slice(
            start + self.horizon,
            start + self.horizon + self.sequence_length,
        )
        token_ids = torch.from_numpy(self.token_ids[input_slice].copy())
        token_mask = torch.from_numpy(self.token_mask[input_slice].copy())
        targets = torch.from_numpy(self.semantics[target_slice].copy())

        if not self.return_metadata:
            return token_ids, token_mask, targets

        input_indices = np.arange(start, start + self.sequence_length, dtype=np.int64)
        target_indices = input_indices + self.horizon
        metadata: dict[str, Any] = {
            "semantics_input": torch.from_numpy(self.semantics[input_slice].copy()),
            "frame_indices_input": torch.from_numpy(input_indices),
            "frame_indices_target": torch.from_numpy(target_indices),
            "episode_id": torch.tensor(self.episode_codes[start], dtype=torch.int64),
        }
        if self.state is not None:
            metadata["state_input"] = torch.as_tensor(
                np.asarray(self.state[input_slice], dtype=np.float32)
            )
            metadata["state_target"] = torch.as_tensor(
                np.asarray(self.state[target_slice], dtype=np.float32)
            )
        if self.actions is not None and (self.include_actions or self.return_metadata):
            metadata["actions_input"] = torch.as_tensor(
                np.asarray(self.actions[input_slice], dtype=np.float32)
            )
            metadata["actions_target"] = torch.as_tensor(
                np.asarray(self.actions[target_slice], dtype=np.float32)
            )
        if self.unity_frames is not None:
            metadata["unity_frames_input"] = torch.as_tensor(
                np.asarray(self.unity_frames[input_slice], dtype=np.int64)
            )
            metadata["unity_frames_target"] = torch.as_tensor(
                np.asarray(self.unity_frames[target_slice], dtype=np.int64)
            )
        if self.image_paths is not None:
            metadata["image_paths_input"] = tuple(self.image_paths[i] for i in input_indices)
            metadata["image_paths_target"] = tuple(self.image_paths[i] for i in target_indices)
        return token_ids, token_mask, targets, metadata
