"""RGB-only windows from the global Unity trail index, with shifted targets."""

from __future__ import annotations

from collections import OrderedDict
import json
from pathlib import Path, PureWindowsPath
import warnings

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


def image_relative_path(record: dict) -> str:
    folder = record.get("path") or record.get("folder") or record.get("path_dir")
    filename = record.get("filename") or record.get("image_filename")
    if folder and filename:
        value = str(PureWindowsPath(str(folder)) / str(filename))
    elif folder and "frame" in record:
        value = str(PureWindowsPath(str(folder)) / f"{int(record['frame'])}.png")
    else:
        value = next((record[k] for k in ("image_path", "imagePath", "relative_path", "image", "file", "filename", "path") if record.get(k)), "")
    relative = PureWindowsPath(str(value))
    if not value or relative.drive or relative.root or ".." in relative.parts:
        raise ValueError(f"Expected a portable relative image path, got {value!r}")
    if relative.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
        raise ValueError(f"Index does not identify an image: {record}")
    return relative.as_posix()


class TrailVisualSequenceDataset(Dataset):
    def __init__(
        self, data_root, sequence_length: int = 25, horizon: int = 1,
        stride: int = 5, image_size: int = 64, split: str = "all",
        val_fraction: float = 0.1, cache_frames: int = 0,
        return_metadata: bool = False, return_targets: bool = True,
    ):
        self.data_root = Path(data_root).expanduser().resolve()
        self.sequence_length, self.horizon = int(sequence_length), int(horizon)
        self.stride, self.image_size = int(stride), int(image_size)
        self.cache_frames = int(cache_frames)
        self.return_metadata = return_metadata
        self.return_targets = return_targets
        if sequence_length < 1 or horizon < 0 or stride < 1 or cache_frames < 0:
            raise ValueError("Invalid window, horizon, stride or cache size")
        if image_size < 16 or image_size % 8:
            raise ValueError("image_size must be >=16 and divisible by 8")
        if split not in {"all", "train", "val"} or not 0 < val_fraction < 1:
            raise ValueError("Invalid split or val_fraction")
        self.image_paths, index_episodes = [], []
        previous_step = None
        with (self.data_root / "frame_index.jsonl").open(encoding="utf-8-sig") as handle:
            for line_number, line in enumerate(handle, 1):
                record = json.loads(line)
                self.image_paths.append(image_relative_path(record))
                index_episodes.append(record.get("episode"))
                if "globalStep" in record:
                    step = int(record["globalStep"])
                    if previous_step is not None and step != previous_step + 1:
                        raise ValueError(f"Non-contiguous globalStep at index line {line_number}")
                    previous_step = step
        self.num_frames = len(self.image_paths)
        if not self.num_frames:
            raise ValueError("Empty frame index")
        episodes_path = self.data_root / "episodes.npy"
        has_index_episodes = all(value is not None for value in index_episodes)
        if episodes_path.exists():
            episodes = np.load(episodes_path, allow_pickle=False)
            if episodes.shape not in {(self.num_frames,), (self.num_frames, 1)}:
                raise ValueError("episodes.npy does not align with the frame index")
            episodes = episodes.reshape(-1)
            if has_index_episodes and not np.array_equal(episodes.astype(str), np.asarray(index_episodes).astype(str)):
                raise ValueError("episodes.npy and frame_index.jsonl disagree")
        elif has_index_episodes:
            episodes = np.asarray(index_episodes)
        else:
            if any(value is not None for value in index_episodes):
                raise ValueError("Incomplete episode IDs in the index and no episodes.npy")
            warnings.warn("No episode IDs; treating this trail as one continuous episode")
            episodes = np.zeros(self.num_frames, dtype=np.int64)
        self.episodes = np.unique(episodes.astype(str), return_inverse=True)[1]
        # Split raw frames before constructing windows, including their future targets.
        boundary = int(self.num_frames * (1 - val_fraction))
        lower, upper = (0, self.num_frames)
        if split == "train":
            upper = boundary
        elif split == "val":
            lower = boundary
        self.sequence_map = []
        start = 0
        for stop in range(1, self.num_frames + 1):
            if stop == self.num_frames or self.episodes[stop] != self.episodes[stop - 1]:
                first, end = max(start, lower), min(stop, upper)
                self.sequence_map.extend(range(first, end - sequence_length - horizon + 1, stride))
                start = stop
        if not self.sequence_map:
            raise ValueError(f"No valid {split} windows; need L+h={sequence_length+horizon} contiguous frames")
        self._cache = OrderedDict()

    def __len__(self):
        return len(self.sequence_map)

    def _image(self, index):
        if index in self._cache:
            self._cache.move_to_end(index)
            return self._cache[index]
        with Image.open(self.data_root / self.image_paths[index]) as source:
            rgb = source.convert("RGB").resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
            tensor = torch.from_numpy(np.array(rgb, copy=True)).permute(2, 0, 1).float().div_(255)
        if self.cache_frames:
            self._cache[index] = tensor
            if len(self._cache) > self.cache_frames:
                self._cache.popitem(last=False)
        return tensor

    def __getitem__(self, index):
        start = self.sequence_map[index]
        inputs = np.arange(start, start + self.sequence_length, dtype=np.int64)
        targets = inputs + self.horizon
        # Decode overlapping input/target images only once per sample.
        needed = np.union1d(inputs, targets) if self.return_targets else inputs
        decoded = {int(i): self._image(int(i)) for i in needed}
        sample = {"images": torch.stack([decoded[int(i)] for i in inputs])}
        if self.return_targets:
            sample["targets"] = torch.stack([decoded[int(i)] for i in targets])
        if self.return_metadata:
            sample.update(
                input_indices=torch.from_numpy(inputs), target_indices=torch.from_numpy(targets),
                episode_id=int(self.episodes[start]),
            )
        return sample
