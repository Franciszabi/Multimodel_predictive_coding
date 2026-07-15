"""Shared tiny trail generator for smoke scripts."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


def write_tiny_trail(
    root: Path,
    tokens_by_frame: list[list[str]],
    episodes: np.ndarray,
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    frame_count = len(tokens_by_frame)
    with (root / "semantics.jsonl").open("w", encoding="utf-8") as handle:
        for frame_tokens in tokens_by_frame:
            handle.write(json.dumps({"modelTokens": frame_tokens}) + "\n")
    with (root / "frame_index.jsonl").open("w", encoding="utf-8") as handle:
        for frame_index, episode_id in enumerate(episodes):
            record = {"path": f"path{int(episode_id)}", "filename": f"{frame_index}.png"}
            handle.write(json.dumps(record) + "\n")

    state = np.stack(
        [
            np.arange(frame_count, dtype=np.float32),
            np.arange(frame_count, dtype=np.float32) * 2,
            np.zeros(frame_count, dtype=np.float32),
        ],
        axis=1,
    )
    np.save(root / "state.npy", state)
    np.save(root / "actions.npy", np.zeros((frame_count, 2), dtype=np.float32))
    np.save(root / "unity_frames.npy", np.arange(frame_count, dtype=np.int64))
    np.save(root / "episodes.npy", np.asarray(episodes, dtype=np.int64))

    vocabulary = sorted({token for frame in tokens_by_frame for token in frame})
    with (root / "meta.json").open("w", encoding="utf-8") as handle:
        json.dump({"semantic_vocabulary": vocabulary}, handle)
    with (root / "object_map.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["modelToken", "cx", "cz"])
        writer.writeheader()
        for index, token in enumerate(vocabulary):
            writer.writerow({"modelToken": token, "cx": index, "cz": index})
