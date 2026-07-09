"""
Create a strongly degenerated semantic dataset by merging selected object columns.

Strong degeneracy = merge multiple object IDs into one semantic column in objects_map,
and write a matching new vocabulary.npy.

Supports two layouts:
  1) path dataset: DATA_ROOT/path0/objects_map.npy, path1/...
  2) predefined/grid dataset: DATA_ROOT/objects_map.npy (+ optional positions.npy/images.npy)

Output is a NEW bundle directory:
  OUT_BUNDLE/
    vocabulary.npy
    <out_data_name>/...

Example (merge all teddy bears):
  python make_semantic_degenerate.py \
    --in_data_root /home/ubuntu/project/data/.../data_11272025_100000samples \
    --out_bundle_root /home/ubuntu/project/data/.../deg_teddybear_100k \
    --merge_regex ".*teddybear.*"
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path
from typing import Iterable

import numpy as np


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build degenerated semantic dataset (column merge).")
    ap.add_argument("--in_data_root", type=str, required=True, help="Input data root")
    ap.add_argument(
        "--in_vocab",
        type=str,
        default="",
        help="Input vocabulary.npy; empty => in_data_root.parent / vocabulary.npy",
    )
    ap.add_argument("--out_bundle_root", type=str, required=True, help="Output bundle root (must be new)")
    ap.add_argument(
        "--out_data_name",
        type=str,
        default="data_degenerate",
        help="Name of output data dir inside out_bundle_root",
    )
    ap.add_argument(
        "--merge_regex",
        type=str,
        action="append",
        required=True,
        help="Regex to select vocab entries to merge; can pass multiple times",
    )
    ap.add_argument(
        "--merged_token_name",
        type=str,
        default="prop_teddybear",
        help="New vocabulary token name for merged column",
    )
    ap.add_argument(
        "--binary_merge",
        action="store_true",
        help="Use OR-style merge (max over columns). Default is sum then clip to [0,1].",
    )
    return ap.parse_args()


def select_merge_indices(vocabulary: np.ndarray, regexes: Iterable[str]) -> list[int]:
    names = [str(x) for x in np.atleast_1d(vocabulary)]
    patterns = [re.compile(r) for r in regexes]
    idxs: list[int] = []
    for i, name in enumerate(names):
        if any(p.search(name) for p in patterns):
            idxs.append(i)
    return idxs


def merge_columns(objects_map: np.ndarray, merge_idxs: list[int], binary_merge: bool) -> np.ndarray:
    if objects_map.ndim != 2:
        raise ValueError(f"Expected objects_map ndim=2, got {objects_map.ndim}, shape={objects_map.shape}")
    v = objects_map.shape[1]
    keep_idxs = [i for i in range(v) if i not in set(merge_idxs)]
    merged_src = objects_map[:, merge_idxs]
    if binary_merge:
        merged = merged_src.max(axis=1, keepdims=True)
    else:
        merged = merged_src.sum(axis=1, keepdims=True)
        merged = np.clip(merged, 0.0, 1.0)
    out = np.concatenate([objects_map[:, keep_idxs], merged], axis=1)
    return out.astype(np.float32)


def process_path_layout(
    in_data_root: Path,
    out_data_root: Path,
    merge_idxs: list[int],
    binary_merge: bool,
) -> tuple[int, int]:
    episode_dirs = sorted([p for p in in_data_root.iterdir() if p.is_dir()])
    if not episode_dirs:
        raise FileNotFoundError(f"No episode dirs found in {in_data_root}")

    n_episodes = 0
    n_frames_total = 0
    for ep in episode_dirs:
        src_obj = ep / "objects_map.npy"
        if not src_obj.exists():
            continue
        rel = ep.relative_to(in_data_root)
        dst_ep = out_data_root / rel
        dst_ep.mkdir(parents=True, exist_ok=True)

        # copy non-objects_map files
        for item in ep.iterdir():
            dst_item = dst_ep / item.name
            if item.name == "objects_map.npy":
                continue
            if item.is_file():
                shutil.copy2(item, dst_item)

        obj = np.load(src_obj, allow_pickle=True).astype(np.float32)
        obj_new = merge_columns(obj, merge_idxs, binary_merge=binary_merge)
        np.save(dst_ep / "objects_map.npy", obj_new)
        n_episodes += 1
        n_frames_total += int(obj.shape[0])
    return n_episodes, n_frames_total


def process_predefined_layout(
    in_data_root: Path,
    out_data_root: Path,
    merge_idxs: list[int],
    binary_merge: bool,
) -> tuple[int, int]:
    out_data_root.mkdir(parents=True, exist_ok=True)
    src_obj = in_data_root / "objects_map.npy"
    if not src_obj.exists():
        raise FileNotFoundError(f"Expected {src_obj}")

    for item in in_data_root.iterdir():
        if item.name == "objects_map.npy":
            continue
        if item.is_file():
            shutil.copy2(item, out_data_root / item.name)

    obj = np.load(src_obj, allow_pickle=True).astype(np.float32)
    if obj.ndim == 3:
        n, l, v = obj.shape
        flat = obj.reshape(n * l, v)
        merged = merge_columns(flat, merge_idxs, binary_merge=binary_merge).reshape(n, l, -1)
        out = merged
        n_frames_total = int(n * l)
    elif obj.ndim == 2:
        out = merge_columns(obj, merge_idxs, binary_merge=binary_merge)
        n_frames_total = int(obj.shape[0])
    else:
        raise ValueError(f"Unsupported predefined objects_map shape: {obj.shape}")

    np.save(out_data_root / "objects_map.npy", out.astype(np.float32))
    return 1, n_frames_total


def main() -> None:
    args = parse_args()

    in_data_root = Path(args.in_data_root)
    if not in_data_root.exists():
        raise FileNotFoundError(f"in_data_root not found: {in_data_root}")

    in_vocab = Path(args.in_vocab) if args.in_vocab.strip() else in_data_root.parent / "vocabulary.npy"
    if not in_vocab.exists():
        raise FileNotFoundError(f"vocabulary.npy not found: {in_vocab}")

    out_bundle_root = Path(args.out_bundle_root)
    out_data_root = out_bundle_root / args.out_data_name
    out_vocab = out_bundle_root / "vocabulary.npy"

    if out_bundle_root.exists() and any(out_bundle_root.iterdir()):
        raise FileExistsError(
            f"Output bundle root is not empty: {out_bundle_root}. "
            "Use a new directory to avoid overwriting."
        )
    out_data_root.mkdir(parents=True, exist_ok=True)

    vocabulary = np.load(in_vocab, allow_pickle=True)
    merge_idxs = select_merge_indices(vocabulary, args.merge_regex)
    if len(merge_idxs) < 2:
        raise ValueError(
            f"Need >=2 matched vocab columns to merge, got {len(merge_idxs)}. "
            f"regex={args.merge_regex}"
        )
    merge_set = set(merge_idxs)
    keep_idxs = [i for i in range(len(vocabulary)) if i not in merge_set]
    merged_names = [str(vocabulary[i]) for i in merge_idxs]

    # Write new vocabulary first
    vocab_new = np.array([str(vocabulary[i]) for i in keep_idxs] + [args.merged_token_name], dtype=object)
    out_bundle_root.mkdir(parents=True, exist_ok=True)
    np.save(out_vocab, vocab_new, allow_pickle=True)

    # Detect layout
    has_predefined_obj = (in_data_root / "objects_map.npy").exists()
    has_episode_dirs = any(p.is_dir() for p in in_data_root.iterdir())
    if has_predefined_obj:
        mode = "grid_predefined"
        n_units, n_frames = process_predefined_layout(
            in_data_root=in_data_root,
            out_data_root=out_data_root,
            merge_idxs=merge_idxs,
            binary_merge=bool(args.binary_merge),
        )
    elif has_episode_dirs:
        mode = "path_episodes"
        n_units, n_frames = process_path_layout(
            in_data_root=in_data_root,
            out_data_root=out_data_root,
            merge_idxs=merge_idxs,
            binary_merge=bool(args.binary_merge),
        )
    else:
        raise ValueError(f"Could not detect dataset layout at {in_data_root}")

    print(f"[done] mode={mode}")
    print(f"[in ] data_root={in_data_root}")
    print(f"[in ] vocab={in_vocab}")
    print(f"[out] bundle={out_bundle_root}")
    print(f"[out] data_root={out_data_root}")
    print(f"[out] vocab={out_vocab}")
    print(f"[merge] matched={len(merge_idxs)} names={merged_names}")
    print(f"[merge] new_token={args.merged_token_name}")
    print(f"[stats] units={n_units} frames={n_frames}")


if __name__ == "__main__":
    main()

