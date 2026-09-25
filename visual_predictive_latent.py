"""Export fixed-context visual latents; state is post-hoc metadata only."""

from __future__ import annotations

import argparse
import hashlib
import json

import numpy as np
import torch
from torch.utils.data import Subset

from src.data.trail_visual_dataset import TrailVisualSequenceDataset
from train_visual_predictive import load_visual_checkpoint, make_loader, resolve_device, resolve_path


@torch.no_grad()
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data_root", required=True)
    p.add_argument("--out_npz", required=True)
    p.add_argument("--layer", default="final", help="encoder, temporal_1, temporal_2, ... or final")
    p.add_argument("--pool", choices=("none", "spatial_mean"), default="none")
    p.add_argument("--stride", type=int, default=None, help="Defaults to checkpoint sequence_length")
    p.add_argument("--split", choices=("all", "train", "val"), default="all")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--limit_samples", type=int, default=0)
    p.add_argument("--max_output_mb", type=int, default=2048)
    p.add_argument("--device", default="auto")
    args = p.parse_args(argv)
    if args.batch_size < 1 or args.num_workers < 0 or args.limit_samples < 0 or args.max_output_mb < 1:
        raise ValueError("Invalid loader or output size settings")
    out = resolve_path(args.out_npz)
    if out.suffix != ".npz" or out.exists():
        raise ValueError("out_npz must be a new .npz file")
    args.device = resolve_device(args.device)
    model, checkpoint = load_visual_checkpoint(args.ckpt, args.device)
    checkpoint_hash = hashlib.sha256()
    with resolve_path(args.ckpt).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            checkpoint_hash.update(block)
    model.eval()
    config = checkpoint["config"]
    valid_layers = {"encoder", "final"} | {f"temporal_{i+1}" for i in range(len(model.blocks))}
    if args.layer not in valid_layers:
        raise ValueError(f"Layer must be one of {sorted(valid_layers)}")
    dataset = TrailVisualSequenceDataset(
        resolve_path(args.data_root), sequence_length=config["sequence_length"], horizon=config["horizon"],
        stride=args.stride if args.stride is not None else config["sequence_length"],
        image_size=config["image_size"], split=args.split, val_fraction=config["val_fraction"],
        return_metadata=True, return_targets=False,
    )
    count = min(args.limit_samples, len(dataset)) if args.limit_samples else len(dataset)
    side = config["image_size"] // 8 if args.pool == "none" else 1
    shape = (count, 128, side, side)
    estimated_mb = np.prod(shape) * 4 / 2**20
    if estimated_mb > args.max_output_mb:
        raise ValueError(f"Latents need {estimated_mb:.1f} MiB; increase stride, use spatial_mean, or raise max_output_mb")
    latents = np.empty(shape, dtype=np.float32)
    indices = np.empty(count, dtype=np.int64)
    episodes = np.empty(count, dtype=np.int64)
    loader = make_loader(Subset(dataset, range(count)), args)
    offset = 0
    for batch in loader:
        layers = model.forward_features(batch["images"].to(args.device), return_layers=True)
        value = layers[args.layer][:, -1]
        if args.pool == "spatial_mean":
            value = value.mean(dim=(-2, -1), keepdim=True)
        size = len(value)
        latents[offset:offset + size] = value.float().cpu().numpy()
        indices[offset:offset + size] = batch["input_indices"][:, -1].numpy()
        episodes[offset:offset + size] = batch["episode_id"].numpy()
        offset += size
        if offset == count or offset % (args.batch_size * 100) == 0:
            print(f"[export] {offset}/{count}", flush=True)
    metadata = dict(
        format="visual_latent_v1", data_root=str(dataset.data_root), checkpoint=str(resolve_path(args.ckpt)),
        checkpoint_epoch=checkpoint["epoch"], model_type=config["model_type"],
        checkpoint_sha256=checkpoint_hash.hexdigest(),
        layer=args.layer, pool=args.pool, temporal_position="last_input_frame",
        sequence_length=config["sequence_length"], horizon=config["horizon"],
        image_size=config["image_size"], stride=dataset.stride, split=args.split,
        state_used_as_input=False, state_columns=["x", "z", "yaw"],
    )
    arrays = dict(latents=latents, input_indices=indices, target_indices=indices + config["horizon"],
                  episode_ids=episodes, metadata_json=np.asarray(json.dumps(metadata)),
                  image_paths=np.asarray([dataset.image_paths[i] for i in indices]))
    state_path = dataset.data_root / "state.npy"
    if state_path.exists():
        state = np.load(state_path, mmap_mode="r", allow_pickle=False)
        if state.ndim != 2 or state.shape[0] != dataset.num_frames or state.shape[1] < 3:
            raise ValueError("Expected state.npy [global_frames, >=3] ordered x,z,yaw")
        arrays["state"] = np.asarray(state[indices, :3], dtype=np.float32)
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(out)
    print(f"[done] {out} latents={latents.shape}; state was never passed to the model", flush=True)
    return out


if __name__ == "__main__":
    main()
