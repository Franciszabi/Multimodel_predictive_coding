"""Evaluate a saved visual baseline on a trail without fitting any weights."""

import argparse

from torch.utils.data import Subset

from src.data.trail_visual_dataset import TrailVisualSequenceDataset
from train_visual_predictive import (
    amp_dtype, evaluate, load_visual_checkpoint, make_loader,
    resolve_device, resolve_path, save_json, save_preview,
)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data_root", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--split", choices=("all", "train", "val"), default="all")
    p.add_argument("--stride", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--limit_samples", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--no_progress", action="store_true", help="Hide batch progress bars")
    p.add_argument("--amp", choices=("auto", "off", "fp16", "bf16"), default="auto")
    args = p.parse_args(argv)
    if args.batch_size < 1 or min(args.num_workers, args.limit_samples) < 0:
        raise ValueError("Invalid batch size, worker count or sample limit")
    out = resolve_path(args.out_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Choose a new evaluation directory: {out}")
    args.device = resolve_device(args.device)
    amp, dtype = amp_dtype(args.amp, args.device)
    model, checkpoint = load_visual_checkpoint(args.ckpt, args.device)
    config = checkpoint["config"]
    dataset = TrailVisualSequenceDataset(
        resolve_path(args.data_root), sequence_length=config["sequence_length"], horizon=config["horizon"],
        image_size=config["image_size"], stride=args.stride if args.stride is not None else config["stride"],
        split=args.split, val_fraction=config["val_fraction"],
    )
    data = Subset(dataset, range(min(args.limit_samples, len(dataset)))) if args.limit_samples else dataset
    metrics, preview = evaluate(model, make_loader(data, args), args.device, dtype,
                                progress=not args.no_progress, description="Evaluate")
    metrics.update(checkpoint=str(resolve_path(args.ckpt)), checkpoint_epoch=checkpoint["epoch"],
                   data_root=str(dataset.data_root), split=args.split, amp=amp,
                   sequence_length=dataset.sequence_length, horizon=dataset.horizon, stride=dataset.stride)
    out.mkdir(parents=True, exist_ok=True)
    save_json(out / "metrics.json", metrics)
    save_preview(preview, out / "prediction.png", dataset.horizon)
    print(f"[eval] mse={metrics['mse']:.6f} persistence={metrics['persistence_mse']:.6f} last_mse={metrics['last_mse']:.6f}", flush=True)
    return metrics


if __name__ == "__main__":
    main()
