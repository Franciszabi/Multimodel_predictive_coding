"""Train or benchmark the independent causal RGB baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from src.data.trail_visual_dataset import TrailVisualSequenceDataset
from src.models.visual_predictive import VisualPredictiveModel
from src.training_progress import EpochTimer, batch_progress, timing_summary


ROOT = Path(__file__).resolve().parent
MODEL_KEYS = ("num_heads", "num_layers", "position_encoding", "model_type")


def resolve_path(value):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def resolve_device(value):
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if value == "auto" else value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return device


def read_checkpoint(path):
    payload = torch.load(resolve_path(path), map_location="cpu", weights_only=True)
    if payload.get("format") != "visual_predictive_v1":
        raise ValueError("Expected a new visual_predictive_v1 checkpoint, not a legacy checkpoint")
    return payload


def load_visual_checkpoint(path, device):
    payload = read_checkpoint(path)
    model = VisualPredictiveModel(**{key: payload["config"][key] for key in MODEL_KEYS})
    model.load_state_dict(payload["model_state_dict"])
    return model.to(device), payload


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def save_checkpoint(path, payload):
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def amp_dtype(requested, device):
    if requested == "auto":
        requested = ("bf16" if torch.cuda.is_bf16_supported() else "fp16") if device.type == "cuda" else "off"
    if requested != "off" and device.type != "cuda":
        raise ValueError("Use --amp off for a non-CUDA device")
    if requested == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("This CUDA device does not support bf16; use fp16")
    return requested, {"off": None, "bf16": torch.bfloat16, "fp16": torch.float16}[requested]


def make_loader(dataset, args, shuffle=False):
    return DataLoader(
        dataset, batch_size=args.batch_size, shuffle=shuffle,
        num_workers=args.num_workers, pin_memory=args.device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        **({"prefetch_factor": 2} if args.num_workers else {}),
    )


def train_step(model, batch, optimizer, scaler, device, dtype, grad_clip):
    images = batch["images"].to(device, non_blocking=True)
    targets = batch["targets"].to(device, non_blocking=True)
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device.type, dtype=dtype, enabled=dtype is not None):
        prediction = model(images)
        loss = F.mse_loss(prediction.float(), targets)
    if not torch.isfinite(loss):
        raise RuntimeError("Non-finite visual training loss")
    scaler.scale(loss).backward()
    if grad_clip > 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip, error_if_nonfinite=not scaler.is_enabled())
    old_scale = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    return float(loss.detach()), len(images), scaler.get_scale() >= old_scale


def train_batches(
    model, loader, optimizer, scaler, device, dtype, grad_clip, steps=None, scheduler=None,
    *, progress=True, description="Train",
):
    model.train()
    steps = len(loader) if steps is None else steps
    sync(device)
    start = time.perf_counter()
    data_seconds, loss_sum, samples, skipped_steps = 0.0, 0.0, 0, 0
    with batch_progress(range(steps), desc=description, enabled=progress) as batches:
        iterator = iter(loader)
        for _ in batches:
            waiting = time.perf_counter()
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            data_seconds += time.perf_counter() - waiting
            loss, count, updated = train_step(model, batch, optimizer, scaler, device, dtype, grad_clip)
            skipped_steps += int(not updated)
            if scheduler is not None and updated:
                scheduler.step()
            loss_sum += loss * count
            samples += count
            batches.set_postfix(loss=f"{loss_sum / samples:.6f}", refresh=False)
    sync(device)
    seconds = time.perf_counter() - start
    return {
        "loss": loss_sum / samples, "seconds": seconds, "data_wait_seconds": data_seconds,
        "steps": steps, "samples": samples, "samples_per_second": samples / seconds,
        "input_frames_per_second": samples * batch["images"].shape[1] / seconds,
        "skipped_optimizer_steps": skipped_steps,
        "milliseconds_per_step": seconds * 1000 / steps,
    }


@torch.no_grad()
def evaluate(model, loader, device, dtype, *, progress=True, description="Val"):
    model.eval()
    totals = dict(mse=0.0, persistence_mse=0.0, last_mse=0.0, last_persistence_mse=0.0)
    samples = 0
    preview = None
    sync(device)
    start = time.perf_counter()
    with batch_progress(loader, desc=description, enabled=progress) as batches:
        for batch in batches:
            images = batch["images"].to(device, non_blocking=True)
            targets = batch["targets"].to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=dtype, enabled=dtype is not None):
                predictions = model(images)
            errors = (predictions.float() - targets).square().mean(dim=(2, 3, 4))
            baseline = (images - targets).square().mean(dim=(2, 3, 4))
            totals["mse"] += errors.mean(1).sum().item()
            totals["persistence_mse"] += baseline.mean(1).sum().item()
            totals["last_mse"] += errors[:, -1].sum().item()
            totals["last_persistence_mse"] += baseline[:, -1].sum().item()
            samples += len(images)
            batches.set_postfix(loss=f"{totals['mse'] / samples:.6f}", copy=f"{totals['persistence_mse'] / samples:.6f}", refresh=False)
            if preview is None:
                preview = [item[0].float().cpu() for item in (images, predictions, targets)]
    result = {key: value / samples for key, value in totals.items()}
    if not all(np.isfinite(value) for value in result.values()):
        raise RuntimeError("Non-finite validation metric")
    result["skill_vs_persistence"] = (
        1 - result["mse"] / result["persistence_mse"] if result["persistence_mse"] > 0 else None
    )
    result["last_skill_vs_persistence"] = (
        1 - result["last_mse"] / result["last_persistence_mse"] if result["last_persistence_mse"] > 0 else None
    )
    result.update(samples=samples, seconds=time.perf_counter() - start)
    return result, preview


def save_preview(preview, path, horizon):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    columns = np.unique(np.linspace(0, len(preview[0]) - 1, min(6, len(preview[0])), dtype=int))
    fig, axes = plt.subplots(3, len(columns), figsize=(2 * len(columns), 6), squeeze=False)
    for row, (label, images) in enumerate(zip(("Input", "Prediction", "Target"), preview)):
        for col, index in enumerate(columns):
            ax = axes[row, col]
            ax.imshow(images[index].permute(1, 2, 0).clamp(0, 1).numpy())
            ax.set_title(f"{label}: t={index if row == 0 else index + horizon}", fontsize=9)
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_root", required=True)
    p.add_argument("--val_data_root", default="")
    p.add_argument("--out_dir", default="experiments/visual_pc_v1")
    p.add_argument("--model_type", choices=("predictive", "autoencoder"), default="predictive")
    p.add_argument("--sequence_length", type=int, default=25)
    p.add_argument("--horizon", type=int, default=None)
    p.add_argument("--stride", type=int, default=5)
    p.add_argument("--image_size", type=int, default=64)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--position_encoding", choices=("none", "sinusoidal"), default="none")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--optimizer", choices=("adamw", "sgd"), default="adamw")
    p.add_argument("--scheduler", choices=("none", "onecycle"), default="none")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--early_stopping_patience", type=int, default=10)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--cache_frames", type=int, default=0)
    p.add_argument("--val_fraction", type=float, default=0.1)
    p.add_argument("--limit_train_samples", type=int, default=0)
    p.add_argument("--limit_val_samples", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--no_progress", action="store_true", help="Hide batch progress bars; keep epoch summaries")
    p.add_argument("--amp", choices=("auto", "off", "fp16", "bf16"), default="auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--benchmark_steps", type=int, default=0)
    p.add_argument("--warmup_steps", type=int, default=5)
    p.add_argument("--resume", default="")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if min(args.batch_size, args.epochs) < 1 or min(args.num_workers, args.benchmark_steps, args.warmup_steps) < 0:
        raise ValueError("Invalid batch size, epochs, workers or benchmark length")
    if min(args.limit_train_samples, args.limit_val_samples) < 0 or args.lr <= 0 or args.weight_decay < 0:
        raise ValueError("Invalid sample limit, learning rate or weight decay")
    if args.horizon is None:
        args.horizon = 0 if args.model_type == "autoencoder" else 1
    if (args.model_type == "predictive" and args.horizon < 1) or (args.model_type == "autoencoder" and args.horizon != 0):
        raise ValueError("Predictive models require h>=1; same-frame AE requires h=0")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.data_root = resolve_path(args.data_root)
    val_root = resolve_path(args.val_data_root) if args.val_data_root else None
    if val_root == args.data_root:
        raise ValueError("Use a distinct validation trail, or omit val_data_root for a disjoint temporal split")
    out = resolve_path(args.out_dir)
    if out.exists() and any(out.iterdir()) and not args.resume:
        raise FileExistsError(f"Output directory is not empty: {out}. Choose a new run or use --resume.")
    if args.resume and args.benchmark_steps:
        raise ValueError("Run benchmarks in a fresh output directory without --resume")
    if args.resume and resolve_path(args.resume) != out / "last.ckpt":
        raise ValueError("Resume from last.ckpt in the same out_dir; use a fresh directory for a new experiment")
    args.device = resolve_device(args.device)
    amp, dtype = amp_dtype(args.amp, args.device)
    common = dict(sequence_length=args.sequence_length, horizon=args.horizon, stride=args.stride,
                  image_size=args.image_size, val_fraction=args.val_fraction, cache_frames=args.cache_frames)
    train_raw = TrailVisualSequenceDataset(args.data_root, split="all" if val_root else "train", **common)
    train_data = Subset(train_raw, range(min(args.limit_train_samples, len(train_raw)))) if args.limit_train_samples > 0 else train_raw
    train_loader = make_loader(train_data, args, shuffle=True)
    model = VisualPredictiveModel(**{key: getattr(args, key) for key in MODEL_KEYS}).to(args.device)
    optimizer = (
        torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        if args.optimizer == "adamw" else
        torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0.9, nesterov=True)
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp == "fp16")
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=len(train_loader)) if args.scheduler == "onecycle" else None
    config = {**vars(args), "data_root": str(args.data_root), "val_data_root": str(val_root or ""),
              "out_dir": str(out), "device": str(args.device), "amp_effective": amp,
              "normalization": "per_frame_groupnorm_and_layernorm", "pixel_range": "RGB_[0,1]",
              "state_used_as_input": False, "actions_used_as_input": False, "semantic_used_as_input": False,
              "parameters": sum(p.numel() for p in model.parameters()), "torch_version": str(torch.__version__),
              "python_version": platform.python_version(), "torch_num_threads": torch.get_num_threads(),
              "gpu": torch.cuda.get_device_name(args.device) if args.device.type == "cuda" else None,
              "train_windows": len(train_data), "train_split": "all" if val_root else "chronological_train"}
    start_epoch, best, stale = 1, float("inf"), 0
    if args.resume:
        saved = read_checkpoint(args.resume)
        fixed = (*MODEL_KEYS, "sequence_length", "horizon", "image_size", "stride", "optimizer", "scheduler", "lr", "weight_decay", "grad_clip", "seed", "batch_size", "train_windows", "limit_val_samples", "data_root", "val_data_root", "val_fraction")
        if args.scheduler == "onecycle":
            fixed += ("epochs",)
        for key in fixed:
            if saved["config"][key] != config[key]:
                raise ValueError(f"Resume configuration mismatch for {key}")
        model.load_state_dict(saved["model_state_dict"])
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        scaler.load_state_dict(saved["scaler_state_dict"])
        if scheduler is not None:
            scheduler.load_state_dict(saved["scheduler_state_dict"])
        start_epoch, best, stale = saved["epoch"] + 1, saved["best_val_loss"], saved["stale_epochs"]
        if args.epochs < start_epoch:
            raise ValueError(f"Run already completed {saved['epoch']} epochs; increase --epochs to continue")
        torch.set_rng_state(saved["rng_state"])
        if args.device.type == "cuda" and saved.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state_all(saved["cuda_rng_state"])
    out.mkdir(parents=True, exist_ok=True)
    save_json(out / "config.json", config)
    print(f"[setup] model={args.model_type} L={args.sequence_length} h={args.horizon} images={args.image_size} heads={args.num_heads} layers={args.num_layers}", flush=True)
    print(f"[setup] device={args.device} amp={amp} params={config['parameters']:,} train_windows={len(train_data):,} workers={args.num_workers}", flush=True)
    if args.benchmark_steps:
        warmup = train_batches(model, train_loader, optimizer, scaler, args.device, dtype, args.grad_clip, args.warmup_steps,
                               progress=not args.no_progress, description="Warmup") if args.warmup_steps else None
        if args.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(args.device)
        report = train_batches(model, train_loader, optimizer, scaler, args.device, dtype, args.grad_clip, args.benchmark_steps,
                               progress=not args.no_progress, description="Benchmark")
        report.update(warmup=warmup, amp=amp, gpu=config["gpu"], workers=args.num_workers,
                      batch_size=args.batch_size, sequence_length=args.sequence_length,
                      horizon=args.horizon, image_size=args.image_size, train_windows=len(train_data),
                      torch_num_threads=config["torch_num_threads"],
                      peak_allocated_mb=torch.cuda.max_memory_allocated(args.device) / 2**20 if args.device.type == "cuda" else None,
                      estimated_train_epoch_seconds=report["seconds"] / report["steps"] * len(train_loader),
                      note="Includes loader waits, transfers, forward, backward and optimizer; excludes validation/checkpoint I/O. Warmup is separate; frames are window occurrences, not unique images.")
        save_json(out / "benchmark.json", report)
        print(json.dumps(report, indent=2), flush=True)
        return report
    val_raw = TrailVisualSequenceDataset(val_root or args.data_root, split="all" if val_root else "val", **common)
    val_data = Subset(val_raw, range(min(args.limit_val_samples, len(val_raw)))) if args.limit_val_samples > 0 else val_raw
    val_loader = make_loader(val_data, args)
    timer = EpochTimer(args.epochs, start_epoch)
    for epoch in range(start_epoch, args.epochs + 1):
        if args.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(args.device)
        training = train_batches(model, train_loader, optimizer, scaler, args.device, dtype, args.grad_clip, scheduler=scheduler,
                                 progress=not args.no_progress, description=f"Train {epoch}/{args.epochs}")
        validation, preview = evaluate(model, val_loader, args.device, dtype,
                                       progress=not args.no_progress, description=f"Val {epoch}/{args.epochs}")
        improved = validation["mse"] < best
        best, stale = (validation["mse"], 0) if improved else (best, stale + 1)
        saving = time.perf_counter()
        payload = dict(format="visual_predictive_v1", config=config, epoch=epoch, best_val_loss=best,
                       stale_epochs=stale, model_state_dict=model.state_dict(), optimizer_state_dict=optimizer.state_dict(),
                       scaler_state_dict=scaler.state_dict(), scheduler_state_dict=scheduler.state_dict() if scheduler else None,
                       rng_state=torch.get_rng_state(), cuda_rng_state=torch.cuda.get_rng_state_all() if args.device.type == "cuda" else None)
        save_checkpoint(out / "last.ckpt", payload)
        if improved:
            save_checkpoint(out / "best.ckpt", payload)
            save_preview(preview, out / "best_prediction.png", args.horizon)
        timing = timer.finish_epoch(epoch)
        record = dict(epoch=epoch, train=training, validation=validation, best_val_loss=best,
                      peak_allocated_mb=torch.cuda.max_memory_allocated(args.device) / 2**20 if args.device.type == "cuda" else None,
                      save_seconds=time.perf_counter() - saving, **timing)
        with (out / "train_log.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, allow_nan=False) + "\n")
        print(f"[epoch {epoch:03d}] train={training['loss']:.6f} val={validation['mse']:.6f} copy={validation['persistence_mse']:.6f} last={validation['last_mse']:.6f} train_s={training['seconds']:.1f} val_s={validation['seconds']:.1f} {timing_summary(timing)}", flush=True)
        if args.early_stopping_patience > 0 and stale >= args.early_stopping_patience:
            print("[early_stop] no validation improvement", flush=True)
            break
    print(f"[done] best_val_loss={best:.6f} checkpoint={out / 'best.ckpt'}", flush=True)


if __name__ == "__main__":
    main()
