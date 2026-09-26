"""James-style CNN position decoder on frozen, full visual latent maps."""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
from torch import nn
from tqdm.auto import tqdm

from src.spatial_plotting import load_map_overlay, draw_map_overlay, overlay_report, validate_map_positions
from train_visual_predictive import resolve_path, resolve_device, save_checkpoint, save_json
from visual_latent_analysis import read_latents


MATCH_KEYS = ("checkpoint_sha256", "layer", "pool", "horizon", "image_size")


class PositionDecoder(nn.Module):
    """Architecture from the public PositionDecoder, including no post-conv ReLU."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1), nn.MaxPool2d(2), nn.Flatten(1),
            nn.Linear(256 * 4 * 4, 64), nn.ReLU(), nn.Linear(64, 2),
        )

    def forward(self, inputs):
        if inputs.ndim != 4 or tuple(inputs.shape[1:]) != (128, 8, 8):
            raise ValueError("James PositionDecoder requires full [N,128,8,8] latents; export 64px images with --pool none")
        return self.net(inputs)


def frame_span(indices, metadata):
    return (int(indices.min()) - int(metadata["sequence_length"]) + 1,
            int(indices.max()) + int(metadata["horizon"]))


def check_representation(train_meta, test_meta, *, allow_context_shift=False):
    for key in MATCH_KEYS:
        if train_meta[key] != test_meta[key]:
            raise ValueError(f"Decoder exports must match on {key}")
    context_shift = {}
    for key in ("sequence_length", "sampling_protocol"):
        train_value = train_meta.get(key, "trail_windows") if key == "sampling_protocol" else train_meta[key]
        test_value = test_meta.get(key, "trail_windows") if key == "sampling_protocol" else test_meta[key]
        if train_value != test_value:
            context_shift[key] = dict(training=train_value, evaluation=test_value)
    if context_shift and not allow_context_shift:
        differences = "; ".join(f"{key}: {value['training']} -> {value['evaluation']}"
                                for key, value in context_shift.items())
        raise ValueError(f"Decoder context differs ({differences}). Use --allow_context_shift only for an "
                         "intentional cross-context evaluation; model weights/layer/shape checks remain strict.")
    return context_shift


def overlaps_training(training, indices, metadata):
    if training["metadata"]["data_root"] != metadata["data_root"]:
        return False
    first, last = frame_span(indices, metadata)
    train_first, train_last = training["frame_span"]
    return not (last < train_first or train_last < first)


def fit_decoder(model, latents, positions, args, out, training):
    # Only this auxiliary decoder is optimized; the visual model is not loaded.
    inputs = torch.from_numpy(latents).to(args.device)
    targets = torch.as_tensor(positions, dtype=torch.float32, device=args.device) / args.coordinate_scale
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_step, gamma=0.1)
    rng = np.random.default_rng(args.seed)
    started = time.perf_counter()
    config = dict(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, lr_step=args.lr_step,
                  lr_gamma=0.1, weight_decay=0.01, coordinate_scale=args.coordinate_scale,
                  seed=args.seed, architecture="Conv128-256_k3_MaxPool2_Flatten_Linear4096-64_ReLU_Linear64-2",
                  remainder_policy="drop_last_if_at_least_one_full_batch; otherwise_use_all",
                  precision="float32", visual_model_frozen=True)
    payload = dict(format="visual_position_decoder_v1", config=config, training=training)
    model.train()
    with (out / "decoder_train_log.jsonl").open("w", encoding="utf-8") as log:
        with tqdm(range(1, args.epochs + 1), desc="Position decoder", unit="epoch", disable=args.no_progress) as progress:
            for epoch in progress:
                order = rng.permutation(len(latents))
                if len(order) >= args.batch_size:
                    order = order[:len(order) // args.batch_size * args.batch_size]
                loss_sum = torch.zeros((), device=args.device)
                samples = 0
                rate = optimizer.param_groups[0]["lr"]
                for start in range(0, len(order), args.batch_size):
                    idx = torch.as_tensor(order[start:start + args.batch_size], device=args.device)
                    optimizer.zero_grad(set_to_none=True)
                    loss = nn.functional.mse_loss(model(inputs[idx]), targets[idx])
                    if not torch.isfinite(loss):
                        raise RuntimeError("Non-finite position decoder loss")
                    loss.backward()
                    optimizer.step()
                    loss_sum += loss.detach() * len(idx)
                    samples += len(idx)
                scheduler.step()
                value = float(loss_sum / samples)
                entry = dict(epoch=epoch, normalized_mse=value,
                             coordinate_mse=value * args.coordinate_scale ** 2,
                             lr=rate, samples=samples, elapsed_seconds=time.perf_counter() - started)
                log.write(json.dumps(entry, allow_nan=False) + "\n")
                log.flush()
                progress.set_postfix(mse=f"{entry['coordinate_mse']:.5g}", lr=f"{rate:.1g}")
                if epoch % args.save_every == 0 or epoch == args.epochs:
                    payload.update(epoch=epoch, model_state_dict=model.state_dict(),
                                   optimizer_state_dict=optimizer.state_dict(), scheduler_state_dict=scheduler.state_dict())
                    save_checkpoint(out / "decoder.ckpt", payload)
    return payload


@torch.no_grad()
def predict(model, latents, device, batch_size, scale):
    model.eval()
    values = []
    for start in range(0, len(latents), batch_size):
        batch = torch.from_numpy(latents[start:start + batch_size]).to(device)
        values.append((model(batch) * scale).cpu().numpy())
    result = np.concatenate(values)
    if not np.isfinite(result).all():
        raise RuntimeError("Non-finite decoded positions")
    return result


def plot_error_map(positions, errors, out, *, gridsize=27, vmax=None, overlay=None, protocol=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.5, 7))
    extent = overlay["extent"] if overlay else None
    values = ax.hexbin(positions[:, 0], positions[:, 1], C=errors, gridsize=gridsize,
                       reduce_C_function=np.mean, mincnt=1, cmap="inferno", vmin=0, vmax=vmax,
                       extent=extent)
    draw_map_overlay(ax, overlay)
    ax.set(xlabel="World x", ylabel="World z", aspect="equal",
           title=f"Position decoding error: {protocol}")
    fig.colorbar(values, ax=ax, label="Mean Euclidean error (world coordinate units)")
    fig.tight_layout()
    fig.savefig(out / "error_map.png", dpi=180)
    np.savez_compressed(out / "error_map.npz", hex_centers=values.get_offsets(),
                        mean_error=np.asarray(values.get_array()), gridsize=gridsize)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz", required=True, help="Latents to evaluate")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--train_npz", help="Fit on this separate export; evaluate on --npz")
    mode.add_argument("--fit_on_eval", action="store_true", help="Explicit same-sample fit, as in James's public notebook; NOT held-out error")
    mode.add_argument("--ckpt", help="Evaluate a saved decoder without retraining")
    parser.add_argument("--allow_context_shift", action="store_true",
                        help="Allow different input lengths/sampling protocols; record this distribution shift in the report")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--map_root", default="", help="Optional Unity map metadata directory; auto-detected from --npz otherwise")
    parser.add_argument("--epochs", type=int, default=8000, help="Public-code default; paper text reports 2000 instead")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_step", type=int, default=4000, help="Public-code LR decay epoch; paper text reports 1000")
    parser.add_argument("--coordinate_scale", type=float, default=30.0)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--save_every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no_progress", action="store_true")
    parser.add_argument("--gridsize", type=int, default=27)
    parser.add_argument("--error_vmax", type=float, default=None, help="Shared color maximum for model comparisons; default auto")
    args = parser.parse_args(argv)
    if min(args.epochs, args.lr_step, args.batch_size, args.save_every, args.gridsize) < 1 or args.seed < 0:
        raise ValueError("Epoch, batch, save and grid settings must be positive; seed must be nonnegative")
    if not np.isfinite([args.lr, args.coordinate_scale]).all() or min(args.lr, args.coordinate_scale) <= 0:
        raise ValueError("Learning rate and coordinate scale must be finite and positive")
    if args.error_vmax is not None and (not np.isfinite(args.error_vmax) or args.error_vmax <= 0):
        raise ValueError("error_vmax must be finite and positive")
    args.device = resolve_device(args.device)
    out = resolve_path(args.out_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Choose a new decoder output directory: {out}")
    latents, positions, indices, metadata = read_latents(args.npz)
    if latents.shape[1:] != (128, 8, 8):
        raise ValueError("Export full [N,128,8,8] latent maps at image_size=64; pooled latents cannot be decoded")
    torch.manual_seed(args.seed)
    model = PositionDecoder().to(args.device)
    overlay = load_map_overlay(resolve_path(args.map_root) if args.map_root else None,
                               data_root=metadata.get("data_root"))
    validate_map_positions(overlay, positions)
    if args.ckpt:
        payload = torch.load(resolve_path(args.ckpt), map_location="cpu", weights_only=True)
        if payload.get("format") != "visual_position_decoder_v1":
            raise ValueError("Expected a visual_position_decoder_v1 checkpoint")
        training = payload["training"]
        context_shift = check_representation(training["metadata"], metadata,
                                             allow_context_shift=args.allow_context_shift)
        model.load_state_dict(payload["model_state_dict"])
        protocol = "overlapping_training_support" if overlaps_training(training, indices, metadata) else "separate_export"
        out.mkdir(parents=True, exist_ok=True)
    else:
        if args.fit_on_eval:
            train, train_positions, train_indices, train_meta = latents, positions, indices, metadata
        else:
            train, train_positions, train_indices, train_meta = read_latents(args.train_npz)
        context_shift = check_representation(train_meta, metadata, allow_context_shift=args.allow_context_shift)
        if train.shape[1:] != (128, 8, 8):
            raise ValueError("Training latents must be [N,128,8,8]")
        training = dict(npz=str(resolve_path(args.train_npz or args.npz)), samples=len(train),
                        metadata=train_meta, frame_span=list(frame_span(train_indices, train_meta)),
                        mean_position=train_positions.mean(axis=0).tolist())
        if not args.fit_on_eval and overlaps_training(training, indices, metadata):
            raise ValueError("Decoder train/evaluation supports overlap; use independent exports or explicitly --fit_on_eval")
        protocol = "same_sample_fit" if args.fit_on_eval else "separate_export"
        out.mkdir(parents=True, exist_ok=True)
        if context_shift:
            print(f"[context_shift] {json.dumps(context_shift)}; fitting uses training latents only", flush=True)
        print(f"[decoder] protocol={protocol} train={len(train)} eval={len(latents)} epochs={args.epochs} "
              f"device={args.device}; visual model is frozen", flush=True)
        payload = fit_decoder(model, train, train_positions, args, out, training)
    if context_shift:
        protocol += "_context_shift"
        print(f"[context_shift] {json.dumps(context_shift)}; results are cross-context, not matched-context errors", flush=True)
    prediction = predict(model, latents, args.device, args.batch_size, payload["config"]["coordinate_scale"])
    errors = np.linalg.norm(prediction - positions, axis=1)
    baseline = np.linalg.norm(np.asarray(training["mean_position"]) - positions, axis=1)
    report = dict(method="james_public_code_position_decoder", protocol=protocol,
                  context_shift=context_shift, allow_context_shift=args.allow_context_shift,
                  training=training, evaluation_metadata=metadata, config=payload["config"],
                  decoder_epoch=payload["epoch"], source_checkpoint=str(resolve_path(args.ckpt)) if args.ckpt else str(out / "decoder.ckpt"),
                  samples=len(latents), mean_distance=float(errors.mean()), median_distance=float(np.median(errors)),
                  rmse_distance=float(np.sqrt(np.mean(errors ** 2))),
                  train_mean_baseline_distance=float(baseline.mean()),
                  map_overlay=overlay_report(overlay),
                  error_map=dict(statistic="mean Euclidean distance per hexagon", axes=["x", "z"],
                                 gridsize=args.gridsize, vmax=args.error_vmax),
                  note="Same-sample/overlapping errors are fit errors, not generalization. Separate exports are held out "
                       "from this decoder only; duplicate recordings under different roots cannot be detected. "
                       "A context_shift changes input length and/or sampling protocol, so errors combine trajectory "
                       "generalization with a context distribution shift. "
                       "Public code uses 8000 epochs/decay at 4000 and no post-conv ReLU; paper prose differs. "
                       "Unlike the source diagnostic, logged coordinate MSE uses consistent units.")
    np.savez_compressed(out / "position_predictions.npz", predicted=prediction, true=positions, error=errors,
                        input_indices=indices, metadata_json=np.asarray(json.dumps(report)))
    plot_error_map(positions, errors, out, gridsize=args.gridsize, vmax=args.error_vmax,
                   overlay=overlay, protocol=protocol)
    save_json(out / "report.json", report)
    print(f"[done] {out} mean_distance={errors.mean():.6f} protocol={protocol}", flush=True)
    return report


if __name__ == "__main__":
    main()
