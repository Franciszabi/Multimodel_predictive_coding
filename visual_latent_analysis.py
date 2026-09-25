"""Occupancy-normalized activation maps and an optional held-out position probe."""

from __future__ import annotations

import argparse
import json

import numpy as np

from train_visual_predictive import resolve_path, save_json


def read_latents(path):
    with np.load(resolve_path(path), allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata_json"]))
        if metadata["format"] != "visual_latent_v1" or "state" not in archive:
            raise ValueError("Expected visual_latent_v1 with state.npy metadata")
        latents = archive["latents"].astype(np.float32)
        positions = archive["state"][:, :2].astype(np.float64)
        indices = archive["input_indices"].copy()
    if not np.isfinite(latents).all() or not np.isfinite(positions).all():
        raise ValueError("Non-finite latents or positions")
    return latents, positions, indices, metadata


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--npz", required=True, help="Evaluation trail latent export")
    p.add_argument("--train_npz", default="", help="Optional separate training export for a spatial-mean Ridge probe")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--bins", type=int, default=30)
    p.add_argument("--min_occupancy", type=int, default=3)
    p.add_argument("--units", type=int, default=16)
    p.add_argument("--ridge_alpha", type=float, default=1.0)
    args = p.parse_args(argv)
    if min(args.bins, args.min_occupancy, args.units) < 1 or args.ridge_alpha <= 0:
        raise ValueError("Plot and probe settings must be positive")
    out = resolve_path(args.out_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Choose a new analysis directory: {out}")
    latents, positions, indices, metadata = read_latents(args.npz)
    out.mkdir(parents=True, exist_ok=True)
    flat = latents.reshape(len(latents), -1)
    # Fixed evenly spaced units, not selected by visual appearance or test scores.
    selected = np.unique(np.linspace(0, flat.shape[1] - 1, min(args.units, flat.shape[1]), dtype=int))
    occupancy, xedges, zedges = np.histogram2d(positions[:, 0], positions[:, 1], bins=args.bins)
    maps = []
    for unit in selected:
        sums, _, _ = np.histogram2d(positions[:, 0], positions[:, 1], bins=(xedges, zedges), weights=flat[:, unit])
        average = np.full_like(sums, np.nan)
        np.divide(sums, occupancy, out=average, where=occupancy >= args.min_occupancy)
        maps.append(average)
    np.savez_compressed(out / "activation_maps.npz", means=np.stack(maps), occupancy=occupancy,
                        unit_indices=selected, x_edges=xedges, z_edges=zedges)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    columns = min(4, len(maps) + 1)
    rows = (len(maps) + 1 + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(3.3 * columns, 3 * rows), squeeze=False)
    for index, ax in enumerate(axes.flat):
        if index > len(maps):
            ax.axis("off")
            continue
        values = np.where(occupancy > 0, occupancy, np.nan) if index == 0 else maps[index - 1]
        plotted = ax.pcolormesh(xedges, zedges, values.T, shading="auto", cmap="viridis" if index == 0 else "coolwarm")
        ax.set_title("Occupancy (export samples)" if index == 0 else f"Mean activation: unit {selected[index - 1]}")
        ax.set(xlabel="x", ylabel="z", aspect="equal")
        fig.colorbar(plotted, ax=ax, shrink=0.75)
    fig.tight_layout()
    fig.savefig(out / "activation_maps.png", dpi=140)
    plt.close(fig)
    report = dict(samples=len(latents), metadata=metadata, min_occupancy=args.min_occupancy,
                  note="Exploratory mean-activation maps, not evidence of significant place/grid cells by themselves.")
    if args.train_npz:
        train, train_positions, train_indices, train_meta = read_latents(args.train_npz)
        for key in ("checkpoint_sha256", "layer", "pool", "sequence_length", "horizon", "image_size"):
            if train_meta[key] != metadata[key]:
                raise ValueError(f"Probe exports must match on {key}")
        if train_meta["data_root"] == metadata["data_root"]:
            # Chronological splits are allowed only when complete window supports are disjoint.
            train_span = (train_indices.min() - train_meta["sequence_length"] + 1,
                          train_indices.max() + train_meta["horizon"])
            test_span = (indices.min() - metadata["sequence_length"] + 1, indices.max() + metadata["horizon"])
            if not (train_span[1] < test_span[0] or test_span[1] < train_span[0]):
                raise ValueError("Probe train/test frame supports overlap; export independent trails or chronological splits")
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import Ridge
        from sklearn.metrics import r2_score
        # Keep probe capacity and cost fixed across input resolutions.
        train_x = train.mean(axis=(-2, -1))
        test_x = latents.mean(axis=(-2, -1))
        probe = make_pipeline(StandardScaler(), Ridge(alpha=args.ridge_alpha))
        probe.fit(train_x, train_positions)
        prediction = probe.predict(test_x)
        errors = np.linalg.norm(prediction - positions, axis=1)
        baseline = np.linalg.norm(train_positions.mean(0) - positions, axis=1)
        report["position_probe"] = dict(
            representation="spatial_mean_128_channels", alpha=args.ridge_alpha,
            train_samples=len(train), test_samples=len(latents), mean_distance=float(errors.mean()),
            median_distance=float(np.median(errors)), train_mean_baseline_distance=float(baseline.mean()),
            r2=float(r2_score(positions, prediction)) if len(positions) >= 2 else None,
            note="Train-fitted scaler and Ridge; evaluation coordinates were not used to fit the probe.",
        )
        np.savez_compressed(out / "position_probe.npz", predicted=prediction, true=positions, error=errors)
    save_json(out / "report.json", report)
    print(json.dumps(report, indent=2), flush=True)
    return report


if __name__ == "__main__":
    main()
