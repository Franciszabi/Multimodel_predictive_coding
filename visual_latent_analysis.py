"""Channel-wise percentile place fields; position decoding is a separate analysis."""

from __future__ import annotations

import argparse
import json

import numpy as np

from src.spatial_plotting import load_map_overlay, draw_map_overlay, overlay_report, validate_map_positions
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
    if latents.ndim != 4 or min(latents.shape) < 1:
        raise ValueError("Expected nonempty latents shaped [N,C,H,W]")
    if positions.shape != (len(latents), 2) or indices.shape != (len(latents),):
        raise ValueError("Latents, x/z positions and input indices must align")
    return latents, positions, indices, metadata


def compute_place_fields(latents, positions, *, bins=30, quantile=0.9, spatial_range=None):
    """Match the reference PlaceFields: spatial means, strict quantile, raw counts."""
    if latents.ndim != 4 or min(latents.shape) < 1 or positions.shape != (len(latents), 2):
        raise ValueError("Expected [N,C,H,W] latents and aligned [N,2] positions")
    if not np.isfinite(latents).all() or not np.isfinite(positions).all():
        raise ValueError("Non-finite latents or positions")
    if not np.isfinite(quantile) or not 0 < quantile < 1:
        raise ValueError("quantile must be between zero and one")
    if spatial_range is not None:
        bounds = np.asarray(spatial_range, dtype=np.float64)
        if bounds.shape != (2, 2) or not np.isfinite(bounds).all() or np.any(bounds[:, 0] >= bounds[:, 1]):
            raise ValueError("spatial_range must contain increasing finite x and z bounds")
        if np.any(positions < bounds[:, 0]) or np.any(positions > bounds[:, 1]):
            raise ValueError("spatial_range must include every sample; refusing to silently drop positions")
    activations = latents.mean(axis=(-2, -1))
    thresholds = np.quantile(activations, quantile, axis=0)
    active = activations > thresholds
    occupancy, xedges, zedges = np.histogram2d(
        positions[:, 0], positions[:, 1], bins=bins, range=spatial_range,
    )
    channels = activations.shape[1]
    counts = np.zeros((channels, *occupancy.shape), dtype=np.int64)
    means = np.full((channels, 2), np.nan)
    covariances = np.full((channels, 2, 2), np.nan)
    gaussian = np.full(counts.shape, np.nan, dtype=np.float64)
    areas = np.full(channels, np.nan)
    status = np.full(channels, "too_few_active_samples", dtype="U32")
    centers = np.stack(np.meshgrid((xedges[:-1] + xedges[1:]) / 2,
                                   (zedges[:-1] + zedges[1:]) / 2, indexing="ij"), axis=-1)
    for channel in range(channels):
        points = positions[active[:, channel]]
        counts[channel] = np.histogram2d(points[:, 0], points[:, 1], bins=(xedges, zedges))[0]
        if len(points):
            means[channel] = points.mean(axis=0)
        if len(points) >= 2:
            covariances[channel] = np.cov(points, rowvar=False)
        if len(points) < 3:
            continue
        covariance = covariances[channel]
        # Do not invent spatial spread by regularizing singular or collinear fields.
        if np.linalg.eigvalsh(covariance).min() <= 0:
            status[channel] = "singular_covariance"
            continue
        sign, logdet = np.linalg.slogdet(covariance)
        if sign <= 0:
            status[channel] = "singular_covariance"
            continue
        try:
            precision = np.linalg.inv(covariance)
        except np.linalg.LinAlgError:
            status[channel] = "singular_covariance"
            continue
        delta = centers - means[channel]
        squared_distance = np.einsum("...i,ij,...j->...", delta, precision, delta)
        gaussian[channel] = np.exp(-np.log(2 * np.pi) - logdet / 2 - squared_distance / 2)
        # This is the public code's approx_areas, not a fixed-density contour area.
        areas[channel] = np.pi * np.exp(logdet / 2)
        status[channel] = "ok"
    binary = counts > 0
    return dict(
        channel_activations=activations, thresholds=thresholds, active_samples=active,
        high_activation_counts=counts, binary_fields=binary, occupancy=occupancy.astype(np.int64),
        x_edges=xedges, z_edges=zedges, channel_indices=np.arange(channels), quantile=np.asarray(quantile),
        active_sample_counts=active.sum(axis=0), gaussian_mean=means, gaussian_covariance=covariances,
        gaussian_density=gaussian, gaussian_fit_status=status, gaussian_area_1sigma=areas,
        fields_per_position=binary.sum(axis=0), positions_per_field=binary.sum(axis=(1, 2)),
    )


def plot_place_fields(fields, out, selected, min_occupancy, overlay=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xedges, zedges = fields["x_edges"], fields["z_edges"]
    occupancy = fields["occupancy"]
    visible = occupancy >= min_occupancy
    columns = min(4, len(selected) + 1)
    rows = (len(selected) + 1 + columns - 1) // columns
    for filename, label, key in (
        ("place_fields.png", "High-activation count", "high_activation_counts"),
        ("place_fields_binary.png", "Active field", "binary_fields"),
        ("place_fields_gaussian.png", "Gaussian density", "gaussian_density"),
    ):
        fig, axes = plt.subplots(rows, columns, figsize=(3.3 * columns, 3 * rows), squeeze=False)
        for index, ax in enumerate(axes.flat):
            if index > len(selected):
                ax.axis("off")
                continue
            if index == 0:
                values = np.where(occupancy > 0, occupancy, np.nan)
                title, cmap = "Occupancy (export samples)", "viridis"
            else:
                channel = selected[index - 1]
                values = np.where(visible, fields[key][channel], np.nan)
                title, cmap = f"{label}: ch {channel}", "Blues"
                if key == "gaussian_density" and fields["gaussian_fit_status"][channel] != "ok":
                    title = f"Ch {channel}: Gaussian unavailable"
            limits = dict(vmin=0, vmax=1) if key == "binary_fields" and index else {}
            plotted = ax.pcolormesh(xedges, zedges, values.T, shading="auto", cmap=cmap, **limits)
            ax.set_title(title, fontsize=9)
            ax.set(xlabel="x", ylabel="z", aspect="equal")
            draw_map_overlay(ax, overlay)
            fig.colorbar(plotted, ax=ax, shrink=0.75)
        fig.tight_layout()
        fig.savefig(out / filename, dpi=140)
        plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
    areas = fields["gaussian_area_1sigma"]
    areas = areas[np.isfinite(areas)]
    if len(areas):
        axes[0].hist(areas, bins=min(20, len(areas)))
    else:
        axes[0].text(0.5, 0.5, "No nonsingular Gaussian fits", ha="center", transform=axes[0].transAxes)
    axes[0].set(xlabel="1-sigma ellipse area (coordinate units squared)", ylabel="Channels")
    coverage = fields["fields_per_position"]
    coverage = np.sort(coverage[coverage > 0])
    axes[1].bar(np.arange(len(coverage)), coverage, width=1)
    axes[1].set(xlabel="Covered spatial bins (sorted)", ylabel="Active channels")
    sizes = np.sort(fields["positions_per_field"])
    axes[2].bar(np.arange(len(sizes)), sizes, width=1)
    axes[2].set(xlabel="Channels (sorted, including empty fields)", ylabel="Active spatial bins")
    fig.tight_layout()
    fig.savefig(out / "place_field_statistics.png", dpi=140)
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--npz", required=True, help="Evaluation trail latent export")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--bins", type=int, nargs="+", default=[30], help="One bin count, or separate x and z counts")
    p.add_argument("--spatial_range", type=float, nargs=4, metavar=("XMIN", "XMAX", "ZMIN", "ZMAX"),
                   help="Fixed world bounds; otherwise use this export's position range")
    p.add_argument("--quantile", type=float, default=0.9, help="Per-channel quantile across exported samples")
    p.add_argument("--min_occupancy", type=int, default=1,
                   help="Display mask only; does not change thresholds, counts, fits or statistics")
    p.add_argument("--units", type=int, default=16, help="Channels to plot; ALL channels are analyzed and saved")
    p.add_argument("--map_root", default="", help="Directory containing map/navigation/occupancy; defaults to export data_root if available")
    args = p.parse_args(argv)
    if len(args.bins) not in (1, 2) or min(*args.bins, args.min_occupancy, args.units) < 1:
        raise ValueError("Plot settings must be positive")
    out = resolve_path(args.out_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Choose a new analysis directory: {out}")
    latents, positions, indices, metadata = read_latents(args.npz)
    bins = args.bins[0] if len(args.bins) == 1 else tuple(args.bins)
    spatial_range = None if args.spatial_range is None else np.asarray(args.spatial_range).reshape(2, 2)
    overlay = load_map_overlay(resolve_path(args.map_root) if args.map_root else None,
                               data_root=metadata.get("data_root"))
    validate_map_positions(overlay, positions)
    if spatial_range is None and overlay is not None:
        spatial_range = np.asarray(overlay["extent"]).reshape(2, 2)
    fields = compute_place_fields(latents, positions, bins=bins, quantile=args.quantile, spatial_range=spatial_range)
    channels = latents.shape[1]
    # Fixed channel indices, not ranked by spatial localization or held-out scores.
    selected = np.unique(np.linspace(0, channels - 1, min(args.units, channels), dtype=int))
    occupancy = fields["occupancy"]
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "place_fields.npz", **fields, plotted_channel_indices=selected)
    plot_place_fields(fields, out, selected, args.min_occupancy, overlay)
    report = dict(samples=len(latents), metadata=metadata, min_occupancy=args.min_occupancy,
                  map_overlay=overlay_report(overlay),
                  analysis_method="channel_percentile_place_fields_v1",
                  place_fields=dict(
                      representation="spatial_mean_per_channel", channels=channels, plotted_channels=selected.tolist(),
                      quantile=args.quantile, comparison="strict_greater_than",
                      threshold_population="all_samples_in_analysis_export",
                      map_statistic="raw_high_activation_sample_count", occupancy_normalized=False,
                      bins=list(occupancy.shape),
                      spatial_range=[[float(fields["x_edges"][0]), float(fields["x_edges"][-1])],
                                     [float(fields["z_edges"][0]), float(fields["z_edges"][-1])]],
                      visited_bins=int((occupancy > 0).sum()),
                      covered_bins=int((fields["fields_per_position"] > 0).sum()),
                      active_sample_count_summary=dict(min=int(fields["active_sample_counts"].min()),
                                                       median=float(np.median(fields["active_sample_counts"])),
                                                       max=int(fields["active_sample_counts"].max())),
                      channels_with_field=int((fields["positions_per_field"] > 0).sum()),
                      gaussian_fit_status_counts={str(status): int((fields["gaussian_fit_status"] == status).sum())
                                                  for status in np.unique(fields["gaussian_fit_status"])},
                      gaussian_area_definition="pi * sqrt(det(sample_covariance)); public-code approx_areas (1-sigma ellipse), not a fixed-density contour",
                      min_occupancy_scope="map_display_only",
                  ),
                  note="Reference-style descriptive place fields, not a significance test. Sampling/heading/history must be matched separately; trail exports are not controlled grid scans.")
    save_json(out / "report.json", report)
    print(json.dumps(report, indent=2), flush=True)
    return report


if __name__ == "__main__":
    main()
