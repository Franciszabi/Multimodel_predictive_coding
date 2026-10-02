"""Plot saved position predictions against true trajectories, without inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.spatial_plotting import draw_map_overlay, load_map_overlay, validate_map_positions


def read_predictions(path):
    with np.load(path, allow_pickle=False) as archive:
        required = {"true", "predicted", "input_indices"}
        if not required.issubset(archive.files):
            raise ValueError("Use the decoder's position_predictions.npz, not a latent export")
        true = np.asarray(archive["true"], dtype=np.float64)
        predicted = np.asarray(archive["predicted"], dtype=np.float64)
        indices = archive["input_indices"]
        report = json.loads(str(archive["metadata_json"])) if "metadata_json" in archive.files else {}
    if true.ndim != 2 or true.shape[1] != 2 or predicted.shape != true.shape:
        raise ValueError("Expected true and predicted arrays with matching [N,2] shapes")
    if not len(true) or not np.isfinite(true).all() or not np.isfinite(predicted).all():
        raise ValueError("Position arrays must be nonempty and finite")
    if indices.shape != (len(true),) or not np.issubdtype(indices.dtype, np.integer):
        raise ValueError("Expected one integer input_indices entry per position")
    if np.any(indices < 0) or np.any(np.diff(indices) <= 0):
        raise ValueError("input_indices must be nonnegative and strictly increasing")
    return true, predicted, indices, report


def segment_indices(indices, metadata, data_root, max_gap=None):
    breaks = np.zeros(len(indices), dtype=bool)
    breaks[0] = True
    gaps = np.diff(indices)
    expected = metadata.get("stride")
    if max_gap is None:
        max_gap = int(expected) if expected is not None else (int(np.median(gaps)) if len(gaps) else 1)
    if max_gap < 1:
        raise ValueError("max_gap must be positive")
    breaks[1:] |= gaps > max_gap
    episode_source = None
    if data_root is not None and (data_root / "episodes.npy").is_file():
        episodes = np.load(data_root / "episodes.npy", mmap_mode="r", allow_pickle=False)
        if episodes.ndim == 2 and episodes.shape[1] == 1:
            episodes = episodes[:, 0]
        if episodes.ndim != 1 or len(episodes) <= indices[-1]:
            raise ValueError("episodes.npy does not cover the selected input indices")
        # Detect any boundary between endpoints, even if an episode ID is reused.
        changes = np.r_[0, np.cumsum(episodes[1:] != episodes[:-1])]
        breaks[1:] |= changes[indices[1:]] != changes[indices[:-1]]
        episode_source = str(data_root / "episodes.npy")
    starts = np.flatnonzero(breaks)
    segments = [slice(int(start), int(stop)) for start, stop in zip(starts, np.r_[starts[1:], len(indices)])]
    return segments, max_gap, episode_source


def plot_paths(ax, true, predicted, segments, *, relative=False, overlay=None):
    from matplotlib.lines import Line2D

    colors = ("#147D92", "#D25436")
    if not relative:
        draw_map_overlay(ax, overlay)
    plotted = []
    for segment in segments:
        for coordinates, color, marker in zip((true, predicted), colors, ("o", "x")):
            points = coordinates[segment]
            if relative:
                points = points - points[0]
            plotted.append(points)
            ax.plot(points[:, 0], points[:, 1], color=color, marker=marker,
                    markersize=3, linewidth=1.2, alpha=0.85, zorder=8)
            ax.scatter(*points[0], marker="^", s=65, color=color, edgecolors="white", linewidths=0.6, zorder=9)
            ax.scatter(*points[-1], marker="s", s=35, color=color, edgecolors="white", linewidths=0.6, zorder=9)
    all_points = np.concatenate(plotted)
    if overlay is not None and not relative:
        xmin, xmax, zmin, zmax = overlay["extent"]
        all_points = np.concatenate((all_points, [[xmin, zmin], [xmax, zmax]]))
    low, high = all_points.min(axis=0), all_points.max(axis=0)
    margin = np.maximum(high - low, 1.0) * 0.06
    # Include out-of-map predictions rather than clipping the decoding failures.
    ax.set(xlim=(low[0] - margin[0], high[0] + margin[0]),
           ylim=(low[1] - margin[1], high[1] + margin[1]), aspect="equal",
           xlabel="Displacement x" if relative else "World x",
           ylabel="Displacement z" if relative else "World z",
           title="Relative motion: each segment starts at zero" if relative else "True and decoded positions")
    handles = [Line2D([], [], color=color, marker=marker, label=label)
               for color, marker, label in zip(colors, ("o", "x"), ("True", "Decoded"))]
    handles += [Line2D([], [], color="#555555", marker=marker, linestyle="none", label=label)
                for marker, label in (("^", "Segment start"), ("s", "Segment end"))]
    ax.legend(handles=handles, loc="best", fontsize=8)
    ax.grid(alpha=0.15)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz", required=True, help="position_predictions.npz from visual_position_decoder.py")
    parser.add_argument("--out", help="Output PNG; default: trajectory_START_END.png beside the NPZ. Replaces that PNG only.")
    parser.add_argument("--start_frame", type=int, help="Inclusive original input frame index; default: first saved frame")
    stop = parser.add_mutually_exclusive_group()
    stop.add_argument("--end_frame", type=int, help="Exclusive original input frame index; default: last saved frame + 1")
    stop.add_argument("--num_frames", type=int, help="Original frame span beginning at start_frame, NOT the number of exported points")
    parser.add_argument("--relative", action="store_true", help="Add a start-zeroed motion panel; no rotation, scaling, or fitted alignment")
    parser.add_argument("--map_root", help="Optional matching Unity map metadata for room/obstacle overlays")
    parser.add_argument("--data_root", help="Override the source trail root, e.g. after moving data; used to detect episode boundaries")
    parser.add_argument("--max_gap", type=int, help="Break lines at larger frame gaps; default: export stride, or median saved gap")
    args = parser.parse_args(argv)
    if args.start_frame is not None and args.start_frame < 0:
        raise ValueError("start_frame must be nonnegative")
    if args.num_frames is not None and args.num_frames < 1:
        raise ValueError("num_frames must be positive")
    if args.max_gap is not None and args.max_gap < 1:
        raise ValueError("max_gap must be positive")
    source = Path(args.npz).expanduser().resolve()
    true, predicted, indices, report = read_predictions(source)
    metadata = report.get("evaluation_metadata", {})
    if metadata.get("sampling_protocol", "trail_windows").startswith("scan_"):
        raise ValueError("Scan exports contain teleports, not a continuous exploration trajectory; use a trail decoder export")
    start = int(indices[0]) if args.start_frame is None else args.start_frame
    end = start + args.num_frames if args.num_frames is not None else args.end_frame
    end = int(indices[-1]) + 1 if end is None else end
    if end <= start:
        raise ValueError("end_frame must be greater than start_frame")
    selected = (indices >= start) & (indices < end)
    if selected.sum() < 2:
        raise ValueError(f"Range [{start},{end}) contains fewer than two exported points; increase num_frames or change the range")
    true, predicted, indices = true[selected], predicted[selected], indices[selected]
    root = args.data_root or metadata.get("data_root")
    data_root = Path(root).expanduser().resolve() if root else None
    segments, gap, episode_source = segment_indices(indices, metadata, data_root, args.max_gap)
    overlay = load_map_overlay(Path(args.map_root).expanduser().resolve() if args.map_root else None,
                               data_root=data_root)
    validate_map_positions(overlay, true)
    output = Path(args.out).expanduser().resolve() if args.out else source.with_name(f"trajectory_{start}_{end}.png")
    if output.suffix.lower() != ".png":
        raise ValueError("out must be a .png image")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    columns = 2 if args.relative else 1
    fig, axes = plt.subplots(1, columns, figsize=(8 * columns, 7), squeeze=False)
    plot_paths(axes[0, 0], true, predicted, segments, overlay=overlay)
    if args.relative:
        plot_paths(axes[0, 1], true, predicted, segments, relative=True)
    gaps = np.diff(indices)
    fig.suptitle(f"Input frames {indices[0]} to {indices[-1]} | {len(indices)} sampled points | {len(segments)} segment(s)", fontsize=12)
    fig.text(0.5, 0.015, f"Sampled endpoints only; frame gaps min/median/max = {gaps.min()}/{np.median(gaps):g}/{gaps.max()}. "
             "Lines do not reconstruct unobserved intermediate motion.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)
    error = np.linalg.norm(predicted - true, axis=1)
    increments = [np.linalg.norm(np.diff(predicted[s], axis=0) - np.diff(true[s], axis=0), axis=1)
                  for s in segments if s.stop - s.start > 1]
    print(f"[selection] requested=[{start},{end}) saved_frames={indices[0]}..{indices[-1]} points={len(indices)} segments={len(segments)}")
    print(f"[sampling] frame_gap min/median/max={gaps.min()}/{np.median(gaps):g}/{gaps.max()}; break_above={gap}")
    print(f"[error] mean_position_distance={error.mean():.6f}")
    if increments:
        print(f"[motion] mean_displacement_error_per_sampled_interval={np.concatenate(increments).mean():.6f}")
    if episode_source is None:
        print("[note] episodes.npy unavailable; only saved frame gaps can identify breaks. Use --data_root if the trail moved.")
    print(f"[done] {output}")
    return output


if __name__ == "__main__":
    main()
