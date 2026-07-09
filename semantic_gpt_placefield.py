"""
semantic_gpt_placefield.py

Pseudo-code / implementation skeleton for:
  - loading semantic_gpt cached latents + grid positions
  - computing per-dimension (channel) place fields
  - thresholding top activations by quantile (e.g. 0.9)
  - projecting selected timepoints onto (x,y) grid
  - ranking channels and plotting montage

INPUT (from semantic_gpt latent cache you saved):
  latents_on_grid.npz:
    z:          (N, L, D, 1, 1)   # semantic_gpt latent per frame
    positions: (N, L, 3)         # x,y,(angle) ; we use first 2 dims
    semantics: (N, L, V)         # optional, only for future concept analyses

GOAL:
  For each latent dimension d in 0..D-1:
    1) get activation a(t) = mean(z(t)[d]) across spatial (here already 1x1 so just z)
    2) compute threshold thr = quantile(a(t), q)
    3) select timepoints t where a(t) > thr
    4) take corresponding positions p(t) and build 2D histogram on map grid
    5) rank channels by simple spatial metrics (area_nonzero, sparsity, peak, ...)
    6) plot top-K channels montage

NOTE:
  This file is intentionally NOT a full runnable implementation.
  Fill in the "PSEUDO IMPLEMENTATION" blocks as needed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple, Dict, List

import numpy as np

import matplotlib
matplotlib.use("Agg")  # headless safe
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import csv
import json
import argparse


#
# -------------------- User-config section (edit values) --------------------
#
DEFAULT_NPZ = Path(
    "/home/ubuntu/project/analysis_out/semantic_gpt_predefined_latents/latents_on_grid.npz"
)

OUT_DIR = Path(
    "/home/ubuntu/project/analysis_out/semantic_gpt_predefined_latents/placefields"
)

QUANTILE = 0.9
BINS = (41, 66)          # x bins, y bins (match your previous scripts)
TOPK = 16
STEPMODE = "all"         # "all" or "last"
ROT90_K = 1              # keep if your plot orientation needs rotation
TWIN_MANSION_EXTENT = (-32.0, 32.0, -32.0, 32.0)  # xmin, xmax, ymin, ymax


def draw_twin_mansion_layout(ax: plt.Axes) -> None:
    """Overlay Twin Mansion room/corridor boundaries in world coordinates."""
    rooms = [
        ("R1", -32.0, -32.0, 24.0, 24.0),
        ("R2", 8.0, -32.0, 24.0, 24.0),
        ("R3", 8.0, 8.0, 24.0, 24.0),
        ("R4", -32.0, 8.0, 24.0, 24.0),
    ]
    corridors = [
        ("C12", -8.0, -24.0, 16.0, 8.0),
        ("C23", 16.0, -8.0, 8.0, 16.0),
        ("C34", -8.0, 16.0, 16.0, 8.0),
    ]
    for _, x, y, w, h in corridors:
        ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor="0.35", linewidth=0.8, linestyle="--"))
    for name, x, y, w, h in rooms:
        ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor="0.05", linewidth=1.1))
        ax.text(x + 1.0, y + h - 2.2, name, fontsize=6, color="0.15", va="top")
    ax.set_xlim(TWIN_MANSION_EXTENT[0], TWIN_MANSION_EXTENT[1])
    ax.set_ylim(TWIN_MANSION_EXTENT[2], TWIN_MANSION_EXTENT[3])
    ax.set_aspect("equal", adjustable="box")

#
# -------------------- Helper: reshape z and positions --------------------
#
def reshape_to_acts_pos(
    z: np.ndarray,
    positions: np.ndarray,
    *,
    stepmode: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    z: (N, L, D, 1, 1)
    positions: (N, L, 3)

    returns:
      acts: (T, D)  # T = N*L if stepmode=all, else T=N
      pos2: (T, 2)  # use (x,y)
    """
    if z.ndim != 5:
        raise ValueError(f"Expected z with shape (N,L,D,1,1). Got z.shape={z.shape}")
    if positions.ndim != 3:
        raise ValueError(f"Expected positions with shape (N,L,3). Got positions.shape={positions.shape}")
    if z.shape[0] != positions.shape[0] or z.shape[1] != positions.shape[1]:
        raise ValueError(
            f"z and positions mismatch: z (N={z.shape[0]},L={z.shape[1]}) vs positions (N={positions.shape[0]},L={positions.shape[1]})"
        )

    # acts_frame: (N, L, D)
    acts_frame = z.mean(axis=(3, 4))
    N, L, D = acts_frame.shape

    if stepmode == "all":
        # (N,L,D)->(N*L,D), (N,L,3)->(N*L,2)
        acts = acts_frame.reshape(-1, D)
        pos2 = positions[:, :, :2].reshape(-1, 2)
    elif stepmode == "last":
        acts = acts_frame[:, -1, :]  # (N,D)
        pos2 = positions[:, -1, :2]  # (N,2)
    else:
        raise ValueError("stepmode must be 'all' or 'last'")

    return acts.astype(np.float32), pos2.astype(np.float32)


#
# -------------------- Place field computation per latent dim --------------------
#
def compute_placefield_histograms(
    acts: np.ndarray,
    pos2: np.ndarray,
    *,
    quantile: float,
    bins: Tuple[int, int],
) -> np.ndarray:
    """
    acts: (T, D)
    pos2: (T, 2)
    returns:
      hist: (D, bins_x, bins_y)
    """
    if acts.ndim != 2 or pos2.ndim != 2 or pos2.shape[1] != 2:
        raise ValueError(f"Expected acts (T,D) and pos2 (T,2). Got acts={acts.shape}, pos2={pos2.shape}")
    T, D = acts.shape

    if pos2.shape[0] != T:
        raise ValueError(f"acts and pos2 length mismatch: T(acts)={T}, pos2={pos2.shape[0]}")

    xmin, xmax, ymin, ymax = TWIN_MANSION_EXTENT

    hist = np.zeros((D, bins[0], bins[1]), dtype=np.float32)

    for d in range(D):
        thr = float(np.quantile(acts[:, d], quantile))
        sel = acts[:, d] > thr
        units = pos2[sel]
        if units.shape[0] == 0:
            continue
        H, _, _ = np.histogram2d(
            units[:, 0],
            units[:, 1],
            bins=bins,
            range=[[xmin, xmax], [ymin, ymax]],
        )
        hist[d] = H.astype(np.float32)

    return hist


#
# -------------------- Rank channels (heuristics) --------------------
#
def rank_channels_by_spatial_metrics(hist: np.ndarray) -> List[Dict[str, float]]:
    """
    hist: (D, bins_x, bins_y)
    return rows sorted by "place-like" heuristics.
    """
    if hist.ndim != 3:
        raise ValueError(f"Expected hist with shape (D,bins_x,bins_y). Got {hist.shape}")
    D = hist.shape[0]
    rows: List[Dict[str, float]] = []

    eps = 1e-12
    for d in range(D):
        H = hist[d] #histogram的D维
        area_nonzero = float((H > 0).sum()) #直方图里非零的格子数，集中=非零面积小
        peak = float(H.max()) #直方图里最大的值
        s = float(H.sum()) #直方图里所有值的和
        sparsity = float(peak / (s + eps)) if s > 0 else 0.0 #稀疏性，峰值/总和

        if peak > 0:
            peak_idx = int(np.argmax(H))
            px, py = np.unravel_index(peak_idx, H.shape)
        else:
            px, py = 0, 0

        rows.append(
            {
                "channel": float(d),
                "area_nonzero": area_nonzero,
                "peak": peak,
                "sum": s,
                "sparsity": sparsity,
                "peak_bin_x": float(px),
                "peak_bin_y": float(py),
            }
        )

    # Heuristic: "place-like" tends to be spatially compact (small area),
    # and concentrated (large sparsity), and have a strong peak.
    # TODO：256个channel，现在这个sorting方式比较随便可能会漏东西
    rows_sorted = sorted(rows, key=lambda r: (r["area_nonzero"], -r["sparsity"], -r["peak"]))
    return rows_sorted


#
# -------------------- Plot montage of top channels --------------------
#
def plot_montage_from_hist(
    hist: np.ndarray,
    top_channels: List[int],
    out_png: Path,
    *,
    rot90_k: int = 1,
    cmap: str = "Blues",
    draw_layout: bool = True,
) -> None:
    """
    PSEUDO IMPLEMENTATION:
      - create matplotlib figure grid
      - for each channel in top_channels:
          binary or raw mask from hist[d]
          optionally rotate (rot90_k) to match room layout
          imshow on subplot
      - save figure to out_png
    """
    if len(top_channels) == 0:
        return

    import math

    n = len(top_channels)
    ncols = int(math.ceil(math.sqrt(n)))
    nrows = int(math.ceil(n / ncols))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(3 * ncols, 3 * nrows))
    axes = np.array(axes).reshape(-1)

    for ax in axes:
        ax.axis("off")

    for ax, d in zip(axes, top_channels):
        H = hist[int(d)]
        # binary visualization (place field "visited bins")
        mask = (H > 0).astype(np.float32)
        ax.imshow(
            mask.T,
            cmap=cmap,
            origin="lower",
            extent=TWIN_MANSION_EXTENT,
            alpha=mask.T * 0.9,
            interpolation="nearest",
        )
        if draw_layout:
            draw_twin_mansion_layout(ax)
        ax.set_title(f"ch {int(d)}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

    # turn off leftover axes
    for ax in axes[len(top_channels) :]:
        ax.axis("off")

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


#
# -------------------- Main --------------------
#
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", type=str, default=str(DEFAULT_NPZ), help="Path to latents_on_grid.npz")
    ap.add_argument("--out_dir", type=str, default=str(OUT_DIR), help="Base output directory")
    ap.add_argument("--stepmode", type=str, default=STEPMODE, choices=["all", "last"])
    ap.add_argument("--quantile", type=float, default=float(QUANTILE))
    ap.add_argument("--bins", type=str, default=f"{BINS[0]},{BINS[1]}", help="like '41,66'")
    ap.add_argument("--mode", type=str, default="rank", choices=["rank", "channels"])
    ap.add_argument("--channels", type=str, default="", help="Comma-separated channel ids, e.g. '3,7,20'")
    ap.add_argument("--topk", type=int, default=int(TOPK), help="topK channels for mode=rank")
    ap.add_argument("--rot90_k", type=int, default=int(ROT90_K))
    args = ap.parse_args()

    npz_path = Path(args.npz)
    if not npz_path.exists():
        raise FileNotFoundError(f"npz not found: {npz_path}")

    bins_parts = [int(x.strip()) for x in args.bins.split(",") if x.strip()]
    if len(bins_parts) != 2:
        raise ValueError("--bins must be like '41,66'")
    bins = (bins_parts[0], bins_parts[1])

    # load npz
    data = np.load(npz_path, allow_pickle=True)
    if "z" not in data or "positions" not in data:
        raise ValueError(f"Expected keys 'z' and 'positions' in npz. Got keys={list(data.keys())}")
    z = data["z"]
    positions = data["positions"]

    # reshape to (acts, pos2)
    acts, pos2 = reshape_to_acts_pos(z, positions, stepmode=args.stepmode)

    # compute per-dim histograms
    hist = compute_placefield_histograms(acts, pos2, quantile=args.quantile, bins=bins)

    # choose channels
    if args.mode == "channels":
        if not args.channels.strip():
            raise ValueError("--channels must be provided when --mode=channels")
        top_channels = [int(x.strip()) for x in args.channels.split(",") if x.strip()]
        if any(c < 0 or c >= hist.shape[0] for c in top_channels):
            raise ValueError(f"--channels contains out-of-range id. hist has D={hist.shape[0]}")
        ranks = None
    else:
        ranks = rank_channels_by_spatial_metrics(hist)
        top_channels = [int(r["channel"]) for r in ranks[: int(args.topk)]]

    # output dir
    out_dir = Path(args.out_dir) / args.stepmode / f"q{args.quantile}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # save hist
    np.save(out_dir / "hist.npy", hist)

    # save ranks csv (only in rank mode)
    if args.mode == "rank" and ranks is not None:
        rank_path = out_dir / f"rank_top{int(args.topk)}.csv"
        with rank_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "channel",
                    "area_nonzero",
                    "peak",
                    "sum",
                    "sparsity",
                    "peak_bin_x",
                    "peak_bin_y",
                ],
            )
            w.writeheader()
            for row in ranks:
                w.writerow(row)

    # save summary json
    ch_tag = "_".join([f"ch{c}" for c in top_channels])
    summary = {
        "npz": str(npz_path),
        "stepmode": args.stepmode,
        "acts_shape": list(acts.shape),
        "pos2_shape": list(pos2.shape),
        "hist_shape": list(hist.shape),
        "quantile": float(args.quantile),
        "bins": list(bins),
        "top_channels": top_channels,
        "topk": int(args.topk),
        "mode": args.mode,
        "requested_channels": args.channels if args.mode == "channels" else None,
        "channel_tag": ch_tag,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # plot montage (one figure)
    if args.mode == "channels":
        out_png = out_dir / f"montage_placefields_{args.stepmode}_{ch_tag}.png"
    else:
        out_png = out_dir / f"montage_placefields_{args.stepmode}_top{int(args.topk)}.png"

    plot_montage_from_hist(hist, top_channels=top_channels, out_png=out_png, rot90_k=args.rot90_k)

    print("[done]")
    print(f"acts={acts.shape}, pos2={pos2.shape}, hist={hist.shape}")
    print(f"top_channels={top_channels}")
    print(f"saved: {out_dir}")


if __name__ == "__main__":
    main()

