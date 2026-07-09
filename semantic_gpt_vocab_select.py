"""
semantic_gpt_vocab_select.py (pseudo-code skeleton)

Goal (per your request):
  1) vocabulary粒度（object class / V 维）而不是 subword 粒度
  2) 对每个 vocab id，找出“激活最高的高激活 latent channels”
  3) 激活 score 先用：该 vocab 出现帧的 latent 平均激活（mean activation）
  4) 只需要“终端打印 + 数据保存”，不做可视化

Input:
  - npz from semantic_gpt_latent extraction on grid:
      latents_on_grid.npz
        z:          (N, L, D, 1, 1)
        positions: (N, L, 3)          # optional for this script
        semantics: (N, L, V)          # multi-hot vocab labels per frame

Optional:
  - vocabulary.npy to map vocab id -> readable object name

Outputs:
  - vocab_select_topk.json / .csv / .npz (choose one; skeleton shows both)
  - terminal prints for sanity check

IMPORTANT:
  This file is intentionally NOT fully implemented. It contains:
    - function skeletons
    - shape-aware pseudo-code
    - TODO blocks where you (or later we) fill in details
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import argparse

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

"""
GPT:
python semantic_gpt_vocab_select.py \
  --npz "/home/ubuntu/project/analysis_out/semantic_gpt_predefined_latents/latents_on_grid.npz" \
  --out_dir "/home/ubuntu/project/analysis_out/semantic_gpt_predefined_latents/vocab_select" \
  --vocab_npy "/home/ubuntu/project/data/data_11272025_twinmansion/vocabulary.npy" \
  --step_mode all --topk 10 --thr_sem 0.5 \
  --enable_heatmap \
  --heatmap_channels "114" \
  --heatmap_out "/home/ubuntu/project/analysis_out/semantic_ae_predefined" \
  --heatmap_zscore row \
  --enable_vocab_lineplot

AE:
python semantic_gpt_vocab_select.py \
  --npz "/home/ubuntu/project/analysis_out/semantic_ae_predefined_latents/latents_on_grid_ae.npz" \
  --out_dir "/home/ubuntu/project/analysis_out/semantic_ae_predefined_latents/vocab_select" \
  --vocab_npy "/home/ubuntu/project/data/data_11272025_twinmansion/vocabulary.npy" \
  --step_mode all --topk 10 --thr_sem 0.5 \
  --enable_heatmap \
  --heatmap_channels "191" \
  --heatmap_out "/home/ubuntu/project/analysis_out/semantic_ae_predefined" \
  --heatmap_zscore row \
  --enable_vocab_lineplot

Full channel×vocab heatmap with rows sorted by each channel's peak vocab (exploratory):
  add --heatmap_row_sort peak_vocab and omit --heatmap_channels (or pass only when subsetting).

Windows (repo defaults): do not type literal ... as an argument.
  python semantic_gpt_vocab_select.py --enable_heatmap --heatmap_row_sort peak_vocab --step_mode all --heatmap_zscore row
"""
# ------------------------ config (edit) ------------------------
DEFAULT_NPZ = Path(
    r"D:\workspace\PRED_CODING\A_gs_project\analysis_out\semantic_gpt_predefined_latents\latents_on_grid.npz"
)
DEFAULT_VOCABULARY_NPY = Path(
    r"D:\workspace\PRED_CODING\A_gs_project\data\data_11272025_twinmansion\vocabulary.npy"
)  # optional; set to None if you don't want names

OUT_DIR = Path(
    r"D:\workspace\PRED_CODING\A_gs_project\analysis_out\semantic_gpt_predefined_latents\vocab_select"
)

TOPK = 10                 # how many channels to report per vocab id
STEP_MODE = "all"         # "all" or "last"
THRESH_SEM = 0.5          # semantics active threshold
ENABLE_HEATMAP = False   # set True to generate channel x vocab heatmap
HEATMAP_OUT = OUT_DIR / "channel_x_vocab_heatmap.png"
# ---------------------------------------------------------------


def load_npz(npz_path: Path) -> Dict[str, np.ndarray]:
    """
    Load cached arrays:
      z: (N, L, D, 1, 1)
      semantics: (N, L, V)
    """
    d = np.load(npz_path, allow_pickle=True)
    keys = list(d.keys())
    if "z" not in d or "semantics" not in d:
        raise ValueError(f"Expected keys 'z' and 'semantics' in {npz_path}, got keys={keys}")
    return {k: d[k] for k in keys}


def load_vocabulary_names(vocab_npy: Optional[Path]) -> Optional[np.ndarray]:
    """
    Returns:
      vocab: 1D array of object names, length V
    """
    if vocab_npy is None:
        return None
    if not vocab_npy.exists():
        raise FileNotFoundError(f"vocabulary.npy not found: {vocab_npy}")
    vocab = np.load(vocab_npy, allow_pickle=True)
    return vocab


def reshape_acts_semantics(
    z: np.ndarray,
    semantics: np.ndarray,
    *,
    stepmode: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Shapes:
      z: (N, L, D, 1, 1)
      semantics: (N, L, V)

    Returns:
      acts: (T, D)          # per-frame latent activation used as "channel activation"
      sem:  (T, V)          # multi-hot vocab labels aligned with acts

    PSEUDO IMPLEMENTATION:
      acts_frame = z.mean(axis=(3,4))  # (N, L, D)

      if stepmode == "all":
         acts = acts_frame.reshape(-1, D)          # T = N*L
         sem  = semantics.reshape(-1, V)           # T = N*L
      elif stepmode == "last":
         acts = acts_frame[:, -1, :]              # (N, D)
         sem  = semantics[:, -1, :]              # (N, V)
      else:
         raise ValueError
    """
    if z.ndim != 5:
        raise ValueError(f"Expected z with shape (N,L,D,1,1). Got {z.shape}")
    if semantics.ndim != 3:
        raise ValueError(f"Expected semantics with shape (N,L,V). Got {semantics.shape}")
    if z.shape[0] != semantics.shape[0] or z.shape[1] != semantics.shape[1]:
        raise ValueError(
            f"z and semantics mismatch: z (N={z.shape[0]},L={z.shape[1]}) vs semantics (N={semantics.shape[0]},L={semantics.shape[1]})"
        )

    acts_frame = z.mean(axis=(3, 4))  # (N, L, D)
    N, L, D = acts_frame.shape
    N2, L2, V = semantics.shape
    assert N2 == N and L2 == L

    if stepmode == "all":
        acts = acts_frame.reshape(-1, D)        # (T, D)
        sem = semantics.reshape(-1, V)         # (T, V)
    elif stepmode == "last":
        acts = acts_frame[:, -1, :]            # (N, D)
        sem = semantics[:, -1, :]             # (N, V)
    else:
        raise ValueError("stepmode must be 'all' or 'last'")

    return acts.astype(np.float32), sem.astype(np.float32)


def compute_top_channels_per_vocab(
    acts: np.ndarray,     # (T, D)
    sem: np.ndarray,      # (T, V)
    *,
    topk: int,
    thr_sem: float,
) -> Tuple[List[int], np.ndarray, np.ndarray, np.ndarray]:
    """
    For each vocab id c in [0..V-1]:
      mask_on = sem[:, c] > thr_sem          # boolean for frames where vocab c is active

      mean_on_d = mean(acts[mask_on, d] for each d)    # (D,)
      score[d] = mean_on_d (your requested "average activation" score)

      ranking = argsort(score descending)
      select topk channels

    Outputs:
      top_channels_per_vocab: list/array with shape (V, topk)
      top_scores_per_vocab:   same shape (V, topk)
      vocab_ids: [0..V-1] (or equivalent)

    NOTE:
      If a vocab c never appears (mask_on all False), set its scores to -inf and channels empty or filled with -1.
    """
    T, D = acts.shape
    T2, V = sem.shape
    assert T2 == T

    top_channels_per_vocab: List[List[int]] = []
    top_scores_per_vocab: List[List[float]] = []
    score_matrix = np.full((V, D), -np.inf, dtype=np.float32)  # (V, D) keep for potential heatmap usage

    for c in range(V):
        mask_on = sem[:, c] > thr_sem
        if mask_on.sum() == 0:
            # never appears
            top_channels_per_vocab.append([-1] * topk)
            top_scores_per_vocab.append([float("-inf")] * topk)
            continue

        # score[d] = mean activation over frames where vocab c is active
        mean_on = acts[mask_on].mean(axis=0)  # (D,)
        score_matrix[c] = mean_on

        # rank channels by mean_on descending
        idx = np.argsort(mean_on)[::-1][:topk]
        top_channels_per_vocab.append([int(x) for x in idx])
        top_scores_per_vocab.append([float(mean_on[x]) for x in idx])

    # (V, topk)
    top_channels_arr = np.array(top_channels_per_vocab, dtype=np.int64)
    top_scores_arr = np.array(top_scores_per_vocab, dtype=np.float32)
    vocab_ids = list(range(V))
    return vocab_ids, top_channels_arr, top_scores_arr, score_matrix


def sort_heatmap_rows_by_peak_vocab(heat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reorder channel rows for visualization: primary key = vocab index of the row's
    maximum mean-on activation (finite entries only); secondary = descending peak value.
    Rows with no finite entry sort last.

    heat: (D, V) mean activation matrix; may contain -inf where a vocab never appears.

    Returns:
      heat_sorted: (D, V) rows permuted
      perm: (D,) int64 — perm[i] is the original row index placed at visual row i
    """
    if heat.ndim != 2:
        raise ValueError(f"Expected heat (D,V), got shape={heat.shape}")
    D, V = heat.shape
    peak_vocab = np.zeros(D, dtype=np.int64)
    peak_val = np.full(D, -np.inf, dtype=np.float64)
    for j in range(D):
        row = heat[j].astype(np.float64)
        ok = np.isfinite(row)
        if not ok.any():
            peak_vocab[j] = V
            continue
        masked = np.where(ok, row, -np.inf)
        vi = int(np.argmax(masked))
        peak_vocab[j] = vi
        peak_val[j] = float(row[vi])
    order = np.lexsort((-peak_val, peak_vocab))
    perm = order.astype(np.int64)
    return heat[perm], perm


def zscore_rows_across_vocab(heat: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    For each channel row d, z-score across vocab columns v:
      z[d,v] = (heat[d,v] - mean_v heat[d,:]) / (std_v heat[d,:] + eps)
    heat: (D, V)
    """
    if heat.ndim != 2:
        raise ValueError(f"Expected heat (D,V), got shape={heat.shape}")
    mu = heat.mean(axis=1, keepdims=True)
    sigma = heat.std(axis=1, keepdims=True)
    return (heat - mu) / (sigma + eps)


def _truncate_label(s: str, max_len: int = 28) -> str:
    s = str(s)
    if len(s) <= max_len:
        return s
    return s[: max_len - 3] + "..."


def plot_vocab_profiles_psth(
    heat_plot: np.ndarray,
    *,
    channel_ids: List[int],
    vocab_names: Optional[np.ndarray],
    step_mode: str,
    heatmap_zscore: str,
    out_png: Path,
) -> None:
    """
    PSTH-like line plot: x = vocab index (labeled by name), y = value per vocab for each channel row.
    heat_plot: (n_channels, V), same matrix as heatmap after optional row z-score.
    """
    n_ch, V = heat_plot.shape
    if n_ch != len(channel_ids):
        raise ValueError(f"heat_plot rows {n_ch} != len(channel_ids) {len(channel_ids)}")
    if V == 0:
        raise ValueError("empty vocab dimension")

    x = np.arange(V, dtype=np.int32)
    if vocab_names is not None and len(vocab_names) == V:
        xlabels = [_truncate_label(vocab_names[i]) for i in range(V)]
    else:
        xlabels = [str(i) for i in range(V)]

    z_tag = "zscore_row" if heatmap_zscore == "row" else "raw"
    out_png.parent.mkdir(parents=True, exist_ok=True)

    fig_h = max(3.0, 2.2 * n_ch)
    fig_w = max(10.0, min(28.0, 0.22 * V + 4.0))
    fig, axes = plt.subplots(
        n_ch,
        1,
        sharex=True,
        figsize=(fig_w, fig_h),
        squeeze=False,
    )
    for i, ch in enumerate(channel_ids):
        ax = axes[i, 0]
        y = heat_plot[i].astype(np.float64)
        ax.plot(x, y, color="C0", linewidth=1.2, marker="o", markersize=2.5, alpha=0.85)
        ax.axhline(0.0, color="0.5", linewidth=0.6, linestyle="--", alpha=0.7)
        ax.set_ylabel(f"ch {ch}\n({z_tag})")
        ax.grid(True, alpha=0.25)
    axes[-1, 0].set_xticks(x)
    axes[-1, 0].set_xticklabels(xlabels, rotation=90, ha="right", fontsize=7)
    axes[-1, 0].set_xlabel("vocabulary (object name or id)")
    fig.suptitle(f"Channel tuning across vocabulary ({z_tag}, stepmode={step_mode})", fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_results_json(
    out_dir: Path,
    payload: Dict[str, Any],
    filename: str,
) -> Path:
    """
    PSEUDO:
      out_dir.mkdir(parents=True, exist_ok=True)
      path = out_dir / filename
      write json.dumps(payload)
    """
    import json

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


def main() -> None:
    """
    End-to-end pseudo pipeline:
      1) load npz (z, semantics)
      2) acts, sem = reshape_acts_semantics(...)
      3) compute top channels per vocab using mean activation score
      4) load vocabulary.npy names (optional)
      5) terminal print:
           for each vocab c:
             print name (if available), c, top_channels list, top_scores
      6) save to json/npz
    """
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--npz",
        type=str,
        default=str(DEFAULT_NPZ),
        help="Path to latents npz containing z/semantics",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default=str(OUT_DIR),
        help="Output directory for optional heatmap",
    )
    ap.add_argument(
        "--vocab_npy",
        type=str,
        default=str(DEFAULT_VOCABULARY_NPY),
        help="Path to vocabulary.npy (set empty string to disable names)",
    )
    ap.add_argument(
        "--step_mode",
        type=str,
        default=STEP_MODE,
        choices=["all", "last"],
        help="Use all frames or last frame only",
    )
    ap.add_argument(
        "--topk",
        type=int,
        default=int(TOPK),
        help="Top-k channels per vocab",
    )
    ap.add_argument(
        "--thr_sem",
        type=float,
        default=float(THRESH_SEM),
        help="Semantics active threshold",
    )
    ap.add_argument(
        "--enable_heatmap",
        action="store_true",
        help="Generate channel x vocab heatmap (overrides config ENABLE_HEATMAP if set)",
    )
    ap.add_argument(
        "--heatmap_channels",
        type=str,
        default="",
        help="Comma-separated channel ids to plot only those rows, e.g. '149,157'. Empty means plot all.",
    )
    ap.add_argument(
        "--heatmap_out",
        type=str,
        default="",
        help="Override heatmap output png path (optional).",
    )
    ap.add_argument(
        "--heatmap_zscore",
        type=str,
        default="row",
        choices=["row", "none"],
        help="row: z-score each channel across vocab (default); none: raw mean activation",
    )
    ap.add_argument(
        "--heatmap_row_sort",
        type=str,
        default="none",
        choices=["none", "peak_vocab"],
        help="peak_vocab: reorder rows by argmax vocab (then peak value) for exploratory "
        "full-map readability; none keeps native channel index order",
    )
    ap.add_argument(
        "--enable_vocab_lineplot",
        action="store_true",
        help="With --enable_heatmap: also save PSTH-like line plot (vocab on x, score on y) per selected channel",
    )
    ap.add_argument(
        "--vocab_lineplot_out",
        type=str,
        default="",
        help="Output png for vocab line plot; empty = derive from heatmap_out name",
    )
    args = ap.parse_args()

    npz_path = Path(args.npz)
    out_dir = Path(args.out_dir)
    vocab_npy = Path(args.vocab_npy) if str(args.vocab_npy).strip() else None
    step_mode = args.step_mode
    topk = int(args.topk)
    thr_sem = float(args.thr_sem)

    enable_heatmap = bool(ENABLE_HEATMAP) or bool(args.enable_heatmap)
    selected_channels: List[int] = []
    if args.heatmap_channels.strip():
        selected_channels = [int(x.strip()) for x in args.heatmap_channels.split(",") if x.strip()]
        if len(selected_channels) == 0:
            raise ValueError("--heatmap_channels parsed to empty list")

    heatmap_out = Path(args.heatmap_out) if args.heatmap_out.strip() else out_dir / HEATMAP_OUT.name
    heatmap_zscore = args.heatmap_zscore
    heatmap_row_sort = args.heatmap_row_sort
    enable_vocab_lineplot = bool(args.enable_vocab_lineplot)
    vocab_lineplot_out_arg = Path(args.vocab_lineplot_out) if str(args.vocab_lineplot_out).strip() else None

    if not npz_path.exists():
        raise FileNotFoundError(f"npz not found: {npz_path}")

    d = load_npz(npz_path)
    z = d["z"]
    semantics = d["semantics"]

    acts, sem = reshape_acts_semantics(z, semantics, stepmode=step_mode)
    T, D = acts.shape
    V = sem.shape[1]

    vocab_names = load_vocabulary_names(vocab_npy)
    if vocab_names is not None and len(vocab_names) != V:
        print(f"[warn] vocabulary length={len(vocab_names)} but semantics V={V}. Use ids only.")
        vocab_names = None

    vocab_ids, top_channels, top_scores, score_matrix_vd = compute_top_channels_per_vocab(
        acts,
        sem,
        topk=topk,
        thr_sem=thr_sem,
    )

    # ---- terminal prints ----
    print(f"[semantic_gpt_vocab_select] stepmode={step_mode} T={T} D={D} V={V} topk={topk}")
    print(f"[semantic_gpt_vocab_select] THRESH_SEM={thr_sem}")

    for c in vocab_ids:
        name = f"(id={c})"
        if vocab_names is not None:
            name = str(vocab_names[c])

        chans = top_channels[c]
        scores = top_scores[c]
        valid = chans[0] != -1
        if not valid:
            print(f"  vocab[{c}] {name}: never appears (no frames above threshold)")
            continue

        # show channel indices and scores
        ch_list = ", ".join([f"{int(ch)}:{scores[i]:.4f}" for i, ch in enumerate(chans) if ch >= 0])
        print(f"  vocab[{c}] {name}: top{topk} channels -> {ch_list}")

    # ---- optional heatmap ----
    if enable_heatmap:
        # score_matrix_vd: (V,D); we want (D,V) for plotting with channels on y-axis
        heat = score_matrix_vd.T  # (D,V)

        heatmap_out_local = heatmap_out
        channel_ids_for_lineplot: List[int] = []
        if selected_channels:
            D_all = heat.shape[0]
            if any((ch < 0 or ch >= D_all) for ch in selected_channels):
                raise ValueError(f"--heatmap_channels contains out-of-range id. heat has D={D_all}")
            heat = heat[selected_channels, :]  # (len(selected_channels), V)
            channel_ids_for_lineplot = list(selected_channels)

            ch_tag = "_".join([f"ch{ch}" for ch in selected_channels])
            heatmap_out_local = heatmap_out.with_name(heatmap_out.stem + f"_{ch_tag}" + heatmap_out.suffix)
        else:
            channel_ids_for_lineplot = []

        if heatmap_row_sort == "peak_vocab":
            heat, perm = sort_heatmap_rows_by_peak_vocab(heat)
            if channel_ids_for_lineplot:
                channel_ids_for_lineplot = [channel_ids_for_lineplot[int(i)] for i in perm]
            sort_tag = "_rowsort_peakvocab"
            heatmap_out_local = heatmap_out_local.with_name(
                heatmap_out_local.stem + sort_tag + heatmap_out_local.suffix
            )
            print(
                "[heatmap] rows reordered by peak vocab index (exploratory viz; "
                "native channel order is arbitrary)"
            )

        heat = np.nan_to_num(heat, nan=0.0, posinf=0.0, neginf=0.0)

        if heatmap_zscore == "row":
            heat_plot = zscore_rows_across_vocab(heat.astype(np.float64)).astype(np.float32)
            cbar_label = "z-score (per channel across vocab)"
        else:
            heat_plot = heat
            cbar_label = "mean activation (on frames)"

        heatmap_out_local.parent.mkdir(parents=True, exist_ok=True)

        y_rows = heat_plot.shape[0]
        plt.figure(figsize=(max(6, V * 0.8), max(4, y_rows * 0.015)))
        plt.imshow(heat_plot, aspect="auto", cmap="viridis")
        plt.colorbar(label=cbar_label)
        plt.xlabel("vocab id")
        plt.ylabel("channel/unit id")
        z_tag = "zscore_row" if heatmap_zscore == "row" else "raw"
        sort_note = ", rows sorted by peak vocab" if heatmap_row_sort == "peak_vocab" else ""
        plt.title(f"channel x vocab heatmap ({z_tag}, stepmode={step_mode}{sort_note})")
        if selected_channels:
            plt.yticks(
                range(len(channel_ids_for_lineplot)),
                [str(ch) for ch in channel_ids_for_lineplot],
                fontsize=8,
            )
        plt.tight_layout()
        plt.savefig(heatmap_out_local, dpi=200)
        plt.close()
        print(f"[heatmap] saved to {heatmap_out_local}")

        if enable_vocab_lineplot:
            if not selected_channels:
                print(
                    "[vocab_lineplot] skipped: set --heatmap_channels to one or more channel ids "
                    "(line plot is only generated for selected rows)."
                )
            else:
                if vocab_lineplot_out_arg is not None:
                    line_out = vocab_lineplot_out_arg
                else:
                    line_out = heatmap_out_local.with_name(
                        heatmap_out_local.stem + "_vocab_profiles" + heatmap_out_local.suffix
                    )
                plot_vocab_profiles_psth(
                    heat_plot,
                    channel_ids=channel_ids_for_lineplot,
                    vocab_names=vocab_names,
                    step_mode=step_mode,
                    heatmap_zscore=heatmap_zscore,
                    out_png=line_out,
                )
                print(f"[vocab_lineplot] saved to {line_out}")

    print("[done]")

if __name__ == "__main__":
    main()