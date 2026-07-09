"""
Find latent channels that look like “one vocab high, others low” on the same heatmap as
semantic_gpt_vocab_select.py (mean activation on frames where each vocab is on).

Default ranking (--metric heatmap_peak_sparse):
  For each channel j, build the row heat[j, :] = mean(z_j | vocab c on), same as vocab_select.
  Score = peak − median(rest): among vocabs with enough data, take the max mean, subtract the
  median of the *other* means. Large values ≈ one sharp peak vs flat/off background.

Optional (--metric roc_auc): pairwise ROC-AUC vs on/off (often noisier for sparse concepts).

Latent / step_mode (same as vocab_select):
  - z: (N, L, D, 1, 1); pooled to (N, L, D), then step_mode "all" → (N*L, D) or "last" → (N, D).

Outputs:
  - selectivity_top_channels.json
  - channel_x_vocab_*_topK heatmap + vocab_profiles line plot (like vocab_select)

Example:
  python semantic_gpt_vocab_selectivity.py \\
    --npz path/to/latents_on_grid.npz \\
    --out_dir path/to/out \\
    --vocab_npy path/to/vocabulary.npy \\
    --metric heatmap_peak_sparse --step_mode all --thr_sem 0.5 --top_channels 8 \\
    --heatmap_zscore row
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
from scipy.stats import rankdata

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from semantic_gpt_vocab_select import (
    compute_top_channels_per_vocab,
    load_npz,
    reshape_acts_semantics,
    load_vocabulary_names,
    plot_vocab_profiles_psth,
    zscore_rows_across_vocab,
)


# Defaults aligned with semantic_gpt_vocab_select.py
DEFAULT_NPZ = Path(
    r"D:\workspace\PRED_CODING\A_gs_project\analysis_out\semantic_gpt_predefined_latents\latents_on_grid.npz"
)
DEFAULT_VOCABULARY_NPY = Path(
    r"D:\workspace\PRED_CODING\A_gs_project\data\data_11272025_twinmansion\vocabulary.npy"
)
OUT_DIR = Path(
    r"D:\workspace\PRED_CODING\A_gs_project\analysis_out\semantic_gpt_predefined_latents\demo_vocab_selectivity"
)


def compute_auc_matrix_mann_whitney(
    acts: np.ndarray,
    sem: np.ndarray,
    *,
    thr_sem: float,
) -> np.ndarray:
    """
    Pairwise AUC for each (channel j, vocab c): ROC-AUC of acts[t,j] vs binary sem[t,c] > thr_sem.

    Uses the Mann–Whitney U / rank-sum formula (same as sklearn when ties use average ranks).
    Shape: (D, V), NaN when vocab never on or never off.
    """
    T, D = acts.shape
    if sem.shape[0] != T:
        raise ValueError("acts and sem row count mismatch")
    V = sem.shape[1]
    y_all = sem > thr_sem
    auc = np.full((D, V), np.nan, dtype=np.float64)
    ranks = rankdata(acts, axis=0, method="average")

    for c in range(V):
        y = y_all[:, c]
        n_pos = int(y.sum())
        n_neg = int(T - n_pos)
        if n_pos == 0 or n_neg == 0:
            continue
        rank_sum_pos = ranks[y, :].sum(axis=0)
        auc[:, c] = (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)

    return auc.astype(np.float32)


def channel_selectivity_scores(auc_jv: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Per channel j: score_j = max_c max(AUC, 1-AUC) so inverted tuning counts.
    Returns (scores_d, argmax_vocab_index optional — we compute argmax in caller if needed).
    """
    valid = np.isfinite(auc_jv)
    # symmetrize separation from chance
    sep = np.maximum(auc_jv, 1.0 - auc_jv)
    sep = np.where(valid, sep, np.nan)
    scores = np.nanmax(sep, axis=1)
    return scores.astype(np.float32), sep


def rank_top_channels(scores: np.ndarray, k: int) -> np.ndarray:
    k = min(k, scores.shape[0])
    order = np.argsort(-np.nan_to_num(scores, nan=-1.0))
    return order[:k].astype(np.int64)


def mean_on_heatmap_from_vocab_select(
    acts: np.ndarray,
    sem: np.ndarray,
    *,
    thr_sem: float,
) -> np.ndarray:
    """
    Same matrix as vocab_select heatmap: shape (D, V), heat[d,v] = mean(acts[t,d] | sem[t,v] > thr).
    Entries are -inf where vocab v never appears (mask_on empty).
    """
    _, _, _, score_matrix_vd = compute_top_channels_per_vocab(
        acts,
        sem,
        topk=1,
        thr_sem=thr_sem,
    )
    return score_matrix_vd.T.astype(np.float64)


def sparse_peak_minus_median_rest(heat_dv: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Per channel: score = max_v x_v - median({ x_v : v != v* }) over finite entries only.
    Returns (scores shape (D,), peak_vocab_idx shape (D,) with -1 if undefined).
    """
    D, V = heat_dv.shape
    scores = np.full(D, np.nan, dtype=np.float64)
    peak_vocab = np.full(D, -1, dtype=np.int64)

    for j in range(D):
        row = heat_dv[j].astype(np.float64)
        ok = np.isfinite(row)
        if ok.sum() < 2:
            continue
        vals = row[ok]
        idx_global = np.where(ok)[0]
        li = int(np.argmax(vals))
        v_peak = int(idx_global[li])
        peak_vocab[j] = v_peak
        others = np.delete(vals, li)
        scores[j] = float(vals[li] - np.median(others))

    return scores.astype(np.float32), peak_vocab


def main() -> None:
    ap = argparse.ArgumentParser(description="Pick sparse vocab-tuned channels (heatmap-based by default)")
    ap.add_argument("--npz", type=str, default=str(DEFAULT_NPZ))
    ap.add_argument("--out_dir", type=str, default=str(OUT_DIR))
    ap.add_argument("--vocab_npy", type=str, default=str(DEFAULT_VOCABULARY_NPY))
    ap.add_argument("--step_mode", type=str, default="all", choices=["all", "last"])
    ap.add_argument("--thr_sem", type=float, default=0.5)
    ap.add_argument("--top_channels", type=int, default=8, help="Plot/save this many top channels")
    ap.add_argument(
        "--metric",
        type=str,
        default="heatmap_peak_sparse",
        choices=["heatmap_peak_sparse", "roc_auc"],
        help="heatmap_peak_sparse: peak mean − median(rest); roc_auc: max symmetrized AUC",
    )
    ap.add_argument(
        "--heatmap_zscore",
        type=str,
        default="row",
        choices=["row", "none"],
        help="row: z-score each channel row across vocab for visualization",
    )
    args = ap.parse_args()

    npz_path = Path(args.npz)
    out_dir = Path(args.out_dir)
    vocab_npy = Path(args.vocab_npy) if str(args.vocab_npy).strip() else None
    if not npz_path.exists():
        raise FileNotFoundError(f"npz not found: {npz_path}")

    out_dir.mkdir(parents=True, exist_ok=True)

    d = load_npz(npz_path)
    z = d["z"]
    semantics = d["semantics"]
    acts, sem = reshape_acts_semantics(z, semantics, stepmode=args.step_mode)
    T, D = acts.shape
    V = sem.shape[1]

    vocab_names = load_vocabulary_names(vocab_npy)
    if vocab_names is not None and len(vocab_names) != V:
        print(f"[warn] vocabulary length={len(vocab_names)} but semantics V={V}. Names disabled.")
        vocab_names = None

    print(
        f"[semantic_gpt_vocab_selectivity] metric={args.metric} step_mode={args.step_mode} "
        f"T={T} D={D} V={V} thr_sem={args.thr_sem}"
    )
    print(
        "  Note: step_mode 'all' uses every frame (N*L); 'last' uses only the last frame per sequence."
    )

    metric_tag = str(args.metric)
    if args.metric == "heatmap_peak_sparse":
        heat_dv = mean_on_heatmap_from_vocab_select(acts, sem, thr_sem=args.thr_sem)
        scores, peak_vocab_idx = sparse_peak_minus_median_rest(heat_dv)
        top_idx = rank_top_channels(scores, int(args.top_channels))
        plot_values = heat_dv[top_idx, :]
        metric_name = "heatmap_peak_minus_median_rest"
    else:
        auc_jv = compute_auc_matrix_mann_whitney(acts, sem, thr_sem=args.thr_sem)
        scores, sep_jv = channel_selectivity_scores(auc_jv)
        top_idx = rank_top_channels(scores, int(args.top_channels))
        peak_vocab_idx = np.nanargmax(sep_jv, axis=1)
        plot_values = auc_jv[top_idx, :]
        metric_name = "roc_auc_on_vs_off"

    best_c = peak_vocab_idx[top_idx]

    payload: Dict[str, Any] = {
        "npz": str(npz_path),
        "step_mode": args.step_mode,
        "thr_sem": args.thr_sem,
        "T": int(T),
        "D": int(D),
        "V": int(V),
        "metric": metric_name,
        "top_channel_indices": [int(x) for x in top_idx],
        "top_channel_scores": [float(scores[j]) for j in top_idx],
        "peak_vocab_per_top_channel": [int(best_c[i]) for i in range(len(top_idx))],
    }
    json_path = out_dir / "selectivity_top_channels.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[save] {json_path}")

    for i, j in enumerate(top_idx):
        c = int(best_c[i])
        name = str(vocab_names[c]) if vocab_names is not None and c >= 0 else str(c)
        print(f"  #{i+1} channel={j}  score={scores[j]:.6g}  peak_vocab[{c}]={name}")

    # Heatmap + line plot (mean-on matrix by default; optional row z-score)
    heat = np.array(plot_values, copy=True)
    heat = np.nan_to_num(heat, nan=0.0, posinf=0.0, neginf=0.0)

    if args.heatmap_zscore == "row":
        heat_plot = zscore_rows_across_vocab(heat.astype(np.float64)).astype(np.float32)
        cbar_label = "z-score (per channel across vocab)"
        z_tag = "zscore_row"
    else:
        heat_plot = heat.astype(np.float32)
        if args.metric == "heatmap_peak_sparse":
            cbar_label = "mean activation (on frames)"
        else:
            cbar_label = "ROC-AUC"
        z_tag = "raw"

    stem = "mean_on_top" if args.metric == "heatmap_peak_sparse" else "auc_top"
    heatmap_path = out_dir / f"channel_x_vocab_{stem}{len(top_idx)}.png"
    plt.figure(figsize=(max(6, V * 0.25), max(4, len(top_idx) * 0.45)))
    plt.imshow(heat_plot, aspect="auto", cmap="viridis" if args.metric == "heatmap_peak_sparse" else "magma", vmin=None, vmax=None)
    plt.colorbar(label=cbar_label)
    plt.xlabel("vocab id")
    plt.ylabel("channel id")
    plt.title(f"channel × vocab ({metric_tag}, {z_tag}, step_mode={args.step_mode})")
    plt.yticks(range(len(top_idx)), [str(int(j)) for j in top_idx], fontsize=8)
    plt.tight_layout()
    plt.savefig(heatmap_path, dpi=200)
    plt.close()
    print(f"[heatmap] saved to {heatmap_path}")

    line_out = heatmap_path.with_name(heatmap_path.stem + "_vocab_profiles.png")
    plot_vocab_profiles_psth(
        heat_plot,
        channel_ids=[int(j) for j in top_idx],
        vocab_names=vocab_names,
        step_mode=args.step_mode,
        heatmap_zscore=args.heatmap_zscore,
        out_png=line_out,
    )
    print(f"[vocab_lineplot] saved to {line_out}")
    print("[done]")


if __name__ == "__main__":
    main()
