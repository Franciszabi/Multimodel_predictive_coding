"""
semantic_gpt_error_map.py

Decode cached latents -> (x,y) with a small MLP, then plot error map (hex or square bins).

Key design choices
  - Target: positions[:, -1, :2] (last frame).
  - decoder_arch conv1d (default): z[:, :, :, 0,0] -> (N, L, D) -> (B, D, L) Conv1d over time,
    then last-timestep vector + small MLP -> (x, y).
  - decoder_arch mlp: z[:, -1, :, 0, 0] -> (N, D) MLP only (no temporal conv).
  - Same npz layout for SemanticGPT or AE exports from train_semantic_ae.py:
      z (N,L,C,1,1), positions (N,L,3)

Usage:
  python semantic_gpt_error_map.py --npz .../latents_on_grid_ae.npz --out_dir .../error_map_ae \\
    --map_mode square --bins 41,66
  python semantic_gpt_error_map.py --map_mode hex --hex_gridsize 27

Caches (default on, disable with --no_error_cache):
  error_map_last_frame_samples.npz — pos_last_xy, errors, source_npz
  error_map_last_frame_square_grid.npz — mean_error (bins_x x bins_y), edges, counts, ...
  error_map_last_frame_hex_grid.npz — hex_offsets, hex_mean_error, hex_gridsize
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    # Inputs
    npz_path: Path = Path("/home/ubuntu/project/analysis_out/semantic_gpt_predefined_latents/latents_on_grid.npz")

    # Outputs
    out_dir: Path = Path("/home/ubuntu/project/analysis_out/semantic_gpt_predefined_latents/error_map")

    # Position decoder
    decoder_arch: str = "conv1d"  # "conv1d" | "mlp"
    # mlp: Linear+ReLU chain then Linear -> 2 (only when decoder_arch == "mlp")
    mlp_hidden_dims: tuple[int, ...] = (512, 256, 128, 64)
    # conv1d: channels over frame index L; D = in_channels
    conv_channels: tuple[int, ...] = (256, 256, 128)
    conv_kernel_size: int = 3
    conv_head_hidden: int = 128  # 0 = single Linear(C, 2) after conv

    # Training params
    device: str = "cuda"  # "cuda" or "cpu"
    lr: float = 1e-4
    weight_decay: float = 0.0
    batch_size: int = 512
    epochs: int = 2000  # tune
    num_steps: int | None = None  # optional early stop by gradient steps

    # Position scaling (mimic src.analysis.PositionDecoder)
    pos_scale: float = 30.0

    # Plotting
    map_mode: str = "square"  # "square" | "hex"
    bins_xy: tuple[int, int] = (41, 66)  # square: histogram2d bins (x, y)
    hex_gridsize: int = 27  # hex: plt.hexbin gridsize

    # Save binned mean-error grids (+ per-sample errors) as .npz for numerical comparison
    cache_error_data: bool = True


class SemanticGPTPositionDecoderMLP(nn.Module):
    """
    MLP regression head for decoding (B,D) -> (B,2).

    Structure: D -> [Linear(h0)->ReLU -> Linear(h1)->ReLU -> ...] -> Linear(2).
    Example mlp_hidden_dims=(512,256,128,64): 4 hidden layers, 5 Linear maps total.
    """

    def __init__(self, d_in: int, hidden_dims: Sequence[int], out_dim: int = 2):
        super().__init__()
        if len(hidden_dims) < 1:
            raise ValueError("hidden_dims must contain at least one layer width")
        layers: list[nn.Module] = []
        prev = d_in
        for h in hidden_dims:
            layers.append(nn.Linear(prev, int(h)))
            layers.append(nn.ReLU())
            prev = int(h)
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SemanticGPTPositionDecoderConv1d(nn.Module):
    """
    Conv1d over sequence length L with D channels: input (B, D, L).
    Same padding keeps length L; use features at last frame (aligned with target time).
    Then Linear(+ReLU)+Linear to (x, y) in scaled space.
    """

    def __init__(
        self,
        d_in: int,
        channels: Sequence[int],
        kernel_size: int,
        head_hidden: int,
        out_dim: int = 2,
    ):
        super().__init__()
        if len(channels) < 1:
            raise ValueError("conv_channels must contain at least one width")
        ks = int(kernel_size)
        pad = ks // 2
        blocks: list[nn.Module] = []
        c_prev = int(d_in)
        for c in channels:
            c = int(c)
            blocks.append(nn.Conv1d(c_prev, c, ks, padding=pad))
            blocks.append(nn.ReLU())
            c_prev = c
        self.backbone = nn.Sequential(*blocks)
        hh = int(head_hidden)
        if hh > 0:
            self.head: nn.Module = nn.Sequential(
                nn.Linear(c_prev, hh),
                nn.ReLU(),
                nn.Linear(hh, out_dim),
            )
        else:
            self.head = nn.Linear(c_prev, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, D, L)
        h = self.backbone(x)
        h_last = h[:, :, -1]
        return self.head(h_last)


def build_position_decoder(z: np.ndarray, cfg: Config) -> nn.Module:
    """z: (N, L, D, 1, 1)."""
    _, _, d_model, h, w = z.shape
    if h != 1 or w != 1:
        raise ValueError(f"Expected z spatial dims 1x1, got {h}x{w}")
    arch = cfg.decoder_arch.strip().lower()
    if arch == "mlp":
        return SemanticGPTPositionDecoderMLP(
            d_in=int(d_model), hidden_dims=cfg.mlp_hidden_dims, out_dim=2
        )
    if arch == "conv1d":
        return SemanticGPTPositionDecoderConv1d(
            d_in=int(d_model),
            channels=cfg.conv_channels,
            kernel_size=cfg.conv_kernel_size,
            head_hidden=cfg.conv_head_hidden,
            out_dim=2,
        )
    raise ValueError(f"decoder_arch must be 'mlp' or 'conv1d', got {cfg.decoder_arch!r}")


def _decoder_inputs_tensor(z: np.ndarray, cfg: Config) -> torch.Tensor:
    """Batch tensor on CPU float32: mlp -> (N, D), conv1d -> (N, D, L)."""
    arch = cfg.decoder_arch.strip().lower()
    if arch == "mlp":
        return torch.from_numpy(z[:, -1, :, 0, 0]).float()
    if arch == "conv1d":
        seq = torch.from_numpy(z[:, :, :, 0, 0]).float()
        return seq.permute(0, 2, 1)
    raise ValueError(f"decoder_arch must be 'mlp' or 'conv1d', got {cfg.decoder_arch!r}")


def load_cached_latents(npz_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      z: (N,L,D,1,1)
      positions: (N,L,3)
    """
    d = np.load(npz_path, allow_pickle=True)
    z = d["z"]
    positions = d["positions"]
    return z, positions


def select_last_frame(z: np.ndarray, positions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Per your decision:
      latent selection: z[:, -1, ...]
      target selection: positions[:, -1, :2]
    """
    # z: (N,L,D,1,1) -> (N,D)
    z_last = z[:, -1, :, 0, 0]
    # positions: (N,L,3) -> (N,2)
    pos_last_xy = positions[:, -1, :2]
    return z_last, pos_last_xy


def train_decoder(decoder: nn.Module, z: np.ndarray, pos_last_xy: np.ndarray, cfg: Config):
    """
    PSEUDOCODE training loop (mimic `src.analysis.PositionDecoder.train()`):

    1) scale positions:
         y = pos_last_xy / cfg.pos_scale
    2) optimizer:
         optim = AdamW(decoder.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    3) for epoch in range(cfg.epochs):
         - shuffle indices
         - for each mini-batch:
             inputs = z[idx]  # (B,D) mlp or (B,D,L) conv1d
             targets = y[idx]                 # (B,2)
             pred = decoder(inputs)           # (B,2)
             loss = MSE(pred, targets)
             loss.backward()
             optim.step(); optim.zero_grad()
    4) optional: print loss every N steps
    """
    decoder.train()

    device = next(decoder.parameters()).device
    z_t = _decoder_inputs_tensor(z, cfg)
    pos_t = torch.from_numpy(pos_last_xy).float()  # (N,2)
    targets = pos_t / float(cfg.pos_scale)  # scale to stabilize regression

    n = z_t.shape[0]
    if n != targets.shape[0]:
        raise ValueError(f"batch N={n} mismatch targets N={targets.shape[0]}")

    optimizer = torch.optim.AdamW(
        decoder.parameters(),
        lr=float(cfg.lr),
        weight_decay=float(cfg.weight_decay),
    )
    # Optional schedule: keep same spirit as PositionDecoder (StepLR with long step)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4000, gamma=0.1)

    step = 0
    for epoch in range(int(cfg.epochs)):
        perm = torch.randperm(n)
        for start in range(0, n, int(cfg.batch_size)):
            batch_idx = perm[start : start + int(cfg.batch_size)]
            inputs = z_t[batch_idx].to(device, non_blocking=True)
            batch_targets = targets[batch_idx].to(device, non_blocking=True)

            pred = decoder(inputs)  # (B,2) in scaled coord space
            loss = F.mse_loss(pred, batch_targets)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()

            step += 1
            if step % 200 == 0:
                print(f"[train] epoch={epoch+1}/{cfg.epochs} step={step} loss={loss.item():.6f}")

            if cfg.num_steps is not None and step >= int(cfg.num_steps):
                print(f"[train] reached num_steps={cfg.num_steps}, stopping early.")
                return


@torch.no_grad()
def compute_errors(decoder: nn.Module, z: np.ndarray, pos_last_xy: np.ndarray, cfg: Config) -> np.ndarray:
    """
    PSEUDOCODE:
      - decoder.eval()
      - pred_scaled = decoder(z_tensor)          # (N,2)
      - pred = pred_scaled.cpu().numpy() * cfg.pos_scale
      - error = L2(pred - pos_last_xy) along axis=1
    """
    decoder.eval()
    device = next(decoder.parameters()).device

    z_t = _decoder_inputs_tensor(z, cfg)
    pos_np = np.asarray(pos_last_xy, dtype=np.float32)

    n = z_t.shape[0]
    errors = np.zeros((n,), dtype=np.float32)

    batch_size = int(cfg.batch_size)
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            inputs = z_t[start:end].to(device, non_blocking=True)
            pred_scaled = decoder(inputs).detach().cpu().numpy()  # (B,2)
            pred_xy = pred_scaled * float(cfg.pos_scale)  # back to original units

            err = np.linalg.norm(pred_xy - pos_np[start:end], axis=1)
            errors[start:end] = err.astype(np.float32)

    return errors


def plot_error_hexbin(
    pos_last_xy: np.ndarray,
    errors: np.ndarray,
    out_png: Path,
    *,
    gridsize: int = 27,
    cache_npz: Path | None = None,
) -> dict[str, Any] | None:
    """
    Mimic `src.analysis.error_map()`:
      hexbin(-y, x, C=error, reduce_C_function=np.mean)

    If cache_npz is set, writes hex cell centers and mean error per cell (matplotlib hexbin).
    """
    out_png.parent.mkdir(parents=True, exist_ok=True)

    plt.figure()
    hb = plt.hexbin(
        -pos_last_xy[:, 1],
        pos_last_xy[:, 0],
        C=errors,
        gridsize=int(gridsize),
        cmap="inferno",
        vmin=0,
        vmax=float(errors.max()) if errors.size else 1.0,
        reduce_C_function=np.mean,
    )
    plt.colorbar(label=r"Error ($\Vert x - \hat{x} \Vert$) (lattice units)")
    plt.xlabel("x-axis (lattice units)")
    plt.ylabel("y-axis (lattice units)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

    meta: dict[str, Any] | None = None
    if cache_npz is not None:
        cache_npz.parent.mkdir(parents=True, exist_ok=True)
        arr = hb.get_array()
        off = hb.get_offsets()
        np.savez(
            cache_npz,
            hex_gridsize=np.int32(gridsize),
            hex_offsets=np.asarray(off, dtype=np.float64),
            hex_mean_error=np.asarray(arr, dtype=np.float64),
            x_plot_label=np.array("plotted x = -position_y"),
            y_plot_label=np.array("plotted y = position_x"),
        )
        meta = {"cache_npz": str(cache_npz)}
    return meta


def plot_error_square_grid(
    pos_last_xy: np.ndarray,
    errors: np.ndarray,
    out_png: Path,
    bins_xy: tuple[int, int],
    cache_npz: Path | None = None,
):
    """
    Square-bin mean error map (aligns with fixed grid like placefield scripts).
    Uses same orientation as hex: x_plot=-y, y_plot=x.

    mean_error in cache: shape (bins_x, bins_y) = np.histogram2d first dim = x_plot, second = y_plot.
    imshow uses mean_error.T (shape bins_y x bins_x).
    """
    out_png.parent.mkdir(parents=True, exist_ok=True)
    x_plot = -pos_last_xy[:, 1]
    y_plot = pos_last_xy[:, 0]
    eps = 1e-9

    xmin, xmax = float(x_plot.min() - 2), float(x_plot.max() + 2)
    ymin, ymax = float(y_plot.min() - 2), float(y_plot.max() + 2)
    range_xy = [[xmin, xmax], [ymin, ymax]]

    err_sum, x_edges, y_edges = np.histogram2d(
        x_plot, y_plot, bins=bins_xy, range=range_xy, weights=errors
    )
    counts, _, _ = np.histogram2d(x_plot, y_plot, bins=bins_xy, range=range_xy)
    mean_err = err_sum / (counts + eps)
    mean_err[counts == 0] = np.nan

    if cache_npz is not None:
        cache_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            cache_npz,
            mean_error=np.asarray(mean_err, dtype=np.float64),
            err_sum=np.asarray(err_sum, dtype=np.float64),
            counts=np.asarray(counts, dtype=np.int64),
            x_edges=np.asarray(x_edges, dtype=np.float64),
            y_edges=np.asarray(y_edges, dtype=np.float64),
            range_xmin=np.float64(xmin),
            range_xmax=np.float64(xmax),
            range_ymin=np.float64(ymin),
            range_ymax=np.float64(ymax),
            bins_x=np.int32(bins_xy[0]),
            bins_y=np.int32(bins_xy[1]),
            x_plot_label=np.array("plotted x = -position_y"),
            y_plot_label=np.array("plotted y = position_x"),
        )

    plt.figure()
    img = mean_err.T
    plt.imshow(
        img,
        origin="lower",
        extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
        aspect="auto",
        cmap="inferno",
        vmin=0,
        vmax=float(np.nanmax(img)) if np.isfinite(np.nanmax(img)) else 1.0,
    )
    plt.colorbar(label=r"Mean error per bin ($\Vert x - \hat{x} \Vert$)")
    plt.xlabel("plotted x (-y)")
    plt.ylabel("plotted y (x)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def main(cfg: Config):
    # 1) load cached arrays
    z, positions = load_cached_latents(cfg.npz_path)

       # 2) last-frame (x,y) targets; decoder may use full z[:,:,:,0,0] (conv1d) or last slice (mlp)
    _, pos_last_xy = select_last_frame(z, positions)

    # 3) build decoder
    decoder = build_position_decoder(z, cfg)

    device = torch.device(cfg.device if torch.cuda.is_available() or cfg.device == "cpu" else "cpu")
    decoder.to(device)
    print(f"[decoder] arch={cfg.decoder_arch!r} n_params={sum(p.numel() for p in decoder.parameters())}")

    # 4) train decoder (pseudo)
    train_decoder(decoder, z, pos_last_xy, cfg)

    # 5) compute errors (pseudo)
    errors = compute_errors(decoder, z, pos_last_xy, cfg)

    # 6) plot error map
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    if cfg.cache_error_data:
        samples_npz = cfg.out_dir / "error_map_last_frame_samples.npz"
        np.savez(
            samples_npz,
            pos_last_xy=np.asarray(pos_last_xy, dtype=np.float32),
            errors=np.asarray(errors, dtype=np.float32),
            source_npz=np.array(str(cfg.npz_path)),
        )
        print(f"[cache] per-sample: {samples_npz}")

    if cfg.map_mode == "square":
        out_png = cfg.out_dir / "error_map_last_frame_square.png"
        grid_npz = cfg.out_dir / "error_map_last_frame_square_grid.npz" if cfg.cache_error_data else None
        plot_error_square_grid(
            pos_last_xy, errors, out_png, bins_xy=cfg.bins_xy, cache_npz=grid_npz
        )
        if cfg.cache_error_data:
            print(f"[cache] square grid: {grid_npz}")
    elif cfg.map_mode == "hex":
        out_png = cfg.out_dir / "error_map_last_frame_hex.png"
        grid_npz = cfg.out_dir / "error_map_last_frame_hex_grid.npz" if cfg.cache_error_data else None
        plot_error_hexbin(
            pos_last_xy, errors, out_png, gridsize=cfg.hex_gridsize, cache_npz=grid_npz
        )
        if cfg.cache_error_data:
            print(f"[cache] hex grid: {grid_npz}")
    else:
        raise ValueError(f"map_mode must be 'square' or 'hex', got {cfg.map_mode!r}")
    print(f"[done] saved: {out_png}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Train position decoder (conv1d over time or MLP) on cached latents + plot error map."
    )
    ap.add_argument("--npz", type=str, default="", help="Path to latents .npz (z, positions)")
    ap.add_argument("--out_dir", type=str, default="", help="Output directory for error_map_last_frame.png")
    ap.add_argument("--device", type=str, default="", help="cuda or cpu (empty = use Config)")
    ap.add_argument(
        "--map_mode",
        type=str,
        default="",
        choices=["", "square", "hex"],
        help="square: fixed rectangular bins; hex: hexbin. Empty = use Config",
    )
    ap.add_argument(
        "--bins",
        type=str,
        default="",
        help="For square mode: 'nx,ny' e.g. 41,66. Empty = use Config",
    )
    ap.add_argument(
        "--hex_gridsize",
        type=int,
        default=0,
        help="For hex mode: hexbin gridsize.0 = use Config",
    )
    ap.add_argument(
        "--no_error_cache",
        action="store_true",
        help="Do not write error_map_last_frame_* .npz caches (png only)",
    )
    args = ap.parse_args()

    cfg = Config()
    if args.npz.strip():
        cfg.npz_path = Path(args.npz)
    if args.out_dir.strip():
        cfg.out_dir = Path(args.out_dir)
    if args.device.strip():
        cfg.device = args.device.strip()
    if args.map_mode.strip():
        cfg.map_mode = args.map_mode.strip()
    if args.bins.strip():
        parts = [int(x.strip()) for x in args.bins.split(",") if x.strip()]
        if len(parts) != 2:
            raise ValueError("--bins must be like '41,66'")
        cfg.bins_xy = (parts[0], parts[1])
    if args.hex_gridsize > 0:
        cfg.hex_gridsize = int(args.hex_gridsize)
    if args.no_error_cache:
        cfg.cache_error_data = False

    main(cfg)

