"""World-coordinate plot overlays from exported Unity map metadata."""

from pathlib import Path
import json

import numpy as np


def load_map_overlay(map_root=None, *, data_root=None):
    root = Path(map_root or data_root or "").expanduser()
    if not map_root and not (root / "map.json").is_file() and not (root / "navigation.json").is_file():
        return None
    if not root.is_dir():
        raise FileNotFoundError(f"Map directory not found: {root}")
    geometry = {}
    if (root / "map.json").is_file():
        geometry = json.loads((root / "map.json").read_text(encoding="utf-8-sig"))
    elif (root / "meta.json").is_file():
        geometry = json.loads((root / "meta.json").read_text(encoding="utf-8-sig")).get("map", {})
    navigation = {}
    if (root / "navigation.json").is_file():
        navigation = json.loads((root / "navigation.json").read_text(encoding="utf-8-sig"))
    extent, occupancy = None, None
    if geometry:
        x, z, dx, dz = [float(geometry[k]) for k in ("minX", "minZ", "cellW", "cellH")]
        width, height = int(geometry["W"]), int(geometry["H"])
        if not np.isfinite([x, z, dx, dz]).all() or min(dx, dz, width, height) <= 0:
            raise ValueError("Invalid map dimensions")
        extent = (x, x + dx * width, z, z + dz * height)
        if (root / "occupancy.npy").is_file():
            occupancy = np.load(root / "occupancy.npy", allow_pickle=False)
            if occupancy.shape != (height, width) or not np.isfinite(occupancy).all():
                raise ValueError("occupancy.npy must have shape [H,W] with rows along world z")
            occupancy = occupancy != 0
    rooms = navigation.get("rooms", [])
    spacing = float(navigation.get("roomSpacingMeters", 0))
    if not np.isfinite(spacing) or spacing < 0:
        raise ValueError("Invalid room spacing")
    if extent is None and rooms and spacing > 0:
        xs = [float(room["center"]["x"]) for room in rooms]
        zs = [float(room["center"]["z"]) for room in rooms]
        extent = (min(xs) - spacing / 2, max(xs) + spacing / 2,
                  min(zs) - spacing / 2, max(zs) + spacing / 2)
    if extent is None:
        raise ValueError(f"No usable map bounds or room layout in {root}")
    return dict(root=str(root.resolve()), extent=extent, occupancy=occupancy,
                rooms=rooms, spacing=spacing, doors=navigation.get("doors", []),
                occupancy_inflation=navigation.get("occupancyInflationMeters"))


def overlay_report(overlay):
    if overlay is None:
        return None
    return dict(root=overlay["root"], extent=list(overlay["extent"]), rooms=len(overlay["rooms"]),
                occupancy_loaded=overlay["occupancy"] is not None,
                occupancy_inflation_meters=overlay["occupancy_inflation"],
                note="Dashed room cells use exported centers and spacing, not exact wall polygons. "
                     "Gray obstacles use the exported (potentially inflated) navigation occupancy; circles mark doors.")


def validate_map_positions(overlay, positions):
    if overlay is None:
        return
    bounds = np.asarray(overlay["extent"]).reshape(2, 2)
    if np.any(positions < bounds[:, 0]) or np.any(positions > bounds[:, 1]):
        raise ValueError("Map bounds do not contain every sample; use metadata from the matching environment")


def draw_map_overlay(ax, overlay):
    if overlay is None:
        return
    import matplotlib.patheffects as effects
    from matplotlib.collections import LineCollection
    from matplotlib.colors import ListedColormap

    occupancy, extent = overlay["occupancy"], overlay["extent"]
    if occupancy is not None:
        ax.imshow(np.ma.masked_where(~occupancy, occupancy), origin="lower", extent=extent,
                  cmap=ListedColormap(["#454545"]), interpolation="nearest", alpha=0.30, zorder=3)
    segments = set()
    half = overlay["spacing"] / 2
    for room in overlay["rooms"]:
        x, z = float(room["center"]["x"]), float(room["center"]["z"])
        if half:
            corners = [(x - half, z - half), (x + half, z - half),
                       (x + half, z + half), (x - half, z + half)]
            for a, b in zip(corners, corners[1:] + corners[:1]):
                segments.add(tuple(sorted((a, b))))
        ax.text(x, z, str(room["roomId"]), ha="center", va="center", fontsize=6,
                color="#202020", zorder=6,
                path_effects=[effects.withStroke(linewidth=1.8, foreground="white")])
    if segments:
        lines = LineCollection(sorted(segments), colors="#202020", linewidths=0.65,
                               linestyles="dashed", zorder=5)
        lines.set_path_effects([effects.Stroke(linewidth=1.5, foreground="white"), effects.Normal()])
        ax.add_collection(lines)
    if overlay["doors"]:
        ax.scatter([d["position"]["x"] for d in overlay["doors"]],
                   [d["position"]["z"] for d in overlay["doors"]],
                   s=8, facecolors="white", edgecolors="#202020", linewidths=0.45, zorder=7)
    ax.set(xlim=extent[:2], ylim=extent[2:], aspect="equal")
