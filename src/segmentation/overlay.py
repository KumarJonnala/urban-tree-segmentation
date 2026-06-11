from __future__ import annotations
"""Rasterization and overlay composition: multi-class .npy + PNG outputs."""

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from pyproj import Transformer
from rasterio.features import rasterize
from rasterio.transform import from_bounds

_to_utm = Transformer.from_crs("EPSG:4326", "EPSG:25832", always_xy=True)

# Class indices: 0=other, 1=tree, 2=road, 3=building
CLASS_COLORS = {
    0: [0.88, 0.88, 0.80],
    1: [0.15, 0.60, 0.15],
    2: [0.55, 0.55, 0.55],
    3: [0.85, 0.28, 0.28],
}
CLASS_LABELS = {0: "Other", 1: "Tree", 2: "Road", 3: "Building"}

# RGBA alpha values for the semi-transparent overlay compositing
_OVERLAY_ALPHA = {0: 0.0, 1: 0.55, 2: 0.65, 3: 0.70}


def make_transform(bbox: dict, width_px: int, height_px: int):
    """Return a rasterio Affine transform for bbox (WGS84) at the given pixel dimensions."""
    min_x, min_y = _to_utm.transform(bbox["west"], bbox["south"])
    max_x, max_y = _to_utm.transform(bbox["east"], bbox["north"])
    return from_bounds(min_x, min_y, max_x, max_y, width_px, height_px)


def rasterize_layer(gdf: gpd.GeoDataFrame, out_shape: tuple, transform) -> np.ndarray:
    """Burn GeoDataFrame geometries onto a pixel grid. Returns a bool array.

    Returns zeros if gdf is None or empty.
    """
    if gdf is None or len(gdf) == 0:
        return np.zeros(out_shape, dtype=bool)
    shapes = [(geom, 1) for geom in gdf.geometry if geom is not None]
    if not shapes:
        return np.zeros(out_shape, dtype=bool)
    return rasterize(shapes, out_shape=out_shape, transform=transform, fill=0, dtype="uint8").astype(bool)


def save_segmentation(
    img: np.ndarray,
    bbox: dict,
    building_gdf: gpd.GeoDataFrame,
    road_gdf: gpd.GeoDataFrame,
    tree_mask: np.ndarray,
    out_dir: Path,
    stem: str,
) -> tuple[Path, Path]:
    """Rasterize all layers, compose a multi-class array, and save outputs.

    Priority stacking: tree (1) > road (2) > building (3) > other (0).

    Parameters
    ----------
    img         : RGB uint8 (H, W, 3) orthophoto for the overlay PNG
    bbox        : WGS84 bounding box dict (west/south/east/north)
    building_gdf, road_gdf : GeoDataFrames in EPSG:25832
    tree_mask   : bool (H, W) from vegetation.vari_mask()
    out_dir     : directory to write outputs (created if missing)
    stem        : filename prefix, e.g. "dop20_ovgu" or "ovgu_bbox_tile_0_0"

    Returns
    -------
    (npy_path, png_path)
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    height_px, width_px = img.shape[:2]
    transform = make_transform(bbox, width_px, height_px)
    out_shape = (height_px, width_px)

    building_mask = rasterize_layer(building_gdf, out_shape, transform)
    road_mask = rasterize_layer(road_gdf, out_shape, transform)

    seg = np.zeros(out_shape, dtype=np.uint8)
    seg[building_mask] = 3
    seg[road_mask] = 2
    seg[tree_mask] = 1

    npy_path = out_dir / f"{stem}_seg.npy"
    np.save(npy_path, seg)

    # Vectorised RGBA lookup (avoids per-pixel Python loop)
    n_classes = max(CLASS_COLORS) + 1
    rgb_lut = np.array([CLASS_COLORS[k] for k in range(n_classes)], dtype=float)
    alpha_lut = np.array([_OVERLAY_ALPHA[k] for k in range(n_classes)], dtype=float)
    overlay_rgb = rgb_lut[seg]
    overlay_alpha = alpha_lut[seg, np.newaxis]
    overlay_rgba = np.concatenate([overlay_rgb, overlay_alpha], axis=-1)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img)
    ax.imshow(overlay_rgba)
    ax.axis("off")
    fig.tight_layout(pad=0)

    png_path = out_dir / f"{stem}_seg_overlay.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return npy_path, png_path
