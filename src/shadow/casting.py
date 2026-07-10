from __future__ import annotations
"""Tree shadow casting: height estimation and geometric shadow projection."""

import math
import datetime as dt

import geopandas as gpd
import numpy as np
from pyproj import Transformer
from rasterio.features import shapes as rio_shapes, rasterize as rio_rasterize
from rasterio.transform import from_bounds
from scipy.ndimage import label as cc_label
from shapely.geometry import shape as sg_shape

from src.config import MAX_CROWN_RADIUS_M, ALLOMETRIC_A, ALLOMETRIC_B, ALLOMETRIC_PROFILES
from src.shadow.solar import sun_position, _tile_center

_to_utm = Transformer.from_crs("EPSG:4326", "EPSG:25832", always_xy=True)

_8CONN = np.ones((3, 3), dtype=int)


def _pixel_size_m(bbox: dict, shape: tuple[int, int]) -> float:
    """Return average metres-per-pixel for a tile."""
    H, W = shape
    west_m, south_m = _to_utm.transform(bbox["west"], bbox["south"])
    east_m, north_m = _to_utm.transform(bbox["east"], bbox["north"])
    return ((east_m - west_m) / W + (north_m - south_m) / H) / 2.0


def _bbox_to_transform(bbox: dict, shape: tuple[int, int]):
    """Return rasterio Affine mapping pixel space → EPSG:25832 for the tile."""
    H, W = shape
    west_m, south_m = _to_utm.transform(bbox["west"], bbox["south"])
    east_m, north_m = _to_utm.transform(bbox["east"], bbox["north"])
    return from_bounds(west_m, south_m, east_m, north_m, W, H)


def _shift_mask(mask: np.ndarray, dr: int, dc: int) -> np.ndarray:
    """Translate a boolean mask by (dr rows, dc cols), clipping at image edges."""
    H, W = mask.shape
    out = np.zeros_like(mask)
    src_r0 = max(0, -dr);  src_r1 = H - max(0, dr)
    dst_r0 = max(0,  dr);  dst_r1 = H + min(0, dr)
    src_c0 = max(0, -dc);  src_c1 = W - max(0, dc)
    dst_c0 = max(0,  dc);  dst_c1 = W + min(0, dc)
    if dst_r1 > dst_r0 and dst_c1 > dst_c0:
        out[dst_r0:dst_r1, dst_c0:dst_c1] = mask[src_r0:src_r1, src_c0:src_c1]
    return out


def estimate_tree_heights(
    tree_mask: np.ndarray,
    pixel_size_m: float,
    min_component_pixels: int = 50,
    max_crown_radius_m: float = MAX_CROWN_RADIUS_M,
    dominant_genus: str | None = None,
) -> tuple[np.ndarray, dict[int, tuple[float, float]]]:
    """Estimate per-tree-cluster height from canopy area using a power-law allometric model.

    For each connected component:
        crown_area_m² = n_pixels × pixel_size_m²
        height_m      = exp(ALLOMETRIC_A + ALLOMETRIC_B × ln(crown_area_m²))
                        (log-linear form: ln(H) = A + B·ln(CPA), fitted on Schmucker et al. 2022)

    Components whose equivalent crown radius exceeds max_crown_radius_m are split
    via watershed before height estimation, preventing the sqrt(N) height inflation
    that occurs when N merged trees are treated as one giant tree.

    Parameters
    ----------
    tree_mask : np.ndarray
        Bool (H, W) — True where pixels are classified as trees (class 1).
    pixel_size_m : float
        Metres per pixel (derived from tile bbox and image dimensions).
    min_component_pixels : int
        Components smaller than this are skipped (noise).
    max_crown_radius_m : float
        Single-tree radius threshold. Larger components are watershed-split.
        Default 8 m ≈ 200 m² crown area.

    Returns
    -------
    labeled : np.ndarray
        Integer (H, W) array of component labels.
    heights : dict[int, tuple[float, float]]
        Mapping of label → (tree_height_m, crown_radius_m).
    """
    from scipy.ndimage import distance_transform_edt
    from skimage.feature import peak_local_max
    from skimage.segmentation import watershed

    A, B = ALLOMETRIC_PROFILES.get(dominant_genus, (ALLOMETRIC_A, ALLOMETRIC_B)) \
           if dominant_genus else (ALLOMETRIC_A, ALLOMETRIC_B)

    labeled, n = cc_label(tree_mask, structure=_8CONN)
    labeled = labeled.astype(np.int32)
    heights: dict[int, tuple[float, float]] = {}
    px_area = pixel_size_m ** 2
    next_label = int(labeled.max()) + 1
    min_dist_px = max(1, int(max_crown_radius_m * 0.5 / pixel_size_m))

    for k in range(1, n + 1):
        comp_mask = labeled == k
        n_pixels = int(comp_mask.sum())
        if n_pixels < min_component_pixels:
            continue

        crown_area_m2 = n_pixels * px_area
        crown_radius_m = math.sqrt(crown_area_m2 / math.pi)

        if crown_radius_m <= max_crown_radius_m:
            heights[k] = (math.exp(A + B * math.log(crown_area_m2)), crown_radius_m)
            continue

        # Large cluster: watershed to recover individual crowns
        dist = distance_transform_edt(comp_mask)
        coords = peak_local_max(dist, min_distance=min_dist_px, labels=comp_mask.astype(np.int32))

        if len(coords) <= 1:
            # Only one peak — cap the area to a single-crown max before estimating height
            capped_radius = min(crown_radius_m, max_crown_radius_m)
            capped_area = math.pi * capped_radius ** 2
            heights[k] = (math.exp(A + B * math.log(capped_area)), capped_radius)
            continue

        peaks_mask = np.zeros(dist.shape, dtype=bool)
        peaks_mask[tuple(coords.T)] = True
        markers, _ = cc_label(peaks_mask)
        split = watershed(-dist, markers, mask=comp_mask)

        for sub_val in np.unique(split[comp_mask]):
            if sub_val == 0:
                continue
            sub_mask = split == sub_val
            sub_pixels = int(sub_mask.sum())
            if sub_pixels < min_component_pixels:
                labeled[sub_mask] = 0
                continue
            sub_area_m2 = sub_pixels * px_area
            sub_radius_m = min(math.sqrt(sub_area_m2 / math.pi), max_crown_radius_m)
            labeled[sub_mask] = next_label
            heights[next_label] = (math.exp(A + B * math.log(sub_area_m2)), sub_radius_m)
            next_label += 1

        # Zero pixels still carrying original label k (small watershed regions)
        labeled[comp_mask & (labeled == k)] = 0

    return labeled, heights


def vectorize_trees(
    tree_mask: np.ndarray,
    bbox: dict,
    vegetation_model: str,
    min_component_pixels: int = 50,
    dominant_genus: str | None = None,
    max_crown_radius_m: float = MAX_CROWN_RADIUS_M,
) -> gpd.GeoDataFrame:
    """Convert a tree mask to georeferenced polygon features in EPSG:25832.

    Runs plain connected-component labeling (no watershed splitting) and the
    allometric height formula, then traces each component into a Shapely polygon
    via rasterio.features.shapes(). One polygon per canopy blob.

    Returns a GeoDataFrame with columns:
      tree_id (int), geometry (Polygon), height_m, crown_radius_m,
      crown_area_m2 (float), vegetation_model (str).
    """
    H, W = tree_mask.shape
    pixel_size_m = _pixel_size_m(bbox, (H, W))
    transform = _bbox_to_transform(bbox, (H, W))
    px_area = pixel_size_m ** 2

    A, B = ALLOMETRIC_PROFILES.get(dominant_genus, (ALLOMETRIC_A, ALLOMETRIC_B)) \
           if dominant_genus else (ALLOMETRIC_A, ALLOMETRIC_B)

    labeled, n = cc_label(tree_mask, structure=_8CONN)
    labeled = labeled.astype(np.int32)

    records = []
    for k in range(1, n + 1):
        comp_mask = labeled == k
        n_pixels = int(comp_mask.sum())
        if n_pixels < min_component_pixels:
            continue
        crown_area_m2 = n_pixels * px_area
        crown_radius_m = math.sqrt(crown_area_m2 / math.pi)
        height_m = math.exp(A + B * math.log(crown_area_m2))

        component = comp_mask.astype(np.uint8)
        polys = [
            sg_shape(geom)
            for geom, val in rio_shapes(component, mask=component, transform=transform)
            if val == 1
        ]
        if not polys:
            continue
        geom = max(polys, key=lambda g: g.area)
        records.append({
            "tree_id": k,
            "geometry": geom,
            "height_m": height_m,
            "allometric_height_m": height_m,
            "crown_radius_m": crown_radius_m,
            "crown_area_m2": crown_area_m2,
            "vegetation_model": vegetation_model,
        })

    if not records:
        return gpd.GeoDataFrame(
            columns=["tree_id", "geometry", "height_m", "allometric_height_m",
                     "crown_radius_m", "crown_area_m2", "vegetation_model"],
            crs="EPSG:25832",
        )
    return gpd.GeoDataFrame(records, crs="EPSG:25832")


def vectorize_shadows(
    shadow_mask: np.ndarray,
    bbox: dict,
    dt_utc: dt.datetime,
    vegetation_model: str,
) -> gpd.GeoDataFrame:
    """Convert a boolean shadow mask to georeferenced polygon features in EPSG:25832.

    Returns a GeoDataFrame with columns:
      geometry (Polygon), datetime_utc (str), vegetation_model (str), area_m2 (float).
    """
    H, W = shadow_mask.shape
    transform = _bbox_to_transform(bbox, (H, W))
    mask_u8 = shadow_mask.astype(np.uint8)

    records = []
    for geom, val in rio_shapes(mask_u8, mask=mask_u8, transform=transform):
        if val == 1:
            poly = sg_shape(geom)
            records.append({
                "geometry": poly,
                "datetime_utc": dt_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "vegetation_model": vegetation_model,
                "area_m2": round(poly.area, 2),
            })

    if not records:
        return gpd.GeoDataFrame(
            columns=["geometry", "datetime_utc", "vegetation_model", "area_m2"],
            crs="EPSG:25832",
        )
    return gpd.GeoDataFrame(records, crs="EPSG:25832")


def cast_tree_shadows(
    seg_map: np.ndarray,
    bbox: dict,
    dt_utc: dt.datetime,
    min_elevation_deg: float = 5.0,
    max_shadow_factor: float = 5.0,
    min_component_pixels: int = 50,
    tree_gdf: gpd.GeoDataFrame | None = None,
    max_crown_radius_m: float = MAX_CROWN_RADIUS_M,
) -> np.ndarray:
    """Compute where tree canopies cast shadows given a sun position.

    Parameters
    ----------
    seg_map : np.ndarray
        uint8 (H, W) segmentation map: 0=other, 1=tree, 2=road, 3=building.
    bbox : dict
        WGS84 tile bounding box {west, east, south, north} — same dict as
        returned by tiles_for_area().
    dt_utc : datetime.datetime
        Timezone-aware UTC datetime for solar position calculation.
    min_elevation_deg : float
        Sun elevation floor. Below this (incl. nighttime), returns an all-False mask.
    max_shadow_factor : float
        Cap shadow length at this multiple of crown radius to avoid extreme
        shadows at very low sun angles.
    min_component_pixels : int
        Tree clusters smaller than this many pixels are ignored.

    Returns
    -------
    np.ndarray
        Bool (H, W) — True where a tree shadow falls. Source tree pixels are
        excluded (the tree itself is still green in the overlay).
    """
    H, W = seg_map.shape
    pixel_size_m = _pixel_size_m(bbox, (H, W))

    lat, lon = _tile_center(bbox)
    azimuth_deg, elevation_deg = sun_position(lat, lon, dt_utc)

    if elevation_deg < min_elevation_deg:
        return np.zeros((H, W), dtype=bool)

    elevation_rad = math.radians(elevation_deg)
    shadow_az_rad = math.radians((azimuth_deg + 180.0) % 360.0)

    if tree_gdf is not None and len(tree_gdf) > 0:
        transform = _bbox_to_transform(bbox, (H, W))
        labeled = np.zeros((H, W), dtype=np.int32)
        heights: dict[int, tuple[float, float]] = {}
        for row in tree_gdf.itertuples():
            burned = rio_rasterize(
                [(row.geometry, int(row.tree_id))],
                out_shape=(H, W), transform=transform,
                fill=0, dtype=np.int32,
            )
            labeled = np.where(burned > 0, burned, labeled)
            heights[int(row.tree_id)] = (float(row.height_m), float(row.crown_radius_m))
    else:
        labeled, heights = estimate_tree_heights(
            seg_map == 1, pixel_size_m, min_component_pixels, max_crown_radius_m
        )

    shadow_mask = np.zeros((H, W), dtype=bool)
    building_mask = seg_map == 3

    for k, (tree_height_m, crown_radius_m) in heights.items():
        shadow_length_m = min(
            tree_height_m / math.tan(elevation_rad),
            max_shadow_factor * crown_radius_m,
        )
        # Image convention: row 0 is north (top), col 0 is west (left)
        dx_px = shadow_length_m * math.sin(shadow_az_rad) / pixel_size_m
        dy_px = -shadow_length_m * math.cos(shadow_az_rad) / pixel_size_m

        comp_mask = labeled == k
        dy_i = int(round(dy_px))
        dx_i = int(round(dx_px))
        n_steps = max(abs(dy_i), abs(dx_i), 1)
        for step in range(n_steps + 1):
            shifted = _shift_mask(
                comp_mask,
                int(round(dy_px * step / n_steps)),
                int(round(dx_px * step / n_steps)),
            )
            shadow_mask |= shifted
            if (shifted & building_mask).any():
                break  # building occludes further shadow propagation

    shadow_mask &= ~(seg_map == 1)
    shadow_mask &= ~building_mask
    return shadow_mask
