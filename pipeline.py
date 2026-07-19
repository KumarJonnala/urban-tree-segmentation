from __future__ import annotations
"""Pipeline entry point.

Subcommands:
  download          — fetch orthophoto tile grid + full-area overview image
  segment           — run OSM + vegetation segmentation on downloaded tiles
  compare models    — run all vegetation methods side-by-side and print metrics
  compare sizes     — pairwise IoU across tile sizes on a shared UTM grid
  shadow            — cast tree shadows from segmentation maps for a given datetime
  merge             — merge per-tile tree and shadow FGBs into one full-area FGB per size
  render            — render merged tree/shadow polygons over the full-area orthophoto
  species-grid      — multi-panel PNG showing dominant BK species per tile for all tile sizes
  diurnal           — hourly shadow table + overlays for one tile across a full day
  status            — show what has been computed per area and tile size
  tune              — grid-search hyperparameters for a tunable model against a reference
  all               — download + segment + shadow (default when no subcommand given)

Tile size flags (available on download, segment, shadow, all):
  --tile-size M    run for a single tile size M (metres)
  --all-sizes      run for all sizes in TILE_SIZES_M [100, 250, 500, 1000]
  (default: TILE_SIZE_M = 250)

Examples:
  python pipeline.py                                           # download + segment + shadow @ 250m
  python pipeline.py --all-sizes                               # full pipeline for all tile sizes
  python pipeline.py download --dry-run --all-sizes            # preview all tile grids
  python pipeline.py download --tile-size 500                  # download 500m tiles only
  python pipeline.py segment --vegetation-model vari
  python pipeline.py compare models --area ovgu_bbox
  python pipeline.py compare sizes --resolution 1.0 --reference-size 250
  python pipeline.py shadow --datetime-utc "2026-06-21T09:00:00" --all-sizes
  python pipeline.py merge --all-sizes                               # merge trees + shadows for all sizes
  python pipeline.py merge --layer trees --tile-size 250             # trees only at 250m
  python pipeline.py render --all-sizes                              # full-area overlay image for all sizes
  python pipeline.py render --tile-size 500                          # single size
  python pipeline.py species-grid                                    # dominant-species grid for all tile sizes
  python pipeline.py diurnal --date-utc "2026-06-21" --tile 1_1 --tile-size 250
  python pipeline.py status --all-sizes
  python pipeline.py tune --model vari --tile 0_0
"""

import argparse
import time

import numpy as np
from PIL import Image

from src.config import AREAS, DEFAULT_VEGETATION_MODEL, MAX_CROWN_RADIUS_M, OUTPUT_DIR, TILE_SIZE_M, TILE_SIZES_M
from src.data_preprocessing import fetch_area_grid, fetch_full_area_image, fetch_buildings, fetch_roads, tiles_for_area
from src.segmentation import (
    compare_vegetation,
    save_segmentation,
    vari_mask,
)

VEGETATION_MODELS = ("vari", "deepforest", "samgeo", "segformer_b5", "deeplab",
                     "tcd_segformer", "ensemble")
TUNABLE_MODELS = ("vari", "deepforest", "samgeo")


def _merge_layer(area_name: str, vegetation_model: str, layer: str, out_dir):
    """Concatenate per-tile FGB files for *layer* ('trees' or 'shadow') into one full-area FGB.

    Writes {area}_{model}_{layer}_merged.fgb into out_dir and returns the path,
    or None if no tile files are found.
    """
    import geopandas as gpd
    import pandas as pd
    from pathlib import Path as _Path

    out_dir = _Path(out_dir)
    fgb_files = sorted(out_dir.glob(f"{area_name}_tile_*_{vegetation_model}_{layer}.fgb"))
    if not fgb_files:
        return None
    gdfs = [gdf for f in fgb_files for gdf in (gpd.read_file(f),) if len(gdf) > 0]
    if not gdfs:
        return None  # all tile FGBs exist but are empty (e.g. no shadows cast)
    merged = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs="EPSG:25832")
    out_path = out_dir / f"{area_name}_{vegetation_model}_{layer}_merged.fgb"
    merged.to_file(out_path, driver="FlatGeobuf")
    return out_path


def _dissolve_for_render(gdf: "gpd.GeoDataFrame", snap_m: float = 0.3) -> "gpd.GeoDataFrame":
    """Merge touching polygons to remove watershed ridge and tile-boundary seams for rendering.

    buffer(+snap) closes 1-pixel gaps → unary_union merges touching polygons →
    buffer(-snap) restores the original boundary extent.
    Returns a geometry-only GeoDataFrame; attributes are not preserved.
    snap_m=0.3 is just above the 0.2 m/pixel DOP20 resolution.
    """
    import geopandas as gpd
    from shapely.ops import unary_union
    union = unary_union(gdf.geometry.buffer(snap_m))
    dissolved = gpd.GeoDataFrame(
        geometry=gpd.GeoSeries([union], crs=gdf.crs).explode(index_parts=False),
        crs=gdf.crs,
    ).reset_index(drop=True)
    dissolved["geometry"] = dissolved.geometry.buffer(-snap_m)
    return dissolved


def _load_vegetation_model(name: str):
    """Load the requested model and return a callable mask_fn(img) -> bool array."""
    if name == "vari":
        return None, vari_mask

    if name == "deepforest":
        from src.segmentation import deepforest_mask, load_deepforest
        print(f"  Loading DeepForest model...")
        m = load_deepforest()
        return m, lambda img, model=m: deepforest_mask(img, model=model)

    if name == "samgeo":
        from src.segmentation import load_samgeo, samgeo_mask
        print(f"  Loading SamGeo...")
        m = load_samgeo()
        return m, lambda img, model=m: samgeo_mask(img, model=model)

    if name == "segformer_b5":
        from src.segmentation import load_segformer_b5, segformer_b5_mask
        print(f"  Loading SegFormer-B5 model...")
        proc, mdl = load_segformer_b5()
        return (proc, mdl), lambda img, p=proc, m=mdl: segformer_b5_mask(img, processor=p, model=m)

    if name == "deeplab":
        from src.segmentation import deeplab_mask, load_deeplab
        print(f"  Loading DeepLab model...")
        m = load_deeplab()
        return m, lambda img, model=m: deeplab_mask(img, model=model)

    if name == "tcd_segformer":
        from src.segmentation import load_tcd_segformer, tcd_segformer_mask
        print(f"  Loading TCD SegFormer (restor/tcd-segformer-mit-b5)...")
        proc, mdl = load_tcd_segformer()
        return (proc, mdl), lambda img, p=proc, m=mdl: tcd_segformer_mask(img, processor=p, model=m)

    if name == "ensemble":
        from src.segmentation import load_deepforest, ensemble_mask
        print(f"  Loading DeepForest for ensemble (VARI ∩ DeepForest)...")
        m = load_deepforest()
        return m, lambda img, model=m: ensemble_mask(img, df_model=model)

    raise ValueError(f"Unknown vegetation model: {name!r}. Choose from {VEGETATION_MODELS}")


def cmd_download(dry_run: bool = False, tile_size_m: int | None = None, all_sizes: bool = False) -> None:
    sizes = TILE_SIZES_M if all_sizes else [tile_size_m or TILE_SIZE_M]
    for area_name, area in AREAS.items():
        if not dry_run:
            t0 = time.perf_counter()
            full_path = fetch_full_area_image(area_name)
            print(f"\n--- {area_name}: full-area image → {full_path.name} ({time.perf_counter()-t0:.1f}s) ---")
        for size in sizes:
            tiles = tiles_for_area(area, size)
            print(f"\n--- {area_name} @ {size}m: {len(tiles)} tile(s) ---")
            if dry_run:
                for t in tiles:
                    print(f"  tile ({t['ix']},{t['iy']})  W={t['west']:.6f} S={t['south']:.6f} E={t['east']:.6f} N={t['north']:.6f}")
            else:
                t0 = time.perf_counter()
                paths = fetch_area_grid(area_name, tile_size_m=size)
                elapsed = time.perf_counter() - t0
                n = len(paths)
                print(f"  {n} tile(s) saved — {elapsed:.1f}s total, {elapsed/n:.2f}s/tile")


def _save_tile_summary(area_name: str, seg_dir, merged_gdf, bk_gdf, tile_size_m: int) -> list:
    """Save a 6-panel tile-summary PNG alongside the merged FGB.

    Panels (2 × 3):
      1. Majority BK genus        2. Height bias (m)       3. Height MAE (m)
      4. BK match rate (%)        5. Crown area ratio       6. Mean tree height (m)
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import matplotlib.patches as mpatches
    from pathlib import Path as _Path
    from PIL import Image as _PILImage
    from pyproj import Transformer as _Transformer
    import geopandas as gpd
    from shapely.geometry import box as _box
    from src.data_preprocessing import tiles_for_area

    _to_utm = _Transformer.from_crs("EPSG:4326", "EPSG:25832", always_xy=True)

    out_path = _Path(seg_dir) / f"tile_summary_{tile_size_m}m.png"
    full_img_path = OUTPUT_DIR / area_name / f"{area_name}_full.png"
    if not full_img_path.exists():
        print(f"  [tile_summary] orthophoto not found, skipping: {full_img_path}")
        return

    full_img = np.array(_PILImage.open(full_img_path).convert("RGB"))
    ovgu = AREAS[area_name]
    west_m, south_m = _to_utm.transform(ovgu["west"], ovgu["south"])
    east_m, north_m = _to_utm.transform(ovgu["east"], ovgu["north"])
    extent = [west_m, east_m, south_m, north_m]

    tiles = tiles_for_area(AREAS[area_name], tile_size_m)
    N_MIN = 5  # minimum matched trees to show error stats (else grey)

    # Pre-compute per-tile stats
    records = []
    for t in tiles:
        tw_m, ts_m = _to_utm.transform(t["west"], t["south"])
        te_m, tn_m = _to_utm.transform(t["east"], t["north"])
        tile_poly = _box(tw_m, ts_m, te_m, tn_m)
        tile_frame = gpd.GeoDataFrame(geometry=[tile_poly], crs="EPSG:25832")

        # Pipeline trees whose centroid falls in this tile
        pipe = merged_gdf[merged_gdf.geometry.centroid.within(tile_poly)]
        n_pipe = len(pipe)
        matched = pipe[pipe["height_source"] == "measured"] if "height_source" in pipe.columns else pipe.iloc[:0]
        n_matched = len(matched)

        bias, mae, match_rate, mean_h = None, None, None, None
        rel_bias = mape = r2 = None
        bias_global = mae_global = mape_global = r2_global = None
        if n_matched >= N_MIN:
            err = matched["allometric_height_m"] - matched["height_m"]
            bias = float(err.mean())
            mae  = float(err.abs().mean())
            rel_bias = float((err / matched["height_m"]).mean() * 100)
            mape     = float((err.abs() / matched["height_m"]).mean() * 100)
            ss_res = float((err ** 2).sum())
            ss_tot = float(((matched["height_m"] - matched["height_m"].mean()) ** 2).sum())
            r2     = float(1 - ss_res / ss_tot) if ss_tot > 0 else None
            if "h_global_m" in matched.columns:
                err_g = matched["h_global_m"] - matched["height_m"]
                ss_res_g = float((err_g ** 2).sum())
                bias_global  = float(err_g.mean())
                mae_global   = float(err_g.abs().mean())
                mape_global  = float((err_g.abs() / matched["height_m"]).mean() * 100)
                r2_global    = float(1 - ss_res_g / ss_tot) if ss_tot > 0 else None
        if n_pipe > 0:
            match_rate = n_matched / n_pipe * 100
            mean_h = float(pipe["height_m"].mean())

        # BK clip for crown area ratio and dominant genus
        bk_clip = bk_gdf.clip(tile_frame).reset_index(drop=True) if bk_gdf is not None else gpd.GeoDataFrame()
        bk_valid = bk_clip[bk_clip["Kronendurchmesser"] > 0] if len(bk_clip) > 0 else bk_clip
        bk_area = float((3.14159 * (bk_valid["Kronendurchmesser"] / 2) ** 2).sum()) if len(bk_valid) > 0 else 0
        pipe_area = float(pipe["crown_area_m2"].sum()) if "crown_area_m2" in pipe.columns else 0
        ratio = pipe_area / bk_area if bk_area > 0 else None

        tile_area_m2 = (te_m - tw_m) * (tn_m - ts_m)
        pipe_cov = pipe_area / tile_area_m2 * 100
        bk_cov   = bk_area  / tile_area_m2 * 100

        genus = None
        if len(bk_clip) > 0:
            vc = bk_clip["Gattung lang"].str.split(",").str[0].str.strip().value_counts()
            genus = vc.index[0] if len(vc) > 0 else None

        records.append({
            "ix": t["ix"], "iy": t["iy"],
            "geometry": tile_poly,
            "genus": genus or "—",
            "bias": bias, "mae": mae,
            "rel_bias": rel_bias, "mape": mape, "r2": r2,
            "bias_global": bias_global, "mae_global": mae_global,
            "mape_global": mape_global, "r2_global": r2_global,
            "match_rate": match_rate,
            "ratio": ratio,
            "mean_h": mean_h,
            "n_matched": n_matched,
            "n_pipe": n_pipe,
            "n_bk": len(bk_valid),
            "pipe_area": pipe_area,
            "bk_area": bk_area,
            "pipe_cov": pipe_cov,
            "bk_cov": bk_cov,
            "tile_area_m2": tile_area_m2,
        })

    # Build genus colour map (categorical)
    all_genera = sorted({r["genus"] for r in records})
    tab20 = plt.cm.tab20.colors
    genus_color = {g: tab20[i % len(tab20)] for i, g in enumerate(all_genera)}

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f"{area_name}  |  {tile_size_m} m tiles  |  tile summary", fontsize=13)

    panel_cfg = [
        # (ax, value_key, title, cmap, norm, fmt, use_genus_color)
        (axes[0, 0], "genus",      "Majority BK genus",         None,                                 None,                                                       None,    True),
        (axes[0, 1], "bias",       "Allometric height bias (m)", plt.cm.RdBu_r,                        None,                                                       "{:+.1f}", False),
        (axes[0, 2], "mae",        "Height MAE (m)",             plt.cm.YlOrRd,                        None,                                                       "{:.1f}",  False),
        (axes[1, 0], "match_rate", "BK match rate (%)",          plt.cm.YlGn,                          mcolors.Normalize(vmin=0, vmax=100),                        "{:.0f}%", False),
        (axes[1, 1], "ratio",      "Crown area ratio (pipe/BK)", plt.cm.RdYlGn,                        None,                                                       "{:.2f}",  False),
        (axes[1, 2], "mean_h",     "Mean tree height (m)",       plt.cm.viridis,                       None,                                                       "{:.1f}",  False),
    ]

    # Compute dynamic norms for panels without a fixed norm
    def _safe_norm(vals, centre=None):
        vals = [v for v in vals if v is not None]
        if not vals:
            return mcolors.Normalize(0, 1)
        lo, hi = min(vals), max(vals)
        if centre is not None:
            lim = max(abs(lo - centre), abs(hi - centre), 0.1)
            return mcolors.TwoSlopeNorm(vmin=centre - lim, vcenter=centre, vmax=centre + lim)
        return mcolors.Normalize(vmin=lo, vmax=max(hi, lo + 0.1))

    dynamic_norms = {
        "bias":   _safe_norm([r["bias"]  for r in records], centre=0),
        "mae":    _safe_norm([r["mae"]   for r in records]),
        "ratio":  _safe_norm([r["ratio"] for r in records], centre=1),
        "mean_h": _safe_norm([r["mean_h"] for r in records]),
    }

    def _abbrev_species(name: str) -> str:
        """Abbreviate 'Tilia cordata' → 'T. cordata' to fit inside tile labels."""
        parts = name.split()
        if len(parts) >= 2:
            return f"{parts[0][0]}. {' '.join(parts[1:])}"
        return name

    for ax, key, title, cmap, norm, fmt, use_genus in panel_cfg:
        ax.imshow(full_img, extent=extent, origin="upper", aspect="equal")
        if use_genus:
            legend_patches = []
        else:
            actual_norm = norm if norm is not None else dynamic_norms.get(key)

        for r in records:
            val = r[key]
            gdf_t = gpd.GeoDataFrame([{"geometry": r["geometry"]}], crs="EPSG:25832")
            if use_genus:
                color = genus_color[val]
                gdf_t.plot(ax=ax, facecolor=color, edgecolor="white", linewidth=1, alpha=0.55, zorder=2)
                lbl = _abbrev_species(val) if val != "—" else "—"
            elif val is None:
                gdf_t.plot(ax=ax, facecolor="#888888", edgecolor="white", linewidth=1, alpha=0.5, zorder=2)
                lbl = "—"
            else:
                color = cmap(actual_norm(val))
                gdf_t.plot(ax=ax, facecolor=color, edgecolor="white", linewidth=1, alpha=0.6, zorder=2)
                if key == "match_rate":
                    lbl = f"{r['n_matched']} / {r['n_pipe']}\n{val:.0f}%"
                elif key == "ratio":
                    lbl = f"{r['pipe_area']:.0f} / {r['bk_area']:.0f} m²\n= {val:.2f}"
                else:
                    lbl = fmt.format(val)

            cx, cy = r["geometry"].centroid.x, r["geometry"].centroid.y
            ax.text(cx, cy, lbl, ha="center", va="center", fontsize=7, fontweight="bold",
                    bbox=dict(facecolor="white", alpha=0.65, edgecolor="none", pad=1.5), zorder=3)

        ax.set_xlim(west_m, east_m); ax.set_ylim(south_m, north_m)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Easting (m)", fontsize=8); ax.set_ylabel("Northing (m)", fontsize=8)
        ax.tick_params(labelsize=7)

        if use_genus:
            handles = [mpatches.Patch(color=genus_color[g], label=g) for g in all_genera]
            ax.legend(handles=handles, fontsize=6, loc="lower right", framealpha=0.85, ncol=2)
        else:
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=actual_norm)
            sm.set_array([])
            plt.colorbar(sm, ax=ax, shrink=0.65, pad=0.02)

    # Aggregate totals for all footer lines
    total_matched   = sum(r["n_matched"]  for r in records)
    total_pipe      = sum(r["n_pipe"]     for r in records)
    total_pipe_area = sum(r["pipe_area"]  for r in records)
    total_bk_area   = sum(r["bk_area"]   for r in records)
    match_pct_total = total_matched / total_pipe * 100 if total_pipe > 0 else 0
    ratio_total     = total_pipe_area / total_bk_area  if total_bk_area > 0 else 0

    bias_vals     = [r["bias"]     for r in records if r["bias"]     is not None]
    mae_vals      = [r["mae"]      for r in records if r["mae"]      is not None]
    mean_h_vals   = [r["mean_h"]   for r in records if r["mean_h"]   is not None]
    rel_bias_vals = [r["rel_bias"] for r in records if r["rel_bias"] is not None]
    mape_vals     = [r["mape"]     for r in records if r["mape"]     is not None]
    r2_vals       = [r["r2"]       for r in records if r["r2"]       is not None]
    total_bias     = sum(bias_vals)     / len(bias_vals)     if bias_vals     else None
    total_mae      = sum(mae_vals)      / len(mae_vals)      if mae_vals      else None
    total_mean_h   = sum(mean_h_vals)   / len(mean_h_vals)   if mean_h_vals   else None
    total_rel_bias = sum(rel_bias_vals) / len(rel_bias_vals) if rel_bias_vals else None
    total_mape     = sum(mape_vals)     / len(mape_vals)     if mape_vals     else None
    total_r2       = sum(r2_vals)       / len(r2_vals)       if r2_vals       else None
    total_tile_area = sum(r["tile_area_m2"] for r in records)
    total_pipe_cov  = total_pipe_area / total_tile_area * 100 if total_tile_area > 0 else 0
    total_bk_cov    = total_bk_area   / total_tile_area * 100 if total_tile_area > 0 else 0

    # Write tile summary JSON (before figure, so it saves even if matplotlib fails)
    import json as _json
    json_path = _Path(seg_dir) / f"tile_summary_{tile_size_m}m.json"
    json_data = {
        "area_name": area_name,
        "tile_size_m": tile_size_m,
        "n_tiles": len(records),
        "summary": {
            "bias_m":             round(total_bias,     3) if total_bias     is not None else None,
            "mae_m":              round(total_mae,      3) if total_mae      is not None else None,
            "rel_bias_pct":       round(total_rel_bias, 1) if total_rel_bias is not None else None,
            "mape_pct":           round(total_mape,     1) if total_mape     is not None else None,
            "r2":                 round(total_r2,       3) if total_r2       is not None else None,
            "mean_height_m":      round(total_mean_h,   3) if total_mean_h   is not None else None,
            "n_matched":          total_matched,
            "n_segmented":        total_pipe,
            "match_rate_pct":     round(match_pct_total, 1),
            "pipe_crown_area_m2": round(total_pipe_area, 1),
            "bk_crown_area_m2":   round(total_bk_area,   1),
            "crown_area_ratio":   round(ratio_total, 3) if total_bk_area > 0 else None,
            "pipe_canopy_pct":    round(total_pipe_cov, 1),
            "bk_canopy_pct":      round(total_bk_cov,  1),
        },
        "tiles": [
            {
                "ix": r["ix"], "iy": r["iy"],
                "genus":              r["genus"],
                "bias_m":             round(r["bias"],     3) if r["bias"]     is not None else None,
                "mae_m":              round(r["mae"],      3) if r["mae"]      is not None else None,
                "rel_bias_pct":       round(r["rel_bias"], 1) if r["rel_bias"] is not None else None,
                "mape_pct":           round(r["mape"],     1) if r["mape"]     is not None else None,
                "r2":                 round(r["r2"],       3) if r["r2"]       is not None else None,
                "bias_global_m":      round(r["bias_global"],  3) if r["bias_global"]  is not None else None,
                "mae_global_m":       round(r["mae_global"],   3) if r["mae_global"]   is not None else None,
                "mape_global_pct":    round(r["mape_global"],  1) if r["mape_global"]  is not None else None,
                "r2_global":          round(r["r2_global"],    3) if r["r2_global"]    is not None else None,
                "match_rate_pct":     round(r["match_rate"], 1) if r["match_rate"] is not None else None,
                "n_matched":          r["n_matched"],
                "n_segmented":        r["n_pipe"],
                "n_bk":               r["n_bk"],
                "pipe_crown_area_m2": round(r["pipe_area"], 1),
                "bk_crown_area_m2":   round(r["bk_area"],  1),
                "crown_area_ratio":   round(r["ratio"],     3) if r["ratio"]   is not None else None,
                "pipe_canopy_pct":    round(r["pipe_cov"],  1),
                "bk_canopy_pct":      round(r["bk_cov"],   1),
                "mean_height_m":      round(r["mean_h"],    3) if r["mean_h"]  is not None else None,
            }
            for r in records
        ],
    }
    json_path.write_text(_json.dumps(json_data, indent=2))
    print(f"  tile summary JSON → {json_path.name}")

    # Reserve bottom margin so footers don't overlap panel content
    plt.tight_layout(rect=[0, 0.05, 1, 1])

    footers = [
        (axes[0, 1], f"Area total  bias = {total_bias:+.2f} m"  if total_bias  is not None else "—"),
        (axes[0, 2], f"Area total  MAE = {total_mae:.2f} m"      if total_mae   is not None else "—"),
        (axes[1, 0], f"Total: {total_matched} matched / {total_pipe} segmented ({match_pct_total:.0f}%)"),
        (axes[1, 1], f"Total: {total_pipe_area:.0f} m² pipeline / {total_bk_area:.0f} m² BK  (ratio {ratio_total:.2f})"),
        (axes[1, 2], f"Area mean height = {total_mean_h:.1f} m"  if total_mean_h is not None else "—"),
    ]
    for ax_f, txt in footers:
        ax_f.text(0.5, -0.08, txt, transform=ax_f.transAxes,
                  ha="center", va="top", fontsize=9)

    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  tile summary → {out_path.name}")
    return records


def _save_tile_summary_pct(area_name: str, seg_dir, records: list, tile_size_m: int) -> None:
    """Save a second 6-panel tile summary PNG with normalised percentage metrics.

    Uses the same records list returned by _save_tile_summary (no recomputation).
    Panels: genus | rel_bias% | MAPE% | match_rate% | pipe_canopy% | R²
    Saved as tile_summary_pct_{tile_size_m}m.png in seg_dir.
    """
    from pathlib import Path as _Path
    from pyproj import Transformer as _Transformer
    import numpy as _np
    from PIL import Image as _PILImage
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import matplotlib.patches as mpatches
    import geopandas as gpd

    out_path = _Path(seg_dir) / f"tile_summary_pct_{tile_size_m}m.png"
    _to_utm  = _Transformer.from_crs("EPSG:4326", "EPSG:25832", always_xy=True)

    # Need orthophoto + extent (reuse same logic as _save_tile_summary)
    ovgu = AREAS[area_name]
    west_m, south_m = _to_utm.transform(ovgu["west"], ovgu["south"])
    east_m, north_m = _to_utm.transform(ovgu["east"], ovgu["north"])
    extent = [west_m, east_m, south_m, north_m]

    img_dir  = OUTPUT_DIR / area_name
    img_path = img_dir / f"{area_name}_full.png"
    if not img_path.exists():
        print(f"  [tile_summary_pct] orthophoto not found, skipping: {img_path}")
        return
    full_img = _np.array(_PILImage.open(img_path).convert("RGB"))

    # Aggregate totals for footers
    rel_bias_vals = [r["rel_bias"] for r in records if r["rel_bias"] is not None]
    mape_vals     = [r["mape"]     for r in records if r["mape"]     is not None]
    r2_vals       = [r["r2"]       for r in records if r["r2"]       is not None]
    mean_h_vals   = [r["mean_h"]   for r in records if r["mean_h"]   is not None]
    total_matched   = sum(r["n_matched"] for r in records)
    total_pipe      = sum(r["n_pipe"]    for r in records)
    total_pipe_area = sum(r["pipe_area"] for r in records)
    total_bk_area   = sum(r["bk_area"]  for r in records)
    total_tile_area = sum(r["tile_area_m2"] for r in records)
    match_pct_total = total_matched / total_pipe * 100 if total_pipe > 0 else 0
    total_rel_bias  = sum(rel_bias_vals) / len(rel_bias_vals) if rel_bias_vals else None
    total_mape      = sum(mape_vals)     / len(mape_vals)     if mape_vals     else None
    total_r2        = sum(r2_vals)       / len(r2_vals)       if r2_vals       else None
    total_mean_h    = sum(mean_h_vals)   / len(mean_h_vals)   if mean_h_vals   else None
    total_pipe_cov  = total_pipe_area / total_tile_area * 100 if total_tile_area > 0 else 0
    total_bk_cov    = total_bk_area   / total_tile_area * 100 if total_tile_area > 0 else 0

    # Genus colour map
    all_genera = sorted({r["genus"] for r in records})
    tab20 = plt.cm.tab20.colors
    genus_color = {g: tab20[i % len(tab20)] for i, g in enumerate(all_genera)}

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f"{area_name}  |  {tile_size_m} m tiles  |  normalised metrics", fontsize=13)

    def _safe_norm_pct(vals, centre=None):
        vals = [v for v in vals if v is not None]
        if not vals:
            return mcolors.Normalize(0, 1)
        lo, hi = min(vals), max(vals)
        if centre is not None:
            lim = max(abs(lo - centre), abs(hi - centre), 1.0)
            return mcolors.TwoSlopeNorm(vmin=centre - lim, vcenter=centre, vmax=centre + lim)
        return mcolors.Normalize(vmin=lo, vmax=max(hi, lo + 0.1))

    dynamic_norms = {
        "rel_bias": _safe_norm_pct([r["rel_bias"] for r in records], centre=0),
        "mape":     _safe_norm_pct([r["mape"]     for r in records]),
        "pipe_cov": _safe_norm_pct([r["pipe_cov"] for r in records]),
        "r2":       mcolors.Normalize(vmin=0, vmax=1),
    }

    def _abbrev_species(name: str) -> str:
        parts = name.split()
        if len(parts) >= 2:
            return f"{parts[0][0]}. {' '.join(parts[1:])}"
        return name

    panel_cfg = [
        (axes[0, 0], "genus",      "Majority BK genus",          None,            None,                                  None,     True),
        (axes[0, 1], "rel_bias",   "Relative height bias (%)",   plt.cm.RdBu_r,   None,                                  "{:+.0f}%", False),
        (axes[0, 2], "mape",       "Height MAPE (%)",            plt.cm.YlOrRd,   None,                                  "{:.0f}%",  False),
        (axes[1, 0], "match_rate", "BK match rate (%)",          plt.cm.YlGn,     mcolors.Normalize(vmin=0, vmax=100),   "{:.0f}%",  False),
        (axes[1, 1], "pipe_cov",   "Pipeline canopy cover (%)",  plt.cm.YlGn,     None,                                  "{:.0f}%",  False),
        (axes[1, 2], "r2",         "Allometric R² (per tile)",   plt.cm.viridis,  mcolors.Normalize(vmin=0, vmax=1),     "{:.2f}",   False),
    ]

    for ax, key, title, cmap, norm, fmt, use_genus in panel_cfg:
        ax.imshow(full_img, extent=extent, origin="upper", aspect="equal")
        if use_genus:
            legend_patches = []
        else:
            actual_norm = norm if norm is not None else dynamic_norms.get(key)

        for r in records:
            val = r[key]
            gdf_t = gpd.GeoDataFrame([{"geometry": r["geometry"]}], crs="EPSG:25832")
            if use_genus:
                color = genus_color[val]
                gdf_t.plot(ax=ax, facecolor=color, edgecolor="white", linewidth=1, alpha=0.55, zorder=2)
                lbl = _abbrev_species(val) if val != "—" else "—"
            elif val is None:
                gdf_t.plot(ax=ax, facecolor="#888888", edgecolor="white", linewidth=1, alpha=0.5, zorder=2)
                lbl = "—"
            else:
                color = cmap(actual_norm(val))
                gdf_t.plot(ax=ax, facecolor=color, edgecolor="white", linewidth=1, alpha=0.6, zorder=2)
                if key == "pipe_cov":
                    lbl = f"{val:.0f}%\n(BK {r['bk_cov']:.0f}%)"
                elif key == "match_rate":
                    lbl = f"{r['n_matched']} / {r['n_pipe']}\n{val:.0f}%"
                else:
                    lbl = fmt.format(val)

            cx, cy = r["geometry"].centroid.x, r["geometry"].centroid.y
            ax.text(cx, cy, lbl, ha="center", va="center", fontsize=7, fontweight="bold",
                    bbox=dict(facecolor="white", alpha=0.65, edgecolor="none", pad=1.5), zorder=3)

        ax.set_xlim(west_m, east_m); ax.set_ylim(south_m, north_m)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Easting (m)", fontsize=8); ax.set_ylabel("Northing (m)", fontsize=8)
        ax.tick_params(labelsize=7)

        if use_genus:
            handles = [mpatches.Patch(color=genus_color[g], label=g) for g in all_genera]
            ax.legend(handles=handles, fontsize=6, loc="lower right", framealpha=0.85, ncol=2)
        else:
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=actual_norm)
            sm.set_array([])
            plt.colorbar(sm, ax=ax, shrink=0.65, pad=0.02)

    plt.tight_layout(rect=[0, 0.05, 1, 1])

    footers = [
        (axes[0, 1], f"Area rel. bias = {total_rel_bias:+.1f} %"  if total_rel_bias is not None else "—"),
        (axes[0, 2], f"Area MAPE = {total_mape:.1f} %"             if total_mape     is not None else "—"),
        (axes[1, 0], f"Total: {total_matched} matched / {total_pipe} segmented ({match_pct_total:.0f}%)"),
        (axes[1, 1], f"Pipeline {total_pipe_cov:.1f} % / BK {total_bk_cov:.1f} % canopy cover"),
        (axes[1, 2], f"Mean R² = {total_r2:.2f}  |  mean height = {total_mean_h:.1f} m"
                     if total_r2 is not None else "—"),
    ]
    for ax_f, txt in footers:
        ax_f.text(0.5, -0.08, txt, transform=ax_f.transAxes,
                  ha="center", va="top", fontsize=9)

    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  tile summary (pct) → {out_path.name}")


def _save_watershed_comparison(
    area_name: str,
    seg_dir,
    tile_size_m: int,
    vegetation_model: str,
    max_crown_radius_m: float = MAX_CROWN_RADIUS_M,
) -> None:
    """Compare crown polygon counts and area distributions pre vs post watershed.

    Loads each per-tile .npy segmentation map, runs vectorize_trees twice
    (apply_watershed=False and True), and saves a 2-panel PNG:
      Left  — log-scale crown area histogram, pre (blue) vs post (orange)
      Right — per-tile tree-count grouped bar chart with Δn annotations
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path as _Path
    from src.data_preprocessing import tiles_for_area
    from src.shadow.casting import vectorize_trees

    out_path = _Path(seg_dir) / f"watershed_comparison_{tile_size_m}m.png"
    tiles = tiles_for_area(AREAS[area_name], tile_size_m)

    rows = []
    all_pre, all_post = [], []

    for t in tiles:
        seg_path = (
            _Path(seg_dir)
            / f"{area_name}_tile_{t['ix']}_{t['iy']}_{vegetation_model}_seg.npy"
        )
        if not seg_path.exists():
            continue
        seg_map   = np.load(seg_path)
        tree_mask = seg_map == 1
        gdf_pre  = vectorize_trees(tree_mask, t, vegetation_model,
                                   apply_watershed=False,
                                   max_crown_radius_m=max_crown_radius_m)
        gdf_post = vectorize_trees(tree_mask, t, vegetation_model,
                                   apply_watershed=True,
                                   max_crown_radius_m=max_crown_radius_m)
        n_pre, n_post = len(gdf_pre), len(gdf_post)
        n_oversized = int(
            (gdf_pre["crown_area_m2"] > np.pi * max_crown_radius_m ** 2).sum()
        )
        rows.append({
            "tile":       f"({t['ix']},{t['iy']})",
            "n_pre":      n_pre,
            "n_post":     n_post,
            "n_oversized": n_oversized,
            "mean_pre":   float(gdf_pre["crown_area_m2"].mean()) if n_pre else 0.0,
            "mean_post":  float(gdf_post["crown_area_m2"].mean()) if n_post else 0.0,
        })
        all_pre.extend(gdf_pre["crown_area_m2"].tolist())
        all_post.extend(gdf_post["crown_area_m2"].tolist())

    if not rows:
        print(f"  [watershed_comparison] no .npy tiles found, skipping")
        return

    total_pre  = sum(r["n_pre"]  for r in rows)
    total_post = sum(r["n_post"] for r in rows)
    total_split = sum(r["n_oversized"] for r in rows)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # --- Panel 1: crown area histogram (log scale) ---
    ax = axes[0]
    bins = np.logspace(
        np.log10(max(min(all_pre + all_post), 1)),
        np.log10(max(all_pre + all_post) * 1.05),
        40,
    )
    ax.hist(all_pre,  bins=bins, alpha=0.55, color="steelblue",  label=f"Pre-watershed  (n={total_pre})")
    ax.hist(all_post, bins=bins, alpha=0.55, color="darkorange", label=f"Post-watershed (n={total_post})")
    ax.set_xscale("log")
    ax.set_xlabel("Crown area (m²)")
    ax.set_ylabel("Polygon count")
    ax.set_title(f"Crown area distribution — {tile_size_m} m tiles")
    ax.axvline(np.pi * max_crown_radius_m ** 2, color="red", linestyle="--", linewidth=1,
               label=f"Watershed threshold ({max_crown_radius_m:.0f} m radius)")
    ax.legend(fontsize=8)

    # --- Panel 2: per-tile count grouped bars ---
    ax2 = axes[1]
    tile_labels = [r["tile"] for r in rows]
    n_tiles = len(rows)
    x = np.arange(n_tiles)
    w = 0.35
    b_pre  = ax2.bar(x - w / 2, [r["n_pre"]  for r in rows], w, color="steelblue",  alpha=0.8, label="Pre")
    b_post = ax2.bar(x + w / 2, [r["n_post"] for r in rows], w, color="darkorange", alpha=0.8, label="Post")
    for rect_pre, rect_post, r in zip(b_pre, b_post, rows):
        delta = r["n_post"] - r["n_pre"]
        if delta != 0:
            ax2.text(
                rect_post.get_x() + rect_post.get_width() / 2,
                rect_post.get_height() + 1,
                f"+{delta}" if delta > 0 else str(delta),
                ha="center", va="bottom", fontsize=8,
                color="seagreen" if delta > 0 else "crimson",
            )
    ax2.set_xticks(x)
    ax2.set_xticklabels(tile_labels, rotation=45, ha="right", fontsize=8)
    ax2.set_ylabel("Tree polygon count")
    ax2.set_title(f"Per-tile polygon count — pre vs post watershed")
    ax2.legend(fontsize=8)

    plt.suptitle(
        f"Watershed comparison — {area_name} @ {tile_size_m} m  "
        f"[pre: {total_pre}, post: {total_post}, "
        f"oversized blobs split: {total_split}]",
        y=1.01,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  watershed comparison → {out_path.name}")

    # Summary table
    print(f"\n  {'Tile':>8}  {'Pre':>5}  {'Post':>5}  {'Δ':>4}  "
          f"{'Oversized':>10}  {'Mean area pre':>14}  {'Mean area post':>15}")
    print(f"  {'-'*70}")
    for r in rows:
        print(f"  {r['tile']:>8}  {r['n_pre']:>5}  {r['n_post']:>5}  "
              f"{r['n_post']-r['n_pre']:>+4}  {r['n_oversized']:>10}  "
              f"{r['mean_pre']:>14.1f}  {r['mean_post']:>15.1f}")
    print(f"  {'-'*70}")
    print(f"  {'TOTAL':>8}  {total_pre:>5}  {total_post:>5}  "
          f"{total_post-total_pre:>+4}  {total_split:>10}")


def _save_location_map(area_name: str, seg_dir, merged_gdf, bk_gdf, tile_size_m: int) -> None:
    """Save a 2-panel location map: BK crown circles (left) and pipeline trees (right).

    Both panels share an orthophoto background and tile-grid overlay.
    Saved as location_map_{tile_size_m}m.png in seg_dir.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from pathlib import Path as _Path
    from PIL import Image as _PILImage
    from pyproj import Transformer as _Transformer
    import geopandas as gpd
    from shapely.geometry import box as _box
    from src.data_preprocessing import tiles_for_area

    out_path = _Path(seg_dir) / f"location_map_{tile_size_m}m.png"
    full_img_path = OUTPUT_DIR / area_name / f"{area_name}_full.png"
    if not full_img_path.exists():
        print(f"  [location_map] orthophoto not found, skipping: {full_img_path}")
        return

    _to_utm = _Transformer.from_crs("EPSG:4326", "EPSG:25832", always_xy=True)
    full_img = np.array(_PILImage.open(full_img_path).convert("RGB"))
    ovgu = AREAS[area_name]
    west_m, south_m = _to_utm.transform(ovgu["west"], ovgu["south"])
    east_m, north_m = _to_utm.transform(ovgu["east"], ovgu["north"])
    extent = [west_m, east_m, south_m, north_m]
    area_bbox = gpd.GeoDataFrame(
        geometry=[_box(west_m, south_m, east_m, north_m)], crs="EPSG:25832"
    )

    # Tile grid boundaries for overlay
    tiles = tiles_for_area(AREAS[area_name], tile_size_m)
    tile_boundaries = [
        gpd.GeoDataFrame(
            geometry=[_box(
                *_to_utm.transform(t["west"], t["south"]),
                *_to_utm.transform(t["east"], t["north"])
            )],
            crs="EPSG:25832",
        )
        for t in tiles
    ]

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle(
        f"{area_name}  |  {tile_size_m} m tiles  |  location map",
        fontsize=13,
    )

    # ── Left panel: BK crown circles ──────────────────────────────────────────
    ax = axes[0]
    ax.imshow(full_img, extent=extent, origin="upper", aspect="equal")

    if bk_gdf is not None:
        bk_clip = bk_gdf.clip(area_bbox).reset_index(drop=True)
        bk_no_crown = bk_clip[bk_clip["Kronendurchmesser"] <= 0]
        bk_valid    = bk_clip[bk_clip["Kronendurchmesser"] > 0].copy()

        if len(bk_no_crown) > 0:
            bk_no_crown.plot(ax=ax, color="#888888", markersize=2, alpha=0.5, zorder=2)

        if len(bk_valid) > 0:
            bk_circles = bk_valid.copy()
            bk_circles["geometry"] = bk_circles.apply(
                lambda r: r.geometry.buffer(r["Kronendurchmesser"] / 2), axis=1
            )
            cd = bk_valid["Kronendurchmesser"]
            vmin_cd = float(cd.quantile(0.05))
            vmax_cd = float(cd.quantile(0.95))
            norm_cd = mcolors.Normalize(vmin=vmin_cd, vmax=vmax_cd)
            colors_cd = plt.cm.YlGn(norm_cd(cd.values))
            bk_circles.plot(ax=ax, facecolor=colors_cd, edgecolor="none", alpha=0.7, zorder=3)
            sm = plt.cm.ScalarMappable(cmap=plt.cm.YlGn, norm=norm_cd)
            sm.set_array([])
            plt.colorbar(sm, ax=ax, label="Crown diameter (m)", shrink=0.65)
            ax.set_title(
                f"BK registered trees  (n={len(bk_valid)} with crown, "
                f"{len(bk_no_crown)} without)",
                fontsize=10,
            )
    else:
        ax.set_title("BK data not available", fontsize=10)

    for tb in tile_boundaries:
        tb.boundary.plot(ax=ax, color="white", linewidth=0.8, alpha=0.6, zorder=4)
    ax.set_xlim(west_m, east_m); ax.set_ylim(south_m, north_m)
    ax.set_xlabel("Easting (m)", fontsize=8); ax.set_ylabel("Northing (m)", fontsize=8)
    ax.tick_params(labelsize=7)

    # ── Right panel: pipeline tree polygons ───────────────────────────────────
    ax = axes[1]
    ax.imshow(full_img, extent=extent, origin="upper", aspect="equal")

    if len(merged_gdf) > 0 and "height_m" in merged_gdf.columns:
        h = merged_gdf["height_m"]
        vmin_h = float(h.quantile(0.05))
        vmax_h = float(h.quantile(0.95))
        norm_h = mcolors.Normalize(vmin=vmin_h, vmax=max(vmax_h, vmin_h + 0.1))
        colors_h = plt.cm.viridis(norm_h(h.values))
        merged_gdf.plot(ax=ax, facecolor=colors_h, edgecolor="none", alpha=0.7, zorder=3)
        sm2 = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=norm_h)
        sm2.set_array([])
        plt.colorbar(sm2, ax=ax, label="Tree height (m)", shrink=0.65)

    for tb in tile_boundaries:
        tb.boundary.plot(ax=ax, color="white", linewidth=0.8, alpha=0.6, zorder=4)
    ax.set_title(f"Pipeline segmented trees  (n={len(merged_gdf)})", fontsize=10)
    ax.set_xlim(west_m, east_m); ax.set_ylim(south_m, north_m)
    ax.set_xlabel("Easting (m)", fontsize=8); ax.set_ylabel("Northing (m)", fontsize=8)
    ax.tick_params(labelsize=7)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  location map  → {out_path.name}")


def cmd_segment(vegetation_model: str = DEFAULT_VEGETATION_MODEL, tile_size_m: int | None = None, all_sizes: bool = False) -> None:
    import geopandas as gpd
    from shapely.geometry import box as shapely_box
    from pyproj import Transformer
    from src.shadow.casting import vectorize_trees
    from src.shadow.cadastre import enrich_from_baumkataster, tile_dominant_genus
    from src.config import BAUMKATASTER_PATH, ALLOMETRIC_PROFILES, CROWN_RADIUS_BY_GENUS

    sizes = TILE_SIZES_M if all_sizes else [tile_size_m or TILE_SIZE_M]
    _, mask_fn = _load_vegetation_model(vegetation_model)
    _to_utm = Transformer.from_crs("EPSG:4326", "EPSG:25832", always_xy=True)

    bk_gdf = None
    if BAUMKATASTER_PATH and BAUMKATASTER_PATH.exists():
        bk_gdf = gpd.read_file(BAUMKATASTER_PATH)
        print(f"  Baumkataster: {len(bk_gdf):,} trees loaded from {BAUMKATASTER_PATH.name}")

    for area_name, area in AREAS.items():
        buildings = fetch_buildings(area, cache_path=OUTPUT_DIR / f"buildings_{area_name}.fgb")
        roads = fetch_roads(area, cache_path=OUTPUT_DIR / f"roads_{area_name}.fgb")
        print(f"  OSM: {len(buildings)} buildings, {len(roads)} roads")

        for size in sizes:
            tiles = tiles_for_area(area, size)
            tile_dir = OUTPUT_DIR / area_name / f"{size}m"
            seg_dir = OUTPUT_DIR / "segments" / f"{size}m"
            print(f"\n--- {area_name} @ {size}m: segmenting {len(tiles)} tile(s) [{vegetation_model}] ---")

            size_t0 = time.perf_counter()
            processed = 0
            for t in tiles:
                base_stem = f"{area_name}_tile_{t['ix']}_{t['iy']}"
                tile_path = tile_dir / f"{base_stem}.png"
                if not tile_path.exists():
                    print(f"  [skip] {base_stem}.png not found — run 'download' first")
                    continue

                tile_t0 = time.perf_counter()
                img = np.array(Image.open(tile_path).convert("RGB"))
                tree_mask = mask_fn(img)
                stem = f"{base_stem}_{vegetation_model}"

                # Baumkataster enrichment: tile clip → dominant genus → species allometry
                bk_tile = None
                genus = None
                max_r = MAX_CROWN_RADIUS_M
                if bk_gdf is not None:
                    w_m, s_m = _to_utm.transform(t["west"], t["south"])
                    e_m, n_m = _to_utm.transform(t["east"], t["north"])
                    tile_poly = gpd.GeoDataFrame(
                        geometry=[shapely_box(w_m, s_m, e_m, n_m)], crs="EPSG:25832"
                    )
                    bk_tile = bk_gdf.clip(tile_poly)
                    genus = tile_dominant_genus(bk_tile)
                    max_r = CROWN_RADIUS_BY_GENUS.get(genus, MAX_CROWN_RADIUS_M) if genus else MAX_CROWN_RADIUS_M

                npy_path, png_path = save_segmentation(
                    img, t, buildings, roads, tree_mask,
                    out_dir=seg_dir, stem=stem,
                )
                tree_gdf = vectorize_trees(
                    tree_mask, t, vegetation_model,
                    dominant_genus=genus, max_crown_radius_m=max_r,
                )
                if bk_tile is not None:
                    tree_gdf = enrich_from_baumkataster(tree_gdf, bk_tile)
                fgb_path = seg_dir / f"{stem}_trees.fgb"
                tree_gdf.to_file(fgb_path, driver="FlatGeobuf")
                tile_elapsed = time.perf_counter() - tile_t0
                processed += 1
                genus_tag = f" [{genus}]" if genus else ""
                n_matched = (tree_gdf["height_source"] == "measured").sum() if "height_source" in tree_gdf.columns else 0
                print(f"  {base_stem}: {len(tree_gdf)} tree(s){genus_tag}, {n_matched} BK-matched — {tile_elapsed:.1f}s")

            if processed:
                size_elapsed = time.perf_counter() - size_t0
                print(f"  Timing [{size}m]: {processed} tile(s) in {size_elapsed:.1f}s — avg {size_elapsed/processed:.1f}s/tile")

            merged_path = _merge_layer(area_name, vegetation_model, "trees", seg_dir)
            if merged_path:
                print(f"  merged → {merged_path.name}")
                merged = gpd.read_file(merged_path)
                if "allometric_height_m" in merged.columns and "height_source" in merged.columns:
                    m = merged[merged["height_source"] == "measured"]
                    if len(m) > 0:
                        err = m["allometric_height_m"] - m["height_m"]
                        bias = err.mean()
                        rmse = (err ** 2).mean() ** 0.5
                        mae  = err.abs().mean()
                        print(f"  height validation (n={len(m)} BK-matched): "
                              f"bias={bias:+.1f} m  MAE={mae:.1f} m  RMSE={rmse:.1f} m")
                records = _save_tile_summary(area_name, seg_dir, merged, bk_gdf, size)
                _save_tile_summary_pct(area_name, seg_dir, records, size)
                _save_watershed_comparison(area_name, seg_dir, size, vegetation_model)
                _save_location_map(area_name, seg_dir, merged, bk_gdf, size)


def cmd_compare(area_filter: str | None = None) -> None:
    seg_dir = OUTPUT_DIR / "segments"
    areas = {k: v for k, v in AREAS.items() if area_filter is None or k == area_filter}
    if not areas:
        print(f"No area named {area_filter!r}. Available: {list(AREAS)}")
        return

    # Load all models once up front
    print("Loading models...")
    models = {}
    for name in VEGETATION_MODELS:
        if name == "vari":
            continue
        loaded, _ = _load_vegetation_model(name)
        # sam_prompted needs its components split out for compare_vegetation
        models[name] = loaded
    print("  All models ready.\n")

    all_metrics = []

    for area_name, _ in areas.items():
        tiles = tiles_for_area(AREAS[area_name], TILE_SIZE_M)
        tile_dir = OUTPUT_DIR / area_name
        print(f"--- {area_name}: comparing {len(tiles)} tile(s) ---")

        for t in tiles:
            stem = f"{area_name}_tile_{t['ix']}_{t['iy']}"
            tile_path = tile_dir / f"{stem}.png"
            if not tile_path.exists():
                print(f"  [skip] {stem}.png not found — run 'download' first")
                continue

            img = np.array(Image.open(tile_path).convert("RGB"))
            result = compare_vegetation(
                img,
                methods=VEGETATION_MODELS,
                out_dir=seg_dir,
                stem=stem,
                models=models,
            )
            metrics = result["metrics"].reset_index()
            metrics.insert(0, "tile", stem)
            all_metrics.append(metrics)
            print(result["metrics"].to_string())
            print()

    if all_metrics:
        import pandas as pd
        combined = pd.concat(all_metrics, ignore_index=True)
        csv_path = seg_dir / "comparison_metrics.csv"
        combined.to_csv(csv_path, index=False)
        print(f"Metrics saved to {csv_path}")


def cmd_compare_sizes(
    area_filter: str | None = None,
    vegetation_model: str = DEFAULT_VEGETATION_MODEL,
    resolution_m: float = 1.0,
    reference_size_m: int = 250,
) -> None:
    """Rasterize tree polygons from each tile size onto a shared UTM grid and compute pairwise IoU."""
    import geopandas as gpd
    import pandas as pd
    from pyproj import Transformer
    from rasterio.features import rasterize as rio_rasterize
    from rasterio.transform import from_bounds

    _to_utm = Transformer.from_crs("EPSG:4326", "EPSG:25832", always_xy=True)
    areas = {k: v for k, v in AREAS.items() if area_filter is None or k == area_filter}
    if not areas:
        print(f"No area named {area_filter!r}. Available: {list(AREAS)}")
        return

    for area_name, area in areas.items():
        west_m, south_m = _to_utm.transform(area["west"], area["south"])
        east_m, north_m = _to_utm.transform(area["east"], area["north"])
        width = int(round((east_m - west_m) / resolution_m))
        height = int(round((north_m - south_m) / resolution_m))
        transform = from_bounds(west_m, south_m, east_m, north_m, width, height)

        print(f"\n{area_name} — common grid {width}×{height} px @ {resolution_m:.1f} m/px  [{vegetation_model}]")
        print(f"  {'Size':>6}  {'Trees':>7}  {'Coverage':>10}")
        print(f"  {'-'*29}")

        masks: dict[int, np.ndarray] = {}
        tree_counts: dict[int, int] = {}
        for size in TILE_SIZES_M:
            seg_dir = OUTPUT_DIR / "segments" / f"{size}m"
            fgb_files = sorted(seg_dir.glob(f"{area_name}_*_{vegetation_model}_trees.fgb")) if seg_dir.exists() else []
            if not fgb_files:
                print(f"  {size:>4}m  [no data — run 'segment' first]")
                continue

            gdfs = [gpd.read_file(f) for f in fgb_files]
            gdf = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs="EPSG:25832")
            shapes = [(geom, 1) for geom in gdf.geometry if geom is not None and not geom.is_empty]

            if shapes:
                raster = rio_rasterize(shapes, out_shape=(height, width),
                                       transform=transform, fill=0, dtype="uint8")
                masks[size] = raster.astype(bool)
            else:
                masks[size] = np.zeros((height, width), dtype=bool)

            tree_counts[size] = len(gdf)
            print(f"  {size:>4}m  {len(gdf):>7}  {masks[size].mean()*100:>9.2f}%")

        if len(masks) < 2:
            print("  [skip] need ≥2 sizes with data to compare")
            continue

        sizes = sorted(masks)
        col_w = 7

        # Pairwise IoU table
        print(f"\n  Pairwise IoU:")
        print(f"  {'':>6}" + "".join(f"  {s:>{col_w}}m" for s in sizes))
        records = []
        for s1 in sizes:
            row = f"  {s1:>4}m "
            rec: dict = {"size_m": s1, "trees": tree_counts.get(s1, 0),
                         "coverage_pct": round(float(masks[s1].mean() * 100), 3)}
            for s2 in sizes:
                inter = int((masks[s1] & masks[s2]).sum())
                union = int((masks[s1] | masks[s2]).sum())
                iou = inter / union if union > 0 else 1.0
                row += f"  {iou:>{col_w}.3f}"
                rec[f"iou_{s2}m"] = round(iou, 4)
            print(row)
            records.append(rec)

        # Precision / recall vs reference size
        ref = reference_size_m if reference_size_m in masks else max(sizes)
        ref_mask = masks[ref]
        print(f"\n  Precision / Recall vs {ref}m reference:")
        print(f"  {'Size':>6}  {'Prec':>8}  {'Rec':>8}  {'F1':>8}")
        for s in sizes:
            if s == ref:
                print(f"  {s:>4}m  {'(ref)':>8}")
                continue
            tp = int((masks[s] & ref_mask).sum())
            fp = int((masks[s] & ~ref_mask).sum())
            fn = int((~masks[s] & ref_mask).sum())
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec_val = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * prec * rec_val / (prec + rec_val) if (prec + rec_val) > 0 else 0.0
            print(f"  {s:>4}m  {prec:>8.3f}  {rec_val:>8.3f}  {f1:>8.3f}")
            for r in records:
                if r["size_m"] == s:
                    r[f"prec_vs_{ref}m"] = round(prec, 4)
                    r[f"rec_vs_{ref}m"] = round(rec_val, 4)
                    r[f"f1_vs_{ref}m"] = round(f1, 4)

        # Save CSV
        out_csv = OUTPUT_DIR / "segments" / f"{area_name}_{vegetation_model}_size_comparison.csv"
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(records).to_csv(out_csv, index=False)
        print(f"\n  Saved → {out_csv}")


def cmd_shadow(
    area_filter: str | None = None,
    vegetation_model: str = DEFAULT_VEGETATION_MODEL,
    datetime_utc: str | None = None,
    tile_size_m: int | None = None,
    all_sizes: bool = False,
) -> None:
    import datetime as dt
    import geopandas as gpd
    from src.shadow import cast_tree_shadows, save_shadow_overlay, vectorize_shadows

    if datetime_utc is None:
        when = dt.datetime.now(tz=dt.timezone.utc)
    else:
        when = dt.datetime.fromisoformat(datetime_utc).replace(tzinfo=dt.timezone.utc)

    sizes = TILE_SIZES_M if all_sizes else [tile_size_m or TILE_SIZE_M]
    areas = {k: v for k, v in AREAS.items() if area_filter is None or k == area_filter}
    if not areas:
        print(f"No area named {area_filter!r}. Available: {list(AREAS)}")
        return

    print(f"Shadow casting for {when.strftime('%Y-%m-%d %H:%M UTC')}")

    for area_name, _ in areas.items():
        for size in sizes:
            tiles = tiles_for_area(AREAS[area_name], size)
            shadow_dir = OUTPUT_DIR / "shadows" / f"{size}m"
            seg_dir = OUTPUT_DIR / "segments" / f"{size}m"
            print(f"\n--- {area_name} @ {size}m: {len(tiles)} tile(s) ---")

            size_t0 = time.perf_counter()
            processed = 0
            for t in tiles:
                stem = f"{area_name}_tile_{t['ix']}_{t['iy']}"
                seg_path = seg_dir / f"{stem}_{vegetation_model}_seg.npy"
                img_path = OUTPUT_DIR / area_name / f"{size}m" / f"{stem}.png"

                if not seg_path.exists():
                    print(f"  [skip] {seg_path.name} not found — run 'segment' first")
                    continue
                if not img_path.exists():
                    print(f"  [skip] {img_path.name} not found — run 'download' first")
                    continue

                tile_t0 = time.perf_counter()
                seg_map = np.load(seg_path)
                img = np.array(Image.open(img_path).convert("RGB"))

                fgb_path = seg_dir / f"{stem}_{vegetation_model}_trees.fgb"
                tree_gdf_shadow = gpd.read_file(fgb_path) if fgb_path.exists() else None
                shadow_mask = cast_tree_shadows(seg_map, t, when, tree_gdf=tree_gdf_shadow)
                coverage_pct = shadow_mask.mean() * 100

                out_path = shadow_dir / f"{stem}_{vegetation_model}_shadow.png"
                save_shadow_overlay(
                    img, seg_map, shadow_mask, out_path,
                    title=f"{stem} — {when.strftime('%Y-%m-%d %H:%M UTC')}",
                )

                shadow_gdf = vectorize_shadows(shadow_mask, t, when, vegetation_model)
                fgb_out = shadow_dir / f"{stem}_{vegetation_model}_shadow.fgb"
                shadow_gdf.to_file(fgb_out, driver="FlatGeobuf")
                tile_elapsed = time.perf_counter() - tile_t0
                processed += 1
                print(f"  {stem}: shadow={coverage_pct:.1f}% — {tile_elapsed:.1f}s")

            if processed:
                size_elapsed = time.perf_counter() - size_t0
                print(f"  Timing [{size}m]: {processed} tile(s) in {size_elapsed:.1f}s — avg {size_elapsed/processed:.1f}s/tile")

            merged_path = _merge_layer(area_name, vegetation_model, "shadow", shadow_dir)
            if merged_path:
                print(f"  merged → {merged_path.name}")


def cmd_tune(
    model: str = "vari",
    tile_key: str = "0_0",
    area_filter: str | None = None,
    reference_model: str = "deepforest",
) -> None:
    from src.segmentation.tuning import tune_vari, tune_deepforest, tune_samgeo

    if model not in TUNABLE_MODELS:
        print(f"No tuning available for {model!r}. Tunable models: {TUNABLE_MODELS}")
        return

    seg_dir = OUTPUT_DIR / "segments"
    areas = {k: v for k, v in AREAS.items() if area_filter is None or k == area_filter}
    if not areas:
        print(f"No area named {area_filter!r}. Available: {list(AREAS)}")
        return

    try:
        ix, iy = (int(x) for x in tile_key.split("_"))
    except ValueError:
        print(f"Invalid tile key {tile_key!r}. Expected format: 'ix_iy', e.g. '0_0'")
        return

    for area_name in areas:
        tile_path = OUTPUT_DIR / area_name / f"{area_name}_tile_{ix}_{iy}.png"
        if not tile_path.exists():
            print(f"[skip] {tile_path.name} not found — run 'download' first")
            continue

        img = np.array(Image.open(tile_path).convert("RGB"))
        print(f"\n--- {area_name} tile {ix}_{iy}: tuning {model!r} vs reference {reference_model!r} ---")

        print(f"  Loading reference model ({reference_model})...")
        _, ref_fn = _load_vegetation_model(reference_model)
        ref_mask = ref_fn(img)
        print(f"  Reference coverage: {ref_mask.mean()*100:.1f}%")

        stem = f"{area_name}_tile_{ix}_{iy}_{model}"
        if model == "vari":
            df = tune_vari(img, ref_mask, out_dir=seg_dir, stem=stem)
        elif model == "deepforest":
            from src.segmentation import load_deepforest
            m = load_deepforest()
            df = tune_deepforest(img, ref_mask, model=m, out_dir=seg_dir, stem=stem)
        elif model == "samgeo":
            from src.segmentation import load_samgeo
            m = load_samgeo()
            df = tune_samgeo(img, ref_mask, model=m, out_dir=seg_dir, stem=stem)

        print(f"\nTop 5 parameter combinations:")
        print(df.head(5).to_string(index=False))


def cmd_status(
    area_filter: str | None = None,
    vegetation_model: str = DEFAULT_VEGETATION_MODEL,
    tile_size_m: int | None = None,
    all_sizes: bool = False,
) -> None:
    """Print what has been computed for each area and tile size."""
    sizes = TILE_SIZES_M if all_sizes else [tile_size_m or TILE_SIZE_M]
    areas = {k: v for k, v in AREAS.items() if area_filter is None or k == area_filter}
    if not areas:
        print(f"No area named {area_filter!r}. Available: {list(AREAS)}")
        return

    for area_name in areas:
        full_img = OUTPUT_DIR / area_name / f"{area_name}_full.png"
        print(f"\n{area_name}  (full image: {'✓' if full_img.exists() else '✗'})")
        print(f"  {'Size':>6}  {'Tiles':>9}  {'Segments':>10}  {'Shadows':>9}")
        print(f"  {'-'*42}")
        for size in sizes:
            tiles = tiles_for_area(AREAS[area_name], size)
            total = len(tiles)
            tile_dir = OUTPUT_DIR / area_name / f"{size}m"
            seg_dir = OUTPUT_DIR / "segments" / f"{size}m"
            shadow_dir = OUTPUT_DIR / "shadows" / f"{size}m"
            n_tiles = sum(
                1 for t in tiles
                if (tile_dir / f"{area_name}_tile_{t['ix']}_{t['iy']}.png").exists()
            )
            n_seg = len(list(seg_dir.glob(f"{area_name}_*_{vegetation_model}_seg.npy"))) if seg_dir.exists() else 0
            n_shadow = len(list(shadow_dir.glob(f"{area_name}_*_{vegetation_model}_shadow.png"))) if shadow_dir.exists() else 0
            print(f"  {size:>4}m  {n_tiles:>4}/{total:<4}  {n_seg:>4}/{total:<4}   {n_shadow:>4}/{total}")


def cmd_diurnal(
    area_filter: str | None = None,
    vegetation_model: str = DEFAULT_VEGETATION_MODEL,
    date_utc: str | None = None,
    tile_key: str = "0_0",
    tile_size_m: int | None = None,
) -> None:
    """Print hourly shadow coverage for one tile across a full day and save shadow overlays."""
    import datetime as dt
    import geopandas as gpd
    from src.shadow import cast_tree_shadows, save_shadow_overlay
    from src.shadow.solar import sun_position, _tile_center

    date = dt.date.fromisoformat(date_utc) if date_utc else dt.date.today()
    size = tile_size_m or TILE_SIZE_M
    areas = {k: v for k, v in AREAS.items() if area_filter is None or k == area_filter}
    if not areas:
        print(f"No area named {area_filter!r}. Available: {list(AREAS)}")
        return

    try:
        ix, iy = (int(x) for x in tile_key.split("_"))
    except ValueError:
        print(f"Invalid tile key {tile_key!r}. Expected format: 'ix_iy', e.g. '0_0'")
        return

    for area_name in areas:
        tiles = tiles_for_area(AREAS[area_name], size)
        tile = next((t for t in tiles if t["ix"] == ix and t["iy"] == iy), None)
        if tile is None:
            print(f"  [skip] tile ({ix},{iy}) not in {area_name} @ {size}m grid")
            continue

        stem = f"{area_name}_tile_{ix}_{iy}"
        seg_dir = OUTPUT_DIR / "segments" / f"{size}m"
        seg_path = seg_dir / f"{stem}_{vegetation_model}_seg.npy"
        fgb_path = seg_dir / f"{stem}_{vegetation_model}_trees.fgb"
        img_path = OUTPUT_DIR / area_name / f"{size}m" / f"{stem}.png"

        if not seg_path.exists():
            print(f"  [skip] {seg_path.name} not found — run 'segment' first")
            continue
        if not img_path.exists():
            print(f"  [skip] {img_path.name} not found — run 'download' first")
            continue

        seg_map = np.load(seg_path)
        img = np.array(Image.open(img_path).convert("RGB"))
        tree_gdf = gpd.read_file(fgb_path) if fgb_path.exists() else None
        lat, lon = _tile_center(tile)

        diurnal_dir = OUTPUT_DIR / "shadows" / f"{size}m" / "diurnal"

        print(f"\n{area_name} tile ({ix},{iy}) @ {size}m — {date}")
        print(f"  {'Hour UTC':>8}  {'Elev (°)':>8}  {'Az (°)':>8}  {'Shadow %':>9}")
        print(f"  {'-'*42}")

        for hour in range(4, 21):
            when = dt.datetime(date.year, date.month, date.day, hour, 0, 0, tzinfo=dt.timezone.utc)
            az, el = sun_position(lat, lon, when)
            if el < 5.0:
                continue
            shadow_mask = cast_tree_shadows(seg_map, tile, when, tree_gdf=tree_gdf)
            pct = shadow_mask.mean() * 100
            print(f"  {hour:>7}:00  {el:>8.1f}  {az:>8.1f}  {pct:>9.2f}%")
            out_path = diurnal_dir / f"{stem}_{vegetation_model}_{hour:02d}h_shadow.png"
            save_shadow_overlay(img, seg_map, shadow_mask, out_path,
                                title=f"{stem} — {when.strftime('%Y-%m-%d %H:%M UTC')}")


def cmd_merge(
    area_filter: str | None = None,
    vegetation_model: str = DEFAULT_VEGETATION_MODEL,
    tile_size_m: int | None = None,
    all_sizes: bool = False,
    layers: tuple[str, ...] = ("trees", "shadow"),
) -> None:
    """Merge per-tile tree and shadow FGB files into one full-area FGB per size."""
    sizes = TILE_SIZES_M if all_sizes else [tile_size_m or TILE_SIZE_M]
    areas = {k: v for k, v in AREAS.items() if area_filter is None or k == area_filter}
    if not areas:
        print(f"No area named {area_filter!r}. Available: {list(AREAS)}")
        return

    for area_name in areas:
        for size in sizes:
            dirs = {
                "trees": OUTPUT_DIR / "segments" / f"{size}m",
                "shadow": OUTPUT_DIR / "shadows" / f"{size}m",
            }
            for layer in layers:
                out_dir = dirs[layer]
                merged_path = _merge_layer(area_name, vegetation_model, layer, out_dir)
                if merged_path:
                    import geopandas as gpd
                    n = len(gpd.read_file(merged_path))
                    print(f"  {area_name} @ {size}m [{layer}]: {n} features → {merged_path.name}")
                else:
                    n_files = len(list(dirs[layer].glob(f"{area_name}_tile_*_{vegetation_model}_{layer}.fgb"))) if dirs[layer].exists() else 0
                    reason = "all tile FGBs are empty" if n_files > 0 else "no tile FGBs found — run segment/shadow first"
                    print(f"  {area_name} @ {size}m [{layer}]: {reason}")


def cmd_render(
    area_filter: str | None = None,
    vegetation_model: str = DEFAULT_VEGETATION_MODEL,
    tile_size_m: int | None = None,
    all_sizes: bool = False,
) -> None:
    """Render merged tree + shadow FGB polygons over the full-area orthophoto and save as PNG."""
    import geopandas as gpd
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    from pyproj import Transformer

    _to_utm = Transformer.from_crs("EPSG:4326", "EPSG:25832", always_xy=True)

    sizes = TILE_SIZES_M if all_sizes else [tile_size_m or TILE_SIZE_M]
    areas = {k: v for k, v in AREAS.items() if area_filter is None or k == area_filter}
    if not areas:
        print(f"No area named {area_filter!r}. Available: {list(AREAS)}")
        return

    for area_name, area in areas.items():
        full_img_path = OUTPUT_DIR / area_name / f"{area_name}_full.png"
        if not full_img_path.exists():
            print(f"  [skip] {full_img_path.name} not found — run 'download' first")
            continue

        img = np.array(Image.open(full_img_path).convert("RGB"))
        west_m, south_m = _to_utm.transform(area["west"], area["south"])
        east_m, north_m = _to_utm.transform(area["east"], area["north"])
        extent = [west_m, east_m, south_m, north_m]

        buildings_path = OUTPUT_DIR / f"buildings_{area_name}.fgb"
        buildings_gdf = gpd.read_file(buildings_path) if buildings_path.exists() else None

        for size in sizes:
            seg_dir = OUTPUT_DIR / "segments" / f"{size}m"
            shadow_dir = OUTPUT_DIR / "shadows" / f"{size}m"
            trees_path = seg_dir / f"{area_name}_{vegetation_model}_trees_merged.fgb"
            shadow_path = shadow_dir / f"{area_name}_{vegetation_model}_shadow_merged.fgb"

            if not trees_path.exists():
                print(f"  [skip] {trees_path.name} not found — run 'merge' first")
                continue

            trees_gdf = gpd.read_file(trees_path)
            shadow_gdf = gpd.read_file(shadow_path) if shadow_path.exists() and shadow_path.stat().st_size > 0 else None

            # Dissolve for rendering only — on-disk FGBs with attributes are unchanged
            trees_render  = _dissolve_for_render(trees_gdf) if len(trees_gdf) > 0 else trees_gdf
            shadow_render = _dissolve_for_render(shadow_gdf) if shadow_gdf is not None and len(shadow_gdf) > 0 else shadow_gdf

            # --- Figure 1: post-segmentation ---
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.imshow(img, extent=extent, origin="upper", aspect="equal")
            if buildings_gdf is not None and len(buildings_gdf) > 0:
                buildings_gdf.plot(ax=ax, facecolor="#d94747", edgecolor="none", alpha=0.55, zorder=2)
            if len(trees_render) > 0:
                trees_render.plot(ax=ax, facecolor="#267326", edgecolor="#90ee90", linewidth=0.4, alpha=0.65, zorder=4)
            ax.set_xlim(west_m, east_m)
            ax.set_ylim(south_m, north_m)
            ax.set_xlabel("Easting (m, EPSG:25832)")
            ax.set_ylabel("Northing (m, EPSG:25832)")
            ax.set_title(f"{area_name} @ {size}m tiles — segmentation [{vegetation_model}]", fontsize=11)
            legend = [mpatches.Patch(color="#267326", label=f"Trees ({len(trees_gdf)})")]
            if buildings_gdf is not None:
                legend.append(mpatches.Patch(color="#d94747", label=f"Buildings ({len(buildings_gdf)})"))
            ax.legend(handles=legend, loc="lower right", fontsize=9, framealpha=0.85)
            fig.tight_layout()
            seg_render_path = seg_dir / f"{area_name}_{vegetation_model}_{size}m_seg_render.png"
            fig.savefig(seg_render_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  {area_name} @ {size}m [seg]:    → {seg_render_path.name}  ({len(trees_gdf)} trees)")

            # --- Figure 2: post-shadow ---
            if shadow_render is not None and len(shadow_render) > 0:
                fig, ax = plt.subplots(figsize=(10, 10))
                ax.imshow(img, extent=extent, origin="upper", aspect="equal")
                if buildings_gdf is not None and len(buildings_gdf) > 0:
                    buildings_gdf.plot(ax=ax, facecolor="#d94747", edgecolor="none", alpha=0.55, zorder=2)
                shadow_render.plot(ax=ax, facecolor="#1a1a4d", edgecolor="none", alpha=0.50, zorder=3)
                trees_render.plot(ax=ax, facecolor="#267326", edgecolor="#90ee90", linewidth=0.4, alpha=0.65, zorder=4)
                ax.set_xlim(west_m, east_m)
                ax.set_ylim(south_m, north_m)
                ax.set_xlabel("Easting (m, EPSG:25832)")
                ax.set_ylabel("Northing (m, EPSG:25832)")
                ax.set_title(f"{area_name} @ {size}m tiles — segmentation + shadows [{vegetation_model}]", fontsize=11)
                legend = [
                    mpatches.Patch(color="#267326", label=f"Trees ({len(trees_gdf)})"),
                    mpatches.Patch(color="#d94747", label=f"Buildings ({len(buildings_gdf) if buildings_gdf is not None else 0})"),
                    mpatches.Patch(color="#1a1a4d", label=f"Shadows ({len(shadow_gdf)})"),
                ]
                ax.legend(handles=legend, loc="lower right", fontsize=9, framealpha=0.85)
                fig.tight_layout()
                shadow_render_path = shadow_dir / f"{area_name}_{vegetation_model}_{size}m_shadow_render.png"
                fig.savefig(shadow_render_path, dpi=150, bbox_inches="tight")
                plt.close(fig)
                print(f"  {area_name} @ {size}m [shadow]: → {shadow_render_path.name}  ({len(shadow_gdf)} shadows)")
            else:
                print(f"  {area_name} @ {size}m [shadow]: no shadow data — skipped")

    cmd_species_grid(area_filter=area_filter)


def cmd_species_grid(area_filter: str | None = None) -> None:
    """Render dominant Baumkataster species per tile for all tile sizes as a single multi-panel PNG."""
    import geopandas as gpd
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from shapely.geometry import box
    from pyproj import Transformer
    from src.shadow.cadastre import tile_dominant_genus
    from src.config import BAUMKATASTER_PATH

    if BAUMKATASTER_PATH is None or not BAUMKATASTER_PATH.exists():
        print("  [skip] BAUMKATASTER_PATH not set or missing — species grid unavailable")
        return

    _to_utm = Transformer.from_crs("EPSG:4326", "EPSG:25832", always_xy=True)
    bk = gpd.read_file(BAUMKATASTER_PATH)

    areas = {k: v for k, v in AREAS.items() if area_filter is None or k == area_filter}

    for area_name, area in areas.items():
        full_img_path = OUTPUT_DIR / area_name / f"{area_name}_full.png"
        if not full_img_path.exists():
            print(f"  [skip] {full_img_path.name} not found — run 'download' first")
            continue

        img = np.array(Image.open(full_img_path).convert("RGB"))
        west_m, south_m = _to_utm.transform(area["west"], area["south"])
        east_m, north_m = _to_utm.transform(area["east"], area["north"])
        extent = [west_m, east_m, south_m, north_m]

        # Compute dominant species per tile for every tile size
        all_species: set[str] = set()
        all_tile_data: dict[int, list] = {}
        for size in TILE_SIZES_M:
            tiles = tiles_for_area(area, size)
            records = []
            for t in tiles:
                w_m, s_m = _to_utm.transform(t["west"], t["south"])
                e_m, n_m = _to_utm.transform(t["east"], t["north"])
                tile_poly = box(w_m, s_m, e_m, n_m)
                bk_tile = bk.clip(gpd.GeoDataFrame(geometry=[tile_poly], crs="EPSG:25832"))
                genus = tile_dominant_genus(bk_tile)
                label = genus if genus else "—"
                records.append({"geometry": tile_poly, "species": label,
                                "label": label, "n_bk": len(bk_tile)})
                all_species.add(label)
            all_tile_data[size] = records

        species_list = sorted(all_species - {"—"})
        palette = plt.cm.Set2.colors + plt.cm.Set1.colors
        species_color = {s: palette[i % len(palette)] for i, s in enumerate(species_list)}
        species_color["—"] = "#cccccc"

        fig, axes = plt.subplots(2, 2, figsize=(18, 18))
        fig.suptitle(f"Dominant Baumkataster species per tile — {area_name}",
                     fontsize=15, fontweight="bold")

        for ax, size in zip(axes.flat, TILE_SIZES_M):
            records = all_tile_data[size]
            tile_gdf = gpd.GeoDataFrame(records, crs="EPSG:25832")

            ax.imshow(img, extent=extent, origin="upper", aspect="equal")
            for _, row in tile_gdf.iterrows():
                color = species_color.get(row["label"], "#cccccc")
                gpd.GeoDataFrame([row], crs="EPSG:25832").plot(
                    ax=ax, facecolor=color, edgecolor="white", linewidth=1.2, alpha=0.45, zorder=2
                )
                cx, cy = row.geometry.centroid.x, row.geometry.centroid.y
                fontsize = max(5, min(9, size // 60))
                ax.text(cx, cy, row["label"], ha="center", va="center",
                        fontsize=fontsize, fontweight="bold", color="white",
                        bbox=dict(facecolor="black", alpha=0.45, edgecolor="none", pad=1.5),
                        zorder=3)

            ax.set_xlim(west_m, east_m)
            ax.set_ylim(south_m, north_m)
            ax.set_title(f"{size} m tiles  ({len(records)} tiles)", fontsize=11)
            ax.set_xlabel("Easting (m)", fontsize=8)
            ax.set_ylabel("Northing (m)", fontsize=8)
            ax.tick_params(labelsize=7)

        legend_patches = [mpatches.Patch(color=species_color[s], label=s) for s in species_list]
        legend_patches.append(mpatches.Patch(color="#cccccc", label="no BK data"))
        fig.legend(handles=legend_patches, loc="lower center", ncol=6,
                   fontsize=9, framealpha=0.9, title="Dominant genus",
                   bbox_to_anchor=(0.5, 0.01))
        plt.tight_layout(rect=[0, 0.06, 1, 0.97])

        out = OUTPUT_DIR / "segments" / f"{area_name}_species_grid.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  {area_name} → {out.name}")


def cmd_all(dry_run: bool = False, vegetation_model: str = DEFAULT_VEGETATION_MODEL, datetime_utc: str | None = None, tile_size_m: int | None = None, all_sizes: bool = False) -> None:
    cmd_download(dry_run=dry_run, tile_size_m=tile_size_m, all_sizes=all_sizes)
    if not dry_run:
        cmd_segment(vegetation_model=vegetation_model, tile_size_m=tile_size_m, all_sizes=all_sizes)
        cmd_shadow(vegetation_model=vegetation_model, datetime_utc=datetime_utc, tile_size_m=tile_size_m, all_sizes=all_sizes)
        cmd_species_grid()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Urban shadow analysis pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--tile-size", type=int, default=None, metavar="M", help="Single tile size in metres")
    parser.add_argument("--all-sizes", action="store_true", help=f"Run for all TILE_SIZES_M {TILE_SIZES_M}")
    sub = parser.add_subparsers(dest="command")

    dl = sub.add_parser("download", help="Fetch orthophoto tile grid")
    dl.add_argument("--dry-run", action="store_true", help="Show tile layout without downloading")
    dl.add_argument("--tile-size", type=int, default=None, metavar="M", help="Single tile size in metres (overrides config default)")
    dl.add_argument("--all-sizes", action="store_true", help=f"Download for all TILE_SIZES_M {TILE_SIZES_M}")

    seg = sub.add_parser("segment", help="Segment downloaded tiles (OSM + vegetation)")
    seg.add_argument(
        "--vegetation-model",
        choices=VEGETATION_MODELS,
        default=DEFAULT_VEGETATION_MODEL,
        help=f"Vegetation segmentation method (default: {DEFAULT_VEGETATION_MODEL})",
    )
    seg.add_argument("--tile-size", type=int, default=None, metavar="M", help="Single tile size in metres")
    seg.add_argument("--all-sizes", action="store_true", help=f"Segment for all TILE_SIZES_M {TILE_SIZES_M}")

    cmp = sub.add_parser("compare", help="compare models | compare sizes")
    cmp_sub = cmp.add_subparsers(dest="compare_command")

    cmp_models = cmp_sub.add_parser("models", help="Run all vegetation methods side-by-side")
    cmp_models.add_argument("--area", default=None, help="Limit to a single area name")

    cmp_sizes = cmp_sub.add_parser("sizes", help="Pairwise IoU across tile sizes on a shared UTM grid")
    cmp_sizes.add_argument("--area", default=None, help="Limit to a single area name")
    cmp_sizes.add_argument(
        "--vegetation-model",
        choices=VEGETATION_MODELS,
        default=DEFAULT_VEGETATION_MODEL,
        help=f"Which tree .fgb files to load (default: {DEFAULT_VEGETATION_MODEL})",
    )
    cmp_sizes.add_argument(
        "--resolution",
        type=float,
        default=1.0,
        metavar="M",
        help="Common grid resolution in metres/px (default: 1.0)",
    )
    cmp_sizes.add_argument(
        "--reference-size",
        type=int,
        default=250,
        metavar="M",
        help="Tile size used as precision/recall reference (default: 250)",
    )

    shd = sub.add_parser("shadow", help="Cast tree shadows from segmentation maps")
    shd.add_argument("--area", default=None, help="Limit to a single area name")
    shd.add_argument(
        "--vegetation-model",
        choices=VEGETATION_MODELS,
        default=DEFAULT_VEGETATION_MODEL,
        help=f"Which segmentation files to load (default: {DEFAULT_VEGETATION_MODEL})",
    )
    shd.add_argument(
        "--datetime-utc",
        default=None,
        help='ISO datetime in UTC, e.g. "2024-06-21T11:00:00". Defaults to now.',
    )
    shd.add_argument("--tile-size", type=int, default=None, metavar="M", help="Single tile size in metres")
    shd.add_argument("--all-sizes", action="store_true", help=f"Cast shadows for all TILE_SIZES_M {TILE_SIZES_M}")

    rnd = sub.add_parser("render", help="Render merged tree/shadow polygons over the full-area orthophoto")
    rnd.add_argument("--area", default=None, help="Limit to a single area name")
    rnd.add_argument(
        "--vegetation-model",
        choices=VEGETATION_MODELS,
        default=DEFAULT_VEGETATION_MODEL,
        help=f"Which merged FGBs to render (default: {DEFAULT_VEGETATION_MODEL})",
    )
    rnd.add_argument("--tile-size", type=int, default=None, metavar="M", help="Single tile size in metres")
    rnd.add_argument("--all-sizes", action="store_true", help=f"Render for all TILE_SIZES_M {TILE_SIZES_M}")

    mrg = sub.add_parser("merge", help="Merge per-tile tree and shadow FGBs into one full-area FGB")
    mrg.add_argument("--area", default=None, help="Limit to a single area name")
    mrg.add_argument(
        "--vegetation-model",
        choices=VEGETATION_MODELS,
        default=DEFAULT_VEGETATION_MODEL,
        help=f"Which FGB files to merge (default: {DEFAULT_VEGETATION_MODEL})",
    )
    mrg.add_argument("--tile-size", type=int, default=None, metavar="M", help="Single tile size in metres")
    mrg.add_argument("--all-sizes", action="store_true", help=f"Merge for all TILE_SIZES_M {TILE_SIZES_M}")
    mrg.add_argument(
        "--layer",
        choices=("trees", "shadow", "both"),
        default="both",
        help="Which layer to merge (default: both)",
    )

    tun = sub.add_parser("tune", help="Tune vegetation model hyperparameters against a reference")
    tun.add_argument(
        "--model",
        choices=TUNABLE_MODELS,
        default="vari",
        help="Model to tune (default: vari)",
    )
    tun.add_argument(
        "--tile",
        default="0_0",
        metavar="IX_IY",
        help="Tile index to tune on, e.g. '0_0' (default: 0_0)",
    )
    tun.add_argument("--area", default=None, help="Limit to a single area name")
    tun.add_argument(
        "--reference",
        choices=VEGETATION_MODELS,
        default="deepforest",
        help="Reference model used as proxy ground truth (default: deepforest)",
    )

    sta = sub.add_parser("status", help="Show what has been computed per area and tile size")
    sta.add_argument("--area", default=None, help="Limit to a single area name")
    sta.add_argument(
        "--vegetation-model",
        choices=VEGETATION_MODELS,
        default=DEFAULT_VEGETATION_MODEL,
        help=f"Model to check segment/shadow outputs for (default: {DEFAULT_VEGETATION_MODEL})",
    )
    sta.add_argument("--tile-size", type=int, default=None, metavar="M", help="Single tile size in metres")
    sta.add_argument("--all-sizes", action="store_true", help=f"Show status for all TILE_SIZES_M {TILE_SIZES_M}")

    spg = sub.add_parser("species-grid", help="Multi-panel PNG: dominant BK species per tile for all tile sizes")
    spg.add_argument("--area", default=None, help="Limit to a single area name")

    diu = sub.add_parser("diurnal", help="Hourly shadow table + overlays for one tile across a day")
    diu.add_argument("--area", default=None, help="Limit to a single area name")
    diu.add_argument(
        "--vegetation-model",
        choices=VEGETATION_MODELS,
        default=DEFAULT_VEGETATION_MODEL,
        help=f"Which segmentation files to load (default: {DEFAULT_VEGETATION_MODEL})",
    )
    diu.add_argument("--date-utc", default=None, help='ISO date in UTC, e.g. "2026-06-21". Defaults to today.')
    diu.add_argument("--tile", default="0_0", metavar="IX_IY", help="Tile index, e.g. '0_0' (default: 0_0)")
    diu.add_argument("--tile-size", type=int, default=None, metavar="M", help="Tile size in metres")

    all_cmd = sub.add_parser("all", help="Download, segment, and cast shadows")
    all_cmd.add_argument("--dry-run", action="store_true")
    all_cmd.add_argument(
        "--vegetation-model",
        choices=VEGETATION_MODELS,
        default=DEFAULT_VEGETATION_MODEL,
        help=f"Vegetation segmentation method (default: {DEFAULT_VEGETATION_MODEL})",
    )
    all_cmd.add_argument(
        "--datetime-utc",
        default=None,
        help='ISO datetime in UTC for shadow casting, e.g. "2024-06-21T11:00:00". Defaults to now.',
    )
    all_cmd.add_argument("--tile-size", type=int, default=None, metavar="M", help="Single tile size in metres")
    all_cmd.add_argument("--all-sizes", action="store_true", help=f"Run for all TILE_SIZES_M {TILE_SIZES_M}")

    args = parser.parse_args()

    if args.command == "download":
        cmd_download(dry_run=args.dry_run, tile_size_m=args.tile_size, all_sizes=args.all_sizes)
    elif args.command == "segment":
        cmd_segment(vegetation_model=args.vegetation_model, tile_size_m=args.tile_size, all_sizes=args.all_sizes)
    elif args.command == "compare":
        if args.compare_command == "sizes":
            cmd_compare_sizes(
                area_filter=args.area,
                vegetation_model=args.vegetation_model,
                resolution_m=args.resolution,
                reference_size_m=args.reference_size,
            )
        elif args.compare_command == "models" or args.compare_command is None:
            cmd_compare(area_filter=getattr(args, "area", None))
        else:
            cmp.print_help()
    elif args.command == "shadow":
        cmd_shadow(
            area_filter=args.area,
            vegetation_model=args.vegetation_model,
            datetime_utc=args.datetime_utc,
            tile_size_m=args.tile_size,
            all_sizes=args.all_sizes,
        )
    elif args.command == "render":
        cmd_render(area_filter=args.area, vegetation_model=args.vegetation_model,
                   tile_size_m=args.tile_size, all_sizes=args.all_sizes)
    elif args.command == "species-grid":
        cmd_species_grid(area_filter=args.area)
    elif args.command == "merge":
        layers = ("trees", "shadow") if args.layer == "both" else (args.layer,)
        cmd_merge(area_filter=args.area, vegetation_model=args.vegetation_model,
                  tile_size_m=args.tile_size, all_sizes=args.all_sizes, layers=layers)
    elif args.command == "status":
        cmd_status(area_filter=args.area, vegetation_model=args.vegetation_model, tile_size_m=args.tile_size, all_sizes=args.all_sizes)
    elif args.command == "diurnal":
        cmd_diurnal(area_filter=args.area, vegetation_model=args.vegetation_model, date_utc=args.date_utc, tile_key=args.tile, tile_size_m=args.tile_size)
    elif args.command == "tune":
        cmd_tune(
            model=args.model,
            tile_key=args.tile,
            area_filter=args.area,
            reference_model=args.reference,
        )
    elif args.command == "all":
        cmd_all(dry_run=args.dry_run, vegetation_model=args.vegetation_model, datetime_utc=args.datetime_utc, tile_size_m=args.tile_size, all_sizes=args.all_sizes)
    else:
        cmd_all(tile_size_m=args.tile_size, all_sizes=args.all_sizes)
