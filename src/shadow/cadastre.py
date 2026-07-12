from __future__ import annotations
"""Spatial join of Baumkataster measurements onto pipeline tree polygons."""

import math

import geopandas as gpd
import pandas as pd

from src.config import ALLOMETRIC_A, ALLOMETRIC_B, ALLOMETRIC_PROFILES

# Genera known to be deciduous in the Magdeburg climate zone
DECIDUOUS_GENERA = {
    "Tilia", "Acer", "Quercus", "Fraxinus", "Robinia", "Aesculus",
    "Carpinus", "Prunus", "Platanus", "Populus", "Ulmus", "Betula",
    "Salix", "Fagus", "Sorbus", "Pyrus", "Malus",
}


def tile_dominant_genus(bk_tile: gpd.GeoDataFrame) -> str | None:
    """Return the most frequent genus in a BK tile clip, or None if empty."""
    if len(bk_tile) == 0:
        return None
    vc = bk_tile["Gattung lang"].str.split().str[0].value_counts()
    return vc.index[0] if len(vc) > 0 else None


def enrich_from_baumkataster(
    tree_gdf: gpd.GeoDataFrame,
    bk_gdf: gpd.GeoDataFrame,
    match_radius_m: float = 15.0,
) -> gpd.GeoDataFrame:
    """Spatial join BK measurements onto pipeline tree polygons (left join).

    For each pipeline polygon the nearest BK tree within match_radius_m is found.
    Matched trees have height_m / crown_radius_m overwritten with measured values.
    All rows gain six new columns; unmatched rows keep allometric estimates.

    New columns
    -----------
    species               Gattung lang value (None for unmatched)
    height_source         "measured" | "allometric"
    is_deciduous          True/False (None for unmatched)
    trunk_circumference_cm Stammumfang in cm (None for unmatched or 0-valued)
    planting_year         Pflanzjahr (None for unmatched or 0-valued)
    bk_match_dist_m       centroid-to-point distance in metres (None for unmatched)
    """
    result = tree_gdf.copy()
    result["allometric_height_m"] = result["height_m"]
    result["species"] = None
    result["height_source"] = "allometric"
    result["is_deciduous"] = None
    result["trunk_circumference_cm"] = None
    result["planting_year"] = None
    result["bk_match_dist_m"] = None

    if bk_gdf is None or len(bk_gdf) == 0:
        return result

    bk_valid = bk_gdf[
        (bk_gdf["Baumhoehe"] > 1) & (bk_gdf["Kronendurchmesser"] > 0.5)
    ].copy()
    if len(bk_valid) == 0:
        return result

    # Buffer BK points by max(crown_radius, 2 m) for intersection candidate test
    bk_circles = bk_valid.copy()
    bk_circles["geometry"] = bk_circles.apply(
        lambda r: r.geometry.buffer(max(r["Kronendurchmesser"] / 2, 2.0)), axis=1
    )

    # Left sjoin — all pipeline polygons retained
    joined = gpd.sjoin(
        result[["tree_id", "geometry"]],
        bk_circles[["Gattung lang", "Baumhoehe", "Kronendurchmesser",
                     "Stammumfang", "Pflanzjahr", "geometry"]],
        how="left",
        predicate="intersects",
    )

    # Distance from pipeline polygon centroid to BK point (for tie-breaking)
    tree_centroids = result.set_index("tree_id").geometry.centroid
    bk_pts = bk_valid.geometry

    def _dist(row):
        if pd.isna(row.get("index_right")):
            return float("inf")
        return tree_centroids[row["tree_id"]].distance(bk_pts.loc[row["index_right"]])

    joined["_dist"] = joined.apply(_dist, axis=1)

    # Keep nearest match within radius per pipeline polygon
    best = (
        joined[joined["_dist"] <= match_radius_m]
        .sort_values("_dist")
        .groupby("tree_id")
        .first()[["Gattung lang", "Baumhoehe", "Kronendurchmesser",
                   "Stammumfang", "Pflanzjahr", "_dist"]]
    )

    if len(best) == 0:
        return result

    # Merge best matches and update matched rows
    result = result.merge(best.reset_index(), on="tree_id", how="left")
    matched = result["Baumhoehe"].notna()

    result.loc[matched, "height_m"] = result.loc[matched, "Baumhoehe"]
    result.loc[matched, "crown_radius_m"] = result.loc[matched, "Kronendurchmesser"] / 2
    result.loc[matched, "species"] = result.loc[matched, "Gattung lang"]
    result.loc[matched, "height_source"] = "measured"

    # Recompute allometric_height_m for matched trees using per-species (A, B).
    # The initial value (set at line 44) used the tile-dominant genus; now that we
    # know each tree's exact species we can use the species-specific curve instead.
    if "crown_area_m2" in result.columns:
        for idx in result.index[matched]:
            sp = result.at[idx, "Gattung lang"]
            genus = sp.split()[0] if pd.notna(sp) else None
            A, B = ALLOMETRIC_PROFILES.get(genus, (ALLOMETRIC_A, ALLOMETRIC_B))
            area = result.at[idx, "crown_area_m2"]
            result.at[idx, "allometric_height_m"] = math.exp(
                A + B * math.log(max(area, 1e-6))
            )

    result.loc[matched, "is_deciduous"] = result.loc[matched, "species"].apply(
        lambda s: (s.split()[0] in DECIDUOUS_GENERA) if pd.notna(s) else None
    )
    result.loc[matched, "trunk_circumference_cm"] = result.loc[matched, "Stammumfang"].where(
        result.loc[matched, "Stammumfang"] > 0
    )
    result.loc[matched, "planting_year"] = result.loc[matched, "Pflanzjahr"].where(
        result.loc[matched, "Pflanzjahr"] > 0
    )
    result.loc[matched, "bk_match_dist_m"] = result.loc[matched, "_dist"]

    result = result.drop(columns=["Baumhoehe", "Kronendurchmesser",
                                   "Gattung lang", "Stammumfang", "Pflanzjahr", "_dist"])
    return gpd.GeoDataFrame(result, geometry="geometry", crs=tree_gdf.crs)
