from __future__ import annotations
"""OSM vector layer downloader: buildings and buffered road polygons."""

from pathlib import Path

import geopandas as gpd
import osmnx as ox

ROAD_BUFFER = {
    "motorway": 7.0,       "motorway_link": 7.0,
    "trunk": 7.0,          "trunk_link": 7.0,
    "primary": 5.5,        "primary_link": 5.5,
    "secondary": 5.5,      "secondary_link": 5.5,
    "tertiary": 3.5,       "tertiary_link": 3.5,
    "residential": 3.5,
    "service": 2.5,        "unclassified": 2.5,    "living_street": 2.5,
    "footway": 1.5,        "cycleway": 1.5,         "path": 1.5,        "track": 1.5,
}
DEFAULT_ROAD_BUFFER = 2.5


HEIGHT_COLS = ["building:levels", "height", "building:height", "min_height", "roof:levels"]


def fetch_buildings(
    bbox: dict,
    cache_path: Path | None = None,
    timeout: int = 180,
) -> gpd.GeoDataFrame:
    """Return building polygons in EPSG:25832 for bbox (WGS84).

    Columns: osmid, geometry, plus any available height fields
    (building:levels, height, building:height, min_height, roof:levels).
    Loads from cache_path (.fgb) if it exists; downloads and saves otherwise.
    """
    if cache_path is not None and Path(cache_path).exists():
        return gpd.read_file(cache_path)

    ox.settings.timeout = timeout
    ox.settings.use_cache = True
    raw = ox.features_from_bbox(
        bbox=(bbox["west"], bbox["south"], bbox["east"], bbox["north"]),
        tags={"building": True},
    )
    gdf = raw[raw.geometry.type.isin(["Polygon", "MultiPolygon"])].copy()
    gdf = gdf.to_crs("EPSG:25832")

    keep = ["geometry"] + [c for c in HEIGHT_COLS if c in gdf.columns]
    gdf = gdf[keep]
    gdf.index = gdf.index.droplevel("element")
    gdf.index.name = "osmid"
    gdf = gdf.reset_index()

    if cache_path is not None:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        gdf.to_file(cache_path, driver="FlatGeobuf")

    return gdf


def fetch_roads(
    bbox: dict,
    cache_path: Path | None = None,
    road_buffer: dict | None = None,
    default_buffer: float = DEFAULT_ROAD_BUFFER,
    timeout: int = 180,
) -> gpd.GeoDataFrame:
    """Return buffered road polygons in EPSG:25832 for bbox (WGS84).

    Loads from cache_path if it exists; downloads, buffers, and saves otherwise.
    """
    if cache_path is not None and Path(cache_path).exists():
        return gpd.read_file(cache_path)

    buf = road_buffer if road_buffer is not None else ROAD_BUFFER

    ox.settings.timeout = timeout
    ox.settings.use_cache = True
    raw = ox.features_from_bbox(
        bbox=(bbox["west"], bbox["south"], bbox["east"], bbox["north"]),
        tags={"highway": True},
    )
    gdf = raw[raw.geometry.notna()].copy()
    gdf = gdf.to_crs("EPSG:25832")

    hw = gdf.get("highway", "").astype(str)
    gdf["highway"] = hw
    gdf["buffer_m"] = hw.map(lambda h: buf.get(h, default_buffer))
    gdf["geometry"] = gdf.geometry.buffer(gdf["buffer_m"])
    gdf = gdf[["geometry", "highway", "buffer_m"]]

    if cache_path is not None:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        gdf.to_file(cache_path, driver="FlatGeobuf")

    return gdf
