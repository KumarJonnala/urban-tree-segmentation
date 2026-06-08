"""Orchestrates tiling + fetching: splits an area into a grid and downloads each tile."""

from pathlib import Path

from src.config import AREAS, TILE_SIZE_M, IMAGE_WIDTH, IMAGE_HEIGHT, OUTPUT_DIR
from src.data_preprocessing.fetcher import fetch_and_save
from src.data_preprocessing.tiling import tiles_for_area


def fetch_full_area_image(area_name, width=IMAGE_WIDTH, height=IMAGE_HEIGHT, output_root=None):
    """Fetch the entire area bbox as a single overview PNG (pre-tiled image).

    Saved to OUTPUT_DIR/{area_name}/{area_name}_full.png at the requested pixel dimensions.
    """
    if area_name not in AREAS:
        raise KeyError(area_name)
    out_root = Path(output_root) if output_root else OUTPUT_DIR
    out_dir = out_root / area_name
    out_dir.mkdir(parents=True, exist_ok=True)
    return fetch_and_save(AREAS[area_name], f"{area_name}_full.png", out_dir, width=width, height=height)


def fetch_area_grid(area_name, tile_size_m=None, width=IMAGE_WIDTH, height=IMAGE_HEIGHT, output_root=None, fetch=True, max_tiles=None):
    """Download all tiles for area_name; returns list of saved paths (empty if fetch=False).

    Tiles are saved under OUTPUT_DIR/{area_name}/{tile_size}m/.
    """
    if area_name not in AREAS:
        raise KeyError(area_name)
    area = AREAS[area_name]
    tile_size = tile_size_m or TILE_SIZE_M
    out_root = Path(output_root) if output_root else OUTPUT_DIR
    out_dir = out_root / area_name / f"{tile_size}m"
    out_dir.mkdir(parents=True, exist_ok=True)
    tiles = tiles_for_area(area, tile_size)
    saved = []
    for i, t in enumerate(tiles):
        if max_tiles and i >= max_tiles:
            break
        fname = f"{area_name}_tile_{t['ix']}_{t['iy']}.png"
        if fetch:
            saved.append(fetch_and_save(t, fname, out_dir, width=width, height=height))
    return saved


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fetch grid of orthophotos for configured areas")
    parser.add_argument("area", nargs="?", help="Area name from config (or omit for all)")
    parser.add_argument("--tile-size", type=int, help="Tile size in meters (overrides config)")
    parser.add_argument("--width", type=int, default=IMAGE_WIDTH)
    parser.add_argument("--height", type=int, default=IMAGE_HEIGHT)
    parser.add_argument("--output-root", help="Output root directory")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-tiles", type=int, help="Limit number of tiles to fetch")
    args = parser.parse_args()

    output_root = Path(args.output_root) if args.output_root else None
    targets = [args.area] if args.area else list(AREAS.keys())
    for name in targets:
        fetch_area_grid(name, tile_size_m=args.tile_size, width=args.width, height=args.height, output_root=output_root, fetch=not args.dry_run, max_tiles=args.max_tiles)
