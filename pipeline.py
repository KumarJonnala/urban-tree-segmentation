"""Pipeline entry point.

Subcommands:
  download   — fetch orthophoto tile grid + full-area overview image
  segment    — run OSM + vegetation segmentation on downloaded tiles
  compare    — run all vegetation methods side-by-side and print metrics
  shadow     — cast tree shadows from segmentation maps for a given datetime
  diurnal    — hourly shadow table + overlays for one tile across a full day
  status     — show what has been computed per area and tile size
  tune       — grid-search hyperparameters for a tunable model against a reference
  all        — download + segment + shadow (default when no subcommand given)

Tile size flags (available on download, segment, shadow, all):
  --tile-size M    run for a single tile size M (metres)
  --all-sizes      run for all sizes in TILE_SIZES_M [100, 250, 500, 1000]
  (default: TILE_SIZE_M = 250)

Examples:
  python pipeline.py                                         # download + segment + shadow @ 250m
  python pipeline.py --all-sizes                             # full pipeline for all tile sizes
  python pipeline.py download --dry-run --all-sizes          # preview all tile grids
  python pipeline.py download --tile-size 500                # download 500m tiles only
  python pipeline.py segment --vegetation-model vari
  python pipeline.py compare --area ovgu_bbox
  python pipeline.py shadow --datetime-utc "2026-06-21T09:00:00" --all-sizes
  python pipeline.py diurnal --date-utc "2026-06-21" --tile 1_1 --tile-size 250
  python pipeline.py status --all-sizes
  python pipeline.py tune --model vari --tile 0_0
"""

import argparse

import numpy as np
from PIL import Image

from src.config import AREAS, DEFAULT_VEGETATION_MODEL, OUTPUT_DIR, TILE_SIZE_M, TILE_SIZES_M
from src.data_preprocessing import fetch_area_grid, fetch_full_area_image, fetch_buildings, fetch_roads, tiles_for_area
from src.segmentation import (
    compare_vegetation,
    save_segmentation,
    vari_mask,
)

VEGETATION_MODELS = ("vari", "deepforest", "samgeo", "segformer_b5", "deeplab",
                     "tcd_segformer", "ensemble")
TUNABLE_MODELS = ("vari", "deepforest", "samgeo")


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
            full_path = fetch_full_area_image(area_name)
            print(f"\n--- {area_name}: full-area image → {full_path.name} ---")
        for size in sizes:
            tiles = tiles_for_area(area, size)
            print(f"\n--- {area_name} @ {size}m: {len(tiles)} tile(s) ---")
            if dry_run:
                for t in tiles:
                    print(f"  tile ({t['ix']},{t['iy']})  W={t['west']:.6f} S={t['south']:.6f} E={t['east']:.6f} N={t['north']:.6f}")
            else:
                paths = fetch_area_grid(area_name, tile_size_m=size)
                print(f"  {len(paths)} tile(s) saved")


def cmd_segment(vegetation_model: str = DEFAULT_VEGETATION_MODEL, tile_size_m: int | None = None, all_sizes: bool = False) -> None:
    from src.shadow.casting import vectorize_trees
    sizes = TILE_SIZES_M if all_sizes else [tile_size_m or TILE_SIZE_M]
    _, mask_fn = _load_vegetation_model(vegetation_model)

    for area_name, area in AREAS.items():
        buildings = fetch_buildings(area, cache_path=OUTPUT_DIR / f"buildings_{area_name}.fgb")
        roads = fetch_roads(area, cache_path=OUTPUT_DIR / f"roads_{area_name}.fgb")
        print(f"  OSM: {len(buildings)} buildings, {len(roads)} roads")

        for size in sizes:
            tiles = tiles_for_area(area, size)
            tile_dir = OUTPUT_DIR / area_name / f"{size}m"
            seg_dir = OUTPUT_DIR / "segments" / f"{size}m"
            print(f"\n--- {area_name} @ {size}m: segmenting {len(tiles)} tile(s) [{vegetation_model}] ---")

            for t in tiles:
                base_stem = f"{area_name}_tile_{t['ix']}_{t['iy']}"
                tile_path = tile_dir / f"{base_stem}.png"
                if not tile_path.exists():
                    print(f"  [skip] {base_stem}.png not found — run 'download' first")
                    continue

                img = np.array(Image.open(tile_path).convert("RGB"))
                tree_mask = mask_fn(img)
                stem = f"{base_stem}_{vegetation_model}"
                npy_path, png_path = save_segmentation(
                    img, t, buildings, roads, tree_mask,
                    out_dir=seg_dir, stem=stem,
                )
                print(f"  {base_stem}: saved {npy_path.name}, {png_path.name}")
                tree_gdf = vectorize_trees(tree_mask, t, vegetation_model)
                fgb_path = seg_dir / f"{stem}_trees.fgb"
                tree_gdf.to_file(fgb_path, driver="FlatGeobuf")
                print(f"  {base_stem}: vectorized {len(tree_gdf)} tree(s) → {fgb_path.name}")


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

            for t in tiles:
                stem = f"{area_name}_tile_{t['ix']}_{t['iy']}"
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
                if tree_gdf is not None:
                    print(f"  {stem}: using stored tree vectors ({len(tree_gdf)} trees)")

                shadow_mask = cast_tree_shadows(seg_map, t, when, tree_gdf=tree_gdf)
                coverage_pct = shadow_mask.mean() * 100

                out_path = shadow_dir / f"{stem}_{vegetation_model}_shadow.png"
                save_shadow_overlay(
                    img, seg_map, shadow_mask, out_path,
                    title=f"{stem} — {when.strftime('%Y-%m-%d %H:%M UTC')}",
                )

                shadow_gdf = vectorize_shadows(shadow_mask, t, when, vegetation_model)
                fgb_out = shadow_dir / f"{stem}_{vegetation_model}_shadow.fgb"
                shadow_gdf.to_file(fgb_out, driver="FlatGeobuf")
                print(f"  {stem}: shadow={coverage_pct:.1f}%  → {out_path.name}, {fgb_out.name}")


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


def cmd_all(dry_run: bool = False, vegetation_model: str = DEFAULT_VEGETATION_MODEL, datetime_utc: str | None = None, tile_size_m: int | None = None, all_sizes: bool = False) -> None:
    cmd_download(dry_run=dry_run, tile_size_m=tile_size_m, all_sizes=all_sizes)
    if not dry_run:
        cmd_segment(vegetation_model=vegetation_model, tile_size_m=tile_size_m, all_sizes=all_sizes)
        cmd_shadow(vegetation_model=vegetation_model, datetime_utc=datetime_utc, tile_size_m=tile_size_m, all_sizes=all_sizes)


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

    cmp = sub.add_parser("compare", help="Compare all vegetation methods side-by-side")
    cmp.add_argument("--area", default=None, help="Limit to a single area name")

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
        cmd_compare(area_filter=args.area)
    elif args.command == "shadow":
        cmd_shadow(
            area_filter=args.area,
            vegetation_model=args.vegetation_model,
            datetime_utc=args.datetime_utc,
            tile_size_m=args.tile_size,
            all_sizes=args.all_sizes,
        )
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
