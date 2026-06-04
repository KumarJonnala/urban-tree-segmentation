"""Pipeline entry point.

Subcommands:
  download   — fetch orthophoto tile grid for all configured areas
  segment    — run OSM + vegetation segmentation on downloaded tiles
  compare    — run all vegetation methods side-by-side and print metrics
  shadow     — cast tree shadows from segmentation maps for a given datetime
  tune       — grid-search hyperparameters for a tunable model against a reference
  all        — download + segment + shadow (default when no subcommand given)

Examples:
  python pipeline.py                                      # download + segment + shadow (tcd_segformer)
  python pipeline.py download                             # tiles only
  python pipeline.py download --dry-run                   # preview tile layout
  python pipeline.py segment                              # segment with tcd_segformer (default)
  python pipeline.py segment --vegetation-model deepforest
  python pipeline.py segment --vegetation-model segformer_b5
  python pipeline.py segment --vegetation-model deeplab
  python pipeline.py compare                              # all methods, all areas
  python pipeline.py compare --area ovgu_bbox             # single area
  python pipeline.py all --vegetation-model deepforest    # download + segment + shadow
  python pipeline.py shadow                               # shadows for all tiles (now UTC)
  python pipeline.py shadow --datetime-utc "2026-05-21T11:00:00"
  python pipeline.py shadow --area ovgu_bbox --vegetation-model vari
  python pipeline.py tune --model vari --tile 0_0        # tune VARI vs deepforest reference
  python pipeline.py tune --model samgeo --reference segformer_b5
"""

import argparse

import numpy as np
from PIL import Image

from src.config import AREAS, DEFAULT_VEGETATION_MODEL, OUTPUT_DIR, TILE_SIZE_M
from src.data_preprocessing import fetch_area_grid, fetch_buildings, fetch_roads, tiles_for_area
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


def cmd_download(dry_run: bool = False) -> None:
    for area_name, area in AREAS.items():
        tiles = tiles_for_area(area, TILE_SIZE_M)
        print(f"\n--- {area_name}: {len(tiles)} tile(s) ---")
        if dry_run:
            for t in tiles:
                print(f"  tile ({t['ix']},{t['iy']})  W={t['west']:.6f} S={t['south']:.6f} E={t['east']:.6f} N={t['north']:.6f}")
        else:
            paths = fetch_area_grid(area_name)
            print(f"  {len(paths)} tile(s) saved")


def cmd_segment(vegetation_model: str = DEFAULT_VEGETATION_MODEL) -> None:
    from src.shadow.casting import vectorize_trees
    seg_dir = OUTPUT_DIR / "segments"
    _, mask_fn = _load_vegetation_model(vegetation_model)

    for area_name, area in AREAS.items():
        tiles = tiles_for_area(area, TILE_SIZE_M)
        tile_dir = OUTPUT_DIR / area_name
        print(f"\n--- {area_name}: segmenting {len(tiles)} tile(s) [{vegetation_model}] ---")

        buildings = fetch_buildings(area, cache_path=OUTPUT_DIR / f"buildings_{area_name}.fgb")
        roads = fetch_roads(area, cache_path=OUTPUT_DIR / f"roads_{area_name}.fgb")
        print(f"  OSM: {len(buildings)} buildings, {len(roads)} roads")

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
) -> None:
    import datetime as dt
    import geopandas as gpd
    from src.shadow import cast_tree_shadows, save_shadow_overlay, vectorize_shadows

    if datetime_utc is None:
        when = dt.datetime.now(tz=dt.timezone.utc)
    else:
        when = dt.datetime.fromisoformat(datetime_utc).replace(tzinfo=dt.timezone.utc)

    shadow_dir = OUTPUT_DIR / "shadows"
    seg_dir = OUTPUT_DIR / "segments"
    areas = {k: v for k, v in AREAS.items() if area_filter is None or k == area_filter}
    if not areas:
        print(f"No area named {area_filter!r}. Available: {list(AREAS)}")
        return

    print(f"Shadow casting for {when.strftime('%Y-%m-%d %H:%M UTC')}")

    for area_name, _ in areas.items():
        tiles = tiles_for_area(AREAS[area_name], TILE_SIZE_M)
        print(f"\n--- {area_name}: {len(tiles)} tile(s) ---")

        for t in tiles:
            stem = f"{area_name}_tile_{t['ix']}_{t['iy']}"
            seg_path = seg_dir / f"{stem}_{vegetation_model}_seg.npy"
            fgb_path = seg_dir / f"{stem}_{vegetation_model}_trees.fgb"
            img_path = OUTPUT_DIR / area_name / f"{stem}.png"

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


def cmd_all(dry_run: bool = False, vegetation_model: str = DEFAULT_VEGETATION_MODEL, datetime_utc: str | None = None) -> None:
    cmd_download(dry_run=dry_run)
    if not dry_run:
        cmd_segment(vegetation_model=vegetation_model)
        cmd_shadow(vegetation_model=vegetation_model, datetime_utc=datetime_utc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Urban shadow analysis pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command")

    dl = sub.add_parser("download", help="Fetch orthophoto tile grid")
    dl.add_argument("--dry-run", action="store_true", help="Show tile layout without downloading")

    seg = sub.add_parser("segment", help="Segment downloaded tiles (OSM + vegetation)")
    seg.add_argument(
        "--vegetation-model",
        choices=VEGETATION_MODELS,
        default=DEFAULT_VEGETATION_MODEL,
        help=f"Vegetation segmentation method (default: {DEFAULT_VEGETATION_MODEL})",
    )

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

    args = parser.parse_args()

    if args.command == "download":
        cmd_download(dry_run=args.dry_run)
    elif args.command == "segment":
        cmd_segment(vegetation_model=args.vegetation_model)
    elif args.command == "compare":
        cmd_compare(area_filter=args.area)
    elif args.command == "shadow":
        cmd_shadow(
            area_filter=args.area,
            vegetation_model=args.vegetation_model,
            datetime_utc=args.datetime_utc,
        )
    elif args.command == "tune":
        cmd_tune(
            model=args.model,
            tile_key=args.tile,
            area_filter=args.area,
            reference_model=args.reference,
        )
    elif args.command == "all":
        cmd_all(dry_run=args.dry_run, vegetation_model=args.vegetation_model, datetime_utc=args.datetime_utc)
    else:
        cmd_all()
