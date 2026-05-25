"""Hyperparameter tuning for vegetation segmentation methods.

Uses SegFormer-B5 as the reference mask (most stable semantic coverage)
and computes Dice + coverage for each parameter combination.
"""

from pathlib import Path

import numpy as np
import pandas as pd


def _dice(a: np.ndarray, b: np.ndarray) -> float:
    denom = a.sum() + b.sum()
    if denom == 0:
        return float("nan")
    return float(2 * (a & b).sum() / denom)


def tune_deepforest(
    img: np.ndarray,
    reference_mask: np.ndarray,
    model=None,
    score_thresholds: tuple = (0.1, 0.2, 0.3, 0.4, 0.5),
    patch_sizes: tuple = (300, 400, 500),
    patch_overlaps: tuple = (0.05, 0.15),
    out_dir: Path | None = None,
    stem: str = "deepforest_tune",
) -> pd.DataFrame:
    """Grid-search DeepForest hyperparameters against a reference mask.

    Parameters
    ----------
    img : np.ndarray
        Shape (H, W, 3), dtype uint8, RGB.
    reference_mask : np.ndarray
        Boolean (H, W) reference mask to score against (e.g. SegFormer-B5).
    model : optional
        Pre-loaded DeepForest model. Loaded once if None.
    score_thresholds : tuple of float
        Detection confidence thresholds to try.
    patch_sizes : tuple of int
        Sliding window sizes in pixels to try.
    patch_overlaps : tuple of float
        Window overlap fractions to try.
    out_dir : Path, optional
        Save results CSV to out_dir/stem_results.csv.
    stem : str
        Filename stem for saved outputs.

    Returns
    -------
    pd.DataFrame
        Sorted by dice_vs_reference descending. Columns:
        score_threshold, patch_size, patch_overlap, coverage_pct, dice_vs_reference.
    """
    from src.segmentation.vegetation import deepforest_mask, load_deepforest

    if model is None:
        model = load_deepforest()

    rows = []
    total = len(score_thresholds) * len(patch_sizes) * len(patch_overlaps)
    done = 0

    for patch_size in patch_sizes:
        for patch_overlap in patch_overlaps:
            # Run DeepForest once per (patch_size, overlap) — score filtering is cheap
            predictions = model.predict_tile(
                image=img,
                patch_size=patch_size,
                patch_overlap=patch_overlap,
                iou_threshold=0.15,
            )

            for score_thresh in score_thresholds:
                done += 1
                height, width = img.shape[:2]
                mask = np.zeros((height, width), dtype=np.uint8)

                if predictions is not None and len(predictions) > 0:
                    for _, row in predictions[predictions["score"] >= score_thresh].iterrows():
                        x0 = int(np.clip(row["xmin"], 0, width))
                        y0 = int(np.clip(row["ymin"], 0, height))
                        x1 = int(np.clip(row["xmax"], 0, width))
                        y1 = int(np.clip(row["ymax"], 0, height))
                        mask[y0:y1, x0:x1] = 1

                mask = mask.astype(bool)
                rows.append({
                    "score_threshold": score_thresh,
                    "patch_size": patch_size,
                    "patch_overlap": patch_overlap,
                    "coverage_pct": round(float(mask.mean() * 100), 2),
                    "dice_vs_reference": round(_dice(mask, reference_mask), 4),
                })
                print(f"  [{done}/{total}] patch={patch_size} overlap={patch_overlap} score={score_thresh}"
                      f" → coverage={rows[-1]['coverage_pct']:.1f}% dice={rows[-1]['dice_vs_reference']:.4f}")

    df = pd.DataFrame(rows).sort_values("dice_vs_reference", ascending=False).reset_index(drop=True)

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_dir / f"{stem}_results.csv", index=False)
        print(f"\nSaved to {out_dir / f'{stem}_results.csv'}")

    return df


def tune_samgeo(
    img: np.ndarray,
    reference_mask: np.ndarray,
    model=None,
    vari_thresholds: tuple = (0.02, 0.05, 0.08, 0.10, 0.15),
    min_segment_sizes: tuple = (100, 200, 500, 1000),
    out_dir: Path | None = None,
    stem: str = "samgeo_tune",
) -> pd.DataFrame:
    """Grid-search SamGeo hyperparameters against a reference mask.

    SAM is run once (expensive); only the VARI filter is varied per combination.

    Parameters
    ----------
    img : np.ndarray
        Shape (H, W, 3), dtype uint8, RGB.
    reference_mask : np.ndarray
        Boolean (H, W) reference mask to score against (e.g. SegFormer-B5).
    model : optional
        Pre-loaded SamGeo model. Loaded once if None.
    vari_thresholds : tuple of float
        Mean VARI threshold for segment inclusion.
    min_segment_sizes : tuple of int
        Minimum segment size in pixels to consider.
    out_dir : Path, optional
        Save results CSV to out_dir/stem_results.csv.
    stem : str
        Filename stem for saved outputs.

    Returns
    -------
    pd.DataFrame
        Sorted by dice_vs_reference descending.
    """
    from src.segmentation.vegetation import load_samgeo

    if model is None:
        model = load_samgeo()

    # Compute VARI once
    r = img[:, :, 0].astype(float)
    g = img[:, :, 1].astype(float)
    b = img[:, :, 2].astype(float)
    vari = (g - r) / (g + r - b + 1e-6)

    # Run SAM once — expensive step
    print("  Running SAM automatic mask generation (once)...")
    model.generate(img, output=None)
    segments = model.masks
    print(f"  {len(segments)} segments found.")

    height, width = img.shape[:2]
    rows = []
    total = len(vari_thresholds) * len(min_segment_sizes)
    done = 0

    for vari_thresh in vari_thresholds:
        for min_size in min_segment_sizes:
            done += 1
            mask = np.zeros((height, width), dtype=bool)
            for seg in segments:
                if seg["area"] < min_size:
                    continue
                if vari[seg["segmentation"]].mean() > vari_thresh:
                    mask |= seg["segmentation"]

            rows.append({
                "vari_threshold": vari_thresh,
                "min_segment_size": min_size,
                "coverage_pct": round(float(mask.mean() * 100), 2),
                "dice_vs_reference": round(_dice(mask, reference_mask), 4),
            })
            print(f"  [{done}/{total}] vari>{vari_thresh} min_size={min_size}"
                  f" → coverage={rows[-1]['coverage_pct']:.1f}% dice={rows[-1]['dice_vs_reference']:.4f}")

    df = pd.DataFrame(rows).sort_values("dice_vs_reference", ascending=False).reset_index(drop=True)

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_dir / f"{stem}_results.csv", index=False)
        print(f"\nSaved to {out_dir / f'{stem}_results.csv'}")

    return df
