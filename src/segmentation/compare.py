from __future__ import annotations
"""Compare vegetation segmentation methods: metrics and visual output."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _dice(a: np.ndarray, b: np.ndarray) -> float:
    denom = a.sum() + b.sum()
    if denom == 0:
        return float("nan")
    return float(2 * (a & b).sum() / denom)


def compare_vegetation(
    img: np.ndarray,
    methods: list | tuple = ("vari", "deepforest", "segformer_b5", "deeplab"),
    out_dir: Path | None = None,
    stem: str = "comparison",
    models: dict | None = None,
) -> dict:
    """Run multiple vegetation segmentation methods and compare their outputs.

    Parameters
    ----------
    img : np.ndarray
        Shape (H, W, 3), dtype uint8, RGB order.
    methods : sequence of str
        Subset of {"vari", "deepforest", "segformer_b5", "samgeo", "deeplab"} to run.
    out_dir : Path, optional
        Directory to save {stem}_comparison.png. Skipped if None.
    stem : str
        Filename prefix for saved outputs.
    models : dict, optional
        Pre-loaded models keyed by method name. Supported keys:
          "deepforest"   → deepforest model
          "segformer_b5" → (processor, model) tuple
          "samgeo"       → SamGeo model
          "deeplab"      → deeplab model
        Methods not present in models are loaded on-the-fly (slow).

    Returns
    -------
    dict with keys:
      "masks"   : {method: bool (H,W)} for each method run
      "metrics" : pandas DataFrame with coverage_pct and pairwise Dice columns
    """
    from src.segmentation.vegetation import (
        deepforest_mask,
        deeplab_mask,
        ensemble_mask,
        samgeo_mask,
        segformer_b5_mask,
        tcd_segformer_mask,
        vari_mask,
    )

    models = models or {}
    masks = {}

    for method in methods:
        if method == "vari":
            masks["vari"] = vari_mask(img)

        elif method == "deepforest":
            masks["deepforest"] = deepforest_mask(img, model=models.get("deepforest"))

        elif method == "segformer_b5":
            proc, mdl = models.get("segformer_b5", (None, None))
            masks["segformer_b5"] = segformer_b5_mask(img, processor=proc, model=mdl)

        elif method == "samgeo":
            masks["samgeo"] = samgeo_mask(img, model=models.get("samgeo"))

        elif method == "deeplab":
            masks["deeplab"] = deeplab_mask(img, model=models.get("deeplab"))

        elif method == "tcd_segformer":
            proc, mdl = models.get("tcd_segformer", (None, None))
            masks["tcd_segformer"] = tcd_segformer_mask(img, processor=proc, model=mdl)

        elif method == "ensemble":
            masks["ensemble"] = ensemble_mask(img, df_model=models.get("ensemble"))

        else:
            raise ValueError(f"Unknown method: {method!r}")

    # --- metrics ---
    method_names = list(masks.keys())
    rows = []
    for name in method_names:
        m = masks[name]
        row = {"method": name, "coverage_pct": round(float(m.mean() * 100), 2)}
        for other in method_names:
            if other != name:
                row[f"dice_vs_{other}"] = round(_dice(m, masks[other]), 4)
        rows.append(row)
    metrics_df = pd.DataFrame(rows).set_index("method")

    # --- visual comparison ---
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        n = len(masks)
        fig, axes = plt.subplots(1, n + 1, figsize=(4 * (n + 1), 4))
        axes[0].imshow(img)
        axes[0].set_title("Orthophoto")
        axes[0].axis("off")

        for ax, (name, mask) in zip(axes[1:], masks.items()):
            overlay = np.zeros((*img.shape[:2], 4), dtype=float)
            overlay[mask] = [0.15, 0.75, 0.15, 0.6]
            ax.imshow(img)
            ax.imshow(overlay)
            cov = metrics_df.loc[name, "coverage_pct"]
            ax.set_title(f"{name}\n{cov:.1f}% coverage")
            ax.axis("off")

        fig.tight_layout()
        save_path = out_dir / f"{stem}_comparison.png"
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {save_path.name}")

    return {"masks": masks, "metrics": metrics_df}
