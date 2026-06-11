from __future__ import annotations
"""Shadow overlay visualisation."""

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from src.segmentation.overlay import CLASS_COLORS, CLASS_LABELS, _OVERLAY_ALPHA

SHADOW_COLOR = [0.10, 0.10, 0.30]
SHADOW_ALPHA = 0.55


def save_shadow_overlay(
    img: np.ndarray,
    seg_map: np.ndarray,
    shadow_mask: np.ndarray,
    out_path: Path,
    title: str | None = None,
) -> Path:
    """Save orthophoto with segmentation + shadow overlay as a PNG.

    Rendering order: class colours first (trees green, roads grey, buildings red),
    then shadow layer on top (dark navy). Tree pixels are excluded from the shadow
    mask by the caller, so there is no colour conflict.

    Parameters
    ----------
    img : np.ndarray
        uint8 (H, W, 3) RGB orthophoto.
    seg_map : np.ndarray
        uint8 (H, W) segmentation map (same class encoding as overlay.py).
    shadow_mask : np.ndarray
        bool (H, W) — True where shadow falls.
    out_path : Path
        Full output path including filename (e.g. .../shadows/tile_0_0_shadow.png).
    title : str, optional
        Figure title.

    Returns
    -------
    Path
        Resolved out_path.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    H, W = seg_map.shape
    n_classes = max(CLASS_COLORS) + 1
    rgb_lut = np.array([CLASS_COLORS[k] for k in range(n_classes)], dtype=float)
    alpha_lut = np.array([_OVERLAY_ALPHA[k] for k in range(n_classes)], dtype=float)

    overlay_rgb = rgb_lut[seg_map]
    overlay_alpha = alpha_lut[seg_map, np.newaxis]
    overlay_rgba = np.concatenate([overlay_rgb, overlay_alpha], axis=-1)

    # Apply shadow layer on top
    overlay_rgba[shadow_mask] = [*SHADOW_COLOR, SHADOW_ALPHA]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img)
    ax.imshow(overlay_rgba)
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=10)

    legend_patches = [
        mpatches.Patch(color=CLASS_COLORS[k], label=CLASS_LABELS[k])
        for k in sorted(CLASS_LABELS)
    ]
    legend_patches.append(mpatches.Patch(color=SHADOW_COLOR, label="Shadow"))
    ax.legend(handles=legend_patches, loc="lower right", fontsize=8,
              framealpha=0.8, ncol=2)

    fig.tight_layout(pad=0)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return out_path
