"""Vegetation detection: VARI spectral index and ML-based alternatives."""

import numpy as np
from skimage.morphology import closing, disk, remove_small_objects


def _best_device() -> str:
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# ---------------------------------------------------------------------------
# VARI — rule-based spectral baseline
# ---------------------------------------------------------------------------

def _vari(img: np.ndarray) -> np.ndarray:
    r = img[:, :, 0].astype(float)
    g = img[:, :, 1].astype(float)
    b = img[:, :, 2].astype(float)
    return (g - r) / (g + r - b + 1e-6)


def vari_mask(
    img: np.ndarray,
    threshold: float = 0.05,
    min_size: int = 500,
    closing_radius: int = 4,
) -> np.ndarray:
    """Vegetation mask from RGB image using the VARI spectral index.

    Parameters
    ----------
    img : np.ndarray
        Shape (H, W, 3), dtype uint8, channels in RGB order.
    threshold : float
        VARI value above which a pixel is classified as vegetation.
    min_size : int
        Minimum connected-component size in pixels to retain (exclusive).
        At DOP20 resolution (20 cm/px), 500 px ≈ 20 m².
    closing_radius : int
        Disk radius for morphological closing to fill small canopy gaps.

    Returns
    -------
    np.ndarray
        Boolean array of shape (H, W).
    """
    vari = _vari(img)
    raw = vari > threshold
    mask = remove_small_objects(raw, max_size=min_size - 1)
    mask = closing(mask, disk(closing_radius))
    return mask


# ---------------------------------------------------------------------------
# DeepForest — pre-trained aerial tree crown detector
# ---------------------------------------------------------------------------

def load_deepforest():
    """Load the DeepForest pretrained tree crown model. Call once per process and reuse."""
    from deepforest import main as df_main
    model = df_main.deepforest()
    model.load_model()  # loads default pretrained weights from HuggingFace config
    return model


def deepforest_mask(
    img: np.ndarray,
    model=None,
    score_threshold: float = 0.3,
    patch_size: int = 400,
    patch_overlap: float = 0.05,
    iou_threshold: float = 0.15,
    min_size: int = 500,
    closing_radius: int = 4,
) -> np.ndarray:
    """Vegetation mask via DeepForest tree crown bounding box detection.

    Parameters
    ----------
    img : np.ndarray
        Shape (H, W, 3), dtype uint8, RGB order.
    model : deepforest model, optional
        Pre-loaded via load_deepforest(). If None, loads on every call (slow).
    score_threshold : float
        Minimum detection confidence to include (0.3 keeps ~half of detections).
    patch_size : int
        Sliding window size in pixels for predict_tile().
    patch_overlap : float
        Fractional overlap between adjacent patches.
    iou_threshold : float
        IoU threshold for NMS across window boundaries.
    min_size, closing_radius : int
        Same post-processing as vari_mask for output consistency.

    Returns
    -------
    np.ndarray
        Boolean array of shape (H, W).
    """
    if model is None:
        model = load_deepforest()

    height, width = img.shape[:2]
    predictions = model.predict_tile(
        image=img,
        patch_size=patch_size,
        patch_overlap=patch_overlap,
        iou_threshold=iou_threshold,
    )

    mask = np.zeros((height, width), dtype=np.uint8)
    if predictions is None or len(predictions) == 0:
        return mask.astype(bool)

    for _, row in predictions[predictions["score"] >= score_threshold].iterrows():
        x0 = int(np.clip(row["xmin"], 0, width))
        y0 = int(np.clip(row["ymin"], 0, height))
        x1 = int(np.clip(row["xmax"], 0, width))
        y1 = int(np.clip(row["ymax"], 0, height))
        mask[y0:y1, x0:x1] = 1

    mask = mask.astype(bool)
    mask = remove_small_objects(mask, max_size=min_size - 1)
    mask = closing(mask, disk(closing_radius))
    return mask


# ---------------------------------------------------------------------------
# SegFormer — transformer-based semantic segmentation (ADE20K)
# ---------------------------------------------------------------------------

# ADE20K class indices relevant to vegetation
_SEGFORMER_VEGETATION_CLASSES = (4, 9, 17)  # tree, grass, plant


# ---------------------------------------------------------------------------
# SegFormer-B5 — transformer semantic segmentation, ADE20K
# ---------------------------------------------------------------------------

_SEGFORMER_B5_CHECKPOINT = "nvidia/segformer-b5-finetuned-ade-640-640"


def load_segformer_b5(device: str | None = None):
    """Load SegFormer-B5 finetuned on ADE20K. Returns (processor, model)."""
    from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
    proc = SegformerImageProcessor.from_pretrained(_SEGFORMER_B5_CHECKPOINT)
    mdl = SegformerForSemanticSegmentation.from_pretrained(_SEGFORMER_B5_CHECKPOINT)
    dev = device or _best_device()
    return proc, mdl.eval().to(dev)


def segformer_b5_mask(
    img: np.ndarray,
    processor=None,
    model=None,
    vegetation_classes: tuple = _SEGFORMER_VEGETATION_CLASSES,
    device: str | None = None,
) -> np.ndarray:
    """Vegetation mask via SegFormer-B5 (ADE20K).

    Combines ADE20K classes 4=tree, 9=grass, 17=plant into a single mask.

    Parameters
    ----------
    img : np.ndarray
        Shape (H, W, 3), dtype uint8, RGB order.
    processor, model : optional
        Pre-loaded via load_segformer_b5(). If None, loads on every call (slow).
    vegetation_classes : tuple of int
        ADE20K class indices to treat as vegetation.
    device : str, optional
        Torch device string. Defaults to MPS > CUDA > CPU.

    Returns
    -------
    np.ndarray
        Boolean array of shape (H, W).
    """
    import torch
    import torch.nn.functional as F
    from PIL import Image as PilImage

    dev = device or _best_device()
    if processor is None or model is None:
        processor, model = load_segformer_b5(device=dev)

    inputs = processor(images=PilImage.fromarray(img), return_tensors="pt")
    inputs = {k: v.to(dev) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits  # (1, 150, H/4, W/4)

    upsampled = F.interpolate(logits, size=img.shape[:2], mode="bilinear", align_corners=False)
    pred = upsampled.argmax(dim=1).squeeze().cpu().numpy()
    return np.isin(pred, list(vegetation_classes))


# ---------------------------------------------------------------------------
# SamGeo — SAM automatic mask generation filtered by VARI spectral signal
# ---------------------------------------------------------------------------

def load_samgeo():
    """Load SamGeo using the local ViT-B SAM checkpoint.

    Always runs on CPU — SAM's automatic mask generator requires float64, which
    MPS does not support.
    """
    from pathlib import Path
    from samgeo import SamGeo

    checkpoint = str(
        Path(__file__).resolve().parents[2] / "data" / "sam_checkpoints" / "sam_vit_b_01ec64.pth"
    )
    sam = SamGeo(
        model_type="vit_b",
        checkpoint=checkpoint,
        device="cpu",  # MPS lacks float64 support required by SAM
        automatic=True,
    )
    return sam


def samgeo_mask(
    img: np.ndarray,
    model=None,
    vari_threshold: float = 0.05,
    min_segment_size: int = 200,
) -> np.ndarray:
    """Vegetation mask via SAM automatic segmentation filtered by VARI.

    SAM provides precise object boundaries; VARI selects which segments are
    vegetation. This avoids VARI bleeding across object edges.

    Parameters
    ----------
    img : np.ndarray
        Shape (H, W, 3), dtype uint8, RGB order.
    model : SamGeo, optional
        Pre-loaded via load_samgeo(). If None, loads on every call.
    vari_threshold : float
        Minimum mean VARI within a segment to classify it as vegetation.
    min_segment_size : int
        Discard segments smaller than this many pixels.

    Returns
    -------
    np.ndarray
        Boolean array of shape (H, W).
    """
    if model is None:
        model = load_samgeo()

    vari = _vari(img)

    # Run SAM automatic mask generation (populates model.masks)
    model.generate(img, output=None)
    segments = model.masks  # list of dicts with 'segmentation' (bool H×W) and 'area'

    height, width = img.shape[:2]
    mask = np.zeros((height, width), dtype=bool)

    for seg in segments:
        region = seg["segmentation"]  # bool (H, W)
        if seg["area"] < min_segment_size:
            continue
        if vari[region].mean() > vari_threshold:
            mask |= region

    return mask


# ---------------------------------------------------------------------------
# TCD SegFormer — tree cover delineation, trained on global aerial imagery
# ---------------------------------------------------------------------------

_TCD_SEGFORMER_CHECKPOINT = "restor/tcd-segformer-mit-b5"
_TCD_TREE_CLASS = 1  # binary: 0=background, 1=tree


def load_tcd_segformer(
    model_id: str = _TCD_SEGFORMER_CHECKPOINT,
    device: str | None = None,
):
    """Load the TCD SegFormer model trained on aerial imagery. Returns (processor, model)."""
    from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
    proc = AutoImageProcessor.from_pretrained(model_id)
    mdl = SegformerForSemanticSegmentation.from_pretrained(model_id)
    dev = device or _best_device()
    return proc, mdl.eval().to(dev)


def tcd_segformer_mask(
    img: np.ndarray,
    processor=None,
    model=None,
    device: str | None = None,
    resize_to: int | None = 1024,
) -> np.ndarray:
    """Tree cover mask via TCD SegFormer (restor/tcd-segformer-mit-b5).

    Trained on global high-resolution aerial imagery (~10 cm/px). Outputs a
    binary tree/no-tree mask.  The model is scale-sensitive: resize_to crops
    inference to a safe tile size before upsampling back to original shape.

    Parameters
    ----------
    img : np.ndarray
        Shape (H, W, 3), dtype uint8, RGB order.
    processor, model : optional
        Pre-loaded via load_tcd_segformer(). If None, loads on every call.
    device : str, optional
        Torch device string. Defaults to MPS > CUDA > CPU.
    resize_to : int or None
        Resize the longer side to this value before inference, then upsample
        output back to original shape. None disables resizing.

    Returns
    -------
    np.ndarray
        Boolean array of shape (H, W). True = tree.
    """
    import torch
    import torch.nn.functional as F
    from PIL import Image as PilImage

    dev = device or _best_device()
    if processor is None or model is None:
        processor, model = load_tcd_segformer(device=dev)

    orig_h, orig_w = img.shape[:2]
    pil_img = PilImage.fromarray(img)

    if resize_to is not None:
        scale = resize_to / max(orig_h, orig_w)
        new_h, new_w = int(orig_h * scale), int(orig_w * scale)
        pil_img = pil_img.resize((new_w, new_h), PilImage.BILINEAR)

    inputs = processor(images=pil_img, return_tensors="pt")
    inputs = {k: v.to(dev) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits  # (1, 2, H', W')

    upsampled = F.interpolate(
        logits, size=(orig_h, orig_w), mode="bilinear", align_corners=False
    )
    pred = upsampled.argmax(dim=1).squeeze().cpu().numpy()
    return pred == _TCD_TREE_CLASS


# ---------------------------------------------------------------------------
# Ensemble — VARI ∩ / ∪ DeepForest
# ---------------------------------------------------------------------------

def ensemble_mask(
    img: np.ndarray,
    df_model=None,
    mode: str = "intersection",
    vari_threshold: float = 0.05,
    score_threshold: float = 0.3,
    min_size: int = 500,
    closing_radius: int = 4,
) -> np.ndarray:
    """Tree mask by combining VARI and DeepForest predictions.

    Parameters
    ----------
    img : np.ndarray
        Shape (H, W, 3), dtype uint8, RGB order.
    df_model : optional
        Pre-loaded DeepForest model. Loaded on first call if None.
    mode : str
        ``"intersection"`` (VARI ∩ DeepForest) — high precision, fewer
        false positives from grass/shrubs.
        ``"union"`` (VARI ∪ DeepForest) — high recall, fewer missed crowns.
    vari_threshold, min_size, closing_radius : float / int
        Forwarded to vari_mask().
    score_threshold : float
        Forwarded to deepforest_mask().

    Returns
    -------
    np.ndarray
        Boolean array of shape (H, W).
    """
    vari = vari_mask(img, threshold=vari_threshold, min_size=min_size,
                     closing_radius=closing_radius)
    df = deepforest_mask(img, model=df_model, score_threshold=score_threshold,
                         min_size=min_size, closing_radius=closing_radius)
    if mode == "intersection":
        return vari & df
    return vari | df


# ---------------------------------------------------------------------------
# DeepLab — torchvision DeepLabV3 ResNet50 (PASCAL VOC / COCO)
# ---------------------------------------------------------------------------

# VOC class 16 = pottedplant; only available vegetation proxy in this 21-class model.
# Performance on aerial orthophotos is expected to be poor — included for comparison.
_DEEPLAB_VEGETATION_CLASS = 16  # pottedplant


def load_deeplab(device: str | None = None):
    """Load DeepLabV3 ResNet50 pretrained on COCO/VOC."""
    from torchvision.models.segmentation import deeplabv3_resnet50
    dev = device or _best_device()
    model = deeplabv3_resnet50(weights="DEFAULT")
    return model.eval().to(dev)


def deeplab_mask(
    img: np.ndarray,
    model=None,
    vegetation_class: int = _DEEPLAB_VEGETATION_CLASS,
    device: str | None = None,
) -> np.ndarray:
    """Vegetation mask via DeepLabV3 ResNet50 (VOC class 16 = pottedplant).

    Note: VOC has no general outdoor vegetation class. Class 16 (pottedplant)
    is the only vegetation proxy available. Expect low recall on aerial imagery.

    Parameters
    ----------
    img : np.ndarray
        Shape (H, W, 3), dtype uint8, RGB order.
    model : optional
        Pre-loaded via load_deeplab(). If None, loads on every call.
    vegetation_class : int
        VOC class index to use as vegetation (default 16 = pottedplant).
    device : str, optional
        Torch device string.

    Returns
    -------
    np.ndarray
        Boolean array of shape (H, W).
    """
    import torch
    import torchvision.transforms.functional as TF

    dev = device or _best_device()
    if model is None:
        model = load_deeplab(device=dev)

    # Normalize to ImageNet stats expected by torchvision models
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    tensor = TF.to_tensor(img)  # (3, H, W) float32 in [0,1]
    tensor = TF.normalize(tensor, mean=mean, std=std).unsqueeze(0).to(dev)

    with torch.no_grad():
        out = model(tensor)["out"]  # (1, 21, H, W)

    pred = out.argmax(dim=1).squeeze().cpu().numpy()
    return pred == vegetation_class
