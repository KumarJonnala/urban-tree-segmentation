# Urban Shadow Analysis, Magdeburg

Urban satellite image segmentation and shadow modelling for shaded routing.

---

## Phases

1. **Data acquisition** — DOP20 orthophotos (Sachsen-Anhalt WMS, 20 cm/px) tiled at configurable sizes (100 / 250 / 500 / 1000 m); building and road footprints from OpenStreetMap
2. **Urban segmentation** — 4-class pixel map: `other | tree | road | building` using one of 7 vegetation models (default: TCD SegFormer)
3. **Shadow modelling** — Per-tree shadow projection using solar position, allometric height estimation, and watershed crown separation

---

## Usage

```bash
# Full pipeline at default tile size (250m)
python pipeline.py

# Full pipeline for all tile sizes [100, 250, 500, 1000]
python pipeline.py --all-sizes

# Download only — with preview
python pipeline.py download --dry-run --all-sizes
python pipeline.py download --tile-size 500

# Segment and shadow at a specific size
python pipeline.py segment --tile-size 250
python pipeline.py shadow --datetime-utc "2026-06-21T09:00:00" --tile-size 250

# Hourly shadow table for one tile across a day
python pipeline.py diurnal --date-utc "2026-06-21" --tile 1_1 --tile-size 250

# Check what has been computed
python pipeline.py status --all-sizes

# Benchmark all vegetation models side-by-side
python pipeline.py compare

# Tune hyperparameters for a model against a reference
python pipeline.py tune --model vari --tile 0_0

# All options
python pipeline.py --help
```

Interactive exploration: open `test_notebooks/shadow_analysis.ipynb` in Jupyter.

---

## Vegetation models

| Model | Type | Notes |
|---|---|---|
| `tcd_segformer` | Transformer (aerial) | **Default** — `restor/tcd-segformer-mit-b5` |
| `vari` | Spectral index | Fast, no GPU, tends to over-segment |
| `segformer_b5` | Transformer (ADE20K) | Broad greenness, over-segments |
| `samgeo` | SAM + VARI filter | Precise boundaries, slow |
| `deepforest` | Crown detector | Low recall on German urban trees |
| `ensemble` | VARI ∩ DeepForest | High precision, low recall |
| `deeplab` | CNN (VOC) | Not usable on aerial imagery |

**Model references**

- **TCD SegFormer** — SegFormer-MiT-B5 fine-tuned for tree cover delineation by Restor (restor.eco). Architecture: Xie, S. et al. (2021) — see SegFormer below. Training data and metrics: [`restor/tcd-segformer-mit-b5`](https://huggingface.co/restor/tcd-segformer-mit-b5) *(verify specific training paper from model card before citing).*
- **SegFormer** — Xie, S. et al. (2021). *SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers.* NeurIPS 2021. mIoU 51.8% on ADE20K (B5). HuggingFace: [`nvidia/segformer-b5-finetuned-ade-640-640`](https://huggingface.co/nvidia/segformer-b5-finetuned-ade-640-640)
- **SAM** — Kirillov, A. et al. (2023). *Segment Anything.* ICCV 2023. [`facebookresearch/segment-anything`](https://github.com/facebookresearch/segment-anything)
- **SamGeo** — Wu, Q. & Osco, L. (2023). *samgeo: A Python package for segmenting geospatial data with the Segment Anything Model (SAM).* Journal of Open Source Software, 8(89), 5663. [`opengeos/segment-geospatial`](https://github.com/opengeos/segment-geospatial)
- **DeepForest** — Weinstein, B.G. et al. (2020). *DeepForest: A Python package for RGB deep learning tree crown delineation.* Methods in Ecology and Evolution, 11(12), 1743–1751. F1 ≈ 0.66 on NEON benchmark. [`weecology/DeepForest`](https://github.com/weecology/DeepForest)
- **DeepLab** — Chen, L.-C. et al. (2017). *Rethinking Atrous Convolution for Semantic Image Segmentation.* arXiv:1706.05587. torchvision: `deeplabv3_resnet50`
- **VARI** — Gitelson, A.A. et al. (2002). *Novel algorithms for remote estimation of vegetation fraction.* Remote Sensing of Environment, 80(1), 76–87.

---

## Shadow modelling

**Height estimation** — allometric formula per crown cluster:
```
crown_radius = sqrt(area / π)
height       = 2 × crown_radius × 0.7
```

**Watershed split** — clusters with `crown_radius > MAX_CROWN_RADIUS_M` (default 8 m, set in `config.py`) are split into individual crowns via distance-transform watershed before height estimation, preventing sqrt(N) inflation from merged trees.

**Shadow projection** — sun azimuth and elevation from `pysolar`; shadow swept as a continuous pixel stripe from canopy edge to tip; propagation stops at building walls.

**Outputs per tile** — segmentation mask (`.npy`), overlay (`.png`), tree polygons with height and crown attributes (`.fgb`), shadow mask (`.png` + `.fgb`).

**Shadow modelling references**

- **Solar position** — Reda, I. & Andreas, A. (2004). *Solar position algorithm for solar radiation applications.* Solar Energy, 76(5), 577–589. Implemented via [`pysolar`](https://pysolar.readthedocs.io)
- **Allometric height** — Jucker, T. et al. (2017). *Allometric equations for integrating remote sensing imagery into forest monitoring programmes.* Global Change Biology, 23(1), 177–190. *(Provides the general RS-allometric framework; the specific H = 0.7 × crown_diameter ratio used here should be verified against urban tree allometry literature, e.g. Pretzsch et al.)*
- **Watershed crown delineation** — Beucher, S. & Meyer, F. (1993). *The morphological approach to segmentation: the watershed transformation.* In Mathematical Morphology in Image Processing, pp. 433–481. Distance-transform watershed applied via `scipy.ndimage` + `skimage.segmentation.watershed`.
- **Shadow geometry** — Lindberg, F. & Grimmond, C.S.B. (2011). *The influence of vegetation and building morphology on shadow patterns and mean radiant temperatures in urban areas.* Theoretical and Applied Climatology, 105(3–4), 311–323.

---

## Project structure

```
urban-shadow-analysis/
├── test_notebooks/         # exploratory Jupyter notebooks
├── src/
│   ├── config.py           # WMS settings, TILE_SIZE_M, TILE_SIZES_M, model defaults
│   ├── data_preprocessing/ # orthophoto fetching + tiling + OSM building/road fetch
│   ├── segmentation/       # vegetation models, overlay composition, comparison, tuning
│   └── shadow/             # solar position, height estimation, shadow casting, viz
├── data/
│   ├── orthophotos/
│   │   ├── {area}/
│   │   │   ├── {area}_full.png   # full-area overview image
│   │   │   ├── 100m/             # tile grids per size
│   │   │   ├── 250m/
│   │   │   └── ...
│   │   ├── segments/
│   │   │   ├── 100m/
│   │   │   ├── 250m/
│   │   │   └── ...
│   │   └── shadows/
│   │       ├── 100m/
│   │       │   └── diurnal/      # hourly outputs from `diurnal` command
│   │       ├── 250m/
│   │       └── ...
│   ├── sentinel2/          # Sentinel-2 time-series GeoTIFFs
│   └── sam_checkpoints/    # SAM ViT-B weights
├── pipeline.py             # CLI: download / segment / compare / shadow / diurnal / status / tune / all
├── requirements.txt
└── README.md
```
