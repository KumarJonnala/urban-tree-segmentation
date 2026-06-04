# Urban Shadow Analysis, Magdeburg

Urban satellite image segmentation and shadow modelling for shaded routing.

---

## Phases

1. **Data acquisition** — DOP20 orthophotos (Sachsen-Anhalt WMS, 20 cm/px) tiled at 250 m × 250 m; building and road footprints from OpenStreetMap
2. **Urban segmentation** — 4-class pixel map: `other | tree | road | building` using one of 7 vegetation models (default: TCD SegFormer)
3. **Shadow modelling** — Per-tree shadow projection using solar position, allometric height estimation, and watershed crown separation

---

## Usage

```bash
# Download, segment, and cast shadows (default model: tcd_segformer)
python pipeline.py

# Choose a vegetation model
python pipeline.py all --vegetation-model vari

# Cast shadows for a specific datetime
python pipeline.py shadow --datetime-utc "2026-05-21T11:00:00" --area ovgu_bbox

# Benchmark all vegetation models side-by-side
python pipeline.py compare

# Tune hyperparameters for a model against a reference
python pipeline.py tune --model vari --tile 0_0

# Preview tile layout without downloading
python pipeline.py download --dry-run

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

---

## Project structure

```
urban-shadow-analysis/
├── test_notebooks/         # exploratory Jupyter notebooks
├── src/
│   ├── config.py           # WMS settings, tile size, model defaults, MAX_CROWN_RADIUS_M
│   ├── data_preprocessing/ # orthophoto fetching + tiling + OSM building/road fetch
│   ├── segmentation/       # vegetation models, overlay composition, comparison, tuning
│   └── shadow/             # solar position, height estimation, shadow casting, viz
├── data/
│   ├── orthophotos/        # downloaded tiles, segmentation masks, shadow outputs
│   ├── sentinel2/          # Sentinel-2 time-series GeoTIFFs
│   └── sam_checkpoints/    # SAM ViT-B weights
├── pipeline.py             # CLI: download / segment / compare / shadow / tune / all
├── requirements.txt
└── README.md
```

---

## Reference

> Lindberg, F. et al. — *Modelling sunlight and shading distribution on 3D Trees and Buildings*
