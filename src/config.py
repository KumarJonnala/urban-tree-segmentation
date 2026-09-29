"""Project configuration: geographic areas and global settings."""

from pathlib import Path

# WMS orthogonal images data source
WMS_URL   = "https://www.geodatenportal.sachsen-anhalt.de/wss/service/ST_LVermGeo_DOP_WMS_OpenData/guest"
WMS_LAYER = "lsa_lvermgeo_dop20_2"

# Default output resolution for fetched tiles (pixels)
IMAGE_WIDTH  = 1200
IMAGE_HEIGHT = 1200

# Root directory for downloaded orthophotos
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data" / "orthophotos"

# Tile size (meters) for subdivision of areas into grid, optimmal size needs to be determined experimentally
TILE_SIZE_M = 250

# All tile sizes (meters) used for multi-size comparative analysis
TILE_SIZES_M = [100, 250, 500, 1000]

# Default vegetation segmentation model used by segment, shadow, and all subcommands
DEFAULT_VEGETATION_MODEL = "tcd_segformer"

# Maximum single-tree crown radius (metres) for height estimation.
# Components exceeding this are watershed-split into individual crowns before
# applying the allometric formula. 8 m ≈ 200 m² crown area.
MAX_CROWN_RADIUS_M = 8.0

# --- Vectorisation and watershed splitting (src/shadow/casting.py) ---

# Connected components smaller than this many pixels are discarded as noise,
# both when vectorising and when accepting watershed sub-regions. At the 250 m
# tile resolution (~0.21 m/px) 50 px ≈ 2.2 m² of canopy.
MIN_COMPONENT_PIXELS = 50

# Minimum separation between watershed seed peaks, as a fraction of the
# applicable crown radius. Peaks closer than
# max_crown_radius_m * this / pixel_size_m are merged into one crown, so lower
# values split more aggressively.
WATERSHED_PEAK_SEP_FRAC = 0.5

# --- Shadow casting (src/shadow/casting.py) ---

# Sun elevation floor (degrees). At or below this — including night — the tile
# produces an all-False shadow mask rather than implausibly long shadows.
MIN_SUN_ELEVATION_DEG = 5.0

# Shadow length is capped at this multiple of the crown radius, which bounds
# the shadow cast by a low sun (length grows as 1/tan(elevation)).
MAX_SHADOW_FACTOR = 5.0

# Optional Baumkataster GeoPackage for enriching tree FGBs with measured heights/species.
# Set to None to disable enrichment (pipeline runs without it).
BAUMKATASTER_PATH = Path(__file__).resolve().parents[1] / "references" / "Baeume_SFM_2026.gpkg"

# All tree registries, merged on load by load_baumkataster(). They are complementary
# rather than overlapping: SFM covers public green and street trees (85,302 points),
# Liegenschaftsservice covers municipal-property trees (5,665). No LS point lies within
# 1 m of an SFM point citywide, so the union double-counts nothing. LS contributes zero
# trees inside ovgu_bbox and 26 inside ovgu_bbox2.
BAUMKATASTER_PATHS = [
    BAUMKATASTER_PATH,
    Path(__file__).resolve().parents[1] / "references" / "Baeume_Liegenschaftsservice_2026.gpkg",
]

# --- Baumkataster matching (src/shadow/cadastre.py) ---

# A pipeline crown polygon matches the nearest BK tree whose buffered point it
# intersects, provided the polygon centroid lies within this distance of that
# point. Observed matches on ovgu_bbox @ 250 m sit well inside it
# (median 1.2 m, max 14.2 m), so it is a safety ceiling rather than a tight fit.
BK_MATCH_RADIUS_M = 15.0

# BK records below either floor are treated as unmeasured and excluded from
# matching: they carry placeholder zeros rather than real survey values.
BK_MIN_HEIGHT_M = 1.0        # Baumhoehe (m)
BK_MIN_CROWN_DIAM_M = 0.5    # Kronendurchmesser (m)

# BK points are buffered by max(Kronendurchmesser/2, this) to generate match
# candidates. The floor keeps small-crowned or newly planted trees reachable.
BK_MATCH_BUFFER_MIN_M = 2.0

# Allometric height model: H = exp(ALLOMETRIC_A + ALLOMETRIC_B * ln(CPA_m2))
# Power-law form: direct OLS of ln(H) ~ ln(CPA) on Magdeburg Baumkataster 2026
# (n=84,081 trees, all species, CPA = π(Kronendurchmesser/2)²). Not street trees only:
# by Objektart lang that population is 63% Öffentliches Grün / 36% AMT 66 (street) / 2%
# Spielplatz, and park and street trees have measurably different allometry — see below.
# R²=0.53 (ln-space); predictions match cadastre medians within ±10% for crown diameters 4–12 m.
# Source: Baeume_SFM_2026.gpkg (references/), fitted in test_notebooks/shadow_analysis.ipynb.
#
# Held out on ovgu_bbox (n=1,045 registry trees), 5-fold CV, seed 42 — see
# test_notebooks/holdout_validation.ipynb:
#   out-of-fold ln-R² +0.661 vs +0.656 scored circularly, so the fit is NOT overfit;
#   it also sits at 93% of the +0.708 ceiling any function of CPA alone could reach,
#   because Kronendurchmesser is field-estimated to the nearest metre.
#   In metres the error is a tilt rather than scatter: trees under 10 m come out ~2 m
#   too tall and trees over 22 m ~6 m too short, so the largest under-prediction lands
#   on the trees that cast the longest shadows.
#   A local refit would give A=1.169, B=0.350; not adopted (it buys ~0.005 out-of-fold
#   and would invalidate allometric_height_m / h_global_m in every FGB until re-segment).
ALLOMETRIC_A = 1.317   # ln-space intercept
ALLOMETRIC_B = 0.318   # power-law exponent on CPA (m²)

# --- Vegetation segmentation models (src/segmentation/vegetation.py) ---
# Only the TCD_SEGFORMER_* value affects the default pipeline
# (DEFAULT_VEGETATION_MODEL = "tcd_segformer"); the rest apply to the
# alternative models reachable via --vegetation-model.

# TCD SegFormer is scale-sensitive: inference runs on the image resized so its
# longer side is this many pixels, then the mask is upsampled back. Set to None
# to run at native resolution.
TCD_SEGFORMER_RESIZE_TO = 1024

# VARI (Visible Atmospherically Resistant Index) vegetation masking. Shared by
# vari_mask, samgeo_mask and ensemble_mask. Tunable via tune_vari()
# in src/segmentation/tuning.py, which overrides these with its own grid.
VARI_THRESHOLD = 0.05        # index value above which a pixel counts as vegetation
VARI_MIN_SIZE = 500          # drop mask components below this pixel count
VARI_CLOSING_RADIUS = 4      # morphological closing disk radius (px)

# DeepForest crown detection. Inference is tiled: patch_size with patch_overlap
# fraction, then boxes are merged by non-max suppression at iou_threshold.
DEEPFOREST_SCORE_THRESHOLD = 0.3
DEEPFOREST_PATCH_SIZE = 400
DEEPFOREST_PATCH_OVERLAP = 0.05
DEEPFOREST_IOU_THRESHOLD = 0.15

# SAM-geo: discard segments below this area (px) before the VARI greenness test.
SAMGEO_MIN_SEGMENT_SIZE = 200

# Per-genus allometric profiles: (A, B) pairs for H = exp(A + B*ln(CPA_m2)).
# Keyed by Gattung-lang value (full species string). Falls back to ALLOMETRIC_A/B if absent.
# Fitted via OLS ln(H) ~ ln(CPA) on Baumkataster 2026 (n_min=200 per species).
ALLOMETRIC_PROFILES: dict[str, tuple[float, float]] = {
    "Tilia cordata": (1.0799, 0.3864),  # R²=0.665, n=10435
    "Acer platanoides": (1.5002, 0.2716),  # R²=0.47, n=8119
    "Robinia pseudoacacia": (1.8419, 0.2185),  # R²=0.31, n=5779
    "Fraxinus excelsior": (1.4301, 0.3106),  # R²=0.5, n=5607
    "Quercus robur": (1.3798, 0.312),  # R²=0.572, n=4858
    "Acer pseudoplatanus": (1.5826, 0.2594),  # R²=0.419, n=4031
    "Acer campestre": (1.4289, 0.2706),  # R²=0.483, n=3924
    "Aesculus hippocastanum": (1.3895, 0.3096),  # R²=0.569, n=2479
    "Carpinus betulus": (1.3151, 0.3183),  # R²=0.592, n=2196
    "Prunus avium": (1.3153, 0.2522),  # R²=0.428, n=1750
    "Acer negundo": (1.5969, 0.2079),  # R²=0.327, n=1736
    "Platanus acerifolia": (1.1471, 0.3503),  # R²=0.652, n=1510
    "Pyrus communis": (1.1895, 0.2476),  # R²=0.558, n=1225
    "Tilia platyphyllos": (1.5648, 0.2706),  # R²=0.46, n=1209
    "Populus canadensis Hybride": (2.241, 0.2012),  # R²=0.29, n=1109
    "Tilia cordata 'Greenspire'": (1.4216, 0.2484),  # R²=0.687, n=863
    "Ulmus laevis": (1.6273, 0.2622),  # R²=0.508, n=849
    "Tilia euchlora": (1.2496, 0.3605),  # R²=0.602, n=838
    "Populus nigra 'Italica'": (1.7121, 0.471),  # R²=0.658, n=761
    "Malus spec.": (1.039, 0.1942),  # R²=0.357, n=757
    "Pinus sylvestris": (2.4931, 0.1042),  # R²=0.082, n=737
    "Prunus padus": (1.4955, 0.1688),  # R²=0.365, n=694
    "Populus nigra": (1.9586, 0.2363),  # R²=0.515, n=686
    "Salix alba": (1.3447, 0.2721),  # R²=0.402, n=670
    "Acer platanoides 'Columnare'": (1.4811, 0.2896),  # R²=0.625, n=667
    "Carpinus betulus 'Fastigiata'": (1.4099, 0.2381),  # R²=0.424, n=629
    "Pinus nigra": (1.8269, 0.3129),  # R²=0.466, n=628
    "Populus canadensis": (1.8341, 0.2743),  # R²=0.556, n=591
    "Betula pendula": (1.6957, 0.266),  # R²=0.424, n=589
    "Crataegus monogyna": (1.4232, 0.1304),  # R²=0.135, n=481
    "Quercus rubra": (1.1, 0.3689),  # R²=0.691, n=479
    "Juglans regia": (1.1396, 0.2878),  # R²=0.514, n=456
    "Ailanthus altissima": (1.376, 0.3031),  # R²=0.574, n=420
    "Corylus colurna": (1.2245, 0.3268),  # R²=0.567, n=415
    "Prunus spec.": (1.2495, 0.2267),  # R²=0.431, n=374
    "Prunus serrulata 'Kanzan'": (1.336, 0.1607),  # R²=0.441, n=331
    "Styphnolobium japonicum": (1.1277, 0.2794),  # R²=0.769, n=322
    "Malus sylvestris (communis)": (0.9744, 0.223),  # R²=0.346, n=313
    "Alnus glutinosa": (1.5591, 0.2703),  # R²=0.46, n=302
    "Populus spec.": (1.7247, 0.2763),  # R²=0.597, n=300
    "Prunus mahaleb": (1.169, 0.2582),  # R²=0.402, n=280
    "Sorbus aria": (1.0874, 0.3165),  # R²=0.463, n=274
    "Aesculus carnea": (0.9382, 0.3378),  # R²=0.577, n=272
    "Ulmus glabra": (1.4804, 0.2898),  # R²=0.633, n=263
    "Gleditsia triacanthos": (1.3762, 0.3001),  # R²=0.688, n=256
    "Salix spec.": (1.5703, 0.2418),  # R²=0.378, n=249
    "Fraxinus ornus": (1.2656, 0.2218),  # R²=0.554, n=242
    "Acer platanoides 'Globosum'": (0.8476, 0.2581),  # R²=0.601, n=241
    "Pyrus calleryana 'Chanticleer'": (1.3303, 0.2133),  # R²=0.417, n=241
    "Populus canescens": (1.6423, 0.3165),  # R²=0.527, n=238
    "Tilia tomentosa": (1.1331, 0.3086),  # R²=0.752, n=238
    "Tilia spec.": (1.321, 0.326),  # R²=0.643, n=237
    "Sorbus aucuparia": (1.2865, 0.2072),  # R²=0.497, n=219
    "Sorbus intermedia": (1.1311, 0.2875),  # R²=0.529, n=219
    "Quercus robur 'Fastigiata'": (1.7241, 0.3339),  # R²=0.758, n=218
    "Liquidambar styraciflua": (1.3298, 0.2521),  # R²=0.803, n=215
    "Ulmus carpinifolia": (1.5622, 0.2665),  # R²=0.382, n=206
}

# Per-genus 95th-percentile crown radius (m) used as watershed split threshold.
# Keyed by Gattung-lang value. Falls back to MAX_CROWN_RADIUS_M if absent.
# Using p95 so the threshold is tight enough that single large crowns still merge,
# while true multi-tree clusters get split. Computed from Baumkataster 2026 (n_min=200).
CROWN_RADIUS_BY_GENUS: dict[str, float] = {
    "Tilia cordata": 6.5,  # median=3.5 m, n=10444
    "Acer platanoides": 6.5,  # median=3.5 m, n=8120
    "Robinia pseudoacacia": 6.5,  # median=3.5 m, n=5781
    "Fraxinus excelsior": 7.5,  # median=3.5 m, n=5609
    "Quercus robur": 9.0,  # median=4.0 m, n=4923
    "Acer pseudoplatanus": 7.0,  # median=3.5 m, n=4033
    "Acer campestre": 6.0,  # median=3.0 m, n=3925
    "Aesculus hippocastanum": 7.0,  # median=3.5 m, n=2483
    "Carpinus betulus": 6.5,  # median=3.0 m, n=2197
    "Prunus avium": 5.0,  # median=2.5 m, n=1752
    "Acer negundo": 7.5,  # median=4.0 m, n=1736
    "Platanus acerifolia": 8.5,  # median=4.0 m, n=1510
    "Pyrus communis": 5.0,  # median=2.0 m, n=1225
    "Tilia platyphyllos": 7.0,  # median=4.0 m, n=1209
    "Populus canadensis Hybride": 8.8,  # median=4.0 m, n=1109
    "Tilia cordata 'Greenspire'": 5.0,  # median=3.0 m, n=863
    "Ulmus laevis": 7.5,  # median=3.0 m, n=849
    "Tilia euchlora": 7.0,  # median=4.5 m, n=838
    "Populus nigra 'Italica'": 4.0,  # median=1.5 m, n=761
    "Malus spec.": 4.0,  # median=2.0 m, n=757
    "Pinus sylvestris": 4.0,  # median=2.5 m, n=737
    "Prunus padus": 5.0,  # median=2.5 m, n=694
    "Populus nigra": 10.0,  # median=5.0 m, n=686
    "Salix alba": 10.0,  # median=5.0 m, n=671
    "Acer platanoides 'Columnare'": 4.0,  # median=2.5 m, n=667
    "Carpinus betulus 'Fastigiata'": 4.3,  # median=2.5 m, n=630
    "Pinus nigra": 5.3,  # median=3.0 m, n=629
    "Populus canadensis": 8.0,  # median=4.0 m, n=592
    "Betula pendula": 5.0,  # median=3.0 m, n=589
    "Crataegus monogyna": 5.0,  # median=2.5 m, n=481
    "Quercus rubra": 11.0,  # median=4.0 m, n=479
    "Juglans regia": 6.0,  # median=3.0 m, n=456
    "Ailanthus altissima": 7.5,  # median=4.0 m, n=420
    "Corylus colurna": 5.0,  # median=3.0 m, n=415
    "Prunus spec.": 5.0,  # median=2.5 m, n=374
    "Prunus serrulata 'Kanzan'": 4.5,  # median=3.0 m, n=331
    "Styphnolobium japonicum": 7.0,  # median=3.5 m, n=322
    "Malus sylvestris (communis)": 4.0,  # median=2.5 m, n=313
    "Alnus glutinosa": 5.0,  # median=2.5 m, n=302
    "Populus spec.": 8.0,  # median=3.5 m, n=300
    "Prunus mahaleb": 5.0,  # median=3.0 m, n=280
    "Sorbus aria": 3.0,  # median=2.0 m, n=274
    "Aesculus carnea": 6.0,  # median=3.5 m, n=272
    "Ulmus glabra": 7.5,  # median=3.0 m, n=263
    "Gleditsia triacanthos": 8.0,  # median=5.0 m, n=256
    "Salix spec.": 9.0,  # median=4.0 m, n=249
    "Fraxinus ornus": 4.5,  # median=2.5 m, n=242
    "Pyrus calleryana 'Chanticleer'": 3.0,  # median=2.0 m, n=241
    "Acer platanoides 'Globosum'": 4.0,  # median=2.5 m, n=241
    "Populus canescens": 8.0,  # median=4.0 m, n=238
    "Tilia tomentosa": 6.0,  # median=3.5 m, n=238
    "Tilia spec.": 4.1,  # median=1.8 m, n=237
    "Sorbus aucuparia": 3.0,  # median=2.0 m, n=219
    "Sorbus intermedia": 4.5,  # median=2.5 m, n=219
    "Quercus robur 'Fastigiata'": 5.6,  # median=1.9 m, n=218
    "Liquidambar styraciflua": 4.0,  # median=2.0 m, n=215
    "Ulmus carpinifolia": 6.0,  # median=2.5 m, n=206
}

# Geographic areas to fetch (bounding boxes in WGS84 lat/lon)
AREAS = {
    "ovgu_bbox": {
        "west":  11.639779,
        "east":  11.652739,
        "south": 52.137663,
        "north": 52.145538,
    },

    "ovgu_bbox2": {
        "west":  11.630389,
        "east":  11.662135,
        "south": 52.131955,
        "north": 52.151244,
    },

    # "magdeburg_bbox": {
    #     "west": 11.5,
    #     "east": 12.0,
    #     "south": 52.0,
    #     "north": 52.3,
    # },
}