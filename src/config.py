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

# Optional Baumkataster GeoPackage for enriching tree FGBs with measured heights/species.
# Set to None to disable enrichment (pipeline runs without it).
BAUMKATASTER_PATH = Path(__file__).resolve().parents[1] / "references" / "Baeume_SFM_2026.gpkg"

# Allometric height model: H = exp(ALLOMETRIC_A + ALLOMETRIC_B * ln(CPA_m2))
# Power-law form: direct OLS of ln(H) ~ ln(CPA) on Magdeburg Baumkataster 2026
# (n=84,081 street trees, all species, CPA = π(Kronendurchmesser/2)²).
# R²=0.53; predictions match cadastre medians within ±10% for crown diameters 4–12 m.
# Source: Baeume_SFM_2026.gpkg (references/), fitted in test_notebooks/shadow_analysis.ipynb.
ALLOMETRIC_A = 1.317   # ln-space intercept
ALLOMETRIC_B = 0.318   # power-law exponent on CPA (m²)

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
    # "ovgu_bbox": {
    #     "west":  11.639779,
    #     "east":  11.652739,
    #     "south": 52.137663,
    #     "north": 52.145538,
    # },
    
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