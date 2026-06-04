"""Project configuration: geographic areas and global settings."""

from pathlib import Path

# WMS orthogonal images data source
WMS_URL   = "https://www.geodatenportal.sachsen-anhalt.de/wss/service/ST_LVermGeo_DOP_WMS_OpenData/guest"
WMS_LAYER = "lsa_lvermgeo_dop20_2"

# Default output resolution for fetched tiles (pixels), 1200 for smaller preview tiles and 4096 for full 20cm tiles, 
IMAGE_WIDTH  = 1200
IMAGE_HEIGHT = 1200

# Root directory for downloaded orthophotos
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data" / "orthophotos"

# Tile size (meters) for subdivision of areas into grid, optimmal size needs to be determined experimentally
TILE_SIZE_M = 250

# Default vegetation segmentation model used by segment, shadow, and all subcommands
DEFAULT_VEGETATION_MODEL = "tcd_segformer"

# Geographic areas to fetch (bounding boxes in WGS84 lat/lon)
AREAS = {
    "ovgu_bbox": {
        "west": 11.639779,
        "east": 11.652739,
        "south": 52.137663,
        "north": 52.145538,
     },
    # "magdeburg_bbox": {
    #     "west": 11.5,
    #     "east": 12.0,
    #     "south": 52.0,
    #     "north": 52.3,
    # },
}