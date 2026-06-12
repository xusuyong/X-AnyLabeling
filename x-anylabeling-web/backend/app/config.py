"""X-AnyLabeling Web Backend Configuration"""

import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).parent
DATA_DIR = Path(os.getenv("XANYLABELING_DATA_DIR",
                           Path.home() / ".xanylabeling_data"))
PROJECTS_DIR = DATA_DIR / "projects"
MODELS_DIR = DATA_DIR / "models"

# Ensure directories exist
PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Server settings
HOST = os.getenv("XANYLABELING_HOST", "0.0.0.0")
PORT = int(os.getenv("XANYLABELING_PORT", "8000"))
DEBUG = os.getenv("XANYLABELING_DEBUG", "false").lower() == "true"

# CORS settings
CORS_ORIGINS = os.getenv(
    "XANYLABELING_CORS_ORIGINS",
    "http://localhost:5173,http://localhost:3000"
).split(",")

# Model settings - look for models.yaml in the original project
# Path: x-anylabeling-web/backend/app/config.py -> X-AnyLabeling/anylabeling/configs/models.yaml
_XANYLABELING_ROOT = Path(__file__).resolve().parent.parent.parent.parent
MODEL_REGISTRY_PATH = _XANYLABELING_ROOT / "anylabeling" / "configs" / "models.yaml"
# Fallback: also check environment variable
if not MODEL_REGISTRY_PATH.exists():
    _env_path = os.getenv("XANYLABELING_MODEL_REGISTRY", "")
    if _env_path:
        MODEL_REGISTRY_PATH = Path(_env_path)

# Supported image extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

# Annotation file extension
ANNOTATION_EXTENSION = ".json"
