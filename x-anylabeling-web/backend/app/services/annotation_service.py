"""Annotation management service - handles loading and saving annotations."""

import json
import os
from pathlib import Path
from typing import Optional

from app.config import PROJECTS_DIR
from app.app_info import __version__


# X-AnyLabeling annotation format (from schema.py)
XLABEL_BASIC_FIELDS = [
    "version",
    "flags",
    "checked",
    "shapes",
    "imagePath",
    "imageData",
    "imageHeight",
    "imageWidth",
]


def create_xlabel_template(
    version: str = __version__,
    flags: Optional[dict] = None,
    checked: bool = False,
    shapes: Optional[list] = None,
    image_path: str = "",
    image_data: Optional[str] = None,
    image_height: int = -1,
    image_width: int = -1,
) -> dict:
    """Create a new annotation template."""
    return {
        "version": version,
        "flags": flags if flags is not None else {},
        "checked": checked,
        "shapes": shapes if shapes is not None else [],
        "imagePath": image_path,
        "imageData": image_data,
        "imageHeight": image_height,
        "imageWidth": image_width,
    }


class AnnotationService:
    """Manages annotations for images in projects."""

    def get_annotation(self, project_id: str, image_id: str) -> Optional[dict]:
        """Load annotation for an image."""
        ann_path = (
            PROJECTS_DIR / project_id / "annotations" / f"{image_id}.json"
        )
        if not ann_path.exists():
            return None
        try:
            with open(ann_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return None

    def save_annotation(
        self, project_id: str, image_id: str, data: dict
    ) -> dict:
        """Save annotation for an image."""
        ann_dir = PROJECTS_DIR / project_id / "annotations"
        ann_dir.mkdir(parents=True, exist_ok=True)

        ann_path = ann_dir / f"{image_id}.json"

        # Ensure the annotation has the correct format
        if "version" not in data:
            data["version"] = __version__

        # Never store image_data in the JSON (too large)
        data["imageData"] = None

        with open(ann_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        return data

    def create_default_annotation(
        self, project_id: str, image_id: str, image_filename: str,
        image_height: int, image_width: int,
    ) -> dict:
        """Create a default annotation for a new image."""
        return create_xlabel_template(
            image_path=image_filename,
            image_height=image_height,
            image_width=image_width,
        )


# Singleton instance
annotation_service = AnnotationService()
