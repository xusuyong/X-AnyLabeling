"""Annotation management API routes."""

from fastapi import APIRouter, HTTPException

from app.services.project_service import project_service
from app.services.annotation_service import annotation_service

router = APIRouter()


@router.get("")
async def get_annotations(project_id: str, image_id: str):
    """Get annotations for an image."""
    # Verify project and image exist
    _verify_project_and_image(project_id, image_id)

    ann = annotation_service.get_annotation(project_id, image_id)
    if ann is None:
        # Return default empty annotation
        from PIL import Image

        image_path = project_service.get_image_path(
            project_id, image_id
        )
        try:
            with Image.open(str(image_path)) as img:
                width, height = img.size
        except Exception:
            width, height = -1, -1

        ann = annotation_service.create_default_annotation(
            project_id, image_id,
            image_path.name if image_path else "",
            height, width,
        )
    return ann


@router.put("")
async def save_annotations(project_id: str, image_id: str, data: dict):
    """Save annotations for an image."""
    _verify_project_and_image(project_id, image_id)

    # Validate the annotation format
    if "shapes" not in data:
        raise HTTPException(
            status_code=400, detail="Missing 'shapes' field"
        )

    saved = annotation_service.save_annotation(
        project_id, image_id, data
    )
    return saved


def _verify_project_and_image(project_id: str, image_id: str):
    """Verify that project and image exist, raise 404 if not."""
    project = project_service.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    image_path = project_service.get_image_path(project_id, image_id)
    if not image_path:
        raise HTTPException(status_code=404, detail="Image not found")
