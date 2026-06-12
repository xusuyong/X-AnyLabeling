"""Project management API routes."""

from fastapi import APIRouter, HTTPException, UploadFile, File
from typing import List, Optional

from app.services.project_service import project_service
from app.services.annotation_service import annotation_service

router = APIRouter()


@router.get("")
async def list_projects():
    """List all projects."""
    return project_service.list_projects()


@router.post("")
async def create_project(name: str, description: str = ""):
    """Create a new project."""
    if not name.strip():
        raise HTTPException(status_code=400, detail="Name is required")
    return project_service.create_project(name.strip(), description)


@router.get("/{project_id}")
async def get_project(project_id: str):
    """Get project details."""
    project = project_service.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return project


@router.delete("/{project_id}")
async def delete_project(project_id: str):
    """Delete a project."""
    if not project_service.delete_project(project_id):
        raise HTTPException(status_code=404, detail="Project not found")
    return {"status": "deleted"}


@router.get("/{project_id}/images")
async def list_images(project_id: str):
    """List all images in a project."""
    project = project_service.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return project_service.list_images(project_id)


@router.put("/{project_id}/images")
async def upload_images(
    project_id: str,
    files: List[UploadFile] = File(...),
):
    """Upload one or more images to a project."""
    project = project_service.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    uploaded = []
    for file in files:
        content = await file.read()
        image_info = project_service.save_uploaded_image(
            project_id, file.filename, content
        )
        uploaded.append(image_info)
    return {"uploaded": uploaded, "count": len(uploaded)}


@router.get("/{project_id}/images/{image_id}")
async def get_image(project_id: str, image_id: str):
    """Get image file. Returns the raw image file."""
    from fastapi.responses import FileResponse

    image_path = project_service.get_image_path(project_id, image_id)
    if not image_path or not image_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(
        str(image_path),
        media_type=_get_media_type(image_path),
        filename=image_path.name,
    )


@router.delete("/{project_id}/images/{image_id}")
async def delete_image(project_id: str, image_id: str):
    """Delete an image and its annotation."""
    if not project_service.delete_image(project_id, image_id):
        raise HTTPException(status_code=404, detail="Image not found")
    return {"status": "deleted"}


def _get_media_type(path) -> str:
    """Get MIME type from file extension."""
    suffix = path.suffix.lower()
    media_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".bmp": "image/bmp",
        ".tiff": "image/tiff",
        ".tif": "image/tiff",
        ".webp": "image/webp",
    }
    return media_types.get(suffix, "application/octet-stream")
