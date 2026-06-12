"""Project management service - handles projects and images on the filesystem."""

import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Optional

from app.config import IMAGE_EXTENSIONS, PROJECTS_DIR


class ProjectService:
    """Manages annotation projects on the local filesystem."""

    def list_projects(self):
        """List all projects."""
        projects = []
        if not PROJECTS_DIR.exists():
            return projects
        for d in sorted(PROJECTS_DIR.iterdir()):
            if d.is_dir() and (d / "_meta.json").exists():
                meta = self._read_json(d / "_meta.json")
                if meta:
                    projects.append(meta)
        return projects

    def get_project(self, project_id: str) -> Optional[dict]:
        """Get project metadata."""
        meta_path = PROJECTS_DIR / project_id / "_meta.json"
        return self._read_json(meta_path)

    def create_project(self, name: str, description: str = "") -> dict:
        """Create a new project."""
        project_id = str(uuid.uuid4())[:8]
        project_dir = PROJECTS_DIR / project_id
        project_dir.mkdir(parents=True, exist_ok=True)
        (project_dir / "images").mkdir(exist_ok=True)
        (project_dir / "annotations").mkdir(exist_ok=True)

        meta = {
            "id": project_id,
            "name": name,
            "description": description,
            "image_count": 0,
            "created_at": "",
            "updated_at": "",
        }
        self._write_json(project_dir / "_meta.json", meta)
        return meta

    def delete_project(self, project_id: str) -> bool:
        """Delete a project and all its data."""
        project_dir = PROJECTS_DIR / project_id
        if project_dir.exists():
            shutil.rmtree(project_dir)
            return True
        return False

    def list_images(self, project_id: str) -> list:
        """List all images in a project."""
        images_dir = PROJECTS_DIR / project_id / "images"
        if not images_dir.exists():
            return []
        images = []
        for f in sorted(images_dir.iterdir()):
            if f.suffix.lower() in IMAGE_EXTENSIONS:
                images.append({
                    "id": f.stem,
                    "filename": f.name,
                    "path": str(f),
                    "size": f.stat().st_size,
                })
        return images

    def get_image_path(self, project_id: str, image_id: str) -> Optional[Path]:
        """Get the file path for an image."""
        images_dir = PROJECTS_DIR / project_id / "images"
        if not images_dir.exists():
            return None
        for f in images_dir.iterdir():
            if f.stem == image_id and f.suffix.lower() in IMAGE_EXTENSIONS:
                return f
        return None

    def save_uploaded_image(
        self, project_id: str, filename: str, content: bytes
    ) -> dict:
        """Save an uploaded image to a project."""
        images_dir = PROJECTS_DIR / project_id / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        # Generate unique image ID
        stem = Path(filename).stem
        suffix = Path(filename).suffix.lower()
        if suffix not in IMAGE_EXTENSIONS:
            suffix = ".jpg"

        # Avoid name collisions
        target = images_dir / filename
        image_id = stem
        counter = 1
        while target.exists():
            image_id = f"{stem}_{counter}"
            target = images_dir / f"{image_id}{suffix}"
            counter += 1

        with open(target, "wb") as f:
            f.write(content)

        # Update project metadata
        meta = self.get_project(project_id)
        if meta:
            meta["image_count"] = len(self.list_images(project_id))
            self._write_json(
                PROJECTS_DIR / project_id / "_meta.json", meta
            )

        return {
            "id": image_id,
            "filename": target.name,
            "path": str(target),
            "size": len(content),
        }

    def delete_image(self, project_id: str, image_id: str) -> bool:
        """Delete an image and its annotation."""
        image_path = self.get_image_path(project_id, image_id)
        if image_path and image_path.exists():
            image_path.unlink()
            # Also delete annotation if exists
            ann_path = (
                PROJECTS_DIR
                / project_id
                / "annotations"
                / f"{image_id}.json"
            )
            if ann_path.exists():
                ann_path.unlink()
            return True
        return False

    @staticmethod
    def _read_json(path: Path) -> Optional[dict]:
        """Read a JSON file."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return None

    @staticmethod
    def _write_json(path: Path, data: dict):
        """Write a JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)


# Singleton instance
project_service = ProjectService()
