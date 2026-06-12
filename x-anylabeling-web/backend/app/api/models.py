"""Model management API routes."""

from fastapi import APIRouter, HTTPException

from app.config import MODEL_REGISTRY_PATH

router = APIRouter()

# In-memory model status cache
_model_status = {}


@router.get("")
async def list_models():
    """List all available models from the registry."""
    import yaml

    if not MODEL_REGISTRY_PATH.exists():
        return []

    with open(MODEL_REGISTRY_PATH, "r", encoding="utf-8") as f:
        models = yaml.safe_load(f) or []

    # Add status info
    for model in models:
        model_name = model.get("model_name", "")
        model["status"] = _model_status.get(model_name, "not_loaded")

    return models


@router.get("/{model_id:path}")
async def get_model_detail(model_id: str):
    """Get detailed info about a specific model."""
    import yaml

    if not MODEL_REGISTRY_PATH.exists():
        raise HTTPException(status_code=404, detail="Model registry not found")

    with open(MODEL_REGISTRY_PATH, "r", encoding="utf-8") as f:
        models = yaml.safe_load(f) or []

    for model in models:
        if model.get("model_name") == model_id:
            model["status"] = _model_status.get(model_id, "not_loaded")
            return model

    raise HTTPException(status_code=404, detail="Model not found")


@router.post("/{model_id:path}/load")
async def load_model(model_id: str):
    """Load a model into memory (async, progress via WebSocket)."""
    # Actual model loading will be handled by ModelService
    # This endpoint just triggers the load process
    _model_status[model_id] = "loading"
    return {
        "status": "loading",
        "model_id": model_id,
        "message": "Model loading initiated. Connect to WebSocket for progress.",
    }


@router.post("/{model_id:path}/unload")
async def unload_model(model_id: str):
    """Unload a model from memory."""
    _model_status[model_id] = "not_loaded"
    return {"status": "unloaded", "model_id": model_id}


@router.get("/{model_id:path}/status")
async def model_status(model_id: str):
    """Get current model loading status."""
    status = _model_status.get(model_id, "not_loaded")
    return {"model_id": model_id, "status": status}
