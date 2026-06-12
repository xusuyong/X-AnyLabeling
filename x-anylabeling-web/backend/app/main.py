"""X-AnyLabeling Web Backend - FastAPI Application"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import CORS_ORIGINS, DEBUG, PROJECTS_DIR
from app.api import projects, annotations, models as models_api
from app.ws import inference

app = FastAPI(
    title="X-AnyLabeling Web",
    description="AI-powered annotation tool - Web API",
    version="0.1.0",
    debug=DEBUG,
)

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API routers
app.include_router(
    projects.router,
    prefix="/api/projects",
    tags=["projects"],
)
app.include_router(
    annotations.router,
    prefix="/api/projects/{project_id}/images/{image_id}/annotations",
    tags=["annotations"],
)
app.include_router(
    models_api.router,
    prefix="/api/models",
    tags=["models"],
)
app.include_router(
    inference.router,
    prefix="/api/ws",
    tags=["websocket"],
)


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok", "version": "0.1.0"}


@app.on_event("startup")
async def startup():
    """Initialize services on startup"""
    # Ensure projects directory exists
    PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
