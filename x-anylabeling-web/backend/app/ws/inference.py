"""WebSocket endpoint for real-time inference."""

import asyncio
import json
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.services.project_service import project_service
from app.services.annotation_service import annotation_service
from app.config import PROJECTS_DIR

router = APIRouter()

# Thread pool for running blocking model inference
_executor = ThreadPoolExecutor(max_workers=2)


class InferenceSession:
    """Manages a WebSocket inference session."""

    def __init__(self, websocket: WebSocket):
        self.websocket = websocket
        self.loaded_model = None
        self.model_config = None
        self._cancel_event = asyncio.Event()

    async def send(self, message: dict):
        """Send a JSON message to the client."""
        await self.websocket.send_json(message)

    async def run_prediction(
        self, image_path: str, params: dict
    ):
        """Run model prediction on an image."""
        if self.loaded_model is None:
            await self.send({
                "type": "prediction_error",
                "message": "No model loaded",
            })
            return

        self._cancel_event.clear()
        await self.send({"type": "prediction_started"})

        try:
            # Run prediction in thread pool (blocking call)
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                _executor,
                self._predict_sync,
                image_path,
                params,
            )
            await self.send({
                "type": "prediction_result",
                "shapes": result.get("shapes", []),
                "replace": result.get("replace", True),
                "image_path": result.get("image_path", ""),
            })
        except Exception as e:
            await self.send({
                "type": "prediction_error",
                "message": str(e),
            })
        finally:
            await self.send({"type": "prediction_finished"})

    def _predict_sync(self, image_path: str, params: dict) -> dict:
        """Synchronous prediction (runs in thread pool)."""
        import cv2
        import numpy as np

        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot read image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Run prediction
        text_prompt = params.get("text_prompt")
        if text_prompt:
            result = self.loaded_model.predict_shapes(
                image, image_path, text_prompt=text_prompt
            )
        else:
            result = self.loaded_model.predict_shapes(image, image_path)

        # Convert result to JSON-serializable format
        shapes = []
        if hasattr(result, "shapes"):
            for shape in result.shapes:
                shapes.append(shape.to_dict() if hasattr(shape, "to_dict") else shape)

        return {
            "shapes": shapes,
            "replace": getattr(result, "replace", True),
            "image_path": getattr(result, "image_path", image_path),
        }


@router.websocket("/inference/{session_id}")
async def inference_websocket(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time model inference."""
    await websocket.accept()
    session = InferenceSession(websocket)

    try:
        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type")

            if msg_type == "run_prediction":
                project_id = data.get("project_id")
                image_id = data.get("image_id")
                params = data.get("params", {})

                if not project_id or not image_id:
                    await session.send({
                        "type": "prediction_error",
                        "message": "Missing project_id or image_id",
                    })
                    continue

                image_path = project_service.get_image_path(
                    project_id, image_id
                )
                if not image_path:
                    await session.send({
                        "type": "prediction_error",
                        "message": f"Image not found: {image_id}",
                    })
                    continue

                await session.run_prediction(str(image_path), params)

            elif msg_type == "cancel_prediction":
                session._cancel_event.set()
                await session.send({
                    "type": "model_status",
                    "message": "Prediction cancelled",
                })

            elif msg_type == "ping":
                await session.send({"type": "pong"})

    except WebSocketDisconnect:
        pass
    except Exception as e:
        traceback.print_exc()
        try:
            await session.send({
                "type": "error",
                "message": str(e),
            })
        except Exception:
            pass
