import os
import numpy as np
from typing import List

from PyQt6 import QtCore
from PyQt6.QtCore import QCoreApplication
from PyQt6.QtGui import QImage

from anylabeling.views.labeling.shape import Shape
from anylabeling.views.labeling.logger import logger
from .model import Model
from .types import AutoLabelingResult


class YOLOUltralytics(Model):
    class Meta:
        required_config_names = [
            "type",
            "name",
            "display_name",
            "model_path",
        ]
        widgets = [
            "button_run",
            "input_conf",
            "edit_conf",
            "input_iou",
            "edit_iou",
            "toggle_preserve_existing_annotations",
            "button_classes_filter",
        ]
        output_modes = {
            "rectangle": QCoreApplication.translate("Model", "Rectangle"),
        }
        default_output_mode = "rectangle"

    def __init__(self, model_config, on_message) -> None:
        super().__init__(model_config, on_message)

        model_abs_path = self.get_model_abs_path(self.config, "model_path")
        if not model_abs_path or not os.path.isfile(model_abs_path):
            raise FileNotFoundError(
                QCoreApplication.translate(
                    "Model",
                    f"Could not find model file: {model_abs_path}",
                )
            )

        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                "Ultralytics package is required for this model. Install it with: uv pip install ultralytics"
            )

        task = self.config.get("task", "detect")
        self.model = YOLO(model_abs_path, task=task)
        self.classes = list(self.model.names.values())
        self.conf_thres = self.config.get("conf_threshold", 0.25)
        self.iou_thres = self.config.get("iou_threshold", 0.45)
        self.replace = True
        self.filter_classes = self.config.get("filter_classes", None)

    def set_auto_labeling_conf(self, value):
        if value > 0:
            self.conf_thres = value

    def set_auto_labeling_iou(self, value):
        if value > 0:
            self.iou_thres = value

    def set_auto_labeling_preserve_existing_annotations_state(self, state):
        self.replace = not state

    def set_auto_labeling_filter_classes(self, class_names: List[str]) -> None:
        if not class_names or len(class_names) == len(self.classes):
            self.filter_classes = None
        else:
            self.filter_classes = class_names

    def predict_shapes(self, image, image_path=None):
        if image is None:
            return []

        if image_path and os.path.isfile(image_path):
            source = image_path
        else:
            logger.warning(f"Could not load image from path: {image_path}. Trying to load from QImage.")
            try:
                image = image.convertToFormat(QImage.Format.Format_RGB888)
                ptr = image.bits()
                ptr.setsize(image.height() * image.width() * 3)
                source = np.array(ptr).reshape(image.height(), image.width(), 3).copy()
            except Exception as e:
                logger.warning(f"Could not inference model from QImage: {e}")
                return []

        results = self.model.predict(
            source,
            conf=self.conf_thres,
            iou=self.iou_thres,
            verbose=False,
        )

        if not results:
            return AutoLabelingResult([], replace=self.replace)
        result = results[0]

        shapes = []
        if result.boxes is None:
            return AutoLabelingResult(shapes, replace=self.replace)

        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            score = float(box.conf[0])
            cls_id = int(box.cls[0])
            label = result.names[cls_id]

            if self.filter_classes and label not in self.filter_classes:
                continue

            shape = Shape(
                label=label,
                score=score,
                shape_type="rectangle",
                flags={},
            )
            shape.add_point(QtCore.QPointF(x1, y1))
            shape.add_point(QtCore.QPointF(x2, y1))
            shape.add_point(QtCore.QPointF(x2, y2))
            shape.add_point(QtCore.QPointF(x1, y2))
            shapes.append(shape)

        return AutoLabelingResult(shapes, replace=self.replace)

    def unload(self):
        del self.model
