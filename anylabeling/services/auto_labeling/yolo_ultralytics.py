import os
from typing import List

import cv2
import numpy as np
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
            "output_label",
            "output_select_combobox",
            "toggle_preserve_existing_annotations",
            "button_classes_filter",
            "mask_fineness_slider",
            "mask_fineness_value_label",
        ]
        output_modes = {
            "rectangle": QCoreApplication.translate("Model", "Rectangle"),
            "polygon": QCoreApplication.translate("Model", "Polygon"),
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
        self.task = getattr(self.model, "task", task)

        # Set default output mode according to task
        if self.task in ("segment", "semantic"):
            self.output_mode = "polygon"
            self.Meta.default_output_mode = "polygon"
        else:
            self.output_mode = "rectangle"
            self.Meta.default_output_mode = "rectangle"

        self._classes = None
        self.conf_thres = self.config.get("conf_threshold", 0.25)
        self.iou_thres = self.config.get("iou_threshold", 0.45)
        self.epsilon = 0.001
        self.replace = True
        self.filter_classes = self.config.get("filter_classes", None)

    @property
    def classes(self):
        # 懒加载：不在 __init__ 中访问 self.model.names。
        # 对于导出格式（torchscript/onnx/engine 等），YOLO() 构造不会真正加载模型，
        # 访问 .names 会触发 setup_model 创建 AutoBackend 并加载权重；而构造阶段此时
        # self.predictor 尚未建立，该临时加载会被丢弃，导致模型被无谓地加载两次
        # （一次取 names，一次首次 predict）。延迟到首次 predict 之后，self.predictor
        # 已存在，访问 .names 走缓存路径不再重复加载。
        if self._classes is None:
            self._classes = list(self.model.names.values())
        return self._classes

    def set_auto_labeling_conf(self, value):
        if value > 0:
            self.conf_thres = value

    def set_auto_labeling_iou(self, value):
        if value > 0:
            self.iou_thres = value

    def set_mask_fineness(self, epsilon):
        if epsilon > 0:
            self.epsilon = epsilon

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

        # 1. 优先处理语义分割 (Semantic Segmentation)
        if hasattr(result, "semantic_mask") and result.semantic_mask is not None:
            mask_data = result.semantic_mask.data.cpu().numpy().astype(np.uint8)
            img_area = mask_data.shape[0] * mask_data.shape[1]

            unique_cls_ids = np.unique(mask_data)
            for cls_id in unique_cls_ids:
                cls_id_int = int(cls_id)
                label = result.names.get(cls_id_int, str(cls_id_int))
                if self.filter_classes and label not in self.filter_classes:
                    continue

                binary_mask = (mask_data == cls_id).astype(np.uint8) * 255
                contours, _ = cv2.findContours(
                    binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                for contour in contours:
                    # 过滤过小轮廓
                    if cv2.contourArea(contour) < max(10, img_area * 0.00005):
                        continue

                    if self.output_mode == "rectangle":
                        x, y, w, h = cv2.boundingRect(contour)
                        shape = Shape(
                            label=label,
                            score=1.0,
                            shape_type="rectangle",
                            flags={},
                        )
                        shape.add_point(QtCore.QPointF(x, y))
                        shape.add_point(QtCore.QPointF(x + w, y))
                        shape.add_point(QtCore.QPointF(x + w, y + h))
                        shape.add_point(QtCore.QPointF(x, y + h))
                        shape.close()
                        shapes.append(shape)
                    else:
                        # polygon 模式
                        arc_len = cv2.arcLength(contour, True)
                        epsilon = self.epsilon * arc_len
                        approx = cv2.approxPolyDP(contour, epsilon, True)
                        pts = approx.reshape(-1, 2)
                        if len(pts) < 3:
                            continue

                        shape = Shape(
                            label=label,
                            score=1.0,
                            shape_type="polygon",
                            flags={},
                        )
                        for pt in pts:
                            shape.add_point(QtCore.QPointF(float(pt[0]), float(pt[1])))
                        shape.close()
                        shapes.append(shape)

            return AutoLabelingResult(shapes, replace=self.replace)

        # 2. 实例分割 (Instance Segmentation)
        if (
            self.output_mode == "polygon"
            and hasattr(result, "masks")
            and result.masks is not None
        ):
            xy_segments = getattr(result.masks, "xy", None)
            boxes = result.boxes
            if xy_segments is not None and len(xy_segments) > 0:
                for i, segment in enumerate(xy_segments):
                    if len(segment) < 3:
                        continue

                    score = 1.0
                    label = "object"
                    if boxes is not None and i < len(boxes):
                        score = float(boxes.conf[i])
                        cls_id = int(boxes.cls[i])
                        label = result.names.get(cls_id, str(cls_id))

                    if self.filter_classes and label not in self.filter_classes:
                        continue

                    # 平滑简化多边形
                    pts_np = np.array(segment, dtype=np.int32).reshape((-1, 1, 2))
                    arc_len = cv2.arcLength(pts_np, True)
                    epsilon = self.epsilon * arc_len
                    approx = cv2.approxPolyDP(pts_np, epsilon, True)
                    pts = approx.reshape(-1, 2)
                    if len(pts) < 3:
                        continue

                    shape = Shape(
                        label=label,
                        score=score,
                        shape_type="polygon",
                        flags={},
                    )
                    for pt in pts:
                        shape.add_point(QtCore.QPointF(float(pt[0]), float(pt[1])))
                    shape.close()
                    shapes.append(shape)

                return AutoLabelingResult(shapes, replace=self.replace)

        # 3. 目标检测 / 矩形输出 (Object Detection / Boxes)
        if result.boxes is None:
            return AutoLabelingResult(shapes, replace=self.replace)

        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            score = float(box.conf[0])
            cls_id = int(box.cls[0])
            label = result.names.get(cls_id, str(cls_id))

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
            shape.close()
            shapes.append(shape)

        return AutoLabelingResult(shapes, replace=self.replace)

    def unload(self):
        del self.model
