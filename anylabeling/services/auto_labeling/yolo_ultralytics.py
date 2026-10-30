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
            "point": QCoreApplication.translate("Model", "Point"),
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
        if self.task == "detect":
            m = getattr(self.model, "model", None)
            if m is not None and hasattr(m, "kpt_shape") and m.kpt_shape is not None:
                self.task = "pose"

        # Create instance-level Meta so each model config has tailored output_modes and widgets
        class InstanceMeta:
            required_config_names = list(self.Meta.required_config_names)
            widgets = list(self.Meta.widgets)
            output_modes = dict(self.Meta.output_modes)
            default_output_mode = self.Meta.default_output_mode

        self.Meta = InstanceMeta

        # Set output modes & default output mode according to task
        if self.task == "pose":
            self.Meta.output_modes = {
                "point": QCoreApplication.translate("Model", "Point"),
                "rectangle": QCoreApplication.translate("Model", "Rectangle"),
            }
            self.output_mode = self.config.get("output_mode", "point")
            self.Meta.default_output_mode = self.output_mode
            self.show_boxes = self.config.get("show_boxes", True)
            self.kpt_threshold = self.config.get("kpt_threshold", 0.1)
            if "mask_fineness_slider" in self.Meta.widgets:
                self.Meta.widgets.remove("mask_fineness_slider")
            if "mask_fineness_value_label" in self.Meta.widgets:
                self.Meta.widgets.remove("mask_fineness_value_label")
        elif self.task in ("segment", "semantic"):
            self.Meta.output_modes = {
                "polygon": QCoreApplication.translate("Model", "Polygon"),
                "rectangle": QCoreApplication.translate("Model", "Rectangle"),
            }
            self.output_mode = self.config.get("output_mode", "polygon")
            self.Meta.default_output_mode = self.output_mode
            self.show_boxes = self.config.get("show_boxes", False)
            self.kpt_threshold = 0.1
        else:
            self.Meta.output_modes = {
                "rectangle": QCoreApplication.translate("Model", "Rectangle"),
            }
            self.output_mode = self.config.get("output_mode", "rectangle")
            self.Meta.default_output_mode = self.output_mode
            self.show_boxes = self.config.get("show_boxes", False)
            self.kpt_threshold = 0.1
            if "mask_fineness_slider" in self.Meta.widgets:
                self.Meta.widgets.remove("mask_fineness_slider")
            if "mask_fineness_value_label" in self.Meta.widgets:
                self.Meta.widgets.remove("mask_fineness_value_label")

        self._classes = None
        self._model_channels = None
        self.conf_thres = self.config.get("conf_threshold", 0.25)
        self.iou_thres = self.config.get("iou_threshold", 0.45)
        self.epsilon = 0.001
        self.replace = True
        self.filter_classes = self.config.get("filter_classes", None)

    @property
    def model_channels(self) -> int:
        """Get expected input channels of the model (1 for grayscale, 3 for color)."""
        if self._model_channels is not None:
            return self._model_channels

        self._model_channels = 3
        try:
            m = getattr(self.model, "model", None)
            if m is not None:
                yaml_cfg = getattr(m, "yaml", {})
                if isinstance(yaml_cfg, dict) and "channels" in yaml_cfg:
                    self._model_channels = int(yaml_cfg["channels"])
                    return self._model_channels
                if hasattr(m, "model") and len(m.model) > 0:
                    first = m.model[0]
                    if hasattr(first, "conv") and hasattr(first.conv, "in_channels"):
                        self._model_channels = int(first.conv.in_channels)
                        return self._model_channels
                    if hasattr(first, "in_channels"):
                        self._model_channels = int(first.in_channels)
                        return self._model_channels
                if hasattr(m, "channels"):
                    self._model_channels = int(m.channels)
                    return self._model_channels

            pred = getattr(self.model, "predictor", None)
            if pred is not None and hasattr(pred, "model") and hasattr(pred.model, "channels"):
                self._model_channels = int(pred.model.channels)
                return self._model_channels
        except Exception as e:
            logger.warning(f"Could not detect model channels, defaulting to 3: {e}")

        return self._model_channels

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

    def _get_keypoint_names(
        self, cls_id: int, cls_name: str, num_kpts: int
    ) -> List[str]:
        """Resolve keypoint names from config or model metadata."""
        # 1. Config classes dict: {cls_name: [kpt0, kpt1, ...]} or {cls_id: [...]}
        classes_cfg = self.config.get("classes")
        if isinstance(classes_cfg, dict):
            if cls_name in classes_cfg and isinstance(
                classes_cfg[cls_name], (list, tuple)
            ):
                return [str(k) for k in classes_cfg[cls_name]]
            if cls_id in classes_cfg and isinstance(
                classes_cfg[cls_id], (list, tuple)
            ):
                return [str(k) for k in classes_cfg[cls_id]]

        # 2. Config keypoints list: [kpt0, kpt1, ...]
        if "keypoints" in self.config and isinstance(
            self.config["keypoints"], (list, tuple)
        ):
            return [str(k) for k in self.config["keypoints"]]

        # 3. Model attribute kpt_names
        m = getattr(self.model, "model", None)
        kpt_names = getattr(m, "kpt_names", None) if m else None
        if isinstance(kpt_names, dict):
            if cls_id in kpt_names and isinstance(
                kpt_names[cls_id], (list, tuple)
            ):
                return [str(k) for k in kpt_names[cls_id]]
            if cls_name in kpt_names and isinstance(
                kpt_names[cls_name], (list, tuple)
            ):
                return [str(k) for k in kpt_names[cls_name]]
        elif isinstance(kpt_names, (list, tuple)):
            return [str(k) for k in kpt_names]

        # 4. Fallback: '0', '1', '2', ...
        return [str(i) for i in range(num_kpts)]

    def predict_shapes(self, image, image_path=None):
        if image is None and not image_path:
            return []

        if image_path and os.path.isfile(image_path) and not image_path.lower().endswith(".raw"):
            source = image_path
        else:
            if not image_path or not image_path.lower().endswith(".raw"):
                logger.warning(
                    f"Could not load image from path: {image_path}. Trying to load from QImage."
                )
            try:
                channels = self.model_channels
                if hasattr(image, "convertToFormat"):
                    if channels == 1:
                        img = image.convertToFormat(QImage.Format.Format_Grayscale8)
                        w, h, bpl = img.width(), img.height(), img.bytesPerLine()
                        ptr = img.bits()
                        ptr.setsize(h * bpl)
                        arr = np.frombuffer(ptr, dtype=np.uint8).reshape((h, bpl))
                        if bpl != w:
                            arr = arr[:, :w]
                        source = arr.copy()
                    else:
                        img = image.convertToFormat(QImage.Format.Format_BGR888)
                        w, h, bpl = img.width(), img.height(), img.bytesPerLine()
                        ptr = img.bits()
                        ptr.setsize(h * bpl)
                        arr = np.frombuffer(ptr, dtype=np.uint8).reshape((h, bpl))
                        arr = arr[:, : w * 3].reshape((h, w, 3))
                        source = arr.copy()
                elif isinstance(image, np.ndarray):
                    arr = image
                    if channels == 1:
                        if arr.ndim == 3 and arr.shape[2] == 3:
                            source = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
                        elif arr.ndim == 3 and arr.shape[2] == 1:
                            source = arr[:, :, 0]
                        else:
                            source = arr
                    else:
                        if arr.ndim == 2:
                            source = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
                        elif arr.ndim == 3 and arr.shape[2] == 1:
                            source = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
                        else:
                            source = arr
                else:
                    source = np.array(image)
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

        # 3. 姿态估计 (Pose Estimation / Keypoints)
        if (
            self.task == "pose"
            or (hasattr(result, "keypoints") and result.keypoints is not None)
        ) and self.output_mode == "point":
            boxes = result.boxes
            keypoints = result.keypoints

            if boxes is None or len(boxes) == 0:
                return AutoLabelingResult(shapes, replace=self.replace)

            kpts_xy = (
                keypoints.xy.cpu().numpy()
                if keypoints is not None and hasattr(keypoints, "xy")
                else None
            )
            kpts_conf = (
                keypoints.conf.cpu().numpy()
                if keypoints is not None and keypoints.conf is not None
                else None
            )

            for i, box in enumerate(boxes):
                score = float(box.conf[0])
                cls_id = int(box.cls[0])
                label = result.names.get(cls_id, str(cls_id))

                if self.filter_classes and label not in self.filter_classes:
                    continue

                group_id = (
                    int(box.id[0])
                    if (hasattr(box, "id") and box.id is not None)
                    else int(i)
                )

                if self.show_boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    box_shape = Shape(
                        label=label,
                        score=score,
                        shape_type="rectangle",
                        flags={},
                        group_id=group_id,
                    )
                    box_shape.add_point(QtCore.QPointF(x1, y1))
                    box_shape.add_point(QtCore.QPointF(x2, y1))
                    box_shape.add_point(QtCore.QPointF(x2, y2))
                    box_shape.add_point(QtCore.QPointF(x1, y2))
                    box_shape.close()
                    shapes.append(box_shape)

                if kpts_xy is not None and i < len(kpts_xy):
                    inst_kpts = kpts_xy[i]
                    num_kpts = len(inst_kpts)
                    kpt_names = self._get_keypoint_names(
                        cls_id, label, num_kpts
                    )

                    for j in range(num_kpts):
                        x, y = float(inst_kpts[j][0]), float(inst_kpts[j][1])
                        kpt_score = (
                            float(kpts_conf[i][j])
                            if (kpts_conf is not None and j < len(kpts_conf[i]))
                            else score
                        )

                        if (
                            (x <= 0 and y <= 0)
                            or np.isnan(x)
                            or np.isnan(y)
                            or kpt_score < self.kpt_threshold
                        ):
                            continue

                        kpt_label = (
                            kpt_names[j] if j < len(kpt_names) else str(j)
                        )
                        pt_shape = Shape(
                            label=kpt_label,
                            score=kpt_score,
                            shape_type="point",
                            flags={},
                            group_id=group_id,
                        )
                        pt_shape.add_point(QtCore.QPointF(x, y))
                        pt_shape.close()
                        shapes.append(pt_shape)

            return AutoLabelingResult(shapes, replace=self.replace)

        # 4. 目标检测 / 矩形输出 (Object Detection / Boxes)
        if result.boxes is None:
            return AutoLabelingResult(shapes, replace=self.replace)

        for i, box in enumerate(result.boxes):
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            score = float(box.conf[0])
            cls_id = int(box.cls[0])
            label = result.names.get(cls_id, str(cls_id))

            if self.filter_classes and label not in self.filter_classes:
                continue

            group_id = (
                int(box.id[0])
                if (hasattr(box, "id") and box.id is not None)
                else (int(i) if self.task == "pose" else None)
            )

            shape = Shape(
                label=label,
                score=score,
                shape_type="rectangle",
                flags={},
                group_id=group_id,
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
