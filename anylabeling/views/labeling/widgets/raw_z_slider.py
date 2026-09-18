"""Z-axis slider and volume inspection widget for 3D RAW volume navigation."""

from typing import Optional, Any
from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import Qt

from anylabeling.views.labeling.utils.theme import get_theme
from anylabeling.views.labeling.utils.raw_reader import RawFileInfo


class RawZSlider(QtWidgets.QWidget):
    """Bottom navigation and inspection widget for 3D RAW volumes."""

    z_changed = QtCore.pyqtSignal(int)
    orthogonal_view_toggled = QtCore.pyqtSignal(bool)
    crosshair_toggled = QtCore.pyqtSignal(bool)
    decompose_requested = QtCore.pyqtSignal()

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.file_info: Optional[RawFileInfo] = None
        self._current_z: int = 0
        self._setup_ui()

    def _setup_ui(self):
        main_layout = QtWidgets.QHBoxLayout(self)
        main_layout.setContentsMargins(10, 4, 10, 4)
        main_layout.setSpacing(10)

        t = get_theme()

        # ==================== Left Half: RAW volume tools & voxel inspection ====================
        left_container = QtWidgets.QWidget()
        left_layout = QtWidgets.QHBoxLayout(left_container)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(8)

        # 1. Orthogonal Side Views Button
        self._ortho_btn = QtWidgets.QPushButton(self.tr("◫ 侧视图"))
        self._ortho_btn.setCheckable(True)
        self._ortho_btn.setToolTip(self.tr("打开/关闭 3D 正交侧视图 (XZ 冠状面 & YZ 矢状面切片)  (Shift+V)"))
        self._ortho_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._ortho_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {t['surface']};
                border: 1px solid {t['border_light']};
                border-radius: 4px;
                color: {t['text']};
                padding: 3px 8px;
                font-size: 11px;
                font-weight: 500;
                min-height: 22px;
            }}
            QPushButton:hover {{
                background-color: {t['surface_hover']};
                border-color: {t['primary']};
            }}
            QPushButton:checked {{
                background-color: {t['primary']};
                color: #ffffff;
                border-color: {t['primary']};
                font-weight: bold;
            }}
        """)
        self._ortho_btn.toggled.connect(self.orthogonal_view_toggled.emit)
        left_layout.addWidget(self._ortho_btn)

        # 2. Crosshair Inspection Button
        self._crosshair_btn = QtWidgets.QPushButton(self.tr("✛ 十字线"))
        self._crosshair_btn.setCheckable(True)
        self._crosshair_btn.setToolTip(self.tr("开启/关闭十字准星交互 (点击切片任意位置定位体数据)"))
        self._crosshair_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._crosshair_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {t['surface']};
                border: 1px solid {t['border_light']};
                border-radius: 4px;
                color: {t['text']};
                padding: 3px 8px;
                font-size: 11px;
                font-weight: 500;
                min-height: 22px;
            }}
            QPushButton:hover {{
                background-color: {t['surface_hover']};
                border-color: {t['primary']};
            }}
            QPushButton:checked {{
                background-color: {t['primary']};
                color: #ffffff;
                border-color: {t['primary']};
                font-weight: bold;
            }}
        """)
        self._crosshair_btn.toggled.connect(self.crosshair_toggled.emit)
        left_layout.addWidget(self._crosshair_btn)

        # 3. Decompose Slices Button
        self._decompose_btn = QtWidgets.QPushButton(self.tr("⇱ 分解切片"))
        self._decompose_btn.setToolTip(self.tr("将当前 RAW 标注分解为单张切片图像 (jpg/png/bmp) 与对应标注 JSON，保存至同级新文件夹"))
        self._decompose_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._decompose_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {t['surface']};
                border: 1px solid {t['border_light']};
                border-radius: 4px;
                color: {t['text']};
                padding: 3px 8px;
                font-size: 11px;
                font-weight: 500;
                min-height: 22px;
            }}
            QPushButton:hover {{
                background-color: {t['surface_hover']};
                border-color: {t['primary']};
            }}
        """)
        self._decompose_btn.clicked.connect(self.decompose_requested.emit)
        left_layout.addWidget(self._decompose_btn)

        # 3. Voxel inspector text badge
        self._voxel_info_label = QtWidgets.QLabel()
        self._voxel_info_label.setStyleSheet(f"""
            QLabel {{
                color: {t['text']};
                font-size: 11px;
                background-color: {t['background']};
                border: 1px solid {t['border_light']};
                border-radius: 3px;
                padding: 2px 6px;
                min-height: 18px;
            }}
        """)
        self._voxel_info_label.setText(self.tr("点击画面或开启十字线查看体素"))
        left_layout.addWidget(self._voxel_info_label)
        left_layout.addStretch(1)

        main_layout.addWidget(left_container, 1)

        # Separator line between left tools and right Z navigation
        sep = QtWidgets.QFrame()
        sep.setFrameShape(QtWidgets.QFrame.Shape.VLine)
        sep.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        sep.setStyleSheet(f"color: {t['border']};")
        main_layout.addWidget(sep)

        # ==================== Right Half: Z-axis Navigation & Precision Slider ====================
        right_container = QtWidgets.QWidget()
        right_layout = QtWidgets.QHBoxLayout(right_container)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(6)

        # Title
        self._title_label = QtWidgets.QLabel(self.tr("Z轴:"))
        self._title_label.setStyleSheet(
            f"font-weight: bold; font-size: 12px; color: {t['text']};"
        )
        right_layout.addWidget(self._title_label)

        # Slider
        self._slider = QtWidgets.QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(0, 0)
        self._slider.setValue(0)
        self._slider.setTracking(True)
        self._slider.setStyleSheet(f"""
            QSlider::groove:horizontal {{
                height: 6px;
                background: {t['background']};
                border-radius: 3px;
                border: 1px solid {t['border']};
            }}
            QSlider::sub-page:horizontal {{
                background: {t['primary']};
                border-radius: 3px;
            }}
            QSlider::handle:horizontal {{
                background: #ffffff;
                border: 2px solid {t['primary']};
                width: 14px;
                margin-top: -5px;
                margin-bottom: -5px;
                border-radius: 7px;
            }}
            QSlider::handle:horizontal:hover {{
                background: {t['primary_hover']};
                border: 2px solid #ffffff;
            }}
        """)
        self._slider.valueChanged.connect(self._on_slider_value_changed)
        right_layout.addWidget(self._slider, 1)

        # Previous slice button
        self._prev_btn = QtWidgets.QToolButton()
        self._prev_btn.setText("◀")
        self._prev_btn.setToolTip(self.tr("上一层 (PageUp / [)"))
        self._prev_btn.setFixedSize(26, 24)
        self._prev_btn.setStyleSheet(f"""
            QToolButton {{
                background-color: {t['surface']};
                border: 1px solid {t['border_light']};
                border-radius: 4px;
                color: {t['text']};
                font-size: 11px;
            }}
            QToolButton:hover {{
                background-color: {t['surface_hover']};
            }}
            QToolButton:pressed {{
                background-color: {t['surface_pressed']};
            }}
            QToolButton:disabled {{
                color: {t['text_secondary']};
                border-color: {t['border']};
            }}
        """)
        self._prev_btn.clicked.connect(self.prev_slice)
        right_layout.addWidget(self._prev_btn)

        # Input box
        self._validator = QtGui.QIntValidator(0, 0, self)
        self._edit = QtWidgets.QLineEdit()
        self._edit.setValidator(self._validator)
        self._edit.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._edit.setText("0")
        self._edit.setFixedWidth(52)
        self._edit.setStyleSheet(f"""
            QLineEdit {{
                background-color: {t['background']};
                color: {t['text']};
                border: 1px solid {t['border_light']};
                border-radius: 4px;
                padding: 2px 4px;
                font-size: 12px;
                min-height: 20px;
            }}
            QLineEdit:focus {{
                border: 1px solid {t['primary']};
            }}
        """)
        self._edit.editingFinished.connect(self._on_edit_submitted)
        right_layout.addWidget(self._edit)

        # Next slice button
        self._next_btn = QtWidgets.QToolButton()
        self._next_btn.setText("▶")
        self._next_btn.setToolTip(self.tr("下一层 (PageDown / ])"))
        self._next_btn.setFixedSize(26, 24)
        self._next_btn.setStyleSheet(f"""
            QToolButton {{
                background-color: {t['surface']};
                border: 1px solid {t['border_light']};
                border-radius: 4px;
                color: {t['text']};
                font-size: 11px;
            }}
            QToolButton:hover {{
                background-color: {t['surface_hover']};
            }}
            QToolButton:pressed {{
                background-color: {t['surface_pressed']};
            }}
            QToolButton:disabled {{
                color: {t['text_secondary']};
                border-color: {t['border']};
            }}
        """)
        self._next_btn.clicked.connect(self.next_slice)
        right_layout.addWidget(self._next_btn)

        # Info label
        self._info_label = QtWidgets.QLabel()
        self._info_label.setStyleSheet(
            f"font-size: 11px; color: {t['text_secondary']}; padding-left: 4px;"
        )
        right_layout.addWidget(self._info_label)

        main_layout.addWidget(right_container, 1)

        # Container styling
        self.setStyleSheet(f"""
            RawZSlider {{
                background-color: {t['surface']};
                border-top: 1px solid {t['border']};
                border-bottom: 1px solid {t['border']};
            }}
        """)

        self.setVisible(False)

    def set_orthogonal_view_checked(self, checked: bool):
        """Update side view button check state without re-emitting signal."""
        with QtCore.QSignalBlocker(self._ortho_btn):
            self._ortho_btn.setChecked(checked)

    def set_crosshair_checked(self, checked: bool):
        """Update crosshair button check state without re-emitting signal."""
        with QtCore.QSignalBlocker(self._crosshair_btn):
            self._crosshair_btn.setChecked(checked)

    def set_voxel_info(
        self,
        x: int,
        y: int,
        z: int,
        val: Any,
        phys_z: Optional[float] = None,
        unit: Optional[str] = None,
    ):
        """Update voxel inspector readout."""
        z_str = f"Z: {z}"
        if phys_z is not None:
            u_str = f" {unit}" if unit else ""
            z_str += f" ({phys_z:.1f}{u_str})"

        val_str = str(val) if not hasattr(val, "tolist") else str(val.tolist())
        self._voxel_info_label.setText(
            f"📍 X: {x}   Y: {y}   {z_str}   |   值: {val_str}"
        )

    def set_volume_info(self, info: RawFileInfo, initial_z: int = 0):
        """Configure slider with volume metadata and set current slice."""
        self.file_info = info
        depth = max(1, info.depth)
        target_z = max(0, min(depth - 1, initial_z))

        self._validator.setRange(0, depth - 1)

        with QtCore.QSignalBlocker(self._slider):
            self._slider.setRange(0, depth - 1)
            self._slider.setValue(target_z)

        with QtCore.QSignalBlocker(self._edit):
            self._edit.setText(str(target_z))

        self._current_z = target_z
        self._update_info_display()
        self._update_button_states()

    def set_current_z(self, z: int):
        """Update current Z without emitting z_changed."""
        if not self.file_info:
            return
        z = max(0, min(self.file_info.depth - 1, z))
        if z != self._current_z:
            self._current_z = z
            with QtCore.QSignalBlocker(self._slider):
                self._slider.setValue(z)
            with QtCore.QSignalBlocker(self._edit):
                self._edit.setText(str(z))
            self._update_info_display()
            self._update_button_states()

    def get_current_z(self) -> int:
        return self._current_z

    def prev_slice(self):
        if self._current_z > 0:
            self.set_z_and_emit(self._current_z - 1)

    def next_slice(self):
        if self.file_info and self._current_z < self.file_info.depth - 1:
            self.set_z_and_emit(self._current_z + 1)

    def set_z_and_emit(self, z: int):
        if not self.file_info:
            return
        z = max(0, min(self.file_info.depth - 1, z))
        if z != self._current_z:
            self._current_z = z
            with QtCore.QSignalBlocker(self._slider):
                self._slider.setValue(z)
            with QtCore.QSignalBlocker(self._edit):
                self._edit.setText(str(z))
            self._update_info_display()
            self._update_button_states()
            self.z_changed.emit(z)

    def _on_slider_value_changed(self, value: int):
        if value != self._current_z:
            self._current_z = value
            with QtCore.QSignalBlocker(self._edit):
                self._edit.setText(str(value))
            self._update_info_display()
            self._update_button_states()
            self.z_changed.emit(value)

    def _on_edit_submitted(self):
        text = self._edit.text().strip()
        if not text:
            self._edit.setText(str(self._current_z))
            return
        try:
            val = int(text)
        except ValueError:
            self._edit.setText(str(self._current_z))
            return

        if not self.file_info:
            return
        val = max(0, min(self.file_info.depth - 1, val))
        self._edit.setText(str(val))
        if val != self._current_z:
            self.set_z_and_emit(val)

    def _update_info_display(self):
        if not self.file_info:
            self._info_label.setText("")
            return

        total_depth = self.file_info.depth
        slice_idx = self._current_z
        phys_z = self.file_info.get_slice_z_coord(slice_idx)

        z_str = f"Z: {phys_z}" if phys_z is not None else f"#{slice_idx}"
        dim_str = f"{self.file_info.width}×{self.file_info.height}"

        voxel_str = ""
        if self.file_info.pixel_width > 0:
            voxel_str = f" | {self.file_info.pixel_width:.1f} {self.file_info.unit}"

        self._info_label.setText(
            f"切片: {slice_idx + 1}/{total_depth} ({z_str}) | {dim_str}{voxel_str}"
        )

    def _update_button_states(self):
        if not self.file_info:
            self._prev_btn.setEnabled(False)
            self._next_btn.setEnabled(False)
            return
        self._prev_btn.setEnabled(self._current_z > 0)
        self._next_btn.setEnabled(self._current_z < self.file_info.depth - 1)

    def wheelEvent(self, event):
        """Scroll wheel adjusts Z slice when hovering over slider widget."""
        delta = event.angleDelta().y()
        if delta > 0:
            self.prev_slice()
        elif delta < 0:
            self.next_slice()
        event.accept()
