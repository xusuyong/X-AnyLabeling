"""Z-axis slider and volume inspection widget for 3D RAW volume navigation."""

from typing import Optional, Any, Dict, List, Set, Union
from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import Qt

from anylabeling.views.labeling.utils.theme import get_theme
from anylabeling.views.labeling.utils.raw_reader import RawFileInfo


class _AnnotatedSlider(QtWidgets.QSlider):
    """Horizontal slider with visual markers on slices containing annotations."""

    def __init__(
        self,
        orientation: Qt.Orientation = Qt.Orientation.Horizontal,
        parent: Optional[QtWidgets.QWidget] = None,
    ):
        super().__init__(orientation, parent)
        self._annotated_slices: Dict[int, int] = {}

    def set_annotated_slices(self, slices: Dict[int, int]):
        self._annotated_slices = dict(slices)
        self.update()

    def paintEvent(self, event: QtGui.QPaintEvent):
        super().paintEvent(event)
        if not self._annotated_slices or self.maximum() <= self.minimum():
            return

        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)

        opt = QtWidgets.QStyleOptionSlider()
        self.initStyleOption(opt)
        groove_rect = self.style().subControlRect(
            QtWidgets.QStyle.ComplexControl.CC_Slider,
            opt,
            QtWidgets.QStyle.SubControl.SC_SliderGroove,
            self,
        )
        handle_rect = self.style().subControlRect(
            QtWidgets.QStyle.ComplexControl.CC_Slider,
            opt,
            QtWidgets.QStyle.SubControl.SC_SliderHandle,
            self,
        )

        min_val = self.minimum()
        max_val = self.maximum()
        span = groove_rect.width() - handle_rect.width()
        if span <= 0:
            return

        half_handle = handle_rect.width() / 2.0
        y_center = groove_rect.center().y()
        marker_w = 4.0
        marker_h = 13.0

        for slice_idx, count in self._annotated_slices.items():
            if slice_idx < min_val or slice_idx > max_val:
                continue
            x_pos = QtWidgets.QStyle.sliderPositionFromValue(
                min_val, max_val, slice_idx, span, opt.upsideDown
            )
            cx = groove_rect.x() + half_handle + x_pos
            rect = QtCore.QRectF(
                cx - marker_w / 2.0,
                y_center - marker_h / 2.0,
                marker_w,
                marker_h,
            )
            painter.setBrush(QtGui.QColor("#FF9F0A"))
            painter.setPen(QtGui.QPen(QtGui.QColor("#C77700"), 1.0))
            painter.drawRoundedRect(rect, 2.0, 2.0)


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
        self._annotated_slices: Dict[int, int] = {}
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

        # Slider with annotation markers
        self._slider = _AnnotatedSlider(Qt.Orientation.Horizontal)
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

        # Previous annotated slice button
        self._prev_annotated_btn = QtWidgets.QToolButton()
        self._prev_annotated_btn.setText("⏮")
        self._prev_annotated_btn.setToolTip(
            self.tr("上一个标注层 (Alt+[ / Ctrl+PageUp)")
        )
        self._prev_annotated_btn.setFixedSize(26, 24)
        self._prev_annotated_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._prev_annotated_btn.setStyleSheet(f"""
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
        self._prev_annotated_btn.clicked.connect(self.jump_to_prev_annotated)
        right_layout.addWidget(self._prev_annotated_btn)

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

        # Next annotated slice button
        self._next_annotated_btn = QtWidgets.QToolButton()
        self._next_annotated_btn.setText("⏭")
        self._next_annotated_btn.setToolTip(
            self.tr("下一个标注层 (Alt+] / Ctrl+PageDown)")
        )
        self._next_annotated_btn.setFixedSize(26, 24)
        self._next_annotated_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._next_annotated_btn.setStyleSheet(f"""
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
        self._next_annotated_btn.clicked.connect(self.jump_to_next_annotated)
        right_layout.addWidget(self._next_annotated_btn)

        # Annotated slices summary badge & jump menu
        self._annotated_btn = QtWidgets.QPushButton(self.tr("🏷️ 无标注"))
        self._annotated_btn.setToolTip(
            self.tr("查看体数据所有已标注层 (点击快速跳转)")
        )
        self._annotated_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._annotated_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {t['background']};
                border: 1px solid {t['border_light']};
                border-radius: 4px;
                color: {t['text_secondary']};
                font-size: 11px;
                padding: 2px 7px;
                min-height: 22px;
            }}
        """)
        self._annotated_btn.clicked.connect(self._show_annotated_menu)
        right_layout.addWidget(self._annotated_btn)

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

    def set_volume_info(
        self,
        info: RawFileInfo,
        initial_z: int = 0,
        annotated_slices: Optional[Union[Dict[int, int], Set[int], List[int]]] = None,
    ):
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
        if annotated_slices is not None:
            self.set_annotated_slices(annotated_slices)
        else:
            self._update_annotated_badge()
            self._update_info_display()
            self._update_button_states()

    def set_annotated_slices(
        self, slices: Union[Dict[int, int], Set[int], List[int]]
    ):
        """Update the set/dict of slices that contain annotations."""
        if isinstance(slices, dict):
            self._annotated_slices = {
                int(k): int(v) for k, v in slices.items() if v > 0
            }
        else:
            self._annotated_slices = {int(z): 1 for z in slices}

        self._slider.set_annotated_slices(self._annotated_slices)
        self._update_annotated_badge()
        self._update_button_states()
        self._update_info_display()

    def get_annotated_slices(self) -> Dict[int, int]:
        """Return dict of {slice_index: count} for annotated slices."""
        return dict(self._annotated_slices)

    def jump_to_prev_annotated(self):
        """Jump to the nearest annotated slice with z < current_z, or wrap around to the last."""
        if not self._annotated_slices:
            return
        candidates = [z for z in self._annotated_slices if z < self._current_z]
        if candidates:
            self.set_z_and_emit(max(candidates))
        else:
            self.set_z_and_emit(max(self._annotated_slices.keys()))

    def jump_to_next_annotated(self):
        """Jump to the nearest annotated slice with z > current_z, or wrap around to the first."""
        if not self._annotated_slices:
            return
        candidates = [z for z in self._annotated_slices if z > self._current_z]
        if candidates:
            self.set_z_and_emit(min(candidates))
        else:
            self.set_z_and_emit(min(self._annotated_slices.keys()))

    def _update_annotated_badge(self):
        t = get_theme()
        sorted_slices = sorted(self._annotated_slices.keys())
        total_annotated = len(sorted_slices)

        if total_annotated == 0:
            self._annotated_btn.setText(self.tr("🏷️ 无标注"))
            self._annotated_btn.setToolTip(self.tr("当前体数据暂无标注层"))
            self._annotated_btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: {t['background']};
                    border: 1px solid {t['border_light']};
                    border-radius: 4px;
                    color: {t['text_secondary']};
                    font-size: 11px;
                    padding: 2px 7px;
                    min-height: 22px;
                }}
                QPushButton:hover {{
                    border-color: {t['border']};
                }}
            """)
            return

        total_shapes = sum(self._annotated_slices.values())
        btn_text = f"🏷️ {total_annotated}层已标注"
        self._annotated_btn.setText(btn_text)

        lines = [
            self.tr(
                "体数据已标注切片 (共 {count} 层, {shapes} 个对象，点击选择跳转):"
            ).format(count=total_annotated, shapes=total_shapes)
        ]
        for z in sorted_slices[:25]:
            phys_z = (
                self.file_info.get_slice_z_coord(z)
                if self.file_info
                else None
            )
            phys_str = f", Z={phys_z:.2f}" if phys_z is not None else ""
            cnt = self._annotated_slices[z]
            lines.append(f"  • 切片 #{z} (第 {z + 1} 层{phys_str}): {cnt} 个标注")
        if total_annotated > 25:
            lines.append(f"  ... 另有 {total_annotated - 25} 层")

        self._annotated_btn.setToolTip("\n".join(lines))
        self._annotated_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: rgba(255, 159, 10, 0.12);
                border: 1px solid #FF9F0A;
                border-radius: 4px;
                color: #FF9F0A;
                font-size: 11px;
                font-weight: bold;
                padding: 2px 7px;
                min-height: 22px;
            }}
            QPushButton:hover {{
                background-color: rgba(255, 159, 10, 0.25);
                border-color: #FFAE33;
            }}
            QPushButton:pressed {{
                background-color: rgba(255, 159, 10, 0.35);
            }}
        """)

    def _show_annotated_menu(self):
        sorted_slices = sorted(self._annotated_slices.keys())
        if not sorted_slices:
            QtWidgets.QToolTip.showText(
                self._annotated_btn.mapToGlobal(
                    QtCore.QPoint(0, self._annotated_btn.height())
                ),
                self.tr("当前体数据暂无标注层"),
                self._annotated_btn,
            )
            return

        menu = QtWidgets.QMenu(self)
        for z in sorted_slices:
            phys_z = (
                self.file_info.get_slice_z_coord(z)
                if self.file_info
                else None
            )
            phys_str = f" (Z={phys_z:.2f})" if phys_z is not None else ""
            cnt = self._annotated_slices[z]
            is_cur = "● " if z == self._current_z else "   "
            title = f"{is_cur}切片 #{z} (第 {z + 1} 层{phys_str})  —  {cnt} 个标注"
            action = menu.addAction(title)
            action.triggered.connect(
                lambda checked=False, target_z=z: self.set_z_and_emit(target_z)
            )

        menu.exec(
            self._annotated_btn.mapToGlobal(
                QtCore.QPoint(0, self._annotated_btn.height())
            )
        )

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
        dim_str = f"{self.file_info.width}×{self.file_info.height}"

        voxel_str = ""
        if self.file_info.pixel_width > 0:
            voxel_str = f" | {self.file_info.pixel_width:.1f} {self.file_info.unit}"

        self._info_label.setText(f"切片: {slice_idx + 1}/{total_depth} | {dim_str}{voxel_str}")

    def _update_button_states(self):
        if not self.file_info:
            self._prev_btn.setEnabled(False)
            self._next_btn.setEnabled(False)
            self._prev_annotated_btn.setEnabled(False)
            self._next_annotated_btn.setEnabled(False)
            return
        self._prev_btn.setEnabled(self._current_z > 0)
        self._next_btn.setEnabled(self._current_z < self.file_info.depth - 1)

        has_ann = bool(self._annotated_slices)
        self._prev_annotated_btn.setEnabled(has_ann)
        self._next_annotated_btn.setEnabled(has_ann)

    def wheelEvent(self, event):
        """Scroll wheel adjusts Z slice when hovering over slider widget."""
        delta = event.angleDelta().y()
        if delta > 0:
            self.prev_slice()
        elif delta < 0:
            self.next_slice()
        event.accept()
