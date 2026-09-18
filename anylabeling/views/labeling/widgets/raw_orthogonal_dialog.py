"""Orthogonal Views (XZ and YZ side views) widgets for 3D RAW volume inspection."""

from typing import Optional, Tuple
from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import Qt

from anylabeling.views.labeling.utils.theme import get_theme
from anylabeling.views.labeling.utils.raw_reader import (
    RawVolume,
    RawFileInfo,
    slice_to_qimage,
)


class OrthogonalCanvasWidget(QtWidgets.QWidget):
    """Interactive canvas widget displaying a 2D orthogonal slice with crosshairs.

    Orientations:
    - "XZ" (attached BELOW XY):
        - Image width = W (horizontal is X, matching XY's X)
        - Image height = D (vertical is Z, depth)
        - Crosshair: vertical line at X, horizontal line at Z
        - Mouse click/drag emits point_clicked(x, z)
    - "YZ" (attached to the RIGHT of XY):
        - Image width = D (horizontal is Z, depth)
        - Image height = H (vertical is Y, matching XY's Y)
        - Crosshair: horizontal line at Y, vertical line at Z
        - Mouse click/drag emits point_clicked(y, z)
    """

    point_clicked = QtCore.pyqtSignal(int, int)  # (first_coord, second_coord)

    def __init__(
        self,
        view_type: str = "XZ",
        parent: Optional[QtWidgets.QWidget] = None,
    ):
        super().__init__(parent)
        self.view_type = view_type.upper()  # "XZ" or "YZ"
        self._pixmap: Optional[QtGui.QPixmap] = None
        self._cross_x: int = 0  # For XZ
        self._cross_y: int = 0  # For YZ
        self._cross_z: int = 0  # For both
        self._aspect_ratio_z: float = 1.0  # slice_thickness / pixel_width

        self._img_w: int = 0
        self._img_h: int = 0
        self._draw_rect = QtCore.QRectF()

        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
        self.setMinimumSize(120, 100)

    def set_slice_data(
        self,
        slice_arr,
        cross_primary: int,
        cross_z: int,
        aspect_ratio_z: float = 1.0,
    ):
        """Update slice image and crosshair position.

        For XZ: cross_primary is X, cross_z is Z.
        For YZ: cross_primary is Y, cross_z is Z.
        """
        if slice_arr is None or slice_arr.size == 0:
            self._pixmap = None
            self._img_w = 0
            self._img_h = 0
            self.update()
            return

        qimg = slice_to_qimage(slice_arr)
        if not qimg.isNull():
            self._pixmap = QtGui.QPixmap.fromImage(qimg)
            self._img_w = self._pixmap.width()
            self._img_h = self._pixmap.height()
        else:
            self._pixmap = None
            self._img_w = 0
            self._img_h = 0

        if self.view_type == "XZ":
            self._cross_x = cross_primary
        else:
            self._cross_y = cross_primary
        self._cross_z = cross_z
        self._aspect_ratio_z = max(0.01, aspect_ratio_z)
        self.update()

    def update_crosshair(self, cross_primary: int, cross_z: int):
        """Update only the crosshair position without re-loading pixmap."""
        if self.view_type == "XZ":
            self._cross_x = cross_primary
        else:
            self._cross_y = cross_primary
        self._cross_z = cross_z
        self.update()

    def paintEvent(self, event: QtGui.QPaintEvent):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, False)
        painter.setRenderHint(QtGui.QPainter.RenderHint.SmoothPixmapTransform, True)

        # Background
        t = get_theme()
        bg_color = QtGui.QColor(t.get("background_secondary", "#1a1a1a"))
        painter.fillRect(self.rect(), bg_color)

        if self._pixmap is None or self._img_w == 0 or self._img_h == 0:
            painter.setPen(QtGui.QColor(t.get("text_secondary", "#888888")))
            painter.drawText(
                self.rect(),
                Qt.AlignmentFlag.AlignCenter,
                self.tr("无切片数据"),
            )
            return

        # Calculate display rect preserving aspect ratio
        widget_w = max(10, self.width() - 8)
        widget_h = max(10, self.height() - 8)

        if self.view_type == "XZ":
            # Width is X, height is Z
            effective_img_w = float(self._img_w)
            effective_img_h = float(self._img_h) * self._aspect_ratio_z
        else:
            # Width is Z, height is Y
            effective_img_w = float(self._img_w) * self._aspect_ratio_z
            effective_img_h = float(self._img_h)

        scale_x = widget_w / max(1.0, effective_img_w)
        scale_y = widget_h / max(1.0, effective_img_h)
        scale = min(scale_x, scale_y)

        draw_w = effective_img_w * scale
        draw_h = effective_img_h * scale
        offset_x = 4 + (widget_w - draw_w) / 2.0
        offset_y = 4 + (widget_h - draw_h) / 2.0
        self._draw_rect = QtCore.QRectF(offset_x, offset_y, draw_w, draw_h)

        # Draw image
        painter.drawPixmap(self._draw_rect.toRect(), self._pixmap)

        # Border around image
        painter.setPen(QtGui.QPen(QtGui.QColor(t.get("border", "#444444")), 1))
        painter.drawRect(self._draw_rect.adjusted(-1, -1, 1, 1))

        # Calculate crosshair positions
        if self.view_type == "XZ":
            u_ratio = self._cross_x / max(1, self._img_w - 1) if self._img_w > 1 else 0.5
            v_ratio = self._cross_z / max(1, self._img_h - 1) if self._img_h > 1 else 0.5
        else:
            u_ratio = self._cross_z / max(1, self._img_w - 1) if self._img_w > 1 else 0.5
            v_ratio = self._cross_y / max(1, self._img_h - 1) if self._img_h > 1 else 0.5

        cx = offset_x + u_ratio * draw_w
        cy = offset_y + v_ratio * draw_h

        # Contrast outline
        pen_bg = QtGui.QPen(QtGui.QColor(0, 0, 0, 200), 2)
        pen_bg.setCosmetic(True)
        painter.setPen(pen_bg)
        painter.drawLine(QtCore.QPointF(offset_x, cy), QtCore.QPointF(offset_x + draw_w, cy))
        painter.drawLine(QtCore.QPointF(cx, offset_y), QtCore.QPointF(cx, offset_y + draw_h))

        # Vibrant inner line (yellow)
        pen_fg = QtGui.QPen(QtGui.QColor("#FFFF00"), 1)
        pen_fg.setCosmetic(True)
        painter.setPen(pen_fg)
        painter.drawLine(QtCore.QPointF(offset_x, cy), QtCore.QPointF(offset_x + draw_w, cy))
        painter.drawLine(QtCore.QPointF(cx, offset_y), QtCore.QPointF(cx, offset_y + draw_h))


    def _map_widget_to_image(self, pos: QtCore.QPointF) -> Tuple[int, int]:
        """Convert widget coordinates to primary and secondary coordinates."""
        if self._draw_rect.isEmpty() or self._img_w == 0 or self._img_h == 0:
            return 0, 0

        rx = (pos.x() - self._draw_rect.left()) / max(1e-4, self._draw_rect.width())
        ry = (pos.y() - self._draw_rect.top()) / max(1e-4, self._draw_rect.height())

        rx = max(0.0, min(1.0, rx))
        ry = max(0.0, min(1.0, ry))

        if self.view_type == "XZ":
            x = int(round(rx * (self._img_w - 1)))
            z = int(round(ry * (self._img_h - 1)))
            return x, z
        else:
            z = int(round(rx * (self._img_w - 1)))
            y = int(round(ry * (self._img_h - 1)))
            return y, z

    def mousePressEvent(self, event: QtGui.QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            c1, c2 = self._map_widget_to_image(event.position())
            self.point_clicked.emit(c1, c2)
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent):
        if event.buttons() & Qt.MouseButton.LeftButton:
            c1, c2 = self._map_widget_to_image(event.position())
            self.point_clicked.emit(c1, c2)
            event.accept()
        else:
            super().mouseMoveEvent(event)


class RawOrthoCornerWidget(QtWidgets.QFrame):
    """Corner widget placed at bottom-right (below YZ, right of XZ).

    Displays live coordinates, physical values, and controls.
    """

    close_requested = QtCore.pyqtSignal()
    popout_requested = QtCore.pyqtSignal()

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.setObjectName("RawOrthoCornerWidget")
        self.setMinimumSize(120, 100)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(4)

        # Header with badge and close button
        header_layout = QtWidgets.QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 0)

        self.title_label = QtWidgets.QLabel(self.tr("📍 正交空间联动"))
        self.title_label.setStyleSheet("font-weight: bold; color: #4FC3F7;")
        header_layout.addWidget(self.title_label)
        header_layout.addStretch()

        self.close_btn = QtWidgets.QToolButton(self)
        self.close_btn.setText("✕")
        self.close_btn.setToolTip(self.tr("关闭侧视图"))
        self.close_btn.setStyleSheet(
            "QToolButton { border: none; font-size: 12px; font-weight: bold; color: #888; padding: 2px; } "
            "QToolButton:hover { color: #f44336; }"
        )
        self.close_btn.clicked.connect(self.close_requested.emit)
        header_layout.addWidget(self.close_btn)

        layout.addLayout(header_layout)

        # Volume shape info
        self.vol_info_label = QtWidgets.QLabel(self.tr("尺寸: - × - × -"))
        self.vol_info_label.setStyleSheet("color: #AAA; font-size: 11px;")
        layout.addWidget(self.vol_info_label)

        # Coordinates info
        self.coords_label = QtWidgets.QLabel(self.tr("X: 0 | Y: 0 | Z: 0"))
        self.coords_label.setStyleSheet("font-weight: 500; font-size: 11px; color: #E0E0E0;")
        layout.addWidget(self.coords_label)

        # Physical Z
        self.phys_z_label = QtWidgets.QLabel(self.tr("高度: -"))
        self.phys_z_label.setStyleSheet("color: #FFB74D; font-size: 11px;")
        layout.addWidget(self.phys_z_label)

        # Voxel value
        self.val_label = QtWidgets.QLabel(self.tr("体素值: -"))
        self.val_label.setStyleSheet("color: #81C784; font-size: 11px; font-weight: bold;")
        layout.addWidget(self.val_label)

        layout.addStretch()

        self.setStyleSheet(
            "#RawOrthoCornerWidget {"
            " background-color: #212121;"
            " border: 1px solid #383838;"
            " border-radius: 4px;"
            "}"
        )

    def set_volume_info(self, info: RawFileInfo):
        """Set volume dimensions and pixel resolution."""
        if info:
            self.vol_info_label.setText(
                f"{info.width} × {info.height} × {info.depth} ({info.pixel_width} {info.unit})"
            )

    def set_coordinates(
        self,
        x: int,
        y: int,
        z: int,
        val,
        phys_z: Optional[float] = None,
        unit: str = "µm",
    ):
        """Update live coordinate and voxel values."""
        self.coords_label.setText(f"X: {x}   Y: {y}   Z: {z}")
        if phys_z is not None:
            self.phys_z_label.setText(f"物理Z: {phys_z:.1f} {unit}")
        else:
            self.phys_z_label.setText(f"切片: #{z}")

        if isinstance(val, (tuple, list, QtGui.QColor)) or hasattr(val, "__len__"):
            if hasattr(val, "red"):
                self.val_label.setText(f"RGB: ({val.red()}, {val.green()}, {val.blue()})")
            elif len(val) >= 3:
                self.val_label.setText(f"RGB: ({val[0]}, {val[1]}, {val[2]})")
            else:
                self.val_label.setText(f"值: {val}")
        else:
            self.val_label.setText(f"体素值: {val}")


class RawOrthogonalViewDialog(QtWidgets.QDialog):
    """Optional pop-out dialog for orthogonal views (retained for compatibility)."""

    point_changed = QtCore.pyqtSignal(int, int, int)  # (x, y, z)
    visibility_changed = QtCore.pyqtSignal(bool)

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle(self.tr("3D 正交侧视图 (Orthogonal Views)"))
        self.setWindowFlags(
            Qt.WindowType.Window
            | Qt.WindowType.WindowCloseButtonHint
            | Qt.WindowType.WindowMinMaxButtonsHint
        )
        self.resize(720, 480)

        self.raw_volume: Optional[RawVolume] = None
        self.raw_info: Optional[RawFileInfo] = None
        self.curr_x: int = 0
        self.curr_y: int = 0
        self.curr_z: int = 0

        layout = QtWidgets.QGridLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        self.xz_canvas = OrthogonalCanvasWidget(view_type="XZ", parent=self)
        self.yz_canvas = OrthogonalCanvasWidget(view_type="YZ", parent=self)
        self.corner_widget = RawOrthoCornerWidget(parent=self)

        layout.addWidget(self.yz_canvas, 0, 1)
        layout.addWidget(self.xz_canvas, 1, 0)
        layout.addWidget(self.corner_widget, 1, 1)

        self.xz_canvas.point_clicked.connect(self._on_xz_clicked)
        self.yz_canvas.point_clicked.connect(self._on_yz_clicked)
        self.corner_widget.close_requested.connect(self.close)

    def set_volume_data(
        self,
        volume: RawVolume,
        info: RawFileInfo,
        x: int,
        y: int,
        z: int,
    ):
        self.raw_volume = volume
        self.raw_info = info
        self.curr_x = x
        self.curr_y = y
        self.curr_z = z
        self.corner_widget.set_volume_info(info)
        self.update_coordinates(x, y, z)

    def update_coordinates(self, x: int, y: int, z: int):
        if self.raw_volume is None or self.raw_info is None:
            return
        self.curr_x = x
        self.curr_y = y
        self.curr_z = z

        aspect_z = 1.0
        if self.raw_info.pixel_width > 0:
            aspect_z = self.raw_info.slice_thickness / self.raw_info.pixel_width

        coronal = self.raw_volume.get_coronal_slice(y)
        self.xz_canvas.set_slice_data(coronal, x, z, aspect_ratio_z=aspect_z)

        sagittal = self.raw_volume.get_sagittal_slice(x)
        self.yz_canvas.set_slice_data(sagittal, y, z, aspect_ratio_z=aspect_z)

        val = self.raw_volume.get_voxel_value(x, y, z)
        phys_z = self.raw_info.get_slice_z_coord(z)
        self.corner_widget.set_coordinates(x, y, z, val, phys_z=phys_z, unit=self.raw_info.unit)

    def _on_xz_clicked(self, x: int, z: int):
        self.curr_x = x
        self.curr_z = z
        self.point_changed.emit(x, self.curr_y, z)

    def _on_yz_clicked(self, y: int, z: int):
        self.curr_y = y
        self.curr_z = z
        self.point_changed.emit(self.curr_x, y, z)

    def closeEvent(self, event):
        self.visibility_changed.emit(False)
        super().closeEvent(event)
