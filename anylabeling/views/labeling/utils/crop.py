import json
import multiprocessing
import os
import os.path as osp
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
    QFileDialog,
    QGridLayout,
    QLabel,
    QLineEdit,
    QHBoxLayout,
    QProgressDialog,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QMessageBox,
)

from anylabeling.views.labeling.logger import logger
from anylabeling.views.labeling.widgets import Popup
from anylabeling.views.labeling.utils.general import (
    resolve_export_directory,
    resolve_path_within_directory,
)
from anylabeling.views.labeling.utils.qt import new_icon_path
from anylabeling.views.labeling.utils.style import (
    get_cancel_btn_style,
    get_checkbox_indicator_style,
    get_export_option_style,
    get_ok_btn_style,
    get_msg_box_style,
    get_progress_dialog_style,
    get_spinbox_style,
)

__all__ = ["save_crop"]


def process_single_image(args):
    """Process a single image with cropping parameters

    Args:
        args: Tuple containing
        (image_file, label_dir_path, save_path, min_width, min_height, padding,
         draw_box, box_thickness, label_start_indices)
    """
    (
        image_file,
        label_dir_path,
        save_path,
        min_width,
        min_height,
        padding,
        draw_box,
        box_thickness,
        label_start_indices,
    ) = args
    try:
        image_name = osp.basename(image_file)
        label_file = osp.join(
            label_dir_path, osp.splitext(image_name)[0] + ".json"
        )

        if not osp.exists(label_file):
            return True

        with open(label_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        shapes = data.get("shapes", [])
        image_path = Path(image_file)
        orig_filename = image_path.stem

        try:
            image = cv2.imdecode(
                np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR
            )
            if image is None:
                raise ValueError(f"Failed to read image: {image_file}")
        except Exception as e:
            logger.error(f"Error reading image: {str(e)}")
            return False

        for shape in shapes:
            label = shape.get("label", "")
            points = np.array(shape.get("points", [])).astype(np.int32)
            shape_type = shape.get("shape_type", "")

            if (
                shape_type not in ["rectangle", "polygon", "rotation"]
                or len(points) < 3
            ):
                continue

            current_index = label_start_indices.get(label, 0) + 1
            label_start_indices[label] = current_index

            x, y, w, h = cv2.boundingRect(points)
            if w < min_width or h < min_height:
                continue

            height, width = image.shape[:2]
            xmin, ymin = max(0, x - padding), max(0, y - padding)
            xmax, ymax = min(width, x + w + padding), min(height, y + h + padding)

            if xmin >= xmax or ymin >= ymax:
                logger.warning(
                    f"Invalid crop region: xmin={xmin}, xmax={xmax}, ymin={ymin}, ymax={ymax}"
                )
                continue

            cropped_image = image[ymin:ymax, xmin:xmax]
            if cropped_image.size == 0:
                logger.warning(f"Empty cropped image for {orig_filename}, skipping")
                continue

            # === 在裁剪图像上画框（相对坐标） ===
            if draw_box:
                shifted_points = points - [xmin, ymin]  # 把点移动到裁剪区域坐标系
                cv2.polylines(
                    cropped_image,
                    [shifted_points],
                    isClosed=True,
                    color=(0, 255, 0),
                    thickness=box_thickness,
                )

            dst_path = Path(save_path) / label
            dst_path.mkdir(parents=True, exist_ok=True)

            dst_file = resolve_path_within_directory(
                dst_path / f"{orig_filename}_{current_index}-{shape_type}.jpg",
                save_path,
            )

            try:
                is_success, buf = cv2.imencode(".jpg", cropped_image)
                if is_success and buf is not None:
                    with open(str(dst_file), "wb") as f:
                        f.write(buf.tobytes())
                else:
                    raise ValueError(f"Failed to save image: {dst_file}")
            except Exception as e:
                logger.error(f"Error saving image: {str(e)}")

        return True
    except Exception as e:
        logger.error(f"Error processing {image_file}: {str(e)}")
        return False


def save_crop(self):
    """Save the cropped image with multiprocessing optimization"""

    if not self.filename:
        popup = Popup(
            self.tr("Please load an image folder before proceeding!"),
            self,
            msec=1000,
            icon=new_icon_path("warning", "svg"),
        )
        popup.show_popup(self, position="center")
        return

    dialog = QDialog(self)
    dialog.setWindowTitle(self.tr("Cropped Image Options"))
    dialog.setMinimumWidth(500)
    dialog.setStyleSheet(get_export_option_style())

    layout = QVBoxLayout()
    layout.setContentsMargins(24, 24, 24, 24)
    layout.setSpacing(16)

    path_label = QLabel(self.tr("Save Path"))
    layout.addWidget(path_label)

    path_input_layout = QHBoxLayout()
    path_input_layout.setSpacing(8)

    path_edit = QLineEdit()
    path_edit.setText(
        osp.realpath(osp.join(osp.dirname(self.filename), "..", "crops"))
    )
    path_edit.setPlaceholderText(self.tr("Select Save Directory"))

    def browse_export_path():
        path = QFileDialog.getExistingDirectory(
            self,
            self.tr("Select Save Directory"),
            path_edit.text(),
            QFileDialog.Option.DontUseNativeDialog,
        )
        if path:
            path_edit.setText(path)

    path_button = QPushButton(self.tr("Browse"))
    path_button.clicked.connect(browse_export_path)
    path_button.setStyleSheet(get_cancel_btn_style())

    path_input_layout.addWidget(path_edit)
    path_input_layout.addWidget(path_button)
    layout.addLayout(path_input_layout)

    # Grid keeps spinboxes and buttons in the same right-side column
    grid = QGridLayout()
    grid.setHorizontalSpacing(12)
    grid.setVerticalSpacing(12)
    grid.setColumnStretch(0, 1)

    min_width_label = QLabel(self.tr("Minimum width:"))
    min_width_spin = QSpinBox()
    min_width_spin.setRange(0, 10000)
    min_width_spin.setValue(0)
    min_width_spin.setStyleSheet(get_spinbox_style())
    grid.addWidget(min_width_label, 0, 0)
    grid.addWidget(min_width_spin, 0, 1)

    min_height_label = QLabel(self.tr("Minimum height:"))
    min_height_spin = QSpinBox()
    min_height_spin.setRange(0, 10000)
    min_height_spin.setValue(0)
    min_height_spin.setStyleSheet(get_spinbox_style())
    grid.addWidget(min_height_label, 1, 0)
    grid.addWidget(min_height_spin, 1, 1)

    padding_label = QLabel(self.tr("Padding (pixels):"))
    padding_label.setToolTip(
        self.tr(
            "Expand the crop region outward by this many pixels. "
            "Set to 0 to crop tightly."
        )
    )
    padding_spin = QSpinBox()
    padding_spin.setRange(0, 10000)
    padding_spin.setValue(30)
    padding_spin.setSuffix(" px")
    padding_spin.setToolTip(padding_label.toolTip())
    padding_spin.setStyleSheet(get_spinbox_style())
    grid.addWidget(padding_label, 2, 0)
    grid.addWidget(padding_spin, 2, 1)

    draw_box_label = QLabel(self.tr("Draw box:"))
    draw_box_label.setToolTip(
        self.tr(
            "Draw a green outline of the annotation on the cropped image."
        )
    )
    draw_box_checkbox = QCheckBox()
    draw_box_checkbox.setChecked(True)
    draw_box_checkbox.setToolTip(draw_box_label.toolTip())
    draw_box_checkbox.setStyleSheet(get_checkbox_indicator_style())
    grid.addWidget(draw_box_label, 3, 0)
    grid.addWidget(draw_box_checkbox, 3, 1)

    thickness_label = QLabel(self.tr("Box thickness:"))
    thickness_label.setToolTip(
        self.tr("Line thickness of the drawn box (in pixels).")
    )
    thickness_spin = QSpinBox()
    thickness_spin.setRange(1, 100)
    thickness_spin.setValue(2)
    thickness_spin.setSuffix(" px")
    thickness_spin.setToolTip(thickness_label.toolTip())
    thickness_spin.setStyleSheet(get_spinbox_style())
    grid.addWidget(thickness_label, 4, 0)
    grid.addWidget(thickness_spin, 4, 1)

    def _update_thickness_enabled(_state):
        thickness_spin.setEnabled(draw_box_checkbox.isChecked())
        thickness_label.setEnabled(draw_box_checkbox.isChecked())

    draw_box_checkbox.stateChanged.connect(_update_thickness_enabled)

    cancel_button = QPushButton(self.tr("Cancel"))
    cancel_button.clicked.connect(dialog.reject)
    cancel_button.setStyleSheet(get_cancel_btn_style())

    ok_button = QPushButton(self.tr("OK"))
    ok_button.clicked.connect(dialog.accept)
    ok_button.setStyleSheet(get_ok_btn_style())

    button_layout = QHBoxLayout()
    button_layout.setContentsMargins(0, 4, 0, 0)
    button_layout.setSpacing(8)
    button_layout.addWidget(cancel_button)
    button_layout.addWidget(ok_button)
    grid.addLayout(button_layout, 5, 1)

    layout.addLayout(grid)

    dialog.setLayout(layout)
    result = dialog.exec()

    if not result:
        return

    save_path = path_edit.text()

    if osp.exists(save_path):
        msg_box = QMessageBox(self)
        msg_box.setIcon(QMessageBox.Icon.Warning)
        msg_box.setWindowTitle(self.tr("Output Directory Exists!"))
        msg_box.setText(self.tr("Directory already exists. Choose an action:"))
        msg_box.setInformativeText(
            self.tr(
                "• Overwrite - Overwrite existing directory\n"
                "• Cancel - Abort export"
            )
        )

        msg_box.addButton(self.tr("Overwrite"), QMessageBox.ButtonRole.YesRole)
        cancel_button = msg_box.addButton(
            self.tr("Cancel"), QMessageBox.ButtonRole.RejectRole
        )
        msg_box.setStyleSheet(get_msg_box_style())
        msg_box.exec()

        clicked_button = msg_box.clickedButton()
        if clicked_button == cancel_button:
            return
        else:
            shutil.rmtree(save_path)
            os.makedirs(save_path)
    else:
        os.makedirs(save_path)

    image_file_list = (
        [self.filename] if not self.image_list else self.image_list
    )
    label_dir_path = self.output_dir or osp.dirname(self.filename)

    progress_dialog = QProgressDialog(
        self.tr("Processing..."),
        self.tr("Cancel"),
        0,
        len(image_file_list),
        self,
    )
    progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
    progress_dialog.setWindowTitle(self.tr("Progress"))
    progress_dialog.setMinimumWidth(400)
    progress_dialog.setMinimumHeight(150)
    progress_dialog.setStyleSheet(
        get_progress_dialog_style(color="#1d1d1f", height=20)
    )
    progress_dialog.show()

    QApplication.processEvents()

    try:
        process_args = [
            (
                image_file,
                label_dir_path,
                save_path,
                min_width_spin.value(),
                min_height_spin.value(),
                padding_spin.value(),
                draw_box_checkbox.isChecked(),
                thickness_spin.value(),
                {},  # 每张图独立计数，文件名用 orig_filename 区分
            )
            for image_file in image_file_list
        ]

        is_frozen = getattr(sys, "frozen", False)

        if is_frozen:
            logger.info(
                "Running in PyInstaller environment, using single-thread processing"
            )
            for i, args in enumerate(process_args):
                process_single_image(args)
                progress_dialog.setValue(i + 1)
                QApplication.processEvents()

                if progress_dialog.wasCanceled():
                    return
        else:
            # Use multiprocessing to process images in parallel in the dev environment only.
            num_cores = max(1, int(multiprocessing.cpu_count() * 0.9))
            with multiprocessing.Pool(processes=num_cores) as pool:
                for i, _ in enumerate(
                    pool.imap(process_single_image, process_args)
                ):
                    progress_dialog.setValue(i + 1)
                    QApplication.processEvents()

                    if progress_dialog.wasCanceled():
                        pool.terminate()
                        pool.join()
                        return

        progress_dialog.close()
        popup = Popup(
            self.tr(
                f"Cropped images successfully!\nResults have been saved to:\n{save_path}"
            ),
            self,
            msec=3000,
            icon=new_icon_path("copy-green", "svg"),
        )
        popup.show_popup(self, popup_height=65, position="center")

    except Exception as e:
        logger.error(f"Error occurred while exporting cropped images: {e}")
        popup = Popup(
            self.tr(f"Error occurred while exporting cropped images!"),
            self,
            msec=3000,
            icon=new_icon_path("error", "svg"),
        )
        popup.show_popup(self, position="center")
