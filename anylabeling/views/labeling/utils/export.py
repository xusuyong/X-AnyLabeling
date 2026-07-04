import json
import os
import os.path as osp
import pathlib
import shutil
import time

import cv2
import numpy as np
import yaml

from PyQt6 import QtGui, QtWidgets
from PyQt6.QtCore import QCoreApplication, Qt, QSize, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QVBoxLayout,
    QProgressDialog,
)

from anylabeling.views.labeling.label_converter import (
    LabelConverter,
    PoseClassError,
    PoseGroupError,
)
from anylabeling.views.labeling.logger import logger
from anylabeling.views.labeling.widgets import Popup
from anylabeling.views.labeling.utils.colormap import label_colormap
from anylabeling.views.labeling.utils.qt import new_icon_path
from anylabeling.views.labeling.utils.style import *


class ExportThread(QThread):
    finished = pyqtSignal(bool, str)

    def __init__(
        self,
        converter,
        image_list,
        label_dir_path,
        save_path,
        mode,
        prefix=None,
    ):
        super().__init__()
        self.converter = converter
        self.image_list = image_list
        self.label_dir_path = label_dir_path
        self.save_path = save_path
        self.mode = mode
        self.prefix = prefix

    def run(self):
        try:
            time.sleep(1)

            if self.mode == "vlm_r1_ovd":
                self.converter.custom_to_vlm_r1_ovd(
                    self.image_list,
                    self.label_dir_path,
                    self.save_path,
                    self.prefix,
                )
            elif self.mode == "mot":
                self.converter.custom_to_mot(
                    self.label_dir_path, self.save_path
                )
            elif self.mode == "mots":
                self.converter.custom_to_mots(
                    self.label_dir_path, self.save_path
                )
            elif self.mode == "odvg":
                self.converter.custom_to_odvg(
                    self.image_list, self.label_dir_path, self.save_path
                )
            else:
                self.converter.custom_to_coco(
                    self.image_list,
                    self.label_dir_path,
                    self.save_path,
                    self.mode,
                )
            self.finished.emit(True, "")
        except Exception as e:
            self.finished.emit(False, str(e))


def _check_filename_exist(self):
    if not self.may_continue():
        return False

    if not self.filename:
        popup = Popup(
            self.tr("Please load an image folder before proceeding!"),
            self,
            icon=new_icon_path("warning", "svg"),
        )
        popup.show_popup(self, position="center")
        return False

    return True


def _show_yolo_export_error(parent, image_file, error):
    image_path = osp.abspath(image_file) if image_file else None
    message = (
        QCoreApplication.translate("LabelingWidget", "Failed on image: %s")
        % image_path
        if image_path
        else QCoreApplication.translate("LabelingWidget", "Export failed.")
    )
    if isinstance(error, PoseGroupError):
        message += "\n\n" + QCoreApplication.translate(
            "LabelingWidget",
            "Reason: Pose instance grouping is incomplete or mismatched.\n"
            "Please ensure that each instance has one bounding box and that "
            "its bounding box and keypoints use the same numeric group ID.",
        )
    elif isinstance(error, PoseClassError):
        message += "\n\n" + QCoreApplication.translate(
            "LabelingWidget",
            "Reason: The bounding box label is not defined in the pose "
            "configuration.\nPlease ensure that the bounding box label is "
            "listed under classes in the pose YAML file.",
        )
    elif str(error):
        message += "\n\n" + QCoreApplication.translate(
            "LabelingWidget", "Reason: %s"
        ) % str(error)

    msg_box = QtWidgets.QMessageBox(parent)
    msg_box.setIcon(QtWidgets.QMessageBox.Icon.Critical)
    msg_box.setWindowTitle(
        QCoreApplication.translate("LabelingWidget", "Export Failed")
    )
    msg_box.setText(message)
    msg_box.addButton(QtWidgets.QMessageBox.StandardButton.Ok)
    msg_box.setStyleSheet(get_msg_box_style())
    msg_box.exec()

    loaded_image_path = (
        osp.abspath(parent.filename) if parent.filename else None
    )
    if image_path and image_path != loaded_image_path:
        parent.load_file(image_file)


def _get_yolo_source_root(filename, last_open_dir):
    source_root = osp.dirname(osp.abspath(filename))
    if not last_open_dir:
        return source_root

    last_open_dir = osp.abspath(last_open_dir)
    try:
        if osp.commonpath((last_open_dir, source_root)) == last_open_dir:
            return last_open_dir
    except ValueError:
        pass
    return source_root


def _validate_yolo_export_path(source_root, save_path):
    if not save_path:
        raise ValueError("Please select an export root directory.")

    source_root = osp.realpath(source_root)
    save_path = osp.realpath(save_path)
    if osp.exists(save_path) and not osp.isdir(save_path):
        raise ValueError("The export root path must be a directory.")

    try:
        common_path = osp.commonpath((source_root, save_path))
    except ValueError:
        return
    if common_path == source_root:
        raise ValueError(
            "The export root directory cannot be the loaded image directory "
            "or one of its subdirectories."
        )
    if common_path == save_path:
        raise ValueError(
            "The export root directory cannot contain the loaded image "
            "directory."
        )


def _get_yolo_export_files(image_list, source_root, save_path):
    export_files = []
    label_destinations = {}
    for image_file in image_list:
        try:
            relative_image_path = osp.relpath(image_file, source_root)
        except ValueError:
            relative_image_path = osp.basename(image_file)
        if (
            relative_image_path == osp.pardir
            or relative_image_path.startswith(osp.pardir + osp.sep)
        ):
            relative_image_path = osp.basename(image_file)
        relative_label_path = osp.splitext(relative_image_path)[0] + ".txt"
        destination_key = osp.normcase(osp.normpath(relative_label_path))
        if destination_key in label_destinations:
            raise ValueError(
                "Multiple images map to the same YOLO label file "
                f"'{relative_label_path}':\n"
                f"{label_destinations[destination_key]}\n{image_file}"
            )
        label_destinations[destination_key] = image_file
        export_files.append(
            (
                image_file,
                osp.join(save_path, relative_label_path),
                osp.join(save_path, relative_image_path),
            )
        )
    return export_files


def export_yolo_annotation(self, mode):
    if not _check_filename_exist(self):
        return

    # Handle config/classes file selection based on mode
    if mode == "pose":
        filter = QCoreApplication.translate(
            "LabelingWidget", "Classes Files (*.yaml);;All Files (*)"
        )
        self.yaml_file, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            QCoreApplication.translate(
                "LabelingWidget",
                "Select a specific yolo-pose config file",
            ),
            "",
            filter,
        )
        if not self.yaml_file:
            return
        try:
            converter = LabelConverter(pose_cfg_file=self.yaml_file)
        except Exception as e:
            logger.error(f"Failed to load pose config: {self.yaml_file}: {e}")
            popup = Popup(
                QCoreApplication.translate(
                    "LabelingWidget", "Invalid pose config file:\n%s"
                )
                % str(e),
                self,
                icon=new_icon_path("error", "svg"),
            )
            popup.show_popup(self, popup_height=65, position="center")
            return

    elif mode in ["hbb", "obb", "seg"]:
        filter = QCoreApplication.translate(
            "LabelingWidget", "Classes Files (*.txt);;All Files (*)"
        )
        self.classes_file, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            QCoreApplication.translate(
                "LabelingWidget", "Select a specific classes file"
            ),
            "",
            filter,
        )
        if not self.classes_file:
            return
        converter = LabelConverter(classes_file=self.classes_file)

    source_root = _get_yolo_source_root(
        self.filename, getattr(self, "last_open_dir", None)
    )

    dialog = QtWidgets.QDialog(self)
    dialog.setWindowTitle(
        QCoreApplication.translate("LabelingWidget", "Export options")
    )
    dialog.setMinimumWidth(500)
    dialog.setStyleSheet(get_export_option_style())

    layout = QVBoxLayout()
    layout.setContentsMargins(24, 24, 24, 24)
    layout.setSpacing(16)

    path_layout = QVBoxLayout()
    path_label = QtWidgets.QLabel(
        QCoreApplication.translate("LabelingWidget", "Export path")
    )
    path_layout.addWidget(path_label)

    path_input_layout = QHBoxLayout()
    path_input_layout.setSpacing(8)

    path_edit = QtWidgets.QLineEdit()
    path_edit.setText(osp.realpath(osp.join(source_root, "..", "labels")))
    path_edit.setPlaceholderText(
        QCoreApplication.translate("LabelingWidget", "Select Export Directory")
    )

    def browse_export_path():
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            QCoreApplication.translate(
                "LabelingWidget", "Select Export Directory"
            ),
            path_edit.text(),
            QtWidgets.QFileDialog.Option.DontUseNativeDialog,
        )
        if path:
            path_edit.setText(path)

    path_button = QtWidgets.QPushButton(
        QCoreApplication.translate("LabelingWidget", "Browse")
    )
    path_button.clicked.connect(browse_export_path)
    path_button.setStyleSheet(get_cancel_btn_style())

    path_input_layout.addWidget(path_edit)
    path_input_layout.addWidget(path_button)
    path_layout.addLayout(path_input_layout)
    layout.addLayout(path_layout)

    options_label = QtWidgets.QLabel(
        QCoreApplication.translate("LabelingWidget", "Export Options")
    )
    layout.addWidget(options_label)

    save_images_checkbox = QtWidgets.QCheckBox(
        QCoreApplication.translate("LabelingWidget", "Save with images?")
    )
    save_images_checkbox.setChecked(False)
    layout.addWidget(save_images_checkbox)

    skip_empty_files_checkbox = QtWidgets.QCheckBox(
        QCoreApplication.translate("LabelingWidget", "Skip empty labels?")
    )
    skip_empty_files_checkbox.setChecked(False)
    layout.addWidget(skip_empty_files_checkbox)

    button_layout = QHBoxLayout()
    button_layout.setContentsMargins(0, 16, 0, 0)
    button_layout.setSpacing(8)

    cancel_button = QtWidgets.QPushButton(
        QCoreApplication.translate("LabelingWidget", "Cancel")
    )
    cancel_button.clicked.connect(dialog.reject)
    cancel_button.setStyleSheet(get_cancel_btn_style())

    ok_button = QtWidgets.QPushButton(
        QCoreApplication.translate("LabelingWidget", "OK")
    )
    ok_button.clicked.connect(dialog.accept)
    ok_button.setStyleSheet(get_ok_btn_style())

    button_layout.addStretch()
    button_layout.addWidget(cancel_button)
    button_layout.addWidget(ok_button)
    layout.addLayout(button_layout)

    dialog.setLayout(layout)
    result = dialog.exec()

    if not result:
        return

    save_images = save_images_checkbox.isChecked()
    skip_empty_files = skip_empty_files_checkbox.isChecked()
    save_path = path_edit.text()
    image_list = self.image_list if self.image_list else [self.filename]

    try:
        _validate_yolo_export_path(source_root, save_path)
        export_files = _get_yolo_export_files(
            image_list, source_root, save_path
        )
    except ValueError as error:
        _show_yolo_export_error(self, None, error)
        return

    def get_label_file(image_file):
        label_file_name = osp.splitext(osp.basename(image_file))[0] + ".json"
        label_dir = self.output_dir or osp.dirname(image_file)
        return osp.join(label_dir, label_file_name)

    obb_boundary_policy = "keep"
    if mode == "obb":
        out_of_bounds_count = 0
        for image_file in image_list:
            label_file = get_label_file(image_file)
            if not osp.exists(label_file):
                continue
            data = converter.read_json(label_file)
            image_width = data["imageWidth"]
            image_height = data["imageHeight"]
            for shape in data["shapes"]:
                points = shape["points"]
                if shape["shape_type"] != "rotation" or len(points) != 4:
                    continue
                if any(
                    point[0] < 0
                    or point[0] > image_width
                    or point[1] < 0
                    or point[1] > image_height
                    for point in points
                ):
                    out_of_bounds_count += 1

        if out_of_bounds_count:
            msg_box = QtWidgets.QMessageBox(self)
            msg_box.setIcon(QtWidgets.QMessageBox.Icon.Warning)
            msg_box.setWindowTitle(
                QCoreApplication.translate(
                    "LabelingWidget", "Out-of-bounds OBBs"
                )
            )
            msg_box.setText(
                QCoreApplication.translate(
                    "LabelingWidget",
                    "Detected %d oriented bounding boxes with points outside "
                    "the image boundaries. Keep them?",
                )
                % out_of_bounds_count
            )
            msg_box.setStandardButtons(
                QtWidgets.QMessageBox.StandardButton.Yes
                | QtWidgets.QMessageBox.StandardButton.No
                | QtWidgets.QMessageBox.StandardButton.Cancel
            )
            msg_box.setDefaultButton(QtWidgets.QMessageBox.StandardButton.Yes)
            msg_box.setStyleSheet(get_msg_box_style())
            response = QtWidgets.QMessageBox.StandardButton(msg_box.exec())
            if response == QtWidgets.QMessageBox.StandardButton.Cancel:
                return
            if response == QtWidgets.QMessageBox.StandardButton.No:
                obb_boundary_policy = "skip"

    if osp.exists(save_path):
        msg_box = QtWidgets.QMessageBox(self)
        msg_box.setIcon(QtWidgets.QMessageBox.Icon.Warning)
        msg_box.setWindowTitle(
            QCoreApplication.translate(
                "LabelingWidget", "Output Directory Exists!"
            )
        )
        msg_box.setText(
            QCoreApplication.translate(
                "LabelingWidget",
                "Directory already exists. Choose an action:",
            )
        )
        msg_box.setInformativeText(
            QCoreApplication.translate(
                "LabelingWidget",
                "• Yes    - Merge with existing files\n"
                "• No     - Delete existing directory\n"
                "• Cancel - Abort export",
            )
        )

        msg_box.addButton(
            QCoreApplication.translate("LabelingWidget", "Yes"),
            QtWidgets.QMessageBox.ButtonRole.YesRole,
        )
        no_button = msg_box.addButton(
            QCoreApplication.translate("LabelingWidget", "No"),
            QtWidgets.QMessageBox.ButtonRole.NoRole,
        )
        cancel_button = msg_box.addButton(
            QCoreApplication.translate("LabelingWidget", "Cancel"),
            QtWidgets.QMessageBox.ButtonRole.RejectRole,
        )
        msg_box.setStyleSheet(get_msg_box_style())
        msg_box.exec()

        clicked_button = msg_box.clickedButton()
        if clicked_button == no_button:
            shutil.rmtree(save_path)
            os.makedirs(save_path)
        elif clicked_button == cancel_button:
            return
    else:
        os.makedirs(save_path)

    progress_dialog = QProgressDialog(
        QCoreApplication.translate("LabelingWidget", "Exporting..."),
        QCoreApplication.translate("LabelingWidget", "Cancel"),
        0,
        len(image_list),
        self,
    )
    progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
    progress_dialog.setWindowTitle(
        QCoreApplication.translate("LabelingWidget", "Progress")
    )
    progress_dialog.setMinimumWidth(500)
    progress_dialog.setMinimumHeight(150)
    progress_dialog.setStyleSheet(
        get_progress_dialog_style(color="#1d1d1f", height=20)
    )

    current_image_file = None
    try:
        for i, (image_file, dst_file, image_dst) in enumerate(export_files):
            current_image_file = image_file
            src_file = get_label_file(image_file)
            os.makedirs(osp.dirname(dst_file), exist_ok=True)

            is_empty_file = converter.custom_to_yolo(
                src_file,
                dst_file,
                mode,
                skip_empty_files=skip_empty_files,
                obb_boundary_policy=obb_boundary_policy,
            )

            if save_images and not (skip_empty_files and is_empty_file):
                os.makedirs(osp.dirname(image_dst), exist_ok=True)
                shutil.copy(image_file, image_dst)

            if skip_empty_files and is_empty_file and osp.exists(dst_file):
                os.remove(dst_file)

            progress_dialog.setValue(i)
            if progress_dialog.wasCanceled():
                break

        current_image_file = None
        progress_dialog.close()
        template = QCoreApplication.translate(
            "LabelingWidget",
            "Exporting annotations successfully!\n"
            "Results have been saved to:\n"
            "%s",
        )
        message_text = template % save_path
        popup = Popup(
            message_text,
            self,
            icon=new_icon_path("copy-green", "svg"),
        )
        popup.show_popup(self, popup_height=65, position="center")

    except Exception as e:
        progress_dialog.close()
        failed_image_path = (
            osp.abspath(current_image_file) if current_image_file else None
        )
        if failed_image_path:
            logger.error(
                "Error occurred while exporting annotations for image:\n"
                f"{failed_image_path}\n{e}"
            )
        else:
            logger.error(f"Error occurred while exporting annotations: {e}")

        _show_yolo_export_error(self, current_image_file, e)


def export_voc_annotation(self, mode):
    if not _check_filename_exist(self):
        return

    dialog = QtWidgets.QDialog(self)
    dialog.setWindowTitle(self.tr("Export options"))
    dialog.setMinimumWidth(500)
    dialog.setStyleSheet(get_export_option_style())

    layout = QVBoxLayout()
    layout.setContentsMargins(24, 24, 24, 24)
    layout.setSpacing(16)

    path_layout = QVBoxLayout()
    path_label = QtWidgets.QLabel(self.tr("Export path"))
    path_layout.addWidget(path_label)

    path_input_layout = QHBoxLayout()
    path_input_layout.setSpacing(8)

    path_edit = QtWidgets.QLineEdit()
    path_edit.setText(
        osp.realpath(osp.join(osp.dirname(self.filename), "..", "Annotations"))
    )
    path_edit.setPlaceholderText(self.tr("Select Export Directory"))

    def browse_export_path():
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            self.tr("Select Export Directory"),
            path_edit.text(),
            QtWidgets.QFileDialog.Option.DontUseNativeDialog,
        )
        if path:
            path_edit.setText(path)

    path_button = QtWidgets.QPushButton(self.tr("Browse"))
    path_button.clicked.connect(browse_export_path)
    path_button.setStyleSheet(get_cancel_btn_style())

    path_input_layout.addWidget(path_edit)
    path_input_layout.addWidget(path_button)
    path_layout.addLayout(path_input_layout)
    layout.addLayout(path_layout)

    options_label = QtWidgets.QLabel(self.tr("Export Options"))
    layout.addWidget(options_label)

    save_images_checkbox = QtWidgets.QCheckBox(self.tr("Save with images?"))
    save_images_checkbox.setChecked(False)
    layout.addWidget(save_images_checkbox)

    skip_empty_files_checkbox = QtWidgets.QCheckBox(
        self.tr("Skip empty labels?")
    )
    skip_empty_files_checkbox.setChecked(False)
    layout.addWidget(skip_empty_files_checkbox)

    button_layout = QHBoxLayout()
    button_layout.setContentsMargins(0, 16, 0, 0)
    button_layout.setSpacing(8)

    cancel_button = QtWidgets.QPushButton(self.tr("Cancel"))
    cancel_button.clicked.connect(dialog.reject)
    cancel_button.setStyleSheet(get_cancel_btn_style())

    ok_button = QtWidgets.QPushButton(self.tr("OK"))
    ok_button.clicked.connect(dialog.accept)
    ok_button.setStyleSheet(get_ok_btn_style())

    button_layout.addStretch()
    button_layout.addWidget(cancel_button)
    button_layout.addWidget(ok_button)
    layout.addLayout(button_layout)

    dialog.setLayout(layout)
    result = dialog.exec()

    if not result:
        return

    save_images = save_images_checkbox.isChecked()
    skip_empty_files = skip_empty_files_checkbox.isChecked()
    save_path = path_edit.text()

    if osp.exists(save_path):
        msg_box = QtWidgets.QMessageBox(self)
        msg_box.setIcon(QtWidgets.QMessageBox.Icon.Warning)
        msg_box.setWindowTitle(self.tr("Output Directory Exists!"))
        msg_box.setText(self.tr("Directory already exists. Choose an action:"))
        msg_box.setInformativeText(
            self.tr(
                "• Yes    - Merge with existing files\n"
                "• No     - Delete existing directory\n"
                "• Cancel - Abort export"
            )
        )

        msg_box.addButton(
            self.tr("Yes"), QtWidgets.QMessageBox.ButtonRole.YesRole
        )
        no_button = msg_box.addButton(
            self.tr("No"), QtWidgets.QMessageBox.ButtonRole.NoRole
        )
        cancel_button = msg_box.addButton(
            self.tr("Cancel"), QtWidgets.QMessageBox.ButtonRole.RejectRole
        )
        msg_box.setStyleSheet(get_msg_box_style())
        msg_box.exec()

        clicked_button = msg_box.clickedButton()
        if clicked_button == no_button:
            shutil.rmtree(save_path)
            os.makedirs(save_path)
        elif clicked_button == cancel_button:
            return
    else:
        os.makedirs(save_path)

    converter = LabelConverter()

    image_list = self.image_list if self.image_list else [self.filename]

    progress_dialog = QProgressDialog(
        self.tr("Exporting..."), self.tr("Cancel"), 0, len(image_list), self
    )
    progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
    progress_dialog.setWindowTitle(self.tr("Progress"))
    progress_dialog.setMinimumWidth(500)
    progress_dialog.setMinimumHeight(150)
    progress_dialog.setStyleSheet(
        get_progress_dialog_style(color="#1d1d1f", height=20)
    )

    try:
        for i, image_file in enumerate(image_list):
            image_file_name = osp.basename(image_file)
            label_file_name = osp.splitext(image_file_name)[0] + ".json"
            dst_file_name = osp.splitext(image_file_name)[0] + ".xml"

            if self.output_dir:
                src_file = osp.join(self.output_dir, label_file_name)
            else:
                src_file = osp.join(osp.dirname(image_file), label_file_name)
            dst_file = osp.join(save_path, dst_file_name)

            is_empty_file = converter.custom_to_voc(
                image_file, src_file, dst_file, mode, skip_empty_files
            )

            if save_images and not (skip_empty_files and is_empty_file):
                image_dst = osp.join(save_path, image_file_name)
                shutil.copy(image_file, image_dst)

            if skip_empty_files and is_empty_file and osp.exists(dst_file):
                os.remove(dst_file)

            progress_dialog.setValue(i)
            if progress_dialog.wasCanceled():
                break

        progress_dialog.close()
        template = self.tr(
            "Exporting annotations successfully!\n"
            "Results have been saved to:\n"
            "%s"
        )
        message_text = template % save_path
        popup = Popup(
            message_text,
            self,
            icon=new_icon_path("copy-green", "svg"),
        )
        popup.show_popup(self, popup_height=65, position="center")

    except Exception as e:
        message = f"Error occurred while exporting annotations: {str(e)}"
        progress_dialog.close()
        logger.error(message)
        popup = Popup(
            message,
            self,
            icon=new_icon_path("error", "svg"),
        )
        popup.show_popup(self, position="center")


def export_coco_annotation(self, mode):
    if not _check_filename_exist(self):
        return

    if mode == "pose":
        filter = "Classes Files (*.yaml);;All Files (*)"
        self.yaml_file, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            self.tr("Select a specific coco-pose config file"),
            "",
            filter,
        )
        if not self.yaml_file:
            return
        try:
            converter = LabelConverter(pose_cfg_file=self.yaml_file)
        except Exception as e:
            logger.error(f"Failed to load pose config: {self.yaml_file}: {e}")
            popup = Popup(
                self.tr("Invalid pose config file:\n%s") % str(e),
                self,
                icon=new_icon_path("error", "svg"),
            )
            popup.show_popup(self, popup_height=65, position="center")
            return
    elif mode in ["rectangle", "polygon"]:
        filter = "Classes Files (*.txt);;All Files (*)"
        self.classes_file, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            self.tr("Select a specific classes file"),
            "",
            filter,
        )
        if not self.classes_file:
            return
        converter = LabelConverter(classes_file=self.classes_file)

    dialog = QtWidgets.QDialog(self)
    dialog.setWindowTitle(self.tr("Export options"))
    dialog.setMinimumWidth(500)
    dialog.setStyleSheet(get_export_option_style())

    layout = QVBoxLayout()
    layout.setContentsMargins(24, 24, 24, 24)
    layout.setSpacing(16)

    path_layout = QVBoxLayout()
    path_label = QtWidgets.QLabel(self.tr("Export path"))
    path_layout.addWidget(path_label)

    path_input_layout = QHBoxLayout()
    path_input_layout.setSpacing(8)

    label_dir_path = osp.dirname(self.filename)
    if self.output_dir:
        label_dir_path = self.output_dir

    path_edit = QtWidgets.QLineEdit()
    path_edit.setText(
        osp.realpath(osp.join(label_dir_path, "..", "annotations"))
    )
    path_edit.setPlaceholderText(self.tr("Select Export Directory"))

    def browse_export_path():
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            self.tr("Select Export Directory"),
            path_edit.text(),
            QtWidgets.QFileDialog.Option.DontUseNativeDialog,
        )
        if path:
            path_edit.setText(path)

    path_button = QtWidgets.QPushButton(self.tr("Browse"))
    path_button.clicked.connect(browse_export_path)
    path_button.setStyleSheet(get_cancel_btn_style())

    path_input_layout.addWidget(path_edit)
    path_input_layout.addWidget(path_button)
    path_layout.addLayout(path_input_layout)
    layout.addLayout(path_layout)

    button_layout = QHBoxLayout()
    button_layout.setContentsMargins(0, 16, 0, 0)
    button_layout.setSpacing(8)

    cancel_button = QtWidgets.QPushButton(self.tr("Cancel"))
    cancel_button.clicked.connect(dialog.reject)
    cancel_button.setStyleSheet(get_cancel_btn_style())

    ok_button = QtWidgets.QPushButton(self.tr("OK"))
    ok_button.clicked.connect(dialog.accept)
    ok_button.setStyleSheet(get_ok_btn_style())

    button_layout.addStretch()
    button_layout.addWidget(cancel_button)
    button_layout.addWidget(ok_button)
    layout.addLayout(button_layout)

    dialog.setLayout(layout)
    result = dialog.exec()

    if not result:
        return

    save_path = path_edit.text()
    if osp.exists(save_path):
        msg_box = QtWidgets.QMessageBox(self)
        msg_box.setIcon(QtWidgets.QMessageBox.Icon.Warning)
        msg_box.setWindowTitle(self.tr("Output Directory Exists!"))
        msg_box.setText(self.tr("Directory already exists. Choose an action:"))
        msg_box.setInformativeText(
            self.tr(
                "• Overwrite - Overwrite existing directory\n"
                "• Cancel - Abort export"
            )
        )

        msg_box.addButton(
            self.tr("Overwrite"), QtWidgets.QMessageBox.ButtonRole.YesRole
        )
        cancel_button = msg_box.addButton(
            self.tr("Cancel"), QtWidgets.QMessageBox.ButtonRole.RejectRole
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

    image_list = self.image_list if self.image_list else [self.filename]
    progress_dialog = QProgressDialog(
        self.tr("Exporting..."), self.tr("Cancel"), 0, 0, self
    )
    progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
    progress_dialog.setWindowTitle(self.tr("Progress"))
    progress_dialog.setMinimumWidth(500)
    progress_dialog.setMinimumHeight(150)
    progress_dialog.setRange(0, 0)
    progress_dialog.setStyleSheet(get_progress_dialog_style())

    self.export_thread = ExportThread(
        converter, image_list, label_dir_path, save_path, mode
    )

    def on_export_finished(success, error_msg):
        progress_dialog.close()
        if success:
            template = self.tr(
                "Exporting annotations successfully!\n"
                "Results have been saved to:\n"
                "%s"
            )
            message_text = template % save_path
            popup = Popup(
                message_text,
                self,
                icon=new_icon_path("copy-green", "svg"),
            )
            popup.show_popup(self, popup_height=65, position="center")
        else:
            message = (
                f"Error occurred while exporting annotations: {str(error_msg)}"
            )
            logger.error(message)
            popup = Popup(
                message,
                self,
                icon=new_icon_path("error", "svg"),
            )
            popup.show_popup(self, position="center")

    self.export_thread.finished.connect(on_export_finished)

    progress_dialog.show()
    self.export_thread.start()

    progress_dialog.canceled.connect(self.export_thread.terminate)


def export_dota_annotation(self):
    if not _check_filename_exist(self):
        return

    filter = "Classes Files (*.txt);;All Files (*)"
    self.classes_file, _ = QtWidgets.QFileDialog.getOpenFileName(
        self,
        self.tr("Select a specific classes file"),
        "",
        filter,
    )
    if not self.classes_file:
        return

    dialog = QtWidgets.QDialog(self)
    dialog.setWindowTitle(self.tr("Export options"))
    dialog.setMinimumWidth(500)
    dialog.setStyleSheet(get_export_option_style())

    layout = QVBoxLayout()
    layout.setContentsMargins(24, 24, 24, 24)
    layout.setSpacing(16)

    path_layout = QVBoxLayout()
    path_label = QtWidgets.QLabel(self.tr("Export path"))
    path_layout.addWidget(path_label)

    path_input_layout = QHBoxLayout()
    path_input_layout.setSpacing(8)

    path_edit = QtWidgets.QLineEdit()
    path_edit.setText(
        osp.realpath(osp.join(osp.dirname(self.filename), "..", "labelTxt"))
    )
    path_edit.setPlaceholderText(self.tr("Select Export Directory"))

    def browse_export_path():
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            self.tr("Select Export Directory"),
            path_edit.text(),
            QtWidgets.QFileDialog.Option.DontUseNativeDialog,
        )
        if path:
            path_edit.setText(path)

    path_button = QtWidgets.QPushButton(self.tr("Browse"))
    path_button.clicked.connect(browse_export_path)
    path_button.setStyleSheet(get_cancel_btn_style())

    path_input_layout.addWidget(path_edit)
    path_input_layout.addWidget(path_button)
    path_layout.addLayout(path_input_layout)
    layout.addLayout(path_layout)

    button_layout = QHBoxLayout()
    button_layout.setContentsMargins(0, 16, 0, 0)
    button_layout.setSpacing(8)

    cancel_button = QtWidgets.QPushButton(self.tr("Cancel"))
    cancel_button.clicked.connect(dialog.reject)
    cancel_button.setStyleSheet(get_cancel_btn_style())

    ok_button = QtWidgets.QPushButton(self.tr("OK"))
    ok_button.clicked.connect(dialog.accept)
    ok_button.setStyleSheet(get_ok_btn_style())

    button_layout.addStretch()
    button_layout.addWidget(cancel_button)
    button_layout.addWidget(ok_button)
    layout.addLayout(button_layout)

    dialog.setLayout(layout)
    result = dialog.exec()

    if not result:
        return

    save_path = path_edit.text()

    if osp.exists(save_path):
        msg_box = QtWidgets.QMessageBox(self)
        msg_box.setIcon(QtWidgets.QMessageBox.Icon.Warning)
        msg_box.setWindowTitle(self.tr("Output Directory Exists!"))
        msg_box.setText(self.tr("Directory already exists. Choose an action:"))
        msg_box.setInformativeText(
            self.tr(
                "• Yes    - Merge with existing files\n"
                "• No     - Delete existing directory\n"
                "• Cancel - Abort export"
            )
        )

        msg_box.addButton(
            self.tr("Yes"), QtWidgets.QMessageBox.ButtonRole.YesRole
        )
        no_button = msg_box.addButton(
            self.tr("No"), QtWidgets.QMessageBox.ButtonRole.NoRole
        )
        cancel_button = msg_box.addButton(
            self.tr("Cancel"), QtWidgets.QMessageBox.ButtonRole.RejectRole
        )
        msg_box.setStyleSheet(get_msg_box_style())
        msg_box.exec()

        clicked_button = msg_box.clickedButton()
        if clicked_button == no_button:
            shutil.rmtree(save_path)
            os.makedirs(save_path)
        elif clicked_button == cancel_button:
            return
    else:
        os.makedirs(save_path)

    converter = LabelConverter(classes_file=self.classes_file)

    image_list = self.image_list if self.image_list else [self.filename]

    progress_dialog = QProgressDialog(
        self.tr("Exporting..."), self.tr("Cancel"), 0, len(image_list), self
    )
    progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
    progress_dialog.setWindowTitle(self.tr("Progress"))
    progress_dialog.setMinimumWidth(500)
    progress_dialog.setMinimumHeight(150)
    progress_dialog.setStyleSheet(
        get_progress_dialog_style(color="#1d1d1f", height=20)
    )

    try:
        for i, image_file in enumerate(image_list):
            image_file_name = osp.basename(image_file)
            label_file_name = osp.splitext(image_file_name)[0] + ".json"
            dst_file_name = osp.splitext(image_file_name)[0] + ".txt"

            if self.output_dir:
                src_file = osp.join(self.output_dir, label_file_name)
            else:
                src_file = osp.join(osp.dirname(image_file), label_file_name)
            dst_file = osp.join(save_path, dst_file_name)

            if not osp.exists(src_file):
                pathlib.Path(dst_file).touch()
            else:
                converter.custom_to_dota(src_file, dst_file)

            progress_dialog.setValue(i)
            if progress_dialog.wasCanceled():
                break

        progress_dialog.close()
        template = self.tr(
            "Exporting annotations successfully!\n"
            "Results have been saved to:\n"
            "%s"
        )
        message_text = template % save_path
        popup = Popup(
            message_text,
            self,
            icon=new_icon_path("copy-green", "svg"),
        )
        popup.show_popup(self, popup_height=65, position="center")

    except Exception as e:
        message = f"Error occurred while exporting annotations: {str(e)}"
        progress_dialog.close()
        logger.error(message)
        popup = Popup(
            message,
            self,
            icon=new_icon_path("error", "svg"),
        )
        popup.show_popup(self, position="center")


def _export_mask_files(
    converter,
    image_list,
    output_dir,
    save_path,
    mapping_table,
    include_null_images,
    only_checked_images,
    progress_dialog,
):
    for i, image_file in enumerate(image_list):
        image_file_name = osp.basename(image_file)
        label_file_name = osp.splitext(image_file_name)[0] + ".json"
        dst_file_name = osp.splitext(image_file_name)[0] + ".png"

        if output_dir:
            src_file = osp.join(output_dir, label_file_name)
        else:
            src_file = osp.join(osp.dirname(image_file), label_file_name)
        dst_file = osp.join(save_path, dst_file_name)

        if osp.exists(src_file):
            if (
                not only_checked_images
                or converter.read_json(src_file).get("checked", False) is True
            ):
                converter.custom_to_mask(src_file, dst_file, mapping_table)
        elif include_null_images and not only_checked_images:
            converter.custom_image_to_empty_mask(
                image_file, dst_file, mapping_table
            )

        progress_dialog.setValue(i + 1)
        if progress_dialog.wasCanceled():
            break


def _collect_polygon_labels(self):
    """Collect all unique polygon labels from label files in the image list."""
    labels = set()
    image_list = self.image_list if self.image_list else [self.filename]
    output_dir = self.output_dir
    for image_file in image_list:
        label_file_name = osp.splitext(osp.basename(image_file))[0] + ".json"
        if output_dir:
            label_file = osp.join(output_dir, label_file_name)
        else:
            label_file = osp.join(osp.dirname(image_file), label_file_name)
        if not osp.exists(label_file):
            continue
        try:
            with open(label_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            for shape in data.get("shapes", []):
                if shape.get("shape_type") == "polygon":
                    label = shape.get("label", "")
                    if label:
                        labels.add(str(label))
        except Exception:
            continue
    return sorted(labels)


def export_mask_annotation(self):
    if not _check_filename_exist(self):
        return

    # Collect polygon labels upfront (used for both preview and mapping)
    all_labels = _collect_polygon_labels(self)
    cmap = label_colormap(max(len(all_labels) + 1, 256))

    dialog = QtWidgets.QDialog(self)
    dialog.setWindowTitle(self.tr("Export Mask Annotations"))
    dialog.setMinimumWidth(560)
    dialog.setStyleSheet(get_export_option_style())

    layout = QVBoxLayout()
    layout.setContentsMargins(24, 24, 24, 24)
    layout.setSpacing(16)

    # --- Mode selection ---
    mode_group = QtWidgets.QGroupBox(self.tr("Mask Mode"))
    mode_layout = QVBoxLayout()
    mode_layout.setSpacing(8)
    bw_radio = QtWidgets.QRadioButton(self.tr("Black & White (binary mask)"))
    bw_radio.setToolTip(
        self.tr("All targets are white (255), background is black (0).")
    )
    multi_radio = QtWidgets.QRadioButton(
        self.tr("Multi-class semantic segmentation")
    )
    multi_radio.setToolTip(
        self.tr("Each label is auto-assigned a unique color.")
    )
    ultralytics_radio = QtWidgets.QRadioButton(
        self.tr("Ultralytics semantic segmentation (class-index mask)")
    )
    ultralytics_radio.setToolTip(
        self.tr(
            "Single-channel PNG where pixel value equals the class "
            "index. Background is index 0. Matches the masks_dir "
            "format used by Ultralytics semantic segmentation."
        )
    )
    bw_radio.setChecked(True)
    mode_layout.addWidget(bw_radio)
    mode_layout.addWidget(multi_radio)
    mode_layout.addWidget(ultralytics_radio)
    mode_group.setLayout(mode_layout)
    layout.addWidget(mode_group)

    # --- Label color preview ---
    preview_title = QtWidgets.QLabel(self.tr("Label color preview"))
    layout.addWidget(preview_title)

    preview_list = QtWidgets.QListWidget()
    preview_list.setIconSize(QSize(16, 16))
    preview_list.setMaximumHeight(140)
    layout.addWidget(preview_list)

    def build_preview():
        preview_list.clear()
        for i, label in enumerate(all_labels):
            if multi_radio.isChecked():
                color = cmap[i + 1]  # index 0 is background (black)
                rgb = (int(color[0]), int(color[1]), int(color[2]))
                item_text = label
            elif ultralytics_radio.isChecked():
                # Show the flat gray value corresponding to the class
                # index so users get a visual sense of separation; the
                # actual saved pixel value is the raw index (i + 1).
                gray = min(255, (i + 1) * 40)
                rgb = (gray, gray, gray)
                item_text = f"{label}  (index {i + 1})"
            else:
                rgb = (255, 255, 255)
                item_text = label
            pix = QtGui.QPixmap(16, 16)
            pix.fill(QtGui.QColor(*rgb))
            item = QtWidgets.QListWidgetItem(item_text)
            item.setIcon(QtGui.QIcon(pix))
            preview_list.addItem(item)

    build_preview()

    def on_mode_toggled():
        build_preview()

    bw_radio.toggled.connect(on_mode_toggled)
    multi_radio.toggled.connect(on_mode_toggled)
    ultralytics_radio.toggled.connect(on_mode_toggled)

    # --- Export path ---
    path_layout = QVBoxLayout()
    path_label = QtWidgets.QLabel(self.tr("Export path"))
    path_layout.addWidget(path_label)

    path_input_layout = QHBoxLayout()
    path_input_layout.setSpacing(8)

    label_dir_path = osp.dirname(self.filename)
    if self.output_dir:
        label_dir_path = self.output_dir

    path_edit = QtWidgets.QLineEdit()
    path_edit.setText(osp.realpath(osp.join(label_dir_path, "..", "masks")))
    path_edit.setPlaceholderText(self.tr("Select Export Directory"))

    def browse_export_path():
        path = QtWidgets.QFileDialog.getExistingDirectory(
            dialog,
            self.tr("Select Export Directory"),
            path_edit.text(),
            QtWidgets.QFileDialog.Option.DontUseNativeDialog,
        )
        if path:
            path_edit.setText(path)

    path_button = QtWidgets.QPushButton(self.tr("Browse"))
    path_button.clicked.connect(browse_export_path)
    path_button.setStyleSheet(get_cancel_btn_style())

    path_input_layout.addWidget(path_edit)
    path_input_layout.addWidget(path_button)
    path_layout.addLayout(path_input_layout)
    layout.addLayout(path_layout)

    options_label = QtWidgets.QLabel(self.tr("Export Options"))
    layout.addWidget(options_label)

    include_null_images_checkbox = QtWidgets.QCheckBox(
        self.tr("Include images without labels?")
    )
    include_null_images_checkbox.setChecked(False)
    layout.addWidget(include_null_images_checkbox)

    only_checked_images_checkbox = QtWidgets.QCheckBox(
        self.tr("Only export checked images?")
    )
    only_checked_images_checkbox.setChecked(False)
    layout.addWidget(only_checked_images_checkbox)

    button_layout = QHBoxLayout()
    button_layout.setContentsMargins(0, 16, 0, 0)
    button_layout.setSpacing(8)

    cancel_button = QtWidgets.QPushButton(self.tr("Cancel"))
    cancel_button.clicked.connect(dialog.reject)
    cancel_button.setStyleSheet(get_cancel_btn_style())

    ok_button = QtWidgets.QPushButton(self.tr("OK"))
    ok_button.clicked.connect(dialog.accept)
    ok_button.setStyleSheet(get_ok_btn_style())

    button_layout.addStretch()
    button_layout.addWidget(cancel_button)
    button_layout.addWidget(ok_button)
    layout.addLayout(button_layout)

    dialog.setLayout(layout)
    result = dialog.exec()

    if not result:
        return

    # --- Build mapping_table according to selected mode (no file needed) ---
    if multi_radio.isChecked():
        mapping_table = {
            "type": "rgb",
            "colors": {
                label: [int(c) for c in cmap[i + 1]]
                for i, label in enumerate(all_labels)
            },
        }
    elif ultralytics_radio.isChecked():
        # Background is index 0; each label gets the next contiguous
        # index. This matches Ultralytics' masks_dir semantic
        # segmentation format (pixel value == class id).
        mapping_table = {
            "type": "index",
            "background": 0,
            "colors": {
                label: i + 1 for i, label in enumerate(all_labels)
            },
        }
    else:
        # Black & White: every polygon label maps to 255 (white) on black background
        mapping_table = {
            "type": "grayscale",
            "colors": {label: 255 for label in all_labels},
        }

    save_path = path_edit.text()
    include_null_images = include_null_images_checkbox.isChecked()
    only_checked_images = only_checked_images_checkbox.isChecked()
    if osp.exists(save_path):
        msg_box = QtWidgets.QMessageBox(self)
        msg_box.setIcon(QtWidgets.QMessageBox.Icon.Warning)
        msg_box.setWindowTitle(self.tr("Output Directory Exists!"))
        msg_box.setText(self.tr("Directory already exists. Choose an action:"))
        msg_box.setInformativeText(
            self.tr(
                "• Overwrite - Overwrite existing directory\n"
                "• Cancel - Abort export"
            )
        )

        msg_box.addButton(
            self.tr("Overwrite"), QtWidgets.QMessageBox.ButtonRole.YesRole
        )
        cancel_button = msg_box.addButton(
            self.tr("Cancel"), QtWidgets.QMessageBox.ButtonRole.RejectRole
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

    converter = LabelConverter()
    image_list = self.image_list if self.image_list else [self.filename]

    progress_dialog = QProgressDialog(
        self.tr("Exporting..."), self.tr("Cancel"), 0, len(image_list), self
    )
    progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
    progress_dialog.setWindowTitle(self.tr("Progress"))
    progress_dialog.setMinimumWidth(500)
    progress_dialog.setMinimumHeight(150)
    progress_dialog.setStyleSheet(
        get_progress_dialog_style(color="#1d1d1f", height=20)
    )
    progress_dialog.show()
    QtWidgets.QApplication.processEvents()

    canceled = False
    failed_files = []

    try:
        _export_mask_files(
            converter,
            image_list,
            self.output_dir,
            save_path,
            mapping_table,
            include_null_images,
            only_checked_images,
            progress_dialog,
        )

        if progress_dialog.wasCanceled():
            canceled = True

        progress_dialog.close()

        if canceled:
            popup = Popup(
                self.tr("Export canceled."),
                self,
                icon=new_icon_path("warning", "svg"),
            )
            popup.show_popup(self, position="center")
        else:
            if mapping_table["type"] == "index":
                # Write index -> label reference so users can copy it
                # straight into their Ultralytics dataset yaml `names`.
                names_path = osp.join(save_path, "names.txt")
                try:
                    with open(names_path, "w", encoding="utf-8") as f:
                        f.write("names:\n")
                        f.write(
                            f"  {mapping_table.get('background', 0)}: background\n"
                        )
                        for label, idx in mapping_table["colors"].items():
                            f.write(f"  {idx}: {label}\n")
                except Exception as e:
                    logger.error(f"Failed to write names.txt: {e}")

            template = self.tr(
                "Exporting annotations successfully!\n"
                "Results have been saved to:\n"
                "%s"
            )
            message_text = template % save_path
            if failed_files:
                message_text += self.tr(
                    "\n\n%d file(s) failed to convert, see log for details."
                ) % len(failed_files)
            popup = Popup(
                message_text,
                self,
                icon=new_icon_path("copy-green", "svg"),
            )
            popup.show_popup(self, popup_height=65, position="center")

    except Exception as e:
        message = f"Error occurred while exporting annotations: {str(e)}"
        progress_dialog.close()
        logger.error(message)
        popup = Popup(
            message,
            self,
            icon=new_icon_path("error", "svg"),
        )
        popup.show_popup(self, position="center")


def _get_image_shape(image_file, label_data=None):
    """Get (height, width) for an image, preferring label json metadata."""
    if label_data:
        w = label_data.get("imageWidth")
        h = label_data.get("imageHeight")
        if w and h:
            return int(h), int(w)
    img = cv2.imdecode(
        np.fromfile(str(image_file), dtype=np.uint8), cv2.IMREAD_COLOR
    )
    if img is None:
        raise ValueError(f"Failed to read image: {image_file}")
    return img.shape[:2]


def export_ultralytics_semantic_annotation(self):
    """
    Export images + PNG masks in the Ultralytics semantic segmentation
    format: https://docs.ultralytics.com/datasets/semantic/#dataset-yaml-format

    Whatever folder is currently open is treated as one split (its
    basename becomes the split name, e.g. "train"/"val"). Running this
    export again on a different split folder, pointed at the same output
    root, extends the same dataset.yaml (consistent class IDs, added
    train/val/test keys) instead of overwriting it.
    """
    if not _check_filename_exist(self):
        return

    all_labels = _collect_polygon_labels(self)

    dialog = QtWidgets.QDialog(self)
    dialog.setWindowTitle(self.tr("Export Ultralytics Semantic Segmentation"))
    dialog.setMinimumWidth(560)
    dialog.setStyleSheet(get_export_option_style())

    layout = QVBoxLayout()
    layout.setContentsMargins(24, 24, 24, 24)
    layout.setSpacing(16)

    path_label = QtWidgets.QLabel(self.tr("Dataset root directory"))
    layout.addWidget(path_label)

    path_input_layout = QHBoxLayout()
    path_input_layout.setSpacing(8)

    source_dir = osp.dirname(self.filename)
    path_edit = QtWidgets.QLineEdit()
    path_edit.setText(
        osp.realpath(osp.join(source_dir, "..", "ultralytics_semantic"))
    )
    path_edit.setPlaceholderText(self.tr("Select Dataset Root Directory"))

    def browse_export_path():
        path = QtWidgets.QFileDialog.getExistingDirectory(
            dialog,
            self.tr("Select Dataset Root Directory"),
            path_edit.text(),
            QtWidgets.QFileDialog.Option.DontUseNativeDialog,
        )
        if path:
            path_edit.setText(path)

    path_button = QtWidgets.QPushButton(self.tr("Browse"))
    path_button.clicked.connect(browse_export_path)
    path_button.setStyleSheet(get_cancel_btn_style())

    path_input_layout.addWidget(path_edit)
    path_input_layout.addWidget(path_button)
    layout.addLayout(path_input_layout)

    hint_label = QtWidgets.QLabel(
        self.tr(
            "The current folder's name is used as the split (e.g. "
            "'train' or 'val'). Run this export again on other split "
            "folders, pointed at the same root, to build the full dataset."
        )
    )
    hint_label.setStyleSheet(
        "color: gray; font-style: italic; padding-left: 5px;"
    )
    hint_label.setWordWrap(True)
    layout.addWidget(hint_label)

    button_layout = QHBoxLayout()
    button_layout.setContentsMargins(0, 16, 0, 0)
    button_layout.setSpacing(8)

    cancel_button = QtWidgets.QPushButton(self.tr("Cancel"))
    cancel_button.clicked.connect(dialog.reject)
    cancel_button.setStyleSheet(get_cancel_btn_style())

    ok_button = QtWidgets.QPushButton(self.tr("OK"))
    ok_button.clicked.connect(dialog.accept)
    ok_button.setStyleSheet(get_ok_btn_style())

    button_layout.addStretch()
    button_layout.addWidget(cancel_button)
    button_layout.addWidget(ok_button)
    layout.addLayout(button_layout)

    dialog.setLayout(layout)
    result = dialog.exec()

    if not result:
        return

    output_root = path_edit.text()
    if not output_root:
        return

    split_name = osp.basename(osp.normpath(source_dir)) or "train"

    images_out = osp.join(output_root, "images", split_name)
    masks_out = osp.join(output_root, "masks", split_name)
    os.makedirs(images_out, exist_ok=True)
    os.makedirs(masks_out, exist_ok=True)

    # --- Load existing dataset.yaml (if any) to keep class IDs consistent
    # across multiple runs (e.g. once for train, once for val). ---
    yaml_path = osp.join(output_root, "dataset.yaml")
    names = {0: "background"}
    if osp.exists(yaml_path):
        try:
            with open(yaml_path, "r", encoding="utf-8") as f:
                existing = yaml.safe_load(f) or {}
            existing_names = existing.get("names") or {}
            names = {int(k): v for k, v in existing_names.items()}
            if 0 not in names:
                names[0] = "background"
        except Exception as e:
            logger.warning(f"Could not read existing dataset.yaml: {e}")

    label_to_id = {v: k for k, v in names.items() if k != 0}
    next_id = (max(names.keys()) + 1) if names else 1
    for label in all_labels:
        if label not in label_to_id:
            label_to_id[label] = next_id
            names[next_id] = label
            next_id += 1

    mapping_table = {"type": "grayscale", "colors": label_to_id}

    converter = LabelConverter()
    image_list = self.image_list if self.image_list else [self.filename]

    progress_dialog = QProgressDialog(
        self.tr("Exporting..."), self.tr("Cancel"), 0, len(image_list), self
    )
    progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
    progress_dialog.setWindowTitle(self.tr("Progress"))
    progress_dialog.setMinimumWidth(500)
    progress_dialog.setMinimumHeight(150)
    progress_dialog.setStyleSheet(
        get_progress_dialog_style(color="#1d1d1f", height=20)
    )
    progress_dialog.show()
    QtWidgets.QApplication.processEvents()

    canceled = False
    failed_files = []

    try:
        for i, image_file in enumerate(image_list):
            image_file_name = osp.basename(image_file)
            stem = osp.splitext(image_file_name)[0]
            label_file_name = stem + ".json"
            dst_mask_file = osp.join(masks_out, stem + ".png")

            if self.output_dir:
                src_label_file = osp.join(self.output_dir, label_file_name)
            else:
                src_label_file = osp.join(
                    osp.dirname(image_file), label_file_name
                )

            try:
                shutil.copy2(
                    image_file, osp.join(images_out, image_file_name)
                )

                has_polygons = False
                label_data = None
                if osp.exists(src_label_file):
                    with open(src_label_file, "r", encoding="utf-8") as f:
                        label_data = json.load(f)
                    has_polygons = any(
                        shape.get("shape_type") == "polygon"
                        for shape in label_data.get("shapes", [])
                    )

                if has_polygons:
                    converter.custom_to_mask(
                        src_label_file, dst_mask_file, mapping_table
                    )
                else:
                    # No shapes (or no label file): write an all-background mask
                    height, width = _get_image_shape(image_file, label_data)
                    blank_mask = np.zeros((height, width), dtype=np.uint8)
                    cv2.imencode(".png", blank_mask)[1].tofile(
                        dst_mask_file
                    )
            except Exception as e:
                failed_files.append(image_file_name)
                logger.error(
                    f"Failed to export {image_file_name}: {e}"
                )

            progress_dialog.setValue(i + 1)
            QtWidgets.QApplication.processEvents()
            if progress_dialog.wasCanceled():
                canceled = True
                break

        progress_dialog.close()

        if canceled:
            popup = Popup(
                self.tr("Export canceled."),
                self,
                icon=new_icon_path("warning", "svg"),
            )
            popup.show_popup(self, position="center")
        else:
            # --- Write / update dataset.yaml ---
            yaml_data = {"path": osp.realpath(output_root)}
            for split in ("train", "val", "test"):
                if osp.isdir(osp.join(output_root, "images", split)):
                    yaml_data[split] = f"images/{split}"
            yaml_data["masks_dir"] = "masks"
            yaml_data["names"] = dict(sorted(names.items()))

            with open(yaml_path, "w", encoding="utf-8") as f:
                yaml.dump(
                    yaml_data, f, allow_unicode=True, sort_keys=False
                )

            template = self.tr(
                "Exporting annotations successfully!\n"
                "Results have been saved to:\n"
                "%s"
            )
            message_text = template % output_root
            if failed_files:
                message_text += self.tr(
                    "\n\n%d file(s) failed to export, see log for details."
                ) % len(failed_files)
            popup = Popup(
                message_text,
                self,
                icon=new_icon_path("copy-green", "svg"),
            )
            popup.show_popup(self, popup_height=65, position="center")

    except Exception as e:
        message = f"Error occurred while exporting annotations: {str(e)}"
        progress_dialog.close()
        logger.error(message)
        popup = Popup(
            message,
            self,
            icon=new_icon_path("error", "svg"),
        )
        popup.show_popup(self, position="center")


def export_mot_annotation(self, mode):
    if not _check_filename_exist(self):
        return

    converter = LabelConverter()
    if mode in ["mot", "mots", "odvg"]:
        filter = "Classes Files (*.txt);;All Files (*)"
        self.classes_file, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            self.tr("Select a specific classes file"),
            "",
            filter,
        )
        if not self.classes_file:
            return
        converter = LabelConverter(classes_file=self.classes_file)

    dialog = QtWidgets.QDialog(self)
    dialog.setWindowTitle(self.tr("Export options"))
    dialog.setMinimumWidth(500)
    dialog.setStyleSheet(get_export_option_style())

    layout = QVBoxLayout()
    layout.setContentsMargins(24, 24, 24, 24)
    layout.setSpacing(16)

    path_layout = QVBoxLayout()
    path_label = QtWidgets.QLabel(self.tr("Export path"))
    path_layout.addWidget(path_label)

    path_input_layout = QHBoxLayout()
    path_input_layout.setSpacing(8)

    label_dir_path = osp.dirname(self.filename)
    if self.output_dir:
        label_dir_path = self.output_dir

    path_edit = QtWidgets.QLineEdit()
    path_edit.setText(osp.realpath(osp.join(label_dir_path, "..", mode)))
    path_edit.setPlaceholderText(self.tr("Select Export Directory"))

    def browse_export_path():
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            self.tr("Select Export Directory"),
            path_edit.text(),
            QtWidgets.QFileDialog.Option.DontUseNativeDialog,
        )
        if path:
            path_edit.setText(path)

    path_button = QtWidgets.QPushButton(self.tr("Browse"))
    path_button.clicked.connect(browse_export_path)
    path_button.setStyleSheet(get_cancel_btn_style())

    path_input_layout.addWidget(path_edit)
    path_input_layout.addWidget(path_button)
    path_layout.addLayout(path_input_layout)
    layout.addLayout(path_layout)

    button_layout = QHBoxLayout()
    button_layout.setContentsMargins(0, 16, 0, 0)
    button_layout.setSpacing(8)

    cancel_button = QtWidgets.QPushButton(self.tr("Cancel"))
    cancel_button.clicked.connect(dialog.reject)
    cancel_button.setStyleSheet(get_cancel_btn_style())

    ok_button = QtWidgets.QPushButton(self.tr("OK"))
    ok_button.clicked.connect(dialog.accept)
    ok_button.setStyleSheet(get_ok_btn_style())

    button_layout.addStretch()
    button_layout.addWidget(cancel_button)
    button_layout.addWidget(ok_button)
    layout.addLayout(button_layout)

    dialog.setLayout(layout)
    result = dialog.exec()

    if not result:
        return

    save_path = path_edit.text()
    if osp.exists(save_path):
        msg_box = QtWidgets.QMessageBox(self)
        msg_box.setIcon(QtWidgets.QMessageBox.Icon.Warning)
        msg_box.setWindowTitle(self.tr("Output Directory Exists!"))
        msg_box.setText(self.tr("Directory already exists. Choose an action:"))
        msg_box.setInformativeText(
            self.tr(
                "• Overwrite - Overwrite existing directory\n"
                "• Cancel - Abort export"
            )
        )

        msg_box.addButton(
            self.tr("Overwrite"), QtWidgets.QMessageBox.ButtonRole.YesRole
        )
        cancel_button = msg_box.addButton(
            self.tr("Cancel"), QtWidgets.QMessageBox.ButtonRole.RejectRole
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

    image_list = self.image_list if self.image_list else [self.filename]
    progress_dialog = QProgressDialog(
        self.tr("Exporting..."), self.tr("Cancel"), 0, 0, self
    )
    progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
    progress_dialog.setWindowTitle(self.tr("Progress"))
    progress_dialog.setMinimumWidth(500)
    progress_dialog.setMinimumHeight(150)
    progress_dialog.setRange(0, 0)
    progress_dialog.setStyleSheet(get_progress_dialog_style())

    self.export_thread = ExportThread(
        converter, image_list, label_dir_path, save_path, mode
    )

    def on_export_finished(success, error_msg):
        progress_dialog.close()
        if success:
            template = self.tr(
                "Exporting annotations successfully!\n"
                "Results have been saved to:\n"
                "%s"
            )
            message_text = template % save_path
            popup = Popup(
                message_text,
                self,
                icon=new_icon_path("copy-green", "svg"),
            )
            popup.show_popup(self, popup_height=65, position="center")
        else:
            message = (
                f"Error occurred while exporting annotations: {str(error_msg)}"
            )
            logger.error(message)
            popup = Popup(
                message,
                self,
                icon=new_icon_path("error", "svg"),
            )
            popup.show_popup(self, position="center")

    self.export_thread.finished.connect(on_export_finished)

    progress_dialog.show()
    self.export_thread.start()

    progress_dialog.canceled.connect(self.export_thread.terminate)


def export_odvg_annotation(self):
    export_mot_annotation(self, "odvg")


def export_pporc_annotation(self, mode):
    if not _check_filename_exist(self):
        return

    dialog = QtWidgets.QDialog(self)
    dialog.setWindowTitle(self.tr("Export options"))
    dialog.setMinimumWidth(500)
    dialog.setStyleSheet(get_export_option_style())

    layout = QVBoxLayout()
    layout.setContentsMargins(24, 24, 24, 24)
    layout.setSpacing(16)

    path_layout = QVBoxLayout()
    path_label = QtWidgets.QLabel(self.tr("Export path"))
    path_layout.addWidget(path_label)

    path_input_layout = QHBoxLayout()
    path_input_layout.setSpacing(8)

    label_dir_path = osp.dirname(self.filename)
    if self.output_dir:
        label_dir_path = self.output_dir

    path_edit = QtWidgets.QLineEdit()
    path_edit.setText(
        osp.realpath(osp.join(label_dir_path, "..", f"ppocr_{mode}"))
    )
    path_edit.setPlaceholderText(self.tr("Select Export Directory"))

    def browse_export_path():
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            self.tr("Select Export Directory"),
            path_edit.text(),
            QtWidgets.QFileDialog.Option.DontUseNativeDialog,
        )
        if path:
            path_edit.setText(path)

    path_button = QtWidgets.QPushButton(self.tr("Browse"))
    path_button.clicked.connect(browse_export_path)
    path_button.setStyleSheet(get_cancel_btn_style())

    path_input_layout.addWidget(path_edit)
    path_input_layout.addWidget(path_button)
    path_layout.addLayout(path_input_layout)
    layout.addLayout(path_layout)

    button_layout = QHBoxLayout()
    button_layout.setContentsMargins(0, 16, 0, 0)
    button_layout.setSpacing(8)

    cancel_button = QtWidgets.QPushButton(self.tr("Cancel"))
    cancel_button.clicked.connect(dialog.reject)
    cancel_button.setStyleSheet(get_cancel_btn_style())

    ok_button = QtWidgets.QPushButton(self.tr("OK"))
    ok_button.clicked.connect(dialog.accept)
    ok_button.setStyleSheet(get_ok_btn_style())

    button_layout.addStretch()
    button_layout.addWidget(cancel_button)
    button_layout.addWidget(ok_button)
    layout.addLayout(button_layout)

    dialog.setLayout(layout)
    result = dialog.exec()

    if not result:
        return

    save_path = path_edit.text()
    if osp.exists(save_path):
        msg_box = QtWidgets.QMessageBox(self)
        msg_box.setIcon(QtWidgets.QMessageBox.Icon.Warning)
        msg_box.setWindowTitle(self.tr("Output Directory Exists!"))
        msg_box.setText(self.tr("Directory already exists. Choose an action:"))
        msg_box.setInformativeText(
            self.tr(
                "• Overwrite - Overwrite existing directory\n"
                "• Cancel - Abort export"
            )
        )

        msg_box.addButton(
            self.tr("Overwrite"), QtWidgets.QMessageBox.ButtonRole.YesRole
        )
        cancel_button = msg_box.addButton(
            self.tr("Cancel"), QtWidgets.QMessageBox.ButtonRole.RejectRole
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

    if mode == "rec":
        save_crop_img_path = osp.join(save_path, "crop_img")
        if osp.exists(save_crop_img_path):
            shutil.rmtree(save_crop_img_path)
        os.makedirs(save_crop_img_path, exist_ok=True)
        for fname in ("Label.txt", "rec_gt.txt"):
            fpath = osp.join(save_path, fname)
            if osp.exists(fpath):
                os.remove(fpath)
    elif mode == "kie":
        total_class_set = set()
        class_list_file = osp.join(save_path, "class_list.txt")
        ppocr_kie_file = osp.join(save_path, "ppocr_kie.json")
        if osp.exists(ppocr_kie_file):
            os.remove(ppocr_kie_file)

    converter = LabelConverter()

    image_list = self.image_list if self.image_list else [self.filename]
    progress_dialog = QProgressDialog(
        self.tr("Exporting..."), self.tr("Cancel"), 0, len(image_list), self
    )
    progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
    progress_dialog.setWindowTitle(self.tr("Progress"))
    progress_dialog.setMinimumWidth(500)
    progress_dialog.setMinimumHeight(150)
    progress_dialog.setStyleSheet(
        get_progress_dialog_style(color="#1d1d1f", height=20)
    )

    try:
        for i, image_file in enumerate(image_list):
            image_file_name = osp.basename(image_file)
            label_file_name = osp.splitext(image_file_name)[0] + ".json"
            label_file = osp.join(osp.dirname(image_file), label_file_name)
            if mode == "rec":
                converter.custom_to_ppocr(
                    image_file, label_file, save_path, mode
                )
            elif mode == "kie":
                class_set = converter.custom_to_ppocr(
                    image_file, label_file, save_path, mode
                )
                total_class_set = total_class_set.union(class_set)

            progress_dialog.setValue(i)
            if progress_dialog.wasCanceled():
                break

        if mode == "kie":
            with open(class_list_file, "w") as f:
                for c in total_class_set:
                    f.writelines(f"{c.upper()}\n")

        progress_dialog.close()

        template = self.tr(
            "Exporting annotations successfully!\n"
            "Results have been saved to:\n"
            "%s"
        )
        message_text = template % save_path
        popup = Popup(
            message_text,
            self,
            icon=new_icon_path("copy-green", "svg"),
        )
        popup.show_popup(self, popup_height=65, position="center")

    except Exception as e:
        message = f"Error occurred while exporting annotations: {str(e)}"
        progress_dialog.close()
        logger.error(message)
        popup = Popup(
            message,
            self,
            icon=new_icon_path("error", "svg"),
        )
        popup.show_popup(self, position="center")


def export_vlm_r1_ovd_annotation(self):
    if not _check_filename_exist(self):
        return

    dialog = QtWidgets.QDialog(self)
    dialog.setWindowTitle(self.tr("Export VLM-R1 OVD Annotation"))
    dialog.setMinimumWidth(500)
    dialog.setStyleSheet(get_export_option_style())

    main_layout = QVBoxLayout()
    main_layout.setContentsMargins(24, 24, 24, 24)
    main_layout.setSpacing(16)

    # --- File path selection ---
    path_layout = QVBoxLayout()
    path_label = QtWidgets.QLabel(self.tr("Export to"))
    path_layout.addWidget(path_label)

    path_input_layout = QHBoxLayout()
    path_input_layout.setSpacing(8)

    # Default export path and filename
    label_dir_path = osp.dirname(self.filename)
    if self.output_dir:
        label_dir_path = self.output_dir
    default_export_path = osp.realpath(
        osp.join(label_dir_path, "..", "vlm_r1_ovd.jsonl")
    )

    path_edit = QtWidgets.QLineEdit()
    path_edit.setText(default_export_path)
    path_edit.setPlaceholderText(self.tr("Select Export File"))

    def browse_export_file():
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            dialog,
            self.tr("Select Export File"),
            path_edit.text(),
            "JSONL Files (*.jsonl)",
            options=QtWidgets.QFileDialog.Option.DontUseNativeDialog,
        )
        if path:
            if not path.endswith(".jsonl"):
                path += ".jsonl"
            path_edit.setText(path)

    path_button = QtWidgets.QPushButton(self.tr("Browse"))
    path_button.clicked.connect(browse_export_file)
    path_button.setStyleSheet(get_cancel_btn_style())

    path_input_layout.addWidget(path_edit)
    path_input_layout.addWidget(path_button)
    path_layout.addLayout(path_input_layout)
    main_layout.addLayout(path_layout)

    # --- Prefix input ---
    prefix_layout = QVBoxLayout()
    prefix_layout.setSpacing(8)

    prefix_label = QHBoxLayout()
    prefix_label.setSpacing(2)

    prefix_title_label = QtWidgets.QLabel(self.tr("Prefix:"))
    prefix_preview_label = QtWidgets.QLabel("")
    prefix_preview_label.setStyleSheet(
        "color: gray; font-style: italic; padding-left: 5px;"
    )

    prefix_label.addWidget(prefix_title_label)
    prefix_label.addWidget(prefix_preview_label)
    prefix_label.addStretch()

    prefix_edit = QtWidgets.QLineEdit()
    prefix_edit_placeholder = self.tr(
        "Optional prefix for image filenames (e.g., 'path/to/images/')"
    )
    prefix_edit.setPlaceholderText(prefix_edit_placeholder)

    prefix_layout.addLayout(prefix_label)
    prefix_layout.addWidget(prefix_edit)
    main_layout.addLayout(prefix_layout)

    def _update_preview():
        prefix = prefix_edit.text()
        if not prefix:
            prefix = "demo.jpg"
        else:
            prefix += "demo.jpg"
        preview_text = self.tr("{}").format(prefix)
        prefix_preview_label.setText(preview_text)

    prefix_edit.textChanged.connect(_update_preview)
    _update_preview()

    # --- Class Filtering ---
    self.classes_file = None

    # --- Class Label ---
    class_label = QtWidgets.QLabel(self.tr("Use specific classes? (Optional)"))
    main_layout.addWidget(class_label)

    # --- Class Path Layout ---
    class_path_layout = QHBoxLayout()
    class_path_layout.setSpacing(8)

    class_path_edit = QtWidgets.QLineEdit()
    class_path_edit.setPlaceholderText(
        self.tr("Select a specific classes file")
    )

    def _handle_class_file_upload():
        filter = "Classes Files (*.txt);;All Files (*)"
        classes_file, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            self.tr("Select a specific classes file"),
            "",
            filter,
        )
        class_path_edit.setText(classes_file)

    class_path_button = QtWidgets.QPushButton(self.tr("Upload"))
    class_path_edit.textChanged.connect(
        lambda text: setattr(self, "classes_file", text)
    )
    class_path_button.clicked.connect(_handle_class_file_upload)
    class_path_button.setStyleSheet(get_cancel_btn_style())

    class_path_layout.addWidget(class_path_edit)
    class_path_layout.addWidget(class_path_button)
    main_layout.addLayout(class_path_layout)

    # --- Hint Label ---
    class_hint_label = QtWidgets.QLabel(
        self.tr(
            "Hint: If you don't upload a specific classes file, all unique labels found in one of the annotations will be used for the export."
        )
    )
    class_hint_label.setStyleSheet(
        "color: gray; font-style: italic; padding-left: 5px;"
    )
    class_hint_label.setWordWrap(True)
    main_layout.addWidget(class_hint_label)

    # --- Buttons layout ---
    button_layout = QHBoxLayout()
    button_layout.setContentsMargins(0, 16, 0, 0)
    button_layout.setSpacing(8)

    cancel_button = QtWidgets.QPushButton(self.tr("Cancel"))
    cancel_button.clicked.connect(dialog.reject)
    cancel_button.setStyleSheet(get_cancel_btn_style())

    ok_button = QtWidgets.QPushButton(self.tr("OK"))
    ok_button.clicked.connect(dialog.accept)
    ok_button.setStyleSheet(get_ok_btn_style())

    button_layout.addWidget(cancel_button)
    button_layout.addWidget(ok_button)
    main_layout.addLayout(button_layout)

    dialog.setLayout(main_layout)
    result = dialog.exec()

    if not result:
        return

    save_path = path_edit.text()
    prefix = prefix_edit.text().strip()

    # --- File Exists Check ---
    if osp.exists(save_path):
        msg_box = QtWidgets.QMessageBox(self)
        msg_box.setIcon(QtWidgets.QMessageBox.Icon.Warning)
        msg_box.setWindowTitle(self.tr("File Exists!"))
        msg_box.setText(self.tr("File already exists. Choose an action:"))
        msg_box.setInformativeText(
            self.tr(
                "• Overwrite - Replace existing file\n"  # Escaped newline for informative text
                "• Cancel - Abort export"
            )
        )
        _ = msg_box.addButton(
            self.tr("Overwrite"), QtWidgets.QMessageBox.ButtonRole.YesRole
        )
        cancel_msg_button = msg_box.addButton(
            self.tr("Cancel"), QtWidgets.QMessageBox.ButtonRole.RejectRole
        )
        msg_box.setDefaultButton(cancel_msg_button)

        msg_box.setStyleSheet(get_msg_box_style())
        msg_box.exec()

        clicked_button = msg_box.clickedButton()
        if clicked_button == cancel_msg_button:
            return

    image_list = self.image_list if self.image_list else [self.filename]
    label_dir_path = osp.dirname(self.filename)
    if self.output_dir:
        label_dir_path = self.output_dir

    # --- Attempt to create LabelConverter first ---
    try:
        converter = LabelConverter(classes_file=self.classes_file)
    except Exception as e:
        logger.error(f"Failed to initialize LabelConverter: {e}")
        template = self.tr("Error initializing export: %s")
        popup = Popup(
            template % e,
            self,
            icon=new_icon_path("error", "svg"),
        )
        popup.show_popup(self, position="center")
        return

    # --- Progress Dialog ---
    progress_dialog = QProgressDialog(
        self.tr("Exporting..."), self.tr("Cancel"), 0, 0, self
    )
    progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
    progress_dialog.setWindowTitle(self.tr("Progress"))
    progress_dialog.setMinimumWidth(500)
    progress_dialog.setMinimumHeight(150)
    progress_dialog.setRange(0, 0)
    progress_dialog.setStyleSheet(get_progress_dialog_style())

    try:
        self.export_thread = ExportThread(
            converter,
            image_list,
            label_dir_path,
            save_path,
            "vlm_r1_ovd",
            prefix=prefix,
        )

        def on_export_finished(success, error_msg):
            progress_dialog.close()
            if success:
                template = self.tr(
                    "Exporting annotations successfully!\n"
                    "Results have been saved to:\n"
                    "%s"
                )
                message_text = template % save_path
                popup = Popup(
                    message_text,
                    self,
                    icon=new_icon_path("copy-green", "svg"),
                )
                popup.show_popup(self, popup_height=65, position="center")
            else:
                message = f"Error occurred while exporting annotations: {str(error_msg)}"
                logger.error(message)
                popup = Popup(
                    message,
                    self,
                    icon=new_icon_path("error", "svg"),
                )
                popup.show_popup(self, position="center")

        self.export_thread.finished.connect(on_export_finished)

        progress_dialog.show()
        self.export_thread.start()

        progress_dialog.canceled.connect(self.export_thread.terminate)

    except Exception as e:
        message = f"Error occurred while exporting annotations: {str(e)}"
        progress_dialog.close()
        logger.error(message)
        popup = Popup(
            message,
            self,
            icon=new_icon_path("error", "svg"),
        )
        popup.show_popup(self, position="center")
