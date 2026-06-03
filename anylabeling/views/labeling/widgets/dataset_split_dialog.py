import os
import random
import shutil

from pathlib import Path

from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QDialog

from anylabeling.views.labeling.logger import logger
from anylabeling.views.labeling.utils.qt import new_icon_path
from anylabeling.views.labeling.utils.style import (
    get_cancel_btn_style,
    get_dialog_style,
    get_double_spinbox_style,
    get_lineedit_style,
    get_msg_box_style,
    get_ok_btn_style,
    get_theme,
)
from anylabeling.views.labeling.widgets.popup import Popup

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class DatasetSplitDialog(QDialog):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent_widget = parent
        self.is_processing = False
        self.cancel_requested = False
        self._setup_ui()

    def _setup_ui(self):
        self.setWindowTitle(self.tr("Dataset Split"))
        self.resize(680, 520)
        self.setStyleSheet(get_dialog_style())
        theme = get_theme()

        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(12)

        # --- Source path ---
        path_label = QtWidgets.QLabel(self.tr("Source Path:"))
        path_layout = QtWidgets.QHBoxLayout()
        self.path_edit = QtWidgets.QLineEdit(self)
        self.path_edit.setReadOnly(True)
        self.path_edit.setStyleSheet(get_lineedit_style())
        self.browse_button = QtWidgets.QPushButton(self.tr("Browse"))
        self.browse_button.setIcon(
            QtGui.QIcon(new_icon_path("folder", "svg"))
        )
        self.browse_button.setFixedWidth(100)
        self.browse_button.clicked.connect(self._browse_folder)
        path_layout.addWidget(self.path_edit, 1)
        path_layout.addWidget(self.browse_button, 0)
        main_layout.addWidget(path_label)
        main_layout.addLayout(path_layout)

        # --- Ratios ---
        ratio_group = QtWidgets.QGroupBox(self.tr("Split Ratios"))
        ratio_group.setStyleSheet(f"""
            QGroupBox {{
                font-weight: bold;
                border: 1px solid {theme["border_light"]};
                border-radius: 6px;
                margin-top: 10px;
                padding-top: 16px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 6px;
            }}
        """)
        ratio_layout = QtWidgets.QGridLayout()
        ratio_layout.setHorizontalSpacing(16)
        ratio_layout.setVerticalSpacing(8)

        spin_style = get_double_spinbox_style()

        self.train_spin = self._create_ratio_spin(0.80, spin_style)
        self.val_spin = self._create_ratio_spin(0.20, spin_style)
        self.test_spin = self._create_ratio_spin(0.00, spin_style)

        self.train_label = QtWidgets.QLabel("80%")
        self.val_label = QtWidgets.QLabel("20%")
        self.test_label = QtWidgets.QLabel("0%")

        ratio_layout.addWidget(QtWidgets.QLabel(self.tr("Train:")), 0, 0)
        ratio_layout.addWidget(self.train_spin, 0, 1)
        ratio_layout.addWidget(self.train_label, 0, 2)

        ratio_layout.addWidget(QtWidgets.QLabel(self.tr("Val:")), 1, 0)
        ratio_layout.addWidget(self.val_spin, 1, 1)
        ratio_layout.addWidget(self.val_label, 1, 2)

        ratio_layout.addWidget(QtWidgets.QLabel(self.tr("Test:")), 2, 0)
        ratio_layout.addWidget(self.test_spin, 2, 1)
        ratio_layout.addWidget(self.test_label, 2, 2)

        self.total_ratio_label = QtWidgets.QLabel(self.tr("Total: 100%"))
        self.total_ratio_label.setStyleSheet(
            f"color: {theme['text']}; font-weight: bold;"
        )
        ratio_layout.addWidget(self.total_ratio_label, 0, 3, 3, 1)

        ratio_group.setLayout(ratio_layout)
        main_layout.addWidget(ratio_group)

        # --- Info label ---
        self.info_label = QtWidgets.QLabel("")
        self.info_label.setStyleSheet(f"color: {theme['text_secondary']};")
        main_layout.addWidget(self.info_label)

        # --- Progress bar ---
        self.progress_bar = QtWidgets.QProgressBar(self)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setStyleSheet(f"""
            QProgressBar {{
                background-color: {theme["surface"]};
                border: 1px solid {theme["border_light"]};
                border-radius: 0px;
                text-align: center;
                color: {theme["text"]};
                font-size: 12px;
                font-weight: 500;
                min-height: 22px;
            }}
            QProgressBar::chunk {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #0066FF,
                    stop:0.5 #00A6FF,
                    stop:1 #0066FF);
                border-radius: 0px;
            }}
        """)
        main_layout.addWidget(self.progress_bar)

        # --- Log terminal ---
        self.log_terminal = QtWidgets.QListWidget(self)
        self.log_terminal.setIconSize(QtCore.QSize(16, 16))
        self.log_terminal.setSpacing(2)
        self.log_terminal.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.NoSelection
        )
        self.log_terminal.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.log_terminal.setTextElideMode(Qt.TextElideMode.ElideNone)
        self.log_terminal.setHorizontalScrollMode(
            QtWidgets.QAbstractItemView.ScrollMode.ScrollPerPixel
        )
        self.log_terminal.setStyleSheet(f"""
            QListWidget {{
                background-color: {theme["background_secondary"]};
                color: {theme["text"]};
                border: 1px solid {theme["border"]};
                border-radius: 0px;
                padding: 6px;
                outline: none;
            }}
            QListWidget::item {{
                border: none;
                padding: 6px 8px;
                margin: 2px 0px;
            }}
            QListWidget::item:selected {{
                background-color: {theme["selection"]};
                color: {theme["selection_text"]};
            }}
        """)
        main_layout.addWidget(self.log_terminal, 1)

        # --- Buttons ---
        button_layout = QtWidgets.QHBoxLayout()
        button_layout.addStretch(1)
        self.cancel_button = QtWidgets.QPushButton(self.tr("Cancel"))
        self.cancel_button.setStyleSheet(get_cancel_btn_style())
        self.start_button = QtWidgets.QPushButton(self.tr("Start"))
        self.start_button.setStyleSheet(get_ok_btn_style())
        self.start_button.setDefault(True)
        button_layout.addWidget(self.cancel_button)
        button_layout.addWidget(self.start_button)
        main_layout.addLayout(button_layout)

        # --- Signals ---
        self.train_spin.valueChanged.connect(self._update_ratio_labels)
        self.val_spin.valueChanged.connect(self._update_ratio_labels)
        self.test_spin.valueChanged.connect(self._update_ratio_labels)
        self.cancel_button.clicked.connect(self._on_cancel_clicked)
        self.start_button.clicked.connect(self._on_start_clicked)

        # --- Icons ---
        self.check_icon = QtGui.QIcon(new_icon_path("check", "svg"))
        self.error_icon = QtGui.QIcon(new_icon_path("error", "svg"))
        self.folder_icon = QtGui.QIcon(new_icon_path("folder", "svg"))

        # --- Init ---
        self._init_source_path()
        self._update_ratio_labels()
        self._set_progress(0, 1)

    def _create_ratio_spin(self, default, style):
        spin = QtWidgets.QDoubleSpinBox(self)
        spin.setRange(0.0, 1.0)
        spin.setSingleStep(0.05)
        spin.setDecimals(2)
        spin.setValue(default)
        spin.setStyleSheet(style)
        return spin

    def _init_source_path(self):
        src_path = ""
        if hasattr(self.parent_widget, "last_open_dir") and self.parent_widget.last_open_dir:
            src_path = self.parent_widget.last_open_dir
        elif hasattr(self.parent_widget, "filename") and self.parent_widget.filename:
            src_path = os.path.dirname(str(self.parent_widget.filename))
        self.path_edit.setText(src_path)
        self._update_info_label()

    def _browse_folder(self):
        current = self.path_edit.text() or "."
        dir_path = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            self.tr("Select Dataset Folder"),
            current,
        )
        if dir_path:
            self.path_edit.setText(dir_path)
            self._update_info_label()

    def _update_ratio_labels(self):
        t = self.train_spin.value()
        v = self.val_spin.value()
        te = self.test_spin.value()
        self.train_label.setText(f"{t * 100:.0f}%")
        self.val_label.setText(f"{v * 100:.0f}%")
        self.test_label.setText(f"{te * 100:.0f}%")
        total = t + v + te
        theme = get_theme()
        if abs(total - 1.0) < 0.001:
            self.total_ratio_label.setText(self.tr("Total: 100%"))
            self.total_ratio_label.setStyleSheet(
                f"color: {theme['success']}; font-weight: bold;"
            )
        else:
            self.total_ratio_label.setText(
                self.tr(f"Total: {total * 100:.0f}%")
            )
            self.total_ratio_label.setStyleSheet(
                f"color: #FF4444; font-weight: bold;"
            )

    def _update_info_label(self):
        src_path = self.path_edit.text()
        if not src_path or not os.path.isdir(src_path):
            self.info_label.setText(
                self.tr("Please select a valid source folder.")
            )
            return
        pair_count = self._count_valid_pairs(src_path)
        self.info_label.setText(
            self.tr(f"Detected {pair_count} valid image-label pairs.")
        )

    def _get_valid_pairs(self, src_path):
        pairs = []
        try:
            for f in os.listdir(src_path):
                ext = os.path.splitext(f)[1].lower()
                if ext in IMAGE_EXTENSIONS:
                    json_path = os.path.join(
                        src_path, os.path.splitext(f)[0] + ".json"
                    )
                    if os.path.exists(json_path):
                        pairs.append(os.path.join(src_path, f))
        except OSError:
            pass
        return pairs

    def _count_valid_pairs(self, src_path):
        return len(self._get_valid_pairs(src_path))

    def _set_progress(self, current, total):
        safe_total = max(total, 1)
        self.progress_bar.setRange(0, safe_total)
        self.progress_bar.setValue(min(current, safe_total))
        percent = int(current * 100 / safe_total)
        self.progress_bar.setFormat(f"{current}/{total} ({percent}%)")

    def _append_success_log(self, text):
        item = QtWidgets.QListWidgetItem(self.check_icon, f" {text}")
        self.log_terminal.addItem(item)
        self.log_terminal.scrollToBottom()

    def _append_error_log(self, text):
        item = QtWidgets.QListWidgetItem(self.error_icon, f" {text}")
        self.log_terminal.addItem(item)
        self.log_terminal.scrollToBottom()

    def _append_info_log(self, text):
        item = QtWidgets.QListWidgetItem(self.folder_icon, f" {text}")
        self.log_terminal.addItem(item)
        self.log_terminal.scrollToBottom()

    def _set_controls_enabled(self, enabled):
        self.path_edit.setEnabled(enabled)
        self.browse_button.setEnabled(enabled)
        self.train_spin.setEnabled(enabled)
        self.val_spin.setEnabled(enabled)
        self.test_spin.setEnabled(enabled)
        self.start_button.setEnabled(enabled)
        self.cancel_button.setEnabled(True)

    def _on_cancel_clicked(self):
        if self.is_processing:
            self.cancel_requested = True
            self.cancel_button.setEnabled(False)
            return
        self.reject()

    def _confirm_split(self, total_count):
        response = QtWidgets.QMessageBox(self)
        response.setIcon(QtWidgets.QMessageBox.Icon.Warning)
        response.setWindowTitle(self.tr("Warning"))
        response.setText(
            self.tr(
                "This will move your image and label files into train/val/test subfolders."
            )
        )
        response.setInformativeText(
            self.tr(
                f"Are you sure you want to split {total_count} pairs of data?"
            )
        )
        response.setStandardButtons(
            QtWidgets.QMessageBox.StandardButton.Cancel
            | QtWidgets.QMessageBox.StandardButton.Ok
        )
        response.setDefaultButton(QtWidgets.QMessageBox.StandardButton.Ok)
        response.setStyleSheet(get_msg_box_style())
        return response.exec() == QtWidgets.QMessageBox.StandardButton.Ok

    def _on_start_clicked(self):
        if self.is_processing:
            return

        src_path = self.path_edit.text().strip()
        if not src_path or not os.path.isdir(src_path):
            QtWidgets.QMessageBox.warning(
                self,
                self.tr("Error"),
                self.tr("Please select a valid source folder."),
            )
            return

        train_ratio = self.train_spin.value()
        val_ratio = self.val_spin.value()
        test_ratio = self.test_spin.value()
        total_ratio = train_ratio + val_ratio + test_ratio

        if total_ratio <= 0:
            QtWidgets.QMessageBox.warning(
                self,
                self.tr("Error"),
                self.tr("Split ratios must be greater than 0."),
            )
            return

        # Normalize ratios
        train_r = train_ratio / total_ratio
        val_r = val_ratio / total_ratio
        test_r = test_ratio / total_ratio

        # Find valid pairs
        try:
            valid_pairs = self._get_valid_pairs(src_path)
        except OSError as e:
            QtWidgets.QMessageBox.warning(
                self,
                self.tr("Error"),
                self.tr(f"Failed to read source folder: {e}"),
            )
            return

        total_count = len(valid_pairs)
        if total_count == 0:
            QtWidgets.QMessageBox.warning(
                self,
                self.tr("Error"),
                self.tr(
                    "No valid image-label pairs found. "
                    "Please check the source folder."
                ),
            )
            return

        if not self._confirm_split(total_count):
            return

        # Shuffle
        random.shuffle(valid_pairs)

        # Calculate split indices
        train_num = int(total_count * train_r)
        val_num = int(total_count * val_r)

        train_files = valid_pairs[:train_num]
        if test_ratio == 0:
            val_files = valid_pairs[train_num:]
            test_files = []
        else:
            val_files = valid_pairs[train_num : train_num + val_num]
            test_files = valid_pairs[train_num + val_num :]

        splits = {"train": train_files, "val": val_files, "test": test_files}

        # Start processing
        self.log_terminal.clear()
        self.cancel_requested = False
        self.is_processing = True
        self._set_controls_enabled(False)

        total_moves = sum(len(f) for f in splits.values())
        self._set_progress(0, max(total_moves, 1))

        self._append_info_log(
            self.tr(f"Total: {total_count} pairs")
        )
        self._append_info_log(
            self.tr(
                f"Train: {len(train_files)}, "
                f"Val: {len(val_files)}, "
                f"Test: {len(test_files)}"
            )
        )

        processed_count = 0
        error_count = 0
        src_path_obj = Path(src_path)

        for split_name, files in splits.items():
            if not files:
                continue

            if self.cancel_requested:
                break

            target_dir = src_path_obj / split_name
            target_dir.mkdir(exist_ok=True)
            self._append_info_log(
                self.tr(f"Creating {split_name} folder: {target_dir}")
            )

            for img_path_str in files:
                if self.cancel_requested:
                    break

                img_path = Path(img_path_str)
                json_path = img_path.with_suffix(".json")

                try:
                    shutil.move(
                        str(img_path), str(target_dir / img_path.name)
                    )
                    shutil.move(
                        str(json_path), str(target_dir / json_path.name)
                    )
                    self._append_success_log(
                        f"{split_name}/{img_path.name}"
                    )
                except Exception as e:
                    error_count += 1
                    logger.error(
                        f"Error moving {img_path.name}: {e}"
                    )
                    self._append_error_log(
                        f"{img_path.name}: {e}"
                    )

                processed_count += 1
                self._set_progress(processed_count, total_moves)
                QtWidgets.QApplication.processEvents()

        was_canceled = self.cancel_requested
        self.is_processing = False
        self.cancel_requested = False
        self._set_controls_enabled(True)

        if was_canceled:
            popup = Popup(
                self.tr("Dataset split canceled."),
                self.parent_widget,
                msec=1200,
                icon=new_icon_path("error", "svg"),
            )
            popup.show_popup(self.parent_widget, position="center")
        elif error_count > 0:
            popup = Popup(
                self.tr("Dataset split completed with errors."),
                self.parent_widget,
                msec=1200,
                icon=new_icon_path("error", "svg"),
            )
            popup.show_popup(self.parent_widget, position="center")
        else:
            popup = Popup(
                self.tr("Dataset split completed successfully!"),
                self.parent_widget,
                msec=1000,
                icon=new_icon_path("copy-green", "svg"),
            )
            popup.show_popup(
                self.parent_widget, popup_height=65, position="center"
            )
            # Auto-open train folder
            train_dir = os.path.join(src_path, "train")
            if os.path.isdir(train_dir):
                self.parent_widget.import_image_folder(train_dir)
