import os
import tempfile
import unittest
import numpy as np

from PyQt6 import QtCore, QtWidgets
from anylabeling.views.labeling.utils.raw_reader import (
    parse_raw_file_info,
    RawVolume,
    slice_to_png_bytes,
    slice_to_qimage,
)
from anylabeling.views.labeling.utils.opencv import qt_img_to_rgb_cv_img
from anylabeling.views.labeling.widgets.raw_z_slider import RawZSlider
from anylabeling.views.labeling.widgets.raw_orthogonal_dialog import (
    RawOrthogonalViewDialog,
    OrthogonalCanvasWidget,
    RawOrthoCornerWidget,
)

app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


class TestRawVolume(unittest.TestCase):
    def test_synthetic_raw_volume(self):
        w, h, d = 50, 40, 10
        total_bytes = w * h * d
        with tempfile.TemporaryDirectory() as temp_dir:
            raw_filename = f"test_volume_{w}x{h}x{d}_2.5um.raw"
            raw_path = os.path.join(temp_dir, raw_filename)
            arr = np.arange(total_bytes, dtype=np.uint8).reshape((d, h, w))
            with open(raw_path, "wb") as f:
                f.write(arr.tobytes())

            info = parse_raw_file_info(raw_path)
            self.assertIsNotNone(info)
            self.assertEqual(info.width, w)
            self.assertEqual(info.height, h)
            self.assertEqual(info.depth, d)
            self.assertEqual(info.data_type, "uint8")
            self.assertAlmostEqual(info.pixel_width, 2.5)

            vol = RawVolume(info, use_memmap=True)
            self.assertEqual(vol.depth, d)
            self.assertEqual(vol.height, h)
            self.assertEqual(vol.width, w)

            slice_3 = vol.get_axial_slice(3)
            self.assertEqual(slice_3.shape, (h, w))
            np.testing.assert_array_equal(slice_3, arr[3])

            png_bytes = slice_to_png_bytes(slice_3)
            self.assertGreater(len(png_bytes), 0)

            qimg = slice_to_qimage(slice_3)
            self.assertFalse(qimg.isNull())
            self.assertEqual(qimg.width(), w)
            self.assertEqual(qimg.height(), h)

            cv_img = qt_img_to_rgb_cv_img(qimg, raw_path)
            self.assertEqual(cv_img.shape, (h, w, 3))

            # Test Coronal and Sagittal slices
            coronal = vol.get_coronal_slice(20)
            self.assertEqual(coronal.shape, (d, w))
            sagittal = vol.get_sagittal_slice(25)
            self.assertEqual(sagittal.shape, (h, d))

            val = vol.get_voxel_value(25, 20, 3)
            self.assertEqual(val, arr[3, 20, 25])

            vol.close()

    def test_user_sample_raw_file_if_available(self):
        candidates = [
            r"D:\raw_test\20260918141911407__pad5-3_X836_Y1177_Z119-215_ROI-300x300x97-5um.raw",
        ]
        sample_path = next((p for p in candidates if os.path.exists(p)), None)
        if not sample_path:
            self.skipTest("Sample grayscale raw file not found")

        info = parse_raw_file_info(sample_path)
        self.assertIsNotNone(info)
        self.assertEqual(info.width, 300)
        self.assertEqual(info.height, 300)
        self.assertIn(info.depth, [97, 557])
        self.assertEqual(info.data_type, "uint8")
        self.assertAlmostEqual(info.pixel_width, 5.0)
        self.assertEqual(info.unit, "µm")

        vol = RawVolume(info, use_memmap=True)
        s0 = vol.get_axial_slice(0)
        self.assertEqual(s0.shape, (300, 300))
        xz = vol.get_coronal_slice(150)
        self.assertEqual(xz.shape, (info.depth, 300))
        yz = vol.get_sagittal_slice(150)
        self.assertEqual(yz.shape, (300, info.depth))
        vol.close()

    def test_user_color_raw_file_if_available(self):
        candidates = [
            r"D:\raw_test\20260918141911407__pad5-3_X836_Y1177_Z119-215_ROI-300x300x97_彩色-5um.raw",
        ]
        sample_path = next((p for p in candidates if os.path.exists(p)), None)
        if not sample_path:
            self.skipTest("Sample color raw file not found")

        info = parse_raw_file_info(sample_path)
        self.assertIsNotNone(info)
        self.assertEqual(info.width, 300)
        self.assertEqual(info.height, 300)
        self.assertIn(info.depth, [97, 557])
        self.assertEqual(info.data_type, "rgb8")

        vol = RawVolume(info, use_memmap=True)
        s0 = vol.get_axial_slice(0)
        self.assertEqual(s0.shape, (300, 300, 3))
        qimg = slice_to_qimage(s0)
        self.assertFalse(qimg.isNull())
        self.assertEqual(qimg.width(), 300)
        self.assertEqual(qimg.height(), 300)

        cv_img = qt_img_to_rgb_cv_img(qimg, sample_path)
        self.assertEqual(cv_img.shape, (300, 300, 3))

        # Test coronal & sagittal color slices
        xz = vol.get_coronal_slice(150)
        self.assertEqual(xz.shape, (info.depth, 300, 3))
        yz = vol.get_sagittal_slice(150)
        self.assertEqual(yz.shape, (300, info.depth, 3))

        vol.close()

    def test_raw_z_slider_layout_and_buttons(self):
        slider = RawZSlider()
        self.assertIsNotNone(slider._ortho_btn)
        self.assertIsNotNone(slider._crosshair_btn)
        self.assertIsNotNone(slider._voxel_info_label)

        # Test signals
        ortho_received = []
        cross_received = []
        slider.orthogonal_view_toggled.connect(lambda b: ortho_received.append(b))
        slider.crosshair_toggled.connect(lambda b: cross_received.append(b))

        slider._ortho_btn.setChecked(True)
        slider._crosshair_btn.setChecked(True)
        self.assertEqual(ortho_received, [True])
        self.assertEqual(cross_received, [True])

        # Test programmatic setter without re-emitting
        slider.set_orthogonal_view_checked(False)
        self.assertFalse(slider._ortho_btn.isChecked())
        slider.set_crosshair_checked(False)
        self.assertFalse(slider._crosshair_btn.isChecked())

        # Test voxel info display
        slider.set_voxel_info(100, 150, 45, 180, phys_z=163.0, unit="µm")
        self.assertIn("100", slider._voxel_info_label.text())
        self.assertIn("180", slider._voxel_info_label.text())

    def test_raw_orthogonal_views_dialog(self):
        w, h, d = 30, 20, 10
        total_bytes = w * h * d
        with tempfile.TemporaryDirectory() as temp_dir:
            raw_path = os.path.join(temp_dir, f"sample_{w}x{h}x{d}_5.0um.raw")
            arr = np.arange(total_bytes, dtype=np.uint8).reshape((d, h, w))
            with open(raw_path, "wb") as f:
                f.write(arr.tobytes())

            info = parse_raw_file_info(raw_path)
            vol = RawVolume(info, use_memmap=True)

            dlg = RawOrthogonalViewDialog()
            points_received = []
            dlg.point_changed.connect(lambda x, y, z: points_received.append((x, y, z)))

            try:
                dlg.set_volume_data(vol, info, x=15, y=10, z=5)
                self.assertEqual(dlg.curr_x, 15)
                self.assertEqual(dlg.curr_y, 10)
                self.assertEqual(dlg.curr_z, 5)

                # Test clicking on XZ canvas: (u=20, v=8) -> x=20, z=8
                dlg._on_xz_clicked(20, 8)
                self.assertEqual(points_received[-1], (20, 10, 8))

                # Test clicking on YZ canvas: (u=15, v=3) -> y=15, z=3
                dlg._on_yz_clicked(15, 3)
                self.assertEqual(points_received[-1], (20, 15, 3))
            finally:
                vol.close()
                dlg.close()

    def test_raw_slice_inference_if_model_available(self):
        model_path = r"E:\xsy\pythoncode\myproj\9skytech\model\beizuan-det_model_library\beizuan-det_model_v1\260917_beizuan-det_v1.onnx"
        if not os.path.exists(model_path):
            self.skipTest(f"ONNX model not found at {model_path}")

        from anylabeling.services.auto_labeling.yolo_ultralytics import YOLOUltralytics

        config = {
            "type": "yolo_ultralytics",
            "name": "beizuan_test",
            "display_name": "Beizuan Test",
            "model_path": model_path,
            "task": "detect",
            "conf_thres": 0.25,
            "iou_thres": 0.45,
        }
        model = YOLOUltralytics(config, lambda msg: None)

        test_files = [
            r"D:\raw_test\20260918141911407__pad5-3_X836_Y1177_Z119-215_ROI-300x300x97_彩色-5um.raw",
            r"D:\raw_test\20260918141911407__pad5-3_X836_Y1177_Z119-215_ROI-300x300x97-5um.raw",
        ]
        for f in test_files:
            if not os.path.exists(f):
                continue
            info = parse_raw_file_info(f)
            self.assertIsNotNone(info)
            vol = RawVolume(info, use_memmap=True)
            s0 = vol.get_axial_slice(0)
            qimg = slice_to_qimage(s0)
            result = model.predict_shapes(qimg, f)
            self.assertIsNotNone(result)
            self.assertGreater(len(result.shapes), 0)
            vol.close()


    def test_orthogonal_canvas_widgets(self):
        w, h, d = 40, 30, 10
        total_bytes = w * h * d
        arr = np.arange(total_bytes, dtype=np.uint8).reshape((d, h, w))

        # Test XZ canvas (below XY)
        xz_canvas = OrthogonalCanvasWidget(view_type="XZ")
        coronal = arr[:, 15, :]  # shape: (10, 40)
        xz_canvas.set_slice_data(coronal, cross_primary=20, cross_z=5, aspect_ratio_z=1.0)
        self.assertEqual(xz_canvas._img_w, w)
        self.assertEqual(xz_canvas._img_h, d)
        self.assertEqual(xz_canvas._cross_x, 20)
        self.assertEqual(xz_canvas._cross_z, 5)

        xz_clicks = []
        xz_canvas.point_clicked.connect(lambda x, z: xz_clicks.append((x, z)))
        xz_canvas.point_clicked.emit(25, 7)
        self.assertEqual(xz_clicks[-1], (25, 7))

        # Test YZ canvas (right of XY)
        yz_canvas = OrthogonalCanvasWidget(view_type="YZ")
        sagittal = np.swapaxes(arr[:, :, 20], 0, 1)  # shape: (30, 10)
        yz_canvas.set_slice_data(sagittal, cross_primary=12, cross_z=5, aspect_ratio_z=1.0)
        self.assertEqual(yz_canvas._img_w, d)
        self.assertEqual(yz_canvas._img_h, h)
        self.assertEqual(yz_canvas._cross_y, 12)
        self.assertEqual(yz_canvas._cross_z, 5)

        yz_clicks = []
        yz_canvas.point_clicked.connect(lambda y, z: yz_clicks.append((y, z)))
        yz_canvas.point_clicked.emit(18, 4)
        self.assertEqual(yz_clicks[-1], (18, 4))

    def test_corner_widget(self):
        corner = RawOrthoCornerWidget()
        corner.set_coordinates(100, 120, 35, 210, phys_z=150.5, unit="µm")
        self.assertIn("100", corner.coords_label.text())
        self.assertIn("120", corner.coords_label.text())
        self.assertIn("35", corner.coords_label.text())
        self.assertIn("150.5", corner.phys_z_label.text())
        self.assertIn("210", corner.val_label.text())

    def test_raw_decompose(self):
        import json
        from pathlib import Path
        from tools.raw_decompose import (
            clean_shape_to_standard_2d,
            decompose_single_raw,
            decompose_raw_directory,
        )

        # 1. Test clean_shape_to_standard_2d
        dirty_shape = {
            "label": "scratch",
            "points": [[10.0, 10.0], [20.0, 20.0]],
            "shape_type": "rectangle",
            "slice_index": 2,
            "z": 15.0,
            "other_data": {
                "slice_index": 2,
                "z": 15.0,
                "raw_info": "sample",
                "custom_prop": "important_value",
            },
        }
        cleaned = clean_shape_to_standard_2d(dirty_shape)
        self.assertEqual(cleaned["label"], "scratch")
        self.assertNotIn("slice_index", cleaned)
        self.assertNotIn("z", cleaned)
        self.assertIn("other_data", cleaned)
        self.assertEqual(cleaned["other_data"]["custom_prop"], "important_value")
        self.assertNotIn("slice_index", cleaned["other_data"])
        self.assertNotIn("z", cleaned["other_data"])

        # 2. Test decomposition with synthetic volume
        w, h, d = 20, 20, 5
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            raw_path = temp_path / f"test_pad_X0_Y0_Z0_ROI-{w}x{h}x{d}-1um.raw"
            arr = np.ones((d, h, w), dtype=np.uint8) * 128
            with open(raw_path, "wb") as f:
                f.write(arr.tobytes())

            json_path = temp_path / f"{raw_path.stem}.json"
            sample_json = {
                "version": "0.4.19",
                "flags": {},
                "shapes": [
                    {
                        "label": "defect",
                        "points": [[2.0, 2.0], [5.0, 5.0]],
                        "shape_type": "rectangle",
                        "slice_index": 1,
                        "other_data": {"slice_index": 1},
                    },
                    {
                        "label": "bubble",
                        "points": [[10.0, 10.0], [12.0, 12.0]],
                        "shape_type": "rectangle",
                        "slice_index": 3,
                        "other_data": {"slice_index": 3},
                    },
                ],
                "imagePath": raw_path.name,
                "imageData": None,
                "imageHeight": h,
                "imageWidth": w,
            }
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(sample_json, f)

            # Test JPG decomposition (default)
            out_dir = temp_path.parent / f"{temp_path.name}_slices"
            count = decompose_single_raw(raw_path, json_path, out_dir, only_annotated=True, img_format="jpg")
            self.assertEqual(count, 2)
            self.assertTrue((out_dir / f"{raw_path.stem}_slice0001.jpg").exists())
            self.assertTrue((out_dir / f"{raw_path.stem}_slice0001.json").exists())
            self.assertTrue((out_dir / f"{raw_path.stem}_slice0003.jpg").exists())
            self.assertTrue((out_dir / f"{raw_path.stem}_slice0003.json").exists())
            self.assertFalse((out_dir / f"{raw_path.stem}_slice0000.jpg").exists())

            # Verify exported slice JSON structure
            with open(out_dir / f"{raw_path.stem}_slice0001.json", "r", encoding="utf-8") as f:
                s1_data = json.load(f)
            self.assertEqual(s1_data["imagePath"], f"{raw_path.stem}_slice0001.jpg")
            self.assertEqual(s1_data["imageWidth"], w)
            self.assertEqual(s1_data["imageHeight"], h)
            self.assertEqual(len(s1_data["shapes"]), 1)
            self.assertEqual(s1_data["shapes"][0]["label"], "defect")
            self.assertNotIn("slice_index", s1_data["shapes"][0])

            # Test PNG decomposition via directory decomposer
            png_out = temp_path.parent / f"{temp_path.name}_png_slices"
            total, final_dir = decompose_raw_directory(temp_path, output_dir=png_out, only_annotated=True, img_format="png")
            self.assertEqual(total, 2)
            self.assertTrue((png_out / f"{raw_path.stem}_slice0001.png").exists())

            # Test BMP decomposition
            bmp_out = temp_path.parent / f"{temp_path.name}_bmp_slices"
            b_cnt = decompose_single_raw(raw_path, json_path, bmp_out, only_annotated=True, img_format="bmp")
            self.assertEqual(b_cnt, 2)
            self.assertTrue((bmp_out / f"{raw_path.stem}_slice0001.bmp").exists())


if __name__ == "__main__":
    unittest.main()

