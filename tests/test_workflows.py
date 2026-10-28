import _paths  # noqa: F401  (puts src/ and scripts/ on sys.path)
import os
os.environ.setdefault("QT_QPA_PLATFORM","offscreen")
from pathlib import Path
import gc
import tempfile
import tracemalloc
import time
import unittest
from unittest.mock import patch
import numpy as np
import torch
from PyQt5.QtWidgets import QApplication, QFileDialog, QMessageBox
from PyQt5.QtCore import Qt
from PyQt5.QtTest import QTest
from gui_utils import load_gui, wait_jobs
from satsr.geo_io import load_any_user_image, output_valid_mask

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT/"tests"/"fixtures"/"formats"


class MainWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        cls.gui = load_gui("audit_main_gui","app.py")

    def setUp(self):
        self.window = self.gui.MainWindow(); self.window.device = torch.device("cpu")
        self.errors = []
        self.messages = patch.object(QMessageBox,"critical",side_effect=lambda *args:self.errors.append(args[-1]))
        self.messages.start()

    def tearDown(self):
        wait_jobs(self.window)
        self.window.close(); self.window.deleteLater(); QTest.qWait(10)
        self.messages.stop()
        self.assertIsNone(QApplication.overrideCursor())

    def load(self,path):
        self.assertTrue(self.window.load_file(str(path))); wait_jobs(self.window)
        self.assertFalse(self.errors,self.errors)

    def test_every_demo_format_load_enhance_detect_and_replace(self):
        from satsr.geo_io import IMAGE_EXTENSIONS
        for path in sorted(p for p in DEMO.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS):
            with self.subTest(file=path.name):
                self.load(path)
                self.assertEqual(self.window.image_name,path.name)
                self.assertIsNone(self.window.sr_bgrn); self.assertIsNone(self.window.seg)
                self.window.on_detect(); wait_jobs(self.window)
                self.assertFalse(self.errors,self.errors)
                h,w = self.window.input_bgrn.shape[:2]
                self.assertEqual(self.window.sr_bgrn.shape,(2*h,2*w,4))
                self.assertEqual(self.window.seg["class_map"].shape,(2*h,2*w))
                self.assertTrue(self.window.btn_open.isEnabled())
                self.assertTrue(self.window.btn_save_map.isEnabled())
                self.assertNotIn("probabilities",self.window.seg)
                valid = output_valid_mask(self.window.ref_band,self.window.sr_bgrn.shape)
                self.assertTrue((self.window.sr_bgrn[~valid] == 0).all())
                self.assertTrue((self.window.seg["class_map"][~valid] == 255).all())
                for key,_ in self.window.VIEWS:
                    self.window._show_view(key)
                    self.assertEqual(self.window._current_view(),key)

    def test_chooser_reopens_and_loaded_preview_accepts_another_image(self):
        paths = [DEMO/"scene_rgb.png",DEMO/"scene_rgb.jpeg",DEMO/"satellite_4band.tif"]
        with patch.object(self.gui,"open_image_dialog",side_effect=[str(p) for p in paths]) as dialog:
            self.window.choose_image(); wait_jobs(self.window)
            self.window.on_detect(); wait_jobs(self.window)
            self.window.choose_image(); wait_jobs(self.window)
            self.window.on_enhance(); wait_jobs(self.window)
            self.window.show(); QTest.mouseClick(self.window.viewer,Qt.LeftButton); wait_jobs(self.window)
            self.assertEqual(self.window.image_name,paths[-1].name)
            self.assertIsNone(self.window.seg)
            self.assertEqual(dialog.call_count,3)
            for call in dialog.call_args_list:
                self.assertEqual(Path(call.args[2]),ROOT/"demo_images")

    def test_cancel_and_invalid_upload_preserve_valid_scene_and_allow_retry(self):
        self.load(DEMO/"scene_rgb.png")
        self.window.on_detect(); wait_jobs(self.window)
        old = self.window.sr_bgrn
        with patch.object(self.gui,"open_image_dialog",return_value=""):
            self.window.choose_image()
        self.assertIs(self.window.sr_bgrn,old)
        with tempfile.TemporaryDirectory() as td:
            bad = Path(td)/"bad.png"; bad.write_bytes(b"bad")
            self.window.load_file(str(bad)); wait_jobs(self.window)
        self.assertTrue(self.errors)
        self.assertIs(self.window.sr_bgrn,old)
        self.assertTrue(self.window.btn_open.isEnabled())
        self.errors.clear(); self.load(DEMO/"scene_rgb.webp")
        self.assertIsNone(self.window.sr_bgrn)

    def test_upload_during_job_is_rejected_and_callback_failure_recovers(self):
        self.load(DEMO/"scene_rgb.png")
        def operation():
            time.sleep(.05)
            return 1
        def failed_commit(_):
            raise ValueError("commit failed")
        self.window._start_job(operation,failed_commit,"test")
        self.assertFalse(self.window.load_file(str(DEMO/"scene_rgb.jpg")))
        self.assertFalse(self.window.btn_open.isEnabled())
        self.window.close()
        self.assertTrue(self.window._workers)
        wait_jobs(self.window)
        self.assertIn("commit failed",self.errors)
        self.assertTrue(self.window.btn_open.isEnabled())
        self.errors.clear(); self.load(DEMO/"scene_rgb.jpg")

    def test_exports_then_new_scene_has_no_old_downloads(self):
        from PIL import Image
        import rasterio
        self.load(DEMO/"satellite_nodata.tif")
        self.window.on_detect(); wait_jobs(self.window)
        with tempfile.TemporaryDirectory() as td:
            folder = Path(td)
            for method,name in (("on_save_image","enhanced.png"),("on_save_map","map.png"),
                                ("on_save_map","map.tif"),("on_save_geotiff","enhanced.tif"),
                                ("on_save_report","report.md")):
                with patch.object(self.gui,"save_dialog",return_value=str(folder/name)):
                    getattr(self.window,method)()
                self.assertTrue((folder/name).is_file())
            with Image.open(folder/"enhanced.png") as img:
                self.assertEqual(img.mode,"RGBA")
            with rasterio.open(folder/"map.tif") as src:
                self.assertEqual(src.nodata,255)
                self.assertEqual(src.transform.a,5.)
            report = (folder/"report.md").read_text(encoding="utf-8")
            self.assertIn("cpu",report); self.assertIn("satellite_nodata.tif",report)
            self.assertFalse(self.errors,self.errors)
            self.load(DEMO/"scene_rgb.png")
            self.assertFalse(self.window.btn_save_img.isEnabled())
            self.assertFalse(self.window.btn_save_map.isEnabled())
            with patch.object(self.gui,"save_dialog") as dialog:
                self.window.on_save_map(); self.window.on_save_image()
                dialog.assert_not_called()

    def test_model_swap_failure_preserves_models_and_outputs(self):
        self.load(DEMO/"scene_rgb.png"); self.window.on_detect(); wait_jobs(self.window)
        model, result = self.window.sr_engine,self.window.sr_bgrn
        with tempfile.TemporaryDirectory() as td, patch.object(self.gui,"folder_dialog",return_value=td):
            self.window.choose_models(); wait_jobs(self.window)
        self.assertTrue(self.errors)
        self.assertIs(self.window.sr_engine,model); self.assertIs(self.window.sr_bgrn,result)
        self.assertTrue(self.window.btn_open.isEnabled())

    def test_repeated_workflows_release_python_and_array_memory(self):
        self.load(DEMO/"scene_rgb.png"); self.window.on_detect(); wait_jobs(self.window)
        tracemalloc.start(); gc.collect(); baseline = tracemalloc.get_traced_memory()[0]
        for i in range(12):
            self.load(DEMO/("scene_rgb.png" if i%2 else "satellite_4band.tif"))
            self.window.on_detect(); wait_jobs(self.window)
            self.assertFalse(self.window._workers)
        gc.collect(); used = tracemalloc.get_traced_memory()[0]; tracemalloc.stop()
        self.assertLess(used-baseline,5*1024*1024)


class OnnxWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        cls.gui = load_gui("audit_onnx_gui","app_onnx.py")

    def setUp(self):
        self.window = self.gui.MainWindow(); self.errors = []
        self.messages = patch.object(QMessageBox,"critical",side_effect=lambda *args:self.errors.append(args[-1]))
        self.messages.start()

    def tearDown(self):
        wait_jobs(self.window); self.window.close(); self.window.deleteLater(); QTest.qWait(10)
        self.messages.stop()
        self.assertIsNone(QApplication.overrideCursor())

    def test_photo_raster_and_another_photo_without_restarting(self):
        for name in ("scene_rgb.png","satellite_nodata.tif","scene_grayscale16.png"):
            self.window.load_file(str(DEMO/name)); wait_jobs(self.window)
            self.assertIsNone(self.window.result_u16_hwc)
            self.window.on_process(); wait_jobs(self.window)
            self.assertFalse(self.errors,self.errors)
            self.assertIsNotNone(self.window.result_u16_hwc)
            self.assertTrue(self.window.btn_open_image.isEnabled())
            self.assertTrue(self.window.btn_save_image.isEnabled())

    def test_replacing_a_band_clears_outputs_and_model_change_clears_cache(self):
        self.window.load_file(str(DEMO/"satellite_4band.tif")); wait_jobs(self.window)
        self.window.on_process(); wait_jobs(self.window)
        with patch.object(self.gui,"open_image_dialog",return_value=str(DEMO/"satellite_bands"/"B02.tif")):
            self.window.choose_band("B02"); wait_jobs(self.window)
        self.assertIsNone(self.window.result_u16_hwc)
        self.assertFalse(self.window.btn_save_tif.isEnabled())
        self.window.model_edit.setText("invalid.onnx")
        self.assertIsNone(self.window.engine)
        self.window.on_process(); wait_jobs(self.window)
        self.assertTrue(self.errors)
        self.assertTrue(self.window.btn_open_image.isEnabled())

    def test_jpeg_worldfile_crs_and_ungeoreferenced_image_export(self):
        self.window.load_file(str(DEMO/"satellite_4band.tif")); wait_jobs(self.window)
        self.window.on_process(); wait_jobs(self.window)
        with tempfile.TemporaryDirectory() as td:
            folder = Path(td)
            with patch.object(self.gui,"save_dialog",return_value=str(folder/"test.jpg")):
                self.window.on_save_jpg()
            self.assertTrue((folder/"test.prj").is_file())
            world = np.loadtxt(folder/"test.jgw")
            np.testing.assert_allclose(world,[5,0,0,-5,635002.5,5784997.5])
            self.window.load_file(str(DEMO/"scene_rgb.png")); wait_jobs(self.window)
            self.window.on_process(); wait_jobs(self.window)
            self.assertFalse(self.window.btn_save_tif.isEnabled())
            with patch.object(self.gui,"save_dialog",return_value=str(folder/"photo.png")):
                self.window.on_save_image()
            self.assertTrue((folder/"photo.png").is_file())
