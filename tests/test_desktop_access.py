import _paths  # noqa: F401  (puts src/ and scripts/ on sys.path)
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
import zipfile
import numpy as np
import torch  # Must precede Qt on Windows.
from PIL import Image
from PyQt5.QtCore import QTimer
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication, QWidget, QFileDialog, QMessageBox, QLineEdit
from gui_utils import load_gui, wait_jobs
from satsr.gui_support import open_image_dialog, save_dialog
from satsr.geo_io import IMAGE_FILTER

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT/"demo_images"
FIXTURES = ROOT/"tests"/"fixtures"/"formats"


class DialogAccessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.parent = QWidget()

    def tearDown(self):
        self.parent.close(); self.parent.deleteLater(); QTest.qWait(10)

    def interact(self, callback):
        errors = []
        def run():
            dialog = QApplication.activeModalWidget()
            try:
                self.assertIsInstance(dialog, QFileDialog)
                dialog.finished.connect(timer.stop)
                callback(dialog)
            except Exception as error:
                errors.append(error)
                if dialog:
                    dialog.reject()
        QTimer.singleShot(80, run)
        # A failed dialog acceptance must report a test error instead of hanging CI.
        def guard():
            dialog = QApplication.activeModalWidget()
            if isinstance(dialog, QFileDialog):
                errors.append(AssertionError("Dialog interaction did not finish"))
                dialog.reject()
        timer = QTimer(self.parent)
        timer.setSingleShot(True); timer.timeout.connect(guard); timer.start(2500)
        return errors

    def test_real_chooser_opens_demo_with_shortcuts_then_remembers_selected_folder(self):
        with tempfile.TemporaryDirectory() as td:
            image = Path(td).resolve()/"next image.PNG"
            shutil.copyfile(FIXTURES/"scene_rgb.png", image)
            def select(dialog):
                self.assertEqual(Path(dialog.directory().absolutePath()), DEMO)
                self.assertTrue(dialog.testOption(QFileDialog.DontUseNativeDialog))
                self.assertIn(DEMO, [Path(u.toLocalFile()) for u in dialog.sidebarUrls()])
                self.assertGreaterEqual(len(dialog.nameFilters()), 7)
                dialog.findChild(QLineEdit,"fileNameEdit").setText(image.as_posix()); dialog.accept()
            errors = self.interact(select)
            path = open_image_dialog(self.parent, "Choose fixture", DEMO, IMAGE_FILTER)
            self.assertFalse(errors, errors); self.assertEqual(Path(path), image)
            def cancel(dialog):
                self.assertEqual(Path(dialog.directory().absolutePath()), Path(td).resolve())
                dialog.reject()
            errors = self.interact(cancel)
            self.assertEqual(open_image_dialog(self.parent, "Choose again", DEMO, IMAGE_FILTER), "")
            self.assertFalse(errors, errors)

    def test_missing_start_folder_falls_back_to_demo_and_cancellation_is_safe(self):
        def cancel(dialog):
            self.assertEqual(Path(dialog.directory().absolutePath()), DEMO)
            dialog.reject()
        errors = self.interact(cancel)
        self.assertEqual(open_image_dialog(self.parent, "Missing start", ROOT/"does-not-exist", IMAGE_FILTER), "")
        self.assertFalse(errors, errors)

    def test_real_save_dialog_defaults_to_downloads_and_remembers_directory_and_suffix(self):
        expected = Path.home()/"Downloads"
        if not expected.is_dir():
            expected = ROOT
        with tempfile.TemporaryDirectory() as td:
            target = Path(td).resolve()/"result"
            def select(dialog):
                self.assertEqual(Path(dialog.directory().absolutePath()), expected)
                self.assertEqual(dialog.acceptMode(), QFileDialog.AcceptSave)
                self.assertEqual(dialog.defaultSuffix(), "png")
                dialog.selectNameFilter("JPEG (*.jpg *.jpeg)")
                dialog.filterSelected.emit("JPEG (*.jpg *.jpeg)")
                self.assertEqual(dialog.defaultSuffix(), "jpg")
                dialog.findChild(QLineEdit,"fileNameEdit").setText(target.as_posix()); dialog.accept()
            errors = self.interact(select)
            self.assertEqual(Path(save_dialog(self.parent, "Save fixture", "result.png", "PNG (*.png);;JPEG (*.jpg *.jpeg)")), target.with_suffix(".jpg"))
            self.assertFalse(errors, errors)
            def cancel(dialog):
                self.assertEqual(Path(dialog.directory().absolutePath()), Path(td).resolve())
                dialog.reject()
            errors = self.interact(cancel)
            self.assertEqual(save_dialog(self.parent, "Save again", "next.png", "PNG (*.png)"), "")
            self.assertFalse(errors, errors)


class DownloadAccessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        cls.gui = load_gui("download_main", "app.py")

    def setUp(self):
        self.previous_style = self.app.styleSheet()
        self.app.setStyle("Fusion"); self.app.setStyleSheet(self.gui.STYLE)
        self.window = self.gui.MainWindow(); self.window.device = torch.device("cpu")
        self.errors = []
        self.messages = patch.object(QMessageBox, "critical", side_effect=lambda *args:self.errors.append(args[-1]))
        self.messages.start()

    def tearDown(self):
        wait_jobs(self.window)
        self.window.close(); self.window.deleteLater(); QTest.qWait(10)
        self.messages.stop()
        self.app.setStyleSheet(self.previous_style)
        self.assertIsNone(QApplication.overrideCursor())

    def load(self, name):
        self.window.load_file(str(FIXTURES/name)); wait_jobs(self.window)
        self.assertFalse(self.errors, self.errors)

    def process(self):
        self.window.btn_run_all.click(); wait_jobs(self.window)
        self.assertFalse(self.errors, self.errors)

    def test_download_controls_visible_at_default_and_minimum_window_sizes(self):
        self.window.show()
        buttons = [self.window.btn_save_img,self.window.btn_save_map,self.window.btn_save_geotiff,
                   self.window.btn_save_view,self.window.btn_save_scores,self.window.btn_save_report,
                   self.window.btn_save_all,self.window.btn_choose,self.window.btn_run_all,self.window.chk_cpu]
        for width,height in ((1280,800),(980,640)):
            self.window.resize(width,height); QTest.qWait(40)
            for button in buttons:
                with self.subTest(size=(width,height),button=button.text()):
                    self.assertTrue(button.isVisible())
                    self.assertTrue(self.window.rect().contains(button.mapTo(self.window,button.rect().topLeft())))
                    self.assertTrue(self.window.rect().contains(button.mapTo(self.window,button.rect().bottomRight())))
        self.load("scene_rgb.png")
        self.assertTrue(self.window.btn_save_view.isEnabled())
        self.assertFalse(self.window.btn_save_all.isEnabled())
        self.process()
        self.assertTrue(self.window.btn_save_all.isEnabled())
        self.assertFalse(self.window.btn_save_geotiff.isEnabled())
        self.assertTrue(self.window.btn_save_scores.isEnabled())
        self.load("scene_rgb.jpg")
        self.assertFalse(self.window.btn_save_all.isEnabled())
        self.assertFalse(self.window.btn_save_scores.isEnabled())

    def test_full_zip_download_has_geographic_outputs_and_valid_scores_and_report(self):
        self.load("satellite_nodata.tif"); self.process()
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)/"results.zip"
            with patch.object(self.gui,"save_dialog",return_value=str(path)):
                self.window.btn_save_all.click()
                self.assertTrue(self.window._is_busy)
                self.assertFalse(self.window.btn_choose.isEnabled())
                wait_jobs(self.window)
            self.assertFalse(self.errors, self.errors)
            with zipfile.ZipFile(path) as archive:
                names = set(archive.namelist())
                self.assertTrue({"original.png","enhanced.png","enhanced.jpg","land_cover.png","overlay.png",
                                 "scores.png","scores.npy","class_ids.npy","enhanced_bands.npy","report.md",
                                 "report.json","enhanced.tif","land_cover.tif","enhanced.jgw","enhanced.prj"} <= names)
                self.assertIsNone(archive.testzip())
                report = json.loads(archive.read("report.json"))
                self.assertEqual(set(report["files"]),names)
                self.assertIn("bicubic",report["metrics"]["reference"])
                labels = np.load(io.BytesIO(archive.read("class_ids.npy")),allow_pickle=False)
                scores = np.load(io.BytesIO(archive.read("scores.npy")),allow_pickle=False)
                self.assertTrue(np.isnan(scores[labels == 255]).all())
                with Image.open(io.BytesIO(archive.read("enhanced.png"))) as img:
                    self.assertEqual(img.mode,"RGBA")
            self.assertTrue(self.window.btn_choose.isEnabled())

    def test_photo_zip_and_current_compare_view_and_scores_download(self):
        self.load("scene_rgb.png"); self.process()
        with tempfile.TemporaryDirectory() as td:
            folder = Path(td)
            with patch.object(self.gui,"save_dialog",return_value=str(folder/"photo.zip")):
                self.window.on_save_all(); wait_jobs(self.window)
            with zipfile.ZipFile(folder/"photo.zip") as archive:
                self.assertNotIn("enhanced.tif",archive.namelist())
                self.assertNotIn("enhanced.prj",archive.namelist())
                self.assertIn("enhanced_bands.npy",archive.namelist())
                self.assertIn("difference_from_bicubic.png",archive.namelist())
                with Image.open(io.BytesIO(archive.read("enhanced.png"))) as img:
                    self.assertEqual(img.info.get("Derived"),"true")
                    self.assertIn("DERIVED IMAGE",img.info["Description"])
            self.window._show_view("compare")
            with patch.object(self.gui,"save_dialog",return_value=str(folder/"compare.png")):
                self.window.btn_save_view.click()
            h,w = self.window.sr_bgrn.shape[:2]
            with Image.open(folder/"compare.png") as img:
                # original | enhanced | change vs bicubic, plus a caption strip
                self.assertEqual(img.size[0],w*3+16)
                self.assertGreater(img.size[1],h)
                self.assertEqual(img.info.get("Derived"),"true")
            self.assertIn("Change vs bicubic",self.window._report_text())
            with patch.object(self.gui,"save_dialog",return_value=str(folder/"scores.png")):
                self.window.btn_save_scores.click()
            self.assertTrue((folder/"scores.png").is_file())
            self.assertFalse(self.errors, self.errors)

    def test_failed_zip_write_preserves_existing_download_and_controls_recover(self):
        self.load("scene_rgb.png"); self.process()
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)/"old.zip"; path.write_bytes(b"previous download")
            with patch.object(self.gui,"save_dialog",return_value=str(path)), patch("satsr.results_export.zipfile.ZipFile.write",side_effect=OSError("disk full")):
                self.window.on_save_all(); wait_jobs(self.window)
            self.assertEqual(path.read_bytes(),b"previous download")
            self.assertEqual(list(Path(td).iterdir()),[path])
            self.assertIn("disk full",self.errors)
            self.assertTrue(self.window.btn_save_all.isEnabled())
            self.assertTrue(self.window.btn_choose.isEnabled())
