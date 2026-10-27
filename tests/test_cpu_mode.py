"""CPU-only operation and transactional device switching in the full GUI."""
import _paths  # noqa: F401  (puts src/ and scripts/ on sys.path)
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
import torch  # Must precede Qt on Windows.
from PyQt5.QtWidgets import QApplication, QMessageBox
from PyQt5.QtTest import QTest
from gui_utils import load_gui, wait_jobs
from satsr.runtime_support import configure_cpu_threads

ROOT = Path(__file__).resolve().parents[1]
SCENE = ROOT/"demo_images"/"sentinel2_urban_delhi_2023-12-05.tif"


class CpuModeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        cls.gui = load_gui("cpu_mode_gui", "app.py")

    def setUp(self):
        self.windows = []
        self.errors = []
        self.messages = patch.object(QMessageBox, "critical", side_effect=lambda *args: self.errors.append(args[-1]))
        self.messages.start()

    def tearDown(self):
        for window in self.windows:
            wait_jobs(window, 60); window.close(); window.deleteLater()
        QTest.qWait(10); self.messages.stop()
        self.assertIsNone(QApplication.overrideCursor())

    def window(self):
        window = self.gui.MainWindow(); self.windows.append(window)
        return window

    def process(self, window):
        window.load_file(str(SCENE)); wait_jobs(window)
        window.btn_run_all.click(); wait_jobs(window, 60)
        self.assertFalse(self.errors, self.errors)

    def test_no_gpu_full_pipeline_never_initializes_cuda_and_can_choose_again(self):
        with patch.object(torch.cuda, "is_available", return_value=False), \
             patch.object(torch.cuda, "_lazy_init", side_effect=AssertionError("CPU touched CUDA")), \
             patch.object(torch.cuda, "get_device_name", side_effect=AssertionError("CPU queried GPU")):
            window = self.window()
            self.assertEqual(window.device.type, "cpu")
            self.assertTrue(window.chk_cpu.isChecked())
            self.assertFalse(window.chk_cpu.isEnabled())
            window.chk_cpu.setChecked(False)  # Programmatic attempts also stay on CPU.
            self.assertTrue(window.chk_cpu.isChecked())
            self.process(window)
            for engine in (window.sr_engine, window.seg_engine):
                self.assertTrue(engine.weights_loaded)
                self.assertEqual(next(engine.model.parameters()).device.type, "cpu")
            self.assertEqual(window._report_data()["device"], "cpu")
            self.assertTrue(window.btn_save_all.isEnabled())
            self.assertTrue(window.btn_choose.isEnabled())
            window.load_real_sample(); wait_jobs(window)
            self.assertIsNone(window.seg)
            self.assertTrue(window.btn_run_all.isEnabled())

    def test_cpu_is_default_even_if_cuda_is_present_and_busy_toggle_is_guarded(self):
        with patch.object(torch.cuda, "is_available", return_value=True):
            window = self.window()
            self.assertEqual(window.device.type, "cpu")
            self.assertTrue(window.chk_cpu.isChecked())
            self.assertTrue(window.chk_cpu.isEnabled())
            window._busy(True, "test")
            self.assertFalse(window.chk_cpu.isEnabled())
            window.chk_cpu.setChecked(False)
            self.assertTrue(window.chk_cpu.isChecked())
            self.assertEqual(window.device.type, "cpu")
            self.assertEqual(window._workers, [])
            window._busy(False)

    def test_failed_gpu_switch_preserves_cpu_models_input_results_and_downloads(self):
        with patch.object(torch.cuda, "is_available", return_value=True):
            window = self.window()
            self.process(window)
            previous = (window.sr_engine, window.seg_engine, window.input_bgrn, window.sr_bgrn, window.seg)
            with patch.object(self.gui, "L1BSRSR", return_value=window.sr_engine), \
                 patch.object(self.gui, "LandCoverSegmenter", side_effect=RuntimeError("GPU driver unavailable")):
                window.chk_cpu.setChecked(False); wait_jobs(window)
            self.assertEqual(len(self.errors), 1)
            self.assertIn("GPU driver unavailable", self.errors[0])
            self.assertTrue(window.chk_cpu.isChecked())
            self.assertEqual(window.device.type, "cpu")
            for actual, expected in zip((window.sr_engine, window.seg_engine, window.input_bgrn, window.sr_bgrn, window.seg), previous):
                self.assertIs(actual, expected)
            self.assertTrue(window.btn_save_all.isEnabled())
            self.assertIsNone(window._pending_device)
            self.assertIn("Previous results are preserved", window.lbl_status.text())

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA GPU unavailable")
    def test_gpu_cpu_switch_reloads_both_models_and_clears_stale_downloads(self):
        window = self.window()
        window.load_file(str(SCENE)); wait_jobs(window)
        original = window.input_bgrn
        initial_allocation = torch.cuda.memory_allocated()
        window.chk_cpu.setChecked(False); wait_jobs(window)
        self.assertEqual(window.device.type, "cuda")
        window.btn_run_all.click(); wait_jobs(window, 60)
        gpu_sr = window.sr_bgrn.copy()
        gpu_labels = window.seg["class_map"].copy()
        gpu_peak = torch.cuda.memory_allocated()
        self.assertEqual(window._report_data()["device"], "cuda")
        self.assertGreater(gpu_peak, initial_allocation)
        window.chk_cpu.setChecked(True); wait_jobs(window)
        self.assertEqual(window.device.type, "cpu")
        self.assertIs(window.input_bgrn, original)
        self.assertIsNone(window.sr_bgrn); self.assertIsNone(window.seg)
        self.assertFalse(window.btn_save_all.isEnabled())
        self.assertFalse(window.btn_save_geotiff.isEnabled())
        self.assertEqual(window._current_view(), "original")
        # GPU model weights are unloaded from CUDA memory; cuBLAS preserves execution pool
        self.assertLess(torch.cuda.memory_allocated(), gpu_peak - 20 * 1024**2)
        for engine in (window.sr_engine, window.seg_engine):
            self.assertEqual(next(engine.model.parameters()).device.type, "cpu")
        window.btn_run_all.click(); wait_jobs(window, 60)
        self.assertFalse(self.errors, self.errors)
        self.assertEqual(window._report_data()["device"], "cpu")
        self.assertLessEqual(np.abs(window.sr_bgrn.astype(int)-gpu_sr.astype(int)).max(), 2)
        self.assertGreater((window.seg["class_map"] == gpu_labels).mean(), .999)
        self.assertTrue(window.btn_save_all.isEnabled())

    def test_cpu_thread_limit_bounds_oversubscription_and_preserves_smaller_limit(self):
        with patch.object(torch, "get_num_threads", return_value=32), \
             patch("satsr.runtime_support.os.cpu_count", return_value=8), \
             patch.object(torch, "set_num_threads") as set_threads:
            self.assertEqual(configure_cpu_threads(), 4)
            set_threads.assert_called_once_with(4)
        with patch.object(torch, "get_num_threads", return_value=2), \
             patch("satsr.runtime_support.os.cpu_count", return_value=8), \
             patch.object(torch, "set_num_threads") as set_threads:
            self.assertEqual(configure_cpu_threads(), 2)
            set_threads.assert_not_called()
