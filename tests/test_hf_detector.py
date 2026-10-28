"""Offline Hugging Face inference, abstentions, CPU/GPU parity and UI state."""
import _paths  # noqa: F401  (puts src/ and scripts/ on sys.path)
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from PIL import Image
from PyQt5.QtWidgets import QApplication, QMessageBox
from PyQt5.QtTest import QTest
from satsr.models.hf_landcover import HuggingFaceLandCoverEngine, EnsembleLandCoverEngine
from evaluate_hf_accuracy import metrics
from gui_utils import load_gui, wait_jobs

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT/'checkpoints/demo/hf_segformer'


class HuggingFaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.engine = HuggingFaceLandCoverEngine(FOLDER)

    def test_offline_cpu_inference_shapes_finite_scores_and_validation(self):
        rgb = np.asarray(Image.open(ROOT/'tests/fixtures/formats/scene_rgb.png').convert('RGB'))
        p, dry = self.engine.predict_rgb(rgb, True)
        self.assertEqual(p.shape, (6, *rgb.shape[:2]))
        self.assertTrue(np.isfinite(p).all())
        self.assertTrue((p.sum(0) <= 1.00001).all())
        self.assertTrue((dry[3] == 0).all())
        for bad in [rgb.astype(float), np.zeros((1, 1025, 3), np.uint8), rgb[:, :, :2]]:
            with self.assertRaises(ValueError):
                self.engine.predict_rgb(bad)

    def test_rgb_normalization_aspect_ratio_and_background_abstention(self):
        class Capture(torch.nn.Module):
            def forward(self, x):
                self.seen = x.detach().cpu().numpy()
                out = torch.zeros((1, 9, 128, 128), device=x.device)
                out[:, 0] = 100  # Explicit unknown, not a building or bare land.
                return out
        model = Capture()
        rgb = np.full((3, 7, 3), [20, 90, 210], np.uint8)
        image = np.full((3, 7, 4), 1000, np.uint16)
        with patch.object(self.engine, 'model', model):
            result = self.engine.segment(image, rgb=rgb, pixel_size_m=None, output_shape=(6, 14))
        expected = (np.array([20, 90, 210])/255-[.485, .456, .406])/[.229, .224, .225]
        np.testing.assert_allclose(model.seen[0, :, 0, 0], expected, atol=1e-6)
        self.assertEqual(model.seen.shape, (1, 3, 512, 512))
        self.assertTrue((result['class_map'] == 255).all())
        self.assertTrue(np.isnan(result['confidence_map']).all())
        self.assertEqual(result['abstained_native_pixels'], 21)
        self.assertEqual(sum(v['pixel_count'] for v in result['statistics'].values()), 0)

    def test_checksum_or_metadata_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            folder = Path(td)
            meta = json.loads((FOLDER/'source.json').read_text())
            (folder/'model.ts').write_bytes(b'corrupted')
            (folder/'source.json').write_text(json.dumps(meta))
            with self.assertRaisesRegex(ValueError, 'corrupted'):
                HuggingFaceLandCoverEngine(folder)

    def test_ensemble_averages_predictions_and_dry_scores(self):
        engine = EnsembleLandCoverEngine(ROOT/'checkpoints/demo/landcover_real.pt', FOLDER)
        a = np.full((6, 3, 7), 1/6, np.float32)
        b = a.copy(); b[0] += .1; b[1] -= .1
        with patch.object(engine.sparse, 'predict_rgb', return_value=(a, a)), \
             patch.object(engine.hf, 'predict_rgb', return_value=(b, b)):
            result = engine.predict_rgb(np.zeros((3, 7, 3), np.uint8), True)
        for value in result:
            np.testing.assert_allclose(value, (a+b)/2)

    def test_audit_counts_abstentions_as_errors(self):
        matrix = np.zeros((6, 7), np.int64)
        matrix[0, 0] = 80; matrix[0, 6] = 20
        report = metrics(matrix)
        self.assertEqual(report['pixel_accuracy'], .8)
        self.assertEqual(report['abstained_pixels'], 20)
        self.assertEqual(report['per_class']['Buildings']['iou'], .8)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_exported_model_cpu_gpu_score_parity(self):
        rgb = np.random.default_rng(8).integers(0, 256, (97, 151, 3), dtype=np.uint8)
        gpu = HuggingFaceLandCoverEngine(FOLDER, 'cuda')
        cpu_scores, gpu_scores = self.engine.predict_rgb(rgb), gpu.predict_rgb(rgb)
        # CPU/GPU convolution and attention reductions have different rounding.
        np.testing.assert_allclose(cpu_scores, gpu_scores, atol=5e-4, rtol=2e-4)
        self.assertGreater((cpu_scores.argmax(0) == gpu_scores.argmax(0)).mean(), .999)


class HuggingFaceGuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        cls.gui = load_gui('hf_gui', 'app.py')

    def test_select_hf_detect_downloads_then_switch_and_open_another_image(self):
        window = self.gui.MainWindow(); errors = []
        try:
            with patch.object(QMessageBox, 'critical', side_effect=lambda *args:errors.append(args[-1])):
                window.combo_detector.setCurrentIndex(2)
                window.load_file(str(ROOT/'demo_images/sentinel2_water_pavana_2024-02-28.tif')); wait_jobs(window)
                window.btn_run_all.click(); wait_jobs(window, 60)
                self.assertFalse(errors, errors)
                self.assertIn('Hugging Face', window.seg['method'])
                self.assertEqual(window._report_data()['device'], 'cpu')
                self.assertTrue(window.btn_save_all.isEnabled())
                window.combo_detector.setCurrentIndex(3)
                self.assertIsNone(window.seg)
                self.assertFalse(window.btn_save_all.isEnabled())
                window.load_file(str(ROOT/'tests/fixtures/formats/scene_rgb.png')); wait_jobs(window)
                window.btn_run_all.click(); wait_jobs(window, 60)
                self.assertIn('SparseMask', window.seg['method'])
                self.assertFalse(window.seg['measured_nir_used'])
                self.assertTrue(window.btn_choose.isEnabled())
                self.assertFalse(errors, errors)
        finally:
            wait_jobs(window, 60); window.close(); window.deleteLater(); QTest.qWait(10)


if __name__ == '__main__':
    unittest.main()
