import _paths  # noqa: F401  (puts src/ and scripts/ on sys.path)
import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
import rasterio
from rasterio.transform import Affine
from rasterio.crs import CRS
from PIL import Image
from satsr.data import SatelliteDataset, synthetic_scene
from satsr.models.rcan import L1BSRSR
from satsr.models.unet_segmentation import LandCoverSegmenter, SatelliteSegmentationNet
from satsr.models.evaluation_metrics import (reconstruction_metrics, compute_full_sr_evaluation,
                                       segmentation_metrics, calculate_sam)
from satsr.geo_io import (BandData, load_any_user_image, load_band_stack, pixel_area_m2,
                    write_geotiff, write_single_band_geotiff)

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(2)


class PipelineTests(unittest.TestCase):
    def test_reference_metrics_and_bad_shapes(self):
        image = np.full((9, 11, 4), 1000, np.uint16)
        self.assertEqual(reconstruction_metrics(image, image), dict(psnr_db=100., ssim=1., sam_degrees=0.))
        self.assertEqual(calculate_sam(image/10000, image/10000), (0., 0.))
        zeros = np.zeros_like(image)
        self.assertEqual(calculate_sam(zeros, zeros), (0., 0.))
        self.assertEqual(calculate_sam(zeros, image)[0], 90.)
        with self.assertRaises(ValueError):
            reconstruction_metrics(image, image[:3])
        lr, hr, _ = synthetic_scene(991)
        metrics = compute_full_sr_evaluation(lr, L1BSRSR.bicubic_upscale(lr), hr)
        self.assertFalse(metrics["is_ground_truth"])
        self.assertTrue(compute_full_sr_evaluation(lr, hr, hr, hr)["is_ground_truth"])

    def test_segmentation_scores_and_ignore_label(self):
        truth = np.array([[0, 0], [1, 255]], np.uint8)
        pred = np.array([[0, 1], [1, 4]], np.uint8)
        metrics = segmentation_metrics(pred, truth)
        self.assertAlmostEqual(metrics["pixel_accuracy"], 2/3)
        self.assertEqual(metrics["per_class_iou"][:2], [.5, .5])
        self.assertIsNone(metrics["per_class_iou"][2])

    def test_baselines_and_tiny_odd_images(self):
        image = np.full((1, 3, 4), 2000, np.uint16)
        engine = L1BSRSR(device="cpu")
        self.assertFalse(engine.weights_loaded)
        self.assertEqual(engine.super_resolve(image).shape, (2, 6, 4))
        baseline = LandCoverSegmenter(device="cpu")
        result = baseline.segment(image, pixel_size_m=None, apply_morphology=False)
        self.assertEqual(result["method"], "spectral rules")
        self.assertIsNone(result["statistics"]["Water"]["area_hectares"])
        self.assertTrue(np.allclose(result["probabilities"].sum(0), 1))
        with self.assertRaises(ValueError):
            engine.super_resolve(image[..., :3])

    def test_trained_models_on_unseen_scene(self):
        # The synthetic generator's own degradation is only beaten by the synthetic-trained RCAN.
        # The app's rcan.pt is trained on real SEN2VENµS pairs (scored in evaluate_sen2venus.py).
        app_sr = L1BSRSR(ROOT/"checkpoints/demo/rcan.pt", "cpu", tile_size=32)
        self.assertIn("SEN2VENµS", app_sr.provenance)
        sr = L1BSRSR(ROOT/"checkpoints/demo/rcan_synthetic.pt", "cpu", tile_size=32)
        seg = LandCoverSegmenter("cpu", ROOT/"checkpoints/demo/unet.pt", tile_size=32)
        lr, hr, mask = synthetic_scene(300000, 64)
        prediction = sr.super_resolve(lr)
        result = seg.segment(prediction, apply_morphology=False)
        learned = reconstruction_metrics(prediction, hr)
        baseline = reconstruction_metrics(sr.bicubic_upscale(lr), hr)
        self.assertGreater(learned["psnr_db"], baseline["psnr_db"])
        self.assertGreater(segmentation_metrics(result["class_map"], mask)["mean_iou"], .7)
        self.assertTrue(sr.weights_loaded and seg.weights_loaded)
        tiny = seg.segment(hr[:3, :5], apply_morphology=False)
        self.assertEqual(tiny["class_map"].shape, (3, 5))
        valid = np.ones(hr.shape[:2], bool); valid[0] = False
        result = seg.segment(hr, valid_mask=valid, apply_morphology=False)
        self.assertTrue((result["class_map"][0] == 255).all())
        self.assertTrue(np.isnan(result["confidence_map"][0]).all())
        self.assertEqual(sum(v["pixel_count"] for v in result["statistics"].values()), int(valid.sum()))

    def test_checkpoint_rejection(self):
        with tempfile.TemporaryDirectory() as folder:
            bad = Path(folder)/"bad.pt"
            torch.save({"task":"segmentation"}, bad)
            with self.assertRaises(ValueError):
                L1BSRSR(bad, "cpu")
            state = torch.load(ROOT/"checkpoints/demo/rcan.pt", weights_only=True)
            state["state_dict"].pop(next(iter(state["state_dict"])))
            torch.save(state, bad)
            with self.assertRaises(RuntimeError):
                L1BSRSR(bad, "cpu")

    def test_real_dataset_format(self):
        with tempfile.TemporaryDirectory() as folder:
            split = Path(folder)/"train"; split.mkdir()
            lr, hr, mask = synthetic_scene(444)
            np.savez(split/"scene.npz", lr=lr, hr=hr, mask=mask)
            dataset = SatelliteDataset(folder)
            x, target, features, labels = dataset[0]
            self.assertEqual(x.shape, (4, 32, 32))
            self.assertEqual(features.shape, (8, 64, 64))
            self.assertEqual(labels.dtype, torch.int64)
            np.savez(split/"scene.npz", lr=lr, hr=hr, mask=np.full_like(mask, 7))
            with self.assertRaises(ValueError):
                dataset[0]

    def test_georeferencing_and_exports(self):
        with tempfile.TemporaryDirectory() as folder:
            image = np.full((10, 12, 4), 1000, np.uint16)
            ref = BandData("test", image[..., 0], Affine(10, 2, 600000, 1, -10, 5700000), CRS.from_epsg(32634), 12, 10)
            output = L1BSRSR.bicubic_upscale(image)
            self.assertAlmostEqual(pixel_area_m2(ref, output.shape), 25.5)
            path = Path(folder)/"enhanced.tif"
            write_geotiff(path, output, ref)
            with rasterio.open(path) as src:
                self.assertEqual(src.transform, ref.transform*Affine.scale(.5, .5))
                self.assertEqual(src.crs, ref.crs)
                self.assertEqual(src.descriptions, ("B02", "B03", "B04", "B08"))
            ids = np.zeros(output.shape[:2], np.uint8)
            write_single_band_geotiff(Path(folder)/"map.tif", ids, ref)
            with rasterio.open(Path(folder)/"map.tif") as src:
                self.assertEqual(src.nodata, 255)
                self.assertTrue((src.read_masks(1) > 0).all())
            bands, loaded, _, _ = load_any_user_image(str(path), max_side=12)
            self.assertEqual(loaded.shape, image.shape)
            self.assertEqual(bands["B02"].transform, ref.transform)
            png = Path(folder)/"photo.png"
            Image.fromarray(np.full((7, 11, 3), 100, np.uint8)).save(png)
            bands, loaded, _, _ = load_any_user_image(str(png))
            self.assertIsNone(bands["B02"].crs)
            self.assertIsNone(pixel_area_m2(bands["B02"], loaded.shape))
            with self.assertRaises(ValueError):
                write_geotiff(Path(folder)/"fake.tif", loaded, bands["B02"])
            rgba = Path(folder)/"rgba.tif"
            Image.fromarray(np.full((7, 11, 4), 128, np.uint8), "RGBA").save(rgba)
            bands, _, _, meta = load_any_user_image(str(rgba))
            self.assertIn("estimated NIR", meta["source_kind"])

    def test_band_alignment_rejection(self):
        with tempfile.TemporaryDirectory() as folder:
            paths = {}
            for i, key in enumerate(("B02", "B03", "B04", "B08")):
                paths[key] = Path(folder)/(key+".tif")
                with rasterio.open(paths[key], "w", driver="GTiff", height=8, width=8, count=1,
                                   dtype="uint16", crs="EPSG:32634",
                                   transform=Affine(10, 0, 600000+i, 0, -10, 5700000)) as dst:
                    dst.write(np.ones((8, 8), dtype=np.uint16), 1)
            with self.assertRaises(ValueError):
                load_band_stack(paths)


class GuiTests(unittest.TestCase):
    def test_worker_pipeline_and_reset(self):
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
        spec = importlib.util.spec_from_file_location("gui", ROOT/"app.py")
        gui = importlib.util.module_from_spec(spec); spec.loader.exec_module(gui)
        from PyQt5.QtCore import QEventLoop, QTimer
        app = gui.QApplication.instance() or gui.QApplication([])
        window = gui.MainWindow()
        window.device = torch.device("cpu")
        errors = []
        original_critical = gui.QMessageBox.critical
        self.addCleanup(setattr, gui.QMessageBox, "critical", original_critical)
        gui.QMessageBox.critical = lambda *args: errors.append(args[-1])
        window.load_real_sample()
        from gui_utils import wait_jobs
        wait_jobs(window)
        self.assertTrue(window.btn_enhance.isEnabled())
        window.on_detect()
        self.assertTrue(window._is_busy)
        loop = QEventLoop()
        timer = QTimer()
        timer.timeout.connect(lambda: loop.quit() if (not window._workers and not window._is_busy) else None)
        timer.start(20)
        QTimer.singleShot(30000, loop.quit)
        loop.exec_(); timer.stop()
        self.assertFalse(window._is_busy)
        self.assertFalse(errors, errors)
        self.assertIsNotNone(window.seg)
        self.assertTrue(window.btn_save_map.isEnabled())
        window._show_view("confidence")
        self.assertEqual(window._current_view(), "confidence")
        window.load_real_sample()
        wait_jobs(window)
        self.assertIsNone(window.seg)
        self.assertFalse(window.btn_save_map.isEnabled())
        def failed_operation():
            raise ValueError("Deliberate worker failure")
        window._start_job(failed_operation, lambda _: None, "Testing failure recovery")
        failure_loop = QEventLoop()
        failure_timer = QTimer()
        failure_timer.timeout.connect(lambda: failure_loop.quit() if not window._workers else None)
        failure_timer.start(20); QTimer.singleShot(5000, failure_loop.quit)
        failure_loop.exec_(); failure_timer.stop()
        self.assertFalse(window._is_busy)
        self.assertTrue(window.btn_enhance.isEnabled())
        self.assertIn("Deliberate worker failure", errors)
        window.close(); app.processEvents()


if __name__ == "__main__":
    unittest.main()
