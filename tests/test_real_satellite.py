"""Offline provenance and full application workflows on actual Sentinel-2 crops."""
import _paths  # noqa: F401  (puts src/ and scripts/ on sys.path)
import os
os.environ.setdefault("QT_QPA_PLATFORM","offscreen")
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile
import numpy as np
import torch
import rasterio
from rasterio.transform import Affine
from rasterio.windows import Window
from PyQt5.QtWidgets import QApplication, QMessageBox
from PyQt5.QtTest import QTest
from satsr.geo_io import load_any_user_image, load_band_stack, output_valid_mask, IMAGE_EXTENSIONS
from download_sentinel_demo import asset_calibration, BANDS
from generate_test_fixtures import generate_test_fixtures
from gui_utils import load_gui, wait_jobs

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT/"demo_images"
RECORDS = json.loads((DEMO/"manifest.json").read_text(encoding="utf-8"))


class RealSourceTests(unittest.TestCase):
    def test_only_real_geotiffs_and_source_pixels_native_grid_and_provenance(self):
        self.assertEqual({r["category"] for r in RECORDS},{"urban","agriculture","vegetation","water","mixed"})
        files = {p.name for p in DEMO.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS}
        self.assertEqual(files,{r["file"] for r in RECORDS})
        for record in RECORDS:
            with self.subTest(file=record["file"]):
                path = DEMO/record["file"]
                self.assertEqual(path.suffix,".tif")
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),record["sha256"])
                item = json.loads((DEMO/record["stac_metadata_file"]).read_text(encoding="utf-8"))
                self.assertEqual(item["collection"],"sentinel-2-l2a")
                self.assertEqual(item["id"],record["stac_item_id"])
                self.assertEqual(item["properties"]["datetime"],record["acquisition_datetime"])
                with rasterio.open(path) as src:
                    self.assertEqual(src.count,4); self.assertEqual(src.res,(10.,10.))
                    self.assertEqual(src.descriptions,tuple(BANDS))
                    self.assertEqual(src.crs.to_string(),record["crs"])
                    native = src.read()
                    self.assertEqual(hashlib.sha256(native.tobytes()).hexdigest(),record["source_dn_sha256"])
                    asset = item["assets"]["blue"]
                    grid = Affine(*asset["proj:transform"][:6])
                    window = Window(**record["source_window"])
                    self.assertEqual(src.transform,rasterio.windows.transform(window,grid))
                    self.assertEqual(src.tags()["acquisition_datetime"],record["acquisition_datetime"])
                    self.assertEqual(src.scales,tuple(record["scales"]))
                    self.assertEqual(src.offsets,(0.,)*4)
                for key,asset in BANDS.items():
                    self.assertEqual(record["source_assets"][key],item["assets"][asset]["href"])
                    self.assertTrue(record["source_assets"][key].startswith("https://sentinel-cogs.s3."))
                bands,image,_,meta = load_any_user_image(path)
                # These COGs are already BOA corrected; applying -0.1 again destroys spectra.
                expected = np.moveaxis(native,0,-1).copy()
                expected[~bands["B02"].valid_mask] = 0
                np.testing.assert_array_equal(image,expected)
                self.assertEqual(int(bands["B02"].valid_mask.sum()),record["valid_pixels"])
                self.assertIn("real Sentinel-2",meta["source_kind"])
                self.assertEqual(bands["B02"].source_metadata["stac_item_id"],item["id"])
                with rasterio.open(DEMO/record["quality_file"]) as src:
                    scl = src.read(1)
                self.assertGreaterEqual(np.isin(scl,[2,4,5,6,7]).mean(),.98)

    def test_all_separate_measured_bands_match_four_band_scene(self):
        for record in RECORDS:
            with self.subTest(file=record["file"]):
                ref,image,_ = load_band_stack({key:DEMO/record["single_band_directory"]/(key+".tif") for key in BANDS})
                expected = load_any_user_image(DEMO/record["file"])[1]
                np.testing.assert_array_equal(image,expected)
                self.assertIn("real Sentinel-2",ref.source_kind)
                self.assertEqual(ref.source_metadata["stac_item_id"],record["stac_item_id"])

    def test_boa_offset_is_applied_once_and_invalid_calibration_is_rejected(self):
        asset = {"raster:bands":[dict(scale=.0001,offset=-.1)]}
        self.assertEqual(asset_calibration({"properties":{"earthsearch:boa_offset_applied":True}},asset),(.0001,0.))
        self.assertEqual(asset_calibration({"properties":{"earthsearch:boa_offset_applied":False}},asset),(.0001,-.1))
        with self.assertRaises(ValueError):
            asset_calibration({"properties":{}},{"raster:bands":[dict(scale=1,offset=0)]})
        with self.assertRaisesRegex(ValueError,"reserved for real"):
            generate_test_fixtures(DEMO)


class RealApplicationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        cls.main = load_gui("real_main_gui","app.py")
        cls.onnx = load_gui("real_onnx_gui","app_onnx.py")

    def setUp(self):
        self.errors = []
        self.messages = patch.object(QMessageBox,"critical",side_effect=lambda *args:self.errors.append(args[-1]))
        self.messages.start(); self.windows=[]

    def tearDown(self):
        for window in self.windows:
            wait_jobs(window,60); window.close(); window.deleteLater()
        QTest.qWait(10); self.messages.stop()
        self.assertIsNone(QApplication.overrideCursor())

    def test_all_real_scenes_select_load_enhance_detect_and_download_all_formats(self):
        window = self.main.MainWindow(); self.windows.append(window)
        self.assertTrue(window.chk_cpu.isChecked())
        self.assertEqual(window.device.type, "cpu")
        self.assertEqual(window.combo_sample.count(),7)
        self.assertEqual({Path(p).name for p in window._demo_paths.values()}, {r["file"] for r in RECORDS} | {"palu_8.tif"})
        with tempfile.TemporaryDirectory() as td:
            for record in RECORDS:
                with self.subTest(file=record["file"]):
                    index = next(i for i,p in window._demo_paths.items() if Path(p).name == record["file"])
                    window._on_sample(index); wait_jobs(window,60)
                    self.assertIsNone(window.sr_bgrn); self.assertIsNone(window.seg)
                    self.assertEqual(window.image_name,record["file"])
                    window.btn_run_all.click(); wait_jobs(window,60)
                    self.assertFalse(self.errors,self.errors)
                    self.assertEqual(window.sr_bgrn.shape,(512,512,4))
                    self.assertEqual(window.seg["class_map"].shape,(512,512))
                    valid = output_valid_mask(window.ref_band,window.sr_bgrn.shape)
                    self.assertTrue(np.isfinite(window.seg["confidence_map"][valid]).all())
                    self.assertTrue(np.isnan(window.seg["confidence_map"][~valid]).all())
                    self.assertTrue(np.isin(window.seg["class_map"][valid],[0,1,2,3,4,5]).all())
                    self.assertTrue((window.seg["class_map"][~valid] == 255).all())
                    self.assertTrue(window.btn_save_geotiff.isEnabled())
                    for key,_ in window.VIEWS:
                        window._show_view(key)
                    folder = Path(td)/record["category"]; folder.mkdir()
                    for method,name in (("on_save_image","enhanced.png"),("on_save_map","classes.png"),
                                        ("on_save_map","classes.tif"),("on_save_geotiff","enhanced.tif"),
                                        ("on_save_scores","scores.png"),("on_save_report","report.md"),
                                        ("on_save_all","results.zip")):
                        with patch.object(self.main,"save_dialog",return_value=str(folder/name)):
                            getattr(window,method)(); wait_jobs(window,60)
                        self.assertTrue((folder/name).is_file())
                    for name in ("enhanced.tif","classes.tif"):
                        with rasterio.open(folder/name) as src:
                            self.assertEqual(src.crs.to_string(),record["crs"])
                            self.assertEqual(src.res,(5.,5.))
                            self.assertEqual(src.bounds,rasterio.transform.array_bounds(256,256,window.ref_band.transform))
                            self.assertEqual(src.tags()["stac_item_id"],record["stac_item_id"])
                    with zipfile.ZipFile(folder/"results.zip") as archive:
                        report=json.loads(archive.read("report.json"))
                        self.assertEqual(report["device"], "cpu")
                        self.assertEqual(report["source_metadata"]["acquisition_datetime"],record["acquisition_datetime"])
                        self.assertFalse(report["metrics"]["is_ground_truth"])
                        self.assertIn("enhanced.tif",archive.namelist())
                        self.assertEqual(np.load(io.BytesIO(archive.read("class_ids.npy")),allow_pickle=False).shape,(512,512))
                    self.assertFalse(self.errors,self.errors)
            self.assertTrue(window.btn_choose.isEnabled())

    def test_onnx_process_and_geotiff_export_all_real_scenes_without_restart(self):
        window=self.onnx.MainWindow(); self.windows.append(window)
        with tempfile.TemporaryDirectory() as td:
            for record in RECORDS:
                with self.subTest(file=record["file"]):
                    window.load_file(str(DEMO/record["file"])); wait_jobs(window,60)
                    window.on_process(); wait_jobs(window,60)
                    self.assertFalse(self.errors,self.errors)
                    self.assertEqual(window.result_u16_hwc.shape,(512,512,4))
                    path=Path(td)/record["file"]
                    with patch.object(self.onnx,"save_dialog",return_value=str(path)):
                        window.on_save_tif()
                    with rasterio.open(path) as src:
                        self.assertEqual(src.res,(5.,5.))
                        self.assertEqual(src.tags()["stac_item_id"],record["stac_item_id"])
                    self.assertTrue(window.btn_open_image.isEnabled())
