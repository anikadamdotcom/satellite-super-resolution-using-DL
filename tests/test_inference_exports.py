import _paths  # noqa: F401  (puts src/ and scripts/ on sys.path)
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
import torch.nn.functional as F
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine
from PIL import Image
from satsr.data import synthetic_scene
from satsr.models.common import tiled_predict, read_checkpoint
from satsr.models.rcan import L1BSRSR
from satsr.models.unet_segmentation import LandCoverSegmenter
from satsr.onnx_engine import ONNXModel
from satsr.geo_io import (BandData, load_any_user_image, save_image, write_geotiff,
    write_single_band_geotiff, _atomic_write, resolve_output_path, save_jpeg_with_jgw)
from satsr.models.evaluation_metrics import (reconstruction_metrics, calculate_psnr, calculate_ssim,
    calculate_sam, calculate_land_cover_stats, segmentation_metrics)

ROOT = Path(__file__).resolve().parents[1]


class InferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.sr = L1BSRSR(ROOT/"checkpoints/demo/rcan.pt","cpu",tile_size=16)
        cls.onnx = ONNXModel(ROOT/"checkpoints/demo/rcan_model.onnx",tile_size=16)
        cls.seg = LandCoverSegmenter("cpu",ROOT/"checkpoints/demo/unet.pt",tile_size=16)

    def test_backend_parity_small_odd_tiny_and_multitile(self):
        rng = np.random.default_rng(12)
        for h,w in ((1,1),(1,13),(13,1),(3,5),(17,19),(64,80)):
            with self.subTest(shape=(h,w)):
                image = rng.integers(200,4200,(h,w,4),dtype=np.uint16)
                pytorch = self.sr.super_resolve(image); onnx = self.onnx.super_resolve(image)
                self.assertEqual(pytorch.shape,(h*2,w*2,4))
                self.assertLessEqual(np.max(np.abs(pytorch.astype(int)-onnx.astype(int))),1)
                result = self.seg.segment(pytorch,apply_morphology=False)
                self.assertTrue(np.isfinite(result["confidence_map"]).all())
                self.assertTrue(np.allclose(result["probabilities"].sum(0),1,atol=1e-6))

    def test_invalid_inputs_both_backends(self):
        cases = (np.zeros((0,2,4)),np.zeros((2,2,3)),np.full((2,2,4),np.nan),
                 np.full((2,2,4),-1),np.full((2,2,4),65536),np.zeros((2,2,4),complex))
        for image in cases:
            for engine in (self.sr,self.onnx):
                with self.subTest(shape=image.shape,engine=type(engine).__name__), self.assertRaises(ValueError):
                    engine.super_resolve(image)

    def test_tiles_preserve_pooling_origin_and_output_coordinates(self):
        class Pool(torch.nn.Module):
            def forward(self,x):
                return F.interpolate(F.avg_pool2d(x,8,8),scale_factor=8,mode="nearest")
        chw = np.arange(3*64*64,dtype=np.float32).reshape(3,64,64)
        full = Pool()(torch.from_numpy(chw)[None])[0].numpy()
        tiled = tiled_predict(Pool(),chw,"cpu",tile_size=13,halo=9,min_size=8,alignment=8)
        np.testing.assert_array_equal(full,tiled)
        class Repeat(torch.nn.Module):
            def forward(self,x):
                return F.interpolate(x,scale_factor=2,mode="nearest")
        actual = tiled_predict(Repeat(),chw,"cpu",scale=2,tile_size=13,halo=3)
        np.testing.assert_array_equal(actual,Repeat()(torch.from_numpy(chw)[None])[0].numpy())

    def test_nonfinite_model_output_and_bad_tiles(self):
        class Bad(torch.nn.Module):
            def forward(self,x):
                return x*float("nan")
        chw = np.ones((4,8,8),np.float32)
        with self.assertRaisesRegex(ValueError,"non-finite"):
            tiled_predict(Bad(),chw,"cpu")
        for size in (0,-1,1.5,True):
            with self.assertRaises(ValueError):
                tiled_predict(Bad(),chw,"cpu",tile_size=size)

    def test_nan_checkpoint_and_excessive_architecture_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)/"invalid.pt"
            state = torch.load(ROOT/"checkpoints/demo/rcan.pt",weights_only=True)
            state["state_dict"]["head.0.weight"][0,0,0,0] = float("nan"); torch.save(state,path)
            with self.assertRaisesRegex(ValueError,"non-finite"):
                read_checkpoint(path,"sr")
            state = torch.load(ROOT/"checkpoints/demo/rcan.pt",weights_only=True)
            state["model_config"]["n_feats"] = 100000; torch.save(state,path)
            with self.assertRaises(ValueError):
                read_checkpoint(path,"sr")

    @unittest.skipUnless(torch.cuda.is_available(),"CUDA unavailable")
    def test_cuda_matches_cpu_and_releases_transient_memory(self):
        gpu = L1BSRSR(ROOT/"checkpoints/demo/rcan.pt","cuda",tile_size=16)
        seg = LandCoverSegmenter("cuda",ROOT/"checkpoints/demo/unet.pt",tile_size=16)
        image,_,_ = synthetic_scene(80808,64)
        expected = self.sr.super_resolve(image)
        warm = gpu.super_resolve(image); seg.segment(warm,return_probabilities=False)
        torch.cuda.synchronize(); baseline = torch.cuda.memory_allocated()
        for _ in range(6):
            actual = gpu.super_resolve(image)
            result = seg.segment(actual,return_probabilities=False)
            self.assertTrue(np.isfinite(result["confidence_map"]).all())
            self.assertLessEqual(np.abs(actual.astype(int)-expected.astype(int)).max(),2)
        torch.cuda.synchronize()
        self.assertLessEqual(torch.cuda.memory_allocated()-baseline,1024*1024)


class ExportMetricTests(unittest.TestCase):
    def test_masked_metrics_ignore_invalid_values(self):
        a = np.full((12,12,4),1000,np.uint16); b = a.copy(); b[:6] = 60000
        valid = np.ones((12,12),bool); valid[:6] = False
        scores = reconstruction_metrics(a,b,valid_mask=valid)
        self.assertEqual(scores,dict(psnr_db=100.,ssim=1.,sam_degrees=0.))
        with self.assertRaises(ValueError):
            calculate_psnr(a,b,max_val=float("nan"))
        with self.assertRaises(ValueError):
            calculate_ssim(a,b,win_size=0)
        with self.assertRaises(ValueError):
            reconstruction_metrics(a,b,valid_mask=np.zeros((12,12),bool))

    def test_metrics_labels_confusion_and_tiny_spectra(self):
        a = np.full((3,3,4),1e-8)
        self.assertEqual(calculate_sam(a,a)[0],0.)
        for matrix in (np.ones((2,2)),np.full((5,5),-1),np.zeros((5,5),float)):
            with self.assertRaises(ValueError):
                segmentation_metrics(confusion=matrix)
        stats = calculate_land_cover_stats(np.array([[0,255],[1,255]],np.uint8),
                                          {0:dict(name="A",hex="#000000"),1:dict(name="B",hex="#ffffff")},pixel_size_m=None)
        self.assertEqual(stats["A"]["percentage"],50.)
        self.assertIsNone(stats["A"]["area_ha"])

    def test_atomic_write_failure_preserves_previous_file(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)/"output.txt"; path.write_text("original")
            def failed(temp):
                temp.write_text("partial")
                raise OSError("simulated write failure")
            with self.assertRaises(OSError):
                _atomic_write(path,failed)
            self.assertEqual(path.read_text(),"original")
            self.assertEqual(list(Path(td).iterdir()),[path])

    def test_bundle_failure_preserves_all_previous_files(self):
        from satsr.geo_io import _atomic_bundle
        with tempfile.TemporaryDirectory() as td:
            paths = [Path(td)/name for name in ("image.jpg","image.jgw","image.prj")]
            for path in paths:
                path.write_text("old")
            def failed(temp):
                raise OSError("worldfile failure")
            with self.assertRaises(OSError):
                _atomic_bundle([(paths[0],lambda temp:temp.write_text("new")),(paths[1],failed),
                                (paths[2],lambda temp:temp.write_text("new"))])
            self.assertTrue(all(path.read_text() == "old" for path in paths))

    def test_extensions_validation_and_transparent_png(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); rgb = np.full((5,7,3),120,np.uint8)
            path = save_image(root/"without_extension",rgb)
            self.assertEqual(path.suffix,".png")
            valid = np.ones((5,7),bool); valid[0] = False
            save_image(root/"transparent.png",rgb,valid)
            with Image.open(root/"transparent.png") as image:
                self.assertEqual(image.mode,"RGBA")
                self.assertTrue((np.asarray(image)[0,:,3] == 0).all())
            with self.assertRaises(ValueError):
                save_image(root/"fake.gif",rgb)
            with self.assertRaises(ValueError):
                save_image(root/"bad.png",rgb.astype(np.float32))

    def test_geotiff_roundtrip_mask_palette_and_existing_mask(self):
        with tempfile.TemporaryDirectory() as td:
            folder = Path(td)
            bands,image,_,_ = load_any_user_image(ROOT/"tests/fixtures/formats/satellite_nodata.tif")
            ref = bands["B02"]; prediction = L1BSRSR.bicubic_upscale(image)
            path = folder/"result.tif"
            path.with_name(path.name+".msk").write_bytes(b"stale mask")
            write_geotiff(path,prediction,ref)
            self.assertFalse(Path(str(path)+".msk").exists())
            with rasterio.open(path) as src:
                self.assertFalse(src.read_masks(1)[:20].any())
                self.assertTrue((src.read()[:,:20] == 0).all())
            ids = np.zeros(prediction.shape[:2],np.uint8)
            write_single_band_geotiff(folder/"classes.tif",ids,ref)
            with rasterio.open(folder/"classes.tif") as src:
                self.assertEqual(src.colormap(1)[0],(230,57,70,255))
                self.assertEqual(src.nodata,255)
                self.assertTrue((src.read(1)[:20] == 255).all())
                self.assertTrue((src.read(1)[20:] == 0).all())
            with self.assertRaises(ValueError):
                write_geotiff(folder/"float.tif",prediction.astype(float),ref)

    def test_worldfile_rotated_affine_uses_actual_dimensions(self):
        with tempfile.TemporaryDirectory() as td:
            ref = BandData("source",np.zeros((10,12),np.uint16),Affine(10,2,600000,1,-10,5000000),
                           CRS.from_epsg(32634),12,10)
            path = Path(td)/"result.jpg"
            save_jpeg_with_jgw(path,np.zeros((20,24,3),np.uint8),ref)
            np.testing.assert_allclose(np.loadtxt(path.with_suffix(".jgw")),[5,.5,1,-5,600003,4999997.75])
            self.assertIn("UTM",path.with_suffix(".prj").read_text(encoding="utf-8"))
