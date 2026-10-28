"""Real-trained model provenance, preprocessing, spectral gating and app state."""
import _paths  # noqa: F401  (puts src/ and scripts/ on sys.path)
import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from PIL import Image
import rasterio
from PyQt5.QtWidgets import QApplication, QMessageBox
from PyQt5.QtTest import QTest
from satsr.geo_io import load_any_user_image, write_single_band_geotiff
from satsr.models.real_landcover import RealLandCoverEngine, MAPPING
from satsr.models.unet_segmentation import LandCoverSegmenter
from satsr.models.evaluation_metrics import segmentation_metrics
from gui_utils import load_gui, wait_jobs

ROOT=Path(__file__).resolve().parents[1]
WEIGHTS=ROOT/'checkpoints/demo/landcover_real.pt'
SCENE=ROOT/'demo_images/sentinel2_water_pavana_2024-02-28.tif'


class RealDetectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.engine=RealLandCoverEngine(WEIGHTS)

    def test_checkpoint_source_and_real_annotated_road_building_water_predictions(self):
        meta=json.loads((ROOT/'checkpoints/demo/landcover_real_source.json').read_text())
        self.assertEqual(hashlib.sha256(WEIGHTS.read_bytes()).hexdigest(),meta['checkpoint_sha256'])
        self.assertIn('OpenEarthMap',self.engine.provenance)
        fixture=ROOT/'tests/fixtures/real_landcover'
        source=np.asarray(Image.open(fixture/'labels/houston_16.tif'))
        rgb=np.asarray(Image.open(fixture/'images/houston_16.tif').convert('RGB'))
        pred=self.engine.predict_rgb(rgb).argmax(0).astype(np.uint8)
        truth=np.full(source.shape,255,np.uint8)
        for original,target in enumerate(MAPPING,1):
            truth[source==original]=target
        metric=segmentation_metrics(pred,truth,num_classes=6)
        # A regression on known upstream examples, not independent accuracy evidence.
        self.assertGreater(metric['pixel_accuracy'],.75)
        for cid,minimum in [(0,.6),(1,.5),(3,.8)]:
            self.assertGreater(metric['per_class_iou'][cid],minimum)

    def test_rgb_channel_order_normalization_and_non_square_padding(self):
        class Capture(torch.nn.Module):
            def __init__(self):
                super().__init__();self.seen=None
            def forward(self,x):
                self.seen=x.clone()
                return torch.zeros((1,8,8,8),device=x.device)
        rgb=np.full((3,7,3),[20,90,210],np.uint8)
        model=Capture()
        with patch.object(self.engine,'model',model):
            p=self.engine.predict_rgb(rgb)
        expected=(np.array([210,90,20])/255-np.array([.4325,.4483,.3879]))/np.array([.0195,.0169,.0179])
        np.testing.assert_allclose(model.seen[0,:,0,0].numpy(),expected,rtol=1e-5)
        self.assertEqual(tuple(model.seen.shape),(1,3,1024,1024))
        self.assertEqual(p.shape,(6,3,7))
        np.testing.assert_allclose(p.sum(0),1,atol=1e-6)
        self.assertAlmostEqual(float(p[2,0,0]),3/8)
        self.assertAlmostEqual(float(p[5,0,0]),1/8)

    def test_measured_nir_only_corrections_and_invalid_pixels_output_replication(self):
        image=np.full((2,3,4),[600,800,400,4200],np.uint16)
        image[0,0]=[800,800,600,200]  # measured-water spectrum
        valid=np.ones((2,3),bool);valid[1,2]=False
        p=np.full((6,2,3),.01,np.float32);p[1]=.95
        dry=p.copy();dry/=dry.sum(0)
        def predict(rgb,return_nonwater=False):
            return (p.copy(),dry.copy()) if return_nonwater else p.copy()
        with patch.object(self.engine,'predict_rgb',side_effect=predict):
            rgb_only=self.engine.segment(image,measured_nir=False,pixel_size_m=None)
            physical=self.engine.segment(image,measured_nir=True,valid_mask=valid,
                output_shape=(4,6),pixel_size_m=None,pixel_area_m2=25.)
        self.assertTrue((rgb_only['class_map']==1).all())
        self.assertFalse(rgb_only['measured_nir_used'])
        self.assertEqual(physical['class_map'][0,0],3)
        self.assertEqual(physical['class_map'][2,0],2)
        self.assertTrue((physical['class_map'][2:,4:]==255).all())
        self.assertTrue(np.isnan(physical['confidence_map'][2:,4:]).all())
        self.assertEqual(sum(v['pixel_count'] for v in physical['statistics'].values()),20)
        self.assertEqual(physical['prediction_input_shape'],[2,3])
        np.testing.assert_allclose(physical['probabilities'][:,:2,:2].sum(0),1,atol=1e-6)

    def test_tiling_covers_every_pixel_and_matches_constant_full_image(self):
        class Fixed(torch.nn.Module):
            def forward(self,x):
                out=torch.zeros((1,8,4,4),device=x.device);out[:,3]=2.
                return out
        rgb=np.full((17,23,3),120,np.uint8)
        with patch.object(self.engine,'model',Fixed()):
            full=self.engine.predict_rgb(rgb)
            with patch.object(self.engine,'tile_size',9):
                tiled=self.engine.predict_rgb(rgb)
        np.testing.assert_allclose(full,tiled,atol=1e-6)
        self.assertTrue(np.isfinite(tiled).all())
        self.assertTrue((tiled.argmax(0)==1).all())

    def test_rejects_incompatible_metadata_nonfinite_weights_and_bad_input(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'bad.pt'
            ck=torch.load(WEIGHTS,weights_only=True)
            ck['channel_order']='RGB';torch.save(ck,path)
            with self.assertRaisesRegex(ValueError,'metadata'):
                RealLandCoverEngine(path)
            ck['channel_order']='BGR';next(v for v in ck['state_dict'].values() if v.dtype.is_floating_point).view(-1)[0]=float('nan')
            torch.save(ck,path)
            with self.assertRaisesRegex(ValueError,'non-finite'):
                RealLandCoverEngine(path)
        for rgb in [np.zeros((0,3,3),np.uint8),np.zeros((3,3,4),np.uint8),np.zeros((3,3,3),np.float32)]:
            with self.assertRaises(ValueError):
                self.engine.predict_rgb(rgb)
        with self.assertRaises(ValueError):
            self.engine.segment(np.full((2,2,4),np.nan))

    def test_class_five_geotiff_has_correct_palette_and_name(self):
        bands,image,_,_=load_any_user_image(SCENE)
        labels=np.full(image.shape[:2],5,np.uint8)
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'developed.tif'
            write_single_band_geotiff(path,labels,bands['B02'])
            with rasterio.open(path) as src:
                self.assertEqual(src.colormap(1)[5],(164,139,191,255))
                self.assertIn('5 Developed space',src.tags()['class_names'])
                self.assertEqual(src.crs,bands['B02'].crs)


class RealDetectorGuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app=QApplication.instance() or QApplication([])
        cls.gui=load_gui('real_detector_gui','app.py')

    def test_default_native_detection_model_change_clears_map_keeps_enhancement_and_reopens(self):
        window=self.gui.MainWindow();errors=[]
        try:
            with patch.object(QMessageBox,'critical',side_effect=lambda *args:errors.append(args[-1])):
                window.load_file(str(SCENE));wait_jobs(window)
                window.btn_run_all.click();wait_jobs(window,60)
                self.assertFalse(errors,errors)
                self.assertIsNotNone(window.seg_engine.real_engine)
                self.assertEqual(window.seg['prediction_input_shape'],[256,256])
                self.assertTrue(window.seg['measured_nir_used'])
                enhanced=window.sr_bgrn
                window.combo_detector.setCurrentIndex(1)
                self.assertIs(window.sr_bgrn,enhanced)
                self.assertIsNone(window.seg)
                self.assertFalse(window.btn_save_all.isEnabled())
                window.btn_detect.click();wait_jobs(window,60)
                self.assertEqual(window.seg['method'],'U-Net')
                self.assertTrue(window.btn_save_all.isEnabled())
                window.combo_detector.setCurrentIndex(0)
                window.btn_detect.click();wait_jobs(window,60)
                self.assertFalse(errors,errors)
                self.assertIn('OpenEarthMap',window.seg['method'])
                window.load_file(str(ROOT/'tests/fixtures/formats/scene_rgb.png'));wait_jobs(window)
                window.btn_run_all.click();wait_jobs(window,60)
                self.assertFalse(window.seg['measured_nir_used'])
                self.assertTrue(window.btn_choose.isEnabled())
                self.assertFalse(errors,errors)
        finally:
            wait_jobs(window,60);window.close();window.deleteLater();QTest.qWait(10)


if __name__=='__main__':
    unittest.main()
