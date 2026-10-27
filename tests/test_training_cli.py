import _paths  # noqa: F401  (puts src/ and scripts/ on sys.path)
from pathlib import Path
from types import SimpleNamespace
import json
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
import rasterio
from rasterio.transform import from_origin
from satsr.data import SatelliteDataset, synthetic_scene, degrade
from prepare_dataset import prepare
from train import preflight_datasets
from predict import run_prediction, publish_outputs
from convert_to_onnx import export_checkpoint

ROOT = Path(__file__).resolve().parents[1]


def write_raw(root, split, seed, extension=".tiff"):
    image,hr,mask = synthetic_scene(seed,32)
    for key,data,res in (("hr",hr,5),("lr",image,10),("masks",mask,5)):
        folder = root/split/key; folder.mkdir(parents=True,exist_ok=True)
        with rasterio.open(folder/("scene"+extension),"w",driver="GTiff",
                           height=data.shape[0],width=data.shape[1],count=4 if data.ndim == 3 else 1,
                           dtype=data.dtype,crs="EPSG:32634",transform=from_origin(635000,5785000,res,res)) as dst:
            dst.write(np.moveaxis(data,-1,0) if data.ndim == 3 else data[None])


class TrainingTests(unittest.TestCase):
    def test_prepare_tiff_train_validate_and_test_on_real_dataset_contract(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for i,split in enumerate(("train","val","test")):
                write_raw(root/"raw",split,99111+i)
            count = prepare(root/"raw",root/"dataset")
            self.assertEqual(count,3)
            result = subprocess.run([sys.executable,str(ROOT/"scripts"/"train.py"),"--data",str(root/"dataset"),
                        "--output",str(root/"models"),"--epochs","1","--batch-size","1","--device","cpu"],
                        cwd=root,text=True,capture_output=True,timeout=60)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            report = json.loads((root/"models/training_report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["split_sizes"],dict(train=1,val=1,test=1))
            self.assertEqual(len(report["history"]["sr"]),1)
            self.assertTrue((root/"models/unet.pt").is_file())

    def test_failed_preparation_does_not_leave_half_dataset(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            write_raw(root/"raw","train",99111)
            with self.assertRaises(ValueError):
                prepare(root/"raw",root/"prepared")
            self.assertFalse((root/"prepared").exists())
            self.assertFalse(list(root.glob(".l1bsr-prepare-*")))
            with self.assertRaises(ValueError):
                prepare(root/"raw",root/"raw")

    def test_data_split_leakage_and_batch_shape_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            lr,hr,mask = synthetic_scene(444)
            for split in ("train","val","test"):
                (root/split).mkdir(); np.savez(root/split/"scene.npz",lr=lr,hr=hr,mask=mask)
            datasets = {split:SatelliteDataset(root,split) for split in ("train","val","test")}
            with self.assertRaisesRegex(ValueError,"leakage"):
                preflight_datasets(datasets)
            lr,hr,mask = synthetic_scene(445,32)
            np.savez(root/"train/other.npz",lr=lr,hr=hr,mask=mask)
            with self.assertRaisesRegex(ValueError,"same dimensions"):
                preflight_datasets({"train":SatelliteDataset(root,"train")})

    def test_dataset_and_degradation_bad_inputs(self):
        for size in (0,15,17,1.5):
            with self.assertRaises(ValueError):
                synthetic_scene(1,size)
        with self.assertRaises(ValueError):
            degrade(np.zeros((3,5,4),np.uint16))
        with self.assertRaises(ValueError):
            SatelliteDataset(demo=True,samples=100000)
        with self.assertRaises(ValueError):
            SatelliteDataset(demo=True,split="wrong")
        dataset = SatelliteDataset(demo=True,samples=1)
        with self.assertRaises(IndexError):
            dataset.arrays(-1)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)/"train"; path.mkdir()
            np.savez(path/"scene.npz",hr=np.zeros((32,32,4),np.uint16))
            with self.assertRaisesRegex(ValueError,"hr and mask"):
                SatelliteDataset(td).arrays(0)

    def test_nan_learning_rate_rejected_before_writing_checkpoints(self):
        with tempfile.TemporaryDirectory() as td:
            output = Path(td)/"weights"
            result = subprocess.run([sys.executable,str(ROOT/"scripts"/"train.py"),"--demo","--lr","nan","--output",str(output)],
                                    capture_output=True,text=True,timeout=30)
            self.assertNotEqual(result.returncode,0)
            self.assertFalse(output.exists())


class CommandLineTests(unittest.TestCase):
    def args(self,output,image=None,demo=False):
        return SimpleNamespace(output=str(output),image=str(image) if image else None,bands=None,
                               demo=demo,models=str(ROOT/"checkpoints/demo"),device="cpu")

    def test_prediction_rerun_removes_stale_geo_and_ground_truth_files(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); output = root/"result"
            first = run_prediction(self.args(output,demo=True))
            self.assertTrue((output/"enhanced.tif").is_file())
            (output/"notes.txt").write_text("keep me")
            second = run_prediction(self.args(output,ROOT/"tests/fixtures/formats/scene_rgb.png"))
            self.assertFalse((output/"enhanced.tif").exists())
            self.assertFalse((output/"land_cover.tif").exists())
            self.assertFalse((output/"ground_truth.png").exists())
            self.assertEqual((output/"notes.txt").read_text(),"keep me")
            self.assertNotIn("enhanced.tif",second["files"])
            self.assertEqual(json.loads((output/"report.json").read_text(encoding="utf-8"))["input"],
                             str(ROOT/"tests/fixtures/formats/scene_rgb.png"))

    def test_invalid_prediction_preserves_existing_outputs(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); output = root/"result"; output.mkdir()
            report = output/"report.json"; report.write_text("original")
            with self.assertRaises(ValueError):
                run_prediction(self.args(output,root/"missing.png"))
            self.assertEqual(report.read_text(),"original")
            collision = output/"enhanced.png"; collision.write_bytes(b"input")
            with self.assertRaisesRegex(ValueError,"overwrite an input"):
                run_prediction(self.args(output,collision))
            self.assertEqual(collision.read_bytes(),b"input")

    def test_output_publish_rolls_back_commit_failure(self):
        import os
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); stage = root/"stage"; stage.mkdir()
            output = root/"out"; output.mkdir()
            (output/"report.json").write_text("old")
            (stage/"report.json").write_text("new"); (stage/"enhanced.png").write_text("new image")
            original = os.replace
            def failed(src,dst):
                if Path(src) == stage/"report.json":
                    raise OSError("simulated failure")
                return original(src,dst)
            with patch("predict.os.replace",side_effect=failed), self.assertRaises(OSError):
                publish_outputs(stage,output)
            self.assertEqual((output/"report.json").read_text(),"old")
            self.assertFalse((output/"enhanced.png").exists())

    def test_cli_from_another_working_directory_uses_project_models(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            result = subprocess.run([sys.executable,str(ROOT/"scripts"/"predict.py"),"--image",str(ROOT/"tests/fixtures/formats/scene_tiny.png"),
                                     "--device","cpu","--output",str(root/"prediction")],
                                    cwd=root,capture_output=True,text=True,timeout=30)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            ids = np.load(root/"prediction/class_ids.npy")
            self.assertEqual(ids.shape,(2,2))

    def test_onnx_export_validates_before_replacing_existing_model(self):
        with tempfile.TemporaryDirectory() as td:
            target = Path(td)/"model.onnx"
            export_checkpoint(ROOT/"checkpoints/demo/rcan.pt",target)
            original = target.read_bytes()
            with patch("numpy.testing.assert_allclose",side_effect=AssertionError("parity failed")), self.assertRaises(AssertionError):
                export_checkpoint(ROOT/"checkpoints/demo/rcan.pt",target)
            self.assertEqual(target.read_bytes(),original)
