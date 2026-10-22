"""SEN2VENµS preparation, the SR-only patch dataset, the compensated loss and train.py --task sr."""
from pathlib import Path
import json
import subprocess
import sys
import tempfile
import unittest
import numpy as np
import torch
from data import SRPatchDataset
from prepare_sen2venus import prepare, patch_filter
from train import compensated_l1
from models.common import read_checkpoint

ROOT = Path(__file__).resolve().parents[1]


def fake_site(root, site, tile="31TCJ", date="2020-01-01", n=6, seed=0):
    """Mimic one SEN2VENµS acquisition: int16 (N,4,H,W) 10 m and 5 m tensors."""
    rng = np.random.default_rng(seed)
    hr = rng.integers(200, 4000, (n, 4, 64, 64)).astype(np.int16)
    lr = torch.nn.functional.avg_pool2d(torch.from_numpy(hr).float(), 2).round().short()
    folder = Path(root)/site/site
    folder.mkdir(parents=True, exist_ok=True)
    torch.save(lr, folder/f"{site}_{tile}_{date}_10m_b2b3b4b8.pt")
    torch.save(torch.from_numpy(hr), folder/f"{site}_{tile}_{date}_05m_b2b3b4b8.pt")
    return lr, torch.from_numpy(hr)


class Sen2VenusTests(unittest.TestCase):
    def test_prepare_filters_clips_and_writes_memmappable_pairs(self):
        with tempfile.TemporaryDirectory() as td:
            lr, hr = fake_site(td, "AAA")
            lr[0, 0, :20] = -5        # >10% negative -> dropped
            lr[1, 0, 0, 0] = -5       # tiny negative -> clipped to 0
            hr[2, :, 0, 0] = -10000   # nodata -> dropped
            folder = Path(td)/"AAA/AAA"
            torch.save(lr, next(folder.glob("*_10m_*.pt"))); torch.save(hr, next(folder.glob("*_05m_*.pt")))
            for site, seed in (("BBB", 1), ("CCC", 2)):
                fake_site(td, site, seed=seed)
            summary = prepare(td, Path(td)/"npy", dict(train=["AAA"], val=["BBB"], test=["CCC"]))
            self.assertEqual(summary["AAA"], dict(split="train", kept=4, total=6))
            data = SRPatchDataset(Path(td)/"npy", "train")
            self.assertEqual(len(data), 4)
            small, large = data.arrays(0)
            self.assertEqual((small.dtype, small.shape, large.shape), (np.uint16, (32, 32, 4), (64, 64, 4)))
            self.assertEqual(int(small[0, 0, 0]), 0)
            self.assertTrue((Path(td)/"npy/train/patches.csv").read_text().startswith("site,tile,date"))
            data.close(); del small, large

    def test_patch_dataset_crops_augments_and_rejects_bad_files(self):
        with tempfile.TemporaryDirectory() as td:
            folder = Path(td)/"train"; folder.mkdir()
            lr = np.random.default_rng(0).integers(0, 5000, (3, 32, 32, 4)).astype(np.uint16)
            hr = np.repeat(np.repeat(lr, 2, 1), 2, 2)
            np.save(folder/"x_lr.npy", lr); np.save(folder/"x_hr.npy", hr)
            data = SRPatchDataset(td, "train", crop=16, augment=True)
            for index in range(3):
                small, large = data[index]
                self.assertEqual((tuple(small.shape), tuple(large.shape)), ((4, 16, 16), (4, 32, 32)))
                # The HR crop must stay aligned with the LR crop through every augmentation.
                torch.testing.assert_close(torch.nn.functional.avg_pool2d(large[None], 2)[0], small)
            with self.assertRaises(ValueError):
                SRPatchDataset(td, "train", crop=64)
            with self.assertRaises(IndexError):
                data.arrays(3)
            data.close(); del small, large
            np.save(folder/"x_hr.npy", hr[:, :-2])
            with self.assertRaises(ValueError):
                SRPatchDataset(td, "train")

    def test_compensated_l1_ignores_small_shifts_and_band_offsets(self):
        target = torch.rand(2, 4, 32, 32)
        shifted = torch.roll(target, (1, -2), (2, 3)) + torch.tensor([.01, -.02, 0, .03]).view(1, 4, 1, 1)
        self.assertLess(float(compensated_l1(shifted, target).max()), 1e-6)
        self.assertGreater(float(torch.nn.functional.l1_loss(shifted, target)), .1)
        self.assertGreater(float(compensated_l1(torch.rand(2, 4, 32, 32), target).min()), .1)
        keep, _ = patch_filter(torch.zeros(1, 4, 8, 8, dtype=torch.int16), torch.ones(1, 4, 16, 16, dtype=torch.int16), .1)
        self.assertFalse(bool(keep[0]))  # constant fill

    def test_sr_only_training_cli_writes_app_compatible_checkpoint(self):
        with tempfile.TemporaryDirectory() as td:
            for site, seed in (("AAA", 1), ("BBB", 2), ("CCC", 3)):
                fake_site(td, site, seed=seed, n=4)
            prepare(td, Path(td)/"npy", dict(train=["AAA"], val=["BBB"], test=["CCC"]))
            result = subprocess.run([sys.executable, str(ROOT/"train.py"), "--task", "sr", "--data", str(Path(td)/"npy"),
                                     "--output", str(Path(td)/"out"), "--epochs", "1", "--batch-size", "2", "--crop", "16",
                                     "--workers", "0", "--device", "cpu", "--lr", "2e-4"],
                                    cwd=td, text=True, capture_output=True, timeout=120)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            checkpoint = read_checkpoint(Path(td)/"out/rcan.pt", "sr")
            self.assertIn("SEN2VENµS", checkpoint["data_source"])
            self.assertEqual(checkpoint["model_config"]["n_feats"], 32)
            report = json.loads((Path(td)/"out/training_report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["split_sizes"], dict(train=4, val=4, test=4))
            self.assertEqual(report["sites"]["test"], ["CCC"])
            self.assertIn("bicubic_compensated_l1", report["test"])
            self.assertFalse((Path(td)/"out/unet.pt").exists())


if __name__ == "__main__":
    unittest.main()
