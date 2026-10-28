"""SEN2VENµS preparation, the SR-only patch dataset, the compensated loss and train.py --task sr."""
import _paths  # noqa: F401  (puts src/ and scripts/ on sys.path)
from pathlib import Path
import json
import subprocess
import sys
import tempfile
import unittest
import numpy as np
import torch
from satsr.data import SRPatchDataset
from prepare_sen2venus import prepare, patch_filter
from train import compensated_l1
from satsr.models.common import read_checkpoint

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
                small, large, shift = data[index]
                self.assertEqual(tuple(shift.shape), (2,))
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

    def test_subpixel_alignment_estimates_and_follows_augmentation(self):
        from satsr.models.alignment import shift_image, estimate_shift, augment_shift, aligned_l1
        torch.manual_seed(0)
        base = torch.nn.functional.interpolate(torch.rand(2, 4, 16, 16), size=64, mode="bicubic", align_corners=False)
        true = torch.tensor([[0.625, -1.25], [-0.5, 0.375]])
        hr = shift_image(base, true)
        torch.testing.assert_close(estimate_shift(torch.nn.functional.avg_pool2d(base, 2), hr), true)
        self.assertLess(float(aligned_l1(base + .01, hr, true).max()), 1e-5)
        image, moved = base[0].permute(1, 2, 0).numpy(), hr[0].permute(1, 2, 0).numpy()
        for k in range(4):
            for flip in (False, True):
                a, b = np.rot90(image, k), np.rot90(moved, k)
                if flip:
                    a, b = a[:, ::-1], b[:, ::-1]
                out = shift_image(torch.from_numpy(a.copy()).permute(2, 0, 1)[None], torch.tensor([augment_shift(true[0], k, flip)]))
                torch.testing.assert_close(out[0, :, 6:-6, 6:-6], torch.from_numpy(b.copy()).permute(2, 0, 1)[:, 6:-6, 6:-6], atol=1e-5, rtol=0)

    def test_radiometric_match_restores_sentinel2_contrast_and_mean(self):
        from satsr.models.alignment import radiometric_match
        torch.manual_seed(1)
        hr = 2000 + 800*torch.nn.functional.interpolate(torch.rand(3, 4, 8, 8), size=64, mode="bicubic", align_corners=False)
        lr = torch.nn.functional.avg_pool2d(hr, 2)
        venus = .9*hr + 50                                          # VENµS-like: 10% less contrast, offset
        gain, offset = radiometric_match(lr, venus, torch.zeros(3, 2)).unbind(1)
        torch.testing.assert_close(gain, torch.full_like(gain, 1/.9), rtol=1e-3, atol=0)
        torch.testing.assert_close(venus*gain[..., None, None] + offset[..., None, None], hr, rtol=0, atol=.5)
        flat = radiometric_match(torch.full((1, 4, 32, 32), 500.), torch.full((1, 4, 64, 64), 480.), torch.zeros(1, 2))
        torch.testing.assert_close(flat[:, 0], torch.ones(1, 4))    # flat patch keeps gain 1

    def test_striping_score_detects_two_pixel_stripes(self):
        from plot_compare_striping import striping
        rng = np.random.default_rng(0)
        smooth = rng.normal(size=(64, 64)).cumsum(0).cumsum(1)
        stripes = smooth + 20*np.tile([1., -1.], 32)[None, :]
        self.assertLess(striping(smooth), .01)
        self.assertGreater(striping(stripes), .2)

    def test_consistent_rcan_preserves_input_at_10m(self):
        from satsr.models.rcan import RCAN
        model = RCAN(n_feats=8, n_resgroups=1, n_resblocks=1, residual=True, consistent=True).eval()
        torch.nn.init.normal_(model.tail[-1].weight, std=.1)
        x = torch.rand(1, 4, 12, 12)
        with torch.no_grad():
            y = model(x)
        torch.testing.assert_close(torch.nn.functional.avg_pool2d(y, 2), x, atol=1e-6, rtol=0)

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
            result = subprocess.run([sys.executable, str(ROOT/"scripts"/"train.py"), "--task", "sr", "--data", str(Path(td)/"npy"),
                                     "--output", str(Path(td)/"out"), "--epochs", "1", "--batch-size", "2", "--crop", "16",
                                     "--workers", "0", "--device", "cpu", "--lr", "2e-4"],
                                    cwd=td, text=True, capture_output=True, timeout=120)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            checkpoint = read_checkpoint(Path(td)/"out/rcan.pt", "sr")
            self.assertIn("SEN2VENµS", checkpoint["data_source"])
            self.assertEqual(checkpoint["model_config"]["n_feats"], 32)
            self.assertFalse(checkpoint["model_config"]["consistent"])  # opt-in only (--consistent)
            report = json.loads((Path(td)/"out/training_report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["split_sizes"], dict(train=4, val=4, test=4))
            self.assertEqual(report["sites"]["test"], ["CCC"])
            self.assertIn("bicubic_aligned_psnr", report["test"])
            self.assertFalse((Path(td)/"out/unet.pt").exists())


if __name__ == "__main__":
    unittest.main()
