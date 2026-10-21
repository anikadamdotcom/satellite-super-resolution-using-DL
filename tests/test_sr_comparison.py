"""Checks for the SR comparison and mapping experiment code (optional research tools)."""
import importlib.util
import unittest

import numpy as np
import torch

from download_mapping_benchmark import CLASSES, WORLDCOVER_TO_CLASS, worldcover_tile
from models.sen2sr_lite import DEFAULT_DIR, FROM_SEN2SR, TO_SEN2SR, block_mean

torch.set_num_threads(2)
HAVE_SEN2SR = importlib.util.find_spec("sen2sr") is not None and (DEFAULT_DIR/"model.safetensor").is_file()


class ComparisonHelperTests(unittest.TestCase):
    def test_block_mean_averages_whole_blocks(self):
        image = np.arange(4*6*2, dtype=np.float64).reshape(4, 6, 2)
        out = block_mean(image, 2)
        self.assertEqual(out.shape, (2, 3, 2))
        self.assertAlmostEqual(out[0, 0, 0], image[:2, :2, 0].mean())
        self.assertEqual(block_mean(image[:3, :5], 2).shape, (1, 2, 2))  # partial blocks dropped
        with self.assertRaises(ValueError):
            block_mean(image[:1], 2)

    def test_band_order_round_trip(self):
        bgrn = np.array([1, 2, 3, 4])
        self.assertEqual(list(bgrn[TO_SEN2SR]), [3, 2, 1, 4])  # B04, B03, B02, B08
        self.assertEqual(list(bgrn[TO_SEN2SR][FROM_SEN2SR]), [1, 2, 3, 4])

    def test_worldcover_tiles_and_classes(self):
        self.assertEqual(worldcover_tile(75.98, 30.95), "N30E075")
        self.assertEqual(worldcover_tile(77.209, 28.614), "N27E075")
        self.assertEqual(worldcover_tile(-0.5, -1.0), "S03W003")
        self.assertEqual(set(WORLDCOVER_TO_CLASS), {10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100})
        self.assertEqual(set(WORLDCOVER_TO_CLASS.values()), set(range(len(CLASSES))))

    def test_mapping_scores(self):
        from mapping_experiment import scores
        matrix = np.zeros((len(CLASSES), len(CLASSES)), np.int64)
        matrix[0, 0], matrix[0, 1], matrix[1, 1] = 8, 2, 10
        result = scores(matrix)
        self.assertAlmostEqual(result["overall_accuracy"], .9)
        self.assertAlmostEqual(result["per_class_f1"]["Built-up"], round(16/18, 4))
        self.assertIsNone(result["per_class_f1"]["Cropland"])  # absent classes are not averaged
        self.assertAlmostEqual(result["mean_iou"], (.8+10/12)/2)


@unittest.skipUnless(HAVE_SEN2SR, "sen2sr package or SEN2SR weights not installed")
class Sen2srModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from models.sen2sr_lite import SEN2SRLite
        cls.model = SEN2SRLite()

    def test_shapes_and_odd_sizes(self):
        rng = np.random.default_rng(0)
        image = rng.integers(200, 3000, (70, 150, 4)).astype(np.uint16)
        x4 = self.model.super_resolve_x4(image)
        self.assertEqual((x4.shape, x4.dtype), ((280, 600, 4), np.uint16))
        self.assertEqual(self.model.super_resolve(image).shape, (140, 300, 4))

    def test_flat_scene_stays_flat_and_consistent(self):
        image = np.zeros((128, 128, 4), np.uint16)
        image[...] = [500, 800, 700, 3000]
        out = self.model.super_resolve(image).astype(float)
        self.assertTrue(np.allclose(out.mean((0, 1)), [500, 800, 700, 3000], rtol=.02))

    def test_output_matches_input_when_averaged_back(self):
        rng = np.random.default_rng(1)
        image = (np.kron(rng.integers(300, 4000, (32, 32, 4)), np.ones((8, 8, 1)))).astype(np.uint16)
        back = block_mean(self.model.super_resolve_x4(image), 4)
        error = np.abs(back-image).mean()/image.mean()
        self.assertLess(error, .05)


if __name__ == "__main__":
    unittest.main()
