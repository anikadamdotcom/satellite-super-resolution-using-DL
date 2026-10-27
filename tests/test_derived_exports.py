"""Difference-from-bicubic map and derived-imagery labels on enhanced exports."""
import _paths  # noqa: F401  (puts src/ and scripts/ on sys.path)
from pathlib import Path
import tempfile
import unittest

import numpy as np
import rasterio
from PIL import Image

from satsr.geo_io import derived_note, load_any_user_image, save_image, save_jpeg_with_jgw, write_geotiff
from satsr.models.rcan import L1BSRSR
from satsr.results_export import (COMPARE_GAP, DIFF_RAMP, DIFF_SCALE_DN, compose_compare,
                            difference_from_bicubic, difference_rgb)

ROOT = Path(__file__).resolve().parents[1]


class DifferenceMapTests(unittest.TestCase):
    def test_identical_images_show_no_change(self):
        bands = np.full((6, 8, 4), 1200, np.uint16)
        magnitude, stats = difference_from_bicubic(bands, bands)
        self.assertFalse(magnitude.any())
        self.assertEqual((stats["mean_reflectance"], stats["changed_pct"]), (0., 0.))
        self.assertTrue((difference_rgb(magnitude) == DIFF_RAMP[0]).all())

    def test_magnitude_scale_and_validity(self):
        a = np.zeros((2, 2, 4), np.uint16); b = a.copy()
        b[0, 0] = 200; b[0, 1] = [0, 0, 0, 4000]; b[1, 1] = 50000
        magnitude, stats = difference_from_bicubic(a, b, valid=np.array([[1, 1], [1, 0]], bool))
        self.assertEqual(magnitude[0, 0], 200)          # mean over 4 bands
        self.assertEqual(magnitude[0, 1], 1000)
        self.assertAlmostEqual(stats["changed_pct"], 66.67)  # invalid pixel ignored
        rgb = difference_rgb(magnitude, valid=np.array([[1, 1], [1, 0]], bool))
        self.assertTrue((rgb[0, 1] == DIFF_RAMP[-1]).all())  # clipped at the scale maximum
        self.assertTrue((rgb[1, 1] == 0).all())
        middle = difference_rgb(np.array([[DIFF_SCALE_DN/2]]))[0, 0].astype(int)
        self.assertTrue(DIFF_RAMP[-1].sum() < middle.sum() < DIFF_RAMP[0].sum())
        with self.assertRaises(ValueError):
            difference_from_bicubic(a, b[:1])

    def test_compare_layout_and_mask(self):
        for size in (16, 64, 300):
            panel = np.zeros((size, size, 3), np.uint8)
            valid = np.ones((size, size), bool); valid[0, 0] = False
            arr, mask = compose_compare(panel, panel, panel, valid)
            self.assertEqual(arr.shape[1], 3*size+2*COMPARE_GAP)
            self.assertGreater(arr.shape[0], size)
            self.assertEqual(mask.shape, arr.shape[:2])
            self.assertFalse(mask[0, 0]); self.assertFalse(mask[0, size+COMPARE_GAP])
            self.assertTrue(mask[-1].all())  # caption strip is opaque
        with self.assertRaises(ValueError):
            compose_compare(panel, panel[:5], panel, valid)


class DerivedLabelTests(unittest.TestCase):
    def test_png_and_jpeg_carry_label(self):
        note = derived_note("RCAN (test)")
        self.assertIn("Not measured imagery", note)
        with tempfile.TemporaryDirectory() as td:
            rgb = np.full((5, 7, 3), 90, np.uint8)
            save_image(Path(td)/"a.png", rgb, description=note)
            save_image(Path(td)/"b.jpg", rgb, description=note)
            save_image(Path(td)/"plain.png", rgb)
            with Image.open(Path(td)/"a.png") as image:
                self.assertEqual((image.info["Description"], image.info["Derived"]), (note, "true"))
            with Image.open(Path(td)/"b.jpg") as image:
                self.assertEqual(image.getexif()[0x010E], note)
            with Image.open(Path(td)/"plain.png") as image:
                self.assertNotIn("Derived", image.info)

    def test_geotiff_and_worldfile_jpeg_labels(self):
        bands, image, _, _ = load_any_user_image(ROOT/"demo_images"/"sentinel2_urban_delhi_2023-12-05.tif")
        ref = bands["B02"]
        with tempfile.TemporaryDirectory() as td:
            path = write_geotiff(Path(td)/"enhanced.tif", L1BSRSR.bicubic_upscale(image), ref, sr_model="RCAN (test)")
            with rasterio.open(path) as src:
                tags = src.tags()
                self.assertEqual(tags["derived_product"], "true")
                self.assertEqual((tags["native_pixel_size"], tags["output_pixel_size"]), ("10 m", "5 m"))
                self.assertIn("DERIVED IMAGE", tags["TIFFTAG_IMAGEDESCRIPTION"])
                self.assertEqual(tags["stac_item_id"], ref.source_metadata["stac_item_id"])  # provenance kept
            jpg = save_jpeg_with_jgw(Path(td)/"enhanced.jpg", np.zeros((512, 512, 3), np.uint8), ref,
                                     description=derived_note())
            with Image.open(jpg) as image:
                self.assertIn("DERIVED IMAGE", image.getexif()[0x010E])


if __name__ == "__main__":
    unittest.main()
