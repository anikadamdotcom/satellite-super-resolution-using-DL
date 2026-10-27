import _paths  # noqa: F401  (puts src/ and scripts/ on sys.path)
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
import warnings
import numpy as np
import rasterio
from rasterio.transform import Affine
from PIL import Image
from satsr.geo_io import (load_any_user_image, read_band, render_rgb, IMAGE_EXTENSIONS,
                    percentile_stretch, to_uint16)

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT/"tests"/"fixtures"/"formats"


class FormatTests(unittest.TestCase):
    def test_full_range_scaled_reflectance_roundtrip_and_real_overflow_rejection(self):
        original = np.array([0, 1, 32768, 65534, 65535], np.uint16)
        physical = original.astype(np.float64)*.0001
        np.testing.assert_array_equal(to_uint16(physical, reflectance=True), original)
        with self.assertRaisesRegex(ValueError, "Scaled reflectance"):
            to_uint16(np.array([6.5536]), reflectance=True)
        # Actual tagged GeoTIFF decoding follows the same full-range contract.
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)/"full_range.tif"
            data = np.tile(original, (4, 4, 1))
            with rasterio.open(path, "w", driver="GTiff", count=4, width=5, height=4,
                               dtype="uint16", crs="EPSG:32643", transform=Affine(10,0,100,0,-10,200)) as dst:
                dst.write(data); dst.scales = (.0001,)*4
                dst.descriptions = ("B02", "B03", "B04", "B08")
            np.testing.assert_array_equal(load_any_user_image(path)[1], np.moveaxis(data, 0, -1))

    def test_all_demo_files_and_manifest(self):
        records = json.loads((DEMO/"manifest.json").read_text(encoding="utf-8"))
        extensions = set()
        for record in records:
            with self.subTest(file=record["file"]):
                path = DEMO/record["file"]; extensions.add(path.suffix.lower())
                bands, image, rgb, meta = load_any_user_image(path)
                self.assertEqual(image.shape, (record["loaded_height"],record["loaded_width"],4))
                self.assertEqual(rgb.dtype,np.uint8)
                self.assertEqual(image.dtype,np.uint16)
                self.assertEqual(bands["B02"].valid_mask.sum(),record["valid_pixels"])
                self.assertTrue(np.array_equal(rgb,render_rgb(image,bands["B02"])))
        self.assertEqual(extensions,set(IMAGE_EXTENSIONS))

    def test_preserve_photo_colors_and_scalar_image_brightness(self):
        bands, image, rgb, _ = load_any_user_image(DEMO/"scene_rgb.png")
        with Image.open(DEMO/"scene_rgb.png") as source:
            self.assertTrue(np.array_equal(rgb,np.asarray(source)))
        self.assertTrue((percentile_stretch(np.full((4,4),100,np.uint8)) == 100).all())

    def test_alpha_and_nodata(self):
        for name in ("scene_transparent.png","scene_transparent.webp","scene_rgba.tiff","satellite_nodata.tif"):
            with self.subTest(file=name):
                bands, image, rgb, _ = load_any_user_image(DEMO/name)
                valid = bands["B02"].valid_mask
                self.assertTrue(valid.any() and not valid.all())
                self.assertTrue((image[~valid] == 0).all())
                self.assertTrue((rgb[~valid] == 0).all())
        with tempfile.TemporaryDirectory() as td:
            file = Path(td)/"empty_alpha.png"
            Image.fromarray(np.zeros((8,8,4),np.uint8)).save(file)
            with self.assertRaisesRegex(ValueError,"transparent"):
                load_any_user_image(file)

    def test_equivalent_spectral_encodings(self):
        original = load_any_user_image(DEMO/"satellite_4band.tif")[1]
        for name in ("satellite_float_reflectance.tiff","satellite_scaled_dn.tif","satellite_reordered.tif"):
            with self.subTest(file=name):
                actual = load_any_user_image(DEMO/name)[1]
                np.testing.assert_array_equal(actual,original)

    def test_exif_and_grayscale16(self):
        rgb = load_any_user_image(DEMO/"scene_exif_rotated.jpeg")[2]
        self.assertEqual(rgb.shape[:2],(96,64))
        for name in ("scene_grayscale16.png","scene_grayscale16.tiff"):
            rgb = load_any_user_image(DEMO/name)[2]
            self.assertGreater(int(rgb.max())-int(rgb.min()),200)
            self.assertGreater(np.unique(rgb).size,100)

    def test_unicode_path_and_file_handle_release(self):
        with tempfile.TemporaryDirectory() as td:
            dest = Path(td)/"image \u0938\u0945\u091f\u0947\u0932\u093e\u0907\u091f.png"
            shutil.copyfile(DEMO/"scene_rgb.png",dest)
            load_any_user_image(dest)
            dest.rename(dest.with_name("renamed.png"))
            dest.with_name("renamed.png").unlink()

    def test_invalid_uploads(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cases = {"empty.png":b"", "corrupt.jpg":b"not an image",
                     "fake.png":(DEMO/"scene_rgb.jpg").read_bytes(),
                     "unsupported.txt":(DEMO/"scene_rgb.png").read_bytes(),
                     "truncated.png":(DEMO/"scene_rgb.png").read_bytes()[:80]}
            for name,data in cases.items():
                file = root/name; file.write_bytes(data)
                with self.subTest(file=name), self.assertRaises(ValueError):
                    load_any_user_image(file)
            for path in (root,root/"missing.png"):
                with self.assertRaises(ValueError):
                    load_any_user_image(path)

    def test_animated_and_multipage_rejection(self):
        with tempfile.TemporaryDirectory() as td:
            frame = Image.new("RGB",(8,8),"red"); other = Image.new("RGB",(8,8),"green")
            for ext in (".webp",".tiff"):
                path = Path(td)/("animated"+ext)
                frame.save(path,save_all=True,append_images=[other],duration=100)
                with self.assertRaises(ValueError):
                    load_any_user_image(path)

    def test_resource_and_data_validation(self):
        for bound in (0,-1,1.5,True):
            with self.assertRaises(ValueError):
                load_any_user_image(DEMO/"scene_rgb.png",max_side=bound)
        with patch("satsr.geo_io.MAX_PHOTO_PIXELS",10), self.assertRaisesRegex(ValueError,"pixel limit"):
            load_any_user_image(DEMO/"scene_rgb.png")
        for array in (np.array([-1]),np.array([65536]),np.array([np.nan]),np.array([1j]),np.array([])):
            with self.assertRaises(ValueError):
                to_uint16(array)
        with self.assertRaises(ValueError):
            percentile_stretch(np.full((2,2),np.nan))

    def test_masked_float_and_no_valid_pixels(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)/"float.tif"
            data = np.full((4,8,8),.25,np.float32); data[:,0] = np.nan
            with rasterio.open(path,"w",driver="GTiff",height=8,width=8,count=4,
                               dtype="float32",nodata=np.nan,transform=Affine(10,0,500000,0,-10,5000000),crs="EPSG:32634") as dst:
                dst.write(data)
            bands, loaded, _, _ = load_any_user_image(path)
            self.assertTrue((loaded[1:] == 2500).all())
            self.assertFalse(bands["B02"].valid_mask[0].any())
            data[:] = np.nan
            with rasterio.open(path,"r+") as dst:
                dst.write(data)
            with self.assertRaisesRegex(ValueError,"valid pixels"):
                load_any_user_image(path)

    def test_rgba_tiff_internal_validity_mask_and_bad_band_labels(self):
        from rasterio.enums import ColorInterp
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)/"rgba.tif"
            with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True):
                with rasterio.open(path,"w",driver="GTiff",height=8,width=8,count=4,dtype="uint8",
                                   crs="EPSG:32634",transform=Affine(10,0,500000,0,-10,5000000)) as dst:
                    dst.write(np.full((4,8,8),255,np.uint8))
                    dst.colorinterp = (ColorInterp.red,ColorInterp.green,ColorInterp.blue,ColorInterp.alpha)
                    mask = np.full((8,8),255,np.uint8); mask[0] = 0; dst.write_mask(mask)
            bands,_,_,_ = load_any_user_image(path)
            self.assertFalse(bands["B02"].valid_mask[0].any())
            wrong = Path(td)/"wrong.tif"
            with rasterio.open(wrong,"w",driver="GTiff",height=8,width=8,count=4,dtype="uint16",
                               crs="EPSG:32634",transform=Affine(10,0,500000,0,-10,5000000)) as dst:
                dst.write(np.full((4,8,8),1000,np.uint16))
                for i,key in enumerate(("B02","B03","B04","B04")):
                    dst.set_band_description(i+1,key)
            with self.assertRaisesRegex(ValueError,"descriptions"):
                load_any_user_image(wrong)
