"""Generate deterministic software format fixtures, separate from real satellite demos."""
from _bootstrap import ROOT  # also puts src/ on sys.path
from pathlib import Path
import json
import numpy as np
import rasterio
from rasterio.transform import from_origin
from PIL import Image
from satsr.data import synthetic_scene
from satsr.geo_io import load_any_user_image, IMAGE_EXTENSIONS



def generate_test_fixtures(folder=None):
    folder = Path(folder or ROOT/"tests"/"fixtures"/"formats").resolve()
    if folder == (ROOT/"demo_images").resolve():
        raise ValueError("demo_images is reserved for real satellite data; choose a fixture folder")
    folder.mkdir(parents=True, exist_ok=True)
    _, scene, labels = synthetic_scene(777123, 96)
    rgb = np.clip(scene[..., [2,1,0]].astype(float)/3500*255, 0, 255).astype(np.uint8)
    photo = Image.fromarray(rgb)
    for name in ("scene_rgb.png","scene_rgb.jpg","scene_rgb.jpeg","scene_rgb.bmp","scene_rgb.webp","scene_rgb.tif","scene_rgb.tiff"):
        photo.save(folder/name, **({"quality":95} if name.endswith((".jpg",".jpeg",".webp")) else {}))
    photo.convert("L").save(folder/"scene_grayscale.png")
    for name in ("scene_palette.png","scene_palette.tiff"):
        photo.quantize(colors=32).save(folder/name)
    for name in ("scene_cmyk.jpg","scene_cmyk.tiff"):
        photo.convert("CMYK").save(folder/name)
    exif = Image.Exif(); exif[274] = 6
    photo.resize((96,64)).save(folder/"scene_exif_rotated.jpeg", exif=exif)
    rgba = photo.convert("RGBA"); alpha = np.full((96,96),255,np.uint8)
    alpha[:12] = 0; alpha[12:24] = 128; rgba.putalpha(Image.fromarray(alpha))
    for name in ("scene_transparent.png","scene_transparent.webp","scene_rgba.tiff"):
        rgba.save(folder/name)
    gray16 = np.arange(96*96,dtype=np.uint16).reshape(96,96)*7
    for name in ("scene_grayscale16.png","scene_grayscale16.tiff"):
        Image.fromarray(gray16).save(folder/name)
    photo.resize((1377,129)).save(folder/"scene_large.bmp")
    photo.resize((1,1)).save(folder/"scene_tiny.png")
    photo.save(folder/"scene_uppercase.PNG")
    transform = from_origin(635000,5785000,10,10)
    def raster(name, values, nodata=False, descriptions=None, scale=None):
        count = 4 if values.ndim == 3 else 1
        with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True):
            with rasterio.open(folder/name,"w",driver="GTiff",height=96,width=96,count=count,
                               dtype=values.dtype,crs="EPSG:32634",transform=transform) as dst:
                dst.write(np.moveaxis(values,-1,0) if count == 4 else values[None])
                dst.update_tags(source_kind="synthetic illustration with illustrative georeference")
                if descriptions:
                    for i,key in enumerate(descriptions):
                        dst.set_band_description(i+1,key)
                if nodata:
                    valid = np.full((96,96),255,np.uint8); valid[:10] = 0; dst.write_mask(valid)
                if scale:
                    dst.scales = (scale,)*count
    keys = ("B02","B03","B04","B08")
    raster("satellite_4band.tif",scene,descriptions=keys)
    raster("satellite_float_reflectance.tiff",scene.astype(np.float32)/10000,descriptions=keys)
    raster("satellite_scaled_dn.tif",scene,descriptions=keys,scale=.0001)
    raster("satellite_nodata.tif",scene,nodata=True,descriptions=keys)
    raster("satellite_reordered.tif",scene[..., [3,2,1,0]],descriptions=keys[::-1])
    (folder/"satellite_bands").mkdir(exist_ok=True)
    for i,key in enumerate(keys):
        raster("satellite_bands/"+key+".tif",scene[..., i])
    np.save(folder/"synthetic_labels.npy",labels)
    records = []
    for path in sorted(folder.iterdir()):
        if path.suffix.lower() in IMAGE_EXTENSIONS:
            bands, _, _, meta = load_any_user_image(path)
            records.append(dict(file=path.name,format=IMAGE_EXTENSIONS[path.suffix.lower()],
                loaded_width=meta["width"],loaded_height=meta["height"],source_kind=meta["source_kind"],
                valid_pixels=int(bands["B02"].valid_mask.sum())))
    (folder/"manifest.json").write_text(json.dumps(records,indent=2),encoding="utf-8")
    (folder/"README.md").write_text(
        "# Software format fixtures\n\nGenerated inputs for unit/regression tests only. Real satellite demo inputs are in demo_images/.\n\n"
        "Open any top-level image in either GUI. PNG, JPG, JPEG, BMP, WebP, TIF and TIFF are included, "
        "plus grayscale, palette, alpha, CMYK, EXIF rotation, 16-bit, tiny and large-image cases.\n\n"
        "Use satellite_4band.tif for full mapping and GeoTIFF export. Its 10 m grid and location are illustrative. "
        "Float, scale-tagged and reordered TIFFs encode the same spectra. satellite_nodata.tif exercises masks.\n\n"
        "Select the four TIFFs in satellite_bands/ for the individual-band workflow. Ordinary photos use estimated NIR.\n\n"
        "manifest.json records expected loaded dimensions and validity. synthetic_labels.npy holds generator class labels. "
        "Regenerate with python scripts/generate_test_fixtures.py.\n",encoding="utf-8")
    print(f"Validated {len(records)} demo images in {folder.resolve()}")
    return records


if __name__ == "__main__":
    generate_test_fixtures()
