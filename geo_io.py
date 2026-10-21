"""Validated raster/photo ingestion and atomic, geospatially correct exports."""
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
import os
import tempfile
import warnings
import numpy as np
import rasterio
from rasterio.enums import ColorInterp, Resampling
from rasterio.transform import Affine
from PIL import Image, ImageOps, UnidentifiedImageError
from PIL.PngImagePlugin import PngInfo

BAND_ORDER = ("B02", "B03", "B04", "B08")
IMAGE_EXTENSIONS = {".png":"PNG", ".jpg":"JPEG", ".jpeg":"JPEG", ".bmp":"BMP",
                    ".webp":"WEBP", ".tif":"TIFF", ".tiff":"TIFF"}
IMAGE_FILTER = ("All supported images (*.png *.jpg *.jpeg *.bmp *.webp *.tif *.tiff);;"
                "PNG (*.png);;JPEG (*.jpg *.jpeg);;TIFF (*.tif *.tiff);;"
                "BMP (*.bmp);;WebP (*.webp);;All files (*)")
MAX_PHOTO_PIXELS = 40_000_000


@dataclass
class BandData:
    path: str
    arr: np.ndarray
    transform: Affine
    crs: Any
    width: int
    height: int
    valid_mask: Any = None
    source_kind: str = "multispectral"
    display_limits: Any = None
    original_grid: Any = None
    source_metadata: Any = None


def validate_path(path, extensions=None):
    path = Path(path).expanduser()
    if not path.is_file():
        raise ValueError(f"Image file does not exist: {path}")
    if path.stat().st_size == 0:
        raise ValueError("The selected file is empty.")
    if extensions is not None and path.suffix.lower() not in extensions:
        raise ValueError("Unsupported file extension. Use PNG, JPG, JPEG, BMP, WebP, TIFF or TIF.")
    return path


def _size(height, width, max_side):
    if not isinstance(max_side, int) or isinstance(max_side, bool) or max_side < 1:
        raise ValueError("max_side must be a positive integer")
    ratio = min(1., max_side/max(height, width))
    return max(1, round(height*ratio)), max(1, round(width*ratio))


def to_uint16(array, reflectance=None):
    array = np.asarray(array)
    if array.size == 0 or array.dtype.kind not in "uif" or not np.isfinite(array).all():
        raise ValueError("Band values must be non-empty, real, finite numbers.")
    if array.min() < 0 or array.max() > 65535:
        raise ValueError("Band values must be between 0 and 65535; no wrapping or silent truncation is allowed.")
    if reflectance is True or (reflectance is None and array.dtype.kind == "f" and array.max() <= 1.5):
        array = array.astype(np.float64)*10000
    # A valid scale/offset round trip can land a few ulps above 65535.
    # Accept floating-point noise only, while still rejecting true overflow.
    if array.max() > 65535 + 1e-8:
        raise ValueError("Scaled reflectance exceeds the uint16 range")
    return np.rint(np.clip(array, 0, 65535)).astype(np.uint16)


def _read_scaled(src, indexes, shape):
    data = src.read(indexes, out_shape=(len(indexes), *shape), resampling=Resampling.bilinear)
    valid = np.all(src.read_masks(indexes, out_shape=(len(indexes), *shape)) > 0, axis=0)
    valid &= np.isfinite(data).all(axis=0)
    scales = np.array([src.scales[i-1] for i in indexes], dtype=float)
    offsets = np.array([src.offsets[i-1] for i in indexes], dtype=float)
    tagged = np.any(scales != 1) or np.any(offsets != 0)
    if tagged:
        if not np.isfinite(scales).all() or not np.isfinite(offsets).all() or (scales <= 0).any():
            raise ValueError("Raster has invalid radiometric scale/offset metadata")
        data = np.maximum(data.astype(np.float64)*scales[:, None, None]+offsets[:, None, None], 0)
    data = np.where(valid[None], data, 0)
    bands = to_uint16(np.moveaxis(data, 0, -1), reflectance=True if tagged else None)
    if not valid.any():
        raise ValueError("The raster contains no valid pixels.")
    return bands, valid


def read_band(path, max_side=1024):
    path = validate_path(path, {".tif", ".tiff", ".jp2"})
    with rasterio.open(path) as src:
        if src.count != 1:
            raise ValueError("Choose a single-band raster for each Sentinel-2 band.")
        shape = _size(src.height, src.width, max_side)
        bands, valid = _read_scaled(src, [1], shape)
        transform = src.transform * Affine.scale(src.width/shape[1], src.height/shape[0])
        return BandData(str(path), bands[..., 0], transform, src.crs, shape[1], shape[0], valid,
                        source_kind=src.tags().get("source_kind","multispectral"),
                        original_grid=(src.width, src.height, src.crs, src.transform),source_metadata=src.tags())


def stretch_limits(arr, valid_mask=None, p_low=2., p_high=98.):
    arr = np.asarray(arr)
    if arr.ndim == 2:
        arr = arr[..., None]
    if arr.ndim != 3 or arr.size == 0 or arr.dtype.kind not in "uif" or not np.isfinite(arr).all():
        raise ValueError("Display image must contain finite real pixels")
    if not 0 <= p_low < p_high <= 100:
        raise ValueError("Stretch percentiles must satisfy 0 <= low < high <= 100")
    valid = np.ones(arr.shape[:2], bool) if valid_mask is None else np.asarray(valid_mask, bool)
    if valid.shape != arr.shape[:2] or not valid.any():
        raise ValueError("Display mask must match the image and contain valid pixels")
    limits = []
    for i in range(arr.shape[-1]):
        lo, hi = np.percentile(arr[..., i][valid], [p_low, p_high])
        if hi <= lo:
            lo, hi = 0., max(255. if arr.dtype == np.uint8 else 10000., float(hi))
        limits.append((float(lo), float(hi)))
    return np.array(limits, dtype=float)


def percentile_stretch(arr, p_low=2., p_high=98., valid_mask=None, limits=None):
    arr = np.asarray(arr)
    if arr.dtype.kind not in "uif" or arr.size == 0 or not np.isfinite(arr).all():
        raise ValueError("Display image must contain finite real pixels")
    single = arr.ndim == 2
    image = arr[..., None] if single else arr
    bounds = stretch_limits(arr, valid_mask, p_low, p_high) if limits is None else np.asarray(limits)
    if image.ndim != 3 or bounds.shape != (image.shape[-1], 2) or not np.isfinite(bounds).all() or (bounds[:, 1] <= bounds[:, 0]).any():
        raise ValueError("Invalid display stretch bounds")
    out = np.clip((image.astype(np.float32)-bounds[:, 0])/(bounds[:, 1]-bounds[:, 0])*255, 0, 255).round().astype(np.uint8)
    if valid_mask is not None:
        out[~np.asarray(valid_mask, bool)] = 0
    return out[..., 0] if single else out


def render_rgb(image, ref):
    return percentile_stretch(image[..., [2, 1, 0]], valid_mask=output_valid_mask(ref, image.shape),
                              limits=ref.display_limits)


def _photo(path, max_side):
    with warnings.catch_warnings():
        warnings.simplefilter("error", Image.DecompressionBombWarning)
        with Image.open(path) as src:
            if src.format != IMAGE_EXTENSIONS[path.suffix.lower()]:
                raise ValueError("File contents do not match the selected image extension.")
            # GDAL internal-mask IFDs are not photo frames and Pillow cannot decode them.
            # TIFF multi-page validation is handled by rasterio's subdatasets above.
            if path.suffix.lower() not in (".tif", ".tiff") and getattr(src, "n_frames", 1) != 1:
                raise ValueError("Animated images and multi-page TIFFs are unsupported. Export one still frame first.")
            if src.width*src.height > MAX_PHOTO_PIXELS:
                raise ValueError("Photo exceeds the 40-million-pixel limit. Crop it first.")
            src.load()
            img = ImageOps.exif_transpose(src)
            shape = _size(img.height, img.width, max_side)
            if img.mode in ("I", "F") or img.mode.startswith("I;16"):
                gray = percentile_stretch(to_uint16(np.asarray(img)))
                img = Image.fromarray(gray).convert("RGB")
            if "A" in img.getbands() or "transparency" in img.info:
                rgba = img.convert("RGBA").resize((shape[1], shape[0]), Image.Resampling.LANCZOS)
                valid = np.asarray(rgba.getchannel("A")) > 0
                rgb = np.asarray(Image.alpha_composite(Image.new("RGBA", rgba.size, "white"), rgba).convert("RGB")).copy()
            else:
                rgb = np.asarray(img.convert("RGB").resize((shape[1], shape[0]), Image.Resampling.LANCZOS)).copy()
                valid = np.ones(shape, bool)
            if not valid.any():
                raise ValueError("The image is completely transparent.")
            rgb[~valid] = 0
            return rgb, valid


def load_any_user_image(path, max_side=1024):
    path = validate_path(path, IMAGE_EXTENSIONS)
    _size(1, 1, max_side)
    transform, crs, original = Affine.identity(), None, None
    multispectral = False
    source_kind = "RGB with estimated NIR (illustration only)"
    source_metadata = {}
    try:
        if path.suffix.lower() in (".tif", ".tiff"):
            with rasterio.open(path) as src:
                if src.driver != "GTiff":
                    raise ValueError("TIFF extension requires TIFF file contents")
                if src.subdatasets:
                    raise ValueError("Multi-page TIFFs are unsupported; export one raster first.")
                shape = _size(src.height, src.width, max_side)
                original = (src.width, src.height, src.crs, src.transform)
                transform = src.transform * Affine.scale(src.width/shape[1], src.height/shape[0])
                crs = src.crs
                source_metadata = src.tags()
                interps = src.colorinterp
                rgb_interps = (ColorInterp.red, ColorInterp.green, ColorInterp.blue)
                photo = (ColorInterp.alpha in interps or ColorInterp.palette in interps or
                         (src.count == 4 and src.dtypes[0] == "uint8" and interps[:3] == rgb_interps))
                try:
                    with Image.open(path) as probe:
                        photo = photo or probe.mode in ("CMYK", "LA", "RGBA", "P")
                        if crs is not None and probe.getexif().get(274, 1) != 1:
                            raise ValueError("Oriented geospatial TIFF has ambiguous pixel coordinates; remove its orientation tag first.")
                except UnidentifiedImageError:
                    pass
                multispectral = src.count == 4 and not photo
                if multispectral:
                    descriptions = [str(d or "").strip().upper() for d in src.descriptions]
                    if set(descriptions) & set(BAND_ORDER) and set(descriptions) != set(BAND_ORDER):
                        raise ValueError("Satellite band descriptions must include B02,B03,B04,B08 exactly once")
                    indexes = [descriptions.index(key)+1 for key in BAND_ORDER] if set(descriptions) == set(BAND_ORDER) else [1, 2, 3, 4]
                    bgrn, valid = _read_scaled(src, indexes, shape)
                    source_kind = src.tags().get("source_kind", src.tags().get("data_source", "multispectral (B02,B03,B04,B08 order)"))
                    limits = stretch_limits(bgrn[..., [2, 1, 0]], valid)
                    rgb = percentile_stretch(bgrn[..., [2, 1, 0]], valid_mask=valid, limits=limits)
                elif not photo and src.count in (1, 3):
                    data, valid = _read_scaled(src, list(range(1, src.count+1)), shape)
                    if src.count == 1:
                        data = np.repeat(data, 3, -1)
                    if src.dtypes[0] == "uint8" and all(s == 1 for s in src.scales) and all(o == 0 for o in src.offsets):
                        rgb = data.astype(np.uint8); rgb[~valid] = 0
                    else:
                        rgb = percentile_stretch(data, valid_mask=valid)
                elif not photo:
                    raise ValueError("Use a grayscale/RGB photo TIFF or a four-band satellite TIFF.")
                if photo:
                    photo_valid = np.all(src.read_masks(out_shape=(src.count, *shape)) > 0, axis=0)
            if photo:
                rgb, valid = _photo(path, max_side)
                if rgb.shape[:2] != shape:
                    raise ValueError("Oriented geospatial TIFF has ambiguous pixel coordinates; remove its orientation tag first.")
                valid &= photo_valid
                if not valid.any():
                    raise ValueError("The TIFF contains no valid pixels")
                rgb[~valid] = 0
        else:
            rgb, valid = _photo(path, max_side)
        h, w = rgb.shape[:2]
        if not multispectral:
            r, g, b = np.moveaxis(rgb.astype(np.float32)/255, -1, 0)
            nir = np.clip(1.35*g+.30*r-.25*b, .02, 1.)
            bgrn = np.rint(np.stack([b, g, r, nir], -1)*8000+200).astype(np.uint16)
            bgrn[~valid] = 0
            limits = np.array([[200., 8200.]]*3)
        bands = {key: BandData(str(path), bgrn[..., i], transform, crs, w, h, valid, source_kind, limits, original, source_metadata)
                 for i, key in enumerate(BAND_ORDER)}
        return bands, np.ascontiguousarray(bgrn), np.ascontiguousarray(rgb), dict(
            name=path.name, width=w, height=h, source_kind=source_kind,
            resampled=bool(original and (original[0], original[1]) != (w, h)))
    except (rasterio.errors.RasterioError, OSError, SyntaxError, Image.DecompressionBombWarning, Image.DecompressionBombError) as error:
        raise ValueError(f"Cannot decode {path.name}: {error}") from error


def stack_loaded_bands(bands):
    if set(bands) != set(BAND_ORDER) or any(bands[k] is None for k in BAND_ORDER):
        raise ValueError("All four bands B02,B03,B04,B08 are required")
    ref = bands["B02"]
    for key, band in bands.items():
        if (band.width, band.height, band.crs) != (ref.width, ref.height, ref.crs) or not band.transform.almost_equals(ref.transform, precision=1e-8):
            raise ValueError(f"{key} is not aligned with B02. Reproject bands to the same grid first.")
        if ref.original_grid is not None and band.original_grid is not None and ref.original_grid != band.original_grid:
            raise ValueError(f"{key} has a different original pixel grid from B02")
    valid = np.logical_and.reduce([np.ones((b.height, b.width), bool) if b.valid_mask is None else b.valid_mask for b in bands.values()])
    if not valid.any():
        raise ValueError("The bands have no overlapping valid pixels")
    image = np.stack([bands[k].arr for k in BAND_ORDER], -1); image[~valid] = 0
    ref = replace(ref, valid_mask=valid, display_limits=stretch_limits(image[..., [2, 1, 0]], valid))
    return ref, image, render_rgb(image, ref)


def load_band_stack(paths):
    if set(paths) != set(BAND_ORDER):
        raise ValueError("Select paths for all four satellite bands")
    return stack_loaded_bands({key: read_band(paths[key]) for key in BAND_ORDER})


def output_transform(ref, shape):
    if len(shape) < 2 or min(shape[:2]) < 1 or ref.width < 1 or ref.height < 1:
        raise ValueError("Invalid raster dimensions")
    if not np.isfinite(tuple(ref.transform)).all() or ref.transform.determinant == 0:
        raise ValueError("Invalid source affine transform")
    return ref.transform * Affine.scale(ref.width/shape[1], ref.height/shape[0])


def pixel_area_m2(ref, shape):
    if ref.crs is None or not ref.crs.is_projected:
        return None
    transform = output_transform(ref, shape)
    area = abs(transform.a*transform.e-transform.b*transform.d)*ref.crs.linear_units_factor[1]**2
    if not np.isfinite(area) or area <= 0:
        raise ValueError("Invalid geospatial pixel area")
    return area


def output_valid_mask(ref, shape):
    if ref.valid_mask is None:
        return np.ones(shape[:2], dtype=bool)
    mask = np.asarray(ref.valid_mask, bool)
    if mask.shape != (ref.height, ref.width):
        raise ValueError("Source validity mask dimensions do not match the source image")
    return np.asarray(Image.fromarray(mask.astype(np.uint8)).resize(
        (shape[1], shape[0]), Image.Resampling.NEAREST)).astype(bool)


def resolve_output_path(path, allowed, default):
    path = Path(path)
    if not path.suffix:
        path = path.with_suffix(default)
    if path.suffix.lower() not in allowed:
        raise ValueError(f"Unsupported output extension. Use {', '.join(sorted(allowed))}.")
    if not path.parent.is_dir() or path.is_dir():
        raise ValueError("Output must be a file in an existing directory")
    return path


def _atomic_write(path, writer):
    fd, name = tempfile.mkstemp(prefix=".l1bsr-", suffix=path.suffix, dir=path.parent)
    os.close(fd); temp = Path(name)
    try:
        writer(temp); os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)
        Path(str(temp)+".msk").unlink(missing_ok=True)


def derived_note(model="RCAN"):
    """Label for every super-resolved export: it is a model estimate, not an observation."""
    return (f"DERIVED IMAGE: 2x super-resolution estimate by {model}. Not measured imagery; "
            "the sensor's native resolution is unchanged and added detail may be inaccurate.")


def _image_metadata(suffix, description):
    """PNG text chunks or JPEG EXIF ImageDescription carrying the derived-image label."""
    if not description:
        return {}
    if suffix == ".png":
        info = PngInfo()
        info.add_text("Description", description)
        info.add_text("Derived", "true")
        return {"pnginfo": info}
    exif = Image.Exif()
    exif[0x010E] = description  # ImageDescription
    return {"exif": exif}


def save_image(path, rgb, valid_mask=None, description=None):
    path = resolve_output_path(path, {".png", ".jpg", ".jpeg"}, ".png")
    rgb = np.asarray(rgb)
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.size == 0:
        raise ValueError("Visual exports require a non-empty RGB uint8 image")
    image = Image.fromarray(rgb)
    if valid_mask is not None and path.suffix.lower() == ".png":
        valid = np.asarray(valid_mask, bool)
        if valid.shape != rgb.shape[:2]:
            raise ValueError("Output mask shape mismatch")
        if not valid.all():
            image.putalpha(Image.fromarray(valid.astype(np.uint8)*255))
    suffix = ".png" if path.suffix.lower() == ".png" else ".jpg"
    options = dict(_image_metadata(suffix, description), **({"quality":95, "subsampling":0} if suffix == ".jpg" else {}))
    _atomic_write(path, lambda temp: image.save(temp, format="PNG" if suffix == ".png" else "JPEG", **options))
    return path


def save_text(path, text):
    path = Path(path)
    if not path.parent.is_dir() or path.is_dir():
        raise ValueError("Output must be a file in an existing directory")
    _atomic_write(path, lambda temp: temp.write_text(text, encoding="utf-8"))
    return path


def _atomic_bundle(items):
    """Stage every file before committing and restore the previous bundle on failure."""
    paths = [Path(path) for path, _ in items]
    if len(set(paths)) != len(paths) or len({path.parent.resolve() for path in paths}) != 1:
        raise ValueError("Bundle paths must be distinct files in the same directory")
    parent = paths[0].parent.resolve()
    if not parent.is_dir() or any(path.exists() and not path.is_file() for path in paths):
        raise ValueError("Bundle output paths must be files in an existing directory")
    with tempfile.TemporaryDirectory(prefix=".l1bsr-export-",dir=parent) as name:
        workspace = Path(name).resolve()
        if workspace.parent != parent:
            raise ValueError("Invalid export workspace")
        staged = []
        for i, (path, writer) in enumerate(items):
            temp = workspace/(str(i)+Path(path).suffix)
            writer(temp); staged.append(temp)
        moved, committed = [], []
        try:
            for i, path in enumerate(paths):
                if path.exists():
                    backup = workspace/("backup-"+str(i))
                    os.replace(path,backup); moved.append((backup,path))
            for path,temp in zip(paths,staged):
                os.replace(temp,path); committed.append(path)
        except Exception:
            for path in committed:
                path.unlink(missing_ok=True)
            for backup,path in moved:
                os.replace(backup,path)
            raise


def write_geotiff(out_path, data, ref, scale=None, sr_model="RCAN"):
    path = resolve_output_path(out_path, {".tif", ".tiff"}, ".tif")
    data = np.asarray(data)
    if ref.crs is None:
        raise ValueError("This image has no CRS. Save PNG/JPEG or supply georeferenced satellite bands.")
    if data.dtype != np.uint16 or data.ndim != 3 or data.shape[2] != 4 or data.size == 0:
        raise ValueError("Enhanced GeoTIFF must be HxWx4 uint16")
    valid = output_valid_mask(ref, data.shape)
    out_transform = output_transform(ref, data.shape)
    units = "m" if ref.crs.is_projected else "degrees"
    derived = dict(TIFFTAG_IMAGEDESCRIPTION=derived_note(sr_model), derived_product="true",
                   product_type="AI super-resolved estimate (not measured reflectance at this pixel size)",
                   sr_model=str(sr_model),
                   native_pixel_size=f"{abs(ref.transform.a):g} {units}",
                   output_pixel_size=f"{abs(out_transform.a):g} {units}")
    def write(temp):
        with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True):
            with rasterio.open(temp, "w", driver="GTiff", height=data.shape[0], width=data.shape[1],
                               count=4, dtype="uint16", crs=ref.crs, transform=out_transform, compress="lzw") as dst:
                bands = np.moveaxis(data, -1, 0).copy(); bands[:, ~valid] = 0
                dst.write(bands); dst.write_mask(valid.astype(np.uint8)*255)
                dst.scales = (.0001,)*4; dst.offsets = (0.,)*4
                for i, key in enumerate(BAND_ORDER):
                    dst.set_band_description(i+1, key)
                source = {key:value for key,value in (ref.source_metadata or {}).items() if key in
                          ("satellite","acquisition_datetime","stac_item_id","stac_item_url","attribution","location")}
                dst.update_tags(**source,source_kind=ref.source_kind,calibration="reflectance = stored value / 10000",
                                **derived)
    _atomic_write(path, write)
    Path(str(path)+".msk").unlink(missing_ok=True)
    return path


def write_single_band_geotiff(out_path, data_u8, ref, scale=None):
    path = resolve_output_path(out_path, {".tif", ".tiff"}, ".tif")
    if ref.crs is None:
        raise ValueError("This image has no CRS. Save a PNG map instead.")
    data = np.asarray(data_u8)
    if data.dtype != np.uint8 or data.ndim != 2 or data.size == 0 or not np.isin(data, [0, 1, 2, 3, 4, 5, 255]).all():
        raise ValueError("Land-cover GeoTIFF needs a uint8 class map with IDs 0..5 or 255")
    data = data.copy(); data[~output_valid_mask(ref, data.shape)] = 255
    def write(temp):
        with rasterio.open(temp, "w", driver="GTiff", height=data.shape[0], width=data.shape[1],
                           count=1, dtype="uint8", crs=ref.crs, transform=output_transform(ref, data.shape), compress="lzw", nodata=255) as dst:
            dst.write(data, 1)
            dst.write_colormap(1, {0:(230,57,70,255), 1:(73,80,87,255), 2:(42,157,143,255), 3:(29,53,87,255), 4:(233,196,106,255), 5:(164,139,191,255), 255:(0,0,0,0)})
            source = {key:value for key,value in (ref.source_metadata or {}).items() if key in
                      ("satellite","acquisition_datetime","stac_item_id","stac_item_url","attribution","location")}
            dst.update_tags(**source,source_kind=ref.source_kind)
            dst.update_tags(class_names="0 Buildings;1 Roads;2 Vegetation;3 Water;4 Bare Land;5 Developed space;255 NoData")
    _atomic_write(path, write)
    Path(str(path)+".msk").unlink(missing_ok=True)
    return path


def save_jpeg_with_jgw(path, rgb, ref, description=None):
    path = resolve_output_path(path, {".jpg", ".jpeg"}, ".jpg")
    if ref.crs is None:
        raise ValueError("World files require georeferenced inputs")
    transform = output_transform(ref, rgb.shape)
    xc, yc = transform*(.5, .5)
    world = [transform.a, transform.d, transform.b, transform.e, xc, yc]
    rgb = np.asarray(rgb)
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.size == 0:
        raise ValueError("JPEG output requires RGB uint8")
    world_text = "".join(f"{v:.12f}\n" for v in world)
    metadata = _image_metadata(".jpg", description)
    _atomic_bundle([(path,lambda temp:Image.fromarray(rgb).save(temp,format="JPEG",quality=95,subsampling=0,**metadata)),
                    (path.with_suffix(".jgw"),lambda temp:temp.write_text(world_text,encoding="utf-8")),
                    (path.with_suffix(".prj"),lambda temp:temp.write_text(ref.crs.to_wkt(),encoding="utf-8"))])
    return path
