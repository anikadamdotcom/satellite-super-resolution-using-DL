"""Prepare aligned four-band GeoTIFF patches in split/{hr,lr,masks}/NAME.tif."""
import _bootstrap  # noqa: F401  (puts src/ on sys.path)
import argparse
from pathlib import Path
import tempfile
import numpy as np
import rasterio
from rasterio.enums import ColorInterp
from rasterio.transform import Affine
from satsr.data import degrade
from satsr.geo_io import _read_scaled, BAND_ORDER


def read_image(path):
    with rasterio.open(path) as src:
        if src.count != 4 or ColorInterp.alpha in src.colorinterp:
            raise ValueError(f"{path}: expected four measured bands, not RGBA")
        if src.width*src.height > 1_048_576:
            raise ValueError(f"{path}: crop training patches no larger than 1,048,576 pixels")
        if not (src.read_masks() > 0).all():
            raise ValueError(f"{path}: crop valid imagery before preparing training data")
        descriptions = [str(d or "").upper() for d in src.descriptions]
        indexes = [descriptions.index(key)+1 for key in BAND_ORDER] if set(descriptions) == set(BAND_ORDER) else [1,2,3,4]
        image, valid = _read_scaled(src,indexes,(src.height,src.width))
        if not valid.all():
            raise ValueError(f"{path}: imagery includes non-finite values")
        return image, src.crs, src.transform


def prepare(input_dir, output_dir):
    source, output = Path(input_dir).resolve(), Path(output_dir).resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("Input and output directories must be separate")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError("Choose an absent or empty output directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with tempfile.TemporaryDirectory(prefix=".l1bsr-prepare-",dir=output.parent) as name:
        workspace = Path(name).resolve()
        if workspace.parent != output.parent:
            raise ValueError("Invalid preparation workspace")
        stage = workspace/"prepared"; stage.mkdir()
        for split in ("train","val","test"):
            folder = source/split
            hr_folder = folder/"hr"
            files = sorted(p for p in hr_folder.iterdir() if p.is_file() and p.suffix.lower() in (".tif",".tiff")) if hr_folder.is_dir() else []
            if not files:
                raise ValueError(f"No HR TIFF scenes in {hr_folder}")
            destination = stage/split; destination.mkdir()
            for path in files:
                hr, crs, transform = read_image(path)
                if min(hr.shape[:2]) < 16 or any(n % 2 for n in hr.shape[:2]):
                    raise ValueError(f"{path}: HR dimensions must be even and at least 16")
                with rasterio.open(folder/"masks"/path.name) as src:
                    if src.count != 1 or src.crs != crs or not src.transform.almost_equals(transform,precision=1e-8):
                        raise ValueError(f"{path}: mask must share the HR CRS and transform")
                    mask = src.read(1)
                    if mask.dtype.kind not in "ui" or mask.shape != hr.shape[:2]:
                        raise ValueError(f"{path}: mask must be an integer array matching HR")
                    mask = np.where(src.read_masks(1) > 0,mask,255)
                    if not np.isin(mask,[0,1,2,3,4,255]).all() or not (mask != 255).any():
                        raise ValueError(f"{path}: masks need IDs 0..4 and 255 for ignored labels")
                lr_path = folder/"lr"/path.name
                if lr_path.exists():
                    lr, lr_crs, lr_transform = read_image(lr_path)
                    if lr_crs != crs or not lr_transform.almost_equals(transform*Affine.scale(2,2),precision=1e-8):
                        raise ValueError(f"{lr_path}: LR must exactly cover the HR footprint")
                    if hr.shape[:2] != (lr.shape[0]*2,lr.shape[1]*2):
                        raise ValueError(f"{lr_path}: HR must be exactly 2x LR dimensions")
                else:
                    lr = degrade(hr)
                result = destination/(path.stem+".npz")
                if result.exists():
                    raise ValueError("Duplicate scene stems within a split")
                np.savez_compressed(result,hr=hr,lr=lr,mask=mask.astype(np.uint8)); count += 1
        if output.exists():
            output.rmdir()  # Only the verified, empty user-selected output directory.
        stage.rename(output)
    print(f"Prepared {count} scenes in {output}")
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",required=True); parser.add_argument("--output",required=True)
    args = parser.parse_args()
    prepare(args.input,args.output)


if __name__ == "__main__":
    main()
