"""Download a small real Sentinel-2 + ESA WorldCover land-cover benchmark for India.

Each site gives one 512 x 512 native 10 m Sentinel-2 L2A crop (B02, B03, B04, B08)
from the 2021-22 dry season, plus the ESA WorldCover 2021 v200 map resampled onto
the same 10 m grid. Splits are fixed per site, so train, validation and test
areas never overlap.

WorldCover is itself a model output (about 75% accurate globally), so scores
against it measure agreement with WorldCover, not absolute truth.

    python download_mapping_benchmark.py --output datasets/mapping_benchmark
"""
from _bootstrap import ROOT  # also puts src/ on sys.path
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import urllib.parse

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject, transform_bounds

from download_sentinel_demo import API, COLLECTION, GDAL_OPTIONS, get_json, read_crop

WORLDCOVER = "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/ESA_WorldCover_10m_2021_v200_{tile}_Map.tif"
DATES = "2021-11-01T00:00:00Z/2022-03-15T23:59:59Z"

# Benchmark classes. WorldCover codes -> project benchmark IDs.
CLASSES = ["Built-up", "Tree cover", "Shrub / grass", "Cropland", "Water / wetland", "Bare / sparse"]
WORLDCOVER_TO_CLASS = {50: 0, 10: 1, 20: 2, 30: 2, 40: 3, 80: 4, 90: 4, 95: 4, 60: 5, 70: 5, 100: 5}
IGNORE = 255

# (slug, kind, lon, lat, split). Split is chosen per site to cover every kind.
SITES = [
    ("mumbai", "urban", 72.877, 19.076, "train"), ("bengaluru", "urban", 77.594, 12.972, "train"),
    ("kolkata", "urban", 88.364, 22.573, "train"), ("chennai", "urban", 80.270, 13.083, "train"),
    ("ahmedabad", "urban", 72.571, 23.023, "train"), ("jaipur", "urban", 75.787, 26.912, "train"),
    ("pune", "urban", 73.856, 18.520, "train"), ("bhopal", "urban", 77.413, 23.260, "train"),
    ("hyderabad", "urban", 78.487, 17.385, "test"), ("lucknow", "urban", 80.946, 26.847, "test"),
    ("nagpur", "urban", 79.088, 21.146, "val"),
    ("ludhiana_fields", "agriculture", 75.984, 30.947, "train"), ("meerut_rural", "agriculture", 77.800, 28.980, "train"),
    ("indore_rural", "agriculture", 75.700, 22.600, "train"), ("thanjavur", "agriculture", 79.140, 10.790, "train"),
    ("nashik", "agriculture", 73.950, 20.050, "train"), ("guntur", "agriculture", 80.450, 16.300, "test"),
    ("bathinda", "agriculture", 74.950, 30.210, "test"), ("karnal", "agriculture", 76.990, 29.690, "val"),
    ("koyna", "forest", 73.724, 17.956, "train"), ("kanha", "forest", 80.610, 22.330, "train"),
    ("nilgiris", "forest", 76.700, 11.400, "train"), ("wayanad", "forest", 76.080, 11.690, "test"),
    ("similipal", "forest", 86.330, 21.850, "val"),
    ("vembanad", "water", 76.380, 9.600, "train"), ("sundarbans", "water", 88.900, 21.950, "train"),
    ("sambhar", "water", 75.100, 26.950, "train"), ("hirakud", "water", 83.870, 21.530, "test"),
    ("chilika", "water", 85.320, 19.710, "val"),
    ("rann_of_kutch", "bare", 70.000, 23.900, "train"), ("leh", "bare", 77.580, 34.150, "train"),
    ("jaisalmer", "bare", 70.910, 26.920, "test"),
    ("guwahati", "mixed", 91.740, 26.140, "train"), ("dehradun", "mixed", 78.030, 30.320, "train"),
    ("bhubaneswar", "mixed", 85.820, 20.300, "test"), ("coimbatore", "mixed", 76.960, 11.020, "val"),
    ("varanasi", "mixed", 83.000, 25.320, "train"),
]


def worldcover_tile(lon, lat):
    """WorldCover 3x3 degree tile name from its lower-left corner, e.g. N27E075."""
    lat0, lon0 = int(np.floor(lat/3)*3), int(np.floor(lon/3)*3)
    return f"{'N' if lat0 >= 0 else 'S'}{abs(lat0):02d}{'E' if lon0 >= 0 else 'W'}{abs(lon0):03d}"


def search(lon, lat):
    parameters = dict(collections=COLLECTION, bbox=f"{lon-.03},{lat-.03},{lon+.03},{lat+.03}",
                      datetime=DATES, limit=100, query=json.dumps({"eo:cloud_cover": {"lt": 5}}))
    result = get_json(API+"/search?"+urllib.parse.urlencode(parameters))
    return sorted(result["features"], key=lambda item: (item["properties"]["eo:cloud_cover"], item["id"]))


def read_labels(crs, grid, size, bounds_wgs84):
    """Resample every WorldCover tile touching the crop onto the crop's 10 m grid (mode)."""
    west, south, east, north = bounds_wgs84
    tiles = {worldcover_tile(x, y) for x in (west, east) for y in (south, north)}
    codes = np.zeros((size, size), np.uint8)
    with rasterio.Env(**GDAL_OPTIONS):
        for tile in sorted(tiles):
            with rasterio.open(WORLDCOVER.format(tile=tile)) as src:
                part = np.zeros((size, size), np.uint8)
                reproject(rasterio.band(src, 1), part, dst_transform=grid, dst_crs=crs,
                          resampling=Resampling.mode, src_nodata=0, dst_nodata=0)
                codes = np.where(codes == 0, part, codes)
    labels = np.full(codes.shape, IGNORE, np.uint8)
    for code, cls in WORLDCOVER_TO_CLASS.items():
        labels[codes == code] = cls
    return labels, sorted(tiles)


def download_site(site, output, size):
    slug, kind, lon, lat, split = site
    target = output/f"{slug}.npz"
    if target.exists():
        print(f"{slug}: already downloaded", flush=True)
        return json.loads(str(np.load(target)["record"]))
    error = "no low-cloud scenes"
    for item in search(lon, lat)[:15]:
        try:
            data, valid, scl, crs, grid, window, _ = read_crop(dict(lon=lon, lat=lat), item, size)
            break
        except (ValueError, rasterio.errors.RasterioError) as exc:
            error = str(exc)
    else:
        print(f"{slug}: SKIPPED ({error})", flush=True)
        return None
    bounds = transform_bounds(crs, "EPSG:4326", *rasterio.transform.array_bounds(size, size, grid))
    labels, tiles = read_labels(crs, grid, size, bounds)
    labels[~valid] = IGNORE
    image = np.moveaxis(data, 0, -1)  # H x W x B02,B03,B04,B08, reflectance x 10000
    image[~valid] = 0
    record = dict(slug=slug, kind=kind, split=split, lon=lon, lat=lat, size=size,
                  stac_item_id=item["id"], acquisition_datetime=item["properties"]["datetime"],
                  cloud_cover=item["properties"]["eo:cloud_cover"], crs=crs.to_string(), transform=list(grid)[:6],
                  worldcover_tiles=tiles, clear_fraction=float(valid.mean()),
                  class_fractions={name: float((labels == i).mean()) for i, name in enumerate(CLASSES)},
                  image_sha256=hashlib.sha256(image.tobytes()).hexdigest())
    np.savez_compressed(target, image=image, labels=labels, valid=valid, record=json.dumps(record))
    print(f"{slug} ({split}): {item['id']} " + ", ".join(f"{n} {v:.0%}" for n, v in record["class_fractions"].items() if v > .02), flush=True)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", default=str(ROOT/"datasets"/"mapping_benchmark"))
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        records = [r for r in pool.map(lambda s: download_site(s, output, args.size), SITES) if r]
    manifest = dict(classes=CLASSES, worldcover_mapping={str(k): v for k, v in WORLDCOVER_TO_CLASS.items()},
                    ignore_index=IGNORE, dates=DATES, sites=records,
                    attribution=["Contains modified Copernicus Sentinel data (2021/2022)",
                                 "ESA WorldCover project 2021 / Contains modified Copernicus Sentinel data (2021) processed by ESA WorldCover consortium, CC BY 4.0"])
    (output/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
    counts = {s: sum(r["split"] == s for r in records) for s in ("train", "val", "test")}
    print(f"Ready: {len(records)} of {len(SITES)} sites in {output} ({counts})", flush=True)


if __name__ == "__main__":
    main()
