"""Download native-resolution Sentinel-2 L2A crops from public Earth Search COGs."""
from _bootstrap import ROOT  # also puts src/ on sys.path
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tempfile
import urllib.parse
import urllib.request
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import transform, transform_bounds
from rasterio.windows import Window, from_bounds
from satsr.geo_io import load_any_user_image

API = "https://earth-search.aws.element84.com/v1"
COLLECTION = "sentinel-2-l2a"
BANDS = {"B02":"blue", "B03":"green", "B04":"red", "B08":"nir"}
SITES = [
    dict(slug="urban_delhi", category="urban", name="Central New Delhi", lon=77.209, lat=28.614),
    dict(slug="agriculture_punjab", category="agriculture", name="Farmland east of Ludhiana, Punjab", lon=75.984, lat=30.947),
    dict(slug="vegetation_western_ghats", category="vegetation", name="Western Ghats near Koyna", lon=73.724, lat=17.956),
    dict(slug="water_pavana", category="water", name="Pavana Reservoir, Maharashtra", lon=73.481, lat=18.670),
    dict(slug="mixed_pune", category="mixed", name="Pune Mula-Mutha river and surrounding city", lon=73.883, lat=18.535),
]
GDAL_OPTIONS = dict(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
                    CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif", GDAL_HTTP_CONNECTTIMEOUT="15",
                    GDAL_HTTP_TIMEOUT="45", GDAL_HTTP_MAX_RETRY="2", GDAL_HTTP_RETRY_DELAY="1")


def get_json(url):
    request = urllib.request.Request(url, headers={"User-Agent":"L1BSR-Sentinel-demo/1.0"})
    with urllib.request.urlopen(request, timeout=45) as response:
        return json.load(response)


def candidates(site):
    parameters = dict(collections=COLLECTION, bbox=",".join(str(v) for v in
        (site["lon"]-.015,site["lat"]-.015,site["lon"]+.015,site["lat"]+.015)),
        datetime="2023-11-01T00:00:00Z/2024-02-29T23:59:59Z", limit=100,
        query=json.dumps({"eo:cloud_cover":{"lt":5}}))
    result = get_json(API+"/search?"+urllib.parse.urlencode(parameters))
    return sorted(result["features"], key=lambda item:(item["properties"]["eo:cloud_cover"],item["id"]))


def asset_calibration(item, asset):
    band = asset["raster:bands"][0]
    scale, offset = float(band.get("scale",1)), float(band.get("offset",0))
    # Legacy Earth Search items can retain the original JP2 offset in raster:bands
    # after their COG DN values have already been offset-corrected. Do not subtract twice.
    if item["properties"].get("earthsearch:boa_offset_applied") is True:
        offset = 0.
    if scale != .0001 or not np.isfinite(offset):
        raise ValueError("Unexpected Sentinel-2 reflectance calibration")
    return scale, offset


def read_crop(site, item, size):
    with rasterio.Env(**GDAL_OPTIONS):
        with rasterio.open(item["assets"]["blue"]["href"]) as src:
            if src.crs is None or tuple(src.res) != (10.,10.):
                raise ValueError("Expected projected 10 m Sentinel-2 source")
            x,y = transform("EPSG:4326",src.crs,[site["lon"]],[site["lat"]])
            row,col = src.index(x[0],y[0])
            window = Window(col-size//2,row-size//2,size,size)
            if min(window.col_off,window.row_off) < 0 or window.col_off+size > src.width or window.row_off+size > src.height:
                raise ValueError("Crop crosses the source tile boundary")
            crs, grid = src.crs, src.transform
            cropped_transform = src.window_transform(window)
            bounds = rasterio.windows.bounds(window,grid)
            with rasterio.open(item["assets"]["scl"]["href"]) as scl_source:
                if scl_source.crs != crs:
                    raise ValueError("SCL grid CRS does not match reflectance bands")
                scl = scl_source.read(1,window=from_bounds(*bounds,transform=scl_source.transform),
                                      out_shape=(size,size),resampling=Resampling.nearest)
            usable = np.isin(scl,[2,4,5,6,7])
            if usable.mean() < .98:
                raise ValueError(f"Only {usable.mean():.1%} clear pixels in crop")
        def read_band(key):
            asset = item["assets"][BANDS[key]]
            with rasterio.Env(**GDAL_OPTIONS):
                with rasterio.open(asset["href"]) as band:
                    if band.crs != crs or band.transform != grid or band.count != 1:
                        raise ValueError("Source bands do not share one native grid")
                    values = band.read(1,window=window)
                    valid = band.read_masks(1,window=window) > 0
            scale, offset = asset_calibration(item,asset)
            return values, valid, scale, offset
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(read_band,BANDS))
        data = np.stack([r[0] for r in results])
        valid = usable & np.logical_and.reduce([r[1] for r in results])
        if data.dtype != np.uint16 or data.shape != (4,size,size) or valid.mean() < .98:
            raise ValueError("Invalid or mostly missing reflectance crop")
        return data, valid, scl, crs, cropped_transform, window, results


def write_crop(path, data, valid, crs, grid, scales, offsets, tags, descriptions):
    with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True):
        with rasterio.open(path,"w",driver="GTiff",count=data.shape[0],height=data.shape[1],width=data.shape[2],
                           dtype=data.dtype,crs=crs,transform=grid,compress="deflate",predictor=2) as dst:
            dst.write(data); dst.write_mask(valid.astype(np.uint8)*255)
            dst.scales = tuple(scales); dst.offsets = tuple(offsets)
            dst.update_tags(**tags)
            for i,key in enumerate(descriptions):
                dst.set_band_description(i+1,key)


def download(output=None, size=256):
    output = Path(output or ROOT/"demo_images").resolve()
    if not isinstance(size,int) or size < 64 or size > 512 or size%2:
        raise ValueError("Crop size must be an even integer from 64 to 512")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output folder must be empty; existing data will not be overwritten")
    output.parent.mkdir(parents=True,exist_ok=True)
    with ThreadPoolExecutor(max_workers=5) as pool:
        search_results = list(pool.map(candidates,SITES))
    records = []
    with tempfile.TemporaryDirectory(prefix=".sentinel-demo-",dir=output.parent) as directory:
        stage = Path(directory).resolve()
        if stage.parent != output.parent:
            raise ValueError("Invalid download staging folder")
        (stage/"sources").mkdir(); (stage/"quality").mkdir(); (stage/"satellite_bands").mkdir()
        for site,items in zip(SITES,search_results):
            last_error = "No low-cloud scenes found"
            for item in items[:12]:
                print(f"Reading {site['slug']}: {item['id']}...",flush=True)
                try:
                    data,valid,scl,crs,grid,window,raw = read_crop(site,item,size)
                    break
                except (ValueError,rasterio.errors.RasterioError) as error:
                    last_error = str(error); print("Skipping candidate: "+last_error,flush=True)
            else:
                raise ValueError(f"No usable source for {site['slug']}: {last_error}")
            date = item["properties"]["datetime"][:10]
            filename = f"sentinel2_{site['slug']}_{date}.tif"
            url = API+f"/collections/{COLLECTION}/items/{item['id']}"
            source_kind = "real Sentinel-2 L2A surface reflectance; measured B02/B03/B04/B08"
            tags = dict(source_kind=source_kind,satellite=item["properties"].get("platform","sentinel-2"),
                        acquisition_datetime=item["properties"]["datetime"],stac_item_id=item["id"],
                        stac_item_url=url,attribution="Contains modified Copernicus Sentinel data (2023/2024)",
                        crop_processing="Native 10 m integer window; original COG DN preserved",
                        calibration="physical reflectance = stored DN * per-band scale + per-band offset",
                        boa_offset_already_applied=str(item["properties"].get("earthsearch:boa_offset_applied",False)),
                        location=site["name"],quality_mask="SCL 2,4,5,6,7 plus original band validity")
            scales = [r[2] for r in raw]; offsets = [r[3] for r in raw]
            write_crop(stage/filename,data,valid,crs,grid,scales,offsets,tags,list(BANDS))
            bands_folder = stage/"satellite_bands"/site["slug"]; bands_folder.mkdir()
            for i,key in enumerate(BANDS):
                write_crop(bands_folder/(key+".tif"),data[i:i+1],valid,crs,grid,scales[i:i+1],offsets[i:i+1],tags,[key])
            write_crop(stage/"quality"/(site["slug"]+"_SCL.tif"),scl[None],np.ones(valid.shape,bool),crs,grid,[1],[0],
                       dict(source_kind="real Sentinel-2 scene classification; nearest-neighbor 20 m to 10 m quality mask"),["SCL"])
            (stage/"sources"/(site["slug"]+".stac.json")).write_text(json.dumps(item,indent=2),encoding="utf-8")
            loaded = load_any_user_image(stage/filename)
            bbox = transform_bounds(crs,"EPSG:4326",*rasterio.transform.array_bounds(size,size,grid))
            record = dict(file=filename,format="TIFF",loaded_width=size,loaded_height=size,source_kind=source_kind,
                valid_pixels=int(valid.sum()),category=site["category"],location=site["name"],center_lon=site["lon"],center_lat=site["lat"],
                acquisition_datetime=item["properties"]["datetime"],platform=tags["satellite"],
                stac_item_id=item["id"],stac_item_url=url,stac_metadata_file="sources/"+site["slug"]+".stac.json",
                source_assets={key:item["assets"][asset]["href"] for key,asset in BANDS.items()},
                scl_asset=item["assets"]["scl"]["href"],quality_file="quality/"+site["slug"]+"_SCL.tif",
                single_band_directory="satellite_bands/"+site["slug"],crs=crs.to_string(),transform=list(grid)[:6],bbox_wgs84=list(bbox),
                native_pixel_size_m=10,band_order=list(BANDS),scales=scales,offsets=offsets,
                boa_offset_already_applied=item["properties"].get("earthsearch:boa_offset_applied",False),
                original_stac_offsets=[item["assets"][a]["raster:bands"][0].get("offset",0) for a in BANDS.values()],
                calibration_policy="Use STAC scale; offset zero when earthsearch:boa_offset_applied is true; otherwise STAC offset",
                source_window=dict(col_off=int(window.col_off),row_off=int(window.row_off),width=size,height=size),
                source_dn_sha256=hashlib.sha256(data.tobytes()).hexdigest(),
                sha256=hashlib.sha256((stage/filename).read_bytes()).hexdigest(),
                scene_cloud_cover_percent=item["properties"]["eo:cloud_cover"],clear_crop_percent=round(valid.mean()*100,3),
                downloaded_at_utc=datetime.now(timezone.utc).isoformat(),
                attribution=tags["attribution"],license_url="https://sentinels.copernicus.eu/documents/247904/690755/Sentinel_Data_Legal_Notice")
            if loaded[1].shape != (size,size,4):
                raise ValueError("Application did not load crop correctly")
            records.append(record)
            print(f"Saved {filename}: {valid.mean():.1%} clear; {crs}; {size}x{size}",flush=True)
        (stage/"manifest.json").write_text(json.dumps(records,indent=2)+"\n",encoding="utf-8")
        lines = ["# Real Sentinel-2 demonstration dataset", "", "Five genuine Copernicus Sentinel-2 L2A crops, downloaded from public Earth Search COGs. No generated scenes, screenshots or ordinary photographs are used in this folder.", "",
                 "| File | Area | Acquisition (UTC) | Native resolution |", "| --- | --- | --- | --- |"]
        lines += [f"| {r['file']} | {r['location']} | {r['acquisition_datetime'][:10]} | 10 m |" for r in records]
        lines += ["", "Each GeoTIFF contains measured B02/B03/B04/B08 uint16 values, original UTM CRS and native pixel grid, band descriptions, scale/offset calibration and an internal validity mask. The crop preserves the source DN values; the app converts them to reflectance times 10000. Native pixels are cropped without image synthesis or resampling.", "",
                  "Open a top-level TIFF and press Run all. Downloads supports images, maps, scores, reports, GeoTIFF and ZIP. Separate original bands are in satellite_bands/<area>/. quality/ contains actual SCL data used only for quality screening; SCL is not a five-class ground-truth label map.", "",
                  "manifest.json contains acquisition dates, source asset URLs, crop windows, calibration, CRS, footprints and SHA-256 hashes. sources/ contains the original STAC metadata. Crops have at least 98% usable pixels after SCL/no-data screening.", "",
                  "Effective calibration accounts for earthsearch:boa_offset_applied, so an offset already applied to the COG pixels is not subtracted again. Both the original STAC offset and the effective offset are recorded.", "",
                  "Source: [Sentinel-2 COGs on AWS](https://registry.opendata.aws/sentinel-2-l2a-cogs/) via [Earth Search](https://github.com/Element84/earth-search). Contains modified Copernicus Sentinel data (2023/2024). See each manifest record for the legal notice and source item.", "",
                  "These are real inference/workflow test inputs. Included model weights are still trained on synthetic data: predictions are unvalidated on these locations. No HR reference or independent land-cover labels are provided, so this dataset is not an accuracy benchmark or supervised training dataset.", "",
                  "Re-download to an EMPTY folder: python download_sentinel_demo.py --output new_demo_images. Source selection is recorded in manifest.json; a new search can select another acquisition if the archive changes."]
        (stage/"README.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
        if output.exists():
            output.rmdir()  # Only the previously checked empty directory; never recursive.
        stage.rename(output)
    print(f"Ready: {len(records)} real satellite crops in {output}",flush=True)
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",default=str(ROOT/"demo_images"))
    parser.add_argument("--size",type=int,default=256)
    args = parser.parse_args()
    download(args.output,args.size)
