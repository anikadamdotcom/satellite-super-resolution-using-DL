# Real Sentinel-2 demonstration dataset

Five genuine Copernicus Sentinel-2 L2A crops, downloaded from public Earth Search COGs. No generated scenes, screenshots or ordinary photographs are used in this folder.

| File | Area | Acquisition (UTC) | Native resolution |
| --- | --- | --- | --- |
| sentinel2_urban_delhi_2023-12-05.tif | Central New Delhi | 2023-12-05 | 10 m |
| sentinel2_agriculture_punjab_2023-12-03.tif | Farmland east of Ludhiana, Punjab | 2023-12-03 | 10 m |
| sentinel2_vegetation_western_ghats_2023-12-30.tif | Western Ghats near Koyna | 2023-12-30 | 10 m |
| sentinel2_water_pavana_2024-02-28.tif | Pavana Reservoir, Maharashtra | 2024-02-28 | 10 m |
| sentinel2_mixed_pune_2024-02-28.tif | Pune Mula-Mutha river and surrounding city | 2024-02-28 | 10 m |

Each GeoTIFF contains measured B02/B03/B04/B08 uint16 values, original UTM CRS and native pixel grid, band descriptions, scale/offset calibration and an internal validity mask. The source COGs have earthsearch:boa_offset_applied=true, so their effective offset is zero despite the original JP2 offset of -0.1 in the catalog. It is not subtracted again. The crop preserves source COG DN values; the app converts them to reflectance times 10000. Native pixels are cropped without image synthesis or resampling.

Open a top-level TIFF and press Run all. Downloads supports images, maps, scores, reports, GeoTIFF and ZIP. Separate original bands are in satellite_bands/<area>/. quality/ contains actual SCL data used only for quality screening; SCL is not a five-class ground-truth label map.

manifest.json contains acquisition dates, source asset URLs, crop windows, calibration, CRS, footprints and SHA-256 hashes. sources/ contains the original STAC metadata. Crops have at least 98% usable pixels after SCL/no-data screening.

Source: [Sentinel-2 COGs on AWS](https://registry.opendata.aws/sentinel-2-l2a-cogs/) via [Earth Search](https://github.com/Element84/earth-search). Contains modified Copernicus Sentinel data (2023/2024). See each manifest record for the legal notice and source item.

These are real inference/workflow test inputs. The default land-cover model is pretrained on real fine-resolution OpenEarthMap imagery, with measured-NIR consistency checks. RCAN is trained on real Sentinel-2/VENµS pairs; the optional U-Net remains synthetic-trained. Independent classification accuracy is unvalidated on these locations; see ../docs/LAND_COVER.md. No HR reference or independent land-cover labels are provided, so this dataset is not an accuracy benchmark or supervised training dataset.

Re-download to an EMPTY folder: python scripts/download_sentinel_demo.py --output new_demo_images. Source selection is recorded in manifest.json; a new search can select another acquisition if the archive changes.
