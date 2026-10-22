# Offline showcase results

These outputs use the five genuine Sentinel-2 inputs and CPU inference.
Land cover uses real-pretrained OpenEarthMap weights and measured-NIR checks. RCAN is trained on real Sentinel-2/VENµS pairs (small measured gain over bicubic).
Road/building accuracy at Sentinel-2's 10 m resolution is unvalidated. Scores are not calibrated accuracy.

![Original, enhanced, predicted map and overlay](overview.png)

Each scene folder includes original/enhanced PNGs, maps, overlay, scores, arrays, GeoTIFFs, report and results.zip.
The ZIPs are backups for a live demonstration. summary.json records predicted shares without inventing accuracy.

See ../SHOWCASE_GUIDE.md for the live demonstration and presentation wording.
