# Real-imagery detection update

**Latest update:** the default combines this SparseMask detector with a Hugging Face SegFormer. See [HF_MODEL.md](HF_MODEL.md) and `hf_accuracy_report.json`: the fixed LoveDA audit improved from 77.17% to 79.80%, below the requested 85%. The results below describe the preceding SparseMask-only comparison, not the new combined model.

The previous real detector uses **OpenEarthMap SparseMask/MobileNetV2**, pretrained on real manually annotated aerial/satellite imagery. Its eight source classes are correctly mapped to buildings, roads, vegetation (rangeland/trees/crops), water, bare land and a separate developed-space class. Parking/developed surfaces are not silently called roads or buildings. The original synthetic U-Net remains available for the reproducible training experiment.

The [official OpenEarthMap code page](https://open-earth-map.org/code.html) links the [model author's implementation and pretrained weights](https://github.com/cliffbb/oem-lightweight). The MIT architecture, license, converted state dictionary, source URLs, pinned revision and original/converted SHA-256 hashes are bundled. No additional package or model download is required at runtime. `download_real_model.py` can rebuild the checkpoint using checksum-verified upstream downloads. All learned tensors load strictly; optimizer and profiling counters are removed.

Detection consumes the **original observed RGB**, not the RCAN output. BGR channel order, author normalization and 1024-pixel network input match the published contract. Resizing is adapted with aspect-preserving padding rather than distorting nonsquare images. Larger images use overlapping native tiles with weighted blending. Native predictions are replicated to the enhanced display grid; this does not create new observed resolution.

For real four-band inputs only, strong measured-NIR evidence corrects water/vegetation conflicts: NDVI > 0.55 with negative NDWI for green vegetation, and NDWI > 0.15 with NDVI < 0.15 and NIR reflectance < 0.12 for water. Water predictions inconsistent with strong measured spectra are reconsidered using the network's other logits. These are explicit heuristic checks, not learned Sentinel-2 calibration. RGB inputs use the real RGB network alone; their estimated NIR is **never** used for these checks. Scores remain uncalibrated and the exported report identifies the hybrid method.

## What was measured

Run `python evaluate_real_accuracy.py` to reproduce `real_accuracy_report.json` on CPU. It records every included Sentinel-2 scene and both annotated examples, including regressions.

- On Pavana, water intersection-over-union against the **automatic SCL mask** rose from **0.2023 to 0.9412**. The incorrect road share fell from about 47.5% to 0.08%. SCL is a 20 m automatic quality classification resampled to the scene grid: this is reference consistency, **not independent human-labeled accuracy**.
- Vegetation consistency improved in Delhi and Pune, but declined in the Western Ghats and Pavana. Agriculture and Pune's narrow river remain difficult. There is no claim that every class or scene improved.
- Two genuine Maxar/xBD satellite crops, with OpenEarthMap human annotations, give a combined **80.91% pixel agreement** across the six mapped classes. These are upstream demo examples and may be in upstream train/validation. This is an **in-domain sanity check**, not held-out generalization performance and not Sentinel-2 accuracy. The class IoUs and full confusion matrices are included; roads remain imperfect.

The reference images/masks in `tests/fixtures/real_landcover/` are unchanged upstream data, with checksums and attribution. They are licensed CC BY-NC-SA 4.0 for noncommercial use, as listed by [OpenEarthMap](https://open-earth-map.org/attribution.html); providers are Maxar/xBD and OpenEarthMap annotation authors. The five primary demo scenes remain genuine georeferenced Sentinel-2 data.

## Practical limits

This model was trained on **0.25-0.5 m** imagery. Sentinel-2's B02/B03/B04/B08 observations have **10 m** pixels. Small roads/buildings may occupy only part of a pixel, and the model has substantial domain shift. Building-like farmland patches, urban vegetation and small river errors remain possible. Finer real imagery is much more suitable for road/building demonstrations. You can open the real high-resolution `demo_images/high_resolution/palu_8.tif` example in the app; it has native RGB and an original EPSG:4326 footprint, but no measured NIR. Geographic exports preserve that footprint; physical areas are unavailable for this geographic CRS and NIR checks are disabled.

There is **no measured independent accuracy for arbitrary real satellite uploads** and no promise of mistake-free classification. Accurate deployment needs geographically separate labeled data at the target resolution, calibration and per-class evaluation. RCAN is now trained on real Sentinel-2/VENµS pairs with a small measured gain over bicubic (README); sharpening and 2x sampling are not evidence that subpixel buildings have been recovered.

Use **Real imagery model** for actual uploads and leave **Run without GPU (CPU)** checked on the friend's laptop. Changing detectors clears stale maps/scores/downloads while retaining the original image and existing enhancement. The real backend participates in CPU/GPU switching and all image/GeoTIFF/ZIP exports.
