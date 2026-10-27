# Land-cover detector: models, accuracy and limits

The app's default **Combined real models** detector averages two models pretrained on real, human-labelled aerial and satellite imagery:

- **OpenEarthMap SparseMask** (MobileNetV2): eight source classes mapped to buildings, roads, vegetation, water, bare land and a separate developed-space class.
- **SegFormer-B0** ([boehnen/satlens-segformer](https://huggingface.co/boehnen/satlens-segformer)), fine-tuned on OpenEarthMap. It averages predictions over the original image and three reflected views.

Neither model was fine-tuned on any of the evaluation images below.

The detector runs on the **original image, not the RCAN output**; see [SR_COMPARISON.md](SR_COMPARISON.md) for why. Its predictions are copied onto the enhanced 2x grid for display, which adds no new observed resolution.

## Measured accuracy

Reproduce with `python scripts/evaluate_hf_accuracy.py` (full per-image results, class IoUs, confusion matrices and CPU timings in `results/hf_accuracy_report.json`).

**Pixel accuracy:**

| Evaluation set | SparseMask | SegFormer alone | Combined (default) |
| --- | ---: | ---: | ---: |
| 8 fixed LoveDA validation images | 77.17% | 78.54% | **79.80%** |
| 2 OpenEarthMap examples (may overlap training data) | 80.91% | 80.07% | 84.11% |

**Combined detector, per class (LoveDA set):**

| Class | IoU | Recall |
| --- | ---: | ---: |
| Buildings | 55.26% | 65.91% |
| Roads | 38.61% | 39.52% |
| Vegetation | 79.25% | 96.33% |
| Water | 62.78% | 65.15% |
| Bare land | 0.21% | 0.21% |

Notes on how these numbers were made:

- **How the LoveDA images were chosen.** They were picked before inference by evenly spacing eight IDs in a pinned validation listing.
- **Class mapping.** LoveDA forest and agriculture map to Vegetation.
- **Excluded pixels.** LoveDA's background class has no equivalent in the app, so it is excluded along with no-data: 2,196,159 of 8,388,608 pixels. Abstentions on labelled pixels count as errors.
- **Not comparable to leaderboards.** Because of the mapping and exclusions, these are not comparable with published seven-class LoveDA scores.
- **Interpretation.** Vegetation dominates the images, which flatters overall accuracy. Bare land is effectively not detected.

## Sentinel-2 consistency checks

`python scripts/evaluate_real_accuracy.py` writes `results/real_accuracy_report.json`. It covers the five demo scenes and two annotated examples, including regressions. It was run with SparseMask alone, before the combined detector:

- **Pavana reservoir: water.** Water IoU against the automatic Sentinel-2 SCL mask rose from 0.20 to 0.94 compared with the earlier synthetic U-Net. The share of pixels wrongly called road fell from 47.5% to 0.08%. SCL is an automatic 20 m quality layer, so this measures consistency, not human-labelled accuracy.
- **Other scenes: vegetation.** Vegetation consistency improved in Delhi and Pune but declined in the Western Ghats and Pavana. Farmland and Pune's narrow river remain difficult.
- **Two Maxar/xBD crops with OpenEarthMap labels.** These give 80.91% pixel agreement. They may be in the upstream training data, so this is a sanity check only.

## Spectral corrections (measured NIR only)

For real four-band inputs, strong measured-NIR evidence overrides water/vegetation conflicts:

- **Green vegetation:** NDVI > 0.55 with negative NDWI.
- **Water:** NDWI > 0.15, NDVI < 0.15 and NIR reflectance < 0.12.

These are explicit heuristics, not a learned Sentinel-2 calibration. RGB-only inputs never use estimated NIR for these checks.

## Limits

- **Resolution gap.** Both models were trained on 0.25–0.5 m imagery, while Sentinel-2 pixels are 10 m. Small roads and buildings occupy part of a pixel, so expect errors such as building-like farmland and missed narrow rivers. For road and building examples, use the fine-resolution sample `demo_images/high_resolution/palu_8.tif`.
- **Scores.** Scores are uncalibrated and are not accuracy estimates.
- **Abstentions.** Background-dominant predictions abstain (ID 255) instead of being forced into a class.
- **No general accuracy.** There is no measured accuracy for arbitrary uploads. Reliable deployment would need labelled data from the target region and resolution.

## Model sources and preprocessing

**SegFormer**

| Item | Detail |
| --- | --- |
| Pinned revision | `19cdd879e535d76e7fcbbd1e8b54dac9c1da90d6` |
| License | MIT |
| Source safetensors SHA-256 | `68f1c70396d3e1b5a9b752408cadbafa90c655da816564ce7d9646b112392ac0` |
| Runtime form | 3.72 M parameters, exported to a ~15 MB TorchScript graph. No Transformers install or network is needed at runtime. |
| Export check | Maximum logit difference vs. the original model below 2e-6 in the recorded check |
| Rebuild | `scripts/download_hf_model.py` (needs `transformers` and `safetensors`) |

Preprocessing:

1. Edge-pad non-square inputs to a square.
2. Bilinear resize to 512.
3. Rescale by 1/255 and apply ImageNet mean/std normalisation.

The model card, configs and checksums are bundled in `checkpoints/demo/hf_segformer/`.

**SparseMask**

- **Source:** the OpenEarthMap [author's implementation and weights](https://github.com/cliffbb/oem-lightweight), MIT.
- **Bundled files:** a strictly loaded state dict with source URLs, a pinned revision and SHA-256 hashes.
- **Rebuild:** `scripts/download_real_model.py`.
- **Input:** BGR order, the author's normalisation and a 1024-pixel network input. Large images use overlapping tiles with weighted blending.

## Data attribution

- **LoveDA:** Wang et al., NeurIPS Datasets and Benchmarks 2021 ([GitHub](https://github.com/Junjue-Wang/LoveDA)). Academic use only; Google Earth terms apply. The fixed selection, URLs and hashes are in `tests/fixtures/loveda/manifest.json`. Re-download with `python scripts/download_accuracy_samples.py`.
- **OpenEarthMap reference images and masks** (`tests/fixtures/real_landcover/`): CC BY-NC-SA 4.0. The imagery is from Maxar/xBD.
