# User guide

How to use the desktop apps and command-line tools, train on your own data, and read the outputs correctly. For results, see the [README](../README.md), [SR_COMPARISON.md](SR_COMPARISON.md) and [LAND_COVER.md](LAND_COVER.md).

## Desktop app

```bash
python -m pip install -r requirements.txt
python app.py
```

**Setup**

- **Python:** 3.10 or newer.
- **Weights:** `checkpoints/demo/` already holds the trained weights:
  - `rcan.pt`: super-resolution model.
  - `landcover_real.pt` and `hf_segformer/`: land-cover detectors.
  - `unet.pt`: synthetic U-Net.
- **Windows:** you can instead run `setup_cpu.bat` once, then `run_app.bat`. Setup creates `.venv-cpu` with the CPU build of PyTorch. No CUDA or NVIDIA driver is needed, and after setup everything runs offline.
- **CPU or GPU:** CPU is the default. To use an NVIDIA GPU, uncheck **Run without GPU (CPU)**. Switch devices only while the app is idle; if the new device fails to load, the app restores the previous one.

**Workflow**

1. **Open an image.** Use **Choose image**, **Open image**, **Open four satellite bands** (aligned single-band rasters) or the sample list.
2. **Process it.** **Run all (enhance + map)** produces every result. **Enhance (2x)** and **Detect** run the two steps separately.
3. **Inspect the views.** The views are Original, Enhanced, Compare, Map, Map on photo and Scores. **Compare** shows the original, the RCAN result and a change-vs-bicubic map (darker blue means RCAN changed the pixel more, up to 0.05 reflectance). The map shows where the model acted; it does not prove the added detail is real.
4. **Download results.** The **Downloads** panel exports:
   - images as PNG/JPG and maps as PNG/TIF
   - GeoTIFF (needs a georeferenced source)
   - the current view, the scores PNG and the report
   - **All results ZIP**

   Enhanced exports are tagged as derived imagery (`derived_product=true` in GeoTIFF tags, and in PNG/JPEG metadata).
5. **Pick a detector.**
   - **Combined real models** is the default.
   - **Hugging Face SegFormer (fast)** is quicker.
   - **SparseMask** and **U-Net experiment / custom model** keep the earlier workflows.

   Changing the detector clears stale maps but keeps the enhancement.
6. **Use your own weights.** **Choose model folder** loads custom weights:
   - Combined detector: `rcan.pt`, `landcover_real.pt` and `hf_segformer/`.
   - SegFormer alone: `rcan.pt` and `hf_segformer/`.
   - Synthetic mode: `rcan.pt` and `unet.pt`.

   Missing weights raise an error rather than silently falling back.

**Processing details**

- **Background processing:** processing runs in a worker thread.
- **Tiling:** large inputs use overlapping tiles, so results can differ slightly from whole-image inference.
- **Size cap:** inputs are capped at 1024 px on the longest side. Larger rasters are resampled with their footprint and transform updated; use smaller crops to keep native pixel spacing.

## Command line

```bash
# A four-band GeoTIFF or an ordinary image
python scripts/predict.py --image demo_images/sentinel2_urban_delhi_2023-12-05.tif --output outputs/urban_delhi
# Separate single-band files, in B02 B03 B04 B08 order
python scripts/predict.py --bands B02.tif B03.tif B04.tif B08.tif --output outputs/scene
# Pick the detector: auto (default), ensemble, hf, real or synthetic
python scripts/predict.py --image demo_images/high_resolution/palu_8.tif --detector hf --output outputs/palu
```

Prediction runs on CPU by default; pass `--device cuda` for a GPU.

## Demo data

`demo_images/` holds five genuine Sentinel-2 L2A crops, each 256×256 at 10 m. Each crop has:

- measured B02/B03/B04/B08 bands
- the original UTM georeferencing
- calibration and validity masks

| Scene | Acquired (UTC) | Area |
| --- | --- | --- |
| Urban | 2023-12-05 | Central New Delhi |
| Agriculture | 2023-12-03 | Farmland east of Ludhiana, Punjab |
| Vegetation | 2023-12-30 | Western Ghats near Koyna |
| Water | 2024-02-28 | Pavana Reservoir |
| Mixed | 2024-02-28 | Pune and the Mula-Mutha river |

![Real Sentinel-2 crops](sentinel2_dataset.png)

**Provenance and supporting files**

- **Source records:** source URLs, dates, crop windows, calibration and SHA-256 hashes are in `demo_images/manifest.json` and `demo_images/sources/`.
- **Separate bands:** each scene's single-band files are in `demo_images/satellite_bands/<area>/`.
- **Quality layer:** `demo_images/quality/` holds the Sentinel-2 SCL layer. It is used only for quality screening and is not a land-cover label.
- **High-resolution sample:** `demo_images/high_resolution/palu_8.tif` is a fine-resolution Maxar crop with RGB only, useful for road and building examples.
- **Re-download:** run `python scripts/download_sentinel_demo.py --output <empty folder>`.

The source COGs already have their BOA offset applied, so the downloader does not subtract it again.

## Real-data RCAN training (SEN2VENµS)

**Dataset.** SEN2VENµS (Michel et al., 2022, *Data* 7(7):96; Zenodo 6514159; CC BY-NC 4.0). It pairs Sentinel-2 10 m B02/B03/B04/B08 patches (128×128) with VENµS 5 m patches (256×256) taken the same day. The splits are by site:

| Split | Sites | Patches |
| --- | --- | ---: |
| Train | ATTO, ESTUAMAR, SUDOUE-4, K34-AMAZ, MAD-AMBO, LERIDA-1, JAM2018 | 11,745 |
| Validation | ALSACE | 2,647 |
| Test | BENGA (West Bengal), KUDALIAR (Telangana) | 13,039 |

```bash
python scripts/download_sen2venus.py      # ~17 GB, resumable, MD5-checked
python scripts/prepare_sen2venus.py       # -> datasets/sen2venus_npy/{train,val,test}
python scripts/train.py --task sr --data datasets/sen2venus_npy --output checkpoints/real/runA --epochs 80 --batch-size 16 --crop 96 --lr 2e-4
python scripts/evaluate_sen2venus.py --models old_synthetic=checkpoints/demo/rcan_synthetic.pt new_real=checkpoints/demo/rcan.pt
```

**Architecture.** 32 features, 3 residual groups × 4 blocks and a bicubic skip connection, under 1 M parameters.

**Loss.** VENµS and Sentinel-2 are misregistered by about one 5 m pixel. The loss is therefore a **shift- and bias-compensated L1**:

- The best integer shift within ±2 px is removed before scoring.
- Each band's mean offset is removed too.
- A weak term (weight 0.1) keeps the 2×2 average of the output equal to the input.

This stops the model from learning to shift or recolour the image.

**Training run.** bf16 mixed precision, about 60 s per epoch on an RTX 5050 Laptop GPU. The checkpoint with the best validation loss came at epoch 6 of 80. Results are in `results/sen2venus_test_runA.json` and the README.

## Synthetic experiment (original baseline)

```bash
python scripts/train.py --demo --epochs 30 --samples 128 --output checkpoints/demo
python scripts/predict.py --demo --output outputs/synthetic_experiment
python scripts/plot_training.py
```

**Data.** 128 training, 32 validation and 32 test scenes, from disjoint seeds. Each scene is 64×64 four-band HR with five-class labels; LR is made by antialiased bicubic downsampling.

**Training.** RCAN uses an L1 loss and the U-Net uses cross-entropy. The best checkpoint per task is chosen by validation loss.

**Outputs.** The RCAN from this run is kept as `checkpoints/demo/rcan_synthetic.pt`, with an ONNX copy as `rcan_synthetic.onnx`. The U-Net (`unet.pt`) and `training_report.json` also come from this run.

| Held-out synthetic test | Bicubic | RCAN |
| --- | ---: | ---: |
| PSNR | 30.08 dB | 35.74 dB |
| SSIM | 0.8928 | 0.9717 |
| Spectral angle | 5.02° | 2.39° |

The RCAN → U-Net pipeline reached 98.1% pixel accuracy and 94.4% mean IoU on these synthetic scenes. This gain did **not** transfer to real imagery (−0.29 dB vs bicubic on the SEN2VENµS test), which is why the model was retrained on real pairs.

![Training and test results](training_results.png)

## Train on your own data

**Input format**

- **Imagery:** co-registered B02, B03, B04, B08 in that order, as uint16 reflectance × 10000.
- **Masks:** single-channel GeoTIFFs with these IDs: 0 buildings, 1 roads, 2 vegetation, 3 water, 4 bare land, 255 ignored.
- **Splits:** split by geographic scene **before** extracting patches.

```text
raw_data/
  train/  hr/scene_001.tif  masks/scene_001.tif  lr/scene_001.tif (optional)
  val/    hr/...            masks/...            lr/... (optional)
  test/   hr/...            masks/...            lr/... (optional)
```

```bash
python scripts/prepare_dataset.py --input raw_data --output datasets/satellite
python scripts/train.py --data datasets/satellite --epochs 100 --batch-size 8 --output checkpoints/real
python scripts/predict.py --image your_four_band_scene.tif --models checkpoints/real --output outputs/real
```

**Requirements**

- **Rasters:** HR and masks must share size, CRS and transform. HR dimensions must be even and at least 16×16. Measured LR must cover the same footprint at half size.
- **Missing LR:** if LR is missing, a degraded LR is generated. This then tests reconstruction under that degradation, not recovery of independently observed detail.
- **NPZ alternative:** you can supply NPZ files with `hr` (2H×2W×4 uint16), optional `lr` (H×W×4) and `mask` (2H×2W).

## Reading the outputs

- **App metrics.** With no HR reference, the app's metrics measure similarity to bicubic, not accuracy. Higher edge energy means sharper output, not real recovered structure.
- **Scores.** Softmax scores are uncalibrated.
- **Images without NIR.** PNG/JPEG/RGB TIFF inputs get an estimated NIR so the four-band model can run. The land-cover detector uses their RGB only, and estimated NIR is never used for spectral checks.
- **Images without a CRS.** Ordinary images keep no invented CRS, so their area is unknown. Projected rasters compute pixel area from the transform and CRS units.
- **GeoTIFF exports.** They keep the source CRS, extent, band descriptions and validity masks. 2x output halves the pixel spacing; that is resampling, not twice the real resolving power.
- **Label GeoTIFFs.** They include a class palette, with 255 as NoData.
- **Indices.** Four bands support NDVI and green/NIR NDWI. NDBI needs SWIR, which is not used here.
- **Rejected inputs.** Corrupt, truncated, mislabelled, fully transparent and fully NoData inputs are rejected, and the previous valid image is kept.

## CPU-only ONNX app

A second app runs super-resolution only, with ONNX Runtime and without PyTorch:

```bash
python -m pip install -r requirements/onnx.txt
python app_onnx.py
# Re-export (needs the PyTorch requirements too):
python scripts/convert_to_onnx.py --checkpoint checkpoints/demo/rcan.pt --output checkpoints/demo/rcan_model.onnx
```

**Export checks.** The export script checks the ONNX model against PyTorch on three input sizes, including a single pixel. Band order and normalisation are stored in the model and checked by the app.

**What the app supports**

- inputs: ordinary images and four-band TIFFs
- overlapping-tile processing
- exports: PNG/JPEG, GeoTIFF, and JPEG with JGW and PRJ sidecars

![ONNX app](onnx_preview.png)
