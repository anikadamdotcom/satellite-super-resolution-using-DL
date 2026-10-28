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
| Train | ANJI, ATTO, BAMBENW2, ES-LTERA, ESGISB-1, ESTUAMAR, FGMANAUS, FR-LQ1, JAM2018, K34-AMAZ, LERIDA-1, MAD-AMBO, NARYN, SO2, SUDOUE-4, SUDOUE-5, SUDOUE-6 | 41,704 |
| Validation | ALSACE | 2,647 |
| Test | BENGA (West Bengal), KUDALIAR (Telangana) | 13,039 |

```bash
python scripts/download_sen2venus.py      # 20 sites, ~37 GB, resumable, MD5-checked
python scripts/prepare_sen2venus.py       # -> datasets/sen2venus_npy/{train,val,test}, with per-patch shift and contrast
python scripts/train.py --task sr --data datasets/sen2venus_npy --output checkpoints/real/runB_s42 --epochs 25 --early-stop 6 --batch-size 16 --crop 96 --lr 1e-4 --seed 42
python scripts/evaluate_sen2venus.py --models old_synthetic=checkpoints/demo/rcan_synthetic.pt runA=checkpoints/demo/rcan_runA.pt runB=checkpoints/demo/rcan.pt --sen2sr --limit 0
```

**Architecture.** 32 features, 3 residual groups × 4 blocks and a bicubic skip connection, under 1 M parameters.

**Preparing the pairs.** `scripts/prepare_sen2venus.py` drops patches with nodata, saturation or many negative values. For every patch it also stores:

- the sub-pixel shift between VENµS and Sentinel-2, estimated to 1/8 pixel (`src/satsr/models/alignment.py`);
- a per-band gain and offset that put VENµS on Sentinel-2 contrast, measured at 40 m (VENµS has 5–9% less contrast).

**Loss.** L1 between the prediction, moved by the patch's known shift, and the contrast-matched VENµS patch, with each band's mean offset removed. An RMSE term (weight 0.2) keeps the output, smoothly downsampled to 10 m, equal to the Sentinel-2 input, which anchors brightness. A strict 2×2-mean layer (`--consistent`) stayed below bicubic and is off by default.

**Training run (Run B).** bf16 mixed precision, about 3 minutes per epoch on an RTX 5050 Laptop GPU, learning rate 1e-4, early stopping on validation. Three seeds were trained; seed 42 (best validation loss, epoch 5) is `checkpoints/demo/rcan.pt`. The first real-data run (Run A: 7 sites, best-integer-shift loss, no contrast matching) flattened Sentinel-2 contrast and is kept as `rcan_runA.pt` for comparison. Results: [SR_COMPARISON.md](SR_COMPARISON.md), Experiment 3.

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
