# Handoff: retrain RCAN on real Sentinel-2 data

Written 22 October 2026 on the team's first laptop (Windows, CPU + NVIDIA T500) for the Claude Code session on the second laptop (Windows, **RTX 5050 Laptop GPU, 8 GB**). Same user: Aniruddha Kadam. Read this before doing anything.

## 1. The project in one paragraph

Semester-7 deep-learning project (team: Shantanu Vasagadekar, Aniruddha Kadam, Shrihari Kshirsagar). A PyQt desktop app (`L1BSR-GUI.py`) takes Sentinel-2 imagery (B02, B03, B04, B08; uint16, reflectance × 10000), upsamples it 2x with an RCAN model (10 m → 5 m grid), maps land cover with pretrained SegFormer + SparseMask, and exports PNG/GeoTIFF/ZIP. See `README.md`. Background research is in `RESEARCH.md` and `RESEARCH_LANDSCAPE.md`. The research's top recommendation is to **retrain RCAN on real image pairs (SEN2VENµS)**. That is your job.

## 2. What happened in the last session (all committed)

1. **SEN2SR comparison model.** `models/sen2sr_lite.py` wraps ESA OpenSR's SEN2SRLite (RGBN, 4x, trained on real Sentinel-2/NAIP pairs). Its 4x output is area-averaged to 5 m to compare with RCAN. It's optional and not used by the app. Weights: `python download_sen2sr.py` (goes to `checkpoints/external/`, gitignored).
2. **Real benchmark data.** `download_mapping_benchmark.py` downloads 37 Indian sites: a 512×512 Sentinel-2 L2A crop from the 2021-22 dry season plus ESA WorldCover 2021 labels for each. Fixed site splits: 24 train / 5 val / 8 test. Output goes to `datasets/mapping_benchmark/` (gitignored, about 58 MB). Re-download with the script.
3. **Image-quality test** (`compare_sr_models.py` → `results/sr_comparison.json`, `docs/sr_comparison.png`). Reduced-scale test on 42 real scenes (each 10 m image area-averaged to 20 m, rebuilt to 10 m, scored against the real 10 m image):

   | Method | PSNR vs bicubic | Scenes better than bicubic | SSIM | SAM (deg) |
   |---|---:|---:|---:|---:|
   | Bicubic | 0 | n/a | 0.9366 | 1.743 |
   | **RCAN (current, synthetic-trained)** | **−0.63 dB** | 2 / 42 | 0.9324 | 2.053 |
   | SEN2SR (ESA, real-trained) | −0.12 dB | 15 / 42 | 0.9412 | 1.776 |

   RCAN adds about 3x more high-frequency energy than bicubic, but that detail is wrong. Visible striping appears in `docs/sr_visual_comparison.png`. This is the synthetic-training gap that retraining should fix.
4. **Mapping experiment** (`mapping_experiment.py`, finished). The same U-Net is trained on native / bicubic / RCAN / SEN2SR inputs (3 seeds, 1200 steps, CPU), scored against WorldCover on the 8 test sites. Test mean IoU: native 47.8%, **bicubic 49.3%, RCAN 49.3%**, SEN2SR 48.8% (seed spread about ±2). SR adds nothing over bicubic for mapping. Full write-up: **`SR_COMPARISON.md`**. Don't re-run it.
5. **App changes.** The Compare view now shows Original | Enhanced | **change-vs-bicubic map** (`results_export.compose_compare`, `difference_from_bicubic`). Every enhanced export (PNG text chunk, JPEG EXIF, GeoTIFF tags `derived_product=true`, `TIFFTAG_IMAGEDESCRIPTION`) is labelled as derived imagery (`geo_io.derived_note`). Tests: `tests/test_derived_exports.py`, `tests/test_sr_comparison.py`.
6. Test status on laptop 1: all suites pass in `.venv-cpu` once `requirements-onnx.txt` is installed.

## 3. Is the RTX 5050 (8 GB) good enough? Yes, use it

- The RCAN being trained is small. The current config is `n_feats=32, n_resgroups=3, n_resblocks=4, residual=True`, well under 1 M parameters. Even a larger RCAN (`n_feats=64`, 5–10 groups) with 128×128 LR patches and batch 16 fits in 8 GB, especially with mixed precision.
- Local training has no session time limit, and the dataset subset (about 17 GB) stays on disk. Colab free (T4, 16 GB) would mean re-downloading data each session and dealing with disconnects.
- **Blackwell catch:** the RTX 5050 is compute capability **12.0 (sm_120)**. It needs a PyTorch build for **CUDA 12.8 or newer** and an NVIDIA driver from the R570 series or newer. Older CUDA 11.8/12.1 wheels fail with "no kernel image is available". Verify first (section 4).
- **Use Google Colab only if** the GPU check below fails and can't be fixed by updating the driver or PyTorch build.

## 4. Environment setup on this laptop

```powershell
py -3.14 -m venv .venv-gpu          # or the Python 3.10+ that is installed
.venv-gpu\Scripts\python -m pip install --upgrade pip
.venv-gpu\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
.venv-gpu\Scripts\python -m pip install -r requirements.txt -r requirements-dev.txt -r requirements-onnx.txt
.venv-gpu\Scripts\python -c "import torch;print(torch.__version__,torch.version.cuda,torch.cuda.get_device_name(0),torch.cuda.get_device_capability(0));x=torch.randn(2048,2048,device='cuda');print((x@x).sum().item())"
```

Expected: a CUDA version of 12.8 or newer, `NVIDIA GeForce RTX 5050 ...`, capability `(12, 0)`, and a number printed without error. Then run the test suite: `.venv-gpu\Scripts\python -m unittest discover -s tests`. `.venv*` folders are gitignored. Don't copy virtual environments between laptops.

For the SEN2SR comparison later: `pip install -r requirements-research.txt` in a separate env, or in `.venv-gpu` if versions agree, then `python download_sen2sr.py`.

## 5. The retraining plan

Goal: an RCAN trained on **real Sentinel-2 10 m → VENµS 5 m pairs**, compared fairly against bicubic, the old synthetic RCAN and SEN2SR.

### 5.1 Data: SEN2VENµS (Zenodo record 6514159, v1.0.0, CC BY-NC 4.0)

- 30 site archives (`.7z`, 0.07–9.6 GB) plus `MD5SUMS`. List them: `https://zenodo.org/api/records/6514159`. Download: `https://zenodo.org/records/6514159/files/<SITE>.7z`.
- **Two sites appear to be in India: `BENGA` (3.7 GB) and `KUDALIAR` (5.0 GB).** Confirm their locations from the dataset paper or metadata.
- Suggested subset (about 17 GB), split **by site**:
  - train: `ATTO, ESTUAMAR, SUDOUE-4, K34-AMAZ, MAD-AMBO, LERIDA-1, JAM2018`
  - val: `ALSACE`
  - test: `BENGA, KUDALIAR` (the story is "trained elsewhere, tested on Indian sites")
  - Adjust if a site turns out unsuitable. Check MD5 sums after download. Extract with 7-Zip (`7z x`) or `py7zr`. Put everything under `datasets/sen2venus/` (gitignored).
- **Verify the file format before writing code.** From the dataset paper (Michel et al., 2022, *Data* 7(7):96): each site has PyTorch tensor files with patches of 10 m bands B2, B3, B4, B8 (128×128), 20 m bands, and 5 m VENµS references (256×256), stored as int16 to be divided by 10,000 (the same scale as this project). There is also an index/CSV of patches. Filenames and tensor layouts here are from memory and **unverified**, so inspect an extracted site first: shapes, dtype, value range, band order.

### 5.2 Code to write

1. **`prepare_sen2venus.py`**: convert the selected sites into the NPZ layout `train.py` reads (`datasets/sen2venus_npz/{train,val,test}/*.npz` with `lr` H×W×4, `hr` 2H×2W×4, uint16, band order B02, B03, B04, B08). Drop patches with nodata/negative values. Record site, date and patch ID per file. Many patches per NPZ file (or one per patch) is fine if `SatelliteDataset` loads them. Read `data.py` first.
2. **`train.py`: add an SR-only mode** (e.g. `--task sr`). Right now `--data` training always trains RCAN *and* the U-Net and requires `mask` arrays. SEN2VENµS has no land-cover masks. With `--task sr`, masks must be optional and the segmentation phase and its test metrics skipped. Keep the existing checkpoint format (`save_checkpoint`, `read_checkpoint` in `models/common.py`), so `band_order` = B02, B03, B04, B08, `normalization` = 10000, scale 2, 4 channels. Set `data_source` to something like `"SEN2VENµS real Sentinel-2 10 m / VENµS 5 m pairs (sites: ...)"`, because the app shows this string.
3. Use GPU training niceties: `torch.amp` autocast, `num_workers` > 0 and `pin_memory`, random flips/rot90, random LR patch crops (e.g. 64–96 px from the 128 px patches). Keep `cudnn.deterministic` optional.
4. Write any new tests in the `tests/` style. Don't break the existing suite.

### 5.3 Training runs

- **Run A (required):** same architecture as the current app model (`n_feats=32, n_resgroups=3, n_resblocks=4, residual=True`, zero-initialised last tail conv so it starts as bicubic). L1 loss, Adam 1e-4 to 2e-4 with a cosine schedule. Select by validation L1. This gives a like-for-like "synthetic vs real data" comparison and stays CPU-friendly for the app and ONNX.
- **Run B (optional, if time):** larger RCAN (e.g. `n_feats=64, n_resgroups=5, n_resblocks=10`). `read_checkpoint` caps `n_feats²·groups·blocks` at 2,000,000. Check CPU inference time in the app, since the target laptops run on CPU.
- Save to `checkpoints/real/` (gitignored) while experimenting. Copy the chosen model in at the end (section 5.5).

### 5.4 Evaluation to report

1. **SEN2VENµS test sites (BENGA, KUDALIAR):** PSNR / SSIM / SAM against the real VENµS 5 m reference for bicubic, old synthetic RCAN, new real RCAN and SEN2SR (4x, area-averaged to 5 m). This is the cross-sensor test the project has never had. Note that VENµS and Sentinel-2 differ slightly in band response and registration.
2. **Reduced-scale test on Indian scenes:** after `python download_mapping_benchmark.py`, run `python compare_sr_models.py --rcan checkpoints/real/rcan.pt --output results/sr_comparison_real_rcan.json` and compare with `results/sr_comparison.json` (old RCAN: −0.63 dB vs bicubic).
3. Look at the app's Compare view: the striping in the change-vs-bicubic panel should be gone or reduced.
4. Report honestly whatever happens, including if real-trained RCAN only matches bicubic. The literature (RESEARCH.md §9, RESEARCH_LANDSCAPE.md §5) expects small gains at 2x.

### 5.5 Putting the new model in the app (only after the user agrees)

- Keep the old model: rename `checkpoints/demo/rcan.pt` to `checkpoints/demo/rcan_synthetic.pt` and put the new one at `checkpoints/demo/rcan.pt` (the app loads this path), with its `training_report.json`.
- Re-export ONNX: `python convert_to_onnx.py` (check its arguments), replacing `rcan_model.onnx`.
- Update README's training-results section, `PROJECT_REPORT.md` and the sidebar text in `L1BSR-GUI.py` (it says "synthetic-trained RCAN").
- Run the full test suite. Some tests may assert the synthetic provenance string or exact metric values; update them deliberately.

## 6. House rules (from how the user has worked so far)

- Report results honestly with their limits. The project's docs are deliberately candid. Never claim real 5 m detail.
- Commit only when asked. Use clear commit messages. Data, venvs and large weights stay out of git (`.gitignore` already covers `datasets/`, `.venv*/`, `checkpoints/real/`, `checkpoints/external/`).
- Run `git pull` before starting and before committing. When retraining is done, add the new RCAN's numbers to `SR_COMPARISON.md` next to the old ones.
- Keep explanations to the user short and plain.
