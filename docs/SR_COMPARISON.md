# Does super-resolution help? Results on real Sentinel-2

October 2026. Experiments on **real** Sentinel-2 imagery compare plain bicubic interpolation, this project's RCAN and ESA's SEN2SRLite (trained on real Sentinel-2/NAIP pairs). Raw numbers: `results/sr_comparison*.json`, `results/mapping_experiment.json`, `results/sen2venus_test_full.json`, `results/compare_striping.json`.

**Which RCAN?** The project has had three RCANs with the same small architecture:
- **Synthetic RCAN** (`checkpoints/demo/rcan_synthetic.pt`): trained on procedural scenes. It was the app model when Experiment 2 ran, so that result describes it.
- **Run A** (`checkpoints/demo/rcan_runA.pt`): first training on real Sentinel-2 10 m / VENµS 5 m pairs (SEN2VENµS, 7 sites). It learned VENµS's lower contrast and flattens Sentinel-2 images; kept for comparison only.
- **Run B** (`checkpoints/demo/rcan.pt`, the current app model): retrained on 17 sites with sub-pixel alignment and contrast matching. Experiments 1 and 3 include it; Experiment 2 **has not been re-run** with it.

![Summary](sr_comparison.png)

## Short answer

- **Image quality, synthetic RCAN:** **worse than bicubic** on real images (−0.63 dB PSNR; better on only 2 of 42 scenes). It adds about 3x more fine detail than bicubic, but the detail doesn't match the real image. Visible striping appears below. SEN2SR roughly matches bicubic (−0.12 dB, slightly higher SSIM).
- **Image quality, real RCAN (Run B):** against real VENµS 5 m images at two held-out Indian sites it **beats bicubic by a small margin**: +0.20 dB PSNR (+0.11 dB with no corrections), better on 83% of 13,039 patches (Experiment 3). It is also the only model above bicubic on the reduced-scale test: +0.20 dB, 38 of 42 scenes (Experiment 1). The synthetic RCAN scores −0.26 dB and SEN2SR −0.04 dB on the cross-sensor test. An earlier "+0.47 dB" for Run A came from a more lenient score and hid a contrast-flattening flaw; it is withdrawn.
- **Mapping (synthetic RCAN):** the same U-Net trained on bicubic, RCAN or SEN2SR input gives **the same land-cover accuracy** (mean IoU 49.3%, 49.3%, 48.8%). All three beat the raw 10 m pixels by about 1–1.5 points. That gain comes from the smoother 5 m grid, which bicubic already provides, not from any model-added detail. With a 0.2 dB image gain and 10 m labels, the real RCAN is unlikely to change this, but it has not been measured.
- This matches the literature ([RELATED_WORK.md](RELATED_WORK.md) §5): super-resolution makes images look sharper far more than it makes maps better.

![Visual comparison](sr_visual_comparison.png)

## Experiment 1: image quality (reduced-scale test)

**Data:** 42 real Sentinel-2 L2A crops: 37 Indian sites from `scripts/download_mapping_benchmark.py` plus the 5 demo scenes. Neither model was trained on any of them.

**Method:** each real 10 m crop is area-averaged to 20 m. Each method upsamples it 2x back to 10 m and is scored against the real 10 m image. SEN2SR runs at 4x and is area-averaged by 2. Script: `scripts/compare_sr_models.py`.

| Method | PSNR (dB) | Change vs bicubic | Scenes better than bicubic | SSIM | Spectral angle (deg) |
|---|---:|---:|---:|---:|---:|
| Bicubic | 39.34 | 0 | n/a | 0.9366 | 1.74 |
| Synthetic RCAN | 38.71 | −0.63 | 2 / 42 | 0.9324 | 2.05 |
| SEN2SR (ESA) | 39.22 | −0.12 | 15 / 42 | 0.9412 | 1.78 |
| Run A (real data, flattened contrast) | 37.76 | −1.57 | 0 / 42 | 0.9285 | 2.25 |
| **Run B (app)** | **39.53** | **+0.20** | **38 / 42** | **0.9428** | 1.77 |

Run B's seeds 1 and 2 score +0.24 and +0.28 dB.

**Native-scale checks** (real 10 m input → 5 m output, no reference image):

| Method | Consistency PSNR (output averaged back to 10 m vs input) | Fine detail vs bicubic |
|---|---:|---:|
| Bicubic | 53.6 dB | 1.0x |
| Synthetic RCAN | 55.1 dB | 2.9x |
| SEN2SR | 54.5 dB | 2.4x |
| Run A | 43.9 dB | 0.86x |
| Run B (app) | 54.7 dB | 1.7x |

The synthetic RCAN stays consistent with its input yet adds the most "detail". Combined with the reduced-scale result, much of that detail is invented. Run A's low consistency (43.9 dB) is its contrast flattening: averaged back to 10 m, its output no longer matches the input. Run B adds less detail than the synthetic RCAN or SEN2SR, but the detail it adds scores better.

**Limits:** both models were built for 10 m input, so 20 m input is outside their training conditions. This tests detail recovery at a different scale, a standard but imperfect stand-in for a true 5 m reference. Experiment 3 adds the real cross-sensor test.

**Striping:** the striping in the figure above comes from the synthetic RCAN. `docs/compare_striping.png` shows the app's change-vs-bicubic map on the five demo scenes for all three RCANs: the regular 2-pixel stripes (striping score 0.034) are gone in Run B (0.0012), and Run B changes only edges and features, not flat water or fields.

![Change vs bicubic for the three RCANs](compare_striping.png)

## Experiment 2: does SR input improve land-cover mapping? (synthetic RCAN)

**Data:** 37 Indian sites, each a 512×512 Sentinel-2 crop (2021-22 dry season) with ESA WorldCover 2021 labels in 6 classes: built-up, tree cover, shrub/grass, cropland, water/wetland, bare/sparse. Split by site: 24 train, 5 validation, 8 test (Hyderabad, Lucknow, Guntur, Bathinda, Wayanad, Hirakud, Jaisalmer, Bhubaneswar).

**Method:** one U-Net (the project's `SatelliteSegmentationNet`, 1.9 M parameters) trained from scratch separately on each input, all on a 5 m grid. Same labels, seeds (0, 1, 2), 1,200 steps, class weights and augmentation for every input. The checkpoint with the best validation mean IoU is scored once on the test sites. Script: `scripts/mapping_experiment.py`.

| Input | Overall accuracy | Mean IoU | Mean F1 | Mean IoU change vs bicubic (same seed) |
|---|---:|---:|---:|---:|
| Original 10 m (pixels repeated) | 71.0% ± 2.9 | 47.8% ± 2.2 | 59.7% | −1.5 (range −2.7 to +0.4) |
| Bicubic 2x | 73.5% ± 1.9 | 49.3% ± 1.6 | 61.1% | n/a |
| RCAN 2x | 73.2% ± 2.8 | 49.3% ± 2.1 | 61.2% | 0.0 (range −0.8 to +1.0) |
| SEN2SR (4x → 5 m) | 72.6% ± 2.1 | 48.8% ± 1.8 | 60.7% | −0.5 (range −0.8 to −0.4) |

± is the standard deviation over 3 seeds.

Per-class F1 (mean of 3 seeds):

| Class | Original | Bicubic | RCAN | SEN2SR |
|---|---:|---:|---:|---:|
| Built-up | 0.869 | 0.874 | 0.873 | 0.875 |
| Tree cover | 0.661 | 0.711 | 0.701 | 0.694 |
| Shrub / grass | 0.185 | 0.174 | 0.173 | 0.177 |
| Cropland | 0.313 | 0.350 | 0.359 | 0.341 |
| Water / wetland | 0.914 | 0.917 | 0.918 | 0.911 |
| Bare / sparse | 0.642 | 0.641 | 0.646 | 0.643 |

**Reading it:** the differences between bicubic, RCAN and SEN2SR are smaller than the variation between seeds, so none is reliably better. The original 10 m input is slightly worse, mostly at the forested Wayanad site (53% vs 67% accuracy), where blocky 10 m pixels hurt. Shrub/grass and cropland are weak for every input, which is also true of global 10 m products.

**Limits:**
- WorldCover is itself a model output (about 75% accurate globally), not human ground truth.
- Its labels are 10 m, so it **cannot reward detail finer than 10 m**. This test shows SR doesn't help match a 10 m map; it cannot rule out gains that only finer labels would reveal.
- 8 test sites and 3 seeds is a small sample; confidence intervals are wide.
- Training was short (1,200 steps on CPU). Longer training would raise all scores, probably by a similar amount for each input.
- The RCAN input here is the **synthetic** RCAN. The real RCAN has not been run through this experiment.

## Experiment 3: real cross-sensor test (Sentinel-2 vs VENµS 5 m)

**Data:** all 13,039 patches of SEN2VENµS test sites BENGA (West Bengal) and KUDALIAR (Telangana). Each 10 m Sentinel-2 patch is upsampled 2x and scored against a real VENµS 5 m image of the same place on the same day. Neither site was used in training or validation. Script: `scripts/evaluate_sen2venus.py`.

**Two sensor differences.** Run A exposed both:
1. **Misregistration.** VENµS and Sentinel-2 are offset by fractions of a 5 m pixel. With a leftover sub-pixel offset, blur lowers the error, so a loss that only searches whole-pixel shifts rewards blur. `scripts/prepare_sen2venus.py` estimates each patch's shift to 1/8 pixel from bicubic vs VENµS, once, the same for every method.
2. **Contrast.** At every scale from 10 to 40 m, VENµS has 5–9% less contrast than Sentinel-2 in all four bands. A model trained on raw VENµS learns to flatten Sentinel-2. Each reference is rescaled per band to Sentinel-2's mean and contrast, measured at 40 m.

**Scores.** *Raw*: VENµS as it is. *Aligned*: prediction moved by the patch's known shift, band offsets removed, 4 px border dropped. *Harmonised* (main score): aligned, against the contrast-matched reference, i.e. VENµS's spatial detail on Sentinel-2's radiometry, which is what the app needs.

| Method | Harmonised gain vs bicubic (dB) | Patches better | Aligned gain | Raw gain | SSIM (harm.) | Spectral angle (harm.) |
|---|---:|---:|---:|---:|---:|---:|
| Bicubic (40.54 dB) | 0 | n/a | 0 | 0 | 0.9684 | 1.90° |
| Synthetic RCAN | −0.26 | 2% | −0.25 | −0.25 | 0.9659 | 1.97° |
| Run A | +0.12 | 69% | +0.51 | +0.10 | 0.9706 | 1.78° |
| SEN2SRLite (4x, averaged to 5 m) | −0.04 | 31% | −0.06 | −0.07 | 0.9679 | 1.88° |
| **Run B (app)** | **+0.20** | **83%** | +0.18 | **+0.11** | 0.9703 | 1.80° |

Per site (harmonised gain): Run B +0.34 dB at KUDALIAR, +0.03 dB at BENGA. Seeds 42, 1 and 2 give +0.19 to +0.21 dB; seed 42 was picked by validation loss alone.

**Training (Run B).** 17 training sites (41,704 patches), validation ALSACE. The loss is L1 after applying the known shift, against the contrast-matched reference, with band offsets removed, plus an RMSE term keeping the output (downsampled to 10 m) equal to the input. Learning rate 1e-4, early stopping; best epoch 5. Pilots that failed on validation: a strict 2×2-mean output layer (below bicubic), no brightness anchor (drifted, −2.4 dB raw), an L1 anchor without contrast matching (contrast stayed ~8% low).

**Reading it:** Run A wins the *aligned* score because it copied VENµS's lower contrast, which matches the raw reference but is wrong for Sentinel-2 input; the reduced-scale test (−1.57 dB) shows the damage. Run B is the only model better than bicubic on both the cross-sensor and the reduced-scale test.

**Limits:**
- The gain is small (0.1–0.3 dB), as expected for 2x on Sentinel-2.
- VENµS is a different sensor. This measures "closer to a real 5 m image after correcting known sensor differences", not recovery of objects under 10 m. The shift and contrast corrections are estimates.
- 2 test sites, both Indian, and the gain differs between them.

## Conclusions

1. The synthetic RCAN's gap was its **synthetic training data**. Retraining on real Sentinel-2/VENµS pairs, with the sensors' misregistration and contrast difference corrected, closed it: the app's RCAN now beats bicubic on real imagery by a small margin (Experiments 1 and 3).
2. For mapping, the app is right to classify the **original** image rather than the enhanced one ([LAND_COVER.md](LAND_COVER.md)).
3. The app's Compare view now shows a change-vs-bicubic map, and enhanced exports are labelled as derived imagery, so users can see where the model acted without mistaking it for measured detail.

## Reproduce

```powershell
py -3.14 -m venv .venv-research
.venv-research\Scripts\python -m pip install -r requirements/research.txt
.venv-research\Scripts\python scripts/download_sen2sr.py
.venv-research\Scripts\python scripts/download_mapping_benchmark.py
.venv-research\Scripts\python scripts/compare_sr_models.py
.venv-research\Scripts\python scripts/mapping_experiment.py --steps 1200 --eval-every 200 --seeds 0 1 2
.venv-research\Scripts\python scripts/plot_sr_results.py
```

Runtime on a CPU laptop (8 threads): about 5 minutes for the comparison and about 4.5 hours for the 12 mapping runs.

Experiment 3 needs the SEN2VENµS sites (~37 GB) and the steps in the README ("Reproduce the training and evaluation"). Then:

```powershell
python scripts/evaluate_sen2venus.py --models old_synthetic=checkpoints/demo/rcan_synthetic.pt runA=checkpoints/demo/rcan_runA.pt runB=checkpoints/demo/rcan.pt --sen2sr --limit 0
python scripts/compare_sr_models.py --rcan checkpoints/demo/rcan.pt --output results/sr_comparison_runB.json
python scripts/plot_compare_striping.py old_synthetic=checkpoints/demo/rcan_synthetic.pt runA=checkpoints/demo/rcan_runA.pt runB=checkpoints/demo/rcan.pt
```

**Data credits:** contains modified Copernicus Sentinel data (2021–2024). ESA WorldCover 2021 (CC BY 4.0). SEN2SRLite by ESA OpenSR (MIT). SEN2VENµS (Michel et al., 2022; CC BY-NC 4.0).
