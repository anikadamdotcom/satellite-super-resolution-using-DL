# Research Review: Deep-Learning Super-Resolution and Land-Cover Mapping from Sentinel-2

A literature and background review for this project, compiled October 2026. It covers the problem, the methods, the datasets, how results are judged, the land-cover side, and where this project sits. For competitors, a timeline of prior work and the 2025–2026 research frontier, see [RESEARCH_LANDSCAPE.md](RESEARCH_LANDSCAPE.md). Every external claim links to its source in the [References](#references). Numbers quoted from other papers come from their abstracts or project pages; check the full paper before citing a figure in the final report.

---

## 1. Summary

- **The problem is real and active.** Sentinel-2 gives free, global imagery every five days, but its best bands are 10 m. Many objects that matter (narrow roads, small buildings, field boundaries, small ponds) are at or below that size. Making Sentinel-2 look like 2.5–5 m imagery is an active research area, with an ESA-funded project (OpenSR) dedicated to it.
- **The field has moved from "synthetic" to "real" training data.** Early work, and this project's RCAN, trains on HR images that are artificially blurred and downsampled. Models trained this way often fail on real Sentinel-2 because real blur, noise and sensor effects differ. Since 2022, datasets that pair real Sentinel-2 with real higher-resolution sensors (SEN2VENµS, WorldStrat, SEN2NAIP, S2-NAIP) have become the standard.
- **PSNR/SSIM are no longer enough.** A model can raise PSNR without adding true detail, and generative models (GANs, diffusion) can add detail that is not on the ground ("hallucination"). New benchmarks such as `opensr-test` measure consistency with the input, how much detail is added, and whether that detail is correct.
- **Super-resolution does not reliably help mapping yet.** The evidence is mixed. Joint SR-plus-segmentation models and multi-image methods report gains. Simple "super-resolve, then segment" pipelines often perform no better than bicubic interpolation; one 2025 study on building delineation found no advantage over interpolation.
- **"Super-resolution mapping" has a second, older meaning.** In remote sensing it originally meant *sub-pixel mapping*: predicting a land-cover map finer than the image pixels, from class fractions. The project title uses the phrase; know both meanings for the viva.
- **Best next step for this project:** retrain or fine-tune RCAN on **SEN2VENµS**. It pairs real Sentinel-2 B2/B3/B4/B8 at 10 m with real 5 m VENµS images: exactly this project's four bands and 2× scale. Then evaluate with `opensr-test` and with a bicubic-vs-RCAN downstream segmentation comparison.

---

## 2. What the project does, in context

The project ([README.md](README.md), [PROJECT_REPORT.md](PROJECT_REPORT.md)) has two parts:

1. **Super-resolution.** A small RCAN (residual channel attention network) upsamples four Sentinel-2 bands (B02 blue, B03 green, B04 red, B08 NIR) by 2×, from 10 m to a 5 m grid. It is trained on procedurally generated scenes, with LR made by bicubic downsampling.
2. **Land-cover mapping.** A pretrained ensemble (Hugging Face SegFormer-B0 and OpenEarthMap SparseMask) labels buildings, roads, vegetation, water, bare land and developed space. A synthetic-trained U-Net is also available. Results are exported as maps and GeoTIFFs.

In research terms this is **single-image super-resolution (SISR)** of the **10 m bands** at **2×**, followed by **semantic segmentation**. The sections below place each of those choices among the alternatives.

---

## 3. Background: Sentinel-2

| Property | Value | Why it matters here |
| --- | --- | --- |
| Mission | Copernicus Sentinel-2A/2B (and 2C), MultiSpectral Instrument (MSI) | Free, global, 5-day revisit: the main reason to super-resolve it rather than buy commercial imagery |
| 10 m bands | B2 (490 nm), B3 (560 nm), B4 (665 nm), B8 (842 nm) | The four bands this project uses |
| 20 m bands | B5, B6, B7, B8A (red edge / narrow NIR), B11, B12 (SWIR) | SWIR is needed for NDBI and for separating built-up from bare soil; the project notes this gap |
| 60 m bands | B1, B9, B10 | Atmospheric bands; rarely super-resolved for mapping |
| MTF at Nyquist (10 m bands) | Specified 0.15–0.30 | The image is already soft at its own pixel scale. This limits how much real detail SR can recover; L1BSR's authors cap the realistic gain at about 2× for this reason |
| Product levels | L1C (top-of-atmosphere), L2A (bottom-of-atmosphere surface reflectance) | The project uses L2A, scaled by 10,000, from the Earth Search COG archive |

Sources: ESA's MSI performance guide, the Sentinel-2 product specifications and the L1BSR paper.

### Four ways to "super-resolve" Sentinel-2

| Task | Input → output | Typical scale | Examples |
| --- | --- | --- | --- |
| **Band sharpening** (intra-sensor) | 20 m / 60 m bands → 10 m, guided by the 10 m bands | 2× and 6× | DSen2 (Lanaras et al., 2018), Sen2-RDSR |
| **Single-image SR** of the 10 m bands | One 10 m image → 5 m or 2.5 m | 2×–4× | L1BSR, SEN2SR, RCAN/EDSR/SwinIR adaptations, **this project** |
| **Multi-image SR (MISR)** | Several revisits of the same place → one sharper image | 2×–4× | HighRes-net, DeepSUM (PROBA-V challenge), WorldStrat baselines, Satlas |
| **Cross-sensor SR** | Sentinel-2 → appearance of another sensor (NAIP, SPOT, VENµS, Pléiades) | 2×–8× | SEN2NAIP, S2-NAIP/Satlas, SEN4X |

DSen2 relies on a useful trick: band sharpening can be trained on real Sentinel-2 alone. The authors degrade 20 m → 40 m and learn 40 → 20, then apply the model at 20 → 10. They report about 50% lower RMSE than the best competing method at reduced scale, and a model that works on new scenes without retraining. This is the **Wald protocol** idea: test at reduced scale, where a reference exists, and assume the model transfers across scales.

---

## 4. Super-resolution methods

### 4.1 Model families (computer vision first, then adapted to EO)

| Year | Model | Idea | Relevance |
| --- | --- | --- | --- |
| 2014 | SRCNN | First CNN for SR: three convolution layers | Historical baseline |
| 2017 | EDSR | Deep residual network without batch norm | Common EO baseline |
| 2018 | **RCAN** | Residual-in-residual groups plus **channel attention** | **The model this project adapts.** Channel attention suits multispectral data because it learns to weight bands |
| 2018 | ESRGAN | GAN loss plus perceptual loss for sharp textures | Basis of Satlas SR and SEG-ESRGAN; sharp but can hallucinate |
| 2021 | Real-ESRGAN | Trains on a random chain of blur, noise, resize and JPEG ("blind SR") | Main fix for the synthetic-to-real gap when no real pairs exist |
| 2021 | SwinIR | Swin Transformer blocks for restoration | Transformer baseline in recent EO benchmarks |
| 2022 | HAT | Hybrid attention transformer | Strong PSNR results in recent comparisons |
| 2021+ | SR3, latent diffusion | Iterative denoising produces HR samples | Current EO frontier (Donike et al. 2025; DiffFuSR 2025); highest perceptual quality, highest hallucination risk, slow |

### 4.2 Sentinel-2-specific methods

- **DSen2** (Lanaras et al., 2018): globally trained CNN that brings the 20 m and 60 m bands to 10 m.
- **L1BSR** (Nguyen et al., CVPR EarthVision 2023): self-supervised SR of the 10 m bands on raw **L1B** data. Neighbouring Sentinel-2 detectors overlap, so the same ground is imaged twice at slightly different positions. That overlap supplies the training signal; no HR ground truth is needed. A cross-spectral registration network also aligns the bands. The authors report results comparable to supervised methods and target 2× as the realistic limit. **Note:** this project's GUI file names (`L1BSR-GUI.py`, `L1-BSR-GUI-ONNX.py`) refer to this method. The project's RCAN is not L1BSR; it runs on L2A data and trains on synthetic pairs. State this clearly if asked.
- **Satlas super-resolution** (Allen AI): ESRGAN trained on Sentinel-2/NAIP pairs. It takes 6–18 Sentinel-2 images per location and outputs 4× imagery. Its own page warns that outputs are sometimes wrong.
- **Multi-image SR with radiometric-consistency losses** (Razzak et al., 2021): keeps Sentinel-2's spectral calibration during SR and measures the effect on building delineation.
- **SEN2SR / ESA OpenSR**: an ESA-funded effort for trustworthy open-source SR of Sentinel-2 up to 2.5 m (University of València, University of Oxford, Brockmann Consult).
- **Latent diffusion for multispectral Sentinel-2** (Donike et al., IEEE JSTARS 2025): aims to add high-frequency detail without hallucination or spectral distortion, and outputs a per-pixel uncertainty map.
- **DiffFuSR** (Sarmad et al., IEEE TGRS 2025): a diffusion model sharpens RGB, then a fusion network brings all 12 L2A bands to a common 2.5 m grid. The authors report better reflectance fidelity and less hallucination than earlier methods on the OpenSR benchmark.
- **SEN4X**: combines multi-image and single-image SR to reach about 2.5 m, using Pléiades Neo as a prior. It is evaluated on urban land-cover segmentation in Hanoi.

### 4.3 The synthetic-degradation problem

This project's own report states the issue: LR made by bicubic downsampling tests "reconstruction under the chosen degradation, rather than recovery of independently observed spatial detail." The literature agrees. The `opensr-test` authors note that most models are tested on synthetic data, which raises doubts about real-world performance.

Three ways to close the gap, in order of strength:

1. **Real paired data** from a second sensor: SEN2VENµS, WorldStrat, SEN2NAIP (Section 5).
2. **Self-supervision** on real Sentinel-2 alone: L1BSR's detector overlap, or DSen2-style reduced-scale training.
3. **Realistic synthetic degradation**: blur with the Sentinel-2 MTF rather than a bicubic kernel, add sensor noise, and randomise the degradation (Real-ESRGAN style). SEN2NAIP learned a degradation model from real pairs to create 35,314 more synthetic pairs.

---

## 5. Datasets

### 5.1 Super-resolution

| Dataset | LR | HR | Scale | Size | Bands | Fit for this project |
| --- | --- | --- | --- | --- | --- | --- |
| **SEN2VENµS** (Michel et al., 2022) | Sentinel-2 10 m and 20 m, same-day | VENµS 5 m | **2×** (10→5), 4× (20→5) | 132,955 patch triplets, 29 sites, ~116 GB | **B2, B3, B4, B8**, B5, B6, B7, B8A (later versions add B11, B12) | **Best match**: same four bands, same 2× scale, real sensor pairs. Stored as int16, divide by 10,000: the project's scaling convention |
| **SEN2NAIP** (Aybar et al., Scientific Data 2024) | Sentinel-2 L2A 10 m | NAIP aerial, 2.5 m | 4× | 2,851 real pairs (1.46 km² each), plus 35,314 synthetic pairs | RGB + NIR | Good match (RGB-NIR); USA only; 4× |
| **S2-NAIP** (Satlas; Wolters et al., 2023) | Sentinel-2 time series (≥18 per tile) | NAIP | 4× | ~1.2 million pairs | RGB (+ others) | Large; multi-image; USA only |
| **WorldStrat** (Cornebise et al., NeurIPS 2022) | Multiple Sentinel-2 revisits | SPOT 6/7, 1.5 m | ~6× | ~10,000 km², stratified by land use worldwide | RGB-NIR (+ S2 bands) | Global coverage, including India; HR imagery is CC BY-NC |
| **PROBA-V** (ESA Kelvins challenge, 2019) | PROBA-V 300 m (multi-frame) | PROBA-V 100 m | 3× | ~1,450 scenes | Red, NIR | Classic multi-image SR benchmark; not Sentinel-2 |
| **L1BSR overlap set** (Zenodo, 2023) | Sentinel-2 L1B detector overlaps | none (self-supervised) | 2× | 3,740 crop pairs | 10 m bands | For self-supervised training |

### 5.2 Land cover

| Dataset | Imagery | Classes | Notes |
| --- | --- | --- | --- |
| **OpenEarthMap** (Xia et al., WACV 2023) | Aerial/satellite, 0.25–0.5 m | 8 | 5,000 images, 97 regions, 44 countries. **The project's SparseMask and SegFormer were trained on this.** |
| **LoveDA** (Wang et al., NeurIPS 2021) | 0.3 m, three Chinese cities | 7 | 5,987 images; urban vs rural domain shift. **The project's accuracy audit uses 8 LoveDA images.** |
| **Dynamic World** (Brown et al., Scientific Data 2022) | Sentinel-2, 10 m | 9 | Fully convolutional network; near-real-time global labels; usable as Sentinel-2 training labels |
| **ESA WorldCover** (2020, 2021) | Sentinel-1 + Sentinel-2, 10 m | 11 | Global annual map; random-forest based; stated target 75% overall accuracy |

The project's real detector was trained on 0.25–0.5 m imagery and runs on 10 m Sentinel-2: a 20–40× resolution gap. The project documents this domain shift. Sentinel-2-native labels (Dynamic World, WorldCover) are the obvious way to close it.

---

## 6. How super-resolution is evaluated

### 6.1 Classic metrics, and why they are not enough

| Metric | Measures | Limitation |
| --- | --- | --- |
| **PSNR** | Pixel-wise error (log of inverse MSE) | Rewards blurry averages; ignores structure |
| **SSIM** | Local luminance, contrast and structure similarity | Better than PSNR, still favours smooth outputs |
| **SAM** (spectral angle) | Angle between true and predicted spectra per pixel | Essential for multispectral data; this project reports it |
| **LPIPS / FID / NIQE** | Perceptual similarity or "naturalness" | Can reward invented texture |
| **Downstream task score** | Segmentation IoU, detection F1 on SR output | Closest to real usefulness; needs labels |

The **perception–distortion trade-off** (Blau & Michaeli, 2018) proves that no method can minimise distortion (PSNR, SSIM) and maximise perceptual quality at the same time. That is why GAN and diffusion models look sharper but score lower PSNR than L1-trained models such as this project's RCAN. The 2026 ISPRS benchmark by Mühlhaus et al. found the same trade-off among SR architectures on Sentinel-2 and VENµS, and therefore added a downstream field-boundary task.

### 6.2 Evaluation protocols

- **Reduced-resolution (Wald) protocol**: degrade real imagery, super-resolve it, compare with the original. A reference exists, but this assumes the model transfers across scales. This is the project's current protocol, with synthetic scenes.
- **Cross-sensor protocol**: compare SR Sentinel-2 with a same-day image from a finer sensor (VENµS, NAIP, SPOT). Most realistic, but registration errors and sensor differences add noise.
- **No-reference checks**: downsample the SR output and compare with the original LR input (consistency); measure added high-frequency energy. The project's GUI reports similarity to bicubic and edge energy, and its documentation correctly says these are not accuracy.

### 6.3 `opensr-test`: the current Sentinel-2 benchmark

Aybar et al. (IEEE GRSL, 2024) built a benchmark specifically for Sentinel-2 SR, with curated cross-sensor datasets at several scale factors. It scores three independent groups:

1. **Consistency**: does the SR output, degraded back to 10 m, still match the original image? (Reflectance and spectral preservation.)
2. **Synthesis**: how much new high-frequency detail was added?
3. **Correctness**: of the added detail, how much is real (matches the HR reference) and how much is hallucinated or omitted? The package reports percentages for hallucination, omission and improvement.

It installs with `pip install opensr-test` and is the natural next evaluation for this project's RCAN.

---

## 7. "Super-resolution mapping": the original meaning

In remote sensing, **super-resolution mapping (SRM)**, also called **sub-pixel mapping**, predates deep-learning image SR. Peter Atkinson introduced it in the late 1990s. The method:

1. **Soft classification / spectral unmixing** estimates the *fraction* of each class inside every coarse pixel (for example, 40% water, 60% vegetation).
2. **Sub-pixel allocation** splits each coarse pixel into finer cells and assigns classes so that the fractions are honoured and the map is spatially smooth (the *spatial dependence* assumption).

Classic allocation methods include pixel swapping, spatial attraction models, Hopfield neural networks, Markov random fields and feed-forward neural networks. **Ling & Foody (2019)** first applied deep learning: a CNN learns the coarse-fractions-to-fine-map relation directly.

The project title ("Super-Resolution *Mapping* from Medium-Resolution Satellite Imagery") matches this term. The implementation does a newer variant: **image SR first, then classification**. A third variant, used by SEG-ESRGAN and Google's building/road model, predicts a **fine-resolution map directly from coarse imagery** without producing an SR image. Naming all three in the report and viva shows command of the field:

| Approach | Output | Example |
| --- | --- | --- |
| Classic SRM / sub-pixel mapping | Fine map from class fractions | Atkinson (2005); Ling & Foody (2019) |
| SR image → segmentation (two stages) | SR image, then map | **This project**; Hollendonner (2025) |
| Direct fine-resolution mapping (one model) | Fine map from coarse image | SEG-ESRGAN (2022); Sirko et al. (2023) |

---

## 8. Land-cover mapping from Sentinel-2

### 8.1 Segmentation models

- **U-Net** (2015): encoder–decoder with skip connections; still the standard EO baseline. The PANGAEA benchmark found a plain U-Net competitive with many foundation models.
- **DeepLabV3+, HRNet**: common stronger CNN baselines.
- **SegFormer** (2021): lightweight transformer encoder with a simple MLP decoder; used in this project (B0, the smallest variant).
- **Dynamic World's FCNN**: the production model behind a global 10 m map.

### 8.2 Geospatial foundation models

Large models pretrained on unlabeled satellite data, then fine-tuned with few labels. Examples: **Prithvi** (NASA/IBM; trained on Harmonized Landsat-Sentinel-2), **SatMAE**, **CROMA**, **DOFA** (trained on Sentinel-1, Sentinel-2, Gaofen, NAIP and EnMAP), **Clay**. Benchmarks: **GEO-Bench** and **PANGAEA** (2024). PANGAEA found that models pretrained on lower-resolution imagery often fall behind on high-resolution tasks, except CROMA and DOFA, and that a U-Net remains a strong baseline. A foundation model fine-tuned on Sentinel-2 labels is a credible "future work" alternative to the project's high-resolution-trained detector.

### 8.3 How accurate are 10 m land-cover maps?

A 2022 comparison of the three global 10 m products found overall accuracy of about **75% (Esri), 72% (Dynamic World) and 65% (WorldCover)** against global reference data, and a different order against European LUCAS data (WorldCover 71%, Dynamic World 66%, Esri 63%). Water was the most accurate class (92%), followed by built area (83%). Grass (34%), shrub (47%) and bare ground (57%) were the weakest. Other validations rank the products differently, so accuracy depends heavily on region and reference data.

This puts the project's numbers in context. Its combined detector reached **79.80% pixel accuracy** on eight LoveDA images, with **bare land at 0.21% IoU**. That is in the range of production 10 m global maps, and bare ground is a known weak class across the field. (The datasets and class sets differ, so this is context, not a like-for-like comparison.)

---

## 9. Does super-resolution help mapping?

The most important question for this project, and the literature has no settled answer.

| Study | Setup | Finding |
| --- | --- | --- |
| Hollendonner (TU Wien thesis, 2025) | Sentinel-2 10 m → 2.5 m SR vs interpolation, U-Net building delineation, Austria | "Super-resolution currently offers no advantage over interpolation methods" for building delineation |
| Sirko et al. (Google, 2023/2024) | Multiple Sentinel-2 frames → 50 cm building/road masks; student model learns from a high-resolution teacher | 79.0% building mIoU vs 85.5% for the high-resolution teacher; building-count R² 0.91 |
| SEG-ESRGAN (Salgueiro et al., 2022) | Joint SR + segmentation, Sentinel-2 / WorldView-2 pairs, 5× | Outperforms U-Net, DeepLabV3+ and HRNet on 10 m inputs |
| SEN4X | Multi-image + single-image SR with Pléiades Neo prior | Reports clearly better urban land-cover segmentation (Hanoi) |
| Mühlhaus et al. (ISPRS Annals, 2026) | Benchmark of SR models on Sentinel-2 / VENµS + field-boundary detection | Trade-off between distortion and perceptual metrics; downstream task added as a validation step |
| Politecnico di Torino thesis (2025) | SR for downstream use | Notes that SR output is often unsuitable as input for land-cover segmentation |

**The pattern:** gains appear when (a) SR and segmentation are trained together, (b) multiple revisits add real information, or (c) a high-resolution teacher supervises the output. Plain "SR image, then off-the-shelf classifier" rarely beats feeding interpolated imagery to a classifier trained on it. A single image at 10 m holds limited information, and SR cannot create information that was never measured.

The project already handles this carefully: its real detector classifies the **original** image, not the RCAN output, and the documentation says the 2× GeoTIFF "changes sampling, without demonstrating twice the real spatial resolving power." The literature supports that choice.

---

## 10. Where this project stands

### Strengths, judged against the literature

- **Honest evaluation.** It separates synthetic metrics from real-world claims, labels GUI metrics as similarity to bicubic, and does not claim accuracy it has not measured. This matches the field's main recommendation since 2022.
- **Multispectral SR with a spectral metric.** It super-resolves four bands including NIR and reports SAM; many student projects stop at RGB and PSNR.
- **A sensible architecture.** RCAN with a bicubic skip and a zero-initialised head starts exactly at the bicubic baseline, a standard and stable design. Channel attention suits multispectral data.
- **Real data and provenance.** Genuine Sentinel-2 L2A crops with STAC metadata, checksums and preserved georeferencing.
- **Correct pipeline order.** Classifying the original image avoids feeding hallucinated detail to the classifier.
- **Deployable.** CPU and ONNX export, tiled inference and GeoTIFF output.

### Gaps, in order of importance

1. **RCAN is trained only on synthetic scenes with bicubic degradation.** The main gap relative to current work. It cannot show that RCAN recovers real detail on Sentinel-2.
2. **There is no real HR reference for the Sentinel-2 demo scenes**, so SR quality on real data is unmeasured.
3. **The detector was trained on 0.25–0.5 m imagery and runs on 10 m Sentinel-2.** This domain shift explains weak road, building and bare-land results.
4. **Downstream benefit is untested.** No experiment compares map quality from bicubic input vs RCAN input.
5. **No SWIR bands.** B11/B12 help separate built-up from bare land (NDBI), the weakest class.

---

## 11. Recommendations (ordered by value for effort)

1. **Fine-tune RCAN on SEN2VENµS.** Its 10 m Sentinel-2 patches (128×128) contain B2, B3, B4 and B8, and its VENµS references are 5 m (256×256): exactly the project's 4-band, 2× contract and its 10,000 scaling. A few sites suffice for a student project; the full dataset is ~116 GB. The existing `prepare_dataset.py` already accepts measured LR, so this mostly needs a converter. Report PSNR, SSIM and SAM against **bicubic** on held-out sites.
2. **Run `opensr-test`** on the retrained model to report consistency, synthesis and hallucination, the measures current Sentinel-2 SR papers use.
3. **Add a downstream ablation.** Use the same segmentation model and labels, and compare three inputs: (a) native 10 m, (b) bicubic 2×, (c) RCAN 2×. Dynamic World or ESA WorldCover can supply 10 m labels. Whatever the result, it answers the project's own objective ("assess whether super-resolution improves geographic feature extraction and segmentation") and matches the open question in Section 9.
4. **If real pairs are not possible, make the synthetic degradation realistic.** Replace bicubic with a Gaussian blur matched to the Sentinel-2 MTF (0.15–0.30 at Nyquist), add noise, and randomise the parameters (Real-ESRGAN style). Then evaluate on real Sentinel-2 using the reduced-resolution protocol: 10 m → 20 m → back to 10 m.
5. **Fine-tune a segmentation model on Sentinel-2-native labels** (Dynamic World, WorldCover) to close the resolution domain gap. A small U-Net or SegFormer on four bands, or a foundation model such as Prithvi, would work.
6. **Optional extensions:** add B11/B12 with DSen2-style band sharpening; multi-image SR using several Sentinel-2 revisits (the project already downloads by date); try a joint SR + segmentation loss (SEG-ESRGAN idea).

---

## 12. Likely viva questions and short answers

- **Does your model create real 5 m resolution?** No. It produces a 5 m *grid*. On real Sentinel-2 the model is not yet validated against a real 5 m reference. The 10 m bands have an MTF of 0.15–0.30 at Nyquist, so physically recoverable detail is limited; L1BSR's authors treat about 2× as the realistic limit.
- **Why RCAN rather than a GAN or diffusion model?** RCAN trained with L1 loss stays close to the measured data and rarely invents structures. GAN and diffusion models look sharper but can hallucinate: the perception–distortion trade-off.
- **Why is PSNR not enough?** It rewards smooth, average-looking outputs and ignores whether added detail is real. Current benchmarks add spectral (SAM), consistency and hallucination measures, plus downstream task scores.
- **Why classify the original image instead of the enhanced one?** The detector was trained on real imagery, not SR output. Recent studies show SR input often does not improve segmentation and can add false detail.
- **What does "super-resolution mapping" mean?** Originally, sub-pixel mapping: a fine land-cover map from coarse class fractions (Atkinson; Ling & Foody 2019). This project does SR of the image, then classification.
- **How would you validate on real data?** SEN2VENµS (same-day 5 m VENµS), WorldStrat (SPOT 1.5 m), or SEN2NAIP (NAIP 2.5 m) as references; `opensr-test` for scoring; Dynamic World/WorldCover labels for the downstream test.
- **Why is 98% synthetic accuracy not meaningful?** The procedural scenes have simple shapes and distinct spectra. Real 10 m maps from Google, Esri and ESA reach about 65–75% overall accuracy on global reference data.

---

## References

**Sentinel-2 and evaluation**
- ESA. *Sentinel-2 MSI Technical Guide: Mission Performance* (MTF specification). https://sentinel.esa.int/web/sentinel/technical-guides/sentinel-2-msi/mission-performance
- Aybar, C., Montero, D., Donike, S., Kalaitzis, F., Gómez-Chova, L. (2024). A Comprehensive Benchmark for Optical Remote Sensing Image Super-Resolution. *IEEE GRSL* 21. DOI 10.1109/LGRS.2024.3401394. Paper: https://roderic.uv.es/bitstreams/6b2cfbe3-8b73-4852-aa72-dfa2c9ca0d23/download · Package: https://pypi.org/project/opensr-test/
- Blau, Y., Michaeli, T. (2018). The Perception-Distortion Tradeoff. *CVPR*. https://arxiv.org/abs/1711.06077
- Wald, L., Ranchin, T., Mangolini, M. (1997). Fusion of satellite images of different spatial resolutions: assessing the quality of resulting images. *Photogrammetric Engineering & Remote Sensing* 63(6).
- Mühlhaus et al. (2026). SR benchmark on Sentinel-2 and VENµS with downstream field-boundary detection. *ISPRS Annals* XI-3-2026, 215. https://isprs-annals.copernicus.org/articles/XI-3-2026/215/2026/
- Seydi, S. T., Arefi, H. (2023). Comparison of deep-learning SR models on Sentinel-2 urban imagery. *ISPRS Annals* X-1-W1-2023, 1021. https://isprs-annals.copernicus.org/articles/X-1-W1-2023/1021/2023/
- Wang, P., Bayram, B., Sertel, E. (2022). A comprehensive review on deep learning based remote sensing image super-resolution methods. *Earth-Science Reviews* 232.

**Super-resolution models**
- Dong, C. et al. (2014). SRCNN. https://arxiv.org/abs/1501.00092
- Lim, B. et al. (2017). EDSR. https://arxiv.org/abs/1707.02921
- Zhang, Y. et al. (2018). RCAN: Image Super-Resolution Using Very Deep Residual Channel Attention Networks. https://arxiv.org/abs/1807.02758
- Wang, X. et al. (2018). ESRGAN. https://arxiv.org/abs/1809.00219
- Wang, X. et al. (2021). Real-ESRGAN. https://arxiv.org/abs/2107.10833
- Liang, J. et al. (2021). SwinIR. https://arxiv.org/abs/2108.10257
- Chen, X. et al. (2022). HAT. https://arxiv.org/abs/2205.04437
- Saharia, C. et al. (2021). SR3: Image Super-Resolution via Iterative Refinement. https://arxiv.org/abs/2104.07636

**Sentinel-2 super-resolution**
- Lanaras, C. et al. (2018). Super-resolution of Sentinel-2 images: Learning a globally applicable deep neural network (DSen2). https://arxiv.org/abs/1803.04271 · Code: https://github.com/lanha/DSen2
- Nguyen, N. L. et al. (2023). L1BSR: Exploiting Detector Overlap for Self-Supervised SISR of Sentinel-2 L1B Imagery. *CVPRW*. https://arxiv.org/abs/2304.06871 · Code: https://github.com/centreborelli/L1BSR · Data: https://zenodo.org/record/7826696
- Razzak, M. T. et al. (2021). Multi-Spectral Multi-Image Super-Resolution of Sentinel-2 with Radiometric Consistency Losses and Its Effect on Building Delineation. https://arxiv.org/abs/2111.03231
- Donike, S. et al. (2025). Trustworthy Super-Resolution of Multispectral Sentinel-2 Imagery With Latent Diffusion. *IEEE JSTARS*.
- Sarmad, M. et al. (2025). DiffFuSR: Super-Resolution of all Sentinel-2 Multispectral Bands using Diffusion Models. *IEEE TGRS* 63. https://arxiv.org/abs/2506.11764
- Wolters, P., Bastani, F., Kembhavi, A. (2023). Zooming Out on Zooming In: Advancing Super-Resolution for Remote Sensing (S2-NAIP). https://arxiv.org/abs/2311.18082 · Demo: https://satlas.allen.ai/superres
- ESA OpenSR project (SEN2SR, opensr-test). https://github.com/ESAOpenSR
- Deudon, M. et al. (2020). HighRes-net: Recursive Fusion for Multi-Frame Super-Resolution of Satellite Imagery. https://arxiv.org/abs/2002.06460
- Molini, A. B. et al. (2019). DeepSUM: Deep neural network for super-resolution of unregistered multitemporal images. https://arxiv.org/abs/1907.06490

**Super-resolution datasets**
- Michel, J. et al. (2022). SEN2VENµS, a Dataset for the Training of Sentinel-2 Super-Resolution Algorithms. *Data* 7(7), 96. https://hal.inrae.fr/hal-03904203 · Data: https://zenodo.org/record/6514159
- Aybar, C. et al. (2024). SEN2NAIP: A large-scale dataset for Sentinel-2 Image Super-Resolution. *Scientific Data*. https://doaj.org/article/bc28e820c10b4c1d86a63ae4846ed128
- Cornebise, J., Oršolić, I., Kalaitzis, F. (2022). Open High-Resolution Satellite Imagery: The WorldStrat Dataset. *NeurIPS Datasets & Benchmarks*. https://arxiv.org/abs/2207.06418

**Super-resolution mapping (sub-pixel)**
- Atkinson, P. M. (2005). Sub-pixel target mapping from soft-classified remotely sensed imagery. *Photogrammetric Engineering & Remote Sensing* 71(7).
- Ling, F., Foody, G. M. (2019). Super-resolution land cover mapping by deep learning. *Remote Sensing Letters*. https://nottingham-repository.worktribe.com/preview/1615988/RSL-DeepSRM-Revision-clean-0216.pdf
- Nguyen, M. Q., Atkinson, P. M., Lewis, H. G. (2006). Superresolution mapping using a Hopfield neural network with fused images. *IEEE TGRS* 44(3).

**Land cover and downstream use**
- Ronneberger, O. et al. (2015). U-Net. https://arxiv.org/abs/1505.04597
- Xie, E. et al. (2021). SegFormer. https://arxiv.org/abs/2105.15203
- Xia, J. et al. (2023). OpenEarthMap: A Benchmark Dataset for Global High-Resolution Land Cover Mapping. *WACV*. https://openaccess.thecvf.com/content/WACV2023/papers/Xia_OpenEarthMap_A_Benchmark_Dataset_for_Global_High-Resolution_Land_Cover_Mapping_WACV_2023_paper.pdf
- Wang, J. et al. (2021). LoveDA: A Remote Sensing Land-Cover Dataset for Domain Adaptive Semantic Segmentation. *NeurIPS*. https://arxiv.org/abs/2110.08733
- Brown, C. F. et al. (2022). Dynamic World, Near real-time global 10 m land use land cover mapping. *Scientific Data* 9.
- Venter, Z. S. et al. (2022). Global 10 m Land Use Land Cover Datasets: A Comparison of Dynamic World, World Cover and Esri Land Cover. *Remote Sensing* 14(16), 4101. https://www.mdpi.com/2072-4292/14/16/4101
- Land & Carbon Lab. Global land cover maps' accuracy and applications explained. https://landcarbonlab.org/insights/global-land-cover-maps-accuracy-applications/
- Marsocci, V. et al. (2024). PANGAEA: A Global and Inclusive Benchmark for Geospatial Foundation Models. https://arxiv.org/abs/2412.04204
- Salgueiro, L. et al. (2022). SEG-ESRGAN: A Multi-Task Network for Super-Resolution and Semantic Segmentation of Remote Sensing Images. *Remote Sensing* 14. https://accedacris.ulpgc.es/handle/10553/121364
- Sirko, W. et al. (2023/2024). High-Resolution Building and Road Detection from Sentinel-2. https://arxiv.org/abs/2310.11622
- Hollendonner, S. (2025). Evaluating Sentinel-2 super-resolution algorithms for automated building delineation. Diploma thesis, TU Wien. https://repositum.tuwien.at/handle/20.500.12708/221077 · Data: https://researchdata.tuwien.ac.at/records/vzxjz-q2h34
- SEN4X summary. https://www.emergentmind.com/topics/sen4x
