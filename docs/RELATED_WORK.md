# Related work: prior work, competing tools and the research frontier

This review covers **who else is doing this**, **what has already been done**, and **what is newest**, as of October 2026.

Commercial claims below come from vendor websites and are **not independently validated** unless stated. Paper results are taken from abstracts or papers as noted.

---

## 1. Summary

1. **The idea of this project is already a commercial product category.** At least four companies sell or offer AI-enhanced satellite imagery: Gamma Earth (Sentinel-2 to 1 m), DigiFarm (Sentinel-2 to 1 m for field boundaries), Planet (PlanetScope 3 m to 2 m) and Maxar (30 cm to 15 cm "HD"). Free alternatives include Allen AI's Satlas and ESA's open-source OpenSR models (SEN2SR, LDSR-S2).
2. **The newest downstream evidence shows a small gain.** A May 2026 flow-matching paper (Hester et al.) gave three segmentation models the same task with different inputs. Super-resolved Sentinel-2 raised average F1 by about **0.16 points** over plain Lanczos upsampling (77.56% vs 77.40%), while real 2.5 m aerial imagery raised it by about **3.3 points** (80.74%). Lanczos had the highest overall accuracy of the Sentinel-2 inputs for every model. SR made images look better far more than it made maps better.
3. **The frontier has three directions:**
   - *Better generators*: diffusion → **flow matching** (one-step sampling, tunable sharpness) and Mamba state-space models.
   - *Trustworthiness*: per-pixel **uncertainty maps**, **hallucination metrics**, radiometric-consistency constraints and confidence layers in commercial products.
   - *Skipping the SR image*: predicting **fine-resolution maps directly** (Google's Open Buildings 2.5D Temporal, label super-resolution) or using **foundation-model embeddings** (Google DeepMind's AlphaEarth, IBM/ESA TerraMind).
4. **This project's distinctive niche** is an offline, CPU-only desktop tool that combines SR, land-cover mapping and GeoTIFF export, with candid metric reporting. No competitor found offers all of that together. Its main weakness against the field was training on synthetic data; RCAN has since been retrained on real Sentinel-2/VENµS pairs.
5. **A real research gap the project could fill:** published Sentinel-2 SR studies with Indian reference imagery are scarce. The closest found (GINet, 2025) only *validates* on Delhi. India's free 5.8 m **Resourcesat LISS-IV** could serve as a cross-sensor reference for Indian scenes. Open access to LISS-IV needs checking on Bhoonidhi (Section 7).

---

## 2. What has already been done: timeline

| Year | Milestone | Why it matters |
| --- | --- | --- |
| 1997–2005 | **Super-resolution mapping / sub-pixel mapping** (Atkinson) | Original meaning of the project title: fine land-cover maps from coarse class fractions |
| 2014 | **SRCNN** | First deep-learning SR |
| 2017–2018 | **EDSR**, **RCAN**, **ESRGAN** | RCAN is this project's model; ESRGAN underlies Satlas and Planet SuperRes |
| 2018 | **DSen2** (Lanaras et al.) | Global CNN sharpening Sentinel-2's 20/60 m bands to 10 m, trained only on Sentinel-2 itself |
| 2019 | **ESA PROBA-V challenge**; HighRes-net, DeepSUM | Made multi-image SR a mainstream research problem |
| 2019 | **Label super-resolution networks** (Malkin et al., ICLR) | 30 m labels used to train 1 m land-cover maps: map SR without image SR |
| 2020 | **Maxar HD** (15 cm from 30 cm) | First major commercial SR product for satellite imagery |
| 2022 | **WorldStrat**, **SEN2VENµS** datasets | Real cross-sensor training pairs become available |
| 2022 | **SEG-ESRGAN**; **L2HNet** | Joint SR + segmentation; low-to-high resolution land-cover training |
| 2023 | **L1BSR**; Nguyen et al. on aliasing and band shift | Self-supervised SR from detector overlap; physical reason Sentinel-2 SR works |
| 2023 | **S2-NAIP / Satlas SR** (Allen AI); **MuS2** benchmark (WorldView-2 reference) | Large-scale multi-image SR products and benchmarks |
| 2023 | **S2DR3** (Yosef Akhtman, October 2023) | Commercial-grade 12-band Sentinel-2 SR at a 1 m grid |
| 2023–2024 | **High-resolution building and road detection from Sentinel-2** (Google; Sirko et al.) | Multi-frame student model predicts 50 cm masks; basis of Open Buildings 2.5D Temporal |
| 2024 | **SEN2NAIP**; **opensr-test**; **DiffusionSat** (ICLR) | Standard real dataset, standard benchmark, first generative foundation model for satellite imagery |
| 2025 | **LDSR-S2** (latent diffusion with uncertainty), **DiffFuSR**, **SEN4X**, **SEN2SR** | Diffusion and Mamba SR for Sentinel-2; trust becomes central |
| 2025 | **TerraMind** (ICCV), **Galileo** (ICML), **Prithvi-EO-2.0**, **AlphaEarth Foundations** | Foundation models and embeddings reshape land-cover mapping |
| 2025 | Hollendonner (TU Wien) | SR showed no advantage over interpolation for building delineation |
| 2026 | **Flow matching for Sentinel-2** (Hester et al.), **FlowGS**, self-supervised SR uncertainty (UAI), ISPRS SR benchmark | Current frontier: one-step generative SR, arbitrary scale, calibrated uncertainty, downstream-task benchmarks |
| 2026 | **Planet SuperRes** scenes launch (May 2026) | Commercial SR with a per-pixel confidence layer |

---

## 3. Competitors

### 3.1 Commercial super-resolution products

| Provider | Product | Input → output | Bands | Method (as disclosed) | Trust features | Access |
| --- | --- | --- | --- | --- | --- | --- |
| **Gamma Earth** (Switzerland) | S2DR ("Sentinel-2 Deep Resolution"): S2DR3, S2DR4 | Sentinel-2 10 m → **1 m** grid | 10–12 bands; RGB, NDVI, IRP outputs | Proprietary single-image network. S2DR3's author says real detail is reconstructed down to about 3 m. S2DR4 adds cloud, haze and shadow removal | None published | API and SDK billed per km², free tier and Colab; research credits. Claims 250 M km² processed in 180 countries |
| **DigiFarm** (Norway) | Deep Resolution + field boundaries | Sentinel-2 10 m → **1 m** | Multispectral, calibrated | Proprietary 10× SR, then a boundary-detection network | Vendor-reported 0.94+ IoU on field boundaries | SaaS/API. Says it won an Alberta (AFSC) tender against PlanetScope |
| **Planet** | **SuperRes** (PlanetScope scenes and mosaics) | PlanetScope 3 m → **2 m** | 4-band | ESRGAN trained on 120,000+ SkySat/PlanetScope pairs | **Per-pixel confidence layer**; Planet advises human review of AI-enhanced data | Planet Insights Platform; scenes launched May 2026 |
| **Maxar** | **HD** imagery | 30 cm → **15 cm**, 50 cm → 30 cm | Commercial VHR | Proprietary; algorithm not published | States native resolution is unchanged | Commercial licence |

**Free or research-oriented services**

| Provider | Product | What it offers |
| --- | --- | --- |
| **Allen AI** | Satlas super-resolution | ESRGAN on 6–18 Sentinel-2 images per place → 4×, trained on Sentinel-2/NAIP; open code; site warns outputs are sometimes wrong |
| **ESA OpenSR** | SEN2SR, LDSR-S2, opensr-test, web viewer | Open models to 2.5 m, benchmark and validation tooling (Section 3.2) |
| **Google** | Open Buildings 2.5D Temporal | Annual building presence, count and height for 2016–2023 from Sentinel-2, at about 4 m effective resolution; Global South; CC BY 4.0 / ODbL |

### 3.2 Open-source tools (closest to this project)

| Tool | What it does | Scale / output | Notes |
| --- | --- | --- | --- |
| **SEN2SR** (ESA OpenSR) | Sentinel-2 SR package | 10 m and 20 m bands → **2.5 m** | Lite CNN model and a full model using a **Mamba** architecture; weights via `mlstac`; also published as an Esri ArcGIS item |
| **opensr-model / LDSR-S2** (ESA OpenSR) | Latent-diffusion SR of RGB-NIR | 10 m → 2.5 m | Out of the experimental phase at v1.0; MIT licence; **pixel-level uncertainty maps**; IEEE JSTARS 2025 |
| **opensr-test** | Benchmark | Consistency, synthesis, hallucination | `pip install opensr-test` |
| **ESA OpenSR SRGAN** | Configurable SRGAN-family framework for multispectral EO | Any | Described as a practical tool, not state of the art |
| **ESA OpenSR validation repos** | `Segmentation-Models-Benchmark`, `spectral_usecases`, `SEN2NEON` | Downstream, spectral and SWIR validation of SR | Shows ESA treats downstream validation as essential |
| **SR4RS** (OTBTF) | CNN SR trained on 250 SPOT-6/7 scenes paired with Sentinel-2 | RGB-NIR 10 m → **2.5 m** | Docker; users can retrain on their own data |
| **DSen2** | Band sharpening | 20/60 m → 10 m | Classic baseline |
| **s2super** | Inception-ResNet band sharpener | 20 m → 10 m | Sharpens only where the model is confident |
| **L1BSR** | Self-supervised SISR on L1B | 10 m → 5 m | GPL-3.0; the method this project's GUI file names refer to |
| **satlas-super-resolution** | Multi-image ESRGAN | 4× | Training and inference code |
| **WorldStrat package** | Dataset rebuild and MISR baselines | ~6× | BSD code |
| **Esri `arcgis.learn` SuperResolution** | Train-your-own SR inside ArcGIS | Any | Trains on downsampled aerial imagery: the synthetic-pair approach this project originally used |
| **TorchGeo / TerraTorch** | PyTorch geospatial datasets, samplers and foundation-model fine-tuning | n/a | Infrastructure rather than SR models |

### 3.3 Land-cover products (competitors for the mapping half)

| Product | Resolution | Method | Reported accuracy |
| --- | --- | --- | --- |
| **Google Dynamic World** | 10 m, near real time | FCNN per Sentinel-2 scene | ~72% overall (2022 global comparison) |
| **ESA WorldCover** | 10 m, 2020/2021 | Random forest, Sentinel-1 + 2 | ~65% global, 71% Europe (2022 comparison); ~77% in its own validation |
| **Esri / Impact Observatory LULC** | 10 m, annual 2017 onward (v3 to 2024) | Deep learning on Sentinel-2 | ">75%" average (vendor) |
| **AlphaEarth Foundations** (Google DeepMind) | 10 m, 64-band annual embeddings, 2017 onward | Foundation-model embedding field | 23.9% average error reduction vs other featurisations (authors) |
| **Hester et al. Chesapeake product** | 2.5 m annual land cover, 2020–2025 | Flow-matching SR + segmentation | 89.11% overall accuracy on 25,000 points (5 classes) |
| **SinoLC-1** | 1 m, all of China | L2HNet trained on low-resolution labels | 74% OA, kappa 0.65 |

### 3.4 Where this project fits

| Feature | This project | Gamma Earth S2DR | SEN2SR / LDSR-S2 | Satlas | Planet SuperRes |
| --- | --- | --- | --- | --- | --- |
| Input | Sentinel-2 4 bands; any RGB image | Sentinel-2 | Sentinel-2 | Sentinel-2 time series | PlanetScope |
| Scale | 2× | 10× (1 m grid) | 4× (2.5 m) | 4× | 1.5× |
| Trained on real sensor pairs | **Yes (SEN2VENµS, since Oct 2026)** | Undisclosed | Yes (SEN2NAIP family) | Yes (NAIP) | Yes (SkySat) |
| Land-cover mapping included | **Yes** | NDVI only | No (separate benchmarks) | Separate Satlas models | No |
| Uncertainty / confidence | Uncalibrated softmax scores only | No | **Yes** (LDSR-S2) | No | **Yes** |
| Offline, CPU, desktop GUI | **Yes** | No (cloud) | Python library | No | No |
| GeoTIFF export with CRS | **Yes** | Yes | Yes | Yes | Yes |
| Open source | Yes | No | Yes (MIT) | Yes | No |
| Candid limits documented | **Yes, extensive** | Limited | Yes | Brief warning | Yes |

**Positioning in one sentence:** a transparent, offline, end-to-end SR-plus-mapping teaching and prototyping tool, not a competitor to commercial SR quality. To be credible against the open tools, RCAN needs real-pair training, or SEN2SR should be offered as a comparison model inside the app.

---

## 4. The research frontier (2025–2026)

### 4.1 Generative SR: from GANs to diffusion to flow matching

- **Diffusion models** produce the most realistic textures but need many steps and can hallucinate.
  - *LDSR-S2* (Donike et al., JSTARS 2025) conditions on the encoded low-resolution image for spectral consistency and is probabilistic, so it can output a pixel-wise uncertainty map.
  - *DiffFuSR* (Sarmad et al., TGRS 2025) applies diffusion to RGB, then fuses the result into all 12 bands at 2.5 m.
  - *DiffusionSat* (Khanna et al., ICLR 2024) conditions on latitude, longitude, date and ground-sampling distance; it handles SR, inpainting and temporal generation through ControlNet-style adapters.
- **Flow matching** is the newest family. *Hester et al. (2026)* trained on 120,851 same-day Sentinel-2/NAIP pairs for 4× SR of RGB-NIR:

  | Method | PSNR (dB) | SSIM | LPIPS (lower is better) |
  | --- | --- | --- | --- |
  | Lanczos upsampling | 32.99 | 0.813 | 0.907 |
  | Real-ESRGAN | 33.29 | 0.786 | **0.194** |
  | Diffusion DDPM, 100 steps | 33.31 | 0.803 | 0.336 |
  | **Flow matching, 1 Euler step** | **34.95** | **0.832** | 0.374 |
  | Flow matching, 20 Midpoint steps | 32.64 | 0.771 | 0.247 |

  One step gives the most accurate output; more steps give sharper but less faithful output. This lets a user choose a point on the perception–distortion trade-off **at inference time without retraining**. Diffusion failed completely at one step (6 dB PSNR).
- **FlowGS** (arXiv, May 2026) combines flow matching with 2D Gaussian splatting for **arbitrary-scale** RS super-resolution (any zoom factor, not just 2× or 4×).
- **Semantic-guided flow matching** (arXiv 2510.23816) conditions cross-sensor SR on semantic maps as well as the LR image.

### 4.2 New backbones: Mamba / state-space models

Mamba models process images as sequences with linear cost, giving transformer-like global context at CNN-like speed. Examples: **FMSR** (frequency-assisted Mamba for RS SR, 2024), the full **SEN2SR** model, and **RoMA** (a Mamba-based RS foundation model, 2025). Two 2025 surveys cover Vision Mamba in remote sensing.

### 4.3 Physics-aware SR

- **Why Sentinel-2 SR works at all:** Nguyen et al. (2023) attribute single-image Sentinel-2 SR to the sensor's **aliasing and inter-band shift**. Each band samples the ground at a slightly different position, so the bands together hold sub-pixel information. They also found that training with a plain **L1 loss gives results without hallucinated details**. This project's RCAN uses L1 loss, which is good, but the L2A products it uses are resampled to a common grid, which removes much of the inter-band shift. L1BSR works on L1B for this reason.
- **Radiometric consistency:** constraining the SR output so that, degraded back to 10 m, it reproduces the measured Sentinel-2 values (Razzak et al. 2021; the opensr-test consistency metrics).
- **Band sharpening with geometry:** *GINet* (2025) uses 10 m band geometry to guide 20 m SR, validated on Delhi.

### 4.4 Trust: uncertainty and hallucination

- **Uncertainty maps**: LDSR-S2 (diffusion sampling); **Zheng, Dewil & Arias (UAI 2026)** estimate calibrated SR uncertainty **without HR ground truth**, using a self-supervised loss, tested on SkySat.
- **Hallucination scoring**: opensr-test reports how much added detail is correct, hallucinated or omitted.
- **Product practice**: Planet ships a per-pixel confidence layer and recommends human review. Maxar says native resolution is unchanged. Satlas warns outputs are sometimes wrong. The field's norm is now to **label SR imagery as derived, never as observed**.
- **Misuse risk**: a 2021 University of Washington study showed AI can create convincing fake satellite images of real places ("deepfake geography") and called for geographic fact-checking. Labelled outputs and stored provenance, as this project's reports already do, are the accepted mitigation.

### 4.5 Skip the SR image: map directly

A strong trend: if the goal is a map, predict the fine map directly.

- **Teacher–student mapping**: Google trains a Sentinel-2 "student" to copy a high-resolution "teacher". Result: 50 cm building/road masks with 79.0% building mIoU (teacher 85.5%) and a building-count R² of 0.91. Now a public dataset (Open Buildings 2.5D Temporal).
- **Label super-resolution**: train fine-resolution maps from coarse labels (Malkin et al. 2019: 30 m labels → 1 m maps). **L2HNet** extends this to national scale (SinoLC-1, 1 m China). Newer work (a 2025 dual-branch local–global framework) tackles L2HNet's boundary noise.
- **Embeddings instead of pixels**: AlphaEarth gives a 64-number description of every 10 m pixel each year. A simple classifier on these embeddings can replace a full segmentation network for many tasks.

### 4.6 Foundation models for land cover

| Model | From | Key idea |
| --- | --- | --- |
| **TerraMind** | IBM, ESA, Jülich (ICCV 2025) | Any-to-any generative model over nine modalities. "Thinking-in-Modalities" generates missing inputs (for example, a land-cover map or SAR from optical) |
| **Prithvi-EO-2.0** | NASA, IBM (2024) | 300 M/600 M-parameter multi-temporal model on HLS 30 m data; +8% over v1 on GEO-Bench |
| **Galileo** | NASA Harvest et al. (ICML 2025) | One encoder for optical, SAR, elevation, weather and land-cover inputs; beats specialist models on 11 benchmarks |
| **DOFA, CROMA, SatMAE, Clay** | various | Multi-sensor or multispectral pretraining; PANGAEA found CROMA and DOFA strongest on high-resolution tasks |
| **AlphaEarth Foundations** | Google DeepMind (2025) | Annual global 10 m embedding layers in Earth Engine |

The PANGAEA benchmark adds a caution: a well-trained **U-Net is still competitive** with many foundation models.

### 4.7 Promptable and open-vocabulary segmentation

**SegEarth-OV** (2024) and **SegEarth-OV3** (2025, built on Meta's SAM 3) segment remote-sensing images from **text prompts without training**: "roads", "solar panels", "ponds". SegEarth-OV3 reports evaluation on 20 segmentation datasets and adds open-vocabulary change detection. These could replace a fixed six-class detector with user-chosen classes, though they target high-resolution imagery.

---

## 5. What the evidence says, overall

| Question | Current answer | Strength of evidence |
| --- | --- | --- |
| Does deep SR beat interpolation on image metrics? | Yes, by 1–6 dB PSNR depending on data (2 dB on real Sentinel-2/NAIP in Hester 2026; 5.7 dB on this project's synthetic data) | Strong |
| Does SR add *real* detail on Sentinel-2? | Some, up to about 2× (aliasing and band shift); beyond that, mostly learned priors | Moderate (L1BSR, Nguyen 2023, S2DR3 author's 3 m claim) |
| Does SR make land-cover maps better? | **Marginally**: ~0.1–0.2 F1 points over Lanczos in the best 2026 study; no gain for buildings in 2025 study; larger gains only from multi-image or jointly trained methods | Moderate and consistent |
| Is real HR imagery still better than SR? | Yes: about +3 F1 points over any Sentinel-2 input in Hester 2026 | Strong |
| Do generative models hallucinate? | Yes, more with more sampling steps; L1-trained and one-step models less | Strong |
| Are commercial 1 m Sentinel-2 products validated? | Only by vendors (for example, DigiFarm's 0.94+ IoU); no independent validation found | Weak |

**Implication for this project:** the right claim is that SR improves *viewing and sampling*, and that its value for *mapping* is small and must be measured. The project's documentation already takes this position. The literature now supports it with numbers.

---

## 6. Open problems and opportunities

Areas the field has not settled, roughly ordered by fit for this project:

1. **Regional evaluation outside the USA and Europe.** Most real-pair datasets are American (NAIP) or European (VENµS sites, Pléiades). India-specific Sentinel-2 SR evaluation is scarce. Using **LISS-IV 5.8 m** as a same-season reference for Indian scenes would be a genuine contribution (note: 5.8 m vs 10 m is less than 2×, so it suits checking a 2× model only approximately; acquisition dates and spectral bands also differ).
2. **Honest downstream baselines.** Many SR papers omit a plain interpolation baseline. A clean three-way test (native, bicubic/Lanczos, SR) on Indian land cover is a useful contribution.
3. **Uncertainty in user tools.** Few end-user tools show where SR output is unreliable. A per-pixel uncertainty or "difference from bicubic" layer in the GUI would match current best practice (Planet, LDSR-S2).
4. **Cheap trustworthy SR.** Flow matching's one-step mode is fast and accurate; distilling it into a small CPU model like this project's RCAN is an open engineering problem.
5. **Hallucination detection without a reference.** Still unsolved in general; the UAI 2026 self-supervised uncertainty work is the closest.
6. **Arbitrary-scale SR** (FlowGS) and **semantic-guided SR** are early-stage and active.

---

## 7. Data alternatives to super-resolution

Sometimes the better answer is finer real data.

| Source | Resolution | Cost / access | Note |
| --- | --- | --- | --- |
| **Resourcesat-2/2A LISS-IV** (ISRO) | 5.8 m multispectral (green, red, NIR; no blue) | NRSC material states products coarser than 5 m are free and open; confirm the current policy on Bhoonidhi | Most relevant free finer-resolution source for India |
| **Planet NICFI basemaps** | 4.77 m RGB-NIR, tropics | Free programme **ended 24 January 2025** | No longer an option for new work |
| **PlanetScope / SkySat** | 3 m / 0.5 m | Commercial; research and education programmes exist | SuperRes adds 2 m |
| **NAIP** | 0.6–1 m, USA only | Free | Basis of SEN2NAIP and S2-NAIP |
| **SPOT 6/7 (via WorldStrat)** | 1.5 m | Non-commercial (CC BY-NC) for the WorldStrat subset | Global sample sites |
| **Sentinel-2C / 2D** | 10 m | Free | Continuity, not finer resolution |
| **Sentinel-2 Next Generation** | Finer than current MSI (AMSI instrument; a panchromatic band studied) | Future | Exact resolution and launch date not confirmed in sources found |

---

## 8. Status of this project against the review

Done:
- SEN2SR added as a comparison model (`src/satsr/models/sen2sr_lite.py`, `scripts/compare_sr_models.py`).
- RCAN retrained on SEN2VENµS; it now beats bicubic by +0.47 dB (compensated) on held-out Indian sites.
- Downstream test in the style of Hester et al.: same U-Net on native, bicubic, RCAN and SEN2SR inputs ([SR_COMPARISON.md](SR_COMPARISON.md)).
- Change-vs-bicubic map in the app's Compare view; enhanced exports labelled as derived imagery.

Still open:
- Score SEN2SR and the real-trained RCAN with `opensr-test` and on the SEN2VENµS test.
- Re-run the mapping experiment with the real-trained RCAN.
- An India-specific reference using same-season LISS-IV scenes.

---

## References

**Commercial products and services**
- Gamma Earth: Sentinel-2 Deep Resolution (S2DR3/S2DR4). https://www.gamma.earth/ · Research tier: https://www.gamma.earth/industries/science/
- S2DR3 description (Yosef Akhtman, 2023). https://medium.com/@ya_71389/c71a601a2253 · Comparison with SR4RS: https://www.geodose.com/2025/07/sentinel-2-super-resolution-sr4rs-s2dr3.html
- DigiFarm. https://earthdaily.com/blog/digifarm-field-boundaries-for-precise-in-season-analytics · https://business.esa.int/projects/digifarm
- Planet SuperRes. https://docs.planet.com/data/imagery/superres/ · Technical overview: https://www.planet.com/pulse/planet-superres-a-technical-overview · Scene launch: https://community.planet.com/product-updates/introducing-visual-analysis-and-superres-planetscope-scenes-6487
- Maxar 15 cm HD. https://blog.maxar.com/earth-intelligence/2020/introducing-15-cm-hd-the-highest-clarity-from-commercial-satellite-imagery · Native vs synthetic: https://blog.maxar.com/earth-intelligence/2020/satellite-imagery-native-resolution-compared-to-synthetic-resolution
- Allen AI Satlas super-resolution. https://satlas.allen.ai/superres
- Google Open Buildings 2.5D Temporal. https://sites.research.google/gr/open-buildings/temporal · Earth Engine: https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_Research_open-buildings-temporal_v1

**Open-source tools**
- ESA OpenSR organisation (SEN2SR, opensr-model, opensr-test, SRGAN, validation repos). https://github.com/ESAOpenSR · SEN2SR: https://pypi.org/project/sen2sr/ · LDSR-S2: https://github.com/ESAOpenSR/opensr-model · Project page: https://cin.philab.esa.int/databases/projects/towards-explainable-ai-application-to-trustworthy-super-resolution-opensr-robust-accountable-super-resolution-for-sentinel-2-and-beyond
- SEN2SR as an Esri item. https://esri.hub.arcgis.com/content/79e770c5e09b4f31b7899291868c735a
- SR4RS. https://github.com/remicres/sr4rs
- s2super. https://pypi.org/project/s2super/
- Esri arcgis.learn SuperResolution. https://developers.arcgis.com/python/latest/samples/increase-image-resolution-using-superresolution/
- TorchGeo. https://arxiv.org/abs/2111.08872

**Generative and frontier SR**
- Hester, D. et al. (2026). Flow matching for Sentinel-2 super-resolution: implementation, application, and implications. https://arxiv.org/abs/2605.00367
- FlowGS: Flow-based Gaussian Splatting for Continuous-Scale Remote Sensing Image Super-Resolution (2026). https://arxiv.org/abs/2605.22147
- Semantic-Guided Cross-Sensor Super Resolution: A Gated Dual Conditioning Flow Matching Model (2025). https://arxiv.org/abs/2510.23816
- Donike, S. et al. (2025). Trustworthy Super-Resolution of Multispectral Sentinel-2 Imagery With Latent Diffusion. *IEEE JSTARS*. DOI 10.1109/JSTARS.2025.3542220
- Sarmad, M. et al. (2025). DiffFuSR. *IEEE TGRS* 63. https://arxiv.org/abs/2506.11764
- Khanna, S. et al. (2024). DiffusionSat: A Generative Foundation Model for Satellite Imagery. *ICLR*. https://arxiv.org/abs/2312.03606
- Retnanto, A. et al. (2025). Beyond Pretty Pictures: Combined Single- and Multi-Image Super-resolution for Sentinel-2 Images (SEN4X). https://arxiv.org/abs/2505.24799
- Pereira-Sánchez, I. et al. (2025). Super-Resolution of Sentinel-2 Images Using a Geometry-Guided Back-Projection Network with Self-Attention (GINet). https://arxiv.org/abs/2508.04729
- Cross-sensor super-resolution of irregularly sampled Sentinel-2 time series (2024). https://arxiv.org/abs/2404.16409
- Frequency-Assisted Mamba for Remote Sensing Image Super-Resolution (FMSR, 2024). https://arxiv.org/abs/2405.04964
- Vision Mamba in Remote Sensing: survey (2025). https://arxiv.org/abs/2505.00630
- Nguyen, N. L. et al. (2023). On the Role of Alias and Band-Shift for Sentinel-2 Super-Resolution. https://arxiv.org/abs/2302.11494
- Kawulok, M. et al. (2023). MuS2: A Real-World Benchmark for Sentinel-2 Multi-Image Super-Resolution. *Scientific Data*. https://arxiv.org/abs/2210.02745
- Mühlhaus et al. (2026). SR benchmark with downstream field-boundary detection. *ISPRS Annals* XI-3-2026. https://isprs-annals.copernicus.org/articles/XI-3-2026/215/2026/

**Trust, uncertainty and misuse**
- Zheng, Dewil, Arias (2026). Self-Supervised Uncertainty Estimation for Super-Resolution of Satellite Images. *UAI*. https://arxiv.org/abs/2603.14074
- University of Washington (2021). A growing problem of "deepfake geography". https://www.washington.edu/news/2021/04/21/a-growing-problem-of-deepfake-geography-how-ai-falsifies-satellite-images/

**Direct fine-resolution mapping**
- Sirko, W. et al. (2023/2024). High-Resolution Building and Road Detection from Sentinel-2. https://arxiv.org/abs/2310.11622
- Malkin, K. et al. (2019). Label Super-Resolution Networks. *ICLR*. https://www.microsoft.com/en-us/research/publication/label-super-resolution-networks/
- Li, Z. et al. (2022). Breaking the resolution barrier: A low-to-high network for large-scale high-resolution land-cover mapping using low-resolution labels (L2HNet). *ISPRS J. Photogramm. Remote Sens.* 192, 244–267.
- SinoLC-1: National-scale 1-m land-cover mapping for China (2023). https://arxiv.org/abs/2303.05305
- A Dual-Branch Local-Global Framework for Cross-Resolution Land Cover Mapping (2025). https://arxiv.org/abs/2512.19990

**Foundation models and segmentation**
- Jakubik, J. et al. (2025). TerraMind: Large-Scale Generative Multimodality for Earth Observation. *ICCV*. https://openaccess.thecvf.com/content/ICCV2025/html/Jakubik_TerraMind_Large-Scale_Generative_Multimodality_for_Earth_Observation_ICCV_2025_paper.html · Models: https://huggingface.co/ibm-esa-geospatial
- Szwarcman, D. et al. (2024). Prithvi-EO-2.0. https://arxiv.org/abs/2412.02732
- Tseng, G. et al. (2025). Galileo: Learning Global & Local Features of Many Remote Sensing Modalities. *ICML*. https://arxiv.org/abs/2502.09356
- Brown, C. F. et al. (2025). AlphaEarth Foundations. https://arxiv.org/abs/2507.22291
- RoMA: Scaling up Mamba-based Foundation Models for Remote Sensing (2025). https://arxiv.org/abs/2503.10392
- Marsocci, V. et al. (2024). PANGAEA benchmark. https://arxiv.org/abs/2412.04204
- SegEarth-OV (2024). https://arxiv.org/abs/2410.01768 · SegEarth-OV3 (2025). https://arxiv.org/abs/2512.08730

**Land-cover products**
- Esri / Impact Observatory 10 m Annual LULC. https://registry.opendata.aws/io-lulc/ · https://gee-community-catalog.org/projects/S2TSLULC/
- Venter, Z. S. et al. (2022). Global 10 m LULC datasets comparison. *Remote Sensing* 14(16), 4101. https://www.mdpi.com/2072-4292/14/16/4101

**Data sources**
- NRSC data policy and holdings (Bhoonidhi). https://www.nrsc.gov.in/sites/default/files/pdf/ebooks/UIM-2024/3_UIM_2024.pdf · Resourcesat-2 handbook: https://lps16.esa.int/posterfiles/paper1213/[RD13]_Resourcesat-2_Handbook.pdf
- Planet NICFI programme end. https://community.planet.com/tropical-basemaps-85/nicfi-satellite-data-program-prolongs-public-access-to-high-resolution-rainforest-satellite-images-6020
- Sentinel-2 Next Generation phase-0 study. https://rosa.ro/sentinel-2-next-generation-phase-0-scientific-support-study-expro-3/
