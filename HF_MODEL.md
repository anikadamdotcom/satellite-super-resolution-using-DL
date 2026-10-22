# Hugging Face integration and measured accuracy

The default **Combined real models** detector averages the previous OpenEarthMap
SparseMask model with **boehnen/satlens-segformer**, a SegFormer-B0 fine-tuned on
real OpenEarthMap imagery. SegFormer predictions average the original image and
three reflected views. No model was fine-tuned on the audit samples.

**The requested 85% general accuracy has not been demonstrated.** Accepting a
file does not establish accurate interpretation of its contents. Use top-down
satellite/aerial imagery; ordinary photographs, clouds, different sensors and
coarse pixels can produce incorrect predictions. Vegetation combines trees,
rangeland and agriculture; the system does not count individual objects.

## Reproducible audit

Run `python evaluate_hf_accuracy.py`. Full per-image results, class IoUs, recall,
confusion matrices, exclusions and CPU timings are in `hf_accuracy_report.json`.

| Dataset and scope | Previous SparseMask | HF SegFormer alone | Combined default |
| --- | ---: | ---: | ---: |
| Two OEM examples, possible training overlap | 80.91% | 80.07% | 84.11% |
| Eight fixed LoveDA validation images | 77.17% | 78.54% | 79.80% |

These are **pixel accuracy**, not object detection success rates. The two OEM
examples were used to compare candidate approaches, so they are not an
independent test. The combined result on Palu alone is 85.68%; selecting that
number would not establish the requested accuracy on other images.

The LoveDA sample was selected before inference by evenly spacing eight numeric
IDs in a pinned validation-file listing. It is a separate dataset from the
models' documented OpenEarthMap training; geographic overlap has not been
independently excluded. These are a small cross-dataset audit, not an official
full LoveDA benchmark. None of its labels were used for training or tuning.

LoveDA forest and agriculture map to Vegetation. Its background class has no
equivalent in the app and is excluded along with no-data: **2,196,159 of
8,388,608 pixels are excluded**. Accuracy is measured on the remaining
6,192,449 pixels. Abstentions on labeled pixels count as errors. This mapping
and exclusion mean these scores must not be compared directly with published
seven-class LoveDA leaderboard scores.

| Class | Combined LoveDA IoU | Combined recall |
| --- | ---: | ---: |
| Buildings | 55.26% | 65.91% |
| Roads | 38.61% | 39.52% |
| Vegetation | 79.25% | 96.33% |
| Water | 62.78% | 65.15% |
| Bare land | 0.21% | 0.21% |

The standalone HF option scores better for buildings and roads on this sample;
the combined model scores better overall, especially water. Bare-ground
performance remains poor. High vegetation prevalence makes overall pixel
accuracy look better than several individual classes. Neither the model nor
its scores justify a universal 85% claim.

## Run and model choices

Leave **Run without GPU (CPU)** checked. The default combined option works
offline on CPU. **Hugging Face SegFormer (fast)** uses one model and one view.
**SparseMask (previous model)** preserves the previous detector, and **U-Net
experiment / custom model** preserves the synthetic research workflow.

The sample list now includes **Palu: fine-resolution roads and buildings** in
addition to the five Sentinel-2 scenes. Run all, switch views, and use the
Downloads panel or All results ZIP. Select another sample without restarting.

```powershell
python predict.py --image demo_images/high_resolution/palu_8.tif --detector ensemble --output outputs/combined --device cpu
python predict.py --image demo_images/high_resolution/palu_8.tif --detector hf --output outputs/huggingface --device cpu
```

Both real models classify the original decoded image; RCAN enhancement is
independent. Physical spectral checks require measured NIR. Sentinel-2's 10 m
pixels cannot resolve many individual buildings or narrow roads. RCAN is now
trained on real Sentinel-2/VENµS pairs (small measured gain over bicubic, see
README) but still does not establish new observed ground detail.

## Source, conversion and preprocessing

- Model: https://huggingface.co/boehnen/satlens-segformer
- Pinned revision: `19cdd879e535d76e7fcbbd1e8b54dac9c1da90d6`.
- Model card license: MIT. Original card, class configuration, preprocessing
  configuration and checksums are bundled in `checkpoints/demo/hf_segformer/`.
- Upstream training code:
  https://github.com/boehnen/satlens/blob/b96058915b2e151012b3fbc7ca0bb63a464722d7/notebooks/train.ipynb
- Source safetensors SHA256:
  `68f1c70396d3e1b5a9b752408cadbafa90c655da816564ce7d9646b112392ac0`.
- 3.72 million parameters; an approximately 15 MB TorchScript graph
  runs through the existing PyTorch dependency. No Hugging Face account,
  Transformers installation, network connection or GPU is needed at runtime.
- Export compared with the original Transformers model: maximum absolute
  logit difference below `0.000002` in the recorded random-input check.
- RGB, PIL bilinear resize to 512 square, rescale by 1/255, ImageNet mean
  `[0.485,0.456,0.406]` and standard deviation `[0.229,0.224,0.225]`.
  Nonsquare inputs are edge-padded first, preserving aspect ratio. Inputs are
  bounded at 1024 pixels on their longest side by the image loader.
- Nine source classes are mapped explicitly to the app's six land-cover
  categories. Background-dominant HF predictions abstain with ID 255; they
  are not silently called buildings or bare ground. Reports count abstained
  native pixels. Scores remain uncalibrated, not accuracy estimates.
- `download_hf_model.py` downloads pinned safetensors, verifies its hash,
  recreates and checks the export. Rebuilding only requires the optional
  developer dependencies `transformers==5.5.0` and `safetensors`.

## Audit data attribution

LoveDA: Junjue Wang, Zhuo Zheng, Ailong Ma, Xiaoyan Lu and Yanfei Zhong,
RSIDEA, Wuhan University, NeurIPS Datasets and Benchmarks 2021.
https://github.com/Junjue-Wang/LoveDA

The original license permits academic purposes only and prohibits commercial
use; underlying Google Earth terms apply. Image/mask URLs, hashes and fixed
selection are in `tests/fixtures/loveda/manifest.json`. Re-download with
`python download_accuracy_samples.py`. The original PNGs have no embedded CRS;
no geospatial metadata was fabricated. They are test fixtures, not replacements
for the georeferenced Sentinel-2 and Maxar demo imagery.
