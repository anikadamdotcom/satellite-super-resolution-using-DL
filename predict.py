"""Run the complete mapping pipeline without launching the GUI."""
import argparse
import json
import os
import tempfile
from pathlib import Path
import numpy as np
from PIL import Image
from rasterio.transform import from_origin
from rasterio.crs import CRS
from data import synthetic_scene
from geo_io import (BandData, load_any_user_image, load_band_stack, render_rgb, stretch_limits,
    pixel_area_m2, output_valid_mask, write_geotiff, write_single_band_geotiff, save_image, save_text,
    derived_note)
from models.rcan import L1BSRSR
from models.unet_segmentation import LandCoverSegmenter
from models.evaluation_metrics import compute_full_sr_evaluation, segmentation_metrics

ROOT = Path(__file__).resolve().parent
OUTPUT_NAMES = {"input.png","enhanced.png","land_cover.png","overlay.png","confidence.png",
                "class_ids.npy","confidence.npy","enhanced.tif","land_cover.tif","report.json","ground_truth.png"}


def publish_outputs(stage, output):
    """Replace this pipeline's files with rollback; preserve unrelated files."""
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".l1bsr-backup-",dir=output.parent) as name:
        backup = Path(name).resolve()
        if backup.parent != output.parent.resolve():
            raise ValueError("Invalid output backup path")
        moved, created = [], []
        try:
            for filename in sorted(OUTPUT_NAMES):
                target = output/filename
                if target.exists():
                    if not target.is_file():
                        raise ValueError(f"Output path is a directory: {target}")
                    os.replace(target,backup/filename); moved.append(filename)
            for path in stage.iterdir():
                if path.name not in OUTPUT_NAMES:
                    raise ValueError("Unknown pipeline artifact")
                os.replace(path,output/path.name); created.append(path.name)
        except Exception:
            for filename in created:
                (output/filename).unlink(missing_ok=True)
            for filename in moved:
                os.replace(backup/filename,output/filename)
            raise
    # Remove old external GDAL masks only for the TIFFs this pipeline owns.
    for filename in ("enhanced.tif.msk","land_cover.tif.msk"):
        (output/filename).unlink(missing_ok=True)


def run_prediction(args):
    output = Path(args.output).resolve()
    if output.exists() and not output.is_dir():
        raise ValueError("Output must be a directory")
    inputs = ([args.image] if args.image else args.bands or [])
    if any(Path(p).resolve() in {output/name for name in OUTPUT_NAMES} for p in inputs):
        raise ValueError("Output paths would overwrite an input image; choose another directory")
    truth, mask = None, None
    if args.demo:
        image, truth, mask = synthetic_scene(200100,128)
        ref = BandData("synthetic demo",image[...,0],from_origin(635000,5785000,10,10),
                      CRS.from_epsg(32634),image.shape[1],image.shape[0],
                      source_kind="synthetic illustration with illustrative georeference",
                      display_limits=stretch_limits(image[..., [2,1,0]]))
        input_rgb = render_rgb(image,ref)
    elif args.bands:
        ref, image, _ = load_band_stack(dict(zip(("B02","B03","B04","B08"),args.bands)))
        input_rgb = render_rgb(image,ref)
    else:
        bands, image, input_rgb, _ = load_any_user_image(args.image); ref = bands["B02"]
    sr = L1BSRSR(Path(args.models)/"rcan.pt",args.device)
    detector = getattr(args,"detector","auto")
    if detector == "auto":
        detector = "synthetic" if args.demo else "ensemble"
    real = detector != "synthetic"
    seg = (LandCoverSegmenter(args.device,
        real_weights_path=Path(args.models)/"landcover_real.pt" if detector in ("real", "ensemble") else None,
        hf_weights_path=Path(args.models)/"hf_segformer" if detector in ("hf", "ensemble") else None)
        if real else LandCoverSegmenter(args.device,Path(args.models)/"unet.pt"))
    enhanced = sr.super_resolve(image)
    valid = output_valid_mask(ref,enhanced.shape); enhanced[~valid] = 0
    if real:
        result = seg.segment(image,rgb=input_rgb,measured_nir=not args.demo and "estimated NIR" not in ref.source_kind,
            output_shape=tuple(enhanced.shape[:2]),pixel_size_m=None,
            pixel_area_m2=pixel_area_m2(ref,enhanced.shape),valid_mask=output_valid_mask(ref,image.shape),return_probabilities=False)
    else:
        result = seg.segment(enhanced,apply_morphology=False,pixel_size_m=None,
            pixel_area_m2=pixel_area_m2(ref,enhanced.shape),valid_mask=valid,return_probabilities=False)
    rgb = render_rgb(enhanced,ref)
    bicubic = sr.bicubic_upscale(image); bicubic[~valid] = 0
    metrics = compute_full_sr_evaluation(image,bicubic,enhanced,truth,valid_mask=valid)
    report = dict(input=ref.path,source_kind=ref.source_kind,sr_model_source=sr.provenance,
                  source_metadata=ref.source_metadata or {},
                  segmentation_model_source=seg.provenance,metrics=metrics,statistics=result["statistics"],
                  confidence_note=result["confidence_note"],device=str(sr.device))
    report["detection_details"] = {key:result[key] for key in ("method", "prediction_source", "prediction_input_shape",
        "measured_nir_used", "spectral_correction_pixels", "abstained_native_pixels", "resolution_note") if key in result}
    if mask is not None and not real:
        report["segmentation"] = segmentation_metrics(result["class_map"],mask)
    output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".l1bsr-predict-",dir=output.parent) as name:
        stage = Path(name).resolve()
        if stage.parent != output.parent:
            raise ValueError("Invalid inference workspace")
        save_image(stage/"input.png",input_rgb,ref.valid_mask)
        note = derived_note("RCAN ("+sr.provenance+")" if sr.weights_loaded else "bicubic baseline")
        report["enhanced_imagery"] = note
        save_image(stage/"enhanced.png",rgb,valid,description=note)
        save_image(stage/"land_cover.png",result["color_map"],valid)
        overlay = (rgb*.45+result["color_map"]*.55).astype(np.uint8); overlay[~valid] = 0
        save_image(stage/"overlay.png",overlay,valid,description=note)
        score = np.nan_to_num(result["confidence_map"],nan=0.)
        Image.fromarray((score*255).astype(np.uint8)).save(stage/"confidence.png")
        np.save(stage/"class_ids.npy",result["class_map"]); np.save(stage/"confidence.npy",result["confidence_map"])
        if ref.crs:
            write_geotiff(stage/"enhanced.tif",enhanced,ref,sr_model="RCAN ("+sr.provenance+")" if sr.weights_loaded else "bicubic baseline")
            write_single_band_geotiff(stage/"land_cover.tif",result["class_map"],ref)
        if truth is not None:
            save_image(stage/"ground_truth.png",render_rgb(truth,ref))
        report["files"] = sorted([p.name for p in stage.iterdir()]+["report.json"])
        save_text(stage/"report.json",json.dumps(report,indent=2,allow_nan=False))
        publish_outputs(stage,output)
    print(f"Saved results to {output}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--demo",action="store_true"); inputs.add_argument("--image")
    inputs.add_argument("--bands",nargs=4,metavar=("B02","B03","B04","B08"))
    parser.add_argument("--models",default=str(ROOT/"checkpoints"/"demo"))
    parser.add_argument("--detector",default="auto",choices=("auto","ensemble","hf","real","synthetic"),
                        help="Auto: combined real models for images; synthetic U-Net for --demo. hf: fast Hugging Face model; real: previous SparseMask")
    parser.add_argument("--output",default="outputs/demo"); parser.add_argument("--device",default="cpu",choices=("cpu","cuda"))
    args = parser.parse_args()
    try:
        from runtime_support import configure_cpu_threads
        configure_cpu_threads()
        run_prediction(args)
    except (ValueError,OSError,RuntimeError) as error:
        parser.exit(1,f"Prediction failed: {error}\n")


if __name__ == "__main__":
    main()
