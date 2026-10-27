"""Reproducible real-data checks, explicitly separating references and accuracy."""
from _bootstrap import ROOT  # also puts src/ on sys.path
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image
import rasterio
from satsr.geo_io import load_any_user_image, save_text
from satsr.models.unet_segmentation import LandCoverSegmenter
from satsr.models.real_landcover import MAPPING, SOURCE_CLASSES
from satsr.models.rcan import L1BSRSR
from satsr.models.evaluation_metrics import segmentation_metrics
from satsr.runtime_support import configure_cpu_threads



def binary_metrics(pred, target, valid):
    tp=int((pred & target & valid).sum())
    fp=int((pred & ~target & valid).sum())
    fn=int((~pred & target & valid).sum())
    return dict(iou=tp/(tp+fp+fn) if tp+fp+fn else None,
                precision=tp/(tp+fp) if tp+fp else None,
                recall=tp/(tp+fn) if tp+fn else None,
                reference_pixels=int((target & valid).sum()))


def evaluate(output):
    configure_cpu_threads()
    new=LandCoverSegmenter('cpu',real_weights_path=ROOT/'checkpoints/demo/landcover_real.pt')
    old=LandCoverSegmenter('cpu',ROOT/'checkpoints/demo/unet.pt')
    sr=L1BSRSR(ROOT/'checkpoints/demo/rcan.pt','cpu')
    report=dict(real_model=new.provenance,device='cpu',
        sentinel_reference_note='SCL is an automatic 20 m quality classifier resampled to 10 m, NOT human ground truth. Only water/vegetation consistency is compared. No road/building accuracy can be inferred.',
        annotated_reference_note='OpenEarthMap upstream demo examples with real human annotations, potentially included in upstream train/validation. Sanity checks only, NOT an independent test set.',
        independent_real_satellite_accuracy=None,sentinel_checks=[],annotated_examples=[])
    records=json.loads((ROOT/'demo_images/manifest.json').read_text(encoding='utf-8'))
    for record in records:
        bands,image,rgb,_=load_any_user_image(ROOT/'demo_images'/record['file'])
        valid=bands['B02'].valid_mask
        with rasterio.open(ROOT/'demo_images'/record['quality_file']) as src:
            scl=src.read(1)
        # Reproduce the old complete SR->synthetic U-Net pipeline on the same scenes.
        baseline=old.segment(sr.super_resolve(image),apply_morphology=False,return_probabilities=False)['class_map'][::2,::2]
        start=time.perf_counter()
        result=new.segment(image,rgb=rgb,measured_nir=True,valid_mask=valid,
                           pixel_size_m=None,return_probabilities=False)
        labels=result['class_map']
        selected=valid & np.isin(scl,[4,5,6])
        entry=dict(scene=record['file'],category=record['category'],new_detection_seconds=round(time.perf_counter()-start,3),
            before_percentages={str(k):round(float((baseline[valid]==k).mean()*100),2) for k in range(5)},
            after_percentages={name:v['percentage'] for name,v in result['statistics'].items()},
            spectral_correction_pixels=result['spectral_correction_pixels'])
        for name,cid,reference in [('water',3,6),('vegetation',2,4)]:
            entry[name]=dict(before=binary_metrics(baseline==cid,scl==reference,selected),
                            after=binary_metrics(labels==cid,scl==reference,selected))
        report['sentinel_checks'].append(entry)
        print(record['category'],entry['water'],flush=True)
    fixture=ROOT/'tests/fixtures/real_landcover'
    manifest=json.loads((fixture/'manifest.json').read_text())
    for file in manifest['files']:
        if hashlib.sha256((fixture/file['file']).read_bytes()).hexdigest() != file['sha256']:
            raise ValueError('Annotated example checksum mismatch')
    combined=np.zeros((6,6),np.int64)
    for image_path in sorted((fixture/'images').glob('*.tif')):
        rgb=np.asarray(Image.open(image_path).convert('RGB'))
        # RGB network only: no estimated NIR is ever used for reference evaluation.
        labels=new.real_engine.predict_rgb(rgb).argmax(0).astype(np.uint8)
        source=np.asarray(Image.open(fixture/'labels'/image_path.name))
        if not np.isin(source,range(9)).all():
            raise ValueError('Unexpected OpenEarthMap reference label')
        truth=np.full(source.shape,255,np.uint8)
        for original,target in enumerate(MAPPING,1):
            truth[source==original]=target
        metrics=segmentation_metrics(labels,truth,num_classes=6)
        combined += np.asarray(metrics['confusion_matrix'])
        report['annotated_examples'].append(dict(image=image_path.name,metrics=metrics,
            class_names=['Buildings','Roads','Vegetation','Water','Bare Land','Developed space']))
        print(image_path.name,metrics['pixel_accuracy'],flush=True)
    report['annotated_combined_sanity_check']=segmentation_metrics(confusion=combined,num_classes=6)
    save_text(output,json.dumps(report,indent=2,allow_nan=False))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default=str(ROOT/'results/real_accuracy_report.json'))
    evaluate(parser.parse_args().output)
