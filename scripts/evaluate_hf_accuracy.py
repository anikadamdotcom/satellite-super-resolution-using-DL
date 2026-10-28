"""Evaluate fixed real-image audit samples; never report confidence as accuracy."""
from _bootstrap import ROOT  # also puts src/ on sys.path
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image
from satsr.models.real_landcover import RealLandCoverEngine, MAPPING
from satsr.models.hf_landcover import HuggingFaceLandCoverEngine, EnsembleLandCoverEngine
from satsr.runtime_support import configure_cpu_threads

NAMES = ['Buildings', 'Roads', 'Vegetation', 'Water', 'Bare Land', 'Developed space']


def metrics(matrix):
    # Final prediction column is abstention. It remains in the denominator.
    matrix = np.asarray(matrix, np.int64)
    correct = np.diag(matrix[:, :6]).astype(float)
    actual = matrix.sum(1)
    predicted = matrix[:, :6].sum(0)
    union = actual+predicted-correct
    iou = np.divide(correct, union, out=np.full(6, np.nan), where=union > 0)
    return dict(pixel_accuracy=float(correct.sum()/matrix.sum()) if matrix.sum() else None,
        mean_iou=float(np.nanmean(iou)) if np.isfinite(iou).any() else None,
        per_class={name:dict(iou=float(iou[i]) if np.isfinite(iou[i]) else None,
            recall=float(correct[i]/actual[i]) if actual[i] else None,
            reference_pixels=int(actual[i])) for i, name in enumerate(NAMES)},
        abstained_pixels=int(matrix[:, 6].sum()), evaluated_pixels=int(matrix.sum()),
        confusion_matrix=matrix.tolist())


def evaluate(output):
    configure_cpu_threads()
    folder = ROOT/'checkpoints/demo'
    models = dict(sparsemask=RealLandCoverEngine(folder/'landcover_real.pt'),
        huggingface=HuggingFaceLandCoverEngine(folder/'hf_segformer'),
        combined=EnsembleLandCoverEngine(folder/'landcover_real.pt', folder/'hf_segformer'))
    report = dict(target_pixel_accuracy=.85, target_demonstrated=False,
        universal_accuracy_guarantee=False, fine_tuning_performed=False,
        oem_note='Two upstream examples may overlap training. Used to compare candidates, NOT an independent test.',
        loveda_note='Eight fixed LoveDA validation images, separate from documented OpenEarthMap training; small cross-dataset audit, not the full benchmark. No parameters tuned using these results. Geographic overlap not independently excluded.',
        mapping_note='OEM tree/rangeland/agriculture and LoveDA forest/agriculture map to Vegetation. LoveDA 0=no-data and 1=background are excluded because background has no equivalent class. Exclusion counts are reported. Abstentions count as errors.',
        device='cpu', source=json.loads((folder/'hf_segformer/source.json').read_text()), datasets={})
    for dataset in ['real_landcover', 'loveda']:
        root = ROOT/'tests/fixtures'/dataset
        manifest = json.loads((root/'manifest.json').read_text())
        for item in manifest['files']:
            if hashlib.sha256((root/item['file']).read_bytes()).hexdigest() != item['sha256']:
                raise ValueError(f'Fixture checksum mismatch: {item["file"]}')
        totals = {key:np.zeros((6, 7), np.int64) for key in models}
        results = []
        for path in sorted((root/'images').iterdir()):
            rgb = np.asarray(Image.open(path).convert('RGB'))
            original = np.asarray(Image.open(root/'labels'/path.name))
            mapping = (255, *MAPPING) if dataset == 'real_landcover' else (255, 255, 0, 1, 3, 4, 2, 2)
            truth = np.asarray(mapping, np.uint8)[original]
            valid = truth != 255
            entry = dict(image=path.name, total_pixels=int(truth.size), excluded_pixels=int((~valid).sum()), models={})
            for key, model in models.items():
                start = time.perf_counter()
                probabilities = model.predict_rgb(rgb)
                pred = probabilities.argmax(0)
                pred[probabilities.sum(0) == 0] = 6
                matrix = np.bincount(truth[valid].astype(int)*7+pred[valid], minlength=42).reshape(6, 7)
                totals[key] += matrix
                entry['models'][key] = dict(**metrics(matrix), seconds=round(time.perf_counter()-start, 3))
            results.append(entry)
            print(dataset, path.name, {k:round(v['pixel_accuracy'], 4) for k, v in entry['models'].items()}, flush=True)
        report['datasets'][dataset] = dict(examples=results, combined={k:metrics(v) for k, v in totals.items()})
    # This flag means the specific cross-dataset audit threshold, never all images.
    report['target_demonstrated'] = report['datasets']['loveda']['combined']['combined']['pixel_accuracy'] >= .85
    from satsr.geo_io import save_text
    save_text(output, json.dumps(report, indent=2, allow_nan=False))
    return report


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default=str(ROOT/'results/hf_accuracy_report.json'))
    evaluate(parser.parse_args().output)
