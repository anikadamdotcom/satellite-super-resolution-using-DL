"""Rebuild the bundled Hugging Face model; optional developer dependency:
pip install transformers==5.5.0 safetensors
Normal application use needs neither package nor an internet connection.
"""
from _bootstrap import ROOT  # also puts src/ on sys.path
import hashlib
import json
from pathlib import Path
import tempfile
import urllib.request
import warnings
import torch

REPO = 'boehnen/satlens-segformer'
REVISION = '19cdd879e535d76e7fcbbd1e8b54dac9c1da90d6'
SOURCE_SHA = '68f1c70396d3e1b5a9b752408cadbafa90c655da816564ce7d9646b112392ac0'


class LogitsOnly(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.net = model

    def forward(self, image):
        return self.net(pixel_values=image, return_dict=False)[0]


def rebuild(destination):
    from transformers import SegformerForSemanticSegmentation
    from satsr.models.hf_landcover import CLASSES
    from satsr.geo_io import _atomic_write, save_text
    from satsr.runtime_support import configure_cpu_threads
    configure_cpu_threads()
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='l1bsr-hf-') as tmp:
        source = Path(tmp)
        for name in ('config.json', 'preprocessor_config.json', 'README.md', 'model.safetensors'):
            with urllib.request.urlopen(f'https://huggingface.co/{REPO}/resolve/{REVISION}/{name}', timeout=120) as response:
                (source/name).write_bytes(response.read())
        if hashlib.sha256((source/'model.safetensors').read_bytes()).hexdigest() != SOURCE_SHA:
            raise ValueError('Hugging Face weights checksum mismatch')
        model = LogitsOnly(SegformerForSemanticSegmentation.from_pretrained(source, local_files_only=True).eval())
        torch.manual_seed(123)
        with torch.inference_mode(), warnings.catch_warnings():
            warnings.simplefilter('ignore')
            # Keep parameters as module state. Frozen tensor constants can remain
            # in JIT execution-plan caches after a CUDA->CPU device switch.
            traced = torch.jit.trace(model, torch.zeros(1, 3, 512, 512), check_trace=True).eval()
            sample = torch.randn(1, 3, 512, 512)
            error = float((model(sample)-traced(sample)).abs().max())
        if error > 1e-4:
            raise ValueError(f'Export parity failed: {error}')
        _atomic_write(destination/'model.ts', lambda temp: traced.save(str(temp)))
        for name in ('config.json', 'preprocessor_config.json', 'README.md'):
            save_text(destination/name, (source/name).read_text())
        meta = dict(repository=REPO, revision=REVISION, license='MIT', source_sha256=SOURCE_SHA,
            runtime_sha256=hashlib.sha256((destination/'model.ts').read_bytes()).hexdigest(),
            format_version=1, architecture='SegFormer-B0', input_size=512, class_names=CLASSES,
            trace_parity_max_abs=error,
            source_training_code='https://github.com/boehnen/satlens/blob/b96058915b2e151012b3fbc7ca0bb63a464722d7/notebooks/train.ipynb')
        save_text(destination/'source.json', json.dumps(meta, indent=2))
    print(f'Exported verified offline model to {destination}')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default=str(ROOT/'checkpoints/demo/hf_segformer'))
    rebuild(parser.parse_args().output)
