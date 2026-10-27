"""Pinned Hugging Face SegFormer-B0, exported locally for offline CPU inference.

The TorchScript graph has a fixed 512-square input. Runtime preprocessing uses
the publisher's RGB / PIL bilinear / ImageNet contract. No remote code is run.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from .real_landcover import RealLandCoverEngine, MAPPING

REPOSITORY = 'boehnen/satlens-segformer'
REVISION = '19cdd879e535d76e7fcbbd1e8b54dac9c1da90d6'
CLASSES = ['Background', 'Bareland', 'Rangeland', 'Developed space', 'Road',
           'Tree', 'Water', 'Agriculture land', 'Building']


class HuggingFaceLandCoverEngine(RealLandCoverEngine):
    def __init__(self, folder, device='cpu', augment=False):
        folder = Path(folder)
        meta = json.loads((folder/'source.json').read_text())
        path = folder/'model.ts'
        if (meta.get('repository') != REPOSITORY or meta.get('revision') != REVISION or
                meta.get('input_size') != 512 or meta.get('class_names') != CLASSES or
                meta.get('format_version') != 1 or not path.is_file() or
                path.stat().st_size > 32*1024**2 or
                hashlib.sha256(path.read_bytes()).hexdigest() != meta.get('runtime_sha256')):
            raise ValueError('Incompatible or corrupted Hugging Face model')
        self.device = torch.device(device)
        self.model = torch.jit.load(str(path), map_location=self.device).eval()
        self.augment = bool(augment)
        self.method_name = 'Hugging Face OpenEarthMap SegFormer-B0'
        self.provenance = f'Hugging Face {REPOSITORY}@{REVISION}; real OpenEarthMap imagery; MIT'

    @torch.inference_mode()
    def predict_rgb(self, rgb, return_nonwater=False):
        rgb = np.asarray(rgb)
        if (rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8 or
                min(rgb.shape[:2]) < 1 or max(rgb.shape[:2]) > 1024):
            raise ValueError('Hugging Face input must be RGB uint8, with dimensions 1..1024')
        h, w = rgb.shape[:2]
        side = max(h, w)
        padded = np.pad(rgb, ((0, side-h), (0, side-w), (0, 0)), mode='edge')
        resized = np.asarray(Image.fromarray(padded).resize((512, 512), Image.Resampling.BILINEAR)).copy()
        tensor = torch.from_numpy(resized).permute(2, 0, 1)[None].to(self.device).float()/255
        mean = tensor.new_tensor([.485, .456, .406])[None, :, None, None]
        std = tensor.new_tensor([.229, .224, .225])[None, :, None, None]
        tensor = (tensor-mean)/std
        sources, dry_sources = [], []
        for axes in ([(), (2,), (3,), (2, 3)] if self.augment else [()]):
            logits = self.model(tensor.flip(axes) if axes else tensor)
            if tuple(logits.shape) != (1, 9, 128, 128) or not torch.isfinite(logits).all():
                raise ValueError('Hugging Face model returned invalid logits')
            if axes:
                logits = logits.flip(axes)
            logits = F.interpolate(logits, size=(side, side), mode='bilinear', align_corners=False)
            sources.append(logits.softmax(1)[0, :, :h, :w].cpu().numpy())
            if return_nonwater:
                logits[:, 6] = -torch.inf
                dry_sources.append(logits.softmax(1)[0, :, :h, :w].cpu().numpy())
        def remap(values):
            source = np.mean(values, axis=0)
            out = np.zeros((6, h, w), np.float32)
            for original, target in enumerate(MAPPING, 1):
                out[target] += source[original]
            # Background is an abstention, never silently renamed bare land.
            out[:, source[0] >= out.max(0)] = 0
            return out
        probs = remap(sources)
        return (probs, remap(dry_sources)) if return_nonwater else probs


class EnsembleLandCoverEngine(RealLandCoverEngine):
    def __init__(self, sparse_path, hf_folder, device='cpu'):
        self.sparse = RealLandCoverEngine(sparse_path, device)
        self.hf = HuggingFaceLandCoverEngine(hf_folder, device, augment=True)
        self.device = torch.device(device)
        self.model = torch.nn.ModuleList([self.sparse.model, self.hf.model]).eval()
        self.method_name = 'OpenEarthMap combined SparseMask + Hugging Face SegFormer-B0 (4 flips)'
        self.provenance = self.sparse.provenance + '; combined equally with ' + self.hf.provenance

    def predict_rgb(self, rgb, return_nonwater=False):
        a = self.sparse.predict_rgb(rgb, return_nonwater)
        b = self.hf.predict_rgb(rgb, return_nonwater)
        if return_nonwater:
            return tuple((x+y)*.5 for x, y in zip(a, b))
        return (a+b)*.5
