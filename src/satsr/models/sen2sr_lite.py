"""ESA OpenSR SEN2SRLite (RGBN, 4x) as a comparison model for RCAN.

SEN2SRLite is a small SPAN network trained by ESA OpenSR on SEN2NAIPv2: real
Sentinel-2 10 m images paired with real 2.5 m NAIP aerial images. It is used here
only as an external reference model. It is not part of the desktop app's
default pipeline.

Install the optional dependencies with:  pip install -r requirements/research.txt
Download the weights with:               python download_sen2sr.py
"""
from pathlib import Path

import numpy as np
import torch

from .common import NORMALIZATION, validate_image
from ..paths import PROJECT_ROOT

MODEL_URL = "https://huggingface.co/tacofoundation/sen2sr/resolve/main/SEN2SRLite/NonReference_RGBN_x4/mlm.json"
DEFAULT_DIR = PROJECT_ROOT/"checkpoints"/"external"/"SEN2SRLite_RGBN"
# The project stores B02,B03,B04,B08. SEN2SR expects B04,B03,B02,B08.
TO_SEN2SR = [2, 1, 0, 3]
FROM_SEN2SR = [2, 1, 0, 3]


def block_mean(image, factor):
    """Area-average an H x W x C array by an integer factor (2.5 m -> 5 m, 5 m -> 10 m)."""
    image = np.asarray(image)
    h, w = image.shape[0]//factor*factor, image.shape[1]//factor*factor
    if h == 0 or w == 0:
        raise ValueError("Image is smaller than the averaging factor")
    blocks = image[:h, :w].astype(np.float64).reshape(h//factor, factor, w//factor, factor, *image.shape[2:])
    return blocks.mean(axis=(1, 3))


class SEN2SRLite:
    """Pretrained SEN2SRLite RGBN x4 with the same uint16 interface as L1BSRSR."""
    provenance = ("ESA OpenSR SEN2SRLite NonReference_RGBN_x4 (SPAN, 472k parameters), "
                  "trained on real Sentinel-2 / NAIP pairs (SEN2NAIPv2); MIT licence")

    def __init__(self, model_dir=DEFAULT_DIR, device="cpu", tile_size=128):
        try:
            import safetensors.torch
            from sen2sr.models.opensr_baseline.cnn import CNNSR
            from sen2sr.models.tricks import HardConstraint
            from sen2sr.nonreference import srmodel
        except ImportError as error:
            raise ImportError("SEN2SR needs the optional packages in requirements/research.txt") from error
        model_dir = Path(model_dir)
        weights, constraint = model_dir/"model.safetensor", model_dir/"hard_constraint.safetensor"
        if not weights.is_file() or not constraint.is_file():
            raise FileNotFoundError(f"SEN2SR weights not found in {model_dir}; run download_sen2sr.py")
        self.device = torch.device(device)
        self.tile_size = tile_size
        # Same construction as the model card's load.py, without executing downloaded code.
        network = CNNSR(4, 4, 24, 4, True, False, 6)
        network.load_state_dict(safetensors.torch.load_file(weights), strict=True)
        network.eval()
        hard = HardConstraint(low_pass_mask=safetensors.torch.load_file(constraint)["weights"].to(self.device),
                              device=self.device)
        self.model = srmodel(network, hard, self.device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad = False

    @torch.inference_mode()
    def super_resolve_x4(self, img_bgrn_u16):
        """10 m B02,B03,B04,B08 uint16 -> 2.5 m grid, same band order, uint16."""
        image = validate_image(img_bgrn_u16)
        chw = np.moveaxis(image[..., TO_SEN2SR].astype(np.float32)/NORMALIZATION, -1, 0)
        out = self._tiled(chw)
        out = np.moveaxis(out, 0, -1)[..., FROM_SEN2SR]*NORMALIZATION
        return np.clip(out, 0, 65535).round().astype(np.uint16)

    def _tiled(self, chw, overlap=32):
        """The model's hard constraint only accepts exactly tile x tile inputs, so
        use fixed-size overlapping windows and blend them with a smooth weight."""
        t, scale = self.tile_size, 4
        c, h, w = chw.shape
        ph, pw = max(0, t-h), max(0, t-w)
        if ph or pw:
            chw = np.pad(chw, ((0, 0), (0, ph), (0, pw)), mode="reflect" if min(h, w) > 1 else "edge")
        H, W = chw.shape[1:]
        step = t-overlap

        def starts(n):
            positions = list(range(0, max(1, n-t+1), step))
            if positions[-1]+t < n:
                positions.append(n-t)
            return positions

        ramp = np.minimum(np.arange(t*scale)+.5, t*scale-np.arange(t*scale)-.5)
        ramp = np.clip(ramp/(overlap*scale/2), 1e-3, 1.)
        weight = np.outer(ramp, ramp).astype(np.float32)
        total = np.zeros((c, H*scale, W*scale), np.float32)
        norm = np.zeros((H*scale, W*scale), np.float32)
        for y in starts(H):
            for x in starts(W):
                tile = torch.from_numpy(np.ascontiguousarray(chw[:, y:y+t, x:x+t]))[None].to(self.device)
                out = self.model(tile)[0].float().cpu().numpy()
                if out.shape != (c, t*scale, t*scale) or not np.isfinite(out).all():
                    raise ValueError("SEN2SR returned non-finite pixels or an incorrect shape")
                total[:, y*scale:(y+t)*scale, x*scale:(x+t)*scale] += out*weight
                norm[y*scale:(y+t)*scale, x*scale:(x+t)*scale] += weight
        return (total/norm)[:, :h*scale, :w*scale]

    def super_resolve(self, img_bgrn_u16):
        """2x output (5 m grid) to compare with RCAN: 4x prediction area-averaged by 2."""
        x4 = self.super_resolve_x4(img_bgrn_u16)
        return np.clip(block_mean(x4, 2), 0, 65535).round().astype(np.uint16)
