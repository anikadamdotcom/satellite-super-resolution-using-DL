"""Real-data OpenEarthMap inference with optional measured-NIR consistency checks.

The upstream network consumes BGR, not RGB, with its published normalization.
Its weights were trained on 0.25--0.5 m imagery. Sentinel-2 is an out-of-domain
input: small roads/buildings are candidates, not resolved object footprints.
"""
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from .common import validate_image
from .oem_sparsemask.sparse_mask_eval_mode import SparseMask

SOURCE_CLASSES = ['bareland', 'rangeland', 'developed space', 'road',
                  'tree', 'water', 'agriculture land', 'buildings']
MAPPING = (4, 2, 5, 1, 2, 3, 2, 0)


class RealLandCoverEngine:
    def __init__(self, path, device='cpu', tile_size=1024):
        path = Path(path)
        if not path.is_file() or path.stat().st_size > 32*1024**2:
            raise ValueError('Real land-cover checkpoint must exist and be at most 32 MB')
        ck = torch.load(path, map_location='cpu', weights_only=True)
        if (not isinstance(ck, dict) or ck.get('format_version') != 1 or
                ck.get('task') != 'real_landcover' or
                ck.get('architecture') != 'OEM-SparseMask-MobileNetV2' or
                ck.get('class_names') != SOURCE_CLASSES or ck.get('channel_order') != 'BGR' or
                ck.get('mean') != [.4325, .4483, .3879] or
                ck.get('std') != [.0195, .0169, .0179] or ck.get('input_size') != 1024):
            raise ValueError('Incompatible real land-cover checkpoint metadata')
        mask = np.asarray(ck.get('mask'))
        if mask.shape != (9, 9, 2) or not np.isin(mask, [0, 1]).all():
            raise ValueError('Invalid SparseMask architecture')
        state = ck.get('state_dict')
        if (not isinstance(state, dict) or not state or
                any(not isinstance(v, torch.Tensor) or not torch.isfinite(v).all() for v in state.values())):
            raise ValueError('Real checkpoint contains missing or non-finite tensors')
        self.model = SparseMask(mask, 'mobilenet_v2', depth=64, in_channels=3, num_classes=8)
        self.model.load_state_dict(state, strict=True)
        self.device = torch.device(device)
        self.model.to(self.device).eval()
        self.provenance = ck['data_source']
        if type(tile_size) is not int or not 1 <= tile_size <= 1024:
            raise ValueError('Real inference tile size must be between 1 and 1024')
        self.tile_size = tile_size

    @torch.inference_mode()
    def predict_rgb(self, rgb, return_nonwater=False):
        """Overlap/blend native-image tiles; preserve aspect ratio and bound memory."""
        rgb = np.asarray(rgb)
        if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8 or min(rgb.shape[:2]) < 1:
            raise ValueError('Real land-cover input must be non-empty RGB uint8')
        h, w = rgb.shape[:2]
        sums = np.zeros((6, h, w), np.float32)
        dry_sums = np.zeros_like(sums) if return_nonwater else None
        weights = np.zeros((h, w), np.float32)
        side = self.tile_size
        step = max(1, side*3//4)
        def positions(length):
            return sorted(set(range(0, max(1, length-side+1), step)) | {max(0, length-side)})
        mean = torch.tensor([.4325,.4483,.3879], device=self.device)[None,:,None,None]
        std = torch.tensor([.0195,.0169,.0179], device=self.device)[None,:,None,None]
        for y in positions(h):
            for x in positions(w):
                patch = rgb[y:y+side, x:x+side]
                ph, pw = patch.shape[:2]
                bgr = torch.from_numpy(patch[..., ::-1].copy()).permute(2,0,1)[None].to(self.device).float()/255
                # Square padding rather than distortion; upsampling is not new resolution.
                padded = F.pad(bgr, (0, max(ph,pw)-pw, 0, max(ph,pw)-ph), mode='replicate')
                inp = F.interpolate(padded, size=(1024,1024), mode='bilinear', align_corners=False)
                logits = self.model((inp-mean)/std)
                if logits.ndim != 4 or logits.shape[1] != 8 or not torch.isfinite(logits).all():
                    raise ValueError('Real model returned invalid logits')
                logits = F.interpolate(logits, size=(max(ph,pw),)*2, mode='bilinear', align_corners=True)
                source = logits.softmax(1)[0, :, :ph, :pw].cpu().numpy()
                if return_nonwater:
                    dry_logits = logits.clone()
                    dry_logits[:,5] = -torch.inf
                    dry_source = dry_logits.softmax(1)[0,:,:ph,:pw].cpu().numpy()
                probs = np.zeros((6, ph, pw), np.float32)
                for src, target in enumerate(MAPPING):
                    probs[target] += source[src]
                # Nonzero tent weights include image boundaries and avoid seam overwrites.
                wy = np.minimum(np.arange(ph)+1, np.arange(ph,0,-1)).astype(np.float32)
                wx = np.minimum(np.arange(pw)+1, np.arange(pw,0,-1)).astype(np.float32)
                blend = wy[:,None]*wx[None,:]
                sums[:,y:y+ph,x:x+pw] += probs*blend
                weights[y:y+ph,x:x+pw] += blend
                if return_nonwater:
                    for src,target in enumerate(MAPPING):
                        dry_sums[target,y:y+ph,x:x+pw] += dry_source[src]*blend
        probs = sums/weights[None]
        return (probs, dry_sums/weights[None]) if return_nonwater else probs

    def segment(self, image, rgb=None, measured_nir=False, valid_mask=None,
                output_shape=None, pixel_size_m=None, pixel_area_m2=None, return_probabilities=True):
        from .unet_segmentation import REAL_CLASS_DEFINITIONS
        image = validate_image(image)
        h, w = image.shape[:2]
        valid = np.ones((h,w), bool) if valid_mask is None else np.asarray(valid_mask, dtype=bool)
        if valid.shape != (h,w):
            raise ValueError('valid_mask shape does not match observed image')
        if rgb is None:
            from ..geo_io import percentile_stretch
            rgb = percentile_stretch(image[..., [2,1,0]], valid_mask=valid)
        if np.asarray(rgb).shape != (h,w,3):
            raise ValueError('RGB display and observed bands must have matching shapes')
        if measured_nir:
            probs, dry_probs = self.predict_rgb(rgb, return_nonwater=True)
        else:
            probs = self.predict_rgb(rgb)
        corrected = np.zeros((h,w), bool)
        if measured_nir:
            # Never apply physical spectral indices to the loader's estimated RGB NIR.
            green, red, nir = (image[...,i].astype(np.float32)/10000 for i in (1,2,3))
            ndvi = (nir-red)/(nir+red+1e-6)
            ndwi = (green-nir)/(green+nir+1e-6)
            # Conservative strong-green-vegetation evidence; mixed pixels stay neural.
            vegetation = valid & (ndvi > .55) & (ndwi < 0)
            water = valid & (ndwi > .15) & (ndvi < .15) & (nir < .12)
            wrong_water = valid & (probs.argmax(0) == 3) & ((nir >= .15) | (ndwi < -.10)) & ~water
            # Re-softmax without water avoids zero-sum underflow on saturated logits.
            probs[:,wrong_water] = dry_probs[:,wrong_water]
            for cid, selected in ((2,vegetation), (3,water)):
                changed = selected & (probs.argmax(0) != cid)
                # Explicit heuristic distribution, NOT a calibrated neural probability.
                probs[:,changed] = .05
                probs[cid,changed] = .75
                corrected |= changed
            corrected |= wrong_water
        abstained = valid & (probs.sum(0) == 0)
        valid = valid & ~abstained
        labels = probs.argmax(0).astype(np.uint8)
        confidence = np.take_along_axis(probs,labels[None].astype(int),axis=0)[0]
        target = (h,w) if output_shape is None else tuple(output_shape[:2])
        if len(target) != 2 or any(type(v) is not int or v < 1 for v in target):
            raise ValueError('Invalid segmentation output shape')
        def resize(arr):
            # Pixel labels and scores remain native predictions, replicated for display.
            tensor=torch.from_numpy(np.ascontiguousarray(arr)).float()
            batch=tensor[None,None] if arr.ndim==2 else tensor[None]
            return F.interpolate(batch,size=target,mode='nearest')[0].numpy()
        if target != (h,w):
            labels=resize(labels)[0].astype(np.uint8)
            confidence=resize(confidence)[0]
            valid=resize(valid)[0].astype(bool)
            if return_probabilities:
                probs=resize(probs)
        labels[~valid]=255
        confidence=confidence.astype(np.float32)
        confidence[~valid]=np.nan
        area = pixel_area_m2 if pixel_area_m2 is not None else (pixel_size_m**2 if pixel_size_m is not None else None)
        if ((pixel_size_m is not None and (not np.isfinite(pixel_size_m) or pixel_size_m <= 0)) or
                (area is not None and (not np.isfinite(area) or area <= 0))):
            raise ValueError('Pixel size and area must be positive and finite')
        color=np.zeros((*target,3), np.uint8)
        stats={}
        total=int(valid.sum())
        for cid,info in REAL_CLASS_DEFINITIONS.items():
            selected=labels==cid
            color[selected]=info['color']
            count=int(selected.sum())
            stats[info['name']]=dict(class_id=cid,color_hex=info['hex'],pixel_count=count,
                percentage=round(100*count/total,2) if total else 0.,
                area_hectares=round(count*area/10000,3) if area is not None else None,
                area_km2=round(count*area/1e6,4) if area is not None else None,description=info['desc'])
        result=dict(class_map=labels,color_map=color,confidence_map=confidence,uncertainty_map=1-confidence,
            statistics=stats,method=getattr(self, 'method_name', 'OpenEarthMap SparseMask') + (' + measured-NIR checks' if measured_nir else ' (RGB only)'),
            provenance=self.provenance,confidence_note='Uncalibrated model scores; NIR corrections are heuristic scores',
            prediction_input_shape=[h,w],prediction_source='Original observed image; independent of super-resolution',
            spectral_correction_pixels=int(corrected.sum()),measured_nir_used=bool(measured_nir),
            abstained_native_pixels=int(abstained.sum()),
            resolution_note='Model trained on 0.25-0.5 m imagery. Coarse imagery can miss or confuse roads/buildings; 2x display adds no observed detail.')
        if return_probabilities:
            probs[:,~valid]=0
            result['probabilities']=probs
        return result
