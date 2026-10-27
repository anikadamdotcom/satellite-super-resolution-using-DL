# models/unet_segmentation.py
# Deep Learning Semantic Segmentation for Satellite Land Cover Mapping
# Implements Slide 7 ("From Enhanced Image to Intelligent Map")
# Classifies Sentinel-2 Multi-Spectral Imagery into 5 Target Geospatial Classes:
# 1. Buildings  2. Roads  3. Vegetation  4. Water  5. Bare Land

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import binary_opening, binary_closing


# Standard Color Palette for Geospatial Land Cover Classes (Slide 7)
CLASS_DEFINITIONS = {
    0: {"name": "Buildings", "color": [230, 57, 70], "hex": "#E63946", "desc": "Urban structures & residential footprints"},
    1: {"name": "Roads", "color": [73, 80, 87], "hex": "#495057", "desc": "Transportation corridors & paved surfaces"},
    2: {"name": "Vegetation", "color": [42, 157, 143], "hex": "#2A9D8F", "desc": "Dense forests, crop fields & grassland"},
    3: {"name": "Water", "color": [29, 53, 87], "hex": "#1D3557", "desc": "Rivers, reservoirs, lakes & canals"},
    4: {"name": "Bare Land", "color": [233, 196, 106], "hex": "#E9C46A", "desc": "Exposed soil, fallow ground & sand"}
}

# Preserve the distinction between a road and a parking lot/paved developed area.
REAL_CLASS_DEFINITIONS = {**CLASS_DEFINITIONS, 5: {
    "name": "Developed space", "color": [164, 139, 191], "hex": "#A48BBF",
    "desc": "Developed surfaces distinct from road and building predictions"}}


class ConvBlock(nn.Module):
    """Double 3x3 Conv + BatchNorm + LeakyReLU"""
    def __init__(self, in_ch: int, out_ch: int):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class SatelliteSegmentationNet(nn.Module):
    """
    Multi-Scale Deep U-Net for Satellite Semantic Segmentation.
    Fuses Multi-Spectral Sentinel-2 Bands with Remote Sensing Indices (NDVI, NDWI, red/NIR contrast).
    """
    def __init__(self, in_channels: int = 8, num_classes: int = 5):
        super(SatelliteSegmentationNet, self).__init__()
        
        # Contracting Path (Encoder)
        self.enc1 = ConvBlock(in_channels, 32)
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = ConvBlock(32, 64)
        self.pool2 = nn.MaxPool2d(2)
        
        self.enc3 = ConvBlock(64, 128)
        self.pool3 = nn.MaxPool2d(2)
        
        # Bottleneck with residual bridge
        self.bottleneck = ConvBlock(128, 256)
        
        # Expanding Path (Decoder)
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(256, 128)
        
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(128, 64)
        
        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(64, 32)
        
        # Classification Head
        self.out_conv = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        
        # Bottleneck
        b = self.bottleneck(self.pool3(e3))
        
        # Decoder with Skip Connections
        d3 = self.up3(b)
        if d3.shape != e3.shape:
            d3 = F.interpolate(d3, size=e3.shape[2:], mode='bilinear', align_corners=False)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        
        d2 = self.up2(d3)
        if d2.shape != e2.shape:
            d2 = F.interpolate(d2, size=e2.shape[2:], mode='bilinear', align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        
        d1 = self.up1(d2)
        if d1.shape != e1.shape:
            d1 = F.interpolate(d1, size=e1.shape[2:], mode='bilinear', align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        
        logits = self.out_conv(d1)
        return logits


class LandCoverSegmenter:
    """Trained U-Net probabilities, or explicitly labeled spectral-rule scores."""
    def __init__(self, device=None, weights_path=None, tile_size=128, real_weights_path=None, hf_weights_path=None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = None
        self.weights_loaded = False
        self.tile_size = tile_size
        self.provenance = "Untrained spectral-rule baseline"
        self.real_engine = None
        if hf_weights_path:
            from .hf_landcover import HuggingFaceLandCoverEngine, EnsembleLandCoverEngine
            self.real_engine = (EnsembleLandCoverEngine(real_weights_path, hf_weights_path, self.device)
                if real_weights_path else HuggingFaceLandCoverEngine(hf_weights_path, self.device))
            self.model = self.real_engine.model
            self.weights_loaded = True
            self.provenance = self.real_engine.provenance
        elif real_weights_path:
            from .real_landcover import RealLandCoverEngine
            self.real_engine = RealLandCoverEngine(real_weights_path, self.device)
            self.model = self.real_engine.model
            self.weights_loaded = True
            self.provenance = self.real_engine.provenance
        elif weights_path:
            self.load_weights(weights_path)

    def load_weights(self, path):
        from .common import read_checkpoint
        checkpoint = read_checkpoint(path, "segmentation")
        model = SatelliteSegmentationNet(**checkpoint["model_config"])
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        self.model = model.to(self.device).eval()
        self.weights_loaded = True
        self.provenance = checkpoint.get("data_source", "unspecified training data")

    @staticmethod
    def extract_features(image):
        from .common import extract_features
        return extract_features(image)

    @torch.inference_mode()
    def segment(self, img_bgrn_u16, apply_morphology=True, pixel_size_m=5.0,
                pixel_area_m2=None, valid_mask=None, return_probabilities=True,
                rgb=None, measured_nir=False, output_shape=None):
        if self.real_engine is not None:
            return self.real_engine.segment(img_bgrn_u16, rgb=rgb, measured_nir=measured_nir,
                output_shape=output_shape, valid_mask=valid_mask, pixel_size_m=pixel_size_m,
                pixel_area_m2=pixel_area_m2, return_probabilities=return_probabilities)
        from .common import tiled_predict
        if pixel_size_m is not None and (not np.isfinite(pixel_size_m) or pixel_size_m <= 0):
            raise ValueError("Pixel size must be positive and finite")
        feat, ind = self.extract_features(img_bgrn_u16)
        h, w = feat.shape[:2]
        if self.weights_loaded:
            logits = tiled_predict(self.model, np.moveaxis(feat, -1, 0), self.device,
                                   tile_size=self.tile_size, halo=64, min_size=8, alignment=8)
            logits -= logits.max(axis=0, keepdims=True)
            probs = np.exp(logits)
        else:
            water = (ind["ndwi"] > .15) | ((ind["b08"] < .04) & (ind["b02"] > .07))
            veg = (ind["ndvi"] > .38) & ~water
            road = (ind["grad"] > .12) & (ind["ndvi"] < .30) & ~water & ~veg
            building = (ind["lum"] > .18) & (ind["ndvi"] < .35) & ~water & ~veg & ~road
            bare = ~(water | veg | road | building)
            probs = np.stack([building, road, veg, water, bare]).astype(np.float32) * .9 + .02
        probs /= probs.sum(axis=0, keepdims=True)
        raw = probs.argmax(axis=0).astype(np.uint8)
        class_map = raw.copy()
        if apply_morphology:
            # Remove isolated components as well as filling holes. Resolve conflicts by score.
            eligible = np.ones_like(probs, dtype=bool)
            for cid in (0, 1, 3):
                cleaned = binary_opening(binary_closing(raw == cid, border_value=1),
                                         structure=np.ones((2, 2)))
                eligible[cid] = cleaned
            scores = np.where(eligible, probs, -1.0)
            class_map = scores.argmax(axis=0).astype(np.uint8)
        confidence = np.take_along_axis(probs, class_map[None].astype(int), axis=0)[0]
        valid = np.ones((h, w), dtype=bool) if valid_mask is None else np.asarray(valid_mask, dtype=bool)
        if valid.shape != (h, w):
            raise ValueError("valid_mask shape does not match image")
        confidence = confidence.astype(np.float32)
        confidence[~valid] = np.nan
        class_map[~valid] = 255
        color = np.zeros((h, w, 3), dtype=np.uint8)
        stats = {}
        area = pixel_size_m ** 2 if pixel_area_m2 is None and pixel_size_m is not None else pixel_area_m2
        if area is not None and (not np.isfinite(area) or area <= 0):
            raise ValueError("Pixel area must be finite and positive")
        total = int(valid.sum())
        for cid, info in CLASS_DEFINITIONS.items():
            mask = class_map == cid
            color[mask] = info["color"]
            count = int(mask.sum())
            stats[info["name"]] = dict(class_id=cid, color_hex=info["hex"], pixel_count=count,
                percentage=round(100 * count / total, 2) if total else 0.0,
                area_hectares=round(count * area / 10000, 3) if area is not None else None,
                area_km2=round(count * area / 1e6, 4) if area is not None else None,
                description=info["desc"])
        result = dict(class_map=class_map, color_map=color, confidence_map=confidence,
                    uncertainty_map=1-confidence, statistics=stats,
                    method="U-Net" if self.weights_loaded else "spectral rules",
                    provenance=self.provenance, confidence_note="Uncalibrated model probabilities" if self.weights_loaded else "Heuristic scores; not calibrated probabilities")
        if return_probabilities:
            result["probabilities"] = probs
        return result
