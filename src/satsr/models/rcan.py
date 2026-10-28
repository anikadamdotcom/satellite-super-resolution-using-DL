# models/rcan.py
# Deep Learning Super-Resolution Architecture: Residual Channel Attention Network (RCAN)
# Tailored for Sentinel-2 4-Band Multi-Spectral Satellite Imagery (B02, B03, B04, B08)

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def default_conv(in_channels: int, out_channels: int, kernel_size: int, bias: bool = True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias
    )


class Upsampler(nn.Sequential):
    """Sub-pixel Convolutional Upsampler (PixelShuffle)"""
    def __init__(self, conv, scale: int, n_feats: int, bn: bool = False, act: bool = False, bias: bool = True):
        if type(scale) is not int or scale < 1:
            raise ValueError("Upsampling scale must be a positive integer")
        modules = []
        if (scale & (scale - 1)) == 0:
            for _ in range(int(math.log(scale, 2))):
                modules.append(conv(n_feats, 4 * n_feats, 3, bias))
                modules.append(nn.PixelShuffle(2))
                if bn:
                    modules.append(nn.BatchNorm2d(n_feats))
                if act == "relu":
                    modules.append(nn.ReLU(True))
                elif act == "prelu":
                    modules.append(nn.PReLU(n_feats))
        elif scale == 3:
            modules.append(conv(n_feats, 9 * n_feats, 3, bias))
            modules.append(nn.PixelShuffle(3))
            if bn:
                modules.append(nn.BatchNorm2d(n_feats))
            if act == "relu":
                modules.append(nn.ReLU(True))
            elif act == "prelu":
                modules.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError(f"Scale {scale} not supported.")
        super(Upsampler, self).__init__(*modules)


class CALayer(nn.Module):
    """Channel Attention (CA) Layer to exploit inter-channel correlations among spectral bands"""
    def __init__(self, channel: int, reduction: int = 16):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, max(1, channel // reduction), 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(1, channel // reduction), channel, 1, padding=0, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class RCAB(nn.Module):
    """Residual Channel Attention Block (RCAB) combining residual learning with channel attention"""
    def __init__(self, conv, n_feat: int, kernel_size: int, reduction: int = 16, bias: bool = True, bn: bool = False, act=nn.ReLU(True)):
        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn:
                modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0:
                modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x) + x


class ResidualGroup(nn.Module):
    """Residual Group (RG) containing multiple RCAB blocks with short-cut connection"""
    def __init__(self, conv, n_feat: int, kernel_size: int, reduction: int, n_resblocks: int):
        super(ResidualGroup, self).__init__()
        modules_body = [
            RCAB(conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True))
            for _ in range(n_resblocks)
        ]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x) + x


class RCAN(nn.Module):
    """
    Residual Channel Attention Network for 4-Band Sentinel-2 Super-Resolution (2x scale).
    Inputs: 4 Channels (B02 Blue, B03 Green, B04 Red, B08 NIR)
    Outputs: 4 Channels at 2x spatial resolution.
    """
    def __init__(self, n_colors: int = 4, scale: int = 2, n_feats: int = 64, n_resgroups: int = 10, n_resblocks: int = 20, conv=default_conv, residual: bool = False, consistent: bool = False):
        super(RCAN, self).__init__()
        if any(type(v) is not int or v < 1 for v in (n_colors, n_feats, n_resgroups, n_resblocks)):
            raise ValueError("RCAN channel and block counts must be positive integers")
        self.scale = scale
        self.residual = residual
        # Exact consistency: the 2x2 mean of the output equals the input pixel, so the
        # measured 10 m radiometry is preserved and only sub-pixel detail is added.
        self.consistent = consistent
        kernel_size = 3
        reduction = 16
        
        # Shallow feature extraction head
        self.head = nn.Sequential(conv(n_colors, n_feats, kernel_size))
        
        # Deep residual feature extraction trunk with channel attention
        modules_body = [
            ResidualGroup(conv, n_feats, kernel_size, reduction, n_resblocks=n_resblocks)
            for _ in range(n_resgroups)
        ]
        modules_body.append(conv(n_feats, n_feats, kernel_size))
        self.body = nn.Sequential(*modules_body)
        
        # Upsampling and reconstruction head
        self.tail = nn.Sequential(
            Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, n_colors, kernel_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_head = self.head(x)
        res = self.body(x_head)
        res = res + x_head
        out = self.tail(res)
        if self.residual:
            out = out + F.interpolate(x, scale_factor=self.scale, mode="bicubic", align_corners=False)
        if self.consistent:
            out = out + F.interpolate(x - F.avg_pool2d(out, self.scale), scale_factor=self.scale, mode="nearest")
        return out


class L1BSRSR:
    """Checkpoint-backed RCAN; explicit bicubic baseline when no weights exist."""
    def __init__(self, weights_path=None, device=None, tile_size=128):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = None
        self.weights_loaded = False
        self.loaded_weights_path = None
        self.tile_size = tile_size
        self.provenance = "Untrained bicubic baseline"
        if weights_path:
            self.load_weights(weights_path)

    def load_weights(self, weights_path):
        from .common import read_checkpoint
        checkpoint = read_checkpoint(weights_path, "sr")
        model = RCAN(**checkpoint["model_config"])
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        self.model = model.to(self.device).eval()
        self.loaded_weights_path = str(weights_path)
        self.weights_loaded = True
        self.provenance = checkpoint.get("data_source", "unspecified training data")

    @torch.inference_mode()
    def super_resolve(self, img_bgrn_u16):
        from .common import validate_image, tiled_predict, NORMALIZATION
        image = validate_image(img_bgrn_u16)
        if not self.weights_loaded:
            return self.bicubic_upscale(image)
        chw = np.moveaxis(image.astype(np.float32) / NORMALIZATION, -1, 0)
        out = tiled_predict(self.model, chw, self.device, scale=2, tile_size=self.tile_size)
        return np.clip(np.moveaxis(out, 0, -1) * NORMALIZATION, 0, 65535).round().astype(np.uint16)

    @staticmethod
    def bicubic_upscale(img_bgrn_u16, scale=2):
        from .common import validate_image
        if not isinstance(scale, int) or scale < 1:
            raise ValueError("scale must be a positive integer")
        image = validate_image(img_bgrn_u16)
        x = torch.from_numpy(image.astype(np.float32)).permute(2, 0, 1)[None]
        out = F.interpolate(x, scale_factor=scale, mode="bicubic", align_corners=False)[0]
        return np.clip(np.moveaxis(out.numpy(), 0, -1), 0, 65535).round().astype(np.uint16)
