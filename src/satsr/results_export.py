"""Export a complete GUI result as one atomic ZIP download."""
import json
import tempfile
import zipfile
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from .geo_io import (resolve_output_path, save_image, save_text, output_valid_mask,
                    write_geotiff, write_single_band_geotiff, save_jpeg_with_jgw,
                    _atomic_write, derived_note)

# Difference map: one blue hue, light (no change) to dark (large change), fixed scale
# so maps from different images are comparable. 500 = 0.05 reflectance.
DIFF_SCALE_DN = 500.
DIFF_RAMP = np.array([[0xfc, 0xfc, 0xfb], [0xcd, 0xe2, 0xfb], [0x9e, 0xc5, 0xf4], [0x6d, 0xa7, 0xec],
                      [0x39, 0x87, 0xe5], [0x25, 0x6a, 0xbf], [0x18, 0x4f, 0x95], [0x0d, 0x36, 0x6b]], float)
COMPARE_GAP = 8


def difference_from_bicubic(enhanced_bands, bicubic_bands, valid=None):
    """Per-pixel mean absolute difference over the four bands (reflectance x 10000).

    Shows where the model changed the image relative to plain bicubic
    interpolation. A change is not proof of real detail: it may be invented."""
    a, b = np.asarray(enhanced_bands, np.float32), np.asarray(bicubic_bands, np.float32)
    if a.shape != b.shape or a.ndim != 3 or a.size == 0:
        raise ValueError("Enhanced and bicubic bands must have the same H x W x bands shape")
    magnitude = np.abs(a-b).mean(-1)
    keep = np.ones(magnitude.shape, bool) if valid is None else np.asarray(valid, bool)
    values = magnitude[keep] if keep.any() else np.zeros(1, np.float32)
    stats = dict(mean_reflectance=round(float(values.mean())/10000, 5),
                 p95_reflectance=round(float(np.percentile(values, 95))/10000, 5),
                 changed_pct=round(100*float((values > 100).mean()), 2),
                 note="Mean |enhanced - bicubic| over 4 bands; % of pixels changed by more than 0.01 reflectance")
    return magnitude, stats


def difference_rgb(magnitude, valid=None):
    t = np.clip(np.nan_to_num(np.asarray(magnitude, np.float32))/DIFF_SCALE_DN, 0, 1)*(len(DIFF_RAMP)-1)
    low = np.floor(t).astype(int).clip(0, len(DIFF_RAMP)-2)
    frac = (t-low)[..., None]
    rgb = (DIFF_RAMP[low]*(1-frac)+DIFF_RAMP[low+1]*frac).round().astype(np.uint8)
    if valid is not None:
        rgb[~np.asarray(valid, bool)] = 0
    return rgb


def _font(size):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


def compose_compare(original, enhanced, diff_rgb, valid):
    """Original | enhanced | difference, with a caption strip naming each panel and the scale."""
    h, w = enhanced.shape[:2]
    if original.shape[:2] != (h, w) or diff_rgb.shape[:2] != (h, w):
        raise ValueError("Compare panels must share one size")
    gap = np.full((h, COMPARE_GAP, 3), 255, np.uint8)
    panels = np.concatenate([original, gap, enhanced, gap, diff_rgb], axis=1)
    size = max(9, min(24, w//20))
    caption_h = max(30, int(size*2.8))
    canvas = Image.new("RGB", (panels.shape[1], caption_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    font, ink, muted = _font(size), (11, 11, 11), (82, 81, 78)
    titles = ("Original", "Enhanced (RCAN)", "Change vs bicubic")
    for i, title in enumerate(titles[:2]):
        x0 = i*(w+COMPARE_GAP)
        draw.text((x0+w/2, caption_h/2), title, fill=ink, font=font, anchor="mm")
    # Third panel: title on the left, then a gradient legend from 0 to the scale maximum.
    x0 = 2*(w+COMPARE_GAP)
    draw.text((x0+4, caption_h*.3), titles[2], fill=ink, font=font, anchor="lm")
    bar_y, bar_h = int(caption_h*.58), max(4, caption_h//6)
    label_w = draw.textlength("0.05+", font=_font(max(8, size-2)))
    bar_x0, bar_x1 = x0+4+int(draw.textlength("0 ", font=_font(max(8, size-2)))), x0+w-int(label_w)-8
    if bar_x1 > bar_x0+8:
        ramp = difference_rgb(np.linspace(0, DIFF_SCALE_DN, bar_x1-bar_x0)[None].repeat(bar_h, 0))
        canvas.paste(Image.fromarray(ramp), (bar_x0, bar_y))
        small = _font(max(8, size-2))
        draw.text((x0+4, bar_y+bar_h/2), "0", fill=muted, font=small, anchor="lm")
        draw.text((bar_x1+4, bar_y+bar_h/2), "0.05+", fill=muted, font=small, anchor="lm")
    full = np.concatenate([panels, np.asarray(canvas)], axis=0)
    gap_valid = np.ones((h, COMPARE_GAP), bool)
    full_valid = np.concatenate([np.concatenate([valid, gap_valid, valid, gap_valid, valid], axis=1),
                                 np.ones((caption_h, panels.shape[1]), bool)], axis=0)
    return full, full_valid


def score_rgb(confidence):
    score = np.nan_to_num(confidence, nan=0.)
    rgb = np.stack([255*(1-score), 255*score, np.zeros_like(score)], -1).astype(np.uint8)
    rgb[~np.isfinite(confidence)] = 0
    return rgb


def export_results_zip(path, original, enhanced_rgb, enhanced_bands, ref, segmentation,
                       report_text, report_data, bicubic_bands=None, sr_model="RCAN"):
    path = resolve_output_path(path, {".zip"}, ".zip")
    if not path.parent.is_dir():
        raise ValueError("Choose an existing download folder")
    valid = output_valid_mask(ref, enhanced_bands.shape)
    with tempfile.TemporaryDirectory(prefix=".l1bsr-download-", dir=path.parent) as name:
        stage = Path(name).resolve()
        if stage.parent != path.parent.resolve():
            raise ValueError("Invalid download workspace")
        note = derived_note(sr_model)
        save_image(stage/"original.png", original, ref.valid_mask)
        save_image(stage/"enhanced.png", enhanced_rgb, valid, description=note)
        save_image(stage/"enhanced.jpg", enhanced_rgb, description=note)
        save_image(stage/"land_cover.png", segmentation["color_map"], valid)
        overlay = (enhanced_rgb.astype(np.float32)*.45 +
                   segmentation["color_map"].astype(np.float32)*.55).astype(np.uint8)
        save_image(stage/"overlay.png", overlay, valid, description=note)
        save_image(stage/"scores.png", score_rgb(segmentation["confidence_map"]), valid)
        np.save(stage/"class_ids.npy", segmentation["class_map"], allow_pickle=False)
        np.save(stage/"scores.npy", segmentation["confidence_map"], allow_pickle=False)
        np.save(stage/"enhanced_bands.npy", enhanced_bands, allow_pickle=False)
        if bicubic_bands is not None:
            magnitude, _ = difference_from_bicubic(enhanced_bands, bicubic_bands, valid)
            save_image(stage/"difference_from_bicubic.png", difference_rgb(magnitude, valid), valid)
        if ref.crs is not None:
            write_geotiff(stage/"enhanced.tif", enhanced_bands, ref, sr_model=sr_model)
            write_single_band_geotiff(stage/"land_cover.tif", segmentation["class_map"], ref)
            save_jpeg_with_jgw(stage/"enhanced.jpg", enhanced_rgb, ref, description=note)
        report = dict(report_data)
        report["enhanced_imagery"] = note
        report["files"] = sorted([p.name for p in stage.iterdir()] + ["report.md", "report.json"])
        save_text(stage/"report.md", report_text)
        save_text(stage/"report.json", json.dumps(report, indent=2, allow_nan=False))
        def write_zip(temp):
            with zipfile.ZipFile(temp, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for file in sorted(stage.iterdir()):
                    archive.write(file, file.name)
        _atomic_write(path, write_zip)
    return path
