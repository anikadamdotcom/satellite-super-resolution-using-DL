"""Validated reconstruction and classification metrics, with optional NoData masks."""
import numpy as np
from scipy.ndimage import uniform_filter, laplace


def validate_pair(img1, img2, max_val=10000.0):
    if not isinstance(img1, np.ndarray) or not isinstance(img2, np.ndarray) or img1.shape != img2.shape or img1.ndim not in (2, 3) or img1.size == 0:
        raise ValueError("Metrics require non-empty images with matching shapes")
    if img1.dtype.kind not in "uif" or img2.dtype.kind not in "uif" or not np.isfinite(img1).all() or not np.isfinite(img2).all() or not np.isfinite(max_val) or max_val <= 0:
        raise ValueError("Metrics require finite real pixels and a positive finite data range")


def _valid(image, mask):
    valid = np.ones(image.shape[:2], bool) if mask is None else np.asarray(mask, bool)
    if valid.shape != image.shape[:2] or not valid.any():
        raise ValueError("Metric mask must match the image and contain valid pixels")
    return valid


def calculate_psnr(img1, img2, max_val=10000.0, valid_mask=None):
    validate_pair(img1, img2, max_val)
    valid = _valid(img1, valid_mask)
    error = img1[valid].astype(np.float64)-img2[valid].astype(np.float64)
    mse = np.mean(error**2)
    return 100.0 if mse <= 1e-10 else float(round(min(100., 10*np.log10(max_val**2/mse)), 2))


def _ssim_channel(a, b, max_val, win_size, valid):
    a, b = a.astype(np.float64), b.astype(np.float64)
    if valid.all():
        weight = 1.
    else:
        weight = np.maximum(uniform_filter(valid.astype(float), win_size), 1e-12)
        a = np.where(valid, a, 0.); b = np.where(valid, b, 0.)
    mu1, mu2 = uniform_filter(a, win_size)/weight, uniform_filter(b, win_size)/weight
    var1 = np.maximum(0., uniform_filter(a*a, win_size)/weight-mu1*mu1)
    var2 = np.maximum(0., uniform_filter(b*b, win_size)/weight-mu2*mu2)
    cov = uniform_filter(a*b, win_size)/weight-mu1*mu2
    c1, c2 = (.01*max_val)**2, (.03*max_val)**2
    scores = (2*mu1*mu2+c1)*(2*cov+c2)/((mu1*mu1+mu2*mu2+c1)*(var1+var2+c2))
    return float(np.mean(np.clip(scores[valid], -1, 1)))


def calculate_ssim(img1, img2, max_val=10000.0, win_size=7, valid_mask=None):
    validate_pair(img1, img2, max_val)
    if type(win_size) is not int or win_size < 1 or win_size % 2 == 0:
        raise ValueError("SSIM window must be a positive odd integer")
    valid = _valid(img1, valid_mask)
    a, b = (img1[..., None], img2[..., None]) if img1.ndim == 2 else (img1, img2)
    return float(round(np.mean([_ssim_channel(a[..., c], b[..., c], max_val, win_size, valid) for c in range(a.shape[2])]), 4))


def calculate_sam(img1, img2, valid_mask=None):
    validate_pair(img1, img2)
    if img1.ndim != 3:
        raise ValueError("SAM requires H x W x channels arrays")
    x, y = img1.astype(np.float64), img2.astype(np.float64)
    nx, ny = np.linalg.norm(x, axis=-1), np.linalg.norm(y, axis=-1)
    denominator = nx*ny
    cosine = np.divide(np.sum(x*y, axis=-1), denominator, out=np.zeros_like(nx), where=denominator > 0)
    valid = _valid(img1, valid_mask) & ((nx > 0) | (ny > 0))
    mean = float(np.arccos(np.clip(cosine[valid], -1, 1)).mean()) if valid.any() else 0.
    return round(float(np.degrees(mean)), 3), round(mean, 4)


def calculate_edge_energy(img, valid_mask=None):
    validate_pair(img, img)
    valid = _valid(img, valid_mask)
    lum = img.astype(np.float64).mean(-1) if img.ndim == 3 else img.astype(np.float64)
    values = laplace(lum)
    # Boundary artifacts around NoData do not count as image detail.
    interior = valid & (uniform_filter(valid.astype(float), 3) >= 1-1e-12)
    return round(float(np.var(values[interior])), 2) if interior.any() else 0.


def reconstruction_metrics(prediction, reference, max_val=10000.0, valid_mask=None):
    sam, _ = calculate_sam(prediction, reference, valid_mask)
    return dict(psnr_db=calculate_psnr(prediction, reference, max_val, valid_mask),
                ssim=calculate_ssim(prediction, reference, max_val, valid_mask=valid_mask), sam_degrees=sam)


def compute_full_sr_evaluation(lr_img, bicubic_img, sr_img, reference=None, max_val=10000.0, valid_mask=None):
    if sr_img.ndim != 3 or lr_img.ndim != 3 or sr_img.shape != (lr_img.shape[0]*2, lr_img.shape[1]*2, lr_img.shape[2]):
        raise ValueError("SR output must be 2x LR dimensions with the same bands")
    validate_pair(bicubic_img, sr_img, max_val)
    target = bicubic_img if reference is None else reference
    result = reconstruction_metrics(sr_img, target, max_val, valid_mask)
    _, result["sam_radians"] = calculate_sam(sr_img, target, valid_mask)
    baseline_edge, sr_edge = calculate_edge_energy(bicubic_img, valid_mask), calculate_edge_energy(sr_img, valid_mask)
    result.update(reference="bicubic similarity (no HR ground truth)" if reference is None else "HR ground truth",
                  is_ground_truth=reference is not None, edge_energy_sr=sr_edge, edge_energy_bicubic=baseline_edge,
                  edge_gain_pct=round(100*(sr_edge-baseline_edge)/(baseline_edge+1e-6), 1),
                  valid_pixels=int(_valid(sr_img, valid_mask).sum()))
    if reference is not None:
        result["bicubic"] = reconstruction_metrics(bicubic_img, reference, max_val, valid_mask)
    return result


def segmentation_metrics(prediction=None, target=None, confusion=None, num_classes=5):
    if type(num_classes) is not int or num_classes < 1:
        raise ValueError("num_classes must be positive")
    if confusion is None:
        prediction, target = np.asarray(prediction), np.asarray(target)
        if prediction.ndim != 2 or prediction.shape != target.shape or prediction.size == 0:
            raise ValueError("Segmentation maps must be non-empty, matching 2D arrays")
        valid = target != 255
        if prediction.dtype.kind not in "ui" or target.dtype.kind not in "ui" or not np.isin(target[valid], np.arange(num_classes)).all() or not np.isin(prediction[valid], np.arange(num_classes)).all():
            raise ValueError("Invalid integer class IDs")
        confusion = np.bincount(num_classes*target[valid].astype(int)+prediction[valid].astype(int), minlength=num_classes**2).reshape(num_classes, num_classes)
    confusion = np.asarray(confusion)
    if confusion.shape != (num_classes, num_classes) or confusion.dtype.kind not in "ui" or (confusion < 0).any():
        raise ValueError("Confusion matrix must contain nonnegative integer counts and match num_classes")
    confusion = confusion.astype(np.int64)
    tp = np.diag(confusion).astype(float)
    union = confusion.sum(0)+confusion.sum(1)-tp
    iou = np.divide(tp, union, out=np.full(num_classes, np.nan), where=union > 0)
    return dict(pixel_accuracy=float(tp.sum()/confusion.sum()) if confusion.sum() else None,
                mean_iou=float(np.nanmean(iou)) if np.isfinite(iou).any() else None,
                per_class_iou=[float(v) if np.isfinite(v) else None for v in iou], confusion_matrix=confusion.tolist())


def calculate_land_cover_stats(class_map, class_defs, pixel_size_m=5.0):
    class_map = np.asarray(class_map)
    if class_map.ndim != 2 or class_map.size == 0 or class_map.dtype.kind not in "ui" or not np.isin(class_map, list(class_defs)+[255]).all():
        raise ValueError("Invalid land-cover class map")
    if pixel_size_m is not None and (not np.isfinite(pixel_size_m) or pixel_size_m <= 0):
        raise ValueError("Pixel size must be positive and finite")
    total = int((class_map != 255).sum())
    area = pixel_size_m**2 if pixel_size_m is not None else None
    return {info["name"]:dict(class_id=cid, hex=info["hex"], count=int((class_map == cid).sum()),
                percentage=round(100*int((class_map == cid).sum())/total, 2) if total else 0.,
                area_ha=round(int((class_map == cid).sum())*area/10000, 3) if area is not None else None,
                area_km2=round(int((class_map == cid).sum())*area/1e6, 4) if area is not None else None)
            for cid, info in class_defs.items()}
