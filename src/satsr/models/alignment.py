"""Sub-pixel registration between Sentinel-2 and VENµS patches.

Convention: a shift s = (sy, sx) in HR (5 m) pixels means the HR reference shows the
scene displaced by +s relative to the Sentinel-2 grid, i.e. hr(y, x) ~ up(y - sy, x - sx),
where `up` is any image on the Sentinel-2 grid at HR sampling (bicubic or a model output).
`shift_image(img, s)` applies exactly that displacement, so a prediction is compared with
the reference as shift_image(prediction, s) vs hr.
"""
import torch
import torch.nn.functional as F


def shift_image(image, shift):
    """Displace NCHW `image` content by `shift` (N,2: dy, dx, HR pixels), bicubic, border padded."""
    n, _, h, w = image.shape
    shift = shift.to(image.dtype).to(image.device).view(n, 1, 1, 2)
    ys = torch.arange(h, device=image.device, dtype=image.dtype)
    xs = torch.arange(w, device=image.device, dtype=image.dtype)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    # Output pixel (y, x) samples the input at (y - sy, x - sx).
    sample_y = gy[None] - shift[..., 0]
    sample_x = gx[None] - shift[..., 1]
    grid = torch.stack((2*sample_x/(w-1) - 1, 2*sample_y/(h-1) - 1), -1)
    return F.grid_sample(image, grid, mode="bicubic", padding_mode="border", align_corners=True)


def aligned_l1(prediction, target, shift, margin=4):
    """Per-patch L1 after applying the known sub-pixel shift and removing each band's mean offset."""
    aligned = shift_image(prediction, shift)[..., margin:-margin, margin:-margin]
    diff = aligned - target[..., margin:-margin, margin:-margin]
    return (diff - diff.mean((2, 3), keepdim=True)).abs().mean((1, 2, 3))


@torch.no_grad()
def estimate_shift(lr, hr, radius=3.0, margin=12):
    """Sub-pixel shift (N,2) of HR relative to bicubic-upsampled LR (float tensors, NCHW).

    Grid search at 0.5 px within +-radius, then refinement at 0.125 px within +-0.5 px.
    The per-band mean offset is ignored so sensor radiometry does not bias the estimate.
    """
    up = F.interpolate(lr, scale_factor=2, mode="bicubic", align_corners=False)
    n = lr.shape[0]

    def search(centres, step, half_width):
        offsets = torch.arange(-half_width, half_width + 1e-6, step, device=lr.device)
        best_err = torch.full((n,), float("inf"), device=lr.device)
        best = centres.clone()
        for dy in offsets:
            for dx in offsets:
                candidate = centres + torch.stack((dy, dx))
                err = aligned_l1(up, hr, candidate, margin)
                better = err < best_err
                best_err = torch.where(better, err, best_err)
                best[better] = candidate[better]
        return best

    coarse = search(torch.zeros(n, 2, device=lr.device), 0.5, radius)
    return search(coarse, 0.125, 0.5)


def augment_shift(shift, k, flip):
    """Shift vector after np.rot90(image, k) (axes 0,1) and then an optional horizontal flip."""
    dy, dx = float(shift[0]), float(shift[1])
    for _ in range(k % 4):
        dy, dx = -dx, dy
    if flip:
        dx = -dx
    return dy, dx


@torch.no_grad()
def radiometric_match(lr, hr, shift, coarse=4, limits=(0.8, 1.25), min_std=10.0):
    """Per-patch, per-band gain and offset mapping the VENµS reference onto Sentinel-2 radiometry.

    VENµS references have about 5-9% less contrast than Sentinel-2 at every scale tested
    (10-40 m), a sensor/processing difference rather than spatial detail. Both images are
    compared at `coarse` x 10 m (default 40 m), where detail and residual misregistration
    barely matter: gain = std(S2) / std(VENµS), offset matches the means. Apply as
    hr' = hr * gain + offset. Returns (N, 2, C) float32: [gain, offset] in input units.
    """
    aligned = shift_image(hr, -shift)                     # VENµS content back on the S2 grid
    a = F.avg_pool2d(lr, coarse).flatten(2)
    b = F.avg_pool2d(aligned, 2*coarse).flatten(2)
    sa, sb = a.std(2), b.std(2)
    # Flat patches (water, uniform fields; std below min_std input units) keep gain 1.
    gain = torch.where((sa > min_std) & (sb > min_std), sa/sb.clamp(min=1e-6), torch.ones_like(sa)).clamp(*limits)
    offset = a.mean(2) - gain*b.mean(2)
    return torch.stack((gain, offset), 1)
