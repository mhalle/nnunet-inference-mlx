"""SynthStrip brain extraction (skull-strip) as an nnseg engine.

SynthStrip is a contrast-agnostic learned brain-mask generator (a small 3D UNet,
Hoopes et al. 2022) - a different algorithm family from nnU-Net and FastSurfer. It
is deliberately the LIGHTEST engine: torch + numpy + scipy + SimpleITK (all
already in nnseg's stack) + the vendored :mod:`._synthstrip_model` (~80 lines) +
a 29 MB ``model_state_dict``. No surfa, no FreeSurfer, no FastSurferCNN.

Shape mirrors :mod:`nnseg.engines.fastsurfer` (current per-engine pattern; the
Ecosystem/Engine re-architecture is a later holistic pass): conform in memory
(SimpleITK, surfa-free), run the net -> a signed distance transform (SDT), restore
the *graded* SDT to the input grid (physical-space ``grid_sample``, reusing the
FastSurfer restore geometry), then threshold + largest-component -> a 1-label
brain mask :class:`nnseg.result.Segmentation`. Proven torch-only in a spike
(2026-08-26): correct 1326 ml mask, zero missing state_dict keys.
"""
from __future__ import annotations

import os

import numpy as np

WEIGHTS_ID = "synthstrip"
WEIGHTS_VERSION = "v1"                       # synthstrip.1.pt (MGH, 2022-04-28)
MODEL_URL = ("https://ftp.nmr.mgh.harvard.edu/pub/dist/freesurfer/synthstrip/"
             "models/synthstrip.1.pt")
DEFAULT_MODEL_PATH = "/opt/synthstrip/synthstrip.1.pt"   # baked into the worker image
BRAIN_LABEL = 1


def weights_installed() -> list[dict]:
    """The engine's weights identity for the result-cache key - one source of
    truth for both the API-side describe and the worker re-key (same contract as
    :func:`nnseg.engines.fastsurfer.weights_installed`)."""
    return [{"id": WEIGHTS_ID, "version": WEIGHTS_VERSION}]


_MODELS: dict = {}                           # device -> StripModel, cached across jobs


def _get_model(device: str):
    """Build StripModel and load the bundled weights ONCE per device (the model
    is input-independent). Weights path: ``NNSEG_SYNTHSTRIP_MODEL`` or the baked
    default; loaded with ``weights_only=True`` (no pickle execution)."""
    import torch

    key = str(device)
    m = _MODELS.get(key)
    if m is not None:
        return m
    from ._synthstrip_model import StripModel

    path = os.environ.get("NNSEG_SYNTHSTRIP_MODEL", DEFAULT_MODEL_PATH)
    if not os.path.exists(path):
        raise RuntimeError(
            f"SynthStrip weights not found at {path!r}; bake {MODEL_URL} into the "
            "image or set NNSEG_SYNTHSTRIP_MODEL to its path")
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    m = StripModel()
    m.load_state_dict(sd)
    m.eval()
    m = m.to(device)
    _MODELS[key] = m
    return m


def conform(t1_img):
    """SynthStrip's trained-input conform, surfa-free (SimpleITK): resample to
    1 mm, reorient to LIA, crop to the nonzero bbox, pad/crop to a multiple of 64
    per axis in [192, 320], and intensity-normalize (``-min``, ``/p99``, clip
    [0,1]). Returns ``(conf_sitk, arr_zyx)`` - the SimpleITK image carries the
    conformed geometry the restore resamples FROM; ``arr_zyx`` is the normalized
    model input.

    NOTE: an approximate replica of surfa's conform (spike-verified to a correct
    mask volume); boundary parity against surfa is a follow-up before this is
    trusted for clinical use, the same trained-input-contract caution as FastSurfer.
    """
    import SimpleITK as sitk

    sp = t1_img.GetSpacing()
    size = [int(round(t1_img.GetSize()[i] * sp[i])) for i in range(3)]
    rs = sitk.ResampleImageFilter()
    rs.SetOutputSpacing((1.0, 1.0, 1.0)); rs.SetSize(size)
    rs.SetOutputOrigin(t1_img.GetOrigin()); rs.SetOutputDirection(t1_img.GetDirection())
    rs.SetInterpolator(sitk.sitkLinear)
    iso = sitk.DICOMOrient(rs.Execute(t1_img), "LIA")

    a = sitk.GetArrayFromImage(iso)                       # (z, y, x)
    nz = np.argwhere(a > float(a.min()))
    lo, hi = nz.min(0), nz.max(0) + 1                     # (z, y, x)
    crop = sitk.RegionOfInterest(iso, [int(hi[2]-lo[2]), int(hi[1]-lo[1]), int(hi[0]-lo[0])],
                                 [int(lo[2]), int(lo[1]), int(lo[0])])   # sitk order (x,y,z)
    csz = crop.GetSize()                                  # (x, y, z)
    tgt = [int(np.clip(int(np.ceil(s / 64)) * 64, 192, 320)) for s in csz]
    lower, upper, roi_lo, roi_sz = [], [], [], []
    for i in range(3):
        d = tgt[i] - csz[i]
        if d >= 0:
            lower.append(d // 2); upper.append(d - d // 2); roi_lo.append(0); roi_sz.append(csz[i])
        else:
            lower.append(0); upper.append(0); roi_lo.append((-d) // 2); roi_sz.append(tgt[i])
    conf = sitk.ConstantPad(sitk.RegionOfInterest(crop, roi_sz, roi_lo), lower, upper, 0.0)

    arr = sitk.GetArrayFromImage(conf).astype(np.float32)
    arr = arr - float(arr.min())
    p99 = float(np.percentile(arr, 99)) or 1.0
    arr = np.clip(arr / p99, 0.0, 1.0)
    return conf, arr


def _resample_affine(source_ref, target_ref):
    """3x3 A and offset t mapping a TARGET voxel index (x,y,z) to the continuous
    SOURCE voxel index (x,y,z), composing both SimpleITK grids' transforms (same
    derivation as the FastSurfer restore; source direction is orthonormal)."""
    O_s = np.asarray(source_ref.GetOrigin(), dtype=np.float64)
    O_t = np.asarray(target_ref.GetOrigin(), dtype=np.float64)
    D_s = np.asarray(source_ref.GetDirection(), dtype=np.float64).reshape(3, 3)
    D_t = np.asarray(target_ref.GetDirection(), dtype=np.float64).reshape(3, 3)
    S_s = np.asarray(source_ref.GetSpacing(), dtype=np.float64)
    S_t = np.asarray(target_ref.GetSpacing(), dtype=np.float64)
    inv_s = np.diag(1.0 / S_s) @ D_s.T
    return inv_s @ D_t @ np.diag(S_t), inv_s @ (O_t - O_s)


def restore_sdt_gpu(sdt, source_ref, target_ref, device="cuda", outside=100.0):
    """Trilinear-resample the 1-channel SDT from ``source_ref``'s grid onto
    ``target_ref``'s grid (physical-space ``grid_sample``, full affine so any
    orientation/rotation is handled). Voxels sampling outside the source get
    ``outside`` (a large positive distance = far from brain, matching surfa's
    ``fill=100``). ``sdt`` is a torch tensor already ``(Zs,Ys,Xs)`` on a device
    (on-GPU path) or a numpy array. Returns the resampled SDT ``(Zt,Yt,Xt)``."""
    import torch
    import torch.nn.functional as F

    dev = torch.device(device)
    if isinstance(sdt, torch.Tensor):
        Zs, Ys, Xs = (int(s) for s in sdt.shape)
        field = sdt.to(dev).float()
    else:
        Zs, Ys, Xs = sdt.shape
        field = torch.from_numpy(np.ascontiguousarray(sdt)).to(dev, torch.float32)
    # Shift by -outside so grid_sample's zero-padding REPRESENTS the outside fill:
    # outside/partial-edge taps contribute 0 -> +outside after, matching SimpleITK's
    # constant fill exactly (else zero-padding would drag the edge SDT toward 0 and
    # spuriously threshold as brain). Makes this bit-match restore_sdt_cpu.
    field = (field - float(outside))[None, None]
    tgt = target_ref.GetSize()
    Xt, Yt, Zt = int(tgt[0]), int(tgt[1]), int(tgt[2])
    A, t = _resample_affine(source_ref, target_ref)
    zz, yy, xx = torch.meshgrid(torch.arange(Zt), torch.arange(Yt), torch.arange(Xt),
                                indexing="ij")
    idx = torch.stack([xx, yy, zz], dim=-1).to(dev, torch.float64)
    src = idx @ torch.as_tensor(A, device=dev, dtype=torch.float64).T \
        + torch.as_tensor(t, device=dev, dtype=torch.float64)
    N = torch.as_tensor([Xs, Ys, Zs], device=dev, dtype=torch.float64)
    grid = ((src + 0.5) * 2.0 / N - 1.0).to(torch.float32)[None]
    out = F.grid_sample(field, grid, mode="bilinear", padding_mode="zeros",
                        align_corners=False)[0, 0] + float(outside)
    return out.float().cpu().numpy()


def restore_sdt_cpu(sdt_zyx, source_ref, target_ref, outside=100.0):
    """Memory-frugal fallback: one SimpleITK linear resample of the SDT (single
    channel, so cheap even on CPU). ``outside`` is the fill for voxels off the
    source grid (surfa uses 100)."""
    import SimpleITK as sitk

    ch = sitk.GetImageFromArray(np.ascontiguousarray(sdt_zyx.astype(np.float32)))
    ch.CopyInformation(source_ref)
    out = sitk.Resample(ch, target_ref, sitk.Transform(), sitk.sitkLinear,
                        float(outside), sitk.sitkFloat32)
    return sitk.GetArrayFromImage(out)


def _capture_sdt(t1_img, device: str, on_gpu: bool = True):
    """Conform (SimpleITK) + run the cached model -> the SDT field in the
    conformed frame. Keeps the SDT on the device when ``on_gpu`` (no transfer;
    the restore resamples it there). Returns ``(sdt, conf_sitk)``."""
    import torch

    model = _get_model(device)
    conf, arr = conform(t1_img)
    with torch.no_grad():
        inp = torch.from_numpy(arr[None, None]).to(device)
        sdt = model(inp).squeeze()                        # (Zs,Ys,Xs) on device
    return (sdt if on_gpu else sdt.float().cpu().numpy()), conf


def _fs_version() -> str:
    return WEIGHTS_VERSION


def segment(t1_input, *, out_dir=None, device: str = "cuda", restore: str = "auto",
            border: float = 1.0, self_check: bool = True):
    """Skull-strip a brain image with SynthStrip and return a 1-label brain-mask
    :class:`nnseg.result.Segmentation` on the input's grid.

    ``t1_input`` is a SimpleITK image (memory-in) or a path. ``restore`` selects
    the SDT-restore backend (``"gpu"``/``"cpu"``/``"auto"`` = GPU on CUDA). The
    graded SDT is resampled to the input grid, then thresholded at ``border`` mm
    (SynthStrip's default) and reduced to the largest filled connected component -
    thresholding *after* the resample gives a sub-voxel mask boundary. ``out_dir``
    is accepted for call-site compatibility and unused (no temp files)."""
    import time

    import SimpleITK as sitk

    from ..grid import Grid
    from ..result import Segmentation
    from ..values import LabelSchema

    if isinstance(t1_input, sitk.Image):
        t1_img = t1_input
    else:
        from .. import io
        t1_img = io.read_image(str(t1_input))

    use_gpu = restore == "gpu" or (restore == "auto" and str(device).startswith("cuda"))
    timings: dict[str, float] = {}
    _t = time.perf_counter()
    sdt, conf_orig = _capture_sdt(t1_img, device, on_gpu=use_gpu)
    timings["capture"] = time.perf_counter() - _t

    _t = time.perf_counter()
    if use_gpu:
        sdt_native = restore_sdt_gpu(sdt, conf_orig, t1_img, device)
    else:
        arr = sdt if isinstance(sdt, np.ndarray) else sdt.float().cpu().numpy()
        sdt_native = restore_sdt_cpu(arr, conf_orig, t1_img)
    timings["restore"] = time.perf_counter() - _t

    from scipy import ndimage
    mask = sdt_native < border
    lbl, n = ndimage.label(mask)
    if n:
        biggest = 1 + int(np.argmax(np.bincount(lbl.flat)[1:]))
        mask = ndimage.binary_fill_holes(lbl == biggest)
    nfg = int(mask.sum())
    print(f"[synthstrip] conf={conf_orig.GetSize()} t1={t1_img.GetSize()} "
          f"restore={'gpu' if use_gpu else 'cpu'} components={n} "
          f"brain_voxels={nfg}", flush=True)
    if nfg == 0:
        raise RuntimeError(
            f"synthstrip mask is empty: no voxel had SDT < {border} mm "
            f"(conf={conf_orig.GetSize()}, t1={t1_img.GetSize()}); "
            "conform or restore geometry is likely wrong")

    labels_arr = mask.astype(np.uint16)
    out_img = sitk.GetImageFromArray(labels_arr)
    out_img.CopyInformation(t1_img)
    grid = Grid(shape=tuple(int(s) for s in labels_arr.shape),
                spacing=tuple(float(s) for s in reversed(t1_img.GetSpacing())),
                origin=tuple(float(o) for o in reversed(t1_img.GetOrigin())))
    prov = {"engine": "synthstrip", "synthstrip_version": WEIGHTS_VERSION,
            "network": "SynthStrip UNet (SDT)",
            "restore": f"sdt-graded ({'gpu' if use_gpu else 'cpu'})",
            "border_mm": border, "device": device}
    return Segmentation(labels=out_img, schema=LabelSchema(names={BRAIN_LABEL: "Brain"}),
                        grid=grid, spec=None, timings=timings, provenance=prov)
