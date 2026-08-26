"""SynthStrip brain extraction (skull-strip) as an nnseg engine.

SynthStrip is a contrast-agnostic learned brain-mask generator (a small 3D UNet,
Hoopes et al. 2022) - a different algorithm family from nnU-Net and FastSurfer.

**The model + conform now live in the standalone ``synthstrip-torch`` package**
(mirroring how FastSurfer's CNN lives in ``fastsurfer-lean``): it owns the surfa
trained-conform and the net, and returns the signed distance transform (SDT). This
module is the thin nnseg ADAPTER: it keeps the restore geometry (physical-space
``grid_sample``, shared with FastSurfer), the mask cleanup, and the wrapping into an
nnseg :class:`~nnseg.result.Segmentation`, plus the result-cache weights identity.

Flow: ``synthstrip_torch.predict_sdt`` (surfa conform 1 mm/LIA + net -> conformed
SDT handed back as a SimpleITK image) -> restore the *graded* SDT to the input grid
-> threshold at ``border`` mm + largest filled component -> a 1-label brain mask.
"""
from __future__ import annotations

import os

import numpy as np

WEIGHTS_ID = "synthstrip"
WEIGHTS_VERSION = "v1"                       # synthstrip.1.pt (MGH, 2022-04-28)
BRAIN_LABEL = 1


def weights_installed() -> list[dict]:
    """The engine's weights identity for the result-cache key - one source of
    truth for both the API-side describe and the worker re-key (same contract as
    :func:`nnseg.engines.fastsurfer.weights_installed`)."""
    return [{"id": WEIGHTS_ID, "version": WEIGHTS_VERSION}]


def _get_model(device: str):
    """The cached SynthStrip net for ``device`` (built + weights loaded once per
    device by ``synthstrip_torch``). ``NNSEG_SYNTHSTRIP_MODEL`` overrides the
    weights path; otherwise the package fetches + caches ``synthstrip.1.pt``."""
    import synthstrip_torch

    return synthstrip_torch.load_model(
        path=os.environ.get("NNSEG_SYNTHSTRIP_MODEL"), device=device)


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


def _capture_sdt(t1_img, device: str):
    """SynthStrip's conform + net via ``synthstrip_torch`` -> ``(sdt_zyx, sdt_sitk)``:
    the SDT array ``(z,y,x)`` and its geometry as a SimpleITK image (so the restore
    needs no manual axis conversion)."""
    import synthstrip_torch

    return synthstrip_torch.predict_sdt(t1_img, model=_get_model(device), device=device)


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
    sdt_zyx, conf_orig = _capture_sdt(t1_img, device)     # surfa conform + net (1-channel SDT)
    timings["capture"] = time.perf_counter() - _t

    _t = time.perf_counter()
    if use_gpu:
        sdt_native = restore_sdt_gpu(sdt_zyx, conf_orig, t1_img, device)
    else:
        sdt_native = restore_sdt_cpu(sdt_zyx, conf_orig, t1_img)
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
