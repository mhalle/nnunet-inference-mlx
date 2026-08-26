"""FastSurfer whole-brain parcellation as an nnseg engine.

FastSurferVINN is a 2.5D view-aggregation network, not an nnU-Net model, so it
is an *engine* (a different runner) rather than an ecosystem entry. This module
produces a :class:`nnseg.result.Segmentation` on the input's grid so it flows
through the same cache / preview / statistics / client path as an nnU-Net run.

The value nnseg adds over FastSurfer's own output: FastSurfer argmaxes at its
conformed 1 mm grid and only nearest-neighbor reorients the labelmap back to the
input orientation - it never restores to the input *grid*, and never at logit
grade. Here we capture the pre-argmax logit field, resample **it** to the input
grid (physical-space, so oblique acquisitions are handled), and argmax after -
sub-voxel boundary placement instead of a blocky label resample. Proven on Modal
2026-08-26 (1 mm self-check reproduces FastSurfer's own labels exactly; the
graded restore de-stairsteps when upsampling).

FastSurfer itself is imported lazily inside :func:`segment` - importing nnseg,
or this module, never requires FastSurfer to be installed. The restore geometry
(:func:`restore_logits`) is dependency-light and unit-tested on its own.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

_LUT_PATH = Path(__file__).resolve().parent.parent / "data" / "fastsurfer_lut.json"

WEIGHTS_ID = "fastsurfer"
WEIGHTS_VERSION = "vinn-v2"


def weights_installed() -> list[dict]:
    """The engine's weights identity for the result-cache key.

    FastSurfer bakes its checkpoints into the worker image (not nnseg's weights
    volume), so this is a fixed version string, not read from an install
    sidecar. It is the ONE source of truth: the API-side describe
    (:meth:`ecosystems.FastSurferEcosystem.info`) and the worker-side re-key
    (``modal_app._FastSurferShim``) must both return this, or the key the worker
    stores a finished result under and the key a plain GET computes diverge -
    and every bare read then 404s a result that is actually cached. (Bug found
    2026-08-26: worker keyed ``fastsurfer=vinn-v2``, API keyed ``unknown``.)"""
    return [{"id": WEIGHTS_ID, "version": WEIGHTS_VERSION}]


def load_lut() -> dict[int, dict]:
    """FastSurfer output labels -> {name, color}. The segment table for the
    ``.seg.nrrd`` (names) and the canonical FreeSurfer colors."""
    raw = json.loads(_LUT_PATH.read_text())
    return {int(k): v for k, v in raw.items()}


def sitk_to_nibabel(img):
    """A SimpleITK image -> an in-memory nibabel Nifti1Image, so FastSurfer's
    ``conform`` (which is nibabel-coupled, and which we deliberately do not
    reimplement) can consume SimpleITK-decoded data without a file round-trip.

    The one geometry conversion on the way in: SimpleITK is LPS with array order
    (z, y, x); nibabel is RAS with array order (i, j, k) = (x, y, z). So the
    data is transposed to (x, y, z) and the affine is built from the direction/
    spacing/origin with the first two axes negated (LPS -> RAS). Round-trip
    tested."""
    import nibabel as nib
    import SimpleITK as sitk

    arr = sitk.GetArrayFromImage(img)                       # (z, y, x)
    data = np.ascontiguousarray(np.transpose(arr, (2, 1, 0)))   # (x, y, z)
    sp = np.asarray(img.GetSpacing(), dtype=np.float64)     # (sx, sy, sz)
    D = np.asarray(img.GetDirection(), dtype=np.float64).reshape(3, 3)
    aff = np.eye(4)
    aff[:3, :3] = D * sp[np.newaxis, :]                     # columns scaled by spacing (LPS)
    aff[:3, 3] = np.asarray(img.GetOrigin(), dtype=np.float64)
    aff = np.diag([-1.0, -1.0, 1.0, 1.0]) @ aff             # LPS -> RAS
    return nib.Nifti1Image(data, aff)


def nibabel_to_sitk(nb):
    """The inverse of :func:`sitk_to_nibabel`: an in-memory nibabel image back to
    a SimpleITK image. Used to recover the conformed-orig geometry from the
    conformed nibabel image FastSurfer produces, so the logit restore's source
    grid needs no file round-trip. Assumes an orthogonal affine (direction *
    spacing, no shear) - true for FastSurfer's conformed output and for medical
    acquisitions. Round-trip tested against sitk_to_nibabel."""
    import SimpleITK as sitk

    data = np.asanyarray(nb.dataobj)                        # (x, y, z)
    arr = np.ascontiguousarray(np.transpose(data, (2, 1, 0)))   # (z, y, x) for sitk
    aff = np.diag([-1.0, -1.0, 1.0, 1.0]) @ np.asarray(nb.affine, dtype=np.float64)  # RAS -> LPS
    M = aff[:3, :3]
    sp = np.linalg.norm(M, axis=0)                          # column norms = spacing
    D = M / sp[np.newaxis, :]                               # unit columns = direction cosines
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing(tuple(float(s) for s in sp))
    img.SetOrigin(tuple(float(o) for o in aff[:3, 3]))
    img.SetDirection(tuple(float(d) for d in D.flatten()))
    return img


def restore_logits(logit_zyx, source_ref, target_ref):
    """Resample a per-class logit field from ``source_ref``'s grid onto
    ``target_ref``'s grid (SimpleITK physical space, so any orientation/spacing
    difference is handled), and argmax over classes -> a class-index volume on
    the target grid.

    ``logit_zyx`` is ``(Z, Y, X, K)`` (SimpleITK array order) aligned with
    ``source_ref`` (a SimpleITK image carrying the source geometry).
    ``target_ref`` is a SimpleITK image whose grid is the desired output.

    Streams one channel at a time with a running argmax, so peak memory is one
    source channel + one target channel + the O(target) accumulators - never the
    full resampled K-channel volume. This is the only interpolation in the
    output path; the coordinate change (LIA -> input orientation) is exact and
    lives in the caller's capture.
    """
    import SimpleITK as sitk

    K = logit_zyx.shape[3]
    tgt_size = target_ref.GetSize()
    tgt_shape = (tgt_size[2], tgt_size[1], tgt_size[0])   # sitk size is (x,y,z)
    best = np.full(tgt_shape, -np.inf, dtype=np.float32)
    idx = np.zeros(tgt_shape, dtype=np.int32)
    for k in range(K):
        ch = sitk.GetImageFromArray(np.ascontiguousarray(logit_zyx[..., k]))
        ch.CopyInformation(source_ref)
        native = sitk.GetArrayFromImage(
            sitk.Resample(ch, target_ref, sitk.Transform(),
                          sitk.sitkLinear, 0.0, sitk.sitkFloat32))
        up = native > best
        best[up] = native[up]
        idx[up] = k
    return idx


def _resample_affine(source_ref, target_ref):
    """The 3x3 matrix A and offset t that map a TARGET voxel index (x,y,z) to the
    continuous SOURCE voxel index (x,y,z), composing SimpleITK's physical-space
    transforms of both grids (identity transform between them, like
    ``restore_logits``). physical = origin + direction @ (spacing * index); the
    source direction is orthonormal so its inverse is its transpose."""
    O_s = np.asarray(source_ref.GetOrigin(), dtype=np.float64)
    O_t = np.asarray(target_ref.GetOrigin(), dtype=np.float64)
    D_s = np.asarray(source_ref.GetDirection(), dtype=np.float64).reshape(3, 3)
    D_t = np.asarray(target_ref.GetDirection(), dtype=np.float64).reshape(3, 3)
    S_s = np.asarray(source_ref.GetSpacing(), dtype=np.float64)
    S_t = np.asarray(target_ref.GetSpacing(), dtype=np.float64)
    inv_s = np.diag(1.0 / S_s) @ D_s.T
    A = inv_s @ D_t @ np.diag(S_t)
    t = inv_s @ (O_t - O_s)
    return A, t


def restore_logits_gpu(logits_in, source_ref, target_ref, device="cuda"):
    """GPU equivalent of :func:`restore_logits`: trilinear-resample the whole
    K-channel logit field from ``source_ref``'s grid onto ``target_ref``'s grid
    and argmax over classes, in one batched ``grid_sample`` on ``device`` instead
    of 79 CPU SimpleITK resamples. Uses the FULL physical-space affine (via
    :func:`_resample_affine`, composing both grids' direction cosines), so
    orientation, spacing AND oblique rotation are handled uniformly; same
    half-pixel (voxel-center) convention as SimpleITK (``align_corners=False``,
    zero padding outside).

    ``logits_in`` is either a torch tensor already ``(K, Zs, Ys, Xs)`` on a device
    (the on-GPU path - no host<->device copy) or a numpy ``(Zs, Ys, Xs, K)`` field
    (moved to ``device`` here). Needs the field resident on the device (~5 GB fp32
    at 256^3 x 79); the CPU :func:`restore_logits` is the memory-frugal fallback.
    Returns a ``(Z, Y, X)`` int32 class-index volume (target array order)."""
    import torch
    import torch.nn.functional as F

    dev = torch.device(device)
    if isinstance(logits_in, torch.Tensor):              # already (K,Zs,Ys,Xs) on device
        K, Zs, Ys, Xs = (int(s) for s in logits_in.shape)
        logits = logits_in.to(dev).float()[None]         # (1,K,Zs,Ys,Xs) fp32
    else:                                                # numpy (Zs,Ys,Xs,K)
        Zs, Ys, Xs, K = logits_in.shape
        logits = (torch.from_numpy(np.ascontiguousarray(logits_in))
                  .permute(3, 0, 1, 2).contiguous()[None].to(dev, torch.float32))
    tgt = target_ref.GetSize()                            # (Xt, Yt, Zt)
    Xt, Yt, Zt = int(tgt[0]), int(tgt[1]), int(tgt[2])
    A, t = _resample_affine(source_ref, target_ref)

    # target voxel indices (x,y,z) for every output voxel, array order (z,y,x)
    zz, yy, xx = torch.meshgrid(torch.arange(Zt), torch.arange(Yt), torch.arange(Xt),
                                indexing="ij")
    idx_t = torch.stack([xx, yy, zz], dim=-1).to(dev, torch.float64)   # (Zt,Yt,Xt,3)
    A_t = torch.as_tensor(A, device=dev, dtype=torch.float64)
    off = torch.as_tensor(t, device=dev, dtype=torch.float64)
    src = idx_t @ A_t.T + off                             # continuous source index (x,y,z)
    # -> normalized [-1,1], voxel-center convention (align_corners=False)
    N = torch.as_tensor([Xs, Ys, Zs], device=dev, dtype=torch.float64)
    grid = ((src + 0.5) * 2.0 / N - 1.0).to(torch.float32)[None]        # (1,Zt,Yt,Xt,3)

    out = F.grid_sample(logits, grid, mode="bilinear",
                        padding_mode="zeros", align_corners=False)      # (1,K,Zt,Yt,Xt)
    idx = out.argmax(dim=1)[0].to(torch.int32).cpu().numpy()            # (Zt,Yt,Xt)
    return idx


_RUNNERS: dict = {}          # (device, batch_size) -> RunModelOnData, cached across jobs


def _get_runner(device: str, batch_size: int):
    """A FastSurfer ``RunModelOnData`` (the three view models + LUT), built ONCE
    per (device, batch_size) and reused across jobs. This is the expensive,
    input-independent setup - checkpoint load, arch build, device upload - so
    caching it turns per-job model reload (dominant in the warm case) into a
    one-time cost. Defaults (checkpoint/config/LUT paths, conform knobs) are
    taken from FastSurfer's own argument parser so we track upstream, not a
    hardcoded copy. FastSurfer is imported here and nowhere else."""
    key = (device, int(batch_size))
    runner = _RUNNERS.get(key)
    if runner is not None:
        return runner
    from FastSurferCNN import run_prediction as rp
    from FastSurferCNN.utils.checkpoint import (
        get_checkpoints, get_config_file, load_checkpoint_config_defaults)

    args = rp.make_parser().parse_args(
        ["--t1", "x", "--sd", "x", "--device", device,
         "--batch_size", str(int(batch_size)), "--viewagg_device", "auto"])
    cfg_file = get_config_file("FastSurferCNN")
    get_checkpoints(args.ckpt_ax, args.ckpt_cor, args.ckpt_sag,   # no-op once downloaded
                    urls=load_checkpoint_config_defaults("url", filename=cfg_file))
    # Mirror main()'s constructor EXACTLY, every knob from the parsed args - the
    # init defaults are NOT the CLI defaults (e.g. image_size init=True but CLI
    # "auto"), and a wrong conform knob yields a degenerate segmentation.
    runner = rp.RunModelOnData(
        lut=args.lut, ckpt_ax=args.ckpt_ax, ckpt_sag=args.ckpt_sag,
        ckpt_cor=args.ckpt_cor, cfg_ax=args.cfg_ax, cfg_sag=args.cfg_sag,
        cfg_cor=args.cfg_cor, device=args.device, viewagg_device=args.viewagg_device,
        threads=args.threads, batch_size=args.batch_size, vox_size=args.vox_size,
        orientation=args.orientation, image_size=args.image_size,
        async_io=args.async_io, conform_to_1mm_threshold=args.conform_to_1mm_threshold)
    _RUNNERS[key] = runner
    return runner


def _capture_logits(t1_sitk, device: str, batch_size: int = 8, on_gpu: bool = True):
    """Segment an in-memory SimpleITK image with a CACHED FastSurfer model,
    capturing the pre-argmax logit field in the conformed-orig frame. Drives
    conform + get_prediction directly (no ``rp.main``, no SubjectList): the input
    goes through nibabel in memory, conform runs on it, and NOTHING is written -
    the conformed orig, the segfile, brainmask/aseg/CC that ``main`` would emit
    are all skipped (we only need the logits).

    We conform to LIA (the default), so the LIA inference frame IS the conformed
    frame and ``n2l.inverse`` is the identity - the orientation change is deferred
    into the restore's physical-space resample. When ``on_gpu`` and that reorder
    is identity, the K-channel field stays on the GPU (returned as a torch tensor
    ``(K, Zs, Ys, Xs)``) - no host<->device copy, no per-channel reorder. The
    fallback (``on_gpu=False`` or a non-identity reorder) returns numpy
    ``(Zs, Ys, Xs, K)`` as before. Returns
    (logits, conf_orig_sitk, fs_labels_zyx, class_labels)."""
    import torch
    from FastSurferCNN.data_loader.conform import Reorientation, conform, is_conform
    import FastSurferCNN.data_loader.data_utils as du

    r = _get_runner(device, batch_size)
    orig = sitk_to_nibabel(t1_sitk)                   # the SITK -> nibabel bridge
    orig_data = np.asanyarray(orig.dataobj)
    # conform in memory, no file writes (conform_and_save_orig minus the IO);
    # reuse FastSurfer's own conform kwargs so we match its trained-input contract
    ck = r._RunModelOnData__conform_kwargs()          # name-mangled: FastSurfer's exact knobs
    if not is_conform(orig, **r._RunModelOnData__conform_kwargs(verbose=False)):
        orig = conform(orig, **ck)
        orig_data = np.asanyarray(orig.dataobj)

    zoom = np.asarray(orig.header.get_zooms())
    n2l = Reorientation.from_target_orientation(
        orig.affine, "soft LIA", orig_data.shape, zoom)
    orig_in_lia = n2l(orig_data, order=1)
    shape = orig_in_lia.shape + (r.get_num_classes(),)
    pred_prob = torch.zeros(shape, device=r.viewagg_device,
                            dtype=torch.float16, requires_grad=False)
    for plane, model in r.models.items():
        r.set_model(plane)
        pred_prob = model.run(pred_prob, "image", orig_in_lia,
                              n2l.reorder_axes(zoom), out=pred_prob)

    inv = n2l.inverse                                 # LIA -> conformed-orig frame
    identity = inv.is_identity()                      # true when conformed to LIA (default)

    # FastSurfer's own labels, for the self-check (single channel, cheap to move)
    pred_classes = inv(torch.argmax(pred_prob, 3), order=0)
    pred_classes = du.map_label2aparc_aseg(pred_classes, r.labels)
    fs_labels = du.split_cortex_labels(pred_classes.cpu().numpy())      # (X,Y,Z)
    fs_labels_zyx = np.ascontiguousarray(np.transpose(fs_labels, (2, 1, 0)))
    conf_orig = nibabel_to_sitk(orig)                 # conformed geometry, no file round-trip

    if on_gpu and identity:
        # keep the field on the device; (X,Y,Z,K) -> (K,Z,Y,X) for the resampler.
        # The orientation change is left to the restore's affine (no reorder here).
        logits = pred_prob.permute(3, 2, 1, 0).contiguous()            # (K,Zs,Ys,Xs) on device
    else:
        pp = pred_prob.float().cpu().numpy()          # (X, Y, Z, K) nibabel order
        if identity:
            logit_conf = pp
        else:                                         # generic reorder (rare: non-LIA conform)
            logit_conf = np.empty(inv(pp[..., 0], order=1).shape + (pp.shape[3],), np.float32)
            for k in range(pp.shape[3]):
                logit_conf[..., k] = np.asarray(inv(pp[..., k], order=1))
        logits = np.ascontiguousarray(np.transpose(logit_conf, (2, 1, 0, 3)))   # (Z,Y,X,K)
    del pred_prob
    return logits, conf_orig, fs_labels_zyx, r.labels


def _fs_version() -> str:
    try:
        import FastSurferCNN
        return getattr(FastSurferCNN, "__version__", "unknown")
    except Exception:
        return "unknown"


def segment(t1_input, *, out_dir=None, device: str = "cuda", batch_size: int = 8,
            logit_grade: bool = True, self_check: bool = True, restore: str = "auto"):
    """Segment a T1 with FastSurfer and return an :class:`nnseg.result.Segmentation`
    on the input's grid.

    ``t1_input`` is a SimpleITK image (the memory-in path - what nnseg's reader
    / read-ahead produces) or a path (read with nnseg's ``io.read_image`` so its
    IPP/affine geometry fixes apply). Either way the data reaches FastSurfer
    through nibabel in memory; nothing is written to disk (``out_dir`` is accepted
    for call-site compatibility and unused - the engine writes no temp files).

    ``logit_grade`` restores the captured logit field to the input grid and
    argmaxes after (sub-voxel boundaries); ``False`` falls back to a
    nearest-neighbor resample of FastSurfer's own labelmap. ``self_check``
    verifies (loudly) that argmax at the conformed grid reproduces FastSurfer's
    own labels before trusting the restore.

    ``restore`` selects the logit-restore backend: ``"gpu"`` (batched
    ``grid_sample``, fast, needs the whole field on the device), ``"cpu"``
    (per-channel SimpleITK, slow but memory-frugal - for limited local hosts), or
    ``"auto"`` (GPU on a CUDA device, CPU otherwise). The two are numerically
    equivalent (same physical-space mapping and half-pixel convention).
    """
    import SimpleITK as sitk
    import FastSurferCNN.data_loader.data_utils as du
    import torch

    from ..grid import Grid
    from ..result import Segmentation
    from ..values import LabelSchema

    import time

    if isinstance(t1_input, sitk.Image):
        t1_img = t1_input                             # memory-in (read-ahead / caller)
    else:
        from .. import io
        t1_img = io.read_image(str(t1_input))         # path: geometry-correct read
    use_gpu = restore == "gpu" or (restore == "auto" and str(device).startswith("cuda"))
    timings: dict[str, float] = {}
    _t = time.perf_counter()
    logits, conf_orig, fs_labels_zyx, class_labels = _capture_logits(
        t1_img, device, batch_size, on_gpu=use_gpu)
    timings["capture"] = time.perf_counter() - _t     # conform + VINN inference (model cached)

    def to_fs(idx_zyx):
        m = du.map_label2aparc_aseg(torch.from_numpy(idx_zyx.astype(np.int64)),
                                    class_labels)
        return du.split_cortex_labels(m.cpu().numpy()).astype(np.int32)

    def _source_argmax(lg):                           # argmax over classes -> (Zs,Ys,Xs)
        if isinstance(lg, torch.Tensor):              # (K,Zs,Ys,Xs) on device
            return lg.argmax(dim=0).to(torch.int32).cpu().numpy()
        return np.argmax(lg, axis=3).astype(np.int32)  # (Zs,Ys,Xs,K) numpy

    if self_check:                                    # argmax at conformed == FastSurfer's labels
        my = to_fs(_source_argmax(logits))
        agree = float((my == fs_labels_zyx).mean())
        if agree < 0.999:
            raise RuntimeError(f"FastSurfer logit self-check failed: {agree:.4%} "
                               "of conformed voxels match FastSurfer's own labels")

    def _extent(im):
        import numpy as _np
        lo = _np.array(im.TransformIndexToPhysicalPoint((0, 0, 0)))
        sz = im.GetSize()
        hi = _np.array(im.TransformIndexToPhysicalPoint((sz[0]-1, sz[1]-1, sz[2]-1)))
        return _np.minimum(lo, hi), _np.maximum(lo, hi)
    clo, chi = _extent(conf_orig); tlo, thi = _extent(t1_img)
    print(f"[fastsurfer] logits={tuple(logits.shape)} conf={conf_orig.GetSize()} "
          f"t1={t1_img.GetSize()} restore={'gpu' if use_gpu else 'cpu'} "
          f"conf_ext={clo.round(1)}..{chi.round(1)} "
          f"t1_ext={tlo.round(1)}..{thi.round(1)}", flush=True)

    if logit_grade:
        _t = time.perf_counter()
        if use_gpu:
            idx_native = restore_logits_gpu(logits, conf_orig, t1_img, device)
        else:
            idx_native = restore_logits(logits, conf_orig, t1_img)
        timings["restore"] = time.perf_counter() - _t   # physical-space resample + argmax
        labels_arr = to_fs(idx_native)                # (Z,Y,X) FreeSurfer ids on input grid
        nfg = int((labels_arr > 0).sum())
        print(f"[fastsurfer] restored foreground voxels={nfg}/{labels_arr.size}", flush=True)
        if nfg == 0:
            raise RuntimeError(
                f"logit-grade restore is empty: logits={tuple(logits.shape)}, "
                f"conf={conf_orig.GetSize()} ext {clo.round(1)}..{chi.round(1)}, "
                f"t1={t1_img.GetSize()} ext {tlo.round(1)}..{thi.round(1)}; "
                "conf/t1 physical extents likely do not overlap")
    else:
        conf_seg = sitk.GetImageFromArray(fs_labels_zyx.astype(np.uint16))
        conf_seg.CopyInformation(conf_orig)
        labels_arr = sitk.GetArrayFromImage(
            sitk.Resample(conf_seg, t1_img, sitk.Transform(),
                          sitk.sitkNearestNeighbor, 0, sitk.sitkUInt16)).astype(np.int32)

    out_img = sitk.GetImageFromArray(labels_arr.astype(np.uint16))
    out_img.CopyInformation(t1_img)                   # input grid + orientation

    lut = load_lut()
    present = sorted(int(v) for v in np.unique(labels_arr) if v)
    names = {v: lut.get(v, {}).get("name", f"label_{v}") for v in present}
    grid = Grid(shape=tuple(int(s) for s in labels_arr.shape),
                spacing=tuple(float(s) for s in reversed(t1_img.GetSpacing())),
                origin=tuple(float(o) for o in reversed(t1_img.GetOrigin())))
    prov = {"engine": "fastsurfer", "fastsurfer_version": _fs_version(),
            "network": "FastSurferVINN (2.5D view aggregation)",
            "restore": (f"logit-grade (physical-space, {'gpu' if use_gpu else 'cpu'})"
                        if logit_grade else "label-nn"),
            "self_check": "reproduces FastSurfer labels at conformed grid" if self_check else "skipped",
            "device": device}
    seg = Segmentation(labels=out_img, schema=LabelSchema(names=names),
                       grid=grid, spec=None, timings=timings, provenance=prov)
    return seg
