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


def _capture_logits(t1_sitk, out_dir: str, device: str):
    """Run FastSurfer seg-only on an in-memory SimpleITK image, capturing the
    pre-argmax logit field in the conformed-orig frame. The input reaches
    FastSurfer through nibabel in memory (no file decode): ``du.load_image`` -
    FastSurfer's one file-read seam - is patched to return the SimpleITK image
    converted to nibabel, so FastSurfer's own ``conform`` runs on it directly.
    Returns (logit_zyx, conf_orig_sitk, fs_labels_zyx, class_labels).
    FastSurfer is imported here and nowhere else."""
    import os

    import nibabel as nib
    import SimpleITK as sitk
    import torch
    from FastSurferCNN import run_prediction as rp
    from FastSurferCNN.data_loader.conform import Reorientation
    import FastSurferCNN.data_loader.data_utils as du

    nib_img = sitk_to_nibabel(t1_sitk)                # the SITK -> nibabel bridge
    orig_load_image = du.load_image

    def load_image_from_memory(file, name="image", **kw):
        return nib_img, np.asanyarray(nib_img.dataobj)

    du.load_image = load_image_from_memory
    # a tiny valid placeholder so SubjectList's existence check passes; its
    # content is never read (load_image is patched to the in-memory image)
    ph = Path(out_dir) / "placeholder.nii.gz"
    ph.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(np.zeros((2, 2, 2), np.uint8), np.eye(4)), str(ph))

    cap: dict = {}

    def capturing_get_prediction(self, image_name, orig_data, zoom, affine):
        _zoom = np.asarray(zoom)
        n2l = Reorientation.from_target_orientation(
            affine, "soft LIA", orig_data.shape, _zoom)
        orig_in_lia = n2l(orig_data, order=1)
        shape = orig_in_lia.shape + (self.get_num_classes(),)
        pred_prob = torch.zeros(shape, device=self.viewagg_device,
                                dtype=torch.float16, requires_grad=False)
        for plane, model in self.models.items():
            self.set_model(plane)
            pred_prob = model.run(pred_prob, image_name, orig_in_lia,
                                  n2l.reorder_axes(_zoom), out=pred_prob)
        inv = n2l.inverse                             # exact discrete reorder LIA->conformed
        pp = pred_prob.float().cpu().numpy()          # (X, Y, Z, K) nibabel order
        conf0 = np.asarray(inv(pp[..., 0], order=1))
        logit_conf = np.empty(conf0.shape + (pp.shape[3],), np.float32)
        logit_conf[..., 0] = conf0
        for k in range(1, pp.shape[3]):
            logit_conf[..., k] = np.asarray(inv(pp[..., k], order=1))
        cap["logit_conf"] = logit_conf
        cap["labels"] = self.labels
        pred_classes = torch.argmax(pred_prob, 3)
        del pred_prob
        pred_classes = n2l.inverse(pred_classes, order=0)
        pred_classes = du.map_label2aparc_aseg(pred_classes, self.labels)
        cap["fs_labels"] = du.split_cortex_labels(pred_classes.cpu().numpy())
        return cap["fs_labels"]

    rp.RunModelOnData.get_prediction = capturing_get_prediction
    conf_path = f"{out_dir}/spike/mri/orig.nii.gz"
    parser = rp.make_parser()
    args = parser.parse_args([
        "--t1", str(ph), "--sid", "spike", "--sd", out_dir,
        "--asegdkt_segfile", f"{out_dir}/spike/mri/aparc.DKTatlas+aseg.deep.nii.gz",
        "--conformed_name", conf_path, "--device", device,
        "--batch_size", "8", "--viewagg_device", "auto"])
    try:
        rc = rp.main(**vars(args))
    finally:
        du.load_image = orig_load_image               # never leave the patch installed
    if "logit_conf" not in cap:
        raise RuntimeError(f"FastSurfer produced no logit field (rc={rc})")

    conf_orig = sitk.ReadImage(conf_path)             # correct geometry, (z,y,x)
    logit_zyx = np.ascontiguousarray(
        np.transpose(cap["logit_conf"], (2, 1, 0, 3)))  # (X,Y,Z,K) -> (Z,Y,X,K)
    fs_labels_zyx = np.ascontiguousarray(np.transpose(cap["fs_labels"], (2, 1, 0)))
    return logit_zyx, conf_orig, fs_labels_zyx, cap["labels"]


def _fs_version() -> str:
    try:
        import FastSurferCNN
        return getattr(FastSurferCNN, "__version__", "unknown")
    except Exception:
        return "unknown"


def segment(t1_input, *, out_dir, device: str = "cuda",
            logit_grade: bool = True, self_check: bool = True):
    """Segment a T1 with FastSurfer and return an :class:`nnseg.result.Segmentation`
    on the input's grid.

    ``t1_input`` is a SimpleITK image (the memory-in path - what nnseg's reader
    / read-ahead produces) or a path (read with nnseg's ``io.read_image`` so its
    IPP/affine geometry fixes apply). Either way the data reaches FastSurfer
    through nibabel in memory; no temp NIfTI of the volume is written.

    ``logit_grade`` restores the captured logit field to the input grid and
    argmaxes after (sub-voxel boundaries); ``False`` falls back to a
    nearest-neighbor resample of FastSurfer's own labelmap. ``self_check``
    verifies (loudly) that argmax at the conformed grid reproduces FastSurfer's
    own labels before trusting the restore.
    """
    import SimpleITK as sitk
    import FastSurferCNN.data_loader.data_utils as du
    import torch

    from ..grid import Grid
    from ..result import Segmentation
    from ..values import LabelSchema

    if isinstance(t1_input, sitk.Image):
        t1_img = t1_input                             # memory-in (read-ahead / caller)
    else:
        from .. import io
        t1_img = io.read_image(str(t1_input))         # path: geometry-correct read
    logit_zyx, conf_orig, fs_labels_zyx, class_labels = _capture_logits(
        t1_img, str(out_dir), device)

    def to_fs(idx_zyx):
        m = du.map_label2aparc_aseg(torch.from_numpy(idx_zyx.astype(np.int64)),
                                    class_labels)
        return du.split_cortex_labels(m.cpu().numpy()).astype(np.int32)

    if self_check:                                    # argmax at conformed == FastSurfer's labels
        my = to_fs(np.argmax(logit_zyx, axis=3))
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
    print(f"[fastsurfer] logit_zyx={logit_zyx.shape} conf={conf_orig.GetSize()} "
          f"t1={t1_img.GetSize()} conf_ext={clo.round(1)}..{chi.round(1)} "
          f"t1_ext={tlo.round(1)}..{thi.round(1)}", flush=True)

    if logit_grade:
        idx_native = restore_logits(logit_zyx, conf_orig, t1_img)
        labels_arr = to_fs(idx_native)                # (Z,Y,X) FreeSurfer ids on input grid
        nfg = int((labels_arr > 0).sum())
        print(f"[fastsurfer] restored foreground voxels={nfg}/{labels_arr.size}", flush=True)
        if nfg == 0:
            raise RuntimeError(
                f"logit-grade restore is empty: logit_zyx={logit_zyx.shape}, "
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
            "restore": "logit-grade (physical-space)" if logit_grade else "label-nn",
            "self_check": "reproduces FastSurfer labels at conformed grid" if self_check else "skipped",
            "device": device}
    seg = Segmentation(labels=out_img, schema=LabelSchema(names=names),
                       grid=grid, spec=None, timings={}, provenance=prov)
    return seg
