"""VoxTell free-text promptable segmentation as an nnseg engine.

VoxTell (MIC-DKFZ, CVPR 2026) maps free-form text - a word or a clinical sentence -
onto a 3D mask, zero-shot across CT / MR / PET. That makes it a different *kind* of
engine from the other three: there is no fixed label set, because **the prompt is an
input**. One catalog task (``voxtell:text``) covers everything a caller can ask for,
and the prompts ride in the job's ``options``, where they hash into the result-cache
key - so two different prompt lists are two different cached results, for free.

Output shape. VoxTell returns one BINARY mask per prompt, ``(num_prompts, X, Y, Z)``,
and those masks may overlap ("liver" and "liver tumor"). nnseg's Segmentation carries
a single labelmap, so we paint them in prompt order: label ``i+1`` is prompt ``i``,
the schema names it with the prompt text, and a later prompt wins where they overlap.
That keeps the whole existing path (seg.nrrd, preview, statistics, the client) working
unchanged; the overlap that is lost is recorded in the provenance.

**Axis order is the trap here, and it is not the obvious one.** VoxTell's contract is
whatever its training reader gives it, and that reader is nnU-Net's
``NibabelIOWithReorient`` - which reorients to RAS and then, in its own words,
"transpose image to be consistent with the way SimpleITk reads images. Yeah. Annoying.",
transposes to **SimpleITK's (Z, Y, X)** order. So "nibabel reader" does *not* mean
nibabel's (X, Y, Z) index order, and the array a SimpleITK reader already holds is
exactly right: reorient to RAS, hand it over, put the masks back in the caller's frame.

Transposing to (X, Y, Z) "to match nibabel" is the bug this warns about - it was written
that way first. Nothing catches it downstream: the model happily segments a transposed
volume and returns plausible, non-empty masks of the wrong anatomy, and a shape check
passes because the masks come back transposed too. Only the anatomy is wrong. Hence the
convention is documented here and asserted in the tests against the reader's source.

It stays in SimpleITK throughout: unlike the FastSurfer engine, which crosses into
nibabel because FastSurfer's ``conform`` takes a nibabel *image*, VoxTell takes a plain
numpy array - so there is nothing to bridge, and reorientation is SimpleITK's own.
"""
from __future__ import annotations

import numpy as np

ENGINE = "voxtell"

# The weights identity lives in the engine registry (one literal, read by both the
# API-side describe and the worker's re-key - see engines/fastsurfer.py).
from . import registry as _registry                                    # noqa: E402

WEIGHTS_ID = ENGINE
WEIGHTS_VERSION = _registry.ENGINES[ENGINE].weights_identity()[0]["version"]


def weights_installed() -> list[dict]:
    """The engine's weights identity for the result-cache key (from the registry)."""
    return _registry.ENGINES[ENGINE].weights_identity()


_PREDICTORS: dict = {}          # device -> VoxTellPredictor, cached across jobs


def _get_predictor(device: str):
    """A ``VoxTellPredictor`` built once per device and reused across jobs - the
    expensive, input-independent setup (checkpoint load, network build, device
    upload, and the precomputed embedding bank). The predictor also lazily loads
    the Qwen3 text backbone the first time a prompt is not in the bank; pointing
    ``HF_HOME`` at the persistent weights volume is what keeps that a
    once-ever download rather than once per cold container."""
    import torch

    key = str(device)
    p = _PREDICTORS.get(key)
    if p is not None:
        return p
    from voxtell.inference.predictor import VoxTellPredictor

    # model_dir=None -> $VOXTELL_MODEL, else the default model from the HF hub.
    p = VoxTellPredictor(device=torch.device(device))
    _PREDICTORS[key] = p
    return p


def normalize_prompts(prompts) -> list[str]:
    """Accept a string or a list of strings; return a clean list.

    Raises :class:`~nnseg.errors.InputError` on an empty or malformed prompt list -
    at the engine boundary, where the message can say what the caller should have
    sent, rather than deep inside the model."""
    from ..errors import InputError

    if prompts is None:
        raise InputError(
            "voxtell needs at least one text prompt: pass options "
            '{"prompts": ["liver", "spleen"]} (a string is accepted for one prompt)')
    if isinstance(prompts, str):
        prompts = [prompts]
    if not isinstance(prompts, (list, tuple)):
        raise InputError(f"voxtell prompts must be a string or a list of strings, "
                         f"got {type(prompts).__name__}")
    out = [str(p).strip() for p in prompts if str(p).strip()]
    if not out:
        raise InputError("voxtell prompts are empty; give at least one non-blank prompt")
    return out


def to_ras(image):
    """``(array (Z, Y, X) in RAS, the RAS image, the input's orientation code)``.

    VoxTell's contract is whatever its training reader produces, and that reader is
    nnU-Net's ``NibabelIOWithReorient``, which reorients to RAS and then - per its own
    comment, "transpose image to be consistent with the way SimpleITk reads images.
    Yeah. Annoying." - transposes to **SimpleITK's (Z, Y, X)** order (spacing reversed
    to match). Their reference server says the same: ``(1, Z, Y, X), RAS``.

    So the array a SimpleITK reader already holds is the right one; all that is needed
    is the reorientation to RAS. **Do not transpose here** - an axis-order "fix" would
    feed the model a transposed volume, which still produces plausible non-empty masks
    and is therefore invisible except in the anatomy.
    """
    from .. import io as nio

    sitk = nio._sitk()
    original = nio.orientation_of(image)
    ras = image if original == "RAS" else sitk.DICOMOrient(image, "RAS")
    return np.ascontiguousarray(sitk.GetArrayFromImage(ras)), ras, original


def segment(image_input, prompts, *, out_dir=None, device: str = "cuda",
            self_check: bool = True, progress=None, cancel=None):
    """Segment whatever ``prompts`` describe and return a
    :class:`nnseg.result.Segmentation` on the input's grid.

    ``image_input`` is a SimpleITK image (memory-in) or a path. ``prompts`` is a
    string or list of strings. Label ``i+1`` is prompt ``i``, painted in order, so a
    later prompt wins where masks overlap. ``out_dir`` is accepted for call-site
    compatibility and unused (no temp files)."""
    import time

    import SimpleITK as sitk

    from .. import io as nio
    from ..errors import Cancelled, InputError
    from ..grid import Grid
    from ..progress import Reporter
    from ..result import Segmentation
    from ..values import LabelSchema

    names = normalize_prompts(prompts)
    report = Reporter.of(progress, cancel=cancel)

    if isinstance(image_input, sitk.Image):
        img = image_input
    else:
        img = nio.read_image(str(image_input))

    timings: dict[str, float] = {}
    _t = time.perf_counter()
    data_zyx, ras_img, original = to_ras(img)
    timings["read"] = time.perf_counter() - _t

    predictor = _get_predictor(device)

    def _tick(done: int, total: int) -> bool:
        # VoxTell's progress_callback returns False to stop, which is exactly our cancel
        # token - so a cancelled job stops INSIDE the model rather than running to the
        # end. Reporter.tick raises on cancel; catch it here so VoxTell unwinds its own
        # way, and re-raise below once it has returned.
        try:
            report.tick(done, total)
            return True
        except Cancelled:
            return False

    _t = time.perf_counter()
    masks = predictor.predict_single_image(data_zyx, names, progress_callback=_tick)
    timings["network"] = time.perf_counter() - _t
    tok = report.cancel
    if tok is not None and tok.cancelled:        # stopped cooperatively above
        raise Cancelled("voxtell run cancelled")

    masks = np.asarray(masks)
    if masks.ndim != 4 or masks.shape[0] != len(names):
        raise InputError(f"voxtell returned {masks.shape}, expected "
                         f"({len(names)}, Z, Y, X) for {len(names)} prompt(s)")
    if self_check and tuple(masks.shape[1:]) != tuple(data_zyx.shape):
        # The masks must land on the grid the model was given; anything else means the
        # array went over in the wrong axis order.
        raise InputError(
            f"voxtell masks are {masks.shape[1:]} but the input is {data_zyx.shape}; "
            "the array handed to the model is not in the orientation it expects")

    # Paint in prompt order on the RAS grid: label i+1 = prompt i, later wins.
    _t = time.perf_counter()
    labels_zyx = np.zeros(masks.shape[1:], dtype=np.uint16)
    overlap = 0
    for i, mask in enumerate(masks):
        m = np.asarray(mask).astype(bool)
        overlap += int(np.count_nonzero(m & (labels_zyx > 0)))
        labels_zyx[m] = i + 1
    timings["compose"] = time.perf_counter() - _t

    # Back to SimpleITK on the RAS grid we ran on, then into the caller's own frame.
    # Reorientation only permutes and flips, so the round trip is exact - no resampling.
    out = sitk.GetImageFromArray(np.ascontiguousarray(labels_zyx))
    out.CopyInformation(ras_img)
    if original != "RAS":
        out = nio.restore_orientation(out, original)
    arr = sitk.GetArrayFromImage(out)

    found = {i + 1: int(np.count_nonzero(labels_zyx == i + 1)) for i in range(len(names))}
    print(f"[voxtell] prompts={names} voxels={[found[k] for k in sorted(found)]} "
          f"overlap_overwritten={overlap}", flush=True)

    grid = Grid(shape=tuple(int(s) for s in arr.shape),
                spacing=tuple(float(s) for s in reversed(out.GetSpacing())),
                origin=tuple(float(o) for o in reversed(out.GetOrigin())))
    prov = {"engine": ENGINE, "voxtell_version": WEIGHTS_VERSION,
            "network": "VoxTell (free-text promptable)",
            "prompts": names, "device": device,
            "empty_prompts": [names[k - 1] for k, n in sorted(found.items()) if n == 0],
            # painting is lossy where prompts overlap; say so rather than hide it
            "overlap_voxels_overwritten": overlap,
            "composite": "labelmap painted in prompt order (later prompt wins)"}
    return Segmentation(labels=out,
                        schema=LabelSchema(names={i + 1: n for i, n in enumerate(names)}),
                        grid=grid, spec=None, timings=timings, provenance=prov)
