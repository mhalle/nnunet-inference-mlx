"""
Label-scheme post-processing: logits → integer segmentation.

Two schemes are supported, dispatched by inspecting ``dataset.json``:

* **Standard N-class softmax.** Each label is a single int. The model has
  one softmax output per class; conversion is plain ``argmax(axis=0)``.
* **Region-based.** Each "label" is a union of underlying classes (a list
  of ints). The model has one *independent sigmoid* head per region — they
  are not softmax-related. Conversion is "paint each region in priority
  order from ``regions_class_order``." Used by BraTS-style brain-tumor
  models and several medical-imaging-challenge datasets.

Standard:                  Region-based:
  "labels": {                "labels": {
    "background": 0,           "background": 0,
    "spleen":     1,           "WT":  [1, 2, 3],
    "kidney":     2            "TC":  [1, 3],
    ...                        "ET":  3
  }                          },
                             "regions_class_order": [2, 1, 3]

Region-based dispatch is automatic — callers don't need to inspect the
dataset.json themselves.
"""

from __future__ import annotations

import numpy as np


def has_regions(dataset_json: dict) -> bool:
    """True iff ``dataset_json["labels"]`` defines any label as a union
    of base classes (a list/tuple of ints) — the region-based scheme.

    A single int value is a plain class; a list/tuple of ints is a region.
    Mixed dicts (e.g. ``background=0`` plus regions) count as region-based.
    """
    labels = dataset_json.get("labels", {})
    return any(isinstance(v, (list, tuple)) for v in labels.values())


def regions_class_order(dataset_json: dict) -> tuple[int, ...]:
    """Region paint-priority order. Empty tuple for non-region datasets.

    Required for region-based datasets; raises ``ValueError`` if missing.
    """
    if not has_regions(dataset_json):
        return ()
    order = dataset_json.get("regions_class_order")
    if order is None:
        raise ValueError(
            "dataset_json has region-based labels but no regions_class_order; "
            "cannot deterministically convert region predictions to labels."
        )
    return tuple(int(c) for c in order)


def label_dtype(dataset_json: dict) -> np.dtype:
    """Smallest unsigned integer dtype that fits every label value the
    dataset can produce.

    Inspects both ``labels`` (region member classes and standard labels)
    and ``regions_class_order`` (the integer values actually painted at
    inference for region-based models). Returns ``uint8`` / ``uint16`` /
    ``uint32`` per nnUNetv2's convention — small enough to be efficient
    on disk and in memory, large enough to hold every label without
    silent truncation.
    """
    labels = dataset_json.get("labels", {})
    max_val = 0
    for v in labels.values():
        if isinstance(v, (list, tuple)):
            if v:
                max_val = max(max_val, max(int(x) for x in v))
        else:
            max_val = max(max_val, int(v))
    for v in dataset_json.get("regions_class_order", ()):
        max_val = max(max_val, int(v))

    if max_val <= np.iinfo(np.uint8).max:
        return np.dtype(np.uint8)
    if max_val <= np.iinfo(np.uint16).max:
        return np.dtype(np.uint16)
    return np.dtype(np.uint32)


def sigmoid_inplace(arr: np.ndarray) -> np.ndarray:
    """Apply the logistic sigmoid in place to a float32 array.

    Numerically safe for any logit range. Inputs are clipped to ``±88``
    before ``exp`` to keep float32 ``exp(-x)`` in finite range; outside
    that band the sigmoid is already ``0`` or ``1`` to float32 precision,
    so the clip is exact for the values we care about.
    """
    np.clip(arr, -88.0, 88.0, out=arr)
    np.negative(arr, out=arr)
    np.exp(arr, out=arr)
    arr += 1.0
    np.reciprocal(arr, out=arr)
    return arr


def convert_logits_to_segmentation(
    pred: np.ndarray,
    dataset_json: dict,
    threshold: float = 0.0,
    dtype: str | np.dtype | None = None,
) -> np.ndarray:
    """Convert per-channel predictions into an integer segmentation map.

    Parameters
    ----------
    pred : np.ndarray
        Shape ``(K, ...)``, float32. For standard models, ``K`` is the
        class count and these are logits or softmax probabilities (the
        result is the same after ``argmax``). For region-based models,
        ``K`` is the number of regions and these are *per-region* sigmoid
        logits or post-sigmoid probabilities.
    dataset_json : dict
        Parsed dataset.json; determines the label scheme.
    threshold : float, default 0.0
        Cut for region-based predictions. ``0.0`` matches "logit > 0",
        i.e. "sigmoid > 0.5"; pass ``0.5`` if you've already applied
        sigmoid and have probabilities. Ignored for standard datasets.
    dtype : str, np.dtype, or None
        Output integer dtype. ``None`` (default) auto-picks the smallest
        unsigned integer that fits every label value (``uint8`` / ``uint16``
        / ``uint32``) via :func:`label_dtype`. Pass an explicit dtype
        (``"uint16"``, ``np.int32``, etc.) to override — useful when a
        downstream tool requires a specific integer width regardless of
        what the dataset actually needs.

    Returns
    -------
    np.ndarray
        Shape ``pred.shape[1:]``, dtype as resolved above.
    """
    out_dtype = np.dtype(dtype) if dtype is not None else label_dtype(dataset_json)

    if not has_regions(dataset_json):
        return np.argmax(pred, axis=0).astype(out_dtype)

    order = regions_class_order(dataset_json)
    if pred.shape[0] != len(order):
        raise ValueError(
            f"Region prediction has {pred.shape[0]} channels but "
            f"regions_class_order has {len(order)} entries."
        )
    seg = np.zeros(pred.shape[1:], dtype=out_dtype)
    for region_idx, label in enumerate(order):
        seg[pred[region_idx] > threshold] = label
    return seg


__all__ = [
    "has_regions",
    "regions_class_order",
    "label_dtype",
    "convert_logits_to_segmentation",
    "sigmoid_inplace",
]
