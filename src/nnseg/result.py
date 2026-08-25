"""What :func:`nnseg.segment` hands back.

A bare ``(image, schema, timings)`` tuple makes every caller unpack three things to use one,
and cannot grow without breaking them. :class:`Segmentation` carries the same content plus the
two things consumers kept having to reconstruct: a structure-by-name accessor, and a record of
what actually ran.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Segmentation:
    """The labels, what they mean, and how they were produced.

    ``labels`` is the label volume as a SimpleITK image, on the requested grid and in the
    *input's* orientation - the thing you save. It is deliberately not called ``image``: in this
    domain that word means the intensity volume that went *in*, which is what ``segment()``'s
    first parameter is. ``array`` is the same label data as numpy (Z, Y, X). ``provenance``
    records the models, folds, device and preprocessing policy the run actually used; for a
    medical toolkit that is not decoration - it is what makes a result reproducible.
    """

    labels: object                      # SimpleITK.Image of the label volume
    schema: object                      # nnseg.values.LabelSchema
    grid: object                        # nnseg.grid.Grid - the grid the labels live on
    spec: object                        # nnseg.tasks.TaskSpec - what was asked for
    timings: dict = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)

    # -- data ---------------------------------------------------------------
    @property
    def array(self) -> np.ndarray:
        """The label volume as numpy (Z, Y, X). A fresh array each call - SimpleITK owns the buffer."""
        import SimpleITK as sitk
        return sitk.GetArrayFromImage(self.labels)

    @property
    def names(self) -> dict[int, str]:
        """``{label value: structure name}``, background excluded."""
        return dict(self.schema.names)

    def label_of(self, name: str) -> int:
        """The integer label for a structure name."""
        for value, n in self.schema.names.items():
            if n == name:
                return int(value)
        raise KeyError(f"no structure named {name!r}; have {sorted(self.schema.names.values())[:8]}...")

    def mask(self, which) -> np.ndarray:
        """A boolean (Z, Y, X) mask for one structure, by name or label value.

        The common case - a caller wants *the liver*, not a 117-label volume they then have to
        look the id up in.
        """
        value = self.label_of(which) if isinstance(which, str) else int(which)
        return self.array == value

    def present(self) -> dict[int, str]:
        """The structures actually found in this volume (nonzero voxel count)."""
        found = set(int(v) for v in np.unique(self.array)) - {0}
        return {v: n for v, n in sorted(self.schema.names.items()) if int(v) in found}

    def volumes_ml(self) -> dict[str, float]:
        """Physical volume per present structure, in millilitres."""
        arr = self.array
        per_voxel = float(np.prod(self.grid.spacing)) / 1000.0
        counts = np.bincount(arr.reshape(-1))
        return {n: float(counts[v]) * per_voxel
                for v, n in sorted(self.schema.names.items())
                if int(v) < counts.size and counts[int(v)]}

    # -- output -------------------------------------------------------------
    def save(self, path) -> Path:
        """Write the labels anywhere SimpleITK writes (``.nii.gz``, ``.nrrd``, ``.mha``, ...).

        ``.nrrd`` / ``.seg.nrrd`` outputs carry the segmentation's *meaning* in the
        header - per-segment names, label values, extents and colors in 3D Slicer's
        ``.seg.nrrd`` conventions (the file loads as a named segmentation there),
        plus the full provenance as an ``nnseg_provenance`` key - one self-contained
        artifact. NIfTI has nowhere to put any of that (an 80-character ``descrip``),
        which is why it is the lossy option here, not the default: the service
        stores and serves ``.seg.nrrd`` (decided 2026-08-24).
        """
        import SimpleITK as sitk
        p = Path(path).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        img = self.labels
        if p.name.endswith((".nrrd", ".seg.nrrd")):
            img = sitk.Image(self.labels)          # shallow copy: metadata only
            for k, v in self._seg_nrrd_metadata().items():
                img.SetMetaData(k, v)
        sitk.WriteImage(img, str(p), True)
        return p

    def _seg_nrrd_metadata(self) -> dict:
        """Slicer ``.seg.nrrd`` keys for every present structure + nnseg provenance."""
        import colorsys
        import json
        arr = self.array
        try:                                        # one-pass bounding boxes
            from scipy import ndimage
            objects = ndimage.find_objects(arr)
        except ImportError:
            objects = None
        md = {
            "Segmentation_MasterRepresentation": "Binary labelmap",
            "Segmentation_ContainedRepresentationNames": "Binary labelmap|",
            "Segmentation_ReferenceImageExtentOffset": "0 0 0",
            "nnseg_provenance": json.dumps(self.provenance),
        }
        present = sorted(int(v) for v in np.unique(arr) if v != 0)
        for i, value in enumerate(present):
            name = self.schema.names.get(value, f"label_{value}")
            h = (value * 0.61803398875) % 1.0
            r, g, b = colorsys.hsv_to_rgb(h, 0.55 + 0.35 * ((value * 7) % 5) / 4.0,
                                          0.70 + 0.30 * ((value * 3) % 4) / 3.0)
            md[f"Segment{i}_ID"] = f"Segment_{value}"
            md[f"Segment{i}_Name"] = name
            md[f"Segment{i}_NameAutoGenerated"] = "0"
            md[f"Segment{i}_LabelValue"] = str(value)
            md[f"Segment{i}_Layer"] = "0"
            md[f"Segment{i}_Color"] = f"{r:.3f} {g:.3f} {b:.3f}"
            md[f"Segment{i}_ColorAutoGenerated"] = "1"
            sl = objects[value - 1] if objects is not None and value - 1 < len(objects) else None
            if sl is not None:
                z, y, x = sl                        # array is (Z, Y, X); extents are IJK = x y z
                md[f"Segment{i}_Extent"] = (f"{x.start} {x.stop - 1} {y.start} "
                                            f"{y.stop - 1} {z.start} {z.stop - 1}")
        return md

    @property
    def seconds(self) -> float:
        return float(self.timings.get("total", 0.0))

    def __repr__(self) -> str:
        n = len(self.present())
        task = getattr(self.spec, "name", "?")
        return (f"Segmentation(task={task!r}, {n}/{len(self.schema.names)} structures present, "
                f"grid={tuple(self.grid.shape)}, {self.seconds:.1f}s)")
