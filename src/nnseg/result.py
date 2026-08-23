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

    ``image`` is a SimpleITK image on the requested grid, in the *input's* orientation - the
    thing you save. ``array`` is the same data as numpy (Z, Y, X). ``provenance`` records the
    models, folds, device and preprocessing policy the run actually used; for a medical toolkit
    that is not decoration, it is what makes a result reproducible and reviewable.
    """

    image: object                       # SimpleITK.Image
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
        return sitk.GetArrayFromImage(self.image)

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
        """Write the labels anywhere SimpleITK writes (``.nii.gz``, ``.nrrd``, ``.mha``, ...)."""
        import SimpleITK as sitk
        p = Path(path).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        sitk.WriteImage(self.image, str(p))
        return p

    @property
    def seconds(self) -> float:
        return float(self.timings.get("total", 0.0))

    def __repr__(self) -> str:
        n = len(self.present())
        task = getattr(self.spec, "name", "?")
        return (f"Segmentation(task={task!r}, {n}/{len(self.schema.names)} structures present, "
                f"grid={tuple(self.grid.shape)}, {self.seconds:.1f}s)")
