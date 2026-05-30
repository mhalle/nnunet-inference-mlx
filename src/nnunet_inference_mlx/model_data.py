"""ModelData — the IO/compute seam.

``ModelData`` is the *fully-read, zero-GPU* representation of a model: parsed
plans + dataset + per-fold weight dicts (already MLX arrays) + provenance.
Reading produces it; :func:`build.build_engine` consumes it. It holds no
compiled network and allocates no Metal state — it's cheap data you can
construct, inspect, and pass between layers. (It is the model in its *data*
form, between "downloaded files on disk" and "loaded engine in memory".)

Derived accessors mirror what the network builder and preprocessing need
(patch size, target spacing with the transpose applied, label schema, input
channel count), so consumers never re-parse ``plans``/``dataset`` by hand.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import mlx.core as mx

from .values import LabelSchema


@dataclass(frozen=True)
class Provenance:
    """Where an artifact came from — for display, logging, and cache identity."""

    ecosystem: str            # "nnunet" | "totalsegmentator" | "moose" | "local" | ...
    id: int | str             # dataset id / folder name / path
    version: str | None = None


@dataclass(frozen=True, eq=False)
class ModelData:
    """Parsed model data with no GPU state.

    Parameters
    ----------
    plans, dataset :
        Parsed ``plans.json`` / ``dataset.json``.
    fold_weights :
        One ``{param_name: mx.array}`` dict per cross-validation fold.
    provenance :
        Where it came from.
    configuration :
        Which entry of ``plans["configurations"]`` to use. ``None`` → resolve
        to ``"3d_fullres"`` if present, else the first.
    """

    plans: Mapping
    dataset: Mapping
    fold_weights: tuple[Mapping[str, mx.array], ...]
    provenance: Provenance
    configuration: str | None = None

    # ----- label schema -----
    @property
    def schema(self) -> LabelSchema:
        return LabelSchema.from_dataset_json(self.dataset)

    # ----- configuration resolution -----
    @property
    def config_name(self) -> str:
        configs = self.plans.get("configurations", {})
        if self.configuration is not None:
            return self.configuration
        if "3d_fullres" in configs:
            return "3d_fullres"
        if configs:
            return next(iter(configs))
        raise KeyError("plans.json has no configurations")

    @property
    def _config(self) -> Mapping:
        return self.plans["configurations"][self.config_name]

    # ----- axis permutation -----
    @property
    def transpose_forward(self) -> tuple[int, int, int]:
        t = self.plans.get("transpose_forward", (0, 1, 2))
        return tuple(int(x) for x in t)

    @property
    def transpose_backward(self) -> tuple[int, int, int]:
        t = self.plans.get("transpose_backward", (0, 1, 2))
        return tuple(int(x) for x in t)

    # ----- spacing / patch -----
    @property
    def target_spacing_zyx(self) -> tuple[float, float, float]:
        """Training spacing in canonical (Z, Y, X), with transpose undone.

        plans.json stores spacing in the model's transposed axis order; we
        expose it in canonical order so it lines up with caller arrays.
        """
        spacing = tuple(float(s) for s in self._config["spacing"])
        tb = self.transpose_backward
        if tb == (0, 1, 2):
            return spacing
        return tuple(spacing[i] for i in tb)

    @property
    def patch_size_zyx(self) -> tuple[int, int, int]:
        return tuple(int(p) for p in self._config["patch_size"])

    @property
    def normalization_schemes(self) -> tuple[str, ...]:
        return tuple(self._config.get("normalization_schemes", ["CTNormalization"]))

    # ----- channels / folds -----
    @property
    def num_input_channels(self) -> int:
        ch = self.dataset.get("channel_names", self.dataset.get("modality", {}))
        return len(ch) if ch else 1

    @property
    def num_folds(self) -> int:
        return len(self.fold_weights)

    @property
    def num_outputs(self) -> int:
        return self.schema.num_outputs

    # ----- estimated resident footprint (MB) -----
    @property
    def weights_mb(self) -> float:
        """Approximate size of one fold's weights in MB (for cache budgeting)."""
        if not self.fold_weights:
            return 0.0
        total = sum(int(a.size) * a.dtype.size for a in self.fold_weights[0].values())
        return total / (1024 * 1024)

    # ----- construction from disk -----
    @staticmethod
    def read_folder(
        folder: str | Path,
        *,
        folds: int | "list[int]" | str = "all",
        dtype: str | None = None,
        provenance: Provenance | None = None,
    ) -> "ModelData":
        """Read a model config folder (``{trainer}__{plans}__{config}``) into an
        artifact.

        During migration this delegates to the proven folder reader; at cutover
        the orchestration moves here and the old reader is deleted.
        """
        from .engine import ModelBundle

        folder = Path(folder)
        bundle = ModelBundle.from_folder(folder, folds=folds, dtype=dtype)
        return ModelData(
            plans=bundle.plans,
            dataset=bundle.dataset,
            fold_weights=tuple(bundle.fold_weights),
            provenance=provenance or Provenance(ecosystem="local", id=str(folder)),
        )


__all__ = ["ModelData", "Provenance"]
