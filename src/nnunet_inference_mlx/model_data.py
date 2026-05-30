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

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

import mlx.core as mx

from .values import LabelSchema


@dataclass(frozen=True, eq=False)
class ModelData:
    """Parsed model data with no GPU state.

    Parameters
    ----------
    plans, dataset :
        Parsed ``plans.json`` / ``dataset.json``.
    fold_weights :
        One ``{param_name: mx.array}`` dict per cross-validation fold.
    metadata :
        The checkpoint metadata read from disk (``init_args``,
        ``inference_allowed_mirroring_axes``, …). Carried so the built model
        resolves the right configuration and mirroring axes from the real
        training-time settings rather than a guess. Empty for hand-built data.
    ecosystem, id, version :
        Where it came from — ecosystem (``"totalsegmentator"`` / ``"moose"``
        / ``"local"`` / …), the id within that ecosystem (dataset id or
        folder name), and an optional version string. For display/logging.
    configuration :
        Which entry of ``plans["configurations"]`` to use. ``None`` → resolve
        to ``"3d_fullres"`` if present, else the first.
    """

    plans: Mapping
    dataset: Mapping
    fold_weights: tuple[Mapping[str, mx.array], ...]
    metadata: Mapping = field(default_factory=dict)
    ecosystem: str = "local"
    id: int | str = ""
    version: str | None = None
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
        ecosystem: str = "local",
        id: int | str | None = None,
        version: str | None = None,
    ) -> "ModelData":
        """Read a model config folder into :class:`ModelData`.

        Accepts either the trainer/config folder
        (``.../nnUNetTrainer__nnUNetPlans__3d_fullres``) or the dataset folder
        (``.../Dataset297_...``). Reads ``plans.json`` / ``dataset.json`` and the
        per-fold checkpoint weights + metadata directly — no engine/bundle.

        ``folds``: ``int`` (single), iterable of ints, or ``"all"`` (auto-detect
        every ``fold_*`` subdir). ``dtype`` casts weights on load.
        """
        from .weights import discover_folds, load_checkpoint_with_metadata

        folder = Path(folder).expanduser()
        # Accept a dataset dir: descend to the single trainer/config subfolder.
        if not (folder / "plans.json").exists():
            trainer_dirs = sorted(folder.glob("*__*__*"))
            if trainer_dirs:
                folder = trainer_dirs[0]

        plans = json.loads((folder / "plans.json").read_text())
        dataset = json.loads((folder / "dataset.json").read_text())

        if isinstance(folds, str):
            if folds != "all":
                raise ValueError(f"folds= must be int, iterable, or 'all'; got {folds!r}")
            fold_ids = discover_folds(folder)
            if not fold_ids:
                raise FileNotFoundError(f"No fold_* subdirs in {folder}")
        elif isinstance(folds, int):
            fold_ids = (folds,)
        else:
            fold_ids = tuple(int(f) for f in folds)
            if not fold_ids:
                raise ValueError("folds= must contain at least one fold ID.")

        fold_weights: list[Mapping[str, mx.array]] = []
        metadata: Mapping = {}
        for i, f in enumerate(fold_ids):
            w, meta = load_checkpoint_with_metadata(folder, fold=f, dtype=dtype)
            fold_weights.append(w)
            if i == 0:
                metadata = meta

        return ModelData(
            plans=plans,
            dataset=dataset,
            fold_weights=tuple(fold_weights),
            metadata=metadata,
            ecosystem=ecosystem,
            id=id if id is not None else str(folder),
            version=version,
        )


__all__ = ["ModelData"]
