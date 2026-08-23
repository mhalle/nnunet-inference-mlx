"""A configured segmenter: execution policy set once, models kept warm between calls.

``nnseg.segment(image, task)`` is the right shape for a script - everything decided per call,
nothing retained. A Slicer module, an MCP tool or a REST endpoint wants the opposite: the
device, weights root and dtype are properties of the *deployment*, fixed at startup, and
reloading a checkpoint on every request is the difference between a service and a toy.

:class:`Segmenter` holds that policy plus a :class:`~nnseg.store.ModelStore`, and exposes the
same operation as a method. Per-call arguments still win, so a caller can override anything for
one request without building a second Segmenter.
"""
from __future__ import annotations

from .store import ModelStore
from .tasks import TaskCatalog

# Set once per deployment, and overridable per call.
POLICY = ("device", "dtype", "model_root", "folds", "accumulate", "batch_size",
          "resampling_order", "envelope_mm", "convention", "interp", "grid", "configuration")


class Segmenter:
    """Segment with a fixed execution policy and warm models.

    >>> seg = Segmenter(device="cuda", model_root="/weights", cache_models=2)
    >>> r = seg.segment("scan.nii.gz", "total_fast")
    >>> r.mask("liver").sum()

    ``cache_models`` is how many models stay resident. Keep it small: warm weights compete for
    the same device memory the sliding-window accumulator needs (see :mod:`nnseg.store`).
    """

    def __init__(self, *, device: str = "auto", dtype: str = "fp16", model_root=None,
                 catalog=None, folds=(0,), accumulate: str = "auto", batch_size="auto",
                 resampling_order: int = 3, envelope_mm: float | None = 20.0,
                 convention: str = "auto", interp: str = "linear", grid="input",
                 configuration: str | None = None, cache_models: int = 1):
        self.catalog = catalog if catalog is not None else TaskCatalog("totalsegmentator")
        self.models = ModelStore(capacity=cache_models)
        self.policy = dict(device=device, dtype=dtype, model_root=model_root, folds=folds,
                           accumulate=accumulate, batch_size=batch_size,
                           resampling_order=resampling_order, envelope_mm=envelope_mm,
                           convention=convention, interp=interp, grid=grid,
                           configuration=configuration)

    # -- the operation ------------------------------------------------------
    def segment(self, image, task, **overrides):
        """Segment ``image`` with ``task``; any policy argument may be overridden for this call."""
        from .pipeline import segment
        unknown = set(overrides) - set(POLICY) - {"progress", "outside"}
        if unknown:
            raise TypeError(f"unknown argument(s) {sorted(unknown)}; policy is {sorted(POLICY)}")
        kw = {**self.policy, **overrides}
        return segment(image, task, catalog=self.catalog, models=self.models, **kw)

    __call__ = segment

    # -- introspection: what can this thing do, before it does it -----------
    def tasks(self) -> list[str]:
        """Every task name in the catalog."""
        return self.catalog.names()

    def describe(self, task) -> dict:
        """What a task is and what it needs, without running or downloading anything."""
        from .pipeline import _resolve_spec
        spec = _resolve_spec(task, self.catalog)
        return {"name": spec.name, "source": spec.source, "modality": spec.modality,
                "shape": spec.shape, "n_structures": len(spec.label_map),
                "structures": [spec.label_map[k] for k in sorted(spec.label_map)],
                "weights": [str(w) for w in spec.weights_ids]}

    def structures(self, task) -> list[str]:
        """The structure names a task produces, in label order."""
        return self.describe(task)["structures"]

    # -- warm models --------------------------------------------------------
    def warm(self, task) -> int:
        """Load this task's models now, so the first real call does not pay for it.

        Returns how many are resident afterwards. Only useful with ``cache_models >= 1``.
        """
        from .pipeline import _resolve_spec
        from .tasks import resolve_model_folder
        spec = _resolve_spec(task, self.catalog)
        native = spec.source == "nnunet"
        for wid in spec.weights_ids:
            folder = resolve_model_folder(wid, ecosystem="nnunet" if native else "totalsegmentator",
                                          model_root=self.policy["model_root"],
                                          configuration=self.policy["configuration"])
            self.models.get(folder, folds=self.policy["folds"], device=self.policy["device"],
                            dtype=self.policy["dtype"], accumulate=self.policy["accumulate"],
                            batch_size=self.policy["batch_size"])
        return len(self.models)

    def clear(self) -> None:
        """Drop every warm model and free the device memory."""
        self.models.clear()

    def __repr__(self) -> str:
        return (f"Segmenter(device={self.policy['device']!r}, dtype={self.policy['dtype']!r}, "
                f"{len(self.catalog)} tasks, {self.models!r})")
