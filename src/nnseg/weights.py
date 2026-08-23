"""Where model weights come from.

A caller should say *which* model they want, not where its files happen to sit. ``model_root``
threaded through the pipeline made the filesystem layout part of the API, and split one concern
across three places: an env-var/default root, a folder resolver, and a downloader.

:class:`WeightsStore` is that one concern. It answers "give me a loadable model folder for this
weights id", fetching on a miss when it knows how. Where it fetches *from* is the store's
business - a directory, a mounted cloud volume, a release download, or a subclass that pulls
from somewhere else entirely.

One thing the abstraction cannot hide: nnU-Net's loader reads ``plans.json``, ``dataset.json``
and ``fold_*/checkpoint_final.pth`` off a real filesystem, so every store must ultimately
materialize into a local directory. ``resolve()`` therefore always returns a local ``Path`` -
the same contract as ``hf_hub_download``. A remote store is a local cache with a fetch policy.
"""
from __future__ import annotations

from pathlib import Path

from .errors import ModelNotFound
from .tasks import ECOSYSTEMS, resolve_model_folder, weights_root


class WeightsStore:
    """Resolves weights ids to local model folders, downloading them if it can.

    >>> store = WeightsStore("/weights")                 # a directory (or a cloud volume mount)
    >>> store.resolve(297)                               # -> .../Dataset297_.../nnUNetTrainer__...
    >>> WeightsStore(ecosystem="nnunet").resolve(500)     # no download source; must be present

    ``fetch=False`` makes the store read-only: a missing model raises instead of reaching for
    the network, which is what you want in a sandbox, an air-gapped install, or a test.
    Subclass and override :meth:`fetch` to pull from somewhere other than the ecosystem's own
    release assets.
    """

    def __init__(self, root=None, *, ecosystem: str = "totalsegmentator", fetch: bool = True,
                 progress=None):
        self.ecosystem = ecosystem
        self._root = Path(root).expanduser() if root is not None else None
        self.fetch_enabled = bool(fetch)
        self.progress = progress

    @property
    def root(self) -> Path:
        """The local directory holding (or receiving) the weights."""
        if self._root is None:
            self._root = weights_root(self.ecosystem, None)
        return self._root

    # -- the interface a caller uses ----------------------------------------
    def have(self, weights_id) -> bool:
        """Is this model already local? Never touches the network."""
        p = Path(str(weights_id)).expanduser()
        if p.is_dir():
            return True
        return bool(sorted(self.root.glob(f"Dataset{weights_id}_*")))

    def resolve(self, weights_id, *, configuration: str | None = None) -> Path:
        """A loadable ``trainer__plans__config`` folder, fetched first if it is missing.

        A path to a model (or dataset) folder passes straight through, so a caller can always
        point at something on disk without involving a store's layout at all.
        """
        if Path(str(weights_id)).expanduser().is_dir():
            return resolve_model_folder(weights_id, configuration=configuration)
        if not self.have(weights_id):
            if not self.fetch_enabled:
                raise ModelNotFound(
                    f"weights {weights_id} are not in {self.root} and this store has fetching "
                    f"disabled; download them or pass fetch=True")
            self.fetch(weights_id)
        return resolve_model_folder(weights_id, ecosystem=self.ecosystem, model_root=self.root,
                                    configuration=configuration)

    def ensure(self, task, *, catalog=None) -> list[Path]:
        """Make every model a task needs local, recursing through cascade crop-from tasks."""
        from .weights_fetch import ensure_task_weights
        if self.ecosystem != "totalsegmentator":
            raise ModelNotFound(
                f"no download source for ecosystem {self.ecosystem!r}; place the weights under "
                f"{self.root} yourself, or subclass WeightsStore.fetch")
        return ensure_task_weights(task, self.root, catalog=catalog, progress=self.progress)

    # -- the override point for a different source --------------------------
    def fetch(self, weights_id) -> Path:
        """Bring one model into :attr:`root`. Override this for a non-default source."""
        from .weights_fetch import fetch_one
        if self.ecosystem != "totalsegmentator":
            raise ModelNotFound(
                f"weights {weights_id} not found under {self.root}, and nnseg has no download "
                f"source for ecosystem {self.ecosystem!r}. Put the model folder there, pass a "
                f"path directly, or subclass WeightsStore.fetch.")
        return fetch_one(weights_id, self.root, progress=self.progress)

    # -- provenance ---------------------------------------------------------
    def describe(self) -> dict:
        return {"kind": type(self).__name__, "ecosystem": self.ecosystem,
                "root": str(self._root) if self._root is not None else "(default)",
                "fetch": self.fetch_enabled}

    def __repr__(self) -> str:
        root = str(self._root) if self._root is not None else "default"
        return f"{type(self).__name__}({root!r}, ecosystem={self.ecosystem!r}, fetch={self.fetch_enabled})"


def as_store(weights, *, ecosystem: str = "totalsegmentator") -> WeightsStore:
    """Coerce what a caller passed into a store: a store, a path, or ``None`` for the default."""
    if isinstance(weights, WeightsStore):
        return weights
    return WeightsStore(weights, ecosystem=ecosystem)


__all__ = ["WeightsStore", "as_store", "ECOSYSTEMS"]
