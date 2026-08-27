"""Task catalog: a named task -> the model(s) to run and how their labels combine.

Reads the same registry JSON the MLX toolkit ships (``ts_tasks.json``), but with no dependency
on that package - it imports mlx, which does not exist off Apple silicon. Only the parts nnseg
executes are modelled here: single-model tasks and label-union tasks. Cascades are recorded
but not runnable yet, and say so.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from .errors import ModelNotFound, UnsupportedModel
from typing import Mapping

WeightsId = int | str

# Where each weights LAYOUT keeps its models: (env vars, default root). A layout is not an
# ecosystem (a catalog) and not an engine (a runtime) - it is just which install tree the
# weights live in. The names match the vocabulary used everywhere else: ``ts`` is
# TotalSegmentator's own tree, ``nnunetv2`` is a stock nnU-Net results tree. The layout
# *under* either root is nnU-Net's own:
# ``Dataset<id>_<name>/<trainer>__<plans>__<config>/``.
# The env vars and default path are TotalSegmentator's and nnU-Net's, so they keep their
# upstream spelling.
LAYOUTS = {
    "ts": (("TOTALSEG_WEIGHTS_PATH", "nnUNet_results"),
           Path("~/.totalsegmentator/nnunet/results")),
    "nnunetv2": (("nnUNet_results",), None),
}


@dataclass(frozen=True)
class UnionPart:
    """One model of a label-union task, and how its local classes map to global labels."""

    weights_id: WeightsId
    label_remap: Mapping[int, int]
    name: str = ""


@dataclass(frozen=True)
class CascadeStep:
    """One stage of a cascade. All but the last exist to crop the next: run the model, take the
    bounding box of ``crop_to_classes`` in its output, dilate by ``dilation_mm``, and restrict
    the following stage to that box. The last stage (``crop_to_classes`` empty) is the target
    whose labels become the result.

    A stage either runs a model (``weights_id``) or reuses another task's output as the crop
    source (``crop_from_task``, e.g. teeth cropping from craniofacial_structures)."""

    weights_id: WeightsId | None = None
    crop_to_classes: tuple[int, ...] = ()
    dilation_mm: float = 10.0
    crop_from_task: str | None = None


@dataclass(frozen=True)
class TaskSpec:
    name: str
    #: Which preprocessing lineage the model was trained under - "ts"
    #: (TotalSegmentator: corner convention, no crop) or "nnunetv2" (stock
    #: nnU-Net: center convention, crop-to-nonzero). NOT the ecosystem that
    #: lists the task, and NOT the engine that runs it.
    lineage: str = "ts"
    modality: str = "CT"
    shape: str = "single"
    single: WeightsId | None = None
    union: tuple[UnionPart, ...] = ()
    cascade: tuple[CascadeStep, ...] = ()
    label_map: Mapping[int, str] = field(default_factory=dict)

    @classmethod
    def from_model_folder(cls, folder, *, name: str | None = None) -> "TaskSpec":
        """A single-model task read straight from a stock nnU-Net result folder.

        Takes ``.../Dataset<id>_<name>/<trainer>__<plans>__<config>/`` (or the dataset folder,
        resolved by :func:`resolve_model_folder`) and builds the spec from its ``dataset.json``:
        labels become the label map, ``channel_names`` the modality. This is how a caller uses
        nnseg with their own nnU-Net model, with no catalog entry anywhere.
        """
        f = resolve_model_folder(folder)
        ds = json.loads((f / "dataset.json").read_text())
        labels = ds.get("labels") or {}
        if any(isinstance(v, (list, tuple)) for v in labels.values()):
            raise UnsupportedModel(
                f"{f.name}: region-based labels (a label mapping to several values) are not "
                "supported yet - nnseg takes the argmax of a softmax head")
        chan = ds.get("channel_names") or ds.get("modality") or {"0": "unknown"}
        return cls(name=name or f.parent.name, lineage="nnunetv2",
                   modality=str(next(iter(chan.values()))),
                   shape="single", single=str(f),
                   label_map={int(v): str(k) for k, v in labels.items() if int(v) != 0})

    @property
    def parts(self) -> list[tuple[WeightsId, Mapping[int, int] | None, str]]:
        """``(weights id, local->global remap or None, part name)`` in paint order (single / union)."""
        if self.single is not None:
            return [(self.single, None, self.name)]
        if self.union:
            return [(p.weights_id, dict(p.label_remap), p.name or str(p.weights_id)) for p in self.union]
        raise NotImplementedError(
            f"task {self.name!r} is a {self.shape!r} task; use .cascade for cascade tasks")

    @property
    def weights_ids(self) -> list[WeightsId]:
        """Every model the task needs, for provisioning - single, union parts, or cascade stages."""
        if self.single is not None:
            return [self.single]
        if self.union:
            return [p.weights_id for p in self.union]
        return [st.weights_id for st in self.cascade if st.weights_id is not None]


class TaskCatalog:
    """The named tasks of an ecosystem, from its registry JSON."""

    def __init__(self, layout: str = "ts", path: str | Path | None = None):
        self.layout = layout
        self._specs: dict[str, TaskSpec] = {}
        self._load(Path(path) if path else self._builtin(layout))

    @staticmethod
    def _builtin(layout: str) -> Path:
        here = Path(__file__).parent / "data"
        name = {"ts": "ts_tasks.json"}.get(layout)
        if name is None:
            raise ValueError(f"no built-in task registry for layout {layout!r}")
        return here / name

    def _load(self, path: Path) -> None:
        raw = json.loads(Path(path).read_text())
        items = raw["tasks"] if isinstance(raw, dict) and "tasks" in raw else raw
        items = list(items.values()) if isinstance(items, dict) else items
        for d in items:
            union = tuple(UnionPart(weights_id=p["weights_id"],
                                    label_remap={int(k): int(v) for k, v in p.get("label_remap", {}).items()},
                                    name=p.get("name", ""))
                          for p in d.get("union") or ())
            cascade = tuple(CascadeStep(weights_id=st.get("weights_id"),
                                        crop_to_classes=tuple(st.get("crop_to_classes") or ()),
                                        dilation_mm=float(st.get("dilation_mm", 10.0)),
                                        crop_from_task=st.get("crop_from_task"))
                            for st in d.get("cascade") or ())
            self._specs[d["name"]] = TaskSpec(
                name=d["name"], lineage=d.get("lineage", "ts"), modality=d.get("modality", "CT"),
                shape=d.get("shape", "single"), single=d.get("single"), union=union,
                cascade=cascade,
                label_map={int(k): str(v) for k, v in (d.get("label_map") or {}).items()})

    def get(self, name) -> TaskSpec:
        """A task by name (or the lineage-qualified ``ts:total`` form); a TaskSpec passes through."""
        if isinstance(name, TaskSpec):
            return name
        if name in self._specs:
            return self._specs[name]
        if ":" in name:
            lineage, _, bare = name.partition(":")
            spec = self._specs.get(bare)
            if spec is not None and spec.lineage == lineage:
                return spec
        try:
            return self._specs[name]
        except KeyError:
            raise LookupError(f"unknown task {name!r}; {len(self._specs)} known, e.g. "
                              f"{sorted(self._specs)[:6]}") from None

    __getitem__ = get

    def __contains__(self, name: str) -> bool:
        return name in self._specs

    def names(self) -> list[str]:
        return sorted(self._specs)

    def __len__(self) -> int:
        return len(self._specs)


def weights_root(layout: str = "ts", explicit=None) -> Path:
    """Explicit argument, then environment, then the layout's default location."""
    if explicit is not None:
        return Path(explicit).expanduser()
    env_vars, default = LAYOUTS.get(layout, ((), None))
    for var in env_vars:
        v = os.environ.get(var)
        if v:
            return Path(v).expanduser()
    if default is None:
        raise ModelNotFound(f"no weights root for layout {layout!r}; pass model_root or set {env_vars}")
    return default.expanduser()


# Preference order when a dataset ships several configurations and the caller named none.
# 3d_fullres is nnU-Net's default and the only one nnseg runs today: the cascade needs a
# lowres prediction as an extra input channel, and 2d needs a slice-wise loop.
CONFIG_PREFERENCE = ("3d_fullres", "3d_lowres", "2d")
UNSUPPORTED_CONFIGS = {"3d_cascade_fullres": "needs the 3d_lowres prediction as an extra input channel"}


def _dataset_dirs(root: Path, weights_id) -> list[Path]:
    """``Dataset<id>_*`` directories, tolerating zero-padded ids (Dataset008 vs Dataset8)."""
    pats = [f"Dataset{weights_id}_*"]
    t = str(weights_id).strip()
    if t.isdigit():
        pats += [f"Dataset{int(t)}_*", f"Dataset{int(t):03d}_*"]
    seen: dict[str, Path] = {}
    for pat in pats:
        for d in sorted(root.glob(pat)):
            seen.setdefault(d.name, d)
    return list(seen.values()) or sorted(root.glob(str(weights_id)))


def resolve_model_folder(weights_id: WeightsId, *, layout: str = "ts", model_root=None,
                         configuration: str | None = None) -> Path:
    """``Dataset<id>_*`` under the weights root -> its ``trainer__plans__config`` folder.

    A model folder path passes through unchanged, so a caller can point nnseg straight at a
    stock nnU-Net result directory. When a dataset ships several configurations (a trained
    nnU-Net commonly has 2d / 3d_lowres / 3d_fullres / 3d_cascade_fullres), ``configuration``
    picks one; otherwise :data:`CONFIG_PREFERENCE` decides, rather than whichever sorts first.
    """
    p = Path(str(weights_id)).expanduser()
    if p.is_dir() and p.name.count("__") == 2:
        return p
    root = Path(p) if p.is_dir() else weights_root(layout, model_root)
    matches = ([root] if p.is_dir() else _dataset_dirs(root, weights_id))
    if not matches:
        raise ModelNotFound(f"no Dataset{weights_id}_* under {root}")
    configs = sorted(c for c in matches[0].iterdir()
                     if c.is_dir() and not c.name.startswith(".") and c.name.count("__") == 2)
    if not configs:
        raise ModelNotFound(f"no trainer__plans__config folder in {matches[0]}")
    by_config = {c.name.rsplit("__", 1)[1]: c for c in configs}
    if configuration is not None:
        if configuration not in by_config:
            raise ModelNotFound(f"configuration {configuration!r} not in {matches[0].name}; "
                                    f"have {sorted(by_config)}")
        return by_config[configuration]
    for name in CONFIG_PREFERENCE:
        if name in by_config:
            return by_config[name]
    if len(configs) == 1:
        return configs[0]
    why = "; ".join(f"{k} ({UNSUPPORTED_CONFIGS[k]})" for k in sorted(by_config) if k in UNSUPPORTED_CONFIGS)
    raise ModelNotFound(
        f"no runnable configuration in {matches[0].name}; have {sorted(by_config)}"
        + (f" - unsupported: {why}" if why else "") + ". Pass configuration=... to choose.")


def _resolve_spec(task, catalog) -> "TaskSpec":
    """A TaskSpec, a catalog name, or a path to a stock nnU-Net model folder. Lives here
    (torch-free) rather than in pipeline so `describe()` and the serve front-end can resolve
    a task without importing the inference stack (torch)."""
    if isinstance(task, TaskSpec):
        return task
    if isinstance(task, Path) or (isinstance(task, str) and Path(task).expanduser().is_dir()):
        return TaskSpec.from_model_folder(task)
    return catalog.get(task)


def _uses_nnunet_preprocessing(spec) -> bool:
    return spec.lineage == "nnunetv2"
