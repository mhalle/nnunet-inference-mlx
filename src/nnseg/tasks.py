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
from typing import Mapping

WeightsId = int | str

# Where each ecosystem keeps its weights: (env vars, default root). The layout under the root
# is nnU-Net's own: ``Dataset<id>_<name>/<trainer>__<plans>__<config>/``.
ECOSYSTEMS = {
    "totalsegmentator": (("TOTALSEG_WEIGHTS_PATH", "nnUNet_results"),
                         Path("~/.totalsegmentator/nnunet/results")),
    "nnunet": (("nnUNet_results",), None),
}


@dataclass(frozen=True)
class UnionPart:
    """One model of a label-union task, and how its local classes map to global labels."""

    weights_id: WeightsId
    label_remap: Mapping[int, int]
    name: str = ""


@dataclass(frozen=True)
class TaskSpec:
    name: str
    source: str = "ts"
    modality: str = "CT"
    shape: str = "single"
    single: WeightsId | None = None
    union: tuple[UnionPart, ...] = ()
    cascade: tuple = ()
    label_map: Mapping[int, str] = field(default_factory=dict)

    @property
    def parts(self) -> list[tuple[WeightsId, Mapping[int, int] | None, str]]:
        """``(weights id, local->global remap or None, part name)`` in paint order."""
        if self.single is not None:
            return [(self.single, None, self.name)]
        if self.union:
            return [(p.weights_id, dict(p.label_remap), p.name or str(p.weights_id)) for p in self.union]
        raise NotImplementedError(
            f"task {self.name!r} is a {self.shape!r} task; nnseg runs single and label_union tasks so far")


class TaskCatalog:
    """The named tasks of an ecosystem, from its registry JSON."""

    def __init__(self, ecosystem: str = "totalsegmentator", path: str | Path | None = None):
        self.ecosystem = ecosystem
        self._specs: dict[str, TaskSpec] = {}
        self._load(Path(path) if path else self._builtin(ecosystem))

    @staticmethod
    def _builtin(ecosystem: str) -> Path:
        here = Path(__file__).parent / "data"
        name = {"totalsegmentator": "ts_tasks.json", "ts": "ts_tasks.json"}.get(ecosystem)
        if name is None:
            raise ValueError(f"no built-in task registry for ecosystem {ecosystem!r}")
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
            self._specs[d["name"]] = TaskSpec(
                name=d["name"], source=d.get("source", "ts"), modality=d.get("modality", "CT"),
                shape=d.get("shape", "single"), single=d.get("single"), union=union,
                cascade=tuple(d.get("cascade") or ()),
                label_map={int(k): str(v) for k, v in (d.get("label_map") or {}).items()})

    def get(self, name: str) -> TaskSpec:
        """By name, or by the source-qualified form the MLX toolkit also accepts (``ts:total``)."""
        if name in self._specs:
            return self._specs[name]
        if ":" in name:
            source, _, bare = name.partition(":")
            spec = self._specs.get(bare)
            if spec is not None and spec.source == source:
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


def weights_root(ecosystem: str = "totalsegmentator", explicit=None) -> Path:
    """Explicit argument, then environment, then the ecosystem's default location."""
    if explicit is not None:
        return Path(explicit).expanduser()
    env_vars, default = ECOSYSTEMS.get(ecosystem, ((), None))
    for var in env_vars:
        v = os.environ.get(var)
        if v:
            return Path(v).expanduser()
    if default is None:
        raise FileNotFoundError(f"no weights root for {ecosystem!r}; pass model_root or set {env_vars}")
    return default.expanduser()


def resolve_model_folder(weights_id: WeightsId, *, ecosystem: str = "totalsegmentator", model_root=None) -> Path:
    """``Dataset<id>_*`` under the weights root -> its ``trainer__plans__config`` folder."""
    root = weights_root(ecosystem, model_root)
    matches = sorted(root.glob(f"Dataset{weights_id}_*")) or sorted(root.glob(str(weights_id)))
    if not matches:
        raise FileNotFoundError(f"no Dataset{weights_id}_* under {root}")
    configs = sorted(p for p in matches[0].iterdir()
                     if p.is_dir() and not p.name.startswith(".") and p.name.count("__") == 2)
    if not configs:
        raise FileNotFoundError(f"no trainer__plans__config folder in {matches[0]}")
    return configs[0]
