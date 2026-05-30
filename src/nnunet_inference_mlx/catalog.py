"""TaskCatalog — an explicit, owned catalog of named task recipes.

The catalog maps a **name** (``"total_fast"``) to a **recipe** (a frozen
:class:`TaskSpec`: single model / cascade / label-union). It is the explicit,
caller-constructed replacement for the module-global task registry — same
no-hidden-state principle as :class:`ModelStore`: you build a catalog, you
hold it, nothing is process-global.

Names are ecosystem-namespaced (``"ts:total"`` / ``"moose:total"``) so two
ecosystems can coexist. A bare name resolves when exactly one source defines
it; collisions raise :class:`AmbiguousTaskError` and must be qualified.

The recipe data classes (``TaskSpec`` / ``CascadeStep`` / ``UnionPart``) and
the JSON reader are reused from ``tasks.py`` during migration; at cutover the
module-global registry in ``tasks.py`` is deleted and this becomes the only
catalog.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .tasks import (  # reuse proven recipe types + JSON reader (no global touched)
    AmbiguousTaskError,
    TaskSpec,
    _taskspec_from_dict,
)


def _builtin_json(ecosystem: str) -> Path | None:
    """Path to a shipped registry JSON for a known ecosystem, if any."""
    here = Path(__file__).parent / "data"
    mapping = {
        "totalsegmentator": here / "ts_tasks.json",
        "ts": here / "ts_tasks.json",
    }
    return mapping.get(ecosystem)


class TaskCatalog:
    """A catalog of named task recipes. Construct from ecosystem name(s) or
    explicit JSON path(s); merge with ``|`` / :meth:`merged_with`.

    >>> catalog = TaskCatalog("totalsegmentator")
    >>> spec = catalog["total_fast"]          # bare name (unambiguous)
    >>> spec = catalog["ts:total"]            # qualified
    """

    def __init__(self, *sources: str, path: str | Path | None = None):
        self._tasks: dict[str, TaskSpec] = {}  # qualified name → recipe
        for src in sources:
            self._load_ecosystem(src)
        if path is not None:
            self._load_json(Path(path))

    # ----- loading -----
    def _load_ecosystem(self, ecosystem: str) -> None:
        jp = _builtin_json(ecosystem)
        if jp is None:
            raise ValueError(
                f"no built-in catalog for ecosystem {ecosystem!r}; "
                f"pass an explicit path=."
            )
        if jp.exists():
            self._load_json(jp)

    def _load_json(self, path: Path) -> None:
        payload = json.loads(path.read_text())
        for entry in payload.get("tasks", []):
            spec = _taskspec_from_dict(entry)
            self._tasks[spec.qualified_name] = spec

    # ----- lookup (bare resolves when unambiguous; else qualify) -----
    def get(self, name: str) -> TaskSpec:
        if ":" in name:
            if name not in self._tasks:
                raise KeyError(f"unknown task: {name!r}. "
                               f"Available: {sorted(self._tasks) or '(empty)'}")
            return self._tasks[name]
        matches = [k for k, s in self._tasks.items() if s.name == name]
        if not matches:
            raise KeyError(f"unknown task: {name!r}. "
                           f"Available: {sorted(self._tasks) or '(empty)'}")
        if len(matches) > 1:
            raise AmbiguousTaskError(
                f"task name {name!r} is defined by multiple sources: "
                f"{sorted(matches)}. Qualify it, e.g. {sorted(matches)[0]!r}."
            )
        return self._tasks[matches[0]]

    def __getitem__(self, name: str) -> TaskSpec:
        return self.get(name)

    def __contains__(self, name: str) -> bool:
        try:
            self.get(name)
            return True
        except (KeyError, AmbiguousTaskError):
            return False

    def names(self, *, source: str | None = None) -> list[str]:
        """Sorted qualified names, optionally filtered to one source."""
        return sorted(
            k for k, s in self._tasks.items()
            if source is None or s.source == source
        )

    def by_modality(self, modality: str) -> list[str]:
        return sorted(k for k, s in self._tasks.items() if s.modality == modality)

    def __len__(self) -> int:
        return len(self._tasks)

    # ----- merge -----
    def merged_with(self, other: "TaskCatalog") -> "TaskCatalog":
        """A new catalog containing both catalogs' tasks (other wins on a
        qualified-name collision)."""
        cat = TaskCatalog()
        cat._tasks = {**self._tasks, **other._tasks}
        return cat

    def __or__(self, other: "TaskCatalog") -> "TaskCatalog":
        return self.merged_with(other)

    def register(self, spec: TaskSpec, *, overwrite: bool = False) -> None:
        """Add a recipe (e.g. a user-defined task) to this catalog."""
        key = spec.qualified_name
        if key in self._tasks and not overwrite:
            raise ValueError(f"task {key!r} already in catalog; overwrite=True to replace.")
        self._tasks[key] = spec


__all__ = ["TaskCatalog", "AmbiguousTaskError"]
