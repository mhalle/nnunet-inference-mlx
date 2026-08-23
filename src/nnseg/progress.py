"""Progress reporting and cooperative cancellation.

A caller driving a UI needs two things a plain function call cannot give: to know how far along
the work is, and to stop it. Both have to be *cooperative* - you cannot interrupt a blocking
torch call from outside - so the run checks a token and emits a snapshot at the one granularity
that already exists in the pipeline: between sliding-window patches.

:class:`Progress` is a frozen snapshot, so it is safe to hand to another thread and safe to
stash for a UI to poll. It prints readably, so an existing ``progress=lambda m: print(m)``
callback keeps working.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass

from .errors import Cancelled

# Rough share of a run spent before/after the network, used only to make the fraction move
# sensibly during load and preprocess rather than sitting at 0 until the first patch.
_PRE = 0.05


@dataclass(frozen=True)
class Progress:
    """Where a run has got to. Immutable, so it crosses threads safely."""

    stage: str                       # loading | preprocess | predict | restore | finalize | queued
    detail: str = ""
    part: int = 0                    # 0-based index of the current model part
    n_parts: int = 1
    step: int = 0                    # patches done within this part
    n_steps: int = 0
    fraction: float = 0.0            # 0..1 over the whole run, best effort
    elapsed: float = 0.0

    def __str__(self) -> str:
        pct = f"{self.fraction * 100:3.0f}%"
        where = f" [{self.part + 1}/{self.n_parts}]" if self.n_parts > 1 else ""
        steps = f" {self.step}/{self.n_steps}" if self.n_steps else ""
        return f"{pct} {self.stage}{where}{steps} {self.detail}".rstrip()


class CancelToken:
    """A flag one thread sets and the running job polls. Idempotent and thread-safe."""

    def __init__(self):
        self._event = threading.Event()

    def cancel(self) -> None:
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def check(self) -> None:
        """Raise :class:`~nnseg.errors.Cancelled` if the token has fired."""
        if self._event.is_set():
            raise Cancelled("segmentation cancelled")

    def __repr__(self) -> str:
        return f"CancelToken({'cancelled' if self.cancelled else 'active'})"


class Reporter:
    """Threads progress and cancellation through one run.

    Deliberately *not* stored on a :class:`~nnseg.network.TorchModel`: models are shared through
    the cache, and per-run state on a shared object is how two jobs corrupt each other. A
    Reporter belongs to the call, and is passed down.
    """

    def __init__(self, progress=None, cancel: CancelToken | None = None, n_parts: int = 1):
        self._cb = progress
        self.cancel = cancel
        self.n_parts = max(1, int(n_parts))
        self.part = 0
        self.stage_name = "starting"
        self.t0 = time.perf_counter()
        self.last: Progress | None = None

    # -- the run calls these ------------------------------------------------
    def check(self) -> None:
        if self.cancel is not None:
            self.cancel.check()

    def enter_part(self, index: int, name: str = "") -> None:
        self.part = int(index)
        self.stage("loading", name)

    def stage(self, name: str, detail: str = "") -> None:
        self.stage_name = name
        self._emit(detail=detail, step=0, n_steps=0, within=_PRE if name != "restore" else 0.95)

    def tick(self, step: int, n_steps: int, detail: str = "") -> None:
        """One patch done. Checks cancellation first - this is the interrupt point."""
        self.check()
        within = _PRE + (0.9 * step / n_steps if n_steps else 0.0)
        self._emit(detail=detail, step=step, n_steps=n_steps, within=within)

    # -- plumbing -----------------------------------------------------------
    def _emit(self, *, detail: str, step: int, n_steps: int, within: float) -> None:
        frac = (self.part + min(max(within, 0.0), 1.0)) / self.n_parts
        p = Progress(stage=self.stage_name, detail=detail, part=self.part, n_parts=self.n_parts,
                     step=step, n_steps=n_steps, fraction=min(max(frac, 0.0), 1.0),
                     elapsed=time.perf_counter() - self.t0)
        self.last = p
        if self._cb is not None:
            self._cb(p)

    @staticmethod
    def of(progress=None, cancel=None, n_parts: int = 1) -> "Reporter":
        """Accept a Reporter, a callback, or nothing, and always get a Reporter."""
        if isinstance(progress, Reporter):
            return progress
        return Reporter(progress=progress, cancel=cancel, n_parts=n_parts)
