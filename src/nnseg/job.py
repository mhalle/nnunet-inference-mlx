"""Running a segmentation without tying up the caller's loop.

A whole-body run is minutes of GPU work. Called directly it freezes a Slicer panel or blocks a
server's event loop, so :class:`Job` runs it on a worker thread and hands back a handle that can
be polled, cancelled and waited on. Deliberately framework-agnostic - Qt polls it on a timer,
asyncio wraps it in ``to_thread``, a script just calls :meth:`Job.result`. Committing the toolkit
to asyncio would exclude the Qt consumer, and ``async def`` around blocking GPU work would only
move the freeze into the event loop.

Threading is enough here because the expensive work - torch, numpy, SimpleITK - releases the GIL,
so the caller's thread keeps running. Pure-Python stretches between patches still hold it briefly;
a consumer needing hard isolation should use a subprocess.

**The device is serial.** Two runs at once would corrupt each other - ``TorchModel`` mutates its
weights in place to load a fold, and models are shared through the cache - and the memory policy
would break independently of that, since ``choose_accumulate`` sizes the accumulator from *free*
memory that a concurrent run is about to take. So every run holds a per-device lock. One
mechanism, both problems.
"""
from __future__ import annotations

import threading
from collections import defaultdict

from .errors import Cancelled
from .progress import CancelToken, Progress, Reporter

_DEVICE_LOCKS: dict[str, threading.RLock] = defaultdict(threading.RLock)
_LOCKS_GUARD = threading.Lock()


def device_lock(device) -> threading.RLock:
    """The lock serializing work on one device. Reentrant, so a job may hold it while the
    pipeline it called re-acquires it on the same thread."""
    key = str(device)
    with _LOCKS_GUARD:
        return _DEVICE_LOCKS[key]


class Job:
    """A segmentation running on a worker thread.

    >>> job = seg.submit("scan.nii.gz", "total")
    >>> while not job.done:              # or a Qt timer, or `await asyncio.to_thread(job.wait)`
    ...     show(job.progress)
    >>> result = job.result()            # re-raises whatever the run raised

    ``progress`` is the latest :class:`~nnseg.progress.Progress` snapshot, or ``None`` before the
    first one. Polling it needs no locking or cross-thread signal marshalling, which is what makes
    it easy to drive from a UI timer.
    """

    def __init__(self, fn, *, device="auto", on_progress=None, name: str = "segment"):
        self.name = name
        self.cancel_token = CancelToken()
        self._on_progress = on_progress
        self._progress: Progress | None = None
        self._result = None
        self._error: BaseException | None = None
        self._finished = threading.Event()
        self._callbacks: list = []
        self._device = device
        self._reporter = Reporter(progress=self._record, cancel=self.cancel_token)
        self._thread = threading.Thread(target=self._run, args=(fn,), name=f"nnseg-{name}", daemon=True)
        self._thread.start()

    # -- what the worker does ------------------------------------------------
    def _record(self, p: Progress) -> None:
        self._progress = p                      # a single atomic rebind; safe to read from anywhere
        if self._on_progress is not None:
            try:
                self._on_progress(p)
            except Exception:                   # a UI callback must never kill the run
                pass

    def _run(self, fn) -> None:
        try:
            self._record(Progress(stage="queued", detail="waiting for the device"))
            with device_lock(self._device):     # serialize: see the module docstring
                self._reporter.stage("starting")
                self._result = fn(self._reporter)
        except BaseException as e:              # noqa: BLE001 - re-raised from result()
            self._error = e
        finally:
            self._finished.set()
            for cb in list(self._callbacks):
                try:
                    cb(self)
                except Exception:
                    pass

    # -- what the caller does ------------------------------------------------
    @property
    def progress(self) -> Progress | None:
        return self._progress

    @property
    def done(self) -> bool:
        return self._finished.is_set()

    @property
    def cancelled(self) -> bool:
        return isinstance(self._error, Cancelled)

    def cancel(self) -> None:
        """Ask the run to stop. It ends at the next patch boundary, not instantly."""
        self.cancel_token.cancel()

    def wait(self, timeout: float | None = None) -> bool:
        """Block until the run finishes. Returns False on timeout."""
        return self._finished.wait(timeout)

    def result(self, timeout: float | None = None):
        """The :class:`~nnseg.result.Segmentation`, re-raising whatever the run raised."""
        if not self._finished.wait(timeout):
            raise TimeoutError(f"{self.name} still running after {timeout}s")
        if self._error is not None:
            raise self._error
        return self._result

    def add_done_callback(self, fn) -> None:
        """Call ``fn(job)`` when the run ends - on the worker thread, so marshal to your UI."""
        self._callbacks.append(fn)
        if self.done:
            fn(self)

    def __repr__(self) -> str:
        state = "cancelled" if self.cancelled else "done" if self.done else "running"
        p = self._progress
        return f"Job({self.name!r}, {state}" + (f", {p}" if p and not self.done else "") + ")"
