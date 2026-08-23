"""Jobs, progress and cancellation - the properties a UI actually depends on.

Driven by fake work rather than a real network: what is under test is that the caller's thread
stays free, that cancellation is honored promptly, and that runs on one device serialize.
"""
import threading
import time

import pytest

from nnseg.errors import Cancelled
from nnseg.job import Job, device_lock
from nnseg.progress import CancelToken, Progress, Reporter


def _slow(n=20, sleep=0.005):
    """Fake work shaped like the sliding window: ticks a reporter per 'patch'."""
    def run(reporter):
        for i in range(n):
            reporter.tick(i + 1, n)
            time.sleep(sleep)
        return "finished"
    return run


# -- cancellation -----------------------------------------------------------------------
def test_cancel_token_raises_at_the_next_check():
    t = CancelToken()
    t.check()                                   # no-op while active
    t.cancel()
    assert t.cancelled
    with pytest.raises(Cancelled):
        t.check()


def test_a_reporter_tick_is_the_cancellation_point():
    t = CancelToken()
    r = Reporter(cancel=t)
    r.tick(1, 10)
    t.cancel()
    with pytest.raises(Cancelled):
        r.tick(2, 10)


def test_cancelling_a_job_stops_it_promptly_and_is_reported():
    job = Job(_slow(n=2000, sleep=0.001), device="cpu")
    while job.progress is None or job.progress.step < 2:
        time.sleep(0.005)
    job.cancel()
    assert job.wait(timeout=5), "job did not stop after cancel"
    assert job.cancelled and job.done
    with pytest.raises(Cancelled):
        job.result()
    assert job.progress.step < 2000             # stopped early, did not run to completion


def test_a_job_that_finishes_returns_its_value():
    job = Job(_slow(n=3), device="cpu")
    assert job.result(timeout=5) == "finished"
    assert job.done and not job.cancelled


def test_an_exception_is_re_raised_from_result_not_swallowed():
    def boom(reporter):
        raise ValueError("kaboom")
    job = Job(boom, device="cpu")
    with pytest.raises(ValueError, match="kaboom"):
        job.result(timeout=5)
    assert job.done


def test_result_times_out_rather_than_blocking_forever():
    job = Job(_slow(n=1000, sleep=0.01), device="cpu")
    with pytest.raises(TimeoutError):
        job.result(timeout=0.05)
    job.cancel()


# -- the point of the exercise: the caller's thread stays free ---------------------------
def test_the_caller_keeps_running_while_the_job_works():
    job = Job(_slow(n=40, sleep=0.005), device="cpu")
    spins = 0
    while not job.done:                          # a UI event loop would be doing this
        spins += 1
        time.sleep(0.001)
    assert spins > 5, "the caller was blocked instead of looping"
    assert job.result() == "finished"


def test_progress_is_pollable_without_a_callback():
    job = Job(_slow(n=30, sleep=0.003), device="cpu")
    seen = []
    while not job.done:
        p = job.progress
        if p is not None:
            seen.append(p.fraction)
        time.sleep(0.002)
    job.result()
    assert seen and max(seen) > min(seen)        # it moved


def test_a_progress_callback_that_raises_does_not_kill_the_run():
    def bad(_p):
        raise RuntimeError("a broken UI callback")
    job = Job(_slow(n=5), device="cpu", on_progress=bad)
    assert job.result(timeout=5) == "finished"


def test_done_callback_fires_and_is_immediate_if_already_finished():
    job = Job(_slow(n=2), device="cpu")
    job.result(timeout=5)
    fired = []
    job.add_done_callback(fired.append)          # already done: called at once
    assert fired == [job]


# -- the device is serial ----------------------------------------------------------------
def test_two_jobs_on_one_device_do_not_overlap():
    """TorchModel mutates its weights in place to load a fold and models are shared through the
    cache, so overlapping runs would corrupt each other - and the memory policy sizes the
    accumulator from free memory a concurrent run is about to take."""
    active = []
    overlapped = []

    def work(reporter):
        active.append(1)
        if len(active) > 1:
            overlapped.append(1)
        time.sleep(0.05)
        active.pop()
        return "ok"

    a = Job(work, device="cpu:test-serial")
    b = Job(work, device="cpu:test-serial")
    assert a.result(timeout=5) == "ok" and b.result(timeout=5) == "ok"
    assert not overlapped, "two jobs ran on the same device at once"


def test_a_queued_job_says_so():
    started = threading.Event()
    release = threading.Event()

    def hold(reporter):
        started.set()
        release.wait(timeout=5)
        return "ok"

    first = Job(hold, device="cpu:test-queued")
    started.wait(timeout=5)
    second = Job(_slow(n=1), device="cpu:test-queued")
    time.sleep(0.02)
    assert second.progress is not None and second.progress.stage == "queued"
    release.set()
    assert first.result(timeout=5) == "ok" and second.result(timeout=5) == "finished"


def test_different_devices_do_not_block_each_other():
    assert device_lock("cuda:0") is not device_lock("cuda:1")
    assert device_lock("cuda:0") is device_lock("cuda:0")


# -- Progress itself ---------------------------------------------------------------------
def test_progress_prints_readably_so_a_print_callback_still_works():
    p = Progress(stage="predict", detail="organs", part=1, n_parts=5, step=3, n_steps=10,
                 fraction=0.34)
    text = str(p)
    assert "34%" in text and "predict" in text and "[2/5]" in text and "3/10" in text


def test_fraction_advances_across_parts():
    r = Reporter(n_parts=2)
    r.enter_part(0, "a"); r.tick(10, 10)
    first = r.last.fraction
    r.enter_part(1, "b"); r.tick(10, 10)
    assert 0.0 < first < r.last.fraction <= 1.0
