"""The accumulator-placement policy: a runtime decision from the device's budget."""
import pytest
import torch

pytest.importorskip("nnseg")
from nnseg.network import ACCUMULATE, choose_accumulate, device_budget_bytes

CPU = torch.device("cpu")
MPS = torch.device("mps")
CHEST_3MM = (236, 167, 167)          # padded model grid of a chest at 3 mm


def test_forced_modes():
    on, why = choose_accumulate("host", device=MPS, K=118, shape=CHEST_3MM)
    assert on is False and "forced host" in why
    on, _ = choose_accumulate("device", device=MPS, K=118, shape=CHEST_3MM)
    assert on is True
    with pytest.raises(ValueError):
        choose_accumulate("gpu", device=MPS, K=118, shape=CHEST_3MM)


def test_cpu_is_always_host():
    on, why = choose_accumulate("auto", device=CPU, K=118, shape=CHEST_3MM)
    assert on is False and "cpu" in why


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs MPS")
def test_auto_scales_with_the_budget(monkeypatch):
    """Same volume, different budgets: the policy is about the machine, not the model.
    The host figure is pinned so the test measures the policy and not the moment it ran."""
    import nnseg.network as N
    monkeypatch.setattr(N, "host_available_bytes", lambda: int(400e9))     # roomy host: the Metal ceiling binds
    budget = N.device_budget_bytes(MPS)
    assert budget > 4e9
    # K=118 chest at 3 mm: 1.57 GB accumulator. Whether it fits depends on what else must fit.
    big, why_big = N.choose_accumulate("auto", device=MPS, K=118, shape=CHEST_3MM, activation_reserve_gb=0.5)
    small, why_small = N.choose_accumulate("auto", device=MPS, K=118, shape=CHEST_3MM,
                                           activation_reserve_gb=budget / 1e9 + 1.0)
    assert big is True and small is False, (why_big, why_small)
    # a tiny volume fits anywhere
    on, _ = N.choose_accumulate("auto", device=MPS, K=4, shape=(32, 32, 32), activation_reserve_gb=0.5)
    assert on is True
    # ... and a busy host takes the option away, whatever the model
    monkeypatch.setattr(N, "host_available_bytes", lambda: int(3.2e9))
    on, why = N.choose_accumulate("auto", device=MPS, K=4, shape=(32, 32, 32), activation_reserve_gb=0.5)
    assert on is False, why


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs MPS")
def test_measured_mode_uses_what_the_network_actually_holds(monkeypatch):
    """After the first patch the network is resident, so the budget already reflects it and only
    a transient margin is reserved - instead of a constant that is 2 GB wrong for some models."""
    import nnseg.network as N
    monkeypatch.setattr(N, "host_available_bytes", lambda: int(400e9))     # take the Metal ceiling
    monkeypatch.setattr(N, "device_working_set_bytes", lambda device: int(2.5e9))
    budget = N.device_budget_bytes(MPS)
    on, why = N.choose_accumulate("auto", device=MPS, K=25, shape=(480, 340, 340), measured=True)
    assert "measured" in why and "network holds 2.50 GB" in why
    # 2.78 GB accumulator + 0.62 GB margin against the real budget
    assert on is (2.78e9 + 0.625e9 <= budget), why
    # the unmeasured path would have demanded a further 4.5 GB on top
    on_est, why_est = N.choose_accumulate("auto", device=MPS, K=25, shape=(480, 340, 340))
    assert "unmeasured" in why_est
    assert not (on_est and not on)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs MPS")
def test_mps_budget_honors_the_watermark_env(monkeypatch):
    """The watermark caps the Metal side of the budget. The host figure is pinned high so that
    side binds - otherwise on a busy machine both values are host-limited and drift between
    calls, which says nothing about the watermark."""
    import nnseg.network as N
    monkeypatch.setattr(N, "host_available_bytes", lambda: int(400e9))
    full = N.device_budget_bytes(MPS)
    monkeypatch.setenv("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.05")
    capped = N.device_budget_bytes(MPS)
    ceiling = torch.mps.recommended_max_memory()
    assert capped <= 0.05 * ceiling
    assert capped < full


def test_host_available_is_reported_here():
    from nnseg.network import host_available_bytes
    host = host_available_bytes()
    assert host is not None and 0 < host < 10_000e9


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs MPS")
def test_mps_budget_is_host_limited_not_the_metal_ceiling(monkeypatch):
    """Unified memory: the Metal ceiling is hardware, not availability. Taking it drives the
    machine into swap (measured 2026-08-22: a 9 GB budget on a 16 GB M2 Air pushed
    kern.memorystatus_level to 10 % and tripped the bench guard)."""
    import nnseg.network as N
    ceiling = torch.mps.recommended_max_memory()
    monkeypatch.setattr(N, "host_available_bytes", lambda: int(6e9))       # a busy machine
    # only a fraction of "available" is durably ours on unified memory (taking it causes swapping)
    assert N.device_budget_bytes(MPS, host_headroom_gb=3.0) == pytest.approx(1.5e9, rel=1e-6)
    assert N.device_budget_bytes(MPS, host_headroom_gb=3.0, unified_fraction=1.0) == pytest.approx(3e9, rel=1e-6)
    monkeypatch.setattr(N, "host_available_bytes", lambda: int(400e9))     # a big workstation
    assert N.device_budget_bytes(MPS, host_headroom_gb=3.0) == pytest.approx(
        ceiling - torch.mps.driver_allocated_memory(), rel=1e-6)           # then the Metal ceiling binds
    monkeypatch.setattr(N, "host_available_bytes", lambda: int(2e9))
    assert N.device_budget_bytes(MPS, host_headroom_gb=3.0) == 0           # nothing to spare -> host


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs MPS")
def test_tight_host_declines_the_device_whatever_the_budget_says(monkeypatch):
    """A snapshot of free memory is optimistic on unified memory: taking the budget is what
    destroys it. Measured 2026-08-22 - 6.9 GB looked available, the policy took a 1.6 GB
    accumulator on device, and the machine then swapped at 4.9 GB/min."""
    import nnseg.network as N
    monkeypatch.setattr(N, "host_available_bytes", lambda: int(400e9))
    monkeypatch.setattr(N, "device_working_set_bytes", lambda device: int(1e9))
    monkeypatch.setattr(N, "host_memory_health", lambda: 20)               # kernel says memory is tight
    on, why = N.choose_accumulate("auto", device=MPS, K=4, shape=(32, 32, 32), measured=True)
    assert on is False and "already tight" in why
    monkeypatch.setattr(N, "host_memory_health", lambda: 70)               # healthy host
    on, why = N.choose_accumulate("auto", device=MPS, K=4, shape=(32, 32, 32), measured=True)
    assert on is True, why


def test_accumulate_names():
    assert ACCUMULATE == ("auto", "device", "host")
