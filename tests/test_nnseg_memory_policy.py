"""The accumulator-placement policy: a runtime decision from the device's budget."""
import pytest
import torch

pytest.importorskip("labelgrid")
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
def test_auto_scales_with_the_budget():
    """Same volume, different budgets: the policy is about the machine, not the model."""
    budget = device_budget_bytes(MPS)
    assert budget and budget > 1e9
    # K=118 chest at 3 mm: 1.6 GB accumulator. Whether it fits depends on the device.
    big, why_big = choose_accumulate("auto", device=MPS, K=118, shape=CHEST_3MM, activation_reserve_gb=0.5)
    small, why_small = choose_accumulate("auto", device=MPS, K=118, shape=CHEST_3MM, activation_reserve_gb=budget / 1e9)
    assert big is True and small is False, (why_big, why_small)
    # a tiny volume fits anywhere
    on, _ = choose_accumulate("auto", device=MPS, K=4, shape=(32, 32, 32), activation_reserve_gb=0.5)
    assert on is True


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs MPS")
def test_mps_budget_honors_the_watermark_env(monkeypatch):
    full = device_budget_bytes(MPS)
    monkeypatch.setenv("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.05")
    capped = device_budget_bytes(MPS)
    ceiling = torch.mps.recommended_max_memory()
    assert capped <= 0.05 * ceiling and capped < full


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
    assert N.device_budget_bytes(MPS, host_headroom_gb=3.0) == pytest.approx(3e9, rel=1e-6)
    monkeypatch.setattr(N, "host_available_bytes", lambda: int(400e9))     # a big workstation
    assert N.device_budget_bytes(MPS, host_headroom_gb=3.0) == pytest.approx(
        ceiling - torch.mps.driver_allocated_memory(), rel=1e-6)           # then the Metal ceiling binds
    monkeypatch.setattr(N, "host_available_bytes", lambda: int(2e9))
    assert N.device_budget_bytes(MPS, host_headroom_gb=3.0) == 0           # nothing to spare -> host


def test_accumulate_names():
    assert ACCUMULATE == ("auto", "device", "host")
