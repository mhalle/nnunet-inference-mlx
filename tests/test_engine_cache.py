"""Tests for the engine_cache module.

The cache mechanics (key construction, get/set/clear, env-var gating) are
tested with sentinel objects in place of real :class:`InferenceEngine`
instances — at runtime the cache is just a dict, so these tests don't
need real weights. Integration with :func:`cached_engine_from_folder` /
:func:`cached_engine_from_task` is covered by the smoke tests with real
weights present.
"""

from __future__ import annotations

import pytest

from nnunet_inference_mlx import (
    cache_enabled,
    cache_engine,
    clear_engine_cache,
    get_cached_engine,
)
from nnunet_inference_mlx.engine_cache import _CACHE, _engine_key


@pytest.fixture(autouse=True)
def _clean_cache():
    """Ensure each test starts and ends with an empty cache."""
    clear_engine_cache()
    yield
    clear_engine_cache()


@pytest.fixture
def cache_on(monkeypatch):
    """Force the cache on for tests that need it regardless of host RAM."""
    monkeypatch.setenv("NNUNET_MLX_CACHE_ENGINES", "1")


@pytest.fixture
def cache_off(monkeypatch):
    """Force the cache off."""
    monkeypatch.setenv("NNUNET_MLX_CACHE_ENGINES", "0")


# ---------------------------------------------------------------------------
# cache_enabled() env-var gating
# ---------------------------------------------------------------------------


def test_cache_enabled_env_on(monkeypatch):
    monkeypatch.setenv("NNUNET_MLX_CACHE_ENGINES", "1")
    assert cache_enabled() is True


def test_cache_enabled_env_off(monkeypatch):
    monkeypatch.setenv("NNUNET_MLX_CACHE_ENGINES", "0")
    assert cache_enabled() is False


@pytest.mark.parametrize("falsy", ["0", "false", "False", "no", "NO", ""])
def test_cache_enabled_env_falsy(monkeypatch, falsy):
    monkeypatch.setenv("NNUNET_MLX_CACHE_ENGINES", falsy)
    assert cache_enabled() is False


@pytest.mark.parametrize("truthy", ["1", "true", "True", "yes", "anything"])
def test_cache_enabled_env_truthy(monkeypatch, truthy):
    monkeypatch.setenv("NNUNET_MLX_CACHE_ENGINES", truthy)
    assert cache_enabled() is True


# ---------------------------------------------------------------------------
# Basic get / set / clear round-trip with sentinel "engines"
# ---------------------------------------------------------------------------


def test_get_returns_none_when_missing(cache_on):
    assert get_cached_engine("missing-key") is None


def test_get_returns_none_when_cache_disabled(cache_off):
    # Even if we cache something with the disabled cache, get returns None
    cache_engine("k", "engine")
    assert get_cached_engine("k") is None


def test_set_get_roundtrip(cache_on):
    sentinel = object()
    cache_engine("my-key", sentinel)
    assert get_cached_engine("my-key") is sentinel


def test_set_is_noop_when_cache_disabled(cache_off):
    cache_engine("k", "engine")
    # Even though the env says off, peek behind the curtain to confirm
    # nothing got stored.
    assert "k" not in _CACHE


def test_clear_engine_cache_empties(cache_on):
    cache_engine("a", object())
    cache_engine("b", object())
    assert len(_CACHE) == 2
    clear_engine_cache()
    assert len(_CACHE) == 0


def test_clear_engine_cache_idempotent(cache_on):
    clear_engine_cache()
    clear_engine_cache()  # should not raise on empty cache


def test_clear_calls_engine_close_when_available(cache_on):
    """clear_engine_cache should call .close() on each cached object."""
    calls = []

    class FakeEngine:
        def close(self):
            calls.append("closed")

    cache_engine("k1", FakeEngine())
    cache_engine("k2", FakeEngine())
    clear_engine_cache()
    assert calls == ["closed", "closed"]


def test_clear_tolerates_missing_close(cache_on):
    """Closing should not crash for objects without a close() method."""
    cache_engine("k", object())
    clear_engine_cache()  # no AttributeError


def test_clear_tolerates_close_raising(cache_on):
    """If close() raises, the cache should still be cleared."""

    class BadEngine:
        def close(self):
            raise RuntimeError("boom")

    cache_engine("k", BadEngine())
    clear_engine_cache()  # swallows the exception
    assert len(_CACHE) == 0


# ---------------------------------------------------------------------------
# Cache key construction — what should/should not invalidate
# ---------------------------------------------------------------------------


def test_key_canonicalizes_int_fold():
    k_int = _engine_key("/m", None, 0, 0.5, True, None, False)
    k_tuple = _engine_key("/m", None, (0,), 0.5, True, None, False)
    assert k_int == k_tuple


def test_key_canonicalizes_iterable_folds():
    k_list = _engine_key("/m", None, [0, 1], 0.5, True, None, False)
    k_tuple = _engine_key("/m", None, (0, 1), 0.5, True, None, False)
    assert k_list == k_tuple


def test_key_preserves_string_folds():
    k = _engine_key("/m", None, "all", 0.5, True, None, False)
    # "all" is stored as-is for cache-key purposes
    assert any(part == "all" for part in k)


def test_key_distinguishes_use_mirroring():
    k_no = _engine_key("/m", None, 0, 0.5, True, None, False)
    k_yes = _engine_key("/m", None, 0, 0.5, True, None, True)
    assert k_no != k_yes


def test_key_distinguishes_step_size():
    k_a = _engine_key("/m", None, 0, 0.5, True, None, False)
    k_b = _engine_key("/m", None, 0, 0.7, True, None, False)
    assert k_a != k_b


def test_key_distinguishes_folder():
    k_a = _engine_key("/m/A", None, 0, 0.5, True, None, False)
    k_b = _engine_key("/m/B", None, 0, 0.5, True, None, False)
    assert k_a != k_b


def test_same_args_yield_same_key(cache_on):
    """Cache hit when the same call args are presented again."""
    k1 = _engine_key("/m", "3d_fullres", 0, 0.5, True, None, False)
    k2 = _engine_key("/m", "3d_fullres", (0,), 0.5, True, None, False)
    sentinel = object()
    cache_engine(k1, sentinel)
    assert get_cached_engine(k2) is sentinel
