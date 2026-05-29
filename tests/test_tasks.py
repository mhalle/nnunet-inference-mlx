"""Tests for the declarative task registry and run_named_task dispatcher.

Four concerns:

1. **TaskSpec validation** — shape ↔ data field consistency, modality /
   source allow-lists, sane error messages.
2. **JSON round-trip** — _taskspec_to_dict / _taskspec_from_dict are
   strict inverses for every shape. JSON keys are str, dict keys are
   int — the reconstruction must coerce correctly.
3. **Registry API** — register/get/unregister/list, name-collision
   handling, modality filtering.
4. **Dispatcher** — run_named_task routes to the right backend by shape,
   reuses engines via the engine_factory hook, preserves orientation and
   geometry. We inject synthetic engines through engine_factory; no
   weight loading happens here.
"""

from __future__ import annotations

import json

import mlx.nn as nn
import numpy as np
import pytest

from nnunet_inference_mlx import (
    AmbiguousTaskError, CascadeStep, InferenceEngine, ModelBundle,
    TaskSpec, UnionPart,
    get_task, list_registered_tasks, list_tasks_by_modality,
    register_task, run_named_task, unregister_task,
)
from nnunet_inference_mlx.plans import build_network_from_plans
from nnunet_inference_mlx.tasks import (
    _BUILTIN_LOADED, _REGISTRY, _taskspec_from_dict, _taskspec_to_dict,
)

sitk = pytest.importorskip("SimpleITK")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_engine(num_classes: int = 3) -> InferenceEngine:
    plans = {
        "configurations": {
            "3d_fullres": {
                "patch_size": [16, 16, 16],
                "spacing": [1.5, 1.5, 1.5],
                "normalization_schemes": ["ZScoreNormalization"],
                "pool_op_kernel_sizes": [[1, 1, 1], [2, 2, 2]],
                "conv_kernel_sizes": [[3, 3, 3], [3, 3, 3]],
                "n_conv_per_stage_encoder": [2, 2],
                "n_conv_per_stage_decoder": [1],
                "UNet_base_num_features": 4,
            }
        },
        "foreground_intensity_properties_per_channel": {},
    }
    dataset = {
        "labels": {"background": 0,
                   **{f"class_{i}": i for i in range(1, num_classes)}},
        "channel_names": {"0": "CT"},
    }
    net = build_network_from_plans(plans, "3d_fullres", 1, num_classes,
                                    deep_supervision=False)
    weights = dict(nn.utils.tree_flatten(net.parameters()))
    bundle = ModelBundle(plans=plans, dataset=dataset,
                          fold_weights=[weights], metadata={}, fold_ids=(0,))
    return InferenceEngine(bundle, verbose=False)


def _make_sitk(shape_zyx=(24, 24, 24), spacing_xyz=(1.0, 1.0, 1.0)):
    arr = np.random.randn(*shape_zyx).astype(np.float32)
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing(spacing_xyz)
    return img


@pytest.fixture(scope="module")
def engine():
    return _make_engine()


@pytest.fixture(autouse=True)
def isolate_registry():
    """Snapshot and restore the registry around each test so that
    register_task calls in one test don't leak to others.

    Triggers the lazy builtin-registry load before snapshotting so the
    snapshot includes the shipped TS entries — otherwise the first
    test that touches the registry would lock in an empty snapshot for
    every subsequent test."""
    list_registered_tasks()       # force lazy load
    saved = dict(_REGISTRY)
    yield
    _REGISTRY.clear()
    _REGISTRY.update(saved)


# ---------------------------------------------------------------------------
# TaskSpec validation
# ---------------------------------------------------------------------------


class TestTaskSpecValidation:
    def test_single_shape(self):
        spec = TaskSpec(name="t", source="ts", modality="CT",
                        shape="single", single=42)
        assert spec.single == 42

    def test_cascade_shape(self):
        spec = TaskSpec(
            name="t", source="ts", modality="CT", shape="cascade",
            cascade=(CascadeStep(weights_id=1, crop_to_classes=(1, 2)),
                     CascadeStep(weights_id=2)),
        )
        assert len(spec.cascade) == 2

    def test_label_union_shape(self):
        spec = TaskSpec(
            name="t", source="ts", modality="CT", shape="label_union",
            union=(UnionPart(weights_id=1, label_remap={1: 10}, name="a"),),
        )
        assert len(spec.union) == 1

    def test_empty_name_rejected(self):
        with pytest.raises(ValueError, match="non-empty"):
            TaskSpec(name="", source="ts", modality="CT",
                     shape="single", single=1)

    def test_bad_source(self):
        with pytest.raises(ValueError, match="source"):
            TaskSpec(name="t", source="bogus", modality="CT",
                     shape="single", single=1)

    def test_bad_modality(self):
        with pytest.raises(ValueError, match="modality"):
            TaskSpec(name="t", source="ts", modality="XR",
                     shape="single", single=1)

    def test_bad_shape(self):
        with pytest.raises(ValueError, match="shape"):
            TaskSpec(name="t", source="ts", modality="CT",
                     shape="quadruple", single=1)

    def test_multiple_shape_fields_rejected(self):
        with pytest.raises(ValueError, match="Exactly one"):
            TaskSpec(
                name="t", source="ts", modality="CT", shape="single",
                single=1, cascade=(CascadeStep(weights_id=2),
                                    CascadeStep(weights_id=3)),
            )

    def test_no_shape_field_rejected(self):
        with pytest.raises(ValueError, match="Exactly one"):
            TaskSpec(name="t", source="ts", modality="CT", shape="single")

    def test_shape_field_mismatch_rejected(self):
        """shape='cascade' but only single= is populated."""
        with pytest.raises(ValueError, match="requires the 'cascade' field"):
            TaskSpec(name="t", source="ts", modality="CT",
                     shape="cascade", single=1)

    def test_cascade_too_short(self):
        with pytest.raises(ValueError, match="at least 2 steps"):
            TaskSpec(name="t", source="ts", modality="CT", shape="cascade",
                     cascade=(CascadeStep(weights_id=1),))

    def test_label_union_empty(self):
        with pytest.raises(ValueError, match="at least 1 part"):
            TaskSpec(name="t", source="ts", modality="CT",
                     shape="label_union", union=())


# ---------------------------------------------------------------------------
# JSON round-trip
# ---------------------------------------------------------------------------


class TestJsonRoundTrip:
    """The dict produced by _taskspec_to_dict must round-trip through
    JSON and reconstruct the same TaskSpec via _taskspec_from_dict.

    Critical detail: JSON object keys are strings, so dict[int, ...]
    fields (label_remap, label_map) need explicit coercion on both
    sides. The round-trip is the strongest test we have of that
    contract.
    """

    def test_single_round_trip(self):
        spec = TaskSpec(
            name="s", source="ts", modality="CT", shape="single",
            single=42, label_map={1: "liver", 2: "spleen"},
            expected_coverage="trunk",
        )
        rt = _taskspec_from_dict(json.loads(json.dumps(_taskspec_to_dict(spec))))
        assert spec == rt

    def test_cascade_round_trip(self):
        spec = TaskSpec(
            name="c", source="ts", modality="CT", shape="cascade",
            cascade=(
                CascadeStep(weights_id=1, crop_to_classes=(2, 3),
                            dilation_mm=15.0),
                CascadeStep(weights_id=4),
            ),
        )
        rt = _taskspec_from_dict(json.loads(json.dumps(_taskspec_to_dict(spec))))
        assert spec == rt

    def test_label_union_round_trip(self):
        spec = TaskSpec(
            name="u", source="ts", modality="CT", shape="label_union",
            union=(
                UnionPart(weights_id=1, label_remap={1: 10, 2: 11},
                          name="organs"),
                UnionPart(weights_id=2, label_remap={1: 20}, name="vert"),
            ),
        )
        rt = _taskspec_from_dict(json.loads(json.dumps(_taskspec_to_dict(spec))))
        assert spec == rt

    def test_int_keys_recovered_in_label_remap(self):
        """JSON converts int keys to str; reconstruction must restore int."""
        spec = TaskSpec(
            name="u", source="user", modality="CT", shape="label_union",
            union=(UnionPart(weights_id=1, label_remap={5: 100, 6: 101},
                              name="x"),),
        )
        as_json = json.dumps(_taskspec_to_dict(spec))
        # JSON serialization: keys should be strings
        assert '"5"' in as_json or "'5'" in as_json
        rt = _taskspec_from_dict(json.loads(as_json))
        # After round-trip: keys are int again
        assert all(isinstance(k, int) for k in rt.union[0].label_remap)
        assert rt.union[0].label_remap == {5: 100, 6: 101}

    def test_optional_fields_omitted_when_default(self):
        """A clean spec should produce a clean dict — no None / 'any' /
        empty {} clutter on disk."""
        spec = TaskSpec(name="s", source="ts", modality="CT",
                        shape="single", single=1)
        d = _taskspec_to_dict(spec)
        assert "weights_url" not in d
        assert "weights_sha256" not in d
        assert "expected_coverage" not in d   # default "any"
        assert "label_map" not in d           # empty dict


# ---------------------------------------------------------------------------
# Registry API
# ---------------------------------------------------------------------------


class TestRegistryApi:
    def test_register_then_get(self):
        spec = TaskSpec(name="mytask", source="user", modality="CT",
                        shape="single", single=99)
        register_task(spec)
        assert get_task("mytask") is spec

    def test_get_unknown_raises(self):
        with pytest.raises(KeyError, match="unknown task"):
            get_task("totally_made_up")

    def test_duplicate_raises_without_overwrite(self):
        spec1 = TaskSpec(name="dup", source="user", modality="CT",
                         shape="single", single=1)
        spec2 = TaskSpec(name="dup", source="user", modality="CT",
                         shape="single", single=2)
        register_task(spec1)
        with pytest.raises(ValueError, match="already registered"):
            register_task(spec2)

    def test_overwrite_allowed_explicitly(self):
        spec1 = TaskSpec(name="dup", source="user", modality="CT",
                         shape="single", single=1)
        spec2 = TaskSpec(name="dup", source="user", modality="CT",
                         shape="single", single=2)
        register_task(spec1)
        register_task(spec2, overwrite=True)
        assert get_task("dup").single == 2

    def test_unregister(self):
        spec = TaskSpec(name="temp", source="user", modality="CT",
                        shape="single", single=1)
        register_task(spec)
        unregister_task("temp")
        with pytest.raises(KeyError):
            get_task("temp")

    def test_list_returns_sorted_qualified_names(self):
        for n in ["zeta", "alpha", "mu"]:
            register_task(TaskSpec(name=n, source="user", modality="CT",
                                    shape="single", single=1))
        names = list_registered_tasks()
        assert names == sorted(names)
        # Qualified ("source:name") form keeps the listing unambiguous.
        assert {"user:zeta", "user:alpha", "user:mu"}.issubset(names)

    def test_list_filtered_by_source(self):
        register_task(TaskSpec(name="mine", source="user", modality="CT",
                                shape="single", single=1))
        ts_only = list_registered_tasks(source="ts")
        user_only = list_registered_tasks(source="user")
        assert all(k.startswith("ts:") for k in ts_only)
        assert "user:mine" in user_only
        assert "user:mine" not in ts_only

    def test_list_by_modality(self):
        register_task(TaskSpec(name="ct1", source="user", modality="CT",
                                shape="single", single=1))
        register_task(TaskSpec(name="ct2", source="user", modality="CT",
                                shape="single", single=2))
        register_task(TaskSpec(name="mr1", source="user", modality="MR",
                                shape="single", single=3))
        ct_tasks = list_tasks_by_modality("CT")
        mr_tasks = list_tasks_by_modality("MR")
        # Returns qualified keys
        assert "user:ct1" in ct_tasks and "user:ct2" in ct_tasks
        assert "user:mr1" not in ct_tasks
        assert "user:mr1" in mr_tasks


# ---------------------------------------------------------------------------
# Cross-source name conflict handling (anticipating MOOSE)
# ---------------------------------------------------------------------------


class TestCrossSourceConflicts:
    """Two model systems may ship a task with the same bare name
    (TS's ``total`` vs a hypothetical MOOSE ``total``). The registry keys
    on ``source:name`` so both coexist; bare lookups resolve when
    unambiguous and demand qualification otherwise."""

    def test_name_with_colon_rejected(self):
        """':' is reserved as the qualifier separator and can't appear in
        a bare task name."""
        with pytest.raises(ValueError, match="must not contain ':'"):
            TaskSpec(name="ts:total", source="user", modality="CT",
                     shape="single", single=1)

    def test_qualified_name_property(self):
        spec = TaskSpec(name="organs", source="moose", modality="CT",
                        shape="single", single=1)
        assert spec.qualified_name == "moose:organs"

    def test_same_name_different_sources_coexist(self):
        """The whole point: ts:dup and user:dup don't collide."""
        ts_spec = TaskSpec(name="dup", source="ts", modality="CT",
                           shape="single", single=10)
        user_spec = TaskSpec(name="dup", source="user", modality="CT",
                             shape="single", single=20)
        register_task(ts_spec)
        register_task(user_spec)   # must NOT raise — different qualified key
        assert get_task("ts:dup").single == 10
        assert get_task("user:dup").single == 20

    def test_bare_lookup_ambiguous_raises(self):
        register_task(TaskSpec(name="dup", source="ts", modality="CT",
                               shape="single", single=10))
        register_task(TaskSpec(name="dup", source="user", modality="CT",
                               shape="single", single=20))
        with pytest.raises(AmbiguousTaskError, match="multiple sources"):
            get_task("dup")

    def test_qualified_lookup_disambiguates(self):
        register_task(TaskSpec(name="dup", source="ts", modality="CT",
                               shape="single", single=10))
        register_task(TaskSpec(name="dup", source="user", modality="CT",
                               shape="single", single=20))
        # Qualified form resolves cleanly despite the bare-name conflict.
        assert get_task("ts:dup").single == 10

    def test_unqualified_works_when_unique(self):
        """Bare lookup still works when only one source defines the name —
        the common case today (TS-only registry)."""
        register_task(TaskSpec(name="unique_task", source="user",
                               modality="CT", shape="single", single=5))
        assert get_task("unique_task").single == 5

    def test_unregister_qualified(self):
        register_task(TaskSpec(name="dup", source="ts", modality="CT",
                               shape="single", single=10))
        register_task(TaskSpec(name="dup", source="user", modality="CT",
                               shape="single", single=20))
        unregister_task("user:dup")
        # ts:dup remains; bare lookup is now unambiguous again
        assert get_task("dup").single == 10

    def test_run_named_task_ambiguous_raises(self):
        register_task(TaskSpec(name="dup", source="ts", modality="CT",
                               shape="single", single=10))
        register_task(TaskSpec(name="dup", source="user", modality="CT",
                               shape="single", single=20))
        with pytest.raises(AmbiguousTaskError):
            run_named_task("dup", _make_sitk())


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


class TestRunNamedTaskDispatch:
    """run_named_task routes to the right backend based on shape. We use
    the engine_factory injection point to bypass cached_engine_from_task
    (which would try to load real weights) and substitute synthetic
    engines. The test is then a check that:

      - the dispatcher selects the right backend for each shape
      - the weights_id from the registry is passed to engine_factory
      - the output geometry matches the input (the same correctness
        invariant predict_with_resampling and the workflows guarantee)
    """

    def test_single_dispatch(self, engine):
        register_task(TaskSpec(name="single_t", source="user", modality="CT",
                                shape="single", single=42))
        seen = []
        seg = run_named_task(
            "single_t", _make_sitk(),
            engine_factory=lambda wid: (seen.append(wid), engine)[1],
        )
        assert seen == [42]
        assert seg.GetSize() == (24, 24, 24)

    def test_cascade_dispatch(self, engine):
        register_task(TaskSpec(
            name="cascade_t", source="user", modality="CT", shape="cascade",
            cascade=(CascadeStep(weights_id=10, crop_to_classes=(1, 2),
                                  dilation_mm=5.0),
                     CascadeStep(weights_id=20)),
        ))
        seen = []
        seg = run_named_task(
            "cascade_t", _make_sitk(shape_zyx=(32, 32, 32)),
            engine_factory=lambda wid: (seen.append(wid), engine)[1],
        )
        # Both cascade stages get built (even if the crop is empty,
        # the dispatcher constructs the Stage objects up front).
        assert seen == [10, 20]
        assert seg.GetSize() == (32, 32, 32)

    def test_label_union_dispatch(self, engine):
        register_task(TaskSpec(
            name="union_t", source="user", modality="CT", shape="label_union",
            union=(
                UnionPart(weights_id=100, label_remap={1: 50, 2: 51},
                          name="organs"),
                UnionPart(weights_id=101, label_remap={1: 60}, name="ribs"),
            ),
        ))
        seen = []
        seg = run_named_task(
            "union_t", _make_sitk(),
            engine_factory=lambda wid: (seen.append(wid), engine)[1],
        )
        assert seen == [100, 101]
        assert seg.GetSize() == (24, 24, 24)
        # Output dtype should fit the unified IDs (max 60 → uint8 ample).
        assert sitk.GetArrayFromImage(seg).dtype == np.uint8

    def test_unknown_task_raises(self):
        with pytest.raises(KeyError, match="unknown task"):
            run_named_task("does_not_exist", _make_sitk())

    def test_engine_factory_is_called_per_weights_id(self, engine):
        """The dispatcher must call engine_factory once per distinct
        weights_id mentioned in the spec, in order."""
        register_task(TaskSpec(
            name="three_part", source="user", modality="CT",
            shape="label_union",
            union=tuple(
                UnionPart(weights_id=i, label_remap={1: 10 + i}, name=f"p{i}")
                for i in (7, 8, 9)
            ),
        ))
        calls = []
        run_named_task(
            "three_part", _make_sitk(),
            engine_factory=lambda wid: (calls.append(wid), engine)[1],
        )
        assert calls == [7, 8, 9]


# ---------------------------------------------------------------------------
# Builtin registry load
# ---------------------------------------------------------------------------


class TestBuiltinRegistry:
    """The shipped data/ts_tasks.json is generated by
    scripts/refresh_ts_registry.py against an installed TS. These tests
    sanity-check the data without requiring TS at test time — they exercise
    the schema, the dispatch mechanics, and the breadth of TS coverage.
    """

    def test_builtin_registry_loadable(self):
        """The shipped data/ts_tasks.json must be valid JSON parseable by
        our TaskSpec validator."""
        list_registered_tasks()   # Triggers _load_builtin_registry
        # If the JSON file is malformed, the above call would have raised.

    def test_ts_tasks_present(self):
        """The generator must produce a non-trivial registry. Hard count
        is intentional: drops in this number indicate either a generator
        regression or a TS release we should investigate."""
        # As of TS 2.13.0, generator emits 50 specs. Reasonable bound
        # for any near-future TS version — alert on drop, not on growth.
        n = len(list_registered_tasks())
        assert n >= 40, f"only {n} tasks registered; generator may be broken"

    def test_canonical_ts_tasks_resolve(self):
        """The popular TS tasks (the ones users hit first) must all be
        registered. Catches generator skipping or rename regressions."""
        for name in ["total", "total_fast", "total_fastest", "total_mr",
                     "body", "lung_vessels", "liver_segments",
                     "appendicular_bones", "tissue_types"]:
            spec = get_task(name)
            assert spec.source == "ts", f"{name} should have source='ts'"

    def test_total_is_label_union_with_5_parts(self):
        """The flagship TS task structurally is a 5-part label union."""
        spec = get_task("total")
        assert spec.shape == "label_union"
        assert spec.modality == "CT"
        assert len(spec.union) == 5
        # The expected dataset IDs for v2.13's CT total decomposition.
        assert {p.weights_id for p in spec.union} == {291, 292, 293, 294, 295}

    def test_total_fast_is_single_model(self):
        spec = get_task("total_fast")
        assert spec.shape == "single"
        assert spec.single == 297      # TS v2.13 dataset ID for 3mm total

    def test_lung_vessels_is_cascade(self):
        spec = get_task("lung_vessels")
        assert spec.shape == "cascade"
        assert len(spec.cascade) == 2
        # Stage 1 should be a rough total model (298 default / 297 robust)
        assert spec.cascade[0].weights_id in (297, 298)
        # Stage 2 is the focused vessel model
        assert spec.cascade[1].weights_id == 117      # TS v2.13

    # NOTE: "is the committed ts_tasks.json in sync with the generator?"
    # is deliberately NOT a pytest. It requires provisioning TotalSegmentator
    # (heavy torch stack) which we keep out of the test environment. That
    # check lives in the admin CLI instead:
    #     uv run scripts/refresh_ts_registry.py check
    # (to be wired into CI later). The tests above validate the committed
    # fixture's schema and content against our own code, which is the part
    # that belongs in the unit suite.
