"""Tests for the recipe types — TaskSpec / CascadeStep / UnionPart — and their
JSON round-trip. These survive the Phase 5 cutover (the module-global registry
and run_named_task dispatcher were deleted; lookup is now via TaskCatalog and
dispatch via segment()). Pure data: no engines, no registry, no globals.
"""

from __future__ import annotations

import json

import pytest

from nnunet_inference_mlx import CascadeStep, TaskSpec, UnionPart
from nnunet_inference_mlx.tasks import _taskspec_from_dict, _taskspec_to_dict


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
            TaskSpec(name="", source="ts", modality="CT", shape="single", single=1)

    def test_bad_source(self):
        with pytest.raises(ValueError, match="source"):
            TaskSpec(name="t", source="bogus", modality="CT", shape="single", single=1)

    def test_bad_modality(self):
        with pytest.raises(ValueError, match="modality"):
            TaskSpec(name="t", source="ts", modality="XR", shape="single", single=1)

    def test_bad_shape(self):
        with pytest.raises(ValueError, match="shape"):
            TaskSpec(name="t", source="ts", modality="CT", shape="quadruple", single=1)

    def test_multiple_shape_fields_rejected(self):
        with pytest.raises(ValueError, match="Exactly one"):
            TaskSpec(
                name="t", source="ts", modality="CT", shape="single",
                single=1, cascade=(CascadeStep(weights_id=2), CascadeStep(weights_id=3)),
            )

    def test_no_shape_field_rejected(self):
        with pytest.raises(ValueError, match="Exactly one"):
            TaskSpec(name="t", source="ts", modality="CT", shape="single")

    def test_shape_field_mismatch_rejected(self):
        with pytest.raises(ValueError, match="requires the 'cascade' field"):
            TaskSpec(name="t", source="ts", modality="CT", shape="cascade", single=1)

    def test_cascade_too_short(self):
        with pytest.raises(ValueError, match="at least 2 steps"):
            TaskSpec(name="t", source="ts", modality="CT", shape="cascade",
                     cascade=(CascadeStep(weights_id=1),))

    def test_label_union_empty(self):
        with pytest.raises(ValueError, match="at least 1 part"):
            TaskSpec(name="t", source="ts", modality="CT", shape="label_union", union=())


class TestJsonRoundTrip:
    """_taskspec_to_dict → JSON → _taskspec_from_dict must reconstruct the
    same TaskSpec. JSON stringifies int dict-keys (label_remap/label_map), so
    the round-trip is the strongest test of the int-key coercion contract."""

    def test_single_round_trip(self):
        spec = TaskSpec(name="s", source="ts", modality="CT", shape="single",
                        single=42, label_map={1: "liver", 2: "spleen"},
                        expected_coverage="trunk")
        assert _taskspec_from_dict(json.loads(json.dumps(_taskspec_to_dict(spec)))) == spec

    def test_cascade_round_trip(self):
        spec = TaskSpec(
            name="c", source="ts", modality="CT", shape="cascade",
            cascade=(CascadeStep(weights_id=1, crop_to_classes=(2, 3), dilation_mm=15.0),
                     CascadeStep(weights_id=4)),
        )
        assert _taskspec_from_dict(json.loads(json.dumps(_taskspec_to_dict(spec)))) == spec

    def test_label_union_round_trip(self):
        spec = TaskSpec(
            name="u", source="ts", modality="CT", shape="label_union",
            union=(UnionPart(weights_id=1, label_remap={1: 10, 2: 11}, name="organs"),
                   UnionPart(weights_id=2, label_remap={1: 20}, name="vert")),
        )
        assert _taskspec_from_dict(json.loads(json.dumps(_taskspec_to_dict(spec)))) == spec

    def test_int_keys_recovered_in_label_remap(self):
        spec = TaskSpec(
            name="u", source="user", modality="CT", shape="label_union",
            union=(UnionPart(weights_id=1, label_remap={5: 100, 6: 101}, name="x"),),
        )
        as_json = json.dumps(_taskspec_to_dict(spec))
        assert '"5"' in as_json
        rt = _taskspec_from_dict(json.loads(as_json))
        assert all(isinstance(k, int) for k in rt.union[0].label_remap)
        assert rt.union[0].label_remap == {5: 100, 6: 101}

    def test_optional_fields_omitted_when_default(self):
        spec = TaskSpec(name="s", source="ts", modality="CT", shape="single", single=1)
        d = _taskspec_to_dict(spec)
        assert "weights_url" not in d
        assert "weights_sha256" not in d
        assert "expected_coverage" not in d
        assert "label_map" not in d


class TestStringWeightsId:
    """MOOSE identifies models by string (e.g. 'Dataset123_Organs'), not int
    dataset ID. The recipe schema accepts both and preserves the type."""

    def test_single_accepts_string_id(self):
        spec = TaskSpec(name="clin_ct_organs", source="moose", modality="CT",
                        shape="single", single="Dataset123_Organs")
        assert spec.single == "Dataset123_Organs"

    def test_cascade_step_accepts_string_id(self):
        spec = TaskSpec(
            name="clin_ct_body_composition", source="moose", modality="CT",
            shape="cascade",
            cascade=(CascadeStep(weights_id="Dataset666_FastVert", crop_to_classes=(22,)),
                     CascadeStep(weights_id="Dataset778_BodyComp")),
        )
        assert spec.cascade[0].weights_id == "Dataset666_FastVert"
        assert spec.cascade[1].weights_id == "Dataset778_BodyComp"

    def test_string_id_round_trips(self):
        spec = TaskSpec(name="clin_ct_organs", source="moose", modality="CT",
                        shape="single", single="Dataset123_Organs")
        rt = _taskspec_from_dict(json.loads(json.dumps(_taskspec_to_dict(spec))))
        assert spec == rt
        assert isinstance(rt.single, str)

    def test_int_id_stays_int_through_round_trip(self):
        spec = TaskSpec(name="total_fast", source="ts", modality="CT",
                        shape="single", single=297)
        rt = _taskspec_from_dict(json.loads(json.dumps(_taskspec_to_dict(spec))))
        assert rt.single == 297
        assert isinstance(rt.single, int)


class TestNestedCascadeSpec:
    """A cascade step can reference another task by name (crop_from_task)
    instead of an inline weights_id. Spec-level validation + round-trip;
    flattening/dispatch is exercised in test_segment.py."""

    def test_step_requires_exactly_one_source(self):
        with pytest.raises(ValueError, match="exactly one"):
            CascadeStep()
        with pytest.raises(ValueError, match="exactly one"):
            CascadeStep(weights_id=1, crop_from_task="other")

    def test_crop_from_task_step_valid(self):
        step = CascadeStep(crop_from_task="craniofacial_structures", crop_to_classes=(2, 7))
        assert step.weights_id is None
        assert step.crop_from_task == "craniofacial_structures"

    def test_nested_cascade_round_trips(self):
        spec = TaskSpec(
            name="teeth", source="ts", modality="CT", shape="cascade",
            cascade=(CascadeStep(crop_from_task="craniofacial_structures",
                                 crop_to_classes=(2, 7), dilation_mm=10.0),
                     CascadeStep(weights_id=113)),
        )
        rt = _taskspec_from_dict(json.loads(json.dumps(_taskspec_to_dict(spec))))
        assert spec == rt
        assert rt.cascade[0].crop_from_task == "craniofacial_structures"
        assert rt.cascade[0].weights_id is None
        assert rt.cascade[1].weights_id == 113
