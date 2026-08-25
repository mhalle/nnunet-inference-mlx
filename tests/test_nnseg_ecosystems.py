"""Model ecosystems: the checkpoint is the spec; the registry federates."""
import json

import pytest

from nnseg.ecosystems import (EcosystemCatalog, MooseEcosystem, NativeEcosystem,
                              TSEcosystem, registry)


def _fake_model_folder(root, name="Dataset900_Toy", labels=None):
    d = root / name / "nnUNetTrainer__nnUNetPlans__3d_fullres"
    (d / "fold_all").mkdir(parents=True)
    (d / "dataset.json").write_text(json.dumps({
        "channel_names": {"0": "CT"},
        "labels": {"background": 0, **(labels or {"organ_a": 1, "organ_b": 2})},
        "numTraining": 1, "file_ending": ".nii.gz"}))
    (d / "plans.json").write_text(json.dumps({"configurations": {"3d_fullres": {}}}))
    return root / name


def test_registry_rejects_task_collisions(tmp_path):
    class A:
        name = "a"
        def tasks(self): return ["x"]
    class B:
        name = "b"
        def tasks(self): return ["x"]
    with pytest.raises(ValueError):
        registry([A(), B()])


def test_ts_and_moose_coexist_without_collisions():
    reg = registry(None)
    assert set(reg) == {"ts", "moose"}
    assert "total_fast" in reg["ts"].tasks()
    assert "clin_ct_fast_organs" in reg["moose"].tasks()


def test_moose_info_before_install(tmp_path):
    eco = MooseEcosystem()
    info = eco.info("clin_ct_fast_organs", tmp_path)
    assert info["ecosystem"] == "moose" and info["materialized"] is False
    assert info["modality"] == "CT" and "structures" not in info


def test_moose_materialized_spec_reads_checkpoint(tmp_path):
    eco = MooseEcosystem()
    folder = eco._entries["clin_ct_fast_organs"]["folder"]
    _fake_model_folder(tmp_path / "moose", folder)
    assert eco.materialized("clin_ct_fast_organs", tmp_path)
    spec = eco.spec("clin_ct_fast_organs", tmp_path)
    assert spec.name == "clin_ct_fast_organs"
    assert set(spec.label_map.values()) == {"organ_a", "organ_b"}
    info = eco.info("clin_ct_fast_organs", tmp_path)
    assert info["materialized"] and info["structures"] == ["organ_a", "organ_b"]


def test_native_ecosystem_serves_local_folders(tmp_path):
    folder = _fake_model_folder(tmp_path)
    eco = NativeEcosystem({"my_model": folder})
    assert eco.tasks() == ["my_model"]
    spec = eco.spec("my_model", None)
    assert spec.name == "my_model" and len(spec.label_map) == 2


def test_catalog_federates_and_reports(tmp_path):
    folder = _fake_model_folder(tmp_path)
    cat = EcosystemCatalog([TSEcosystem(), NativeEcosystem({"mine": folder})],
                           root=tmp_path)
    assert "total_fast" in cat.names() and "mine" in cat.names()
    assert cat.info("mine")["ecosystem"] == "native"
    assert cat.info("total_fast")["materialized"] is True
    assert cat.get("mine").label_map[1] == "organ_a"
    with pytest.raises(LookupError):
        cat.info("nope")
