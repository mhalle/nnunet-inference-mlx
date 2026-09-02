"""The MRSegmentator ecosystem: flat zips land whole, the manifest is honest about
what the checkpoint cannot say (LPS), and the pipeline's orientation decision honors it."""
import io
import json
import re
import zipfile
from unittest import mock

import numpy as np
import pytest

from nnseg.ecosystems import (EcosystemCatalog, MRSegmentatorEcosystem, MooseEcosystem,
                              TSEcosystem, registry)
from nnseg.errors import ModelNotFound


def _flat_zip(version="1.2", folds=(0, 1), labels=None) -> bytes:
    """What upstream ships: dataset.json / plans.json / version.json / fold_*/ at top level."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("dataset.json", json.dumps({
            "channel_names": {"0": "MR"},
            "labels": {"background": 0, **(labels or {"spleen": 1, "right_kidney": 2, "left_kidney": 3})},
            "numTraining": 1, "file_ending": ".nii.gz"}))
        z.writestr("plans.json", json.dumps({"image_reader_writer": "SimpleITKIO",
                                             "configurations": {"3d_fullres": {}}}))
        z.writestr("version.json", json.dumps({"weights_version": float(version)}))
        for f in folds:
            z.writestr(f"fold_{f}/checkpoint_final.pth", b"weights")
    return buf.getvalue()


class _Resp(io.BytesIO):
    def __enter__(self): return self
    def __exit__(self, *a): return False


def _serve(payload: bytes):
    import urllib.request
    return mock.patch.object(urllib.request, "urlopen", lambda *a, **k: _Resp(payload))


def _eco_expecting(payload: bytes, task="base") -> MRSegmentatorEcosystem:
    """An ecosystem whose manifest digest is the fake asset's - the real manifest carries
    upstream's sha256, so a fake zip is (correctly) refused unless the entry expects it."""
    import hashlib
    eco = MRSegmentatorEcosystem()
    eco._entries[task] = {**eco._entries[task], "sha256": hashlib.sha256(payload).hexdigest()}
    return eco


def test_manifest_holds_only_what_the_checkpoint_cannot_know():
    eco = MRSegmentatorEcosystem()
    assert eco.tasks() == ["base", "body_comp"]
    for task, e in eco._entries.items():
        assert e["url"].startswith("https://"), task
        assert re.fullmatch(r"[0-9a-f]{64}", e["sha256"]), task          # upstream publishes digests
        assert e["tag"], task
        dataset, config = e["folder"].split("/")
        assert dataset.startswith("Dataset") and config.count("__") == 2, task   # resolvable as-is
        assert config.startswith("nnUNetTrainerNoMirroring__"), task   # trained without mirroring


def test_default_catalog_lists_it_beside_ts_and_moose():
    reg = registry(None)
    assert {"ts", "moose", "mrsegmentator"} <= set(reg)
    assert isinstance(reg["mrsegmentator"], MRSegmentatorEcosystem)
    cat = EcosystemCatalog([TSEcosystem(), MooseEcosystem(), MRSegmentatorEcosystem()])
    assert "mrsegmentator:base" in cat.names() and "mrsegmentator:body_comp" in cat.names()
    assert cat.engine_of("mrsegmentator:base") == "nnunetv2"      # a catalog, not an engine


def test_info_before_install_knows_modality_orientation_and_version(tmp_path):
    info = MRSegmentatorEcosystem().info("base", tmp_path)
    assert info["ecosystem"] == "mrsegmentator" and info["engine"] == "nnunetv2"
    assert info["materialized"] is False and info["task_spec"] is True
    assert info["modality"] == "MR" and info["orientation"] == "LPS" and info["tag"] == "1.2"
    assert "structures" not in info                       # never guessed from a manifest


def test_install_lands_the_flat_zip_in_a_configuration_folder(tmp_path):
    payload = _flat_zip()
    eco = _eco_expecting(payload)
    with _serve(payload):
        eco.ensure("base", tmp_path, progress=None)
    folder = eco._folder("base", tmp_path)
    assert folder.name == "nnUNetTrainerNoMirroring__nnUNetPlans__3d_fullres"
    assert (folder / "dataset.json").is_file() and (folder / "fold_1" / "checkpoint_final.pth").is_file()
    assert not list(folder.parent.glob(".install-*"))     # staging cleaned
    assert eco.materialized("base", tmp_path)
    from nnseg.weights_fetch import installed_version
    assert installed_version(folder)["tag"] == "1.2"
    # the spec is the checkpoint, plus the one fact the checkpoint cannot state
    spec = eco.spec("base", tmp_path)
    assert spec.lineage == "nnunetv2" and spec.modality == "MR" and spec.orientation == "LPS"
    assert set(spec.label_map.values()) == {"spleen", "right_kidney", "left_kidney"}
    info = eco.info("base", tmp_path)
    assert info["materialized"] and info["structures"] == ["left_kidney", "right_kidney", "spleen"]
    # idempotent, and a matching pin is satisfied by the sidecar
    with _serve(b"not a zip"):
        eco.ensure("base", tmp_path)
        eco.ensure("base", tmp_path, version="1.2")


def test_a_digest_mismatch_is_refused_before_anything_is_unpacked(tmp_path):
    eco = MRSegmentatorEcosystem()                     # the REAL manifest digest
    from nnseg.errors import InputError
    with _serve(_flat_zip()), pytest.raises(InputError, match="digest"):
        eco.ensure("base", tmp_path)
    assert not eco.materialized("base", tmp_path)
    assert not list(eco._folder("base", tmp_path).parent.glob(".install-*"))


def test_install_refuses_a_zip_whose_own_version_disagrees_with_the_manifest(tmp_path):
    payload = _flat_zip(version="9.9")
    eco = _eco_expecting(payload)
    with _serve(payload), pytest.raises(ModelNotFound, match="weights_version"):
        eco.ensure("base", tmp_path)
    assert not eco.materialized("base", tmp_path)
    assert not list(eco._folder("base", tmp_path).parent.glob(".install-*"))


def test_install_is_atomic_when_the_asset_is_not_a_configuration_folder(tmp_path):
    """A zip that is not flat (or is missing folds) leaves nothing materialized."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("Dataset001/dataset.json", "{}")           # a MOOSE-style nested zip
    eco = _eco_expecting(buf.getvalue())
    with _serve(buf.getvalue()), pytest.raises(ModelNotFound, match="flat"):
        eco.ensure("base", tmp_path)
    assert not eco.materialized("base", tmp_path)
    assert not eco._folder("base", tmp_path).exists()


def test_pin_checks_the_installed_bytes_not_the_manifest(tmp_path):
    payload = _flat_zip()
    eco = _eco_expecting(payload)
    with pytest.raises(ModelNotFound, match="offers version"):
        eco.ensure("base", tmp_path, version="0.1")
    with _serve(payload):
        eco.ensure("base", tmp_path)
    with pytest.raises(ModelNotFound, match="1.2.*remove"):
        eco.ensure("base", tmp_path, version="1.0")          # installed 1.2, asked for 1.0


def test_catalog_resolves_short_and_pinned_forms(tmp_path):
    cat = EcosystemCatalog([TSEcosystem(), MRSegmentatorEcosystem()], root=tmp_path)
    eco, short, canonical, version = cat.resolve("body_comp@1.0")
    assert (eco.name, short, canonical, version) == ("mrsegmentator", "body_comp", "mrsegmentator:body_comp", "1.0")
    with pytest.raises(ModelNotFound, match="not installed"):
        MRSegmentatorEcosystem().spec("base", tmp_path)


# -- the orientation decision ------------------------------------------------

def _config_folder(root, reader="SimpleITKIO"):
    d = root / "Dataset900_Toy" / "nnUNetTrainer__nnUNetPlans__3d_fullres"
    (d / "fold_0").mkdir(parents=True)
    (d / "dataset.json").write_text(json.dumps({"channel_names": {"0": "MR"},
                                                "labels": {"background": 0, "a": 1}}))
    (d / "plans.json").write_text(json.dumps({"image_reader_writer": reader,
                                              "configurations": {"3d_fullres": {}}}))
    return d


def test_pipeline_orientation_follows_the_reader_unless_the_spec_overrides(tmp_path):
    import dataclasses
    from nnseg.pipeline import canonical_orientation_for
    from nnseg.tasks import TaskCatalog, TaskSpec
    from nnseg.weights import as_store
    store = as_store(tmp_path, layout="nnunetv2")
    plain = TaskSpec.from_model_folder(_config_folder(tmp_path / "plain"))
    assert canonical_orientation_for(plain, store) is None                   # stored order
    reorienting = TaskSpec.from_model_folder(_config_folder(tmp_path / "reo", "SimpleITKIOWithReorient"))
    assert canonical_orientation_for(reorienting, store) == "RAS"
    assert canonical_orientation_for(dataclasses.replace(plain, orientation="LPS"), store) == "LPS"
    ts = TaskCatalog("ts").get("total_fast")
    assert ts.orientation is None and canonical_orientation_for(ts, as_store(tmp_path)) == "RAS"


def test_read_can_target_lps_and_matches_dicomorient(tmp_path):
    import SimpleITK as sitk
    from nnseg import io as nio
    arr = np.arange(2 * 3 * 4, dtype=np.int16).reshape(2, 3, 4)
    img = sitk.GetImageFromArray(arr)
    img.SetDirection((-1, 0, 0, 0, -1, 0, 0, 0, 1))          # stored RAS
    img.SetOrigin((10.0, 20.0, 30.0))
    p = tmp_path / "v.nii.gz"
    sitk.WriteImage(img, str(p))
    got, geo, original = nio.read(p, reorient=True, target="LPS")
    want = sitk.GetArrayFromImage(sitk.DICOMOrient(sitk.ReadImage(str(p)), "LPS"))
    assert original == "RAS" and np.array_equal(got, want)
    assert nio.orientation_of(nio.to_image(got, geo)) == "LPS"
    ras, _, _ = nio.read(p, reorient=True)                   # the default is unchanged
    assert np.array_equal(ras, arr)
