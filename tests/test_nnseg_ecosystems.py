"""Model ecosystems: the checkpoint is the spec; the registry federates."""
import json

import pytest

from nnseg.ecosystems import (EcosystemCatalog, MooseEcosystem, CustomEcosystem,
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


def test_short_name_collisions_resolve_by_prefix(tmp_path):
    """Two ecosystems may ship the same short name: only the short form goes
    ambiguous, and the error names the qualified candidates."""
    class Eco:
        def __init__(self, name):
            self.name = name
        def tasks(self):
            return ["x"]
        def materialized(self, t, root):
            return False
        def info(self, t, root):
            return {"name": t, "ecosystem": self.name, "materialized": False}
    cat = EcosystemCatalog([Eco("a"), Eco("b")], root=tmp_path)
    assert cat.names() == ["a:x", "b:x"]
    with pytest.raises(LookupError, match="ambiguous.*a:x.*b:x"):
        cat.resolve("x")
    eco, short, canonical, version = cat.resolve("a:x")
    assert (eco.name, short, canonical, version) == ("a", "x", "a:x", None)
    eco, short, canonical, version = cat.resolve("b:x@v2")
    assert (canonical, version) == ("b:x", "v2")


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


def test_custom_ecosystem_serves_local_folders(tmp_path):
    folder = _fake_model_folder(tmp_path)
    eco = CustomEcosystem({"my_model": folder})
    assert eco.tasks() == ["my_model"]
    spec = eco.spec("my_model", None)
    assert spec.name == "my_model" and len(spec.label_map) == 2


def test_catalog_federates_and_reports(tmp_path):
    folder = _fake_model_folder(tmp_path)
    cat = EcosystemCatalog([TSEcosystem(), CustomEcosystem({"mine": folder})],
                           root=tmp_path)
    assert "ts:total_fast" in cat.names() and "custom:mine" in cat.names()
    assert cat.info("mine")["ecosystem"] == "custom"
    assert cat.info("mine")["name"] == "custom:mine"       # canonical everywhere
    assert cat.info("ts:total_fast")["materialized"] is True
    spec = cat.get("custom:mine")
    assert spec.name == "custom:mine" and spec.label_map[1] == "organ_a"
    assert cat.get("mine").name == "custom:mine"           # short form converges
    with pytest.raises(LookupError):
        cat.info("nope")


def test_version_selector_pins_installs(tmp_path):
    """@version installs the pinned release or refuses - never a silent
    wrong-version serve."""
    folder = _fake_model_folder(tmp_path)
    cat = EcosystemCatalog([CustomEcosystem({"mine": folder})], root=tmp_path)
    with pytest.raises(Exception, match="no version metadata"):
        cat.get("custom:mine@v9")
    from nnseg.weights_fetch import _write_sidecar
    _write_sidecar(folder, "mine", "v9", {"url": "local"}, None)
    assert cat.get("custom:mine@v9").name == "custom:mine"  # matching pin passes
    with pytest.raises(Exception, match="records"):
        cat.get("custom:mine@v8")

    moose = MooseEcosystem()
    with pytest.raises(Exception, match="offers tag"):
        moose.ensure("clin_ct_fast_organs", tmp_path, version="not-a-tag")


def test_ts_pin_refuses_unknown_installed_version(tmp_path, monkeypatch):
    """Round 4 (E1): a sidecar-less weights folder (TS's own downloader, or a
    hand copy) must NOT silently satisfy an @version pin."""
    from nnseg.ecosystems import TSEcosystem
    from nnseg.errors import ModelNotFound

    eco = TSEcosystem()
    folder = tmp_path / "Dataset297_Total"
    folder.mkdir()
    (folder / "dataset.json").write_text("{}")     # present, but NO version sidecar
    monkeypatch.setattr("nnseg.weights_fetch.ensure_task_weights",
                        lambda *a, **k: [folder])
    with pytest.raises(ModelNotFound, match="unknown|remove"):
        eco.ensure("total", tmp_path, version="v2.0.0")


def test_moose_pin_checks_installed_not_manifest_tag(tmp_path):
    """Round 4 (E2): matching the manifest tag is not proof the bytes on disk
    are that release - a stale folder with an old sidecar must be refused."""
    from nnseg.ecosystems import MooseEcosystem
    from nnseg.errors import ModelNotFound
    from nnseg.weights_fetch import _write_sidecar

    moose = MooseEcosystem()
    task = moose.tasks()[0]
    entry = moose._entries[task]
    folder = moose._folder(task, tmp_path)
    folder.mkdir(parents=True)
    (folder / "dataset.json").write_text("{}")     # materialized...
    _write_sidecar(folder, "x", "OLD-RELEASE", {"url": "u"}, None)
    with pytest.raises(ModelNotFound, match="OLD-RELEASE|remove"):
        moose.ensure(task, tmp_path, version=entry["tag"])   # == manifest tag
    # the matching installed tag passes
    _write_sidecar(folder, "x", entry["tag"], {"url": "u"}, None)
    moose.ensure(task, tmp_path, version=entry["tag"])        # no raise


def test_moose_extract_is_atomic_and_digest_checked(tmp_path):
    """Round 4 (E3): a digest mismatch is refused, and an interrupted unpack
    leaves nothing materialized (temp dir + os.replace)."""
    import io
    import zipfile

    from nnseg.ecosystems import _download_and_extract_zip
    from nnseg.errors import InputError

    zbuf = io.BytesIO()
    with zipfile.ZipFile(zbuf, "w") as z:
        z.writestr("Dataset001/dataset.json", "{}")
        z.writestr("Dataset001/fold_0/checkpoint.pth", b"weights")
    payload = zbuf.getvalue()

    import urllib.request
    from unittest import mock

    class R(io.BytesIO):
        def __enter__(self): return self
        def __exit__(self, *a): return False

    dest = tmp_path / "moose"
    dest.mkdir()
    with mock.patch.object(urllib.request, "urlopen",
                           lambda *a, **k: R(payload)):
        with pytest.raises(InputError, match="digest"):
            _download_and_extract_zip("http://h/w.zip", dest, sha256="00" * 32)
        assert not any(dest.iterdir())         # nothing left half-written
        _download_and_extract_zip("http://h/w.zip", dest)   # no digest: ok
    assert (dest / "Dataset001" / "dataset.json").exists()
    assert (dest / "Dataset001" / "fold_0" / "checkpoint.pth").exists()
    assert not list(dest.glob(".unzip-*"))     # staging cleaned


def test_fastsurfer_ecosystem_lists_but_refuses_spec(tmp_path):
    """FastSurfer is an engine: it resolves/lists/describes, but spec() refuses
    (no TaskSpec - it runs on a FastSurfer worker, not the nnU-Net pipeline)."""
    from nnseg.ecosystems import EcosystemCatalog, FastSurferEcosystem
    from nnseg.errors import UnsupportedModel
    cat = EcosystemCatalog([FastSurferEcosystem()], root=tmp_path)
    assert cat.resolve("fastsurfer:brain")[2] == "fastsurfer:brain"
    assert cat.resolve("brain")[2] == "fastsurfer:brain"
    fs_info = cat.info("fastsurfer:brain")
    assert fs_info["engine"] == "fastsurfer" and fs_info["task_spec"] is False
    with pytest.raises(UnsupportedModel, match="engine, not an nnU-Net task"):
        cat.get("fastsurfer:brain")


def test_synthstrip_ecosystem_lists_but_refuses_spec(tmp_path):
    """SynthStrip is an engine too: resolves/lists/describes with weights_installed
    for the cache key, but spec() refuses (no TaskSpec)."""
    from nnseg.ecosystems import EcosystemCatalog, SynthStripEcosystem
    from nnseg.errors import UnsupportedModel
    cat = EcosystemCatalog([SynthStripEcosystem()], root=tmp_path)
    assert cat.resolve("synthstrip:mask")[2] == "synthstrip:mask"
    assert cat.resolve("mask")[2] == "synthstrip:mask"
    info = cat.info("synthstrip:mask")
    assert info["engine"] == "synthstrip" and info["task_spec"] is False
    assert info["weights_installed"] == [{"id": "synthstrip", "version": "v1"}]
    with pytest.raises(UnsupportedModel, match="engine, not an nnU-Net task"):
        cat.get("synthstrip:mask")


# -- the seams a many-task engine catalog needs (MONAI is the first; see
#    medseg/docs/monai-bundles.md). Pinned here because all four ecosystems that
#    exist today are one-model engines or nnU-Net catalogs, so nothing else would
#    notice if these regressed.

def test_engine_ecosystem_does_not_imply_image_baked_weights(tmp_path):
    """has_task_spec (can the nnU-Net pipeline run it?) and materialization (where
    do the weights come from?) are independent axes. An engine catalog needs
    has_task_spec=False WITH a real install per task."""
    from pathlib import Path

    from nnseg.ecosystems import EngineEcosystem, ImageBakedEcosystem

    assert EngineEcosystem.has_task_spec is False
    # the base engine class must NOT decide materialization for its subclasses
    assert "materialized" not in vars(EngineEcosystem)
    assert "ensure" not in vars(EngineEcosystem)
    # the image-baked flavor is what supplies the always-materialized behavior
    assert vars(ImageBakedEcosystem)["materialized"] is not None

    class Catalog(EngineEcosystem):          # no TaskSpec, but installs per task
        name, engine = "cat", "fastsurfer"
        def tasks(self): return ["m"]
        def materialized(self, task, root): return (Path(root) / task).is_dir()
        def ensure(self, task, root, progress=None, version=None):
            (Path(root) / task).mkdir(parents=True, exist_ok=True)

    eco = Catalog()
    assert eco.info("m", tmp_path)["materialized"] is False
    eco.ensure("m", tmp_path)
    assert eco.info("m", tmp_path)["materialized"] is True


def test_weights_identity_and_metadata_can_be_answered_per_task(tmp_path):
    """A catalog knows a version per model, not one constant - and reads modality
    and structures from the model itself once installed."""
    from nnseg.ecosystems import EngineEcosystem

    class Catalog(EngineEcosystem):
        name, engine = "cat", "fastsurfer"
        def tasks(self): return ["a", "b"]
        def materialized(self, task, root): return True
        def ensure(self, task, root, progress=None, version=None): return None
        def weights_identity(self, task, root):
            return [{"id": f"bundle-{task}", "version": "1.2.3"}]
        def describe_task(self, task, root):
            return {"modality": "CT", "structures": ["spleen", "liver"]}

    eco = Catalog()
    a, b = eco.info("a", tmp_path), eco.info("b", tmp_path)
    assert a["weights_installed"] == [{"id": "bundle-a", "version": "1.2.3"}]
    assert b["weights_installed"] == [{"id": "bundle-b", "version": "1.2.3"}]
    assert a["structures"] == ["spleen", "liver"] and a["modality"] == "CT"
    # distinct identities per task must reach the cache key, or two models collide
    from nnseg.serve import result_key
    key = lambda i: result_key(("sha256:x",), "cat:t", {},
                               [f"{e['id']}={e['version']}" for e in i["weights_installed"]])
    assert key(a) != key(b)


def test_one_model_engines_still_take_their_identity_from_the_registry(tmp_path):
    """The default path is unchanged: an image-baked engine answers from the
    engine registry's constant, with no per-task work."""
    from nnseg.ecosystems import FastSurferEcosystem
    from nnseg.engines import registry as R
    eco = FastSurferEcosystem()
    assert eco.weights_identity("brain", tmp_path) == R.ENGINES["fastsurfer"].weights_identity()
