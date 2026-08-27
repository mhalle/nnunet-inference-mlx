"""The MONAI ecosystem: a catalog of bundles on its own engine.

No bundle is downloaded here - what can go wrong at this layer is manifest
handling, per-task identity, and reading the bundle's own metadata, so a tiny
fake bundle on disk stands in for a real one.
"""
import json

import pytest

from nnseg.ecosystems import MonaiEcosystem

MANIFEST = {"bundles": {
    "demo_seg": {"version": "1.2.3", "url": "https://example/demo_seg_v1.2.3.zip",
                 "checksum": "abc123", "source": "monaihosting",
                 "modality": "CT", "in_channels": 1,
                 "n_labels": 3, "task": "Demo segmentation", "required_packages": {}},
    "other_seg": {"version": "0.4.0", "url": "https://example/other_seg_v0.4.0.zip",
                  "checksum": "", "source": "huggingface_hub",
                  "modality": "MR", "in_channels": 1,
                  "n_labels": 2, "task": "Other", "required_packages": {}},
}}


@pytest.fixture
def eco(tmp_path):
    p = tmp_path / "manifest.json"
    p.write_text(json.dumps(MANIFEST))
    return MonaiEcosystem(manifest=p)


def _install(eco, tmp_path, bundle="demo_seg", version="1.2.3", labels=None):
    """Write a bundle where ensure() would have put one."""
    d = tmp_path / "monai" / f"{bundle}_v{version}" / "configs"
    d.mkdir(parents=True)
    (d / "metadata.json").write_text(json.dumps({
        "task": "Demo segmentation",
        "network_data_format": {
            "inputs": {"image": {"modality": "CT", "num_channels": 1,
                                 "spatial_shape": [96, 96, 96]}},
            "outputs": {"pred": {"num_channels": len(labels or {}),
                                 "channel_def": labels or {"0": "background",
                                                           "1": "spleen", "2": "liver"}}}}}))
    return d.parent


def test_it_is_an_engine_catalog_not_an_image_baked_one(eco, tmp_path):
    """The shape that motivated splitting the base classes: no nnU-Net TaskSpec,
    but a real install per task rather than weights baked into an image."""
    from nnseg.ecosystems import EngineEcosystem, ImageBakedEcosystem
    assert isinstance(eco, EngineEcosystem) and not isinstance(eco, ImageBakedEcosystem)
    assert eco.has_task_spec is False
    assert eco.materialized("demo_seg", tmp_path) is False      # nothing installed yet
    _install(eco, tmp_path)
    assert eco.materialized("demo_seg", tmp_path) is True


def test_identity_is_per_bundle_so_two_bundles_cannot_share_a_cache_entry(eco, tmp_path):
    from nnseg.serve import result_key
    a = eco.weights_identity("demo_seg", tmp_path)
    b = eco.weights_identity("other_seg", tmp_path)
    assert a == [{"id": "demo_seg", "version": "1.2.3", "sha1": "abc123"}]
    assert "sha1" not in b[0]                    # the zoo omits it on recent releases
    key = lambda i: result_key(("sha256:x",), "monai:t", {},
                               [f"{e['id']}={e['version']}" for e in i])
    assert key(a) != key(b)


def test_labels_come_from_the_installed_bundle_not_the_manifest(eco, tmp_path):
    """The bundle is the spec. The manifest says n_labels=3; the installed bundle
    is what actually defines the names, and a mismatch must follow the bundle."""
    _install(eco, tmp_path, labels={"0": "background", "1": "spleen",
                                    "2": "liver", "3": "kidney"})
    info = eco.info("demo_seg", tmp_path)
    assert info["structures"] == ["spleen", "liver", "kidney"]   # background dropped
    assert info["modality"] == "CT"
    assert info["engine"] == "monai" and info["task_spec"] is False


def test_an_uninstalled_task_still_describes_usefully(eco, tmp_path):
    """/v1/tasks must say something before anything is downloaded - that is what the
    manifest's listing facts are for."""
    info = eco.info("other_seg", tmp_path)
    assert info["materialized"] is False
    assert info["modality"] == "MR" and info["bundle_version"] == "0.4.0"
    assert info["n_structures"] == 1              # n_labels minus background
    assert "structures" not in info               # not knowable without the bundle


def test_a_version_pin_that_disagrees_with_the_build_is_refused(eco, tmp_path):
    from nnseg.errors import ModelNotFound
    with pytest.raises(ModelNotFound, match="curates demo_seg v1.2.3"):
        eco.ensure("demo_seg", tmp_path, version="9.9.9")


def test_an_unknown_bundle_names_what_is_curated(eco, tmp_path):
    from nnseg.errors import ModelNotFound
    with pytest.raises(ModelNotFound, match="curates"):
        eco.ensure("not_a_bundle", tmp_path)


def test_spec_refuses_because_a_bundle_is_not_an_nnunet_task(eco, tmp_path):
    from nnseg.errors import UnsupportedModel
    with pytest.raises(UnsupportedModel, match="monai engine"):
        eco.spec("demo_seg", tmp_path)


def test_the_shipped_manifest_is_curated_and_servable():
    """Guards the generator's two jobs: 3D segmentation only, and nothing
    multi-channel (the job wire takes one input and refuses multi-channel)."""
    eco = MonaiEcosystem()
    assert eco.tasks(), "the shipped manifest is empty"
    for name in eco.tasks():
        entry = eco._entry(name)
        assert int(entry.get("in_channels") or 1) == 1, f"{name} needs multiple channels"
        assert int(entry.get("n_labels") or 0) >= 2, f"{name} has no labels"
        # the zoo moved hosting; what we need is the host monai.bundle.download
        # resolves, not an archive URL we fetch ourselves
        assert entry.get("source") in ("huggingface_hub", "monaihosting", "github"), \
            f"{name} has no resolvable hosting source"
        assert entry.get("version"), f"{name} has no version"
