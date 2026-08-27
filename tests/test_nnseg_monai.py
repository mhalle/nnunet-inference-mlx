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
    """Guards the generator's two jobs: 3D segmentation only, and nothing that
    cannot be served.

    Servable now means single-input OR multi-input with every channel NAMED -
    the wire binds inputs by name, so a bundle that declares four channels and
    names one (renalStructures_CECT) cannot be bound at all and has no business
    in the catalog."""
    eco = MonaiEcosystem()
    assert eco.tasks(), "the shipped manifest is empty"
    for name in eco.tasks():
        entry = eco._entry(name)
        n_in = int(entry.get("in_channels") or 1)
        named = len(entry.get("channel_def") or {})
        assert n_in == 1 or named == n_in, \
            f"{name} declares {n_in} channels but names {named}: not bindable"
        assert int(entry.get("n_labels") or 0) >= 2, f"{name} has no labels"
        # the zoo moved hosting; what we need is the host monai.bundle.download
        # resolves, not an archive URL we fetch ourselves
        assert entry.get("source") in ("huggingface_hub", "monaihosting", "github"), \
            f"{name} has no resolvable hosting source"
        assert entry.get("version"), f"{name} has no version"


def test_the_inference_config_is_found_whatever_it_is_called(tmp_path):
    """Bundles do not agree on the extension - spleen ships inference.json,
    pancreas_ct_dints ships inference.yaml - and hardcoding one is a failure that
    only appears on the bundle nobody has run yet."""
    from nnseg.engines.monai_bundle import inference_config
    cfg = tmp_path / "configs"; cfg.mkdir()
    (cfg / "metadata.json").write_text("{}")
    (cfg / "inference.yaml").write_text("{}")
    assert inference_config(tmp_path).name == "inference.yaml"
    (cfg / "inference.json").write_text("{}")
    assert inference_config(tmp_path).name == "inference.json"   # json wins when both


def test_a_bundle_with_no_inference_config_says_what_it_has(tmp_path):
    from nnseg.engines.monai_bundle import inference_config
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "train.yaml").write_text("{}")
    with pytest.raises(FileNotFoundError, match="train.yaml"):
        inference_config(tmp_path)


def _bundle_with_channels(root, channel_def, name="multi", version="1.0"):
    """A synthetic installed bundle declaring `channel_def` input channels."""
    import json as _json
    d = root / "monai" / f"{name}_v{version}" / "configs"
    d.mkdir(parents=True)
    (d / "metadata.json").write_text(_json.dumps({"network_data_format": {
        "inputs": {"image": {"modality": "MR", "num_channels": len(channel_def),
                             "channel_def": channel_def}},
        "outputs": {"pred": {"channel_def": {"0": "background", "1": "Tumor"}}}}}))
    return d.parent


def _img(size=(4, 5, 6), spacing=(1.0, 1.0, 1.0)):
    import SimpleITK as sitk
    im = sitk.Image(*size, sitk.sitkFloat32)
    im.SetSpacing(spacing)
    return im


def test_channels_are_stacked_in_the_bundles_order_not_the_callers(tmp_path):
    """The request may name its inputs in any order; the tensor's channel order
    is the bundle's to declare. This is the difference that makes MONAI's BraTS
    (T1c first) and nnU-Net's BraTS (FLAIR first) incompatible as positions."""
    from nnseg.engines.monai_bundle import _stack_inputs, input_roles
    root = _bundle_with_channels(tmp_path, {"0": "T1c", "1": "T1", "2": "T2",
                                            "3": "FLAIR"})
    assert input_roles(root) == ["T1c", "T1", "T2", "FLAIR"]
    supplied = {"FLAIR": _img(), "T1": _img(), "T2": _img(), "T1c": _img()}
    ordered = _stack_inputs(supplied, root, "multi", lambda x: x)
    assert [role for role, _ in ordered] == ["T1c", "T1", "T2", "FLAIR"]


def test_inputs_on_different_grids_are_refused_with_the_reason(tmp_path):
    """nnseg does not register images - Slicer does that upstream. What nnseg
    owes the caller is a clear refusal instead of a shape error from inside
    someone else's transform chain."""
    from nnseg.engines.monai_bundle import _stack_inputs
    from nnseg.errors import InputError
    root = _bundle_with_channels(tmp_path, {"0": "T1c", "1": "FLAIR"})
    supplied = {"T1c": _img(), "FLAIR": _img(spacing=(0.86, 0.86, 1.0))}
    with pytest.raises(InputError) as e:
        _stack_inputs(supplied, root, "multi", lambda x: x)
    assert "not on the same grid" in str(e.value)
    assert "does not register images" in str(e.value)


def test_a_missing_channel_is_named(tmp_path):
    from nnseg.engines.monai_bundle import _stack_inputs
    from nnseg.errors import InputError
    root = _bundle_with_channels(tmp_path, {"0": "T1c", "1": "FLAIR"})
    with pytest.raises(InputError) as e:
        _stack_inputs({"T1c": _img(), "T2": _img()}, root, "multi", lambda x: x)
    assert "FLAIR" in str(e.value)


def _bundle_with_outputs(root, channel_def, name="out", version="1.0"):
    import json as _json
    d = root / "monai" / f"{name}_v{version}" / "configs"
    d.mkdir(parents=True)
    (d / "metadata.json").write_text(_json.dumps({"network_data_format": {
        "inputs": {"image": {"modality": "CT", "num_channels": 1}},
        "outputs": {"pred": {"channel_def": channel_def}}}}))
    return d.parent


def test_a_labelmap_bundle_names_its_values(tmp_path):
    from nnseg.engines.monai_bundle import resolve_label_names
    root = _bundle_with_outputs(tmp_path, {"0": "background", "1": "spleen",
                                           "2": "liver"})
    names, prov = resolve_label_names(root, [1, 2])
    assert names == {1: "spleen", 2: "liver"} and prov == {}


def test_a_region_head_is_not_read_as_a_labelmap(tmp_path):
    """brats declares three overlapping REGIONS as output channels, then writes
    BraTS's own 1/2/4 encoding. Read as a labelmap it called its background
    'Tumor core' - 95% of the volume - and never named value 4."""
    from nnseg.engines.monai_bundle import resolve_label_names
    root = _bundle_with_outputs(tmp_path, {"0": "Tumor core", "1": "Whole tumor",
                                           "2": "Enhancing tumor"})
    names, prov = resolve_label_names(root, [1, 2, 4])
    assert names == {1: "label 1", 2: "label 2", 4: "label 4"}
    assert prov["labels_unnamed"] is True
    assert prov["declared_channels"] == ["Tumor core", "Whole tumor",
                                         "Enhancing tumor"]
    assert "does not interpret region heads" in prov["labels_note"]


def test_a_region_head_is_caught_even_when_its_values_look_plausible(tmp_path):
    """The structural signal - no `background` entry - has to carry this, because
    a run that happens to emit only values 1 and 2 is indistinguishable from a
    labelmap by value alone."""
    from nnseg.engines.monai_bundle import resolve_label_names
    root = _bundle_with_outputs(tmp_path, {"0": "Tumor core", "1": "Whole tumor",
                                           "2": "Enhancing tumor"})
    names, prov = resolve_label_names(root, [1, 2])
    assert names == {1: "label 1", 2: "label 2"}
    assert prov["labels_unnamed"] is True
