"""The wire's typed vocabulary: roles, parameter schemas, and published facts.

The thing under test is a contract, so these read as "what does a client see".
The role cases in particular guard a specific silent-wrong: MONAI's BraTS bundle
declares its channels T1c/T1/T2/FLAIR while nnU-Net's own BraTS convention orders
the same four files FLAIR/T1/T1ce/T2. Binding by position would serve one of
those two conventions a confidently wrong segmentation, which is why position is
never a fallback here.
"""
import json

import pytest

from nnseg import schemas as S
from nnseg.ecosystems import MonaiEcosystem
from nnseg.errors import RequestError


# -- the role vocabulary ---------------------------------------------------

@pytest.mark.parametrize("declared,canonical", [
    ("T1c", "T1w-ce"), ("t1ce", "T1w-ce"), ("T1-CE", "T1w-ce"), ("T1Gd", "T1w-ce"),
    ("T1", "T1w"), ("t2", "T2w"), ("FLAIR", "FLAIR"), ("T2-FLAIR", "FLAIR"),
])
def test_spellings_models_actually_use_map_to_one_name(declared, canonical):
    assert S.canonical_role(declared) == canonical


@pytest.mark.parametrize("declared", ["image", "input", "channel_0", "Renal_ZZZ", ""])
def test_an_unmapped_or_generic_name_simply_has_no_alias(declared):
    """The failure mode of the alias table has to be silence, not a wrong guess -
    that is what makes a hand-maintained table safe here when a hand-maintained
    label list would not be."""
    assert S.canonical_role(declared) is None


def test_roles_match_across_spellings_but_never_across_sequences():
    assert S.roles_match("T1c", "t1-ce") and S.roles_match("T1c", "T1w-ce")
    assert not S.roles_match("T1c", "T1")        # the whole point
    assert not S.roles_match("T2", "FLAIR")


def test_input_specs_record_the_models_own_channel_order():
    specs = S.input_specs(["T1c", "T1", "T2", "FLAIR"], modality="MR")
    assert [s["name"] for s in specs] == ["T1c", "T1", "T2", "FLAIR"]
    assert [s["channel"] for s in specs] == [0, 1, 2, 3]
    assert specs[0]["alias"] == "T1w-ce" and specs[0]["required"] is True
    assert all(s["modality"] == "MR" for s in specs)


# -- parameter schemas -----------------------------------------------------

def test_a_typo_is_reported_as_the_typo_not_as_a_missing_field():
    """`promts` makes `prompts` missing too; reporting the missing one sends the
    caller looking for a field they believe they sent."""
    with pytest.raises(RequestError) as e:
        S.validate_options(S.VoxTellParams, {"promts": ["liver"]})
    d = e.value.detail
    assert d["code"] == "unknown_parameter" and d["parameter"] == "promts"
    assert "did you mean 'prompts'" in d["message"]
    assert len(d["errors"]) == 2          # every failure, not just the headline


def test_voxtell_requires_prompts_at_submit():
    with pytest.raises(RequestError) as e:
        S.validate_options(S.VoxTellParams, {})
    assert e.value.detail["code"] == "missing_parameter"
    assert e.value.detail["missing"] == ["prompts"]
    S.validate_options(S.VoxTellParams, {"prompts": ["liver"]})      # the happy path


def test_deployment_policy_is_not_settable_over_the_wire():
    """device/dtype/weights describe the machine, not the requested result."""
    for knob in ("device", "dtype", "weights", "batch_size", "accumulate"):
        with pytest.raises(RequestError) as e:
            S.validate_options(S.ProcessingParams, {knob: "whatever"})
        assert e.value.detail["code"] == "unknown_parameter"


@pytest.mark.parametrize("opts,ok", [
    ({"grid": 1.5}, True), ({"grid": "input"}, True), ({"grid": "model"}, True),
    ({"grid": -1}, False), ({"grid": "coarse"}, False),
    ({"interp": "linear"}, True), ({"interp": "cubic"}, False),
    ({"resampling_order": 3}, True), ({"resampling_order": 9}, False),
])
def test_processing_knobs_accept_what_they_document(opts, ok):
    if ok:
        S.validate_options(S.ProcessingParams, opts)
    else:
        with pytest.raises(RequestError):
            S.validate_options(S.ProcessingParams, opts)


def test_validation_never_rewrites_the_options_it_checked():
    """The options dict is hashed into the result-cache key, so a normalization
    here would silently move every key."""
    opts = {"grid": 1.5, "interp": "nearest"}
    assert S.validate_options(S.ProcessingParams, opts) == {"grid": 1.5,
                                                            "interp": "nearest"}


def test_published_schemas_are_real_json_schema():
    groups = S.parameter_groups(S.VoxTellParams)
    assert groups["algorithm"]["required"] == ["prompts"]
    assert groups["algorithm"]["additionalProperties"] is False
    assert json.loads(json.dumps(groups))          # serializable as-is


def test_an_engine_without_processing_knobs_publishes_an_empty_group():
    """Advertising `interp` on an engine that runs someone else's chain end to
    end would be advertising a knob we do not turn."""
    groups = S.parameter_groups(S.NoParams, processing=False)
    assert not groups["processing"].get("properties")


# -- what describe() publishes per task ------------------------------------

def _bundle(root, name="toy", version="1.0", *, n_in=1, channel_def=None,
            postprocessing=None, config_name="inference.json"):
    """A synthetic installed MONAI bundle: metadata + an inference config."""
    d = root / "monai" / f"{name}_v{version}" / "configs"
    d.mkdir(parents=True)
    inputs = {"image": {"modality": "MR", "num_channels": n_in}}
    if channel_def is not None:
        inputs["image"]["channel_def"] = channel_def
    (d / "metadata.json").write_text(json.dumps({
        "network_data_format": {
            "inputs": inputs,
            "outputs": {"pred": {"channel_def": {"0": "background", "1": "Tumor"}}}}}))
    (d / config_name).write_text(json.dumps(
        {"postprocessing": {"transforms": postprocessing or []}}))
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps({"bundles": {name: {
        "version": version, "source": "monaihosting", "modality": "MR",
        "in_channels": n_in, "n_labels": 2}}}))
    return MonaiEcosystem(manifest=manifest), name


def test_a_bundle_that_names_its_channels_gets_bindable_roles(tmp_path):
    eco, name = _bundle(tmp_path, n_in=4,
                        channel_def={"0": "T1c", "1": "T1", "2": "T2", "3": "FLAIR"})
    d = eco.describe_task(name, tmp_path)
    assert [i["name"] for i in d["inputs"]] == ["T1c", "T1", "T2", "FLAIR"]
    assert d["inputs"][0]["alias"] == "T1w-ce"


def test_a_bundle_that_declares_more_channels_than_it_names_is_not_bindable(tmp_path):
    """renalStructures_CECT says 3 channels and names one. Refuse rather than
    bind by position."""
    eco, name = _bundle(tmp_path, n_in=3, channel_def={"0": "image"})
    d = eco.describe_task(name, tmp_path)
    assert d["inputs"] is None
    assert d["inputs_incomplete"]["channels"] == 3
    assert d["inputs_incomplete"]["named"] == ["image"]


_INVERT = {"_target_": "Invertd", "nearest_interp": False}
_ARGMAX = {"_target_": "AsDiscreted", "argmax": True}


@pytest.mark.parametrize("post,mode,owner", [
    ([_INVERT, _ARGMAX], "graded", "bundle"),          # spleen: probabilities first
    ([_ARGMAX, _INVERT], "label-nearest", "bundle"),   # wholeBody: labelmap first
    ([_ARGMAX], "label-nearest", "nnseg"),             # never inverts; we resample
])
def test_restore_is_published_as_a_fact_read_from_the_bundles_own_config(
        tmp_path, post, mode, owner):
    """The bundle's postprocessing ORDER decides whether boundaries come back
    graded or snapped to the model grid, and it differs between bundles we ship.
    We do not override it - so a client gets to see it."""
    eco, name = _bundle(tmp_path, postprocessing=post)
    fact = eco.describe_task(name, tmp_path)["behavior"]["restore"]
    assert (fact["mode"], fact["owner"]) == (mode, owner)


def test_a_config_we_cannot_parse_says_unknown_rather_than_guessing(tmp_path):
    eco, name = _bundle(tmp_path, config_name="inference.yaml")
    fact = eco.describe_task(name, tmp_path)["behavior"]["restore"]
    assert fact["mode"] == "unknown"


def test_an_uninstalled_multichannel_bundle_does_not_claim_one_input(tmp_path):
    """Pre-install we know the channel COUNT from the manifest but not the names;
    describing that as a single image would be a lie a client acts on."""
    eco, name = _bundle(tmp_path, n_in=4)
    info = eco.info(name, tmp_path / "elsewhere")        # nothing installed there
    assert info["materialized"] is False
    assert info["inputs"] is None and "4 input channels" in info["inputs_hint"]
