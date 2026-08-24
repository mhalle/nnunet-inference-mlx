"""Refreshing the weights manifest, and telling "needs a license" apart from "missing".

The manifest is the provisioning mechanism for any machine without weights on disk, so a gap in
it means a task cannot run there at all. GitHub is stubbed - these must stay offline and fast.
"""
import json

import pytest

from nnseg import weights_fetch as wf


def _release(tag, published, assets):
    return {"tag_name": tag, "published_at": published,
            "assets": [{"name": n, "browser_download_url": f"https://x/{tag}/{n}",
                        "size": 1, **({"digest": f"sha256:{d}"} if d else {})} for n, d in assets]}


@pytest.fixture
def fake_github(monkeypatch):
    releases = [
        _release("v2.0.0", "2023-01-01T00:00:00Z", [("Dataset291_organs.zip", None),
                                                    ("Dataset297_total_3mm.zip", "aaa")]),
        _release("v2.5.0", "2025-01-01T00:00:00Z", [("Dataset305_discs.zip", "bbb"),
                                                    ("Dataset297_total_3mm_v204.zip", "ccc"),
                                                    ("not-a-dataset.txt", None)]),
    ]
    monkeypatch.setattr(wf, "_api", lambda url, token=None: releases)
    return releases


def test_discovery_parses_dataset_assets_and_ignores_others(fake_github):
    found = wf.discover_release_assets()
    assert set(found) == {"291", "297", "305"}          # the .txt is skipped
    assert wf.selected(found["305"])["sha256"] == "bbb"
    assert "sha256" not in wf.selected(found["291"])    # absent digest is simply omitted


def test_discovery_records_every_version_and_defaults_current_to_newest(fake_github):
    found = wf.discover_release_assets()
    assert set(found["297"]["versions"]) == {"v2.0.0", "v2.5.0"}   # both kept, not just one
    assert found["297"]["default"] == "v2.5.0"                     # 2025 beats 2023


def test_refresh_adds_missing_datasets(fake_github, tmp_path):
    m = tmp_path / "w.json"
    m.write_text(json.dumps({"weights": {}}))
    r = wf.refresh_manifest(path=m)
    saved = json.loads(m.read_text())["weights"]
    assert set(r["added"]) == {"291", "297", "305"} and set(saved) == {"291", "297", "305"}


def test_new_versions_are_recorded_without_changing_current(fake_github, tmp_path):
    """Facts (what upstream published) update freely; the decision (which to install) does not."""
    m = tmp_path / "w.json"
    m.write_text(json.dumps({"weights": {"297": {
        "default": "v2.0.0", "versions": {"v2.0.0": {"url": "https://x/v2.0.0/Dataset297_total_3mm.zip"}}}}}))
    r = wf.refresh_manifest(path=m)
    saved = json.loads(m.read_text())["weights"]["297"]
    assert saved["default"] == "v2.0.0"                  # untouched
    assert set(saved["versions"]) == {"v2.0.0", "v2.5.0"}
    assert r["new_versions"]["297"] == ["v2.5.0"] and r["behind_upstream"]["297"] == ("v2.0.0", "v2.5.0")


def test_update_existing_opts_in_to_repointing(fake_github, tmp_path):
    m = tmp_path / "w.json"
    m.write_text(json.dumps({"weights": {"297": {
        "default": "v2.0.0", "versions": {"v2.0.0": {"url": "https://x/v2.0.0/Dataset297_total_3mm.zip"}}}}}))
    wf.refresh_manifest(path=m, update_existing=True)
    assert json.loads(m.read_text())["weights"]["297"]["default"] == "v2.5.0"


def test_dry_run_writes_nothing(fake_github, tmp_path):
    m = tmp_path / "w.json"
    before = json.dumps({"weights": {}})
    m.write_text(before)
    r = wf.refresh_manifest(path=m, write=False)
    assert r["added"] and m.read_text() == before


# -- license-gated weights ----------------------------------------------------------------
def test_coverage_separates_license_required_from_missing():
    c = wf.coverage()
    assert c["covered"] and c["license_required"]
    # every license-gated task's ids really are in the commercial list
    for name, ids in c["license_required"].items():
        assert all(i in wf.LICENSE_GATED for i in ids), name
    # and nothing is reported in two categories at once
    assert not (set(c["license_required"]) & set(c["missing"]))


def test_a_license_gated_id_explains_itself_instead_of_keyerror(tmp_path):
    from nnseg.errors import ModelNotFound
    with pytest.raises(ModelNotFound, match="licensed backend"):
        wf.fetch_one(301, tmp_path)                     # heartchambers_highres


def test_an_unknown_id_points_at_refresh(tmp_path):
    from nnseg.errors import ModelNotFound
    with pytest.raises(ModelNotFound, match="weights refresh"):
        wf.fetch_one(999999, tmp_path)


def test_license_gated_ids_match_totalsegmentators_own_list():
    """LICENSE_GATED mirrors TS's commercial_models; if TS adds one, coverage should notice."""
    assert wf.LICENSE_GATED["301"] == "heartchambers_highres"
    assert wf.LICENSE_GATED["857"] == "thigh_shoulder_muscles"
    assert all(k.isdigit() for k in wf.LICENSE_GATED)


# -- the parallel current/versions schema -------------------------------------------------
def test_legacy_flat_entries_are_lifted_into_the_versioned_shape():
    e = wf._normalize({"297": {"url": "https://x/a.zip", "sha256": "aa"}})["297"]
    assert e["default"] == "unversioned"
    assert e["versions"]["unversioned"]["url"] == "https://x/a.zip"


def test_selected_returns_current_or_a_named_version():
    e = {"default": "v1", "versions": {"v1": {"url": "one"}, "v2": {"url": "two"}}}
    assert wf.selected(e)["url"] == "one"
    assert wf.selected(e, "v2")["url"] == "two"
    with pytest.raises(KeyError, match="v9"):
        wf.selected(e, "v9")


def test_dataset_key_canonicalizes_zero_padding():
    assert wf.dataset_key("008") == wf.dataset_key(8) == wf.dataset_key("8") == "8"
    assert wf.dataset_key("body_mr") == "body_mr"       # non-numeric ids pass through


def test_padded_and_unpadded_entries_merge_into_one_dataset():
    """Dataset008 and Dataset8 are the same model; two keys would be two entries for one thing."""
    n = wf._normalize({"8": {"url": "https://x/Dataset008.zip"},
                       "008": {"default": "v2.4", "versions": {"v2.4": {"url": "https://x/Dataset008.zip"}}}})
    assert list(n) == ["8"] and n["8"]["default"] == "v2.4"


def test_a_zero_padded_folder_on_disk_resolves(tmp_path):
    from nnseg.tasks import resolve_model_folder
    d = tmp_path / "Dataset008_HepaticVessel" / "nnUNetTrainer__nnUNetPlans__3d_fullres"
    d.mkdir(parents=True)
    assert resolve_model_folder(8, model_root=tmp_path) == d       # globbing Dataset8_* would miss
    assert wf.is_present(8, tmp_path)


def test_migration_matches_by_url_and_never_silently_repoints(fake_github, tmp_path, monkeypatch):
    """A legacy entry must keep the weights it already pointed at. 297 is published as both
    v2.0.0 and v2.0.4 and TotalSegmentator pins the older one - adopting 'newest' would change
    which weights download, and therefore the segmentations."""
    m = tmp_path / "w.json"
    m.write_text(json.dumps({"weights": {"297": {"url": "https://x/v2.0.0/Dataset297_total_3mm.zip"}}}))
    r = wf.refresh_manifest(path=m)
    saved = json.loads(m.read_text())["weights"]["297"]
    assert saved["default"] == "v2.0.0"                 # named, not repointed
    assert set(saved["versions"]) == {"v2.0.0", "v2.5.0"}   # the newer one is recorded, not chosen
    assert r["migrated"] == {"297": "v2.0.0"}


def test_a_placeholder_without_a_url_is_not_downloaded(tmp_path, monkeypatch):
    """A license-gated entry has no URL; fetch must explain, not try to download None."""
    from nnseg.errors import ModelNotFound
    monkeypatch.setattr(wf, "_manifest", lambda path=None: {
        "920": {"default": "unversioned", "versions": {"unversioned": {"url": None, "gated": True}}}})
    with pytest.raises(ModelNotFound, match="licensed backend"):
        wf.fetch_one(920, tmp_path)


# -- recording what is installed ------------------------------------------------------------
def _fake_release_zip(tmp_path, folder_name="Dataset297_TotalSegmentator_total_3mm_1559subj"):
    import zipfile
    z = tmp_path / "asset.zip"
    with zipfile.ZipFile(z, "w") as f:
        f.writestr(f"{folder_name}/plans.json", "{}")
        f.writestr(f"{folder_name}/fold_0/checkpoint_final.pth", "weights")
    return z


def _serve(monkeypatch, zip_path):
    """Point urlopen at a local file so fetch_one runs end to end without a network."""
    import contextlib
    data = zip_path.read_bytes()

    class R:
        def read(self, n=-1):
            nonlocal data
            out, data = (data, b"") if n in (-1, None) or n >= len(data) else (data[:n], data[n:])
            return out

    monkeypatch.setattr(wf.urllib.request, "urlopen", lambda *a, **k: contextlib.nullcontext(R()))


def test_fetch_writes_a_sidecar_naming_the_version(tmp_path, monkeypatch):
    import hashlib
    z = _fake_release_zip(tmp_path)
    sha = hashlib.sha256(z.read_bytes()).hexdigest()
    monkeypatch.setattr(wf, "_manifest", lambda path=None: {"297": {
        "default": "v2.0.0-weights",
        "versions": {"v2.0.0-weights": {"url": "https://x/a.zip", "name": "a.zip", "sha256": sha}}}})
    _serve(monkeypatch, z)
    root = tmp_path / "weights"
    dest = wf.fetch_one(297, root)
    rec = wf.installed_version(dest)
    assert rec["tag"] == "v2.0.0-weights" and rec["id"] == "297" and rec["sha256"] == sha
    assert rec["by"] == "nnseg" and "installed" in rec


def test_a_named_version_is_what_gets_recorded(tmp_path, monkeypatch):
    z = _fake_release_zip(tmp_path)
    monkeypatch.setattr(wf, "_manifest", lambda path=None: {"297": {
        "default": "v2.0.0-weights",
        "versions": {"v2.0.0-weights": {"url": "https://x/a.zip"},
                     "v2.0.4-weights": {"url": "https://x/b.zip"}}}})
    _serve(monkeypatch, z)
    dest = wf.fetch_one(297, tmp_path / "w", tag="v2.0.4-weights")
    assert wf.installed_version(dest)["tag"] == "v2.0.4-weights"


def test_weights_we_did_not_install_report_unknown_rather_than_guessing(tmp_path):
    """TotalSegmentator or a human may have put them there; guessing from the manifest would be
    wrong in exactly the case versioning exists for."""
    d = tmp_path / "Dataset297_TotalSegmentator_total_3mm_1559subj"
    d.mkdir()
    assert wf.installed_version(d) is None


def test_a_corrupt_sidecar_is_not_fatal(tmp_path):
    d = tmp_path / "Dataset297_x"; d.mkdir()
    (d / wf.SIDECAR).write_text("{not json")
    assert wf.installed_version(d) is None


# -- current means "what TotalSegmentator installs" -------------------------------------
def test_pins_are_parsed_from_totalsegmentators_config(monkeypatch):
    src = '''
    291: {"foldername": "Dataset291_x", "version": "v2.0.0-weights"},
    297: {
        "foldername": "Dataset297_y",
        "version": "v2.0.0-weights"
    },
    '''
    monkeypatch.setattr(wf.urllib.request, "urlopen",
                        lambda *a, **k: __import__("contextlib").nullcontext(
                            type("R", (), {"read": lambda self: src.encode()})()))
    assert wf.upstream_pins() == {"291": "v2.0.0-weights", "297": "v2.0.0-weights"}


def test_unreachable_pins_fall_back_to_newest_rather_than_failing(monkeypatch):
    def boom(*a, **k):
        raise OSError("no network")
    monkeypatch.setattr(wf.urllib.request, "urlopen", boom)
    assert wf.upstream_pins() == {}          # advisory, never fatal


def test_a_pin_beats_newest_when_choosing_current(fake_github, tmp_path, monkeypatch):
    """297 is published as v2.0.0 and v2.5.0 here; TS pins the older one, so that is current."""
    monkeypatch.setattr(wf, "upstream_pins", lambda repo=None, progress=None: {"297": "v2.0.0"})
    m = tmp_path / "w.json"
    m.write_text(json.dumps({"weights": {}}))
    wf.refresh_manifest(path=m)
    e = json.loads(m.read_text())["weights"]["297"]
    assert e["default"] == "v2.0.0"                  # not v2.5.0, the newest asset
    assert set(e["versions"]) == {"v2.0.0", "v2.5.0"}


def test_without_a_pin_current_falls_back_to_newest(fake_github, tmp_path, monkeypatch):
    monkeypatch.setattr(wf, "upstream_pins", lambda repo=None, progress=None: {})
    m = tmp_path / "w.json"
    m.write_text(json.dumps({"weights": {}}))
    wf.refresh_manifest(path=m)
    assert json.loads(m.read_text())["weights"]["297"]["default"] == "v2.5.0"


def test_our_manifest_agrees_with_totalsegmentator_today():
    """A drift guard: if a refresh ever repoints something away from TS's pin, this fails.
    Uses the local TotalSegmentator clone, and skips where there isn't one."""
    import re
    from pathlib import Path
    cfg = Path(__file__).resolve().parents[2] / "upstream/TotalSegmentator/totalsegmentator/map_tasks_config.py"
    if not cfg.exists():
        pytest.skip("no local TotalSegmentator clone")
    pins = {str(int(m.group(1))): m.group(2) for m in wf.PIN_RE.finditer(cfg.read_text())}
    ours = wf._manifest()
    differ = {k: (ours[k]["default"], pins[k]) for k in set(pins) & set(ours)
              if ours[k]["default"] != pins[k]}
    assert not differ, f"manifest diverges from TotalSegmentator's pins: {differ}"


def test_the_earlier_current_key_still_reads():
    """A manifest written before the rename must keep working."""
    e = wf._normalize({"297": {"current": "v2.0.0", "versions": {"v2.0.0": {"url": "u"}}}})["297"]
    assert e["default"] == "v2.0.0" and "current" not in e
    assert wf.selected(e)["url"] == "u"
