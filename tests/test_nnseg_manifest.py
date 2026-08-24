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
    assert found["305"]["sha256"] == "bbb"
    assert "sha256" not in found["291"]                 # absent digest is simply omitted


def test_the_newest_release_wins_when_a_dataset_appears_twice(fake_github):
    found = wf.discover_release_assets()
    assert found["297"]["tag"] == "v2.5.0"              # 2025 beats 2023
    assert found["297"]["name"] == "Dataset297_total_3mm_v204.zip"


def test_refresh_adds_missing_but_leaves_existing_alone(fake_github, tmp_path, monkeypatch):
    """Repointing a dataset at a newer release changes which weights download, and therefore the
    segmentations. That must not happen silently."""
    m = tmp_path / "w.json"
    m.write_text(json.dumps({"weights": {"297": {"url": "https://x/v2.0.0/Dataset297_total_3mm.zip"}}}))
    monkeypatch.setattr(wf, "_manifest", lambda: json.loads(m.read_text())["weights"])
    r = wf.refresh_manifest(path=m)
    saved = json.loads(m.read_text())["weights"]
    assert set(r["added"]) == {"291", "305"}
    assert set(r["newer_upstream"]) == {"297"}
    assert saved["297"]["url"].endswith("v2.0.0/Dataset297_total_3mm.zip")   # untouched
    assert "291" in saved and "305" in saved


def test_update_existing_opts_in_to_repointing(fake_github, tmp_path, monkeypatch):
    m = tmp_path / "w.json"
    m.write_text(json.dumps({"weights": {"297": {"url": "https://x/v2.0.0/Dataset297_total_3mm.zip"}}}))
    monkeypatch.setattr(wf, "_manifest", lambda: json.loads(m.read_text())["weights"])
    wf.refresh_manifest(path=m, update_existing=True)
    assert json.loads(m.read_text())["weights"]["297"]["tag"] == "v2.5.0"


def test_dry_run_writes_nothing(fake_github, tmp_path, monkeypatch):
    m = tmp_path / "w.json"
    before = json.dumps({"weights": {}})
    m.write_text(before)
    monkeypatch.setattr(wf, "_manifest", lambda: {})
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
