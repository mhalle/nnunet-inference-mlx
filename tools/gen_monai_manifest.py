"""Regenerate src/nnseg/data/monai_bundles.json from the MONAI model zoo.

The bundle is the spec - labels come from each installed bundle's own
``configs/metadata.json`` (``network_data_format.outputs.pred.channel_def``) at
describe time, never from here, which is the same rule that keeps the MOOSE
catalog from drifting. This manifest holds only what you cannot know before
installing: where to download a bundle, its version, and its checksum - plus a
little listing metadata (modality, label count, task line) so ``/v1/tasks`` can
say something useful about a task whose weights are not installed yet.

Two jobs beyond fetching:

* **Filter.** The zoo is not all 3D segmentation - it also ships classification,
  detection, generative, 2D pathology and templates. Only bundles that declare a
  volumetric segmentation output survive.
* **Deduplicate.** ``model_info.json`` is ~500 entries of <bundle>_v<version>;
  keep the newest version per bundle (older ones stay reachable via ``@version``).

    uv run --no-project python tools/gen_monai_manifest.py [--all]

``--all`` keeps every bundle that passes the filter; the default keeps the
curated list below, because a bundle we have never run is not a task we should
offer (each carries its own ``required_packages_version``, and the image has to
carry the union - see medseg/docs/monai-bundles.md).
"""
from __future__ import annotations

import json
import re
import sys
import urllib.request
from pathlib import Path

ZOO_INFO = ("https://raw.githubusercontent.com/Project-MONAI/model-zoo/dev/"
            "models/model_info.json")
METADATA = ("https://raw.githubusercontent.com/Project-MONAI/model-zoo/dev/"
            "models/{bundle}/configs/metadata.json")

#: Bundles we actually intend to serve. Curated rather than "everything that
#: parses": each bundle pins its own dependency versions, the image carries the
#: union, and an untested bundle is not a task worth offering.
CURATED = (
    "spleen_ct_segmentation",             # small, single organ - the smoke case
    "wholeBody_ct_segmentation",          # 104 structures; the ts:total cross-check
    "pancreas_ct_dints_segmentation",
    "swin_unetr_btcv_segmentation",
    "wholeBrainSeg_Large_UNEST_segmentation",
)
# Deliberately NOT curated: brats_mri_segmentation wants 4 co-registered MR
# channels, and the job wire takes a single input and refuses multi-channel at
# submit. A bundle we cannot serve is not a task worth listing. The ecosystem
# still reports `channel_names`, so if a multi-channel bundle is curated later it
# is refused at submit with a clear message rather than deep inside the model.


def fetch_json(url: str):
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.loads(r.read())


def newest_versions(info: dict) -> dict:
    """``{bundle: (version, entry)}`` keeping the highest version per bundle."""
    out: dict[str, tuple[tuple, str, dict]] = {}
    for key, entry in info.items():
        m = re.match(r"^(?P<name>.+)_v(?P<ver>[0-9][0-9.]*)$", key)
        if not m:
            continue
        name, ver = m.group("name"), m.group("ver")
        sortable = tuple(int(p) if p.isdigit() else 0 for p in ver.split("."))
        if name not in out or sortable > out[name][0]:
            out[name] = (sortable, ver, entry)
    return {n: (v, e) for n, (_, v, e) in out.items()}


def segmentation_facts(meta: dict) -> dict | None:
    """Listing facts for a 3D segmentation bundle, or None if it is not one."""
    fmt = meta.get("network_data_format") or {}
    out = (fmt.get("outputs") or {}).get("pred") or {}
    inp = (fmt.get("inputs") or {}).get("image") or {}
    channel_def = out.get("channel_def")
    if not isinstance(channel_def, dict) or len(channel_def) < 2:
        return None                                  # not a labelled segmentation
    shape = inp.get("spatial_shape")
    if not (isinstance(shape, list) and len(shape) == 3):
        return None                                  # not volumetric
    return {"modality": inp.get("modality"),
            "in_channels": inp.get("num_channels"),
            "n_labels": len(channel_def),
            "task": (meta.get("task") or "").strip(),
            "monai_version": meta.get("monai_version"),
            "required_packages": meta.get("required_packages_version") or {}}


def build(keep_all: bool = False) -> dict:
    info = fetch_json(ZOO_INFO)
    latest = newest_versions(info)
    wanted = sorted(latest) if keep_all else [b for b in CURATED if b in latest]
    missing = [] if keep_all else [b for b in CURATED if b not in latest]
    bundles, skipped = {}, []
    for name in wanted:
        version, entry = latest[name]
        try:
            meta = fetch_json(METADATA.format(bundle=name))
        except Exception as e:                       # a bundle without metadata in the repo
            skipped.append(f"{name}: metadata unavailable ({type(e).__name__})")
            continue
        facts = segmentation_facts(meta)
        if facts is None:
            skipped.append(f"{name}: not a 3D labelled segmentation bundle")
            continue
        # The zoo has moved hosting: recent entries' "source" is a Hugging Face
        # REPO PAGE, not a downloadable archive (and they publish no checksum).
        # monai.bundle.download is what knows how to resolve each host, so record
        # WHICH host rather than a URL we would have to fetch ourselves.
        url = entry.get("source") or ""
        host = ("huggingface_hub" if "huggingface.co" in url
                else "github" if "github.com" in url
                else "monaihosting")
        bundles[name] = {"version": version, "source": host, "url": url,
                         "checksum": entry.get("checksum"), **facts}
    return {"source": "Project-MONAI/model-zoo models/model_info.json",
            "note": ("labels are read from each installed bundle's own metadata.json; "
                     "this manifest holds only download + listing facts"),
            "bundles": bundles, "skipped": skipped, "curated_missing": missing}


if __name__ == "__main__":
    data = build(keep_all="--all" in sys.argv)
    dest = Path(__file__).parent.parent / "src/nnseg/data/monai_bundles.json"
    dest.write_text(json.dumps(data, indent=1, sort_keys=False) + "\n")
    print(f"wrote {dest} with {len(data['bundles'])} bundles")
    for line in data["skipped"]:
        print("  skipped:", line)
    for name in data["curated_missing"]:
        print("  MISSING from the zoo:", name)
    pkgs = sorted({p for b in data["bundles"].values() for p in b["required_packages"]})
    print("  union of required packages:", ", ".join(pkgs) or "(none)")
    # The zoo publishes checksums for most historical entries but omits them on
    # recent releases; without one the download cannot be verified, so say which.
    unverifiable = [n for n, b in data["bundles"].items() if not b.get("checksum")]
    if unverifiable:
        print(f"  NO CHECKSUM published (download cannot be verified) for "
              f"{len(unverifiable)}/{len(data['bundles'])}: {', '.join(unverifiable)}")
