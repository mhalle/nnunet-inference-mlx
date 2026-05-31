#!/usr/bin/env python
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "totalsegmentator",
# ]
# ///
"""Generate src/nnunet_inference_mlx/data/ts_weights.json from TotalSegmentator.

The download URLs live in ``totalsegmentator.libs.download_pretrained_weights``
as a flat ``if task_id == N: ... WEIGHTS_URL = url + "/.../DatasetN_...zip"``
chain (most assets are public GitHub release zips; a few tasks fetch via a
license server — those have no public URL and are marked ``gated``). We read
that function's *source* and regex out the active (uncommented) assignment per
id — TS is never imported at our runtime, only here at build time (the same
relationship ``ts_tasks.json`` has).

    uv run --with totalsegmentator python scripts/refresh_ts_weights.py

SHA-256 is left null: TS does not publish per-asset digests. Verification is
therefore skipped at download time until we record digests ourselves.
"""

from __future__ import annotations

import inspect
import json
import re
from pathlib import Path

import totalsegmentator
from totalsegmentator import libs

OUT = Path(__file__).resolve().parents[1] / "src/nnunet_inference_mlx/data/ts_weights.json"


def main() -> None:
    src = inspect.getsource(libs.download_pretrained_weights).splitlines()

    base = None
    for line in src:
        s = line.strip()
        m = re.match(r'url\s*=\s*"(https://\S+?)"', s)
        if m and not s.startswith("#"):
            base = m.group(1)

    weights: dict[int, dict] = {}
    cur: list[int] = []
    for line in src:
        s = line.strip()
        m = re.match(r"(?:el)?if\s+task_id\s*==\s*(\d+)\s*:", s)
        mlist = re.match(r"(?:el)?if\s+task_id\s+in\s*\[([\d,\s]+)\]\s*:", s)
        if m:
            cur = [int(m.group(1))]
        elif mlist:
            cur = [int(x) for x in mlist.group(1).split(",") if x.strip()]
        if not cur or s.startswith("#"):
            continue
        if "download_model_with_license_and_unpack" in s:
            for i in cur:
                weights[i] = {"url": None, "gated": True}
            continue
        mu = re.match(r"WEIGHTS_URL\s*=\s*(.+)", s)
        if not mu:
            continue
        rhs = mu.group(1)
        if "TODO" in rhs:
            continue
        mcat = re.match(r'url\s*\+\s*"([^"]+)"', rhs)
        mlit = re.match(r'"([^"]+)"', rhs)
        url = (base + mcat.group(1)) if (mcat and base) else (mlit.group(1) if mlit else None)
        if url:
            for i in cur:
                weights[i] = {"url": url, "sha256": None}

    payload = {
        "_meta": {
            "schema_version": 1,
            "ts_version": getattr(totalsegmentator, "__version__", "unknown"),
            "generator": "scripts/refresh_ts_weights.py",
            "note": "id→download URL extracted from libs.download_pretrained_weights; "
                    "sha256 null (TS publishes none); gated=license-server only.",
        },
        "weights": {str(k): weights[k] for k in sorted(weights)},
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    pub = sum(1 for v in weights.values() if v.get("url"))
    print(f"wrote {OUT} — {len(weights)} datasets ({pub} public, "
          f"{len(weights) - pub} gated)")


if __name__ == "__main__":
    main()
