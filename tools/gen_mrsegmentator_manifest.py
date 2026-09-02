"""Regenerate src/nnseg/data/mrsegmentator_weights.json from upstream MRSegmentator.

The checkpoint is the spec (labels come from each model's own dataset.json at
install time); this manifest holds only what the checkpoint cannot know - the
task name, where to download it, its digest and version - plus the folder the
flat zip must be installed under, which nnseg needs *before* it can read the
checkpoint. That folder is read FROM the zip without downloading it: two Range
requests fetch the central directory and the small members (plans.json for the
dataset name, version.json for the version, the head of fold_0's checkpoint for
the trainer name), so a 1.1 GB asset costs a few MB to describe.

Usage (stdlib only):  uv run --no-project python tools/gen_mrsegmentator_manifest.py
                      [path/to/MRSegmentator/src/mrsegmentator/config.py]
"""
import ast
import json
import re
import struct
import sys
import urllib.request
import zlib
from pathlib import Path

UPSTREAM_CONFIG = ("https://raw.githubusercontent.com/hhaentze/MRSegmentator/master/"
                   "src/mrsegmentator/config.py")
DEST = Path(__file__).parent.parent / "src/nnseg/data/mrsegmentator_weights.json"


def parse_registry(config_py: str) -> dict:
    """MODEL_REGISTRY = {name: {version, url, sha256, ...}} out of config.py, by AST."""
    tree = ast.parse(config_py)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "MODEL_REGISTRY" for t in node.targets):
            return ast.literal_eval(node.value)
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) \
                and node.target.id == "MODEL_REGISTRY":
            return ast.literal_eval(node.value)
    raise SystemExit("MODEL_REGISTRY not found in config.py")


def _fetch(url: str, start: int, end: int) -> bytes:
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def _central_directory(url: str) -> dict:
    """{member name: (method, compressed size, local header offset)} via two Range reads."""
    with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=120) as r:
        total = int(r.headers["Content-Length"])
    tail = _fetch(url, max(0, total - 66000), total - 1)
    i = tail.rfind(b"PK\x05\x06")
    _, cd_size, cd_off = struct.unpack("<HII", tail[i + 10:i + 20])
    if cd_off == 0xFFFFFFFF:                                  # zip64
        j = tail.rfind(b"PK\x06\x06")
        cd_size, cd_off = struct.unpack("<QQ", tail[j + 40:j + 56])
    cd = _fetch(url, cd_off, cd_off + cd_size - 1)
    out, p = {}, 0
    while p < len(cd) and cd[p:p + 4] == b"PK\x01\x02":
        method, = struct.unpack("<H", cd[p + 10:p + 12])
        csize, usize = struct.unpack("<II", cd[p + 20:p + 28])
        nlen, elen, clen = struct.unpack("<HHH", cd[p + 28:p + 34])
        off, = struct.unpack("<I", cd[p + 42:p + 46])
        name = cd[p + 46:p + 46 + nlen].decode()
        extra = cd[p + 46 + nlen:p + 46 + nlen + elen]
        if 0xFFFFFFFF in (csize, usize, off):
            q = 0
            while q < len(extra):
                hid, hsz = struct.unpack("<HH", extra[q:q + 4])
                if hid == 1:
                    vals = list(struct.unpack("<" + "Q" * (hsz // 8), extra[q + 4:q + 4 + hsz - hsz % 8]))
                    if usize == 0xFFFFFFFF: usize = vals.pop(0)
                    if csize == 0xFFFFFFFF: csize = vals.pop(0)
                    if off == 0xFFFFFFFF: off = vals.pop(0)
                q += 4 + hsz
        out[name] = (method, csize, off)
        p += 46 + nlen + elen + clen
    return out


def _member_head(url: str, cd: dict, name: str, limit: int | None = None) -> bytes:
    method, csize, off = cd[name]
    lh = _fetch(url, off, off + 29)
    nlen, elen = struct.unpack("<HH", lh[26:30])
    start = off + 30 + nlen + elen
    data = _fetch(url, start, start + (csize if limit is None else min(csize, limit)) - 1)
    if method == 8:
        return zlib.decompressobj(-15).decompress(data)
    return data


def describe_zip(url: str) -> dict:
    cd = _central_directory(url)
    names = set(cd)
    if "plans.json" not in names or "dataset.json" not in names:
        raise SystemExit(f"{url}: not a flat nnU-Net configuration folder ({sorted(names)[:8]}...)")
    plans = json.loads(_member_head(url, cd, "plans.json"))
    version = json.loads(_member_head(url, cd, "version.json")) if "version.json" in names else {}
    ckpt = next(n for n in sorted(names) if re.fullmatch(r"fold_\w+/checkpoint_final\.pth", n))
    head = _member_head(url, cd, ckpt, limit=3_000_000)      # data.pkl is the first member
    # the value is a pickle BINUNICODE: opcode 'X', a 4-byte little-endian length, the bytes -
    # read exactly that many, or the memo opcode that follows rides along as garbage
    m = re.search(rb"trainer_name.{0,40}?X(.{4})", head, re.DOTALL)
    if not m:
        raise SystemExit(f"{url}: trainer name not found in the head of {ckpt}")
    n, = struct.unpack("<I", m.group(1))
    trainer = head[m.end():m.end() + n].decode()
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", trainer):
        raise SystemExit(f"{url}: implausible trainer name {trainer!r}")
    configs = [c for c, v in plans["configurations"].items()
               if v.get("patch_size") and len(v["patch_size"]) == 3 and not v.get("previous_stage")]
    config = "3d_fullres" if "3d_fullres" in configs else configs[0]
    folds = sorted(n.split("/")[0] for n in names if re.fullmatch(r"fold_\w+/checkpoint_final\.pth", n))
    return {"folder": f"{plans['dataset_name']}/{trainer}__{plans['plans_name']}__{config}",
            "weights_version": version.get("weights_version"), "folds": folds,
            "structures": sum(1 for v in json.loads(_member_head(url, cd, "dataset.json"))["labels"].values()
                              if v != 0)}


def main(argv: list) -> None:
    if argv:
        config_py = Path(argv[0]).read_text()
        source = str(argv[0])
    else:
        with urllib.request.urlopen(UPSTREAM_CONFIG, timeout=60) as r:
            config_py = r.read().decode()
        source = UPSTREAM_CONFIG
    registry = parse_registry(config_py)
    tasks = {}
    for name, cfg in registry.items():
        if not cfg.get("url"):
            print(f"skip {name}: no download URL", file=sys.stderr)
            continue
        print(f"describing {name} from {cfg['url']} ...", file=sys.stderr)
        d = describe_zip(cfg["url"])
        tag = str(d["weights_version"] if d["weights_version"] is not None else cfg.get("version"))
        if str(cfg.get("version")) != tag:
            print(f"  note: config.py says version {cfg.get('version')}, the zip says {tag}; "
                  "the zip wins (it is what ensure() verifies on disk)", file=sys.stderr)
        rel = re.search(r"/releases/download/([^/]+)/", cfg["url"])
        zen = re.search(r"zenodo\.org/records?/(\d+)", cfg["url"])
        tasks[name] = {"url": cfg["url"], "sha256": cfg.get("sha256"), "tag": tag,
                       "folder": d["folder"],
                       "release": rel.group(1) if rel else (f"zenodo:{zen.group(1)}" if zen else None),
                       "structures": d["structures"]}
        print(f"  {d['folder']}  folds={d['folds']}  structures={d['structures']}", file=sys.stderr)
    DEST.write_text(json.dumps({"source": f"{source} MODEL_REGISTRY + each zip's own plans.json / "
                                          "version.json / fold_0 checkpoint",
                                "tasks": tasks}, indent=1) + "\n")
    print(f"{len(tasks)} models -> {DEST}")


if __name__ == "__main__":
    main(sys.argv[1:])
