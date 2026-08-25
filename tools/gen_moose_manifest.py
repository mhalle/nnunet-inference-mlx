"""Regenerate src/nnseg/data/moose_weights.json from a moosez checkout.

The checkpoint is the spec (labels come from each model's own dataset.json at
install time); this manifest holds only what the checkpoint cannot know -
the task name, where to download it, and the release tag parsed from the
asset filename. Run after updating upstream/MOOSE.
"""
import json
import re
import sys
from pathlib import Path

def parse(models_py: Path) -> dict:
    src = models_py.read_text()
    out = {}
    for m in re.finditer(
            r'"(?P<name>[a-z0-9_]+)"\s*:\s*\{\s*'
            r'KEY_URL:\s*"(?P<url>[^"]+)"\s*,\s*'
            r'KEY_FOLDER_NAME:\s*"(?P<folder>[^"]+)"', src):
        name, url, folder = m.group("name", "url", "folder")
        tag = (re.search(r"_(\d{8})\.zip$", url) or re.search(r"v[\d.]+", url))
        out[name] = {"url": url, "folder": folder,
                     "tag": tag.group(0).strip("_.zip") if tag else "unknown"}
    return out

if __name__ == "__main__":
    moose = Path(sys.argv[1] if len(sys.argv) > 1 else
                 "../../upstream/MOOSE") / "moosez/models.py"
    dest = Path(__file__).parent.parent / "src/nnseg/data/moose_weights.json"
    entries = parse(moose)
    dest.write_text(json.dumps({"source": "moosez/models.py", "tasks": entries},
                               indent=1) + "\n")
    print(f"{len(entries)} models -> {dest}")
