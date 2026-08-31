# /// script
# requires-python = ">=3.11"
# dependencies = ["zarr>=3"]
# ///
"""Rename the IDC case to `idc-torso1` and record the provenance that makes the name checkable.

`chest.nii` covers 709 mm ending well below the diaphragm - its study description is
"06. Chest, Abdomen CE" - so the name described a body region the scan is not. It is now named
after the archive it came from, which stays true.

**CT_Abdo keeps its name.** It is also a torso rather than an abdomen (2.5 L of lung, 497 mL of
heart, reaching colon and kidneys), and its upstream file is named for a cardiac angiogram it
does not match either, but renaming it was not asked for. Its store still gains a provenance
block, because recording what a case actually is costs nothing and is separate from naming it.

Files keep working under their old names: the artifacts are renamed and the old names become
symlinks, because ~19 files in the repo - including recorded benchmark logs and CLAUDE.md's
established facts - refer to `chest.nii`, and those logs recorded a run against a file of that
name. Rewriting them would falsify the record; dangling them would be worse.

Idempotent: re-running after a `rebuild_duckn.py` re-emit restores the provenance blocks.
"""
import os
import sys
from pathlib import Path

import zarr

# The file renames below already ran once and are idempotent; DATA only matters if they have
# not. DEMO is where the stores live and is the argument that actually gets used day to day.
DATA = Path(os.environ.get("NNSEG_DEMO_DATA", "~/tmp/data")).expanduser()
DEMO = (Path(sys.argv[1]) if len(sys.argv) > 1
        else Path(__file__).resolve().parent.parent / "data" / "duckn_demo")

IDC = {
    "case": "idc-torso1",
    "archive": "NCI Imaging Data Commons",
    # `cptac_ccrcc` (clear cell renal cell carcinoma), read from the IDC index - NOT "CPTAC-3",
    # which is a different designation and was recorded here in error. A wrong collection name
    # sends anyone trying to locate the case to the wrong cohort.
    "collection": "cptac_ccrcc",
    "patient_id": "C3N-01524",
    "license": "CC BY 4.0",
    "source_doi": "10.7937/k9/tcia.2018.oblamn27",
    "crdc_series_uuid": "a05fb365-dfd2-4116-ab8e-a7262d2c169c",
    "idc_version": 2,
    "study_description": "06. Chest, Abdomen CE",
    "series_description": "NEPHROGENIC",
    "study_date": "20090728",
    "modality": "CT",
    "manufacturer": "Philips",
    "model": "Brilliance 64",
    "slice_thickness_mm": 2.0,
    "instances": 709,
    "study_instance_uid":
        "1.3.6.1.4.1.14519.5.2.1.2932.1975.277486652714623414151775226101",
    "series_instance_uid":
        "1.3.6.1.4.1.14519.5.2.1.2932.1975.255072988367557196694880426160",
    "frame_of_reference_uid":
        "1.3.6.1.4.1.14519.5.2.1.2932.1975.712737093675822540969939888973",
}

SLICER = {
    "case": "CT_Abdo",
    "archive": "3D Slicer sample data",
    "source_url": "https://www.slicer.org/wiki/File:CTA-cardio.nrrd",
    "modality": "CT",
    "note": "Provenance is the NIfTI `descrip` / `ITK_FileNotes` fields only - no DICOM, no "
            "series UID, no patient identifier, and NOT IDC-derived. The filename is a label, "
            "not a description: the volume is a torso, and the upstream file is named for a "
            "cardiac angiogram whose coverage it does not match either.",
}

NLST = {
    "case": "nlst-217076",
    "archive": "NCI Imaging Data Commons",
    "collection": "nlst",
    "study_description": "NLST-ACRIN",
    "patient_id": "217076",
    "clinical_trial_subject_id": "12540",
    "study_date": "20000102",
    "modality": "CT",
    "body_part_examined": "CHEST",
    "manufacturer": "GE MEDICAL SYSTEMS",
    "model": "LightSpeed QX/i",
    "convolution_kernel": "STANDARD",
    "kvp": 120,
    "series_number": 3,
    "instances": 249,
    "slice_thickness_mm": 1.25,
    "slice_spacing_mm": 1.25,          # contiguous: thickness == spacing, no gap or overlap
    "pixel_spacing_mm": [0.625, 0.625],
    "license": "CC BY 4.0",
    "source_doi": "10.7937/tcia.hmq8-j677",
    "crdc_series_uuid": "4682f41a-65d7-4a7b-8050-952f73abb746",
    "study_instance_uid":
        "1.3.6.1.4.1.14519.5.2.1.7009.9004.247519293920368460447871591111",
    "series_instance_uid":
        "1.3.6.1.4.1.14519.5.2.1.7009.9004.267775549148804835566347044610",
    "frame_of_reference_uid":
        "1.3.6.1.4.1.14519.5.2.1.7009.9004.318987458734756816915383525170",
    # An independent reference segmentation of THIS series, already published in IDC. Recorded
    # rather than fetched: it is 190 MB and nothing needs it yet, but a store that cannot say
    # which reference applies to it is a store nobody can check later. The SEG names its source
    # as "Series 3", which is this series.
    "reference_segmentation": {
        "producer": "TotalSegmentator v1.5.6",
        "of_series_number": 3,
        "segmentation_crdc_series_uuid": "0c3c5072-c8cc-4772-8f52-bca27b3972b8",
        "shape_measurements_crdc_series_uuid": "20f36010-d22a-41d9-9957-44da1539cc19",
        "firstorder_measurements_crdc_series_uuid":
            "fd8bbfbf-3ac8-4765-8b23-d88865b39b52",
    },
}

OPENNEURO = {
    "case": "ds000114_sub-01",
    "archive": "OpenNeuro",
    "dataset": "ds000114",
    "dataset_doi": "10.18112/openneuro.ds000114.v1.0.1",
    "dataset_name": "A test-retest fMRI dataset for motor, language and spatial "
                    "attention functions.",
    "path": "ds000114/sub-01/ses-test/anat/sub-01_ses-test_T1w.nii.gz",
    "subject": "sub-01",
    "session": "ses-test",
    "modality": "MR",
    "contrast": "T1w",
    "license": "CC0",
    "cite": "http://www.ncbi.nlm.nih.gov/pmc/articles/PMC3641991/",
}

RENAMES = [("chest.nii", "idc-torso1.nii"),
           ("chest.zarr.zip", "idc-torso1.zarr.zip"),
           ("chest_nnseg_1.5mm.zarr.zip", "idc-torso1_nnseg_1.5mm.zarr.zip"),
           ("chest_dicom", "idc-torso1_dicom"),
           ("chest.zmp", "idc-torso1.zmp")]

print(f"data artifacts ({DATA})")
for old, new in RENAMES:
    o, n = DATA / old, DATA / new
    if n.exists() and o.is_symlink():
        print(f"  {old:<32} already renamed")
        continue
    if not o.exists():
        print(f"  {old:<32} MISSING - skipped")
        continue
    o.rename(n)
    os.symlink(new, o)                       # relative: the old path keeps resolving
    print(f"  {old:<32} -> {new}   (old name kept as symlink)")


# duckn's `sources` schema is deliberately compact - type, format, path, url, doi, identifier,
# description, created, note - with no slot for a study UID or a scanner model. So the standard
# fields go there, where any duckn reader finds them, and the rest stays namespaced under
# `nnseg.case` rather than being crammed into `note` or silently extending someone else's schema.
DUCKN_SOURCE_KEYS = ("type", "format", "path", "url", "doi", "identifier", "description",
                     "created", "note")


def as_duckn_source(prov):
    """The case block -> one duckn provenance source entry."""
    src = {"type": "dataset", "format": prov.get("format", "DICOM"),
           "description": prov.get("archive", "")}
    for k in ("doi", "url", "note"):
        if k in prov:
            src[k] = prov[k]
    if "source_url" in prov:
        src["url"] = prov["source_url"]
    if "source_doi" in prov:
        src["doi"] = prov["source_doi"]
    src["identifier"] = (prov.get("crdc_series_uuid") or prov.get("series_instance_uid")
                         or prov.get("path") or prov["case"])
    bits = [prov.get("archive"), prov.get("collection") or prov.get("dataset"),
            f"patient {prov['patient_id']}" if "patient_id" in prov else None,
            f"subject {prov['subject']}" if "subject" in prov else None,
            prov.get("modality"), prov.get("study_description")]
    src["description"] = ", ".join(b for b in bits if b)
    if "study_date" in prov:
        d = str(prov["study_date"])
        src["created"] = f"{d[:4]}-{d[4:6]}-{d[6:]}" if len(d) == 8 else d
    return {k: v for k, v in src.items() if k in DUCKN_SOURCE_KEYS and v not in (None, "")}


def stamp(store_name, prov, new_name=None):
    """Fill in the duckn provenance the builder left open, and the namespaced case detail."""
    d = DEMO / store_name
    if not d.exists():
        print(f"  {store_name:<32} MISSING - skipped")
        return
    if new_name:
        d.rename(DEMO / new_name)
        d = DEMO / new_name
    root = zarr.open_group(str(d), mode="r+")
    a = dict(root.attrs.asdict()["duckn"])
    ext = dict(a["extensions"])
    ext["nnseg"] = dict(ext["nnseg"]) | {"case": prov["case"], "case_detail": prov}
    # the builder wrote `processing`; add the source and the licence without discarding it
    pv = dict(ext.get("provenance") or {"version": "1.0"})
    pv["sources"] = [as_duckn_source(prov)]
    if prov.get("license"):
        pv["attribution"] = {"license": prov["license"],
                             **({"cite": prov["cite"]} if prov.get("cite") else {})}
    ext["provenance"] = pv
    a["extensions"] = ext
    root.attrs["duckn"] = a
    steps = len(pv.get("processing", []))
    print(f"  {d.name:<26} case={prov['case']:<18} sources=1 processing={steps} "
          f"archive={prov['archive']}")


print("\ndemo stores")
BY_SUBJECT = {"idc-torso1": IDC, "nlst-217076": NLST, "CT_Abdo": SLICER,
              "ds000114_sub-01": OPENNEURO}

for _d in sorted(DEMO.glob("*/*.duckn")):
    prov = BY_SUBJECT.get(_d.parent.name)
    if prov is None:
        print(f"  {_d.parent.name}/{_d.name}: no provenance registered - skipped")
        continue
    stamp(str(_d.relative_to(DEMO)), prov)

print("\nverify")
for n in sorted(str(p.relative_to(DEMO)) for p in DEMO.glob("*/*.duckn")):
    r = zarr.open_group(str(DEMO / n), mode="r")
    e = r.attrs.asdict()["duckn"]["extensions"]
    print(f"  {n}")
    print(f"    nnseg.case      {e['nnseg']['case']}")
    print(f"    provenance keys {sorted(e['provenance'])}")
    print(f"    parts           {sorted(k for k, _ in r['parts'].groups())}")
print(f"\n  {DATA/'chest.nii'} -> {os.readlink(DATA/'chest.nii')}")
