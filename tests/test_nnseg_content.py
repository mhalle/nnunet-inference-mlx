"""Content addressing: what a digest means, and who is allowed to decide it.

The store itself, without a server. The rule under test throughout is that the
SERVER does the addressing - a digest is a measurement of bytes that arrived,
never a label a caller attached to them.
"""
import pytest

from nnseg.content import (BLOB, TREE, ContentStore, DigestMismatch, digest_dir,
                           digest_file, is_digest, tree_digest)
from nnseg.serve import SeriesCache


@pytest.fixture
def store(tmp_path):
    return ContentStore(SeriesCache(tmp_path / "cache", lambda k, e: None))


def _file(d, name, data):
    p = d / name
    p.write_bytes(data)
    return p


def test_a_blob_keeps_exactly_the_identity_uploads_have_always_had(tmp_path, store):
    """`sha256:<hex>` of the bytes - the same string the upload path produced
    before any of this existed, so no cached result changes key."""
    import hashlib
    p = _file(tmp_path, "scan.nii.gz", b"\x1f\x8bvolume")
    assert digest_file(p) == BLOB + hashlib.sha256(b"\x1f\x8bvolume").hexdigest()


def test_storing_the_same_bytes_twice_costs_one_entry(tmp_path, store):
    a = _file(tmp_path, "a.nii.gz", b"identical")
    b = _file(tmp_path, "b.nii.gz", b"identical")      # different name, same content
    assert store.put_file(a) == store.put_file(b)


def test_a_declared_digest_is_checked_against_the_bytes(tmp_path, store):
    p = _file(tmp_path, "a.nii.gz", b"real bytes")
    with pytest.raises(DigestMismatch) as e:
        store.put_file(p, expect=BLOB + "0" * 64)
    assert e.value.actual == digest_file(p)
    assert not store.has(BLOB + "0" * 64)              # nothing stored under the lie


def test_a_tree_digest_does_not_depend_on_order_or_filenames(tmp_path):
    """A DICOM series orders itself from its headers; the names on disk are an
    accident of whoever exported it, and the order of arrival is the client's
    business. Neither may change the identity."""
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(); b.mkdir()
    for i, data in enumerate([b"s1", b"s2", b"s3"]):
        _file(a, f"IM_{i}.dcm", data)
    for i, data in enumerate([b"s3", b"s1", b"s2"]):   # other order, other names
        _file(b, f"zz{9 - i}.dcm", data)
    assert digest_dir(a) == digest_dir(b)
    assert digest_dir(a).startswith(TREE)


def test_a_tree_and_a_blob_of_the_same_bytes_are_different_things(tmp_path, store):
    """One member is still a tree if it was sent as one - the grammar says what
    the reader gets handed, and a directory means DICOM series to SimpleITK."""
    d = tmp_path / "one"; d.mkdir()
    p = _file(d, "only.dcm", b"solo")
    assert store.put_dir(d) != store.put_file(p)


def test_resolve_hands_back_a_file_for_a_blob_and_a_directory_for_a_tree(tmp_path, store):
    p = _file(tmp_path, "scan.nii.gz", b"volume")
    d = tmp_path / "series"; d.mkdir()
    _file(d, "IM0.dcm", b"a"); _file(d, "IM1.dcm", b"b")
    assert store.resolve(store.put_file(p)).is_file()
    assert store.resolve(store.put_dir(d)).is_dir()


def test_tree_members_are_stored_under_content_derived_names(tmp_path, store):
    """Flattened, and named from the BYTES rather than from the request.

    Two containers writing this entry must write the same directory: the volume
    behind it is not POSIX-coherent, and the justification for having no
    cross-container mutex is that identical bytes produce identical writes. Names
    taken from arrival order or from whatever a zip carried made that false for
    the directory even while it held for each file."""
    d = tmp_path / "nested"; (d / "sub").mkdir(parents=True)
    _file(d / "sub", "IM0.dcm", b"a")
    _file(d, "IM1.dcm", b"b")
    where = store.resolve(store.put_dir(d))
    names = sorted(p.name for p in where.iterdir())
    assert len(names) == 2 and not (where / "sub").exists()
    assert names == sorted(digest_file(x).split(":")[1][:32]
                           for x in (d / "sub" / "IM0.dcm", d / "IM1.dcm"))


def test_the_same_series_stores_identically_however_it_arrived(tmp_path, store):
    """The invariant the cross-container design rests on: same content in, same
    directory out - whatever the members were called or what order they came."""
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(); b.mkdir()
    for i, data in enumerate([b"s1", b"s2", b"s3"]):
        _file(a, f"IM_{i}.dcm", data)
    for i, data in enumerate([b"s3", b"s1", b"s2"]):
        _file(b, f"{i}_whatever", data)
    da, db = store.put_dir(a), ContentStore(
        SeriesCache(tmp_path / "other", lambda k, e: None)).put_dir(b)
    assert da == db
    listing = lambda st, dg: sorted(p.name for p in st.resolve(dg).iterdir())
    assert listing(store, da) == listing(
        ContentStore(SeriesCache(tmp_path / "other", lambda k, e: None)), db)


def test_is_digest_separates_content_keys_from_source_ids():
    assert is_digest(BLOB + "a" * 64) and is_digest(TREE + "b" * 64)
    assert not is_digest("idc:a05fb365-dfd2-4116-ab8e-a7262d2c169c")
    assert not is_digest("deadbeef")


def test_an_empty_directory_is_not_a_tree(tmp_path, store):
    d = tmp_path / "empty"; d.mkdir()
    with pytest.raises(FileNotFoundError):
        store.put_dir(d)


def test_tree_digest_is_stable_across_calls():
    members = [BLOB + "a" * 64, BLOB + "b" * 64]
    assert tree_digest(members) == tree_digest(reversed(members))


@pytest.mark.parametrize("ext", [".nii.gz", ".nii", ".nrrd", ".mha"])
def test_the_format_is_read_from_the_content_not_the_name(tmp_path, ext):
    """A content-addressed store keeps the bytes and nothing else, but SimpleITK
    picks its reader from the EXTENSION - so an entry stored under a nameless
    temp file is unreadable however correct its digest is. Found by a live run,
    because a fake segmenter never opens the file it is handed."""
    sitk = pytest.importorskip("SimpleITK")
    import numpy as np

    from nnseg.content import guess_name
    src = tmp_path / f"volume{ext}"
    sitk.WriteImage(sitk.GetImageFromArray(np.zeros((4, 5, 6), np.int16)), str(src))
    anonymous = tmp_path / "tmp9f3a2b"          # what mkstemp hands you
    anonymous.write_bytes(src.read_bytes())
    assert guess_name(anonymous).endswith(ext)


def test_content_we_cannot_identify_is_refused_rather_than_mislabelled(tmp_path):
    from nnseg.content import guess_name
    p = tmp_path / "junk"
    p.write_bytes(b"not a medical image at all")
    assert guess_name(p) is None


def test_a_gzipped_nrrd_does_not_become_a_nifti(tmp_path):
    """Gzip is unwrapped before deciding. Assuming .nii.gz would store a gzipped
    NRRD under a name that cannot read it - rare, and therefore the case nobody
    notices until it fails."""
    import gzip

    from nnseg.content import guess_name
    p = tmp_path / "blob"
    p.write_bytes(gzip.compress(b"NRRD0004\ntype: short\ndimension: 3\n"))
    assert guess_name(p) == "input.nrrd.gz"


def test_two_members_that_flatten_onto_one_name_both_survive(tmp_path):
    """Dropping one would silently change what the tree IS - and a synthesized
    de-duplication name can itself collide with a real member."""
    import zipfile

    from nnseg.content import extract_zip
    z = tmp_path / "s.zip"
    with zipfile.ZipFile(z, "w") as f:
        f.writestr("a/IM1.dcm", b"one")
        f.writestr("b/IM1.dcm", b"two")
        f.writestr("1_IM1.dcm", b"three")        # collides with a naive rename
    out = tmp_path / "x"
    files = extract_zip(z, out)
    assert len(files) == 3
    assert sorted(p.read_bytes() for p in out.iterdir()) == [b"one", b"three", b"two"]


# -- path components: legal everywhere, on every platform ------------------

def test_every_key_grammar_we_generate_is_a_legal_filename(tmp_path):
    """The store turns keys straight into directory names, and its safe-check
    screened separators and NUL but never the COLON - which every key contains
    (idc:<uuid>, sha256:<hex>, sha256-tree:<hex>) and which Windows reserves as
    the Alternate Data Stream separator. The store layer could not have worked
    there."""
    from nnseg.serve import RESERVED_PATH_CHARS, SeriesCache
    cache = SeriesCache(tmp_path / "c", lambda k, e: None)
    keys = [BLOB + "a" * 64, TREE + "b" * 64,
            "idc:a05fb365-dfd2-4116-ab8e-a7262d2c169c",
            "tcia:1.3.6.1.4.1.14519.5.2.1", "zenodo:10.5281/zenodo.123"]
    for k in keys:
        name = cache._entry(k).name
        assert not any(c in name for c in RESERVED_PATH_CHARS), f"{k!r} -> {name!r}"
    assert len({cache._entry(k).name for k in keys}) == len(keys)


def test_the_encoding_is_injective_including_a_literal_percent(tmp_path):
    """`%` is escaped first, or a key containing a literal `%3A` and a key
    containing a real colon would land on one entry - two different inputs
    sharing a cache slot."""
    from nnseg.serve import safe_path_component
    assert safe_path_component("a:b") != safe_path_component("a%3Ab")
    assert safe_path_component("a%3Ab") == "a%253Ab"


def test_names_are_readable_because_finding_a_bad_entry_by_eye_is_real(tmp_path):
    """Hashing would have been simpler. It also would have made `modal volume ls`
    useless for matching a stored entry against a digest the API reported, which
    is how a poisoned entry was actually found and removed."""
    from nnseg.serve import safe_path_component
    assert safe_path_component(BLOB + "2549be").startswith("sha256%3A2549be")


def test_the_layout_does_not_depend_on_the_host(tmp_path):
    """A per-platform rule would make one key map to different paths on
    different hosts, so a copied volume or a restored backup would stop
    resolving. The encoding takes no platform argument, by construction."""
    import inspect

    from nnseg.serve import safe_path_component
    assert not inspect.signature(safe_path_component).parameters.get("os_name")
    assert len(inspect.signature(safe_path_component).parameters) == 1


def test_an_absurdly_long_key_still_falls_back_to_a_hash(tmp_path):
    from nnseg.serve import SeriesCache
    cache = SeriesCache(tmp_path / "c", lambda k, e: None)
    assert cache._entry("idc:" + "x" * 400).name.startswith("h_")
