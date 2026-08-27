"""Content-addressed storage for inputs: upload once, refer by digest thereafter.

Remote sources have always been shared - two jobs wanting the same IDC series
download it once, because :class:`~nnseg.serve.SeriesCache` keys on the series
identity. Uploads had no such story: they were written into the job directory and
died with it, so the same volume submitted for a second task was re-sent and
re-stored. Multi-input made that four times worse.

This closes the gap **on the seam that already exists** rather than beside it. A
``SeriesCache`` is not really series-specific: it is a keyed store of content
directories with LRU eviction, refcounted pins, and atomic single-writer claims
whose failure modes have been paid for once already. So an upload becomes another
entry in it, under a key that is the digest of its own bytes, and every job -
upload or fetch - resolves its inputs the same way.

**The server does the addressing.** A client may say which digest it *expects*,
and that claim is checked against the bytes as they arrive; it is never taken as
the identity. Trusting it would let a caller store bytes Y under digest X and
silently poison every later job that refers to X - and, because the digest feeds
the result-cache key, every memoized result along with it.

Two key grammars, and the difference is load-bearing:

* ``sha256:<hex>`` - one file. **Exactly today's upload identity**, so every
  existing cache key keeps its value and nothing has to be migrated.
* ``sha256-tree:<hex>`` - a directory, which is what a DICOM series is. The root
  is taken over the sorted digests of the members, so it does not depend on the
  order they arrived, on filenames (a DICOM series orders itself from its
  headers, not its filenames), or on the zip metadata of whatever carried them.
  Hashing a zip's bytes would have been simpler and wrong: the same series zipped
  twice differs in timestamps and member order, so dedupe would quietly stop
  working while appearing to work.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

BLOB = "sha256:"
TREE = "sha256-tree:"
_CHUNK = 1 << 20


def is_digest(key) -> bool:
    """Whether ``key`` is one of our content keys (as opposed to a source id)."""
    k = str(key)
    return k.startswith(BLOB) or k.startswith(TREE)


def digest_file(path) -> str:
    """``sha256:<hex>`` of one file's bytes."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(_CHUNK):
            h.update(chunk)
    return BLOB + h.hexdigest()


def tree_digest(member_digests) -> str:
    """``sha256-tree:<hex>`` over a directory's member digests.

    Sorted before hashing, so the root is a property of the CONTENT and not of
    the order the members arrived in. Filenames are deliberately excluded: a
    DICOM series is ordered by its headers, and the names on disk are an accident
    of whoever exported it.
    """
    h = hashlib.sha256()
    for d in sorted(member_digests):
        h.update(d.encode())
        h.update(b"\n")
    return TREE + h.hexdigest()


def digest_dir(path) -> str:
    """``sha256-tree:<hex>`` of every file under ``path``."""
    return tree_digest(digest_file(p) for p in Path(path).rglob("*") if p.is_file())


class DigestMismatch(ValueError):
    """The bytes that arrived are not the ones the caller said they were."""

    def __init__(self, expected: str, actual: str):
        super().__init__(f"content digest {actual} does not match the declared "
                         f"{expected}; the bytes were discarded")
        self.expected, self.actual = expected, actual


class ContentStore:
    """Inputs addressed by the digest of their own bytes.

    Backed by the same keyed store that stages fetched series, so there is one
    root, one LRU budget, one pin discipline - and an entry is an entry, whether
    a client uploaded it or the server fetched it.
    """

    def __init__(self, cache, *, commit=None, refresh=None, lock=None):
        import contextlib
        self.cache = cache
        # A refresh of a shared backing store can DISCARD writes that are on disk
        # but not yet published - so a store that needs commit/reload also needs
        # the write and its commit to be one indivisible step against any reload.
        # This is the same hazard the jobs volume already guards against, and the
        # reason the hook is a lock rather than a flag.
        self._lock = lock or contextlib.nullcontext()
        # Hooks for a backing store that is not a plain local filesystem. Modal
        # Volumes need an explicit commit to publish a write and an explicit
        # reload to see someone else's - so they are passed in rather than
        # imported, and this module keeps knowing nothing about Modal.
        self._commit = commit
        self._refresh = refresh

    def has(self, digest: str) -> bool:
        """Whether the content is here.

        Optimistic: a local hit answers immediately, and only a MISS pays for
        refreshing a shared backing store. That ordering matters - this is called
        on every submit that refers to a digest, and a refresh per call would put
        a network round trip in front of the common case.
        """
        if self.cache.has(digest):
            return True
        if self._refresh is not None:
            with self._lock:
                self._refresh()
            return self.cache.has(digest)
        return False

    def resolve(self, digest: str) -> Path:
        """What to hand the reader: the FILE for a blob, the DIRECTORY for a tree.

        Driven by the key grammar rather than by looking at what is inside. The
        reader dispatches on this - a directory means "DICOM series" to
        SimpleITK - so guessing from the entry's shape would turn a one-slice
        series into a single-file read.
        """
        if not self.cache.has(digest):
            self.has(digest)               # refresh a shared store before failing
        content = self.cache.path(digest)
        if digest.startswith(TREE):
            return content
        files = [p for p in content.iterdir() if p.is_file()]
        if len(files) != 1:
            raise FileNotFoundError(
                f"blob entry {digest} holds {len(files)} files; expected one")
        return files[0]

    def pin(self, digest: str) -> None:
        self.cache.pin(digest)

    def unpin(self, digest: str) -> None:
        self.cache.unpin(digest)

    def put_file(self, staged, *, expect: str | None = None,
                 name: str | None = None, computed: str | None = None) -> str:
        """Adopt an already-written file, returning the digest it is stored under.

        ``staged`` is a path the caller has finished writing (an upload streamed
        to the job directory, say). The digest is computed HERE, from the bytes
        on disk; ``expect`` is only ever checked against it.

        ``computed`` says the digest was ALREADY taken by this server over these
        exact bytes - the upload path hashes as it streams - which saves re-reading
        a multi-gigabyte file to learn what it just measured. It is not a way in
        for a client's claim: that is ``expect``, and it is checked, never trusted.
        """
        digest = computed or digest_file(staged)
        if expect and expect != digest:
            raise DigestMismatch(expect, digest)
        self._adopt(digest, [(Path(staged), name or Path(staged).name)])
        return digest

    def put_dir(self, staged, *, expect: str | None = None) -> str:
        """Adopt a directory of files (a DICOM series) as one tree entry."""
        members = sorted(p for p in Path(staged).rglob("*") if p.is_file())
        if not members:
            raise FileNotFoundError(f"{staged} holds no files")
        digest = tree_digest(digest_file(p) for p in members)
        if expect and expect != digest:
            raise DigestMismatch(expect, digest)
        # Flattened to basenames on purpose: an archive member may name any path
        # it likes, and a store keyed by content has no business reproducing
        # someone's directory layout - let alone one with `..` in it.
        self._adopt(digest, [(p, p.name) for p in members])
        return digest

    def _adopt(self, digest: str, members) -> None:
        """Copy members into the entry for ``digest``, unless it is already there.

        Goes through the cache's own fetch path, so the claim, the wait-on-another
        -writer and the commit are the ones already in use for fetched series -
        including the part that matters most here: if two callers upload the same
        bytes at once, one writes and the other waits, and the second upload
        costs nothing. An entry that already exists is simply reused; identical
        content means there is nothing to reconcile.
        """
        import shutil

        def write(_key, entry):
            # same contract every source's fetch() honors: build <entry>/series
            # and return it (see nnseg.sources)
            content = Path(entry) / "series"
            content.mkdir(parents=True, exist_ok=True)
            for src, name in members:
                shutil.copyfile(src, content / Path(name).name)
            return content

        with self._lock:               # no reload may land between these two
            self.cache.get_or_fetch(digest, fetch=write)
            if self._commit is not None:
                self._commit()             # publish it to everyone else


#: Magic-number sniffing, because a content-addressed store keeps the BYTES and
#: the bytes alone - and SimpleITK picks its reader from the file EXTENSION. An
#: entry stored under a name with no suffix is unreadable however correct its
#: digest is, which is a failure that only shows up when something actually opens
#: it (a fake segmenter in a test never will).
_NIFTI_MAGIC_OFFSET = 344


def _inner_name(head: bytes) -> str | None:
    if head[:4] == b"NRRD":
        return "input.nrrd"
    if head[:10] == b"ObjectType":
        return "input.mha"
    if head[128:132] == b"DICM":
        return "input.dcm"
    magic = head[_NIFTI_MAGIC_OFFSET:_NIFTI_MAGIC_OFFSET + 4]
    if magic in (b"n+1\x00", b"ni1\x00"):
        return "input.nii"
    return None


def guess_name(path) -> str | None:
    """A filename whose extension names the format, read from the content.

    Gzip is unwrapped before deciding, rather than assumed to be NIfTI: a
    gzipped NRRD is rare but stored under ``.nii.gz`` it would be unreadable,
    and "rare" is exactly the case nobody notices until it fails.
    """
    with open(path, "rb") as f:
        head = f.read(512)
    if head[:2] == b"\x1f\x8b":
        import gzip
        try:
            with gzip.open(path, "rb") as g:
                inner = _inner_name(g.read(512))
        except OSError:
            return None
        return inner + ".gz" if inner else None
    return _inner_name(head)
