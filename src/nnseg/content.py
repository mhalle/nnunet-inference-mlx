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

#: The decoded copy's filename. Looked up BY NAME, never by scanning the
#: directory: the decoder writes to a temp name and renames, so a scan would
#: happily return a `.partial` left by a crash mid-write - handing a truncated
#: volume to a reader as though it were complete, and defeating the atomic
#: rename that exists to prevent exactly that.
DECODED_NAME = "content.nrrd"

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


def _stored_name(path) -> str:
    """The name a member is stored under - derived from CONTENT, not the request.

    This has to be a function of the bytes alone. The backing store is a Modal
    Volume, which is not POSIX-coherent across containers, and the justification
    for running without a cross-container mutex is that "two writers of the same
    digest write identical bytes". That is true of each FILE and was false of the
    DIRECTORY: the four write paths named identical bytes differently
    (`input_<file>` from a submit, the client's `?filename=` from a PUT,
    `guess_name` from a promotion, `{i}_{name}` from a tree), so two containers
    could merge one digest's entry into a multi-file directory - loud for a blob
    ("holds 2 files"), and SILENT for a tree, where a 3-slice series becomes 6
    files handed to the DICOM reader.

    Sniffing the format gives a name that two containers agree on without
    talking - and it is the ONLY input, so nothing the caller sent can reach the
    stored layout. An earlier version fell back to the caller's suffix when
    sniffing failed, which let two clients sending identical bytes with
    different filenames write different directories: exactly what naming by
    content exists to prevent.
    """
    guessed = guess_name(path)
    if not guessed:
        raise UnidentifiedContent(path)
    return guessed


class UnidentifiedContent(ValueError):
    """The bytes are not a medical image this server can read.

    Refused at ingest rather than stored. A blob whose format we cannot
    determine is one nothing can open later - SimpleITK dispatches its reader on
    the extension - so accepting it only defers the failure from the upload to
    the job, minutes later and further from the cause. The sniffer is therefore
    the accept-list: adding a format means teaching :func:`guess_name` its magic
    number, never trusting a name the caller supplied.
    """

    def __init__(self, path):
        super().__init__(
            f"cannot identify {Path(path).name} as a medical image (expected "
            "NIfTI, NRRD, MetaImage or DICOM); nothing was stored")


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

    def __init__(self, cache, *, commit=None, refresh=None, lock=None, decode=None):
        import contextlib
        self.cache = cache
        # decode(src, dst_dir) -> Path | None: materialize a fast-reading copy.
        # Injected rather than imported, the way SeriesCache takes its fetch and
        # ReadAhead its read - this module stays hashlib + pathlib.
        self._decode = decode
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

    def fast_path(self, digest: str) -> Path:
        """What to READ - the decoded form when there is one, else the original.

        Inputs arrive compressed and get read more than once: the same volume
        served to a second task is a second full decode, and gzip is 10-16x
        slower than a raw read (measured: 87 ms vs 9 ms for a 31 MB CT; seconds
        on a whole-body volume, where a warm case measured 82 % decompression).

        The decoded copy lives OUTSIDE ``series/`` so the digest-true bytes stay
        exactly what the client sent - the digest is over those, and re-encoding
        them would break addressing. It is derived, so it is pure cache: losing
        it costs a re-decode and never an answer.

        Materialized lazily, on the first read rather than at ingest, so a
        preloaded input nobody runs never pays for a decode it does not need.
        """
        original = self.resolve(digest)
        if self._decode is None:
            return original
        entry = self.cache.path(digest).parent
        decoded = entry / "decoded" / DECODED_NAME
        try:
            if decoded.is_file():
                return decoded
            with self._lock:
                made = self._decode(original, decoded.parent)
            if made is not None:
                self._restamp(entry)
                return Path(made)
        except Exception:
            pass                        # any failure: read the original
        return original

    def _restamp(self, entry) -> None:
        """Keep the LRU's byte count honest after adding a derived file.

        The marker records the entry's size at commit; a decoded copy added
        later would otherwise be invisible to the budget - an eviction policy
        that cannot see half of what it stores.
        """
        marker = entry / self.cache.MARKER
        try:
            total = sum(p.stat().st_size for p in entry.rglob("*") if p.is_file())
            marker.write_text(str(total))
        except OSError:
            pass

    def pin(self, digest: str) -> None:
        self.cache.pin(digest)

    def unpin(self, digest: str) -> None:
        self.cache.unpin(digest)

    def put_file(self, staged, *, expect: str | None = None,
                 computed: str | None = None) -> str:
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
        self._adopt(digest, [(Path(staged), _stored_name(staged))])
        return digest

    def put_dir(self, staged, *, expect: str | None = None) -> str:
        """Adopt a directory of files (a DICOM series) as one tree entry."""
        members = sorted(p for p in Path(staged).rglob("*") if p.is_file())
        if not members:
            raise FileNotFoundError(f"{staged} holds no files")
        per_member = [(p, digest_file(p)) for p in members]
        digest = tree_digest(d for _, d in per_member)
        if expect and expect != digest:
            raise DigestMismatch(expect, digest)
        # Each member is stored under its OWN digest, so the stored directory is
        # a function of the content and nothing else - not of arrival order, not
        # of the names a zip happened to carry. Two containers writing this entry
        # therefore write the same directory, which is what the missing
        # cross-container mutex is traded against. Flattening also removes the
        # traversal class outright rather than validating against it.
        #
        # No extension: GDCM identifies DICOM by content, and a series of
        # extensionless files reads correctly (verified against a real 30-slice
        # series).
        self._adopt(digest, [(p, d.split(":", 1)[1][:32]) for p, d in per_member])
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
        except Exception:
            # Any failure to decompress means "cannot identify" - never a raised
            # exception. A truncated gzip raises EOFError, not OSError, and this
            # runs on EVERY stored blob, so a narrow except turns a malformed
            # upload into a 500 far from the cause.
            return None
        return inner + ".gz" if inner else None
    try:
        return _inner_name(head)
    except Exception:
        return None


#: A zip is transport, not identity - so extraction is where the hostile cases
#: live, and none of them should reach a store keyed by content.
MAX_MEMBERS = 20_000            # a long CT series is a few thousand instances
MAX_BYTES = 8 << 30


class ArchiveError(ValueError):
    """The archive cannot be safely or sensibly unpacked."""


def extract_zip(archive, dest, *, max_members: int = MAX_MEMBERS,
                max_bytes: int = MAX_BYTES) -> list:
    """Unpack ``archive`` into ``dest``, flattened, returning the files written.

    Every member is written under its BASENAME. A zip may name any path it
    likes - absolute, or salted with ``..`` - and the usual advice is to
    validate those paths; here there is nothing to validate against, because a
    store keyed by content has no business reproducing someone's directory
    layout in the first place. Flattening removes the whole class rather than
    checking for it.

    Sizes are read from the header and enforced against a running total, so an
    archive that expands to a hundred times its size is refused while unpacking
    rather than after filling the disk.
    """
    import zipfile

    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    written, total, seen = [], 0, {}
    try:
        with zipfile.ZipFile(archive) as z:
            members = [m for m in z.infolist() if not m.is_dir()]
            if not members:
                raise ArchiveError("the archive holds no files")
            if len(members) > max_members:
                raise ArchiveError(
                    f"{len(members)} members exceeds the {max_members} limit")
            for m in members:
                name = Path(m.filename.replace("\\", "/")).name
                if not name or name in (".", ".."):
                    continue               # a directory entry in disguise
                total += m.file_size
                if total > max_bytes:
                    raise ArchiveError(
                        f"the archive expands past the {max_bytes} byte limit")
                # Two members can flatten onto one name (a/IM1 and b/IM1), and
                # both must survive: dropping one would silently change what the
                # tree IS. Index every member unconditionally rather than only
                # the collisions - a synthesized "2_IM1.dcm" would otherwise
                # collide with a real member of that name and overwrite it,
                # which is the same silent loss by a longer route. These names
                # are staging only; the store renames members by their own
                # digest.
                seen[name] = seen.get(name, 0) + 1
                out = dest / f"{len(written)}_{name}"
                with z.open(m) as src, open(out, "wb") as f:
                    while chunk := src.read(_CHUNK):
                        f.write(chunk)
                written.append(out)
    except zipfile.BadZipFile as e:
        raise ArchiveError(f"not a readable zip archive: {e}") from e
    return written
