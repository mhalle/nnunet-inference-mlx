"""Pluggable remote data repositories the server can fetch inputs from.

The IDC methodology - a URL path prefix, a strict identifier pattern, and a
fetch that materializes the identified series locally - generalizes to any
repository. A :class:`DataSource` packages those three things, and everything
else (the series cache, the prefetch pipeline, the path surface, result-cache
identity) is source-agnostic plumbing over the registry.

Two ways to add a repository:

- **Programmatically**: subclass :class:`DataSource` when fetching needs code
  (bucket probing, manifest walks, auth handshakes).
- **Data only**: instantiate :class:`UrlTemplateSource` for the simplest case -
  one anonymously fetchable file per identifier at a templated URL.

Identifiers are the *whole* reference: every source declares a fullmatch
regex, and nothing here accepts a client-supplied URL. That is what keeps the
fetch path SSRF-free, so keep patterns strict when adding sources.

Identity strings are ``"<prefix>:<identifier>"`` - the result-cache key
component - so the ``idc`` source reproduces the established ``idc:<uuid>``
identities byte for byte.
"""
import os
import re
from pathlib import Path

from .errors import InputError

__all__ = ["DataSource", "UrlTemplateSource", "IDCSource", "default_sources",
           "IDC_BUCKETS", "CRDC_RE"]

CRDC_RE = r"[0-9a-f]{8}-(?:[0-9a-f]{4}-){3}[0-9a-f]{12}"

# The three public IDC buckets, probed in order (idc-open-data holds 99.5 % of
# series; -two and -cr the rest - found the hard way 2026-08-24). The clean
# upgrade path is resolving per series via idc-index (`series_aws_url`), which we
# take when /v1/resolve lands.
IDC_BUCKETS = ("idc-open-data", "idc-open-data-two", "idc-open-data-cr")


class DataSource:
    """One remote repository: a path prefix, an identifier pattern, a fetch.

    ``prefix`` names the source everywhere: the ``source`` kind at submit,
    the path surface (``/v1/<prefix>/<id>/<task>/labels.seg.nrrd``), the
    series-cache key namespace, and the identity string. Lowercase letters,
    digits, and ``_`` only, so it stays a clean URL segment.

    ``id_pattern`` is fullmatched against every identifier before anything
    else happens - it is the input-validation boundary, keep it strict.

    ``fetch(identifier, dest_dir)`` materializes the input under ``dest_dir``
    and returns the path the pipeline should read (a file, or a directory for
    a DICOM series). It runs on the worker, possibly on a prefetch thread.
    """

    prefix: str = ""
    id_pattern: str = ""
    description: str = ""

    def enabled(self) -> bool:
        """Whether this source can fetch on this install (dependencies etc.)."""
        return True

    def fetch(self, identifier: str, dest_dir: Path, *, credentials=None) -> Path:
        """Materialize the input. ``credentials`` is an optional per-request
        secret (e.g. a bearer token) - a credential in transit: never store,
        log, or record it anywhere durable."""
        raise NotImplementedError

    def identity(self, identifier: str) -> str:
        """The result-cache identity token for one identifier."""
        return f"{self.prefix}:{identifier}"

    def describe(self) -> dict:
        return {"prefix": self.prefix, "id_pattern": self.id_pattern,
                "enabled": self.enabled(), "description": self.description}


class UrlTemplateSource(DataSource):
    """The simplest possible source, defined by data alone: one anonymously
    fetchable file per identifier at ``url_template.format(id=identifier)``.

    The template is fixed at construction and the identifier is validated
    against ``id_pattern`` before substitution, so clients still cannot steer
    the URL anywhere the operator did not choose.
    """

    def __init__(self, prefix: str, id_pattern: str, url_template: str, *,
                 filename: str | None = None, description: str = ""):
        if "{id}" not in url_template:
            raise ValueError("url_template needs an {id} placeholder")
        self.prefix, self.id_pattern = prefix, id_pattern
        self.url_template, self.filename = url_template, filename
        self.description = description or f"single-file fetch from {url_template}"

    def _filename(self, identifier: str) -> str:
        """The saved name defaults to the identifier's basename so the format
        stays detectable (.nii.gz etc.); explicit ``filename`` overrides."""
        if self.filename:
            return self.filename
        name = Path(identifier).name
        return name if name and name not in (".", "..") else "image"

    def fetch(self, identifier: str, dest_dir: Path, *, credentials=None) -> Path:
        if _has_dotdot(identifier):
            raise InputError(f"{self.prefix}:{identifier}: '..' path segment refused")
        url = self.url_template.format(id=identifier)
        dest = Path(dest_dir) / "series"
        dest.mkdir(exist_ok=True)
        out = dest / self._filename(identifier)
        try:
            with _OPENER.open(url, timeout=300) as r, open(out, "wb") as f:
                _copy_capped(r, f, MAX_FETCH_BYTES, f"{self.prefix}:{identifier}")
        except InputError:
            raise
        except Exception as e:
            raise InputError(f"fetch of {self.prefix}:{identifier} failed: {e}") from e
        return dest


class IDCSource(DataSource):
    """NCI Imaging Data Commons: DICOM series by ``crdc_series_uuid`` from the
    public open-data buckets, anonymously, 32 threads (obstore beat s5cmd in
    every measured quadrant). The uuid names a version-pinned series; the
    bucket prefix is probed across the three known buckets rather than
    assumed."""

    prefix = "idc"
    id_pattern = CRDC_RE
    description = "NCI Imaging Data Commons, by crdc_series_uuid"

    def enabled(self) -> bool:
        try:
            import obstore  # noqa: F401
            return True
        except ImportError:
            return False

    def fetch(self, identifier: str, dest_dir: Path, *, credentials=None) -> Path:
        from concurrent.futures import ThreadPoolExecutor

        from obstore.store import S3Store
        keys, store = [], None
        for bucket in IDC_BUCKETS:
            store = S3Store.from_url(f"s3://{bucket}", config={"aws_skip_signature": "true"})
            keys = [(o.get("path") if isinstance(o, dict) else str(o))
                    for b in store.list(prefix=f"{identifier}/") for o in b]
            if keys:
                break
        if not keys:
            raise InputError(f"no objects under {identifier!r}/ in any probed IDC bucket "
                             f"({', '.join(IDC_BUCKETS)}); if the series exists, IDC may "
                             "have added a bucket this server does not know")
        dest = Path(dest_dir) / "series"
        dest.mkdir(exist_ok=True)

        def one(k):
            base = k.rsplit("/", 1)[-1]
            if not base or base in (".", ".."):
                return                     # bucket pseudo-dir key; skip
            with open(dest / base, "wb") as f:
                f.write(bytes(store.get(k).bytes()))

        try:
            with ThreadPoolExecutor(32) as ex:
                list(ex.map(one, keys))
        except Exception as e:
            raise InputError(f"fetch of idc:{identifier} failed: {e}") from e
        return dest


class TCIASource(DataSource):
    """The Cancer Imaging Archive via its NBIA REST API: a DICOM series by
    SeriesInstanceUID, anonymously, for fully public collections. The API
    returns one zip per series; entries are flattened by basename on
    extraction, which also makes zip-slip impossible by construction.

    NOT version-pinned: TCIA serves the collection's current revision, so the
    same SeriesInstanceUID can resolve to different bytes across data
    releases. For version-pinned identity use the idc source - the two doors
    deliberately have distinct identities (tcia:<uid> vs idc:<uuid>)."""

    prefix = "tcia"
    id_pattern = r"(?=.{10,64}$)[0-9]+(?:\.[0-9]+)+"   # DICOM UID: digits, dots, <=64
    description = "The Cancer Imaging Archive (NBIA), by SeriesInstanceUID"
    API = "https://services.cancerimagingarchive.net/nbia-api/services/v1/getImage"

    def fetch(self, identifier: str, dest_dir: Path, *, credentials=None) -> Path:
        import shutil
        import urllib.request
        import zipfile
        dest = Path(dest_dir) / "series"
        dest.mkdir(exist_ok=True)
        tmp = Path(dest_dir) / "series.zip"
        try:
            with urllib.request.urlopen(f"{self.API}?SeriesInstanceUID={identifier}",
                                        timeout=600) as r, open(tmp, "wb") as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
            n = 0
            with zipfile.ZipFile(tmp) as z:
                if sum(zi.file_size for zi in z.infolist()) > MAX_FETCH_BYTES:
                    raise InputError(f"tcia:{identifier}: series exceeds the "
                                     f"{MAX_FETCH_BYTES}-byte cap")
                seen: dict = {}
                for m in z.infolist():
                    name = Path(m.filename).name   # flatten: zip paths never touch disk
                    if m.is_dir() or not name or name.startswith("."):
                        continue
                    if name in seen:
                        seen[name] += 1
                        stem, dot, ext = name.partition(".")
                        name = f"{stem}-{seen[name]}{dot}{ext}"
                    else:
                        seen[name] = 0
                    with z.open(m) as src, open(dest / name, "wb") as out:
                        shutil.copyfileobj(src, out)
                    n += 1
            if n == 0:
                raise InputError(f"TCIA returned an empty series for {identifier!r}")
        except InputError:
            raise
        except Exception as e:
            raise InputError(f"fetch of tcia:{identifier} failed: {e}") from e
        finally:
            tmp.unlink(missing_ok=True)
        return dest


def openneuro_source() -> UrlTemplateSource:
    """OpenNeuro (CC0 neuroimaging, BIDS layout) straight off its public S3
    bucket over HTTPS - the data-only case: identifiers are
    ``ds<number>/<path-to-file>`` (slashed, so cache keys hash; the path
    surface's greedy routes carry the slashes in ordinary URLs)."""
    return UrlTemplateSource(
        "openneuro",
        r"(?!.*(?:^|/)\.\.(?:/|$))ds[0-9]{6}/[A-Za-z0-9][A-Za-z0-9._/-]{0,200}",
        "https://s3.amazonaws.com/openneuro.org/{id}",
        description="OpenNeuro (CC0), by ds<number>/<file path>")


MAX_FETCH_BYTES = int(float(os.environ.get("NNSEG_MAX_FETCH_GB", "16")) * (1 << 30))


def _has_dotdot(identifier: str) -> bool:
    """True if any path segment is '..' - such an identifier steers a
    template/resolve URL off the operator's prefix (the remote or a proxy may
    apply remove_dot_segments). The id_patterns also exclude it; this is the
    belt to that suspenders, at the one place every fetch passes through."""
    return any(seg == ".." for seg in re.split(r"[/\\]", identifier))


def _copy_capped(src, dst, cap: int, what: str) -> int:
    """copyfileobj with a hard byte ceiling - a server that lies about
    Content-Length (or a bomb) cannot fill the disk."""
    n = 0
    while True:
        chunk = src.read(1 << 20)
        if not chunk:
            return n
        n += len(chunk)
        if n > cap:
            raise InputError(f"{what}: exceeded the {cap}-byte fetch cap")
        dst.write(chunk)
    return n


def _safe_opener():
    """A urllib opener that DROPS Authorization on a cross-host redirect.
    urllib copies every header across redirects except content-*, so a
    per-request source token would otherwise follow a first-hop redirect to
    any host it names. httpx (the client) already does this; sources.py did
    not."""
    import urllib.request
    from urllib.parse import urlparse

    def _origin_change(old_url: str, new_url: str) -> bool:
        a, b = urlparse(old_url), urlparse(new_url)
        if a.netloc != b.netloc:
            return True
        # an https->http downgrade on the SAME host still exposes the token in
        # cleartext; strip it (an http->https upgrade is safe and kept, which
        # is exactly what httpx does)
        return a.scheme == "https" and b.scheme == "http"

    class _StripCrossHostAuth(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            new = super().redirect_request(req, fp, code, msg, headers, newurl)
            if new is not None and _origin_change(req.full_url, newurl):
                for k in [h for h in new.headers if h.lower() == "authorization"]:
                    del new.headers[k]
            return new

    return urllib.request.build_opener(_StripCrossHostAuth)


_OPENER = _safe_opener()


class RangeFile:
    """A seekable read-only file over HTTP Range requests, with an LRU block
    cache. Handing one to :class:`zipfile.ZipFile` gives remote archives
    random access - zip64, deflate, and per-member CRC verification all come
    from the stdlib. Redirects are followed per request (CDN URLs expire)."""

    def __init__(self, url: str, size: int, *, headers=None, block: int = 1 << 22,
                 max_blocks: int = 64):
        import collections
        self.url, self.size, self.pos = url, int(size), 0
        self.headers = dict(headers or {})
        self.block_size, self.max_blocks = int(block), int(max_blocks)
        self._blocks = collections.OrderedDict()
        self.requests = 0
        self.fetched = 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]
        return self.pos

    def tell(self):
        return self.pos

    def _block(self, i: int) -> bytes:
        if i in self._blocks:
            self._blocks.move_to_end(i)
            return self._blocks[i]
        import urllib.request
        lo = i * self.block_size
        hi = min(self.size, lo + self.block_size) - 1
        req = urllib.request.Request(self.url,
                                     headers={**self.headers, "Range": f"bytes={lo}-{hi}"})
        with _OPENER.open(req, timeout=300) as r:
            if r.status != 206:
                # a 200 means the server ignored Range and is streaming the
                # WHOLE body - on a multi-GB archive that is an unbounded read
                # into memory; refuse rather than pull it
                raise InputError(
                    f"range request not honored (status {r.status}) by "
                    f"{self.url}; server does not support HTTP Range")
            data = r.read()
        want = hi - lo + 1
        if len(data) != want:
            raise InputError(f"short/over range read from {self.url}: got "
                             f"{len(data)} bytes, asked {want}")
        self._blocks[i] = data
        while len(self._blocks) > self.max_blocks:
            self._blocks.popitem(last=False)
        self.requests += 1
        self.fetched += len(data)
        return data

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        out = bytearray()
        while n > 0 and self.pos < self.size:
            i, off = divmod(self.pos, self.block_size)
            chunk = self._block(i)[off:off + n]
            if not chunk:
                break
            out += chunk
            self.pos += len(chunk)
            n -= len(chunk)
        return bytes(out)


class ArchiveReadingSource(DataSource):
    """Base for hosts that serve files - and, via HTTP Range, individual
    members of zip archives - by identifier.

    The identifier grammar is ``<outer>[!<member>]``: without ``!`` the outer
    file downloads whole; with it, the named member (or every member under a
    trailing-slash prefix) is extracted from the remote zip while fetching
    only the bytes it occupies. Subclasses implement one method:
    ``resolve(outer, credentials) -> (url, size)``. Tar and parquet archives
    are out of scope - only zip has the trailing central directory that makes
    remote random access possible."""

    def resolve(self, outer: str, credentials=None) -> tuple:
        raise NotImplementedError

    def _headers(self, credentials=None) -> dict:
        return {"Authorization": f"Bearer {credentials}"} if credentials else {}

    def _zip(self, outer: str, credentials=None):
        """The parsed archive, cached per ``(outer, credentials)``. Keying on
        the credential is load-bearing, not an optimization: on a cache hit
        ``resolve`` is skipped, and ``resolve`` is the ONLY place the
        restricted-access gate runs and the token is bound to the archive's
        RangeFile. Caching by ``outer`` alone let a tokenless caller reuse an
        earlier caller's authenticated archive - reading gated content and
        replaying that caller's token upstream (round-4 finding)."""
        import zipfile
        cache = self.__dict__.setdefault("_archives", {})
        ck = (outer, credentials)
        z = cache.get(ck)
        if z is None:
            url, size = self.resolve(outer, credentials)
            z = zipfile.ZipFile(RangeFile(url, size, headers=self._headers(credentials)))
            cache[ck] = z
            while len(cache) > 4:
                cache.pop(next(iter(cache)))
        return z

    def fetch(self, identifier: str, dest_dir: Path, *, credentials=None) -> Path:
        import shutil
        import urllib.request
        outer, _, member = identifier.partition("!")
        if _has_dotdot(outer):             # the outer id steers the URL; the
                                           # member is neutralized by flatten
            raise InputError(f"{self.prefix}:{identifier}: '..' path segment refused")
        dest = Path(dest_dir) / "series"
        dest.mkdir(exist_ok=True)
        try:
            if not member:                 # plain file: stream it down whole
                url, _size = self.resolve(outer, credentials)
                name = Path(outer).name
                if not name or name in (".", ".."):
                    name = "image"
                req = urllib.request.Request(url, headers=self._headers(credentials))
                with _OPENER.open(req, timeout=1800) as r, open(dest / name, "wb") as f:
                    _copy_capped(r, f, MAX_FETCH_BYTES, f"{self.prefix}:{identifier}")
                return dest
            z = self._zip(outer, credentials)
            members = ([m for m in z.namelist()
                        if m.startswith(member) and not m.endswith("/")]
                       if member.endswith("/") else [member])
            if not members:
                raise InputError(f"{self.prefix}:{identifier}: no such member in archive")
            # refuse a decompression bomb by declared uncompressed size BEFORE
            # opening anything (the central directory is already parsed)
            want = set(members)
            total = sum(zi.file_size for zi in z.infolist() if zi.filename in want)
            if total > MAX_FETCH_BYTES:
                raise InputError(f"{self.prefix}:{identifier}: members total "
                                 f"{total} bytes, over the {MAX_FETCH_BYTES} cap")
            seen: dict = {}
            for m in members:
                name = Path(m).name        # flatten: archive paths never touch disk
                if not name or name.startswith("."):
                    continue
                if name in seen:           # basename collision: disambiguate
                    seen[name] += 1        # rather than silently overwrite
                    stem, dot, ext = name.partition(".")
                    name = f"{stem}-{seen[name]}{dot}{ext}"
                else:
                    seen[name] = 0
                with z.open(m) as src, open(dest / name, "wb") as f:
                    shutil.copyfileobj(src, f, 1 << 20)
        except InputError:
            raise
        except Exception as e:
            raise InputError(f"fetch of {self.prefix}:{identifier} failed: {e}") from e
        return dest


class ZenodoSource(ArchiveReadingSource):
    """Zenodo records: ``<recid>/<filename>[!member]``. A record id pins one
    published version (new versions mint new record ids; only the concept DOI
    floats), so identities are cache-grade. A personal access token raises
    rate limits and - when the operator opts in - unlocks restricted records;
    by default restricted records are refused, because cached results are
    readable by every cache reader regardless of who fetched the input."""

    prefix = "zenodo"
    id_pattern = (r"(?!.*(?:^|/)\.\.(?:/|!|$))"
                  r"[0-9]{4,9}/[A-Za-z0-9._-]+(?:![A-Za-z0-9._ /-]+)?")
    description = "Zenodo records, by record id / filename (!member for zip contents)"

    def __init__(self, *, allow_restricted: bool = False):
        self.allow_restricted = allow_restricted

    def resolve(self, outer: str, credentials=None) -> tuple:
        import json as _json
        import urllib.request
        recid, _, filename = outer.partition("/")
        req = urllib.request.Request(f"https://zenodo.org/api/records/{recid}",
                                     headers=self._headers(credentials))
        with _OPENER.open(req, timeout=60) as r:
            rec = _json.load(r)
        access = ((rec.get("metadata") or {}).get("access_right")
                  or (rec.get("access") or {}).get("files") or "")
        # fail CLOSED: absent/empty access fields (a narrowed API response, a
        # schema move) must not read as "open"
        if access not in ("open", "public") and not self.allow_restricted:
            raise InputError(
                f"zenodo record {recid} is {access or 'of undeclared access'!r}; "
                "this server only fetches open "
                "records (operator opt-in required for restricted data - cached "
                "results are readable by every cache reader)")
        for f in rec.get("files", []):
            if f.get("key") == filename or f.get("filename") == filename:
                size = f.get("size") or f.get("filesize")
                url = (f.get("links", {}).get("content")
                       or f"https://zenodo.org/records/{recid}/files/{filename}?download=1")
                return url, int(size)
        raise InputError(f"zenodo record {recid} has no file {filename!r}")


class HuggingFaceSource(ArchiveReadingSource):
    """Hugging Face dataset repos: ``<org>/<name>@<commit-sha>/<path>[!member]``.

    The 40-hex commit sha is REQUIRED: branch names float (``main`` is a
    moving target), and a floating reference under a content-addressed cache
    is the stale-identity bug in waiting - the same doctrine as IDC's
    crdc_series_uuid and the task grammar's @version. Gated/private repos
    follow the same operator opt-in rule as Zenodo restricted records."""

    prefix = "hf"
    id_pattern = (r"(?!.*(?:^|/)\.\.(?:/|!|$))"
                  r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*"
                  r"@[0-9a-f]{40}/[A-Za-z0-9._/-]+(?:![A-Za-z0-9._ /-]+)?")
    description = "Hugging Face datasets, by org/name@commit-sha/path (!member for zips)"

    def __init__(self, *, allow_restricted: bool = False):
        self.allow_restricted = allow_restricted

    def resolve(self, outer: str, credentials=None) -> tuple:
        import urllib.request
        repo, _, rest = outer.partition("@")
        sha, _, path = rest.partition("/")
        url = f"https://huggingface.co/datasets/{repo}/resolve/{sha}/{path}"
        req = urllib.request.Request(url, method="HEAD",
                                     headers=self._headers(credentials))
        try:
            with _OPENER.open(req, timeout=60) as r:
                size = int(r.headers.get("Content-Length") or 0)
        except urllib.error.HTTPError as e:
            if e.code in (401, 403):
                raise InputError(
                    f"hf dataset {repo} is gated or private; "
                    + ("a valid token is required"
                       if self.allow_restricted else
                       "this server only fetches public repos (operator opt-in "
                       "required for gated data)")) from e
            raise
        if size <= 0:
            raise InputError(f"hf: could not size {url}")
        return url, size


def default_sources() -> list:
    """The sources a server carries unless told otherwise."""
    return [IDCSource(), TCIASource(), openneuro_source(), ZenodoSource(),
            HuggingFaceSource()]


def registry(sources=None) -> dict:
    """Normalize a list of sources into an ordered ``{prefix: source}`` map."""
    out = {}
    for s in (default_sources() if sources is None else list(sources)):
        if not s.prefix or not s.prefix.replace("_", "").isalnum() or not s.prefix.islower():
            raise ValueError(f"bad source prefix {s.prefix!r}")
        if s.prefix in out or s.prefix in ("jobs", "tasks", "health", "upload", "segmentations", "sources"):
            raise ValueError(f"source prefix {s.prefix!r} collides")
        if not s.id_pattern:
            raise ValueError(f"source {s.prefix!r} declares no id_pattern")
        # identifiers may contain slashes (DOIs, org/name ids): the series
        # cache hashes filesystem-unsafe keys, and the path surface's greedy
        # routes ({ident:path}, parsed right-to-left in serve._mount_source)
        # address them in ordinary URLs - so no pattern restriction is needed.
        out[s.prefix] = s
    return out
