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

    def fetch(self, identifier: str, dest_dir: Path) -> Path:
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
                 filename: str = "image", description: str = ""):
        if "{id}" not in url_template:
            raise ValueError("url_template needs an {id} placeholder")
        self.prefix, self.id_pattern = prefix, id_pattern
        self.url_template, self.filename = url_template, filename
        self.description = description or f"single-file fetch from {url_template}"

    def fetch(self, identifier: str, dest_dir: Path) -> Path:
        import urllib.request
        url = self.url_template.format(id=identifier)
        dest = Path(dest_dir) / "series"
        dest.mkdir(exist_ok=True)
        out = dest / self.filename
        try:
            with urllib.request.urlopen(url, timeout=120) as r, open(out, "wb") as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
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

    def fetch(self, identifier: str, dest_dir: Path) -> Path:
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
            with open(dest / k.rsplit("/", 1)[-1], "wb") as f:
                f.write(bytes(store.get(k).bytes()))

        with ThreadPoolExecutor(32) as ex:
            list(ex.map(one, keys))
        return dest


def default_sources() -> list:
    """The sources a server carries unless told otherwise."""
    return [IDCSource()]


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
        for probe in ("../x", "a/b", "..", "a\\b"):
            if re.fullmatch(s.id_pattern, probe):
                raise ValueError(f"source {s.prefix!r}: id_pattern admits "
                                 f"{probe!r} - identifiers become directory "
                                 "names and must not carry path separators")
        out[s.prefix] = s
    return out
