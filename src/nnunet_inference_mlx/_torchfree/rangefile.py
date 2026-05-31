"""Seekable, range-request-backed file objects for zipfile.

HTTPRangeFile: one network request per read (simple, but bad for many-small-read
               patterns like zipfile reading hundreds of members).
CachingRangeFile: fetches in aligned blocks and caches them, so adjacent/repeated
               small reads are served from memory. Much better for zip access.

Both resolve redirects once to a stable URL and use only forward ranges
(no suffix/negative ranges, which some CDNs reject).

Uses httpx (the package's HTTP client; install the ``remote`` extra). httpx
does not follow redirects by default, so the client is created with
``follow_redirects=True``.
"""
import httpx


class _Base:
    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self._pos
    def seek(self, offset, whence=0):
        if whence == 0: self._pos = offset
        elif whence == 1: self._pos += offset
        elif whence == 2: self._pos = self.size + offset
        return self._pos
    def close(self): pass
    def __enter__(self): return self
    def __exit__(self, *a): self.close()

    def _setup(self, url, session):
        self.session = session or httpx.Client(follow_redirects=True)
        r = self.session.head(url, timeout=30)
        r.raise_for_status()
        self.url = str(r.url)
        self.size = int(r.headers["Content-Length"])
        if r.headers.get("Accept-Ranges") != "bytes":
            t = self.session.get(self.url, headers={"Range": "bytes=0-0"}, timeout=30)
            if t.status_code != 206:
                raise IOError("server does not support range requests")
        self._pos = 0
        self.num_requests = 0
        self.bytes_read = 0

    def _fetch(self, start, end):
        """Fetch inclusive byte range [start, end] from the server."""
        r = self.session.get(self.url, headers={"Range": f"bytes={start}-{end}"}, timeout=120)
        r.raise_for_status()
        self.num_requests += 1
        self.bytes_read += len(r.content)
        return r.content


class HTTPRangeFile(_Base):
    """One request per read. Simplest; fine for few large reads."""
    def __init__(self, url, session=None):
        self._setup(url, session)

    def read(self, n=-1):
        end = self.size - 1 if (n is None or n < 0) else min(self._pos + n - 1, self.size - 1)
        if self._pos > end:
            return b""
        data = self._fetch(self._pos, end)
        self._pos += len(data)
        return data


class CachingRangeFile(_Base):
    """Block-aligned caching range file. Reads fetch the covering blocks
    (default 4 MB) once and serve from cache thereafter — turns zipfile's
    many small reads into a handful of large fetches.
    """
    def __init__(self, url, session=None, block_size=4 * 1024 * 1024):
        self._setup(url, session)
        self.block_size = block_size
        self._cache = {}  # block_index -> bytes

    def _block(self, idx):
        if idx not in self._cache:
            start = idx * self.block_size
            end = min(start + self.block_size, self.size) - 1
            self._cache[idx] = self._fetch(start, end)
        return self._cache[idx]

    def read(self, n=-1):
        end = self.size if (n is None or n < 0) else min(self._pos + n, self.size)
        if self._pos >= end:
            self._pos = end
            return b""
        out = bytearray()
        pos = self._pos
        while pos < end:
            bidx = pos // self.block_size
            blk = self._block(bidx)
            off = pos - bidx * self.block_size
            take = min(len(blk) - off, end - pos)
            out += blk[off:off + take]
            pos += take
        self._pos = end
        return bytes(out)
