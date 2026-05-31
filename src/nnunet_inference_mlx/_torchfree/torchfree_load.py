"""Read a PyTorch .pth (zip) checkpoint into numpy arrays without importing torch.

Handles the new zipfile serialization format (PyTorch >= 1.6, default).
Reconstructs tensors via a restricted unpickler whose find_class only
permits the storage / rebuild symbols a weights checkpoint actually uses.
"""
import io
import json
import pickle
import struct
import zipfile
import numpy as np

# torch Storage class name -> (numpy dtype, itemsize)
_STORAGE_DTYPE = {
    "DoubleStorage": np.dtype("float64"),
    "FloatStorage":  np.dtype("float32"),
    "HalfStorage":   np.dtype("float16"),
    "LongStorage":   np.dtype("int64"),
    "IntStorage":    np.dtype("int32"),
    "ShortStorage":  np.dtype("int16"),
    "CharStorage":   np.dtype("int8"),
    "ByteStorage":   np.dtype("uint8"),
    "BoolStorage":   np.dtype("bool"),
    "BFloat16Storage": np.dtype("uint16"),  # no native np bf16; keep raw bits
}

_BF16_TAGS = {"BFloat16Storage"}


class _StorageType:
    """Placeholder yielded by find_class for torch.*Storage symbols."""
    def __init__(self, name):
        self.name = name
        self.dtype = _STORAGE_DTYPE[name]


class _TorchFreeUnpickler(pickle.Unpickler):
    def __init__(self, file, zf, prefix, byteorder):
        super().__init__(file)
        self._zf = zf
        self._prefix = prefix
        self._byteorder = byteorder
        self._byteorder_swap = (byteorder == "big") != (np.little_endian is False)

    # --- security: only allow the symbols a weights dict needs ---------------
    def find_class(self, module, name):
        if module == "torch._utils" and name in ("_rebuild_tensor_v2", "_rebuild_tensor"):
            return getattr(self, "bound" + name)
        if module == "torch" and name in _STORAGE_DTYPE:
            return _StorageType(name)
        if module == "collections" and name == "OrderedDict":
            from collections import OrderedDict
            return OrderedDict
        # torch.device appears in tensor metadata; return an inert stub
        if module == "torch" and name == "device":
            return _inert_device
        # _rebuild_parameter wraps a tensor as an nn.Parameter; unwrap to tensor
        if module == "torch._utils" and name == "_rebuild_parameter":
            return self.bound_rebuild_parameter
        # numpy reconstruction symbols (both numpy<2 'core' and numpy>=2 '_core')
        if module in ("numpy.core.multiarray", "numpy._core.multiarray"):
            if name == "scalar":
                return np.core.multiarray.scalar
            if name == "_reconstruct":
                return np.core.multiarray._reconstruct
        if module in ("numpy", "numpy._core", "numpy.core") and name == "ndarray":
            return np.ndarray
        if module == "numpy" and name == "dtype":
            return np.dtype
        if module == "_codecs" and name == "encode":
            import _codecs
            return _codecs.encode
        raise pickle.UnpicklingError(
            f"Blocked global during torch-free load: {module}.{name}"
        )

    # --- persistent storage resolution ---------------------------------------
    def persistent_load(self, pid):
        # pid = ("storage", _StorageType, key, location, numel)
        assert isinstance(pid, tuple) and pid[0] == "storage", pid
        _, storage_type, key, location, numel = pid
        # Lazy: do not read bytes yet. Defer the zf.read until the tensor is
        # actually materialized. This lets a weights-only consumer skip
        # optimizer storages entirely (no remote fetch for them).
        return _LazyStorage(self._zf, f"{self._prefix}data/{key}",
                            storage_type, self._byteorder)


class _LazyStorage:
    def __init__(self, zf, member, storage_type, byteorder):
        self._zf = zf
        self._member = member
        self.storage_type = storage_type
        self._byteorder = byteorder
        self._array = None

    @property
    def array(self):
        if self._array is None:
            raw = self._zf.read(self._member)  # remote fetch happens HERE
            # Zero-copy view into the `bytes` buffer. The returned array is
            # read-only — callers must not mutate it in place. Every downstream
            # consumer in this package (mx.array, np.moveaxis + ascontiguousarray)
            # already copies into a fresh buffer, so this saves a full memcpy
            # over the raw storage bytes (~20% of load time on Apple Silicon).
            arr = np.frombuffer(raw, dtype=self.storage_type.dtype)
            if self._byteorder == "big":
                arr = arr.byteswap()
            self._array = arr
        return self._array


def _inert_device(*args, **kwargs):
    # torch.device("cpu") etc. — value is irrelevant for numpy extraction
    return None


def _rebuild_from(storage, storage_offset, size, stride):
    # Returns a lazy proxy; no storage bytes are read until .realize().
    return _LazyTensor(storage, storage_offset, tuple(size), tuple(stride))


class _LazyTensor:
    def __init__(self, storage, offset, size, stride):
        self._storage = storage
        self._offset = offset
        self._size = size
        self._stride = stride

    def realize(self):
        flat = self._storage.array  # triggers fetch
        if len(self._size) == 0:
            return flat[self._offset:self._offset + 1].reshape(())
        itemsize = flat.itemsize
        byte_strides = tuple(s * itemsize for s in self._stride)
        view = np.lib.stride_tricks.as_strided(
            flat[self._offset:], shape=self._size, strides=byte_strides
        )
        return np.ascontiguousarray(view)


def realize_tree(obj):
    """Recursively replace _LazyTensor proxies with numpy arrays."""
    if isinstance(obj, _LazyTensor):
        return obj.realize()
    if isinstance(obj, dict):
        return type(obj)((k, realize_tree(v)) for k, v in obj.items())
    if isinstance(obj, (list, tuple)):
        return type(obj)(realize_tree(v) for v in obj)
    return obj


# Bound method names referenced by find_class:
def _bind(cls):
    def _rebuild_tensor_v2(self, storage, storage_offset, size, stride,
                           requires_grad, backward_hooks, *extra):
        return _rebuild_from(storage, storage_offset, size, stride)

    def _rebuild_tensor(self, storage, storage_offset, size, stride):
        return _rebuild_from(storage, storage_offset, size, stride)

    def _rebuild_parameter(self, data, requires_grad, backward_hooks):
        # nn.Parameter wrapper -> just the underlying tensor proxy
        return data

    cls.bound_rebuild_tensor_v2 = _rebuild_tensor_v2
    cls.bound_rebuild_tensor = _rebuild_tensor
    cls.bound_rebuild_parameter = _rebuild_parameter
    return cls

_bind(_TorchFreeUnpickler)


def _parse_index(zf):
    """Parse the pickle index without reading tensor bytes.
    Returns (tree_of_lazy_proxies, prefix, byteorder, names).
    """
    names = zf.namelist()
    pkl_name = next(n for n in names if n.endswith("data.pkl"))
    prefix = pkl_name[: -len("data.pkl")]
    byteorder = "little"
    bo_name = prefix + "byteorder"
    if bo_name in names:
        byteorder = zf.read(bo_name).decode().strip()
    data = zf.read(pkl_name)
    up = _TorchFreeUnpickler(io.BytesIO(data), zf, prefix, byteorder)
    return up.load(), prefix, byteorder, names


def _storages_in(obj, acc):
    if isinstance(obj, _LazyTensor):
        acc.add(obj._storage._member)
    elif isinstance(obj, dict):
        for v in obj.values():
            _storages_in(v, acc)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _storages_in(v, acc)


def plan_load(zf, weights_key="network_weights"):
    """Inspect the central directory + pickle index (no tensor bytes read)
    and recommend a remote-loading strategy.

    Returns a dict:
      total_bytes        compressed size of all members
      weights_bytes      compressed size of the weights_key subtree's storages
      weights_fraction   weights_bytes / total_bytes
      compressed         True if the .pth's tensor members use deflate
                         (range-slicing individual tensors is then impossible)
      strategy           "lazy_range"  -> skipping non-weights saves real bytes
                         "full_download" -> little to skip, or compressed; just
                                            stream the whole file
      reason             short human explanation
    """
    tree, prefix, byteorder, names = _parse_index(zf)
    csize = {i.filename: i.compress_size for i in zf.infolist()}
    compress_types = {i.filename: i.compress_type for i in zf.infolist()}
    total = sum(csize.values()) or 1

    # are the storage members compressed?
    data_members = [n for n in names if (prefix + "data/") in n]
    compressed = any(compress_types.get(m, 0) != 0 for m in data_members)

    weights_bytes = total
    if isinstance(tree, dict) and weights_key in tree:
        acc = set()
        _storages_in(tree[weights_key], acc)
        weights_bytes = sum(csize.get(m, 0) for m in acc)

    frac = weights_bytes / total

    if compressed:
        strategy = "full_download"
        reason = "tensor members are deflate-compressed; cannot range-slice"
    elif frac >= 0.7:
        strategy = "full_download"
        reason = f"weights are {frac*100:.0f}% of file; little to skip"
    else:
        strategy = "lazy_range"
        reason = f"weights are {frac*100:.0f}% of file; skipping the rest saves bytes"

    return {
        "total_bytes": total,
        "weights_bytes": weights_bytes,
        "weights_fraction": frac,
        "compressed": compressed,
        "strategy": strategy,
        "reason": reason,
    }


def load_from_zip(zf, weights_key="network_weights"):
    """Core loader: take an open ZipFile (local, or one backed by a
    range-request file like CachingRangeFile) and return numpy arrays.

    If weights_key is given and present at the top level, only that subtree
    is materialized (other storages, e.g. optimizer state, are never read —
    important for lazy remote loads). Pass weights_key=None to realize all.
    """
    tree, prefix, byteorder, names = _parse_index(zf)

    if weights_key is not None and isinstance(tree, dict) and weights_key in tree:
        return realize_tree(tree[weights_key])
    return realize_tree(tree)


def load_pth(path_or_file, weights_key="network_weights"):
    """Return a .pth checkpoint as numpy arrays.

    Accepts a filesystem path or any seekable binary file-like object.
    By default returns only the `network_weights` subtree; pass
    weights_key=None to get the full top-level object.
    """
    if not zipfile.is_zipfile(path_or_file):
        raise NotImplementedError(
            "Legacy (non-zip) .pth format not supported by torch-free loader"
        )
    if hasattr(path_or_file, "seek"):
        path_or_file.seek(0)
    with zipfile.ZipFile(path_or_file) as zf:
        return load_from_zip(zf, weights_key=weights_key)


def load_pth_url(url, weights_key="network_weights", *, session=None,
                 block_size=8 * 1024 * 1024):
    """Load a remote .pth via HTTP range requests, downloading only the
    pickle index and the storage members actually realized.

    With the default weights_key='network_weights', optimizer state and
    other large non-weight storages are never fetched.

    Uses our httpx-backed :class:`CachingRangeFile` (install the ``remote``
    extra: httpx) over any server that supports the Range header. Pass
    ``session=`` an ``httpx.Client`` to reuse a connection / set auth+headers.
    """
    from .rangefile import CachingRangeFile
    rf = CachingRangeFile(url, session=session, block_size=block_size)
    with zipfile.ZipFile(rf) as zf:
        return load_from_zip(zf, weights_key=weights_key)

def smart_load_url(url, weights_key="network_weights", session=None,
                   block_size=8 * 1024 * 1024, verbose=False):
    """Load a remote .pth, automatically choosing between lazy range access
    and a full download based on what the central directory reveals.

    Opens a cheap caching range file (cost: the central directory + pickle
    index, a few KB), calls plan_load, then:
      - strategy 'lazy_range'    -> realize only weights_key over range reads
      - strategy 'full_download' -> stream the whole file once, load locally

    Returns (weights_dict, plan). The plan dict explains the choice.
    """
    import io as _io
    import httpx as _httpx
    from .rangefile import CachingRangeFile

    sess = session or _httpx.Client(follow_redirects=True)
    # Plan with small blocks (cheap: just index + central directory)
    plan_rf = CachingRangeFile(url, session=sess, block_size=256 * 1024)
    with zipfile.ZipFile(plan_rf) as zf:
        plan = plan_load(zf, weights_key=weights_key)
        if verbose:
            print(f"[smart_load] {plan['strategy']}: {plan['reason']} "
                  f"(index cost {plan_rf.bytes_read/1e3:.0f} KB)")
        if plan["strategy"] == "lazy_range":
            # Reopen with large blocks for efficient bulk weight reads
            bulk_rf = CachingRangeFile(url, session=sess, block_size=block_size)
            with zipfile.ZipFile(bulk_rf) as zf2:
                weights = load_from_zip(zf2, weights_key=weights_key)
            return weights, plan

    # full_download: one streamed GET, then load from memory
    r = sess.get(plan_rf.url, timeout=300)
    r.raise_for_status()
    buf = _io.BytesIO(r.content)
    weights = load_pth(buf, weights_key=weights_key)
    return weights, plan
