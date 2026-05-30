"""Torch-free loader for PyTorch ``.pth`` checkpoints.

Single-file module, ``numpy`` as the only dependency.

Reads the modern zip-format ``.pth`` (PyTorch >= 1.6) into nested
dict/list/OrderedDict with ``numpy.ndarray`` leaves, using a restricted
unpickler whose ``find_class`` only permits the symbols a weights checkpoint
actually uses.

Why this exists
---------------
``torch.load(weights_only=True)`` permits any "safe global" registered with
torch — a large, growing surface — and many users still pass
``weights_only=False`` because trainer-state checkpoints (e.g. nnUNet's)
contain classes that aren't on the safe list. The allowlist here is
deliberately tiny:

  * tensor rebuild helpers: ``_rebuild_tensor_v2`` / ``_rebuild_tensor`` /
    ``_rebuild_parameter``
  * the ``torch.*Storage`` dtype tags
  * ``collections.OrderedDict``
  * ``torch.device`` (returned as an inert stub)
  * numpy reconstruction: ``ndarray`` / ``dtype`` / ``_reconstruct`` /
    ``scalar``
  * ``_codecs.encode``

Anything else raises ``UnpicklingError``. There is no path to construct
arbitrary objects or execute code from a malicious ``.pth``.

The loader returns numpy arrays so it is framework-agnostic (MLX, JAX,
ONNX, CoreML, or torch via ``torch.from_numpy``).

Usage
-----
::

    from torchfree_pth import load_pth

    # Returns only the top-level ``network_weights`` subtree by default,
    # which is what nnUNet checkpoints contain. Pass weights_key=None to
    # get the full top-level dict (optimizer state, trainer state, ...).
    weights = load_pth("checkpoint_final.pth")
    for name, arr in weights.items():
        print(name, arr.shape, arr.dtype)
"""

from __future__ import annotations

import io
import pickle
import zipfile
import numpy as np


# ---------------------------------------------------------------------------
# Restricted unpickler
# ---------------------------------------------------------------------------

# torch ``*Storage`` class name -> numpy dtype
_STORAGE_DTYPE = {
    "DoubleStorage":  np.dtype("float64"),
    "FloatStorage":   np.dtype("float32"),
    "HalfStorage":    np.dtype("float16"),
    "LongStorage":    np.dtype("int64"),
    "IntStorage":     np.dtype("int32"),
    "ShortStorage":   np.dtype("int16"),
    "CharStorage":    np.dtype("int8"),
    "ByteStorage":    np.dtype("uint8"),
    "BoolStorage":    np.dtype("bool"),
    # No native np bf16; surface raw 16-bit bits. Caller decides how to view.
    "BFloat16Storage": np.dtype("uint16"),
}


class _StorageType:
    """Placeholder yielded by find_class for ``torch.*Storage`` symbols."""
    def __init__(self, name):
        self.name = name
        self.dtype = _STORAGE_DTYPE[name]


def _inert_device(*_a, **_kw):
    # ``torch.device("cpu")`` etc. — value is irrelevant for numpy extraction.
    return None


class _TorchFreeUnpickler(pickle.Unpickler):
    def __init__(self, file, zf, prefix, byteorder):
        super().__init__(file)
        self._zf = zf
        self._prefix = prefix
        self._byteorder = byteorder

    def find_class(self, module, name):
        # tensor rebuild helpers
        if module == "torch._utils" and name in ("_rebuild_tensor_v2", "_rebuild_tensor"):
            return getattr(self, "bound" + name)
        if module == "torch._utils" and name == "_rebuild_parameter":
            return self.bound_rebuild_parameter

        # storage type tags
        if module == "torch" and name in _STORAGE_DTYPE:
            return _StorageType(name)

        # device shows up in tensor metadata
        if module == "torch" and name == "device":
            return _inert_device

        if module == "collections" and name == "OrderedDict":
            from collections import OrderedDict
            return OrderedDict

        # numpy reconstruction (covers numpy<2 ``core`` and numpy>=2 ``_core``)
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

    def persistent_load(self, pid):
        # pid = ("storage", _StorageType, key, location, numel)
        assert isinstance(pid, tuple) and pid[0] == "storage", pid
        _, storage_type, key, _location, _numel = pid
        # Lazy: do not read bytes yet. Defer until the tensor is realized,
        # so a weights-only consumer can skip optimizer storages entirely.
        return _LazyStorage(
            self._zf, f"{self._prefix}data/{key}", storage_type, self._byteorder
        )


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
            raw = self._zf.read(self._member)
            # Zero-copy view into the ``bytes`` buffer. The returned array is
            # read-only — callers must not mutate it in place. Downstream
            # consumers in this loader (np.ascontiguousarray) copy into a
            # fresh buffer anyway, so this saves a full memcpy.
            arr = np.frombuffer(raw, dtype=self.storage_type.dtype)
            if self._byteorder == "big":
                arr = arr.byteswap()
            self._array = arr
        return self._array


def _rebuild_from(storage, storage_offset, size, stride):
    return _LazyTensor(storage, storage_offset, tuple(size), tuple(stride))


class _LazyTensor:
    def __init__(self, storage, offset, size, stride):
        self._storage = storage
        self._offset = offset
        self._size = size
        self._stride = stride

    def realize(self):
        flat = self._storage.array
        if len(self._size) == 0:
            return flat[self._offset:self._offset + 1].reshape(())
        itemsize = flat.itemsize
        byte_strides = tuple(s * itemsize for s in self._stride)
        view = np.lib.stride_tricks.as_strided(
            flat[self._offset:], shape=self._size, strides=byte_strides
        )
        return np.ascontiguousarray(view)


def _realize_tree(obj):
    if isinstance(obj, _LazyTensor):
        return obj.realize()
    if isinstance(obj, dict):
        return type(obj)((k, _realize_tree(v)) for k, v in obj.items())
    if isinstance(obj, (list, tuple)):
        return type(obj)(_realize_tree(v) for v in obj)
    return obj


def _bind(cls):
    def _rebuild_tensor_v2(self, storage, storage_offset, size, stride,
                           requires_grad, backward_hooks, *_extra):
        return _rebuild_from(storage, storage_offset, size, stride)

    def _rebuild_tensor(self, storage, storage_offset, size, stride):
        return _rebuild_from(storage, storage_offset, size, stride)

    def _rebuild_parameter(self, data, requires_grad, backward_hooks):
        return data

    cls.bound_rebuild_tensor_v2 = _rebuild_tensor_v2
    cls.bound_rebuild_tensor = _rebuild_tensor
    cls.bound_rebuild_parameter = _rebuild_parameter
    return cls


_bind(_TorchFreeUnpickler)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _parse_index(zf):
    """Parse the pickle index without reading tensor bytes."""
    names = zf.namelist()
    pkl_name = next(n for n in names if n.endswith("data.pkl"))
    prefix = pkl_name[: -len("data.pkl")]
    byteorder = "little"
    bo_name = prefix + "byteorder"
    if bo_name in names:
        byteorder = zf.read(bo_name).decode().strip()
    data = zf.read(pkl_name)
    up = _TorchFreeUnpickler(io.BytesIO(data), zf, prefix, byteorder)
    return up.load()


def load_from_zip(zf, weights_key="network_weights"):
    """Load checkpoint contents from an open ``ZipFile``.

    If ``weights_key`` is given and present at the top level, only that
    subtree is materialized (other storages -- optimizer state, trainer
    state -- are never read). Pass ``weights_key=None`` to realize the
    full top-level object.
    """
    tree = _parse_index(zf)
    if weights_key is not None and isinstance(tree, dict) and weights_key in tree:
        return _realize_tree(tree[weights_key])
    return _realize_tree(tree)


def load_pth(path_or_file, weights_key="network_weights"):
    """Return a ``.pth`` checkpoint as numpy arrays.

    Accepts a filesystem path or any seekable binary file-like object.
    By default returns only the ``network_weights`` subtree; pass
    ``weights_key=None`` to get the full top-level object.
    """
    if not zipfile.is_zipfile(path_or_file):
        raise NotImplementedError(
            "Legacy (non-zip) .pth format not supported by torch-free loader"
        )
    if hasattr(path_or_file, "seek"):
        path_or_file.seek(0)
    with zipfile.ZipFile(path_or_file) as zf:
        return load_from_zip(zf, weights_key=weights_key)


__all__ = ["load_pth", "load_from_zip"]
