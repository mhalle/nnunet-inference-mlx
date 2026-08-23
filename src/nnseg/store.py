"""Loading models, and optionally keeping them warm.

``segment()`` on its own builds a :class:`~nnseg.network.TorchModel` per call and drops it -
right for a one-shot script, wrong for a server, where every request would re-read the
checkpoint, rebuild the architecture, redo the ShuffleUp surgery and re-upload the weights.
:class:`ModelStore` is the seam: a bounded LRU keyed on the model folder *and* the policy that
was used to build it, so a cached model is only reused when it would be built identically.

Capacity is deliberately small by default. A warm model holds its weights on the device, and a
five-part task with everything cached is five sets of weights competing with the accumulator -
which is exactly the memory the sliding window wants. Keeping one or two warm is the useful
case; keeping all of them is a way to run out of VRAM.
"""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path


class ModelStore:
    """Builds :class:`TorchModel` instances, keeping up to ``capacity`` of them warm.

    ``capacity=0`` (the default) never caches, which is exactly ``segment()``'s historical
    behavior. Anything larger keeps the most recently used models loaded on the device.
    """

    def __init__(self, capacity: int = 0):
        self.capacity = max(0, int(capacity))
        self._warm: OrderedDict[tuple, object] = OrderedDict()
        self.hits = 0
        self.misses = 0

    @staticmethod
    def _key(folder, *, folds, device, dtype, accumulate, batch_size) -> tuple:
        # every argument that changes what gets built or where it lives is part of the key;
        # reusing a model built under a different policy would silently ignore the new one
        f = folds if isinstance(folds, str) else tuple(int(x) for x in folds)
        return (str(Path(folder)), f, str(device), str(dtype), str(accumulate), str(batch_size))

    def get(self, folder, *, folds=(0,), device="auto", dtype="fp16", accumulate="auto",
            batch_size="auto"):
        """A model for ``folder`` under this policy, warm if it is cached."""
        from .network import TorchModel
        key = self._key(folder, folds=folds, device=device, dtype=dtype,
                        accumulate=accumulate, batch_size=batch_size)
        if key in self._warm:
            self.hits += 1
            self._warm.move_to_end(key)
            return self._warm[key]
        self.misses += 1
        model = TorchModel(folder, folds=folds, device=device, dtype=dtype,
                           accumulate=accumulate, batch_size=batch_size).to_device()
        if self.capacity:
            self._warm[key] = model
            while len(self._warm) > self.capacity:
                self._evict(self._warm.popitem(last=False)[1])
        return model

    def release(self, model) -> None:
        """Hand a model back after use: freed now unless the store is keeping it warm."""
        if any(m is model for m in self._warm.values()):
            return
        self._evict(model)

    @staticmethod
    def _evict(model) -> None:
        import torch
        dev = getattr(getattr(model, "device", None), "type", None)
        del model
        if dev == "cuda":
            torch.cuda.empty_cache()
        elif dev == "mps":
            torch.mps.empty_cache()

    def clear(self) -> None:
        """Drop every warm model and free the device memory they held."""
        while self._warm:
            self._evict(self._warm.popitem(last=False)[1])

    def __len__(self) -> int:
        return len(self._warm)

    def __repr__(self) -> str:
        return (f"ModelStore(capacity={self.capacity}, warm={len(self._warm)}, "
                f"hits={self.hits}, misses={self.misses})")
