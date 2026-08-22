"""nnU-Net model folder -> a torch network that runs fp16 on MPS, plus the sliding window."""
from __future__ import annotations

import queue
import threading
import warnings
from pathlib import Path

import numpy as np
import torch

from .shuffleup import swap_transposed

DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
ACCUMULATE = ("auto", "device", "host")

# Empirical reserve for weights + one patch's activations, from E1/E2 on torch 2.13:
# fp16 nnU-Net, 112x112x128 patch, K=118 -> 4.3 GB driver-allocated; 128^3, K=25 -> 2.5 GB.
# Deliberately conservative; override per call when you know your model.
DEFAULT_ACTIVATION_RESERVE_GB = 4.5


def host_available_bytes() -> int | None:
    """Memory the OS could hand out now without swapping, or None if unknown.

    Uses psutil when present; otherwise, on macOS, free + inactive + speculative pages
    from ``vm_stat`` (what the kernel will reclaim without paging out).
    """
    try:
        import psutil
        return int(psutil.virtual_memory().available)
    except Exception:
        pass
    import platform
    import subprocess
    if platform.system() != "Darwin":
        return None
    try:
        out = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=5).stdout
        pages = {}
        for line in out.splitlines():
            if ":" not in line:
                continue
            k, v = line.split(":", 1)
            v = v.strip().rstrip(".")
            if v.isdigit():
                pages[k.strip()] = int(v)
        page = 16384 if platform.machine() == "arm64" else 4096
        got = sum(pages.get(k, 0) for k in ("Pages free", "Pages inactive", "Pages speculative"))
        return int(got * page) if got else None
    except Exception:
        return None


def device_budget_bytes(device: torch.device, *, host_headroom_gb: float = 3.0) -> int | None:
    """Allocatable bytes for a big buffer on ``device`` right now, or None if unknown.

    CUDA: the card's free memory - a discrete pool, so it is exactly the budget.

    **MPS is unified memory**, so ``recommended_max_memory()`` is a hardware ceiling, not
    availability: a GPU allocation consumes the same RAM the OS is using, and taking the
    ceiling drives the machine into swap (measured 2026-08-22 - a 9 GB budget on a 16 GB M2
    Air pushed ``kern.memorystatus_level`` to 10 % and tripped the bench guard). So the MPS
    budget is the *smaller* of the Metal ceiling (honoring
    PYTORCH_MPS_HIGH_WATERMARK_RATIO, which is what actually caps allocations) and what the
    host can spare, keeping ``host_headroom_gb`` for everything else.
    """
    import os
    if device.type == "cuda":
        free, _total = torch.cuda.mem_get_info(device)
        return int(free)
    if device.type != "mps":
        return None
    rec = float(torch.mps.recommended_max_memory())
    ratio = os.environ.get("PYTORCH_MPS_HIGH_WATERMARK_RATIO")
    cap = rec * float(ratio) if ratio else rec
    device_left = max(0.0, cap - torch.mps.driver_allocated_memory())
    host = host_available_bytes()
    if host is None:
        return int(device_left)
    return int(min(device_left, max(0.0, host - host_headroom_gb * 1e9)))


def choose_accumulate(policy: str, *, device: torch.device, K: int, shape, bytes_per_element: int = 2,
                      activation_reserve_gb: float = DEFAULT_ACTIVATION_RESERVE_GB,
                      host_headroom_gb: float = 3.0) -> tuple[bool, str]:
    """Where the sliding-window accumulator lives. Returns ``(on_device, why)``.

    The accumulator is ``K`` channels plus a weight map at the *padded model grid*: on a
    16 GB Apple machine at K=118 that is 1.6 GB and does not fit beside the network, but on
    a 64 GB Mac or a CUDA card with headroom it does - and on-device accumulation is worth
    ~25 % of the per-patch time (`docs/backend-decision.md` E2-accum). So this is a runtime
    decision from the actual budget, never a hard-coded default. ``"device"`` / ``"host"``
    force it; a forced ``"device"`` that OOMs still falls back, with a warning.
    """
    if policy not in ACCUMULATE:
        raise ValueError(f"accumulate must be one of {ACCUMULATE}; got {policy!r}")
    n = 1
    for d in shape:
        n *= int(d)
    need = int((K + 1) * n * bytes_per_element)
    if policy == "host":
        return False, f"forced host (accumulator would be {need / 1e9:.2f} GB)"
    if device.type == "cpu":
        return False, "cpu device: accumulator is already in host memory"
    budget = device_budget_bytes(device, host_headroom_gb=host_headroom_gb)
    if policy == "device":
        return True, f"forced device ({need / 1e9:.2f} GB accumulator, {budget / 1e9:.2f} GB free)" if budget else "forced device"
    if budget is None:
        return False, "no device budget reported"
    reserve = int(activation_reserve_gb * 1e9)
    fits = need + reserve <= budget
    return fits, (f"{'fits' if fits else 'does not fit'}: accumulator {need / 1e9:.2f} GB + reserve "
                  f"{reserve / 1e9:.1f} GB vs {budget / 1e9:.2f} GB free on {device}")


class TorchModel:
    """One nnU-Net configuration folder (``{trainer}__{plans}__{config}``) on torch.

    Uses nnU-Net's own predictor to build the architecture from ``plans.json`` and load
    the checkpoints (so tiling, gaussian and fold handling stay nnU-Net's), then:
    ``ShuffleUp3d`` surgery (exact; lets the decoder run in half precision on MPS),
    ``dtype`` / channels_last_3d, and a sliding-window loop with two accumulator
    placements, chosen at run time from the device's actual free memory
    (``accumulate="auto"``, or force with ``"device"`` / ``"host"``):

    * **device** - accumulate on the GPU. Fastest; needs ``(K + 1) * voxels * 2`` bytes
      beside the network (1.6 GB at K=118 for a chest at 3 mm). The right choice on a
      large Mac or a CUDA card with headroom.
    * **host** - accumulate in host memory, with the fp16 add on a worker thread while the
      GPU computes the next patch: bit-identical to nnU-Net's loop and ~25 % faster than it,
      and it keeps whole-body inference inside a modest GPU budget.
    """

    def __init__(self, folder, *, folds=(0,), device="mps", dtype: str = "fp16", channels_last: bool = True,
                 surgery: bool = True, accumulate: str = "auto", step_size: float = 0.5,
                 activation_reserve_gb: float = DEFAULT_ACTIVATION_RESERVE_GB):
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
        from nnunetv2.inference.sliding_window_prediction import compute_gaussian

        self.folder = Path(folder)
        self.device = torch.device(device)
        self.dtype = DTYPES[dtype]
        if accumulate not in ACCUMULATE:
            raise ValueError(f"accumulate must be one of {ACCUMULATE}; got {accumulate!r}")
        self.accumulate = accumulate
        self.activation_reserve_gb = float(activation_reserve_gb)
        self.accumulate_choice = None                       # set per volume in predict_logits
        p = nnUNetPredictor(tile_step_size=step_size, use_gaussian=True, use_mirroring=False,
                            perform_everything_on_device=False, device=self.device, verbose=False, allow_tqdm=False)
        p.initialize_from_trained_model_folder(str(self.folder), use_folds=tuple(folds), checkpoint_name="checkpoint_final.pth")
        self.predictor = p
        self.plans = p.plans_manager.plans
        self.dataset_json = p.dataset_json
        self.label_manager = p.label_manager
        self.K = int(p.label_manager.num_segmentation_heads)
        self.patch = tuple(int(x) for x in p.configuration_manager.patch_size)
        self.spacing_zyx = tuple(float(x) for x in p.configuration_manager.spacing)
        self.transpose_forward = tuple(p.plans_manager.transpose_forward)
        self.fold_params = list(p.list_of_parameters)
        self.net = p.network.to(self.device).eval()
        self.n_swapped = swap_transposed(self.net) if surgery else 0
        self.net.to(self.dtype)
        if channels_last:
            self.net.to(memory_format=torch.channels_last_3d)
        self.gaussian = compute_gaussian(self.patch, sigma_scale=1. / 8, value_scaling_factor=10, device=self.device)
        self._gaussian_cpu = self.gaussian.cpu()

    # -- normalization (nnU-Net plans) ------------------------------------------
    @property
    def normalization_schemes(self) -> tuple[str, ...]:
        return tuple(self.predictor.configuration_manager.normalization_schemes)

    def intensity_properties(self, channel: int = 0) -> dict:
        return self.plans["foreground_intensity_properties_per_channel"][str(channel)]

    # -- sliding window ------------------------------------------------------------
    def _load_fold(self, i: int) -> None:
        state = self.fold_params[i]
        if self.n_swapped:
            # the checkpoint holds ConvTranspose3d weights; re-derive the shuffled 1x1 convs
            from .shuffleup import ShuffleUp3d
            current = dict(self.net.named_modules())
            tmp = {}
            for name, m in current.items():
                if isinstance(m, ShuffleUp3d):
                    tmp[name] = m
            # simplest exact route: load into a fresh fp32 copy of the architecture is expensive;
            # for now only fold 0 (already loaded at init) is supported with surgery.
            if i != 0:
                raise NotImplementedError("multi-fold with ShuffleUp3d surgery: load folds before surgery (todo)")
            return
        self.net.load_state_dict(state)
        self.net.to(self.dtype)

    @torch.inference_mode()
    def predict_logits(self, x: torch.Tensor) -> torch.Tensor:
        """``(C, Z, Y, X)`` preprocessed float tensor -> ``(K, Z, Y, X)`` logits (fp16).

        The accumulator's placement follows ``self.accumulate`` and the device's free
        memory at this moment (see :func:`choose_accumulate`); the choice, and why, is
        recorded in ``self.accumulate_choice``.
        """
        from acvl_utils.cropping_and_padding.padding import pad_nd_image

        x = x.to(self.dtype)
        padded, revert = pad_nd_image(x, self.patch, "constant", {"value": 0}, True, None)
        slicers = self.predictor._internal_get_sliding_window_slicers(padded.shape[1:])
        on_device, why = choose_accumulate(self.accumulate, device=self.device, K=self.K,
                                           shape=padded.shape[1:],
                                           activation_reserve_gb=self.activation_reserve_gb)
        self.accumulate_choice = {"on_device": on_device, "why": why}
        total = None
        for i in range(len(self.fold_params)):
            self._load_fold(i)
            try:
                acc = self._sliding_window(padded, slicers, on_device=on_device)
            except (RuntimeError, torch.OutOfMemoryError) if hasattr(torch, "OutOfMemoryError") else RuntimeError as e:
                if not on_device or "memory" not in str(e).lower():
                    raise
                warnings.warn(f"on-device accumulation ran out of memory ({e}); falling back to host", stacklevel=2)
                if self.device.type == "mps":
                    torch.mps.empty_cache()
                elif self.device.type == "cuda":
                    torch.cuda.empty_cache()
                on_device = False
                self.accumulate_choice = {"on_device": False, "why": f"{why}; then OOM -> host"}
                acc = self._sliding_window(padded, slicers, on_device=False)
            total = acc if total is None else total.add_(acc)
        if len(self.fold_params) > 1:
            total /= len(self.fold_params)
        return total[(slice(None), *revert[1:])]

    def _sliding_window(self, padded: torch.Tensor, slicers, *, on_device: bool) -> torch.Tensor:
        K, shape = self.K, padded.shape[1:]
        if on_device:
            acc = torch.zeros((K, *shape), dtype=torch.half, device=self.device)
            n_pred = torch.zeros(shape, dtype=torch.half, device=self.device)
            data = padded.to(self.device)
            for sl in slicers:
                pred = self.net(data[sl][None])[0]
                pred *= self.gaussian
                acc[sl] += pred
                n_pred[sl[1:]] += self.gaussian
            torch.div(acc, n_pred, out=acc)
            return acc
        acc = torch.zeros((K, *shape), dtype=torch.half)
        n_pred = torch.zeros(shape, dtype=torch.half)
        gauss_cpu = self._gaussian_cpu
        q: queue.Queue = queue.Queue(maxsize=2)
        err: list = []

        def worker():
            with torch.inference_mode():
                try:
                    while True:
                        item = q.get()
                        if item is None:
                            q.task_done(); break
                        host, sl = item
                        acc[sl] += host
                        n_pred[sl[1:]] += gauss_cpu
                        q.task_done()
                except Exception as e:                                 # pragma: no cover
                    err.append(e)

        t = threading.Thread(target=worker, daemon=True)
        t.start()
        for sl in slicers:
            pred = self.net(padded[sl][None].to(self.device))[0]
            pred *= self.gaussian
            q.put((pred.to("cpu"), sl))                                # the copy is cheap; the add overlaps the next forward
        q.put(None)
        t.join()
        if err:
            raise err[0]
        torch.div(acc, n_pred, out=acc)
        if not torch.isfinite(acc).all():
            raise RuntimeError("non-finite logits after accumulation")
        return acc


def resolve_model_folder(weights_id, *, ecosystem: str = "totalsegmentator", model_root=None) -> Path:
    """``Dataset{id}_*`` -> its configuration folder, through the toolkit's ecosystem table."""
    from nnunet_inference_mlx.store import _ECOSYSTEMS, _resolve_model_root_dir
    root = _resolve_model_root_dir(ecosystem, model_root)
    if root is None:
        raise FileNotFoundError(f"no model root for ecosystem {ecosystem!r}; pass model_root")
    resolve = _ECOSYSTEMS[ecosystem][0]
    return resolve(root, weights_id)
