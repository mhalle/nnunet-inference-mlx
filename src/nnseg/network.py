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

# Fallback reserve for weights + one patch's activations when nothing has been measured yet,
# from E1/E2 on torch 2.13: fp16 nnU-Net, 112x112x128 patch, K=118 -> 4.3 GB driver-allocated;
# 128^3, K=25 -> 2.5 GB. One constant cannot fit both, which is why the real decision is made
# from a measurement of the first patch (see TorchModel._sliding_window); this is only used
# when a caller asks for a decision before any patch has run.
DEFAULT_ACTIVATION_RESERVE_GB = 4.5


def device_working_set_bytes(device: torch.device) -> int | None:
    """What the framework currently holds on ``device`` (weights, activations, cache)."""
    if device.type == "mps":
        return int(torch.mps.driver_allocated_memory())
    if device.type == "cuda":
        return int(torch.cuda.memory_reserved(device))
    return None


def host_memory_health() -> int | None:
    """macOS ``kern.memorystatus_level``: the percentage of memory the kernel considers free.
    Lower than "available" suggests, because it accounts for what is already paged out."""
    import platform
    import subprocess
    if platform.system() != "Darwin":
        return None
    try:
        out = subprocess.run(["sysctl", "-n", "kern.memorystatus_level"], capture_output=True, text=True, timeout=5)
        return int(out.stdout.strip())
    except Exception:
        return None


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


def device_budget_bytes(device: torch.device, *, host_headroom_gb: float = 3.0,
                        unified_fraction: float = 0.5) -> int | None:
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
    # Only a fraction of "available" is durably ours: on a machine with pages already swapped
    # out, reclaiming inactive pages pushes other processes' memory to swap, and the swapping is
    # the thing that hurts. Measured 2026-08-22: a snapshot showing 6.9 GB available let the
    # policy take a 1.6 GB accumulator on device at K=118, after which the machine swapped at
    # 4.9 GB/min. Taking the budget is what destroys the budget, on unified memory.
    return int(min(device_left, max(0.0, (host - host_headroom_gb * 1e9) * unified_fraction)))


CUDA_AUTO_BATCH = 4


def choose_batch(policy, *, device: torch.device, on_device: bool, held_bytes: int | None,
                 budget_bytes: int | None, accumulator_bytes: int) -> tuple[int, str]:
    """Patches per forward pass. An int is taken as given; ``"auto"`` decides from measurements.

    Batch 1 is the measured optimum on Apple silicon (one patch saturates a bandwidth-bound
    GPU). On CUDA, measured on an A10 in one container with alternated conditions: batch 2 is
    12 % faster than 1 with no warmup cost, batch 4 is 18 % faster in steady state but its
    first use of a patch shape pays ~7 s of cuDNN autotuning - still a net win for a whole-body
    job, where five parts share one shape. Activations scale about linearly with batch, so
    the rule is: CUDA, accumulator on the device, and ``CUDA_AUTO_BATCH`` x the measured
    working set still fits beside the accumulator with margin.
    """
    if policy != "auto":
        b = max(1, int(policy))
        return b, f"batch {b} (requested)"
    if device.type != "cuda":
        return 1, "batch 1: not CUDA (1 is the measured optimum on Apple silicon)"
    if not on_device:
        return 1, "batch 1: accumulator is on the host, so the device-side batched loop does not apply"
    if held_bytes is None or budget_bytes is None:
        return 1, "batch 1: no memory measurement available"
    need = CUDA_AUTO_BATCH * held_bytes + accumulator_bytes
    if need + int(0.5e9) <= budget_bytes + held_bytes:      # budget was measured with b=1 already held
        return CUDA_AUTO_BATCH, (f"batch {CUDA_AUTO_BATCH}: {CUDA_AUTO_BATCH}x the measured {held_bytes / 1e9:.2f} GB "
                                 f"working set + {accumulator_bytes / 1e9:.2f} GB accumulator fits in "
                                 f"{(budget_bytes + held_bytes) / 1e9:.2f} GB")
    return 1, (f"batch 1: {CUDA_AUTO_BATCH}x the measured {held_bytes / 1e9:.2f} GB working set would not fit "
               f"beside the {accumulator_bytes / 1e9:.2f} GB accumulator in {(budget_bytes + held_bytes) / 1e9:.2f} GB")


def choose_accumulate(policy: str, *, device: torch.device, K: int, shape, bytes_per_element: int = 2,
                      activation_reserve_gb: float = DEFAULT_ACTIVATION_RESERVE_GB,
                      host_headroom_gb: float = 3.0, measured: bool = False,
                      safety_fraction: float = 0.25, min_host_health: int = 35) -> tuple[bool, str]:
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
    if device.type == "mps":
        health = host_memory_health()
        if health is not None and health < min_host_health:
            return False, (f"host memory is already tight (kern.memorystatus_level {health}% < "
                           f"{min_host_health}%): on unified memory a device accumulator would come "
                           f"out of the same pool and push the machine into swap")
    if measured:
        # The network is already resident, so `budget` is what is genuinely left; reserve only a
        # margin for per-patch transients the caching allocator has not yet settled.
        held = device_working_set_bytes(device) or 0
        margin = max(int(0.5e9), int(safety_fraction * held))
        fits = need + margin <= budget
        return fits, (f"{'fits' if fits else 'does not fit'}: accumulator {need / 1e9:.2f} GB + margin "
                      f"{margin / 1e9:.2f} GB vs {budget / 1e9:.2f} GB free on {device} "
                      f"(measured: network holds {held / 1e9:.2f} GB)")
    reserve = int(activation_reserve_gb * 1e9)
    fits = need + reserve <= budget
    return fits, (f"{'fits' if fits else 'does not fit'}: accumulator {need / 1e9:.2f} GB + estimated reserve "
                  f"{reserve / 1e9:.1f} GB vs {budget / 1e9:.2f} GB free on {device} (unmeasured)")


def available_folds(folder, folds) -> tuple:
    """The requested folds, restricted to the ``fold_*`` directories that exist.

    TotalSegmentator ships fold_0 only, so ``folds=(0,)`` is the right default - but a stock
    nnU-Net result folder may hold any subset (the knee reference model ships only fold_1).
    ``folds="all"`` takes whatever is on disk. Asking for folds that are all missing is an
    error naming what is there, rather than nnU-Net's bare file-not-found.
    """
    root = Path(folder)
    have = sorted(int(p.name.split("_")[1]) for p in root.glob("fold_*")
                  if p.is_dir() and p.name.split("_")[-1].isdigit())
    if not have:
        raise FileNotFoundError(f"no fold_* directory in {root}")
    if folds is None or (isinstance(folds, str) and folds == "all"):
        return tuple(have)
    want = [int(f) for f in ((folds,) if isinstance(folds, int) else folds)]
    keep = [f for f in want if f in have]
    if not keep:
        raise FileNotFoundError(f"{root.name}: requested fold(s) {want} not present; have {have}")
    return tuple(keep)


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

    def __init__(self, folder, *, folds=(0,), device="auto", dtype: str = "fp16", channels_last: bool = True,
                 surgery: bool = True, accumulate: str = "auto", step_size: float = 0.5,
                 activation_reserve_gb: float = DEFAULT_ACTIVATION_RESERVE_GB, batch_size="auto",
                 defer_device: bool = False):
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
        from nnunetv2.inference.sliding_window_prediction import compute_gaussian

        from .resample import resolve_device
        self.folder = Path(folder)
        self.device = resolve_device(device)
        self.dtype = DTYPES[dtype]
        if accumulate not in ACCUMULATE:
            raise ValueError(f"accumulate must be one of {ACCUMULATE}; got {accumulate!r}")
        self.accumulate = accumulate
        self.activation_reserve_gb = float(activation_reserve_gb)
        # Patches per forward pass: an int, or "auto" (decided per volume after the first patch
        # from measured memory - see choose_batch). Only the device-side accumulate batches;
        # the host path keeps its per-patch pipeline.
        self.batch_size = batch_size
        self.batch_choice = None
        self.accumulate_choice = None                       # set per volume in predict_logits
        from .trainers import ensure_trainer
        ensure_trainer(self.folder)                       # shim custom trainers (e.g. SkeletonRecall)
        p = nnUNetPredictor(tile_step_size=step_size, use_gaussian=True, use_mirroring=False,
                            perform_everything_on_device=False, device=self.device, verbose=False, allow_tqdm=False)
        p.initialize_from_trained_model_folder(str(self.folder), use_folds=available_folds(self.folder, folds),
                                               checkpoint_name="checkpoint_final.pth")
        self.predictor = p
        self.plans = p.plans_manager.plans
        self.dataset_json = p.dataset_json
        self.label_manager = p.label_manager
        self.K = int(p.label_manager.num_segmentation_heads)
        self.patch = tuple(int(x) for x in p.configuration_manager.patch_size)
        self.spacing_zyx = tuple(float(x) for x in p.configuration_manager.spacing)
        self.transpose_forward = tuple(p.plans_manager.transpose_forward)
        if tuple(self.transpose_forward) != (0, 1, 2):
            # nnU-Net permutes the spatial axes before preprocessing, and configuration_manager
            # .spacing is already expressed in that permuted frame - so ignoring it silently
            # resamples to the wrong spacing per axis. Refuse rather than be quietly wrong.
            raise NotImplementedError(
                f"{self.folder.name}: plans set transpose_forward={self.transpose_forward}; nnseg "
                "only supports the identity (0, 1, 2) today. The model spacing is expressed in the "
                "transposed frame, so running it unpermuted would resample the wrong axes.")
        self.fold_params = list(p.list_of_parameters)
        # Everything up to here is CPU work (checkpoint read, architecture build, surgery) and
        # safe to do on a helper thread. The device move is deliberately separate: a helper
        # thread copying weights to the GPU contends with a running prediction - measured on
        # an A10 at batch 4, prefetching whole models cost more network time than it saved.
        self.net = p.network.eval()
        self.n_swapped = swap_transposed(self.net) if surgery else 0
        self.net.to(self.dtype)
        self.channels_last = channels_last
        self._gaussian_cpu = compute_gaussian(self.patch, sigma_scale=1. / 8, value_scaling_factor=10,
                                              device=torch.device("cpu"))
        self.gaussian = None
        self._on_device = False
        if not defer_device:
            self.to_device()

    def to_device(self) -> "TorchModel":
        """Move the network to ``self.device``. Idempotent; call from the thread that predicts."""
        if not self._on_device:
            self.net.to(self.device)
            if self.channels_last:
                self.net.to(memory_format=torch.channels_last_3d)
            self.gaussian = self._gaussian_cpu.to(self.device)
            self._on_device = True
        return self

    # -- normalization (nnU-Net plans) ------------------------------------------
    @property
    def normalization_schemes(self) -> tuple[str, ...]:
        return tuple(self.predictor.configuration_manager.normalization_schemes)

    def intensity_properties(self, channel: int = 0) -> dict:
        return self.plans.get("foreground_intensity_properties_per_channel", {}).get(str(channel), {})

    @property
    def use_mask_for_norm(self):
        return getattr(self.predictor.configuration_manager, "use_mask_for_norm", None)

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

        self.to_device()
        x = x.to(self.dtype)
        padded, revert = pad_nd_image(x, self.patch, "constant", {"value": 0}, True, None)
        slicers = self.predictor._internal_get_sliding_window_slicers(padded.shape[1:])
        total = None
        for i in range(len(self.fold_params)):
            self._load_fold(i)
            try:
                acc = self._sliding_window(padded, slicers)
            except (RuntimeError, torch.OutOfMemoryError) if hasattr(torch, "OutOfMemoryError") else RuntimeError as e:
                if not self.accumulate_choice["on_device"] or "memory" not in str(e).lower():
                    raise
                warnings.warn(f"on-device accumulation ran out of memory ({e}); falling back to host", stacklevel=2)
                if self.device.type == "mps":
                    torch.mps.empty_cache()
                elif self.device.type == "cuda":
                    torch.cuda.empty_cache()
                acc = self._sliding_window(padded, slicers, force_host=True)
            total = acc if total is None else total.add_(acc)
        if len(self.fold_params) > 1:
            total /= len(self.fold_params)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        elif self.device.type == "mps":
            torch.mps.synchronize()
        return total[(slice(None), *revert[1:])]

    def _patch(self, padded: torch.Tensor, sl):
        pred = self.net(padded[sl][None].to(self.device))[0]
        pred *= self.gaussian
        return pred

    def _sliding_window(self, padded: torch.Tensor, slicers, *, force_host: bool = False) -> torch.Tensor:
        """Placement is decided after the *first* patch has run, from what the network actually
        holds on the device - no extra compute, and no guessing at an activation reserve that
        varies 2 GB between models."""
        K, shape = self.K, padded.shape[1:]
        with torch.inference_mode():
            first = self._patch(padded, slicers[0])
        if force_host:
            on_device, why = False, "forced host after an out-of-memory fallback"
        else:
            on_device, why = choose_accumulate(self.accumulate, device=self.device, K=K, shape=shape,
                                               activation_reserve_gb=self.activation_reserve_gb, measured=True)
        self.accumulate_choice = {"on_device": on_device, "why": why}
        held = device_working_set_bytes(self.device)
        budget = device_budget_bytes(self.device) if on_device else None
        n = 1
        for d in shape:
            n *= int(d)
        b, bwhy = choose_batch(self.batch_size, device=self.device, on_device=on_device, held_bytes=held,
                               budget_bytes=budget, accumulator_bytes=(K + 1) * n * 2)
        self.batch_choice = {"batch": b, "why": bwhy}
        return self._accumulate(padded, slicers, first, on_device, batch=b)

    @torch.inference_mode()
    def _accumulate(self, padded: torch.Tensor, slicers, first: torch.Tensor, on_device: bool,
                    batch: int = 1) -> torch.Tensor:
        K, shape = self.K, padded.shape[1:]
        if on_device:
            acc = torch.zeros((K, *shape), dtype=torch.half, device=self.device)
            n_pred = torch.zeros(shape, dtype=torch.half, device=self.device)
            acc[slicers[0]] += first
            n_pred[slicers[0][1:]] += self.gaussian
            del first
            rest = slicers[1:]
            B = batch
            if B == 1:
                for sl in rest:
                    pred = self._patch(padded, sl)
                    acc[sl] += pred
                    n_pred[sl[1:]] += self.gaussian
            else:
                data = padded.to(self.device)
                for i in range(0, len(rest), B):
                    group = rest[i:i + B]
                    x = torch.stack([data[sl] for sl in group])          # (b, C, *patch)
                    preds = self.net(x)
                    preds *= self.gaussian                                # broadcasts over the batch
                    for pred, sl in zip(preds, group):
                        acc[sl] += pred
                        n_pred[sl[1:]] += self.gaussian
                    del x, preds
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
        q.put((first.to("cpu"), slicers[0]))
        del first
        for sl in slicers[1:]:
            pred = self._patch(padded, sl)
            q.put((pred.to("cpu"), sl))                                # the copy is cheap; the add overlaps the next forward
        q.put(None)
        t.join()
        if err:
            raise err[0]
        torch.div(acc, n_pred, out=acc)
        if not torch.isfinite(acc).all():
            raise RuntimeError("non-finite logits after accumulation")
        return acc


# re-exported so callers keep importing it from here
from .tasks import resolve_model_folder  # noqa: E402,F401
