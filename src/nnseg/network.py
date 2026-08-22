"""nnU-Net model folder -> a torch network that runs fp16 on MPS, plus the sliding window."""
from __future__ import annotations

import queue
import threading
from pathlib import Path

import numpy as np
import torch

from .shuffleup import swap_transposed

DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


class TorchModel:
    """One nnU-Net configuration folder (``{trainer}__{plans}__{config}``) on torch.

    Uses nnU-Net's own predictor to build the architecture from ``plans.json`` and load
    the checkpoints (so tiling, gaussian and fold handling stay nnU-Net's), then:
    ``ShuffleUp3d`` surgery (exact; lets the decoder run in half precision on MPS),
    ``dtype`` / channels_last_3d, and our own sliding-window loop whose CPU accumulate
    runs on a worker thread while the GPU computes the next patch (bit-identical to
    nnU-Net's loop, ~25 % faster on MPS).
    """

    def __init__(self, folder, *, folds=(0,), device="mps", dtype: str = "fp16", channels_last: bool = True,
                 surgery: bool = True, accumulate_on_device: bool = False, step_size: float = 0.5):
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
        from nnunetv2.inference.sliding_window_prediction import compute_gaussian

        self.folder = Path(folder)
        self.device = torch.device(device)
        self.dtype = DTYPES[dtype]
        self.accumulate_on_device = accumulate_on_device
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

        On the CPU by default (nnU-Net's stock placement; the accumulate is hidden on a
        worker thread); on ``device`` if ``accumulate_on_device`` - which at K=118 does
        not fit on a 16 GB Apple machine next to the network.
        """
        from acvl_utils.cropping_and_padding.padding import pad_nd_image

        x = x.to(self.dtype)
        padded, revert = pad_nd_image(x, self.patch, "constant", {"value": 0}, True, None)
        slicers = self.predictor._internal_get_sliding_window_slicers(padded.shape[1:])
        total = None
        for i in range(len(self.fold_params)):
            self._load_fold(i)
            acc = self._sliding_window(padded, slicers)
            total = acc if total is None else total.add_(acc)
        if len(self.fold_params) > 1:
            total /= len(self.fold_params)
        return total[(slice(None), *revert[1:])]

    def _sliding_window(self, padded: torch.Tensor, slicers) -> torch.Tensor:
        K, shape = self.K, padded.shape[1:]
        if self.accumulate_on_device:
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
