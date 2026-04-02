"""
MLXPredictor — drop-in replacement for nnUNetPredictor.

Exposes the same interface TotalSegmentator calls:
    predictor.initialize_from_trained_model_folder(...)
    predictor.predict_from_files(dir_in, dir_out, ...)
"""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import numpy as np
import nibabel as nib

from .inference import (
    choose_batch_size,
    predict_sliding_window,
)
from .plans import build_network_from_plans
from .weights import fuzzy_load_weights


class MLXPredictor:
    """Drop-in replacement for nnUNetPredictor using MLX.

    Optimizations over the PyTorch original:
      - Batched sliding window (multiple patches per forward pass)
      - FP16 inference (halves bandwidth)
      - mx.compile for fused conv-norm-relu chains
      - Adaptive batch size based on available system memory
    """

    def __init__(
        self,
        tile_step_size: float = 0.5,
        use_gaussian: bool = True,
        use_mirroring: bool = False,
        verbose: bool = False,
        use_fp16: bool = True,
        use_compile: bool = True,
        batch_size: int | None = None,
        max_memory_gb: float | None = None,
    ):
        self.tile_step_size = tile_step_size
        self.use_gaussian = use_gaussian
        self.use_mirroring = use_mirroring
        self.verbose = verbose
        self.use_fp16 = use_fp16
        self.use_compile = use_compile
        self._forced_batch_size = batch_size
        self._max_memory_gb = max_memory_gb

        self.network = None
        self._compiled_network = None
        self.plans = None
        self.dataset_json = None
        self.patch_size = None
        self.allowed_mirroring_axes = None
        self.list_of_parameters = None
        self.num_classes = None
        self._batch_size = None

    def initialize_from_trained_model_folder(
        self,
        model_training_output_dir: str,
        use_folds: tuple | None = None,
        checkpoint_name: str = "checkpoint_final.pth",
        configuration: str = "3d_fullres",
    ):
        """Load plans, build MLX network, convert weights, compile.

        Prefers safetensors weights when available (no torch needed).
        Falls back to .pth if safetensors not found.
        """
        from .weights import load_model_weights

        model_dir = Path(model_training_output_dir)
        self.dataset_json = json.loads((model_dir / "dataset.json").read_text())
        self.plans = json.loads((model_dir / "plans.json").read_text())

        if use_folds is None:
            fold_dirs = sorted(model_dir.glob("fold_*"))
            use_folds = [
                int(d.name.split("_")[1])
                for d in fold_dirs
                if ((d / checkpoint_name).exists()
                    or (d / checkpoint_name.replace(".pth", "_mlx.safetensors")).exists())
                and d.name != "fold_all"
            ]

        # Load metadata from JSON sidecar or .pth checkpoint
        meta_path = model_dir / f"fold_{use_folds[0]}" / checkpoint_name.replace(".pth", ".json")
        if meta_path.exists():
            metadata = json.loads(meta_path.read_text())
            self.allowed_mirroring_axes = metadata.get("inference_allowed_mirroring_axes")
        else:
            # Fall back to loading .pth for metadata
            import torch
            ckpt = torch.load(
                str(model_dir / f"fold_{use_folds[0]}" / checkpoint_name),
                map_location="cpu", weights_only=False,
            )
            self.allowed_mirroring_axes = ckpt.get("inference_allowed_mirroring_axes")

        # Load weights for all folds
        self.list_of_parameters = [
            load_model_weights(model_dir, fold=f, checkpoint_name=checkpoint_name)
            for f in use_folds
        ]

        config = self.plans["configurations"][configuration]
        self.patch_size = tuple(config["patch_size"])

        modalities = self.dataset_json.get(
            "channel_names", self.dataset_json.get("modality", {})
        )
        num_input_channels = len(modalities)

        labels = self.dataset_json.get("labels", {})
        self.num_classes = len(labels)

        self.network = build_network_from_plans(
            self.plans,
            configuration,
            num_input_channels,
            self.num_classes,
            deep_supervision=False,
        )

        self._load_weights(self.list_of_parameters[0])

        if self.use_compile:
            try:
                self._compiled_network = mx.compile(self.network)
                if self.verbose:
                    print("Network compiled with mx.compile")
            except Exception as e:
                if self.verbose:
                    print(f"mx.compile failed ({e}), using uncompiled network")
                self._compiled_network = self.network
        else:
            self._compiled_network = self.network

        if self._forced_batch_size is not None:
            self._batch_size = self._forced_batch_size
        else:
            dtype_bytes = 2 if self.use_fp16 else 4
            self._batch_size = choose_batch_size(
                self.patch_size, num_classes=self.num_classes,
                dtype_bytes=dtype_bytes,
            )

        if self.verbose:
            n_params = sum(v.size for v in self.list_of_parameters[0].values())
            print(
                f"Model: {n_params:,} parameters, "
                f"patch={self.patch_size}, folds={use_folds}"
            )
            print(
                f"batch_size={self._batch_size}, "
                f"fp16={self.use_fp16}, "
                f"compile={self.use_compile}"
            )

    def _load_weights(self, weights: dict):
        """Load weight dict into network, with fallback fuzzy matching."""
        try:
            self.network.load_weights(list(weights.items()))
        except Exception:
            fuzzy_load_weights(self.network, weights, verbose=self.verbose)

    def predict_volume(self, input_data: np.ndarray) -> np.ndarray:
        """Predict on a single preprocessed volume.

        Parameters
        ----------
        input_data : np.ndarray
            Shape (C, D, H, W), float32, preprocessed and resampled.

        Returns
        -------
        np.ndarray
            Predicted logits, shape (num_classes, D, H, W), float32.
        """
        net = self._compiled_network or self.network
        prediction = None

        for i, params in enumerate(self.list_of_parameters):
            if i > 0:
                self._load_weights(params)
                if self.use_compile:
                    try:
                        net = mx.compile(self.network)
                    except Exception:
                        net = self.network

            pred = predict_sliding_window(
                network=net,
                input_image=input_data,
                patch_size=self.patch_size,
                num_classes=self.num_classes,
                tile_step_size=self.tile_step_size,
                use_gaussian=self.use_gaussian,
                use_mirroring=self.use_mirroring,
                mirror_axes=self.allowed_mirroring_axes,
                batch_size=self._batch_size,
                use_fp16=self.use_fp16,
                verbose=self.verbose and i == 0,
            )

            if prediction is None:
                prediction = pred
            else:
                prediction += pred

        if len(self.list_of_parameters) > 1:
            prediction /= len(self.list_of_parameters)

        return prediction

    def predict_from_files(
        self,
        dir_in: str,
        dir_out: str,
        save_probabilities: bool = False,
        overwrite: bool = True,
        **kwargs,
    ):
        """Predict all NIfTI files in dir_in, save results to dir_out."""
        from glob import glob

        dir_in = Path(dir_in)
        dir_out = Path(dir_out)
        dir_out.mkdir(parents=True, exist_ok=True)

        nifti_files = sorted(glob(str(dir_in / "*.nii.gz")))
        if not nifti_files:
            nifti_files = sorted(glob(str(dir_in / "*.nii")))

        for fpath in nifti_files:
            fname = Path(fpath).name
            out_path = dir_out / fname
            if not overwrite and out_path.exists():
                continue

            if self.verbose:
                print(f"Processing {fname}")

            img = nib.load(fpath)
            data = np.asarray(img.dataobj, dtype=np.float32)
            if data.ndim == 3:
                data = data[None]  # (1, D, H, W)

            # WARNING: this assumes data is already preprocessed and resampled
            # to the model's target spacing. For raw NIfTI files, use
            # nnUNetv2_predict_mlx() or TotalSegmentator's pipeline instead,
            # which handle resampling and normalization.
            logits = self.predict_volume(data)
            seg = np.argmax(logits, axis=0).astype(np.uint8)

            seg_img = nib.Nifti1Image(seg, img.affine, img.header)
            nib.save(seg_img, str(out_path))

            if save_probabilities:
                from scipy.special import softmax

                probs = softmax(logits, axis=0).astype(np.float16)
                prob_path = str(out_path).replace(".nii.gz", ".npz")
                np.savez_compressed(prob_path, probabilities=probs)

            if self.verbose:
                print(f"  Saved {out_path}")
