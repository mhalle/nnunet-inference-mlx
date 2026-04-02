"""
Test loading real TotalSegmentator weights (task 297, fast mode).

Run: python tests/test_real_weights.py
"""

import json
import time
from pathlib import Path

import numpy as np
import mlx.core as mx
import mlx.nn as nn

MODEL_DIR = Path.home() / ".totalsegmentator/nnunet/results/Dataset297_TotalSegmentator_total_3mm_1559subj/nnUNetTrainer_4000epochs_NoMirroring__nnUNetPlans__3d_fullres"


def build_from_old_plans(plans: dict, config_name: str, num_input_channels: int, num_classes: int):
    """Build MLX network from old-style nnU-Net plans (no network_arch_init_kwargs)."""
    from nnunet_mlx.model import PlainConvUNet

    config = plans["configurations"][config_name]
    strides = config["pool_op_kernel_sizes"]
    kernel_sizes = config["conv_kernel_sizes"]
    n_stages = len(strides)
    n_conv_enc = config["n_conv_per_stage_encoder"]
    n_conv_dec = config["n_conv_per_stage_decoder"]
    base_features = config.get("UNet_base_num_features", 32)
    max_features = config.get("unet_max_num_features", 320)

    # Compute features per stage: double each time, capped at max
    features = []
    f = base_features
    for _ in range(n_stages):
        features.append(min(f, max_features))
        f *= 2

    return PlainConvUNet(
        in_channels=num_input_channels,
        n_stages=n_stages,
        features_per_stage=features,
        kernel_sizes=kernel_sizes,
        strides=strides,
        n_conv_per_stage=n_conv_enc,
        num_classes=num_classes,
        n_conv_per_stage_decoder=n_conv_dec,
        bias=True,
        norm_kwargs={"eps": 1e-5, "affine": True},
        nonlin_kwargs={"negative_slope": 0.01},
        deep_supervision=False,
    )


def main():
    if not MODEL_DIR.exists():
        print(f"Model not found at {MODEL_DIR}")
        print("Run TotalSegmentator once with --fast to download it")
        return

    # Load plans and dataset info
    plans = json.loads((MODEL_DIR / "plans.json").read_text())
    dataset = json.loads((MODEL_DIR / "dataset.json").read_text())
    num_classes = len(dataset["labels"])
    num_channels = len(dataset["channel_names"])
    config = plans["configurations"]["3d_fullres"]
    patch_size = tuple(config["patch_size"])

    print(f"Model:   Dataset297 (fast, 3mm)")
    print(f"Classes: {num_classes}")
    print(f"Patch:   {patch_size}")
    print(f"Stages:  {len(config['pool_op_kernel_sizes'])}")

    # Build MLX model
    print("\nBuilding MLX model...")
    model = build_from_old_plans(plans, "3d_fullres", num_channels, num_classes)
    n_params = sum(v.size for _, v in nn.utils.tree_flatten(model.parameters()))
    print(f"Parameters: {n_params:,}")

    # Load PyTorch weights
    print("Loading PyTorch checkpoint...")
    import torch
    ckpt = torch.load(
        str(MODEL_DIR / "fold_0" / "checkpoint_final.pth"),
        map_location="cpu", weights_only=False
    )
    pt_weights = ckpt["network_weights"]
    print(f"PyTorch keys: {len(pt_weights)}")

    # Convert weights
    print("Converting weights...")
    from nnunet_mlx.weights import convert_pytorch_weights, fuzzy_load_weights
    mlx_weights = convert_pytorch_weights(pt_weights)
    print(f"MLX keys: {len(mlx_weights)}")

    # Load into model
    print("Loading weights into MLX model...")
    try:
        model.load_weights(list(mlx_weights.items()))
        print("Direct load: OK")
    except Exception as e:
        print(f"Direct load failed: {e}")
        print("Trying fuzzy load...")
        fuzzy_load_weights(model, mlx_weights, verbose=True)

    # Test forward pass with synthetic CT-like data
    # CT values typically range -1024 to 3071 HU
    print("\nForward pass on synthetic volume...")
    volume = np.random.uniform(-1024, 1000, (1, *patch_size)).astype(np.float32)
    # Convert to channels-last
    x = mx.array(volume[0].transpose(1, 2, 0)[None, ..., None])  # (1, H, W, D, 1)
    # Actually: volume is (1, D, H, W), need (1, D, H, W, C)
    x = mx.array(volume.transpose(0, 1, 2, 3)[0][None, ..., None])

    compiled = mx.compile(model)

    t0 = time.perf_counter()
    out = compiled(x)
    mx.eval(out)
    dt = time.perf_counter() - t0

    out_np = np.array(out)
    seg = out_np[0].argmax(axis=-1)

    print(f"Input:   {x.shape}")
    print(f"Output:  {out.shape}")
    print(f"Seg:     {seg.shape}")
    print(f"Labels:  {np.unique(seg).size} unique")
    print(f"Time:    {dt:.2f}s")
    print(f"\nSample predictions (center voxel):")
    center = tuple(s // 2 for s in patch_size)
    logits = out_np[0, center[0], center[1], center[2], :]
    top5 = np.argsort(logits)[-5:][::-1]
    label_names = list(dataset["labels"].keys())
    for idx in top5:
        name = label_names[idx] if idx < len(label_names) else f"class_{idx}"
        print(f"  {name}: {logits[idx]:.4f}")


if __name__ == "__main__":
    main()
