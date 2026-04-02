"""
Parse nnU-Net plans.json and build the corresponding MLX network.

Supports both the new format (network_arch_init_kwargs) and the old
format (UNet_class_name + pool_op_kernel_sizes) used by TotalSegmentator.
"""

from __future__ import annotations

import mlx.nn as nn

from .model import PlainConvUNet, ResidualEncoderUNet


def build_network_from_plans(
    plans: dict,
    configuration: str,
    num_input_channels: int,
    num_classes: int,
    deep_supervision: bool = False,
) -> nn.Module:
    """Build an MLX network from nnU-Net plans.json.

    Auto-detects old vs new plans format.
    """
    config = plans["configurations"][configuration]

    # New format: network_arch_init_kwargs at top level or in config
    arch_kwargs = plans.get(
        "network_arch_init_kwargs",
        config.get("network_arch_init_kwargs", None),
    )

    if arch_kwargs is not None:
        return _build_from_new_plans(plans, config, arch_kwargs,
                                     num_input_channels, num_classes, deep_supervision)
    else:
        return _build_from_old_plans(config, num_input_channels, num_classes, deep_supervision)


def _build_from_new_plans(
    plans: dict,
    config: dict,
    arch_kwargs: dict,
    num_input_channels: int,
    num_classes: int,
    deep_supervision: bool,
) -> nn.Module:
    """Build from new-style plans (network_arch_init_kwargs)."""
    arch_class = plans.get(
        "network_arch_class_name",
        config.get(
            "network_arch_class_name",
            "dynamic_network_architectures.architectures.unet.PlainConvUNet",
        ),
    )

    n_stages = arch_kwargs["n_stages"]
    features = arch_kwargs["features_per_stage"]
    kernel_sizes = arch_kwargs["kernel_sizes"]
    strides = arch_kwargs["strides"]
    bias = arch_kwargs.get("conv_bias", True)

    norm_kwargs = arch_kwargs.get("norm_op_kwargs", {"eps": 1e-5, "affine": True})
    nonlin_kwargs = arch_kwargs.get("nonlin_kwargs", {"inplace": True})
    nonlin_kwargs = {k: v for k, v in nonlin_kwargs.items() if k != "inplace"}
    if "negative_slope" not in nonlin_kwargs:
        nonlin_kwargs["negative_slope"] = 0.01

    if "ResidualEncoder" in arch_class:
        n_blocks = arch_kwargs.get("n_blocks_per_stage", [1] * n_stages)
        n_dec = arch_kwargs.get("n_conv_per_stage_decoder", [1] * (n_stages - 1))
        stem_ch = arch_kwargs.get("stem_channels", None)
        return ResidualEncoderUNet(
            in_channels=num_input_channels,
            n_stages=n_stages,
            features_per_stage=features,
            kernel_sizes=kernel_sizes,
            strides=strides,
            n_blocks_per_stage=n_blocks,
            num_classes=num_classes,
            n_conv_per_stage_decoder=n_dec,
            bias=bias,
            norm_kwargs=norm_kwargs,
            nonlin_kwargs=nonlin_kwargs,
            deep_supervision=deep_supervision,
            stem_channels=stem_ch,
        )
    else:
        n_conv = arch_kwargs.get("n_conv_per_stage", [2] * n_stages)
        n_dec = arch_kwargs.get("n_conv_per_stage_decoder", [2] * (n_stages - 1))
        return PlainConvUNet(
            in_channels=num_input_channels,
            n_stages=n_stages,
            features_per_stage=features,
            kernel_sizes=kernel_sizes,
            strides=strides,
            n_conv_per_stage=n_conv,
            num_classes=num_classes,
            n_conv_per_stage_decoder=n_dec,
            bias=bias,
            norm_kwargs=norm_kwargs,
            nonlin_kwargs=nonlin_kwargs,
            deep_supervision=deep_supervision,
        )


def _build_from_old_plans(
    config: dict,
    num_input_channels: int,
    num_classes: int,
    deep_supervision: bool,
) -> nn.Module:
    """Build from old-style plans (UNet_class_name + pool_op_kernel_sizes).

    Used by TotalSegmentator models (Dataset291-298, etc).
    """
    arch_class = config.get("UNet_class_name", "PlainConvUNet")
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

    norm_kwargs = {"eps": 1e-5, "affine": True}
    nonlin_kwargs = {"negative_slope": 0.01}

    if "ResidualEncoder" in arch_class:
        return ResidualEncoderUNet(
            in_channels=num_input_channels,
            n_stages=n_stages,
            features_per_stage=features,
            kernel_sizes=kernel_sizes,
            strides=strides,
            n_blocks_per_stage=n_conv_enc,
            num_classes=num_classes,
            n_conv_per_stage_decoder=n_conv_dec,
            bias=False,
            norm_kwargs=norm_kwargs,
            nonlin_kwargs=nonlin_kwargs,
            deep_supervision=deep_supervision,
        )
    else:
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
            norm_kwargs=norm_kwargs,
            nonlin_kwargs=nonlin_kwargs,
            deep_supervision=deep_supervision,
        )
