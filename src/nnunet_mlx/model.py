"""
MLX nnU-Net model architectures.

Building blocks and full U-Net architectures (PlainConvUNet, ResidualEncoderUNet)
ported from nnU-Net's dynamic_network_architectures to MLX.

All modules use channels-last layout: (B, D, H, W, C).
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_list(val, ndim: int) -> list:
    """Convert scalar or sequence to list of length ndim."""
    if isinstance(val, (list, tuple)):
        return list(val)
    return [val] * ndim


def _same_padding(kernel_size: list) -> list:
    """Compute 'same' padding for a given kernel size."""
    return [(k - 1) // 2 for k in kernel_size]


# ---------------------------------------------------------------------------
# Building Blocks
# ---------------------------------------------------------------------------

class ConvNormNonlin(nn.Module):
    """Conv3d -> InstanceNorm -> LeakyReLU (the atomic nnU-Net unit)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: list[int],
        stride: list[int],
        bias: bool = True,
        use_norm: bool = True,
        norm_kwargs: dict | None = None,
        use_nonlin: bool = True,
        nonlin_kwargs: dict | None = None,
    ):
        super().__init__()
        self.conv = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=tuple(kernel_size),
            stride=tuple(stride),
            padding=tuple(_same_padding(kernel_size)),
            bias=bias,
        )
        self.norm = None
        if use_norm:
            nk = norm_kwargs or {}
            self.norm = nn.InstanceNorm(
                out_channels, eps=nk.get("eps", 1e-5), affine=nk.get("affine", True)
            )
        self.nonlin = None
        if use_nonlin:
            nk = nonlin_kwargs or {}
            self.nonlin = nn.LeakyReLU(negative_slope=nk.get("negative_slope", 0.01))

    def __call__(self, x):
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.nonlin is not None:
            x = self.nonlin(x)
        return x


class StackedConvBlocks(nn.Module):
    """N sequential ConvNormNonlin blocks (first may be strided)."""

    def __init__(
        self,
        num_convs: int,
        in_channels: int,
        out_channels: int,
        kernel_size: list[int],
        initial_stride: list[int],
        bias: bool = True,
        norm_kwargs: dict | None = None,
        nonlin_kwargs: dict | None = None,
    ):
        super().__init__()
        layers = [
            ConvNormNonlin(
                in_channels, out_channels, kernel_size, initial_stride,
                bias=bias, norm_kwargs=norm_kwargs, nonlin_kwargs=nonlin_kwargs,
            )
        ]
        for _ in range(1, num_convs):
            layers.append(
                ConvNormNonlin(
                    out_channels, out_channels, kernel_size, [1] * len(kernel_size),
                    bias=bias, norm_kwargs=norm_kwargs, nonlin_kwargs=nonlin_kwargs,
                )
            )
        self.convs = layers
        self.output_channels = out_channels

    def __call__(self, x):
        for c in self.convs:
            x = c(x)
        return x


class _AvgPool3d(nn.Module):
    """Simple 3D average pooling (channels-last).

    Assumes kernel_size == stride (non-overlapping), which is always the case
    in nnU-Net residual blocks.
    """

    def __init__(self, kernel_size: list[int], stride: list[int]):
        super().__init__()
        self.kernel_size = tuple(kernel_size)
        self.stride = tuple(stride)

    def __call__(self, x):
        B, D, H, W, C = x.shape
        kd, kh, kw = self.kernel_size
        sd, sh, sw = self.stride
        D2 = (D // sd) * sd
        H2 = (H // sh) * sh
        W2 = (W // sw) * sw
        x = x[:, :D2, :H2, :W2, :]
        x = x.reshape(B, D2 // sd, sd, H2 // sh, sh, W2 // sw, sw, C)
        x = mx.mean(x, axis=(2, 4, 6))
        return x


class BasicBlockD(nn.Module):
    """ResNet-D style basic residual block.

    conv1(strided) -> conv2(no nonlin) -> + skip -> nonlin
    Skip uses avgpool + 1x1 conv when stride > 1 or channels change.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: list[int],
        stride: list[int],
        bias: bool = False,
        norm_kwargs: dict | None = None,
        nonlin_kwargs: dict | None = None,
    ):
        super().__init__()
        self.stride = stride
        ndim = len(kernel_size)

        self.conv1 = ConvNormNonlin(
            in_channels, out_channels, kernel_size, stride,
            bias=bias, norm_kwargs=norm_kwargs, nonlin_kwargs=nonlin_kwargs,
        )
        self.conv2 = ConvNormNonlin(
            out_channels, out_channels, kernel_size, [1] * ndim,
            bias=bias, norm_kwargs=norm_kwargs, use_nonlin=False,
        )

        nk = nonlin_kwargs or {}
        self.nonlin = nn.LeakyReLU(negative_slope=nk.get("negative_slope", 0.01))

        has_stride = any(s != 1 for s in stride)
        needs_proj = in_channels != out_channels

        self.skip_pool = None
        self.skip_conv = None
        if has_stride:
            self.skip_pool = _AvgPool3d(kernel_size=stride, stride=stride)
        if needs_proj:
            self.skip_conv = ConvNormNonlin(
                in_channels, out_channels, [1] * ndim, [1] * ndim,
                bias=False, norm_kwargs=norm_kwargs, use_nonlin=False,
            )

    def __call__(self, x):
        residual = x
        if self.skip_pool is not None:
            residual = self.skip_pool(residual)
        if self.skip_conv is not None:
            residual = self.skip_conv(residual)
        out = self.conv2(self.conv1(x))
        out = out + residual
        return self.nonlin(out)


# ---------------------------------------------------------------------------
# Encoders
# ---------------------------------------------------------------------------

class PlainConvEncoder(nn.Module):
    """Encoder using stacked conv blocks (PlainConvUNet)."""

    def __init__(
        self,
        in_channels: int,
        n_stages: int,
        features_per_stage: list[int],
        kernel_sizes: list,
        strides: list,
        n_conv_per_stage: list[int],
        bias: bool = True,
        norm_kwargs: dict | None = None,
        nonlin_kwargs: dict | None = None,
        ndim: int = 3,
    ):
        super().__init__()
        self.output_channels = features_per_stage
        self.strides = [_to_list(s, ndim) for s in strides]

        stages = []
        ch_in = in_channels
        for s in range(n_stages):
            ks = _to_list(kernel_sizes[s], ndim)
            st = _to_list(strides[s], ndim)
            stages.append(
                StackedConvBlocks(
                    n_conv_per_stage[s], ch_in, features_per_stage[s],
                    ks, st, bias=bias, norm_kwargs=norm_kwargs,
                    nonlin_kwargs=nonlin_kwargs,
                )
            )
            ch_in = features_per_stage[s]
        self.stages = stages

    def __call__(self, x) -> list:
        skips = []
        for stage in self.stages:
            x = stage(x)
            skips.append(x)
        return skips


class ResidualEncoder(nn.Module):
    """Encoder using residual blocks (ResidualEncoderUNet)."""

    def __init__(
        self,
        in_channels: int,
        n_stages: int,
        features_per_stage: list[int],
        kernel_sizes: list,
        strides: list,
        n_blocks_per_stage: list[int],
        bias: bool = False,
        norm_kwargs: dict | None = None,
        nonlin_kwargs: dict | None = None,
        stem_channels: int | None = None,
        ndim: int = 3,
    ):
        super().__init__()
        self.output_channels = features_per_stage
        self.strides = [_to_list(s, ndim) for s in strides]

        sc = stem_channels if stem_channels is not None else features_per_stage[0]
        ks0 = _to_list(kernel_sizes[0], ndim)
        self.stem = StackedConvBlocks(
            1, in_channels, sc, ks0, [1] * ndim,
            bias=bias, norm_kwargs=norm_kwargs, nonlin_kwargs=nonlin_kwargs,
        )
        ch_in = sc

        stages = []
        for s in range(n_stages):
            ks = _to_list(kernel_sizes[s], ndim)
            st = _to_list(strides[s], ndim)
            blocks = []
            blocks.append(
                BasicBlockD(
                    ch_in, features_per_stage[s], ks, st,
                    bias=bias, norm_kwargs=norm_kwargs, nonlin_kwargs=nonlin_kwargs,
                )
            )
            for _ in range(1, n_blocks_per_stage[s]):
                blocks.append(
                    BasicBlockD(
                        features_per_stage[s], features_per_stage[s],
                        ks, [1] * ndim, bias=bias,
                        norm_kwargs=norm_kwargs, nonlin_kwargs=nonlin_kwargs,
                    )
                )
            stages.append(blocks)
            ch_in = features_per_stage[s]
        self.stages = stages

    def __call__(self, x) -> list:
        x = self.stem(x)
        skips = []
        for stage_blocks in self.stages:
            for blk in stage_blocks:
                x = blk(x)
            skips.append(x)
        return skips


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

class UNetDecoder(nn.Module):
    """Decoder with ConvTranspose3d upsampling and skip concatenation."""

    def __init__(
        self,
        encoder_output_channels: list[int],
        encoder_strides: list[list[int]],
        num_classes: int,
        n_conv_per_stage: list[int],
        bias: bool = True,
        norm_kwargs: dict | None = None,
        nonlin_kwargs: dict | None = None,
        deep_supervision: bool = False,
        ndim: int = 3,
    ):
        super().__init__()
        self.deep_supervision = deep_supervision
        n_stages_encoder = len(encoder_output_channels)

        transpconvs = []
        stages = []
        seg_layers = []

        for s in range(1, n_stages_encoder):
            in_ch = encoder_output_channels[-s]
            skip_ch = encoder_output_channels[-(s + 1)]
            stride = encoder_strides[-s]
            ks = _to_list(stride, ndim)

            transpconvs.append(
                nn.ConvTranspose3d(
                    in_channels=in_ch,
                    out_channels=skip_ch,
                    kernel_size=tuple(ks),
                    stride=tuple(stride),
                    padding=(0,) * ndim,
                    bias=bias,
                )
            )
            stages.append(
                StackedConvBlocks(
                    n_conv_per_stage[s - 1], skip_ch * 2, skip_ch,
                    _to_list(3, ndim), [1] * ndim, bias=bias,
                    norm_kwargs=norm_kwargs, nonlin_kwargs=nonlin_kwargs,
                )
            )
            seg_layers.append(
                nn.Conv3d(
                    skip_ch, num_classes, kernel_size=(1,) * ndim,
                    stride=(1,) * ndim, padding=(0,) * ndim, bias=True,
                )
            )

        self.transpconvs = transpconvs
        self.stages = stages
        self.seg_layers = seg_layers

    def __call__(self, skips: list):
        lres = skips[-1]
        seg_outputs = []
        for s in range(len(self.stages)):
            x = self.transpconvs[s](lres)
            x = mx.concatenate([x, skips[-(s + 2)]], axis=-1)
            x = self.stages[s](x)
            if self.deep_supervision:
                seg_outputs.append(self.seg_layers[s](x))
            elif s == len(self.stages) - 1:
                seg_outputs.append(self.seg_layers[-1](x))
            lres = x

        seg_outputs = seg_outputs[::-1]
        return seg_outputs if self.deep_supervision else seg_outputs[0]


# ---------------------------------------------------------------------------
# Full UNet architectures
# ---------------------------------------------------------------------------

class PlainConvUNet(nn.Module):
    """Plain convolutional U-Net (nnU-Net default)."""

    def __init__(
        self,
        in_channels: int,
        n_stages: int,
        features_per_stage: list[int],
        kernel_sizes: list,
        strides: list,
        n_conv_per_stage: list[int],
        num_classes: int,
        n_conv_per_stage_decoder: list[int],
        bias: bool = True,
        norm_kwargs: dict | None = None,
        nonlin_kwargs: dict | None = None,
        deep_supervision: bool = False,
        ndim: int = 3,
    ):
        super().__init__()
        self.encoder = PlainConvEncoder(
            in_channels, n_stages, features_per_stage,
            kernel_sizes, strides, n_conv_per_stage,
            bias=bias, norm_kwargs=norm_kwargs,
            nonlin_kwargs=nonlin_kwargs, ndim=ndim,
        )
        self.decoder = UNetDecoder(
            self.encoder.output_channels,
            self.encoder.strides,
            num_classes, n_conv_per_stage_decoder,
            bias=bias, norm_kwargs=norm_kwargs,
            nonlin_kwargs=nonlin_kwargs,
            deep_supervision=deep_supervision, ndim=ndim,
        )


    def __call__(self, x):
        skips = self.encoder(x)
        return self.decoder(skips)


class ResidualEncoderUNet(nn.Module):
    """Residual encoder U-Net (nnU-Net ResEnc presets)."""

    def __init__(
        self,
        in_channels: int,
        n_stages: int,
        features_per_stage: list[int],
        kernel_sizes: list,
        strides: list,
        n_blocks_per_stage: list[int],
        num_classes: int,
        n_conv_per_stage_decoder: list[int],
        bias: bool = False,
        norm_kwargs: dict | None = None,
        nonlin_kwargs: dict | None = None,
        deep_supervision: bool = False,
        stem_channels: int | None = None,
        ndim: int = 3,
    ):
        super().__init__()
        self.encoder = ResidualEncoder(
            in_channels, n_stages, features_per_stage,
            kernel_sizes, strides, n_blocks_per_stage,
            bias=bias, norm_kwargs=norm_kwargs,
            nonlin_kwargs=nonlin_kwargs,
            stem_channels=stem_channels, ndim=ndim,
        )
        self.decoder = UNetDecoder(
            self.encoder.output_channels,
            self.encoder.strides,
            num_classes, n_conv_per_stage_decoder,
            bias=bias, norm_kwargs=norm_kwargs,
            nonlin_kwargs=nonlin_kwargs,
            deep_supervision=deep_supervision, ndim=ndim,
        )


    def __call__(self, x):
        skips = self.encoder(x)
        return self.decoder(skips)
