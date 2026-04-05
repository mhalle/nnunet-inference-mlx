"""
Sliding window prediction with Gaussian weighting for 3D volumes.
"""

from __future__ import annotations

import itertools
import time

import mlx.core as mx
import numpy as np


def compute_gaussian(
    tile_size: tuple[int, ...],
    sigma_scale: float = 1.0 / 8,
    value_scaling_factor: float = 1.0,
    dtype=np.float32,
) -> np.ndarray:
    """Compute Gaussian importance map for sliding window aggregation."""
    coords = np.meshgrid(
        *(np.arange(s) for s in tile_size),
        indexing="ij",
    )
    center = [s // 2 for s in tile_size]
    sigmas = [s * sigma_scale for s in tile_size]
    gaussian_map = np.exp(-sum(
        (c - cn) ** 2 / (2 * sg ** 2)
        for c, cn, sg in zip(coords, center, sigmas)
    ))
    gaussian_map = gaussian_map / (gaussian_map.max() / value_scaling_factor)
    mask = gaussian_map == 0
    if mask.any():
        gaussian_map[mask] = gaussian_map[~mask].min()
    return gaussian_map.astype(dtype)


def compute_sliding_window_steps(
    image_size: tuple[int, ...],
    tile_size: tuple[int, ...],
    tile_step_size: float,
) -> list[list[int]]:
    """Compute start positions for each dimension of the sliding window."""
    target_steps = [int(i * tile_step_size) for i in tile_size]
    num_steps = [
        int(np.ceil((i - k) / j)) + 1
        for i, j, k in zip(image_size, target_steps, tile_size)
    ]

    steps = []
    for dim in range(len(tile_size)):
        max_step = image_size[dim] - tile_size[dim]
        if num_steps[dim] > 1:
            actual_step = max_step / (num_steps[dim] - 1)
        else:
            actual_step = 99999999
        steps.append(
            [int(np.round(actual_step * i)) for i in range(num_steps[dim])]
        )
    return steps


def predict_sliding_window(
    network,
    input_image: np.ndarray,
    patch_size: tuple[int, ...],
    num_classes: int,
    tile_step_size: float = 0.5,
    use_gaussian: bool = True,
    use_mirroring: bool = False,
    mirror_axes: tuple[int, ...] | None = None,
    batch_size: int = 1,
    use_fp16: bool = True,
    verbose: bool = False,
) -> np.ndarray:
    """Run batched sliding window inference over a 3D volume.

    Parameters
    ----------
    network : callable
        MLX network. Takes (B, D, H, W, C), returns (B, D, H, W, K).
    input_image : np.ndarray
        Shape (C, D, H, W) in float32.
    patch_size : tuple
        Spatial patch size (D, H, W).
    num_classes : int
        Number of output classes.
    tile_step_size : float
        Overlap fraction (0.5 = 50% overlap).
    use_gaussian : bool
        Weight predictions by Gaussian importance map.
    use_mirroring : bool
        Apply test-time augmentation via axis flipping.
    mirror_axes : tuple, optional
        Which spatial axes to mirror (0=D, 1=H, 2=W).
    batch_size : int
        Number of patches to process in parallel.
    use_fp16 : bool
        Run inference in float16 for speed and memory.
    verbose : bool
        Print progress info.

    Returns
    -------
    np.ndarray
        Predicted logits, shape (num_classes, D, H, W), float32.
    """
    spatial_shape = input_image.shape[1:]

    # Pad if smaller than patch — symmetric padding to match nnU-Net
    pad_widths = []
    for s, t in zip(spatial_shape, patch_size):
        total_pad = max(0, t - s)
        pad_before = total_pad // 2
        pad_after = total_pad - pad_before
        pad_widths.append((pad_before, pad_after))
    needs_padding = any(p[0] > 0 or p[1] > 0 for p in pad_widths)
    if needs_padding:
        full_pad = [(0, 0)] + pad_widths
        input_image = np.pad(input_image, full_pad, mode="constant", constant_values=0)
        spatial_shape = input_image.shape[1:]

    # Convert to channels-last: (C, D, H, W) -> (D, H, W, C)
    data = input_image.transpose(1, 2, 3, 0)

    # Sliding window positions
    steps = compute_sliding_window_steps(spatial_shape, patch_size, tile_step_size)
    slicers = [
        (sx, sy, sz) for sx in steps[0] for sy in steps[1] for sz in steps[2]
    ]

    # Gaussian weighting
    gaussian_np = (
        compute_gaussian(patch_size, sigma_scale=1.0 / 8, value_scaling_factor=10)
        if use_gaussian
        else np.ones(patch_size, dtype=np.float32)
    )

    # Accumulators — use fp16 when class count is large to halve memory
    accum_dtype = np.float16 if num_classes > 20 else np.float32
    predicted_logits = np.zeros((num_classes, *spatial_shape), dtype=accum_dtype)
    n_predictions = np.zeros(spatial_shape, dtype=np.float32)

    # TTA: precompute axis combinations
    if use_mirroring and mirror_axes:
        axes_combos = [
            tuple(m + 1 for m in c)
            for k in range(len(mirror_axes))
            for c in itertools.combinations(mirror_axes, k + 1)
        ]
        n_tta = len(axes_combos) + 1
    else:
        axes_combos = []
        n_tta = 1

    total_fwd = int(np.ceil(len(slicers) / batch_size)) * n_tta
    if verbose:
        print(
            f"Sliding window: {len(slicers)} patches, batch_size={batch_size}, "
            f"tta={n_tta}x, total_fwd={total_fwd}, "
            f"image={spatial_shape}, patch={patch_size}, "
            f"dtype={'fp16' if use_fp16 else 'fp32'}"
        )

    for batch_start in range(0, len(slicers), batch_size):
        batch_slicers = slicers[batch_start:batch_start + batch_size]

        patches_np = np.stack([
            data[sx:sx + patch_size[0], sy:sy + patch_size[1], sz:sz + patch_size[2], :]
            for sx, sy, sz in batch_slicers
        ])

        patches = mx.array(patches_np)
        if use_fp16:
            patches = patches.astype(mx.float16)

        pred = network(patches)
        if isinstance(pred, list):
            pred = pred[0]

        # TTA mirroring
        if axes_combos:
            pred_sum = pred.astype(mx.float32)
            for axes in axes_combos:
                flipped_in = mx.array(
                    np.flip(patches_np, axis=list(axes)).copy()
                )
                if use_fp16:
                    flipped_in = flipped_in.astype(mx.float16)
                fp = network(flipped_in)
                if isinstance(fp, list):
                    fp = fp[0]
                fp = mx.array(
                    np.flip(np.array(fp.astype(mx.float32)), axis=list(axes)).copy()
                )
                pred_sum = pred_sum + fp
            pred = pred_sum * (1.0 / n_tta)

        mx.eval(pred)
        pred_np = np.array(pred.astype(mx.float32))

        for i, (sx, sy, sz) in enumerate(batch_slicers):
            p = pred_np[i].transpose(3, 0, 1, 2)
            if use_gaussian:
                p *= gaussian_np[None]
            predicted_logits[
                :,
                sx:sx + patch_size[0],
                sy:sy + patch_size[1],
                sz:sz + patch_size[2],
            ] += p
            n_predictions[
                sx:sx + patch_size[0],
                sy:sy + patch_size[1],
                sz:sz + patch_size[2],
            ] += gaussian_np

    # Normalize — cast to fp32 for the division to avoid fp16 precision issues
    predicted_logits = predicted_logits.astype(np.float32) / n_predictions[None]

    if needs_padding:
        crop = tuple(
            slice(pb, s - pa) if (pb > 0 or pa > 0) else slice(None)
            for s, (pb, pa) in zip(predicted_logits.shape[1:], pad_widths)
        )
        predicted_logits = predicted_logits[(slice(None), *crop)]

    return predicted_logits


def predict_sliding_window_streaming(
    network,
    input_image: np.ndarray,
    patch_size: tuple[int, ...],
    num_classes: int,
    tile_step_size: float = 0.5,
    use_gaussian: bool = True,
    use_mirroring: bool = False,
    mirror_axes: tuple[int, ...] | None = None,
    batch_size: int = 1,
    use_fp16: bool = True,
    verbose: bool = False,
) -> np.ndarray:
    """Sliding window inference with a rolling Z buffer.

    Same interface and output as predict_sliding_window, but uses a
    rolling accumulator along the Z (first spatial) axis. This reduces
    peak memory from O(K * Z * Y * X) to O(K * active_z * Y * X),
    freeing headroom for larger batch sizes.

    Parameters
    ----------
    Same as predict_sliding_window.

    Returns
    -------
    np.ndarray
        Predicted logits, shape (num_classes, D, H, W), float32.
    """
    spatial_shape = input_image.shape[1:]

    # Pad if smaller than patch — symmetric padding to match nnU-Net
    pad_widths = []
    for s, t in zip(spatial_shape, patch_size):
        total_pad = max(0, t - s)
        pad_before = total_pad // 2
        pad_after = total_pad - pad_before
        pad_widths.append((pad_before, pad_after))
    needs_padding = any(p[0] > 0 or p[1] > 0 for p in pad_widths)
    if needs_padding:
        full_pad = [(0, 0)] + pad_widths
        input_image = np.pad(input_image, full_pad, mode="constant", constant_values=0)
        spatial_shape = input_image.shape[1:]

    Z, Y, X = spatial_shape
    pZ, pY, pX = patch_size

    # Convert to channels-last: (C, Z, Y, X) -> (Z, Y, X, C)
    data = input_image.transpose(1, 2, 3, 0)

    # Sliding window positions per axis
    steps = compute_sliding_window_steps(spatial_shape, patch_size, tile_step_size)
    z_steps = steps[0]
    yx_slicers = [
        (sy, sx) for sy in steps[1] for sx in steps[2]
    ]

    # Gaussian weighting
    gaussian_np = (
        compute_gaussian(patch_size, sigma_scale=1.0 / 8, value_scaling_factor=10)
        if use_gaussian
        else np.ones(patch_size, dtype=np.float32)
    )

    # TTA: precompute axis combinations
    if use_mirroring and mirror_axes:
        axes_combos = [
            tuple(m + 1 for m in c)
            for k in range(len(mirror_axes))
            for c in itertools.combinations(mirror_axes, k + 1)
        ]
        n_tta = len(axes_combos) + 1
    else:
        axes_combos = []
        n_tta = 1

    # Determine the rolling buffer size: covers from the current z_step
    # to the end of its patch (current_z + pZ). The buffer must hold all
    # Z slices that any active patch can write to.
    # Active range = [z_steps[i], z_steps[i] + pZ) at most.
    # But patches from the *previous* z_step may still overlap.
    # The maximum active range is [z_steps[i], z_steps[i+1] + pZ) while
    # processing z_steps[i+1]. Simplification: buffer = pZ + max_z_gap.
    if len(z_steps) > 1:
        max_z_gap = max(z_steps[j] - z_steps[j - 1] for j in range(1, len(z_steps)))
    else:
        max_z_gap = 0
    buf_z = pZ + max_z_gap
    # Clamp to actual volume size (small volumes don't need the full buffer)
    buf_z = min(buf_z, Z)

    # Rolling buffer and weight accumulator — channels-first (K, Z, Y, X)
    accum_dtype = np.float16 if num_classes > 20 else np.float32
    buf_logits = np.zeros((num_classes, buf_z, Y, X), dtype=accum_dtype)
    buf_weights = np.zeros((buf_z, Y, X), dtype=np.float32)
    buf_z_start = 0  # which global Z index does buf[0] correspond to

    # Output — we fill this in as slices finalize (unnormalized)
    predicted_logits = np.zeros((num_classes, Z, Y, X), dtype=accum_dtype)
    out_weights = np.zeros((Z, Y, X), dtype=np.float32)

    total_patches = len(z_steps) * len(yx_slicers)
    total_fwd = int(np.ceil(total_patches / batch_size)) * n_tta
    if verbose:
        buf_mb = (buf_logits.nbytes + buf_weights.nbytes) / 1e6
        full_mb = num_classes * Z * Y * X * 4 / 1e6
        print(
            f"Sliding window (streaming): {total_patches} patches, "
            f"batch_size={batch_size}, tta={n_tta}x, total_fwd={total_fwd}, "
            f"image={spatial_shape}, patch={patch_size}, "
            f"buf_z={buf_z} (buffer={buf_mb:.0f}MB vs full={full_mb:.0f}MB), "
            f"dtype={'fp16' if use_fp16 else 'fp32'}"
        )

    _t0 = time.perf_counter()
    _patches_done = 0

    for zi, sz in enumerate(z_steps):
        # Before processing this z_step, flush any finalized slices
        # Slices [buf_z_start, sz) are no longer touched by any future patch
        flush_end = sz
        if flush_end > buf_z_start:
            n_flush = flush_end - buf_z_start
            # Copy raw (unnormalized) accumulated values to output
            buf_local = slice(0, n_flush)
            predicted_logits[:, buf_z_start:flush_end] = buf_logits[:, buf_local]
            out_weights[buf_z_start:flush_end] = buf_weights[buf_local]

            # Shift buffer: move remaining data to front
            remaining = buf_z - n_flush
            if remaining > 0:
                buf_logits[:, :remaining] = buf_logits[:, n_flush:n_flush + remaining]
                buf_weights[:remaining] = buf_weights[n_flush:n_flush + remaining]
            # Zero out the freed tail
            buf_logits[:, remaining:] = 0
            buf_weights[remaining:] = 0
            buf_z_start = flush_end

        # Process all YX patches at this Z step
        all_slicers = [(sz, sy, sx) for sy, sx in yx_slicers]

        for batch_start in range(0, len(all_slicers), batch_size):
            batch_slicers = all_slicers[batch_start:batch_start + batch_size]

            patches_np = np.stack([
                data[s0:s0 + pZ, s1:s1 + pY, s2:s2 + pX, :]
                for s0, s1, s2 in batch_slicers
            ])

            patches = mx.array(patches_np)
            if use_fp16:
                patches = patches.astype(mx.float16)

            pred = network(patches)
            if isinstance(pred, list):
                pred = pred[0]

            # TTA mirroring
            if axes_combos:
                pred_sum = pred.astype(mx.float32)
                for axes in axes_combos:
                    flipped_in = mx.array(
                        np.flip(patches_np, axis=list(axes)).copy()
                    )
                    if use_fp16:
                        flipped_in = flipped_in.astype(mx.float16)
                    fp = network(flipped_in)
                    if isinstance(fp, list):
                        fp = fp[0]
                    fp = mx.array(
                        np.flip(np.array(fp.astype(mx.float32)), axis=list(axes)).copy()
                    )
                    pred_sum = pred_sum + fp
                pred = pred_sum * (1.0 / n_tta)

            mx.eval(pred)
            pred_np = np.array(pred.astype(mx.float32))

            for i, (s0, s1, s2) in enumerate(batch_slicers):
                p = pred_np[i].transpose(3, 0, 1, 2)  # (K, pZ, pY, pX)
                if use_gaussian:
                    p *= gaussian_np[None]

                # Map global Z to buffer-local Z
                bz = s0 - buf_z_start
                buf_logits[:, bz:bz + pZ, s1:s1 + pY, s2:s2 + pX] += p
                buf_weights[bz:bz + pZ, s1:s1 + pY, s2:s2 + pX] += gaussian_np

            _patches_done += len(batch_slicers)
            if verbose:
                elapsed = time.perf_counter() - _t0
                eta = elapsed / _patches_done * (total_patches - _patches_done)
                print(f"\r  patch {_patches_done}/{total_patches} "
                      f"z_step {zi+1}/{len(z_steps)} "
                      f"({elapsed:.1f}s, ~{eta:.0f}s left)", end="", flush=True)

    if verbose:
        print()

    # Flush everything remaining in the buffer
    remaining = Z - buf_z_start
    if remaining > 0:
        buf_local = slice(0, remaining)
        predicted_logits[:, buf_z_start:] = buf_logits[:, buf_local]
        out_weights[buf_z_start:] = buf_weights[buf_local]

    # Normalize — match full version: cast to fp32 then divide
    predicted_logits = predicted_logits.astype(np.float32) / out_weights[None]

    if needs_padding:
        crop = tuple(
            slice(pb, s - pa) if (pb > 0 or pa > 0) else slice(None)
            for s, (pb, pa) in zip(predicted_logits.shape[1:], pad_widths)
        )
        predicted_logits = predicted_logits[(slice(None), *crop)]

    return predicted_logits


def predict_sliding_window_segmentation(
    network,
    input_image: np.ndarray,
    patch_size: tuple[int, ...],
    num_classes: int,
    tile_step_size: float = 0.5,
    use_gaussian: bool = True,
    use_mirroring: bool = False,
    mirror_axes: tuple[int, ...] | None = None,
    batch_size: int = 1,
    use_fp16: bool = True,
    verbose: bool = False,
) -> np.ndarray:
    """Sliding window inference that returns segmentation directly.

    Instead of accumulating all class logits (num_classes * volume = huge),
    tracks only the weighted sum per class at each voxel via a running
    argmax. Memory usage: O(volume) instead of O(num_classes * volume).

    Returns
    -------
    np.ndarray
        Segmentation labels, shape (D, H, W), dtype uint8.
    """
    spatial_shape = input_image.shape[1:]

    # Pad if smaller than patch — symmetric padding to match nnU-Net
    pad_widths = []
    for s, t in zip(spatial_shape, patch_size):
        total_pad = max(0, t - s)
        pad_before = total_pad // 2
        pad_after = total_pad - pad_before
        pad_widths.append((pad_before, pad_after))
    needs_padding = any(p[0] > 0 or p[1] > 0 for p in pad_widths)
    if needs_padding:
        full_pad = [(0, 0)] + pad_widths
        input_image = np.pad(input_image, full_pad, mode="constant", constant_values=0)
        spatial_shape = input_image.shape[1:]

    data = input_image.transpose(1, 2, 3, 0)

    steps = compute_sliding_window_steps(spatial_shape, patch_size, tile_step_size)
    slicers = [
        (sx, sy, sz) for sx in steps[0] for sy in steps[1] for sz in steps[2]
    ]

    gaussian_np = (
        compute_gaussian(patch_size, sigma_scale=1.0 / 8, value_scaling_factor=10)
        if use_gaussian
        else np.ones(patch_size, dtype=np.float32)
    )

    # Lightweight accumulators: top-2 class scores and labels per voxel
    top_k = 2
    top_scores = np.full((*spatial_shape, top_k), -np.inf, dtype=np.float32)
    top_labels = np.zeros((*spatial_shape, top_k), dtype=np.uint8)
    n_predictions = np.zeros(spatial_shape, dtype=np.float32)

    if use_mirroring and mirror_axes:
        axes_combos = [
            tuple(m + 1 for m in c)
            for k in range(len(mirror_axes))
            for c in itertools.combinations(mirror_axes, k + 1)
        ]
        n_tta = len(axes_combos) + 1
    else:
        axes_combos = []
        n_tta = 1

    total_batches = int(np.ceil(len(slicers) / batch_size))
    if verbose:
        total_fwd = total_batches * n_tta
        accum_mb = (top_scores.nbytes + top_labels.nbytes + n_predictions.nbytes) / 1e6
        print(
            f"Sliding window (segmentation mode): {len(slicers)} patches, "
            f"batch_size={batch_size}, tta={n_tta}x, total_fwd={total_fwd}, "
            f"accum={accum_mb:.0f}MB"
        )

    _t0 = time.perf_counter()
    for batch_idx, batch_start in enumerate(range(0, len(slicers), batch_size)):
        if verbose and batch_idx > 0:
            elapsed = time.perf_counter() - _t0
            eta = elapsed / batch_idx * (total_batches - batch_idx)
            print(f"\r  {batch_idx}/{total_batches} "
                  f"({elapsed:.0f}s, ~{eta:.0f}s left)", end="", flush=True)
        batch_slicers = slicers[batch_start:batch_start + batch_size]

        patches_np = np.stack([
            data[sx:sx + patch_size[0], sy:sy + patch_size[1], sz:sz + patch_size[2], :]
            for sx, sy, sz in batch_slicers
        ])

        patches = mx.array(patches_np)
        if use_fp16:
            patches = patches.astype(mx.float16)

        pred = network(patches)
        if isinstance(pred, list):
            pred = pred[0]

        if axes_combos:
            pred_sum = pred.astype(mx.float32)
            for axes in axes_combos:
                flipped_in = mx.array(
                    np.flip(patches_np, axis=list(axes)).copy()
                )
                if use_fp16:
                    flipped_in = flipped_in.astype(mx.float16)
                fp = network(flipped_in)
                if isinstance(fp, list):
                    fp = fp[0]
                fp = mx.array(
                    np.flip(np.array(fp.astype(mx.float32)), axis=list(axes)).copy()
                )
                pred_sum = pred_sum + fp
            pred = pred_sum * (1.0 / n_tta)

        mx.eval(pred)
        pred_np = np.array(pred.astype(mx.float32))
        # pred_np: (B, pD, pH, pW, num_classes)

        for i, (sx, sy, sz) in enumerate(batch_slicers):
            patch_pred = pred_np[i]  # (pD, pH, pW, K)
            if use_gaussian:
                patch_pred = patch_pred * gaussian_np[:, :, :, None]

            # Per-voxel top-2 from this patch
            best_idx = patch_pred.argmax(axis=-1)
            best_score = patch_pred.max(axis=-1)
            # Mask out the best to find second best
            patch_masked = patch_pred.copy()
            np.put_along_axis(patch_masked, best_idx[..., None], -np.inf, axis=-1)
            second_idx = patch_masked.argmax(axis=-1)
            second_score = patch_masked.max(axis=-1)

            top3_idx = np.stack([best_idx, second_idx], axis=-1).astype(np.uint8)
            top3_scores = np.stack([best_score, second_score], axis=-1)

            sd = slice(sx, sx + patch_size[0])
            sh = slice(sy, sy + patch_size[1])
            sw = slice(sz, sz + patch_size[2])

            # Update where this patch's best score beats the current best
            region_best = top_scores[sd, sh, sw, 0]
            patch_best = top3_scores[..., 0]
            better = patch_best > region_best

            # Where this patch wins, replace all top-3
            for k in range(top_k):
                top_labels[sd, sh, sw, k] = np.where(
                    better, top3_idx[..., k], top_labels[sd, sh, sw, k])
                top_scores[sd, sh, sw, k] = np.where(
                    better, top3_scores[..., k], top_scores[sd, sh, sw, k])
            n_predictions[sd, sh, sw] += gaussian_np

    if verbose:
        print()

    if needs_padding:
        crop = tuple(
            slice(pb, s - pa) if (pb > 0 or pa > 0) else slice(None)
            for s, (pb, pa) in zip(top_labels.shape[:3], pad_widths)
        )
        top_labels = top_labels[(*crop, slice(None))]
        top_scores = top_scores[(*crop, slice(None))]

    return top_labels, top_scores


def _estimate_activation_bytes(
    patch_size: tuple[int, ...],
    features: list[int] = (32, 64, 128, 256, 320, 320),
    bytes_per_element: int = 2,
) -> int:
    """Estimate peak activation memory for one patch through a UNet."""
    total = 0
    for i, f in enumerate(features):
        spatial = [p // (2 ** i) if i > 0 else p for p in patch_size]
        total += f * int(np.prod(spatial)) * bytes_per_element * 2
    total *= 2
    return total


def choose_batch_size(
    patch_size: tuple[int, ...],
    num_classes: int = 105,
    dtype_bytes: int = 4,
) -> int:
    """Choose batch size that fits within Metal buffer limits.

    With mx.compile, MLX fuses operations and reuses memory for
    intermediates. Empirically, peak memory per patch is ~3x the naive
    activation estimate (calibrated on M2 17GB: batch=2 works for 25
    classes at 128^3, batch=3 OOMs).
    """
    max_buffer_bytes = _get_metal_max_buffer_bytes()
    act_bytes = _estimate_activation_bytes(patch_size, bytes_per_element=dtype_bytes)
    # With mx.compile, real peak is ~3x naive estimate
    real_peak_bytes = act_bytes * 3
    output_bytes = int(np.prod(patch_size)) * num_classes * dtype_bytes
    per_patch_bytes = real_peak_bytes + output_bytes
    # Stay at 85% of Metal max buffer (mx.compile enables memory reuse)
    usable_bytes = max_buffer_bytes * 0.85
    batch = max(1, int(usable_bytes / per_patch_bytes))
    return min(batch, 8)


def _get_metal_max_buffer_bytes() -> int:
    """Get the Metal max buffer allocation size."""
    try:
        info = mx.device_info()
        return info["max_buffer_length"]
    except Exception:
        # Fallback: assume 8GB
        return 8 * 1024**3


def get_system_memory_gb() -> float:
    """Detect total system memory in GB. Works on macOS and Linux."""
    try:
        import platform

        if platform.system() == "Darwin":
            import subprocess

            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"], capture_output=True, text=True
            )
            return int(result.stdout.strip()) / 1e9
        else:
            import os

            mem = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
            return mem / 1e9
    except Exception:
        return 16.0
