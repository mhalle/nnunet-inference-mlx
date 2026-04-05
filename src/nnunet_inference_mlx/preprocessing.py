"""
nnU-Net preprocessing for MLX inference.

Implements the preprocessing steps that nnUNet applies internally
before feeding data to the network. For TotalSegmentator, this is
CTNormalization: clip to percentile range, then z-score normalize.
"""

from __future__ import annotations

import numpy as np


def ct_normalization(
    data: np.ndarray,
    mean: float,
    std: float,
    lower_clip: float,
    upper_clip: float,
) -> np.ndarray:
    """Apply CT normalization as done by nnU-Net's CTNormalization.

    1. Clip intensities to [lower_clip, upper_clip]
    2. Z-score normalize using dataset-level mean and std

    Parameters
    ----------
    data : np.ndarray
        Raw CT volume, any shape.
    mean : float
        Dataset foreground mean intensity.
    std : float
        Dataset foreground std intensity.
    lower_clip : float
        Lower clipping bound (typically percentile_00_5).
    upper_clip : float
        Upper clipping bound (typically percentile_99_5).

    Returns
    -------
    np.ndarray
        Normalized volume, float32.
    """
    data = np.clip(data, lower_clip, upper_clip).astype(np.float32)
    data = (data - mean) / max(std, 1e-8)
    return data


def zscore_normalization(
    data: np.ndarray,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Apply z-score normalization as done by nnU-Net's ZScoreNormalization.

    Computes mean and std from the image foreground (non-zero voxels),
    not from dataset-level statistics.

    Parameters
    ----------
    data : np.ndarray
        Single-channel volume, any shape.
    mask : np.ndarray, optional
        Boolean mask of foreground voxels. If None, uses non-zero voxels.

    Returns
    -------
    np.ndarray
        Normalized volume, float32.
    """
    data = data.astype(np.float32)
    if mask is None:
        mask = data != 0
    if mask.any():
        mean = data[mask].mean()
        std = data[mask].std()
    else:
        mean = 0.0
        std = 1.0
    return (data - mean) / max(std, 1e-8)


def get_normalization_params(plans: dict, channel: int = 0) -> dict:
    """Extract CT normalization parameters from nnU-Net plans.json.

    Returns dict with keys: mean, std, lower_clip, upper_clip.
    """
    props = plans["foreground_intensity_properties_per_channel"][str(channel)]
    return {
        "mean": props["mean"],
        "std": props["std"],
        "lower_clip": props["percentile_00_5"],
        "upper_clip": props["percentile_99_5"],
    }


def preprocess_volume(
    data: np.ndarray,
    plans: dict,
    configuration: str = "3d_fullres",
) -> np.ndarray:
    """Full preprocessing pipeline for a single volume.

    Applies the normalization scheme specified in the plans.
    Currently supports CTNormalization only.

    Parameters
    ----------
    data : np.ndarray
        Raw volume, shape (D, H, W) or (C, D, H, W).
    plans : dict
        Parsed plans.json.
    configuration : str
        Which configuration to use from plans.

    Returns
    -------
    np.ndarray
        Preprocessed volume, shape (C, D, H, W), float32.
    """
    config = plans["configurations"][configuration]
    norm_schemes = config.get("normalization_schemes", ["CTNormalization"])

    if data.ndim == 3:
        data = data[None]  # add channel dim

    result = np.zeros_like(data, dtype=np.float32)
    for ch in range(data.shape[0]):
        scheme = norm_schemes[ch] if ch < len(norm_schemes) else norm_schemes[0]
        if scheme == "CTNormalization":
            params = get_normalization_params(plans, ch)
            result[ch] = ct_normalization(data[ch], **params)
        elif scheme == "ZScoreNormalization":
            # nnU-Net computes z-score from the image itself, not dataset stats
            result[ch] = zscore_normalization(data[ch])
        elif scheme == "NoNormalization":
            result[ch] = data[ch].astype(np.float32)
        else:
            # Fall back to CT normalization
            params = get_normalization_params(plans, ch)
            result[ch] = ct_normalization(data[ch], **params)

    return result
