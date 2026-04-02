"""
End-to-end test: synthetic CT through the full MLX pipeline.

Uses the real Dataset297 (fast 3mm) weights.
Run: python tests/test_end_to_end.py
"""

import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np

from nnunet_mlx import nnUNetv2_predict_mlx


def main():
    # Create a synthetic CT volume: 120x120x130 at 3mm spacing
    # Roughly matches the median volume size for this model
    shape = (120, 120, 130)
    # CT-like values: air=-1000, soft tissue=40, bone=400
    data = np.random.uniform(-200, 200, shape).astype(np.float32)
    # Add some structure: sphere of "bone" in the center
    center = np.array(shape) // 2
    coords = np.mgrid[:shape[0], :shape[1], :shape[2]]
    dist = np.sqrt(sum((c - cn) ** 2 for c, cn in zip(coords, center)))
    data[dist < 20] = 400   # bone-like
    data[dist > 50] = -800  # air-like

    affine = np.diag([3.0, 3.0, 3.0, 1.0])  # 3mm isotropic
    img = nib.Nifti1Image(data, affine)

    with tempfile.TemporaryDirectory(prefix="mlx_test_") as tmp:
        dir_in = Path(tmp) / "input"
        dir_out = Path(tmp) / "output"
        dir_in.mkdir()

        # Save as nnU-Net expects: {name}_0000.nii.gz
        nib.save(img, str(dir_in / "s01_0000.nii.gz"))

        print("Running MLX inference pipeline...")
        nnUNetv2_predict_mlx(
            dir_in=str(dir_in),
            dir_out=str(dir_out),
            task_id=297,
            trainer="nnUNetTrainer_4000epochs_NoMirroring",
            step_size=0.5,
            quiet=False,
            verbose=True,
        )

        # Check output
        out_path = dir_out / "s01.nii.gz"
        assert out_path.exists(), f"Output not found at {out_path}"

        seg_img = nib.load(str(out_path))
        seg = seg_img.get_fdata()

        print(f"\nInput:  {data.shape}, range [{data.min():.0f}, {data.max():.0f}]")
        print(f"Output: {seg.shape}, {seg.dtype}")
        print(f"Labels: {np.unique(seg).astype(int).tolist()}")
        print(f"Non-background: {(seg > 0).sum()} / {seg.size} voxels "
              f"({100 * (seg > 0).mean():.1f}%)")
        print("\nEnd-to-end test PASSED")


if __name__ == "__main__":
    main()
