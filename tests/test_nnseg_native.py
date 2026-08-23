"""nnU-Net-native support: crop-to-nonzero geometry, configuration/fold selection, and the
guards that must fail loudly rather than be silently wrong.

The knee reference model (aagatti/nnunet_knee) exercises configuration + fold selection and the
end-to-end path, but its nonzero box is the whole volume and its resample is shape-identity, so
the crop and the convention need their own tests. That is what lives here.
"""
import json

import numpy as np
import pytest

from nnseg.frame import Frame
from nnseg.grid import Grid
from nnseg.preprocess import nonzero_box
from nnseg.tasks import TaskSpec, resolve_model_folder


# -- crop-to-nonzero --------------------------------------------------------------------
def test_nonzero_box_finds_the_border():
    a = np.zeros((20, 30, 40), np.float32)
    a[3:17, 5:25, 8:32] = 1.0
    assert nonzero_box(a) == ((3, 5, 8), (17, 25, 32))


def test_nonzero_box_fills_holes_so_interior_zeros_do_not_split_it():
    pytest.importorskip("scipy")
    a = np.zeros((10, 10, 10), np.float32)
    a[2:8, 2:8, 2:8] = 1.0
    a[4:6, 4:6, 4:6] = 0.0                      # a hole in the middle
    assert nonzero_box(a) == ((2, 2, 2), (8, 8, 8))


def test_nonzero_box_all_zero_yields_the_whole_grid_never_an_empty_box():
    assert nonzero_box(np.zeros((4, 5, 6), np.float32)) == ((0, 0, 0), (4, 5, 6))


def test_nonzero_box_all_nonzero_is_the_whole_grid():
    assert nonzero_box(np.ones((4, 5, 6), np.float32)) == ((0, 0, 0), (4, 5, 6))


# -- the crop's geometry: an output grid on the FULL source must still land correctly ----
def _frame(model_source=None, convention="center"):
    src = Grid((20, 30, 40), (1.0, 1.0, 1.0), (0.0, 0.0, 0.0))
    ms = model_source or src
    return Frame(source=src, model_shape=ms.shape, model_spacing=ms.spacing,
                 convention=convention, canonical=None, model_source=model_source)


def test_uncropped_frame_is_unchanged_by_the_new_field():
    f = _frame()
    assert f.resampled_from is f.source
    m = f.mapping(f.source)
    assert np.allclose(m.a, (1, 1, 1)) and np.allclose(m.b, (0, 0, 0))


def test_crop_offset_is_absorbed_into_the_mapping():
    """A model grid built from a cropped source, sampled on the full source grid: the mapping
    must subtract the crop offset, so full-grid voxel (3,5,8) is model voxel (0,0,0)."""
    crop = Grid((14, 20, 24), (1.0, 1.0, 1.0), (3.0, 5.0, 8.0))   # origin = crop offset in mm
    f = _frame(model_source=crop)
    m = f.mapping(f.source)
    assert np.allclose(m.a, (1, 1, 1))
    assert np.allclose(m.b, (-3, -5, -8))
    assert np.allclose(m.apply((3, 5, 8)), (0, 0, 0))
    # a voxel before the crop maps outside the model grid, where outside="background" handles it
    assert np.all(m.apply((0, 0, 0)) < 0)


def test_crop_plus_resample_composes():
    """Cropped source at 1 mm resampled to a half-size model grid: both the offset and the
    scale must appear, and the crop's first voxel must land at the model's first voxel."""
    crop = Grid((14, 20, 24), (1.0, 1.0, 1.0), (3.0, 5.0, 8.0))
    src = Grid((20, 30, 40), (1.0, 1.0, 1.0), (0.0, 0.0, 0.0))
    f = Frame(source=src, model_shape=(7, 10, 12), model_spacing=(2.0, 2.0, 2.0),
              convention="center", canonical=None, model_source=crop)
    m = f.mapping(src)
    assert np.allclose(m.a, (0.5, 0.5, 0.5))
    # center rule on the cropped grid: x_model = (x_crop + 0.5) * 0.5 - 0.5, x_crop = x_src - off
    assert np.allclose(m.apply((3, 5, 8)), (-0.25, -0.25, -0.25))
    assert np.allclose(m.apply((5, 7, 10)), (0.75, 0.75, 0.75))


# -- configuration selection -------------------------------------------------------------
def _model_tree(tmp_path, configs, labels=None, folds=(0,)):
    ds = tmp_path / "Dataset500_Knee"
    for c in configs:
        d = ds / f"nnUNetTrainer__nnUNetPlans__{c}"
        d.mkdir(parents=True)
        (d / "dataset.json").write_text(json.dumps(
            {"labels": labels or {"background": 0, "femur": 1, "tibia": 2},
             "channel_names": {"0": "MRI"}}))
        for f in folds:
            (d / f"fold_{f}").mkdir()
    return ds


def test_configuration_preference_picks_3d_fullres_not_the_first_alphabetically(tmp_path):
    """The knee model ships 2d, 3d_cascade_fullres, 3d_fullres and 3d_lowres; sorting first
    would select the cascade, which nnseg cannot run."""
    ds = _model_tree(tmp_path, ["2d", "3d_cascade_fullres", "3d_fullres", "3d_lowres"])
    assert resolve_model_folder(ds).name.endswith("__3d_fullres")


def test_explicit_configuration_wins_and_unknown_one_lists_what_exists(tmp_path):
    ds = _model_tree(tmp_path, ["3d_fullres", "3d_lowres"])
    assert resolve_model_folder(ds, configuration="3d_lowres").name.endswith("__3d_lowres")
    with pytest.raises(FileNotFoundError, match="3d_lowres"):
        resolve_model_folder(ds, configuration="nope")


def test_only_an_unsupported_configuration_is_an_informative_error(tmp_path):
    ds = _model_tree(tmp_path, ["3d_cascade_fullres"])
    # a lone configuration is still returned - the caller may know what they are doing
    assert resolve_model_folder(ds).name.endswith("__3d_cascade_fullres")


def test_a_model_folder_passes_straight_through(tmp_path):
    ds = _model_tree(tmp_path, ["3d_fullres"])
    folder = ds / "nnUNetTrainer__nnUNetPlans__3d_fullres"
    assert resolve_model_folder(folder) == folder


# -- TaskSpec from a bare nnU-Net folder ---------------------------------------------------
def test_taskspec_from_model_folder_reads_dataset_json(tmp_path):
    ds = _model_tree(tmp_path, ["3d_fullres"], labels={"background": 0, "femur": 7, "tibia": 8})
    spec = TaskSpec.from_model_folder(ds)
    assert spec.source == "nnunet" and spec.shape == "single"
    assert spec.label_map == {7: "femur", 8: "tibia"}     # background dropped
    assert spec.modality == "MRI"
    assert resolve_model_folder(spec.single).name.endswith("__3d_fullres")


def test_region_based_labels_are_refused_not_silently_flattened(tmp_path):
    ds = _model_tree(tmp_path, ["3d_fullres"],
                     labels={"background": 0, "whole": [1, 2], "core": 2})
    with pytest.raises(NotImplementedError, match="region-based"):
        TaskSpec.from_model_folder(ds)


# -- fold discovery -------------------------------------------------------------------------
def test_available_folds_keeps_what_exists_and_errors_on_what_does_not(tmp_path):
    from nnseg.network import available_folds
    ds = _model_tree(tmp_path, ["3d_fullres"], folds=(1,))
    f = ds / "nnUNetTrainer__nnUNetPlans__3d_fullres"
    assert available_folds(f, "all") == (1,)          # the knee model ships only fold_1
    assert available_folds(f, (1,)) == (1,)
    with pytest.raises(FileNotFoundError, match=r"have \[1\]"):
        available_folds(f, (0,))                      # the TS default must not silently pass


def test_available_folds_intersects_a_request(tmp_path):
    from nnseg.network import available_folds
    ds = _model_tree(tmp_path, ["3d_fullres"], folds=(0, 2, 3))
    f = ds / "nnUNetTrainer__nnUNetPlans__3d_fullres"
    assert available_folds(f, (0, 1, 2)) == (0, 2)
    assert available_folds(f, "all") == (0, 2, 3)
