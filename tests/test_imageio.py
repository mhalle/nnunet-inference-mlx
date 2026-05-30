"""Tests for image IO + the Geometry↔SITK bridge (Phase 3)."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from nnunet_inference_mlx.imageio import (
    ArrayReader,
    NiftiReader,
    NiftiWriter,
    geometry_from_sitk,
    sitk_to_volume,
    volume_to_sitk,
)
from nnunet_inference_mlx.values import Geometry, LabelSchema, Segmentation, Volume

sitk = pytest.importorskip("SimpleITK")


def _geom(shape=(8, 10, 12), spacing=(1.0, 0.8, 0.8)):
    return Geometry(spacing_zyx=spacing, shape_zyx=shape,
                    origin_xyz=(5.0, 10.0, 15.0))


class TestGeometryBridge:
    def test_round_trip_geometry(self):
        g = _geom()
        vol = Volume(data=mx.zeros((*g.shape_zyx, 1)), geometry=g)
        img = volume_to_sitk(vol)
        g2 = geometry_from_sitk(img)
        assert g2.shape_zyx == g.shape_zyx
        assert all(abs(a - b) < 1e-6 for a, b in zip(g2.spacing_zyx, g.spacing_zyx))
        assert all(abs(a - b) < 1e-6 for a, b in zip(g2.origin_xyz, g.origin_xyz))

    def test_sitk_size_order(self):
        # SITK GetSize is (X, Y, Z); our shape is (Z, Y, X)
        g = _geom(shape=(8, 10, 12))
        img = volume_to_sitk(Volume(data=mx.zeros((8, 10, 12, 1)), geometry=g))
        assert img.GetSize() == (12, 10, 8)


class TestVolumeRoundTrip:
    def test_volume_to_sitk_to_volume(self):
        g = _geom()
        data = mx.random.normal((*g.shape_zyx, 1))
        vol = Volume(data=data, geometry=g)
        back = sitk_to_volume(volume_to_sitk(vol))
        assert back.geometry.shape_zyx == g.shape_zyx
        assert np.allclose(np.asarray(back.data), np.asarray(data), atol=1e-5)

    def test_multichannel_rejected(self):
        g = _geom()
        vol = Volume(data=mx.zeros((*g.shape_zyx, 2)), geometry=g,
                     channels=("PET", "CT"))
        with pytest.raises(NotImplementedError, match="single-channel"):
            volume_to_sitk(vol)


class TestReadersWriters:
    def test_array_reader_3d_adds_channel(self):
        g = _geom()
        arr = np.random.randn(*g.shape_zyx).astype(np.float32)
        vol = ArrayReader().read(arr, g)
        assert vol.num_channels == 1
        assert vol.shape_zyx == g.shape_zyx

    def test_array_reader_accepts_mx(self):
        g = _geom()
        vol = ArrayReader().read(mx.zeros((*g.shape_zyx, 1)), g, channels=("MR",))
        assert vol.channels == ("MR",)

    def test_nifti_round_trip(self, tmp_path):
        g = _geom()
        data = mx.random.normal((*g.shape_zyx, 1))
        vol = Volume(data=data, geometry=g)
        path = tmp_path / "img.nii.gz"
        NiftiWriter().write(path, vol)
        back = NiftiReader().read(path)
        assert back.geometry.shape_zyx == g.shape_zyx
        assert np.allclose(np.asarray(back.data), np.asarray(data), atol=1e-4)

    def test_write_segmentation(self, tmp_path):
        g = _geom()
        schema = LabelSchema(names={0: "background", 1: "liver"})
        seg = Segmentation(data=mx.zeros(g.shape_zyx, dtype=mx.uint8),
                           geometry=g, schema=schema)
        path = tmp_path / "seg.nii.gz"
        NiftiWriter().write(path, seg)
        assert path.exists()
        assert sitk.ReadImage(str(path)).GetSize() == (g.shape_zyx[2], g.shape_zyx[1], g.shape_zyx[0])
