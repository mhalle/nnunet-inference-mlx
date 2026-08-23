"""Normalization delegates to nnU-Net's own classes: exact for CT and ZScore, single channel."""
import numpy as np
import pytest

pytest.importorskip("nnunetv2")
from nnseg.preprocess import normalize
from nnunetv2.preprocessing.normalization import default_normalization_schemes as N


def test_ct_matches_nnunet_ctnormalization():
    rng = np.random.default_rng(0)
    img = (rng.normal(40, 300, (8, 10, 12))).astype(np.float32)
    props = {"mean": 42.0, "std": 150.0, "percentile_00_5": -500.0, "percentile_99_5": 900.0}
    got = normalize(img, ["CTNormalization"], props)
    want = N.CTNormalization(use_mask_for_norm=False, intensityproperties=props).run(img[None].copy())[0]
    np.testing.assert_array_equal(got, want)


def test_zscore_matches_nnunet_unmasked():
    rng = np.random.default_rng(1)
    img = rng.normal(300, 80, (6, 7, 9)).astype(np.float32)
    got = normalize(img, ["ZScoreNormalization"], {})
    want = N.ZScoreNormalization(use_mask_for_norm=False, intensityproperties={}).run(img[None].copy())[0]
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-6)
    # sanity: zero mean, unit std over the whole image
    assert abs(float(got.mean())) < 1e-4 and abs(float(got.std()) - 1.0) < 1e-4


def test_zscore_masked_uses_nonzero_region():
    img = np.zeros((5, 6, 7), np.float32)
    img[1:4, 1:5, 1:6] = np.random.default_rng(2).normal(500, 50, (3, 4, 5))
    got = normalize(img, ["ZScoreNormalization"], {}, use_mask_for_norm=[True])
    seg = np.where(img[None] != 0, 0, -1).astype(np.int8)
    want = N.ZScoreNormalization(use_mask_for_norm=True, intensityproperties={}).run(img[None].copy(), seg)[0]
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-6)
    # outside the mask stays exactly zero
    assert (got[img == 0] == 0).all()


def test_normalize_does_not_mutate_input():
    img = np.random.default_rng(3).normal(300, 80, (4, 5, 6)).astype(np.float32)
    before = img.copy()
    normalize(img, ["ZScoreNormalization"], {})
    np.testing.assert_array_equal(img, before)


def test_unknown_scheme_and_multichannel_raise():
    img = np.zeros((3, 3, 3), np.float32)
    with pytest.raises(NotImplementedError):
        normalize(img, ["NotARealScheme"], {})
    with pytest.raises(NotImplementedError):
        normalize(img, ["ZScoreNormalization", "ZScoreNormalization"], {})
