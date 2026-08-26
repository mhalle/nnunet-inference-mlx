"""FastSurfer engine: the geometry (restore_logits) and the LUT, tested with
synthetic logits so no FastSurfer install or GPU is needed. The FastSurfer-
dependent compute (conform + inference) is validated by the live Modal smoke."""
import numpy as np
import pytest

sitk = pytest.importorskip("SimpleITK")

from nnseg.engines import fastsurfer as fs


def test_module_imports_without_fastsurfer():
    # importing the engine must not require FastSurfer (lazy inside segment())
    assert callable(fs.segment) and callable(fs.restore_logits)


def test_lut_has_canonical_freesurfer_labels():
    lut = fs.load_lut()
    assert lut[2]["name"] == "Left-Cerebral-White-Matter"
    assert lut[41]["name"] == "Right-Cerebral-White-Matter"
    assert lut[16]["name"] == "Brain-Stem"
    for v in (2, 41, 16):
        assert len(lut[v]["color"]) == 3 and all(0 <= c <= 255 for c in lut[v]["color"])
    assert len(lut) > 70          # the ~95 aparc+aseg structures (minus background)


def _img(arr, spacing, origin=(0., 0., 0.)):
    im = sitk.GetImageFromArray(np.ascontiguousarray(arr))
    im.SetSpacing(spacing); im.SetOrigin(origin)
    return im


def test_restore_logits_places_boundary_at_physical_location():
    """A 2-class field whose boundary is the plane x=c: after restore to a finer
    grid, the argmax boundary must sit at the same physical x=c (sub-voxel), not
    snap to the coarse grid."""
    Zc = Yc = Xc = 24
    sp_c = 1.0
    xc_phys = 11.3                                   # boundary NOT on a coarse voxel center
    # sitk array is (z,y,x); build 2 channels of logits: ch1-ch0 = (x_phys - xc)
    xphys = (np.arange(Xc) * sp_c)[None, None, :] * np.ones((Zc, Yc, Xc))
    d = xphys - xc_phys
    logit = np.stack([-d, d], axis=-1).astype(np.float32)   # argmax=1 where x>xc
    source = _img(np.zeros((Zc, Yc, Xc)), (sp_c, sp_c, sp_c))

    # target: finer grid (0.25mm) over the same FOV -> upsampling
    sp_f = 0.25
    Xf = int(Xc * sp_c / sp_f)
    target = _img(np.zeros((Zc*4, Yc*4, Xf)), (sp_f, sp_f, sp_f))

    idx = fs.restore_logits(logit, source, target)
    # find the crossover column per row: first x where idx==1
    mid = idx[idx.shape[0]//2, idx.shape[1]//2, :]
    cross = np.argmax(mid == 1)                      # first index labelled 1
    cross_phys = cross * sp_f
    assert abs(cross_phys - xc_phys) <= sp_f + 1e-6, (cross_phys, xc_phys)


def test_restore_logits_beats_label_nn_on_a_slanted_boundary():
    """On upsampling, resampling the graded field + argmax matches the true
    boundary better than nearest-neighbor resampling the coarse labelmap."""
    from scipy import ndimage
    Z = Y = X = 20
    # slanted boundary: label 1 where (x + 0.5*y) > c, graded logit = signed dist
    zz, yy, xx = np.meshgrid(np.arange(Z), np.arange(Y), np.arange(X), indexing="ij")
    signed = (xx + 0.5*yy) - 15.0
    logit = np.stack([-signed, signed], axis=-1).astype(np.float32)
    lab_coarse = (signed > 0).astype(np.int32)
    source = _img(np.zeros((Z, Y, X)), (1., 1., 1.))
    f = 4
    target = _img(np.zeros((Z*f, Y*f, X*f)), (0.25, 0.25, 0.25))

    lg = fs.restore_logits(logit, source, target)
    # truth at fine res
    zf, yf, xf = np.meshgrid(*[(np.arange(n*f)*0.25) for n in (Z, Y, X)], indexing="ij")
    truth = ((xf + 0.5*yf) - 15.0 > 0).astype(np.int32)
    nn = ndimage.zoom(lab_coarse, f, order=0)
    # compare the interior only: target voxels near the FOV edge sample outside
    # the source support (default 0) - a test artifact, not a restore effect
    c = tuple(slice(8, -8) for _ in range(3))
    lg_err = float((lg[c] != truth[c]).mean())
    nn_err = float((nn[c] != truth[c]).mean())
    assert lg_err < nn_err, (lg_err, nn_err)        # logit-grade closer to truth
    assert lg_err < 0.005                            # near-exact on a linear field
