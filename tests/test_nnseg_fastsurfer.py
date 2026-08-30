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
    cross = np.argmax(mid == 1)                      # first index labeled 1
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


def test_sitk_to_nibabel_roundtrip_preserves_geometry():
    """The SITK->nibabel bridge must preserve geometry: a marked voxel's RAS
    physical location and value must survive the conversion (LPS->RAS,
    zyx->xyz). A wrong axis/sign here is the silent-mirror class of bug."""
    nib = pytest.importorskip("nibabel")
    # a non-trivial axis-aligned geometry (anisotropic, flipped, offset)
    arr = np.zeros((6, 8, 10), dtype=np.float32)      # sitk (z,y,x)
    arr[1, 2, 3] = 7.0                                # a marked voxel, sitk index (x=3,y=2,z=1)
    im = sitk.GetImageFromArray(arr)
    im.SetSpacing((1.0, 1.5, 2.0)); im.SetOrigin((10.0, -20.0, 5.0))
    im.SetDirection((-1., 0., 0., 0., -1., 0., 0., 0., 1.))   # LPS-ish flip

    nb = fs.sitk_to_nibabel(im)
    data = np.asanyarray(nb.dataobj)
    # value at nibabel index (i=3, j=2, k=1) == the marked voxel
    assert data[3, 2, 1] == 7.0
    # physical location must match: sitk physical point (RAS) of that voxel
    px, py, pz = im.TransformIndexToPhysicalPoint((3, 2, 1))    # LPS
    ras_sitk = np.array([-px, -py, pz])                          # LPS -> RAS
    ras_nib = (nb.affine @ np.array([3, 2, 1, 1.0]))[:3]
    assert np.allclose(ras_sitk, ras_nib, atol=1e-6), (ras_sitk, ras_nib)


def test_restore_gpu_matches_cpu_reference():
    """The GPU restore (grid_sample) must reproduce the SimpleITK CPU restore's
    argmax labels: same physical-space mapping and half-pixel (voxel-center)
    convention. Run on the CPU torch device so the geometry math is verified
    without a GPU. A flipped direction + anisotropic spacing + offset origin +
    upsampling exercises _resample_affine (the silent-bug-prone part)."""
    pytest.importorskip("torch")
    Z = Y = X = 16
    zz, yy, xx = np.meshgrid(np.arange(Z), np.arange(Y), np.arange(X), indexing="ij")
    # smooth per-class linear fields -> well-defined argmax planes, not tie noise
    feats = [xx, yy, zz, xx + yy, (X - 1 - xx)]
    logit = np.stack([f.astype(np.float32) for f in feats], axis=-1)     # (Z,Y,X,K)
    flip = (-1., 0., 0., 0., -1., 0., 0., 0., 1.)
    source = _img(np.zeros((Z, Y, X)), (1.5, 1.25, 1.0), origin=(10., -20., 5.))
    source.SetDirection(flip)
    target = _img(np.zeros((Z * 2, Y * 2, X * 2)), (0.75, 0.6, 0.5), origin=(8., -18., 6.))
    target.SetDirection(flip)

    cpu = fs.restore_logits(logit, source, target)
    gpu = fs.restore_logits_gpu(logit, source, target, device="cpu")
    assert cpu.shape == gpu.shape == (Z * 2, Y * 2, X * 2)
    interior = tuple(slice(3, -3) for _ in range(3))     # edges differ by padding only
    agree = float((cpu[interior] == gpu[interior]).mean())
    assert agree > 0.99, agree


def test_restore_gpu_tensor_input_matches_numpy_input():
    """The on-GPU handoff passes a (K,Zs,Ys,Xs) torch tensor instead of the numpy
    (Zs,Ys,Xs,K) field; both must yield identical labels. Guards the permute
    layout used when the field is kept resident on the device."""
    torch = pytest.importorskip("torch")
    Z = Y = X = 12
    zz, yy, xx = np.meshgrid(np.arange(Z), np.arange(Y), np.arange(X), indexing="ij")
    logit = np.stack([xx, yy, zz, xx + yy], axis=-1).astype(np.float32)   # (Z,Y,X,K)
    source = _img(np.zeros((Z, Y, X)), (1.0, 1.25, 1.5), origin=(3., -4., 5.))
    target = _img(np.zeros((Z * 2, Y * 2, X * 2)), (0.5, 0.625, 0.75), origin=(3., -4., 5.))

    from_numpy = fs.restore_logits_gpu(logit, source, target, device="cpu")
    tens = torch.from_numpy(logit).permute(3, 0, 1, 2).contiguous()       # (K,Z,Y,X)
    from_tensor = fs.restore_logits_gpu(tens, source, target, device="cpu")
    assert np.array_equal(from_numpy, from_tensor)


def test_sitk_nibabel_sitk_roundtrip_is_geometry_exact():
    """sitk -> nibabel -> sitk must recover size/spacing/origin/direction/data
    exactly. This is the bridge that recovers the conformed-orig geometry for the
    logit restore without a file round-trip; a silent error here misplaces every
    boundary."""
    arr = np.arange(6 * 8 * 10, dtype=np.float32).reshape(6, 8, 10)   # sitk (z,y,x)
    im = sitk.GetImageFromArray(arr)
    im.SetSpacing((1.0, 1.5, 2.0)); im.SetOrigin((10.0, -20.0, 5.0))
    im.SetDirection((-1., 0., 0., 0., -1., 0., 0., 0., 1.))

    back = fs.nibabel_to_sitk(fs.sitk_to_nibabel(im))
    assert back.GetSize() == im.GetSize()
    assert np.allclose(back.GetSpacing(), im.GetSpacing(), atol=1e-9)
    assert np.allclose(back.GetOrigin(), im.GetOrigin(), atol=1e-6)
    assert np.allclose(back.GetDirection(), im.GetDirection(), atol=1e-9)
    assert np.array_equal(sitk.GetArrayFromImage(back), arr)


def test_emit_probabilities_hands_over_the_field_with_both_grids():
    """The engine hook: FastSurfer's 79-class field goes through the same encoder as
    nnU-Net's, carrying enough geometry to redo the restore later."""
    from nnseg import ranked

    torch = pytest.importorskip("torch")
    K, Z, Y, X = 6, 4, 5, 6
    lg = torch.randn(K, Z, Y, X)
    conf = sitk.GetImageFromArray(np.zeros((Z, Y, X), np.float32))
    conf.SetSpacing((1.0, 1.0, 1.0)); conf.SetOrigin((-1.5, 2.0, 0.25))
    tgt = sitk.GetImageFromArray(np.zeros((Z + 1, Y, X), np.float32))
    tgt.SetSpacing((0.8, 0.9, 1.1)); tgt.SetOrigin((3.0, -4.0, 5.0))

    got = []
    spec = ranked.RankedSpec(sink=lambda part, code: got.append((part, code)), depth=3)
    fs.emit_probabilities(spec, lg, conf, tgt, list(range(K)))

    assert len(got) == 1
    part, code = got[0]
    assert part == "brain"
    assert code.meta["engine"] == "fastsurfer"          # a reader must know what made it
    assert code.meta["labels"] == list(range(K))
    assert code.meta["source_grid"]["origin_xyz"] == [-1.5, 2.0, 0.25]
    assert code.meta["target_grid"]["spacing_xyz"] == [0.8, 0.9, 1.1]
    assert code.meta["source_grid"]["size_xyz"] != code.meta["target_grid"]["size_xyz"]


def test_emit_probabilities_accepts_the_cpu_paths_axis_order():
    """_capture_logits returns (K,Z,Y,X) torch on the GPU path but (Z,Y,X,K) numpy on the
    CPU one; the encoder only takes the former, so the hook must transpose."""
    from nnseg import ranked

    pytest.importorskip("torch")
    K, Z, Y, X = 5, 3, 4, 5
    rng = np.random.default_rng(0)
    lg = rng.standard_normal((Z, Y, X, K)).astype(np.float32)
    ref = sitk.GetImageFromArray(np.zeros((Z, Y, X), np.float32))

    got = []
    spec = ranked.RankedSpec(sink=lambda part, code: got.append(code), depth=3)
    fs.emit_probabilities(spec, lg, ref, ref, list(range(K)))

    assert got[0].meta["shape"] == [Z, Y, X]            # not the (Z,Y,X,K) misread
    assert got[0].meta["classes"] == K
    # and the winner survives the transpose
    np.testing.assert_array_equal(got[0].ranks[0] - 1, lg.argmax(axis=3).astype(np.uint8))


def test_emit_probabilities_is_a_noop_without_a_spec():
    ref = sitk.GetImageFromArray(np.zeros((2, 2, 2), np.float32))
    fs.emit_probabilities(None, np.zeros((2, 2, 2, 3), np.float32), ref, ref, [0, 1, 2])
