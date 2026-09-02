"""Reference implementation of the client renderer contract (store README 6.2):
selection reseed from ranks[0]+distance, analytic shell integration, glass.
Lungs as a semi-transparent thin shell with a fully transparent interior.

Everything comes from the three render arrays and nothing else - `support` is never read:

  signed field   sign from ranks[0] membership at VOXEL level, magnitude from `distance`,
                 then trilinear on the signed field (interpolating the folded magnitude
                 across a surface would erase the crossing).
  shell          opacity is a tent over |s|: full at the surface, zero at `--shell-mm`.
                 Density-based compositing, so tangential rays (silhouettes) accumulate
                 more than face-on rays - the classic translucent-shell look.
  interior       nothing. A ray inside a lobe adds no opacity at all.
  fissures       the stored distance dips to zero at lobe|lobe surfaces too. The outer-only
                 variant suppresses them with the ranks[1] mask: a surface counts iff
                 exactly one side of it is in the selection. Both variants render, because
                 seeing the fissures IS seeing the mask work.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib
import numpy as np
import torch
import zarr

matplotlib.use("Agg")
import matplotlib.image as mpimg

SEED_FROM = "deficit"
DEMO = Path.home() / "Dropbox/development/medseg/nnunet-inference-mlx/data/duckn_demo"
LOBES = {
    "lung_upper_lobe_right":  (0.36, 0.62, 0.86),
    "lung_middle_lobe_right": (0.30, 0.78, 0.72),
    "lung_lower_lobe_right":  (0.22, 0.45, 0.74),
    "lung_upper_lobe_left":   (0.92, 0.54, 0.40),
    "lung_lower_lobe_left":   (0.80, 0.36, 0.46),
}
VIEWS = {"anterior": (0, 0), "right_oblique": (10, -45), "left_oblique": (10, 45)}


def load(store, device):
    r = zarr.open_group(str(store), mode="r")
    names = {s["name"]: int(s["label_value"])
             for s in r.attrs.asdict()["duckn"]["extensions"]["seg"]["segments"]
             if not isinstance(s["label_value"], list)}
    g = r["parts/0"]                                            # organs
    b = g.attrs.asdict()["duckn"]["extensions"]["ranked"]
    T = float(b["distance_truncation"])
    lut = np.asarray(b["labels"])
    sp = np.array([float(np.linalg.norm(a["space_direction"]))
                   for a in g["ranks"].attrs.asdict()["duckn"]["axes"]
                   if a.get("space_direction")])

    sel_globals = {names[n] for n in LOBES}
    member = np.isin(lut, sorted(sel_globals))                  # channel index -> selected?
    colidx = np.zeros(len(lut) + 1, np.int64)                   # +1 for the rank shift
    colors = [(0.0, 0.0, 0.0)]
    for n, rgb in LOBES.items():
        ch = int(np.nonzero(lut == names[n])[0][0])
        colidx[ch + 1] = len(colors)
        colors.append(rgb)

    rk = np.asarray(g["ranks"][:])
    su = np.asarray(g["support"][:])
    dist = np.asarray(g["distance"][:])
    rk0 = rk[0]

    inside = member[np.clip(rk0.astype(np.int64) - 1, 0, len(lut) - 1)] & (rk0 > 0)
    # crop to the selection plus the band, for speed
    idx = np.nonzero(inside)
    pad = 6
    box = tuple(slice(max(0, int(i.min()) - pad), min(n, int(i.max()) + pad))
                for i, n in zip(idx, inside.shape))
    rk = rk[(slice(None),) + box]
    su = su[(slice(None),) + box]
    rk0, dist, inside = rk0[box], dist[box], inside[box]
    print(f"  crop {inside.shape} of the organs grid; {int(inside.sum()):,} lung voxels")

    # THE SELECTION'S OWN DISTANCE FIELD, re-seeded from the store - the client-reseed path
    # the design promised. The baked field is distance to ANY surface, so a tent over it
    # fires at the liver as happily as at the lung; a gate on sampled membership fixes that
    # at voxel resolution and stipples. Seeding only the edges that matter - selection
    # membership flips for the outer field, any-class flips touching the selection for the
    # all-surfaces field - gives fields that are zero exactly on the wanted surfaces, with
    # sub-voxel crossings from the deficits, and no gate anywhere downstream.
    import importlib.util
    tools = (Path.home() / "Dropbox/development/medseg/nnunet-inference-mlx"
             / "tools" / "ranked_build_store.py")
    spec = importlib.util.spec_from_file_location("rbs", tools)
    rbs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rbs)
    clip = float(b["clip"])

    dmm = np.where(dist > 0, (1.0 - dist.astype(np.float32) / 255.0) * T, T)

    def selection_distance(edge_matters):
        d = np.full(inside.shape, np.inf, np.float32)
        for axis, step in enumerate(float(v) for v in sp):
            lo = [slice(None)] * 3
            hi = [slice(None)] * 3
            lo[axis], hi[axis] = slice(0, -1), slice(1, None)
            lo, hi = tuple(lo), tuple(hi)
            flip = (rk[0][lo] != rk[0][hi]) & edge_matters(lo, hi)
            if not flip.any():
                continue
            at = np.nonzero(flip)
            bt = list(at)
            bt[axis] = at[axis] + 1
            bt = tuple(bt)
            if SEED_FROM == "deficit":
                dq_a = rbs._deficit_at(rk[(slice(None),) + at], su[(slice(None),) + at],
                                       rk[0][bt], clip)
                dp_b = rbs._deficit_at(rk[(slice(None),) + bt], su[(slice(None),) + bt],
                                       rk[0][at], clip)
            else:                       # "baked": crossings from the distance field alone -
                dq_a, dp_b = dmm[at], dmm[bt]   # the planar cosine cancels in the ratio
            den = dq_a + dp_b
            t = np.divide(dq_a, den, out=np.full_like(dq_a, 0.5), where=den > 1e-9)
            np.minimum.at(d, at, t * step)
            np.minimum.at(d, bt, (1.0 - t) * step)
        return rbs._eikonal(d, list(map(float, sp)), T)

    d_outer = selection_distance(lambda lo, hi: inside[lo] != inside[hi])
    d_all = selection_distance(lambda lo, hi: inside[lo] | inside[hi])
    signed = np.where(inside, d_outer, -d_outer).astype(np.float32)
    signed_all = np.where(inside, np.minimum(d_outer, d_all),
                          -np.minimum(d_outer, d_all)).astype(np.float32)
    print(f"  selection fields: outer band {(d_outer < T).mean():.1%}, "
          f"all-surfaces band {(d_all < T).mean():.1%}")

    from scipy import ndimage
    shade = ndimage.gaussian_filter(signed, sigma=1.0)
    shade_all = ndimage.gaussian_filter(signed_all, sigma=1.0)

    dev = torch.device(device)
    # colour of the NEAREST lobe everywhere, not of the sample's own voxel: the shell
    # straddles the surface and the outside half must still know which lobe it wraps.
    colmap = colidx[np.maximum(rk0.astype(np.int64), 0)]
    nz = ndimage.distance_transform_edt(colmap == 0, return_distances=False,
                                        return_indices=True)
    colmap = colmap[nz[0], nz[1], nz[2]]

    def pack(sig, shd):
        grad = torch.stack(torch.gradient(torch.from_numpy(shd),
                                          spacing=[float(v) for v in sp]), 0)
        return torch.from_numpy(sig).to(dev), grad.to(dev).contiguous()

    return (pack(signed, shade), pack(signed_all, shade_all),
            torch.from_numpy(colmap).to(dev), sp, T,
            torch.tensor(colors, dtype=torch.float32, device=dev))


def render(signed, grad, colid, sp, colors, *, width, height, elev, azim,
           shell_mm, alpha, step_mm, device):
    Z, Y, X = signed.shape
    ext = np.array([Z * sp[0], Y * sp[1], X * sp[2]])
    el, az = np.radians(elev), np.radians(azim)
    fwd = np.array([np.sin(el), -np.cos(el) * np.cos(az), np.cos(el) * np.sin(az)])
    fwd /= np.linalg.norm(fwd)
    up0 = np.array([1.0, 0.0, 0.0])
    if abs(float(np.dot(fwd, up0))) > 0.98:
        up0 = np.array([0.0, 1.0, 0.0])
    right = np.cross(fwd, up0); right /= np.linalg.norm(right)
    up = np.cross(right, fwd)

    occ = torch.nonzero(signed > 0)
    spt = torch.tensor(sp, dtype=torch.float32, device=device)
    lo = (occ.min(0).values.float() * spt).cpu().numpy()
    hi = (occ.max(0).values.float() * spt).cpu().numpy()
    centre = torch.tensor((lo + hi) / 2, dtype=torch.float32, device=device)
    span = float(np.linalg.norm(hi - lo)) * 0.42
    depth_mm = float(np.linalg.norm(hi - lo)) * 1.1

    aspect = width / height
    u = torch.linspace(-span * aspect, span * aspect, width, device=device)
    v = torch.linspace(span, -span, height, device=device)
    vv, uu = torch.meshgrid(v, u, indexing="ij")
    R = torch.tensor(right, dtype=torch.float32, device=device)
    U = torch.tensor(up, dtype=torch.float32, device=device)
    F = torch.tensor(fwd, dtype=torch.float32, device=device)
    light = torch.tensor(-fwd * 0.5 + up * 0.6 - right * 0.5,
                         dtype=torch.float32, device=device)
    light /= light.norm()
    LIGHT_RGB = torch.tensor([1.0, 0.98, 0.94], device=device)   # warm-white key light
    view = torch.tensor(-fwd, dtype=torch.float32, device=device)
    half = light + view
    half = half / half.norm()

    shape = torch.tensor([Z, Y, X], dtype=torch.float32, device=device)
    fld = signed[None, None]
    grad_v = grad[None]
    col_f = colid[None, None].float()
    extent_t = torch.tensor(ext, dtype=torch.float32, device=device)
    img = torch.zeros(height, width, 3, device=device)

    def samp(vol, pos, mode="bilinear", vec=False):
        idxp = pos / spt - 0.5
        n = 2.0 * idxp / (shape - 1) - 1.0
        gridp = torch.stack([n[..., 2], n[..., 1], n[..., 0]], -1)[None, None]
        o = torch.nn.functional.grid_sample(vol, gridp, mode=mode, align_corners=True,
                                            padding_mode="zeros")
        return o[0, :, 0].permute(1, 2, 0) if vec else o[0, 0, 0]

    n_steps = int(depth_mm / step_mm)
    t0 = time.perf_counter()
    band = max(1, int(300_000 / width))
    for y0 in range(0, height, band):
        y1 = min(y0 + band, height)
        origin = (centre[None, None, :] + uu[y0:y1][..., None] * R
                  + vv[y0:y1][..., None] * U - F * depth_mm / 2)
        rgb = torch.zeros(y1 - y0, width, 3, device=device)
        trans = torch.ones(y1 - y0, width, device=device)

        def tent_avg(a_, b_):
            """Mean of max(0, 1-u/W) for u sweeping linearly from a_ to b_ (both >= 0).

            Point-sampling this tent is what caused the stipple: with a shell only a few
            steps wide, each pixel's sum depends on where its samples land relative to the
            crossing - a per-pixel phase that supersampling averages but cannot remove. The
            segment integral is closed-form, so the phase drops out exactly.
            """
            W = shell_mm
            lo = torch.minimum(a_, b_).clamp(min=0)
            hi = torch.maximum(a_, b_)
            c = hi.clamp(max=W)
            num = (c - lo) - (c * c - lo * lo) / (2 * W)
            avg = torch.where(hi > lo + 1e-6, num / (hi - lo + 1e-9),
                              torch.clamp(1 - lo / W, 0, 1))
            return torch.where(lo >= W, torch.zeros_like(avg), avg)

        prev_s = prev_ok = None
        for s in range(n_steps):
            if trans.max() < 0.01:
                break
            pos = origin + F * (s * step_mm)
            ok = (pos >= 0).all(-1) & (pos < extent_t).all(-1)
            sv = samp(fld, pos)
            if prev_s is not None and ok.any():
                a0, b0 = prev_s.abs(), sv.abs()
                crossing = (prev_s * sv) < 0
                f0 = a0 / (a0 + b0 + 1e-9)
                w = torch.where(crossing,
                                f0 * tent_avg(a0, torch.zeros_like(a0))
                                + (1 - f0) * tent_avg(torch.zeros_like(b0), b0),
                                tent_avg(a0, b0)) * (ok & prev_ok)
                mid = pos - F * (step_mm / 2)
                lid = samp(col_f, mid, mode="nearest").long().clamp(0, colors.shape[0] - 1)
                if (w > 0).any():
                    gv = samp(grad_v, mid, vec=True)
                    nrm = -gv / (gv.norm(dim=-1, keepdim=True) + 1e-6)
                    # GLASS: opacity follows Fresnel - nearly clear face-on, strong at
                    # grazing incidence - so the colour lives on the silhouettes and the
                    # faces read as clear material rather than fog.
                    ndv = (nrm * view).sum(-1).abs()
                    fresnel = 0.03 + 0.97 * (1.0 - ndv) ** 3
                    a = 1.0 - torch.exp(-alpha * w * fresnel * step_mm * 3.0)
                    lam = (nrm * light).sum(-1).abs()               # two-sided for a shell
                    sh = 0.22 + 0.38 * lam
                    contrib = trans * a
                    rgb += contrib[..., None] * (colors[lid] * sh[..., None])
                    # The highlight is a SURFACE reflection in the light's own colour: it
                    # composites through the accumulated transmittance but is NOT gated by
                    # the shell's transmission alpha - glass reflects however clear its body
                    # is. Presence-weighted by w so it exists only on the shell.
                    spec = (nrm * half).sum(-1).abs().pow(220)
                    # w integrates to ~shell/step over a crossing; dividing by that count
                    # makes the highlight per-SURFACE, not per-sample, so it cannot bloom
                    rgb += (trans * w * (step_mm / shell_mm)
                            * spec * 1.1)[..., None] * LIGHT_RGB
                    trans = trans * (1 - a)
            prev_s, prev_ok = sv, ok
        img[y0:y1] = rgb
    print(f"    {n_steps} steps x {height} rows in {time.perf_counter() - t0:.0f}s")
    return (img.clamp(0, 1) ** (1 / 2.2)).cpu().numpy()


ap = argparse.ArgumentParser()
ap.add_argument("--store", default=str(DEMO / "idc-torso1/total.duckn"))
ap.add_argument("--width", type=int, default=1000)
ap.add_argument("--shell-mm", type=float, default=1.2)
ap.add_argument("--alpha", type=float, default=0.55, help="shell density per mm")
ap.add_argument("--step-mm", type=float, default=0.5)
ap.add_argument("--view", default="anterior")
ap.add_argument("--seed-from", default="deficit", choices=["deficit", "baked"])
ap.add_argument("--out", default=str(Path(__file__).parent / "lung_shell.png"))
a = ap.parse_args()

dev = "mps" if torch.backends.mps.is_available() else "cpu"
SEED_FROM = a.seed_from
print(f"device {dev}  seeds from {SEED_FROM}")
fields_outer, fields_all, colid, sp, T, colors = load(a.store, dev)
for name in a.view.split(","):
    elev, azim = VIEWS[name]
    for variant, (sig, grd) in (("outer", fields_outer), ("all", fields_all)):
        print(f"{name}/{variant}:")
        img = render(sig, grd, colid, sp, colors,
                     width=a.width, height=int(a.width * 0.95), elev=elev, azim=azim,
                     shell_mm=a.shell_mm, alpha=a.alpha, step_mm=a.step_mm, device=dev)
        f = Path(a.out).with_name(f"{Path(a.out).stem}_{name}_{variant}.png")
        mpimg.imsave(str(f), img)
        print(f"    wrote {f.name}")
