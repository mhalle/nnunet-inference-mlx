"""Compress a model's output distribution: top-N ranks plus quantized margins.

A labelmap answers "which class won" and throws away "by how much". This keeps the
second answer at a few tenths of a byte per voxel, computed from the logits at the one
moment they exist - between the network and the restore - so nothing is recomputed and
no K-channel probability volume is ever materialized.

What is stored, per voxel::

    ranks    (N, Z, Y, X)  class + 1 for the N best channels, 0 = "not this class"
    support  (N-1, Z, Y, X)  how far each trails the winner, 255 = tied ... 0 = at the clip
    tail     (Z, Y, X)     probability mass beyond the top N (omitted when exhaustive)

Three things about that layout are load-bearing.

*Margins, not probabilities.* Softmax has a per-voxel gauge freedom, so only differences
between logits are identifiable. Differences also quantize uniformly where a probability
does not, and - the reason this survives resampling - the linear interpolant of a logit
difference is right where the interpolant of a softmax output is not, which matters
exactly at boundaries. Since the gaps ARE logit differences, ``topk`` on the logits
yields them directly: no softmax, no ``log``, no round trip.

*Zero means "nothing here", in every array.* ``ranks`` stores ``class + 1`` because class
0 is legitimately background, and ``support`` counts UP from the clip rather than down
from the winner. So the fill value is 0 throughout, an unwritten chunk decodes to
"absent", and a reader that forgets the fill value - or a zero-cleared buffer, or a
sparse default - gets the safe answer instead of "tied with the winner". Storing the
deficit instead would make that mistake maximally wrong. It is also smaller: with small
integers the high byte is constant, which byte-shuffling compressors get for free.

*The clip bounds each class's spatial support.* Beyond ``clip`` logits behind the winner a
class is indistinguishable from absent, so its rank plane is masked to the sentinel and
whole regions become uniform - which is what makes depth nearly free and lets a store
skip empty blocks entirely. Precision and range trade against each other here: a smaller
clip quantizes finer AND stores smaller, at the cost of dynamic range for the tail.

Two fields fall out of the same bytes, and they are not interchangeable:

``deficit`` (``l_c - max_j l_j``, zero where c wins) differs from the logits by a per-voxel
constant shared by every channel - a gauge transformation - so interpolating it and taking
the argmax is exactly interpolating the logits. **That is the field a restore must use.**

``margin`` (``l_c - max_{j != c} l_j``, positive inside by c's lead) is the signed "is c
winning" field whose zero level set is c's surface, with an interior gradient to shade and
mesh. **That is the field a renderer wants** - and it is NOT gauge-equivalent, because the
lead is added to one channel only, so an argmax over interpolated margins is not an argmax
over interpolated logits.

Region (sigmoid) heads have no winner and no normalizer, so ranking buys nothing there -
but their logit IS already such a margin, referenced to the decision threshold. That is
what :func:`encode_regions` writes, and it decodes through the same ``margin`` call, so a
consumer never needs to know which head produced a file.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import torch

CLIP = 8.0            # logits behind the winner past which a class reads as absent
SUPPORT_MAX = 255     # uint8: 0 = at/beyond the clip, 255 = tied with the winner
ZERO_LEVEL = 128      # regions only: the level the decision boundary lands on exactly
TAIL_MAX = 255
DEFAULT_DEPTH = 6

__all__ = ["CLIP", "DEFAULT_DEPTH", "RankedCode", "RankedSpec", "decode_groups", "deficit",
           "emit", "encode", "encode_regions", "margin", "probabilities", "to_device"]


@dataclass(frozen=True)
class RankedSpec:
    """Ask a run to emit its output distribution, and say where each part's code goes.

    A sink rather than a return value, deliberately: a multi-model task holds one part's
    logits at a time and frees them, and the uncompressed codes are far larger than the
    stored form - so each part is handed over as it is produced and can be written and
    dropped. It is also the shape the layers want, since one part is one independent file.
    """

    sink: Callable[[str, RankedCode], None]
    depth: int = DEFAULT_DEPTH
    clip: float = CLIP


@dataclass
class RankedCode:
    """Encoded output distribution for one model, on that model's own grid.

    ``meta`` travels with the arrays and is what makes a file self-describing: the caller
    (the pipeline) adds geometry and provenance to it before writing.
    """

    ranks: np.ndarray | None
    support: np.ndarray
    tail: np.ndarray | None
    meta: dict = field(default_factory=dict)

    @property
    def nbytes(self) -> int:
        return sum(a.nbytes for a in (self.ranks, self.support, self.tail) if a is not None)

    def __repr__(self) -> str:
        shape = "x".join(str(s) for s in self.support.shape[1:])
        return (f"RankedCode({self.meta.get('mode', '?')}, K={self.meta.get('classes')}, "
                f"depth={self.meta.get('depth')}, {shape}, {self.nbytes / 1e6:.1f} MB raw)")


def _rank_dtype(K: int):
    """``class + 1`` must fit, and the sentinel is 0. Declared in meta, never assumed."""
    return np.uint8 if K + 1 <= np.iinfo(np.uint8).max else np.uint16


def _gap(support, clip: float) -> np.ndarray:
    """Stored support -> the logit deficit it encodes. Support counts UP from the clip."""
    return (1.0 - np.asarray(support, dtype=np.float32) / SUPPORT_MAX) * clip


def _check(logits) -> torch.Tensor:
    if not isinstance(logits, torch.Tensor):
        raise TypeError(f"logits must be a torch.Tensor; got {type(logits).__name__}")
    if logits.ndim != 4:
        raise ValueError(f"logits must be (K, Z, Y, X); got shape {tuple(logits.shape)}")
    if not logits.dtype.is_floating_point:
        raise TypeError(f"logits must be floating point; got {logits.dtype}")
    return logits


def _settle_ties(top: torch.Tensor, idx: torch.Tensor, N: int) -> None:
    """Order equal-valued entries of ``topk`` by ascending class index, in place.

    ``topk`` leaves the order among EQUAL values undefined, and CUDA's is not the CPU's, so
    without this the stored bytes depend on where they were encoded. Measured on real fp16
    logits (which is what the network returns, and which ties far more readily than fp32):
    0.0033 % of voxels, 85 % of them involving background. Two different costs - a tie at
    the winner flips a label whose margin is ~0 either way, but a tie at the DEPTH BOUNDARY
    decides which class is kept and which drops to the sentinel, so the loser decodes to
    ``-clip`` instead of ``-gap``.

    Lower index wins, which is argmax's own convention - numpy and torch both return the
    first maximal index - so ``ranks[0]`` remains exactly the argmax that every labelmap in
    this ecosystem was made with. That inherits argmax's bias toward background rather than
    introducing a new one; the bound is 0.155 % of the worst structure's volume, against
    ~9 % for the linear-vs-nearest choice already in the pipeline.

    Bubble, because a tie can in principle run longer than a pair. ``N`` is small and each
    pass is a few cheap elementwise kernels, so the early-out is deliberately omitted - it
    would cost a device sync per pass to save less than it costs.

    ⚠ This orders the N entries ``topk`` selected; it cannot change WHICH it selected. A tie
    straddling the depth boundary - two classes equal at the last kept slot - is still
    resolved by ``topk``, hence still backend-dependent. Measured on the real organs part at
    depth 6: 194,043 voxels have a boundary tie, but all except **16** are among classes
    beyond the clip, where both candidates mask to the zero sentinel and the choice never
    reaches the stored bytes. For those 16 the contested gap is 6.60-7.97 against a clip of
    8, so the worst decoded margin differs by 1.40 logits, on the least-significant plane,
    for a class already >6.5 logits behind the winner.

    Making even that exact needs the SELECTION to be deterministic - a stable descending sort
    instead of ``topk``, measured at 3.4-3.7x - which is a poor trade against 16 voxels in
    11.5 M on the path that dominates encode time. Revisit only if the store ever needs to be
    content-addressed by checksum, where any difference at all is a difference.
    """
    for _ in range(N - 1):
        for j in range(N - 1):
            swap = (top[j] == top[j + 1]) & (idx[j] > idx[j + 1])
            lo = torch.where(swap, idx[j + 1], idx[j])
            hi = torch.where(swap, idx[j], idx[j + 1])
            idx[j], idx[j + 1] = lo, hi


def encode(logits: torch.Tensor, *, depth: int = DEFAULT_DEPTH, clip: float = CLIP,
           slab: int = 32, with_tail: bool = True) -> RankedCode:
    """``(K, Z, Y, X)`` logits -> :class:`RankedCode`, in slabs on the logits' own device.

    ``depth`` is how many channels are kept per voxel. It is far cheaper than it looks:
    planes past the first few sit at the clip almost everywhere and mask to the sentinel,
    so raising it changes the compressed size by a few percent while shrinking the tail.

    ``slab`` bounds peak memory - one z-slab is promoted to fp32 at a time, not the whole
    volume - so this costs a fraction of the inference that produced ``logits``.
    """
    lg_all = _check(logits)
    K = int(lg_all.shape[0])
    N = max(1, min(int(depth), K))
    Z, Y, X = (int(v) for v in lg_all.shape[1:])
    exhaustive = N >= K
    rdt = _rank_dtype(K)

    ranks = np.empty((N, Z, Y, X), rdt)
    support = np.empty((N - 1, Z, Y, X), np.uint8) if N > 1 else np.empty((0, Z, Y, X), np.uint8)
    tail = None if (exhaustive or not with_tail) else np.empty((Z, Y, X), np.uint8)
    max_tail = 0.0

    for z0 in range(0, Z, max(1, int(slab))):
        z1 = min(z0 + max(1, int(slab)), Z)
        lg = lg_all[:, z0:z1].float()
        top, idx = torch.topk(lg, N, dim=0)              # descending; top[0] is the winner
        _settle_ties(top, idx, N)
        gaps = top[0:1] - top                            # >= 0, and gaps[0] is exactly 0

        r = (idx + 1).to(torch.int32)
        if N > 1:
            beyond = gaps[1:] >= clip
            r[1:][beyond] = 0                            # sentinel: not this class here
            sup = ((1.0 - gaps[1:] / clip).clamp(0, 1) * SUPPORT_MAX).round()
            support[:, z0:z1] = sup.to(torch.uint8).cpu().numpy()
        ranks[:, z0:z1] = r.cpu().numpy().astype(rdt, copy=False)

        if tail is not None:
            # the only place all K channels are looked at: what the top N discards
            z_full = torch.exp(lg - top[0:1]).sum(0)
            z_top = torch.exp(-gaps).sum(0)
            t = ((z_full - z_top) / z_full).clamp(0, 1)
            max_tail = max(max_tail, float(t.max()))
            tail[z0:z1] = (t * TAIL_MAX).round().to(torch.uint8).cpu().numpy()
        del lg, top, idx, gaps

    return RankedCode(ranks=ranks, support=support, tail=tail, meta={
        "mode": "ranked", "classes": K, "depth": N, "clip": float(clip),
        "rank_dtype": np.dtype(rdt).name, "support_max": SUPPORT_MAX,
        "rank_sentinel": 0, "exhaustive": bool(exhaustive),
        "max_tail": (0.0 if exhaustive else max_tail),
        "shape": [Z, Y, X],
    })


def encode_regions(logits: torch.Tensor, *, clip: float = CLIP, threshold: float = 0.0,
                   slab: int = 32) -> RankedCode:
    """Sigmoid-head logits -> one margin plane per region (no ranks, no tail).

    Overlapping regions are independent Bernoullis: nothing sums to one, several channels
    can be present at once, and ranking is meaningless. The logit is already the margin,
    so all that is needed is to reference it to the decision threshold and quantize -
    ``m_c = l_c - threshold``, positive inside. Folding the threshold in here is what lets
    a consumer treat zero as the boundary without knowing the head type.
    """
    lg_all = _check(logits)
    K = int(lg_all.shape[0])
    Z, Y, X = (int(v) for v in lg_all.shape[1:])
    support = np.empty((K, Z, Y, X), np.uint8)
    for z0 in range(0, Z, max(1, int(slab))):
        z1 = min(z0 + max(1, int(slab)), Z)
        m = lg_all[:, z0:z1].float() - float(threshold)
        # 1 = clip below the boundary, ZERO_LEVEL = exactly on it, 255 = clip above. The
        # boundary lands ON a level rather than between two, and 0 stays reserved as the
        # fill sentinel, so an unwritten block still reads as "absent" here too.
        q = ((m / clip).clamp(-1, 1) * (SUPPORT_MAX - ZERO_LEVEL) + ZERO_LEVEL).round()
        support[:, z0:z1] = q.clamp(1, SUPPORT_MAX).to(torch.uint8).cpu().numpy()
        del m, q
    return RankedCode(ranks=None, support=support, tail=None, meta={
        "mode": "regions", "classes": K, "depth": K, "clip": float(clip),
        "threshold": float(threshold), "support_max": SUPPORT_MAX,
        "signed_support": True, "support_zero": ZERO_LEVEL, "exhaustive": True,
        "max_tail": 0.0, "shape": [Z, Y, X],
    })


def emit(spec, part, logits, /, **meta) -> "RankedCode | None":
    """Encode ``logits`` into ``spec``'s sink, stamping ``meta`` onto the code.

    The one seam every engine hands its output distribution through. It exists because the
    logits are alive only between the network and the restore, so each engine must do this
    for itself at its own call site - and four copies would drift, in the depth and clip the
    spec asked for and in what lands in ``meta``.

    ``spec`` may be ``None``, so a call site can be unconditional rather than guarded.

    The first three are positional-only on purpose: ``meta`` is open-ended and engine-chosen,
    and the nnU-Net path really does stamp ``part`` into it, so a keyword ``part`` here would
    collide with the sink key and raise.

    What an engine puts in ``meta`` is its own business - the grids differ, the label
    mapping differs - but it must be enough to redo the restore later: the grid the field
    was computed on, the grid it restores onto, and channel -> label. Without that the
    arrays are only a picture of one run. Stamp ``engine`` too; with more than one engine
    emitting, a reader cannot otherwise tell what produced the file.
    """
    if spec is None:
        return None
    code = encode(logits, depth=spec.depth, clip=spec.clip)
    code.meta.update(meta)
    spec.sink(str(part), code)
    return code


def deficit(code: RankedCode, channel: int) -> np.ndarray:
    """``l_c - max_j l_j`` for one channel: zero where it wins, negative behind.

    **This is the field to interpolate when the answer is a label.** It differs from the
    logits by ``-max_j l_j``, a per-voxel constant shared by EVERY channel, so it is a gauge
    transformation: interpolating it and taking the argmax is exactly interpolating the
    logits and taking the argmax.

    :func:`margin` is not interchangeable here, and the difference is not small. It adds the
    winner's lead to the winning channel only - channel-dependent, so not a gauge - which
    survives an argmax at voxel centers but not after interpolation, where each corner of the
    stencil has a different winner. Measured on a real K=118 case, restoring through ``margin``
    agrees with the logits on 99.43 % of sub-voxel samples and only 84 % of near-tie samples;
    through ``deficit``, 99.98 % and 99.4 %. Nearest-neighbor restore hides the difference
    entirely, because it never mixes voxels.
    """
    clip = float(code.meta["clip"])
    out = margin(code, channel)
    if code.meta.get("mode") == "regions":
        return out                                            # no winner: nothing to remove
    won = code.ranks[0] == int(channel) + 1
    out[won] = 0.0
    return out


def margin(code: RankedCode, channel: int) -> np.ndarray:
    """``l_c - max_{j != c} l_j`` for one channel: the signed "is c winning" field.

    Positive inside by the amount it leads, zero ON its boundary, negative outside, ``-clip``
    where the channel is absent. This is the field for rendering and meshing ONE structure:
    its zero level set is that structure's surface, it has an interior gradient to shade and
    to place a sub-voxel isosurface with, and it behaves like a signed distance field measured
    in confidence.

    Do NOT feed it to a restore. Use :func:`deficit` when the output is a label map - see
    there for why, and for what it costs.
    """
    _host(code, "margin")
    clip = float(code.meta["clip"])
    if code.meta.get("mode") == "regions":
        q = code.support[int(channel)].astype(np.float32)
        out = (q - ZERO_LEVEL) / (SUPPORT_MAX - ZERO_LEVEL) * clip
        out[q == 0] = -clip                                   # the fill sentinel: absent
        return out
    shape = tuple(code.meta["shape"])
    out = np.full(shape, -clip, np.float32)
    want = int(channel) + 1                                   # ranks store class + 1
    sel = code.ranks[0] == want
    if code.support.shape[0]:
        # the winner's margin IS the runner-up's deficit, and support counts up from the
        # clip: support 255 -> gap 0 (a dead tie), support 0 -> gap clip
        out[sel] = _gap(code.support[0][sel], clip)
    else:
        out[sel] = clip                                       # depth 1: no runner-up stored
    for j in range(1, code.ranks.shape[0]):
        sel = code.ranks[j] == want
        out[sel] = -_gap(code.support[j - 1][sel], clip)       # rank j trails the winner
    return out


def to_device(code: RankedCode, device) -> RankedCode:
    """Move the encoded planes onto ``device`` once, so repeated decodes do not re-upload.

    This is the residency the design wants: the encoded form is small (a few MB) and stays
    put; the fields it stands for are large and are materialized transiently by
    :func:`decode_groups`. Without it every decode ships the planes across again - 126 MB
    for a 6-deep organs store - which is wasted on the interactive case the whole scheme
    exists for, where a viewer re-extracts different groups from the same bytes.

    The result is for :func:`decode_groups`. The numpy readers (:func:`margin`,
    :func:`deficit`, :func:`probabilities`) want the host form and say so rather than
    failing obscurely.
    """
    dev = torch.device(device)
    mv = (lambda a: None if a is None else torch.from_numpy(np.ascontiguousarray(a)).to(dev))
    return RankedCode(ranks=mv(code.ranks), support=mv(code.support), tail=mv(code.tail),
                      meta=dict(code.meta))


def _host(code: RankedCode, who: str) -> None:
    if not isinstance(code.support, np.ndarray):
        raise TypeError(
            f"{who}() reads the host arrays, but this code is resident on "
            f"{code.support.device}. Use decode_groups() for a device-resident code, or "
            f"keep the original alongside it - to_device() does not consume it.")


def decode_groups(code: RankedCode, groups, *, device=None,
                  quantize: bool = False) -> torch.Tensor:
    """Encoded planes -> one signed margin field per GROUP, on ``device``, in one pass.

    The inverse of :func:`encode`, and the operation a renderer or mesher actually wants.
    Decode on the device the consumer runs on: the stored planes are a few MB, the K-channel
    field they stand for is gigabytes, so expanding here and uploading would move the large
    thing to avoid moving the small one.

    ``groups`` is a sequence of sequences of channel indices; one structure is a group of
    one. Grouping is a decode-time choice - the store keeps every class - so the same bytes
    answer "118 classes" and "ten groups" with no re-encoding.

    Why one pass suffices: a class at rank ``j`` sits at level ``-gap_j`` relative to the
    winner (rank 0 is level 0 by construction), and gaps only grow with ``j``, so walking the
    N rank planes while keeping ``max`` over members and ``max`` over non-members yields
    ``m_S = d_S - d_notS`` exactly. That is the union's own margin, not ``max`` of the
    members' margins - which gets the sign right but underestimates the magnitude, and would
    put a surface at every internal boundary the group is supposed to dissolve.

    Both running maxima start at ``-clip``, the absent floor. Rank 0 is never the sentinel,
    so one of the two always reaches level 0 and the difference is never a spurious zero.

    ``quantize`` returns uint8 with 128 on the boundary (the convention
    :func:`encode_regions` stores), which is what a GPU texture wants - ten groups at 1 mm
    is 1.5 GB as float32 against 386 MB as uint8.

    Memory is two ``(G, Z, Y, X)`` fp32 accumulators; the second is consumed in place. The
    docs sketch this as ``decode(groups, spacing)`` - resampling is deliberately not here,
    since that belongs to the restore and would bake a grid into a decode.
    """
    clip = float(code.meta["clip"])
    K = int(code.meta["classes"])
    shape = tuple(int(v) for v in code.meta["shape"])
    resident = isinstance(code.support, torch.Tensor)
    if device is not None:
        dev = torch.device(device)
    else:                       # an already-resident code decodes where it lives
        dev = code.support.device if resident else torch.device("cpu")
    groups = [[int(c) for c in g] for g in groups]
    G = len(groups)

    def plane(a):               # upload only if it is not already here
        if isinstance(a, torch.Tensor):
            return a if a.device == dev else a.to(dev)
        return torch.from_numpy(np.ascontiguousarray(a)).to(dev)

    if code.meta.get("mode") == "regions":
        # independent Bernoullis: no competitor set, so a union is just max over members
        sup = plane(code.support)
        out = torch.full((G,) + shape, -clip, dtype=torch.float32, device=dev)
        for g, members in enumerate(groups):
            for c in members:
                q = sup[c].to(torch.float32)
                m = (q - ZERO_LEVEL) / (SUPPORT_MAX - ZERO_LEVEL) * clip
                m = torch.where(q == 0, torch.full_like(m, -clip), m)
                torch.maximum(out[g], m, out=out[g])
        return _quantize_margin(out, clip) if quantize else out

    ranks = plane(code.ranks)
    sup = plane(code.support)
    memb = torch.zeros((G, K + 1), dtype=torch.bool, device=dev)   # indexed BY RANK VALUE
    for g, members in enumerate(groups):
        for c in members:
            memb[g, c + 1] = True                                  # 0 stays False: absent

    d_in = torch.full((G,) + shape, -clip, dtype=torch.float32, device=dev)
    d_out = torch.full((G,) + shape, -clip, dtype=torch.float32, device=dev)
    floor = torch.tensor(-clip, dtype=torch.float32, device=dev)
    for j in range(ranks.shape[0]):
        rj = ranks[j].long()
        present = rj != 0
        if j == 0:
            level = torch.zeros(shape, dtype=torch.float32, device=dev)
        else:
            level = -(1.0 - sup[j - 1].to(torch.float32) / SUPPORT_MAX) * clip
        mine = memb[:, rj]                                         # (G, Z, Y, X)
        torch.maximum(d_in, torch.where(mine & present, level, floor), out=d_in)
        torch.maximum(d_out, torch.where(~mine & present, level, floor), out=d_out)
        del rj, present, level, mine
    out = d_in.sub_(d_out)                                         # in place: reuse d_in
    return _quantize_margin(out, clip) if quantize else out


def _quantize_margin(m: torch.Tensor, clip: float) -> torch.Tensor:
    """Signed margin -> uint8 with 128 exactly on the boundary, 0 reserved as absent."""
    q = (m / clip).clamp(-1, 1) * (SUPPORT_MAX - ZERO_LEVEL) + ZERO_LEVEL
    return q.round().clamp(1, SUPPORT_MAX).to(torch.uint8)


def probabilities(code: RankedCode) -> tuple[np.ndarray, np.ndarray]:
    """``(class_ids, p)`` for the stored channels - the head-specific decode.

    Ranked: ``p_j = exp(-g_j) / Z`` with ``Z = Z_top / (1 - tail)``; exact when exhaustive.
    Regions: independent sigmoids, which do not normalize and must not be read as if they did.

    Where a rank holds the sentinel the class is absent: its id is ``-1`` and its probability
    is reported as 0. Both halves of that matter - ``-1`` would index the LAST class under
    ``np.take_along_axis``, silently attributing this voxel's mass to whatever sorts last,
    so the probability is zeroed here rather than left as ``exp(-clip)/Z`` for a class that
    is not there. Mask on ``ids >= 0`` before using the ids as indices.
    """
    _host(code, "probabilities")
    clip = float(code.meta["clip"])
    if code.meta.get("mode") == "regions":
        m = np.stack([margin(code, c) for c in range(code.meta["classes"])])
        ids = np.arange(code.meta["classes"], dtype=np.int64)
        ids = np.broadcast_to(ids[:, None, None, None], m.shape)
        return ids.copy(), 1.0 / (1.0 + np.exp(-m))
    gaps = np.concatenate([
        np.zeros((1, *code.support.shape[1:]), np.float32),
        _gap(code.support, clip)], axis=0)
    w = np.exp(-gaps)
    z_top = w.sum(axis=0)
    if code.tail is None:
        z = z_top
    else:
        z = z_top / np.clip(1.0 - code.tail.astype(np.float32) / TAIL_MAX, 1e-6, None)
    ids = code.ranks.astype(np.int64) - 1                     # sentinel 0 -> -1 (absent)
    p = w / z
    p[ids < 0] = 0.0
    return ids, p
