"""Compare two tasks that segment the same structures, on one image.

The validation tier that was missing. `total` (five specialized models at 1.5 mm) and
`total_fast` (one model at 3 mm) label the same 117 structures, so their per-structure
volumes should agree to within the resolution difference - and `total`, being finer and
specialized, should not be systematically *worse* anywhere.

That is exactly what a normalization-sharing bug breaks, and it needs no external reference
implementation: the two tasks check each other. When parts 2..N of `total` ran on the organs
model's statistics, its rib family came in at a third of what `total_fast` found, which this
would have reported as a lopsided ratio on a whole family of structures.

    uv run --no-project python tools/cross_task_volumes.py IMAGE [--a total_fast] [--b total]

Calibration, CT_Abdo, fold 0 (2026-08-28). Healthy agreement is tight: median ratio 1.02
over 87 structures, worst 0.76 (common_carotid_artery_right, a small vessel where 3 mm and
1.5 mm legitimately disagree). Against the normalization bug it was not subtle:

    structure            total_fast   total (bug)   ratio    total (fixed)   ratio
    ribs (all 24)             378.8         273.0    0.72            412.9    1.09
    sternum                    56.7          12.1    0.21             59.6    1.05
    costal_cartilages          92.1           5.8    0.06            128.6    1.40

So --flag-below 0.5 catches sternum and costal cartilage outright, with no false positive
against the 0.76 floor. Ribs at 0.72 needs a looser threshold - but the real signal is that a
whole anatomical family moved the same way at once, which no resolution difference explains.
(Costal cartilage at 1.40 after the fix is the expected direction: 1.5 mm resolves a thin
structure that 3 mm undercounts.)
"""
import argparse
import sys
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--a", default="total_fast", help="reference task")
    ap.add_argument("--b", default="total", help="task under test")
    ap.add_argument("--folds", type=int, nargs="+", default=[0])
    ap.add_argument("--min-ml", type=float, default=1.0,
                    help="ignore structures smaller than this in the reference; ratios of tiny "
                         "volumes are noise, not signal")
    ap.add_argument("--flag-below", type=float, default=0.5,
                    help="report structures where b/a falls under this")
    args = ap.parse_args()

    import nnseg

    vols = {}
    for tag, task in (("a", args.a), ("b", args.b)):
        t = time.perf_counter()
        seg = nnseg.segment(args.image, task, folds=tuple(args.folds))
        vols[tag] = seg.volumes_ml()
        print(f"  {task:12} {time.perf_counter() - t:6.1f} s   "
              f"{sum(1 for v in vols[tag].values() if v > 0)} structures present", flush=True)

    a, b = vols["a"], vols["b"]
    shared = [n for n in a if n in b and a[n] >= args.min_ml]
    ratios = sorted(((b[n] / a[n], n) for n in shared))
    print(f"\n  {len(shared)} structures at >= {args.min_ml} mL in {args.a}")
    print(f"  ratio {args.b}/{args.a}: median {ratios[len(ratios) // 2][0]:.2f}, "
          f"worst {ratios[0][0]:.2f} ({ratios[0][1]})")

    low = [(r, n) for r, n in ratios if r < args.flag_below]
    if low:
        print(f"\n  {len(low)} structure(s) under {args.flag_below}x - {args.b} finds far less "
              f"than {args.a} does, which for a finer task wants an explanation:")
        for r, n in low[:25]:
            print(f"    {n:34} {a[n]:8.1f} -> {b[n]:8.1f} mL   {r:.2f}x")
    else:
        print(f"\n  nothing under {args.flag_below}x")
    return 1 if low else 0


if __name__ == "__main__":
    sys.exit(main())
