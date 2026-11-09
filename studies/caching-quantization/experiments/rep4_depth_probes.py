"""EXP 4 -- what information appears at which message-passing depth?

Re-encodes the pooled 13226-molecule corpus with the SAME pretrained CheMeleon
weights but the message-passing loop truncated to depth 1..6, then runs the exact
probe battery from rep2 at every depth.

This is depth truncation of a depth-6 checkpoint -- the same operation the
companion study measured as costing +15-30% downstream RMSE at depth 3. If that
cost is genuine information loss (rather than a readout mismatch) then some probe
targets must improve between depth 3 and depth 6. That is the test.

Stages (each well under 10 min):
  python rep4_depth_probes.py encode     # builds + caches Z at depths 1..6
  python rep4_depth_probes.py probe
"""
import sys
import time

import numpy as np

import rep_lib as R
sys.path.insert(0, R.SCRIPTS)
import qlib

DEPTHS = [1, 2, 3, 4, 5, 6]
NSPLIT = 5
STAGE = sys.argv[1] if len(sys.argv) > 1 else "all"


def pool_smis():
    smis, Z, src = R.pooled()
    Tg, ok = R.cached_obj("pool_targets", lambda: R.rdkit_targets(smis))
    return [s for s, k in zip(smis, ok) if k], Z[ok], Tg, ok


def stage_encode():
    smis, Z0, Tg, ok = pool_smis()
    Wi, Wh, Wo, bo, d_h, depth = qlib.load_chemeleon()
    assert depth == 6, depth
    chunks = R.cached_obj("pool_chunks_meta", lambda: {"n": len(smis)})
    print(f"encoding n={len(smis)} at depths {DEPTHS} (d_h={d_h})")
    ch = qlib.build_chunks(smis, 400)
    for d in DEPTHS:
        t = time.perf_counter()
        key = f"pool_Z_depth{d}"
        Z = R.cached(key, lambda d=d: qlib.encode(ch, Wi, Wh, Wo, bo, d, d_h).numpy())
        err = float(np.abs(Z - Z0).max()) if d == 6 else np.nan
        print(f"  depth {d}: {Z.shape} {time.perf_counter()-t:.0f}s"
              + (f"  max|Z6 - cached Z0| = {err:.2e} (should be ~0)" if d == 6 else ""))


def stage_probe():
    smis, Z0, Tg, ok = pool_smis()
    n = len(smis)
    Y = np.stack([Tg[nm][ok] for nm, _ in R.TARGETS], 1)
    names = [nm for nm, _ in R.TARGETS]
    kinds = dict(R.TARGETS)
    splits = R.splits_k(n, NSPLIT, seed=7)
    out = {"caveat": [
        "Truncating a depth-6 checkpoint is NOT the same as training a depth-d model. "
        "W_h is applied d-1 times instead of 5, so the hidden-state distribution the "
        "readout W_o sees is off-distribution at d<6. Some of the depth effect is "
        "therefore readout mismatch, not information.",
        "The probe is refitted at every depth, which removes the readout-mismatch "
        "confound for the PROBE (a fresh linear map is fitted to whatever the "
        "truncated encoder outputs) but not for the pretrained W_o inside the encoder.",
        "Linear decodability only; see rep2 caveats.",
        "Depth-6 is byte-identical to the cached Z0, which is asserted in the encode "
        "stage.",
    ], "n_mol": n, "n_splits": NSPLIT, "depths": DEPTHS}

    res = {}
    for d in DEPTHS:
        Z = np.load(f"{R.SCRATCH}/pool_Z_depth{d}.npy")
        t = time.perf_counter()
        acc = {nm: [] for nm in names}
        for tr, te in splits:
            P, _ = R.ridge_multi(Z, Y, tr, te, R.LAM_PROBE)
            for j, nm in enumerate(names):
                acc[nm].append(R.score_target(kinds[nm], Y[te, j], P[:, j])[0])
        res[d] = {nm: float(np.mean(acc[nm])) for nm in names}
        print(f"  depth {d} probed in {time.perf_counter()-t:.0f}s")
    out["per_depth"] = res

    print(f"\n{'target':<22}{'kind':>5}" + "".join(f"{'d'+str(d):>8}" for d in DEPTHS)
          + f"{'d6-d1':>8}{'d6-d3':>8}{'frac@d3':>9}")
    rows = []
    for nm in names:
        v = [res[d][nm] for d in DEPTHS]
        base = 0.5 if kinds[nm] == "cls" else 0.0
        span = v[-1] - base
        frac3 = (v[2] - base) / span if span > 1e-6 else np.nan
        rows.append(dict(target=nm, kind=kinds[nm], per_depth=v,
                         d6_minus_d1=v[-1] - v[0], d6_minus_d3=v[-1] - v[2],
                         frac_of_final_at_d3=frac3))
    for r in sorted(rows, key=lambda x: -x["d6_minus_d3"]):
        print(f"{r['target']:<22}{r['kind']:>5}" + "".join(f"{x:>8.3f}" for x in r["per_depth"])
              + f"{r['d6_minus_d1']:>+8.3f}{r['d6_minus_d3']:>+8.3f}"
              + (f"{r['frac_of_final_at_d3']:>9.2f}" if np.isfinite(r["frac_of_final_at_d3"]) else "      nan"))
    out["table"] = rows

    loc = [r for r in rows if r["target"] in R.LOCAL]
    glo = [r for r in rows if r["target"] in R.GLOBAL]
    fg = [r for r in rows if r["target"].startswith("fg_")]
    for lbl, grp in [("LOCAL", loc), ("GLOBAL/topological", glo), ("functional groups", fg)]:
        if not grp:
            continue
        f3 = np.nanmean([r["frac_of_final_at_d3"] for r in grp])
        d63 = np.nanmean([r["d6_minus_d3"] for r in grp])
        print(f"\n{lbl:<20} mean frac-of-final-at-depth-3 = {f3:.3f}   "
              f"mean d6-d3 gain = {d63:+.3f}  (n={len(grp)})")
        out.setdefault("groups", {})[lbl] = dict(frac_at_d3=float(f3),
                                                 gain_d3_to_d6=float(d63), n=len(grp))
    R.jdump(out, f"{R.RESULTS}/results_rep4_depth_probes.json")


if STAGE == "encode":
    stage_encode()
elif STAGE == "probe":
    stage_probe()
else:
    stage_encode()
    stage_probe()
