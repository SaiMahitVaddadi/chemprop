"""EXP 3 -- is CheMeleon just re-deriving classical descriptors?

Three independent angles, all on the same 11 cached MoleculeNet datasets:
  (a) representational similarity: linear CKA + SVCCA between the frozen CheMeleon
      embedding and ECFP4 counts / ECFP4 bits / RDKit 2D descriptors, with a
      random-weight CheMeleon as the reference point for "architecture only";
  (b) downstream head-to-head with an IDENTICAL ridge protocol (dual form, verified
      against qlib.Head to 1e-5), lambda chosen ONCE per representation by the same
      mean-rank procedure scripts/prep.py used for CheMeleon;
  (c) concatenation: does CheMeleon + descriptors beat either alone?

Honest framing: mordred and descriptastorus are NOT installed, so "descriptors"
means RDKit's own ~213 usable 2-D descriptors, not the ~1600 Mordred features
CheMeleon was pretrained on and fastprop uses. This test is therefore WEAKER than
the real fastprop comparison and we say so.
"""
import itertools
import json
import os
import sys
import time

import numpy as np

import rep_lib as R

# Run in stages so each foreground invocation stays well under 10 minutes:
#   python rep3_descriptor_equiv.py sim
#   python rep3_descriptor_equiv.py scan
#   python rep3_descriptor_equiv.py final esol,freesolv,...
#   python rep3_descriptor_equiv.py report
STAGE = sys.argv[1] if len(sys.argv) > 1 else "all"
PART = f"{R.SCRATCH}/rep3"

LAMS = [10, 30, 100, 300, 1000, 3000]
NSPLIT = 10
CLIP = 5.0


def _agg(v, Y, te, tt):
    """Aggregate per-task metrics into one number.

    Regression tasks are normalised by the test target std before averaging,
    because QM9 carries 12 tasks in different physical units and a plain mean of
    raw RMSEs would just be the largest-unit task. For single-task datasets this
    is a constant rescaling and leaves every paired delta unchanged.
    """
    if tt != "reg":
        return float(np.nanmean(v))
    sd = np.nanstd(Y[te], 0)
    return float(np.nanmean(v / np.maximum(sd, 1e-9)))
# lambda is scanned only on the single-task (NaN-free) datasets: the scan needs
# 6 lambdas x 8 representations x 2 splits and the NaN-masked multi-task solves
# would dominate it for no benefit. The chosen lambda is then applied everywhere.
SCAN_DS = ["esol", "freesolv", "lipo", "bace_reg", "qm9", "bace_cls", "bbbp", "hiv"]


def _zn(X):
    """z-score a block so concatenation does not let one block dominate the ridge."""
    X = X.astype(np.float64)
    mu, sd = X.mean(0), X.std(0)
    return (X - mu) / np.where(sd > 1e-9, sd, np.inf)


def save(key, obj):
    os.makedirs(PART, exist_ok=True)
    R.jdump(obj, f"{PART}/{key}.json")


def part(key):
    with open(f"{PART}/{key}.json") as fh:
        return json.load(fh)


CAVEATS = [
        "RDKit 2D (213 usable descriptors) is NOT Mordred (~1600). mordred and "
        "descriptastorus are absent from this environment. CheMeleon was pretrained "
        "to predict Mordred descriptors, so the CKA and the head-to-head here both "
        "UNDERSTATE how descriptor-like CheMeleon is.",
        "CKA is a second-moment similarity. Two representations can have low CKA and "
        "still support identical downstream predictions, and vice versa; (a) and (b) "
        "are complementary, not redundant.",
        "lambda is scanned once per representation on 2 random splits of the 8 "
        "single-task datasets, then frozen. No per-dataset tuning.",
        "The companion study established a ~2.3% paired delta as the split-noise "
        "floor on these datasets. Differences smaller than that are not claims.",
        "A ridge head on a frozen representation is NOT a trained Chemprop. These "
        "numbers say which representation is more linearly useful, not which model "
        "wins after end-to-end training.",
        "SVCCA is computed on a 6000-molecule subsample for tractability.",
]


def stage_sim():

    # ---------------- (a) representational similarity, pooled corpus ----------- #
    smis, Z, src = R.pooled()
    Tg, ok = R.cached_obj("pool_targets", lambda: R.rdkit_targets(smis))
    smis = [s for s, k in zip(smis, ok) if k]
    Z = Z[ok]
    nheavy = Tg["heavy_atoms"][ok]
    P = {
        "chemeleon": Z,
        "chemeleon_sumagg": (Z.astype(np.float64) * nheavy[:, None]).astype(np.float32),
        "randenc": R.cached("pool_randenc0", lambda: R.rand_encoder_Z(smis, seed=0)),
        "ecfp4_counts": R.cached("pool_ecfp4", lambda: R.ecfp4(smis)),
        "ecfp4_bits": R.cached("pool_ecfp4b", lambda: R.ecfp4(smis, counts=False)),
        "rdkit2d": R.cached("pool_rdkit2d", lambda: R.rdkit2d(smis)[0]),
    }
    assert all(v.shape[0] == len(smis) for v in P.values())
    print(f"pooled n={len(smis)}; representations: " +
          ", ".join(f"{k}({v.shape[1]}d)" for k, v in P.items()))
    keys = list(P)
    # Raw CKA is dominated by whichever representation has the largest-variance
    # columns (RDKit 2D collapses to 2 PCA dims at 99% variance because BertzCT and
    # MolWt dwarf everything else), so the z-scored variant is the primary number.
    Pz = {k: _zn(v) for k, v in P.items()}
    # size-free variant: regress log(n_heavy), n_heavy, n_heavy^2 out of BOTH views.
    Sz = np.stack([np.log(nheavy), nheavy, nheavy ** 2.0], 1)
    Sz = np.concatenate([(Sz - Sz.mean(0)) / Sz.std(0), np.ones((len(Sz), 1))], 1)
    Q = np.linalg.qr(Sz)[0]

    def resid(X):
        X = np.nan_to_num(X)
        return X - Q @ (Q.T @ X)
    Pr = {k: resid(v) for k, v in Pz.items()}

    rng = np.random.default_rng(3)
    sub = rng.choice(len(smis), min(6000, len(smis)), replace=False)
    red = {}
    for k in keys:
        red[k] = R.pca_reduce(Pz[k][sub], var=0.99, kmax=256)
        print(f"  pca99(zscored {k}) -> {red[k].shape[1]}d")
    cka, cka_std, cka_sf, cca = {}, {}, {}, {}
    print(f"\n{'pair':<40}{'CKA_raw':>9}{'CKA_z':>8}{'CKA_sizefree':>14}{'SVCCA64':>9}")
    for a, b in itertools.combinations(keys, 2):
        k = f"{a}|{b}"
        cka[k] = R.linear_cka(P[a], P[b])
        cka_std[k] = R.linear_cka(Pz[a], Pz[b])
        cka_sf[k] = R.linear_cka(Pr[a], Pr[b])
        cca[k] = R.svcca_from_reduced(red[a], red[b], k=64)
        print(f"{a+' ~ '+b:<40}{cka[k]:>9.3f}{cka_std[k]:>8.3f}{cka_sf[k]:>14.3f}{cca[k]:>9.3f}")
    save("sim", dict(linear_cka_raw=cka, linear_cka_zscored=cka_std,
                     linear_cka_sizefree=cka_sf, svcca64_zscored=cca,
                     pca99_dims_zscored={k: int(v.shape[1]) for k, v in red.items()},
                     dims={k: int(v.shape[1]) for k, v in P.items()},
                     n_mol=len(smis), n_svcca_sub=len(sub),
                     note="CKA_z is the primary number; CKA_sizefree removes "
                          "log(n_heavy), n_heavy, n_heavy^2 from both views, because "
                          "molecular size is a shared dominant direction that would "
                          "otherwise make every pair of representations look similar."))


# ---------------- (b)+(c) downstream head-to-head -------------------------- #
def build_ds(only=None):
    print("building per-dataset representations ...")
    DS = {}
    for nm in (only or R.DATASETS):
        D = R.load(nm)
        ss = D["smiles"]
        e = R.cached(f"{nm}_ecfp4", lambda ss=ss: R.ecfp4(ss))
        d = R.cached(f"{nm}_rdkit2d", lambda ss=ss: R.rdkit2d(ss)[0])
        rr = R.cached(f"{nm}_randenc", lambda ss=ss: R.rand_encoder_Z(ss, seed=0))
        z = D["Z"].astype(np.float64)
        DS[nm] = (D, {
            "chemeleon": z,
            "randenc": rr,
            "ecfp4_counts": e,
            "rdkit2d": d,
            "chemeleon+ecfp4": np.concatenate([_zn(z), _zn(e)], 1),
            "chemeleon+rdkit2d": np.concatenate([_zn(z), _zn(d)], 1),
            "ecfp4+rdkit2d": np.concatenate([_zn(e), _zn(d)], 1),
            "chemeleon+ecfp4+rdkit2d": np.concatenate([_zn(z), _zn(e), _zn(d)], 1),
        })
        print(f"  {nm:<10} dims " + " ".join(f"{k.split('+')[0][:4]}{'+' * k.count('+')}"
                                             f":{v.shape[1]}" for k, v in DS[nm][1].items()))
    return DS


def stage_scan():
    DS = build_ds(SCAN_DS)
    RN = list(DS[SCAN_DS[0]][1])
    print(f"\n--- lambda scan on {len(SCAN_DS)} single-task datasets, 2 random splits ---")
    t0 = time.perf_counter()
    raw = {rn: [] for rn in RN}
    for nm in SCAN_DS:
        D, reps = DS[nm]
        tt = D["task_type"]
        sp = [s for s in D["splits"] if s[0].startswith("rand")][:2]
        sd = float(np.nanstd(D["Y"]))
        for rn in RN:
            vv = []
            for sn, tr, te in sp:
                dr = R.DualRidge(reps[rn], tr, te, clip=CLIP)
                vv.append([_agg(R.score_split(dr.predict(D["Y"], tr, float(lam)),
                                              D["Y"], te, tt), D["Y"], te, tt)
                           for lam in LAMS])
            v = np.mean(vv, 0)
            raw[rn].append(-v if tt == "reg" else v)
    chosen, scan = {}, {}
    for rn in RN:
        arr = np.array(raw[rn]).T                  # [n_lam, n_ds], higher better
        rk = (-arr).argsort(0).argsort(0) + 1
        chosen[rn] = LAMS[int(rk.mean(1).argmin())]
        scan[rn] = dict(grid=LAMS, mean_rank=rk.mean(1).tolist(), chosen=chosen[rn])
        print(f"  {rn:<26} mean_rank=" + " ".join(f"{v:4.1f}" for v in rk.mean(1)) +
              f"  -> lam={chosen[rn]}")
    print(f"  ({time.perf_counter()-t0:.0f}s)")
    save("scan", dict(lambda_scan=scan, chosen_lambda=chosen))


def stage_final(only):
    chosen = part("scan")["chosen_lambda"]
    DS = build_ds(only)
    RN = list(DS[only[0]][1])
    print(f"\n--- head-to-head, {NSPLIT} cached splits (6 random + 4 scaffold) ---")
    hh = {}
    for nm in only:
        D, reps = DS[nm]
        tt = D["task_type"]
        sp = D["splits"][:NSPLIT]
        t0 = time.perf_counter()
        per = {rn: [] for rn in RN}
        raw_task = {rn: [] for rn in RN}
        for sn, tr, te in sp:
            for rn in RN:
                dr = R.DualRidge(reps[rn], tr, te, clip=CLIP)
                v = R.score_split(dr.predict(D["Y"], tr, float(chosen[rn])), D["Y"], te, tt)
                raw_task[rn].append(v.tolist())
                per[rn].append(_agg(v, D["Y"], te, tt))
        per = {k: np.array(v) for k, v in per.items()}
        base = per["chemeleon"]
        isc = np.array([s[0].startswith("scaf") for s in sp])
        rec = {}
        for rn in RN:
            d = (per[rn] - base) / base * 100
            if tt != "reg":
                d = -d
            rec[rn] = dict(mean=float(np.nanmean(per[rn])), sd=float(np.nanstd(per[rn])),
                           n_valid_splits=int(np.isfinite(per[rn]).sum()),
                           mean_random=float(np.nanmean(per[rn][~isc])),
                           mean_scaffold=float(np.nanmean(per[rn][isc])),
                           delta_vs_chemeleon_pct=float(np.nanmean(d)),
                           delta_sd=float(np.nanstd(d)), per_split=per[rn].tolist(),
                           per_task_per_split=raw_task[rn],
                           split_names=[s[0] for s in sp])
        hh[nm] = dict(task_type=tt,
                      metric="mean normalised RMSE (RMSE / test target std, averaged "
                             "over tasks)" if tt == "reg" else "mean AUROC over tasks",
                      reps=rec)
        print(f"{nm:<10}{'nRMSE' if tt=='reg' else 'AUROC':>6} " +
              " ".join(f"{_ab(rn)}={rec[rn]['mean']:.4f}" for rn in RN) +
              f"  ({time.perf_counter()-t0:.0f}s)")
        save(f"final_{nm}", hh[nm])


def stage_report():
    out = {"caveat": CAVEATS}
    out.update(part("sim") and {"similarity": part("sim")})
    out.update(part("scan"))
    hh = {nm: part(f"final_{nm}") for nm in R.DATASETS}
    out["head_to_head"] = hh
    RN = list(hh[R.DATASETS[0]]["reps"])
    print(f"\n{'representation':<26}{'mean delta vs CheMeleon':>25}{'median':>9}"
          f"{'n_worse':>9}{'n_better':>10}")
    summ = {}
    for rn in RN:
        ds = [hh[nm]["reps"][rn]["delta_vs_chemeleon_pct"] for nm in R.DATASETS]
        summ[rn] = dict(mean=float(np.nanmean(ds)), median=float(np.nanmedian(ds)),
                        per_dataset=dict(zip(R.DATASETS, ds)))
        print(f"{rn:<26}{np.nanmean(ds):>+24.2f}%{np.nanmedian(ds):>+9.2f}"
              f"{sum(1 for d in ds if d > 2.3):>9}{sum(1 for d in ds if d < -2.3):>10}")
    out["summary_delta_vs_chemeleon"] = summ
    print("  (positive delta = WORSE than CheMeleon; counts use the 2.3% noise band)")
    R.jdump(out, f"{R.RESULTS}/results_rep3_descriptor_equiv.json")


def _ab(rn):
    if "+" not in rn:
        return rn[:4]
    return "+".join(p[:2] for p in rn.split("+"))


if STAGE == "sim":
    stage_sim()
elif STAGE == "scan":
    stage_scan()
elif STAGE == "final":
    stage_final(sys.argv[2].split(","))
elif STAGE == "report":
    stage_report()
else:
    stage_sim()
    stage_scan()
    stage_final(R.DATASETS)
    stage_report()
