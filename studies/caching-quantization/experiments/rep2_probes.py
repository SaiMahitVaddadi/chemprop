"""EXP 2 -- what is linearly decodable from the frozen CheMeleon embedding?

Linear (ridge) probes from the 2048-d frozen embedding to 37 interpretable RDKit
targets, on a pooled corpus of all 11 cached MoleculeNet datasets (deduped SMILES).

Probe capacity is CONTROLLED and REPORTED, per the NLP probing methodology:
  mean        -- predict the train mean (R2=0 / AUROC=0.5 by construction)
  gauss2048   -- 2048 iid N(0,1) features: the pure-overfitting floor at this
                 probe capacity and this n
  randenc     -- a frozen RANDOM-weight CheMeleon of identical architecture. This
                 is the honest floor: random message-passing nets are known to be
                 non-trivial featurizers, so "better than randenc" is the claim
                 that pretraining added something.
  control     -- the SAME probe on label-permuted targets (Hewitt & Liang control
                 task). selectivity = R2(real) - R2(control).
  Z_pca64     -- the probe restricted to the top-64 PCA directions of Z: a
                 lower-capacity probe, to show decodability is not a capacity artifact.

Reported for context (the head-to-head lives in rep3): ECFP4 counts, RDKit 2D.
"""
import sys
import time

import numpy as np

import rep_lib as R

NSPLIT = int(sys.argv[1]) if len(sys.argv) > 1 else 5


def main():
    out = {"caveat": [
        "A linear probe measures LINEAR decodability only. A target with low R2 may "
        "still be present nonlinearly; we do not claim absence of information.",
        "Probe capacity is held fixed at 2048 features + ridge lam=100 for every "
        "representation. The gauss2048 and randenc rows are the controls that make "
        "'decodable' mean 'better than chance at this capacity'.",
        "Targets are RDKit descriptors, computed on the same SMILES the encoder saw. "
        "They are deterministic functions of the 2-D graph, so a perfect encoder "
        "could in principle reach R2=1 on all of them.",
        "RDKit 2D is NOT Mordred. mordred/descriptastorus are not installed in this "
        "environment, so 'descriptors' here means RDKit's native ~210-descriptor "
        "list. CheMeleon was pretrained on Mordred, so this understates the overlap.",
        "Pooled corpus mixes 11 datasets with very different chemistry; the pooled "
        "target variance is therefore larger than within any single dataset, which "
        "makes R2 look better than it would within-dataset.",
    ]}
    t0 = time.perf_counter()
    smis, Z, src = R.pooled()
    Tg, ok = R.rdkit_targets(smis)
    smis = [s for s, k in zip(smis, ok) if k]
    Z = Z[ok]
    Y = np.stack([Tg[n][ok] for n, _ in R.TARGETS], 1)
    names = [n for n, _ in R.TARGETS]
    kinds = dict(R.TARGETS)
    n = len(smis)
    print(f"pooled corpus n={n}, {Y.shape[1]} targets  ({time.perf_counter()-t0:.0f}s)")

    rng = np.random.default_rng(11)
    reps = {}
    reps["chemeleon"] = Z
    print("building controls ...")
    t = time.perf_counter()
    reps["randenc"] = R.rand_encoder_Z(smis, seed=0)
    print(f"  random-weight CheMeleon encoder: {time.perf_counter()-t:.0f}s")
    reps["gauss2048"] = rng.standard_normal((n, 2048)).astype(np.float32)
    t = time.perf_counter()
    reps["ecfp4_counts"] = R.ecfp4(smis)
    Xd, dnames = R.rdkit2d(smis)
    reps["rdkit2d"] = Xd
    print(f"  ecfp4 + rdkit2d ({Xd.shape[1]} desc): {time.perf_counter()-t:.0f}s")
    # low-capacity variant of the real representation
    mu, V, S = R.pca_basis(Z, 64)
    reps["chemeleon_pca64"] = ((Z.astype(np.float64) - mu) @ V.T).astype(np.float32)

    # permuted-label control task (same targets, shuffled rows)
    perm = rng.permutation(n)
    Yctl = Y[perm]

    splits = R.splits_k(n, NSPLIT, seed=7)
    res = {}
    for rn, X in reps.items():
        t = time.perf_counter()
        acc = {nm: [] for nm in names}
        accc = {nm: [] for nm in names}
        for tr, te in splits:
            P, ym = R.ridge_multi(X, Y, tr, te, R.LAM_PROBE)
            Pc, _ = R.ridge_multi(X, Yctl, tr, te, R.LAM_PROBE)
            for j, nm in enumerate(names):
                acc[nm].append(_score(kinds[nm], Y[te, j], P[:, j], Y[tr, j]))
                accc[nm].append(_score(kinds[nm], Yctl[te, j], Pc[:, j], Yctl[tr, j]))
        res[rn] = {nm: dict(kind=kinds[nm],
                            score=float(np.mean(acc[nm])), sd=float(np.std(acc[nm])),
                            control=float(np.mean(accc[nm])),
                            selectivity=float(np.mean(acc[nm]) - np.mean(accc[nm])))
                   for nm in names}
        print(f"  {rn:<18} {X.shape[1]:>5}d  {time.perf_counter()-t:.0f}s")
    out["n_mol"] = n
    out["n_splits"] = NSPLIT
    out["rdkit2d_n_desc"] = int(Xd.shape[1])
    out["probes"] = res

    # ------------- table -------------
    print(f"\n{'target':<22}{'kind':>5}{'CheMeleon':>11}{'sd':>7}{'ctrl':>7}"
          f"{'randenc':>9}{'gauss':>8}{'pca64':>8}{'ECFP4':>8}{'RDKit2D':>9}{'lift':>8}")
    rows = []
    for nm in names:
        a = res["chemeleon"][nm]
        rr = res["randenc"][nm]["score"]
        rows.append((nm, a["kind"], a["score"], a["sd"], a["control"], rr,
                     res["gauss2048"][nm]["score"], res["chemeleon_pca64"][nm]["score"],
                     res["ecfp4_counts"][nm]["score"], res["rdkit2d"][nm]["score"],
                     a["score"] - rr))
    for r in sorted(rows, key=lambda x: -x[2]):
        print(f"{r[0]:<22}{r[1]:>5}" + "".join(f"{v:>9.3f}" if i in (1,) else f"{v:>8.3f}"
              for i, v in enumerate(r[2:])))
    out["table"] = rows

    hard = [r for r in rows if r[1] == "reg" and r[2] < 0.7]
    print("\nleast linearly decodable regression targets (R2 < 0.70):")
    for r in sorted(hard, key=lambda x: x[2]):
        print(f"  {r[0]:<22} R2={r[2]:.3f} (randenc {r[5]:.3f}, ECFP4 {r[8]:.3f}, "
              f"RDKit2D {r[9]:.3f})")
    R.jdump(out, f"{R.RESULTS}/results_rep2_probes.json")


def _score(kind, yte, p, ytr):
    if kind == "reg":
        v = yte.var()
        if v <= 0:
            return np.nan
        return float(1 - ((p - yte) ** 2).mean() / v)
    from sklearn.metrics import roc_auc_score
    if len(np.unique(yte)) < 2:
        return np.nan
    return float(roc_auc_score(yte, p))


main()
