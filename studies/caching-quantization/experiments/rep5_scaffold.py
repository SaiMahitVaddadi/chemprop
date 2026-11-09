"""EXP 5 -- is the Bemis-Murcko scaffold linearly decodable from the embedding?

If it is, that is a partial mechanical explanation for why scaffold splits roughly
double RMSE in the companion study: the encoder is in part a scaffold detector, so
a held-out scaffold is literally out of the encoder's trained-on region of space.

Two metrics, both with controls:
  * multi-class scaffold identity: linear (one-vs-rest ridge) probe over every
    Bemis-Murcko scaffold with >= MIN_MEMBERS molecules in the pooled corpus,
    scored as top-1 accuracy against the majority-class baseline;
  * 1-NN scaffold agreement: for each test molecule, is the cosine-nearest train
    molecule in the same scaffold class? This needs no fitted probe at all, so it
    cannot be a probe-capacity artifact.

Controls: random-weight CheMeleon of the same architecture, ECFP4 counts (the
representation everyone already agrees is substructure-based), RDKit 2D, and a
label-permuted control task.
"""
import sys
import time

import numpy as np

import rep_lib as R

MIN_MEMBERS = 15
NSPLIT = 5


def main():
    out = {"caveat": [
        "Scaffold classes with >= 15 members are, by construction, the common "
        "scaffolds of MoleculeNet. Decoding them is an easier task than decoding "
        "scaffold identity on the long tail (most scaffolds are singletons).",
        "Scaffold identity is strongly correlated with gross molecular features "
        "(ring count, size, aromaticity), so high scaffold accuracy does not prove "
        "the encoder stores a scaffold token -- it may be inferring scaffold from "
        "those features. The ECFP4 column is the reference for 'a representation "
        "that is explicitly substructure-based'.",
        "The split is by MOLECULE, not by scaffold: every scaffold class appears in "
        "both train and test. This measures whether the representation separates "
        "known scaffolds, not whether it generalises to unseen ones.",
        "1-NN agreement uses cosine distance on the raw representation, unstandardised.",
    ]}
    from rdkit import Chem, RDLogger
    from rdkit.Chem.Scaffolds import MurckoScaffold
    RDLogger.DisableLog("rdApp.*")

    smis, Z, src = R.pooled()
    Tg, ok = R.cached_obj("pool_targets", lambda: R.rdkit_targets(smis))
    smis = [s for s, k in zip(smis, ok) if k]
    Z = Z[ok]
    nheavy = Tg["heavy_atoms"][ok]

    scaf = []
    for s in smis:
        try:
            scaf.append(MurckoScaffold.MurckoScaffoldSmiles(smiles=s, includeChirality=False))
        except Exception:
            scaf.append("")
    uq, cnt = np.unique(scaf, return_counts=True)
    keep = set(uq[cnt >= MIN_MEMBERS].tolist()) - {""}
    sel = np.array([s in keep for s in scaf])
    classes = sorted(keep)
    cmap = {c: i for i, c in enumerate(classes)}
    lab = np.array([cmap[s] for s, k in zip(scaf, sel) if k])
    print(f"pooled n={len(smis)}; {len(uq)} distinct Bemis-Murcko scaffolds, "
          f"{len(classes)} with >={MIN_MEMBERS} members covering {sel.sum()} molecules "
          f"({sel.mean()*100:.1f}%)")
    out["n_mol_pooled"] = int(len(smis))
    out["n_scaffolds_total"] = int(len(uq))
    out["n_scaffold_classes"] = len(classes)
    out["n_mol_in_classes"] = int(sel.sum())
    out["min_members"] = MIN_MEMBERS
    out["singleton_scaffold_frac"] = float((cnt == 1).mean())

    reps = {
        "chemeleon": Z[sel],
        "chemeleon_sumagg": (Z.astype(np.float64) * nheavy[:, None])[sel],
        "randenc": R.cached("pool_randenc0", lambda: R.rand_encoder_Z(smis, seed=0))[sel],
        "ecfp4_counts": R.cached("pool_ecfp4", lambda: R.ecfp4(smis))[sel],
        "rdkit2d": R.cached("pool_rdkit2d", lambda: R.rdkit2d(smis)[0])[sel],
        "gauss2048": np.random.default_rng(5).standard_normal((int(sel.sum()), 2048)),
    }
    nsel = int(sel.sum())
    T = len(classes)
    Yoh = np.zeros((nsel, T))
    Yoh[np.arange(nsel), lab] = 1.0
    rngp = np.random.default_rng(13)
    labc = lab[rngp.permutation(nsel)]
    Yohc = np.zeros((nsel, T))
    Yohc[np.arange(nsel), labc] = 1.0

    splits = R.splits_k(nsel, NSPLIT, seed=7)
    maj = float(np.mean([(lab[te] == np.bincount(lab[tr], minlength=T).argmax()).mean()
                         for tr, te in splits]))
    print(f"majority-class baseline top-1 = {maj:.4f}  (chance 1/{T} = {1/T:.4f})")
    out["majority_baseline"] = maj
    out["uniform_chance"] = 1.0 / T

    res = {}
    print(f"\n{'representation':<20}{'top1':>8}{'sd':>7}{'ctrl_top1':>11}"
          f"{'1NN_agree':>11}{'1NN_ctrl':>10}")
    for rn, X in reps.items():
        t = time.perf_counter()
        a1, a1c, nn, nnc = [], [], [], []
        Xn = X.astype(np.float64)
        Xn = Xn / np.maximum(np.linalg.norm(Xn, axis=1, keepdims=True), 1e-12)
        for tr, te in splits:
            P, _ = R.ridge_multi(X, Yoh, tr, te, R.LAM_PROBE)
            a1.append(float((P.argmax(1) == lab[te]).mean()))
            Pc, _ = R.ridge_multi(X, Yohc, tr, te, R.LAM_PROBE)
            a1c.append(float((Pc.argmax(1) == labc[te]).mean()))
            S = Xn[te] @ Xn[tr].T
            nnl = lab[tr][S.argmax(1)]
            nn.append(float((nnl == lab[te]).mean()))
            nnc.append(float((labc[tr][S.argmax(1)] == labc[te]).mean()))
        res[rn] = dict(top1=float(np.mean(a1)), top1_sd=float(np.std(a1)),
                       top1_control=float(np.mean(a1c)),
                       nn1_agree=float(np.mean(nn)), nn1_control=float(np.mean(nnc)))
        print(f"{rn:<20}{res[rn]['top1']:>8.4f}{res[rn]['top1_sd']:>7.4f}"
              f"{res[rn]['top1_control']:>11.4f}{res[rn]['nn1_agree']:>11.4f}"
              f"{res[rn]['nn1_control']:>10.4f}   ({time.perf_counter()-t:.0f}s)")
    out["scaffold_identity"] = res

    # --------- depth sweep, if rep4's encodings exist --------------------------- #
    import os
    dres = {}
    for d in [1, 2, 3, 4, 5, 6]:
        p = f"{R.SCRATCH}/pool_Z_depth{d}.npy"
        if not os.path.exists(p):
            continue
        Zd = np.load(p)[sel]
        a1, nn = [], []
        Xn = Zd.astype(np.float64)
        Xn = Xn / np.maximum(np.linalg.norm(Xn, axis=1, keepdims=True), 1e-12)
        for tr, te in splits:
            P, _ = R.ridge_multi(Zd, Yoh, tr, te, R.LAM_PROBE)
            a1.append(float((P.argmax(1) == lab[te]).mean()))
            nn.append(float((lab[tr][(Xn[te] @ Xn[tr].T).argmax(1)] == lab[te]).mean()))
        dres[d] = dict(top1=float(np.mean(a1)), nn1_agree=float(np.mean(nn)))
    if dres:
        print("\nscaffold decodability vs message-passing depth:")
        for d, v in dres.items():
            print(f"  depth {d}  top1={v['top1']:.4f}  1NN={v['nn1_agree']:.4f}")
        out["by_depth"] = dres
    R.jdump(out, f"{R.RESULTS}/results_rep5_scaffold.json")


main()
