"""Stage 1: load every dataset, featurize, encode with the fp32 encoder, cache to
disk, build the split list, and scan the ridge penalty.

Caching the fp32 embedding is the key efficiency trick: the encoder output does
not depend on the split, so every split/head/metric downstream is nearly free.
"""
import json
import os
import pickle
import time

import numpy as np
import torch

import qlib

OUT = os.path.join(qlib.SCRATCH, "cache")
os.makedirs(OUT, exist_ok=True)

DATASETS = [
    ("esol", 2200), ("freesolv", 2200), ("lipo", 2200), ("bace_reg", 2200),
    ("qm9", 2200), ("bace_cls", 2200), ("bbbp", 2200), ("clintox", 2200),
    ("sider", 2200), ("tox21", 2200), ("hiv", 2200),
]
LAMBDAS = [10, 30, 100, 300, 1000, 3000]


def main():
    W_i, W_h, W_o, b_o, d_h, depth = qlib.load_chemeleon()
    meta = {}
    print(f"CheMeleon d_h={d_h} depth={depth}\n")
    hdr = f"{'dataset':<10}{'n':>6}{'T':>4}{'type':>5}{'atoms':>8}{'edges':>8}{'scaf':>6}{'enc_s':>7}"
    print(hdr)
    for name, cap in DATASETS:
        d = qlib.load_dataset(name, cap)
        smis, Y = d["smiles"], d["Y"]
        chunks = qlib.build_chunks(smis, 400)
        t = time.perf_counter()
        Z = qlib.encode(chunks, W_i, W_h, W_o, b_o, depth, d_h)
        enc = time.perf_counter() - t
        splits, n_scaf = qlib.make_splits(smis, n_random=6, n_scaffold=4)
        natoms = sum(c[0].shape[0] for c in chunks)
        nedges = sum(c[2].shape[1] for c in chunks)
        np.save(f"{OUT}/{name}_Z0.npy", Z.numpy())
        with open(f"{OUT}/{name}.pkl", "wb") as fh:
            pickle.dump(dict(smiles=smis, Y=Y, task_type=d["task_type"],
                             tasks=d["tasks"], desc=d["desc"], splits=splits), fh)
        meta[name] = dict(n=len(smis), T=Y.shape[1], task_type=d["task_type"],
                          desc=d["desc"], tasks=d["tasks"], n_atoms=int(natoms),
                          n_edges=int(nedges), atoms_per_mol=natoms / len(smis),
                          n_scaffolds=n_scaf, encode_s=enc,
                          nan_frac=float(np.isnan(Y).mean()))
        print(f"{name:<10}{len(smis):>6}{Y.shape[1]:>4}{d['task_type']:>5}"
              f"{natoms:>8}{nedges:>8}{n_scaf:>6}{enc:>7.1f}")
    with open(f"{OUT}/meta.json", "w") as fh:
        json.dump(meta, fh, indent=1)

    # ---- ridge penalty scan on fp32 embeddings (one shared lambda for all configs,
    #      chosen once so that the quantization comparison is never re-tuned) ----
    print(f"\n{'--- ridge lambda scan (fp32, mean primary metric over splits) ---'}")
    print(f"{'dataset':<10}" + "".join(f"{l:>9}" for l in LAMBDAS))
    score = {l: [] for l in LAMBDAS}
    for name, _ in DATASETS:
        with open(f"{OUT}/{name}.pkl", "rb") as fh:
            D = pickle.load(fh)
        Z = torch.from_numpy(np.load(f"{OUT}/{name}_Z0.npy"))
        Y, tt = D["Y"], D["task_type"]
        row = []
        for lam in LAMBDAS:
            vals = []
            for sn, tr, te in D["splits"][:4]:
                h = qlib.Head(Z, Y, tr, lam)
                m = qlib.eval_metrics(h.predict(Z, te), Y, te, tt)
                key = "rmse" if tt == "reg" else "auroc"
                # normalise regression RMSE by target std so datasets are comparable
                if tt == "reg":
                    sd = np.nanstd(Y[te], 0)
                    vals.append(np.nanmean(m[key] / np.maximum(sd, 1e-9)))
                else:
                    vals.append(np.nanmean(m[key]))
            v = float(np.mean(vals))
            row.append(v)
            # lower-is-better for reg (normalised rmse), higher for cls
            score[lam].append(-v if tt == "reg" else v)
        print(f"{name:<10}" + "".join(f"{v:>9.4f}" for v in row))
    print(f"\n{'mean rank score (higher=better)':<32}")
    ranks = {l: float(np.mean([np.mean(score[l][i]) for i in range(len(score[l]))]))
             for l in LAMBDAS}
    # choose lambda by average rank across datasets (robust to scale differences)
    arr = np.array([score[l] for l in LAMBDAS])            # [n_lam, n_ds]
    rk = (-arr).argsort(0).argsort(0) + 1                   # rank 1 = best
    for i, l in enumerate(LAMBDAS):
        print(f"  lam={l:<6} mean rank={rk[i].mean():.2f}  per-ds ranks={rk[i].tolist()}")
    best = LAMBDAS[int(rk.mean(1).argmin())]
    print(f"\nCHOSEN LAMBDA = {best}")
    with open(f"{OUT}/lambda.json", "w") as fh:
        json.dump(dict(lam=best, scan={str(l): score[l] for l in LAMBDAS}), fh, indent=1)


main()
