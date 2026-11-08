"""Readout mismatch vs information loss -- IDENTIFIABLE version.

WHY THE TWO EARLIER ATTEMPTS FAILED (both persisted, both reported)
The readout maps a 2120-dim pooled pre-readout vector A to the 2048-dim embedding.
Fitting it needs more molecules than 2120. ESOL has 902 train molecules and
FreeSolv 511, so BOTH earlier formulations were under-determined:
  1. results_wo_refit.csv   -- linear head fitted on A from scalar labels:
     catastrophic overfit, and worse than frozen even AT DEPTH 6 where the
     information is identical, which proves it was variance, not information.
  2. results_wo_distill.csv -- ridge from A_k to Z_6 on one dataset's train rows:
     at depth 6 the map Z_6 = W_o A_6 + b_o is EXACTLY linear, so recovery should
     be perfect; measured relative reconstruction error on test was 0.057 (esol,
     902 train rows) and 0.313 (freesolv, 511 rows). Underdetermined, as expected
     from n < d.

THE FIX (this script)
The distillation target is LABEL-FREE, so it can be fitted on far more molecules
than any single labelled dataset. We fit the readout on a POOLED set of molecules
drawn from the OTHER datasets (no molecule overlap with the evaluation sets), with
n_pool >> 2120, then evaluate task accuracy on ESOL and FreeSolv:

    W_o'  =  argmin || A_k(pool) W^T - Z_6(pool) ||^2 + lam||W||^2 ,  n_pool = 8000

    Zhat_k = A_k W_o'^T   ->   standard ridge head, same 10 splits.

Because the pool excludes the evaluation molecules entirely, there is no label
leakage and no transduction. The depth-6 run is the SANITY CHECK: there the map is
exactly linear, so reconstruction must be near-zero error and task RMSE must match
the true depth-6 model. If that check passes, the depth-3 number is trustworthy.

    Zhat_3 recovers depth-6 accuracy -> depth-3 states DO carry the depth-6
                                        information; the frozen-W_o gap was
                                        readout mismatch; the depth-3 case revives.
    Zhat_3 does not                  -> the information is genuinely absent from
                                        the first 3 rounds.

============================== CAVEAT -- READ FIRST ==============================
The CheMeleon weights were PRETRAINED AT DEPTH 6. This frees only the READOUT;
W_i and W_h stay the depth-6 weights and the distillation target is itself the
depth-6 embedding. So this measures whether the depth-6 REPRESENTATION is linearly
recoverable from depth-k states -- not whether a depth-k TRAINED encoder would
work. A from-scratch depth-k encoder could learn different and better features.
High recovery = strong evidence FOR shallow viability. Low recovery = evidence that
depth-6 features specifically are not reconstructible from k rounds, which is NOT
proof that no good depth-k representation exists.
==================================================================================
"""
import json
import os
import pickle
import sys
import time

import numpy as np
import torch

import qlib

CACHE = os.path.join(qlib.SCRATCH, "cache")
LAM_HEAD = json.load(open(f"{CACHE}/lambda.json"))["lam"]
NATIVE = 6
EVAL_DS = ["freesolv", "esol"]
POOL_DS = ["lipo", "bace_reg", "qm9", "bbbp", "clintox", "sider", "tox21", "hiv"]
N_POOL = 8000
DLAM = 10.0           # modest ridge; the system is now overdetermined (8000 >> 2121)
CAVEAT = __doc__
OUT = f"{qlib.SCRATCH}/results_wo_distill_pooled.json"


def encode_pre(chunks, Wi, Wh, Wo, bo, depth, d_h=2048):
    outs = []
    for V, E, ei, rev, bt, nm in chunks:
        n_v = V.shape[0]
        H_0 = torch.nn.functional.linear(torch.cat([V[ei[0]], E], 1), Wi)
        H = torch.relu(H_0)
        idx = ei[1].unsqueeze(1).expand(-1, d_h)
        for _ in range(1, depth):
            M = torch.zeros(n_v, d_h).scatter_reduce_(
                0, idx, H, reduce="sum", include_self=False)[ei[0]] - H[rev]
            H = torch.relu(H_0 + torch.nn.functional.linear(M, Wh))
        M = torch.zeros(n_v, d_h).scatter_reduce_(0, idx, H, reduce="sum",
                                                  include_self=False)
        pre = torch.cat((V, M), 1)
        cnt = torch.zeros(nm).index_add_(0, bt, torch.ones(bt.shape[0])).clamp(min=1)
        outs.append(torch.zeros(nm, pre.shape[1]).index_add_(0, bt, pre) / cnt.unsqueeze(1))
    return torch.cat(outs)


def main(depths):
    W_i, W_h, W_o, b_o, d_h, _ = qlib.load_chemeleon()
    print(CAVEAT)

    # ---- pooled molecule set, disjoint from the evaluation datasets ----
    eval_smiles = set()
    for n in EVAL_DS:
        with open(f"{CACHE}/{n}.pkl", "rb") as fh:
            eval_smiles |= set(pickle.load(fh)["smiles"])
    pool, seen = [], set()
    for n in POOL_DS:
        with open(f"{CACHE}/{n}.pkl", "rb") as fh:
            for s in pickle.load(fh)["smiles"]:
                if s not in eval_smiles and s not in seen:
                    seen.add(s)
                    pool.append(s)
    rng = np.random.default_rng(0)
    if len(pool) > N_POOL:
        pool = [pool[i] for i in np.sort(rng.choice(len(pool), N_POOL, replace=False))]
    print(f"pool: {len(pool)} molecules from {POOL_DS}, "
          f"disjoint from {EVAL_DS} (overlap removed), vs d=2120 -> "
          f"{len(pool)/2121:.1f}x overdetermined\n")
    pool_chunks = qlib.build_chunks(pool, 400)

    ev = {}
    for n in EVAL_DS:
        with open(f"{CACHE}/{n}.pkl", "rb") as fh:
            D = pickle.load(fh)
        ev[n] = dict(Y=D["Y"], tt=D["task_type"], splits=D["splits"],
                     chunks=qlib.build_chunks(D["smiles"], 400))

    recs = json.load(open(OUT))["records"] if os.path.exists(OUT) else []
    done = {(r["dataset"], r["depth"], r["split"]) for r in recs}

    for depth in depths:
        t0 = time.perf_counter()
        # fit the readout on the pool
        Ap = encode_pre(pool_chunks, W_i, W_h, W_o, b_o, depth, d_h)
        A6p = Ap if depth == NATIVE else encode_pre(pool_chunks, W_i, W_h, W_o, b_o,
                                                    NATIVE, d_h)
        Z6p = torch.nn.functional.linear(A6p, W_o, b_o)
        X = Ap.double()
        mu, sg = X.mean(0), X.std(0).clamp(min=1e-8)
        Xs = torch.cat([(X - mu) / sg,
                        torch.ones(X.shape[0], 1, dtype=torch.float64)], 1)
        ym = Z6p.double().mean(0)
        G = Xs.T @ Xs + DLAM * torch.eye(Xs.shape[1], dtype=torch.float64)
        Wd = torch.linalg.solve(G, Xs.T @ (Z6p.double() - ym))
        fit_rel = float((Xs @ Wd + ym - Z6p.double()).norm() / Z6p.double().norm())
        del Ap, A6p, Z6p, X, Xs, G
        print(f"depth={depth}: readout fitted on pool, in-pool rel recon error "
              f"{fit_rel:.4f}  ({time.perf_counter()-t0:.0f}s)", flush=True)

        for name in EVAL_DS:
            d = ev[name]
            A = encode_pre(d["chunks"], W_i, W_h, W_o, b_o, depth, d_h)
            A6 = A if depth == NATIVE else encode_pre(d["chunks"], W_i, W_h, W_o,
                                                      b_o, NATIVE, d_h)
            Z6 = torch.nn.functional.linear(A6, W_o, b_o)
            Zf = torch.nn.functional.linear(A, W_o, b_o)
            Xe = ((A.double() - mu) / sg)
            Xe = torch.cat([Xe, torch.ones(Xe.shape[0], 1, dtype=torch.float64)], 1)
            Zh = (Xe @ Wd + ym).float()
            for sn, tr, te in d["splits"]:
                if (name, depth, sn) in done:
                    continue
                rec = dict(dataset=name, depth=int(depth), split=sn,
                           split_type="scaffold" if sn.startswith("scaf") else "random",
                           pool_n=len(pool), pool_fit_rel=fit_rel,
                           recon_test=float((Zh[te] - Z6[te]).norm() / Z6[te].norm()))
                for tag, F in (("frozen", Zf), ("distill", Zh), ("ref6", Z6)):
                    h = qlib.Head(F, d["Y"], tr, LAM_HEAD)
                    m = qlib.eval_metrics(h.predict(F, te), d["Y"], te, d["tt"])["rmse"]
                    rec[f"{tag}_rmse"] = float(np.nanmean(m))
                recs.append(rec)
            g = [r for r in recs if r["dataset"] == name and r["depth"] == depth]
            print(f"  [{name}] depth={depth} frozen={np.mean([r['frozen_rmse'] for r in g]):.4f}"
                  f" distilled={np.mean([r['distill_rmse'] for r in g]):.4f}"
                  f" true_d6={np.mean([r['ref6_rmse'] for r in g]):.4f}"
                  f" recon_test={np.mean([r['recon_test'] for r in g]):.4f}", flush=True)
            del A, A6, Z6, Zf, Xe, Zh
        with open(OUT, "w") as fh:
            json.dump(dict(caveat=CAVEAT, head_lambda=LAM_HEAD, distill_lambda=DLAM,
                           pool_datasets=POOL_DS, pool_n=len(pool),
                           eval_datasets=EVAL_DS, native_depth=NATIVE,
                           identifiability=f"n_pool={len(pool)} vs d=2121 -> "
                                           f"{len(pool)/2121:.1f}x overdetermined; the two "
                                           f"earlier under-determined attempts are in "
                                           f"results_wo_refit.csv and results_wo_distill.csv",
                           records=recs), fh, indent=1)
        print(f"  persisted {len(recs)} records to {os.path.basename(OUT)}", flush=True)


if __name__ == "__main__":
    main([int(x) for x in sys.argv[1:]] or [6, 3])
