"""Readout mismatch vs genuine information loss -- the WELL-POSED version.

WHY THE OBVIOUS EXPERIMENT FAILED (see results_wo_refit.csv)
My first formulation refit the readout by fitting a linear head directly on the
pooled pre-readout features A (2120-dim). That is exactly the right FUNCTION
CLASS, but it is a hopeless ESTIMATOR here: n_train is 500-900 and d is 2121, and
A's columns are strongly correlated, so ridge-on-A overfits catastrophically even
with lambda selected by inner CV over 1e1..1e7. The diagnostic that proves it is
an artefact: AT DEPTH 6, where the frozen and refit readouts see *identical*
information, the refit is already far worse (freesolv 9.35 vs 1.56 RMSE, esol
0.869 vs 0.751). A difference at depth 6 cannot be information -- it is variance.
So that experiment cannot separate mismatch from information loss and is reported
only as a negative methodological result.

THE WELL-POSED VERSION (this script)
Fit the readout by DISTILLATION instead of from scarce scalar labels:

    W_o'  =  argmin  || A_k W^T  -  Z_6 ||^2  +  lam ||W||^2        (train rows only)

i.e. find the best linear readout of the depth-k states that reproduces the
DEPTH-6 EMBEDDING Z_6. This is well determined because the target has 2048
dimensions x n_train rows instead of one scalar per molecule, and it is exactly
the coordinator's first suggested option ("least squares on the truncated edge
states against the depth-6 W_o targets").

Then run the standard protocol on the distilled embedding Zhat_k = A_k W_o'^T:
ridge head fitted on train, evaluated on test, same 10 splits.

    Zhat_k recovers depth-6 accuracy  ->  the depth-k states DO carry the
                                          depth-6 information; the frozen-W_o gap
                                          was readout mismatch.
    Zhat_k does not                   ->  the information is genuinely absent
                                          from the first k rounds.

Sanity check built in: at k=6 the distillation target equals the input's own
readout, so Zhat_6 must reproduce Z_6 almost exactly. If it does not, the
distillation itself is underfitting and nothing else can be trusted.

Distillation lambda is chosen per (dataset, depth, split) by RECONSTRUCTION error
on an inner 80/20 split of the TRAINING rows -- label-free and never touches test.

============================== CAVEAT -- READ FIRST ==============================
The CheMeleon weights were PRETRAINED AT DEPTH 6. This frees only the READOUT;
W_i and W_h remain the depth-6 weights, and the distillation target is itself the
depth-6 embedding, so this measures "is the depth-6 representation linearly
recoverable from depth-k states", NOT "would a depth-k TRAINED encoder work".
A depth-k encoder trained from scratch could learn different, better features. So
a high recovery score is strong evidence FOR shallow viability, and a low score is
evidence that depth-6 features specifically are not reconstructible -- not proof
that no good depth-k representation exists.
==================================================================================
"""
import json
import os
import pickle

import numpy as np
import torch

import qlib

CACHE = os.path.join(qlib.SCRATCH, "cache")
LAM_HEAD = json.load(open(f"{CACHE}/lambda.json"))["lam"]
DEPTHS = [1, 2, 3, 4, 5, 6]
NATIVE = 6
DS = ["freesolv", "esol"]
DLAMS = [1e-2, 1e0, 1e2, 1e4]
CAVEAT = __doc__


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


def distill(A, Z6, tr, lams, inner_frac=0.8, seed=0):
    """Ridge map A -> Z6 fitted on `tr`; lambda by reconstruction error on an inner
    split of `tr`. Returns (Zhat for ALL rows, chosen lam, inner rel recon error)."""
    rng = np.random.default_rng(seed)
    p = rng.permutation(len(tr))
    k = int(inner_frac * len(tr))
    itr, iva = tr[p[:k]], tr[p[k:]]

    def fit(rows, lam):
        X = A[rows].double()
        mu, sg = X.mean(0), X.std(0).clamp(min=1e-8)
        Xs = torch.cat([(X - mu) / sg, torch.ones(len(rows), 1, dtype=torch.float64)], 1)
        Y = Z6[rows].double()
        ym = Y.mean(0)
        G = Xs.T @ Xs + lam * torch.eye(Xs.shape[1], dtype=torch.float64)
        W = torch.linalg.solve(G, Xs.T @ (Y - ym))
        return mu, sg, ym, W

    def apply(par, rows):
        mu, sg, ym, W = par
        X = ((A[rows].double() - mu) / sg)
        X = torch.cat([X, torch.ones(len(rows), 1, dtype=torch.float64)], 1)
        return X @ W + ym

    best, blam = None, lams[0]
    for lam in lams:
        par = fit(itr, lam)
        pred = apply(par, iva)
        err = float((pred - Z6[iva].double()).norm() / Z6[iva].double().norm())
        if best is None or err < best:
            best, blam = err, lam
    par = fit(tr, blam)
    allr = np.arange(A.shape[0])
    return apply(par, allr).float(), blam, best


def main():
    W_i, W_h, W_o, b_o, d_h, _ = qlib.load_chemeleon()
    recs = []
    print(CAVEAT)
    for name in DS:
        with open(f"{CACHE}/{name}.pkl", "rb") as fh:
            D = pickle.load(fh)
        Y, tt, splits = D["Y"], D["task_type"], D["splits"]
        chunks = qlib.build_chunks(D["smiles"], 400)
        A6 = encode_pre(chunks, W_i, W_h, W_o, b_o, NATIVE, d_h)
        Z6 = torch.nn.functional.linear(A6, W_o, b_o)
        for depth in DEPTHS:
            A = A6 if depth == NATIVE else encode_pre(chunks, W_i, W_h, W_o, b_o,
                                                      depth, d_h)
            Zf = torch.nn.functional.linear(A, W_o, b_o)      # frozen readout
            for si, (sn, tr, te) in enumerate(splits):
                Zh, dlam, recon = distill(A, Z6, tr, DLAMS, seed=si)
                # how well does the distilled embedding match Z6 on TEST molecules?
                rec_te = float((Zh[te] - Z6[te]).norm() / Z6[te].norm())
                out = dict(dataset=name, depth=depth, split=sn,
                           split_type="scaffold" if sn.startswith("scaf") else "random",
                           dlam=float(dlam), recon_inner=float(recon),
                           recon_test=rec_te)
                for tag, F in (("frozen", Zf), ("distill", Zh), ("ref6", Z6)):
                    h = qlib.Head(F, Y, tr, LAM_HEAD)
                    m = qlib.eval_metrics(h.predict(F, te), Y, te, tt)["rmse"]
                    out[f"{tag}_rmse"] = float(np.nanmean(m))
                recs.append(out)
            g = [r for r in recs if r["dataset"] == name and r["depth"] == depth]
            print(f"  [{name}] depth={depth}  frozen={np.mean([r['frozen_rmse'] for r in g]):.4f}"
                  f"  distilled={np.mean([r['distill_rmse'] for r in g]):.4f}"
                  f"  recon_test={np.mean([r['recon_test'] for r in g]):.3f}", flush=True)

    import pandas as pd
    df = pd.DataFrame(recs)
    df.to_csv(f"{qlib.SCRATCH}/results_wo_distill.csv", index=False)
    pd.set_option("display.width", 240, "display.max_columns", 40)

    print("\n" + "=" * 98)
    print("SANITY CHECK: at depth 6 the distilled readout must reproduce Z_6")
    print("=" * 98)
    s = df[df.depth == NATIVE]
    print(s.groupby("dataset")[["recon_test", "frozen_rmse", "distill_rmse",
                                "ref6_rmse"]].mean().round(4).to_string())

    print("\n" + "=" * 98)
    print("RMSE by depth: frozen depth-6 readout vs DISTILLED readout")
    print("=" * 98)
    for t in ("frozen", "distill"):
        print(f"\n  {t}_rmse:")
        print(df.pivot_table(index="dataset", columns="depth",
                             values=f"{t}_rmse").round(4).to_string())
    print("\n  relative reconstruction error of Z_6 on TEST molecules "
          "(0 = perfectly recovered):")
    print(df.pivot_table(index="dataset", columns="depth",
                         values="recon_test").round(3).to_string())

    rows = []
    for t in ("frozen", "distill"):
        base = df[df.depth == NATIVE].set_index(["dataset", "split"])["ref6_rmse"]
        for (ds, dp), g in df.groupby(["dataset", "depth"]):
            d = np.array([(v - base[(ds, s)]) / base[(ds, s)] * 100
                          for v, s in zip(g[f"{t}_rmse"], g.split)])
            se = d.std(ddof=1) / np.sqrt(len(d))
            rows.append(dict(readout=t, dataset=ds, depth=int(dp), d_pct=float(d.mean()),
                             se=float(se), resolved=bool(abs(d.mean()) > 2 * se)))
    R = pd.DataFrame(rows)
    R.to_csv(f"{qlib.SCRATCH}/results_wo_distill_deltas.csv", index=False)
    print("\n" + "=" * 98)
    print("TRUNCATION COST vs the TRUE depth-6 model (paired %, + = worse)")
    print("=" * 98)
    print(R.pivot_table(index=["dataset", "readout"], columns="depth",
                        values="d_pct").round(2).to_string())
    print("\nSE:")
    print(R.pivot_table(index=["dataset", "readout"], columns="depth",
                        values="se").round(2).to_string())
    print("\n" + "=" * 98)
    print("GAP CLOSED BY DISTILLING THE READOUT  "
          "(100*(frozen-distill)/frozen; 100 = fully closed)")
    print("=" * 98)
    f = R[R.readout == "frozen"].set_index(["dataset", "depth"]).d_pct
    r = R[R.readout == "distill"].set_index(["dataset", "depth"]).d_pct
    print(((f - r) / f * 100).unstack().round(1).to_string())

    with open(f"{qlib.SCRATCH}/results_wo_distill.json", "w") as fh:
        json.dump(dict(caveat=CAVEAT, head_lambda=LAM_HEAD, distill_lambdas=DLAMS,
                       native_depth=NATIVE,
                       lambda_selection="distillation lambda by RECONSTRUCTION error on "
                                        "an inner 80/20 split of TRAIN rows; label-free; "
                                        "test never used",
                       records=recs, deltas=rows), fh, indent=1)
    print("\nwrote results_wo_distill.json/.csv and results_wo_distill_deltas.csv")


if __name__ == "__main__":
    main()
