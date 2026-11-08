"""Does refitting the READOUT close the depth-truncation gap?

Scoped down to run in the foreground: 2 single-task regression datasets
(esol n=1127, freesolv n=639), depths 1-6, all 10 splits. No background jobs.

--------------------- WHY THE REFIT IS CHEAP (and where my first
--------------------- attempt at it was WRONG)
MeanAggregation is linear, so the molecule embedding is

    Z = W_o * A + b_o,     A := mean_{v in mol} [ V_v || M_v ]   (2120-dim)

and the ridge head is affine in Z, hence affine in A. So the FUNCTION CLASS of
"refit W_o jointly with the head" is exactly "fit a linear head on A". No
2048x2120 least-squares is needed.

BUT the regularised ESTIMATORS are not equivalent. Ridge penalises ||w||^2 in
whichever basis you fit. W_o is a dense mixing matrix, so ridge-on-Z penalises a
very different (much better conditioned) direction set than ridge-on-A: A's 2120
columns are strongly correlated (M_v is a sum of similar H vectors), so at the
shared lambda=100 ridge-on-A massively overfits -- my first run produced esol
RMSE 13.7 at depth 2 versus 1.03 frozen. That is an artefact of the penalty, not
a property of the readout.

Fix, used here: select lambda from a WIDE grid (1e1..1e7) by INNER validation on
a 80/20 split of the TRAINING set only -- never on test -- and do it
symmetrically for both the frozen and the refit path, so neither is handicapped
and neither gets to peek at test.

============================== CAVEAT -- READ FIRST ==============================
The CheMeleon weights were PRETRAINED AT DEPTH 6. This experiment truncates that
checkpoint and frees ONLY the readout (W_o + head); W_i and W_h remain the
depth-6 weights. So:
  * refit closes most of the gap -> information IS largely present by depth k,
    the deep layers mostly re-encode, and a full retrain is worth running;
  * refit closes little          -> the information genuinely accrues through
    depth 6 and depth reduction is dead WITHOUT a retrain.
Neither case proves what a depth-k TRAINED encoder (new W_i, W_h) would do.
Indicative evidence bounding where the accuracy lives; not a substitute for a
depth-k retrain.
==================================================================================
"""
import json
import os
import pickle

import numpy as np
import torch

import qlib

CACHE = os.path.join(qlib.SCRATCH, "cache")
DEPTHS = [1, 2, 3, 4, 5, 6]
NATIVE = 6
DS = ["freesolv", "esol"]                     # single-task regression, cheapest
LAMS = [1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7]

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


def ridge_path(F, y, tr, ev, lams):
    """Fit ridge on rows `tr` for every lambda; return {lam: rmse on rows `ev`}.
    The Gram matrix is formed once and reused across lambdas."""
    Xt = F[tr].double()
    mu, sg = Xt.mean(0), Xt.std(0).clamp(min=1e-8)
    Xt = (Xt - mu) / sg
    Xt = torch.cat([Xt, torch.ones(len(tr), 1, dtype=torch.float64)], 1)
    Xe = ((F[ev].double() - mu) / sg)
    Xe = torch.cat([Xe, torch.ones(len(ev), 1, dtype=torch.float64)], 1)
    yt = torch.from_numpy(y[tr]).double()
    ym, ys = yt.mean(), yt.std().clamp(min=1e-12)
    G = Xt.T @ Xt
    b = Xt.T @ ((yt - ym) / ys)
    I = torch.eye(G.shape[0], dtype=torch.float64)
    out = {}
    for lam in lams:
        w = torch.linalg.solve(G + lam * I, b)
        p = (Xe @ w) * ys + ym
        e = p.numpy() - y[ev]
        out[lam] = float(np.sqrt((e ** 2).mean()))
    return out


def main():
    W_i, W_h, W_o, b_o, d_h, _ = qlib.load_chemeleon()
    recs = []
    print(CAVEAT)
    for name in DS:
        with open(f"{CACHE}/{name}.pkl", "rb") as fh:
            D = pickle.load(fh)
        y = D["Y"][:, 0].astype(np.float64)
        splits = D["splits"]
        chunks = qlib.build_chunks(D["smiles"], 400)
        for depth in DEPTHS:
            A = encode_pre(chunks, W_i, W_h, W_o, b_o, depth, d_h)
            Z = torch.nn.functional.linear(A, W_o, b_o)
            for sn, tr, te in splits:
                # inner 80/20 of TRAIN for lambda selection (never touches test)
                rng = np.random.default_rng(abs(hash((name, sn))) % 2**31)
                p = rng.permutation(len(tr))
                k = int(0.8 * len(tr))
                itr, iva = tr[p[:k]], tr[p[k:]]
                row = dict(dataset=name, depth=depth, split=sn,
                           split_type="scaffold" if sn.startswith("scaf") else "random")
                for tag, F in (("frozen", Z), ("refit", A)):
                    inner = ridge_path(F, y, itr, iva, LAMS)
                    lam = min(inner, key=inner.get)
                    test = ridge_path(F, y, tr, te, [lam])[lam]
                    row[f"{tag}_rmse"] = test
                    row[f"{tag}_lam"] = lam
                recs.append(row)
            m = lambda t: np.mean([r[f"{t}_rmse"] for r in recs
                                   if r["dataset"] == name and r["depth"] == depth])
            print(f"  [{name}] depth={depth}  frozen={m('frozen'):.4f}  "
                  f"refit={m('refit'):.4f}", flush=True)

    import pandas as pd
    df = pd.DataFrame(recs)
    df.to_csv(f"{qlib.SCRATCH}/results_wo_refit.csv", index=False)
    pd.set_option("display.width", 240, "display.max_columns", 40)

    print("\n" + "=" * 96)
    print("fp32 RMSE by depth: FROZEN depth-6 readout vs REFIT readout (lambda by inner CV)")
    print("=" * 96)
    for t in ("frozen", "refit"):
        print(f"\n  {t}_rmse:")
        print(df.pivot_table(index="dataset", columns="depth",
                             values=f"{t}_rmse").round(4).to_string())
    print("\n  selected lambda (median over splits):")
    for t in ("frozen", "refit"):
        print(f"   {t}: " + df.groupby(["dataset", "depth"])[f"{t}_lam"]
              .median().unstack().to_string().replace("\n", "\n            "))

    # paired truncation deltas vs that readout's own depth-6
    rows = []
    for t in ("frozen", "refit"):
        base = df[df.depth == NATIVE].set_index(["dataset", "split"])[f"{t}_rmse"]
        for (ds, dp), g in df.groupby(["dataset", "depth"]):
            d = np.array([(v - base[(ds, s)]) / base[(ds, s)] * 100
                          for v, s in zip(g[f"{t}_rmse"], g.split)])
            se = d.std(ddof=1) / np.sqrt(len(d))
            rows.append(dict(readout=t, dataset=ds, depth=dp, d_pct=d.mean(), se=se,
                             resolved=bool(abs(d.mean()) > 2 * se)))
    R = pd.DataFrame(rows)
    R.to_csv(f"{qlib.SCRATCH}/results_wo_refit_deltas.csv", index=False)
    print("\n" + "=" * 96)
    print("TRUNCATION COST vs depth 6 (paired %, + = worse), each readout vs its OWN depth 6")
    print("=" * 96)
    print(R.pivot_table(index=["dataset", "readout"], columns="depth",
                        values="d_pct").round(2).to_string())
    print("\nSE:")
    print(R.pivot_table(index=["dataset", "readout"], columns="depth",
                        values="se").round(2).to_string())

    print("\n" + "=" * 96)
    print("HOW MUCH OF THE TRUNCATION GAP DOES REFITTING THE READOUT CLOSE?")
    print("=" * 96)
    f = R[R.readout == "frozen"].set_index(["dataset", "depth"]).d_pct
    r = R[R.readout == "refit"].set_index(["dataset", "depth"]).d_pct
    print("  gap_closed% = 100*(frozen_delta - refit_delta)/frozen_delta")
    print(((f - r) / f * 100).unstack().round(1).to_string())
    print("\n  does the refit help AT ALL in absolute RMSE? (frozen - refit, + = refit better)")
    piv = (df.groupby(["dataset", "depth"])
           .apply(lambda g: (g.frozen_rmse - g.refit_rmse).mean(), include_groups=False)
           .unstack())
    print(piv.round(4).to_string())

    with open(f"{qlib.SCRATCH}/results_wo_refit.json", "w") as fh:
        json.dump(dict(caveat=CAVEAT, lams=LAMS, native_depth=NATIVE,
                       lambda_selection="inner 80/20 split of TRAIN only, per split, "
                                        "symmetric for both readouts; test never used",
                       equivalence_note="refit == linear head on pooled pre-readout A; "
                                        "exact for the function class, but the ridge "
                                        "penalty differs between the A and Z bases, which "
                                        "is why a wide lambda grid is mandatory",
                       records=recs, deltas=rows), fh, indent=1)
    print("\nwrote results_wo_refit.json/.csv and results_wo_refit_deltas.csv")


if __name__ == "__main__":
    main()
