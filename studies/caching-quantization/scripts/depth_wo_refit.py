"""Does refitting the readout close the truncation gap? And are cache headroom and
depth-reduction in tension?

Three outputs, all persisted with the caveat embedded as a top-level JSON field:
  results_depth_wo_refit.json / .csv        per (dataset, depth, split, readout)
  results_depth_tension.json / .csv         the headroom-vs-accuracy correlation

-------------------------- WHY THE REFIT IS EXACT AND CHEAP --------------------------
MeanAggregation is LINEAR, so the molecule embedding is

    Z = W_o * A + b_o,      A := mean_{v in mol} [ V_v || M_v ]      (A is 2120-dim)

and the ridge head applied to Z is affine in Z, hence affine in A. Conversely every
affine function of A equals head'(W_o' A) for some W_o'. Therefore:

    "re-fit W_o end-to-end jointly with the ridge head"
      ==  "fit the ridge head directly on A, the pooled PRE-readout features"

exactly, up to how the ridge penalty is parameterised. So the refit costs one extra
ridge solve per split -- no 2048x2120 least-squares needed. The refit path has strictly
more freedom than frozen-W_o + head, so any remaining gap at depth k is information
that is genuinely absent from the first k message-passing rounds, not readout mismatch.

To avoid handicapping the refit by a penalty chosen for the frozen path, the refit is
reported at the shared lambda=100 AND at the best lambda per (dataset, depth) from a
scan -- the latter is a generous upper bound on what refitting can buy.

============================== CAVEAT -- READ FIRST ==============================
The CheMeleon weights were PRETRAINED AT DEPTH 6. Truncating the recursion is
truncation of a depth-6 model, not a depth-k model. W_h was fitted to be applied 5
times, and the refit here frees ONLY the readout (W_o + head) -- W_i and W_h remain
the depth-6 ones. So:

  * if the refit closes most of the gap -> the information IS largely present by
    depth k and the deep layers mostly re-encode; a full retrain is worth running.
  * if the refit closes little -> the information genuinely accrues through depth 6
    and depth reduction is dead WITHOUT a retrain. It still does not prove a
    depth-k TRAINED encoder would fail, because W_i/W_h are still depth-6 weights.

This is indicative evidence that bounds where the accuracy lives. It is not a
substitute for training a depth-k encoder.
==================================================================================
"""
import json
import os
import pickle

import numpy as np
import torch

import qlib

CACHE = os.path.join(qlib.SCRATCH, "cache")
LAM = json.load(open(f"{CACHE}/lambda.json"))["lam"]
LAM_SCAN = [10, 30, 100, 300, 1000]
DEPTHS = [1, 2, 3, 4, 5, 6]
NATIVE = 6
HITS = json.load(open(f"{qlib.SCRATCH}/results_cache_hits.json"))
DS = list(json.load(open(f"{CACHE}/meta.json")).keys())
PRIMARY = {"reg": "rmse", "cls": "auroc"}


def encode_pre(chunks, Wi, Wh, Wo, bo, depth, d_h=2048):
    """Returns A = pooled PRE-readout features (n_mol, 72+d_h). Z = A @ Wo.T + bo."""
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


def score(Z, Y, tt, splits, lam):
    """mean primary metric per split."""
    out = {}
    for sn, tr, te in splits:
        h = qlib.Head(Z, Y, tr, lam)
        m = qlib.eval_metrics(h.predict(Z, te), Y, te, tt)[PRIMARY[tt]]
        out[sn] = float(np.nanmean(m)) if np.isfinite(m).any() else np.nan
    return out


def main():
    W_i, W_h, W_o, b_o, d_h, _ = qlib.load_chemeleon()
    recs = []
    for name in DS:
        with open(f"{CACHE}/{name}.pkl", "rb") as fh:
            D = pickle.load(fh)
        Y, tt, splits = D["Y"], D["task_type"], D["splits"]
        chunks = qlib.build_chunks(D["smiles"], 400)
        for depth in DEPTHS:
            A = encode_pre(chunks, W_i, W_h, W_o, b_o, depth, d_h)
            Z = torch.nn.functional.linear(A, W_o, b_o)      # frozen depth-6 readout
            froz = score(Z, Y, tt, splits, LAM)
            ref = score(A, Y, tt, splits, LAM)               # readout refit (exact)
            best, best_lam = ref, LAM
            agg = lambda d: np.nanmean(list(d.values()))
            for lam in LAM_SCAN:
                if lam == LAM:
                    continue
                c = score(A, Y, tt, splits, lam)
                better = (agg(c) < agg(best)) if tt == "reg" else (agg(c) > agg(best))
                if better:
                    best, best_lam = c, lam
            for sn, _, te in splits:
                recs.append(dict(dataset=name, depth=depth, split=sn,
                                 split_type="scaffold" if sn.startswith("scaf") else "random",
                                 task_type=tt, metric=PRIMARY[tt],
                                 frozen_Wo=froz[sn], refit_Wo=ref[sn],
                                 refit_Wo_bestlam=best[sn], best_lam=best_lam))
            print(f"  [{name}] depth={depth} frozen={agg(froz):.4f} "
                  f"refit={agg(ref):.4f} refit*={agg(best):.4f} (lam={best_lam})",
                  flush=True)

    import pandas as pd
    df = pd.DataFrame(recs)
    with open(f"{qlib.SCRATCH}/results_depth_wo_refit.json", "w") as fh:
        json.dump(dict(caveat=__doc__, lam_shared=LAM, lam_scan=LAM_SCAN,
                       native_depth=NATIVE,
                       note="refit_Wo == ridge head fitted on the pooled PRE-readout "
                            "features A, which is exactly equivalent to re-fitting W_o "
                            "jointly with the head because mean-pooling is linear.",
                       records=recs), fh, indent=1)
    df.to_csv(f"{qlib.SCRATCH}/results_depth_wo_refit.csv", index=False)

    pd.set_option("display.width", 240, "display.max_columns", 50)
    sign = lambda tt: 1.0 if tt == "reg" else -1.0

    # ---- paired deltas vs depth 6 of the SAME readout ----
    rows = []
    for col in ("frozen_Wo", "refit_Wo", "refit_Wo_bestlam"):
        base = df[df.depth == NATIVE].set_index(["dataset", "split"])[col]
        for (ds, dp), g in df.groupby(["dataset", "depth"]):
            tt = g.task_type.iloc[0]
            d = np.array([sign(tt) * (r - base[(ds, s)]) / abs(base[(ds, s)]) * 100
                          for r, s in zip(g[col], g.split)])
            d = d[np.isfinite(d)]
            b = df[(df.dataset == ds) & (df.depth == NATIVE)
                   & (df.split_type == "random")][col]
            rows.append(dict(readout=col, dataset=ds, depth=dp, task_type=tt,
                             d_pct=d.mean(), se=d.std(ddof=1) / np.sqrt(len(d)),
                             resolved=abs(d.mean()) > 2 * d.std(ddof=1) / np.sqrt(len(d)),
                             noise_cv_pct=b.std(ddof=1) / abs(b.mean()) * 100))
    R = pd.DataFrame(rows)
    R.to_csv(f"{qlib.SCRATCH}/results_depth_wo_refit_deltas.csv", index=False)

    print("\n" + "=" * 104)
    print("TRUNCATION COST vs depth 6, FROZEN W_o  vs  REFIT READOUT   (paired %, +=worse)")
    print("=" * 104)
    for col in ("frozen_Wo", "refit_Wo_bestlam"):
        print(f"\n  readout = {col}")
        print(R[R.readout == col].pivot_table(index="dataset", columns="depth",
                                              values="d_pct").round(2).to_string())

    print("\n" + "=" * 104)
    print("HOW MUCH OF THE GAP DOES REFITTING THE READOUT CLOSE?")
    print("   gap_closed% = 100*(frozen_delta - refit_delta)/frozen_delta  at each depth")
    print("=" * 104)
    f = R[R.readout == "frozen_Wo"].set_index(["dataset", "depth"]).d_pct
    r2 = R[R.readout == "refit_Wo_bestlam"].set_index(["dataset", "depth"]).d_pct
    gc = ((f - r2) / f * 100).unstack()
    print(gc.round(1).to_string())
    print("\n  (negative = refitting made the truncation gap WORSE in relative terms;")
    print("   note the refit also improves the depth-6 baseline it is compared against)")
    print("\n  absolute benefit of refitting at each depth (percentage points of the")
    print("  depth-6-frozen metric, +=refit better):")
    bb = df[df.depth == NATIVE].set_index(["dataset", "split"]).frozen_Wo
    rows2 = []
    for (ds, dp), g in df.groupby(["dataset", "depth"]):
        tt = g.task_type.iloc[0]
        v = np.array([sign(tt) * (fr - rf) / abs(bb[(ds, s)]) * 100
                      for fr, rf, s in zip(g.frozen_Wo, g.refit_Wo_bestlam, g.split)])
        rows2.append(dict(dataset=ds, depth=dp, refit_gain_pp=np.nanmean(v)))
    print(pd.DataFrame(rows2).pivot_table(index="dataset", columns="depth",
                                          values="refit_gain_pp").round(2).to_string())

    # ---- THE TENSION: cache headroom vs truncation cost ----
    print("\n" + "=" * 104)
    print("THE TENSION: is cache headroom anti-correlated with tolerance for shallowness?")
    print("   x = warm-cache hit rate at radius 2 (depth 3), random splits, from study (D)")
    print("   y = truncation cost at depth 3 (paired % vs depth 6)")
    print("=" * 104)
    tens = []
    for ds in DS:
        rnd = np.mean([w["edge_hit"] for w in HITS[ds]["warm"]
                       if not w["split"].startswith("scaf")], 0)
        scf = np.mean([w["edge_hit"] for w in HITS[ds]["warm"]
                       if w["split"].startswith("scaf")], 0)
        row = dict(dataset=ds,
                   hit_r2_random=rnd[2], hit_r2_scaffold=scf[2],
                   hit_r4_random=rnd[4], hit_max_r2=HITS[ds]["edge"][2]["max_hit"])
        for col in ("frozen_Wo", "refit_Wo_bestlam"):
            for dp in (3, 5):
                q = R[(R.readout == col) & (R.dataset == ds) & (R.depth == dp)].d_pct
                row[f"trunc_d{dp}_{col}"] = float(q.iloc[0]) if len(q) else np.nan
        tens.append(row)
    T = pd.DataFrame(tens)
    T.to_csv(f"{qlib.SCRATCH}/results_depth_tension.csv", index=False)
    print(T.round(4).to_string(index=False))
    corr = {}
    for col in ("frozen_Wo", "refit_Wo_bestlam"):
        for hx in ("hit_r2_random", "hit_r2_scaffold", "hit_max_r2"):
            s = T[[hx, f"trunc_d3_{col}"]].dropna()
            if len(s) > 2:
                corr[f"{hx} vs trunc_d3_{col}"] = dict(
                    spearman=float(s.corr(method="spearman").iloc[0, 1]),
                    pearson=float(s.corr(method="pearson").iloc[0, 1]), n=int(len(s)))
    print("\n  rank correlations (UNDERPOWERED -- n is the number of datasets):")
    for k, v in corr.items():
        print(f"    {k:<44} Spearman={v['spearman']:+.3f}  "
              f"Pearson={v['pearson']:+.3f}  n={v['n']}")
    print("\n  INTERPRETATION KEY: a POSITIVE correlation means datasets with MORE cache")
    print("  headroom at radius 2 also LOSE MORE accuracy when truncated to depth 3,")
    print("  i.e. depth reduction and cache efficiency are IN TENSION. A NEGATIVE")
    print("  correlation means they COMPOUND (headroom where truncation is cheap).")
    with open(f"{qlib.SCRATCH}/results_depth_tension.json", "w") as fh:
        json.dump(dict(caveat=__doc__,
                       interpretation="positive correlation = depth reduction and cache "
                                      "efficiency are in TENSION; negative = they compound",
                       underpowered=f"n={len(T)} datasets; rank correlation only",
                       correlations=corr, rows=tens), fh, indent=1)


if __name__ == "__main__":
    main()
