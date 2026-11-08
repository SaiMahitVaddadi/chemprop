"""fp32 task accuracy of CheMeleon at TRUNCATED message-passing depths 1-6.

Writes results_depth_accuracy.json and results_depth_accuracy.csv plus a printed
table (redirect to DEPTH_ACCURACY.txt).

=========================== CAVEAT -- READ FIRST ===========================
The CheMeleon weights (W_i, W_h, W_o) were PRETRAINED AT DEPTH 6. Running the
recursion for fewer iterations is *truncation of a depth-6 model*, not a
depth-k model. Truncation is unfair to shallower settings: W_o was fitted to
consume the depth-6 aggregated state, and W_h was fitted to be applied 5 times.

Therefore this experiment answers exactly one question:

    "How much of the depth-6 accuracy is already present in the first k
     message-passing rounds?"

It is an indicative LOWER BOUND on what a depth-k encoder could achieve. It
does NOT show that a depth-3 encoder TRAINED at depth 3 would match depth 6 --
a retrained shallow encoder would very likely do better than truncation, so a
small truncation gap is strong evidence and a large gap is weak evidence.
Any decision to ship a shallower encoder needs a retrained depth-k checkpoint.
============================================================================

Protocol is identical to the main study: frozen encoder -> mean aggregation ->
ridge head (lambda=100) refitted on the fp32 TRAIN embeddings of that same
depth, 10 splits (6 random + 4 Bemis-Murcko scaffold), paired per-split deltas.
The head is refitted per depth because the embedding changes meaning; this is
the fairest available comparison.
"""
import json
import os
import pickle

import numpy as np
import torch

import qlib

CACHE = os.path.join(qlib.SCRATCH, "cache")
LAM = json.load(open(f"{CACHE}/lambda.json"))["lam"]
DS = ["qm9", "freesolv", "esol", "lipo"]       # cheapest first
DEPTHS = [1, 2, 3, 4, 5, 6]
NATIVE = 6

# study (D) pooled directed-bond hit rates, radius r = depth - 1
HITS = json.load(open(f"{qlib.SCRATCH}/results_cache_hits.json"))


def encode_depth(chunks, Wi, Wh, Wo, bo, depth, d_h=2048):
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
        Hv = torch.nn.functional.linear(torch.cat((V, M), 1), Wo, bo)
        cnt = torch.zeros(nm).index_add_(0, bt, torch.ones(bt.shape[0])).clamp(min=1)
        outs.append(torch.zeros(nm, d_h).index_add_(0, bt, Hv) / cnt.unsqueeze(1))
    return torch.cat(outs)


def main():
    W_i, W_h, W_o, b_o, d_h, _ = qlib.load_chemeleon()
    recs = []
    print(__doc__)
    for name in DS:
        with open(f"{CACHE}/{name}.pkl", "rb") as fh:
            D = pickle.load(fh)
        Y, tt, tasks, splits = D["Y"], D["task_type"], D["tasks"], D["splits"]
        chunks = qlib.build_chunks(D["smiles"], 400)
        for depth in DEPTHS:
            Z = encode_depth(chunks, W_i, W_h, W_o, b_o, depth, d_h)
            for sn, tr, te in splits:
                h = qlib.Head(Z, Y, tr, LAM)
                m = qlib.eval_metrics(h.predict(Z, te), Y, te, tt)
                sd = np.nanstd(Y[te], 0)
                recs.append(dict(
                    dataset=name, depth=depth, split=sn,
                    split_type="scaffold" if sn.startswith("scaf") else "random",
                    rmse=float(np.nanmean(m["rmse"])),
                    mae=float(np.nanmean(m["mae"])),
                    r2=float(np.nanmean(m["r2"])),
                    nrmse=float(np.nanmean(m["rmse"] / np.maximum(sd, 1e-9))),
                    n_test=int(len(te))))
            print(f"  [{name}] depth={depth} done", flush=True)

    with open(f"{qlib.SCRATCH}/results_depth_accuracy.json", "w") as fh:
        json.dump(dict(caveat=__doc__, lam=LAM, native_depth=NATIVE, records=recs), fh,
                  indent=1)
    import pandas as pd
    df = pd.DataFrame(recs)
    df.to_csv(f"{qlib.SCRATCH}/results_depth_accuracy.csv", index=False)

    pd.set_option("display.width", 220, "display.max_columns", 40)
    print("\n" + "=" * 100)
    print("fp32 RMSE by truncated depth (mean over 10 splits; native depth = 6)")
    print("=" * 100)
    piv = df.pivot_table(index="dataset", columns="depth", values="rmse")
    print(piv.round(4).to_string())
    print("\nsd over the 6 RANDOM splits (the noise floor at each depth):")
    nf = (df[df.split_type == "random"].groupby(["dataset", "depth"]).rmse
          .std(ddof=1).unstack())
    print(nf.round(4).to_string())

    print("\n" + "=" * 100)
    print("PAIRED %delta vs depth 6, per split (positive = WORSE than depth 6)")
    print("   +- SE over 10 splits; 'resolved' = |mean| > 2*SE")
    print("=" * 100)
    base = df[df.depth == NATIVE].set_index(["dataset", "split"]).rmse
    df["d_pct"] = [(r.rmse - base[(r.dataset, r.split)]) / base[(r.dataset, r.split)] * 100
                   for r in df.itertuples()]
    rows = []
    for (ds, dp), g in df.groupby(["dataset", "depth"]):
        d = g.d_pct.to_numpy()
        se = d.std(ddof=1) / np.sqrt(len(d))
        # noise floor expressed the same way: fp32 random-split CV at depth 6
        b = df[(df.dataset == ds) & (df.depth == NATIVE) & (df.split_type == "random")].rmse
        rows.append(dict(dataset=ds, depth=dp, d_pct=d.mean(), se=se,
                         resolved=abs(d.mean()) > 2 * se,
                         noise_cv_pct=b.std(ddof=1) / b.mean() * 100,
                         d_over_noise=d.mean() / (b.std(ddof=1) / b.mean() * 100),
                         d_pct_random=g[g.split_type == "random"].d_pct.mean(),
                         d_pct_scaffold=g[g.split_type == "scaffold"].d_pct.mean()))
    R = pd.DataFrame(rows)
    R.to_csv(f"{qlib.SCRATCH}/results_depth_accuracy_deltas.csv", index=False)
    for ds in DS:
        print(f"\n  {ds}:")
        print(R[R.dataset == ds][["depth", "d_pct", "se", "resolved", "noise_cv_pct",
                                  "d_over_noise", "d_pct_random",
                                  "d_pct_scaffold"]].round(3).to_string(index=False))

    print("\n" + "=" * 100)
    print("THE TRADEOFF IN ONE PLACE: accuracy cost of truncation vs cache hit rate")
    print("   depth k uses message-passing radius r = k-1, so the cache entry that")
    print("   must be stored/looked up is the radius-(k-1) bond environment.")
    print("=" * 100)
    pooled = {d["r"]: d["max_hit"] for d in HITS["_pooled"]["edge"]}
    L = lambda k: max(k - 1, 1)
    print(f"{'depth':>6}{'radius':>8}{'pooled hit%':>13}{'MP layers':>11}"
          f"{'rel FLOPs':>11}" + "".join(f"{ds+' d%':>12}" for ds in DS))
    for k in DEPTHS:
        r = k - 1
        hit = pooled.get(r, float('nan')) * 100
        # FLOPs of the W_h stack relative to depth 6, with and without the cache
        rel = (k - 1) / (NATIVE - 1) if NATIVE > 1 else 1.0
        cells = ""
        for ds in DS:
            v = R[(R.dataset == ds) & (R.depth == k)].d_pct
            cells += f"{float(v.iloc[0]):>+11.1f}%" if len(v) else f"{'':>12}"
        print(f"{k:>6}{r:>8}{hit:>12.1f}%{k-1:>11}{rel:>10.2f}x{cells}")
    print("\n  per-dataset warm-cache hit rate at the matching radius "
          "(cache from 80% train, random splits):")
    print(f"{'dataset':<10}" + "".join(f"{'d='+str(k):>9}" for k in DEPTHS))
    for ds in DS:
        rnd = np.mean([w["edge_hit"] for w in HITS[ds]["warm"]
                       if not w["split"].startswith("scaf")], 0)
        print(f"{ds:<10}" + "".join(f"{rnd[k-1]*100:>8.1f}%" for k in DEPTHS))

    print("\n" + "=" * 100)
    print("REMINDER OF THE CAVEAT: these weights were trained at depth 6. The numbers")
    print("above bound how much accuracy lives in the first k rounds; they do NOT show")
    print("that a depth-k TRAINED encoder matches depth 6. A retrained shallow encoder")
    print("would most likely beat truncation, so a small gap here is strong evidence FOR")
    print("shallow being viable, and a large gap is only weak evidence against.")
    print("=" * 100)


if __name__ == "__main__":
    main()
