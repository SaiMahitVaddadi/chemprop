"""Stage 4: evaluate each of the 8 prior findings against the multi-dataset data.

Everything here reads results_per_dataset_config.csv (paired deltas per dataset),
results_per_split.csv (the noise floor), results_depth.json and
results_cache_hits.json. Nothing is hand-entered.

Verdict rules, applied uniformly:
  CONFIRMED        the prior direction and rough magnitude hold on >=80% of
                   datasets and the sign is consistent.
  DATASET-DEPENDENT the sign or the practical conclusion flips between datasets.
  REFUTED          the prior direction is contradicted on a majority of datasets.
  INCONCLUSIVE     the effect is not resolved above the noise floor.
"""
import json

import numpy as np
import pandas as pd

import qlib

S = qlib.SCRATCH
pc = pd.read_csv(f"{S}/results_per_dataset_config.csv")
sp = pd.read_csv(f"{S}/results_per_split.csv")
pd.set_option("display.width", 220, "display.max_columns", 40)

NDS = pc.dataset.nunique()


def row(cfg):
    return pc[pc.config == cfg].set_index("dataset")


def get(cfg, col="d_pct"):
    return row(cfg)[col]


def band(cfg):
    g = row(cfg)
    return (f"mean {g.d_pct.mean():+.2f}%  median {g.d_pct.median():+.2f}%  "
            f"range [{g.d_pct.min():+.2f}, {g.d_pct.max():+.2f}]  "
            f"|snr| med {abs(g.snr).median():.2f}  "
            f"resolved-worse {int(((g.d_pct>0)&g.resolved).sum())}/{len(g)}")


def hdr(n, title):
    print("\n" + "=" * 100)
    print(f"HYPOTHESIS {n}: {title}")
    print("=" * 100)


# ---------------------------------------------------------------- noise floor
print("=" * 100)
print("NOISE FLOOR (fp32 baseline only)")
print("=" * 100)
f = sp[sp.config == "fp32"]
nf = f.groupby("dataset").apply(
    lambda g: pd.Series(dict(
        metric=g.task_type.iloc[0],
        fp32_random=g[g.split_type == "random"].primary.mean(),
        sd_random=g[g.split_type == "random"].primary.std(ddof=1),
        cv_random_pct=g[g.split_type == "random"].primary.std(ddof=1)
        / abs(g[g.split_type == "random"].primary.mean()) * 100,
        fp32_scaffold=g[g.split_type == "scaffold"].primary.mean(),
        sd_scaffold=g[g.split_type == "scaffold"].primary.std(ddof=1))),
    include_groups=False)
print(nf.round(4).to_string())
print(f"\nmedian fp32 random-split CV = {nf.cv_random_pct.median():.2f}%  "
      f"(range {nf.cv_random_pct.min():.2f}-{nf.cv_random_pct.max():.2f}%)")
# smallest resolvable paired delta, empirically
se = pc[pc.config != "fp32"].groupby("dataset").d_pct_se.median()
print("median paired-delta SE per dataset (the real resolution limit, in %):")
print(se.round(3).to_string())
print(f"\n=> a paired delta below about {2*se.median():.2f}% (2*median SE) is NOT "
      f"distinguishable from split noise.")

# ---------------------------------------------------------------- H1
hdr(1, "8-bit weights are free (prior: MXFP8-e4m3 -0.4%, int8-g32 +0.0% on ESOL)")
for c in ["bf16", "fp16", "MXFP8-e4m3", "MXFP8-e4m3-spec448", "MXFP8-e5m2",
          "int8-perchan", "int8-g32"]:
    print(f"  {c:<22} {band(c)}")
print("\n  per-dataset detail (paired %delta):")
print(pc[pc.config.isin(["MXFP8-e4m3", "int8-g32", "int8-perchan", "MXFP8-e5m2"])]
      .pivot_table(index="config", columns="dataset", values="d_pct").round(2).to_string())

# ---------------------------------------------------------------- H2
hdr(2, "6-bit marginal, 4-bit breaks, <=2-bit dead")
for c in ["MXFP6-e3m2", "MXFP6-e2m3", "int6-g32", "int6-perchan",
          "MXFP4-e2m1", "NVFP4", "int4-g32", "int4-perchan",
          "int3-g32", "int2-g32", "ternary-g32", "int1-g32"]:
    print(f"  {c:<22} {band(c)}")
print("\n  is <=2-bit really 'zero signal'? compare to predicting the train mean.")
print("  R2 of the fp32 and the 2-bit encoders on the regression datasets:")
r = sp[sp.task_type == "reg"]
rr = (r[r.config.isin(["fp32", "int2-g32", "ternary-g32", "int1-g32", "int3-g32"])]
      .groupby(["dataset", "config"]).r2.mean().unstack())
print(rr.round(3).to_string())
print("  (R2 <= 0 means no better than the train mean, i.e. genuinely zero signal)")

# ---------------------------------------------------------------- H3
hdr(3, "Sensitivity is INVERTED: W_i (2% of params, applied once) dominates")
for c in ["only-Wi@MXFP4", "only-Wh@MXFP4", "only-Wo@MXFP4",
          "only-Wi@int4g32", "only-Wh@int4g32", "only-Wo@int4g32",
          "only-Wi@int2g32", "only-Wh@int2g32", "only-Wo@int2g32"]:
    print(f"  {c:<22} {band(c)}")
print("\n  per-dataset ranking of the three matrices at MXFP4 "
      "(which matrix hurts most?):")
t = (pc[pc.config.isin(["only-Wi@MXFP4", "only-Wh@MXFP4", "only-Wo@MXFP4"])]
     .pivot_table(index="dataset", columns="config", values="d_pct"))
t["worst_matrix"] = t.idxmax(1).str.replace("only-", "").str.replace("@MXFP4", "")
print(t.round(2).to_string())
print(f"\n  W_i is the worst matrix on {int((t.worst_matrix=='Wi').sum())}/{len(t)} datasets;"
      f" W_o on {int((t.worst_matrix=='Wo').sum())}; W_h on {int((t.worst_matrix=='Wh').sum())}")
print(f"  W_i/W_h damage ratio: median {np.median(t['only-Wi@MXFP4']/t['only-Wh@MXFP4']):.1f}x"
      f"  (prior ESOL claim: 19.3/2.9 = 6.7x)")

try:
    dep = pd.DataFrame(json.load(open(f"{S}/results_depth.json")))
    print("\n  --- MECHANISM TEST 1: does W_i sensitivity scale with depth? ---")
    for ds in dep.dataset.unique():
        g = dep[(dep.dataset == ds) & (dep.residual)]
        print(f"   {ds}: dPrimary%% by depth")
        print(g.pivot_table(index="depth", columns="matrix",
                            values="d_primary_pct").round(1).to_string())
    print("\n  --- MECHANISM TEST 2: pure Jacobian amplification (fixed 1e-3 noise) ---")
    print("      amplification = relL2(embedding) / relL2(weight). If the residual")
    print("      re-injection amplified W_i error ~depth times, this must grow with depth.")
    g = dep[dep.residual].pivot_table(index="depth", columns="matrix", values="amp_noise")
    print(g.round(2).to_string())
    print("\n  --- MECHANISM TEST 3: RESIDUAL ABLATION (H = relu(W_h M), no H_0 re-injection) ---")
    print("      If the residual is the cause, W_i's excess sensitivity must collapse here.")
    for res in (True, False):
        g = dep[dep.residual == res]
        tag = "WITH residual (real model)" if res else "WITHOUT residual (ablated)"
        gg = g[g.depth == 6].groupby("matrix")[["d_primary_pct", "amp_noise",
                                                "rel_out_mxfp4"]].mean()
        print(f"   depth=6, {tag}:")
        print(gg.round(3).to_string())
    print("\n  weight-space MXFP4 error is nearly identical for the three matrices,")
    print("  so the asymmetry is NOT 'W_i is harder to quantize':")
    print(dep.groupby("matrix").w_relL2_mxfp4.first().round(4).to_string())
except FileNotFoundError:
    print("\n  [depth study results not yet available]")

# ---------------------------------------------------------------- H4
hdr(4, "Protecting W_i is nearly free and ~10x effective")
for c in ["MXFP4-e2m1", "Wi@MXFP8_rest@MXFP4", "Wi@bf16_rest@MXFP4",
          "Wi@MXFP8_rest@MXFP6", "Wi@MXFP8_rest@int4g32", "Wi@MXFP8_rest@int3g32",
          "Wi@MXFP8_rest@int2g32", "Wi+Wo@MXFP8_Wh@MXFP4",
          "Wh@MXFP8_rest@MXFP4", "Wh@MXFP6_rest@MXFP4", "int4-g32"]:
    print(f"  {c:<22} {band(c)}")
a, b = get("MXFP4-e2m1"), get("Wi@MXFP8_rest@MXFP4")
j = a.index.intersection(b.index)
print(f"\n  reduction factor (all-MXFP4 -> W_i protected), per dataset:")
print((a[j] / b[j]).round(1).to_string())
print(f"  median reduction = {np.median(a[j]/b[j]):.1f}x   "
      f"(helps on {int((b[j]<a[j]).sum())}/{len(j)} datasets)")

# ---------------------------------------------------------------- H5
hdr(5, "Format should match distribution: MX float wins for weights, int for activations")
print(f"  {'comparison':<42}{'MX':>9}{'int':>9}{'int better on':>16}")
for lbl, x, y in [("WEIGHTS 8b  MXFP8-e4m3 vs int8-g32", "MXFP8-e4m3", "int8-g32"),
                  ("WEIGHTS 6b  MXFP6-e3m2 vs int6-g32", "MXFP6-e3m2", "int6-g32"),
                  ("WEIGHTS 4b  MXFP4-e2m1 vs int4-g32", "MXFP4-e2m1", "int4-g32"),
                  ("CACHE   8b  MXFP8 vs int8-g32", "cache-MXFP8-e4m3", "cache-int8-g32"),
                  ("CACHE   6b  MXFP6 vs int6-g32", "cache-MXFP6-e3m2", "cache-int6-g32"),
                  ("CACHE   4b  MXFP4 vs int4-g32", "cache-MXFP4-e2m1", "cache-int4-g32")]:
    A, B = get(x), get(y)
    j = A.index.intersection(B.index)
    print(f"  {lbl:<42}{A[j].mean():>8.2f}%{B[j].mean():>8.2f}%"
          f"{int((B[j]<A[j]).sum()):>11}/{len(j)}")

# ---------------------------------------------------------------- H6
hdr(6, "NVFP4 underperforms plain MXFP4 (prior: +32.8% vs +20.7%)")
A, B = get("MXFP4-e2m1"), get("NVFP4")
j = A.index.intersection(B.index)
print(pd.DataFrame({"MXFP4": A[j], "NVFP4": B[j],
                    "NVFP4-MXFP4": B[j] - A[j]}).round(2).to_string())
print(f"\n  NVFP4 worse than MXFP4 on {int((B[j]>A[j]).sum())}/{len(j)} datasets "
      f"(weights); mean MXFP4 {A[j].mean():+.2f}% vs NVFP4 {B[j].mean():+.2f}%")
A2, B2 = get("cache-MXFP4-e2m1"), get("cache-NVFP4")
j2 = A2.index.intersection(B2.index)
print(f"  cache: MXFP4 {A2[j2].mean():+.2f}% vs NVFP4 {B2[j2].mean():+.2f}%  "
      f"(NVFP4 worse on {int((B2[j2]>A2[j2]).sum())}/{len(j2)})")
print(f"  weight-space relL2: MXFP4 {row('MXFP4-e2m1').relL2.mean():.4f} vs "
      f"NVFP4 {row('NVFP4').relL2.mean():.4f} (embedding-space)")

# ---------------------------------------------------------------- H7
hdr(7, "The environment cache quantizes far better than the weights")
for c in ["cache-fp16", "cache-bf16", "cache-MXFP8-e4m3", "cache-int8-perrow",
          "cache-int8-g32", "cache-MXFP6-e3m2", "cache-int6-g32",
          "cache-MXFP4-e2m1", "cache-NVFP4", "cache-int4-perrow",
          "cache-int4-g32", "cache-int2-g32"]:
    print(f"  {c:<22} {band(c)}")
print("\n  CACHE vs WEIGHTS at matched bit width (the actual claim):")
print(f"  {'bits':<8}{'weights':>28}{'cache':>28}{'ratio':>10}")
for bits, wc, cc in [("8b int", "int8-g32", "cache-int8-g32"),
                     ("6b int", "int6-g32", "cache-int6-g32"),
                     ("4b int", "int4-g32", "cache-int4-g32"),
                     ("4b MX", "MXFP4-e2m1", "cache-MXFP4-e2m1"),
                     ("2b int", "int2-g32", "cache-int2-g32")]:
    A, B = get(wc), get(cc)
    j = A.index.intersection(B.index)
    print(f"  {bits:<8}{A[j].mean():>26.2f}%{B[j].mean():>26.2f}%"
          f"{A[j].mean()/B[j].mean() if B[j].mean()!=0 else float('nan'):>10.1f}x")

# ---------------------------------------------------------------- H8
hdr(8, "Cosine similarity is a useless diagnostic")
w = pc[(pc.group == "A") & (pc.config != "fp32")]
print(f"  Spearman(cos, damage)   over {len(w)} (dataset,config) weight points: "
      f"{w.cos.corr(w.d_pct, method='spearman'):+.3f}")
print(f"  Spearman(relL2, damage) over the same points:                    "
      f"{w.relL2.corr(w.d_pct, method='spearman'):+.3f}")
print(f"  Pearson(cos, log10(1+damage)):                                   "
      f"{w.cos.corr(np.log10(1+w.d_pct.clip(lower=0)), method='pearson'):+.3f}")
print("\n  worked counter-examples (mean over datasets):")
ex = pc[pc.config.isin(["int1-g32", "ternary-g32", "int2-g32", "MXFP4-e2m1",
                        "only-Wo@MXFP4", "only-Wh@MXFP4", "cache-int4-g32"])]
print(ex.groupby("config")[["cos", "relL2", "d_pct"]].mean()
      .sort_values("cos", ascending=False).round(4).to_string())
print("\n  Pairs with the SAME cosine but very different damage show cos cannot rank:")
for a, b in [("only-Wh@MXFP4", "only-Wo@MXFP4"), ("ternary-g32", "int2-g32")]:
    ga, gb = row(a), row(b)
    print(f"   {a:<18} cos={ga.cos.mean():.5f} damage={ga.d_pct.mean():+8.2f}%")
    print(f"   {b:<18} cos={gb.cos.mean():.5f} damage={gb.d_pct.mean():+8.2f}%")

# ---------------------------------------------------------------- recipe
print("\n" + "=" * 100)
print("DEPLOYMENT CANDIDATES ranked by (bits, measured damage)")
print("=" * 100)
cand = pc[pc.config != "fp32"].groupby(["config", "bits"]).agg(
    mean_pct=("d_pct", "mean"), med_pct=("d_pct", "median"),
    worst_pct=("d_pct", "max"), snr_med=("snr", lambda x: np.median(np.abs(x))),
    n_res=("resolved", "sum"), n=("d_pct", "size")).reset_index()
cand = cand[cand.med_pct < 3.0].sort_values("bits")
print(cand.round(3).to_string(index=False))
