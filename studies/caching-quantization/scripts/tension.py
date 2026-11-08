"""Is cache headroom anti-correlated with tolerance for a shallower encoder?

Uses only data already on disk:
  results_depth_accuracy_deltas.csv  (truncation cost, 4 regression datasets)
  results_cache_hits.json            (hit rates, all 11 datasets)

Writes results_depth_tension.json / .csv with the caveat as a top-level field.

INTERPRETATION KEY
  POSITIVE correlation  = datasets with MORE cache headroom at radius 2 also LOSE
                          MORE accuracy when truncated to depth 3  => depth
                          reduction and cache efficiency are IN TENSION.
  NEGATIVE correlation  = headroom sits where truncation is cheap => they COMPOUND.
"""
import json

import numpy as np
import pandas as pd

import qlib

CAVEAT = (
    "SEVERELY UNDERPOWERED: n=4 datasets. A Spearman rho on 4 points has a 2-sided "
    "p>=0.33 for any |rho|<1.0, so this cannot establish a correlation -- it can only "
    "show whether the sign is consistent and how large the spread is. Truncation cost "
    "is also measured by truncating a DEPTH-6-TRAINED checkpoint (W_i/W_h/W_o are all "
    "depth-6 weights), so it over-states the cost a retrained depth-3 encoder would pay. "
    "Treat this as a hypothesis-generating observation, not a measured relationship."
)

D = pd.read_csv(f"{qlib.SCRATCH}/results_depth_accuracy_deltas.csv")
H = json.load(open(f"{qlib.SCRATCH}/results_cache_hits.json"))

rows = []
for ds in sorted(D.dataset.unique()):
    rnd = np.mean([w["edge_hit"] for w in H[ds]["warm"]
                   if not w["split"].startswith("scaf")], 0)
    scf = np.mean([w["edge_hit"] for w in H[ds]["warm"]
                   if w["split"].startswith("scaf")], 0)
    g = D[D.dataset == ds].set_index("depth")
    rows.append(dict(
        dataset=ds,
        hit_r2_warm_random=rnd[2], hit_r2_warm_scaffold=scf[2],
        hit_r4_warm_random=rnd[4],
        hit_r2_max=H[ds]["edge"][2]["max_hit"], hit_r5_max=H[ds]["edge"][5]["max_hit"],
        trunc_d3_pct=float(g.loc[3, "d_pct"]), trunc_d3_se=float(g.loc[3, "se"]),
        trunc_d4_pct=float(g.loc[4, "d_pct"]), trunc_d5_pct=float(g.loc[5, "d_pct"]),
        noise_cv_pct=float(g.loc[3, "noise_cv_pct"]),
        n_mols=H[ds]["n"], atoms_per_mol=H[ds]["n_atoms"] / H[ds]["n"]))
T = pd.DataFrame(rows)
T.to_csv(f"{qlib.SCRATCH}/results_depth_tension.csv", index=False)

pd.set_option("display.width", 250, "display.max_columns", 40)
print("=" * 104)
print("CACHE HEADROOM vs TRUNCATION COST  (n=4 datasets -- UNDERPOWERED)")
print("=" * 104)
print(T.round(4).to_string(index=False))

corr = {}
for hx in ("hit_r2_warm_random", "hit_r2_warm_scaffold", "hit_r2_max", "hit_r4_warm_random"):
    for ty in ("trunc_d3_pct", "trunc_d4_pct", "trunc_d5_pct"):
        s = T[[hx, ty]].dropna()
        corr[f"{hx} ~ {ty}"] = dict(
            spearman=float(s.corr(method="spearman").iloc[0, 1]),
            pearson=float(s.corr(method="pearson").iloc[0, 1]), n=int(len(s)))
print("\nrank correlations:")
for k, v in corr.items():
    print(f"  {k:<48} Spearman={v['spearman']:+.3f}  Pearson={v['pearson']:+.3f}  n={v['n']}")

# the specific pair the coordinator flagged
lo = T.loc[T.hit_r2_warm_random.idxmin()]
hi = T.loc[T.hit_r2_warm_random.idxmax()]
print(f"\nthe flagged contrast:")
print(f"  LOWEST  headroom: {lo.dataset:<9} hit(r=2)={lo.hit_r2_warm_random:.3f}  "
       f"trunc(d=3)={lo.trunc_d3_pct:+.1f}%")
print(f"  HIGHEST headroom: {hi.dataset:<9} hit(r=2)={hi.hit_r2_warm_random:.3f}  "
       f"trunc(d=3)={hi.trunc_d3_pct:+.1f}%")
print(f"  => the dataset with the MOST headroom loses "
      f"{'MORE' if hi.trunc_d3_pct > lo.trunc_d3_pct else 'LESS'} accuracy than the one "
      f"with the least.")

# confounder: molecule size drives both
print("\npossible confounder -- molecule size drives both quantities:")
for c in ("hit_r2_warm_random", "trunc_d3_pct"):
    s = T[["atoms_per_mol", c]].dropna()
    print(f"  atoms_per_mol ~ {c:<22} Spearman="
          f"{s.corr(method='spearman').iloc[0,1]:+.3f}")

with open(f"{qlib.SCRATCH}/results_depth_tension.json", "w") as fh:
    json.dump(dict(caveat=CAVEAT,
                   interpretation_key={
                       "positive": "more cache headroom <-> more accuracy lost when "
                                   "shallow => depth reduction and caching are IN TENSION",
                       "negative": "headroom where truncation is cheap => they COMPOUND"},
                   correlations=corr, rows=rows), fh, indent=1)
print(f"\nwrote results_depth_tension.json / .csv")
