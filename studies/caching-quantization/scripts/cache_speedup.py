"""Turn the measured hit rates into the implied cached-inference speedup bound.

A depth-6 bond-message encoder does 5 W_h layers per directed edge plus one W_o
pass per atom. With an exact environment cache, an edge whose radius-t key is
already stored needs no work at layers <= t (the descend/ascend schedule in
cached_mpnn3.py). The work actually required at layer t is therefore the miss
fraction (1 - h_t), so

    speedup_MP  =  (depth-1) / sum_{t=1..depth-1} (1 - h_t)

This is an UPPER BOUND on the achievable speedup: it credits every hit as free
and ignores keying, probing and gather costs, which cached_mpnn3.py measured at
~10-20 us/mol.
"""
import json

import numpy as np

import qlib

H = json.load(open(f"{qlib.SCRATCH}/results_cache_hits.json"))
DEPTH = 6
L = DEPTH - 1          # 5 W_h layers


def bound(h):
    miss = sum(1 - h[t] for t in range(1, L + 1))
    return L / miss if miss > 0 else np.inf


print("Implied UPPER-BOUND speedup on the message-passing layers of a depth-6 encoder")
print("(cold = encode the dataset once from empty; warm = new molecules against a")
print(" cache populated from the 80% train split)\n")
print(f"{'dataset':<10}{'cold(intra-ds)':>15}{'warm/random':>13}{'warm/scaffold':>15}"
      f"{'atom-env hit d=6':>18}")
rows = {}
for name, rec in H.items():
    if name.startswith("_"):
        continue
    cold = [d["max_hit"] for d in rec["edge"]]
    rnd = np.mean([w["edge_hit"] for w in rec["warm"]
                   if not w["split"].startswith("scaf")], 0)
    scf = np.mean([w["edge_hit"] for w in rec["warm"]
                   if w["split"].startswith("scaf")], 0)
    a6 = rec["atom"][DEPTH - 1]["max_hit"]
    rows[name] = (bound(cold), bound(rnd), bound(scf), a6)
    print(f"{name:<10}{bound(cold):>14.2f}x{bound(rnd):>12.2f}x{bound(scf):>14.2f}x"
          f"{a6*100:>17.1f}%")
pooled = [d["max_hit"] for d in H["_pooled"]["edge"]]
print(f"{'POOLED':<10}{bound(pooled):>14.2f}x{'':>13}{'':>15}")

v = np.array(list(rows.values()))
print(f"\nrange across the 11 datasets:")
print(f"  cold          {v[:,0].min():.2f}x - {v[:,0].max():.2f}x  (median {np.median(v[:,0]):.2f}x)")
print(f"  warm random   {v[:,1].min():.2f}x - {v[:,1].max():.2f}x  (median {np.median(v[:,1]):.2f}x)")
print(f"  warm scaffold {v[:,2].min():.2f}x - {v[:,2].max():.2f}x  (median {np.median(v[:,2]):.2f}x)")

print("\nmemory for the depth-1..5 edge cache at d_h=2048, per dataset (unique entries):")
print(f"{'dataset':<10}{'entries(r=1..5)':>17}{'fp32 GiB':>10}{'int8 GiB':>10}{'int4 GiB':>10}")
for name, rec in H.items():
    if name.startswith("_"):
        continue
    n = sum(rec["edge"][t]["unique"] for t in range(1, L + 1))
    print(f"{name:<10}{n:>17}{n*2048*4/2**30:>10.3f}{n*2048*8.25/8/2**30:>10.3f}"
          f"{n*2048*4.25/8/2**30:>10.3f}")
