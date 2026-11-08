"""Interleaved round-robin timing: robust to the heavy external CPU load on this box.
Each variant is timed once per round; we take the min over rounds. Drift is reported."""
import time, numpy as np, torch

def race(fns, rounds=9, warmup=2, sync=None):
    names = list(fns)
    for _ in range(warmup):
        for n in names: fns[n]()
    if sync: sync()
    T = {n: [] for n in names}
    for _ in range(rounds):
        for n in names:
            t = time.perf_counter(); fns[n]()
            if sync: sync()
            T[n].append(time.perf_counter() - t)
    return {n: (min(v), float(np.median(v)), max(v)) for n, v in T.items()}

def show(res, base=None, label=""):
    base = base or list(res)[0]
    b = res[base][0]
    print(f"  {label}{'variant':<24}{'min ms':>10}{'med ms':>10}{'max ms':>10}{'speedup':>9}")
    for n, (mn, md, mx) in res.items():
        print(f"  {label}{n:<24}{mn*1e3:>10.2f}{md*1e3:>10.2f}{mx*1e3:>10.2f}{b/mn:>9.2f}")
