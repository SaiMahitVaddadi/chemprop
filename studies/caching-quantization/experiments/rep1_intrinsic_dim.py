"""EXP 1 -- intrinsic dimensionality of the frozen CheMeleon embedding, and whether
a low-rank projection is a viable alternative to int4 for the cache.

Part A: PCA spectrum / effective rank / dead dimensions, per dataset and pooled.
Part B: rank-k projection -> reconstruct -> same ridge head (lam=100) -> downstream
        metric, paired per split, so the degradation is directly comparable to the
        +0.25% RMSE that int4 group-32 costs in the companion study.

PCA basis is fitted on TRAIN molecules only (no leakage). Reconstruction is back to
2048-d so the head protocol is byte-identical to the quantization runs.
"""
import sys
import time

import numpy as np
import torch

import rep_lib as R
sys.path.insert(0, R.SCRIPTS)
import qlib

RANKS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
NSPLIT = int(sys.argv[1]) if len(sys.argv) > 1 else 4
ONLY = sys.argv[2].split(",") if len(sys.argv) > 2 else R.DATASETS


def primary(m, tt):
    return float(np.nanmean(m["rmse"] if tt == "reg" else m["auroc"]))


def main():
    out = {"caveat": [
        "PCA/effective-rank is a SECOND-MOMENT statistic. Low linear rank does not "
        "mean low information content: a 1-d curve embedded in 2048-d has high "
        "nonlinear complexity and low linear rank.",
        "The spectrum is computed on the mean-aggregated MOLECULE embedding. The "
        "object the environment cache actually stores is the per-EDGE hidden state "
        "H; its rank is measured separately in rep6_env_rank.py.",
        "Each dataset here is <=2200 molecules, so at most min(n-1,2048) eigenvalues "
        "are non-zero by construction; n90/n95 are still meaningful but the tail is "
        "sample-limited. Pooled (n=13226) is the reliable spectrum.",
        "Rank-k reconstruction keeps the 2048-d head; it measures information loss, "
        "not the cost of a k-dim head.",
    ]}

    # ---------------- Part A: spectra ----------------
    specA = {}
    for nm in R.DATASETS:
        D = R.load(nm)
        Z = D["Z"]
        lam = R.spectrum(Z)
        st = R.rank_stats(lam, Z)
        st["n_mol"] = int(Z.shape[0])
        # also the correlation (z-scored) spectrum: scale-free view
        Zs = (Z - Z.mean(0)) / np.maximum(Z.std(0), 1e-8)
        st_corr = R.rank_stats(R.spectrum(Zs))
        st["corr_participation_ratio"] = st_corr["participation_ratio"]
        st["corr_n95"] = st_corr["n95"]
        st["corr_entropy_eff_rank"] = st_corr["entropy_eff_rank"]
        st["eigvals_top64"] = lam[:64].tolist()
        specA[nm] = st
        print(f"{nm:<10} n={Z.shape[0]:>5} PR={st['participation_ratio']:7.2f} "
              f"eff_rank={st['entropy_eff_rank']:7.2f} stable={st['stable_rank']:6.2f} "
              f"n90={st['n90']:>4} n95={st['n95']:>4} n99={st['n99']:>4} "
              f"top1={st['top1_var_frac']:.3f} dead={st['n_dead_1e-6']:>3} "
              f"nearconst={st['n_nearconst_1pct']:>4} | corrPR={st['corr_participation_ratio']:7.2f}")

    smis, Zp, src = R.pooled()
    lam = R.spectrum(Zp)
    stp = R.rank_stats(lam, Zp)
    stp["n_mol"] = int(Zp.shape[0])
    Zs = (Zp - Zp.mean(0)) / np.maximum(Zp.std(0), 1e-8)
    stc = R.rank_stats(R.spectrum(Zs))
    stp["corr_participation_ratio"] = stc["participation_ratio"]
    stp["corr_n90"] = stc["n90"]
    stp["corr_n95"] = stc["n95"]
    stp["corr_n99"] = stc["n99"]
    stp["corr_entropy_eff_rank"] = stc["entropy_eff_rank"]
    stp["eigvals"] = lam.tolist()
    print(f"\nPOOLED     n={Zp.shape[0]} PR={stp['participation_ratio']:.2f} "
          f"eff_rank={stp['entropy_eff_rank']:.2f} n90={stp['n90']} n95={stp['n95']} "
          f"n99={stp['n99']} top1={stp['top1_var_frac']:.3f} "
          f"dead={stp['n_dead_1e-6']} nearconst={stp['n_nearconst_1pct']}")
    print(f"POOLED corr-spectrum: PR={stp['corr_participation_ratio']:.2f} "
          f"n90={stp['corr_n90']} n95={stp['corr_n95']} n99={stp['corr_n99']}")
    out["spectrum"] = {"per_dataset": specA, "pooled": stp}

    # ---------------- Part B: rank-k -> downstream ----------------
    print(f"\n--- rank-k reconstruction -> ridge head (lam={R.LAM_DOWNSTREAM}), "
          f"{NSPLIT} random splits ---")
    curves = {}
    for nm in ONLY:
        D = R.load(nm)
        Z = torch.from_numpy(D["Z"])
        Y, tt = D["Y"], D["task_type"]
        key = "rmse" if tt == "reg" else "auroc"
        per = {k: [] for k in RANKS}
        t0 = time.perf_counter()
        for sn, tr, te in [s for s in D["splits"] if s[0].startswith("rand")][:NSPLIT]:
            mu, V, S = R.pca_basis(D["Z"][tr], max(RANKS))
            for k in RANKS:
                Zk = torch.from_numpy(R.rank_k_recon(D["Z"], mu, V, k)).float()
                h = qlib.Head(Zk, Y, tr, R.LAM_DOWNSTREAM)
                per[k].append(primary(qlib.eval_metrics(h.predict(Zk, te), Y, te, tt), tt))
        full = np.array(per[max(RANKS)])
        rec = {}
        for k in RANKS:
            v = np.array(per[k])
            d = (v - full) / full * 100
            if tt != "reg":
                d = -d                      # higher AUROC is better; report as a loss
            rec[k] = dict(mean=float(v.mean()), std=float(v.std()),
                          delta_pct=float(d.mean()), delta_pct_sd=float(d.std()),
                          per_split=v.tolist())
        curves[nm] = dict(task_type=tt, metric=key, ranks=rec)
        sel = [8, 16, 32, 64, 128, 256, 512]
        print(f"{nm:<10}{key:>6} full={full.mean():.4f} | " +
              " ".join(f"k{k}:{rec[k]['delta_pct']:+.1f}%" for k in sel) +
              f"  ({time.perf_counter()-t0:.0f}s)")
    out["rank_k"] = curves

    # smallest k within the 2.3% split-noise band, per dataset
    thresh = {}
    for nm, c in curves.items():
        ks = [k for k in RANKS if abs(c["ranks"][k]["delta_pct"]) <= 2.3]
        thresh[nm] = min(ks) if ks else None
    out["min_rank_within_2p3pct"] = thresh
    print("\nsmallest rank whose paired delta is inside the +-2.3% split-noise band:")
    for nm, k in thresh.items():
        print(f"  {nm:<10} k={k}")
    vals = [v for v in thresh.values() if v]
    if vals:
        print(f"  worst-case over datasets: k={max(vals)}  "
              f"-> {max(vals)*4} B/mol vs 8192 B/mol fp32 = {8192/(max(vals)*4):.1f}x")
    R.jdump(out, f"{R.RESULTS}/results_rep1_intrinsic_dim.json")


main()
