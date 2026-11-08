"""Stage 3: analysis.

Protocol notes that matter for reading the numbers:
  * primary metric = RMSE (regression) or AUROC (classification), averaged over
    tasks within a split first, then over splits.
  * every delta is PAIRED: computed per split against that same split's fp32
    result, then averaged. Paired deltas have far lower variance than the
    difference of two means.
  * sign convention: POSITIVE delta = WORSE than fp32, for both metric families.
  * NOISE FLOOR: two separate quantities are reported.
      - fp32 split-to-split std  : how much the fp32 baseline itself moves when
        you change the split. Any claim about a single split smaller than this is
        unsupportable.
      - paired-delta SE          : the uncertainty on the paired delta itself.
        A delta is called resolved only if |mean| > 2*SE.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

import qlib

SCRATCH = qlib.SCRATCH
PRIMARY = {"reg": "rmse", "cls": "auroc"}


def load(path):
    rows = [json.loads(l) for l in open(path)]
    supp = os.path.join(os.path.dirname(path) or ".", "results_supp.jsonl")
    if os.path.exists(supp):
        rows += [json.loads(l) for l in open(supp)]
    return pd.DataFrame(rows)


def collapse(df):
    """-> one row per (dataset, config, split): primary metric averaged over tasks."""
    out = []
    for (ds, cfg, sp), g in df.groupby(["dataset", "config", "split"], sort=False):
        tt = g["task_type"].iloc[0]
        p = PRIMARY[tt]
        vals = g[p].to_numpy(dtype=float)
        rec = dict(dataset=ds, config=cfg, split=sp,
                   split_type=g["split_type"].iloc[0], task_type=tt,
                   group=g["group"].iloc[0], bits=g["bits"].iloc[0],
                   kind=g["kind"].iloc[0], cos=g["cos"].iloc[0],
                   relL2=g["relL2"].iloc[0], primary=np.nanmean(vals),
                   n_tasks_ok=int(np.isfinite(vals).sum()))
        for sec in ("mae", "r2", "ap"):
            if sec in g:
                v = g[sec].to_numpy(dtype=float)
                if np.isfinite(v).any():
                    rec[sec] = np.nanmean(v)
        out.append(rec)
    return pd.DataFrame(out)


def deltas(col):
    """Add paired delta vs fp32 (positive = worse)."""
    base = (col[col.config == "fp32"]
            .set_index(["dataset", "split"])["primary"])
    col = col.copy()
    col["fp32"] = [base.get((d, s), np.nan) for d, s in zip(col.dataset, col.split)]
    sign = np.where(col.task_type == "reg", 1.0, -1.0)
    col["d_abs"] = sign * (col["primary"] - col["fp32"])
    col["d_pct"] = col["d_abs"] / col["fp32"].abs() * 100
    return col


def noise_floor(col):
    rows = []
    f = col[col.config == "fp32"]
    for ds, g in f.groupby("dataset"):
        tt = g.task_type.iloc[0]
        allv = g.primary.to_numpy()
        rnd = g[g.split_type == "random"].primary.to_numpy()
        scf = g[g.split_type == "scaffold"].primary.to_numpy()
        rows.append(dict(dataset=ds, metric=PRIMARY[tt], fp32_mean=allv.mean(),
                         std_all=allv.std(ddof=1), std_random=rnd.std(ddof=1),
                         std_scaffold=scf.std(ddof=1),
                         cv_all_pct=allv.std(ddof=1) / abs(allv.mean()) * 100,
                         cv_random_pct=rnd.std(ddof=1) / abs(rnd.mean()) * 100,
                         random_mean=rnd.mean(), scaffold_mean=scf.mean()))
    return pd.DataFrame(rows)


def per_config(col, nf):
    """Per (dataset, config): mean paired delta, SE, resolved?

    `snr` expresses the delta in units of the fp32 baseline's own random-split
    std, i.e. "how many baseline noise-widths of damage". |snr| < 1 means the
    damage is smaller than the spread you get just by reshuffling the split.
    """
    nstd = nf.set_index("dataset").std_random.to_dict()
    rows = []
    for (ds, cfg), g in col.groupby(["dataset", "config"]):
        d = g.d_pct.to_numpy(dtype=float)
        da = g.d_abs.to_numpy(dtype=float)
        n = len(d)
        se = d.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
        rows.append(dict(dataset=ds, config=cfg, group=g.group.iloc[0],
                         bits=g.bits.iloc[0], kind=g.kind.iloc[0],
                         task_type=g.task_type.iloc[0],
                         cos=g.cos.iloc[0], relL2=g.relL2.iloc[0],
                         n_splits=n, d_pct=d.mean(), d_pct_sd=d.std(ddof=1),
                         d_pct_se=se, d_abs=da.mean(),
                         resolved=bool(abs(d.mean()) > 2 * se) if n > 1 else False,
                         snr=da.mean() / nstd[ds],
                         d_pct_random=g[g.split_type == "random"].d_pct.mean(),
                         d_pct_scaffold=g[g.split_type == "scaffold"].d_pct.mean()))
    return pd.DataFrame(rows)


def summarize(pc, group=None, kinds=None):
    """Across-dataset summary per config."""
    q = pc
    if group:
        q = q[q.group.isin(group)]
    rows = []
    for cfg, g in q.groupby("config"):
        d = g.d_pct.to_numpy()
        rows.append(dict(config=cfg, group=g.group.iloc[0], bits=g.bits.iloc[0],
                         n_ds=len(g), mean=d.mean(), sd=d.std(ddof=1),
                         med=np.median(d), worst=d.max(), best=d.min(),
                         n_resolved_worse=int(((g.d_pct > 0) & g.resolved).sum()),
                         n_resolved_better=int(((g.d_pct < 0) & g.resolved).sum()),
                         snr=g.snr.mean(), snr_med=np.median(g.snr),
                         cos=g.cos.mean(), relL2=g.relL2.mean(),
                         reg=g[g.task_type == "reg"].d_pct.mean(),
                         cls=g[g.task_type == "cls"].d_pct.mean()))
    return pd.DataFrame(rows).sort_values("bits", ascending=False)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else f"{SCRATCH}/results_main.jsonl"
    df = load(path)
    col = collapse(df)
    # BBBP's deterministic scaffold split has a single-class test set, so AUROC is
    # undefined there for every config alike. Drop it rather than impute; BBBP
    # therefore has 9 usable splits and every other dataset has 10.
    nbad = int(col.primary.isna().sum())
    col = col[col.primary.notna()].copy()
    if nbad:
        print(f"[note] dropped {nbad} (dataset,config,split) rows with an undefined "
              f"metric (single-class test set); affects BBBP/scafdet only.\n")
    col = deltas(col)
    col.to_csv(f"{SCRATCH}/results_per_split.csv", index=False)
    nf = noise_floor(col)
    pc = per_config(col, nf)
    pc.to_csv(f"{SCRATCH}/results_per_dataset_config.csv", index=False)

    pd.set_option("display.width", 250, "display.max_columns", 60,
                  "display.max_rows", 400)

    print("=" * 110)
    print("NOISE FLOOR: fp32 baseline, split-to-split variability (10 splits: 6 random + 4 scaffold)")
    print("=" * 110)
    print(nf.round(4).to_string(index=False))
    print(f"\nmedian fp32 CV across datasets (all splits)   = {nf.cv_all_pct.median():.2f}%")
    print(f"median fp32 CV across datasets (random only)  = {nf.cv_random_pct.median():.2f}%")

    print("\n" + "=" * 110)
    print("(A) UNIFORM WEIGHT QUANTIZATION -- paired %delta vs fp32, positive = worse")
    print("    mean/sd are ACROSS the 11 datasets; each dataset's value is itself a")
    print("    10-split paired mean. 'res-worse' counts datasets where the delta is")
    print("    resolved (|mean| > 2*SE over splits).")
    print("=" * 110)
    sa = summarize(pc, ["A"])
    print(sa[["config", "bits", "mean", "sd", "med", "best", "worst",
              "n_resolved_worse", "n_resolved_better", "reg", "cls", "snr",
              "snr_med", "cos", "relL2"]].round(3).to_string(index=False))

    print("\n" + "=" * 110)
    print("(B) PER-MATRIX SENSITIVITY AND MIXED PRECISION")
    print("=" * 110)
    sb = summarize(pc, ["B"])
    print(sb[["config", "bits", "mean", "sd", "med", "best", "worst",
              "n_resolved_worse", "reg", "cls", "snr", "snr_med", "cos",
              "relL2"]].round(3).to_string(index=False))

    print("\n" + "=" * 110)
    print("(C) ENVIRONMENT-CACHE / ACTIVATION QUANTIZATION (weights fp32)")
    print("=" * 110)
    sc = summarize(pc, ["C"])
    print(sc[["config", "bits", "mean", "sd", "med", "best", "worst",
              "n_resolved_worse", "reg", "cls", "snr", "snr_med", "cos",
              "relL2"]].round(3).to_string(index=False))

    print("\n" + "=" * 110)
    print("PER-DATASET DETAIL for the headline configs (paired %delta +- SE over 10 splits)")
    print("=" * 110)
    key = ["MXFP8-e4m3", "int8-g32", "MXFP6-e3m2", "int6-g32", "MXFP4-e2m1",
           "NVFP4", "int4-g32", "int2-g32", "ternary-g32",
           "only-Wi@MXFP4", "only-Wh@MXFP4", "only-Wo@MXFP4",
           "Wi@MXFP8_rest@MXFP4", "Wi@MXFP8_rest@MXFP6",
           "cache-int8-g32", "cache-int4-g32", "cache-MXFP4-e2m1", "cache-int2-g32"]
    t = pc[pc.config.isin(key)].pivot_table(index="config", columns="dataset",
                                            values="d_pct")
    t = t.reindex(key)
    print(t.round(2).to_string())
    print("\nSE of each entry above (over 10 splits):")
    tse = pc[pc.config.isin(key)].pivot_table(index="config", columns="dataset",
                                              values="d_pct_se").reindex(key)
    print(tse.round(2).to_string())

    print("\n" + "=" * 110)
    print("RANDOM vs SCAFFOLD split: does the quantization cost differ?")
    print("=" * 110)
    cmp = (pc[pc.config.isin(key)]
           .groupby("config")[["d_pct_random", "d_pct_scaffold"]].mean()
           .reindex(key))
    cmp["diff"] = cmp.d_pct_scaffold - cmp.d_pct_random
    print(cmp.round(2).to_string())

    print("\n" + "=" * 110)
    print("FORMAT-vs-DISTRIBUTION (hypothesis 5): MX float vs int at equal bits")
    print("=" * 110)
    for lbl, a, b in [("WEIGHTS 8b : MXFP8-e4m3 vs int8-g32", "MXFP8-e4m3", "int8-g32"),
                      ("WEIGHTS 6b : MXFP6-e3m2 vs int6-g32", "MXFP6-e3m2", "int6-g32"),
                      ("WEIGHTS 4b : MXFP4-e2m1 vs int4-g32", "MXFP4-e2m1", "int4-g32"),
                      ("CACHE   8b : MXFP8 vs int8-g32", "cache-MXFP8-e4m3", "cache-int8-g32"),
                      ("CACHE   6b : MXFP6 vs int6-g32", "cache-MXFP6-e3m2", "cache-int6-g32"),
                      ("CACHE   4b : MXFP4 vs int4-g32", "cache-MXFP4-e2m1", "cache-int4-g32")]:
        x = pc[pc.config == a].set_index("dataset").d_pct
        y = pc[pc.config == b].set_index("dataset").d_pct
        j = x.index.intersection(y.index)
        wins = int((y[j] < x[j]).sum())
        print(f"  {lbl:<38} MX={x[j].mean():>8.2f}%  int={y[j].mean():>8.2f}%   "
              f"int better on {wins}/{len(j)} datasets")

    print("\n" + "=" * 110)
    print("COSINE SIMILARITY AS A DIAGNOSTIC (hypothesis 8)")
    print("=" * 110)
    w = pc[(pc.group == "A") & (pc.config != "fp32")]
    print(f"  Spearman(cos, d_pct) over all (dataset,config) weight points, n={len(w)}: "
          f"{pd.Series(w.cos).corr(pd.Series(w.d_pct), method='spearman'):.3f}")
    print(f"  Spearman(relL2, d_pct): "
          f"{pd.Series(w.relL2).corr(pd.Series(w.d_pct), method='spearman'):.3f}")
    ex = pc[pc.config.isin(["int1-g32", "ternary-g32", "int2-g32", "MXFP4-e2m1"])]
    print("\n  worked example -- high cosine with destroyed accuracy:")
    print(ex.groupby("config")[["cos", "relL2", "d_pct"]].mean().round(4).to_string())


main()
