"""Robustness check: does a trained MLP head reorder the configs?

The primary study uses a closed-form ridge head on purpose -- it is deterministic,
so the measured delta is attributable to the encoder alone. The risk is that a
*trained* head would absorb or amplify quantization error differently. Here we
train a head with chemprop's FFN shape (one hidden layer of 300, ReLU) with Adam
on the fp32 train embeddings, then evaluate it on each quantized encoder's test
embeddings -- same deployment protocol as the ridge study.

3 MLP seeds x all splits, so the optimizer noise is itself quantified.
"""
import json
import os
import pickle

import numpy as np
import torch

import qlib

CACHE = os.path.join(qlib.SCRATCH, "cache")
e8 = lambda W: qlib.mx(W, "e4m3")
e4 = lambda W: qlib.mx(W, "e2m1")
i4g = lambda W: qlib.intq(W, 4, 32)
I = lambda W: W

CFGS = [("fp32", (I, I, I)),
        ("MXFP8-e4m3", (e8, e8, e8)),
        ("MXFP6-e3m2", (lambda W: qlib.mx(W, "e3m2"),) * 3),
        ("MXFP4-e2m1", (e4, e4, e4)),
        ("int4-g32", (i4g, i4g, i4g)),
        ("Wi@MXFP8_rest@MXFP4", (e8, e4, e4))]
DS = ["esol", "bace_cls"]
SEEDS = [0, 1, 2]
HID = 300          # chemprop FFN default hidden size
EPOCHS = 300


def train_mlp(Xtr, Ytr, mask, tt, T, seed):
    torch.manual_seed(seed)
    net = torch.nn.Sequential(torch.nn.Linear(Xtr.shape[1], HID), torch.nn.ReLU(),
                              torch.nn.Linear(HID, T))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
    lossf = (torch.nn.MSELoss(reduction="none") if tt == "reg"
             else torch.nn.BCEWithLogitsLoss(reduction="none"))
    with torch.enable_grad():
        for _ in range(EPOCHS):
            opt.zero_grad()
            out = net(Xtr)
            l = (lossf(out, Ytr) * mask).sum() / mask.sum()
            l.backward()
            opt.step()
    return net.eval()


def main():
    W_i, W_h, W_o, b_o, d_h, depth = qlib.load_chemeleon()
    recs = []
    for name in DS:
        with open(f"{CACHE}/{name}.pkl", "rb") as fh:
            D = pickle.load(fh)
        Y, tt, tasks, splits = D["Y"], D["task_type"], D["tasks"], D["splits"]
        T = Y.shape[1]
        chunks = qlib.build_chunks(D["smiles"], 400)
        Zs = {}
        for cname, (fi, fh_, fo) in CFGS:
            Zs[cname] = qlib.encode(chunks, fi(W_i), fh_(W_h), fo(W_o), b_o, depth, d_h)
        Z0 = Zs["fp32"]
        key = "rmse" if tt == "reg" else "auroc"
        for sn, tr, te in splits:
            mu, sg = Z0[tr].mean(0), Z0[tr].std(0).clamp(min=1e-6)
            Ytr_raw = Y[tr]
            m = ~np.isnan(Ytr_raw)
            ymu = np.where(m.any(0), np.nanmean(np.where(m, Ytr_raw, np.nan), 0), 0.0)
            ysd = np.where(m.any(0), np.nanstd(np.where(m, Ytr_raw, np.nan), 0), 1.0)
            ysd = np.maximum(ysd, 1e-9)
            if tt == "reg":
                Yt = torch.from_numpy(np.nan_to_num((Ytr_raw - ymu) / ysd)).float()
            else:
                Yt = torch.from_numpy(np.nan_to_num(Ytr_raw)).float()
            mask = torch.from_numpy(m.astype(np.float32))
            Xtr = ((Z0[tr] - mu) / sg)
            for seed in SEEDS:
                net = train_mlp(Xtr, Yt, mask, tt, T, seed)
                for cname, _ in CFGS:
                    P = net((Zs[cname][te] - mu) / sg).numpy().astype(np.float64)
                    if tt == "reg":
                        P = P * ysd[None, :] + ymu[None, :]
                    v = qlib.eval_metrics(P, Y, te, tt)[key]
                    if not np.isfinite(v).any():
                        continue
                    recs.append(dict(dataset=name, split=sn, seed=seed, config=cname,
                                     metric=key, value=float(np.nanmean(v))))
            print(f"  [{name}] {sn} done", flush=True)
    with open(f"{qlib.SCRATCH}/results_mlp.json", "w") as fh:
        json.dump(recs, fh, indent=1)

    # ---- report: paired delta vs fp32, per (dataset, seed) and pooled ----
    import pandas as pd
    df = pd.DataFrame(recs)
    print("\n=== MLP head (2048-300-T, Adam): paired %delta vs fp32, positive = worse ===")
    out = []
    for (ds, cfg), g in df.groupby(["dataset", "config"]):
        base = df[(df.dataset == ds) & (df.config == "fp32")].set_index(["split", "seed"]).value
        d = []
        for _, r in g.iterrows():
            b = base.get((r.split, r.seed))
            if b is None or not np.isfinite(b):
                continue
            sign = 1.0 if r.metric == "rmse" else -1.0
            d.append(sign * (r.value - b) / abs(b) * 100)
        out.append(dict(dataset=ds, config=cfg, d_pct=np.mean(d),
                        se=np.std(d, ddof=1) / np.sqrt(len(d)), n=len(d)))
    o = pd.DataFrame(out)
    print(o.pivot_table(index="config", columns="dataset", values="d_pct")
          .reindex([c for c, _ in CFGS]).round(2).to_string())
    print("\nseed-to-seed spread of the fp32 MLP baseline (pure optimizer noise):")
    for ds, g in df[df.config == "fp32"].groupby("dataset"):
        per = g.groupby("split").value.std()
        print(f"  {ds}: mean within-split std across 3 seeds = {per.mean():.5f} "
              f"({per.mean()/g.value.mean()*100:.2f}% of the metric); "
              f"split-to-split std = {g.groupby('seed').value.mean().std():.5f}")


if __name__ == "__main__":
    main()
