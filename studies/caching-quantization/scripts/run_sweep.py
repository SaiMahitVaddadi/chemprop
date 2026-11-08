"""Stage 2: the main sweep.

For each (dataset, config) we encode ONCE and then evaluate over all 10 splits,
because the encoder output is independent of the split. The ridge head is fitted
once per split on the fp32 train embeddings and reused for every config, which is
both the deployment-realistic protocol and the reason this is affordable.

Writes one JSONL record per (dataset, config, split, task).
"""
import json
import os
import pickle
import sys
import time

import numpy as np
import torch

import configs
import qlib

CACHE = os.path.join(qlib.SCRATCH, "cache")
LAM = json.load(open(f"{CACHE}/lambda.json"))["lam"]
DS = list(json.load(open(f"{CACHE}/meta.json")).keys())


def load(name):
    with open(f"{CACHE}/{name}.pkl", "rb") as fh:
        D = pickle.load(fh)
    D["Z0"] = torch.from_numpy(np.load(f"{CACHE}/{name}_Z0.npy"))
    return D


def main(out_path, only=None, depth=6):
    W_i, W_h, W_o, b_o, d_h, DEPTH = qlib.load_chemeleon()
    names = only or DS
    fh = open(out_path, "w")
    for name in names:
        D = load(name)
        Z0, Y, tt, tasks, splits = D["Z0"], D["Y"], D["task_type"], D["tasks"], D["splits"]
        chunks = qlib.build_chunks(D["smiles"], 400)
        # heads: fitted once per split on fp32 train embeddings, reused for all configs
        heads = [(sn, tr, te, qlib.Head(Z0, Y, tr, LAM)) for sn, tr, te in splits]
        t_ds = time.perf_counter()
        for gi, (grp, cname, bits, kind, spec) in enumerate(configs.ALL):
            t0 = time.perf_counter()
            if kind == "w":
                fi, fh_, fo = spec
                Z = qlib.encode(chunks, fi(W_i), fh_(W_h), fo(W_o), b_o, depth, d_h)
            else:
                Z = qlib.encode(chunks, W_i, W_h, W_o, b_o, depth, d_h, cache_q=spec)
            # embedding-space fidelity diagnostics (for the cosine-similarity claim)
            cos = float(torch.nn.functional.cosine_similarity(Z, Z0, dim=1).mean())
            rel = float((Z - Z0).norm() / Z0.norm())
            for sn, tr, te, head in heads:
                P = head.predict(Z, te)
                mt = qlib.eval_metrics(P, Y, te, tt)
                for t, task in enumerate(tasks):
                    rec = dict(dataset=name, group=grp, config=cname, bits=bits,
                               kind=kind, split=sn,
                               split_type="scaffold" if sn.startswith("scaf") else "random",
                               task=task, task_idx=t, task_type=tt,
                               cos=cos, relL2=rel, depth=depth,
                               n_test=int(len(te)))
                    for k, v in mt.items():
                        val = float(v[t]) if np.isfinite(v[t]) else None
                        rec[k] = val
                    if tt == "reg":
                        sd = np.nanstd(Y[te][:, t])
                        rec["target_std"] = float(sd) if np.isfinite(sd) else None
                    fh.write(json.dumps(rec) + "\n")
            fh.flush()
            print(f"  [{name}] {gi+1}/{len(configs.ALL)} {cname:<24} "
                  f"{time.perf_counter()-t0:>5.1f}s cos={cos:.5f} relL2={rel:.3e}", flush=True)
        print(f"[{name}] done in {(time.perf_counter()-t_ds)/60:.1f} min", flush=True)
    fh.close()


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else f"{qlib.SCRATCH}/results_main.jsonl"
    only = sys.argv[2].split(",") if len(sys.argv) > 2 else None
    main(out, only)
