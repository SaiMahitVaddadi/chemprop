"""Supplementary sweep: the per-output-channel variants of int3/int2/int1 and
ternary, which group (A) only covered at group-32 granularity. Appends to a
separate JSONL that analyze.py concatenates with the main results.
"""
import json
import os
import pickle
import time

import numpy as np
import torch

import qlib

CACHE = os.path.join(qlib.SCRATCH, "cache")
LAM = json.load(open(f"{CACHE}/lambda.json"))["lam"]
DS = list(json.load(open(f"{CACHE}/meta.json")).keys())

i3c = lambda W: qlib.intq(W, 3)
i2c = lambda W: qlib.intq(W, 2)
i1c = lambda W: qlib.intq(W, 1)
ternc = lambda W: qlib.intq(W, 2, None, True)

SUPP = [("A", "int3-perchan", 3.0, (i3c,) * 3),
        ("A", "int2-perchan", 2.0, (i2c,) * 3),
        ("A", "int1-perchan", 1.0, (i1c,) * 3),
        ("A", "ternary-perchan", 1.6, (ternc,) * 3)]


def main(out_path):
    W_i, W_h, W_o, b_o, d_h, depth = qlib.load_chemeleon()
    fh = open(out_path, "w")
    for name in DS:
        with open(f"{CACHE}/{name}.pkl", "rb") as f:
            D = pickle.load(f)
        Z0 = torch.from_numpy(np.load(f"{CACHE}/{name}_Z0.npy"))
        Y, tt, tasks, splits = D["Y"], D["task_type"], D["tasks"], D["splits"]
        chunks = qlib.build_chunks(D["smiles"], 400)
        heads = [(sn, tr, te, qlib.Head(Z0, Y, tr, LAM)) for sn, tr, te in splits]
        for grp, cname, bits, (fi, fh_, fo) in SUPP:
            t0 = time.perf_counter()
            Z = qlib.encode(chunks, fi(W_i), fh_(W_h), fo(W_o), b_o, depth, d_h)
            cos = float(torch.nn.functional.cosine_similarity(Z, Z0, dim=1).mean())
            rel = float((Z - Z0).norm() / Z0.norm())
            for sn, tr, te, head in heads:
                mt = qlib.eval_metrics(head.predict(Z, te), Y, te, tt)
                for t, task in enumerate(tasks):
                    rec = dict(dataset=name, group=grp, config=cname, bits=bits,
                               kind="w", split=sn,
                               split_type="scaffold" if sn.startswith("scaf") else "random",
                               task=task, task_idx=t, task_type=tt, cos=cos,
                               relL2=rel, depth=depth, n_test=int(len(te)))
                    for k, v in mt.items():
                        rec[k] = float(v[t]) if np.isfinite(v[t]) else None
                    fh.write(json.dumps(rec) + "\n")
            fh.flush()
            print(f"  [{name}] {cname:<18} {time.perf_counter()-t0:>5.1f}s "
                  f"cos={cos:.5f}", flush=True)
    fh.close()


if __name__ == "__main__":
    main(f"{qlib.SCRATCH}/results_supp.jsonl")
