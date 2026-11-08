"""Correction to the traffic accounting: torch.cat's uncounted full write.

The kernel brief's traffic arithmetic counts the scatter/gather round-trips but not
the two torch.cat calls in Chemprop's forward:

    initialize():  torch.cat([V[src], E], dim=1)   -> writes (2m, d_v + d_e)
    finalize():    torch.cat((V, M), dim=1)        -> writes (n,  d_v + d_h)

finalize()'s cat is the expensive one: at d_h=2048 it materialises a (n, 2120) fp32
buffer purely to feed one matmul.  Both are avoidable with addmm against column
slices of the weight, which is algebraically identical.

If these are a material fraction of the message step, the brief's P1/P3 traffic
ratios are optimistic and must be corrected.

Run: python experiments/cat_cost.py [--quick]
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent / "."))
from hashimoto_bench import build_batch, load_smiles, timeit  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from cp_loader import SimpleMoleculeMolGraphFeaturizer  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "results"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--reps", type=int, default=5)
    a = ap.parse_args()
    reps = 2 if a.quick else a.reps

    feat = SimpleMoleculeMolGraphFeaturizer()
    b = build_batch(load_smiles(limit=100 if a.quick else None), feat)
    V, E, ei = b["V"], b["E"], b["ei"]
    n_v, n_e = V.shape[0], ei.shape[1]
    d_v, d_e = V.shape[1], E.shape[1]

    res = {"n_atom": int(n_v), "n_dir_bond": int(n_e), "d_v": d_v, "d_e": d_e,
           "timings": {}, "bytes": {}, "exactness": {}}
    devices = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])

    for dev in devices:
        Vd, Ed, eid = V.to(dev), E.to(dev), ei.to(dev)
        sync = torch.mps.synchronize if dev == "mps" else None
        for d_h in (300, 2048):
            W_i = torch.nn.Linear(d_v + d_e, d_h, bias=False).to(dev)
            W_o = torch.nn.Linear(d_v + d_h, d_h).to(dev)
            M = torch.randn(n_v, d_h, device=dev)

            # --- initialize(): cat then matmul, vs two matmuls on column slices
            Wi_v, Wi_e = W_i.weight[:, :d_v].contiguous(), W_i.weight[:, d_v:].contiguous()

            def init_cat():
                return W_i(torch.cat([Vd.index_select(0, eid[0]), Ed], dim=1))

            def init_split():
                return Vd.index_select(0, eid[0]) @ Wi_v.T + Ed @ Wi_e.T

            # --- finalize(): cat then matmul, vs addmm on column slices
            Wo_v, Wo_m = W_o.weight[:, :d_v].contiguous(), W_o.weight[:, d_v:].contiguous()

            def fin_cat():
                return W_o(torch.cat((Vd, M), dim=1))

            def fin_split():
                return torch.addmm(W_o.bias, Vd, Wo_v.T).addmm_(M, Wo_m.T)

            with torch.no_grad():
                for nm, fn in (("init_cat", init_cat), ("init_split", init_split),
                               ("finalize_cat", fin_cat), ("finalize_split", fin_split)):
                    res["timings"][f"{dev}|d_h={d_h}|{nm}"] = timeit(fn, reps, sync) * 1e3
                ec = (init_cat() - init_split()).abs().max().item()
                ef = (fin_cat() - fin_split()).abs().max().item()
            res["exactness"][f"{dev}|d_h={d_h}|init_max_abs"] = ec
            res["exactness"][f"{dev}|d_h={d_h}|finalize_max_abs"] = ef

            # --- byte accounting (fp32), per forward
            cat_init = n_e * (d_v + d_e) * 4
            cat_fin = n_v * (d_v + d_h) * 4
            msg_step = n_e * d_h * 4 * 3 + n_v * d_h * 4  # gather + rev + write + nodesum
            res["bytes"][f"d_h={d_h}"] = {
                "cat_initialize_MB": cat_init / 2**20,
                "cat_finalize_MB": cat_fin / 2**20,
                "one_message_step_MB": msg_step / 2**20,
                "cat_total_as_frac_of_one_msg_step": (cat_init + cat_fin) / msg_step,
            }
            print(f"{dev:4s} d_h={d_h:<5d} init cat {res['timings'][f'{dev}|d_h={d_h}|init_cat']:8.3f} "
                  f"-> split {res['timings'][f'{dev}|d_h={d_h}|init_split']:8.3f} ms | "
                  f"finalize cat {res['timings'][f'{dev}|d_h={d_h}|finalize_cat']:8.3f} "
                  f"-> split {res['timings'][f'{dev}|d_h={d_h}|finalize_split']:8.3f} ms "
                  f"| maxdiff {max(ec, ef):.2e}")

    res["caveat"] = (
        "cat appears ONCE per forward (initialize) and ONCE at readout (finalize), not once "
        "per depth iteration, so compare against the TOTAL of depth-1 message steps rather "
        "than one. The split form is algebraically identical but changes float summation "
        "order, so exactness is reported as max abs diff, not bitwise."
    )
    OUT.mkdir(exist_ok=True)
    (OUT / "results_cat_cost.json").write_text(json.dumps(res, indent=2))
    print(f"wrote {OUT / 'results_cat_cost.json'}")


if __name__ == "__main__":
    main()
