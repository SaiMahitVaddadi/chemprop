"""P8/P9 — size-bucketed dense block-diagonal bmm instead of scatter.

B is block-diagonal over molecules.  If molecules are sorted into size buckets and
padded to a per-bucket E_max, the whole message step becomes

    (b, E_max, E_max) @ (b, E_max, d_h)

i.e. pure bmm, with no irregular gather/scatter at all.  That is attractive on MPS
(weak scatter, no sparse support whatsoever) and is fully ONNX/TorchScript-safe.

It lives or dies on bucketing: padding every molecule to the batch maximum inflates
the main W_h GEMM badly, which is the failure mode this script quantifies.

Run: python experiments/bmm_bucketed.py [--quick]
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent / "."))
from hashimoto_bench import load_smiles, timeit  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from cp_loader import SimpleMoleculeMolGraphFeaturizer  # noqa: E402
from rdkit import Chem, RDLogger  # noqa: E402

RDLogger.DisableLog("rdApp.*")
OUT = Path(__file__).resolve().parent.parent / "results"


def per_mol_graphs(smis, feat):
    out = []
    for s in smis:
        g = feat(Chem.MolFromSmiles(s))
        if g.edge_index.shape[1]:
            out.append(g)
    return out


def dense_B(g):
    """Dense Hashimoto block for one molecule: B[e, f] = 1 iff dest(f)==src(e), f!=rev(e)."""
    ei, rev = g.edge_index, g.rev_edge_index
    m = ei.shape[1]
    B = np.zeros((m, m), dtype=np.float32)
    for e in range(m):
        a = ei[0, e]
        for f in range(m):
            if ei[1, f] == a and f != rev[e]:
                B[e, f] = 1.0
    return B


def bucketise(graphs, n_buckets):
    sizes = np.array([g.edge_index.shape[1] for g in graphs])
    qs = np.quantile(sizes, np.linspace(0, 1, n_buckets + 1)[1:])
    edges = np.unique(np.ceil(qs).astype(int))
    assign = np.searchsorted(edges, sizes, side="left")
    return edges, assign, sizes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--n-mol", type=int, default=500)
    ap.add_argument("--buckets", type=int, default=4)
    ap.add_argument("--reps", type=int, default=5)
    a = ap.parse_args()
    n_mol = 100 if a.quick else a.n_mol
    reps = 2 if a.quick else a.reps

    feat = SimpleMoleculeMolGraphFeaturizer()
    graphs = per_mol_graphs(load_smiles(limit=n_mol * 2)[:n_mol], feat)
    edges, assign, sizes = bucketise(graphs, a.buckets)

    res = {"n_mol": len(graphs), "bucket_edges": edges.tolist(),
           "dir_bond_sizes": {"mean": float(sizes.mean()), "max": int(sizes.max()),
                              "p50": float(np.median(sizes)), "p95": float(np.quantile(sizes, .95))},
           "flops": {}, "timings": {}, "exactness": {}}

    # --- P8: FLOP overhead of padding, bucketed vs naive batch-max
    tot_e = int(sizes.sum())
    for d_h in (300, 2048):
        wh = 2 * tot_e * d_h * d_h  # the real W_h GEMM, unpadded
        bucket_bmm = sum(
            2 * (assign == b).sum() * int(edges[b]) ** 2 * d_h for b in range(len(edges))
        )
        bucket_wh = sum(2 * (assign == b).sum() * int(edges[b]) * d_h * d_h for b in range(len(edges)))
        naive_wh = 2 * len(graphs) * int(sizes.max()) * d_h * d_h
        res["flops"][f"d_h={d_h}"] = {
            "Wh_gemm_unpadded_GFLOP": wh / 1e9,
            "bucketed_B_bmm_GFLOP": bucket_bmm / 1e9,
            "bucketed_B_bmm_pct_of_Wh": 100 * bucket_bmm / wh,
            "bucketed_Wh_inflation": bucket_wh / wh,
            "naive_batchmax_Wh_inflation": naive_wh / wh,
        }
        print(f"d_h={d_h:<5d} B-bmm = {100*bucket_bmm/wh:5.1f}% of Wh | "
              f"Wh inflation bucketed {bucket_wh/wh:.3f}x vs naive {naive_wh/wh:.3f}x")

    # --- P9: wall clock, bucketed bmm vs per-molecule scatter, on each device
    devices = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])
    packs = []
    for b in range(len(edges)):
        idx = np.where(assign == b)[0]
        if not len(idx):
            continue
        Em = int(edges[b])
        Bs = np.zeros((len(idx), Em, Em), dtype=np.float32)
        for j, gi in enumerate(idx):
            db = dense_B(graphs[gi])
            Bs[j, : db.shape[0], : db.shape[1]] = db
        packs.append((torch.from_numpy(Bs), Em, len(idx)))

    for dev in devices:
        sync = torch.mps.synchronize if dev == "mps" else None
        for d_h in (300, 2048):
            pk = [(B.to(dev), Em, nb) for B, Em, nb in packs]
            Hs = [torch.randn(nb, Em, d_h, device=dev) for _, Em, nb in pk]

            def run_bmm():
                return [torch.bmm(B, H) for (B, _, _), H in zip(pk, Hs)]

            with torch.no_grad():
                ms = timeit(run_bmm, reps, sync) * 1e3
            res["timings"][f"{dev}|d_h={d_h}|bucketed_bmm"] = ms
            print(f"  {dev:4s} d_h={d_h:<5d} bucketed_bmm {ms:9.3f} ms")

    res["caveat"] = (
        "Compare these timings against msg_* entries in results_hashimoto.json, which are "
        "measured on the same molecule set but as one flat batch. The bmm form does strictly "
        "more FLOPs and is expected to win only where irregular access is expensive (MPS). "
        "Dense B blocks are built in O(m^2) python here; a real implementation builds them in "
        "collate, and the build is excluded from the timings."
    )
    OUT.mkdir(exist_ok=True)
    (OUT / "results_bmm_bucketed.json").write_text(json.dumps(res, indent=2))
    print(f"wrote {OUT / 'results_bmm_bucketed.json'}")


if __name__ == "__main__":
    main()
