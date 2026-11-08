"""P3/P4/P5 — explicit Hashimoto operator vs Chemprop's factored form.

Chemprop computes  M = B H  with  B = S T^T - tau  (the Hashimoto non-backtracking
edge-adjacency matrix) in the FACTORED form  S(T^T H) - tau H.  For edge e=(a->b),

    M_e = sum_{f in In(a), f != rev(e)} H_f

so row e of B has exactly  deg(a) - 1  nonzeros.  Averaged over EDGES (not nodes)
that is  (mean(d^2) - mean(d)) / mean(d), which is < 1.5 for molecules.  The claim is
that evaluating B explicitly beats the factored form whenever

    mean(d^2) < 4*mean(d) + 2      (degree < 4.45 for a regular graph)

Molecules sit at degree ~2.09.  This script measures nnz(B), verifies the explicit
form is algebraically EXACT against the factored form in fp64, and benchmarks both
plus the index_add_ variant on CPU and MPS at both model sizes.

Run: python experiments/hashimoto_bench.py [--quick]
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from cp_loader import SimpleMoleculeMolGraphFeaturizer  # noqa: E402
from rdkit import Chem, RDLogger  # noqa: E402

RDLogger.DisableLog("rdApp.*")
HERE = Path(__file__).resolve().parent.parent
OUT = HERE / "results"
CKPT = Path.home() / "Library/Caches/molfeat/chemeleon/chemeleon_mp.pt"


# ---------------------------------------------------------------- batch building
def load_smiles(limit=None):
    smis, seen = [], set()
    for csv in sorted((HERE / "data").glob("*.csv")):
        head = csv.open().readline().lower().split(",")
        col = next((i for i, h in enumerate(head) if "smile" in h), None)
        if col is None:
            continue
        for line in csv.open().readlines()[1:]:
            parts = line.rstrip("\n").split(",")
            if col >= len(parts):
                continue
            s = parts[col].strip().strip('"')
            if s and s not in seen and Chem.MolFromSmiles(s) is not None:
                seen.add(s)
                smis.append(s)
            if limit and len(smis) >= limit:
                return smis
    return smis


def build_batch(smis, feat):
    mgs = [feat(Chem.MolFromSmiles(s)) for s in smis]
    mgs = [g for g in mgs if g.edge_index.shape[1] > 0]
    V = torch.from_numpy(np.concatenate([g.V for g in mgs])).float()
    E = torch.from_numpy(np.concatenate([g.E for g in mgs]).astype(np.float32))
    eis, revs, oa, oe = [], [], 0, 0
    for g in mgs:
        eis.append(g.edge_index + oa)
        revs.append(g.rev_edge_index + oe)
        oa += g.V.shape[0]
        oe += g.edge_index.shape[1]
    return dict(
        V=V, E=E,
        ei=torch.from_numpy(np.hstack(eis)).long(),
        rev=torch.from_numpy(np.concatenate(revs)).long(),
        n_mol=len(mgs),
    )


# ------------------------------------------------------- Hashimoto bucketisation
def build_buckets(ei, rev, n_v):
    """Group directed edges by nnz(B row) = deg(src)-1.

    Returns {k: (rows, idx)} where idx has shape (len(rows), k) holding, for each
    edge in rows, the k contributing edge indices.  k=0 rows get no entry (M=0).
    Also returns the row-nnz histogram.
    """
    ei_np, rev_np = ei.numpy(), rev.numpy()
    n_e = ei_np.shape[1]
    in_edges = [[] for _ in range(n_v)]
    for e in range(n_e):
        in_edges[ei_np[1, e]].append(e)

    per_k = {}
    nnz = np.empty(n_e, dtype=np.int64)
    for e in range(n_e):
        a = ei_np[0, e]
        r = rev_np[e]
        contrib = [f for f in in_edges[a] if f != r]
        k = len(contrib)
        nnz[e] = k
        if k:
            per_k.setdefault(k, ([], []))
            per_k[k][0].append(e)
            per_k[k][1].append(contrib)

    buckets = {
        k: (torch.tensor(rows, dtype=torch.long), torch.tensor(idx, dtype=torch.long))
        for k, (rows, idx) in sorted(per_k.items())
    }
    hist = np.bincount(nnz) / n_e
    return buckets, nnz, hist


# ------------------------------------------------------------------ message forms
def msg_factored(H, ei, rev, n_v):
    """Chemprop today: .repeat'd int64 index + scatter_reduce_(include_self=False)."""
    idx = ei[1].unsqueeze(1).repeat(1, H.shape[1])
    allsum = torch.zeros(n_v, H.shape[1], dtype=H.dtype, device=H.device).scatter_reduce_(
        0, idx, H, reduce="sum", include_self=False
    )
    return allsum[ei[0]] - H[rev]


def msg_index_add(H, ei, rev, n_v):
    """Same algebra, 1-D index, no materialised index tensor."""
    allsum = torch.zeros(n_v, H.shape[1], dtype=H.dtype, device=H.device).index_add_(0, ei[1], H)
    return allsum.index_select(0, ei[0]) - H.index_select(0, rev)


def msg_hashimoto(H, buckets, n_e):
    """Explicit B: per-bucket gather + sum.  Never forms a sum it then subtracts from."""
    M = torch.zeros(n_e, H.shape[1], dtype=H.dtype, device=H.device)
    for k, (rows, idx) in buckets.items():
        M[rows] = H.index_select(0, idx.reshape(-1)).view(len(rows), k, -1).sum(1)
    return M


# ----------------------------------------------------------------------- harness
def timeit(fn, reps, sync=None):
    fn()
    if sync:
        sync()
    best = float("inf")
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        if sync:
            sync()
        best = min(best, time.perf_counter() - t)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="100 molecules, 2 reps")
    ap.add_argument("--reps", type=int, default=5)
    a = ap.parse_args()

    feat = SimpleMoleculeMolGraphFeaturizer()
    smis = load_smiles(limit=100 if a.quick else None)
    b = build_batch(smis, feat)
    V, E, ei, rev = b["V"], b["E"], b["ei"], b["rev"]
    n_v, n_e = V.shape[0], ei.shape[1]
    reps = 2 if a.quick else a.reps

    # ---- P4: structure of B
    deg = np.bincount(ei[1].numpy(), minlength=n_v)
    md, md2 = deg.mean(), (deg.astype(float) ** 2).mean()
    t0 = time.perf_counter()
    buckets, nnz, hist = build_buckets(ei, rev, n_v)
    t_build = time.perf_counter() - t0
    pred_nnz = (md2 - md) / md

    res = {
        "n_mol": b["n_mol"], "n_atom": int(n_v), "n_dir_bond": int(n_e),
        "mean_degree": float(md), "mean_deg_sq": float(md2),
        "nnz_per_row_measured": float(nnz.mean()),
        "nnz_per_row_predicted": float(pred_nnz),
        "row_nnz_hist": {int(i): float(v) for i, v in enumerate(hist)},
        "empty_row_frac": float(hist[0]) if len(hist) else 0.0,
        "crossover_lhs_mean_d2": float(md2),
        "crossover_rhs_4d_plus_2": float(4 * md + 2),
        "explicit_B_wins_crossover": bool(md2 < 4 * md + 2),
        "bucket_build_seconds": t_build,
        "bucket_sizes": {int(k): int(len(r)) for k, (r, _) in buckets.items()},
        "timings": {},
        "exactness": {},
    }
    print(f"{b['n_mol']} molecules | {n_v} atoms | {n_e} directed bonds")
    print(f"mean degree {md:.3f}  mean(d^2) {md2:.3f}")
    print(f"nnz(B)/row measured {nnz.mean():.3f}  predicted (m(d2)-m(d))/m(d) = {pred_nnz:.3f}")
    print(f"empty rows {hist[0]:.1%}  hist {[round(float(x), 4) for x in hist[:6]]}")
    print(f"crossover: mean(d^2)={md2:.2f} < 4*mean(d)+2={4*md+2:.2f} -> "
          f"{'EXPLICIT B WINS' if md2 < 4*md+2 else 'factored wins'}")

    # ---- exactness in fp64 (the claim is algebraic equality, not approximation)
    H64 = torch.randn(n_e, 64, dtype=torch.float64)
    bk64 = {k: (r, i) for k, (r, i) in buckets.items()}
    m_fac = msg_factored(H64, ei, rev, n_v)
    m_ia = msg_index_add(H64, ei, rev, n_v)
    m_hash = msg_hashimoto(H64, bk64, n_e)
    res["exactness"] = {
        "fp64_max_abs_factored_vs_hashimoto": float((m_fac - m_hash).abs().max()),
        "fp64_max_abs_factored_vs_index_add": float((m_fac - m_ia).abs().max()),
        "fp64_rel_factored_vs_hashimoto": float((m_fac - m_hash).norm() / m_fac.norm()),
    }
    print(f"\nfp64 exactness vs factored: hashimoto {res['exactness']['fp64_max_abs_factored_vs_hashimoto']:.3e}"
          f"  index_add {res['exactness']['fp64_max_abs_factored_vs_index_add']:.3e}")

    # ---- P6: bf16 cancellation, stratified by degree of src
    H32 = torch.randn(n_e, 256)
    ref = msg_factored(H32.double(), ei, rev, n_v)
    strat = {}
    for form, fn in (("factored", lambda h: msg_factored(h, ei, rev, n_v)),
                     ("hashimoto", lambda h: msg_hashimoto(h, buckets, n_e))):
        got = fn(H32.bfloat16()).float().double()
        err = ((got - ref).norm(dim=1) / ref.norm(dim=1).clamp(min=1e-12)).numpy()
        src_deg = deg[ei[0].numpy()]
        strat[form] = {
            int(d): float(np.median(err[src_deg == d]))
            for d in sorted(set(src_deg.tolist())) if (src_deg == d).sum() > 10
        }
    res["bf16_rel_err_median_by_src_degree"] = strat
    print(f"bf16 median rel err by src degree:\n  factored  {strat['factored']}\n"
          f"  hashimoto {strat['hashimoto']}")

    # ---- timings: message step alone, and the W_h GEMM alone (P5 roofline)
    devices = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])
    for dev in devices:
        eid, revd = ei.to(dev), rev.to(dev)
        bkd = {k: (r.to(dev), i.to(dev)) for k, (r, i) in buckets.items()}
        sync = torch.mps.synchronize if dev == "mps" else None
        for d_h in (300, 2048):
            H = torch.randn(n_e, d_h, device=dev)
            W = torch.randn(d_h, d_h, device=dev)
            tags = {
                "msg_factored_repeat": lambda: msg_factored(H, eid, revd, n_v),
                "msg_index_add": lambda: msg_index_add(H, eid, revd, n_v),
                "msg_hashimoto_bucketed": lambda: msg_hashimoto(H, bkd, n_e),
                "Wh_gemm_only": lambda: H @ W,
            }
            with torch.no_grad():
                for name, fn in tags.items():
                    ms = timeit(fn, reps, sync) * 1e3
                    res["timings"][f"{dev}|d_h={d_h}|{name}"] = ms
                    print(f"  {dev:4s} d_h={d_h:<5d} {name:26s} {ms:9.3f} ms")

    OUT.mkdir(exist_ok=True)
    res["caveat"] = (
        "Timings are message-step microbenchmarks on one machine (Apple Silicon, no CUDA). "
        "Exactness is verified in fp64 and is algebraic. The bucket build cost is "
        "amortisable into collate (once per batch, reused every depth iteration) but is "
        "reported separately and is NOT included in the message-step timings."
    )
    (OUT / "results_hashimoto.json").write_text(json.dumps(res, indent=2))
    print(f"\nwrote {OUT / 'results_hashimoto.json'}")


if __name__ == "__main__":
    main()
