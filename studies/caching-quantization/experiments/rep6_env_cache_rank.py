"""EXP 6 -- the low-rank question for the object the cache actually stores.

rep1 measures the rank of the POOLED MOLECULE embedding. The environment cache in
this project stores something else: the per-EDGE hidden state H, 2048-d, at every
message-passing layer. This script measures that object directly, then tests
low-rank projection as a cache format end to end:

    cache_q(H) = mu + (H - mu) V_k V_k^T   applied after EVERY layer

which is the exact hook qlib.encode already exposes for quantized caches, so the
result is directly comparable to the int4-group-32 environment-cache number from
the companion study (+0.25% RMSE at 7.5x on the final embedding; the per-layer
environment-cache variant is in results_main).

Memory accounting: int4 group-32 costs 4.25 bits/dim x 2048 = 1088 B/edge. A
rank-k cache with fp32 coordinates costs 4k B/edge, so k=272 is break-even and
anything above that is a loss. The basis itself (2048 x k fp32) is amortised.
"""
import sys
import time

import numpy as np
import torch

import rep_lib as R
sys.path.insert(0, R.SCRIPTS)
import qlib

RANKS = [16, 32, 64, 128, 256, 512, 1024]
DS = sys.argv[1].split(",") if len(sys.argv) > 1 else ["esol", "lipo", "bace_reg"]
NSPLIT = 4
SAMPLE_EDGES = 20000


def collect_H(chunks, Wi, Wh, Wo, bo, depth, d_h, rng, per_layer=SAMPLE_EDGES):
    """Run the forward pass and sample per-edge hidden states from every layer."""
    acc = {d: [] for d in range(depth)}
    budget = {d: per_layer for d in range(depth)}
    for V, E, ei, rev, bt, nm in chunks:
        n_v = V.shape[0]
        H_0 = torch.nn.functional.linear(torch.cat([V[ei[0]], E], 1), Wi)
        H = torch.relu(H_0)
        idx = ei[1].unsqueeze(1).expand(-1, d_h)
        for d in range(depth):
            if d > 0:
                M = torch.zeros(n_v, d_h).scatter_reduce_(
                    0, idx, H, reduce="sum", include_self=False)[ei[0]] - H[rev]
                H = torch.relu(H_0 + torch.nn.functional.linear(M, Wh))
            if budget[d] > 0:
                k = min(budget[d], H.shape[0])
                sel = rng.choice(H.shape[0], k, replace=False)
                acc[d].append(H[sel].numpy().copy())
                budget[d] -= k
    return {d: np.concatenate(v) for d, v in acc.items() if v}


def main():
    out = {"caveat": [
        "The low-rank basis here is fitted on edge hidden states sampled from the "
        "WHOLE dataset, including test molecules. That is leakage, and it makes this "
        "an OPTIMISTIC upper bound on what a deployed low-rank cache could achieve. "
        "It is deliberate: an optimistic bound is enough to rule the idea out if it "
        "still loses to int4.",
        "One shared basis is used for every layer (a real cache stores one format). "
        "A per-layer basis would be slightly better and is not tested.",
        "The projection is applied in fp32 arithmetic; only the RANK is restricted. "
        "Real storage would also quantize the k coordinates, which would add error.",
        "ReLU makes H non-negative, so its covariance has a large mean component; the "
        "rank is reported for both the centered and the uncentered second moment.",
        "The sparse byte counts are a LAYOUT estimate, not a measured kernel. They "
        "assume the ReLU zero pattern is stored exactly (so no accuracy loss beyond "
        "the value quantization) and ignore alignment, padding and the cost of "
        "gather/scatter on a sparse layout, which on real hardware can easily eat "
        "the saving. No sparse kernel was benchmarked.",
        "Edge counts: a rank-k cache stores 4k bytes/edge in fp32 vs 1088 bytes/edge "
        "for int4 group-32 at d_h=2048, so k=272 is the break-even point.",
    ], "ranks": RANKS, "sample_edges_per_layer": SAMPLE_EDGES}

    Wi, Wh, Wo, bo, d_h, depth = qlib.load_chemeleon()
    res = {}
    for nm in DS:
        D = R.load(nm)
        Y, tt = D["Y"], D["task_type"]
        chunks = qlib.build_chunks(D["smiles"], 400)
        rng = np.random.default_rng(0)
        t = time.perf_counter()
        Hs = collect_H(chunks, Wi, Wh, Wo, bo, depth, d_h, rng)
        print(f"\n{nm}: sampled per-layer edge states in {time.perf_counter()-t:.0f}s "
              f"({ {d: v.shape for d, v in Hs.items()} })")

        spec = {}
        for d, Hd in Hs.items():
            lam = R.spectrum(Hd)
            st = R.rank_stats(lam, Hd)
            # uncentered second moment too (ReLU output is non-negative)
            s = np.linalg.svd(Hd.astype(np.float64), compute_uv=False)
            stu = R.rank_stats(s ** 2 / max(len(Hd) - 1, 1))
            st["uncentered_participation_ratio"] = stu["participation_ratio"]
            st["uncentered_n95"] = stu["n95"]
            st["frac_zero_entries"] = float((Hd == 0).mean())
            nnz = (Hd != 0).sum(1)
            alive = (np.abs(Hd).max(0) > 0).sum()
            st["nnz_per_edge_mean"] = float(nnz.mean())
            st["nnz_per_edge_p99"] = float(np.percentile(nnz, 99))
            st["n_alive_dims"] = int(alive)
            # sparse cache formats, bytes/edge, vs 1088 B/edge for int4 group-32
            st["bytes_per_edge"] = dict(
                fp32_dense=8192,
                int4_g32_dense=1088,
                bitmask_plus_int8=int(alive / 8 + nnz.mean() * 1),
                bitmask_plus_int4=int(alive / 8 + nnz.mean() * 0.5 + alive / 32 * 1),
                csr_int8_u16idx=int(nnz.mean() * 3),
            )
            spec[d] = st
            print(f"  layer {d}: PR={st['participation_ratio']:7.2f} "
                  f"eff_rank={st['entropy_eff_rank']:7.2f} n90={st['n90']:>4} "
                  f"n95={st['n95']:>4} n99={st['n99']:>4} "
                  f"uncentPR={st['uncentered_participation_ratio']:7.2f} "
                  f"zeros={st['frac_zero_entries']*100:.1f}% "
                  f"nnz/edge={st['nnz_per_edge_mean']:.0f} alive={st['n_alive_dims']} "
                  f"bitmask+int4={st['bytes_per_edge']['bitmask_plus_int4']}B/edge")

        # shared basis across layers
        Hall = np.concatenate([Hs[d][: SAMPLE_EDGES // depth] for d in Hs])
        mu = Hall.astype(np.float64).mean(0)
        _, S, Vt = np.linalg.svd(Hall.astype(np.float64) - mu, full_matrices=False)
        sp = D["splits"][:NSPLIT]

        def run(cq):
            Z = qlib.encode(chunks, Wi, Wh, Wo, bo, depth, d_h, cache_q=cq)
            return np.array([R.fit_eval(Z.numpy(), Y, tr, te, tt, R.LAM_DOWNSTREAM)
                             for sn, tr, te in sp])

        base = run(None)
        rec = {"fp32": dict(mean=float(np.nanmean(base)), delta_pct=0.0,
                            bytes_per_edge=8192)}
        print(f"  fp32 cache: {np.nanmean(base):.4f}")
        muT = torch.from_numpy(mu).float()
        for k in RANKS:
            Vk = torch.from_numpy(np.ascontiguousarray(Vt[:k])).float()

            def cq(H, Vk=Vk, muT=muT):
                return ((H - muT) @ Vk.T) @ Vk + muT
            v = run(cq)
            d = (v - base) / base * 100
            if tt != "reg":
                d = -d
            rec[f"rank{k}"] = dict(mean=float(np.nanmean(v)),
                                   delta_pct=float(np.nanmean(d)),
                                   delta_sd=float(np.nanstd(d)),
                                   bytes_per_edge=4 * k)
            print(f"  rank {k:>5}: {np.nanmean(v):.4f}  delta={np.nanmean(d):+.2f}% "
                  f"({4*k} B/edge vs 1088 B/edge for int4-g32)")
        # int4 group-32 reference, identical hook
        v = run(lambda H: qlib.intq(H, 4, 32))
        d = (v - base) / base * 100
        if tt != "reg":
            d = -d
        rec["int4_g32"] = dict(mean=float(np.nanmean(v)), delta_pct=float(np.nanmean(d)),
                               delta_sd=float(np.nanstd(d)), bytes_per_edge=1088)
        print(f"  int4 g32 : {np.nanmean(v):.4f}  delta={np.nanmean(d):+.2f}% (1088 B/edge)")
        res[nm] = dict(task_type=tt, spectrum_per_layer=spec, cache_formats=rec,
                       singular_values_top128=(S[:128] ** 2 / max(len(Hall) - 1, 1)).tolist())
    out["per_dataset"] = res
    R.jdump(out, f"{R.RESULTS}/results_rep6_env_cache_rank.json")


main()
