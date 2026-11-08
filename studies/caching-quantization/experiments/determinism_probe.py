"""Does nondeterminism break the cache's exactness claim?

Part 2 of the series claims that two directed bonds sharing a radius-r WL key have
identical hidden states, "verified to <=7e-7".  That 7e-7 needs an explanation: it is
either (a) imprecision in the key, which would be a correctness problem, or (b) float
accumulation-order nondeterminism in the scatter, which is a property of the device
and not of the cache.  This distinguishes them.

It also measures:
  * whether torch.use_deterministic_algorithms() actually delivers determinism per
    device (on MPS in torch 2.11 it neither raises nor works)
  * what deterministic mode costs in wall-clock
  * whether the Hashimoto gather-and-add form is deterministic *by construction*
    (no atomics, no contention) and therefore needs no flag

Relevant to chemprop: cli/train.py couples --pytorch-seed to lightning's
deterministic=True.  If the flag is a no-op on a device, --pytorch-seed promises
reproducibility it cannot deliver.

Run: python experiments/determinism_probe.py [--quick]
"""
import argparse, json, platform, sys, time, warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from hashimoto_bench import build_batch, build_buckets, load_smiles, msg_factored, \
    msg_hashimoto, msg_index_add, timeit  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from cp_loader import SimpleMoleculeMolGraphFeaturizer  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "results"
CKPT = Path.home() / "Library/Caches/molfeat/chemeleon/chemeleon_mp.pt"


# --------------------------------------------------------------- WL keys (exact)
def s64(v):
    return int(np.int64(np.uint64(v)))


C1, C2, GOLD = s64(0xBF58476D1CE4E5B9), s64(0x94D049BB133111EB), s64(0x9E3779B97F4A7C15)


def mix(x):
    x = x ^ (x >> 30)
    x = x * C1
    x = x ^ (x >> 27)
    x = x * C2
    return x ^ (x >> 31)


def row_keys(T):
    b = T.contiguous().view(torch.int32).view(T.shape[0], -1).long()
    k = torch.full((T.shape[0],), GOLD, dtype=torch.int64)
    for j in range(b.shape[1]):
        k = mix(k ^ b[:, j])
    return k


@torch.no_grad()
def wl_keys(V, E, ei, rev, rounds):
    """Per-layer int64 key per directed edge; order-invariant additive mixing."""
    a = row_keys(V)
    k0 = mix(a[ei[0]] ^ mix(row_keys(E)))
    keys, k, n_v = [k0], k0, V.shape[0]
    for _ in range(rounds):
        km = mix(k)
        acc = torch.zeros(n_v, dtype=torch.int64).index_add_(0, ei[1], km)
        k = mix(k0 ^ mix(acc[ei[0]] - km[rev]))
        keys.append(k)
    return keys


def within_key_spread(H, keys):
    """Max divergence between edges that share a key. 0 == the cache is bitwise exact."""
    groups = defaultdict(list)
    for i, k in enumerate(keys.tolist()):
        groups[k].append(i)
    worst, n_groups, n_members = 0.0, 0, 0
    Hc = H.detach().float().cpu()
    for rows in groups.values():
        if len(rows) < 2:
            continue
        n_groups += 1
        n_members += len(rows)
        blk = Hc[rows]
        worst = max(worst, float((blk - blk[0]).abs().max()))
    return worst, n_groups, n_members


# -------------------------------------------------------------------- the forward
@torch.no_grad()
def forward(V, E, ei, rev, W_i, W_h, W_o, b_o, depth, form, buckets=None):
    H_0 = torch.nn.functional.linear(torch.cat([V[ei[0]], E], 1), W_i)
    H = torch.relu(H_0)
    n_v, n_e = V.shape[0], ei.shape[1]
    per_layer = []
    for _ in range(1, depth):
        if form == "factored":
            M = msg_factored(H, ei, rev, n_v)
        elif form == "index_add":
            M = msg_index_add(H, ei, rev, n_v)
        else:
            M = msg_hashimoto(H, buckets, n_e)
        H = torch.relu(H_0 + torch.nn.functional.linear(M, W_h))
        per_layer.append(H)
    return H, per_layer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--d-h", type=int, default=2048)
    a = ap.parse_args()
    reps = 2 if a.quick else a.repeats
    d_h = 300 if a.quick else a.d_h
    depth = 3 if a.quick else a.depth

    feat = SimpleMoleculeMolGraphFeaturizer()
    b = build_batch(load_smiles(limit=100 if a.quick else 500), feat)
    V, E, ei, rev = b["V"], b["E"], b["ei"], b["rev"]
    n_v, n_e = V.shape[0], ei.shape[1]
    buckets_cpu, _, _ = build_buckets(ei, rev, n_v)
    keys = wl_keys(V, E, ei, rev, depth - 1)

    if CKPT.exists() and d_h == 2048:
        sd = torch.load(CKPT, weights_only=True, map_location="cpu")["state_dict"]
        W_i, W_h, W_o, b_o = (sd["W_i.weight"], sd["W_h.weight"],
                              sd["W_o.weight"], sd["W_o.bias"])
    else:
        torch.manual_seed(0)
        W_i = torch.randn(d_h, 86) * 0.05
        W_h = torch.randn(d_h, d_h) * (1 / d_h) ** 0.5
        W_o = torch.randn(d_h, 72 + d_h) * 0.02
        b_o = torch.zeros(d_h)

    res = {
        "env": {"torch": torch.__version__, "platform": platform.platform(),
                "machine": platform.machine()},
        "batch": {"n_mol": b["n_mol"], "n_atom": int(n_v), "n_dir_bond": int(n_e),
                  "d_h": d_h, "depth": depth},
        "op_level": {}, "forward_level": {}, "cache_exactness": {}, "cost": {},
    }
    devices = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])
    print(f"{b['n_mol']} mols | {n_v} atoms | {n_e} bonds | d_h={d_h} depth={depth}\n")

    # ---- 1. op level: does the flag deliver determinism, and what does it cost?
    print(f"{'device':6}{'det':6}{'op':24}{'repeatable':12}{'ms':>9}")
    for dev in devices:
        eid, revd = ei.to(dev), rev.to(dev)
        H = torch.randn(n_e, d_h, device=dev)
        sync = torch.mps.synchronize if dev == "mps" else None
        for det in (False, True):
            torch.use_deterministic_algorithms(det)
            try:
                for nm, fn in (("index_add_", lambda: msg_index_add(H, eid, revd, n_v)),
                               ("scatter_reduce_", lambda: msg_factored(H, eid, revd, n_v))):
                    try:
                        outs = [fn() for _ in range(reps)]
                        rep = all(torch.equal(outs[0], o) for o in outs[1:])
                        spread = max(float((outs[0] - o).abs().max()) for o in outs[1:])
                        ms = timeit(fn, reps, sync) * 1e3
                        res["op_level"][f"{dev}|det={det}|{nm}"] = {
                            "bitwise_repeatable": rep, "max_spread": spread, "ms": ms}
                        print(f"{dev:6}{str(det):6}{nm:24}{str(rep):12}{ms:>9.3f}")
                    except Exception as e:
                        res["op_level"][f"{dev}|det={det}|{nm}"] = {
                            "raised": f"{type(e).__name__}: {str(e)[:120]}"}
                        print(f"{dev:6}{str(det):6}{nm:24}RAISES {type(e).__name__}")
            finally:
                torch.use_deterministic_algorithms(False)

    # ---- 2. forward level, and 3. the cache-exactness question
    print(f"\n{'device':6}{'det':6}{'form':12}{'fwd repeat':12}"
          f"{'within-key spread':>20}{'groups':>8}")
    for dev in devices:
        Vd, Ed, eid, revd = V.to(dev), E.to(dev), ei.to(dev), rev.to(dev)
        bk = {k: (r.to(dev), i.to(dev)) for k, (r, i) in buckets_cpu.items()}
        Wi, Wh, Wo, bo = (W_i.to(dev), W_h.to(dev), W_o.to(dev), b_o.to(dev))
        for det in (False, True):
            torch.use_deterministic_algorithms(det)
            try:
                for form in ("factored", "index_add", "hashimoto"):
                    try:
                        runs = [forward(Vd, Ed, eid, revd, Wi, Wh, Wo, bo, depth, form, bk)
                                for _ in range(2)]
                        H1, pl1 = runs[0]
                        H2, _ = runs[1]
                        rep = bool(torch.equal(H1, H2))
                        drift = float((H1 - H2).abs().max())
                        # exactness of the cache key at the deepest layer
                        spread, ng, nm_ = within_key_spread(pl1[-1], keys[depth - 1])
                        res["forward_level"][f"{dev}|det={det}|{form}"] = {
                            "bitwise_repeatable": rep, "run_to_run_max_abs": drift}
                        res["cache_exactness"][f"{dev}|det={det}|{form}"] = {
                            "within_key_max_abs": spread, "n_multi_member_groups": ng,
                            "n_members": nm_}
                        print(f"{dev:6}{str(det):6}{form:12}{str(rep):12}"
                              f"{spread:>20.3e}{ng:>8}")
                    except Exception as e:
                        print(f"{dev:6}{str(det):6}{form:12}RAISES {type(e).__name__}")
            finally:
                torch.use_deterministic_algorithms(False)

    res["interpretation"] = {
        "claim_under_test": (
            "Part 2 claims same-WL-key edges agree to <=7e-7. This separates key "
            "imprecision from float accumulation-order nondeterminism."),
        "read_this_way": (
            "within_key_max_abs == 0.0 means the cache is BITWISE exact on that device/form, "
            "so any nonzero figure quoted elsewhere is a device artifact and not a property "
            "of the WL key. A nonzero spread on a form that is bitwise_repeatable would be "
            "the alarming case: it would mean the key is genuinely imprecise."),
        "chemprop_coupling": (
            "cli/train.py:1826-1829 sets lightning deterministic=True iff --pytorch-seed is "
            "given. If op_level shows bitwise_repeatable=False under det=True on a device, "
            "then --pytorch-seed promises reproducibility that device cannot deliver, without "
            "raising."),
        "caveat": (
            "Run-to-run repeatability is tested over a small number of repeats in one process; "
            "absence of observed divergence is weaker evidence than a determinism guarantee. "
            "The Hashimoto form is expected to be deterministic by construction because it is "
            "gather-and-add with no atomics and no write contention."),
    }
    OUT.mkdir(exist_ok=True)
    (OUT / "results_determinism.json").write_text(json.dumps(res, indent=2))
    print(f"\nwrote {OUT / 'results_determinism.json'}")


if __name__ == "__main__":
    main()
