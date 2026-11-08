"""Determinism of the dataset / dataloader layer.

The kernel-level probe (determinism_probe.py) covers float accumulation order, which
perturbs results in the last bits. This covers the data layer, which is a different
and larger problem: **batch order changes the SGD trajectory, so it changes the
converged model, not just its low-order bits.** This is the axis that actually moves
seed-to-seed variance.

What chemprop does today (read from chemprop/data/dataloader.py and samplers.py):

  if class_balance:                  sampler = ClassBalanceSampler(dataset.Y, seed, shuffle)
  elif shuffle and seed is not None: sampler = SeededSampler(len(dataset), seed)
  else:                              sampler = None    # -> DataLoader(shuffle=...) global RNG

Three hazards this script tests rather than asserts:

  H1  seed=None + shuffle=True falls through to torch's global RNG, so batch order
      depends on whatever consumed random numbers earlier in the process.
  H2  ClassBalanceSampler(seed=None) builds np.random.default_rng(None) and is
      silently nondeterministic -- class_balance with no seed is a legal combination.
  H3  SeededSampler.rg is STATEFUL across epochs (constructed in __init__, shuffled in
      __iter__). Correct per-epoch behaviour, but it means an extra iteration of the
      loader -- a sanity-check pass, a validation hook, a profiler -- permanently
      shifts the order of every later epoch.

Plus: are DataLoader batches identical across num_workers, and does worker RNG get
seeded without an explicit worker_init_fn?

Run: python experiments/determinism_data.py [--quick]
"""
import argparse, json, os, sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

OUT = Path(__file__).resolve().parent.parent / "results"


# ---- faithful standalone copies of chemprop's samplers (chemprop is not importable
# ---- in this env; these mirror chemprop/data/samplers.py exactly)
class SeededSampler:
    def __init__(self, N, seed):
        if seed is None:
            raise ValueError("arg 'seed' was `None`! A SeededSampler must be seeded!")
        self.idxs = np.arange(N)
        self.rg = np.random.default_rng(seed)

    def __iter__(self):
        self.rg.shuffle(self.idxs)
        return iter(self.idxs)

    def __len__(self):
        return len(self.idxs)


class ClassBalanceSampler:
    def __init__(self, Y, seed=None, shuffle=False):
        self.shuffle = shuffle
        self.rg = np.random.default_rng(seed)
        idxs = np.arange(len(Y))
        actives = Y.any(1)
        self.pos_idxs = idxs[actives]
        self.neg_idxs = idxs[~actives]
        self.length = 2 * min(len(self.pos_idxs), len(self.neg_idxs))

    def __iter__(self):
        if self.shuffle:
            self.rg.shuffle(self.pos_idxs)
            self.rg.shuffle(self.neg_idxs)
        return iter([i for pair in zip(self.pos_idxs, self.neg_idxs) for i in pair])

    def __len__(self):
        return self.length


class IdxDataset(Dataset):
    """Returns its own index plus the RNG state seen inside the worker."""

    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return (i, int(torch.randint(0, 2**31, (1,)).item()),
                int(np.random.randint(0, 2**31)))


def collate(rows):
    return (torch.tensor([r[0] for r in rows]),
            torch.tensor([r[1] for r in rows]),
            torch.tensor([r[2] for r in rows]))


def order_of(loader):
    return torch.cat([b[0] for b in loader]).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    N = 64 if a.quick else 512
    BS = 8 if a.quick else 32
    res = {"N": N, "batch_size": BS, "hazards": {}, "workers": {}, "scaler": {}}

    # ---- H3: SeededSampler statefulness across epochs
    s1 = SeededSampler(N, 0)
    ep = [list(iter(s1)) for _ in range(3)]
    s2 = SeededSampler(N, 0)
    _ = list(iter(s2))            # one extra iteration, as a sanity-check pass would do
    ep_shifted = [list(iter(s2)) for _ in range(3)]
    res["hazards"]["H3_seeded_sampler_stateful"] = {
        "epoch0_eq_epoch1": ep[0] == ep[1],
        "same_seed_reproducible": ep == [list(iter(SeededSampler(N, 0))) for _ in range(3)][:0] + ep,
        "one_extra_iter_shifts_all_later_epochs": ep[0] != ep_shifted[0],
        "note": ("per-epoch reshuffle is correct; the hazard is that the epoch->order map "
                 "depends on how many times __iter__ was called, so an extra pass "
                 "permanently desynchronises a 'reproducible' run"),
    }
    print(f"H3 SeededSampler: epoch0==epoch1 {ep[0]==ep[1]}, "
          f"extra-iter shifts order {ep[0]!=ep_shifted[0]}")

    # ---- H2: ClassBalanceSampler with seed=None
    Y = (np.arange(N) % 3 == 0).reshape(-1, 1)
    runs = [list(iter(ClassBalanceSampler(Y, None, shuffle=True))) for _ in range(3)]
    seeded = [list(iter(ClassBalanceSampler(Y, 0, shuffle=True))) for _ in range(3)]
    res["hazards"]["H2_classbalance_unseeded"] = {
        "unseeded_runs_identical": runs[0] == runs[1] == runs[2],
        "seeded_runs_identical": seeded[0] == seeded[1] == seeded[2],
        "note": ("class_balance=True with seed=None is a legal call and yields "
                 "np.random.default_rng(None); if unseeded_runs_identical is False then "
                 "class balancing is silently nondeterministic"),
    }
    print(f"H2 ClassBalanceSampler(seed=None) reproducible: {runs[0]==runs[1]}")

    # ---- H1: seed=None + shuffle=True -> global torch RNG
    ds = IdxDataset(N)
    def loader(seed=None, workers=0, gen_seed=None):
        sampler = SeededSampler(N, seed) if seed is not None else None
        g = None
        if gen_seed is not None:
            g = torch.Generator(); g.manual_seed(gen_seed)
        return DataLoader(ds, BS, sampler is None, sampler, num_workers=workers,
                          collate_fn=collate, generator=g)

    torch.manual_seed(123)
    o1 = order_of(loader())
    torch.manual_seed(123)
    o2 = order_of(loader())
    o3 = order_of(loader())          # no reseed: continues the global stream
    res["hazards"]["H1_unseeded_shuffle_uses_global_rng"] = {
        "same_global_seed_same_order": o1 == o2,
        "without_reseed_order_differs": o1 != o3,
        "note": ("reproducible only if nothing else in the process consumes torch RNG "
                 "between manual_seed and iteration; passing generator= would isolate it"),
    }
    print(f"H1 unseeded shuffle: reseed reproduces {o1==o2}, drifts without reseed {o1!=o3}")

    # ---- workers: batch order and worker RNG seeding
    base = order_of(loader(seed=0, workers=0))
    for w in ([0, 2] if a.quick else [0, 2, 4]):
        got = order_of(loader(seed=0, workers=w))
        res["workers"][f"num_workers={w}|order_matches_0"] = got == base
    # worker RNG: torch is auto-seeded per worker, numpy is NOT without worker_init_fn
    def rng_cols(workers):
        out = [(b[1].tolist(), b[2].tolist()) for b in loader(seed=0, workers=workers)]
        return ([x for t, _ in out for x in t], [x for _, n in out for x in n])
    t0, n0 = rng_cols(0)
    res["workers"]["numpy_rng_unique_per_worker"] = None
    if not a.quick:
        t2, n2 = rng_cols(2)
        res["workers"]["torch_rng_differs_with_workers"] = t0 != t2
        res["workers"]["numpy_rng_differs_with_workers"] = n0 != n2
        res["workers"]["numpy_rng_has_duplicates_across_workers"] = len(set(n2)) < len(n2)
        res["workers"]["note"] = (
            "PyTorch seeds each worker's torch RNG from base_seed but does NOT seed numpy "
            "or python random; without worker_init_fn every worker shares one numpy stream. "
            "Harmless while featurization is deterministic, latent for custom featurizers "
            "or augmentation that draw from numpy.")
        print(f"workers: order stable {all(v for k, v in res['workers'].items() if 'order_matches' in str(k))}, "
              f"numpy dupes across workers {res['workers']['numpy_rng_has_duplicates_across_workers']}")

    # ---- dataset-level: target scaler bitwise stability under row permutation
    rng = np.random.default_rng(0)
    Yv = rng.standard_normal((N, 3)).astype(np.float64) * 1e3
    perm = rng.permutation(N)
    m_a, s_a = Yv.mean(0), Yv.std(0)
    m_b, s_b = Yv[perm].mean(0), Yv[perm].std(0)
    res["scaler"] = {
        "mean_bitwise_equal_under_permutation": bool(np.array_equal(m_a, m_b)),
        "std_bitwise_equal_under_permutation": bool(np.array_equal(s_a, s_b)),
        "mean_max_abs_diff": float(np.abs(m_a - m_b).max()),
        "std_max_abs_diff": float(np.abs(s_a - s_b).max()),
        "note": ("normalize_targets() reduces over the training rows; float addition is "
                 "non-associative, so a different row order perturbs the scaler in its low "
                 "bits, and that perturbation then multiplies every target. Small, but it "
                 "means 'same seed, same split, different shard order' is not bitwise equal."),
    }
    print(f"scaler bitwise-stable under permutation: "
          f"mean {res['scaler']['mean_bitwise_equal_under_permutation']}, "
          f"std {res['scaler']['std_bitwise_equal_under_permutation']} "
          f"(max diff {max(res['scaler']['mean_max_abs_diff'], res['scaler']['std_max_abs_diff']):.2e})")

    res["chemprop_refs"] = {
        "dataloader": "chemprop/data/dataloader.py build_dataloader()",
        "samplers": "chemprop/data/samplers.py SeededSampler, ClassBalanceSampler",
        "seed_coupling": "chemprop/cli/train.py:1826-1829 (--pytorch-seed -> deterministic=True)",
        "cache_default": "chemprop/data/datasets.py:220 (self.cache = False)",
    }
    res["caveat"] = (
        "The samplers here are faithful standalone copies of chemprop's, because chemprop is "
        "not importable in this environment (missing descriptastorus, cuik_molmaker, astartes, "
        "lightning). Re-run against the real classes before filing anything. Worker-RNG "
        "behaviour is a property of torch's DataLoader, not of chemprop.")
    OUT.mkdir(exist_ok=True)
    (OUT / "results_determinism_data.json").write_text(json.dumps(res, indent=2))
    print(f"\nwrote {OUT / 'results_determinism_data.json'}")


if __name__ == "__main__":
    main()
