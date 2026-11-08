"""Study (D): environment-cache hit rates on REAL datasets.

Two numbers per dataset and radius r:
  * intra-dataset uniqueness   = 1 - unique_r / total  (theoretical max hit rate if
    the whole dataset is encoded once with a cold cache)
  * warm-cache hit rate        = fraction of test-molecule environments already
    present in a cache populated from the 80% train split (the realistic
    deployment number for inference on new molecules)

Keys are the exact non-backtracking WL refinement of the bond-message recursion
(same keying as cached_mpnn3.py): k_0(e) = h(atom_type(src(e)), bond_type(e)) and
k_{t+1}(e) = h(k_0(e), sum_{f: dest(f)=src(e), f != rev(e)} h(k_t(f))). Two edges
share a key iff their depth-t hidden states are provably identical, so the unique
count is the exact size of the smallest correct cache.
"""
import json
import os
import pickle

import numpy as np
import torch

import qlib

CACHE = os.path.join(qlib.SCRATCH, "cache")
MAX_R = 6


def s64(v):
    return int(np.int64(np.uint64(v)))


C1, C2, GOLD = s64(0xbf58476d1ce4e5b9), s64(0x94d049bb133111eb), s64(0x9e3779b97f4a7c15)


def mix(x):
    x = x ^ (x >> 30)
    x = x * C1
    x = x ^ (x >> 27)
    x = x * C2
    return x ^ (x >> 31)


def wl_keys(aid, bid, ei, rev, n_v, rounds):
    """Returns edge keys per round 0..rounds and atom keys per depth 1..rounds+1."""
    k0 = mix(mix(aid[ei[0]] * GOLD) ^ mix(bid * C1))
    k = k0
    ekeys, akeys = [k0], []
    for _ in range(rounds + 1):
        km = mix(k)
        acc = torch.zeros(n_v, dtype=torch.int64).index_add_(0, ei[1], km)
        # atom environment: [V[v] || sum_{f: dest(f)=v} H_f]  -> W_o input
        akeys.append(mix(mix(aid * GOLD) ^ mix(acc)))
        k = mix(k0 ^ mix(acc[ei[0]] - km[rev]))
        ekeys.append(k)
    return ekeys, akeys


def dataset_keys(smis):
    """Featurize and compute WL keys per molecule, with globally consistent type IDs."""
    from rdkit import Chem
    feat = qlib.featurizer()
    a_tab, b_tab = {}, {}
    per_mol = []
    for s in smis:
        mg = feat(Chem.MolFromSmiles(s))
        V, E, ei, rev = mg.V.astype(np.float32), mg.E.astype(np.float32), mg.edge_index, mg.rev_edge_index
        aid = np.empty(V.shape[0], np.int64)
        for i in range(V.shape[0]):
            k = V[i].tobytes()
            aid[i] = a_tab.setdefault(k, len(a_tab))
        bid = np.empty(E.shape[0], np.int64)
        for i in range(E.shape[0]):
            k = E[i].tobytes()
            bid[i] = b_tab.setdefault(k, len(b_tab))
        ek, ak = wl_keys(torch.from_numpy(aid), torch.from_numpy(bid),
                         torch.from_numpy(ei).long(), torch.from_numpy(rev).long(),
                         V.shape[0], MAX_R)
        per_mol.append(([e.numpy() for e in ek], [a.numpy() for a in ak],
                        V.shape[0], E.shape[0]))
    return per_mol, len(a_tab), len(b_tab)


def main():
    meta = json.load(open(f"{CACHE}/meta.json"))
    out = {}
    print(f"{'dataset':<10}{'n':>6}{'atoms':>7}{'edges':>7}{'atomT':>7}{'bondT':>7}")
    allkeys = {r: set() for r in range(MAX_R + 1)}
    tot_e_all = 0
    for name in meta:
        with open(f"{CACHE}/{name}.pkl", "rb") as fh:
            D = pickle.load(fh)
        smis, splits = D["smiles"], D["splits"]
        per_mol, nat, nbt = dataset_keys(smis)
        n_e = sum(p[3] for p in per_mol)
        n_a = sum(p[2] for p in per_mol)
        tot_e_all += n_e
        print(f"{name:<10}{len(smis):>6}{n_a:>7}{n_e:>7}{nat:>7}{nbt:>7}", flush=True)
        rec = dict(n=len(smis), n_atoms=n_a, n_edges=n_e, n_atom_types=nat,
                   n_bond_types=nbt, edge=[], atom=[], warm=[], warm_atom=[])
        for r in range(MAX_R + 1):
            u = set()
            for ek, _, _, _ in per_mol:
                u.update(ek[r].tolist())
            rec["edge"].append(dict(r=r, unique=len(u), total=n_e,
                                    max_hit=1 - len(u) / n_e, reuse=n_e / len(u)))
            allkeys[r].update(u)
        for r in range(MAX_R + 1):
            u = set()
            for _, ak, _, _ in per_mol:
                u.update(ak[r].tolist())
            rec["atom"].append(dict(depth=r + 1, unique=len(u), total=n_a,
                                    max_hit=1 - len(u) / n_a))
        # warm-cache hit rate: cache populated from train, measured on test
        for sn, tr, te in splits:
            w, wa = [], []
            for r in range(MAX_R + 1):
                seen = set()
                for i in tr:
                    seen.update(per_mol[i][0][r].tolist())
                hit = tot = 0
                for i in te:
                    k = per_mol[i][0][r].tolist()
                    hit += sum(1 for v in k if v in seen)
                    tot += len(k)
                w.append(hit / max(tot, 1))
                seen_a = set()
                for i in tr:
                    seen_a.update(per_mol[i][1][r].tolist())
                hit = tot = 0
                for i in te:
                    k = per_mol[i][1][r]
                    hit += sum(1 for v in k.tolist() if v in seen_a)
                    tot += len(k)
                wa.append(hit / max(tot, 1))
            rec["warm"].append(dict(split=sn, edge_hit=w))
            rec["warm_atom"].append(dict(split=sn, atom_hit=wa))
        out[name] = rec
    out["_pooled"] = dict(n_edges=tot_e_all,
                          edge=[dict(r=r, unique=len(allkeys[r]), total=tot_e_all,
                                     max_hit=1 - len(allkeys[r]) / tot_e_all)
                                for r in range(MAX_R + 1)])
    with open(f"{qlib.SCRATCH}/results_cache_hits.json", "w") as fh:
        json.dump(out, fh, indent=1)

    # ---- report ----
    print("\n=== directed-bond environment reuse (max hit rate = 1 - unique/total) ===")
    print(f"{'dataset':<10}" + "".join(f"{'r='+str(r):>8}" for r in range(MAX_R + 1)))
    for name in meta:
        print(f"{name:<10}" + "".join(f"{d['max_hit']*100:>7.1f}%" for d in out[name]["edge"]))
    print(f"{'POOLED':<10}" + "".join(f"{d['max_hit']*100:>7.1f}%" for d in out["_pooled"]["edge"]))
    print("\n=== warm-cache hit rate on NEW molecules (cache from 80% train) ===")
    print("    random splits (mean over 6)        /  scaffold splits (mean over 4)")
    print(f"{'dataset':<10}" + "".join(f"{'r='+str(r):>8}" for r in range(MAX_R + 1))
          + "  |" + "".join(f"{'r='+str(r):>8}" for r in range(MAX_R + 1)))
    for name in meta:
        rnd = np.mean([w["edge_hit"] for w in out[name]["warm"]
                       if not w["split"].startswith("scaf")], 0)
        scf = np.mean([w["edge_hit"] for w in out[name]["warm"]
                       if w["split"].startswith("scaf")], 0)
        print(f"{name:<10}" + "".join(f"{v*100:>7.1f}%" for v in rnd) + "  |"
              + "".join(f"{v*100:>7.1f}%" for v in scf))
    print("\n=== atom environments (W_o input) at depth d, max hit rate ===")
    print(f"{'dataset':<10}" + "".join(f"{'d='+str(r+1):>8}" for r in range(MAX_R + 1)))
    for name in meta:
        print(f"{name:<10}" + "".join(f"{d['max_hit']*100:>7.1f}%" for d in out[name]["atom"]))


if __name__ == "__main__":
    main()
