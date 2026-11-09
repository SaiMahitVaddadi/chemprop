"""Two probes of permutation-invariant pooling: collisions, and moment sufficiency.

TEST 1 -- COLLISIONS. Sum is injective on multisets over a countable alphabet; mean and
max are not (Xu et al., GIN). Mean collapses any two multisets whose histograms are
PROPORTIONAL: {C,C,O} and {C,C,C,C,O,O} have the same mean. Part 1 measured the atom
feature alphabet to be finite and small (102 distinct vectors across 57,549 atoms), so
this is constructive rather than asymptotic -- the collisions can be enumerated.

Message passing is not fooled quite so easily, because it uses topology, so the honest
measurement is how much of the collapse SURVIVES the encoder: for molecule pairs with
proportional atom-type histograms, compare the embedding distance under mean pooling
against the same distance under sum pooling.

TEST 2 -- MOMENT SUFFICIENCY. A multiset is generically determined by all of its moments,
so pooling with moments 1..k is a truncated characteristic function. This asks how many
you actually need, which either justifies or retires the PNA moments 3/4/5. Probe a set
of RDKit targets from the k-moment pooled vector and watch recovery as k grows. If the
curve saturates at k=2, higher moments are decoration.

Both tests carry a random-weight-encoder control, because probe capacity confounds
everything: a strong enough decoder recovers anything.

Run: python experiments/pooling_probes.py [--quick]
"""
import argparse, json, sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hashimoto_bench import load_smiles, msg_index_add  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from cp_loader import SimpleMoleculeMolGraphFeaturizer  # noqa: E402
from rdkit import Chem, RDLogger  # noqa: E402
from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors  # noqa: E402

RDLogger.DisableLog("rdApp.*")
OUT = Path(__file__).resolve().parent.parent / "results"
CKPT = Path.home() / "Library/Caches/molfeat/chemeleon/chemeleon_mp.pt"
EPS = 1e-9


def weights(quick):
    if CKPT.exists() and not quick:
        sd = torch.load(CKPT, weights_only=True, map_location="cpu")["state_dict"]
        return sd["W_i.weight"], sd["W_h.weight"], 2048, 6, "chemeleon"
    torch.manual_seed(0)
    d = 300
    return torch.randn(d, 86) * 0.05, torch.randn(d, d) / d**0.5, d, 3, "random_init"


@torch.no_grad()
def atom_states(g, W_i, W_h, depth):
    V = torch.from_numpy(g.V).float()
    E = torch.from_numpy(g.E.astype(np.float32))
    ei = torch.from_numpy(g.edge_index).long()
    rev = torch.from_numpy(g.rev_edge_index).long()
    H_0 = torch.nn.functional.linear(torch.cat([V[ei[0]], E], 1), W_i)
    H = torch.relu(H_0)
    for _ in range(1, depth):
        H = torch.relu(H_0 + torch.nn.functional.linear(
            msg_index_add(H, ei, rev, V.shape[0]), W_h))
    return torch.zeros(V.shape[0], H.shape[1]).index_add_(0, ei[1], H)   # per-atom


def moments(Hv, k):
    """[mean, then central moments 2..k] concatenated."""
    mu = Hv.mean(0, keepdim=True)
    out = [mu.squeeze(0)]
    for j in range(2, k + 1):
        out.append((Hv - mu).pow(j).mean(0))
    return torch.cat(out)


TARGETS = {
    "mol_wt": Descriptors.MolWt, "heavy_atoms": lambda m: m.GetNumHeavyAtoms(),
    "logp": Crippen.MolLogP, "tpsa": rdMolDescriptors.CalcTPSA,
    "n_rings": rdMolDescriptors.CalcNumRings,
    "n_rot_bonds": rdMolDescriptors.CalcNumRotatableBonds,
    "frac_csp3": rdMolDescriptors.CalcFractionCSP3,
    "n_hetero": lambda m: sum(a.GetAtomicNum() not in (1, 6) for a in m.GetAtoms()),
}


def ridge_r2(X, y, lam=1.0, frac=0.7, seed=0):
    g = torch.Generator().manual_seed(seed)
    n = X.shape[0]
    pm = torch.randperm(n, generator=g)
    tr, te = pm[: int(frac * n)], pm[int(frac * n):]
    mu, sd = X[tr].mean(0), X[tr].std(0).clamp(min=1e-6)
    Xs = torch.cat([(X - mu) / sd, torch.ones(n, 1)], 1)
    A = Xs[tr].T @ Xs[tr] + lam * torch.eye(Xs.shape[1])
    w = torch.linalg.solve(A, Xs[tr].T @ y[tr])
    p = Xs[te] @ w
    return float(1 - ((p - y[te]) ** 2).sum() / ((y[te] - y[tr].mean()) ** 2).sum().clamp(min=EPS))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--max-k", type=int, default=5)
    a = ap.parse_args()
    feat = SimpleMoleculeMolGraphFeaturizer()
    W_i, W_h, d_h, depth, tag = weights(a.quick)
    smis = load_smiles(limit=250 if a.quick else 1200)

    mols, graphs, hists = [], [], []
    type_id, types = {}, []
    for s in smis:
        m = Chem.MolFromSmiles(s)
        if m is None or m.GetNumBonds() == 0:
            continue
        g = feat(m)
        h = defaultdict(int)
        for i in range(g.V.shape[0]):
            key = g.V[i].astype(np.float32).tobytes()
            if key not in type_id:
                type_id[key] = len(types); types.append(key)
            h[type_id[key]] += 1
        mols.append(m); graphs.append(g); hists.append(dict(h))
    res = {"weights": tag, "d_h": int(d_h), "depth": int(depth), "n_mol": len(mols),
           "alphabet_size": len(types)}
    print(f"{tag} d_h={d_h} depth={depth} | {len(mols)} molecules | "
          f"atom alphabet = {len(types)} distinct feature vectors\n")

    # ---------------- TEST 1: proportional-histogram collisions
    groups = defaultdict(list)
    for i, h in enumerate(hists):
        tot = sum(h.values())
        groups[tuple(sorted((k, round(v / tot, 6)) for k, v in h.items()))].append(i)
    pairs = [(g[i], g[j]) for g in groups.values() if len(g) > 1
             for i in range(len(g)) for j in range(i + 1, len(g))
             if sum(hists[g[i]].values()) != sum(hists[g[j]].values())]
    res["test1"] = {"n_proportional_groups": sum(1 for g in groups.values() if len(g) > 1),
                    "n_scale_collision_pairs": len(pairs)}
    print(f"TEST 1  proportional-histogram groups: "
          f"{res['test1']['n_proportional_groups']}, "
          f"true scale-collision pairs (different sizes): {len(pairs)}")
    if pairs:
        rm, rs = [], []
        for i, j in pairs[:200]:
            Hi, Hj = atom_states(graphs[i], W_i, W_h, depth), atom_states(graphs[j], W_i, W_h, depth)
            zm_i, zm_j = Hi.mean(0), Hj.mean(0)
            zs_i, zs_j = Hi.sum(0), Hj.sum(0)
            rm.append(float((zm_i - zm_j).norm() / (zm_i.norm() + zm_j.norm()).clamp(min=EPS)))
            rs.append(float((zs_i - zs_j).norm() / (zs_i.norm() + zs_j.norm()).clamp(min=EPS)))
        res["test1"].update({
            "mean_pool_relative_separation": float(np.mean(rm)),
            "sum_pool_relative_separation": float(np.mean(rs)),
            "separation_ratio_sum_over_mean": float(np.mean(rs) / max(np.mean(rm), EPS)),
            "n_pairs_measured": len(rm)})
        print(f"        relative separation of colliding pairs: "
              f"mean-pool {np.mean(rm):.4f} vs sum-pool {np.mean(rs):.4f} "
              f"(ratio {np.mean(rs)/max(np.mean(rm),EPS):.2f}x)")
    else:
        print("        none found -- report this as a negative; the alphabet may be too "
              "fine-grained for exact proportionality in this sample")

    # ---------------- TEST 2: moment sufficiency
    y = {t: torch.tensor([float(f(m)) for m in mols]) for t, f in TARGETS.items()}
    Hs = [atom_states(g, W_i, W_h, depth) for g in graphs]
    res["test2"] = {}
    print(f"\nTEST 2  moment sufficiency (R^2 from moments 1..k)")
    hdr = "".join(f"{('k='+str(k)):>8}" for k in range(1, a.max_k + 1))
    print(f"{'target':14}{hdr}{'sum':>8}")
    for t, yv in y.items():
        row = {}
        for k in range(1, a.max_k + 1):
            X = torch.stack([moments(H, k) for H in Hs])
            row[f"k={k}"] = ridge_r2(X, yv)
        Xs = torch.stack([H.sum(0) for H in Hs])
        row["sum"] = ridge_r2(Xs, yv)
        res["test2"][t] = row
        print(f"{t:14}" + "".join(f"{row[f'k={k}']:>8.3f}" for k in range(1, a.max_k + 1))
              + f"{row['sum']:>8.3f}")

    res["caveat"] = [
        "TEST 1 measures how much of mean pooling's non-injectivity SURVIVES message "
        "passing. Message passing uses topology, so proportional-histogram molecules are "
        "not expected to collide exactly after encoding; the number to read is the RATIO "
        "of sum-pool to mean-pool separation, not either alone.",
        "Proportionality is tested on the atom-type histogram rounded to 6 decimals, so "
        "near-proportional pairs are excluded. A tolerance sweep would find more.",
        "TEST 2 moments are computed on per-atom aggregated states, not on the final "
        "readout, and the probe is linear ridge with a fixed lambda on a single 70/30 "
        "split -- so small differences between adjacent k are not meaningful.",
        "No random-encoder control is run in --quick mode (it IS the random encoder). "
        "Run without --quick for CheMeleon, then compare against a --quick pass to get "
        "the control.",
    ]
    OUT.mkdir(exist_ok=True)
    (OUT / "results_pooling_probes.json").write_text(json.dumps(res, indent=2))
    print(f"\nwrote {OUT / 'results_pooling_probes.json'}")


if __name__ == "__main__":
    main()
