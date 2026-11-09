"""Is there a "starting point" for message passing, and does choosing it help?

Tempting idea: start message passing from the most-connected atom, or a random one,
and see whether the choice matters. Before building that, the architecture has to admit
the question -- and Chemprop's does not, for a reason worth measuring rather than
asserting.

`_MessagePassingBase.forward` updates EVERY directed edge simultaneously, `depth` times,
with no ordering anywhere. The update is equivariant to atom reindexing and the readout
is a permutation-invariant pool, so the molecule representation is **invariant to which
atom you call first**. There is no starting point. Test 1 verifies this to float
precision by permuting atom indices and re-running.

So the study only becomes well posed once invariance is deliberately broken. Three ways,
in increasing invasiveness:

  A. ANCHOR POSITIONAL ENCODING (tested here). Pick an anchor atom by some rule, add
     shortest-path-distance-from-anchor as an extra atom feature. No architecture change
     -- it rides in through `--atom-features-path`. Selection rules compared: max degree,
     max betweenness-ish (sum of distances), Morgan canonical rank 0, random, and the
     graph centroid (min eccentricity).
  B. ANCHOR ENSEMBLING. Compute k rooted representations and pool them, recovering
     invariance by averaging over roots. Costs k x compute. Not implemented.
  C. ASYNCHRONOUS/ORDERED SCHEDULE. BFS or priority order from the anchor, as in Part 3.
     Argued down there on three grounds (the Hammett objection, canonical-rank
     instability under small edits, and serialisation killing batching). Not implemented.

Test 2 measures, for each selection rule: how stable the anchor is under a small edit
(the med-chem workload), and -- the consequence that matters for this project -- that
distance-from-anchor is a GLOBAL feature, so unlike RWSE (provably radius-3 local, Part
8) it is NOT environment-cacheable. Quantified as the fraction of atoms whose feature
changes when a single substituent is swapped.

Run: python experiments/anchor_study.py [--quick]
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hashimoto_bench import build_batch, load_smiles, msg_index_add  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from cp_loader import SimpleMoleculeMolGraphFeaturizer  # noqa: E402
from rdkit import Chem, RDLogger  # noqa: E402
from rdkit.Chem import rdmolops  # noqa: E402

RDLogger.DisableLog("rdApp.*")
OUT = Path(__file__).resolve().parent.parent / "results"
TEMPLATE = "O=C(Nc1ccc({R1})cc1)c1ccc({R2})cc1"
SUBS = ["C", "F", "Cl", "OC", "C(F)(F)F", "C#N", "N(C)C", "c1ccncc1", "N1CCOCC1"]


def encode(V, E, ei, rev, W_i, W_h, depth, batch, n_mol):
    H_0 = torch.nn.functional.linear(torch.cat([V[ei[0]], E], 1), W_i)
    H = torch.relu(H_0)
    for _ in range(1, depth):
        H = torch.relu(H_0 + torch.nn.functional.linear(
            msg_index_add(H, ei, rev, V.shape[0]), W_h))
    M = torch.zeros(V.shape[0], H.shape[1]).index_add_(0, ei[1], H)
    cnt = torch.zeros(n_mol).index_add_(0, batch, torch.ones(batch.shape[0])).clamp(min=1)
    return torch.zeros(n_mol, H.shape[1]).index_add_(0, batch, M) / cnt[:, None]


# ------------------------------------------------------------ anchor selection rules
def anchors(mol):
    n = mol.GetNumAtoms()
    D = rdmolops.GetDistanceMatrix(mol)
    deg = np.array([a.GetTotalDegree() for a in mol.GetAtoms()])
    ranks = np.array(list(Chem.CanonicalRankAtoms(mol, breakTies=True)))
    rng = np.random.default_rng(abs(hash(Chem.MolToSmiles(mol))) % (2**31))
    return {
        "max_degree": int(deg.argmax()),
        "min_total_distance": int(D.sum(1).argmin()),      # closeness centre
        "max_total_distance": int(D.sum(1).argmax()),       # periphery
        "centroid": int(D.max(1).argmin()),                 # min eccentricity
        "canonical_rank0": int(ranks.argmin()),
        "random": int(rng.integers(n)),
    }


def dist_feature(mol, anchor):
    return rdmolops.GetDistanceMatrix(mol)[anchor]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    feat = SimpleMoleculeMolGraphFeaturizer()
    res = {"tests": {}}

    # ---- TEST 1: is the representation invariant to atom indexing / "starting point"?
    smis = load_smiles(limit=40 if a.quick else 120)
    torch.manual_seed(0)
    d_h, depth = 300, 3
    W_i, W_h = torch.randn(d_h, 86) * 0.05, torch.randn(d_h, d_h) / d_h**0.5
    # Permute the FEATURIZED GRAPH directly. An earlier version renumbered the mol and
    # round-tripped through SMILES, which does not preserve atom order (the writer
    # reorders and the parser re-perceives), so it compared two different graphs and
    # reported a spurious 3e-2 non-invariance.
    worst = 0.0
    rng = np.random.default_rng(0)
    for s in smis[: 40 if a.quick else 120]:
        mol = Chem.MolFromSmiles(s)
        if mol is None or mol.GetNumBonds() == 0:
            continue
        b0 = build_batch([s], feat)
        n = b0["V"].shape[0]
        perm = torch.from_numpy(rng.permutation(n))        # new_index = perm[old_index]
        inv = torch.empty_like(perm)
        inv[perm] = torch.arange(n)
        V1 = b0["V"][perm]                                  # relabel atoms
        ei1 = inv[b0["ei"]]                                 # remap endpoints
        with torch.no_grad():
            z0 = encode(b0["V"], b0["E"], b0["ei"], b0["rev"], W_i, W_h, depth,
                        torch.zeros(n, dtype=torch.long), 1)
            z1 = encode(V1, b0["E"], ei1, b0["rev"], W_i, W_h, depth,
                        torch.zeros(n, dtype=torch.long), 1)
        worst = max(worst, float((z0 - z1).abs().max() / z0.abs().max().clamp(min=1e-12)))
    res["tests"]["permutation_invariance"] = {
        "max_relative_difference": worst,
        "verdict": ("INVARIANT - synchronous message passing has no starting point to choose"
                    if worst < 1e-5 else "NOT invariant - investigate"),
    }
    print(f"TEST 1  atom-reindexing invariance: max relative diff = {worst:.3e}")
    print(f"        -> {res['tests']['permutation_invariance']['verdict']}\n")

    # ---- TEST 2: anchor stability and locality under a single-substituent swap
    base = Chem.MolFromSmiles(TEMPLATE.replace("{R1}", "C").replace("{R2}", "C"))
    base_anch = anchors(base)
    rules = list(base_anch)
    stab = {r: {"anchor_moved": 0, "n": 0, "frac_atoms_feature_changed": []} for r in rules}
    for r2 in SUBS:
        m2 = Chem.MolFromSmiles(TEMPLATE.replace("{R1}", "C").replace("{R2}", r2))
        if m2 is None:
            continue
        a2 = anchors(m2)
        for r in rules:
            stab[r]["n"] += 1
            stab[r]["anchor_moved"] += int(a2[r] != base_anch[r])
            # locality: fraction of SHARED atoms whose distance-from-anchor changed
            d0, d1 = dist_feature(base, base_anch[r]), dist_feature(m2, a2[r])
            k = min(len(d0), len(d1))
            stab[r]["frac_atoms_feature_changed"].append(
                float((d0[:k] != d1[:k]).mean()))
    print(f"TEST 2  anchor stability under one R-group swap ({len(SUBS)} edits)")
    print(f"{'rule':22}{'anchor moved':>14}{'atoms w/ changed feature':>26}")
    for r in rules:
        v = stab[r]
        frac = float(np.mean(v["frac_atoms_feature_changed"]))
        stab[r]["anchor_moved_frac"] = v["anchor_moved"] / max(v["n"], 1)
        stab[r]["mean_frac_atoms_changed"] = frac
        stab[r].pop("frac_atoms_feature_changed")
        print(f"{r:22}{stab[r]['anchor_moved_frac']:>13.0%}{frac:>26.1%}")
    res["tests"]["anchor_stability"] = stab

    res["caveat"] = [
        "TEST 1 uses random weights; invariance is a property of the architecture, not of "
        "the weights, so this is a structural check rather than a model result. It permutes "
        "the featurized graph directly: an earlier version renumbered the molecule and "
        "round-tripped through SMILES, which does NOT preserve atom order, and reported a "
        "spurious 3e-2 non-invariance. Residual differences at the 1e-7 level are float "
        "summation order in index_add_, not a broken symmetry.",
        "Distance-from-anchor is a GLOBAL feature: it depends on the whole molecule, so "
        "unlike RWSE (provably a function of the radius-ceil(k/2) ball, Part 8) it is NOT "
        "environment-cacheable. The 'atoms with changed feature' column is the direct cost "
        "-- every changed atom is a cache miss under a single substituent swap.",
        "Anchor ensembling (pool over k roots to recover invariance) and asynchronous "
        "BFS/priority schedules are NOT implemented; Part 3 argues the latter down on "
        "physics, canonical-rank-instability and batching grounds.",
        "The edit series is one scaffold with 9 substituents -- a med-chem analog workload, "
        "not a general chemical sample.",
    ]
    OUT.mkdir(exist_ok=True)
    (OUT / "results_anchor_study.json").write_text(json.dumps(res, indent=2))
    print(f"\nwrote {OUT / 'results_anchor_study.json'}")


if __name__ == "__main__":
    main()
