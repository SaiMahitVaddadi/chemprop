"""Audit of what "Bemis-Murcko scaffold split" actually means across implementations.

Produces the three tables in briefs/post-opportunities-brief.md section 1:
  (a) Jaccard overlap of test sets across splitter implementations and seeds
  (b) acyclic-molecule fraction per split (the empty-scaffold cluster)
  (c) duplicate SMILES / conflicting labels per dataset
  (d) leakage audit: near-duplicate rate and Tanimoto-to-train, per splitter

IMPORTANT CAVEAT. The three splitters below are *faithful re-implementations* read out
of upstream source, not the installed packages:
  - astartes_split   <- astartes/main.py::_extrapolative_sampling
                        + samplers/abstract_sampler.py::get_sorted_cluster_counter
                        (what chemprop v2 `--split scaffold_balanced` calls)
  - v1_split         <- chemprop v1 chemprop/data/scaffold.py::scaffold_split(balanced=True)
  - deepchem_split   <- deepchem/splits/splitters.py::ScaffoldSplitter
Neither `chemprop` nor `astartes` imports in the measurement env (descriptastorus
missing), which is why these exist. Re-derive against the real packages before
publishing anything from this script.

Usage:  uv run scripts/split_audit.py [--data-dir PATH]
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path
from random import Random

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")

# (filename, smiles column, target column or None)
DATASETS = [
    ("delaney-processed.csv", "smiles", "measured log solubility in mols per litre"),
    ("SAMPL.csv", "smiles", "expt"),
    ("Lipophilicity.csv", "smiles", "exp"),
    ("bace.csv", "mol", "pIC50"),
    ("BBBP.csv", "smiles", "p_np"),
    ("HIV.csv", "smiles", "HIV_active"),
]

_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def load(path: Path, smi_col: str) -> list[tuple[str, Chem.Mol]]:
    out = []
    with open(path) as f:
        for row in csv.DictReader(f):
            smi = row[smi_col].strip()
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                out.append((smi, mol))
    return out


def scaffold_clusters(mols) -> dict[str, list[int]]:
    """Bemis-Murcko scaffold -> member indices. All acyclic molecules map to ""."""
    clusters: dict[str, list[int]] = defaultdict(list)
    for i, (_, mol) in enumerate(mols):
        clusters[MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)].append(i)
    return clusters


def astartes_split(mols, train=0.8, val=0.1, test=0.1, random_state=0):
    """chemprop v2 `--split scaffold_balanced`, i.e. astartes sampler="scaffold".

    Clusters sorted ASCENDING by size; every cluster of size <= min(n_test, n_val) is
    shuffled with `random_state`; then fill test first, val second, remainder -> train.
    """
    n = len(mols)
    clusters = scaffold_clusters(mols)
    n_train, n_val, n_test = (math.floor(n * f) for f in (train, val, test))
    max_shuf = min(n_test, n_val) if val > 0 else min(n_train, n_test)

    items = sorted(clusters.items(), key=lambda kv: len(kv[1]))
    small = [kv for kv in items if len(kv[1]) <= max_shuf]
    big = [kv for kv in items if len(kv[1]) > max_shuf]
    np.random.default_rng(seed=random_state).shuffle(small)

    tr, va, te = [], [], []
    for _, idxs in small + big:
        if len(te) + len(idxs) <= n_test:
            te += idxs
        elif len(va) + len(idxs) <= n_val:
            va += idxs
        else:
            tr += idxs
    return tr, va, te


def v1_split(mols, sizes=(0.8, 0.1, 0.1), seed=0):
    """chemprop v1 scaffold_split(balanced=True).

    Clusters bigger than half the val or test size go in a "big" bucket, the rest in
    "small"; each bucket shuffled with Random(seed); concatenated big-then-small; then
    greedy fill train first.
    """
    n = len(mols)
    tr_cap, va_cap, te_cap = (f * n for f in sizes)
    big, small = [], []
    for idxs in scaffold_clusters(mols).values():
        (big if (len(idxs) > va_cap / 2 or len(idxs) > te_cap / 2) else small).append(idxs)
    rng = Random(seed)
    rng.shuffle(big)
    rng.shuffle(small)

    tr, va, te = [], [], []
    for idxs in big + small:
        if len(tr) + len(idxs) <= tr_cap:
            tr += idxs
        elif len(va) + len(idxs) <= va_cap:
            va += idxs
        else:
            te += idxs
    return tr, va, te


def deepchem_split(mols, sizes=(0.8, 0.1, 0.1)):
    """deepchem ScaffoldSplitter: clusters sorted DESCENDING, greedy fill train first.
    Fully deterministic -- its `seed` argument is accepted and never referenced."""
    n = len(mols)
    sets = sorted(scaffold_clusters(mols).values(), key=lambda s: (len(s), s[0]), reverse=True)
    tr_cut, va_cut = sizes[0] * n, (sizes[0] + sizes[1]) * n
    tr, va, te = [], [], []
    for idxs in sets:
        if len(tr) + len(idxs) > tr_cut:
            if len(tr) + len(va) + len(idxs) > va_cut:
                te += idxs
            else:
                va += idxs
        else:
            tr += idxs
    return tr, va, te


def random_split(mols, sizes=(0.8, 0.1, 0.1), seed=0):
    idx = list(range(len(mols)))
    Random(seed).shuffle(idx)
    n = len(mols)
    a, b = int(sizes[0] * n), int((sizes[0] + sizes[1]) * n)
    return idx[:a], idx[a:b], idx[b:]


def acyclic_frac(mols, idxs) -> float:
    if not idxs:
        return float("nan")
    return sum(1 for i in idxs if mols[i][1].GetRingInfo().NumRings() == 0) / len(idxs)


def leakage_audit(mols, tr, te):
    """Definitions follow Schuh et al. 2026 / Walters 2024 so numbers are comparable."""
    fp_tr = [_GEN.GetFingerprint(mols[i][1]) for i in tr]
    scaf_tr = {MurckoScaffold.MurckoScaffoldSmiles(mol=mols[i][1]) for i in tr}
    near = shared = 0
    top5, maxsim = [], []
    for i in te:
        sims = np.sort(np.array(DataStructs.BulkTanimotoSimilarity(_GEN.GetFingerprint(mols[i][1]), fp_tr)))
        maxsim.append(sims[-1])
        top5.append(sims[-5:].mean())
        near += sims[-1] >= 0.9
        shared += MurckoScaffold.MurckoScaffoldSmiles(mol=mols[i][1]) in scaf_tr
    n = len(te)
    return near / n, shared / n, float(np.mean(top5)), float(np.mean(maxsim))


SPLITTERS = [
    ("random 80/10/10", random_split),
    ("astartes (cp v2)", astartes_split),
    ("chemprop v1 balanced", v1_split),
    ("deepchem", deepchem_split),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parents[1] / "data")
    args = ap.parse_args()

    for fname, smi_col, tgt_col in DATASETS:
        path = args.data_dir / fname
        if not path.exists():
            print(f"\n### {fname}: not found, skipping")
            continue
        mols = load(path, smi_col)
        n = len(mols)
        clusters = scaffold_clusters(mols)
        n_acyclic = sum(1 for _, m in mols if m.GetRingInfo().NumRings() == 0)
        print(f"\n### {fname}  n={n}  scaffolds={len(clusters)}")
        print(
            f"    acyclic={n_acyclic} ({n_acyclic / n:.1%})"
            f"  empty-scaffold cluster={len(clusters.get('', []))}"
            f"  largest cluster={max(len(v) for v in clusters.values())}"
            f"  singletons={sum(1 for v in clusters.values() if len(v) == 1)}"
        )

        # (c) duplicates and label conflicts
        with open(path) as f:
            rows = list(csv.DictReader(f))
        if tgt_col in rows[0]:
            canon: dict[str, list[str]] = defaultdict(list)
            unparsable = 0
            for row in rows:
                mol = Chem.MolFromSmiles(row[smi_col].strip())
                if mol is None:
                    unparsable += 1
                    continue
                canon[Chem.MolToSmiles(mol)].append(row[tgt_col])
            dup = {k: v for k, v in canon.items() if len(v) > 1}
            conflict = {k: v for k, v in dup.items() if len(set(v)) > 1}
            spreads = []
            for vals in conflict.values():
                try:
                    f_vals = [float(x) for x in vals]
                except ValueError:
                    continue
                spreads.append(max(f_vals) - min(f_vals))
            extra = (
                f" max_spread={max(spreads):.2f} median_spread={statistics.median(spreads):.2f}"
                if spreads
                else ""
            )
            print(
                f"    unparsable={unparsable} unique_canonical={len(canon)}"
                f" dup_groups={len(dup)} dup_rows={sum(len(v) for v in dup.values())}"
                f" conflicting_labels={len(conflict)}{extra}"
            )

        # (b) + (d) per splitter
        hdr = f"    {'split':24s} {'sizes':>18s} {'acyc tr/te':>14s} {'near>=.9':>9s} {'scafshare':>10s} {'top5 Tc':>8s} {'max Tc':>7s}"
        print(hdr)
        tests = {}
        for name, fn in SPLITTERS:
            tr, va, te = fn(mols)
            tests[name] = set(te)
            a, b, c, d = leakage_audit(mols, tr, te)
            print(
                f"    {name:24s} {len(tr)/n:.2f}/{len(va)/n:.2f}/{len(te)/n:.2f}".ljust(47)
                + f"{acyclic_frac(mols, tr):>6.1%}/{acyclic_frac(mols, te):<7.1%}"
                f"{a:>9.3f} {b:>10.3f} {c:>8.3f} {d:>7.3f}"
            )

        # (a) test-set agreement, including astartes against itself on a new seed
        tests["astartes seed 1"] = set(astartes_split(mols, random_state=1)[2])
        jac = lambda x, y: len(x & y) / len(x | y) if (x | y) else float("nan")  # noqa: E731
        pairs = [
            ("astartes (cp v2)", "astartes seed 1"),
            ("astartes (cp v2)", "chemprop v1 balanced"),
            ("astartes (cp v2)", "deepchem"),
            ("chemprop v1 balanced", "deepchem"),
        ]
        print("    test-set Jaccard: " + "  ".join(f"J({a},{b})={jac(tests[a], tests[b]):.3f}" for a, b in pairs))


if __name__ == "__main__":
    main()
