import sys
sys.path.insert(0, "/private/tmp/chemprop-quant-study")
import glob
from collections import defaultdict
import numpy as np
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")
from cp_loader import SimpleMoleculeMolGraphFeaturizer

MAX_DEPTH = 6  # CheMeleon depth

def load_smiles():
    smis, seen = [], set()
    for f in sorted(glob.glob("tests/data/**/*.csv", recursive=True)):
        try:
            with open(f) as fh:
                hdr = fh.readline().split(",")
                if not any("smi" in h.lower() for h in hdr):
                    continue
                ci = [i for i, h in enumerate(hdr) if "smi" in h.lower()][0]
                for line in fh:
                    s = line.split(",")[ci].strip()
                    if not s or s in seen:
                        continue
                    mol = Chem.MolFromSmiles(s)
                    if mol is None or mol.GetNumBonds() == 0:
                        continue
                    seen.add(s); smis.append(s)
        except Exception:
            continue
    return smis

def hash_mol(mg, max_depth):
    """Return (edge_hashes_per_round, atom_keys_per_round) for one molecule."""
    V, E, ei, rev = mg.V, mg.E, mg.edge_index, mg.rev_edge_index
    n_e = ei.shape[1]
    a_inv = [hash(V[i].astype(np.float32).tobytes()) for i in range(V.shape[0])]
    # round-0 edge hash mirrors W_i(V[src] || E[e])
    h = [hash((a_inv[ei[0, e]], hash(E[e].astype(np.float32).tobytes()))) for e in range(n_e)]
    h0 = list(h)
    in_edges = defaultdict(list)          # node -> incoming directed edges
    for e in range(n_e):
        in_edges[ei[1, e]].append(e)
    edge_rounds = [list(h)]
    atom_rounds = []
    for _ in range(max_depth):
        # atom-level key available from the current edge round
        atom_rounds.append([
            hash((a_inv[v], tuple(sorted(h[f] for f in in_edges[v])))) for v in range(V.shape[0])
        ])
        nxt = []
        for e in range(n_e):
            src = ei[0, e]
            msg = tuple(sorted(h[f] for f in in_edges[src] if f != rev[e]))
            nxt.append(hash((h0[e], msg)))
        h = nxt
        edge_rounds.append(list(h))
    return edge_rounds, atom_rounds

def main():
    smis = load_smiles()
    print(f"molecules: {len(smis)}")
    feat = SimpleMoleculeMolGraphFeaturizer()
    edge_seen = [set() for _ in range(MAX_DEPTH + 1)]
    atom_seen = [set() for _ in range(MAX_DEPTH)]
    n_edges = n_atoms = 0
    raw_atom, raw_bond = set(), set()
    curve = []
    for i, s in enumerate(smis):
        mol = Chem.MolFromSmiles(s)
        mg = feat(mol)
        er, ar = hash_mol(mg, MAX_DEPTH)
        for r, hs in enumerate(er):
            edge_seen[r].update(hs)
        for r, hs in enumerate(ar):
            atom_seen[r].update(hs)
        raw_atom.update(mg.V[i2].astype(np.float32).tobytes() for i2 in range(mg.V.shape[0]))
        raw_bond.update(mg.E[i2].astype(np.float32).tobytes() for i2 in range(mg.E.shape[0]))
        n_edges += mg.edge_index.shape[1]; n_atoms += mg.V.shape[0]
        if (i + 1) % 250 == 0:
            curve.append((i + 1, n_atoms, len(atom_seen[2]), len(atom_seen[MAX_DEPTH - 1])))
    print(f"total directed edges: {n_edges}   total atoms: {n_atoms}"
          f"   avg heavy atoms/mol: {n_atoms/len(smis):.1f}")
    print("\n--- INPUT GRAPH PRECOMPUTE: raw feature-vector reuse ---")
    print(f"unique atom feature vectors: {len(raw_atom)} / {n_atoms} atoms"
          f"  -> {1-len(raw_atom)/n_atoms:.2%} of atom featurizer calls are redundant")
    print(f"unique bond feature vectors: {len(raw_bond)} / {n_edges} directed bonds"
          f"  -> {1-len(raw_bond)/n_edges:.2%} of bond featurizer calls are redundant")
    print("\n--- directed-bond environments (hidden state after r message rounds) ---")
    print(f"{'r':>2} {'unique':>8} {'/total':>9} {'reuse factor':>13} {'max hit rate':>13}")
    for r in range(MAX_DEPTH + 1):
        u = len(edge_seen[r])
        print(f"{r:>2} {u:>8} {u/n_edges:>8.1%} {n_edges/u:>13.2f}x {1-u/n_edges:>12.1%}")
    print("\n--- atom environments (final W_o input, i.e. depth = r+1) ---")
    print(f"{'depth':>5} {'unique':>8} {'/total':>9} {'max hit rate':>13}")
    for r in range(MAX_DEPTH):
        u = len(atom_seen[r])
        print(f"{r+1:>5} {u:>8} {u/n_atoms:>8.1%} {1-u/n_atoms:>12.1%}")
    print("\n--- saturation: how fast the cache stops growing ---")
    print(f"{'mols':>6} {'atoms':>7} {'uniq r=2':>9} {'uniq r=5':>9} {'new/atom r=2':>13}")
    prev = (0, 0)
    for nm, na, u2, u5 in curve:
        print(f"{nm:>6} {na:>7} {u2:>9} {u5:>9} {(u2-prev[0])/(na-prev[1]):>13.2%}")
        prev = (u2, na)

main()
