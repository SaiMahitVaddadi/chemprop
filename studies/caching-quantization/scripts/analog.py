"""Med-chem workload: one scaffold, many small R-group edits (SAR enumeration)."""
import sys, time
sys.path.insert(0, "/private/tmp/chemprop-quant-study")
from collections import defaultdict
import numpy as np
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")
from cp_loader import SimpleMoleculeMolGraphFeaturizer

TEMPLATE = "O=C(Nc1ccc({R1})cc1)c1ccc({R2})cc1"   # biaryl amide, 2 vectors
SUBS = ["F","Cl","Br","C","CC","CCC","OC","OCC","C(F)(F)F","C#N","O","N","N(C)C",
        "S(=O)(=O)C","c1ccccc1","c1ccncc1","N1CCOCC1","N1CCCCC1","C(=O)O","C(=O)N"]

def build_library():
    mols, smis = [], []
    for r1 in SUBS:
        for r2 in SUBS:
            s = TEMPLATE.replace("{R1}", r1).replace("{R2}", r2)
            m = Chem.MolFromSmiles(s)
            if m is not None:
                mols.append(m); smis.append(Chem.MolToSmiles(m))
    return mols, smis

feat = SimpleMoleculeMolGraphFeaturizer()
MAXR = 6

def wl(mg, maxr):
    V,E,ei,rev = mg.V, mg.E, mg.edge_index, mg.rev_edge_index
    a = [hash(V[i].astype(np.float32).tobytes()) for i in range(V.shape[0])]
    h0 = [hash((a[ei[0,e]], hash(E[e].astype(np.float32).tobytes()))) for e in range(ei.shape[1])]
    h = list(h0); ine = defaultdict(list)
    for e in range(ei.shape[1]): ine[ei[1,e]].append(e)
    rounds=[list(h)]; arounds=[]
    for _ in range(maxr):
        arounds.append([hash((a[v], tuple(sorted(h[f] for f in ine[v])))) for v in range(V.shape[0])])
        h=[hash((h0[e], tuple(sorted(h[f] for f in ine[ei[0,e]] if f!=rev[e])))) for e in range(ei.shape[1])]
        rounds.append(list(h))
    return rounds, arounds

mols, smis = build_library()
print(f"analog library: {len(mols)} molecules from 1 scaffold x {len(SUBS)}x{len(SUBS)} R-groups")
print(f"avg heavy atoms: {np.mean([m.GetNumAtoms() for m in mols]):.1f}\n")

mgs = [feat(m) for m in mols]
seen = [set() for _ in range(MAXR+1)]
hits = [0]*(MAXR+1); tot = 0
marginal = []   # (mol#, per-radius hit rate so far this molecule)
for i, mg in enumerate(mols):
    rounds, _ = wl(mgs[i], MAXR)
    ne = mgs[i].edge_index.shape[1]; tot += ne
    per = []
    for r in range(MAXR+1):
        hs = rounds[r]
        h_i = sum(1 for x in hs if x in seen[r])
        hits[r] += h_i; seen[r].update(hs)
        per.append(h_i/ne)
    if i in (0,1,2,4,9,19,49,99,199,len(mols)-1):
        marginal.append((i+1, per))

print("--- cumulative cache hit rate over the enumeration (warm cache) ---")
print(f"{'radius':>7}{'unique envs':>13}{'hit rate':>11}")
for r in range(MAXR+1):
    print(f"{r:>7}{len(seen[r]):>13}{hits[r]/tot:>10.1%}")

print("\n--- marginal hit rate for the Nth analog (this is what a chemist feels) ---")
hdr = "".join(f"{('r=%d'%r):>8}" for r in range(MAXR+1))
print(f"{'mol #':>7}{hdr}")
for n, per in marginal:
    print(f"{n:>7}" + "".join(f"{p:>8.1%}" for p in per))

# ---- locality: how many atoms does a single R-group edit actually dirty? ----
print("\n--- locality of a single-site edit (radius-T neighbourhood of the change) ---")
base = Chem.MolFromSmiles(TEMPLATE.replace("{R1}","C").replace("{R2}","C"))
mg_b = feat(base)
rb, ab = wl(mg_b, MAXR)
print(f"{'edit':>22}{'atoms':>7}" + "".join(f"{('dirty r=%d'%r):>11}" for r in (1,2,3,5)))
for r2 in ["F","C(F)(F)F","c1ccncc1","N1CCOCC1"]:
    m2 = Chem.MolFromSmiles(TEMPLATE.replace("{R1}","C").replace("{R2}",r2))
    mg2 = feat(m2); r2s, a2s = wl(mg2, MAXR)
    row = f"{('R2: C -> '+r2):>22}{mg2.V.shape[0]:>7}"
    for r in (1,2,3,5):
        base_set = set(ab[r-1])
        dirty = sum(1 for x in a2s[r-1] if x not in base_set)
        row += f"{dirty/mg2.V.shape[0]:>11.1%}"
    print(row)
