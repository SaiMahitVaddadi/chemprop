import sys, time, glob
sys.path.insert(0, "/private/tmp/claude-502/-Users-svaddadi-Documents-GitHub-chemprop/54c934b6-e3f8-4460-b930-d5d089b68966/scratchpad")
import numpy as np, torch
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")
from cp_loader import SimpleMoleculeMolGraphFeaturizer, MultiHotAtomFeaturizer, MultiHotBondFeaturizer

smis = []
with open("tests/data/regression.csv") as f:
    f.readline()
    for line in f:
        s = line.split(",")[0].strip()
        if Chem.MolFromSmiles(s) is not None:
            smis.append(s)
mols = [Chem.MolFromSmiles(s) for s in smis]
N = len(mols)
feat = SimpleMoleculeMolGraphFeaturizer()
n_atoms = sum(m.GetNumAtoms() for m in mols)
n_bonds = sum(m.GetNumBonds() for m in mols)
print(f"{N} molecules, {n_atoms} atoms, {n_bonds} bonds ({n_atoms/N:.1f} atoms/mol)\n")

def timeit(fn, reps=3):
    best = 1e9
    for _ in range(reps):
        t = time.perf_counter(); fn(); best = min(best, time.perf_counter() - t)
    return best

# ---------- (1) current featurization path ----------
t_feat = timeit(lambda: [feat(m) for m in mols])
print(f"[featurize] chemprop SimpleMoleculeMolGraphFeaturizer : {t_feat*1e3:8.1f} ms  "
      f"({t_feat/N*1e6:7.1f} us/mol)")

# ---------- (2) atom/bond TYPE-TABLE lookup featurization ----------
af, bf = feat.atom_featurizer, feat.bond_featurizer
atom_tab, bond_tab = {}, {}
def atom_key(a):
    return (a.GetAtomicNum(), a.GetTotalDegree(), a.GetFormalCharge(), int(a.GetChiralTag()),
            int(a.GetTotalNumHs()), int(a.GetHybridization()), a.GetIsAromatic())
def bond_key(b):
    return (None if b is None else int(b.GetBondType()), b.GetIsConjugated(),
            b.IsInRing(), int(b.GetStereo()))
def feat_cached(m):
    V = np.empty((m.GetNumAtoms(), af.__len__()), np.float32)
    for i, a in enumerate(m.GetAtoms()):
        k = atom_key(a)
        r = atom_tab.get(k)
        if r is None:
            r = atom_tab[k] = np.asarray(af(a), np.float32)
        V[i] = r
    E = np.empty((2*m.GetNumBonds(), bf.__len__()), np.float32)
    ei = np.empty((2, 2*m.GetNumBonds()), np.int64)
    for j, b in enumerate(m.GetBonds()):
        k = bond_key(b)
        r = bond_tab.get(k)
        if r is None:
            r = bond_tab[k] = np.asarray(bf(b), np.float32)
        E[2*j] = r; E[2*j+1] = r
        u, v = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        ei[0, 2*j], ei[1, 2*j] = u, v
        ei[0, 2*j+1], ei[1, 2*j+1] = v, u
    return V, E, ei
feat_cached(mols[0])  # warm
t_cached = timeit(lambda: [feat_cached(m) for m in mols])
print(f"[featurize] type-table lookup (warm)                  : {t_cached*1e3:8.1f} ms  "
      f"({t_cached/N*1e6:7.1f} us/mol)   speedup {t_feat/t_cached:.2f}x")
print(f"            table sizes: {len(atom_tab)} atom types, {len(bond_tab)} bond types")

# correctness check
mg = feat(mols[7]); Vc, Ec, eic = feat_cached(mols[7])
print(f"            identical to featurizer output: V={np.allclose(mg.V, Vc)} "
      f"E={np.allclose(mg.E, Ec)} edge_index={np.array_equal(mg.edge_index, eic)}")

# ---------- (3) CheMeleon-sized message passing ----------
d_v, d_e, d_h, depth = 72, 14, 2048, 6
W_i = torch.nn.Linear(d_v+d_e, d_h, bias=False)
W_h = torch.nn.Linear(d_h, d_h, bias=False)
W_o = torch.nn.Linear(d_v+d_h, d_h)
mgs = [feat(m) for m in mols]
V = torch.from_numpy(np.concatenate([g.V for g in mgs])).float()
E = torch.from_numpy(np.concatenate([g.E for g in mgs])).float()
off_a = off_e = 0; eis = []; revs = []
for g in mgs:
    eis.append(g.edge_index + off_a); revs.append(g.rev_edge_index + off_e)
    off_a += g.V.shape[0]; off_e += g.edge_index.shape[1]
ei = torch.from_numpy(np.hstack(eis)).long(); rev = torch.from_numpy(np.concatenate(revs)).long()
print(f"\nbatched graph: V={tuple(V.shape)} E={tuple(E.shape)}  (whole 500-mol dataset as 1 batch)")

@torch.no_grad()
def mpnn(V, E, ei, rev):
    H_0 = W_i(torch.cat([V[ei[0]], E], dim=1))
    H = torch.relu(H_0)
    for _ in range(1, depth):
        idx = ei[1].unsqueeze(1).expand(-1, H.shape[1])
        M = torch.zeros(V.shape[0], H.shape[1]).scatter_reduce_(0, idx, H, reduce="sum",
                                                                include_self=False)[ei[0]] - H[rev]
        H = torch.relu(H_0 + W_h(M))
    idx = ei[1].unsqueeze(1).expand(-1, H.shape[1])
    M = torch.zeros(V.shape[0], H.shape[1]).scatter_reduce_(0, idx, H, reduce="sum",
                                                            include_self=False)
    return W_o(torch.cat((V, M), dim=1))

mpnn(V, E, ei, rev)
t_mp = timeit(lambda: mpnn(V, E, ei, rev))
flops = 2*(ei.shape[1]*(d_v+d_e)*d_h + (depth-1)*ei.shape[1]*d_h*d_h + V.shape[0]*(d_v+d_h)*d_h)
print(f"[model] CheMeleon-size MPNN (d_h=2048, depth=6) CPU   : {t_mp*1e3:8.1f} ms  "
      f"({t_mp/N*1e6:7.1f} us/mol)  {flops/t_mp/1e9:.1f} GFLOP/s, {flops/N/1e9:.2f} GFLOP/mol")

# ---------- (4) WL hashing cost = the "search cost" of a model-side cache ----------
from collections import defaultdict
def wl_hash_mol(g, T):
    Vv, Ee, e_i, r_i = g.V, g.E, g.edge_index, g.rev_edge_index
    a = [hash(Vv[i].tobytes()) for i in range(Vv.shape[0])]
    h0 = [hash((a[e_i[0, e]], hash(Ee[e].tobytes()))) for e in range(e_i.shape[1])]
    h = list(h0)
    ine = defaultdict(list)
    for e in range(e_i.shape[1]):
        ine[e_i[1, e]].append(e)
    for _ in range(T):
        h = [hash((h0[e], tuple(sorted(h[f] for f in ine[e_i[0, e]] if f != r_i[e]))))
             for e in range(e_i.shape[1])]
    return h
t_wl = timeit(lambda: [wl_hash_mol(g, depth-1) for g in mgs])
print(f"[cache] python WL hashing, depth 6 (per-layer keys)   : {t_wl*1e3:8.1f} ms  "
      f"({t_wl/N*1e6:7.1f} us/mol)  <-- search cost")

# ---------- (5) vectorized cache lookup: int64 keys -> gather ----------
n_cache = 200_000
cache = torch.randn(n_cache, d_h)
keys = torch.randint(0, 2**62, (n_cache,)).sort().values
q = keys[torch.randint(0, n_cache, (ei.shape[1],))]
def vec_lookup():
    pos = torch.searchsorted(keys, q)
    return cache.index_select(0, pos)
vec_lookup()
t_look = timeit(lambda: vec_lookup())
print(f"[cache] vectorized searchsorted+gather ({ei.shape[1]} edges): {t_look*1e3:8.1f} ms  "
      f"({t_look/N*1e6:7.1f} us/mol)  <-- one layer's lookup")
print(f"\ncache memory for 1M atom envs @ d_h=2048: fp32 {1e6*2048*4/2**30:.1f} GiB, "
      f"int8 {1e6*2048/2**30:.1f} GiB")
