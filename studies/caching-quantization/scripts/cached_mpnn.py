"""Exact environment-cached D-MPNN, with vectorized (order-invariant) WL keying."""
import sys, time
sys.path.insert(0, "/private/tmp/claude-502/-Users-svaddadi-Documents-GitHub-chemprop/54c934b6-e3f8-4460-b930-d5d089b68966/scratchpad")
import numpy as np, torch
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")
from cp_loader import SimpleMoleculeMolGraphFeaturizer
torch.manual_seed(0)

TEMPLATE = "O=C(Nc1ccc({R1})cc1)c1ccc({R2})cc1"
SUBS = ["F","Cl","Br","C","CC","CCC","OC","OCC","C(F)(F)F","C#N","O","N","N(C)C",
        "S(=O)(=O)C","c1ccccc1","c1ccncc1","N1CCOCC1","N1CCCCC1","C(=O)O","C(=O)N"]
feat = SimpleMoleculeMolGraphFeaturizer()
mols = [Chem.MolFromSmiles(TEMPLATE.replace("{R1}",a).replace("{R2}",b))
        for a in SUBS for b in SUBS]
mols = [m for m in mols if m is not None]

def batch(ms):
    mgs=[feat(m) for m in ms]
    V=torch.from_numpy(np.concatenate([g.V for g in mgs])).float()
    E=torch.from_numpy(np.concatenate([g.E for g in mgs]).astype(np.float32))
    oa=oe=0; eis=[]; revs=[]; bt=[]
    for i,g in enumerate(mgs):
        eis.append(g.edge_index+oa); revs.append(g.rev_edge_index+oe); bt += [i]*g.V.shape[0]
        oa+=g.V.shape[0]; oe+=g.edge_index.shape[1]
    return (V,E,torch.from_numpy(np.hstack(eis)).long(),
            torch.from_numpy(np.concatenate(revs)).long(), torch.tensor(bt).long())

def s64(v):            # wrap a uint64 constant into int64
    return int(np.int64(np.uint64(v)))
C1, C2, GOLD = s64(0xbf58476d1ce4e5b9), s64(0x94d049bb133111eb), s64(0x9e3779b97f4a7c15)
def mix(x):            # splitmix64-style finalizer (wrapping int64 arithmetic)
    x = x ^ (x >> 30); x = x * C1
    x = x ^ (x >> 27); x = x * C2
    return x ^ (x >> 31)

def row_keys(T):
    """stable int64 key per row of a float tensor"""
    b = T.contiguous().view(torch.int32).view(T.shape[0], -1).long()
    k = torch.full((T.shape[0],), GOLD, dtype=torch.int64)
    for j in range(b.shape[1]):
        k = mix(k ^ b[:, j])
    return k

@torch.no_grad()
def wl_keys_vectorized(V, E, ei, rev, depth):
    """per-layer int64 keys for every directed edge; order-invariant via additive hashing."""
    a = row_keys(V); e = row_keys(E)
    k0 = mix(a[ei[0]] ^ mix(e))
    keys = [k0]; k = k0
    n_v = V.shape[0]
    for _ in range(depth):
        km = mix(k)
        acc = torch.zeros(n_v, dtype=torch.int64).index_add_(0, ei[1], km)
        msg = acc[ei[0]] - km[rev]          # non-backtracking, order-invariant
        k = mix(k0 ^ mix(msg))
        keys.append(k)
    return keys

class MP(torch.nn.Module):
    def __init__(s, d_h, depth):
        super().__init__()
        s.W_i=torch.nn.Linear(86,d_h,bias=False); s.W_h=torch.nn.Linear(d_h,d_h,bias=False)
        s.W_o=torch.nn.Linear(72+d_h,d_h); s.depth=depth; s.d_h=d_h
    @torch.no_grad()
    def full(s, V,E,ei,rev):
        H_0=s.W_i(torch.cat([V[ei[0]],E],dim=1)); H=torch.relu(H_0)
        for _ in range(1,s.depth):
            idx=ei[1].unsqueeze(1).expand(-1,s.d_h)
            M=torch.zeros(V.shape[0],s.d_h).scatter_reduce_(0,idx,H,reduce="sum",
                                                            include_self=False)[ei[0]]-H[rev]
            H=torch.relu(H_0+s.W_h(M))
        idx=ei[1].unsqueeze(1).expand(-1,s.d_h)
        M=torch.zeros(V.shape[0],s.d_h).scatter_reduce_(0,idx,H,reduce="sum",include_self=False)
        return s.W_o(torch.cat((V,M),dim=1))
    @torch.no_grad()
    def cached(s, V,E,ei,rev, store):
        """store[t] = dict(int key -> row index) + tensor table; exact, not approximate."""
        keys = wl_keys_vectorized(V,E,ei,rev, s.depth-1)
        H_0 = s.W_i(torch.cat([V[ei[0]],E],dim=1)); H = torch.relu(H_0)
        n_gemm = 0
        for t in range(1, s.depth):
            kt = keys[t]
            d, tab = store[t]
            kl = kt.tolist()
            idx_hit=[]; pos_hit=[]
            for i,kk in enumerate(kl):
                p = d.get(kk)
                if p is not None: idx_hit.append(i); pos_hit.append(p)
            miss = torch.ones(len(kl), dtype=torch.bool)
            if idx_hit: miss[torch.tensor(idx_hit)] = False
            # messages need ALL previous-layer values (available), scatter is cheap
            idx=ei[1].unsqueeze(1).expand(-1,s.d_h)
            M=torch.zeros(V.shape[0],s.d_h).scatter_reduce_(0,idx,H,reduce="sum",
                                                            include_self=False)[ei[0]]-H[rev]
            H_new = torch.empty_like(H)
            if idx_hit:
                H_new[torch.tensor(idx_hit)] = tab[torch.tensor(pos_hit)]
            nm = int(miss.sum())
            n_gemm += nm
            if nm:
                H_new[miss] = torch.relu(H_0[miss] + s.W_h(M[miss]))   # GEMM only on misses
                # grow cache
                newk = kt[miss].tolist(); newv = H_new[miss]
                base = tab.shape[0]
                store[t] = (d, torch.cat([tab, newv], 0))
                for j,kk in enumerate(newk):
                    if kk not in d: d[kk] = base + j
            H = H_new
        idx=ei[1].unsqueeze(1).expand(-1,s.d_h)
        M=torch.zeros(V.shape[0],s.d_h).scatter_reduce_(0,idx,H,reduce="sum",include_self=False)
        return s.W_o(torch.cat((V,M),dim=1)), n_gemm

def timeit(fn,reps=3):
    b=1e9; r=None
    for _ in range(reps):
        t=time.perf_counter(); r=fn(); b=min(b,time.perf_counter()-t)
    return b, r

CHUNK = 25
chunks = [batch(mols[i:i+CHUNK]) for i in range(0, len(mols), CHUNK)]
tot_e = sum(c[2].shape[1] for c in chunks)
print(f"{len(mols)} analogs in {len(chunks)} batches of {CHUNK} | {tot_e} directed bonds total\n")

# keying cost
_, _ = timeit(lambda: [wl_keys_vectorized(*c[:4], 5) for c in chunks])
t_key,_ = timeit(lambda: [wl_keys_vectorized(*c[:4], 5) for c in chunks])
print(f"vectorized WL keying (depth 6): {t_key*1e3:.1f} ms total, "
      f"{t_key/len(mols)*1e6:.1f} us/mol   (python version was 94.0 us/mol)\n")

print(f"{'model':<28}{'full fwd':>10}{'cached fwd':>12}{'speedup':>9}{'GEMM rows':>11}{'max err':>10}")
for nm, d_h, depth in [("default d_h=300 depth=3",300,3), ("CheMeleon d_h=2048 depth=6",2048,6)]:
    m = MP(d_h, depth).eval()
    t_full, refs = timeit(lambda: [m.full(*c[:4]) for c in chunks])
    def run():
        store = {t: ({}, torch.empty(0, d_h)) for t in range(1, depth)}
        out=[]; ng=0
        for c in chunks:
            o, n = m.cached(*c[:4], store); out.append(o); ng += n
        return out, ng
    t_c, (outs, ng) = timeit(run, reps=2)
    err = max(float((a-b).abs().max()) for a,b in zip(refs, outs))
    print(f"{nm:<28}{t_full*1e3:>9.1f}m{t_c*1e3:>11.1f}m{t_full/t_c:>9.2f}x"
          f"{ng/(tot_e*(depth-1)):>10.1%}{err:>10.1e}")
