"""Exact environment-cached D-MPNN with fully vectorized keying AND lookup."""
import sys, time
sys.path.insert(0, "/private/tmp/claude-502/-Users-svaddadi-Documents-GitHub-chemprop/54c934b6-e3f8-4460-b930-d5d089b68966/scratchpad")
import numpy as np, torch
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")
from cp_loader import SimpleMoleculeMolGraphFeaturizer
torch.manual_seed(0)

TEMPLATE="O=C(Nc1ccc({R1})cc1)c1ccc({R2})cc1"
SUBS=["F","Cl","Br","C","CC","CCC","OC","OCC","C(F)(F)F","C#N","O","N","N(C)C",
      "S(=O)(=O)C","c1ccccc1","c1ccncc1","N1CCOCC1","N1CCCCC1","C(=O)O","C(=O)N"]
feat=SimpleMoleculeMolGraphFeaturizer()
mols=[m for m in (Chem.MolFromSmiles(TEMPLATE.replace("{R1}",a).replace("{R2}",b))
      for a in SUBS for b in SUBS) if m is not None]

def batch(ms):
    mgs=[feat(m) for m in ms]
    V=torch.from_numpy(np.concatenate([g.V for g in mgs])).float()
    E=torch.from_numpy(np.concatenate([g.E for g in mgs]).astype(np.float32))
    oa=oe=0; eis=[]; revs=[]
    for g in mgs:
        eis.append(g.edge_index+oa); revs.append(g.rev_edge_index+oe)
        oa+=g.V.shape[0]; oe+=g.edge_index.shape[1]
    return (V,E,torch.from_numpy(np.hstack(eis)).long(),
            torch.from_numpy(np.concatenate(revs)).long())

def s64(v): return int(np.int64(np.uint64(v)))
C1,C2,GOLD=s64(0xbf58476d1ce4e5b9),s64(0x94d049bb133111eb),s64(0x9e3779b97f4a7c15)
def mix(x):
    x=x^(x>>30); x=x*C1; x=x^(x>>27); x=x*C2; return x^(x>>31)
def row_keys(T):
    b=T.contiguous().view(torch.int32).view(T.shape[0],-1).long()
    k=torch.full((T.shape[0],),GOLD,dtype=torch.int64)
    for j in range(b.shape[1]): k=mix(k^b[:,j])
    return k

@torch.no_grad()
def wl_keys(V,E,ei,rev,rounds):
    a=row_keys(V); k0=mix(a[ei[0]]^mix(row_keys(E))); keys=[k0]; k=k0
    for _ in range(rounds):
        km=mix(k)
        acc=torch.zeros(V.shape[0],dtype=torch.int64).index_add_(0,ei[1],km)
        k=mix(k0^mix(acc[ei[0]]-km[rev])); keys.append(k)
    return keys

class Table:
    """sorted int64 keys -> rows; vectorized probe, amortized insert."""
    def __init__(s,d_h,cap=1<<14):
        s.keys=torch.empty(0,dtype=torch.int64); s.rows=torch.empty(0,d_h); s.d_h=d_h
    def probe(s,q):
        if s.keys.numel()==0:
            return torch.zeros(q.shape[0],dtype=torch.bool), torch.zeros(q.shape[0],dtype=torch.long)
        pos=torch.searchsorted(s.keys,q).clamp(max=s.keys.numel()-1)
        hit=s.keys[pos]==q
        return hit,pos
    def insert(s,k,v):
        k2=torch.cat([s.keys,k]); v2=torch.cat([s.rows,v])
        k2,order=torch.sort(k2); v2=v2[order]
        uniq=torch.ones(k2.shape[0],dtype=torch.bool)
        uniq[1:]=k2[1:]!=k2[:-1]
        s.keys=k2[uniq]; s.rows=v2[uniq]

class MP(torch.nn.Module):
    def __init__(s,d_h,depth):
        super().__init__()
        s.W_i=torch.nn.Linear(86,d_h,bias=False); s.W_h=torch.nn.Linear(d_h,d_h,bias=False)
        s.W_o=torch.nn.Linear(72+d_h,d_h); s.depth=depth; s.d_h=d_h
    def _agg(s,H,V,ei,rev,sub):
        idx=ei[1].unsqueeze(1).expand(-1,s.d_h)
        A=torch.zeros(V.shape[0],s.d_h).scatter_reduce_(0,idx,H,reduce="sum",include_self=False)
        return A[ei[0]]-H[rev] if sub else A
    @torch.no_grad()
    def full(s,V,E,ei,rev):
        H_0=s.W_i(torch.cat([V[ei[0]],E],dim=1)); H=torch.relu(H_0)
        for _ in range(1,s.depth):
            H=torch.relu(H_0+s.W_h(s._agg(H,V,ei,rev,True)))
        return s.W_o(torch.cat((V,s._agg(H,V,ei,rev,False)),dim=1))
    @torch.no_grad()
    def cached(s,V,E,ei,rev,store,stats):
        keys=wl_keys(V,E,ei,rev,s.depth-1)
        H_0=s.W_i(torch.cat([V[ei[0]],E],dim=1)); H=torch.relu(H_0)
        for t in range(1,s.depth):
            tab=store[t]; q=keys[t]
            hit,pos=tab.probe(q)
            M=s._agg(H,V,ei,rev,True)
            H_new=torch.empty_like(H)
            if hit.any(): H_new[hit]=tab.rows[pos[hit]]
            miss=~hit; nm=int(miss.sum())
            stats[0]+=nm; stats[1]+=hit.numel()
            if nm:
                vals=torch.relu(H_0[miss]+s.W_h(M[miss]))
                H_new[miss]=vals
                tab.insert(q[miss],vals)
            H=H_new
        return s.W_o(torch.cat((V,s._agg(H,V,ei,rev,False)),dim=1))

def timeit(fn,reps=2):
    b=1e9;r=None
    for _ in range(reps):
        t=time.perf_counter();r=fn();b=min(b,time.perf_counter()-t)
    return b,r

CHUNK=25
chunks=[batch(mols[i:i+CHUNK]) for i in range(0,len(mols),CHUNK)]
tot_e=sum(c[2].shape[1] for c in chunks)
print(f"{len(mols)} analogs, {len(chunks)} batches x {CHUNK} | {tot_e} directed bonds\n")
t_key,_=timeit(lambda:[wl_keys(*c,5) for c in chunks])
print(f"vectorized WL keying: {t_key/len(mols)*1e6:.1f} us/mol  (python loops were 94.0)\n")
print(f"{'model':<28}{'full':>9}{'cached':>9}{'speedup':>9}{'GEMM rows':>11}{'max err':>10}")
for nm,d_h,depth in [("default d_h=300 depth=3",300,3),("d_h=600 depth=5",600,5),
                     ("CheMeleon d_h=2048 depth=6",2048,6)]:
    m=MP(d_h,depth).eval()
    t_full,refs=timeit(lambda:[m.full(*c) for c in chunks])
    def run():
        store={t:Table(d_h) for t in range(1,depth)}; st=[0,0]
        return [m.cached(*c,store,st) for c in chunks],st
    t_c,(outs,st)=timeit(run)
    err=max(float((a-b).abs().max()) for a,b in zip(refs,outs))
    print(f"{nm:<28}{t_full*1e3:>8.0f}m{t_c*1e3:>8.0f}m{t_full/t_c:>9.2f}x"
          f"{st[0]/st[1]:>10.1%}{err:>10.1e}")

# steady state: cache already warm (second pass over the library)
print(f"\n--- steady state (cache warmed on the library, then re-run: the screening case) ---")
print(f"{'model':<28}{'full':>9}{'warm':>9}{'speedup':>9}{'GEMM rows':>11}")
for nm,d_h,depth in [("default d_h=300 depth=3",300,3),("d_h=600 depth=5",600,5),
                     ("CheMeleon d_h=2048 depth=6",2048,6)]:
    m=MP(d_h,depth).eval()
    store={t:Table(d_h) for t in range(1,depth)}; st=[0,0]
    for c in chunks: m.cached(*c,store,st)          # warm
    t_full,_=timeit(lambda:[m.full(*c) for c in chunks])
    st2=[0,0]
    t_w,_=timeit(lambda:[m.cached(*c,store,st2) for c in chunks])
    print(f"{nm:<28}{t_full*1e3:>8.0f}m{t_w*1e3:>8.0f}m{t_full/t_w:>9.2f}x{st2[0]/max(st2[1],1):>10.1%}")
