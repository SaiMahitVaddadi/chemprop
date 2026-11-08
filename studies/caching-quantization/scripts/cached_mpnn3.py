"""Exact cached D-MPNN: optimal descend/ascend schedule + type-ID-seeded keys."""
import sys, time
sys.path.insert(0,"/private/tmp/chemprop-quant-study")
import numpy as np, torch
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")
from cp_loader import SimpleMoleculeMolGraphFeaturizer
torch.manual_seed(0)

TEMPLATE="O=C(Nc1ccc({R1})cc1)c1ccc({R2})cc1"
SUBS=["F","Cl","Br","C","CC","CCC","OC","OCC","C(F)(F)F","C#N","O","N","N(C)C",
      "S(=O)(=O)C","c1ccccc1","c1ccncc1","N1CCOCC1","N1CCCCC1","C(=O)O","C(=O)N"]
feat=SimpleMoleculeMolGraphFeaturizer(); af,bf=feat.atom_featurizer,feat.bond_featurizer
mols=[m for m in (Chem.MolFromSmiles(TEMPLATE.replace("{R1}",a).replace("{R2}",b))
      for a in SUBS for b in SUBS) if m is not None]

# ---- input-graph type-ID cache (from the earlier finding): gives V,E AND the WL seed ----
atom_ids,bond_ids={},{}; atom_rows,bond_rows=[],[]
def ids_of(m):
    aid=np.empty(m.GetNumAtoms(),np.int64)
    for i,a in enumerate(m.GetAtoms()):
        k=(a.GetAtomicNum(),a.GetTotalDegree(),a.GetFormalCharge(),int(a.GetChiralTag()),
           int(a.GetTotalNumHs()),int(a.GetHybridization()),a.GetIsAromatic())
        j=atom_ids.get(k)
        if j is None: j=atom_ids[k]=len(atom_rows); atom_rows.append(np.asarray(af(a),np.float32))
        aid[i]=j
    nb=m.GetNumBonds(); bid=np.empty(nb,np.int64); ij=np.empty((nb,2),np.int64)
    for i,b in enumerate(m.GetBonds()):
        k=(int(b.GetBondType()),b.GetIsConjugated(),b.IsInRing(),int(b.GetStereo()))
        j=bond_ids.get(k)
        if j is None: j=bond_ids[k]=len(bond_rows); bond_rows.append(np.asarray(bf(b),np.float32))
        bid[i]=j; ij[i]=(b.GetBeginAtomIdx(),b.GetEndAtomIdx())
    return aid,bid,ij
allids=[ids_of(m) for m in mols]
AT=torch.from_numpy(np.stack(atom_rows)); BT=torch.from_numpy(np.stack(bond_rows))
print(f"type tables: {AT.shape[0]} atom types, {BT.shape[0]} bond types")

def batch(idxs):
    Va=[];Ea=[];eis=[];revs=[];aids=[];bids=[];oa=oe=0
    for i in idxs:
        aid,bid,ij=allids[i]; n=len(aid); nb=len(bid)
        Va.append(aid); aids.append(aid)
        ei=np.empty((2,2*nb),np.int64)
        ei[0,0::2]=ij[:,0]; ei[1,0::2]=ij[:,1]; ei[0,1::2]=ij[:,1]; ei[1,1::2]=ij[:,0]
        eis.append(ei+oa); revs.append(np.arange(2*nb).reshape(-1,2)[:,::-1].ravel()+oe)
        bids.append(np.repeat(bid,2)); oa+=n; oe+=2*nb
    aid=torch.from_numpy(np.concatenate(aids)); bid=torch.from_numpy(np.concatenate(bids))
    return (AT[aid],BT[bid],torch.from_numpy(np.hstack(eis)),
            torch.from_numpy(np.concatenate(revs)),aid,bid)

def s64(v): return int(np.int64(np.uint64(v)))
C1,C2,GOLD=s64(0xbf58476d1ce4e5b9),s64(0x94d049bb133111eb),s64(0x9e3779b97f4a7c15)
def mix(x):
    x=x^(x>>30); x=x*C1; x=x^(x>>27); x=x*C2; return x^(x>>31)

@torch.no_grad()
def wl_keys(aid,bid,ei,rev,n_v,rounds):
    """seeded from type IDs -- no float hashing needed"""
    k0=mix(mix(aid[ei[0]]*GOLD)^mix(bid*C1)); keys=[k0]; k=k0
    for _ in range(rounds):
        km=mix(k)
        acc=torch.zeros(n_v,dtype=torch.int64).index_add_(0,ei[1],km)
        k=mix(k0^mix(acc[ei[0]]-km[rev])); keys.append(k)
    return keys

class Table:
    def __init__(s,d_h):
        s.keys=torch.empty(0,dtype=torch.int64); s.rows=torch.empty(0,d_h)
    def probe(s,q):
        if s.keys.numel()==0:
            return torch.zeros(q.shape[0],dtype=torch.bool),torch.zeros(q.shape[0],dtype=torch.long)
        pos=torch.searchsorted(s.keys,q).clamp(max=s.keys.numel()-1)
        return s.keys[pos]==q,pos
    def insert(s,k,v):
        k2=torch.cat([s.keys,k]); v2=torch.cat([s.rows,v])
        k2,o=torch.sort(k2); v2=v2[o]
        u=torch.ones(k2.shape[0],dtype=torch.bool); u[1:]=k2[1:]!=k2[:-1]
        s.keys=k2[u]; s.rows=v2[u]

class MP(torch.nn.Module):
    def __init__(s,d_h,depth):
        super().__init__()
        s.W_i=torch.nn.Linear(86,d_h,bias=False); s.W_h=torch.nn.Linear(d_h,d_h,bias=False)
        s.W_o=torch.nn.Linear(72+d_h,d_h); s.depth=depth; s.d_h=d_h
    def _agg(s,H,n_v,ei,rev,sub):
        idx=ei[1].unsqueeze(1).expand(-1,s.d_h)
        A=torch.zeros(n_v,s.d_h).scatter_reduce_(0,idx,H,reduce="sum",include_self=False)
        return A[ei[0]]-H[rev] if sub else A
    @torch.no_grad()
    def full(s,V,E,ei,rev,*_):
        H_0=s.W_i(torch.cat([V[ei[0]],E],dim=1)); H=torch.relu(H_0)
        for _ in range(1,s.depth): H=torch.relu(H_0+s.W_h(s._agg(H,V.shape[0],ei,rev,True)))
        return s.W_o(torch.cat((V,s._agg(H,V.shape[0],ei,rev,False)),dim=1))
    @torch.no_grad()
    def cached(s,V,E,ei,rev,aid,bid,store,stats):
        n_v=V.shape[0]
        keys=wl_keys(aid,bid,ei,rev,n_v,s.depth-1)
        hits={};poss={}
        for t in range(1,s.depth): hits[t],poss[t]=store[t].probe(keys[t])
        # deepest fully-cached layer -> everything below it is unnecessary
        t0=0
        for t in range(s.depth-1,0,-1):
            if bool(hits[t].all()): t0=t; break
        H_0=None
        if t0>0:
            H=store[t0].rows[poss[t0]]
        else:
            H_0=s.W_i(torch.cat([V[ei[0]],E],dim=1)); H=torch.relu(H_0)
        for t in range(t0+1,s.depth):
            stats[1]+=hits[t].numel()
            if bool(hits[t].all()):
                H=store[t].rows[poss[t]]; continue
            if H_0 is None: H_0=s.W_i(torch.cat([V[ei[0]],E],dim=1))
            miss=~hits[t]; nm=int(miss.sum()); stats[0]+=nm
            M=s._agg(H,n_v,ei,rev,True)
            Hn=torch.empty_like(H)
            if hits[t].any(): Hn[hits[t]]=store[t].rows[poss[t][hits[t]]]
            vals=torch.relu(H_0[miss]+s.W_h(M[miss])); Hn[miss]=vals
            store[t].insert(keys[t][miss],vals); H=Hn
        stats[2]+=1 if t0==s.depth-1 else 0
        return s.W_o(torch.cat((V,s._agg(H,n_v,ei,rev,False)),dim=1))

def timeit(fn,reps=2):
    b=1e9;r=None
    for _ in range(reps):
        t=time.perf_counter();r=fn();b=min(b,time.perf_counter()-t)
    return b,r
CHUNK=25
chunks=[batch(range(i,min(i+CHUNK,len(mols)))) for i in range(0,len(mols),CHUNK)]
t_key,_=timeit(lambda:[wl_keys(c[4],c[5],c[2],c[3],c[0].shape[0],5) for c in chunks])
print(f"type-ID-seeded WL keying: {t_key/len(mols)*1e6:.1f} us/mol"
      f"   (float-hash version 74.1, python loops 94.0)\n")
print(f"{'model':<26}{'full':>8}{'cold':>8}{'warm':>8}{'cold x':>8}{'warm x':>8}{'miss':>7}{'err':>9}")
for nm,d_h,depth in [("default d_h=300 d=3",300,3),("d_h=600 d=5",600,5),
                     ("CheMeleon d_h=2048 d=6",2048,6)]:
    m=MP(d_h,depth).eval()
    t_full,refs=timeit(lambda:[m.full(*c) for c in chunks])
    store={t:Table(d_h) for t in range(1,depth)}; st=[0,0,0]
    t_cold,outs=timeit(lambda:[m.cached(*c,store,st) for c in chunks],reps=1)
    miss_cold=st[0]/max(st[1],1)
    st2=[0,0,0]
    t_warm,outs2=timeit(lambda:[m.cached(*c,store,st2) for c in chunks])
    err=max(float((a-b).abs().max()) for a,b in zip(refs,outs2))
    print(f"{nm:<26}{t_full*1e3:>7.0f}m{t_cold*1e3:>7.0f}m{t_warm*1e3:>7.0f}m"
          f"{t_full/t_cold:>7.2f}x{t_full/t_warm:>7.2f}x{miss_cold:>7.1%}{err:>9.1e}")
