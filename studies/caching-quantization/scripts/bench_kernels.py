"""Per-layer message+update kernel variants, isolated and timed."""
import sys; sys.path.insert(0,'.')
from common import *

b=build_batch(); V,E,ei,rev=b['V'],b['E'],b['ei'],b['rev']
nV,nE=b['n_atom'],b['n_edge']
src,dst=ei[0].contiguous(),ei[1].contiguous()

# ---- precomputable (collate-time) structures ----
# 1. CSR for the atom-level scatter with dst-sorted edges
perm = torch.argsort(dst, stable=True)
dst_sorted = dst[perm]
offsets = torch.zeros(nV+1, dtype=torch.long)
offsets[1:] = torch.bincount(dst, minlength=nV).cumsum(0)
# 2. E x E non-backtracking operator in CSR:  row e -> {f : dst(f)=src(e), f != rev(e)}
rows=[]; cols=[]
dst_np=dst.numpy(); src_np=src.numpy(); rev_np=rev.numpy()
byd={}
for f,d in enumerate(dst_np): byd.setdefault(int(d),[]).append(f)
for e in range(nE):
    r=rev_np[e]
    for f in byd[int(src_np[e])]:
        if f!=r: rows.append(e); cols.append(f)
nnz=len(rows)
Bcoo=torch.sparse_coo_tensor(torch.tensor([rows,cols]), torch.ones(nnz), (nE,nE)).coalesce()
Bcsr=Bcoo.to_sparse_csr()
print(f"E x E non-backtracking CSR: nnz={nnz} ({nnz/nE:.2f} x E)")
# 3. padded dense neighbour index [nV, maxdeg]
maxdeg=int(torch.bincount(dst,minlength=nV).max())
padidx=torch.full((nV,maxdeg), nE, dtype=torch.long)   # nE = pad row (zeros)
cnt=torch.zeros(nV,dtype=torch.long)
for f,d in enumerate(dst_np):
    padidx[int(d), cnt[int(d)]]=f; cnt[int(d)]+=1
print(f"padded layout: maxdeg={maxdeg}, rows={nV*maxdeg} ({nV*maxdeg/nE:.2f} x E)")

def make(dev, d_h):
    g=dict(src=src.to(dev),dst=dst.to(dev),rev=rev.to(dev),perm=perm.to(dev),
           off=offsets.to(dev),padidx=padidx.to(dev),
           Bcsr=Bcsr.to(dev) if dev=='cpu' else None)
    H=torch.randn(nE,d_h,device=dev); H0=torch.randn(nE,d_h,device=dev)
    W=torch.randn(d_h,d_h,device=dev)/ (d_h**0.5)
    return g,H,H0,W

def variants(g,H,H0,W,d_h,dev):
    src,dst,rev,off,padidx=g['src'],g['dst'],g['rev'],g['off'],g['padidx']
    Wt=W.t().contiguous()
    def v0_repeat():
        idx=dst.unsqueeze(1).repeat(1,d_h)
        S=torch.zeros(nV,d_h,device=dev).scatter_reduce_(0,idx,H,reduce="sum",include_self=False)
        M=S[src]-H[rev]
        return torch.relu(H0+torch.nn.functional.linear(M,W))
    def v1_expand():
        idx=dst.unsqueeze(1).expand(-1,d_h)
        S=torch.zeros(nV,d_h,device=dev).scatter_reduce_(0,idx,H,reduce="sum",include_self=False)
        M=S[src]-H[rev]
        return torch.relu(H0+torch.nn.functional.linear(M,W))
    def v2_indexadd():
        S=torch.zeros(nV,d_h,device=dev).index_add_(0,dst,H)
        M=S[src]-H[rev]
        return torch.relu(H0+torch.nn.functional.linear(M,W))
    def v3_inplace():
        S=torch.zeros(nV,d_h,device=dev).index_add_(0,dst,H)
        M=torch.index_select(S,0,src); M.sub_(torch.index_select(H,0,rev))
        return torch.addmm(H0,M,Wt).relu_()
    def v4_revadd():
        S=torch.zeros(nV,d_h,device=dev).index_add_(0,dst,H)
        M=torch.index_select(S,0,src); M.index_add_(0,rev,H,alpha=-1)
        return torch.addmm(H0,M,Wt).relu_()
    def v5_pairswap():
        S=torch.zeros(nV,d_h,device=dev).index_add_(0,dst,H)
        M=torch.index_select(S,0,src)
        M.sub_(H.view(nE//2,2,d_h).flip(1).reshape(nE,d_h))
        return torch.addmm(H0,M,Wt).relu_()
    def v6_csr():
        M=torch.sparse.mm(g['Bcsr'],H)
        return torch.addmm(H0,M,Wt).relu_()
    def v7_padded():
        Hp=torch.cat([H,torch.zeros(1,d_h,device=dev)],0)
        S=Hp[padidx].sum(1)
        M=torch.index_select(S,0,src); M.sub_(torch.index_select(H,0,rev))
        return torch.addmm(H0,M,Wt).relu_()
    vs=[("v0 .repeat+scatter_reduce (CURRENT)",v0_repeat),
        ("v1 .expand+scatter_reduce",v1_expand),
        ("v2 index_add_",v2_indexadd),
        ("v3 +index_select/sub_/addmm/relu_",v3_inplace),
        ("v4 + rev via index_add_ alpha=-1",v4_revadd),
        ("v5 + rev via pair view-flip",v5_pairswap),
        ("v6 ExE non-backtracking CSR spmm",v6_csr),
        ("v7 padded dense gather+sum",v7_padded)]
    if g['Bcsr'] is None: vs=[x for x in vs if not x[0].startswith('v6')]
    return vs

for dev in (['cpu','mps'] if torch.backends.mps.is_available() else ['cpu']):
    sync=torch.mps.synchronize if dev=='mps' else None
    for d_h in (300,2048):
        g,H,H0,W=make(dev,d_h)
        vs=variants(g,H,H0,W,d_h,dev)
        print(f"\n=== {dev}, d_h={d_h}: ONE message+update layer ===")
        print(f"{'variant':<38}{'ms':>9}{'speedup':>9}{'max|err| vs v0':>17}")
        ref=None; t0=None
        for nm,fn in vs:
            with torch.no_grad():
                o=fn()
                if sync: sync()
                mn,_=timeit(fn,sync=sync)
            if ref is None: ref,t0=o.float().cpu(),mn
            err=(o.float().cpu()-ref).abs().max().item()
            print(f"{nm:<38}{mn*1e3:>9.3f}{t0/mn:>9.2f}{err:>17.3e}")
