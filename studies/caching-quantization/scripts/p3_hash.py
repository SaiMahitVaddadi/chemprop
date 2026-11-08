import sys; sys.path.insert(0,'/private/tmp/claude-502/-Users-svaddadi-Documents-GitHub-chemprop/54c934b6-e3f8-4460-b930-d5d089b68966/scratchpad')
from common import *
from hash_mp import *
from mp_opt import MPOpt
torch.set_grad_enabled(False)

b = build_batch(); V,E,ei,rev = b['V'],b['E'],b['ei'],b['rev']
nV,nE = b['n_atom'], b['n_edge']
print(f"batch: {b['n_mol']} mols, {nV} atoms, {nE} directed bonds")

import time
t=time.perf_counter(); bk = build_buckets(ei, rev, nV); tb=time.perf_counter()-t
print(f"build_buckets: {tb*1e3:.2f} ms (once per batch)   nnz(B)={bk['nnz']}  nnz/row={bk['nnz']/nE:.4f}")
print(f"bucket occupancy: {[(r,c) for r,c in bk['counts'] if c]}  -> {sum(1 for r,c in bk['counts'] if c)} kernels in cat")
idxbytes = sum(t.numel()*8 for r,t in bk['buckets'] if t is not None)
print(f"bucket index memory: {idxbytes/1e6:.3f} MB int64 ({idxbytes/2e6:.3f} MB as int32)")

# ---------- EXACTNESS in fp64 ----------
print("\n=== P3 exactness: explicit Hashimoto vs factored Sigma - H_rev, fp64 ===")
src,dst = ei[0],ei[1]
for trial,(gen) in enumerate([lambda: torch.randn(nE,64,dtype=torch.float64),
                              lambda: torch.randn(nE,64,dtype=torch.float64)*1e3,
                              lambda: torch.rand(nE,64,dtype=torch.float64)]):
    H = gen()
    Mf = fact_message(H, src, dst, rev, nV)       # original edge order
    Mh = hash_message(H[bk['perm']], bk)          # bucket order
    d = (Mh - Mf[bk['perm']]).abs().max().item()
    rel = d/Mf.abs().max().item()
    print(f"  trial {trial}: max|abs diff| = {d:.3e}   rel {rel:.3e}   bitwise-equal {torch.equal(Mh, Mf[bk['perm']])}")
# also against the explicit dense B for a few small molecules (independent check)
print("  (dense-B cross-check on 30 molecules, fp64):")
import numpy as np
from rdkit import Chem
from cp_loader import SimpleMoleculeMolGraphFeaturizer
feat=SimpleMoleculeMolGraphFeaturizer()
smis=[l.split(",")[0].strip() for l in open(f"{REPO}/tests/data/regression.csv").readlines()[1:]][:40]
mx=0.0; cnt=0
for s in smis:
    m=Chem.MolFromSmiles(s)
    if m is None or m.GetNumBonds()==0: continue
    g=feat(m); e2=torch.from_numpy(g.edge_index).long(); rv=torch.from_numpy(g.rev_edge_index).long()
    n=g.V.shape[0]; ne=e2.shape[1]
    B=torch.zeros(ne,ne,dtype=torch.float64)
    for e in range(ne):
        for f in range(ne):
            if e2[1,f]==e2[0,e] and f!=rv[e]: B[e,f]=1
    H=torch.randn(ne,9,dtype=torch.float64)
    bb=build_buckets(e2,rv,n)
    Mh=hash_message(H[bb['perm']],bb); Mb=(B@H)[bb['perm']]
    mx=max(mx,float((Mh-Mb).abs().max())); cnt+=1
print(f"    max|bucketed - dense B@H| over {cnt} molecules: {mx:.3e}")

# ---------- full-forward benchmark ----------
cfgs = [("d_h=300 depth=3", 300, 3), ("d_h=2048 depth=6", 2048, 6)]
for label,d_h,depth in cfgs:
    ref = MPOpt(d_h,depth,mode="repeat").eval()
    if d_h==2048: ref.load_state_dict(load_chemeleon()['state_dict'])
    hm = MPHash(d_h,depth).eval(); hm.load_state_dict(ref.state_dict())
    print(f"\n########## {label} ##########")
    for dev in (['cpu','mps'] if torch.backends.mps.is_available() else ['cpu']):
        sync = torch.mps.synchronize if dev=='mps' else None
        Vd,Ed,eid,revd = V.to(dev),E.to(dev),ei.to(dev),rev.to(dev)
        bkd = bk_to(bk, dev); Ep = E[bk['perm']].to(dev)
        mo = MPOpt(d_h,depth,mode="index_add").eval(); mo.load_state_dict(ref.state_dict()); mo=mo.to(dev)
        mr = MPOpt(d_h,depth,mode="repeat").eval(); mr.load_state_dict(ref.state_dict()); mr=mr.to(dev)
        mh = hm.to(dev)
        o_r = mr(Vd,Ed,eid,revd); o_i = mo(Vd,Ed,eid,revd); o_h = mh(Vd,Ep,bkd)
        if sync: sync()
        t_r,_ = timeit(lambda: mr(Vd,Ed,eid,revd), sync=sync)
        t_i,_ = timeit(lambda: mo(Vd,Ed,eid,revd), sync=sync)
        t_h,_ = timeit(lambda: mh(Vd,Ep,bkd), sync=sync)
        R=o_r.float().cpu()
        print(f"  {dev}: repeat(current) {t_r*1e3:8.1f} ms | index_add {t_i*1e3:8.1f} ms ({t_r/t_i:.2f}x) "
              f"| hashimoto {t_h*1e3:8.1f} ms ({t_r/t_h:.2f}x vs current, {t_i/t_h:.2f}x vs index_add)")
        print(f"       max|diff vs current|: index_add {(o_i.float().cpu()-R).abs().max():.3e}  "
              f"hashimoto {(o_h.float().cpu()-R).abs().max():.3e}   "
              f"bitwise(hash==cur) {torch.equal(o_h.float().cpu(),R)}")
        mh = mh.cpu()
