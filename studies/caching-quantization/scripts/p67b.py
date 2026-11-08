"""P6/P7 (scoped): bf16 cancellation, factored vs Hashimoto, fp64 ref, stratified by deg(src(e)).
Real CheMeleon weights + real activations. 150-mol batch, layers t=0,2,4."""
import sys; sys.path.insert(0,'/private/tmp/claude-502/-Users-svaddadi-Documents-GitHub-chemprop/54c934b6-e3f8-4460-b930-d5d089b68966/scratchpad')
from common import *
from hash_mp import *
torch.set_grad_enabled(False)
b=build_batch(n=150); V,E,ei,rev=b['V'],b['E'],b['ei'],b['rev']; nV,nE=b['n_atom'],b['n_edge']
bk=build_buckets(ei,rev,nV); perm=bk['perm']; src,dst=ei[0],ei[1]
deg=torch.bincount(src,minlength=nV); dg=deg[src]; dgp=dg[perm]
d_h,depth=2048,6
sd=load_chemeleon()['state_dict']; W_i=sd['W_i.weight'].double(); W_h=sd['W_h.weight'].double()
print(f"batch {b['n_mol']} mols {nV} atoms {nE} directed bonds; rows per deg: "
      + " ".join(f"d={k}:{int((dg==k).sum())}({(dg==k).double().mean()*100:.1f}%)" for k in (1,2,3,4)))
H0=torch.cat([V[src],E],1).double()@W_i.t(); Hs=[torch.relu(H0)]
for t in range(1,depth): Hs.append(torch.relu(H0+fact_message(Hs[-1],src,dst,rev,nV)@W_h.t()))
print("\nreal activation magnitudes (fp64):")
for t in (0,2,4): print(f"  t={t}: mean|H| {Hs[t].abs().mean():.4g} max {Hs[t].abs().max():.4g} frac-zero {(Hs[t]==0).double().mean():.3f}")

def run(H64, dt, acc32):
    Hq=H64.to(dt); Hd=Hq.double()
    Mref=fact_message(Hd,src,dst,rev,nV)                 # fp64-exact M of the ROUNDED input
    adt=torch.float32 if acc32 else dt
    N=torch.zeros(nV,d_h,dtype=adt).index_add_(0,dst,Hq.to(adt))
    Mf=(N[src]-Hq.to(adt)[rev]).to(dt)
    Hqp=Hq[perm]
    if acc32:
        parts=[]
        for r,idx in bk['buckets']:
            if idx is None: continue
            if r==0: parts.append(torch.zeros(idx.shape[0],d_h,dtype=torch.float32))
            else:
                acc=Hqp[idx[:,0]].to(torch.float32)
                for k in range(1,r): acc=acc+Hqp[idx[:,k]].to(torch.float32)
                parts.append(acc)
        Mh=torch.cat(parts,0).to(dt)
    else:
        Mh=hash_message(Hqp,bk)
    def rel(M,pm):
        R=(Mref[perm] if pm else Mref).double()
        return (M.double()-R).norm(dim=1)/R.norm(dim=1).clamp(min=1e-300)
    return rel(Mf,False), rel(Mh,True)

for dt,name in ((torch.bfloat16,"bf16"),(torch.float32,"fp32")):
  for acc32 in ((False,True) if dt==torch.bfloat16 else (False,)):
    print(f"\n######## {name}{'  + fp32 reduction (P7)' if acc32 else ''} :: row-wise RELATIVE error of M vs fp64 ########")
    print(f"{'t':>3} {'form':<11}{'p50 all':>10}{'max all':>10}" + "".join(f"{'deg='+str(k):>20}" for k in (1,2,3)))
    print(f"{'':>3} {'':<11}{'':>10}{'':>10}" + "".join(f"{'p50':>10}{'max':>10}" for k in (1,2,3)))
    for t in (0,2,4):
        ef,eh=run(Hs[t],dt,acc32)
        for nm,e,d in (("factored",ef,dg),("hashimoto",eh,dgp)):
            row=f"{t:>3} {nm:<11}{e.median():>10.2e}{e.max():>10.2e}"
            for k in (1,2,3):
                m=d==k
                row+= f"{e[m].median():>10.2e}{e[m].max():>10.2e}" if m.any() else f"{'-':>10}{'-':>10}"
            print(row)

print("\n######## cancellation amplification |N_src(e)| / |M_e| (drives the error) ########")
for t in (0,2,4):
    H=Hs[t]; N=torch.zeros(nV,d_h,dtype=torch.float64).index_add_(0,dst,H)
    Ns=N[src]; M=Ns-H[rev]; amp=Ns.norm(dim=1)/M.norm(dim=1).clamp(min=1e-300)
    out=[]
    for k in (2,3,4):
        m=dg==k
        if not m.any(): continue
        q=torch.quantile(amp[m],torch.tensor([.5,.9,.99],dtype=torch.float64))
        out.append(f"deg{k}: p50 {q[0]:.2f} p90 {q[1]:.2f} p99 {q[2]:.2f} max {amp[m].max():.1f}")
    print(f"  t={t}  "+" | ".join(out))
    m=dg==2; Hr=H[rev][m].abs(); Mm=M[m].abs(); nz=(Hr>0)&(Mm>0)
    r=Hr[nz]/Mm[nz]; q=torch.quantile(r,torch.tensor([.5,.9,.99],dtype=torch.float64))
    print(f"        elementwise |H_rev|/|M| at deg=2: p50 {q[0]:.2f} p90 {q[1]:.2f} p99 {q[2]:.2f} max {r.max():.3g}")
