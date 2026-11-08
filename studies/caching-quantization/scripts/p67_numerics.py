"""P6/P7: bf16 cancellation in the factored Sigma - H_rev form vs explicit Hashimoto,
stratified by deg(src(e)).  Real CheMeleon weights, real activations, fp64 reference."""
import sys; sys.path.insert(0,'/private/tmp/claude-502/-Users-svaddadi-Documents-GitHub-chemprop/54c934b6-e3f8-4460-b930-d5d089b68966/scratchpad')
from common import *
from hash_mp import *
torch.set_grad_enabled(False)
b=build_batch(); V,E,ei,rev=b['V'],b['E'],b['ei'],b['rev']; nV,nE=b['n_atom'],b['n_edge']
bk=build_buckets(ei,rev,nV); perm=bk['perm']
src,dst=ei[0],ei[1]
deg=torch.bincount(src,minlength=nV); degsrc=deg[src]            # original order
degsrc_p=degsrc[perm]
d_h,depth=2048,6
sd=load_chemeleon()['state_dict']
W_i=sd['W_i.weight'].double(); W_h=sd['W_h.weight'].double()

# ---- fp64 forward to get REAL activations H^t ----
H0=(torch.cat([V[src],E],1).double()@W_i.t())
Hs=[torch.relu(H0)]
for t in range(1,depth):
    M=fact_message(Hs[-1],src,dst,rev,nV)
    Hs.append(torch.relu(H0+M@W_h.t()))
print(f"real activation magnitudes |H^t| (fp64, CheMeleon d_h=2048 depth=6):")
for t,H in enumerate(Hs): print(f"  t={t}: mean|H| {H.abs().mean():.4g}  max {H.abs().max():.4g}  frac zero {(H==0).double().mean():.3f}")

def err_table(H64, dt, acc32):
    """H64 fp64 activations -> round to dt, compute M three ways, compare to fp64-exact M of the
    rounded input.  acc32=True does the reduction/subtraction in fp32 (P7)."""
    Hd = H64.to(dt).double()                       # exact value of the rounded input
    Mref = fact_message(Hd, src, dst, rev, nV)     # fp64-exact M of the rounded input
    Hq = H64.to(dt)
    adt = torch.float32 if acc32 else dt
    Nf = torch.zeros(nV,d_h,dtype=adt).index_add_(0,dst,Hq.to(adt))
    Mf = (Nf[src] - Hq.to(adt)[rev]).to(dt)
    Hqp = Hq[perm]
    if acc32:
        parts=[]
        for r,idx in bk['buckets']:
            if idx is None: continue
            if r==0: parts.append(torch.zeros(idx.shape[0],d_h,dtype=torch.float32))
            else:
                acc=Hqp.to(torch.float32)[idx[:,0]]
                for k in range(1,r): acc=acc+Hqp.to(torch.float32)[idx[:,k]]
                parts.append(acc)
        Mh=torch.cat(parts,0).to(dt)
    else:
        Mh = hash_message(Hqp, bk)
    # relative error, denominator = scale of the true message (per row, L2)
    def relerr(M, order_perm):
        R = Mref[perm] if order_perm else Mref
        dn = R.double().norm(dim=1); num=(M.double()-R.double()).norm(dim=1)
        return num/dn.clamp(min=1e-30), dn
    ef,dnf = relerr(Mf, False)
    eh,dnh = relerr(Mh, True)
    return ef, eh, dnf

for dt,name in ((torch.bfloat16,"bf16"),(torch.float32,"fp32")):
  for acc32 in ((False,True) if dt==torch.bfloat16 else (False,)):
    tag=f"{name}{' + fp32 accumulate (P7)' if acc32 else ''}"
    print(f"\n######## {tag} ########")
    print(f"{'layer':>6}  {'form':<12}{'all':>11}" + "".join(f"{'deg='+str(k):>11}" for k in (1,2,3,4)))
    for t in range(0,depth-1):
        ef,eh,_=err_table(Hs[t],dt,acc32)
        for nm,e,dg in (("factored",ef,degsrc),("hashimoto",eh,degsrc_p)):
            row=f"{t:>6}  {nm:<12}{e.max():>11.2e}"
            for k in (1,2,3,4):
                m=dg==k
                row+=f"{(e[m].max() if m.any() else float('nan')):>11.2e}"
            print(row)
        if t==0: print(f"        (rows per degree: " + " ".join(f"d={k}:{int((degsrc==k).sum())}" for k in (1,2,3,4))+")")

# ---- magnitude ratio / cancellation amplification at degree 2 ----
print("\n######## cancellation amplification |N_src| / |M| (the thing that drives the error) ########")
for t in (0,2,4):
    H=Hs[t]; N=torch.zeros(nV,d_h,dtype=torch.float64).index_add_(0,dst,H)
    Nsrc=N[src]; M=Nsrc-H[rev]
    amp=Nsrc.norm(dim=1)/M.norm(dim=1).clamp(min=1e-300)
    for k in (2,3,4):
        m=degsrc==k
        if not m.any(): continue
        a=amp[m]; q=torch.quantile(a,torch.tensor([.5,.9,.99],dtype=torch.float64))
        print(f"  t={t} deg={k}: |N|/|M| median {q[0]:.2f} p90 {q[1]:.2f} p99 {q[2]:.2f} max {a.max():.1f}  (n={int(m.sum())})")
    # elementwise |H_f|/|H_rev| at degree 2
    m=degsrc==2
    Hrev=H[rev][m]; Hf=M[m]
    nz=(Hrev.abs()>0)&(Hf.abs()>0)
    rat=(Hrev.abs()[nz]/Hf.abs()[nz])
    q=torch.quantile(rat,torch.tensor([.5,.9,.99],dtype=torch.float64))
    print(f"        elementwise |H_rev|/|H_f| at deg=2: median {q[0]:.2f} p90 {q[1]:.2f} p99 {q[2]:.2f} max {rat.max():.3g}")
