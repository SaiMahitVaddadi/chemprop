"""P9: dense block-diagonal bmm formulation vs every scatter-based variant."""
import sys; sys.path.insert(0,'/private/tmp/claude-502/-Users-svaddadi-Documents-GitHub-chemprop/54c934b6-e3f8-4460-b930-d5d089b68966/scratchpad')
from common import *
from hash_mp import *
from mp_opt import MPOpt
from robust import race, show
import numpy as np
torch.set_grad_enabled(False)
d_h=int(sys.argv[1]); depth=int(sys.argv[2]); dev=sys.argv[3]
nbuck=int(sys.argv[4]) if len(sys.argv)>4 else 10
rounds=int(sys.argv[5]) if len(sys.argv)>5 else 7
b=build_batch(); V,E,ei,rev=b['V'],b['E'],b['ei'],b['rev']; nV,nE=b['n_atom'],b['n_edge']
bk=build_buckets(ei,rev,nV); bkd=bk_to(bk,dev)
sync=torch.mps.synchronize if dev=='mps' else None
sd=MP(d_h,depth).state_dict()
if d_h==2048: sd=load_chemeleon()['state_dict']

# ---- per-molecule decomposition of the batch ----
# recover molecule boundaries from edge_index via the atom->mol map
batch=b['batch']; src,dst=ei[0],ei[1]
mol_of_edge=batch[src]
atom_off=torch.cat([torch.tensor([0]),torch.cumsum(torch.bincount(batch,minlength=b['n_mol']),0)])
edge_cnt=torch.bincount(mol_of_edge,minlength=b['n_mol'])
edge_off=torch.cat([torch.tensor([0]),torch.cumsum(edge_cnt,0)])
m2=edge_cnt.numpy()
print(f"{b['n_mol']} mols, 2m: mean {m2.mean():.1f} p50 {np.percentile(m2,50):.0f} p95 {np.percentile(m2,95):.0f} max {m2.max()}")

def build_group(mols, Emax):
    """dense padded B, plus scatter maps, for a list of molecule ids."""
    G=len(mols)
    Bp=torch.zeros(G,Emax,Emax)
    eidx=torch.zeros(G,Emax,dtype=torch.long)          # global edge id for each (g,slot); pad->0
    emask=torch.zeros(G,Emax,dtype=torch.bool)
    for g,i in enumerate(mols):
        lo,hi=int(edge_off[i]),int(edge_off[i+1]); ne=hi-lo
        s=src[lo:hi]-atom_off[i]; d=dst[lo:hi]-atom_off[i]; r=rev[lo:hi]-lo
        # B[e,f]=1 iff d[f]==s[e] and f!=rev(e)
        Bl=(d.unsqueeze(0)==s.unsqueeze(1)).float()
        Bl[torch.arange(ne),r]=0.0
        Bp[g,:ne,:ne]=Bl
        eidx[g,:ne]=torch.arange(lo,hi); emask[g,:ne]=True
    return Bp,eidx,emask

class MPBmm(torch.nn.Module):
    def __init__(self,d_h,depth,d_v=72,d_e=14):
        super().__init__()
        self.W_i=torch.nn.Linear(d_v+d_e,d_h,bias=False)
        self.W_h=torch.nn.Linear(d_h,d_h,bias=False)
        self.W_o=torch.nn.Linear(d_v+d_h,d_h); self.depth=depth; self.d_h=d_h
    def forward(self,V,groups):
        """groups: list of (Bp, Vsrc_feat, E_feat, dstflat, mask) all padded, on device"""
        Mn=torch.zeros(V.shape[0],self.d_h,dtype=V.dtype,device=V.device)
        Wt=self.W_h.weight.t()
        for Bp,Vs,Ee,dflat,mask in groups:
            G,Em,_=Bp.shape
            H_0=self.W_i(torch.cat([Vs,Ee],dim=-1))*mask.unsqueeze(-1)
            H=torch.relu(H_0)
            for _ in range(1,self.depth):
                M=torch.bmm(Bp,H)
                H=torch.addmm(H_0.view(-1,self.d_h),M.view(-1,self.d_h),Wt).relu_().view(G,Em,self.d_h)
            Mn.index_add_(0,dflat,(H*mask.unsqueeze(-1)).view(-1,self.d_h))
        return self.W_o(torch.cat((V,Mn),dim=1))

def make_groups(policy):
    order=np.argsort(m2,kind="stable")
    if policy=="batchmax":
        groups=[list(range(b['n_mol']))]
    else:
        gs=int(np.ceil(b['n_mol']/nbuck))
        groups=[order[i:i+gs].tolist() for i in range(0,b['n_mol'],gs)]
    out=[]; pad_edges=0; real_edges=0; bflops=0
    for mols in groups:
        Em=int(max(m2[i] for i in mols))
        Bp,eidx,emask=build_group(mols,Em)
        G=len(mols); pad_edges+=G*Em; real_edges+=sum(int(m2[i]) for i in mols)
        bflops+=2*G*Em*Em*d_h
        Vs=V[src[eidx.view(-1)]].view(G,Em,-1)*emask.unsqueeze(-1)
        Ee=E[eidx.view(-1)].view(G,Em,-1)*emask.unsqueeze(-1)
        dflat=dst[eidx.view(-1)]
        out.append((Bp.to(dev),Vs.to(dev),Ee.to(dev),dflat.to(dev),emask.to(dev)))
    return out,pad_edges,real_edges,bflops

mk=lambda C,**k:(lambda m:(m.load_state_dict(sd),m.eval().to(dev))[1])(C(d_h,depth,**k))
mr=mk(MP,mode="repeat"); mo=mk(MPOpt,mode="index_add"); mho=mk(MPHashSafe); mhc=mk(MPHash); mb=mk(MPBmm)
eid=torch.stack([src,dst]).to(dev); rv=rev.to(dev); Vd,Ed=V.to(dev),E.to(dev); Ep=E[bk['perm']].to(dev)
fns={"current(.repeat)":lambda:mr(Vd,Ed,eid,rv),
     "index_add_+select":lambda:mo(Vd,Ed,eid,rv),
     "hashimoto(copy_)":lambda:mho(Vd,Ep,bkd),
     "hashimoto(cat)":lambda:mhc(Vd,Ep,bkd)}
gw={}
for pol in ("batchmax","bucketed"):
    g,pe,re_,bf=make_groups(pol)
    gw[pol]=g
    gf_wh=2*pe*d_h*d_h*(depth-1); gf_real=2*re_*d_h*d_h*(depth-1)
    print(f"  {pol:<10}: groups={len(g)} padded_edges={pe} real={re_} inflation={pe/re_:.3f}x "
          f"| W_h GEMM {gf_wh/1e9:.1f} GFLOP vs {gf_real/1e9:.1f} real | B-bmm {bf*(depth-1)/1e9:.1f} GFLOP "
          f"= {bf*(depth-1)/gf_real*100:.1f}% of real W_h GEMM")
    fns[f"bmm/{pol}"]=(lambda gg: (lambda: mb(Vd,gg)))(g)
print(f"\n=== FULL forward {dev} d_h={d_h} depth={depth} rounds={rounds} ===")
show(race(fns,rounds=rounds,sync=sync))
# ---- fp64 reference ----
W_i,W_h,W_o,bo=sd['W_i.weight'].double(),sd['W_h.weight'].double(),sd['W_o.weight'].double(),sd['W_o.bias'].double()
H0=torch.cat([V[src],E],1).double()@W_i.t(); H=torch.relu(H0)
for _ in range(1,depth): H=torch.relu(H0+fact_message(H,src,dst,rev,nV)@W_h.t())
Mn=torch.zeros(nV,d_h,dtype=torch.float64).index_add_(0,dst,H)
REF=torch.cat((V.double(),Mn),1)@W_o.t()+bo; sc=float(REF.abs().max())
print(f"  {'variant':<22}{'max|err| vs fp64':>18}{'rel':>12}")
for n,f in fns.items():
    o=f().cpu().double(); e=float((o-REF).abs().max())
    print(f"  {n:<22}{e:>18.3e}{e/sc:>12.3e}")
