"""NOVEL ABLATION: M_e = sum_f H_f - gamma * H_rev(e).
gamma=1 -> chemprop's non-backtracking (Hashimoto B).  gamma=0 -> ordinary random walk
(simple directed-edge adjacency, backtracking allowed).  Also a LEARNED gamma.
How much is non-backtracking actually worth on a real molecular task?
"""
import sys, time
SP="/private/tmp/claude-502/-Users-svaddadi-Documents-GitHub-chemprop/54c934b6-e3f8-4460-b930-d5d089b68966/scratchpad"
sys.path.insert(0,SP)
import numpy as np, torch, torch.nn as nn
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
RDLogger.DisableLog("rdApp.*")
from cp_loader import SimpleMoleculeMolGraphFeaturizer
REPO="/Users/svaddadi/Documents/GitHub/chemprop"
rows=[l.strip().split(",") for l in open(f"{REPO}/tests/data/regression.csv").readlines()[1:]]
pairs=[(s,float(v)) for s,v in rows if Chem.MolFromSmiles(s) is not None]
mols=[Chem.MolFromSmiles(s) for s,_ in pairs]; y=np.array([v for _,v in pairs],dtype=np.float32)
feat=SimpleMoleculeMolGraphFeaturizer()
def pack(idx):
    gs=[feat(mols[i]) for i in idx]
    V=torch.from_numpy(np.concatenate([g.V for g in gs]).astype(np.float32))
    E=torch.from_numpy(np.concatenate([g.E for g in gs]).astype(np.float32))
    oa=oe=0; eis=[];revs=[];bt=[]
    for j,g in enumerate(gs):
        eis.append(g.edge_index+oa); revs.append(g.rev_edge_index+oe); bt+=[j]*g.V.shape[0]
        oa+=g.V.shape[0]; oe+=g.edge_index.shape[1]
    return (V,E,torch.from_numpy(np.hstack(eis)).long(),
            torch.from_numpy(np.concatenate(revs)).long(),
            torch.tensor(bt).long(),len(idx),torch.from_numpy(y[idx]))
class Net(nn.Module):
    def __init__(self,d_v=72,d_e=14,d_h=300,depth=3,gamma=1.0,learn=False):
        super().__init__()
        self.Wi=nn.Linear(d_v+d_e,d_h,bias=False); self.Wh=nn.Linear(d_h,d_h,bias=False)
        self.Wo=nn.Linear(d_v+d_h,d_h); self.depth=depth
        self.head=nn.Sequential(nn.Linear(d_h,300),nn.ReLU(),nn.Linear(300,1))
        if learn: self.g=nn.Parameter(torch.tensor(float(gamma)))
        else: self.register_buffer("g",torch.tensor(float(gamma)))
    def forward(self,V,E,ei,rev,bt,nm):
        H0=self.Wi(torch.cat([V[ei[0]],E],1)); H=torch.relu(H0); dh=H.shape[1]
        for _ in range(1,self.depth):
            idx=ei[1].unsqueeze(1).expand(-1,dh)
            M=torch.zeros(V.shape[0],dh).scatter_reduce_(0,idx,H,reduce="sum",
                  include_self=False)[ei[0]] - self.g*H[rev]
            H=torch.relu(H0+self.Wh(M))
        idx=ei[1].unsqueeze(1).expand(-1,dh)
        M=torch.zeros(V.shape[0],dh).scatter_reduce_(0,idx,H,reduce="sum",include_self=False)
        Hv=torch.relu(self.Wo(torch.cat((V,M),1)))
        c=torch.zeros(nm).index_add_(0,bt,torch.ones(bt.shape[0])).clamp(min=1)
        return self.head(torch.zeros(nm,dh).index_add_(0,bt,Hv)/c.unsqueeze(1)).squeeze(1)
scaf={}
for i,m in enumerate(mols):
    scaf.setdefault(MurckoScaffold.MurckoScaffoldSmiles(mol=m,includeChirality=False),[]).append(i)
folds=[[] for _ in range(5)]
for g in sorted(scaf.values(),key=len,reverse=True): min(folds,key=len).extend(g)
PK=[ (pack([i for k in range(5) if k!=f for i in folds[k]]), pack(folds[f])) for f in range(5)]
def run(depth,gamma,learn,seed):
    torch.manual_seed(seed); errs=[]; gs=[]
    for Btr,Bte in PK:
        ym,ys=float(Btr[6].mean()),float(Btr[6].std())
        net=Net(depth=depth,gamma=gamma,learn=learn)
        opt=torch.optim.Adam(net.parameters(),lr=1e-3)
        sch=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=2e-3,total_steps=200)
        for _ in range(200):
            net.train(); opt.zero_grad()
            ((net(*Btr[:6])-(Btr[6]-ym)/ys)**2).mean().backward(); opt.step(); sch.step()
        net.eval()
        with torch.no_grad():
            p=net(*Bte[:6])*ys+ym; errs.append(float(torch.sqrt(((p-Bte[6])**2).mean())))
        gs.append(float(net.g))
    return float(np.mean(errs)), float(np.mean(gs))
print("ESOL, 5-fold scaffold-grouped CV, d_h=300, 200 full-batch Adam steps, 3 seeds")
print(f"\n{'gamma (reverse-edge coeff)':<36}{'depth':>7}{'RMSE':>9}{'seed sd':>9}{'final gamma':>13}")
for depth in (3,6):
    for lbl,gm,lr in [("0.0  = ordinary RW (backtracking)",0.0,False),
                      ("0.5  = half-suppressed",0.5,False),
                      ("1.0  = chemprop / Hashimoto B",1.0,False),
                      ("learned (init 1.0)",1.0,True)]:
        rs=[run(depth,gm,lr,s) for s in (0,1)]
        print(f"{lbl:<36}{depth:>7}{np.mean([r[0] for r in rs]):>9.4f}"
              f"{np.std([r[0] for r in rs]):>9.4f}{np.mean([r[1] for r in rs]):>13.3f}")
