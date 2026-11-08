"""FAIR test (no frozen-weight handicap): train D-MPNN from scratch on ESOL.
Does input-injected RWSE (local, cacheable) let depth 3 match depth 6?
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
torch.manual_seed(0); np.random.seed(0)
rows=[l.strip().split(",") for l in open(f"{REPO}/tests/data/regression.csv").readlines()[1:]]
pairs=[(s,float(v)) for s,v in rows if Chem.MolFromSmiles(s) is not None]
mols=[Chem.MolFromSmiles(s) for s,_ in pairs]; y=np.array([v for _,v in pairs],dtype=np.float32)
feat=SimpleMoleculeMolGraphFeaturizer()

def rwse(m,K):
    n=m.GetNumAtoms(); A=np.zeros((n,n))
    for b in m.GetBonds(): A[b.GetBeginAtomIdx(),b.GetEndAtomIdx()]=A[b.GetEndAtomIdx(),b.GetBeginAtomIdx()]=1
    d=A.sum(1); d[d==0]=1; P=A/d[:,None]; Pk=np.eye(n); o=np.zeros((n,K))
    for k in range(K): Pk=Pk@P; o[:,k]=np.diag(Pk)
    return o.astype(np.float32)

def pack(idx, K):
    gs=[feat(mols[i]) for i in idx]
    Vs=[]
    for j,i in enumerate(idx):
        V=gs[j].V.astype(np.float32)
        if K: V=np.hstack([V, rwse(mols[i],K)])
        Vs.append(V)
    V=torch.from_numpy(np.concatenate(Vs))
    E=torch.from_numpy(np.concatenate([g.E for g in gs]).astype(np.float32))
    oa=oe=0; eis=[];revs=[];bt=[]
    for j,g in enumerate(gs):
        eis.append(g.edge_index+oa); revs.append(g.rev_edge_index+oe); bt+=[j]*g.V.shape[0]
        oa+=g.V.shape[0]; oe+=g.edge_index.shape[1]
    return (V,E,torch.from_numpy(np.hstack(eis)).long(),
            torch.from_numpy(np.concatenate(revs)).long(),
            torch.tensor(bt).long(), len(idx), torch.from_numpy(y[idx]))

class Net(nn.Module):
    def __init__(self,d_v,d_e=14,d_h=300,depth=3):
        super().__init__()
        self.Wi=nn.Linear(d_v+d_e,d_h,bias=False); self.Wh=nn.Linear(d_h,d_h,bias=False)
        self.Wo=nn.Linear(d_v+d_h,d_h); self.depth=depth
        self.head=nn.Sequential(nn.Linear(d_h,300),nn.ReLU(),nn.Linear(300,1))
    def forward(self,V,E,ei,rev,bt,nm):
        H0=self.Wi(torch.cat([V[ei[0]],E],1)); H=torch.relu(H0); dh=H.shape[1]
        for _ in range(1,self.depth):
            idx=ei[1].unsqueeze(1).expand(-1,dh)
            M=torch.zeros(V.shape[0],dh).scatter_reduce_(0,idx,H,reduce="sum",include_self=False)[ei[0]]-H[rev]
            H=torch.relu(H0+self.Wh(M))
        idx=ei[1].unsqueeze(1).expand(-1,dh)
        M=torch.zeros(V.shape[0],dh).scatter_reduce_(0,idx,H,reduce="sum",include_self=False)
        Hv=torch.relu(self.Wo(torch.cat((V,M),1)))
        c=torch.zeros(nm).index_add_(0,bt,torch.ones(bt.shape[0])).clamp(min=1)
        g=torch.zeros(nm,dh).index_add_(0,bt,Hv)/c.unsqueeze(1)
        return self.head(g).squeeze(1)

scaf={}
for i,m in enumerate(mols):
    scaf.setdefault(MurckoScaffold.MurckoScaffoldSmiles(mol=m,includeChirality=False),[]).append(i)
folds=[[] for _ in range(5)]
for g in sorted(scaf.values(),key=len,reverse=True): min(folds,key=len).extend(g)

def run(depth,K,seed):
    torch.manual_seed(seed); errs=[]
    for f in range(5):
        te=folds[f]; tr=[i for k in range(5) if k!=f for i in folds[k]]
        Btr=pack(tr,K); Bte=pack(te,K)
        ym,ys=float(Btr[6].mean()),float(Btr[6].std())
        net=Net(Btr[0].shape[1],depth=depth)
        opt=torch.optim.Adam(net.parameters(),lr=1e-3)
        sch=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=2e-3,total_steps=200)
        for ep in range(200):
            net.train(); opt.zero_grad()
            p=net(*Btr[:6]); loss=((p-(Btr[6]-ym)/ys)**2).mean(); loss.backward(); opt.step(); sch.step()
        net.eval()
        with torch.no_grad():
            p=net(*Bte[:6])*ys+ym
            errs.append(float(torch.sqrt(((p-Bte[6])**2).mean())))
    return float(np.mean(errs))

print(f"{'config':<34}{'dim_v':>7}{'RMSE (3 seeds)':>18}{'sd':>8}")
t0=time.time()
for nm,depth,K in [("depth 3, no PE",3,0),("depth 6, no PE",6,0),
                   ("depth 3 + RWSE K=6 (radius 3)",3,6),
                   ("depth 3 + RWSE K=12 (radius 6)",3,12),
                   ("depth 6 + RWSE K=6 (radius 3)",6,6)]:
    rs=[run(depth,K,s) for s in (0,1,2)]
    dv=72+K
    print(f"{nm:<34}{dv:>7}{np.mean(rs):>18.4f}{np.std(rs):>8.4f}")
print(f"\n({time.time()-t0:.0f}s; 5-fold scaffold-grouped CV, d_h=300, 200 full-batch Adam steps, 3 seeds)")
