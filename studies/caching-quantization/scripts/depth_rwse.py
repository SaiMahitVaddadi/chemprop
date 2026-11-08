"""Can LOCAL, cacheable random-walk structural encodings substitute for depth?
depth-3 CheMeleon embedding (+/- RWSE_K) vs depth-6, same frozen-encoder+ridge protocol.
"""
import sys
SP="/private/tmp/claude-502/-Users-svaddadi-Documents-GitHub-chemprop/54c934b6-e3f8-4460-b930-d5d089b68966/scratchpad"
sys.path.insert(0,SP)
import numpy as np, torch
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
RDLogger.DisableLog("rdApp.*")
from cp_loader import SimpleMoleculeMolGraphFeaturizer
torch.set_grad_enabled(False)
REPO="/Users/svaddadi/Documents/GitHub/chemprop"
ck=torch.load("/Users/svaddadi/Library/Caches/molfeat/chemeleon/chemeleon_mp.pt",
              weights_only=True,map_location="cpu")
sd=ck["state_dict"]; D_H=ck["hyper_parameters"]["d_h"]
W_i,W_h,W_o,b_o=sd["W_i.weight"],sd["W_h.weight"],sd["W_o.weight"],sd["W_o.bias"]
rows=[l.strip().split(",") for l in open(f"{REPO}/tests/data/regression.csv").readlines()[1:]]
pairs=[(s,float(v)) for s,v in rows if Chem.MolFromSmiles(s) is not None]
mols=[Chem.MolFromSmiles(s) for s,_ in pairs]
feat=SimpleMoleculeMolGraphFeaturizer(); mgs=[feat(m) for m in mols]
y=np.array([v for _,v in pairs]); NM=len(pairs)
V=torch.from_numpy(np.concatenate([g.V for g in mgs])).float()
E=torch.from_numpy(np.concatenate([g.E for g in mgs]).astype(np.float32))
oa=oe=0; eis=[];revs=[];bt=[]
for i,g in enumerate(mgs):
    eis.append(g.edge_index+oa); revs.append(g.rev_edge_index+oe); bt+=[i]*g.V.shape[0]
    oa+=g.V.shape[0]; oe+=g.edge_index.shape[1]
ei=torch.from_numpy(np.hstack(eis)).long(); rev=torch.from_numpy(np.concatenate(revs)).long()
bt=torch.tensor(bt).long()
def enc(depth):
    H_0=torch.nn.functional.linear(torch.cat([V[ei[0]],E],1),W_i); H=torch.relu(H_0)
    for _ in range(1,depth):
        idx=ei[1].unsqueeze(1).expand(-1,D_H)
        M=torch.zeros(V.shape[0],D_H).scatter_reduce_(0,idx,H,reduce="sum",
              include_self=False)[ei[0]]-H[rev]
        H=torch.relu(H_0+torch.nn.functional.linear(M,W_h))
    idx=ei[1].unsqueeze(1).expand(-1,D_H)
    M=torch.zeros(V.shape[0],D_H).scatter_reduce_(0,idx,H,reduce="sum",include_self=False)
    Hv=torch.nn.functional.linear(torch.cat((V,M),1),W_o,b_o)
    cnt=torch.zeros(NM).index_add_(0,bt,torch.ones(bt.shape[0])).clamp(min=1)
    return (torch.zeros(NM,D_H).index_add_(0,bt,Hv)/cnt.unsqueeze(1)).double().numpy()
def rwse_mol(m,K):
    n=m.GetNumAtoms(); A=np.zeros((n,n))
    for b in m.GetBonds(): A[b.GetBeginAtomIdx(),b.GetEndAtomIdx()]=A[b.GetEndAtomIdx(),b.GetBeginAtomIdx()]=1
    d=A.sum(1); d[d==0]=1; P=A/d[:,None]; Pk=np.eye(n); out=np.zeros((n,K))
    for k in range(K): Pk=Pk@P; out[:,k]=np.diag(Pk)
    return out
def rwse_pool(K):
    return np.array([np.concatenate([rwse_mol(m,K).mean(0),rwse_mol(m,K).std(0)]) for m in mols])
scaf={}
for i,m in enumerate(mols):
    scaf.setdefault(MurckoScaffold.MurckoScaffoldSmiles(mol=m,includeChirality=False),[]).append(i)
folds=[[] for _ in range(5)]
for g in sorted(scaf.values(),key=len,reverse=True): min(folds,key=len).extend(g)
def ridge_cv(X):
    best=None
    for a in (1e-2,1e-1,1,10,100,1e3,1e4,1e5):
        errs=[]
        for f in range(5):
            te=np.array(folds[f]); tr=np.array([i for k in range(5) if k!=f for i in folds[k]])
            mu=X[tr].mean(0); s=X[tr].std(0)+1e-8
            Xtr=(X[tr]-mu)/s; Xte=(X[te]-mu)/s; ym=y[tr].mean()
            w=np.linalg.solve(Xtr.T@Xtr+a*np.eye(X.shape[1]),Xtr.T@(y[tr]-ym))
            errs.append(np.sqrt(np.mean((Xte@w+ym-y[te])**2)))
        if best is None or np.mean(errs)<best[0]: best=(float(np.mean(errs)),a,float(np.std(errs)))
    return best
E3,E4,E6=enc(3),enc(4),enc(6)
print(f"{'features':<46}{'dim':>7}{'RMSE':>9}{'foldSD':>9}{'vs d6':>9}")
r6=ridge_cv(E6)[0]
def row(nm,X):
    r,a,s=ridge_cv(X); print(f"{nm:<46}{X.shape[1]:>7}{r:>9.4f}{s:>9.4f}{(r-r6)/r6*100:>+8.1f}%"); return r
row("CheMeleon depth 6 (reference)",E6)
row("CheMeleon depth 3",E3)
row("CheMeleon depth 4",E4)
for K,rad in ((6,3),(8,4),(12,6),(20,10)):
    R=rwse_pool(K)
    row(f"RWSE K={K} alone  (local, cache radius {rad})",R)
    row(f"depth 3 + RWSE K={K} (cache radius max(3,{rad}))",np.hstack([E3,R]))
