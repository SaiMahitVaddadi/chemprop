"""How much ESOL accuracy lives in CheMeleon's DEEP message-passing layers?
Frozen encoder at truncated depth -> mean agg -> ridge head, 5-fold scaffold-ish CV.
CAVEAT: weights were TRAINED at depth 6; truncation is unfair to shallow settings.
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
sd=ck["state_dict"]; hp=ck["hyper_parameters"]; D_H=hp["d_h"]
W_i,W_h,W_o,b_o=sd["W_i.weight"],sd["W_h.weight"],sd["W_o.weight"],sd["W_o.bias"]
rows=[l.strip().split(",") for l in open(f"{REPO}/tests/data/regression.csv").readlines()[1:]]
pairs=[(s,float(v)) for s,v in rows if Chem.MolFromSmiles(s) is not None]
feat=SimpleMoleculeMolGraphFeaturizer()
mols=[Chem.MolFromSmiles(s) for s,_ in pairs]
mgs=[feat(m) for m in mols]
y=np.array([v for _,v in pairs],dtype=np.float64)
V=torch.from_numpy(np.concatenate([g.V for g in mgs])).float()
E=torch.from_numpy(np.concatenate([g.E for g in mgs]).astype(np.float32))
oa=oe=0; eis=[];revs=[];bt=[]
for i,g in enumerate(mgs):
    eis.append(g.edge_index+oa); revs.append(g.rev_edge_index+oe); bt+=[i]*g.V.shape[0]
    oa+=g.V.shape[0]; oe+=g.edge_index.shape[1]
ei=torch.from_numpy(np.hstack(eis)).long(); rev=torch.from_numpy(np.concatenate(revs)).long()
bt=torch.tensor(bt).long(); NM=len(pairs)
diam=[int(Chem.GetDistanceMatrix(m).max()) for m in mols]
print(f"ESOL n={NM}  atoms={V.shape[0]}  mean heavy atoms={np.mean([m.GetNumAtoms() for m in mols]):.1f}"
      f"  mean graph diameter={np.mean(diam):.1f}  median={np.median(diam):.0f}  p90={np.percentile(diam,90):.0f}")

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

# scaffold-grouped 5-fold CV
scaf={}
for i,m in enumerate(mols):
    s=MurckoScaffold.MurckoScaffoldSmiles(mol=m,includeChirality=False)
    scaf.setdefault(s,[]).append(i)
groups=sorted(scaf.values(),key=len,reverse=True)
folds=[[] for _ in range(5)]
for g in groups: min(folds,key=len).extend(g)
def ridge_cv(X,alphas=(1e-2,1e-1,1,10,100,1e3,1e4)):
    best=None
    for a in alphas:
        errs=[]
        for f in range(5):
            te=np.array(folds[f]); tr=np.array([i for k in range(5) if k!=f for i in folds[k]])
            Xtr,Xte=X[tr],X[te]; mu=Xtr.mean(0); sd_=Xtr.std(0)+1e-8
            Xtr=(Xtr-mu)/sd_; Xte=(Xte-mu)/sd_
            ym=y[tr].mean()
            w=np.linalg.solve(Xtr.T@Xtr+a*np.eye(X.shape[1]), Xtr.T@(y[tr]-ym))
            p=Xte@w+ym; errs.append(np.sqrt(np.mean((p-y[te])**2)))
        r=float(np.mean(errs))
        if best is None or r<best[0]: best=(r,a,float(np.std(errs)))
    return best
print(f"\n{'depth':>6}{'RMSE':>9}{'+/-fold sd':>12}{'alpha':>9}{'vs depth 6':>12}")
res={}
for d in (1,2,3,4,5,6,7,8):
    X=enc(d); r,a,s=ridge_cv(X); res[d]=r
    print(f"{d:>6}{r:>9.4f}{s:>12.4f}{a:>9.3g}{'' if d==6 else f'{(r-res.get(6,r))/res.get(6,r)*100:>+11.1f}%'}")
print(f"\ndepth3 vs depth6 RMSE: {res[3]:.4f} vs {res[6]:.4f}  ->  {(res[3]-res[6])/res[6]*100:+.1f}%")
# baseline: atom features only (depth 0-ish) and ECFP for scale
from rdkit.Chem import rdFingerprintGenerator as fpg
gen=fpg.GetMorganGenerator(radius=2,fpSize=2048)
Xf=np.array([gen.GetCountFingerprintAsNumPy(m) for m in mols],dtype=np.float64)
r,a,s=ridge_cv(Xf); print(f"ECFP4-count-2048 + ridge (same CV): RMSE {r:.4f} (alpha {a:g})")
