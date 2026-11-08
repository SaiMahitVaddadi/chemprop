"""Is RWSE locally cacheable? And does canonical traversal give reusable prefixes?"""
import sys
SP="/private/tmp/claude-502/-Users-svaddadi-Documents-GitHub-chemprop/54c934b6-e3f8-4460-b930-d5d089b68966/scratchpad"
sys.path.insert(0,SP)
import numpy as np
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")

REPO="/Users/svaddadi/Documents/GitHub/chemprop"

def adj(m):
    n=m.GetNumAtoms(); A=np.zeros((n,n))
    for b in m.GetBonds():
        i,j=b.GetBeginAtomIdx(),b.GetEndAtomIdx(); A[i,j]=A[j,i]=1
    return A

def rwse(m,K=20):
    A=adj(m); d=A.sum(1); d[d==0]=1
    P=A/d[:,None]; out=np.zeros((A.shape[0],K)); Pk=np.eye(A.shape[0])
    for k in range(K):
        Pk=Pk@P; out[:,k]=np.diag(Pk)
    return out

def ball(m,v,r):
    """atom indices within graph distance r of v"""
    D=Chem.GetDistanceMatrix(m)
    return [i for i in range(m.GetNumAtoms()) if D[v,i]<=r]

def rwse_on_ball(m,v,r,K=20):
    idx=ball(m,v,r)
    sub=Chem.PathToSubmol(m, [b.GetIdx() for b in m.GetBonds()
         if b.GetBeginAtomIdx() in idx and b.GetEndAtomIdx() in idx], atomMap={})
    # build induced-subgraph adjacency directly instead (safer)
    A=adj(m)[np.ix_(idx,idx)]
    d=A.sum(1); d[d==0]=1; P=A/d[:,None]
    pos=idx.index(v); Pk=np.eye(len(idx)); out=np.zeros(K)
    for k in range(K):
        Pk=Pk@P; out[k]=Pk[pos,pos]
    return out

print("="*78)
print("EXP 1: RWSE_k locality -- does the k-step return probability at v depend")
print("        only on the ball of radius floor(k/2) around v?")
print("="*78)
smis=[l.split(",")[0].strip() for l in open(f"{REPO}/tests/data/regression.csv").readlines()[1:]]
mols=[m for m in (Chem.MolFromSmiles(s) for s in smis) if m is not None]
mols=[m for m in mols if 10<=m.GetNumAtoms()<=30][:60]
K=20
maxerr={r:0.0 for r in range(1,11)}
for m in mols:
    full=rwse(m,K)
    for v in range(m.GetNumAtoms()):
        for r in range(1,11):
            got=rwse_on_ball(m,v,r,K)
            for k in range(1,K+1):
                if k//2<=r:   # prediction: exact
                    maxerr[r]=max(maxerr[r],abs(got[k-1]-full[v,k-1]))
print(f"{'ball radius r':>14}{'max |RWSE_k(ball) - RWSE_k(full)| over all k with floor(k/2)<=r':>66}")
for r in range(1,11):
    print(f"{r:>14}{maxerr[r]:>66.3e}")
print("\n(prediction: identically zero -- a closed walk of length k from v cannot")
print(" leave the ball of radius floor(k/2), so RWSE_k is a LOCAL feature.)")
