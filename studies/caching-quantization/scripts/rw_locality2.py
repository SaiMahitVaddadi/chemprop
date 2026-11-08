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

def rwse_full(m,K):
    A=adj(m); d=A.sum(1); d[d==0]=1; P=A/d[:,None]
    Pk=np.eye(A.shape[0]); out=np.zeros((A.shape[0],K))
    for k in range(K): Pk=Pk@P; out[:,k]=np.diag(Pk)
    return out

def rwse_local(m,v,r,K):
    """induced ball of radius r, but TRUE (full-molecule) degrees used for normalisation"""
    A=adj(m); D=Chem.GetDistanceMatrix(m)
    idx=[i for i in range(m.GetNumAtoms()) if D[v,i]<=r]
    As=A[np.ix_(idx,idx)]; dfull=A.sum(1)[idx]; dfull[dfull==0]=1
    P=As/dfull[:,None]
    pos=idx.index(v); Pk=np.eye(len(idx)); out=np.zeros(K)
    for k in range(K): Pk=Pk@P; out[k]=Pk[pos,pos]
    return out, len(idx)

smis=[l.split(",")[0].strip() for l in open(f"{REPO}/tests/data/regression.csv").readlines()[1:]]
mols=[m for m in (Chem.MolFromSmiles(s) for s in smis) if m is not None]
mols=[m for m in mols if 10<=m.GetNumAtoms()<=30][:60]
K=20
print("="*86)
print("RWSE_k exactness from a radius-r ball with TRUE degrees (ball = atoms at dist<=r of v)")
print("="*86)
print(f"{'r':>3}{'avg |ball|':>12}{'max err, k<=2r':>16}{'max err, k=2r+1':>17}{'max err, k=2r+2':>17}")
for r in range(1,9):
    e_in=e_p1=e_p2=0.0; sz=[]
    for m in mols:
        full=rwse_full(m,K)
        for v in range(m.GetNumAtoms()):
            got,n=rwse_local(m,v,r,K); sz.append(n)
            for k in range(1,K+1):
                err=abs(got[k-1]-full[v,k-1])
                if k<=2*r: e_in=max(e_in,err)
                elif k==2*r+1: e_p1=max(e_p1,err)
                elif k==2*r+2: e_p2=max(e_p2,err)
    print(f"{r:>3}{np.mean(sz):>12.1f}{e_in:>16.2e}{e_p1:>17.2e}{e_p2:>17.2e}")
print("\nCONCLUSION: RWSE_k is EXACTLY a function of the radius-ceil(k/2) neighbourhood")
print("(ball atoms + their true degrees). It is therefore locally cacheable with the")
print("same kind of WL-style key as the message states, at cache radius ceil(k/2).")
