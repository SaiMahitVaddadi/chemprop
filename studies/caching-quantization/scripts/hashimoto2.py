"""Spectrum of the Hashimoto operator B on real drug-like molecules:
what does it predict about the marginal value of message-passing depth?"""
import sys
SP="/private/tmp/claude-502/-Users-svaddadi-Documents-GitHub-chemprop/54c934b6-e3f8-4460-b930-d5d089b68966/scratchpad"
sys.path.insert(0,SP)
import numpy as np
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")
from cp_loader import SimpleMoleculeMolGraphFeaturizer
feat=SimpleMoleculeMolGraphFeaturizer()
REPO="/Users/svaddadi/Documents/GitHub/chemprop"
smis=[l.split(",")[0].strip() for l in open(f"{REPO}/tests/data/regression.csv").readlines()[1:]]
rows=[]
for s in smis:
    m=Chem.MolFromSmiles(s)
    if m is None or m.GetNumBonds()<1: continue
    g=feat(m); ei=g.edge_index; rev=g.rev_edge_index; nE=ei.shape[1]
    B=np.zeros((nE,nE))
    for e in range(nE):
        tail=ei[0,e]
        for f in range(nE):
            if ei[1,f]==tail and f!=rev[e]: B[e,f]=1
    ev=np.abs(np.linalg.eigvals(B)); rho=ev.max() if nE else 0.0
    nrings=m.GetRingInfo().NumRings()
    # nonzero NB-walk counts: total walks of length t from any directed bond
    cnt=[float((np.linalg.matrix_power(B,t)).sum()) for t in range(1,9)]
    rows.append((m.GetNumAtoms(),nE,nrings,rho,cnt))
a=np.array([(r[0],r[1],r[2],r[3]) for r in rows])
print(f"{len(rows)} molecules, mean {a[:,0].mean():.1f} heavy atoms, mean {a[:,2].mean():.2f} SSSR rings\n")
print("Perron root rho(B)  ( = 1/ smallest pole of the Ihara zeta function )")
print(f"  acyclic molecules (B nilpotent, rho=0): {np.mean(a[:,3]<1e-9):.1%}")
print(f"  rho < 1  (NB-walk counts DECAY with depth): {np.mean(a[:,3]<1-1e-9):.1%}")
print(f"  rho = 1  (single cycle, counts constant) : {np.mean(np.abs(a[:,3]-1)<1e-9):.1%}")
print(f"  rho > 1  (NB-walk counts GROW)           : {np.mean(a[:,3]>1+1e-9):.1%}")
q=np.quantile(a[a[:,3]>0,3],[0.1,0.25,0.5,0.75,0.9])
print(f"  rho | rho>0  deciles: p10 {q[0]:.3f} p25 {q[1]:.3f} med {q[2]:.3f} p75 {q[3]:.3f} p90 {q[4]:.3f}")
print(f"\nFraction of molecules where NO non-backtracking walk of length t exists at all")
print("(i.e. message-passing layer t+1 injects ZERO new structural information):")
for t in range(1,9):
    dead=np.mean([r[4][t-1]<0.5 for r in rows])
    print(f"   t={t}: {dead:>6.1%}")
print("\nTotal NB-walk count growth, median over molecules, normalised to t=1:")
med=[np.median([r[4][t-1]/max(r[4][0],1) for r in rows]) for t in range(1,9)]
print("   " + "  ".join(f"t={t}:{med[t-1]:.2f}" for t in range(1,9)))
