"""Verify: Chemprop's bond message operator IS the Hashimoto non-backtracking matrix."""
import sys
SP="/private/tmp/claude-502/-Users-svaddadi-Documents-GitHub-chemprop/54c934b6-e3f8-4460-b930-d5d089b68966/scratchpad"
sys.path.insert(0,SP)
import numpy as np, torch
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")
from cp_loader import SimpleMoleculeMolGraphFeaturizer
torch.set_grad_enabled(False)
feat=SimpleMoleculeMolGraphFeaturizer()
REPO="/Users/svaddadi/Documents/GitHub/chemprop"
smis=[l.split(",")[0].strip() for l in open(f"{REPO}/tests/data/regression.csv").readlines()[1:]][:40]
maxerr=0.0; spec=[]
for s in smis:
    m=Chem.MolFromSmiles(s)
    if m is None or m.GetNumBonds()==0: continue
    g=feat(m); ei=g.edge_index; rev=g.rev_edge_index; nE=ei.shape[1]; nV=g.V.shape[0]
    # Hashimoto / oriented-line-digraph adjacency: B[e,f]=1 iff head(f)==tail(e) and f != rev(e)
    B=np.zeros((nE,nE))
    for e in range(nE):
        for f in range(nE):
            if ei[1,f]==ei[0,e] and f!=rev[e]: B[e,f]=1
    H=torch.randn(nE,7).double()
    # chemprop's message(): scatter-sum into tail, gather at src, minus reverse edge
    idx=torch.from_numpy(ei[1]).long().unsqueeze(1).expand(-1,7)
    Mall=torch.zeros(nV,7,dtype=torch.float64).scatter_reduce_(0,idx,H,reduce="sum",
          include_self=False)[torch.from_numpy(ei[0]).long()]
    M_cp=Mall-H[torch.from_numpy(rev).long()]
    M_B=torch.from_numpy(B)@H
    maxerr=max(maxerr,float((M_cp-M_B).abs().max()))
    ev=np.linalg.eigvals(B); spec.append((m.GetNumAtoms(),nE,float(np.abs(ev).max())))
print(f"max |chemprop message(H) - B @ H| over {len(spec)} molecules, random H: {maxerr:.3e}")
print("=> chemprop's BondMessagePassing message operator is EXACTLY the Hashimoto matrix B.\n")
print("Perron root of B (= growth rate of non-backtracking walks = 1/Ihara-zeta pole):")
a=np.array(spec)
print(f"  n_atoms {a[:,0].mean():.1f} avg | rho(B) mean {a[:,2].mean():.3f} "
      f"min {a[:,2].min():.3f} max {a[:,2].max():.3f}")
print(f"  #(NB walks of length t from a bond) ~ rho(B)^t -> at t=6: {a[:,2].mean()**6:.0f}")
print(f"  compare #(backtracking-allowed walks) with mean degree d~2.2: {2.2**6:.0f}")
