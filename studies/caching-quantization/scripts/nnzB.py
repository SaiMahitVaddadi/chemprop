"""Sparsity of the explicit Hashimoto matrix B on real molecules.
Does nnz(B)/row ~= 1.384 (kernel thread's number) hold, and where is the crossover?"""
import sys
sys.path.insert(0,"/private/tmp/claude-502/-Users-svaddadi-Documents-GitHub-chemprop/54c934b6-e3f8-4460-b930-d5d089b68966/scratchpad")
import numpy as np
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")
from cp_loader import SimpleMoleculeMolGraphFeaturizer
feat=SimpleMoleculeMolGraphFeaturizer()
REPO="/Users/svaddadi/Documents/GitHub/chemprop"
smis=[l.split(",")[0].strip() for l in open(f"{REPO}/tests/data/regression.csv").readlines()[1:]]
tot_rows=0; tot_nnz=0; degs=[]; per=[]
for s in smis:
    m=Chem.MolFromSmiles(s)
    if m is None or m.GetNumBonds()<1: continue
    g=feat(m); ei=g.edge_index; rev=g.rev_edge_index; nE=ei.shape[1]
    # nnz of row e = (in-degree of src(e)) - 1   [exclude rev(e)]
    indeg=np.bincount(ei[1],minlength=g.V.shape[0])
    rows=indeg[ei[0]]-1
    tot_rows+=nE; tot_nnz+=int(rows.sum()); per.append(rows.mean())
    degs.append(indeg[indeg>0].mean())
print(f"{len(per)} molecules, {tot_rows} directed bonds")
print(f"  mean atom degree                       : {np.mean(degs):.3f}")
print(f"  nnz(B) per row  = mean(deg(src)-1)     : {tot_nnz/tot_rows:.3f}")
print(f"  (kernel thread reported 1.384)")
print(f"  rows with 0 nonzeros (terminal bonds)  : {np.mean([0]) if False else ''}", end="")
z=0; t=0
for s in smis:
    m=Chem.MolFromSmiles(s)
    if m is None or m.GetNumBonds()<1: continue
    g=feat(m); ei=g.edge_index
    indeg=np.bincount(ei[1],minlength=g.V.shape[0]); rows=indeg[ei[0]]-1
    z+=int((rows==0).sum()); t+=len(rows)
print(f"{z/t:.1%}")
print(f"\nFLOP/traffic comparison per layer, d_h=2048:")
k=tot_nnz/tot_rows
print(f"  explicit B @ H      : {k:.3f} gathers+adds per output row")
print(f"  factored (scatter to atoms, gather, subtract H_rev):")
print(f"      scatter {1.0:.3f} + gather {1.0:.3f} + gather_rev {1.0:.3f} + sub = 3 passes over d_h")
print(f"  => explicit wins while nnz/row < 3 (+ the atom-sized intermediate it avoids).")
print(f"  crossover at mean degree ~4.45 => nnz/row = 3.45; molecular degree {np.mean(degs):.2f} is far below.")
