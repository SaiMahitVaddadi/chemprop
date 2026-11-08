import numpy as np
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")
REPO="/Users/svaddadi/Documents/GitHub/chemprop"
rows=[l.strip().split(",") for l in open(f"{REPO}/tests/data/regression.csv").readlines()[1:]]
pairs=[(s,float(v)) for s,v in rows if Chem.MolFromSmiles(s) is not None]
y=np.array([v for _,v in pairs]); mols=[Chem.MolFromSmiles(s) for s,_ in pairs]
print(f"ESOL target std = {y.std():.4f}  (RMSE of a mean predictor)")
print(f"mean heavy atoms {np.mean([m.GetNumAtoms() for m in mols]):.1f}")
print("\nreceptive-field saturation: fraction of the molecule inside a radius-r ball")
print(f"{'r':>3}{'mean coverage':>15}{'median':>9}{'frac atoms with 100% coverage':>32}")
for r in range(1,9):
    cov=[];full=[]
    for m in mols:
        D=Chem.GetDistanceMatrix(m); n=m.GetNumAtoms()
        c=(D<=r).sum(1)/n; cov.append(c.mean()); full.append((c>=0.999).mean())
    print(f"{r:>3}{np.mean(cov):>15.1%}{np.median(cov):>9.1%}{np.mean(full):>32.1%}")
