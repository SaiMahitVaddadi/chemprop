import numpy as np
try:
    from rdkit import Chem
except Exception as e:
    print("no rdkit", e); raise SystemExit

# grab SMILES from chemprop test data
import glob, csv, itertools, os
cands = glob.glob("tests/data/*.csv") + glob.glob("tests/data/**/*.csv", recursive=True)
smis=[]
for f in cands:
    try:
        with open(f) as fh:
            r=csv.reader(fh); hdr=next(r)
            # find a column that parses as SMILES
            rows=list(itertools.islice(r,0,2000))
            if not rows: continue
            for ci in range(len(hdr)):
                v=rows[0][ci] if ci<len(rows[0]) else ""
                if Chem.MolFromSmiles(v) is not None and len(v)>3:
                    for row in rows:
                        if ci<len(row):
                            smis.append(row[ci])
                    break
    except Exception: pass
smis=[s for s in smis if s]
print("files scanned:", len(cands), "smiles collected:", len(smis))
degs=[]; nA=0; nB=0; nnzB=0; maxrow=0; rowhist={}
nmol=0
for s in smis:
    m=Chem.MolFromSmiles(s)
    if m is None: continue
    nmol+=1
    nA+=m.GetNumAtoms(); nB+=2*m.GetNumBonds()
    d=np.array([a.GetDegree() for a in m.GetAtoms()])
    degs.append(d)
    # for each directed edge e=(u->v), nnz of row e = deg(u)-1
    for b in m.GetBonds():
        for u in (b.GetBeginAtom(), b.GetEndAtom()):
            r=u.GetDegree()-1
            nnzB+=r; maxrow=max(maxrow,r)
            rowhist[r]=rowhist.get(r,0)+1
d=np.concatenate(degs)
print(f"molecules={nmol} atoms={nA} dir_bonds={nB} atoms/mol={nA/nmol:.2f} dirbonds/mol={nB/nmol:.2f}")
print("avg atom degree:", d.mean(), "max:", d.max())
print("degree hist:", {int(k):int(v) for k,v in zip(*np.unique(d,return_counts=True))})
print(f"nnz(B)={nnzB}  nnz/row={nnzB/nB:.3f}  max row nnz (K_pad)={maxrow}")
print("row-nnz hist:", dict(sorted(rowhist.items())))
# traffic model (units of d_h-vectors), per message() call
m2=nB; n=nA
fact = m2 + n + m2 + m2 + m2   # read H(scatter), write N, read N(gather), read H[rev], write M
csr  = nnzB + m2                # read H per nnz, write M
pad  = maxrow*m2 + m2
print(f"\nvector-traffic per message step: factored={fact}  exact-CSR={csr}  padded-gather(K={maxrow})={pad}")
print(f"CSR vs factored speedup on traffic: {fact/csr:.2f}x ; padded vs factored: {fact/pad:.2f}x")
