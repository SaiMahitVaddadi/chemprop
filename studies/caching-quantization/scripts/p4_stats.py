"""P4: nnz(B)/row, row-nnz histogram, mean(d), mean(d^2), crossover, per real dataset."""
import sys, glob, csv, gzip, os
SP="/private/tmp/claude-502/-Users-svaddadi-Documents-GitHub-chemprop/54c934b6-e3f8-4460-b930-d5d089b68966/scratchpad"
sys.path.insert(0,SP)
import numpy as np
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")

def smiles_col(path, limit=None):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as fh:
        r = csv.reader(fh); hdr = next(r)
        rows = [x for x in r if x]
        if limit: rows = rows[:limit]
    # pick column by name first
    ci = None
    for pref in ("smiles","smile","mol","SMILES"):
        for i,h in enumerate(hdr):
            if h.strip().lower()==pref.lower(): ci=i; break
        if ci is not None: break
    if ci is None:
        for i in range(len(hdr)):
            ok=0
            for row in rows[:20]:
                if i<len(row) and len(row[i])>3 and Chem.MolFromSmiles(row[i]) is not None: ok+=1
            if ok>=10: ci=i; break
    if ci is None: return []
    return [row[ci] for row in rows if ci<len(row) and row[ci]]

def stats(smis):
    degs=[]; nA=0; nB=0; nnzB=0; rowhist={}; nmol=0; m2list=[]
    for s in smis:
        m=Chem.MolFromSmiles(s)
        if m is None or m.GetNumBonds()==0: continue
        nmol+=1
        nA+=m.GetNumAtoms(); e2=2*m.GetNumBonds(); nB+=e2; m2list.append(e2)
        d=np.array([a.GetDegree() for a in m.GetAtoms()])
        degs.append(d)
        for b in m.GetBonds():
            for u in (b.GetBeginAtom(), b.GetEndAtom()):
                r=u.GetDegree()-1
                nnzB+=r; rowhist[r]=rowhist.get(r,0)+1
    if not nmol: return None
    d=np.concatenate(degs)
    md=d.mean(); md2=(d.astype(float)**2).mean()
    return dict(nmol=nmol,nA=nA,nB=nB,nnzB=nnzB,nnz_row=nnzB/nB,
                rowhist={k:v/nB for k,v in sorted(rowhist.items())},
                mean_deg=md, mean_deg2=md2, thresh=4*md+2, maxdeg=int(d.max()),
                m2=np.array(m2list))

files = sorted(glob.glob(f"{SP}/data/*.csv")+glob.glob(f"{SP}/data/*.csv.gz"))
REPO="/Users/svaddadi/Documents/GitHub/chemprop"
pred={0:.107,1:.435,2:.407,3:.051}
print(f"{'dataset':<22}{'nmol':>7}{'atoms':>8}{'2m':>9}{'nnz/row':>9}{'mean d':>8}{'mean d2':>9}{'4d+2':>7}{'win?':>6}  row-nnz hist (r=0..4)")
allsmis=[]
rows=[]
for f in files:
    lim = 3000
    smis = smiles_col(f, lim)
    st = stats(smis)
    if st is None: continue
    allsmis += smis
    h=st['rowhist']
    hs="  ".join(f"{r}:{h.get(r,0)*100:5.1f}%" for r in range(5))
    print(f"{os.path.basename(f):<22}{st['nmol']:>7}{st['nA']:>8}{st['nB']:>9}{st['nnz_row']:>9.3f}"
          f"{st['mean_deg']:>8.3f}{st['mean_deg2']:>9.3f}{st['thresh']:>7.2f}"
          f"{'YES' if st['mean_deg2']<st['thresh'] else 'NO':>6}  {hs}")
    rows.append((os.path.basename(f),st))
# pooled
st=stats(allsmis)
h=st['rowhist']
hs="  ".join(f"{r}:{h.get(r,0)*100:5.1f}%" for r in range(5))
print(f"{'POOLED':<22}{st['nmol']:>7}{st['nA']:>8}{st['nB']:>9}{st['nnz_row']:>9.3f}"
      f"{st['mean_deg']:>8.3f}{st['mean_deg2']:>9.3f}{st['thresh']:>7.2f}"
      f"{'YES' if st['mean_deg2']<st['thresh'] else 'NO':>6}  {hs}")
print(f"\nPOOLED vs predicted histogram (abs deviation, predicted {pred}):")
for r in range(4):
    print(f"  r={r}: measured {h.get(r,0)*100:6.2f}%  predicted {pred[r]*100:5.1f}%  dev {abs(h.get(r,0)-pred[r])*100:+5.2f} pp")
print(f"  r>=4: measured {sum(v for k,v in h.items() if k>=4)*100:.3f}%")
print(f"\nPOOLED nnz(B)/row = {st['nnz_row']:.4f} (predicted 1.38, range 1.38-1.41); max degree {st['maxdeg']}")
# traffic ratios
m2=st['nB']; n=st['nA']; nnz=st['nnzB']
cur = 8*(m2/2)*0 + 0  # compute explicitly below
cur = 4*m2 + 2*n + 2*(2*m2)/2*0
# per brief: current = 8m + 2n + 2*(2m) with m undirected => 4*(2m)+2n+2*(2m)
cur = 4*m2 + 2*n + 2*m2
fac = m2 + 2*n + m2 + m2 + m2
has = nnz + m2
print(f"\nvector traffic (units of d_h-vectors): current={cur}  factored/index_add={fac}  hashimoto={has}")
print(f"  index_add vs current {cur/fac:.3f}x ; hashimoto vs current {cur/has:.3f}x ; hashimoto vs index_add {fac/has:.3f}x")
# 2m distribution for P8
q=np.quantile(st['m2'],[.5,.9,.95,.99])
print(f"\nP8: 2m distribution over {len(st['m2'])} mols: mean {st['m2'].mean():.1f} p50 {q[0]:.0f} p90 {q[1]:.0f} p95 {q[2]:.0f} p99 {q[3]:.0f} max {st['m2'].max()}")
np.save(f"{SP}/m2_pooled.npy", st['m2'])
