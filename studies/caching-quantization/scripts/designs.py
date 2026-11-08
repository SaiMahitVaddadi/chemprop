import numpy as np, glob, csv, itertools
from rdkit import Chem
from rdkit import RDLogger; RDLogger.DisableLog('rdApp.*')
smis=[]
for f in glob.glob("tests/data/**/*.csv", recursive=True)+glob.glob("tests/data/*.csv"):
    try:
        with open(f) as fh:
            r=csv.reader(fh); hdr=next(r); rows=list(itertools.islice(r,0,2000))
            if not rows: continue
            for ci in range(len(hdr)):
                v=rows[0][ci] if ci<len(rows[0]) else ""
                if len(v)>3 and Chem.MolFromSmiles(v) is not None:
                    smis += [row[ci] for row in rows if ci<len(row)]; break
    except Exception: pass
mols=[m for m in (Chem.MolFromSmiles(s) for s in set(smis)) if m is not None]
E=np.array([2*m.GetNumBonds() for m in mols]); A=np.array([m.GetNumAtoms() for m in mols])
print(f"{len(mols)} unique mols; 2m per mol: mean={E.mean():.1f} p50={np.percentile(E,50):.0f} p95={np.percentile(E,95):.0f} p99={np.percentile(E,99):.0f} max={E.max()}")

rows=[]
for m in mols:
    for b in m.GetBonds():
        for u in (b.GetBeginAtom(), b.GetEndAtom()):
            rows.append(u.GetDegree()-1)
rows=np.array(rows); twom=len(rows); nnz=rows.sum()
print(f"\n2m={twom} nnz(B)={nnz} nnz/row={nnz/twom:.3f}")
h=np.bincount(rows)
for r,c in enumerate(h):
    if c: print(f"  rows with r={r}: {c:8d} ({100*c/twom:5.2f}%)")

n = sum(a for a in A)
# --- traffic in units of (d_h-vector) reads+writes, per message() step ---
designs = {
 "current (scatter_reduce_ + .repeat idx)": 2*twom*2 + twom + 2*n + twom + twom + twom,  # idx64 counts double (8B vs 4B)
 "index_add_ (1-D idx) factored":            twom + 2*n + twom + twom + twom,
 "exact Hashimoto (CSR / bucketed)":         nnz + twom,
 "padded gather K=2 + r=3 correction":       2*twom + h[3]*1 + twom,
 "padded gather K=max":                      rows.max()*twom + twom,
}
base = designs["current (scatter_reduce_ + .repeat idx)"]
fix  = designs["index_add_ (1-D idx) factored"]
print("\n--- per-message-step vector traffic (lower=better) ---")
for k,v in designs.items():
    print(f"  {k:42s} {v:9d}   vs-current {base/v:5.2f}x   vs-index_add_fix {fix/v:5.2f}x")
print(f"\ncrossover: explicit-B beats factored iff mean(d^2) < 4*mean(d)+2")
deg=np.concatenate([[a.GetDegree() for a in m.GetAtoms()] for m in mols])
print(f"  measured mean(d)={deg.mean():.3f} mean(d^2)={(deg**2).mean():.3f}  threshold={4*deg.mean()+2:.3f}  -> {'WINS' if (deg**2).mean()<4*deg.mean()+2 else 'LOSES'}")
print(f"  regular-graph crossover degree d where d^2=4d+2 -> d={(4+np.sqrt(16+8))/2:.2f}")

# --- B operator storage ---
for dh in (300,2048):
    print(f"\nd_h={dh}: CSR(B) for a 500-mol batch = indptr {(500*E.mean()+1)*4/1e6:.2f}MB + indices {500*E.mean()*nnz/twom*4/1e6:.2f}MB"
          f"  vs H = {500*E.mean()*dh*4/1e6:.1f}MB  ({100*(2*500*E.mean()*nnz/twom*4)/(500*E.mean()*dh*4):.2f}% of H)")

# --- dense block-diagonal bmm design ---
print("\n--- dense per-molecule bmm design (B_i is 2m_i x 2m_i), b=500 molecules ---")
for dh in (300,2048):
    gemm = 2*500*E.mean()*dh*dh
    print(f" d_h={dh}: W_h GEMM on compact rows = {gemm/1e9:.1f} GFLOP")
    for policy,Emax in [("single pad to batch max (p99)", np.percentile(E,99)),
                        ("single pad to p95", np.percentile(E,95)),
                        ("size-bucketed (pad to 1.15x mean)", 1.15*E.mean())]:
        bmm = 500*2*Emax*Emax*dh
        pad_waste = Emax/E.mean()
        gemm_pad = 2*500*Emax*dh*dh
        print(f"   {policy:34s} E_max={Emax:6.1f}  B-bmm={bmm/1e9:6.2f} GFLOP ({100*bmm/gemm:5.1f}% of GEMM)"
              f"  | if H kept padded, W_h GEMM inflates {pad_waste:.2f}x -> {gemm_pad/1e9:.1f} GFLOP")
