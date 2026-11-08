import sys, time, pickle, os
sys.path.insert(0, "/private/tmp/claude-502/-Users-svaddadi-Documents-GitHub-chemprop/54c934b6-e3f8-4460-b930-d5d089b68966/scratchpad")
import numpy as np
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")
from cp_loader import SimpleMoleculeMolGraphFeaturizer

smis=[l.split(",")[0].strip() for l in open("tests/data/regression.csv").readlines()[1:]]
smis=[s for s in smis if Chem.MolFromSmiles(s) is not None]
N=len(smis); feat=SimpleMoleculeMolGraphFeaturizer()
af,bf=feat.atom_featurizer,feat.bond_featurizer
def timeit(fn,reps=5):
    b=1e9
    for _ in range(reps):
        t=time.perf_counter(); fn(); b=min(b,time.perf_counter()-t)
    return b
print(f"{N} molecules | breakdown of the input-graph precompute (per molecule)\n")
print(f"{'stage (cumulative)':<46}{'us/mol':>9}{'delta':>9}")
prev=0.0
def row(nm,t):
    global prev
    us=t/N*1e6; print(f"{nm:<46}{us:>9.1f}{us-prev:>9.1f}"); prev=us

row("SMILES -> rdkit Mol (MolFromSmiles)", timeit(lambda:[Chem.MolFromSmiles(s) for s in smis]))
mols=[Chem.MolFromSmiles(s) for s in smis]
t_parse=prev
def read_props():
    for m in mols:
        for a in m.GetAtoms():
            (a.GetAtomicNum(),a.GetTotalDegree(),a.GetFormalCharge(),int(a.GetChiralTag()),
             int(a.GetTotalNumHs()),int(a.GetHybridization()),a.GetIsAromatic())
        for b in m.GetBonds():
            (int(b.GetBondType()),b.GetIsConjugated(),b.IsInRing(),int(b.GetStereo()),
             b.GetBeginAtomIdx(),b.GetEndAtomIdx())
row("  + read atom/bond properties from rdkit", t_parse_props:=timeit(read_props)+t_parse/1e6*N/1e6*0 or timeit(read_props)+ t_parse*N/1e6)
# recompute cleanly
t_props=timeit(read_props)

# fully vectorized type-table featurizer
atom_ids,bond_ids={},{}
atom_rows,bond_rows=[],[]
def build(m):
    aid=np.empty(m.GetNumAtoms(),np.int32)
    for i,a in enumerate(m.GetAtoms()):
        k=(a.GetAtomicNum(),a.GetTotalDegree(),a.GetFormalCharge(),int(a.GetChiralTag()),
           int(a.GetTotalNumHs()),int(a.GetHybridization()),a.GetIsAromatic())
        j=atom_ids.get(k)
        if j is None:
            j=atom_ids[k]=len(atom_rows); atom_rows.append(np.asarray(af(a),np.float32))
        aid[i]=j
    nb=m.GetNumBonds(); bid=np.empty(nb,np.int32); ij=np.empty((nb,2),np.int64)
    for i,b in enumerate(m.GetBonds()):
        k=(int(b.GetBondType()),b.GetIsConjugated(),b.IsInRing(),int(b.GetStereo()))
        j=bond_ids.get(k)
        if j is None:
            j=bond_ids[k]=len(bond_rows); bond_rows.append(np.asarray(bf(b),np.float32))
        bid[i]=j; ij[i,0]=b.GetBeginAtomIdx(); ij[i,1]=b.GetEndAtomIdx()
    return aid,bid,ij
for m in mols: build(m)
AT=np.stack(atom_rows); BT=np.stack(bond_rows)
print(f"\n  type tables: {AT.shape[0]} atom types {AT.shape}, {BT.shape[0]} bond types {BT.shape}"
      f"  ({(AT.nbytes+BT.nbytes)/1024:.1f} KiB total)\n")
ids=[build(m) for m in mols]
def vec_feat():
    out=[]
    for aid,bid,ij in ids:
        V=AT[aid]; E=np.repeat(BT[bid],2,axis=0)
        ei=np.empty((2,2*len(bid)),np.int64)
        ei[0,0::2]=ij[:,0]; ei[1,0::2]=ij[:,1]; ei[0,1::2]=ij[:,1]; ei[1,1::2]=ij[:,0]
        out.append((V,E,ei))
    return out
print(f"{'variant':<46}{'us/mol':>9}{'vs chemprop':>13}")
t_cp=timeit(lambda:[feat(m) for m in mols])
print(f"{'chemprop SimpleMoleculeMolGraphFeaturizer':<46}{t_cp/N*1e6:>9.1f}{1.0:>12.2f}x")
print(f"{'  rdkit property reads alone (hard floor)':<46}{t_props/N*1e6:>9.1f}{t_cp/t_props:>12.2f}x")
t_b=timeit(lambda:[build(m) for m in mols])
print(f"{'  keys+ids only (warm tables)':<46}{t_b/N*1e6:>9.1f}{t_cp/t_b:>12.2f}x")
t_v=timeit(vec_feat)
print(f"{'  table gather only (ids precomputed)':<46}{t_v/N*1e6:>9.1f}{t_cp/t_v:>12.2f}x")
t_full=timeit(lambda:[ (lambda r:(AT[r[0]],np.repeat(BT[r[1]],2,axis=0)))(build(m)) for m in mols])
print(f"{'  full type-table featurizer (keys+gather)':<46}{t_full/N*1e6:>9.1f}{t_cp/t_full:>12.2f}x")

# verify equivalence
mg=feat(mols[11]); aid,bid,ij=build(mols[11])
print(f"\n  output identical to chemprop: V={np.allclose(mg.V,AT[aid])} "
      f"E={np.allclose(mg.E,np.repeat(BT[bid],2,axis=0))}")

# ---- what a persisted (id-only) representation costs on disk ----
mgs=[feat(m) for m in mols]
full=sum(g.V.nbytes+g.E.nbytes+g.edge_index.nbytes+g.rev_edge_index.nbytes for g in mgs)
f32=sum(g.V.nbytes+g.E.astype(np.float32).nbytes+g.edge_index.astype(np.int32).nbytes
        +g.rev_edge_index.astype(np.int32).nbytes for g in mgs)
ids_b=sum(a.nbytes+b.nbytes+j.astype(np.int32).nbytes for a,b,j in ids)
print(f"\n--- cached input-graph size for {N} molecules ---")
print(f"  MolGraph as chemprop stores it today (E is float64) : {full/1024:>9.1f} KiB "
      f"({full/N:.0f} B/mol)")
print(f"  same, E as float32 + int32 indices                  : {f32/1024:>9.1f} KiB "
      f"({f32/N:.0f} B/mol)   {full/f32:.2f}x smaller")
print(f"  type-ids only (+{(AT.nbytes+BT.nbytes)/1024:.0f} KiB shared tables)              "
      f": {ids_b/1024:>9.1f} KiB ({ids_b/N:.0f} B/mol)   {full/ids_b:.2f}x smaller")
