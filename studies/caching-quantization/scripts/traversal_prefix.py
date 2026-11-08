"""Would a canonical ordered traversal give KV-style reusable PREFIXES across an analog series?"""
import sys, itertools
SP="/private/tmp/claude-502/-Users-svaddadi-Documents-GitHub-chemprop/54c934b6-e3f8-4460-b930-d5d089b68966/scratchpad"
sys.path.insert(0,SP)
import numpy as np
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")

TEMPLATE="O=C(Nc1ccc({R1})cc1)c1ccc({R2})cc1"
SUBS=["F","Cl","Br","C","CC","CCC","OC","OCC","C(F)(F)F","C#N","O","N","N(C)C",
      "S(=O)(=O)C","c1ccccc1","c1ccncc1","N1CCOCC1","N1CCCCC1","C(=O)O","C(=O)N"]

def lcp(a,b):
    n=0
    for x,y in zip(a,b):
        if x!=y: break
        n+=1
    return n

# ---------- A. canonical SMILES prefix sharing (the literal KV-cache analogy) ----------
print("="*80)
print("EXP 2A: RDKit canonical SMILES -- longest common prefix across an analog series")
print("        (this is exactly what an LLM/SMILES-transformer KV cache can reuse)")
print("="*80)
lib=[]
for r1 in SUBS:
    for r2 in SUBS:
        m=Chem.MolFromSmiles(TEMPLATE.replace("{R1}",r1).replace("{R2}",r2))
        if m: lib.append(Chem.MolToSmiles(m))
print(f"library: {len(lib)} analogs of one scaffold, avg canonical SMILES length "
      f"{np.mean([len(s) for s in lib]):.1f} chars")

# pairwise LCP among random pairs
rng=np.random.default_rng(0)
pairs=[(int(i),int(j)) for i,j in rng.integers(0,len(lib),(4000,2)) if i!=j]
fr=[lcp(lib[i],lib[j])/max(len(lib[i]),len(lib[j])) for i,j in pairs]
print(f"  mean pairwise LCP as fraction of string      : {np.mean(fr):6.1%}")
print(f"  median                                       : {np.median(fr):6.1%}")
print(f"  fraction of pairs sharing >50% of the prefix : {np.mean([f>0.5 for f in fr]):6.1%}")

# best case: prefix shared with the single most-similar other member (an ideal cache)
best=[]
for i in range(0,len(lib),7):
    best.append(max(lcp(lib[i],lib[j])/len(lib[i]) for j in range(len(lib)) if j!=i))
print(f"  ORACLE: LCP with the best-matching other analog: {np.mean(best):6.1%}")

# ---------- B. does the scaffold keep stable canonical ranks? ----------
print()
print("="*80)
print("EXP 2B: is the scaffold's canonical atom order STABLE under an R-group swap?")
print("        (a cache key / traversal prefix must be a function of the scaffold alone)")
print("="*80)
def build(r1,r2):
    return Chem.MolFromSmiles(TEMPLATE.replace("{R1}",r1).replace("{R2}",r2))
def n_atoms_of(r2):
    return build("C",r2).GetNumAtoms()-build("C","F").GetNumAtoms()+1

base=build("C","C")
nF=build("C","F").GetNumAtoms()
n_prefix=nF-1-2           # atoms written before the R2 group; last "cc1" = 2 atoms
print(f"  scaffold (R1=C) prefix atoms in SMILES order: 0..{n_prefix-1}, then R2, then 2 ring atoms")

print(f"\n{'R2 swap':>18}{'atoms':>7}{'scaffold ranks changed':>24}{'canon traversal LCP':>21}")
bca=list(Chem.CanonicalRankAtoms(base))
Chem.MolToSmiles(base); border=[int(x) for x in base.GetProp('_smilesAtomOutputOrder')[1:-1].split(',') if x!='']
# map base scaffold atom -> index: prefix atoms identical; suffix 2 atoms at n_prefix+1 .. (base R2="C" is 1 atom)
def scaffold_map(r2):
    nR=n_atoms_of(r2)
    mp={}                                   # base idx -> analog idx
    for i in range(n_prefix): mp[i]=i
    for k in range(2): mp[n_prefix+1+k]=n_prefix+nR+k
    return mp,nR
for r2 in ["F","Cl","C(F)(F)F","C#N","OC","N(C)C","c1ccncc1","N1CCOCC1","S(=O)(=O)C","C(=O)O"]:
    m=build("C",r2); mp,nR=scaffold_map(r2)
    # sanity: elements must match
    ok=all(base.GetAtomWithIdx(b).GetSymbol()==m.GetAtomWithIdx(a).GetSymbol() for b,a in mp.items())
    ca=list(Chem.CanonicalRankAtoms(m))
    # relative order of scaffold atoms under canonical ranking
    sb=sorted(mp.keys(), key=lambda b: bca[b])
    sa=sorted(mp.keys(), key=lambda b: ca[mp[b]])
    changed=sum(1 for x,y in zip(sb,sa) if x!=y)
    # canonical SMILES traversal order restricted to scaffold atoms -> LCP with base
    Chem.MolToSmiles(m); ao=[int(x) for x in m.GetProp('_smilesAtomOutputOrder')[1:-1].split(',') if x!='']
    inv={v:k for k,v in mp.items()}
    tb=[x for x in border if x in mp]
    ta=[inv[x] for x in ao if x in inv]
    print(f"{('C -> '+r2):>18}{m.GetNumAtoms():>7}"
          f"{changed}/{len(mp)} ({changed/len(mp):.0%}){'':>8}"
          f"{lcp(tb,ta)}/{len(tb)} ({lcp(tb,ta)/len(tb):.0%}){'' if ok else '  [ELEM MISMATCH]'}")
