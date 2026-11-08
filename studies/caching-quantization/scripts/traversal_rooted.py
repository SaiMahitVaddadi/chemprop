"""Does a ROOTED canonical traversal restore long reusable prefixes?"""
import sys
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
lib=[]
for r1 in SUBS:
    for r2 in SUBS:
        m=Chem.MolFromSmiles(TEMPLATE.replace("{R1}",r1).replace("{R2}",r2))
        if m: lib.append(m)
# the carbonyl O is atom 0 in every build (SMILES starts "O=C(")
rooted=[Chem.MolToSmiles(m, rootedAtAtom=0, canonical=True) for m in lib]
canon  =[Chem.MolToSmiles(m) for m in lib]
rng=np.random.default_rng(0)
pairs=[(int(i),int(j)) for i,j in rng.integers(0,len(lib),(4000,2)) if i!=j]
for nm,L in (("global canonical SMILES",canon),("scaffold-ROOTED canonical SMILES",rooted)):
    fr=[lcp(L[i],L[j])/max(len(L[i]),len(L[j])) for i,j in pairs]
    orc=[max(lcp(L[i],L[j])/len(L[i]) for j in range(len(L)) if j!=i) for i in range(0,len(L),7)]
    print(f"{nm:<34} mean LCP {np.mean(fr):6.1%}  median {np.median(fr):6.1%}  "
          f">50% {np.mean([f>0.5 for f in fr]):5.1%}  oracle {np.mean(orc):6.1%}")
print()
# how many DISTINCT prefixes of each length -> cacheable prefix tree
for nm,L in (("global",canon),("rooted",rooted)):
    print(f"{nm:>8} distinct prefixes of length k (of {len(L)} analogs):", end="")
    for k in (5,10,15,20,25):
        print(f"  k={k}:{len({s[:k] for s in L})}", end="")
    print()
print("\nA KV cache's win = (tokens whose prefix is already cached)/(total tokens).")
print("With a prefix tree over the library, reusable fraction =")
for nm,L in (("global",canon),("rooted",rooted)):
    tot=sum(len(s) for s in L); nodes=len({s[:k] for s in L for k in range(1,len(s)+1)})
    print(f"  {nm:>8}: {1-nodes/tot:6.1%} of token-positions shared with an earlier analog")
