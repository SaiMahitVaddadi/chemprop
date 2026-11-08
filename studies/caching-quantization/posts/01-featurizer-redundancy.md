# Your featurizer computes 102 distinct atom vectors 57,000 times

*Part 1 of a series on caching and quantizing D-MPNN inference.*

I was looking for somewhere to put a cache in [Chemprop](https://github.com/chemprop/chemprop),
and started by asking a question that seemed too simple to be interesting: across a real dataset,
how many *distinct* atom feature vectors are there?

Chemprop's default atom featurizer produces a 72-dimensional multi-hot vector per atom — atomic
number, degree, formal charge, chiral tag, hydrogen count, hybridization, aromaticity, mass. Bonds
get 14 dimensions. These are computed one atom at a time, in Python, every time a molecule is
featurized.

I pooled 2,727 molecules from Chemprop's own test data and counted:

- **57,549 atoms → 102 distinct atom feature vectors**
- **123,220 directed bonds → 17 distinct bond feature vectors**

So 99.82% of atom featurizer calls, and 99.99% of bond featurizer calls, recompute a vector that
has already been computed. The bond feature space is small enough to enumerate exhaustively at
import time.

That looks like a 576× win sitting in plain sight. It isn't, and the reason why is the actually
useful part of this post.

## The cache that doesn't work

The obvious fix is a lookup table. Key on the atom properties the featurizer reads, memoize the
output vector, and you never compute the same multi-hot twice:

```python
def atom_key(a):
    return (a.GetAtomicNum(), a.GetTotalDegree(), a.GetFormalCharge(),
            int(a.GetChiralTag()), int(a.GetTotalNumHs()),
            int(a.GetHybridization()), a.GetIsAromatic())
```

On a 500-molecule set this table saturates at **40 atom types and 14 bond types** — 12 KiB total.
Every subsequent molecule is pure cache hits. I measured it against the real featurizer, verified
the output is bit-identical, and got… **1.50×**.

The reason is visible as soon as you time the stages rather than the whole thing:

| | µs/mol | speedup |
|---|---|---|
| `SimpleMoleculeMolGraphFeaturizer` | 102.2 | 1.00× |
| `MolFromSmiles` alone | 31.6 | — |
| **RDKit property reads alone** | **55.7** | 1.84× |
| type-table cache, keys computed live | 68.3 | 1.50× |

**55% of featurization time is reading properties out of RDKit in Python.** Computing the
multi-hot vector from those properties — the part a cache eliminates — is the minority of the
cost. And you cannot cache your way past the property reads, because you need them to build the
key. The lookup table is a cache whose key costs more than the value.

This is a general shape worth internalising: *redundancy in the output does not imply the output
is where the time goes.* 99.8% of the work is duplicated and eliminating all of it buys 1.5×,
because the duplicated work was never the bottleneck.

## The cache that does work

If the property reads are the floor, the only way through is to stop doing them — which means
caching at the level of the *molecule*, not the atom, and storing something you can rehydrate
without touching RDKit at all.

Chemprop already has a molecule-level cache. `MolGraphCache` precomputes `MolGraph` objects and
holds them in memory. But a `MolGraph` stores the fully expanded feature matrices: `V` of shape
`(n_atoms, 72)` and `E` of shape `(2·n_bonds, 14)`, as floats. That's **7,530 bytes per molecule**
for what is, information-theoretically, a short list of type indices and an edge list.

So store that instead. Per molecule, keep two small integer arrays — one atom-type ID per atom,
one bond-type ID per bond — plus the edge list. Rehydrate with a NumPy fancy-index against the
shared type tables:

```python
V = ATOM_TABLE[atom_ids]                      # (n_atoms, 72)
E = np.repeat(BOND_TABLE[bond_ids], 2, axis=0) # (2*n_bonds, 14)
```

| | bytes/mol | vs `MolGraph` |
|---|---|---|
| `MolGraph` as Chemprop stores it | 7,530 | 1.00× |
| same, `E` as float32 + int32 indices | 5,676 | 1.33× smaller |
| **type-IDs + edge list** (+12 KiB shared tables) | **217** | **34.7× smaller** |

And rehydration is **4.4 µs/mol — 23.4× faster** than re-featurizing, verified bit-identical to
the featurizer's output. One million molecules fits in about 217 MB, which is memory-mappable, so
the cache can live on disk and survive across runs.

That last property is the one that matters most in practice, because of this line in
`chemprop/data/datasets.py`:

```python
self.cache = False
```

The default is `MolGraphCacheOnTheFly`, which re-featurizes every molecule on **every
`__getitem__`** — so every epoch. And `MolGraphCache`, the eager version, is in-process only: no
deduplication of repeated SMILES, no disk persistence. Run a 100-trial hyperparameter sweep and
you pay full featurization 100 times for identical inputs. At 102 µs/mol that's ten seconds per
pass over 100k molecules, repeated for every epoch of every trial.

## A one-word bug while we're here

In `chemprop/featurizers/molgraph/molecule.py`:

```python
V = np.array([self.atom_featurizer(a) for a in mol.GetAtoms()], dtype=np.single)
E = np.empty((2 * n_bonds, self.bond_fdim))   # <- no dtype
```

NumPy's default is float64. So atom features are float32 and bond features are float64, in the
same object, for data that is almost entirely multi-hot. `collate.py` casts `E` to float32 on the
way into the batch anyway, so the only effects are double the bytes in every cached `MolGraph` and
a wasted conversion per batch. Adding `dtype=np.single` is 1.33× cache memory for one word.

## What this doesn't fix

Chemprop ships `CuikmolmakerMolGraphFeaturizer`, a C++ batched featurizer, and if you can use it
you should — it attacks the same 55% floor from the other direction by getting the property reads
out of Python entirely. The type-ID cache is complementary: it helps on *repeat* exposure to the
same molecules (epochs, sweeps, cross-validation, re-scoring a library), where even a fast
featurizer is doing work it has already done.

None of this touches the model. The next post is about the message-passing loop, where the
redundancy turns out to be exact and cacheable in a much stronger sense — a D-MPNN's hidden state
is a pure function of a non-backtracking walk tree, which makes a Weisfeiler-Lehman hash a sound
cache key rather than an approximation.

## Reproducing

```bash
python scripts/env_cache_study.py     # redundancy counts
python scripts/bench4.py              # the stage breakdown and storage comparison
```

Measured on Apple Silicon, PyTorch 2.11, RDKit 2025.09.3. Numbers here are from pooled Chemprop
test data (2,727 molecules, mean 21.1 heavy atoms); the stage breakdown uses the 500-molecule
ESOL set (mean 13.3 heavy atoms), which is why its type table saturates at 40 rather than 102.
