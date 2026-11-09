# Half of FreeSolv shares one "scaffold", and it is the empty string

*Part 10 of a series on Chemprop internals.*

Scaffold splitting is the field's standard answer to "is my model learning chemistry or memorising
neighbours." You group molecules by Bemis-Murcko scaffold, put whole groups on one side of the
split, and report the result as a harder, more honest number than a random split.

I went to measure how much harder. The answer is: much less than advertised, for a reason sitting
in plain sight.

## Every acyclic molecule has the same scaffold, and that scaffold is `""`

A Bemis-Murcko scaffold is what remains after you strip side chains, keeping ring systems and the
linkers between them. For a molecule with no rings, nothing remains. RDKit returns the empty
string.

So **every acyclic molecule in a dataset lands in a single scaffold group.** Measured on three
standard sets:

| dataset | molecules | unique scaffolds | mol/scaffold | empty-scaffold group | largest group |
|---|---|---|---|---|---|
| ESOL | 1,128 | 269 | 4.19 | **317 (28.1%)** | 317 (28.1%) |
| FreeSolv | 642 | 63 | 10.19 | **320 (49.8%)** | 320 (49.8%) |
| Lipophilicity | 4,200 | 2,408 | 1.74 | 5 (0.1%) | 76 (1.8%) |

**Half of FreeSolv is one scaffold group.** On ESOL it's 28%, and in both cases the empty-string
group is the *largest* group in the dataset. Scaffold splitters typically assign the largest groups
to the training set first, to hit the target ratio — which means on these two datasets every
acyclic molecule is structurally guaranteed to be in train and never in test. Your test set is
all-cyclic by construction, and your reported generalisation is specifically generalisation to
ring-containing chemistry.

Lipophilicity shows the opposite failure, and it's the more general one.

## At 1.74 molecules per scaffold, a scaffold split *is* a random split

Lipophilicity has 2,408 scaffolds for 4,200 molecules. Most groups are singletons. Grouping by a
key that is nearly unique per item and then partitioning the groups is, to a first approximation,
partitioning the items at random.

That is exactly what the measurement literature finds. Fooladi et al. (JCIM 2025) ran 14 models
across 8 datasets and 10 splitting strategies and report median train-test Tanimoto distance of
**0.577 for random, 0.610 for scaffold, 0.614 for generic scaffold** — against **0.705–0.714** for
UMAP or Lo-Hi clustering splits. A scaffold split moves the distance by 0.03; a clustering split
moves it by 0.13. Their stated mechanism is the same number I measured: "on average, there are only
1.93 molecules per scaffold... this low scaffold-to-molecule ratio means that most scaffolds contain
very few molecules, increasing the probability that structurally similar molecules land on opposite
sides of the split."

The summary statistic is the one to remember: in-distribution and out-of-distribution performance
correlate at **r ≈ 0.9 under scaffold splitting** and **r ≈ 0.4 under clustering**. A scaffold split
tells you almost nothing a random split didn't.

And leakage survives it directly. Steshin (NeurIPS 2023 D&B) measured the fraction of test
molecules with a training neighbour at ECFP4 Tanimoto > 0.4: **MoleculeNet HIV, under HIV's own
recommended scaffold split, is 56%.** TDC's CYP2D6_Veith scaffold split is 78%. QM9 is 0.99, QM8
0.98, MUV 0.96. Zero scaffold sharing and near-total neighbour leakage are perfectly compatible,
because scaffold identity is not similarity.

## The convention isn't the benchmark's recommendation

Here's the part that surprised me most. **MoleculeNet prescribed *random* splitting, not scaffold
splitting, for ESOL, FreeSolv, Lipophilicity, SIDER, ClinTox and Tox21.** Scaffold splitting became
the default through convention and through downstream libraries' defaults, not because the
benchmark asked for it. So a large fraction of published comparisons are using a split the
benchmark didn't specify, which is weaker than it looks, on datasets where a quarter to a half of
the molecules can't appear in the test set at all.

## And "scaffold split" is not one thing

This is the part I could not finish verifying, and I'd rather flag it than publish it as settled.

A companion analysis compared the test sets produced by three implementations — `astartes` (which
Chemprop v2 uses), Chemprop v1's own, and DeepChem's — and reports **Jaccard overlap between 0.012
and 0.29**. Two implementations of the same documented concept, agreeing on as little as 1% of
which molecules are in the test set. It also reports Chemprop v2 with a different seed differing
from itself about as much as it differs from DeepChem, and 1.5–6% near-duplicates remaining under
all three despite zero scaffold sharing, with no consistent ordering between them.

**Those numbers come from re-implementations read out of upstream source, because `chemprop` and
`astartes` don't import in the environment where they were measured.** They are plausible — the
implementations differ in whether they use generic scaffolds, how they break ties, whether they
sort groups by size, and how they handle the empty-scaffold group — but they are not verified, and
I'm not going to assert them until they've been re-derived against real installs of all three
libraries plus DeepChem. The empty-scaffold and molecules-per-scaffold numbers in the tables above
*are* mine, measured directly with RDKit.

If it holds up, the implication is uncomfortable: "we used a scaffold split" does not identify an
experiment, and two papers reporting scaffold-split numbers on the same dataset may not be
comparable.

## What to do instead

- **Report molecules-per-scaffold and the empty-scaffold fraction** alongside any scaffold-split
  result. They take one line of RDKit and they tell a reader immediately whether your split is a
  real shift (Lipophilicity: no) or has a structural hole in it (FreeSolv: yes, half the dataset).
- **Use a clustering split when you want a shift.** UMAP or Butina on ECFP4 moves the distance ~4×
  further and drops the ID/OOD correlation from 0.9 to 0.4. Chemprop exposes `KMEANS` and
  `KENNARD_STONE` splitter keys already.
- **Name the implementation and the seed**, not just "scaffold split."
- **Quote the leakage number.** Fraction of test molecules with a training neighbour above a
  Tanimoto threshold is one line of RDKit and it's the number that actually describes the
  difficulty of your split.

## Why this matters for the rest of this series

I have been quoting "median 2.89× cache speedup under scaffold shift" as my conservative number
for seven posts. It isn't conservative. It's a mild-shift case, and the genuinely-novel-chemistry
number is probably lower. I've corrected Part 2 to say so, and re-measuring hit rates under a
clustering split is the obvious follow-up I haven't run.

The same caution applies to every quantization result in Part 5, where the noise floor was
estimated across random and scaffold splits. The floor is real; whether it covers real
distribution shift is now an open question rather than an assumption.

## Reproducing

```bash
python scripts/split_audit.py     # scaffold statistics; the cross-implementation
                                  # comparison needs chemprop + astartes + deepchem installed
```
