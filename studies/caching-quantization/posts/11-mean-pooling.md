# Mean pooling is why your GNN can't count atoms

*Part 11 of a series on Chemprop internals.*

A frozen CheMeleon embedding is 2048 numbers per molecule. I wanted to know what's in them, so I
fit linear probes from the embedding to 37 RDKit properties. Most of it went as expected —
functional groups are trivially decodable, AUROC 0.989 to 1.000 — and then molecular weight came
back at **R² = 0.679**.

That's badly wrong. Molecular weight is a sum over atoms. ECFP4 counts get it at R² = 0.996. Every
published probing result I know of puts molecular weight at essentially 1.00. A pretrained
molecular encoder that can't linearly predict molecular weight is either broken or being measured
wrong.

It was being measured wrong, and the fix is one multiplication.

## The embedding is a mean

Chemprop's default readout is `MeanAggregation`: the molecule vector is the **average** of its
per-atom vectors. So the cached embedding `Z` relates to the sum-aggregated embedding by exactly
one factor:

```
Z_sum = Z × n_heavy_atoms
```

Multiply the embedding by the molecule's heavy-atom count — recovering what sum pooling would have
produced — and re-fit the identical probe:

| target | mean-pooled R² | sum-pooled R² |
|---|---|---|
| molecular weight | 0.679 | **0.999** |
| heavy-atom count | 0.694 | **1.000** |
| TPSA | 0.680 | **0.999** |
| heteroatom count | 0.657 | **1.000** |
| graph diameter | 0.778 | 0.951 |

And in the other direction, for an *intensive* property — one that doesn't scale with size:

| target | mean-pooled R² | sum-pooled R² |
|---|---|---|
| fraction sp³ carbon | **0.972** | 0.862 |

The information was never missing. **Mean pooling divides it out.** An average over atoms is
size-invariant by construction, so every extensive quantity — anything that grows with the number
of atoms — has its scale destroyed at the readout, after the message passing has already computed
it.

This is not a subtle effect. It is the difference between R² = 0.68 and R² = 1.00 on the single
most basic molecular property there is.

## Why this matters beyond probing

Chemprop ships four aggregations — `mean`, `sum`, `norm`, `attentive` — and `mean` is the default.
The choice is in the hyperparameter search space, so a tuned run can find `sum`. But most people
don't tune aggregation, and the default is actively wrong for an identifiable class of targets:

- **Extensive targets** should prefer `sum` or `norm`: anything that scales with molecular size.
  Enthalpies and free energies of formation, molar volume, polarizability, total surface area.
- **Intensive targets** should prefer `mean`: logP, logD, pKa, melting point, fraction sp³.

Solubility and most ADMET endpoints sit awkwardly in between, which is presumably why `mean` became
the default — it's the safer choice when you don't know. But "safer" here means systematically
discarding size, and the probe says the encoder *computed* size and the readout *threw it away*.

`NormAggregation` is the obvious compromise and is already implemented. Worth being exact about
what it does, because the word "norm" means three incompatible things across codebases:

```python
class NormAggregation(SumAggregation):
    def forward(self, H, batch):
        return super().forward(H, batch) / self.norm     # norm = 100.0 by default
```

It divides the **sum** by a *constant*. So it preserves extensive scaling exactly — it is sum
aggregation with the units changed, and the constant is tunable (`--aggregation-norm`, searched
over 1–200 in hpopt). Compare:

| aggregation | formula | extensive info |
|---|---|---|
| `sum` | Σ h_v | preserved |
| **chemprop `norm`** | **Σ h_v / c** | **preserved (rescaled)** |
| `mean` | Σ h_v / n | **destroyed** |
| L2 row-normalisation | h / ‖h‖₂ | destroyed, and magnitude too |

That last row matters because several graph libraries call it "norm aggregation." It is the most
aggressively intensive option of the four, not a compromise. "We used norm aggregation" does not
identify an experiment.

## The principled fix already exists, and it isn't a binary choice

I posed this as extensive-vs-intensive, pick one. That framing is too coarse, and the better answer
is in the **PNA** family (Corso et al., NeurIPS 2020): don't choose an aggregator, use several, and
attach a **degree scaler**

```
S(d, α) = (log(d+1) / δ)^α
```

with α ∈ {−1, 0, +1} giving attenuation, identity and amplification. **α = 0 is mean, α = +1 is
extensive, α = −1 attenuates** — so the mean/sum decision becomes a selectable (or learnable)
exponent rather than a fork in the road. If you're building this, that's the thing to build.

PNA's other half is multiple aggregators in parallel — mean, max, min, std, var, and higher
**moments (3rd, 4th, 5th)**. This is the part I think is genuinely underexplored for molecules,
because mean and sum both collapse the per-atom distribution to its first moment. Neither can
distinguish *a molecule with a few very unusual atoms* from *a molecule of uniformly ordinary
atoms* — the variance and skew of the per-atom embedding carry exactly that heterogeneity, and both
default readouts throw it away. The probe suite above would test it directly: does variance
aggregation decode anything mean and sum cannot?

A wider aggregation design space than Chemprop's four is worth knowing about, and a reference
implementation of most of it exists in [EGAT](https://github.com/SaiMahitVaddadi/EGAT-Private/tree/v2)
(`src/models/aggregators/`): PNA with moments 3–5, weighted sum/mean, attentive variants that
return a global **edge** feature alongside the node one, bond- and edge-level pooling, learned
hierarchical pooling (DiffPool, SAGPool, cluster pooling), and a path-integral pooling layer with
learned per-walk-length weights.

Two of those deserve calling out. **Edge-level readout** is a real structural difference:
Chemprop scatters its final edge states to atoms and then discards them, even though the edge
states are the thing message passing actually computes — and Part 2 measured atom-level
environments caching far worse (12–54%) than edge-level ones, so there's an efficiency argument on
the same side. And **path-integral pooling with learned weights over walk lengths** is a learned
generating function over powers of the walk operator, which is the same object as the Ihara zeta
function from Part 7 — a learned Ihara-style readout over the non-backtracking operator is an
interesting thing nobody appears to have built.

One convenient fact about all of this: **aggregation is downstream of the cache.** The environment
cache from Part 2 stores per-edge hidden states, and readout happens afterwards, so you can change
aggregation freely without touching any of the caching work. The exception is learned hierarchical
pooling (DiffPool, SAGPool), which assigns clusters globally and therefore breaks the
locality rule from Part 3 — those interact with the cache, the rest don't.

I'd temper expectations on the hierarchical-pooling family specifically: their record on molecular
property prediction is mixed, they add learned-assignment cost, and the evidence from Part 8 is
that molecular graphs don't suffer the bottlenecks that motivate them.

There's also a direct tie to Part 6's depth result. Truncating CheMeleon from depth 6 to depth 3
costs 15–30% RMSE, and probing says a large part of what's lost is heavy-atom count and molecular
weight (+0.23 probe score each from depth 4–6). Those are *size* quantities. The network spends
deep layers reconstructing information that mean pooling then partially discards — which is a
strange division of labour, and suggests the readout, not the depth, is the thing to fix first.

## Two findings that should temper the enthusiasm

**Most of what probes detect is architectural, not learned.** I ran the same probes on a
**randomly initialised** CheMeleon. Functional-group AUROC: random encoder **0.993–1.000** against
the pretrained encoder's 0.989–1.000. The lift from pretraining is ≤ +0.005 — within noise. A
control task (probing for a random label) scores 0.48–0.55, so the probes are working; it's the
pretraining that isn't adding anything on this class of target.

Pretraining does add real, specific things, and they're all topological: bridgehead-atom count
**+0.292**, max ring size +0.218, rotatable bonds and valence electrons +0.202, graph diameter
+0.201. So CheMeleon's pretraining buys ring-system and shape awareness, not functional-group
recognition — the message-passing architecture gets the latter for free.

**And against classical descriptors it's a dead heat.** Downstream, with a frozen ridge head across
11 datasets, ~200 native RDKit 2-D descriptors score **−2.19% mean** against CheMeleon — i.e.
slightly *better* — though that mean is dominated by QM9 at −40% and sits around +1.6% excluding
it. ECFP4 counts are +13.59% worse. The interesting number is the combination: CheMeleon plus ECFP4
plus RDKit2D is **−5.44%**, better than CheMeleon alone on 6 of 11 datasets and worse on 1. They
are **complementary, not redundant.**

Linear CKA between CheMeleon and RDKit2D is only 0.189 on z-scored features — they are not
computing the same thing. But under *sum* aggregation that jumps to 0.884, and removing the size
axis drops it back to 0.326. So the apparent similarity between a learned embedding and a
descriptor vector is largely **one shared size direction**, which is exactly the direction mean
pooling removes.

## The bigger loss is bonds, not scale

Everything above is about *how* atoms are pooled. The larger omission is *what* is pooled, and it's
the one I'd now put first.

Chemprop's `MPNN.fingerprint` is three lines:

```python
H_v = self.message_passing(bmg, V_d)   # atom states only
H   = self.agg(H_v, bmg.batch)         # atoms -> molecule
H   = self.bn(H)
```

Message passing computes its hidden states on **directed bonds**. Those states are consumed
*inside* `_MessagePassingBase.forward` — scattered into atoms via `M_v = Σ_{f: dest(f)=v} H_f` —
and never returned. Every readout in the standard model therefore operates on atoms, and the
per-bond states are gone before the aggregation layer ever sees them.

This is not a missing capability. `mol_atom_bond.py:229` already has `edge_finalize(H, E, E_d)`
with its own learned projection `W_eo = nn.Linear(d_e + d_h, d_h)`, and line 304 returns
`(H_v, H_e)`. The bond readout exists, is trained, and the plain `MPNN` has no path to it.

What the atom scatter destroys, precisely:

- **Per-bond identity.** Which bond contributed what is gone.
- **Every non-linear bond statistic.** The atom sum is a *linear* functional of the edge states, so
  anything non-linear across the bonds at an atom — variance, max, min, skew — is unrecoverable
  from it in principle, not just in practice.
- **The directed pairing.** `H_{u→v}` and `H_{v→u}` are different vectors, and the sum at *v*
  receives only the inbound half. The total "mass" survives (each directed edge lands in exactly
  one atom sum) but the pairing does not.

For reactions this is the sharpest version of the problem. In a condensed graph of reaction the
bonds *are* the signal — which bonds break and form is the reaction — and the standard model
projects them through an atom sum before any readout.

So here is the whole family, implemented for edge-level pooling in
`experiments/bond_aggregation.py`, with the measured participation ratio of each on a 100-molecule
batch at `d_h=300` as a conditioning check:

| mode | out dim | participation ratio | character |
|---|---|---|---|
| `sum` | d_h | 5.0 | extensive |
| `norm` (÷ constant) | d_h | 5.0 | extensive, rescaled |
| `mean` | d_h | 15.1 | intensive |
| `l2row` | d_h | 16.5 | intensive, magnitude-free |
| `max` | d_h | **37.9** | order statistic |
| `min` | d_h | 17.9 | order statistic |
| `std` | d_h | **30.3** | spread — *not* recoverable from an atom sum |
| `var` | d_h | 20.7 | spread |
| `moment3/4/5` | d_h | 15.8 / 4.7 / 3.6 | skew, kurtosis |
| `pna` | 4 × \|scalers\| × d_h | 28.4 | PNA stats × degree scalers |
| `attention` | d_h | 15.1 | learned, softmax weighted |
| `gated` | d_h | 4.7 | learned soft selection |
| `bondtype` | n_types × d_h | 18.8 | one pool per bond order |

The participation ratios are the interesting column. `sum` and `norm` sit at **5.0** — nearly all of
their variance in a handful of directions — while `max` reaches **37.9** and `std` **30.3**. The
order statistics and the spread spread information across many more directions than the sum does,
which is consistent with them carrying something the sum cannot express.

Two implementation notes. The `attention` mode subtracts a per-molecule maximum before
exponentiating; Chemprop's `AttentiveAggregation` calls `.exp()` directly on its logits with no
max-subtraction, which is a textbook overflow hazard (`exp` of a large logit gives `inf`, then
`inf/inf` gives `nan`). And `bondtype` pooling — a separate pool per bond order, concatenated — is
the cheapest chemistry-aware option in the list and has no analogue in Chemprop at all.

**What this does not yet show is that any of it helps.** The table measures shape and conditioning
on a model whose weights were trained with an atom-only readout, so these edge states are not what
a bond-readout model would learn. The number that would settle it is the probe suite from earlier
in this post, run on atom-only versus atom-plus-bond readout: if bond pooling decodes bond-flavoured
targets — rotatable-bond count, conjugation, aromatic-bond count, ring-bond count — that the atom
sum cannot, that quantifies the loss instead of arguing it. That experiment is wired up and unrun.

## Caveats

These are **linear** probes on **frozen** embeddings. A linear probe failing does not show
information is absent — only that it isn't linearly available. I have not run nonlinear probes,
and there's a good reason to distrust them here: probe capacity confounds everything, which is why
the control task matters.

The descriptor comparison uses RDKit's ~200 native 2-D descriptors because `mordred` and
`descriptastorus` weren't installed. CheMeleon was pretrained on 1,613 **Mordred** descriptors, so
the real descriptor-equivalence is probably *closer* than measured — this understates the case
against the learned representation.

And the frozen-CheMeleon-versus-descriptors comparison specifically is **already published twice**
(Praski et al., arXiv:2508.06199; Ben Hicham et al., arXiv:2604.16123). The mean-pooling result and
the depth decomposition are, as far as I can find, not.

## Reproducing

```bash
python experiments/rep2_probes.py        # the 37-target probe suite, both aggregations
python experiments/rep3_descriptor_equiv.py
python experiments/rep4_depth_probes.py  # probe score vs truncated depth
```
