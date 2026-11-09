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

`NormAggregation` (divide by a fixed constant rather than by *n*) is the obvious compromise and is
already implemented. It preserves the extensive scaling while keeping activations bounded.

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
