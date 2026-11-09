# A D-MPNN is a non-backtracking walk, so you can cache it exactly

*Part 2 of a series on caching and quantizing D-MPNN inference.*

The previous post found that a molecular featurizer recomputes the same 102 atom vectors tens of
thousands of times, and that eliminating all of it buys only 1.5× because the duplicated work was
never the bottleneck. This post is about the message-passing loop, where the redundancy is both
larger and *exactly* exploitable.

Here is Chemprop's entire bond message, from `nn/message_passing/mixins.py`:

```python
M_all = scatter_sum(H, dest)[src]   # sum over edges incident to the source atom
M_rev = H[rev_edge_index]           # the reverse edge
return M_all - M_rev                # ...excluded
```

That subtraction is the whole story. The message arriving at directed bond `v→w` sums over bonds
`u→v` for every neighbour `u` **except `w`**. Messages never immediately turn around.

Which means the hidden state of edge `e` after `t` rounds is a function of exactly one thing: the
**non-backtracking walk tree of depth `t` rooted at `e`**. Not the molecule. Not the batch. That
subtree, and nothing else.

## Why that makes an exact cache possible

A function of a rooted labelled subtree is precisely what Weisfeiler-Lehman hashing canonicalises.
So you can key a cache on a WL hash computed over Chemprop's own feature vectors, with the
recursion mirroring the message rule exactly — including the non-backtracking exclusion:

```
h_e^(0) = hash(atom_invariant[src(e)], bond_invariant[e])
h_e^(t) = hash(h_e^(0), multiset{ h_f^(t-1) : f ∈ in(src(e)), f ≠ rev(e) })
```

No subgraph isomorphism, no approximation, no learned similarity. Two edges sharing a key
provably have identical hidden states under any fixed weights.

I checked that rather than assuming it. Across 11 MoleculeNet datasets, edges sharing a radius-*r*
key have hidden states agreeing to **≤7e-7** — float32 roundoff. So unique-key counts are not an
estimate of how much reuse exists; they are the exact minimal cache size. That 7e-7 deserves an
explanation, and it gets one further down: it is a property of the device, not of the key.

One subtlety worth flagging, because getting it wrong gives you a cache that is silently wrong:
the WL invariants must match the featurizer's invariants. RDKit's default Morgan invariants omit
hybridization and chirality, both of which Chemprop uses. You have to hash *Chemprop's* feature
vectors, not an off-the-shelf ECFP identifier.

## How much reuse is actually there

Pooled across 11 datasets, maximum achievable hit rate for directed-bond environments:

| radius | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| hit rate | 99.9% | 97.2% | 87.9% | 77.1% | 67.9% | 60.6% |

The decay is the central constraint on this whole idea, and it has a measurable cause: **at radius
6, 69.2% of atom environments are the entire molecule.** A radius-5 environment around an atom in a
21-heavy-atom molecule covers most of it, so sub-molecular caching degenerates toward per-molecule
caching as depth grows. **Reuse is governed by receptive-field radius, not by layer
count** — a distinction that cost me an entire experiment, which is Part 6.

## Two engineering details that decide whether it's worth anything

A cache only helps if looking something up is cheaper than recomputing it. My first implementation
was *slower* than no cache at all, twice, for two different reasons.

**The keys have to be cheap.** Hashing float feature rows in Python cost **94 µs/mol** — more than
the entire default-size forward pass. Vectorising the hash with order-invariant additive mixing
(so no sorting) got it to 74 µs/mol. What actually fixed it was noticing that the input-graph cache
from Part 1 already assigns every atom and bond a small integer type ID, and those IDs *are* valid
WL seeds. Seeding from them instead of hashing floats: **11.8 µs/mol, an 8× reduction.** The two
caches are the same artifact, and the first one pays for itself twice.

**The schedule has to skip work.** My first version computed the neighbour-sum scatter at every
layer whether or not it was needed. With a 100% hit rate and zero GEMMs executed, it still cost
60% of a full forward. The fix is to probe all layers first, find the *deepest* fully-cached layer,
gather it directly and skip everything beneath it — and never run the scatter when a layer is a
complete hit.

With both fixes, on a 400-molecule analog series:

| model | full forward | cached (cold) | cached (warm) | max abs err |
|---|---|---|---|---|
| default `d_h=300, depth=3` | 71 ms | 36 ms (2.00×) | 24 ms (**2.97×**) | 2.4e-7 |
| `d_h=600, depth=5` | 247 ms | 148 ms (1.67×) | 39 ms (**6.30×**) | 3.0e-7 |
| CheMeleon `d_h=2048, depth=6` | 1091 ms | 589 ms (1.85×) | 138 ms (**7.89×**) | 8.3e-7 |

The error column is the point. These are float32 roundoff, which also proves the additive keying
produced no collisions on this data — a collision would show up as a large error, not a small one.

## Where the 7.9× comes from, and why you probably won't get it

That analog series was synthetic: one biaryl amide scaffold × 20 × 20 R-groups. It's the best case
by construction, and it's worth understanding *why*, because the reason generalises.

400 molecules from that enumeration produce only **1,205 distinct radius-6 bond environments**.
The marginal hit rate for the *N*th analog reaches 100% by around the 50th. A combinatorial
enumeration has **O(R) distinct environments for O(R²) molecules** — once you've seen each
substituent in each position once, everything afterwards is a lookup.

That is exactly the workload of medicinal chemistry SAR exploration: one scaffold, small
variations, hundreds of compounds. It is *not* the workload of a diverse screening deck.

On real data, across 11 datasets:

| regime | range | median |
|---|---|---|
| cold (encode a dataset once) | 3.13× – 8.22× | 3.76× |
| warm, random split | 3.02× – 11.43× | 4.05× |
| warm, **scaffold split** | **1.65× – 9.00×** | **2.89×** |

And it is strongly chemistry-dependent in a way that tracks molecule size: the congeneric BACE
series reaches 9–11×, reproducing the synthetic result, while QM9 bottoms out at **1.65×**. Large
drug-like molecules have many repeated local environments; QM9's 5.5-heavy-atom molecules have
almost none.

So the honest headline is **median 2.89× under scaffold shift, 9–11× on congeneric series**, not
the 7.9× the synthetic benchmark advertised.

And a caveat on that bottom row, which I had been treating as the conservative number. **A
Bemis-Murcko scaffold split is a much weaker distribution shift than it sounds.** Median
train-test Tanimoto distance moves only 0.577 → 0.610 going from random to scaffold, against
0.705–0.714 for UMAP or Lo-Hi clustering splits, and in-distribution/out-of-distribution
performance correlates at r ≈ 0.9 under scaffold splitting versus r ≈ 0.4 under clustering
(Fooladi et al., JCIM 2025). The mechanism is that there are only ~1.93 molecules per scaffold on
average, so as the unique-scaffold-to-molecule ratio approaches 1, scaffold splitting *converges
to random splitting*. Measured leakage bears it out: 56% of MoleculeNet HIV test molecules have a
training neighbour at ECFP4 Tanimoto > 0.4 under HIV's own recommended scaffold split, rising to
96–99% for MUV, QM8 and QM9 (Steshin, NeurIPS 2023 D&B).

So **2.89× is not a worst case** — it is a mild-shift case, and a genuine cluster split would
likely push it lower. Re-measuring hit rates under UMAP or Butina splits is the obvious follow-up
and is not done here.

Two further limits. Atom-level environments (the readout's input) hit only **12–54%** at depth 6,
so the final layer caches far worse than the message-passing layers. And memory is modest but
real: 0.06–1.38 GiB in fp32 for the depth-1..5 edge states, which Part 5 shows drops to
0.008–0.183 GiB at int4 for +0.25% RMSE.

I expected to beat that with a low-rank projection, on the reasoning that a 2048-dimensional
embedding can't be using 2048 dimensions. **That was wrong, and informatively so.** The molecule
embedding has a participation ratio of 3.30 and puts 51.8% of its variance in a single direction,
with 90% of variance in 11 directions — but reconstructing from rank *k* and re-running the task
costs +12.5% at k=64, +5.4% at k=128 and +1.0% at k=256, and some datasets need k=512 to stay
inside the noise band. **90% of the variance is in 11 directions and none of the signal is there.**
At matched memory, rank-256 costs +10.9 to +15.5% where int4 group-32 costs +0.09 to +0.54% —
20–60× worse accuracy per byte, and the low-rank basis was fitted with the test molecules included,
so it's the optimistic version.

The per-edge hidden states are worse still for this idea: their rank *grows* with depth
(participation ratio 7.3 at layer 0 to 57.9 at layer 5; n95 from 23 to 708), which is the opposite
of the rank-collapse story the over-smoothing literature would predict.

What the same measurement did turn up is a better lever: **`H` is 94.6–99.0% exact ReLU zeros**,
24–112 non-zeros per edge, with only 852–1892 of the 2048 dimensions ever non-zero in a given
layer. A bitmask-plus-int4 layout storing the zero pattern exactly comes to roughly 144–330 bytes
per edge against int4's 1088 — a further **3.3–7.6×**. That is a layout estimate with no kernel
behind it, but sparsity rather than low rank looks like where the remaining memory is.

## When the cache is valid, and when it is silently wrong

A cached hidden state is only correct for the weights that produced it. Every optimizer step
invalidates the entire cache, so this is an inference and frozen-encoder technique, never a
from-scratch-training one. Three specifics worth knowing:

- **`DropoutEstimator` breaks it.** Chemprop's MC-dropout uncertainty estimator activates dropout
  at inference time, which makes hidden states stochastic. The cache must be bypassed entirely.
  CheMeleon ships with `dropout=0.0`, so it is deterministic even in train mode.
- **Ensembles need one cache per checkpoint** — different weights, different states. But all `M`
  members share the Part 1 input-graph cache, so ensembling amplifies that win proportionally.
- **MVE, evidential, quantile and Dirichlet are fine.** Single deterministic encoder pass, different
  head; cache the encoder output once and fan out.

## An unreasonably cheap side effect

A cache miss at radius *T* means, literally: *this substructure has not appeared in anything I have
seen.* The fraction of novel environments in a molecule is therefore an applicability-domain
signal that falls out of the cache for free — no distance metric to calibrate, no separate model,
no density estimate.

I want to flag this as a hypothesis rather than oversell it, because molecular applicability
domain is a graveyard. GNN **latent-space** distance to the training set has been measured against
prediction error at Spearman ρ ∈ [−0.08, 0.20] — noise (Parrondo-Pizarro et al., JCIM 2026). A
vanilla MPNN's distance-awareness AUC is 0.500 ± 0.004, exactly chance (Han et al., NeurIPS 2021
DistShift). The classic OOD scores are at or *below* chance on molecules: MSP, Energy and ODIN all
land at 45–57 AUROC on BACE/BBBP/HIV while the same backbone reaches 79–97% in-distribution
accuracy (SGOOD, CIKM 2024). And there is a theorem that ReLU networks produce almost always
high-confidence predictions far from the training data (Hein et al., CVPR 2019) — a D-MPNN is a
ReLU network.

Two reasons I still think it is worth testing. It is an **input-space** measure, not a latent one,
and the one consistent positive in this literature is that input-space distance beats latent
distance — MOOD's authors concluded that "the input space distance is a good surrogate for the
internal model representation distance," i.e. the latent space adds nothing over it. And
remarkably, **none of the four graph-OOD benchmarks — GOOD-D, UB-GOLD, PGR-MOOD, SGOOD — contains
a fingerprint or Tanimoto baseline at all.** The obvious input-space control has never been run.

One structural limit to state up front: a novelty-based signal **cannot** flag activity cliffs by
construction, since a cliff compound sits near the training set and is still mispredicted. Cliff
prevalence runs 7–52% depending on target (van Tilborg et al., JCIM 2022), so an honest evaluation
has to decompose errors into "far from training" versus "near but cliff-like."

So: a testable hypothesis with a plausible mechanism and an unoccupied control — not a result.

## Determinism: where that 7e-7 comes from

An exact cache needs an exact forward pass. "Same key implies same hidden state" is only true if
computing the hidden state twice gives the same answer twice — and float addition is
non-associative, so a scatter that accumulates in a different order produces a different result in
the low bits. The 7e-7 above is therefore ambiguous: either the WL key is slightly imprecise, which
would be a correctness bug, or the scatter is nondeterministic, which is a device property.

It's the second. Running the same `index_add_` twice on identical inputs:

| | repeatable across identical runs |
|---|---|
| CPU, `deterministic=False` | **True** |
| CPU, `deterministic=True` | **True** |
| MPS, `deterministic=False` | False |
| **MPS, `deterministic=True`** | **False** |

On CPU the cache is **bitwise** exact and the 7e-7 never appears. On MPS it's exact to ~1e-6 and no
better. Two things follow.

First, a bug worth reporting: **`torch.use_deterministic_algorithms(True)` on MPS neither raises
nor works** for these ops in torch 2.11. It silently ignores the request. And
`chemprop/cli/train.py:1826-1829` couples the two — passing `--pytorch-seed` sets Lightning's
`deterministic=True`, omitting it sets `False`. So on Apple Silicon `--pytorch-seed` gives you a
reproducible *seed* and a non-reproducible *forward pass*, with no warning that the second half of
the promise wasn't kept.

Second, and more usefully: **the cache is itself a determinism mechanism.** A hit returns a stored
value — no reduction, no accumulation, no ordering. So at an 87.9% hit rate, 88% of hidden states
are bitwise reproducible by construction, on any backend. Higher hit rate means more determinism,
for free. Relatedly, the Hashimoto kernel in Part 7 is gather-and-add with no atomics and no write
contention, which makes it deterministic *by construction* — no flag, no slow path. That is an
argument for it I under-weighted when I ranked it on speed alone.

The practical constraint this puts on persistence: if you write an environment cache to disk and
reuse it in a later process, cached values must match what that process would compute. Without
bitwise determinism they don't, and you get batches of mixed provenance whose inconsistency share
*grows with hit rate*. So the cache key has to include the compute environment — torch version,
device, dtype, deterministic flag — not just the molecular environment.

## Determinism will not reduce your seed variance

Worth separating, because the terms get used interchangeably. **Run-to-run variance at fixed seed**
is what `use_deterministic_algorithms` removes. **Seed variance** — different init, batch order,
dropout masks — is untouched by it. **Split variance** dominates both, at 4–12% CV on random splits
and 8–56% including scaffold splits. The ~1e-6 from nondeterminism is four orders of magnitude
below the 2.3% paired-delta resolution limit, and about 100× below the int4 cache quantization error
we accept at +0.25%. For accuracy purposes it is irrelevant.

The data layer is where determinism actually bites, because **batch order changes the optimization
trajectory, so it changes the converged model rather than its low-order bits.** Three hazards in
Chemprop's current path, all measured:

- **`class_balance=True` with `seed=None` is silently nondeterministic.** It's a legal call, and
  `ClassBalanceSampler` builds `np.random.default_rng(None)`. Confirmed: three runs, three
  different orders.
- **`shuffle=True` with `seed=None` falls through to torch's global RNG.** Reproducible only if
  nothing else in the process consumed random numbers between `manual_seed` and iteration. Passing
  `generator=` would isolate it; `build_dataloader` doesn't.
- **`SeededSampler` is stateful across epochs.** The generator is built in `__init__` and shuffled
  in `__iter__`, so the epoch→order mapping depends on *how many times the loader was iterated*. A
  sanity-check pass, a validation hook or a profiler run permanently desynchronises every later
  epoch of an otherwise reproducible run. Per-epoch reshuffling is correct; the coupling to call
  count is the hazard.

And one smaller thing at the dataset level: `normalize_targets()` reduces over the training rows, so
a different row order perturbs the scaler in its low bits (measured max difference 2.3e-13), and
that perturbation then multiplies every target. Tiny, but it means "same seed, same split,
different shard order" is not bitwise identical either.

If you want reproducible Chemprop runs today: pass `--pytorch-seed`, never combine `class_balance`
with an unset seed, don't iterate the training loader outside the training loop, and on Apple
Silicon don't believe the forward pass is deterministic just because you asked.

## Status

Honest scoping: this is a prototype in a standalone harness that reproduces Chemprop's
message-passing math and is verified bitwise-identical to it. It is **not** integrated into
Chemprop, so the speedups are measured against my own faithful reimplementation rather than
against `chemprop train`. Integration mainly means persisting the type-ID cache and wiring the
schedule into `_MessagePassingBase.forward`; the keying is the part that's done.

Next post: why this is *not* a KV cache, despite looking like one — and what the difference
implies about which graph architectures are cacheable at all.

## Reproducing

```bash
python scripts/env_cache_study.py   # hit rate by radius
python scripts/cached_mpnn3.py      # the scheduled cache, exactness + timings
python scripts/cache_hits.py        # the 11-dataset hit-rate study
python experiments/determinism_probe.py   # kernel-level: where the 7e-7 comes from
python experiments/determinism_data.py    # dataloader/sampler/scaler hazards
```
