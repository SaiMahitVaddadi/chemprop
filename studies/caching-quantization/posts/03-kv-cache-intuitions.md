# Why KV-cache intuitions don't transfer to graph networks

*Part 3 of a series on caching and quantizing D-MPNN inference.*

When I started this, my mental model was "a GNN forward pass is like an LLM forward pass, so I
should be able to do something KV-cache-shaped." That intuition is wrong in a specific and
instructive way, and working out exactly how it fails turned out to be more useful than the cache
I eventually built.

## What makes a KV cache work

An LLM's KV cache is not really a memoization table. It works because of causal masking: the keys
and values at position *i* depend only on tokens ≤ *i*, so once computed they are **final**, and
every later decoding step reuses them. The reuse is *intra-sequence* and *temporal*. There is a
direction of time, and the cache exploits it.

Now look at a D-MPNN forward. There is no autoregressive axis. All edges update simultaneously,
`depth` times. Nothing is "final" partway through in a way later steps consume — each round
overwrites the last. There is no temporal direction to exploit, so there is nothing to cache
*within* a single molecule's forward pass.

The reuse that does exist is across *different molecules that happen to share substructures*. The
right analogy is not per-sequence KV reuse; it is **cross-request prefix sharing** in a long-lived
serving process, where two unrelated requests happen to begin with the same tokens.

## The attractive wrong idea

If you accept the prefix-sharing framing, an obvious design follows. Medicinal chemists explore
analog series: one scaffold, many small R-group edits. So cache the scaffold's partial state and
recompute only the changed substituent. That's prefix caching, and the shared prefix is the
scaffold.

It doesn't work, and the reason is structural rather than an implementation detail. **In a D-MPNN,
messages flow inward.** The scaffold atoms receive messages from the substituents. So the
scaffold's representation *depends on* the R-groups — there is no prefix whose value is
independent of the suffix. Unlike attention with causal masking, the dependency runs both ways.

I measured how bad it is. Take one scaffold, change a single R-group, and count what fraction of
atom environments change:

| edit | dirty @ r=1 | @ r=3 | @ r=5 |
|---|---|---|---|
| R₂: CH₃ → F | 11.8% | 35.3% | 47.1% |
| R₂: CH₃ → CF₃ | 25.0% | 45.0% | 55.0% |
| R₂: CH₃ → morpholine | 45.5% | 68.2% | 81.8% |
| R₂: CH₃ → pyridyl | 59.1% | 72.7% | **90.9%** |

Swapping a methyl for a pyridyl invalidates **91% of a 22-atom molecule** at radius 5. There is no
prefix left to reuse. Parent-child diffing — the thing that feels most like a KV cache — is the
one strategy that definitively fails here.

## What actually works is the opposite of locality

The cache from Part 2 gets 9–11× on congeneric series not because consecutive molecules are
similar to their parents, but because **the same substituent-in-context recurs across the whole
library**. 400 analogs from a 20 × 20 enumeration contain only 1,205 distinct radius-6
environments: once you've seen each substituent in each position, everything after is a lookup.

That is *recurrence*, not *locality*. It needs no provenance, no parent pointers, no edit tracking
— just a global content-addressed table. And it explains the chemistry dependence from Part 2:
congeneric libraries have high recurrence; QM9's small rigid molecules have almost none, which is
why it bottoms out at 1.65×.

So the correct serving design is genuinely KV-cache-adjacent, just not in the way I first thought:
a long-lived warm process holding a paged, content-addressed environment table shared across all
requests, with recency-based eviction. Plus the trivial outer layer worth building first — an
exact SMILES → prediction memo, since chemists resubmit identical structures constantly and that
costs nothing.

## What *would* restore prefix caching

Here is the part I find most interesting, and it's an open question rather than a result.

The objection above only applies to *synchronous* message passing, where every edge updates at
once. A **canonical ordered traversal** would impose a causal order — process edges in a
deterministic sequence derived from the graph alone (canonical SMILES order, Morgan canonical
ranking, a minimum-DFS code), updating each as you go. At that point the computation becomes
autoregressive-like, and shared *prefixes of the traversal* across molecules become legitimately
reusable. That is prefix caching in the real sense.

There is published work in the neighbourhood — asynchronous and sequential message passing, which
is motivated by propagating information further per sweep rather than by caching. And the
scheduling connection runs deeper than it looks: priority-ordered relaxation *is* Dijkstra, and
GNN aggregation is known to align with Bellman-Ford, so "traversal order" and "graph algorithm"
are the same question wearing different hats.

Prefix reuse is genuinely available, too: with *rooted* canonical SMILES, **75.9% of token
positions are reusable** across a 400-analog library, against 63.2% unrooted. So the mechanism
isn't the problem.

Three objections are, and the first is the one that actually kills it:

1. **The causality it needs is physically false.** Ordered traversal only buys prefix reuse if the
   scaffold's representation doesn't depend on its substituents. But that dependence is *real
   chemistry* — it is the Hammett effect. An electron-withdrawing para substituent genuinely
   changes the electronic environment of the ring it sits on. So the 47–91% dirty fraction above
   is not cache inefficiency waiting to be engineered away; **it is the model correctly reporting
   a physical effect.** A cache that skipped it would be a worse model. This is the real reason
   the idea fails, and unlike the other two it is not fixable.
2. **Imposing an order breaks permutation invariance, and the canonical order isn't stable.**
   Canonicalisation recovers invariance in principle, but RDKit canonical ranks shift for up to
   **69% of scaffold atoms** on a single R-group swap — so the key is unsound on exactly the edits
   that constitute the workload. It also breaks `mol_atom_bond.py`, where per-atom and per-bond
   targets must stay aligned to input order.
3. **Serialisation is poison for throughput.** Chemprop's forward is fully vectorised over a flat
   batch of ~13,600 directed bonds. A priority-ordered sweep might cross the graph in fewer
   parameterised layers while being far slower in wall-clock. This is the kind of idea that wins
   on hops-per-sweep and loses on GPU utilisation, and I would expect it to lose.

The scheduling idea also has a cleaner negative than I expected. Algorithmic-alignment work
favours **Bellman-Ford** — a *synchronous* relaxation — over Dijkstra's priority order. Chemprop's
forward pass already *is* synchronous Bellman-Ford over the line graph, so it implements the
aligned algorithm; a Dijkstra schedule would align it with the wrong one while destroying batching
and breaking ONNX export, which is already fragile (issue #1335).

RNN- and transformer-over-traversal models (tree-LSTMs over DFS order, SMILES sequence models)
already have the causal property and already support KV caching trivially. They have also
historically underperformed D-MPNN on molecular property prediction, which is some evidence that
the order-imposing trade is not free.

## What the reverse term is actually worth

Chemprop has hard-coded that coefficient at 1 since 2019, justified by analogy to avoiding
"totters". Yang et al. reported the comparison as largely not statistically significant, which has
been read loosely ever since as "it doesn't much matter." It matters. Sweeping γ on ESOL
(`d_h=300`, scaffold-grouped 5-fold, two seeds):

| γ | depth 3 | depth 6 |
|---|---|---|
| 0.0 — ordinary random walk | 1.1982 | 1.3453 |
| 0.5 | 1.1805 | — |
| **1.0 — Chemprop / Hashimoto** | **1.0326** | **1.1031** |
| ≈2.0 | 1.1725 | — |

Non-backtracking is worth **13.8% at depth 3 and 18.0% at depth 6**, and γ=1 is a *sharp optimum*
rather than a plateau — 0.5 and 2.0 are both clearly worse. The benefit **grows with depth**,
which is what the operator algebra predicts: redundant back-and-forth walks accumulate faster the
longer the walk.

The non-significant reading compared three confounded architectures. A one-parameter sweep isn't
confounded, and it says Chemprop's hard-coded `−1` is right — better motivated than its own paper
established.

**And an important negative from the same experiment: a learned γ does not learn.** Initialised at
0, 1 and 2 it converged to 0.004, 1.001 and 1.996 — it stays wherever you put it. The gradient
signal is too weak to move it. So this is a hyperparameter to sweep, not a parameter to fit. I
recommended the learnable version before running it, and that was wrong.

## The design rule this gives you

Generalising the message rule is where most architectural ideas for Chemprop live, and the cache
sorts them cleanly. The message is currently an unweighted sum with a hard-coded coefficient of 1
on the reverse term:

```
M_e = Σ_{f: dest(f)=src(e)} H_f  −  1 · H_rev(e)
```

- **Weighted walks** (learned attention over the directed line graph, or chemistry-derived weights
  from bond order and conjugation): the weights are functions of the local environment, so the WL
  key is unchanged and only the cached *value* changes. **Cacheability preserved exactly.**
- **A γ coefficient on the reverse term** (γ=1 is Chemprop, γ=0 is an ordinary random walk): one
  scalar, same key. **Preserved.** This was the cheapest interesting experiment in the series and
  it has now been run — see below. Sweep γ as a fixed hyperparameter; do *not* make it learnable.
- **Bidirectional channels:** separate forward/reverse weights. **Preserved.**
- **Multi-hop per layer** (aggregate 1-, 2-, 3-hop simultaneously): radius per layer grows, so
  each layer's key covers a larger environment. **Cacheability gets worse.** And it's FLOP-neutral
  — two GEMMs across half the layers — so it buys gradient flow, not speed.
- **Shortest-path distance as a pairwise bias** (Graphormer-style spatial encoding): O(n²) and
  global. **Breaks it outright.**
- **Global attention, dataset-level motif graphs:** break it.

The rule: **anything that stays local preserves the WL key; anything global destroys it.** Which
happens to align with what the molecular GNN literature says actually works — local substructures
carry most of the signal, and depth is the weakest scaling axis.

## Status

The measurements here (dirty-set fractions, environment counts, hit rates) are real. The traversal
direction is a research question I have *not* executed: no canonical-order model was trained, and
the γ ablation is written but unrun. I'd rather label that clearly than imply the conclusion is
settled.

What is *not* executed is a trained canonical-order model, so the traversal idea is argued down on
three grounds rather than measured down. I think the physics objection is decisive on its own and
needs no experiment; the other two are engineering objections a determined implementation might
dent. Also unrun: γ at 0.5 and 2.0 at depth 6, and the wider depth sweep.

## There is no starting point to choose

A natural follow-on: if ordering matters, pick a good *starting* atom — the most connected one, the
centre, a random one — and propagate from there. It's an appealing study and the first result closes
most of it.

Chemprop's update touches every directed edge simultaneously, `depth` times, with no ordering
anywhere. The update is equivariant to atom relabelling and the readout is a permutation-invariant
pool, so the molecule representation **cannot depend on which atom you call first.** Verified by
permuting the featurized graph directly and re-encoding: max relative difference **3.0e-07**, which
is float summation order in `index_add_`, not a broken symmetry.

So "where does message passing start" is not a question the architecture admits. It becomes one
only if you deliberately break invariance, and there are three ways, in increasing invasiveness:

**A. Anchor positional encoding.** Pick an anchor by some rule and add
shortest-path-distance-from-anchor as an extra atom feature. No architecture change — it rides in
through `--atom-features-path`. This is the cheap, testable version, and it has a measurable cost
specific to this project: distance-from-anchor is a **global** feature, so unlike RWSE (provably a
function of the radius-⌈k/2⌉ ball, Part 8) it is **not environment-cacheable.** Measured over nine
single-substituent swaps on one scaffold:

| anchor rule | anchor moves | atoms whose feature changed |
|---|---|---|
| **max degree** | **0%** | **9.2%** |
| **centroid (min eccentricity)** | **0%** | **9.2%** |
| canonical rank 0 | 11% | 17.6% |
| closeness centre | 22% | 27.5% |
| periphery | 22% | 26.1% |
| **random** | **89%** | **84.3%** |

Degree- and centroid-based anchors are stable; random and canonical-rank are not. Note the
canonical-rank row — this is the same instability that sinks the traversal idea above, showing up in
a milder form. And even the best rule changes the feature on ~1 atom in 11 per edit, each of which
is a cache miss. So if you anchor, use max-degree or centroid, and budget for the cache cost.

**B. Anchor ensembling.** Compute *k* rooted representations and pool them, recovering invariance by
averaging over roots. Costs k× compute. Not implemented.

**C. An asynchronous or priority-ordered schedule** from the anchor. This is the traversal idea
argued down above, on physics, canonical-rank instability, and batching grounds.

My honest expectation is that A is worth one afternoon and B and C are not, because the measured
anchor instability and the Hammett objection both point the same way. But A is cheap enough that the
answer should be measured rather than predicted — the code is in `experiments/anchor_study.py`.

## Reproducing

```bash
python scripts/analog.py            # dirty-set locality, analog-series hit rates
python scripts/gamma_ablation.py    # the γ sweep
python scripts/traversal_prefix.py  # rooted-canonical prefix reuse, rank stability
```
