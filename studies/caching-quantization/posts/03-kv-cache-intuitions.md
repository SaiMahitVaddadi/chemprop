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

Two serious objections, both of which I'd want answered before believing it:

1. **Imposing an order breaks permutation invariance.** Canonicalisation recovers it in principle,
   but canonical orders are not stable under small edits — and small edits are precisely the
   workload. A canonical order that reshuffles when you add a fluorine gives you a cache that
   misses on exactly the molecules you cared about.
2. **Serialisation is poison for throughput.** Chemprop's forward is fully vectorised over a flat
   batch of ~13,600 directed bonds. A priority-ordered sweep might cross the graph in fewer
   parameterised layers while being far slower in wall-clock. This is the kind of idea that wins
   on hops-per-sweep and loses on GPU utilisation, and I would expect it to lose.

RNN- and transformer-over-traversal models (tree-LSTMs over DFS order, SMILES sequence models)
already have the causal property and already support KV caching trivially. They have also
historically underperformed D-MPNN on molecular property prediction, which is some evidence that
the order-imposing trade is not free.

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
- **A learned γ on the reverse term** (γ=1 is Chemprop, γ=0 is an ordinary random walk): one
  scalar, same key. **Preserved.** This is also the cheapest interesting experiment in the whole
  series and nobody appears to have run it since 2019; the code is written and unrun.
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
settled, because the honest summary is "the obvious analogy fails, the replacement works, and the
thing that would make the original analogy correct is untested."

## Reproducing

```bash
python scripts/analog.py          # dirty-set locality, analog-series hit rates
python experiments/gamma_ablation.py --quick   # written, not yet run
```
