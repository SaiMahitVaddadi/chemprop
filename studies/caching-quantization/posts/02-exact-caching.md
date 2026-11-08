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
estimate of how much reuse exists; they are the exact minimal cache size.

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

Two further limits. Atom-level environments (the readout's input) hit only **12–54%** at depth 6,
so the final layer caches far worse than the message-passing layers. And memory is modest but
real: 0.06–1.38 GiB in fp32 for the depth-1..5 edge states, which Part 5 shows drops to
0.008–0.183 GiB at int4 for +0.25% RMSE.

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
no density estimate. I think this is the most underrated consequence of keying on exact
environments rather than on learned embeddings, and I haven't evaluated it properly yet.

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
```
