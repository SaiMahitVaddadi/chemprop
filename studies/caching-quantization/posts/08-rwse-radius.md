# The one feature that attacks radius

*Part 8 of a series on caching and quantizing D-MPNN inference. The payoff, such as it is.*

This series has circled one tension for seven posts. The cache from Part 2 is governed by
**receptive-field radius**: 87.9% hit rate at radius 2, 60.6% at radius 5. Everything that would
shrink the radius turned out to cost accuracy (Part 6) or not shrink it at all. Ring 2-cells,
junction trees and multi-hop-per-layer all cut *layer count* by making each layer reach further,
which buys FLOPs and gradient flow and does nothing for the cache.

So the question I ended up with was narrow: **is there anything that gives a D-MPNN longer-range
information without enlarging the radius its hidden states depend on?**

There is one, and it isn't an architecture change.

## Random-walk structural encodings

RWSE is a per-atom feature vector whose *k*-th entry is the return probability of a *k*-step random
walk starting at that atom. It's standard in graph transformers (GraphGPS uses it) and it is cheap
— a few matrix powers of the normalised adjacency, computed once in preprocessing.

Fed into Chemprop as extra atom features via `--atom-features-path`, with no architecture change
at all, on ESOL trained from scratch at depth 3:

| | RMSE | seed sd |
|---|---|---|
| baseline | 1.0609 | — |
| **+ RWSE (K=6)** | **0.9933 (−6.4%)** | **3× lower** |

The variance reduction is arguably the more interesting half. Lower seed-to-seed spread on a small
noisy dataset means the feature is doing work the message passing was previously having to
discover unreliably.

## Why it doesn't break the cache

Here is the part that makes it fit this series rather than being a generic "add features" result.

`RWSE_k` at an atom is an **exact function of the radius-⌈k/2⌉ ball around that atom, plus the true
degrees of the atoms on that ball's boundary.** That's provable and was verified to machine
precision — a *k*-step closed walk cannot stray further than ⌈k/2⌉ hops and still return, and the
only thing it needs from outside the ball is how many ways there are to step off and come back,
which is the boundary degree.

So K=6 is a **radius-3 local feature**. It can be computed from a bounded neighbourhood, cached by
the same WL-style key as everything else in Part 2, and it carries information about 6-step walk
structure into a model whose hidden states only depend on radius 3.

That is the only measured thing I found that attacks radius rather than trading against it.

The cost is real but one-off: adding atom features changes the width of `W_i`, so a pretrained
encoder like CheMeleon would need re-pretraining to use it. For from-scratch Chemprop training it
is a command-line flag.

## While we're here: over-squashing is not Chemprop's problem

A large literature — rewiring, expander propagation, graph transformers — exists to fix
over-squashing, the exponential compression of a growing receptive field into a fixed-size vector.
I spent a while assuming it applied. It doesn't, and the measurement is clean.

Over-squashing severity is governed by how fast walk multiplicity grows with length. For a
*d*-regular graph that's `d^T`. For molecules, the relevant quantity is the spectral radius of the
non-backtracking operator `B` from Part 7, and on ESOL:

- median **ρ(B) = 1.126** — walk multiplicity grows at **1.13^t**, not `d^T`
- **26.8% of molecules are acyclic**, which makes `B` *nilpotent* — non-backtracking walks on a
  tree terminate, so beyond some length there is literally nothing to squash
- **18.6% have no non-backtracking walk of length 6 at all**

Chemprop's non-backtracking message rule is doing more than removing redundancy. It is keeping the
walk count almost flat in depth, which is why a 6-layer D-MPNN on a 20-atom molecule doesn't
exhibit the pathology the rewiring literature is designed around.

What *does* grow is plain coverage — at radius 6, 69.2% of atom environments are the whole molecule.
That is a caching problem, not an over-squashing problem, and it's the one this series actually ran
into. It also explains why Part 6's depth result went the way it did: there is no bottleneck to
relieve, so deepening neither helps via longer range nor hurts via squashing. It just moves you
further from whatever depth you trained at.

## Where that leaves the ranking

| change | measured effect | reduces layers? | reduces radius? | cacheable? |
|---|---|---|---|---|
| **RWSE K=6 atom features** | **−6.4% RMSE, 3× lower seed sd** | no | **effectively yes** | **yes, radius 3** |
| local edge-aware attention in `message()` | best-supported in a tuned 2025 sweep | no | no | yes, if softmax stays local |
| ring/cycle counts as bond features | free, preprocessing only | no | no | yes |
| synthetic distance/angle on bond triplets | mixed — see below | no | no | yes |
| γ on the reverse term | γ=1 already optimal (Part 3) | — | — | yes |
| ring 2-cells / junction trees / multi-hop | cuts layers | **yes** | **no** | worse |
| depth reduction | unresolved (Part 6) | yes | yes | yes |

Two caveats on that table. The synthetic-coordinate approach — the conformer-free half of DimeNet,
which adds distance and angle features to bond triplets without needing 3D — has a specific problem
for Chemprop: its own results show the line-graph configuration *Chemprop is permanently in* is
where the method backfires, with QM9 U₀ going from 106 to 204 meV. And "reduces radius: effectively
yes" for RWSE is a claim about what the model can *know* at a given radius, not a literal shrinking
of the message-passing receptive field.

## The honest summary of eight posts

The cache works, is exact, and is worth a median 2.89× under scaffold shift and 9–11× on
congeneric series. The fastest real win is a one-line kernel fix plus a compile flag, at 4.6×. The
quantization frontier is 8-bit weights and 4-bit cache. Depth reduction is unresolved rather than
dead, and settling it costs a pretraining run.

And the architecture investigation ended up mostly *validating* Chemprop rather than improving it:
the non-backtracking rule is a sharp optimum worth 14–18%, the operator it implements has a name
nobody attached to it, the synchronous schedule is the algorithmically aligned one, and the
over-squashing literature is solving a problem these graphs don't have. The one clear addition is a
preprocessing feature that costs a flag.

I started this expecting to find an architectural lever. The levers were in the featurizer, the
kernel, and the arithmetic.

## Reproducing

```bash
python scripts/depth_rwse.py        # RWSE locality proof + depth grid
python scripts/train_depth_rwse.py  # the from-scratch RWSE comparison
python scripts/nb_stats.py          # Ihara spectrum, nilpotency, coverage
```
