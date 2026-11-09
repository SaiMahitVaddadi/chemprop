# The depth experiment that didn't work

*Part 6 of a series on caching and quantizing D-MPNN inference. This is the one where I was
wrong twice.*

Earlier in this series I built an exact cache for a D-MPNN's hidden states. The depth-*t* hidden
state of a directed bond is a pure function of its non-backtracking walk tree of depth *t*, so a
Weisfeiler-Lehman hash over that tree is a sound cache key — two edges with the same key provably
have the same hidden state, verified to float32 roundoff.

The catch is that reuse falls off hard with the radius of that tree. Pooled across 11 MoleculeNet
datasets, the maximum achievable hit rate by radius:

| radius | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| hit rate | 99.9% | 97.2% | 87.9% | 77.1% | 67.9% | **60.6%** |

[CheMeleon](https://doi.org/10.1021/acs.jcim.6c01546), the pretrained Chemprop foundation
encoder, runs at `depth=6`, which means radius 5, which means the 60.6% column. If it ran at
`depth=3` it would sit at 87.9%.

So I had what looked like a very good idea. If depth 3 matched depth 6 on accuracy, you would get
two compounding wins from one change: roughly **half the FLOPs**, and a cache moving from the 60%
regime into the 88% regime. Nothing else on my list did two things at once.

The literature was encouraging, too. [Sypetkowski et al.
(NeurIPS 2024)](https://arxiv.org/abs/2404.11568) ran the largest molecular GNN scaling study to
date and found depth is the *weakest* of the three scaling axes — Spearman correlation with
downstream performance of **−0.11 to 0.55 for depth** against **0.69–0.82 for width** — with the
explanation that molecular properties "can be well predicted only from small local substructures."
Chemprop's own hyperparameter search has never looked above `depth=6`, and its default is 3.

Then I measured it.

## The measurement

I truncated the pretrained depth-6 checkpoint at each depth, re-ran the standard protocol (frozen
encoder, ridge head fitted on fp32 train embeddings, 10 splits per dataset: 6 random plus 4
Bemis-Murcko scaffold), and paired the deltas per split against depth 6.

Pairing matters here. The fp32 baseline's own split-to-split coefficient of variation is 4–12% on
random splits, so the first thing to establish is the noise floor: **a paired delta below about
2.3% is not distinguishable from split noise.** Without that number none of the rest is readable.

| dataset | noise CV | depth 3 | depth 4 | depth 5 |
|---|---|---|---|---|
| ESOL | 3.97% | **+24.4%** | +14.9% | +2.5% *(unresolved)* |
| FreeSolv | 7.10% | **+23.7%** | +20.3% | +4.6% *(unresolved)* |
| Lipophilicity | 7.10% | **+15.0%** | +8.0% | +4.3% |
| QM9 (12 targets) | 12.14% | **+30.0%** | +16.3% | +7.1% |

Depth 3 costs 2.1 to 6.1 noise widths, on 4 of 4 datasets. That is not a marginal call. Accuracy
keeps accruing all the way to depth 6, with the largest single increment between depth 4 and 5.

Depth 5 is the only near-free truncation — and it is nearly worthless for the cache, because it
moves the pooled hit rate from 60.6% to 67.9%. The regime change I wanted lives at depth 3
specifically, and depth 3 is exactly where the accuracy goes.

## The obvious objection, and why it needed an instrument

There is a real confound in that table. `W_o`, the readout matrix, was trained to consume
depth-5 edge states. Feeding it depth-2 states is a *readout mismatch* stacked on top of whatever
genuine information loss there is. My protocol refits only the ridge head downstream of `W_o`, so
some unknown fraction of that +15–30% could be the readout being pointed at the wrong thing.

If refitting `W_o` closed most of the gap, the information would largely be present at depth 3
and a retrain would be clearly worth running. If it closed little, the information genuinely
accrues and depth reduction is dead.

Testing that turned out to be harder than the original experiment, and the failures are the most
useful thing I learned.

**Attempt 1: fit a fresh linear readout from the truncated pre-readout states to the labels.**
Catastrophic overfitting. The giveaway was that it came out *worse than the frozen readout at
depth 6* — FreeSolv RMSE 9.35 against 1.56. At depth 6 the information is by construction
identical, so a gap there cannot be information loss; it can only be variance. The experiment was
measuring its own estimator.

The arithmetic says why. The readout maps 2120 → 2048 dimensions. ESOL has 902 training molecules
and FreeSolv has 511. Fitting roughly 4.3 million parameters from 511 scalar labels is not an
experiment.

**Attempt 2: distil instead of fit — regress the truncated states onto the depth-6 embedding.**
Better idea, because the target is label-free and 2048-dimensional rather than scalar, so there is
vastly more signal per molecule. Same identifiability problem though: fitted on one dataset's
training rows, test reconstruction error was 0.057 on ESOL and 0.313 on FreeSolv. At depth 6 the
map `Z₆ = W_o A₆ + b_o` is *exactly linear*, so recovery must be perfect, and it wasn't. Still
underdetermined.

**Attempt 3, which works:** the distillation target is label-free, so the fit doesn't need the
evaluation dataset at all. Fit the readout on 8,000 molecules pooled from the *other* eight
datasets, with evaluation-set molecules removed — 3.8× overdetermined, no label leakage, no
transduction.

The same depth-6 identity now serves as a sanity check instead of a failure: in-pool
reconstruction error 0.0014, test reconstruction 0.004–0.007, and distilled task RMSE matching the
true depth-6 model (ESOL 0.7517 vs 0.7508; FreeSolv 1.4735 vs 1.4825). The instrument works.

## The answer

| dataset | depth | reconstruction err | frozen `W_o` | distilled `W_o` | gap closed |
|---|---|---|---|---|---|
| ESOL | 6 | 0.004 | +0.00% | +0.10% | *(reference)* |
| ESOL | 5 | 0.068 | +2.49 ±1.30% | +4.32 ±0.54% | −73% |
| ESOL | **3** | **0.151** | +24.39 ±2.85% | **+30.41 ±3.20%** | **−25%** |
| FreeSolv | 6 | 0.007 | +0.00% | −0.52% | *(reference)* |
| FreeSolv | 5 | 0.084 | +4.61 ±2.75% | +1.49 ±2.28% | +68% |
| FreeSolv | **3** | **0.242** | +23.72 ±5.82% | **+41.79 ±6.81%** | **−76%** |

**The refit closes nothing.** At depth 5 the depth-6 representation genuinely is largely
recoverable by a linear map (reconstruction 0.068–0.084), and on FreeSolv the distilled readout
lands within noise of true depth-6 accuracy. At depth 3 it is not recoverable (0.151–0.242), and
distilling *worsens* task RMSE rather than closing the gap — which is interpretable rather than
anomalous: forcing an imperfect imitation of an unreachable target is worse than letting the ridge
head adapt to whatever the depth-3 embedding actually is.

So the +15–30% is genuine information loss, not readout mismatch. The cheap explanation is gone.

Probing the truncated embeddings says *what* is lost, and it isn't what I'd have guessed. I
expected local properties to saturate early and global ones to keep improving. They improve
equally: the fraction of the depth-6 probe score already reached at depth 3 is **0.762 for local
targets and 0.791 for global ones**, with gains of +0.163 and +0.154 from depth 3 to 6. The only
thing that saturates is functional-group detection, and it saturates at **depth 1** — 99.6% of its
final score, with +0.002 left to gain.

What depth 4–6 actually buys is ring-system topology and size: bridgehead-atom count **+0.318**
(0.335 → 0.653), max ring size +0.287, spiro-atom count +0.245, heavy-atom count and molecular
weight +0.23 each, graph diameter +0.214. So truncating to depth 3 loses **ring topology and
molecular size, not chemistry** — the functional groups are all present after one layer. And size
is on that list for a reason that turns out to be the readout's fault rather than the depth's; see
Part 11.

One asymmetry is worth stating carefully, because it is the kind of thing that is easy to
overclaim in either direction. These are truncations of a checkpoint *trained* at depth 6, with
the depth-6 embedding as the distillation target. A *small* gap would have been strong evidence
that shallow models are viable. A *large* gap is only weak evidence against a **retrained**
depth-3 encoder — it shows depth-6 features are not reconstructible from three rounds, not that no
good depth-3 representation exists. What it does do is remove the cheap explanation. A full
retrain is now the only route, and it is a much more expensive experiment than this one.

## And then the other direction disagreed

Everything above truncates a checkpoint trained at depth 6. The complementary experiment is to
*train* at each depth from scratch and see what you get. On ESOL, `d_h=300`, scaffold-grouped
5-fold, two seeds:

| trained depth | RMSE |
|---|---|
| 3 | **1.0326** |
| 6 | 1.1031 |

**Trained from scratch, depth 3 beats depth 6** — the opposite sign from the truncation result.

Before reading too much into that: these are small from-scratch models on one 500-molecule
dataset, not CheMeleon, and they are not comparable to the frozen-encoder numbers above in
absolute terms. But the sign is the point, and there is a second clue pointing the same way. A
probe around the trained configuration finds error rising on *both* sides of it — depth 7 is +9.7%
and depth 8 is +15.0% relative to the trained depth. Error that grows in both directions is the
signature of **distance from the trained configuration**, not of information content. Which is
precisely what truncation measures and cannot separate.

So the two tests bracket the question rather than settling it. Truncation says depth-6 *features*
need six rounds to construct. From-scratch training says a depth-3 *model* can be at least as good
as a depth-6 one. Both can be true: a shallow network can learn a different, equally predictive
representation rather than a degraded copy of the deep one — which is exactly the scenario the
distillation experiment was unable to rule out, because its target was the depth-6 embedding.

The decisive experiment is to **re-pretrain CheMeleon at depth 3 or 4** and compare on downstream
tasks. That is a far more expensive run than anything in this series, and it is unrun. So the
honest status is: *depth reduction is not dead, it is unresolved, and the cheap ways of resolving
it are exhausted.* I had written "dead" here before the from-scratch numbers came in, which was
overclaiming from one direction of evidence.

## The part I got wrong twice

Having measured this, I then claimed something worse than the null result: that cache headroom and
tolerance for shallowness were *in tension* — that the datasets with the most to gain from caching
were the ones that could least afford a shallower model. I had one comparison in front of me,
Lipophilicity against QM9, and I read the direction off it.

That was backwards. Across all the pairings:

| dataset | atoms/mol | warm hit rate (r=2) | truncation cost (depth 3) |
|---|---|---|---|
| QM9 | 5.5 | **0.820** (lowest) | **+30.0%** (worst) |
| FreeSolv | 8.8 | 0.859 | +23.7% |
| ESOL | 13.3 | 0.887 | +24.4% |
| Lipophilicity | 27.1 | **0.946** (highest) | **+15.0%** (best) |

Spearman **−0.800**, Pearson −0.97. The correlation is *negative*: the datasets with the most
cache headroom lose the **least** accuracy when truncated. The two effects align rather than
conflict.

The mechanism is almost certainly molecule size, which is a perfect rank confounder here —
`atoms_per_mol` against hit rate ranks at **+1.000**. Large drug-like molecules have many repeated
local environments (so more reuse) and more additive, local properties (so a smaller receptive
field suffices). Small rigid molecules like QM9 at 5.5 heavy atoms have global electronic
properties needing the full receptive field, and few repeated environments to cache.

And this is n=4. A Spearman correlation on four points has p ≥ 0.33 for any |ρ| < 1. It is a
consistent sign with a mechanism-plausible confounder, not a measured relationship, and I am
reporting it as one.

So: my original framing was right, my correction to it was wrong, and the corrected version still
doesn't rescue depth 3 — because the alignment is real but the accuracy cost is also real.

## What survives

The depth lever is open but unresolved, and resolving it costs a pretraining run. What stands
regardless is that the caching result holds on its own at depth 6, with no architectural
multiplier: a median **2.89× under scaffold shift**, 9–11× on congeneric
series like BACE, and 1.65× on diverse sets like QM9.

It is also worth noting what the literature does *not* offer here. Across the rewiring and
expander-graph work (EGP, DRew, GTR, FoSR, SDRF) there is not a single explicit depth-reduction
claim with numbers attached. DRew goes to **23 layers** and its selling point is making depth
*usable*. Exphormer's theory explicitly requires O(log n) layers. Graph transformers use the same
or more layers than Chemprop on molecules — GPS uses 10 on ZINC and molhiv, Graphormer 12 — against
Chemprop's default 3.

The methods that do cut layer count — CIN's ring 2-cells at 2 layers, HIMP's junction trees at
2–3 — do it by making each layer reach *further*. That helps optimization and FLOPs. It does
nothing for the cache, because cache hit rate is governed by receptive-field **radius**, not by
layer count. Those are different quantities, and conflating them was the original error behind
this entire line of investigation.

## Reproducing

```bash
python scripts/depth_accuracy.py      # the truncation table
python scripts/wo_distill_pooled.py   # the identifiable readout refit
python scripts/tension.py             # the headroom/cost correlation
python scripts/depth_sweep.py         # from-scratch training at each depth
```

Every results JSON carries its caveat in a top-level `caveat` field, so the depth-6-training
caveat travels with the numbers rather than living only in this post.
