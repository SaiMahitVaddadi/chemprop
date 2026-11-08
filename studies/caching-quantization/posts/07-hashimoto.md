# Chemprop is already computing the Hashimoto matrix, in the wrong factorization

*Part 7 of a series on caching and quantizing D-MPNN inference. This is the most interesting
result in the series and the one I am least ready to recommend.*

Part 2 established that a D-MPNN's message excludes the reverse edge, so its hidden states are
functions of non-backtracking walk trees. That observation has a name in spectral graph theory,
and once you attach it, a kernel optimization falls out that no GNN kernel paper proposes — for a
reason that turns out to be about molecules specifically.

## The identity

Write `S` for the tail incidence operator, `T` for the head incidence operator, and `τ` for the
edge-reversal involution. Chemprop's message step computes

```
M = B H,        B = S Tᵀ − τ
```

`B` is the **Hashimoto non-backtracking edge-adjacency matrix** — the operator behind the Ihara
zeta function, non-backtracking spectral clustering, and the message rule Chemprop has shipped
since 2019. `B[e, f] = 1` exactly when `dest(f) = src(e)` and `f ≠ rev(e)`.

Chemprop doesn't evaluate `B`. It evaluates the factored form:

```
M = S(Tᵀ H) − τ H        # gather-into-nodes, broadcast back out, subtract the reverse
```

That factorization is the right one for high-degree graphs, where summing once per node and
reusing the sum across many incident edges saves enormous work. For molecules it is the wrong one,
and the crossover is derivable.

Row `e` of `B` has exactly `deg(src(e)) − 1` nonzeros. Averaged over *edges* rather than nodes —
which weights by degree — that's `(mean(d²) − mean(d)) / mean(d)`. Comparing traffic for the two
forms gives a clean condition:

> Explicit `B` wins iff **`mean(d²) < 4·mean(d) + 2`** — degree **< 4.45** for a regular graph.

Molecules sit at **2.09**. Reddit, the benchmark that most GNN kernel papers optimize for, sits at
~492. It misses the crossover by three orders of magnitude, which is why nobody has proposed this:
the entire kernel literature is tuned for the regime where the factored form is correct.

## Measured structure

On 6,253 molecules from Chemprop's own test data, with the predicted histogram from an independent
derivation alongside:

| row nnz | predicted | measured |
|---|---|---|
| 0 | 10.7% | **9.91%** |
| 1 | 43.5% | **43.73%** |
| 2 | 40.7% | **41.57%** |
| 3 | 5.1% | **4.78%** |
| ≥4 | 0.05% | ~0.01% |

Maximum deviation 0.87 percentage points. `nnz(B)/row = 1.413` — **`B` is sparser than a gather**,
with roughly 10% of rows entirely empty (terminal atoms, whose only incoming edge is the one they
must exclude).

This was independently reproduced from the opposite direction. A separate strand of this project
arrived at the same operator from inductive-bias reasoning rather than memory traffic, and measured
`nnz(B)/row = 1.360` on a different molecule set — within 2% — along with 12.7% empty rows and the
identity itself at `max |message(H) − B·H| = 0.000e+00`. It also noted a consequence I'd missed:
**`B` is nilpotent on acyclic molecules**, since a non-backtracking walk on a tree must eventually
run out of edges. More on what that implies in Part 8. The crossover condition holds on **all 9 datasets tested**, including HIV with its
larger fused-ring systems (5.17 against a threshold of 10.55). No outliers.

The bucketed implementation is pleasingly small. Group edges by `nnz`, and you get exactly four
groups — `{0: 1734, 1: 5948, 2: 5268, 3: 684}` on the benchmark batch, with no `nnz ≥ 4` tail at
all. Four gather-and-sum kernels, no padding waste, no sparse library. (This is SELL-C-σ, arrived
at from the chemistry rather than from the format.) The index structure costs **0.148 MB — 0.13%
of `H`** at `d_h=2048`, and building it takes 2.39 ms per 500 molecules once the Python loop is
vectorised out, which belongs in `collate` and dataloader workers anyway.

## It is exact, and that is checkable

The claim is algebraic equality, not approximation, so it deserves a stronger test than fp32
agreement:

- fp64, across three activation distributions including one scaled by 1e4: **max absolute
  difference 1.78e-15, relative 2.4e-16.** Roundoff.
- Independent cross-check against a **densely materialised** `B` on 40 real molecules: **max
  difference exactly 0.0.**
- Full forward against an fp64 reference with real CheMeleon weights: Hashimoto 7.3e-7 versus
  stock 6.7e-7 — same order, no accuracy cost.

## The numerical bonus, which is lovely

The factored form computes, for a degree-2 atom:

```
M_e = (H_f + H_rev) − H_rev = H_f
```

It builds a sum and then subtracts one of that sum's own terms. That's catastrophic cancellation,
and **44.5% of all directed bonds are exactly this case.**

In fp32 it's harmless (1.4e-7). In bf16 it is not: measured **8.2e-3 maximum relative error**,
with median 2.0e-3 concentrated precisely at degree 2.

The Hashimoto form's error at degree 2 is **exactly 0.0** — because that row of `B` has a single
nonzero, so the "sum" is a pure copy with no arithmetic at all. There is nothing to cancel.

One honest deflation here: I had predicted this would be worse than it is, on the assumption that
`|H_f|` and `|H_rev|` would differ by 10–100×. Measured amplification `|N_src| / |M|` at degree 2
is only p50 2.0, p99 4.1, max 6.4. The mechanism is exactly as predicted; the magnitude is at the
low end.

And there's a cheaper fix for the same problem: **bf16 GEMM with an fp32 reduction makes the
factored form bit-for-bit identical to Hashimoto** at every layer and every degree. If bf16
message passing is the goal, two lines get you the numerics without the kernel.

## Why I'm not recommending it yet

Three things came out against it, and they are the reason this post exists rather than a victory
lap.

**1. The traffic claim was overstated, by my own accounting error.** My arithmetic put explicit `B`
at 2.06× less traffic than the `index_add_` form. It didn't count `torch.cat` — and assembling `M`
from four buckets via `cat` materialises it twice. In `d_h`-vector units the real message step is
`nnz + 3·2m = 59,438`, not `nnz + 2m = 32,170`. **That drops the traffic advantage from 2.11× to
1.14×.** Measured, a bare `cat` of the four parts costs 5.78 ms on CPU at `d_h=2048` — **30–35% of
the entire message step.** Slice writes instead of `cat` recover 1.68× on CPU and are neutral on
MPS, but see point 3.

**2. On CPU the wall-clock win is near zero.** Full forward against the `index_add_` baseline:
**1.11× at `d_h=300`, 1.00× at `d_h=2048`.** Against *stock* Chemprop it's 2.38× and 1.31×, but
almost all of that is the Part 4 patch, which is one line instead of 150.

On MPS it does much better — **2.92× and 3.08× against stock**, 1.52× and 1.43× against
`index_add_`. But note what that implies given only a 1.14× traffic advantage: **the MPS win isn't
about traffic at all.** It's that MPS `index_add_` is slow (a compare-and-swap loop at these
degrees) while gathers are fast. Right answer, wrong reason — and a reason that won't transfer to
a backend with a good scatter.

**3. It blocks the bigger win.** `torch.compile(dynamic=True)` on top of the Part 4 patch gives
**4.57×** at the default model size, beating the Hashimoto kernel's 2.92×. And the two don't stack:
compiling the bucketed form dies in an Inductor Metal codegen bug (`variable 'tmp_scoped_2'
declared with deduced type 'auto' cannot appear in its own initializer`), reproducibly, at both
model sizes. So today they are mutually exclusive and the simpler option wins.

The fastest formulation I wrote was also abandoned for correctness:
`torch.index_select(..., out=<view with nonzero storage offset>)` is **silently wrong on MPS in
torch 2.11** — the offset is ignored, giving 99% relative error. It was 3.57× faster than the safe
form. The shipped patch uses `view.copy_` with a comment warning against the "optimisation."

## Two literatures that nearly met

The satisfying part of this result is that two independent threads converge on the same operator
from opposite directions, and neither cites the other.

**NBA-GNN** (NeurIPS 2023) defines exactly Chemprop's update rule, including the `\{i}` exclusion,
and identifies it with the non-backtracking matrix `B` — motivated by over-squashing and walk
redundancy, with no mention of Hashimoto, D-MPNN, or Chemprop. Separately, Gasteiger et al.
(2021) state the other half: *"we can transform any MPNN to a directed MPNN simply by executing it
on the directed line graph."* Compose the two and you get the identity in this post. As far as I
can tell **no single paper states it**.

It also explains a result that otherwise looks like a counterexample. NBA-GNN reports its
non-backtracking variant costing 3–4× runtime and 6–7× memory on Cora. Cora's mean degree is
around 4 — just past the 4.45 crossover. The same operator that is a win at degree 2 is a loss at
degree 4, which is the entire point and is invisible unless you work out where the boundary is.

The broader lesson generalises past chemistry: **the right factorization of a sparse operator
depends on the degree distribution, and the GNN kernel literature has a strong implicit prior
toward high-degree graphs.** Molecules are the opposite corner of that design space. Anyone
working on small sparse graphs — not just molecular ones — should check which side of their
crossover they're on before adopting a technique benchmarked on Reddit.

## Status

The structure is measured and the exactness is proven. The *speedup* is real on MPS, negligible on
CPU, and currently mutually exclusive with a better one-line alternative. The patch exists
(`patches/`, opt-in, adds one file and changes no existing behaviour) and I'd ship it only if the
Inductor bug is fixed, or on a backend where the scatter is the bottleneck, or if bf16 message
passing becomes the default — in which case its exactness at degree 2 starts to matter on its own.

One further negative worth recording: I also built the size-bucketed dense block-diagonal `bmm`
formulation, on the theory that converting all irregular access into pure `bmm` would win on MPS
where scatter is weak and there is no sparse support whatsoever. **It never wins.** It beats
`index_add_` at `d_h=2048` (2.58× versus 2.17× against stock) and loses to the Hashimoto kernel
everywhere, while costing +8–11% FLOPs at `d_h=2048` and +35–43% at `d_h=300`. Bucketing *is* what
makes it viable at all — naive padding to batch maximum inflates the main GEMM 10.2× on a pooled
17,176-molecule distribution — but viable isn't the same as better.

## Reproducing

```bash
python experiments/hashimoto_bench.py   # structure, exactness, timings
python scripts/p67b.py                  # bf16 error stratified by degree
python scripts/p9_bmm.py                # the bmm formulation that doesn't win
```
