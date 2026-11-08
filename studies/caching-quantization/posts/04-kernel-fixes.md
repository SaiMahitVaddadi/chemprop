# Two one-line fixes worth 4.6×, and one that looks right and isn't

*Part 4 of a series on caching and quantizing D-MPNN inference.*

Everything so far in this series has been about avoiding work. This post is about doing the same
work faster, and it is the shortest path to a real speedup in Chemprop today: about twenty lines
across three files, plus one runtime flag.

It also contains the clearest cautionary tale I found, because the obvious version of the fix is
wrong — and I got the reason for *that* wrong too, which I'll come back to.

## The line

Here is how Chemprop builds its scatter index, in `nn/message_passing/mixins.py` and seven other
places:

```python
index_torch = bmg.edge_index[1].unsqueeze(1).repeat(1, H.shape[1])
M = torch.zeros(...).scatter_reduce_(0, index_torch, H, reduce="sum", include_self=False)
```

`.repeat` **materialises** that index. At CheMeleon's `d_h=2048` with 13,634 directed bonds, it
allocates and writes a 223 MB **int64** tensor — on every message call, at every depth iteration.
Since int64 is exactly twice fp32, *the index is twice the byte volume of the features it
indexes*, at every `d_h`.

That's the version of the story I believed first, and it's the less interesting half. The bigger
effect is **kernel eligibility**. PyTorch gates its fast GNN scatter paths on the index being
stride-0 along the feature dimension:

- **CPU**: `can_use_expanded_index_path()` in `TensorAdvancedIndexing.cpp` — the code is literally
  comment-headed `// fast paths for GNN usage` — requires `index_strides[0] == 1` and stride-0
  elsewhere. When it fires you get a COO→CSR sort followed by an spmm-reduce: a structurally
  different and much better algorithm. The PR that added it measured `scatter_add_` going from
  37.8 s to 6.0 s.
- **CUDA**: `fast_scatter_kernel_eligible<16>()` requires stride-0 on the feature dim plus 16-byte
  alignment, routing to `tma_scatter_kernel_launch` on sm_90+.

`.repeat` fails that test, so Chemprop is disqualified from both. This is a Chemprop bug, not a
PyTorch limitation — PyG has shipped the correct form for years.

There is a second waste in the same line. `include_self=False` calls
`scatter_reduce_exclude_self_helper()`, which performs **an entire extra scatter traversal**
writing the identity element — to accomplish what `torch.zeros()` already did. Into a fresh zero
buffer it is mathematically redundant. Measured in isolation at `d_h=2048`: **50.3 ms with
`include_self=False` versus 24.7 ms with `include_self=True`.**

Both problems disappear with one substitution: **`index_add_` with a 1-D index.** It takes the
index natively, so materialising one is impossible to write by accident.

## What it's worth

Full forward, 500 molecules (6,635 atoms, 13,634 directed bonds), verified against an fp64
reference:

| variant | MPS `d_h=300 d=3` | MPS `d_h=2048 d=6` | rel err vs fp64 |
|---|---|---|---|
| stock (`.repeat`) | 24.54 ms — 1.00× | 654.6 ms — 1.00× | 5.7e-7 / 7.3e-7 |
| `index_add_` | 12.62 ms — **1.95×** | 302.2 ms — **2.17×** | 5.7e-7 / 6.5e-7 |
| **`index_add_` + `torch.compile(dynamic=True)`** | **5.37 ms — 4.57×** | **178.1 ms — 3.68×** | 6.0e-7 / 6.7e-7 |

On CPU the patch alone gives **2.15×** at the default size and 1.31× at CheMeleon dimensions.

The equivalence is better than "close enough": `index_add_` is **bitwise identical** to the
current output on CPU at both model sizes, including the empty-group edge case in
`agg.py` (a group with no members stays zero under both forms). On MPS the variants differ by
~3e-6, but that's fp32 reduction-order noise — MPS scatter is not bitwise reproducible in *any*
variant, which is worth knowing independently.

The pattern turned out to be far more widespread than I first thought. Nine sites across five
files: `mixins.py` ×2, `base.py`, `mol_atom_bond.py`, `agg.py` ×3 (`MeanAggregation`,
`SumAggregation`, `AttentiveAggregation`), and `ffn.py` ×2 in the constrained-prediction path.

## The fix that looks right

The naive repair is `.expand` instead of `.repeat` — a zero-copy view, same semantics, one word
changed. Index construction goes from 14.477 ms to 0.002 ms, a factor of seven thousand.

It makes the large model **slower**:

| | CPU `d_h=300 d=3` | CPU `d_h=2048 d=6` |
|---|---|---|
| `.repeat` | 91.7 ms | 1279.9 ms |
| `.expand` | 61.1 ms (1.50×) | **1459.9 ms (0.88×)** |
| `index_add_` | **42.5 ms (2.16×)** | **1094.1 ms (1.17×)** |

A contiguous materialised index gives `scatter_reduce_` a better memory access pattern than a
broadcast view does, and at `d_h=2048` that outweighs the 14 ms you saved building it. The
cheapest-looking fix regresses the model you most wanted to speed up.

**Here I need to correct myself.** I also warned that `.expand` was a *correctness* hazard, citing
pytorch#203078 — an open silent wrong-answer bug for indices with strides `(1,0)` where
`self.size(1) > index.size(1)`. On closer inspection that trigger is **structurally unreachable**
in Chemprop's `message()`, because the scatter target and the expanded index both have width
`d_h`, so the sizes always match. The bug is real and worth knowing about if you write this
pattern elsewhere, but it does not fire here. `.expand` fails on performance grounds alone, which
is sufficient — I just had the second reason wrong.

## The flag

`torch.compile(dynamic=True)` roughly doubles the patch's win again, and both of my predictions
about it were wrong.

I expected a recompile per distinct `(n_atoms, n_bonds)` shape, since molecular batches vary. With
three distinct shape pairs I measured **two graph compilations, not three** — torch 2.11's
`automatic_dynamic_shapes` specialises once and then goes dynamic on its own.

I also expected `dynamic=True` to give back most of the fusion win, which is the usual trade. It
gives back **nothing**: 2.16× with `dynamic=True` against 2.15× for the default and 2.02× for
`mode="reduce-overhead"`, with zero recompiles and a single graph. So the configuration with the
best shape behaviour is also the fastest.

Numerics were checked against eager on every compiled variant — max relative difference 1.2e-6,
no mismatches. That check was not optional: pytorch#201455 reports a `torch.compile` + MPS bug
where a reduction fusing with `index_add` produced results off by **exactly 64×**, on the pattern
"average per-atom channels, then `index_add` atoms into molecules" — structurally Chemprop's
`MeanAggregation`. It did not reproduce on 2.11 (the named versions are 2.14.x), but a compiled
speedup without a correctness check would have been worthless.

## Why I kept calling this memory-bound, and why that was wrong

Three times in this project I described the message-passing loop as memory-bandwidth-bound. The
roofline says otherwise, and the correction is more interesting than the error.

Scaling `d_h` from 300 to 2048 (6.83×), the `W_h` GEMM time scales **50.95×** on CPU — almost
exactly the 46.6× that pure FLOP-bound behaviour predicts — running at **637–969 GFLOP/s** against
a measured machine peak around 830. The GEMM is *compute-bound*, with arithmetic intensity of 476
FLOP/byte at `d_h=2048`.

The message step scales 6.38× (byte-bound predicts 6.83×), so it is clearly the bandwidth-ish
half. But it achieves only **2.9–8.1 GB/s on CPU and 15–45 GB/s on MPS** — nowhere near memory
bandwidth. It is **latency- and gather-bound**, not bandwidth-bound. The distinction matters
because it changes what fixes help: reducing bytes moved is worth less than reducing the number of
irregular accesses and kernel launches.

Which is consistent with a wider signal. A Chemprop maintainer notes in discussion #1019 that low
GPU utilisation is expected because "the number of add operations for each batch of graphs is
inconsistent between batches"; `pyg_chemprop` measured 15 s → 7 s per epoch against v1;
TorchMD-Net 2.0 got 2–10× from CUDA graphs alone on molecules under 300 atoms. Small sparse graphs
are an overhead-bound regime, and `torch.compile` beating a 2× traffic reduction is exactly what
that predicts.

One corollary worth stating because it sinks an obvious idea: **low precision does nothing for the
scatter on MPS.** `index_add_` at `d_h=2048` measures 25.53 ms in fp32, 26.01 in bf16, 26.64 in
fp16 — slightly *slower*. It takes the compare-and-swap path, since the upcast only triggers at
`indices/size ≥ 16` and molecules sit at 2.05. Halving the dtype halves nothing. Low precision
only starts paying once the scatter is gone.

## Two more silent-wrong-answer bugs, found the hard way

Both are new, neither is #203078, and both cost real implementation work:

1. **`torch.index_select(..., out=<view with nonzero storage offset>)` is silently wrong on MPS,
   torch 2.11.** The offset is ignored. Measured **99% relative error** on the real message step.
   Offset 0 works; offset ≥ 1 fails; CPU is fine; `index_copy_` and `view.copy_` are fine. This
   was the *fastest* formulation of the next post's kernel — 5.37 ms versus 19.18 ms for the safe
   form, a 3.57× difference — and it had to be abandoned.
2. **`index_copy_` is ~200× slower than contiguous slice writes on MPS**: 295.5 ms versus 1.41 ms
   at `d_h=300`; 2006 ms versus 4.69 ms at `d_h=2048`.

The general lesson, which I'd now apply reflexively: on a less-travelled backend, verify numerics
against a reference *before* you believe a timing. Two of the four fastest formulations I tried
were silently incorrect.

## Measurement hygiene

This machine ran at load average 85–110 on 10 cores from unrelated processes for the whole
session. All CPU numbers come from interleaved round-robin races — every variant timed once per
round, minimum over 7–21 rounds — so **ratios are sound but CPU absolutes drift by about 2×**. MPS
numbers were stable to under 10% spread. I'd treat the MPS table as the reliable one.

One claim I could not verify: that `index_add_` lowering to ONNX `ScatterND` avoids a silent CPU
fallback that `scatter_reduce_`'s `ScatterElements(reduction="add")` triggers on ORT's MLX
execution provider. There's no `onnxruntime` in this environment and I wasn't willing to install
into the system Python to find out.

## What to ship

Patch 1 plus `torch.compile(dynamic=True)`. That's 4.57× at the default model size on MPS, with
numerics verified against fp64 and against eager, no new dependencies, and no export risk if you
keep an uncompiled path. The patch is in `patches/` and applies cleanly.

The next post is about a better kernel than patch 1 — one that is algebraically exact, strictly
more accurate in bf16, and which I am nonetheless not recommending yet.

## Reproducing

```bash
python scripts/bench_scatter.py   # isolated scatter, all variants
python scripts/bench_fwd.py       # end-to-end, both devices and sizes
python scripts/p11.py             # torch.compile recompiles and numerics
python scripts/p5_roofline.py     # the compute-bound / latency-bound split
```
