# Experimental & SOTA kernel designs for D-MPNN message passing — research brief

**Scope.** External literature + kernel-ecosystem survey for making Chemprop v2.3.1's bond-based
D-MPNN faster and more numerically accurate. Target shapes: ~6,600 atoms / ~13,600 directed bonds
per batch of 500 drug-like molecules (~13 atoms, ~27 directed bonds/mol), `d_h` 300–2048, `depth`
3–6. Deliberately *excludes* the hands-on local micro-benchmarking (`.repeat` vs `.expand`,
`index_add_` vs `scatter_reduce_`, `torch.compile`, MPS, sparse-matmul timings) being done
concurrently; where a finding here predicts something that benchmarking can check, it is called out
under **"Falsifiable predictions"**.

**Code read:** `chemprop/nn/message_passing/base.py`, `chemprop/nn/message_passing/mixins.py`,
`chemprop/nn/agg.py`.

---

## 0. The single most important caveat in this whole literature

Almost every GNN-kernel paper (GE-SpMM, FeatGraph, MaxK-GNN, DF-GNN, TLPGNN, GeoT, dgNN, Graphiler)
benchmarks on **one huge graph**: Cora, Citeseer, Pubmed, Reddit/Reddit2, ogbn-products,
ogbn-papers100M, Yelp, AmazonProducts. Mean degrees run from ~4 (Cora) to ~492 (Reddit) to ~50+
(ogbn-products). Their optimizations are overwhelmingly about **load-balancing long rows, reusing a
gathered neighbour across many destinations, and tiling a feature dimension of 16–128**.

Chemprop's regime is the opposite corner of the design space:

| | Reddit | ogbn-products | **Chemprop batch** |
|---|---|---|---|
| nodes | 233 K | 2.4 M | **6.6 K** (500 disconnected components) |
| mean degree | ~492 | ~50 | **2.09** (measured, below) |
| feature dim | 602 | 100 | **300–2048** |
| connectivity | one graph | one graph | **500 tiny components** |
| depth | 2–3 | 2–3 | **3–6, weights shared across depth** |

Consequences that recur throughout this brief:

1. **Degree ~2 inverts the usual algorithmic choice.** The "gather once into nodes, then broadcast"
   factorisation that is mandatory on Reddit is *pessimal* on molecules (§2). This is the central
   finding of this brief.
2. **Load-balancing papers have nothing to offer.** Row lengths are 1–4. There is no long-tail row
   to balance. MaxK-GNN explicitly reports its wins only "for graphs with average degree greater
   than 50" — i.e. never for us.
3. **`d_h` 300–2048 is far outside the 16–128 feature dims these kernels tune for.** GeoT
   explicitly notes it is tuned for "practical GNN feature sizes (16–64)". Feature-dimension tiling
   assumptions do not carry over.
4. **The relevant precedent literature is molecular-simulation GNNs and LLM decode, not graph
   learning.** TorchMD-Net 2.0 (small molecules) and the "weights-stay" megakernel line (same
   projection reused across serial steps) match Chemprop's structure far better than any SpMM paper.
5. **Two partial exceptions worth reading.** **GeoT**'s dataset table includes **Cora (2,708 nodes /
   10,556 edges) and Citeseer (3,327 / 9,104)** — almost exactly a Chemprop batch's magnitude — and
   PyG's CPU-hotspot roadmap has a **GAT + CiteSeer** row. Those are the only published numbers at
   our scale. (Even they are *one connected graph with hub nodes*, versus our hundreds of disjoint
   ≤4-degree components, so contention is lower still for us.) And **Degree-Quant's ZINC table** is
   genuine molecular-graph data.
6. **Nobody has published anything at `d_h` 300–2048 on ~13 K edges.** Every kernel in §3.2 is tuned
   for feature dims 16–128; every low-precision result is for `M ≳ 128,000`. **The crossover between
   "scatter dominates" and "the `d_h × d_h` GEMM dominates" is unmeasured in the literature, and it
   is the single most valuable thing the local benchmarking could establish** — because it decides
   whether to invest in scatter kernels at all. (§3.1 and §5.6 both predict, from independent
   arithmetic, that the GEMM wins decisively once the index bloat is removed.)

---

## 1. Executive summary — best payoff/effort ratio

Ranked. Effort is my estimate for a competent PyTorch/Triton engineer.

### ⚠️ Read this framing first: the workload is probably *launch-overhead* bound, which caps every traffic argument below

Three independent, carefully-measured ecosystem data points converge:

- **pyg-lib #213** got **5.8× (0.29 ms → 0.05 ms, RTX 3090)** on `segment_matmul` purely by replacing
  7 pageable H2D argument copies with one pinned copy — **no kernel change at all**.
- **pytorch/pytorch #203900** (ROCm row scatter-add override) set a **256 MiB work floor** before
  dispatching its custom kernel and **refuses fp32 at every size**, because below that
  "the native route's host cost" dominates. Chemprop's `H` at `d_h=300` is **16 MB — 16× below
  that floor**.
- **ezyang on PyG GCN** (A100): static Inductor **+CUDA Graphs 1.267×**, static without CUDA Graphs
  1.102×, **dynamic shapes 1.021×** — then: *"26% speedup is cool, but in the overhead bound regime
  I would much rather see 2× or faster."*
- Corroborated inside Chemprop: in **discussion #1019** a maintainer (JacksonBurns) states that low
  GPU utilisation is *expected*, because **"Chemprop's update operation doesn't work well on GPUs
  (the number of add operations for each batch of graphs is inconsistent between batches, since the
  number of edges varies), but v2 uses `scatter` which makes it less inefficient"** — "less
  inefficient", not efficient.
- And quantified: **`itakigawa/pyg_chemprop` measured 15 s → 7 s per epoch (2.1×)** vs Chemprop v1
  on ogbg-molhiv (32,901 molecules, batch 50, A100), i.e. **22.8 ms/step for v1 vs 10.6 ms/step**.
  At 6,600 nodes / 13,600 edges, 22.8 ms/step is ~100× more than the kernels should need.

**Implication for the ranking.** My traffic arithmetic (§3.1) is an *upper bound* on achievable
speedup. Where the kernel is launch-bound, a 2× traffic reduction buys close to nothing, and a
reformulation that *adds* kernel launches can be net negative. So the sequencing is:

> **First make the launches cheap and few (and the shapes static). Then the traffic reductions
> start paying, and only then does low precision matter.**

Concretely that reorders things: **#2 (free, also unlocks faster kernels) → #3 (launch count /
static shapes) → #1 (algorithmic traffic win, best delivered as *one* fused kernel or under
`torch.compile`, not as 7 eager ops) → #4 → #5.** The payoff/effort ordering below is kept because
#1 is the most *novel* and most *durable* finding, but the *implementation* order should be
#2, #3, then #1.

### 📌 The one change to make first — three independent wins from one edit

```python
# chemprop/nn/message_passing/mixins.py  and  base.py
# BEFORE
index_torch = bmg.edge_index[1].unsqueeze(1).repeat(1, H.shape[1])
M = torch.zeros(n, d_h, dtype=H.dtype, device=H.device).scatter_reduce_(
        0, index_torch, H, reduce="sum", include_self=False)
# AFTER
M = torch.zeros(n, d_h, dtype=H.dtype, device=H.device).index_add_(0, bmg.edge_index[1], H)
```

1. **Eliminates a 223 MB int64 index tensor per call** (at `d_h=2048`) — 1.80× message-step traffic,
   `d_h`-independent (§3.1).
2. **Unlocks PyTorch's GNN fast paths** that `.repeat` disqualifies — CPU `can_use_expanded_index_path`
   (COO→CSR + spmm-reduce) and CUDA `fast_scatter_kernel_eligible` (TMA / vectorized scatter) — and
   **drops the redundant extra scatter traversal** that `include_self=False` forces (§1 #2).
3. **Improves ONNX export**: `index_add_` decomposes to `index_put_(accumulate=True)` → **`ScatterND`**,
   which the ORT **MLX execution provider GPU-accelerates on Apple Silicon** (and which supports
   fp16/bf16), whereas `scatter_reduce_` → **`ScatterElements(reduction="add")`** is *exactly* the one
   form the MLX EP refuses, silently falling back to the CPU provider at every depth step (§9.1).

🛑 **Do not use `.expand` for the index instead** — pytorch#203078 (open, repros on 2.9/2.15/2.16-dev)
is a **silent wrong-answer bug** for precisely this shape.

### #1 — Explicit Hashimoto (non-backtracking) operator instead of the `Σ − H_rev` factorisation
**Payoff: ~2.1× less message-step memory traffic *on top of* the index fix, plus exact elimination
of the cancellation hazard, plus atomics-free determinism. Effort: low (pure PyTorch) to medium
(custom kernel). Export-safe.**

Chemprop's `message()` computes `M = B H` where `B = S Tᵀ − τ` is *literally* the Hashimoto
non-backtracking edge-adjacency matrix (§2). Today it is evaluated in the factored form
`S(Tᵀ H) − τH`, which is the right factorisation for high-degree graphs and the **wrong** one for
molecules. Measured on 6,253 molecules from Chemprop's own `tests/data`:

- mean atom degree **2.09**, mean(d²) **5.09**
- `nnz(B)/row` = **1.384** (i.e. B is *sparser than a gather*; 10.7% of rows are empty)
- explicit-B traffic / factored traffic = **1 / 2.06**
- crossover: explicit-B wins iff `mean(d²) < 4·mean(d) + 2`; for a regular graph that is
  **degree < 4.45**. Molecules sit at 2.09. Reddit misses it by ~3 orders of magnitude.

It is **algebraically exact** (not an approximation) and **strictly more accurate** in floating
point, because it never forms `Σ_all` only to subtract one of its own terms.

### #2 — Kill the `.repeat`'d int64 index — and this is *not* mainly about bytes
**Payoff: 1.80× traffic *plus* it unlocks entirely different, much faster PyTorch kernels on both
CPU and CUDA. Effort: trivial. Export-safe. Being benchmarked locally — but the mechanism below is
probably not what the benchmark assumes.**

`index_torch = bmg.edge_index[1].unsqueeze(1).repeat(1, H.shape[1])` materialises a `(2m, d_h)`
**int64** tensor — 223 MB at `d_h=2048`, written *and* read every message call, against `H` itself
at 111 MB (int64 is exactly 2× fp32, so the index is **2× the data volume of the features it
indexes**, at every `d_h`). That alone is the 1.80× traffic figure, and it is why the loop profiles
as memory-bound even though the `W_h` GEMM has arithmetic intensity 476 FLOP/byte (§3.1).

**But the bigger effect is kernel eligibility.** PyTorch gates its fast scatter paths on the index
being *stride-0* on the feature dimension — i.e. `.expand`, not `.repeat`:

- **CPU:** `can_use_expanded_index_path()` in `aten/src/ATen/native/TensorAdvancedIndexing.cpp`
  (literally comment-headed `// fast paths for GNN usage`) requires `index_strides[0] == 1` and
  stride-0 elsewhere, plus contiguous `src`/`self`, floating dtype, OpenMP, and
  `index.numel()/index.size(0) >= 16`. When it fires you get `scatter_reduce_expanded_index_stub`
  → **a COO→CSR sort followed by an spmm-reduce** — a structurally different, much better
  algorithm. The PR that added it (#82703) measured `aten::scatter_add_` going
  **37.797 s → 5.989 s (6.3×)** for SAGE (49.0% → 13.2% of self-CPU time) — ⚠️ on Reddit.
- **CUDA:** `fast_scatter_kernel_eligible<16>()` in `aten/src/ATen/native/cuda/IndexKernelUtils.h`
  requires `iter.strides(2)[0] == 0` (stride-0 index on the feature dim), 16-byte alignment of
  pointers and row stride, and `slice_size_bytes >= kFastScatterMinSliceBytes` (**= 256**). When
  eligible it routes to **`tma_scatter_kernel_launch`** (CUDA ≥ 12.8 and sm_90+) or
  **`vectorized_scatter_kernel_launch`**; otherwise it falls back to the generic path.

`.repeat` fails the stride-0 test and therefore **disqualifies the fast path on both backends**.
This is a Chemprop bug, not a PyTorch limitation, and PyG has shipped the correct form for years
(`torch_geometric/utils/_scatter.py` uses `src.view(size).expand_as(ref)`).

> ### 🛑 **But do NOT fix it with `.expand` — use `index_add_` with a 1-D index.**
>
> **`pytorch/pytorch#203078` (open, filed 2026-11-02) is a live silent-wrong-answer bug:**
> `scatter_add`/`scatter_reduce` **write to the wrong rows** when `index` is an expanded view with
> strides `(1, 0)` and `self.size(1) > index.size(1)`. It reproduces on **2.9, 2.15 and
> 2.16-dev**. Chemprop's call is exactly that shape.
>
> `index_add_(0, bmg.edge_index[1], H)` is semantically identical here, takes a 1-D `[2m]` int64
> index (109 KB), makes index materialisation *impossible to write by accident*, and avoids the
> bug entirely. **This matters for the concurrent local benchmarking, which is comparing `.repeat`
> vs `.expand` — a `.expand` win could be a wrong-answer win. Add a numeric equivalence check
> against the `.repeat` result before trusting any `.expand` timing.**
>
> It also wins on export (§9): `index_add_` lowers to **`ScatterND`**, which is GPU-accelerated on
> Apple Silicon, while `scatter_reduce_` lowers to **`ScatterElements`**, which is not. **One edit,
> three independent wins.**

**Two more free fixes in the same line of code:**

1. **Drop `include_self=False`.** `scatter_impl` calls `scatter_reduce_exclude_self_helper()`, which
   performs **an entire extra scatter traversal** writing the identity (0 for SUM) — to accomplish
   what `torch.zeros()` already did. Use `torch.zeros(...).scatter_add_(0, idx_expanded, H)` or
   `index_add_`. Identical semantics into a fresh zero buffer, one fewer full pass. PyG's
   `reduce='sum'` path never calls `scatter_reduce_` at all, for exactly this reason.
2. **Align `d_h`.** The CUDA fast path needs `row_bytes % 16 == 0` and `row_bytes ≥ 256`:
   - fp32, `d_h=300` → 1200 B ✓ (and ≥256 ✓) → **eligible**
   - fp32, `d_h=2048` → 8192 B ✓ → eligible
   - **bf16/fp16, `d_h=300` → 600 B; 600 % 16 = 8 ✗ → INELIGIBLE even with `.expand`.**
     **If Chemprop ever runs bf16 message passing, `d_h` must be a multiple of 8 — 300 → 304.**
     This is a one-line default change with a real performance cliff behind it.

**And `index_add_` vs `scatter_add_` is now a non-question:** on CUDA ≥ 12.8,
`index_add_(0, idx, H)` dispatches *into* `scatter_add_` (pytorch/pytorch #189506, #88729). They
converge on the same kernel, so prefer `index_add_` — it makes index materialisation
*impossible to write by accident*.

### #3 — Fuse the depth loop into one persistent kernel ("weights that stay")
**Payoff: plausibly 1.5–3× at `d_h=300` where launch overhead dominates; 2–10× is the measured
range for the closest published analogue. Effort: high. Breaks export (needs a fallback path).**

`W_h` is reused for all `depth−1` iterations. This is structurally identical to the speculative-
decoding megakernel that keeps one vocabulary projection resident across serial draft steps
(arXiv 2611.00506, up to **11.6× head-level on B200/H100**). At `d_h=300` the *entire* working set
(`H` 16 MB + node buffer 8 MB + `W_h` 0.36 MB) fits in an H100's 50 MB L2 — a depth-loop megakernel
could run depth 3–6 with essentially no HBM traffic after the first load. The strongest
*same-domain* evidence: **TorchMD-Net 2.0 measured 2–10× from CUDA graphs alone on molecules under
~300 atoms, and 2–3× from `torch.compile`**, explicitly because "kernel launching overhead [is] a
performance bottleneck" for small molecular graphs (RTX 4090, PyTorch 2.1).

**Cheap 80% version: CUDA graphs / size-bucketed static shapes.** No custom kernel; just bucket
`n_atoms`/`n_bonds` to powers of two, pad, and capture. This is exactly what TorchMD-Net did and it
is where their 2–10× came from.

### #4 — Numerics: stop computing `Σ − one_term` in reduced precision
**Payoff: removes a real bf16/fp16 correctness trap; no cost. Effort: trivial if you do #1 anyway.**

For a degree-2 atom, `M_e = (H_f + H_rev) − H_rev = H_f` is computed by catastrophic cancellation.
Relative error ≈ `ε · (|H_f| + |H_rev|)/|H_f|`. In fp32 (ε≈6e-8) a 10× magnitude ratio gives ~1e-6 —
harmless. In **bf16 (ε≈4e-3) the same ratio gives ~4% error on the message**. 43% of all directed
bonds in real molecules are exactly this degree-2 case. The Hashimoto form (#1) removes the
subtraction entirely. See §6.

### #5 — Size-bucketed dense block-diagonal `bmm` (the "Apple Silicon escape hatch")
**Payoff: converts all irregular gather/scatter into pure `bmm` for +2.6% FLOPs at `d_h=2048`.
Effort: medium (touches the dataloader). Fully ONNX/TorchScript-safe, deterministic, and the best
option on MPS/Metal where scatter is weak and `bmm` is strong.**

`B` is block-diagonal over molecules. Pad to a per-bucket `E_max` and run
`(b, E_max, E_max) @ (b, E_max, d_h)`. With molecules sorted into size buckets (pad to 1.15× mean,
`E_max≈47`): the `B`-bmm costs **4.46 GFLOP vs the 170 GFLOP `W_h` GEMM — 2.6% overhead** at
`d_h=2048`, and padding inflates the `W_h` GEMM only 1.15×. Details and the honest failure mode
(unbucketed padding is fatal) in §4.

---

## 1b. Master table of techniques

Speedups in **bold** are measured and reported by the cited source (shapes/hardware noted);
everything else is my arithmetic or an estimate and is labelled. "Fits many-tiny-graphs" is the
column that kills most of the GNN-kernel literature.

| technique | reported speedup / on what shapes | maturity | hardware required | many-tiny-graphs? | ONNX/TorchScript risk |
|---|---|---|---|---|---|
| **`index_add_` + 1-D index** (§1 #2) | **1.80× traffic** (my arithmetic, `d_h`-independent) **plus** it unlocks CPU `can_use_expanded_index_path` (COO→CSR+spmm; **6.3×** on `scatter_add_` in the PR that added it, ⚠️ Reddit) and CUDA TMA/vectorized scatter | trivial, pure PyTorch | any | ✅ ideal | **none — and it *improves* export**: lowers to `ScatterND`, which the ORT **MLX EP accelerates** and `ScatterElements` is not |
| **Drop `include_self=False`** (§1 #2) | removes a whole extra scatter traversal in eager **and** a whole extra `ScatterElements` in the ONNX graph | trivial | any | ✅ | none |
| **Pad `d_h` to a multiple of 8** | bf16 at `d_h=300` → 600 B row, `%16≠0` → **no fast path at all**; 304 fixes it | trivial | CUDA (alignment gate) | ✅ | none |
| ❌ **`.expand`'d index** | — | — | — | — | 🛑 **pytorch#203078: silent wrong answer on CPU for this exact shape, open, repros on 2.9/2.15/2.16-dev** |
| **Explicit Hashimoto `B`, bucketed `index_select`+`add`** (§2.6a) | **2.06×** traffic over the index fix, **3.72×** over current (my arithmetic from measured `nnz(B)`) | low effort, pure PyTorch | any (CPU/MPS/CUDA) | ✅ **only works because degree ≈ 2** | **none** (`Gather`+`Add`+`Concat`) |
| Explicit Hashimoto, custom CSR/Triton/Metal kernel (§2.6b) | same traffic, fewer launches; est. extra 1.1–1.4× | medium | per-backend kernel | ✅ | **breaks** — needs custom op + decomposition fallback |
| Fold `B` into GEMM prologue (§2.6c) | removes one full `H` round-trip; est. 1.2–1.5× | high | CUDA (CUTLASS ≥2.9 / CuTe DSL) | ✅ | **breaks** |
| **Dense block-diagonal `bmm`, size-bucketed** (§4.2) | **+2.6% FLOPs** at `d_h=2048` (+17.9% at 300) to delete all irregular access; speedup unvalidated | medium (touches dataloader) | any; **best option on MPS** | ✅ if and only if size-bucketed | **none** (`bmm`) |
| **CUDA graphs over bucketed static shapes** | **2–10× for molecules ≤300 atoms** (TorchMD-Net 2.0, RTX 4090) | production | CUDA | ✅ **best-matched precedent** | none (runtime wrapper; keep an uncaptured path) |
| **`torch.compile`** | **2–3×** (TorchMD-Net 2.0, same regime) | production | any backend | ✅ but recompiles on shape change → bucket or `dynamic=True` | compiled artifact doesn't export; `torch.export` path separate |
| Depth-loop persistent megakernel (§3.3) | **up to 11.6×** head-level for the structurally identical "same projection reused across serial steps" problem (H100/B200); general megakernels **up to 1.7× e2e, 10× vs native PyTorch** | research / bespoke | CUDA (Hopper+) | ✅ in principle | **breaks** |
| **GeoT** tuned segment reduction | **1.68× geomean vs `scatter_reduce`, 1.28× vs `segment_coo`** (A100/H100/3090Ti, Citeseer→Reddit2) | **public code** | CUDA | ⚠️ needs sorted indices; tuned for feature dim 16–64, not 2048 | breaks (custom op) |
| GE-SpMM | **1.41× vs cuSPARSE, 2.39–6.15× vs DGL** | research | CUDA | ❌ gains are load-balance + neighbour-reuse | breaks |
| FeatGraph | **32× CPU / 7× GPU e2e** | TVM, archived | CPU+CUDA | ❌ (idea transfers, code doesn't) | breaks |
| DF-GNN | **7.0× vs DGL sparse, 2.16× avg e2e** | PyTorch-integrated research | CUDA | ❌ attention GNNs only | breaks |
| TLPGNN | **8.6–12.9× avg** | research | CUDA | ❌ | breaks |
| MaxK-GNN | **2.9× vs cuSPARSE, 3.22–4.24× e2e on Reddit** | public | CUDA | ❌ paper states "avg degree > 50" | breaks + changes the model |
| dgNN | **−45.77% (slower)** per GNNBench | research | CUDA | ❌ | n/a |
| `pyg-lib` `segment_matmul`/`grouped_matmul` | **up to 117.5×, 13.65× avg vs CUTLASS** (FASTEN, A100/4090/GH200) — *for per-relation weights* | public, maintained | CUDA | ❌ **no groups exist** — Chemprop shares one `W_h` | breaks |
| Blocked-ELL SpMM | **350× *slower* than CSR** on an unstructured 355 K-nnz matrix | cuSPARSE | CUDA | ❌ `B` has no dense blocks | n/a |
| SELL-C-σ | up to **1.47×** on a well-sorting stencil | cuSPARSE (SpMV) / portable | any | ✅ — and `B` sorts perfectly (§2.7) | none, if expressed as bucketed gathers |
| **bf16/fp16 `H` + fp32 accumulate** | **2.00× ceiling on the streamed tensor**; standard practice (NVIDIA/TPU); fp32 accumulate is free on tensor cores | production | **everywhere incl. MPS/MLX**; no alignment or `d_h` constraints | ✅ | ⚠️ **ORT CPU EP hard-errors on fp16/bf16 scatter-with-reduction**; MLX EP supports bf16 via `ScatterND` |
| MXFP8 `H` | 3.88× byte ceiling; **observed 1.81× vs 1.97× roofline** (torchao, B200); real MXFP8/BF16 ≈ **1.8×** (Cursor, 1000 W B200) | prototype (torchao), TE on sm_100 only | **Blackwell DC only**; block 32, UE8M0 scale, **FP32 accumulator fixed** | ⚠️ **`d_h=300` is categorically excluded (`K%32`, `N%16`)**; padding 300→384 wastes 28% → net ~1.56×, a wash | ❌ |
| NVFP4 `H` | 7.11× byte ceiling; **observed 2.98× vs 3.82× roofline**; **1.739× prefill but 1.058× decode** (i.e. 6% when memory-bound) | prototype | Blackwell; block 16, UE4M3 | ⚠️ same `d_h` exclusion | ❌ |
| fp8 per-tensor GEMM on **Hopper** | 2.0× datasheet, but **accumulates in FP22 (13 mantissa bits), ~2% worst-case error** — needs DeepSeek's "promote to CUDA cores" every 128 K | production (TE) | H100/H200 | ✅ (K is short, so the trap is mild) | ❌ |
| Re-quantizing `H` each depth step | **−40% of matmul time** (−76% with backward transpose-quant) at *large* LLM shapes; **worse at ours** | — | — | ❌ first-order cost | — |
| **INT8 weights+activations** | **ZINC molecular: free or BETTER than fp32** — GCN 0.469→**0.434**, GIN 0.414→**0.357** (Degree-Quant, measured); J.Cheminform W8A8 **+0.8…+1.4%** | QAT, published | any int8 path; **INT8 ≈2× on Apple M5+** | ✅ **molecular graphs lack the high-degree error mechanism** | int8 QDQ is well supported |
| Low-precision storage of `H` (int4 activations) | +1.7% RMSE (companion); `H` is **6.6× (d_h=2048) to 45× (d_h=300)** the bytes of `W_h` → the right tensor to compress | — | — | ✅ | — |
| **Per-tensor int4 `W_h`** | **+28% RMSE (companion); +25.4% W4A4 GCN/ESOL independently (J.Cheminform)** — depth-compounding, §5.1 | — | — | ❌ **do not** | — |
| Block-scaled (NVFP4-granularity) `W_h` | **untested** — MR-GPTQ shows *scale granularity*, not bit width, drives FP4 error; worth retesting before writing off the weight side | — | Blackwell | ? | — |
| LoopQ "loop-aware activation scaling" (shared `W_h`, one scale per depth step) | recovered **23.6% → 58.96%** (FP 64.1%) on a 4-loop W4A4 model | published, trivially portable | any | ✅ **`depth ≤ 6` scalars — nearly free** | none |
| fp8 / MX on **Apple Silicon** | **1.007–1.15× vs fp16 on M5/M5 Max** — *"FP8 has no compute advantage on this chip"*; `matmul2d` is emulated for fp8 at **0.94× fp16** (Rigel, M4 Max) | PyTorch 2.15 has the dtype, no perf claim | **M5/A19 only have neural accelerators at all** | ❌ **storage only** | ❌ |
| MLX `quantized_matmul` 4-bit | **14–41% SLOWER** than dequant+bf16 at `M > 512` (MLX#4621); our `M=13,600` | shipped | Apple | ❌ | — |
| `torch._segment_reduce(offsets=ptr)` | untested at our shapes; GeoT measured `segment_coo` ≈ **1.31×** over `scatter_reduce` incl. Cora/Citeseer (our magnitude) | **native, zero new deps**; private API; **MPS forward still unmerged** (#197014) | CUDA/CPU | ⚠️ needs sorted indices; mean group 2.06 leaves little contention to avoid | breaks (private API) |
| ❌ `pytorch_scatter` dependency | **`scatter_sum` IS `scatter_add_`** — a 3-line wrapper, no kernel. PyG measures **1.01–1.06× for sum** | release-only maintenance; PyG moving to deprecate; **wheels not on PyPI**; **no MPS backend, no arm64 wheels** | — | — | — |
| ❌ Inductor/ROCm "partitioned scatter" | 1.82–10.45× at high contention; **3.1× REGRESSION at low contention on H100** | shipped **disabled by default** | — | ❌ we are the regressing case | — |
| ❌ Triton segment-sum | **0.04–0.69× of torch on H100** (lumen#225); GeoT's author: Triton *"struggles"* with warp divergence under indirect loading | — | — | ❌ | breaks + **doesn't serialize through `torch.export`** (#151746) |
| ❌ `torch.sparse.mm` / ONNX sparse | — | ORT SpMM is **CPU-only, fp32-only, build-gated, buggy**; **no `sparse_csr` on MPS at all** | — | ❌ | ❌ |
| ❌ MPSGraph / MPS sparse SpMM | — | **does not exist** — SDK headers expose only sparse-tensor *constructors* | — | ❌ | — |
| Kahan/Neumaier compensated segment sums | n/a | easy | any | ❌ **unnecessary** — fan-in ≤ 8 (§6.2) | none |
| **`torch.mps.compile_shader`** custom Metal kernel | n/a (enabler) | **public, in-tree, one-line JIT; the mechanism MPSInductor itself uses** | Apple | ✅ **the only route to a step change on this machine** | breaks (needs a pure-ATen fallback) |
| `mx.gather_mm(..., sorted_indices=True)` | fused gather→GEMM, *"more efficient than `take()` then `matmul()`"* | shipped in MLX | Apple | ✅ **rewards pre-sorting the edge list** | n/a (MLX, not PyTorch) |

## 2. The Hashimoto / line-graph non-backtracking formulation

### 2.1 It is exact, and Chemprop is already computing it — in the wrong factorisation

Let `|V| = n`, `m` undirected bonds, `2m` directed bonds. For directed bond `e`, write
`src(e) = edge_index[0,e]`, `dst(e) = edge_index[1,e]`, and `rev(e)` for the opposite-direction twin.
Define three 0/1 matrices:

- `S ∈ {0,1}^{2m×n}`, `S[e,u] = 1 ⟺ src(e) = u`   (tail / source incidence)
- `T ∈ {0,1}^{2m×n}`, `T[e,v] = 1 ⟺ dst(e) = v`   (head / target incidence)
- `τ ∈ {0,1}^{2m×2m}`, `τ[e,f] = 1 ⟺ f = rev(e)`  (edge-reversal involution, `τ = τᵀ = τ⁻¹`)

`_BondMessagePassingMixin.message()` computes, exactly:

```
N = Tᵀ H                 # (n × d_h) node sums:  N[v] = Σ_{dst(f)=v} H_f      <- scatter_reduce_
M = S N − τ H            # gather by src, minus the reverse edge               <- [edge_index[0]] − H[rev_edge_index]
```

so

> **`M = B H`  with  `B := S Tᵀ − τ`.**

`B[e,f] = 1 ⟺ dst(f) = src(e) **and** f ≠ rev(e)`. That is precisely the **Hashimoto /
non-backtracking edge-adjacency operator** of `G` (equivalently the non-backtracking operator on
the line graph `L(G)`); the identity `B = S Tᵀ − τ` is the standard textbook definition
("`B = ST − τ`, where `S` and `T` are incidence matrices and `τ` is the edge-reversal operator").
`B` is the operator whose powers count non-backtracking walks, whose spectrum underpins the
Krzakala et al. community-detection threshold, and which is the subject of the LGNN and NBA-GNN
lines of work.

**Nothing is approximated.** `S Tᵀ − τ` and `B` are the same matrix; the only question is whether
you evaluate the factored form or the explicit form. Chemprop chose factored. For molecules that is
the wrong call.

### 2.2 Structure of `B` on real molecules (measured, not estimated)

Row `e` of `B` has `deg(src(e)) − 1` nonzeros, so

> `nnz(B) = Σ_e (deg(src(e)) − 1) = Σ_v deg(v)(deg(v) − 1) = Σ_v deg(v)² − 2m`

Measured over **6,253 molecules parsed from Chemprop's own `tests/data/*.csv`** (120,735 atoms,
252,646 directed bonds, RDKit heavy-atom graphs — script: `nb_stats.py`, `designs.py` in this
directory):

```
mean atom degree       2.0926      max degree 8
degree histogram       d=1: 28,755   d=2: 54,637   d=3: 34,490   d=4: 2,776   d≥5: 7
nnz(B)                 349,752      nnz(B)/row = 1.384
row-nnz histogram      r=0: 10.7%   r=1: 43.5%   r=2: 40.7%   r=3: 5.1%   r≥4: 0.05%
```

Two things jump out:

- **`B` has fewer than 1.4 nonzeros per row.** It is sparser than a plain `index_select`. 10.7% of
  rows are *identically zero* (bonds leaving a terminal atom receive no message at all).
- **4 buckets (`r = 0,1,2,3`) cover 99.95% of rows.** The `r ≥ 4` tail is 43 rows out of 252,646.

### 2.3 Why the factored form loses at degree 2 — and the exact crossover

Count traffic in units of one `d_h`-vector read or written, per `message()` call:

| formulation | vector traffic | vs current | vs `index_add_` fix |
|---|---|---|---|
| **current** (`scatter_reduce_` + `.repeat`'d int64 index) | `8m + 2n + 2·(2m)` ← index counts 2× (8 B vs 4 B) | 1.00× | 0.55× |
| `index_add_`, 1-D index, still factored | `2m`(read H) + `2n`(zero+write N) + `2m`(gather N) + `2m`(read τH) + `2m`(write M) | **1.81×** | 1.00× |
| **explicit Hashimoto** (CSR or bucketed) | `nnz(B)`(gather H) + `2m`(write M) | **3.72×** | **2.06×** |
| padded gather `K=2` + `r=3` correction | | 2.93× | 1.62× |
| padded gather `K = max_row` (=7) | | 1.12× | 0.62× |

(Measured on the 2,346-unique-molecule subset; ratios are stable across the full set and are
**independent of `d_h`**, since every term scales as `d_h`.)

Setting the two expressions equal gives the crossover:

> **explicit `B` beats the factored form ⟺ `mean(deg²) < 4·mean(deg) + 2`**
> ⟺ for a `d`-regular graph, **`d < 4.45`**

Measured: `mean(deg²) = 5.09` vs threshold `10.46` → explicit `B` wins by ~2×. Cora (mean degree
~4) sits essentially *on* the crossover; everything bigger (Reddit at ~492, ogbn-products at ~50)
loses catastrophically. **This is exactly why no GNN-kernel paper proposes it, and exactly why it
is right for Chemprop.** It is the clearest "under-explored because the benchmark suite is wrong"
opportunity in this whole survey.

NBA-GNN (arXiv 2310.07430, ICLR'24) independently reinvents the non-backtracking update for
over-squashing reasons but implements it by *explicit neighbour exclusion on citation graphs* and
pays for it: **~160 MiB vs ~24 MiB and 38.95 s vs 12.22 s per run on Cora (≈3–4× slower)**. That is
the high-degree penalty, measured. It does **not** cite the Hashimoto matrix, D-MPNN, or Chemprop —
so the connection "D-MPNN's `−H_rev` *is* the Hashimoto operator, and on molecules the explicit
form is *cheaper*" appears to be unstated in the literature.

### 2.3b Is the `−H_rev` term load-bearing? (Yes — so don't "optimise" it away)

Worth settling before proposing anything, since the cheapest "optimisation" would be to delete the
term. The literature says keep it:

- The original D-MPNN work and Chemprop's own documentation state that the reverse bond is excluded
  "to prevent *totters*" (messages bouncing back and forth) and "for increased numerical stability",
  and report that **messages centred on directed bonds outperform the alternatives** for molecular
  property prediction.
- NBA-GNN gives the theoretical version: non-backtracking message passing's sensitivity bound
  decays as `O(d^{−T})` vs `O((d+1)^{−T})` for standard MPNNs (less over-squashing), and it can
  recover sparse SBMs at average degree `ω(1)·n^{o(1)}` vs `Ω(log n)` for standard GNNs.
- "What drives performance in molecular MPNNs? An operator-level factorial benchmark" (84
  configurations × 10 MoleculeNet datasets, depth ∈ {2,…,6}, FFN hidden 128–2048) finds
  **"performance variation is associated primarily with message construction rather than update
  complexity"**, with update operators showing *no statistically significant effect*. That is a
  direct argument that the message-construction operator (i.e. `B`) is the part you must keep
  exact, while the `W_h` update — the expensive part — is where accuracy is least sensitive. It is
  also a hint that low-precision `W_h` may be tolerable at the *architecture* level even though
  §5.1's depth-compounding argument limits how far you can push it.

So: **keep `B` exactly; the proposal in §2 changes only how it is evaluated, not what is computed.**
There is no accuracy risk, because it is the same matrix.

### 2.4 Cost of materialising `B`

CSR with implicit unit values (no `values` array — every nonzero is 1):

- `indptr`: `2m+1` int32; `indices`: `nnz(B)` int32
- For a 500-molecule batch (`2m ≈ 20,200`, `nnz ≈ 28,000`): **0.08 MB + 0.11 MB ≈ 0.19 MB**
- vs `H` at `d_h=2048`: 166 MB → **`B` costs 0.14% of `H`** (0.94% at `d_h=300`)

Built once per batch in `BatchMolGraph` construction, reused for all `depth−1` iterations **and for
the backward pass**. Negligible.

### 2.5 The backward pass is also atomics-free — a second, separate win

`∂L/∂H = Bᵀ (∂L/∂M)`, and

> `Bᵀ = (S Tᵀ − τ)ᵀ = T Sᵀ − τ`

i.e. **`Bᵀ` is the same kind of operator with `src`/`dst` swapped.** Row `f` of `Bᵀ` has
`deg(dst(f)) − 1` nonzeros, so `nnz(Bᵀ) = nnz(B)` — backward costs the same as forward, and runs
through the *same kernel* with swapped incidence.

This matters because **today's backward does use atomics**: the forward `scatter_reduce_(sum)`
backprops to a deterministic gather, but the forward gather `[edge_index[0]]` backprops to
`index_add_`/`scatter_add_`, which PyTorch documents as nondeterministic on CUDA
(`atomicAdd`; listed explicitly in the reproducibility notes alongside `scatter_add_` and
`bincount`). A custom `autograd.Function` applying `Bᵀ` as an explicit gather-sum makes **both
directions bit-deterministic at zero extra cost** — whereas `torch.use_deterministic_algorithms(True)`
today makes `index_add_` either slow or an outright error.

### 2.6 Four implementations, ranked by effort

**(a) Pure-PyTorch bucketed gather-sum — no custom kernel, exports cleanly.**
Permute directed bonds *once*, at `BatchMolGraph` construction, so rows are grouped by
`r = deg(src(e)) − 1`. Then the whole message step is:

```python
# precomputed per batch: nb[r] = (n_r, r) int64 gather indices, rows already in bucket order
M = torch.cat([
    H.new_zeros(n0, d_h),                      # r = 0  (10.7% of rows)
    H[nb1[:, 0]],                              # r = 1  (43.5%)
    H[nb2[:, 0]] + H[nb2[:, 1]],               # r = 2  (40.7%)
    H[nb3[:, 0]] + H[nb3[:, 1]] + H[nb3[:, 2]],# r = 3  ( 5.1%)
], dim=0)
```

- Reads exactly `nnz(B)` vectors; writes `2m`. **Optimal traffic.**
- Only gathers — **no scatter, no atomics, bit-deterministic forward.**
- `r ≥ 4` tail (0.05%): either add a 5th bucket, or *split* a high-`r` row into two `r ≤ 3` partial
  rows and add them — exact, keeps the kernel count at 4.
- Because rows are pre-permuted at batch construction, **there is no permutation in the hot loop**.
  `H_0`, `E`, `rev_edge_index` and the final node-scatter index are permuted once to match.
- Exports to ONNX as `Gather` + `Add` + `Concat`. TorchScript-safe. Identical code path on CPU /
  MPS / CUDA. **Zero new dependencies.**
- Risk: 4 kernels instead of 3, so at `d_h=300` launch overhead may eat the win. `torch.compile`
  should fuse the adds. The `K=2 + correction` variant (2.93× vs current) is a 2-kernel compromise.

**(a′) Implementation notes for the bucketed form — two orderings want to conflict, and the
resolution is easy.**

- The depth loop wants edges ordered **by `r = deg(src(e)) − 1`** (bucket order). The *final* node
  aggregation `M_v = Σ_{dst(f)=v} H_f` wants edges ordered **by `dst`** so it becomes a sorted
  segment reduction (atomics-free, and the one place GeoT-style kernels genuinely apply).
- **Resolution: bucket order wins.** The depth loop runs `depth−1` times (2–5); the node
  aggregation runs *once*. Pay the unsorted scatter once at the end rather than an inverse
  permutation (`+2m` vectors, which would erode the win from 2.06× to ~1.45×) on every iteration.
- Permute once, in `BatchMolGraph`: `edge_index`, `E`, `rev_edge_index`, and the per-bucket gather
  index arrays. `H_0` is then computed directly in bucket order. Nothing in the hot loop permutes.
- **`rev_edge_index` disappears from the hot loop entirely** — it is only still needed for the
  `undirected=True` option's `(H + H[rev])/2` pre-average. That removes one `(2m × d_h)` gather
  per iteration on top of everything else, and it is already counted in the traffic table.

**(b) Custom CSR gather-sum kernel (Triton / Metal / C++).**
A CSR-SpMM with implicit unit values — one block per (row-tile × feature-tile), read `indptr`/
`indices`, accumulate in fp32 registers, one store. ~15 lines of Triton. Zero atomics, exact
`nnz(B)` reads, deterministic. Breaks TorchScript/ONNX unless registered as a custom op with a
decomposed fallback (see §8).

**(c) Fold `B` into the GEMM.** `B` acts on the *edge* index, `W_h` on the *feature* index, so they
commute:

> `H^{t+1} = relu(H_0 + (B H^t) W_hᵀ) = relu(H_0 + B (H^t W_hᵀ))`

So compute `G = H W_hᵀ` as a plain dense GEMM (`M=2m`, `N=K=d_h` — see §3.1; there is nothing
"small-graph" about this shape), then apply `B` either
- in the **GEMM epilogue** as a scatter-accumulate — needs atomics, since several `f` map to one
  `e`; or
- in the **next GEMM's A-operand prologue** as a ≤3-way gather-sum — *no atomics*, strictly better.

CUTLASS has supported gather-A / scatter-D GEMM fusion since **2.9.0**; CUTLASS 4.x's CuTe DSL
(Python) makes a custom A-loader tractable without C++ template work. This is the
"gather→GEMM→scatter" fusion that GE-SpMM/FeatGraph/dgNN/DF-GNN all chase, but with an operator so
sparse (1.4 nnz/row) that the gather is nearly free.

**(d) Depth-loop persistent megakernel.** §3.3.

### 2.7 Which sparse format? (the bucketed design *is* SELL-C-σ)

Row-length distribution of `B` is `{0: 10.7%, 1: 43.5%, 2: 40.7%, 3: 5.1%, ≥4: 0.05%}` — about as
tight as a sparsity pattern ever gets.

- **CSR / CSC** — correct and compact (0.19 MB/batch, §2.4); the natural format for a custom
  gather-sum kernel. `indptr` loads are pure overhead when rows are this short, which is why…
- **ELL (fixed `K`)** — pad every row to `K`. `K = max = 7` wastes 5× (the table in §2.3 shows it
  loses to the factored form at 1.12×). `K = 2` + a correction pass for the 5.1% of `r=3` rows
  gives 2.93×. Viable as a 2-kernel compromise.
- **SELL-C-σ (Sliced ELLPACK with row-length sorting)** — sort rows by length within a window, then
  use a per-slice `K`. **This is exactly the bucketed design in (a).** With `σ` = the whole matrix
  (a global sort, free because we permute at batch-construction anyway) and `C` = one slice per
  distinct row length, **padding waste is identically zero** and you get the optimal `nnz(B)` read
  count. SELL-C-σ is implemented in cuSPARSE (for SpMV) and in portable form
  (e.g. `EllSparse.jl`); reported wins are modest in general (up to 1.47× on a stencil with 0.08%
  slice padding) *because* most matrices don't sort this cleanly. `B` does.
- **Blocked-ELL — avoid.** cuSPARSE's Blocked-ELL SpMM "provides the best performance with
  power-of-2 block sizes" and falls off a cliff otherwise: a documented case shows the same matrix
  (355 K nonzeros) taking **37 µs in CSR vs 13 ms in Blocked-ELL — a 350× gap**. `B` has no dense
  blocks at all (1.4 nnz/row), so Blocked-ELL is the worst possible choice.
- **Dense block-diagonal** — §4.2. Not a sparse format at all; trades FLOPs for regularity.

**Recommendation: bucketed/SELL-C-σ with 4 slices, materialised as 4 plain `index_select` + `add`
in pure PyTorch (no format library, no dependency, exports cleanly).** Keep CSR as the layout for a
later custom kernel.

---

## 3. Fused kernels, Triton/CUTLASS, and the roofline

### 3.1 Roofline: the GEMM is compute-bound; the *message step* is the bandwidth hog

Per message iteration, for the quoted batch (`n=6,600`, `2m=13,600`), fp32:

| `d_h` | `H` | `.repeat`'d int64 index | `W_h` | `W_h` GEMM | GEMM traffic | **GEMM arithmetic intensity** |
|---|---|---|---|---|---|---|
| 300 | 16.3 MB | **32.6 MB** | 0.4 MB | 2.4 GFLOP | 33.0 MB | **74.2 FLOP/byte** |
| 2048 | 111.4 MB | **222.8 MB** | 16.8 MB | 114.1 GFLOP | 239.6 MB | **476.1 FLOP/byte** |

| `d_h` | message traffic: current | with `index_add_` | with Hashimoto | message/GEMM traffic ratio |
|---|---|---|---|---|
| 300 | 146.4 MB | 81.1 MB | **38.9 MB** | 4.44× → **1.18×** |
| 2048 | 999.4 MB | 553.8 MB | **265.6 MB** | 4.17× → **1.11×** |

Read this carefully, because it resolves an apparent contradiction in the brief:

- The `W_h` GEMM has **AI = 476 FLOP/byte** at `d_h=2048`. Every accelerator's ridge point is
  1–100 FLOP/byte (an M4 Max at ~4 TFLOP/s fp32 / 546 GB/s ridges around 7; an H100 bf16 ridges
  around 300). **The GEMM is firmly compute-bound.** It is not the thing that is memory-bound.
- But the *message step* moves **4.2–4.4× more bytes than the GEMM**, almost entirely because of
  the `.repeat`'d int64 index. That is why the loop *as a whole* profiles as bandwidth-bound.
- After both fixes (#1 + #2), message traffic drops to **1.1–1.2× the GEMM's traffic** — i.e. the
  kernel stops being bandwidth-bound and **the GEMM becomes the bottleneck**. Only *then* do
  tensor cores and low precision start to matter (§5). Sequencing matters: low precision before
  the traffic fixes buys much less than its headline numbers suggest.

So: **the thing that is memory-bound is the index tensor and the extra round-trips, not the math.**

### 3.2 Fused-kernel literature — what's real, what transfers

| system | what it fuses | **measured** speedup (hardware / shapes) | impl. maturity | transfers to 13-atom molecules? |
|---|---|---|---|---|
| **GE-SpMM** (SC'20) | coalesced row-major CSR SpMM, merge-based load balance | 1.41× vs cuSPARSE; 2.39–6.15× vs DGL's SpMM; up to 3.67× end-to-end GCN. Real-world graphs | research code | **No.** Load-balancing + neighbour-reuse gains vanish at degree 2. |
| **FusedMM** (IPDPS'21) | SDDMM+SpMM into one kernel | up to ~28× vs unfused on embedding/GNN kernels | research code, CPU-focused | **No.** We have no SDDMM (no edge attention). |
| **FeatGraph** (SC'20, Amazon) | TVM-generated gSpMM/gSDDMM + feature-dim tiling | up to **32× CPU, 7× GPU** end-to-end | TVM-based, archived | **Partially.** The *idea* (schedule the feature dimension, not just the graph) is the one that transfers; feature dims 300–2048 are far outside their tuned range. |
| **Graphiler** (MLSys'22) | UDF message passing → fused ops via compiler | "up to two orders of magnitude" vs UDF baselines; mainly removes launch overhead + bandwidth | research prototype | **Idea yes, code no.** Its win *is* launch-overhead removal — same lever as #3, which is our biggest non-algorithmic one. |
| **Seastar** (EuroSys'21) | vertex-centric → fused kernels | ~2–8× vs DGL/PyG | research | idea only |
| **dgNN** | attention-GNN (GAT) fusion | **Negative result**: GNNBench found dgNN **45.77% slower** than a cuSPARSE baseline and saving only 16.44% memory | research | **No** — and a useful cautionary data point that fusion is not free. |
| **TLPGNN** (HPDC'22 / TOPC'24) | two-level parallelism, kernel fusion, register caching | avg **12.9×/12.1×/9.8×/8.6×** across GNN models vs baselines | research | **No** — vertex-parallel design assumes long rows. |
| **DF-GNN** (NeurIPS'24, SJTU+AWS) | SDDMM+Softmax+SpMM in one kernel (SMMF/PMF) | up to **7.0×** vs DGL sparse; **2.16× avg end-to-end**; beats cuGraph + dgNN | PyTorch-integrated | **No.** Attention GNNs only. |
| **GeoT** (2024) | tuned **segment reduction** (the exact op we need) | vs PyG `scatter_reduce` **1.68× geomean (max 4.67×)**; vs `segment_coo` **1.28× geomean (max 1.85×)**; SpMM vs cuSPARSE 1.34× geomean. A100/H100/RTX3090Ti, Citeseer→Reddit2 | **public: github.com/fishmingyu/GeoT** | **Maybe — the most relevant of the lot.** Caveats: requires **sorted non-decreasing indices**; explicitly tuned for feature sizes 16–64; "underperforms on F=1". Needs re-tuning for `d_h=2048`. |
| **MaxK-GNN** (ASPLOS'24) | MaxK nonlinearity → structured sparsity → SSpMM | 2.9×/2.98× vs cuSPARSE; 3.22×/4.24× end-to-end on Reddit | public | **No.** Paper states gains are "for graphs with average degree greater than 50". |
| **IO-aware GNN layers** (Yandex Research, 2026) | FlashAttention-style tiling for GNN layers; avoids materialising intermediates | code: **github.com/yandex-research/On-Efficient-Scaling-Of-GNNs** | research code | **Promising.** The "don't materialise the intermediate" principle is exactly our `.repeat`-index and `N`-buffer problem. Worth reading for the tiling strategy. |
| **TorchMD-Net 2.0** (2024) | CUDA graphs + custom CUDA neighbour lists + TorchScript/`torch.compile` on **small molecules** | **CUDA graphs 2–10× for molecules ≤~300 atoms; `torch.compile` 2–3×** (RTX 4090, PyTorch 2.1.0) | **production, maintained** | **Yes — the single best-matched precedent.** Same regime, same diagnosis ("kernel launch overhead [becomes] a performance bottleneck"), same fix (pad to static shapes, capture the graph). |

**Verdict on axis 1:** the fused-SpMM literature is a near-total miss for this workload. The two
items worth actually reading are **GeoT** (a real, public, tuned segment-reduction kernel) and
**TorchMD-Net 2.0** (the right regime, with a measured 2–10×). The transferable *idea* from
Graphiler/FeatGraph/DF-GNN is "eliminate launch overhead and intermediate materialisation", which
is #2 and #3 here, not their kernels.

### 3.3 Persistent kernels / megakernels over the `depth` loop

The closest published analogue is strikingly exact. In speculative decoding with serial draft
heads, "the projection weights remain unchanged for the entire walk, but conventional operator
execution retains each tile in registers or shared memory for only one projection". The fix: a
persistent megakernel where each SM **owns fixed row shards of the projection and keeps them in
on-chip storage across all steps**, exchanging only compact state between steps, "decoupling
traffic for the resident weights from the number of steps in the walk". Measured **up to 11.6×
head-level** (B=1, H100/B200) and 8.6–12.3% end-to-end.

Map onto Chemprop: `W_h` ≡ the projection; `depth−1` ≡ the walk length; `H` ≡ the state.

Feasibility arithmetic:
- `d_h=2048`, bf16: `W_h` = 8 MB. An SM owning a 16-column N-tile holds `2048×16×2 = 64 KB` —
  comfortably inside Hopper's 228 KB/SM shared memory. 128 SMs × 16 cols = 2048 → **the entire
  `W_h` is resident on-chip** across the whole depth loop.
- `d_h=300`, fp32: `W_h` = 0.36 MB, `H` = 16 MB, `N` = 8 MB → **~25 MB total, inside an H100's
  50 MB L2.** A depth-loop megakernel could run depth 3–6 with near-zero HBM traffic after load.

> **⚠️ Correction to the analogy — the value here is the `H` round-trips, not the `W_h` residency.**
> Unlike the speculative-decoding case (batch-1 decode, **weight reuse ≈ 1**), our GEMM has
> `M = 13,600` rows, so **`W_h` is reused 13,600× within a single GEMM** and its per-iteration DRAM
> traffic is already ~nothing. `W_h` at 16.8 MB fp32 is a *capacity* figure, not a *traffic* figure,
> and it **already fits in L2 on every part from A100 (40 MB) onward** (H100 50 MB, GB203 65 MB
> measured, B200 126 MB). **So do not sell a megakernel on weight residency.** Sell it on what it
> actually buys here: eliminating the 5–6 HBM round-trips of `H` per depth iteration (which §5.6
> shows is the difference between AI ≈ 102 and AI ≈ 342 FLOP/byte, i.e. between memory-bound and
> compute-bound on every current part) and eliminating the launches. I found **no measured study**
> of the "repeated tall-skinny GEMM with a static L2-resident weight" case.

The megakernel ecosystem is real but all CUDA: Ada-MK (adaptive DAG-based megakernel search),
AutoMegaKernel, Cohere's North Mini serving megakernel, AMD's MI355X megakernel work. Reported
general wins: "up to 1.7× end-to-end vs kernel-per-operator serving, up to 10× over native
PyTorch for Qwen3-1.7B". **Status for us: plausible but unvalidated; high effort; CUDA-only;
export-hostile.** The cheap 80% is CUDA graphs over bucketed static shapes (TorchMD-Net's 2–10×).

### 3.4 Triton / CUTLASS / CuTe practice

- **CUTLASS gather-GEMM-scatter is a shipped feature, not research.** "Gather and Scatter Fusion
  with GEMM can gather inputs and scatter outputs based on indices vectors in the same GEMM
  kernel" — **since CUTLASS 2.9.0**. You pass index-vector pointers for gather-A, gather-B,
  scatter-D in the kernel arguments. Epilogue Visitor Trees (CUTLASS 3.x) are the supported way to
  add bias+ReLU+reduction to the epilogue — exactly our `relu(H_0 + ·)`.
- **CUTLASS 4.x CuTe DSL is Python-native** ("Python native interfaces for writing high-performance
  CUDA kernels... without performance compromises"), supports Python 3.10–3.13, and ships
  Blackwell blockscaled/NVFP4 **grouped** GEMM with "performance on par with CUTLASS C++". This
  dramatically lowers the cost of prototyping a custom gather-GEMM mainloop vs. C++ templates.
- **Triton is mature enough for this**, with one caveat: `tl.atomic_add` "is performant when
  contention is low" and degrades badly under contention; the practical advice is
  `sem="relaxed"` and avoiding atomics altogether. **The Hashimoto gather-sum formulation has no
  atomics at all**, which is precisely why it is the right kernel to write in Triton.
  The best existing template to copy is **ScatterMoE / scatter2scatter**, which "fuses grouped
  GEMMs with scattered read/write via a Triton kernel, eliminating the need to materialise padded
  tensors in HBM" and "pads indices rather than tensors" — the same trick we want (index padding,
  not tensor padding).
- **Nothing in Triton/CUTLASS helps on Apple Silicon.** See §7.

---

## 4. Grouped / batched GEMM for many small molecules

### 4.1 First, a correction: grouped GEMM is the wrong tool for the `W_h` GEMM

This deserves flagging because it is an inviting but wrong inference from "many tiny graphs".

**Every molecule shares the same `W_h`.** Therefore the per-iteration linear layer is *already* a
single dense GEMM with `M = 2m ≈ 13,600–20,200`, `N = K = d_h`. That is a large, well-shaped,
tensor-core-friendly GEMM with AI 74–476 FLOP/byte. There are no "groups".

Grouped/segment GEMM exists to handle **different weights per group**: relational GNNs (RGCN, HGT
— one weight matrix per edge type) and MoE (one per expert). That is what `pyg-lib`'s
`segment_matmul`/`grouped_matmul` are for; they were contributed with a **CUTLASS grouped-GEMM
backend** (rusty1s + NVIDIA's Haicheng Wu) and FASTEN reports **up to 117.5× / 13.65× avg over
CUTLASS and 4.72× over cuBLAS on A100/RTX4090/GH200** — all for *heterogeneous* GNNs with
per-relation weights. **Chemprop has one relation type and one weight matrix. Grouped GEMM would be
strictly worse than the single dense GEMM it already does.** Do not take the `pyg-lib` dependency
for this.

(The one future case where it *would* apply: if Chemprop ever gains per-bond-type or per-edge-type
`W_h` matrices, or a mixture-of-experts readout.)

### 4.2 Where batching genuinely helps: the dense block-diagonal `B`

`B = diag(B_1, …, B_b)` is block-diagonal over molecules, with `B_i` of size `2m_i × 2m_i`. So the
message step *can* be a batched dense GEMM — trading FLOPs (which are cheap and tensor-core-eligible)
for the elimination of all irregular memory access, all atomics, and all index traffic.

Design sketch:

1. **At dataset/collate time**, sort molecules by `2m` and form batches from size buckets, so every
   molecule in a batch is close to `E_max` for that bucket. (Shuffle *within* buckets to avoid
   correlating batch composition with molecule size — the standard length-bucketing caveat from NLP.)
2. Lay out `H` as `(b, E_max, d_h)` with a validity mask.
3. Build `B_pad` as `(b, E_max, E_max)` 0/1 (or bf16) **once per batch**.
4. Message step becomes `M = torch.bmm(B_pad, H)` — one op. `W_h` applies as
   `H.view(b*E_max, d_h) @ W_h.T`.

Measured cost (2,346 unique molecules; `2m` mean 40.6, p50 38, p95 80, p99 139, max 264), `b=500`:

| padding policy | `E_max` | `B`-bmm | as % of `W_h` GEMM (`d_h=2048`, 170 GFLOP) | `W_h` GEMM inflation from padding |
|---|---|---|---|---|
| pad to batch p99 | 139 | 39.6 GFLOP | 23.3% | **3.43×  ← fatal** |
| pad to p95 | 80 | 13.1 GFLOP | 7.7% | 1.97× ← still bad |
| **size-bucketed (1.15× mean)** | **47** | **4.46 GFLOP** | **2.6%** | **1.15×  ← viable** |

At `d_h=300` the same policies cost 158.9% / 52.6% / **17.9%** of the (much smaller) GEMM.

**The honest conclusion:** this design lives or dies on size bucketing. Naive padding to the batch
maximum inflates the *main* `W_h` GEMM by 3.4×, which is far worse than anything it saves. With
tight size bucketing it costs **+2.6% FLOPs at `d_h=2048`** and converts the entire irregular part
of the computation into `bmm`. At `d_h=300` the +17.9% is marginal and the launch-overhead argument
probably dominates anyway.

**Why this design is strategically valuable despite not being FLOP-optimal:**
- It is **pure `bmm` + `reshape` + `relu`** → exports to ONNX trivially, TorchScript-safe, no
  custom ops, no new dependencies.
- It is **bit-deterministic** forward and backward (`bmm` backward is `bmm`).
- It composes perfectly with **CUDA graphs** (static shapes per bucket — exactly TorchMD-Net's
  recipe that measured 2–10×).
- On **MPS/Metal**, `bmm` is a first-class, well-optimised path while scatter is weak — so this may
  be the *fastest available* formulation on Apple Silicon even though it burns extra FLOPs.
- `B_pad` in bf16 makes the message step a **tensor-core** op, which it currently is not at all.

Status: **plausible but unvalidated.** No paper I found does this for molecular GNNs. The FLOP
accounting above is arithmetic, not measurement.

---

## 5. Low precision

*(Detailed hardware/library findings from the dedicated sub-study are in §5.2; §5.1 is the
architecture-specific reasoning.)*

### 5.1 Why this architecture is unusually hostile to *weight* quantization

The companion study's asymmetry (int4 **cache/activations** ≈ +1.7% RMSE; int4 **weights** ≈ +28%
RMSE; 8-bit weights free; 6-bit marginal; ≤2-bit dead) has a clean structural explanation that is
worth stating because it generalises:

**`W_h` is applied `depth−1` times with the *same* quantization error.** Write `Ŵ = W + ΔW`. Over
the depth loop the perturbation compounds roughly like `(I + W⁻¹ΔW)^{depth-1}`, so a relative
weight error `δ` becomes `≈ (depth−1)·δ` in the best (linear, well-conditioned) case and worse
through the ReLU's sign flips. At `depth=6` that is a **5× amplification of weight error** —
whereas activation error at step `t` is attenuated by the remaining `depth−1−t` ReLU+residual steps
and, critically, each activation's error is an *independent* draw rather than a single shared bias.
This is the standard weight-tied/recurrent-network quantization argument, and it predicts the
observed asymmetry quantitatively: int4 weights at +28% vs int4 activations at +1.7% is roughly the
`depth`-amplification factor times the per-step ratio.

**Practical implications:**
1. **Do not quantize `W_h` below 8 bits.** The companion result (8-bit free, 6-bit marginal) is
   consistent with the compounding argument and should be treated as the ceiling.
2. **Quantize `H` (the activations/cache) aggressively instead.** `H` is `2m × d_h` = 111 MB at
   `d_h=2048` fp32; `W_h` is only 16.8 MB. **`H` is 6.6× the bytes of `W_h`.** The traffic-reduction
   leverage is almost entirely in `H`, and `H` is exactly the tensor that tolerates 4 bits.
   This is a happy coincidence: the forgiving tensor is the big one.
3. **The reduction must stay in fp32.** See §6 — this is non-negotiable and is standard practice.
4. **Sequence matters.** Per §3.1, after the traffic fixes the GEMM becomes compute-bound, so
   low-precision *GEMM* (not just storage) is where the remaining win is — and that needs real
   tensor-core support for the format.

**The literature backs the weight-tying explanation, and it is a good match.**

- **The standard LLM/feedforward consensus is the *opposite* of the companion result**: activation
  outliers are ~100× typical activations, weight distributions are "quite uniform and flat, which
  is easy to quantize" — SmoothQuant's whole premise is *migrating difficulty from activations to
  weights*. KV-cache (activation-like) tolerates 3–4 bits at ≤0.21 PPL. **The companion result only
  makes sense because `W_h` is shared.** And Degree-Quant says it explicitly for GNNs with
  per-layer weights: *"Weight elements, however, are consistently unaffected"* going INT8→INT4.
- **Graph-filtering analysis (arXiv 2506.01404)** — iterative message passing with a *shared* graph
  shift operator `S`, structurally identical to our depth loop. Error recursion
  `w̃_t = S(w̃_{t−1} + n_{t−1})`, giving output noise
  `ỹ = Σ_t Σ_{τ≥t} φ_τ S^{τ−t}(S − D) n_{t−1}` — **noise injected at step `t` is amplified by
  `S^{τ−t}` at every later step**, with gain `tr(G_n(S) S Sᵀ)` scaling with `‖S‖₂`. Replace `S`
  with `W_h` and this is the exact formal statement of "weight error compounds multiplicatively
  across the depth loop". They also give a closed-form diagonal feedback correction
  `α_{i,t−1} = [G_n(S)S]_{ii} / [G_n(S)]_{ii}`.
- **LoopQ (arXiv 2605.16343)** — quantization for *looped* language models (one weight-tied block
  applied `T` times): the closest architectural analogue that exists.
  - Prop. 2: `ε_T ≤ Σ_t (∏_{τ>t} γ_τ) ε_t^quant` — each loop's local error is multiplied by
    downstream Lipschitz factors; *"earlier errors pass through a longer chain of transformations."*
  - Prop. 1 adds a **second, distinct** failure mode: *"the same physical block serves different
    computational roles across loops and processes changing hidden-state distributions"* — so the
    *optimal* scale for `W_h` differs per depth iteration, but you only get one `W_h`.
  - Measured: W4A4 on Ouro 1.4B (4 loops) → static baselines **23.6%** vs **64.1%** FP 5-task mean;
    LAMBADA PPL **>1000** (static) vs **8.98** (LoopQ).
  - **Their fix is nearly free for Chemprop: "Loop-Aware Activation Scaling — maintains
    loop-dependent scales while keeping weights shared."** For us that is `depth ≤ 6` scalars
    computed per iteration and applied in the FP32 epilogue. **Steal this.**
- Corroborating: *Quantizing Recursive Reasoning Models* (arXiv 2607.16237) — *"recurrent
  quantization is more brittle than feedforward... Error grows with the number of passes through
  reused weights."* *Quantization Degradation: A Signal–Noise Perspective* (arXiv 2608.08188) gives
  a per-layer propagation gain `G_l` (>1 amplifies, <1 attenuates) — **a directly instrumentable
  diagnostic: measure `‖ΔH_k‖/‖H_k‖` vs depth `k` and check for geometric growth.** Classical RNN
  work agrees ("Binarizing LSTM is more challenging than binarizing the CNN or MLP").
- **The one counter-example, and why it doesn't rescue us.** *"Why Gated DeltaNet Survives 4-Bit
  Quantization"* (arXiv 2609.04098) gets W4A4 NVFP4 working across all 496 linear layers of a
  recurrent half. Four mechanisms: (1) NVFP4's 16-element blocks localise outliers; (2) gates
  compress ~11% GEMM error → ~2% output error; (3) **"noise forgetting" — each write overwrites the
  state along the current key direction**, so noise does not compound; (4) cost distributes across
  context. **Mechanism (3) is the load-bearing one and D-MPNN has no forgetting gate:**
  `H ← relu(H_0 + W_h M)` is a residual accumulator with no decay. (The `H_0` skip *does* re-inject
  clean input every iteration, which is plausibly why +1.7% activation error doesn't blow up.)
  **But mechanism (1) *is* available to us and is cheap** — which is the basis for the §5.6
  recommendation to retest block-scaled weights.

### 5.2 Which formats have real hardware acceleration, at what granularity, with what accumulate dtype

**OCP Microscaling (MX) Specification v1.1.0 (Oct 2026)** defines exactly four families, **all with
block size k = 32 and a single scale type, UE8M0** (8-bit pure exponent, bias 127). v1.1.0 is a
clarification release — it renamed E8M0 → UE8M0 and refined rounding; **it did not add a 16-element
block or an E4M3 scale type.** NVFP4 is NVIDIA's own variant (E2M1 data, block 16, UE4M3 scale).

| Format | Element | Block | Scale dtype | Bytes/elem |
|---|---|---|---|---|
| MXFP8 | E4M3 / E5M2 | 32 | UE8M0 | 1.03125 |
| MXFP6 | E2M3 / E3M2 | 32 | UE8M0 | 0.78125 |
| MXFP4 | E2M1 | 32 | UE8M0 | 0.53125 |
| MXINT8 | int8 | 32 | UE8M0 | 1.03125 |
| NVFP4 | E2M1 | **16** | **UE4M3** | 0.5625 |

**Hardware, precisely:**

- **Hopper (H100/H200, sm_90): NO MX hardware.** FP8 E4M3/E5M2 MMA with **per-tensor scaling
  only**. No FP6, no FP4, no hardware block scaling. Transformer Engine's matrix lists Hopper as
  *MXFP8: No, NVFP4: No*. cuBLAS 12.9's `MATRIX_SCALE_VEC128_32F` / `BLK128x128_32F` modes are
  **software/epilogue** block scaling (the DeepSeek recipe productised), sm_90-only, and **not
  available on Blackwell** — so you need two code paths.
- **Blackwell datacentre (B200/GB200, sm_100/103): full MX + NVFP4.** `tcgen05.mma.block_scale`
  with kinds `mxf8f6f4` (block 32), `mxf4` (block 32), `mxf4nvf4` (block 16 or 32, UE8M0 or UE4M3).
  Scales live in TMEM. **Accumulator is fixed to FP32** for all block-scaled kinds.
- **Blackwell consumer (RTX 50xx / RTX PRO 6000, sm_120): hardware present, different
  instruction.** Warp-level `mma.sync.aligned.kind::mxf8f6f4.block_scale` etc. with register-
  resident scales — *"SM10x GEMM kernels are completely incompatible with SM12x."* CUTLASS supports
  it; **Transformer Engine does not yet (MXFP8 unsupported on CC 12.0; NVFP4 forward-only)** — a
  library gap, not a silicon gap.
- **AMD:** CDNA 3 (MI300X) has FP8 + INT8/INT4, **no FP6/FP4, no MX**. **CDNA 4 (MI350X/MI355X,
  gfx950) adds native MXFP8/MXFP6/MXFP4.**
- **Intel:** Gaudi 3 has FP8 with FP32 accumulators but **no MX, no FP4, and FP8 runs at the same
  rate as BF16** (1,835 TF each) — pure memory play, zero compute upside. Crescent Island (Xe3P) is
  the first MXFP4-capable Intel part; announced, not shipping.

**Datasheet dense TFLOPS vs BF16:** H100/H200 FP8 **2.0×**; B200 FP8 2.0×, FP6 2.0×, **FP4 4.0×**;
MI355X FP8 2.0×, **FP6 4.0× (NVIDIA runs FP6 at the FP8 rate, AMD at the FP4 rate)**, FP4 4.0×;
Gaudi 3 FP8 **1.0×**.

**Measured, not datasheet** — the ratios are not achievable:
- Real block-scaled MXFP8 on a 1000 W B200: **~2,750 TFLOP/s vs 3,300 for per-tensor FP8 vs
  1,500–1,550 BF16** → **real MXFP8/BF16 ≈ 1.8×, not 2×**, and block scaling itself costs ~17% vs
  per-tensor FP8 because scales must co-reside in TMEM.
- Hand-written MXFP8 GEMM vs cuBLAS on B200: **0.87–0.99×** of cuBLAS.
- torchao's own B200 numbers: MXFP8 roofline 1.97× → **observed 1.81×**; NVFP4 theoretical 3.82× →
  **observed 2.98×**.
- FP4 PTQ (MR-GPTQ): layer-wise up to 3.6× on B200 / 6× on RTX 5090, but **end-to-end only
  2.2× / 4×**.

**Accumulate precision — the one real trap, and it is Hopper-specific:**

> **Hopper FP8 tensor cores do not accumulate in FP32.** DeepSeek-V3: *"the accumulation precision
> of FP8 GEMM on NVIDIA H800 GPUs is limited to retaining around 14 bits."* The follow-up paper is
> exact: after aligning 32 mantissa products, *"the Tensor Core only maintains their highest 13
> fraction bits for addition... Addition results are accumulated to **FP22 registers (1 sign,
> 8 exponent, 13 mantissa)**."* Exact FP32 accumulation of 32 FP8×FP8 products needs ≥34 bits.
> Worst case: ~2% error rate. **Fix = DeepSeek's "promotion to CUDA Cores"** — copy the partial
> result to FP32 registers every 128 K-elements and accumulate in true FP32. This is the origin of
> the standard 1×128/128×128 blockwise FP8 recipe (<0.25% loss at 671B/14.8T).
> **Blackwell's block-scaled MMA fixes this: the accumulator is FP32 by construction.**

**Consumer-part accumulate throttle (pure product segmentation).** On RTX 5090, **FP32-accumulate
costs exactly 2×** for FP16/BF16/FP8 (FP16 419 → 209.5 TF; FP8 838 → 419 TF) — but **FP4 and INT8
were never throttled** (FP4 1,676 TF at FP32-acc). The Ada-era fuse-unlock trick does not work on
Blackwell.

**For Chemprop's shapes the accumulate risk is mild**: `K = d_h = 300–2048`, so the reduction is
short; at `K=300` only 2–3 promotions are needed. The accumulate risk is far smaller than the
weight-quantization risk in §5.1.

### 5.3 The decisive constraint: **`d_h = 300` is categorically excluded from every MX path**

Our GEMM is `M = 2m ≈ 13,600`, `K = N = d_h`.

| | `d_h=300` | 512 | 1024 | 2048 |
|---|---|---|---|---|
| `M % 16` | ✓ | ✓ | ✓ | ✓ |
| `M % 128` | ✗ → pad to 13,696 (**+0.7%, trivial**) | same | same | same |
| **`K % 32` (MX minimum)** | **✗** | ✓ | ✓ | ✓ |
| `K % 128` (CUTLASS / `scaled_mm` v1) | **✗** | ✓ | ✓ | ✓ |
| `N % 16` | **✗** (18.75) | ✓ | ✓ | ✓ |

- **Small-M is a non-issue.** 13,600 rows is plenty; pad to 13,696. There is no cliff at 13.6 K.
- **Odd `d_h` is the killer.** `d_h=300` violates `K%32` *and* `N%16`. Padding 300 → 384 to become
  fully legal wastes **28% of the FLOPs to buy a 2× format → net ~1.56× before quantization
  overhead, i.e. a wash or a loss.** Padding to 320 works on the swizzled MX path only.
- **Re-quantization is first-order, not second-order, and must be paid every depth iteration**
  because `H` changes each time. Measured on *much larger* LLM shapes: quantization kernels were
  **~40% of matmul time** (0.44 ms quant vs 1.16 ms FP8 multiply), rising to **76%** including the
  backward transpose-quantize. **At our smaller GEMM sizes this fraction gets worse.**

### 5.4 Can the segment reduction run in low precision?

**The reduction should stay FP32. The storage/movement of `H` is what should narrow.**

- **There is no FP8 `atomicAdd` in CUDA C++.** Supported: `int`, `unsigned`, `ull`, `float`,
  `double`, `__half`, `__half2`, `__nv_bfloat16`, `__nv_bfloat162`, plus `float2`/`float4` on
  CC ≥ 9.x. At PTX level `.e4m3`/`.e5m2` are accepted by `red`/`atom` on **sm_100+** but with an
  explicit **`.add.acc::f16`** qualifier — the FP8 atomic accumulates in **FP16, not FP32**.
  *(medium confidence — the PTX table did not render cleanly.)*
- cuSPARSE supports FP16 and INT8 but **not INT4 and not FP8**. `scatter_add` was not natively
  supported by TensorRT 10.0, and not in INT8 — a recurring deployment blocker for quantized
  message passing.
- **Low-precision reductions do break — but by an N-dependent mechanism we don't have.** JAX
  issue #42211, directly on `segment_sum`: accumulating in bf16, **8192 × 1.25e-3 sums to 0.5
  instead of 10.24** (rel. error 6.9e-2). Split-K bf16 partials on gfx950: **~6% of outputs deviate
  ≥2 ULP.** Megatron: bf16 gradient reductions stable at 96 GPUs, unstable at 384; FP32 fixed it.
  **Every one of these is a saturation effect** — small contributions vanish once the running sum
  is large relative to the accumulator epsilon. **Chemprop's atom aggregation has N = degree
  ≈ 2.06, max 8.** A 2–4 term bf16 sum has essentially no saturation risk. *This is the single
  biggest way our workload differs from the published GNN-quantization literature, and it cuts in
  our favour.* (Note this is a *different* risk from §6's cancellation hazard, which we **do** have.)
- The molecule-level aggregation has fan-in ≈ 19 atoms — also small.

### 5.5 What the GNN-quantization literature actually measured

| work | result | transfers? |
|---|---|---|
| **Degree-Quant** (ICLR'21) — quantizes *everything* incl. aggregation, with stochastic full-precision protection of high-degree nodes | **ZINC MAE (molecular!): INT8 is free or better than FP32** — GCN 0.469→**0.434** (−7.5%), GAT 0.463→0.456, GIN 0.414→**0.357** (−13.8%). INT4: naive QAT **+38–47%**, DQ **+4–14%**. CPU INT8 up to **4.7×** | **ZINC table: YES.** The 4.7× (Reddit, 22-core Xeon, 2020): **no.** |
| Degree-Quant's *mechanism* — aggregation error scales with in-degree, "aggregation is the predominant source of quantization error, concentrated on high-degree nodes" | — | **Mechanism is structurally absent at degree 2–4. In our favour.** |
| Degree-Quant on weights: *"Weight elements, however, are consistently unaffected"* going INT8→INT4 | — | **Opposite of the companion study — and §5.1 explains why: GCN/GAT/GIN have a separate `W` per layer; D-MPNN reuses `W_h` `depth` times.** |
| **J. Cheminform. 17:81 (2025)**, DoReFa PTQ on molecular property prediction | **GCN/ESOL: W8A8 +1.4%, W4A4 +25.4%, W2A2 +322%.** FreeSolv W8A8 −0.3%, W4A4 +44.5%. QM9/GCN W8A8 +0.8%, W4A4 +80% | **YES** — and **+25.4% at W4A4 closely matches the companion study's +28% for int4 weights.** (Discount the GIN rows: W16A16 0.768→1.745 is artifactual — DoReFa clamps activations to [0,1], catastrophic for GIN's sum aggregation.) |
| **SGQuant** (ICTAI'20) | 4.25–31.9× memory, ≤0.4% accuracy drop | **No** — Cora/Citeseer/Reddit; the memory win is dominated by a resident giant node-feature matrix, which we don't have. No speedup reported. |
| **EXACT** (ICLR'22) — random projection + INT2 of stored activations | up to **32× activation memory**, 0.2–0.5% accuracy drop, **+10–25% time** | **No** — training-memory technique that *costs* time. Wrong objective. |
| **MixQ-GNN** (2025) Theorem 1: `Q_y(AX) = C1 ⊙ Q_a(A)Q_x(X) ⊙ C2 + C3` → scales factor out into one epilogue, so aggregation runs as **integer** sparse-dense matmul with no FP32 inner accumulator (INT32 is exact) | 5.5× / 5.1× **BitOPs** | **Theorem: conceptually yes.** BitOPs ≠ wall-clock; no molecular data. |
| **QGTC** (PPoPP'22) — the only work mapping GNN aggregation onto tensor cores, via 1-bit TC primitives and bit-decomposition (adjacency is naturally 1-bit) | **2.6× avg over DGL (Cluster-GCN), 2.8× (batched GIN)**; **accumulates in uint32** | **Spirit yes** (aggregate-then-transform; wide exact accumulator), numbers no (43 K–2.4 M nodes). |

### 5.6 Verdict on axis 4 — demote it

**Agent-independent roofline, which corroborates §3.1 from a different direction.** At `d_h=2048`:
- **perfectly fused** (read `H`, read `H_0`, write `H'`, `W_h` amortised): 334 MB → **AI ≈ 342 FLOP/byte**
- **unfused, as PyTorch actually runs it** (gather → segment-sum → gather+subtract → GEMM → add →
  ReLU, each streaming `H`): ≈ 1.1 GB → **AI ≈ 102 FLOP/byte**

Machine balance (dense BF16 TFLOPS ÷ HBM TB/s): A100 **153**, H100 **295**, H200 206, B200 281,
RTX 5090 117 FLOP/byte.

> **So the fused op would be compute-bound on every one of these parts (342 > all), and the unfused
> op is memory-bound on all of them (102 < all). The "memory-bound at `d_h=2048`" observation is an
> artifact of kernel non-fusion, not of the math.** At `d_h=300`, AI ≈ 49 — memory-bound no matter
> what. **The highest-leverage fix is fusion, not precision.**

**Also: the "fp8 weights free up L2" argument is weak and I should retract it.** `H` is **6.6×
larger than `W_h` at `d_h=2048` and 45× larger at `d_h=300`**; and `W_h` is reused `M = 13,600`
times *within one GEMM* and again across every depth iteration, so its DRAM traffic per iteration is
~nothing — **16.8 MB is a *capacity* figure, not a *traffic* figure.** Published L2: A100 40 MB,
H100 50 MB, GB203 65 MB (measured), B200 126 MB — a 16.8 MB fp32 `W_h` **already fits comfortably**.
Shrinking it frees L2 for streaming `H`, which is real but second-order, and **I found no measured
study quantifying exactly this** (repeated tall-skinny GEMM, static weight, L2-resident).

Every published "quantization helps the memory-bound case" result is about **weight reuse ≈ 1**
(batch-1 GEMV decode): AWQ W4A16 **3.4× on decode / 1.0× on prefill**; Apple M5 fp8 thin rows
(N=1) **2.6–3.0× purely from halving weight bytes with zero compute advantage**; vLLM BF16→FP8
1.4–1.8× at large batch. **We have weight reuse = 13,600 — four orders of magnitude away. These do
not transfer.** torchao's own data makes the point: NVFP4 Llama-3.1-8B is **1.739× on prefill
(compute-bound) but 1.058× on decode (memory-bound)** — *dynamic 4-bit bought 6% on the
memory-bound phase.*

**Ceiling from narrowing the streamed tensor `H`** (the only thing that actually moves):

| `H` dtype | bytes/elem | ceiling vs fp32 | vs bf16 |
|---|---|---|---|
| bf16 / fp16 | 2 | **2.00×** | 1.00× |
| FP8 E4M3 per-tensor | 1 | 4.00× | 2.00× |
| MXFP8 | 1.03125 | 3.88× | 1.94× |
| MXFP6 | 0.78125 | 5.12× | 2.56× |
| MXFP4 | 0.53125 | 7.53× | 3.76× |
| NVFP4 | 0.5625 | 7.11× | 3.56× |

From which subtract: per-iteration re-quantization (40–76% of matmul time at large shapes, worse at
ours), the mandatory FP32 segment-sum buffer (`n × d_h × 4` = 54 MB at `d_h=2048`, i.e. 2× the fp8
`H`), and the observed/roofline gap (1.81/1.97 for MXFP8, 2.98/3.82 for NVFP4).

> ### Recommended verdict: demote axis 4 from *"accelerate the GEMM with MXFP8/MXFP4"* to *"run the whole message-passing loop in bf16/fp16 with FP32 segment-sum accumulators, and fuse it."*
>
> That captures most of the available bandwidth win, works on **CUDA + ROCm + MPS + MLX**, has no
> alignment constraints, no `d_h=300` exclusion, no scale-swizzle churn, and no weight-quantization
> accuracy risk. Treat MXFP8/NVFP4 as a **Blackwell-and-`d_h`≥512-only stretch goal**, contingent on
> (a) fusing the loop first and (b) re-testing block-scaled weights (next point).
>
> **And do not conclude "low-precision weights are impossible" — conclude "*per-tensor* int4
> weights are impossible."** MR-GPTQ found that *"MXFP4's power-of-two scale quantization severely
> degrades accuracy due to high induced error"* while *"NVFP4's small group size provably
> neutralizes traditional outlier mitigation techniques"* — i.e. **scale granularity, not element
> bit-width, is the likely culprit** behind the +28%. Re-run with FP8 E4M3 + per-32 UE8M0 scales,
> and NVFP4 + per-16 UE4M3, before writing off the weight side.

---

## 6. Numerics

### 6.1 The `Σ − one_term` cancellation is real, quantifiable, and removable

`M_e = (Σ_{dst(f)=src(e)} H_f) − H_{rev(e)}`. The subtrahend is one of the summands. Error analysis:

- **`deg(src(e)) = 1`** (28,755 atoms = 23.8% of atoms): `N = H_rev` exactly, so `M = 0` is computed
  by exact cancellation of identical bit patterns. Correct, no error. But note it is computed by
  reading and subtracting a full `d_h` vector to get a guaranteed zero — 10.7% of all rows do this.
  The Hashimoto form writes the zero directly.
- **`deg(src(e)) = 2`** (54,637 atoms = 45.3%; **43.5% of all directed bonds**):
  `M = (H_f + H_rev) − H_rev`. Relative error ≈ `ε · (|H_f| + |H_rev|)/|H_f|`.
  - fp32 (`ε ≈ 6e-8`), magnitude ratio 10 → **~1e-6** — harmless.
  - **bf16 (`ε ≈ 4e-3`), magnitude ratio 10 → ~4e-2 = 4% error on the message.** With 8 mantissa
    bits there is simply no room. Ratio 100 → ~40%.
  - fp16 (`ε ≈ 5e-4`), ratio 10 → ~5e-3.
- **`deg ≥ 3`**: ordinary summation error, no catastrophic cancellation.

**So: the current formulation is fp32-safe but is a genuine bf16/fp16 trap, and it affects 43% of
directed bonds.** The explicit Hashimoto form (§2) sums only the terms that belong, so the
cancellation is *structurally absent* — not mitigated, absent. **This is the strongest argument in
the brief: the faster formulation is also the numerically correct one.**

A directly analogous bug class is well documented in GEMM land: Modular's issue
"Split-K matmul reduces partial sums in the output dtype by default; **bf16 outputs that cancel are
off by up to 40% on Ampere**", and hipBLASLt's "long-k fp16 GEMMs get no split-k, losing accuracy
(8 ulp at k=2²⁰)". Same mechanism — reducing in the narrow dtype when the result is a difference of
larger numbers.

### 6.2 Standard practice in production stacks

- **bf16 inputs, fp32 accumulate is the industry convention** — NVIDIA tensor cores, Google TPUs,
  and standard bf16 training practice all accumulate in fp32. Mechanically: inputs are rounded to
  bf16, "the BF16×BF16 product is exact in FP32 (the 8×8 significand product occupies at most 16
  bits of the 24-bit mantissa)", products accumulate in fp32.
  **Recommendation: if Chemprop ever does bf16 message passing, the segment reduction and the
  `H_0 + W_h M` residual add must be fp32.** This is free on tensor cores.
- **Split-K / pairwise summation**: pairwise summation improves forward error from `O(n)` to
  `O(log₂ n)` ulp. For us `n = deg ≤ 8`, so **fan-in is tiny and pairwise/Kahan is unnecessary** —
  this is one axis where the small-graph regime *helps*. Chemprop's molecule-level aggregation has
  fan-in ≈ 19 atoms, also tiny. Kahan/Neumaier compensation would be pure overhead here.
  (Contrast: Reddit's degree-492 reductions genuinely need it.)
- **Determinism.** PyTorch documents `index_add_`, `scatter_add_`, and `bincount` as using
  `atomicAdd` and therefore nondeterministic on CUDA; `torch.use_deterministic_algorithms(True)`
  makes `index_add_` throw or run a slow path, and "deterministic operations tend to have worse
  performance". **The Hashimoto gather-sum has no atomics, so it is deterministic at full speed in
  both directions (§2.5).** This is a free win that the current design cannot have.
- GPU non-determinism from float non-associativity is actively studied (arXiv 2408.05148 on
  reproducibility impacts; Hawkeye on reproducing GPU-level non-determinism) — worth citing if
  Chemprop wants a reproducibility guarantee as a feature.

### 6.3 Concrete recommendations

1. Adopt the Hashimoto form → cancellation gone, atomics gone, determinism free.
2. Keep all segment reductions and the residual add in **fp32** even if GEMMs go to bf16/fp8.
3. Do **not** add Kahan/Neumaier compensation — fan-in ≤ 8 makes it dead weight.
4. If staying with the factored form, **do not run it in bf16** without first measuring the
   degree-2 cancellation (see predictions).
5. Mean aggregation: `agg.py` already clamps the count (`counts.clamp(min=1)`), which is correct for
   the empty-graph case. Note `batch.max().int() + 1` for `dim_size` is a device sync and a
   correctness hazard if the last graphs in a batch are empty — unrelated to kernels but worth a
   look (and it is exactly the kind of thing the recent `MeanAggregation` ONNX fix, commit
   `fa77f38e`, touches).

---

## 7. Ecosystem signals (GitHub issues, PRs, maintainer positions)

### 7.1 `pytorch_scatter`: **do not take the dependency** — the answer is a clean negative

This was the open question that determined a dependency decision, and it resolves decisively.

> **`torch_scatter.scatter(..., reduce="sum")` is not a kernel.** It is a thin Python wrapper:
> ```python
> def scatter_sum(src, index, dim=-1, out=None, dim_size=None):
>     index = broadcast(index, src, dim)
>     out = torch.zeros(size, dtype=src.dtype, device=src.device)
>     return out.scatter_add_(dim, index, src)
> ```
> `scatter_mul/min/max` dispatch to `torch.ops.torch_scatter.*`; **`sum` does not.** For Chemprop's
> sum aggregation, installing `torch_scatter` **cannot be faster than what Chemprop already calls** —
> it is byte-for-byte the same op.

Corroboration from every direction:

- **PyG's own migration PR #5120** ("Move to PyTorch `scatter_reduce` implementation", merged
  2022-08-03): *"Speedup is not that impressive in all benchmarks, as `torch_scatter` for `sum`
  algorithm already uses PyTorch version under the hood."* Their benchmark table shows
  **2.55–4.33× for `edge_cnn`/`pna` (which use max/min)** but **1.01–1.06× — noise — for GCN/GAT
  (which use sum)**. ⚠️ All on Reddit / ogbn-products / ogbn-mag at hidden=128.
- **What PyG dispatches to today** (`torch_geometric/utils/_scatter.py`, master Nov 2026):
  `reduce='sum'` → *always* `src.new_zeros(size).scatter_add_(dim, broadcast(index, src, dim), src)`.
  torch_scatter is never consulted for sum. Their in-repo benchmark conclusions read:
  *"'sum': Prefer `scatter_add_`"* on **both** GPU and CPU.
- **Maintenance status:** torch_scatter is release-plumbing-only (commit log since mid-2025 is
  entirely "Add PyTorch 2.x support"); issue #513 "Support torch 2.13/2.14" has been open since
  2026-09-18. **PyG wants it dead**: pyg-team/pytorch_geometric#10716 proposes
  *"uploading their latest version with empty content with a user-friendly warning saying that we
  fully migrated to `pyg-lib`."*
- **Practical blocker regardless:** torch_scatter / pyg-lib wheels are **not on PyPI**, so
  `pip install chemprop` could never pull them transitively. For a library, that's a hard cost.

**The only torch_scatter ops that are real kernels and relevant are `segment_coo`/`segment_csr`**
(and `pyg_lib.ops.segment_sum_coo`, where they now live). But even there, temper expectations —
rusty1s's own description of the mechanism explains why:

> *"`segment_coo` parallelizes across the number of input elements. It then performs a parallel
> reduction inside a single warp (32 threads) to accumulate the result and to only write once via
> atomic ops."*

**The mechanism that makes `segment_*` fast is contention avoidance, and Chemprop barely has
contention** — ~13,600 inputs into ~6,600 groups of mean size 2.06, max in-degree ~4. There is
almost nothing for the warp reduction to eliminate. GeoT measured `segment_coo ≈ 1.31×` over
`scatter_reduce`, and **crucially its dataset table includes Cora (2,708 nodes / 10,556 edges) and
Citeseer (3,327 / 9,104) — almost exactly a Chemprop batch's magnitude**, which makes it the
closest published proxy I found. But Cora/Citeseer are one connected graph with hub nodes; a
Chemprop batch is hundreds of disjoint ≤4-degree components, i.e. *even less* contention.

**Zero-dependency alternative worth benchmarking:** `torch._segment_reduce(src, reduce, offsets=ptr,
initial=...)` **already works on 2-D CUDA tensors and is used in production by PyG today**
(`torch_geometric/utils/_segment.py` falls back to it when torch_scatter is absent, and prefers it
for `reduce='mean'` on CUDA). It is a private API and `torch.segment_reduce` is effectively
deprecated, and MPS coverage is incomplete (pytorch/pytorch#197014 "[MPS] Add native segment_reduce
forward pass" is open with 17 comments) — but it is the cheapest route to a sorted-index segment
reduction with no new dependency.

### 7.2 Contention-reduction work exists, has big numbers, and will make us *slower*

This is an important "do not do" with a measured regression:

- **pytorch/pytorch#194078** "[inductor] Extend partitioned scatter to additional ops" (merged
  2026-08-19, MI308X): with `hot` = fraction of writes forced onto one row, speedups are
  1.82× (hot 10%) → 10.45× (hot 100%). **At hot ≈ 0 — our case — the baseline is already fast and
  there is nothing to recover.**
- **pytorch/pytorch#168073** "[ROCm] partitioned buffer approach for scatter add" (merged
  2025-11-18): MI300 all-to-one 85.52 → 3.55 ms (8.03×); H100 all-to-one 19.84 → 0.756 ms. **But at
  low contention on H100 it goes 0.549 → 1.724 ms — a 3.1× REGRESSION.** It shipped **disabled by
  default** for exactly this reason: *"this could potentially benefit H100 on worst-case examples
  but would degrade perf in the best case."* **Molecular graphs are that best case. Leave
  `partitioned_scatter_enabled=False`.**

### 7.3 The 2.14 CUDA `index_add_` regression tells us where the TMA path's crossover is

**pytorch/pytorch#200940** (closed 2026-10-14, GB300, `n_dst=1M`, `n_idx=4M`, fp32):

| row size | 2.13.0 | 2.14.1 | 2.16-dev |
|---|---:|---:|---:|
| 16 B | 0.105 ms | **0.801** | 0.804 |
| 32 B | 0.229 | **0.813** | 0.813 |
| 64 B | 0.465 | **0.831** | 0.831 |
| **256 B** | 1.791 | **0.889** | 0.889 |

The TMA kernel's time is **flat** from 16 B to 256 B: it loses badly below 256 B and wins ~2× at
256 B — which is why `kFastScatterMinSliceBytes = 256` was added afterwards. **Our rows are
1200 B (`d_h=300` fp32) to 8192 B (`d_h=2048` fp32) — comfortably in the regime where the TMA path
is the right one.** ⚠️ But measured at 150–300× our row count, so the launch floor will dominate
at our size and these ratios will not transfer.

Also: **pytorch/pytorch#182818** "Add CuTeDSL scatter_add override for the expanded-1D-index
pattern" (2026-05-07, **not merged**) is named after our *exact* pattern and measures on **B200**
`scatter_add_(0, idx.unsqueeze(-1).expand(-1,D), src)` vs `index_add_(0, idx, src)`:
**5.0× at D=128–256 fp32, 4.3× at D=1024 fp32, 10.8× at D=256 bf16.** ⚠️ 500 K src rows (≈37× our
size), unmerged, B200-only. It explicitly lists the fall-through cases, independently confirming
every gate in §1 #2: *"non-expanded index, dim != 0, non-inner-contiguous inputs, deterministic
mode, unsupported dtypes, row_bytes not 16-aligned fall through to aten."*

### 7.4 Chemprop's own signals

- **Discussion #1019** (2024-09-12, v1.7.1, RTX 4090, 1 M molecules): user reports GPU utilisation
  <10% on predict, <30% and intermittent on train. Maintainer **JacksonBurns**: *"it's not totally
  unexpected... Chemprop's update operation doesn't work well on GPUs (the number of add operations
  for each batch of graphs is inconsistent between batches, since the number of edges varies), but
  v2 uses `scatter` which makes it less inefficient."*
- **#457** (2023): JacksonBurns: *"once the entire dataset is on the GPU and the entire network is
  there as well, we are maximizing our use of it."* — **this claim is inconsistent with #1019 and
  with the launch-overhead analysis.** Residency ≠ utilisation.
- **#477** "GNN backend refactor question" (2023-10-02): davidegraff's rationale for *not* adopting
  PyG/DGL, on speed and control grounds, with *"I'm happy to eat a shoe and be proven wrong if it
  turns out that reimplementing Chemprop using one of those libraries further improves the speed."*
  He also notes **`BatchMolGraph` ≈ `torch_geometric.Data` with renamed attributes** — the data
  structures are already isomorphic, so **swapping only the aggregation kernel is architecturally
  cheap**, which is good news for every proposal in this brief. (His "60% faster" figure is about
  *featurisation*, not message passing.)
- **`itakigawa/pyg_chemprop`** (unmaintained since 2022-06): the only end-to-end measured
  Chemprop-architecture speedup I found. ogbg-molhiv, 32,901 molecules, batch 50, A100-PCIE-40GB,
  **chemprop features, GPU: 15 s → 7 s per epoch = 2.1×**. Derived: 658 steps/epoch →
  **22.8 ms/step (v1) vs 10.6 ms/step**. ⚠️ **Do not cite that repo's "1743 s without
  torch_scatter" row as evidence for torch_scatter** — `pyg_chemprop_naive.py` is a readability-
  oriented Python loop, per its own README, not native `scatter_add_`.
- **Dataloading is a separate, real bottleneck:** chemprop#1234 added `n_workers` to `make_dataset`
  after a contributor identified descriptor calculation as *"the main bottleneck"*; note
  `num_workers` **defaults to 0**. Externally, NVIDIA's **`cuik-molmaker`** C++ featuriser gave
  **2.2× end-to-end finetuning / 2.9× inference and ~34% memory reduction** in arXiv 2510.12719 at
  batch size 32 (RTX A6000) — a *featurisation* win on a GROVER-family pipeline, but it quantifies
  how large the non-GPU fraction is in this class of workload.

**Net: Chemprop maintainers have never profiled the message-passing kernels.** There is no
maintainer position to argue against, only an acknowledgment that utilisation is poor and that the
variable-size scatter is why.

### 7.5 `torch.compile` on these ops: Inductor fuses them *into atomics*, and dynamic shapes eat the win

- **pytorch/pytorch#113232** (opened by GeoT's author): *"the GAS [Gather-Apply-Scatter] model can
  be fused by the Torch Inductor, [but] it currently reduces to an **atomic_add** operation...
  eliminating atomic operations seems feasible only through data transformation, particularly
  because the edge_index might be unsorted."* His four blockers: `segment_reduce` is 1-D-oriented;
  it can't be fused with the gather; `segment_coo` isn't Inductor-compatible; none is easy to
  codegen. The companion **PyTorch Sparse(GNN) Compiler RFC** claims *"the fused scatter_add kernel
  has accounted for over 80% of the running time of typical GNN models"* and targets 4–7× kernel /
  2–3× end-to-end — **claimed, no shapes given, and nothing shipped.** jansel's suggested fixes
  (2-D `segment_reduce`, a dedicated Inductor lowering, a `scatter_mode="sorted_add"` with
  thread-local accumulation) remain unimplemented.
- GeoT's README puts it bluntly: *"In torch.compile, the scatter and message will be fused into a
  **fully-atomic SpMM**, which largely affects the performance."*
- **pytorch/pytorch#94640** (closed, **59 comments**) is a near-perfect match for Chemprop's
  situation, with verbatim logs:
  ```
  torch._dynamo hit config.cache_size_limit (64)
    function: 'scatter'  reasons: ['dim_size == 14737']
    reasons: ["tensor 'edge_index' strides mismatch at index 0. expected 52637, actual 51470"]
  ```
  **Measured outcome: eager 0.7234 s/epoch vs compiled 0.8730 s/epoch — `torch.compile` was 1.21×
  SLOWER**, because every batch brought a new `dim_size` and blew the 64-entry cache.
  ezyang's clean numbers on the same benchmark (A100): **static + CUDA Graphs 1.267×, static
  without CUDA Graphs 1.102×, dynamic shapes 1.021×.**
- **Official guidance.** PyG's "Compiled GNNs" doc reports **GCN 2.83× / GraphSAGE 2.24× /
  GIN 2.29×** — ⚠️ on *one static* synthetic graph, 10 K nodes / 200 K edges (degree 20), feature
  dim 64. NVIDIA's PyG-compile guide measured **30–35%** on real pipelines (g6.4xlarge,
  PyTorch 2.7 + CUDA 12.8) and gives three non-obvious, cheap fixes that apply directly to
  Chemprop:
  1. **Wrap the learning rate in a tensor** — "torch.compile treats floating-point scalar arguments
     as compile-time constants", so an **LR scheduler forces a recompile every step**. Chemprop uses
     an LR schedule. This is a real, one-line bug.
  2. Pass `output_size` to `torch.repeat_interleave` to avoid a host sync.
  3. Accumulate metrics as GPU tensors; `global_mean_pool`-style ops cause graph breaks via device
     syncs without an explicit batch size — relevant to `agg.py`'s `batch.max().int() + 1`.
  NVIDIA also says explicitly to **avoid CUDA Graphs for GNN training with neighbour sampling**
  "because variable mini-batch sizes make repeated kernel recording counterproductive."
- **The gap nobody has filled:** bucketing + padding to a handful of static shapes, to get CUDA
  Graphs *and* stable dynamo guards. I found **no PyG/PyTorch guidance recommending it and no
  benchmark of it** — yet it is the obvious reconciliation of "CUDA Graphs give 1.267×" with "our
  shapes vary", it is what TorchMD-Net did to get 2–10× on small molecules, and it composes with
  the `d_h` 16-byte alignment requirement from §1 #2.

### 7.6 Triton for segment reductions: the honest answer is "don't"

No well-known production Triton segment-sum/SpMM kernel repo exists, and the signal is consistently
negative:

- **lumen#225** "[triton] scatter_add is slower than torch (0.02× at worst)" (closed 2026-11-01),
  **H100, measured**:

  | dtype | n | Triton | torch | ratio |
  |---|---:|---:|---:|---:|
  | fp32 | 262,144 | 118.0 µs | 21.0 µs | 0.18× |
  | fp32 | 1,048,576 | 455.8 µs | 24.9 µs | 0.05× |
  | fp32 | 16,777,216 | 7644.4 µs | 274.0 µs | **0.04×** |
  | bf16 | 4,194,304 | 1836.1 µs | 43.5 µs | **0.02×** |

  Note torch's absolute time is **~21 µs flat from n=65 K to n=1 M — a launch-bound floor, which is
  exactly our regime.**
- **GeoT's author** — who hand-wrote a tuned CUDA segment-reduction library that beats
  `scatter_reduce` by 1.68× — concludes Triton *"struggles with the unique demands of GNN
  workloads"* due to **warp-level divergence** under indirect loading plus variable sparse
  reduction, and that *"Triton's current implementation using the CSR algorithm may actually
  underperform compared to atomic-based versions."*
- The Inductor RFC's Triton blocker: *"The existing Triton scan primitive doesn't support multiple
  inputs and doesn't have a 'carry' output"* — segment reduction can't be expressed well.
- Triton's own atomics issues are correctness/compilation, not performance
  (triton-lang/triton#891 fp16 `atomic_add` segfaults during compilation; PyTorch #192999 / #179833
  fixing Inductor's `tl.atomic_add` lowering).

**Assessment: a hand-written Triton segment-sum for Chemprop is high-risk, low-expected-value.**
Hopper+ hardware atomics are good, our contention is near-zero, and everyone who tried landed on
"CUDA/CuTeDSL yes, Triton no". **This partially undercuts §2.6(b)** — but note the Hashimoto
gather-sum is *not* a segment reduction (no atomics, no variable-length warp reduction, fixed ≤3
loads per output row), which is precisely the shape Triton *is* good at. That distinction is worth
testing rather than assuming.

### 7.7 Determinism and the fp16/bf16 packed-atomic trap

**Determinism has no cheap escape for us.** pytorch/pytorch#203118 (open, 2026-11-02) states the
behaviour authoritatively: under `torch.use_deterministic_algorithms(True)`, *"index_put,
index_copy, index_add, scatter, scatter_add and scatter_reduce route CUDA (and XPU) inputs to a
sort-based implementation, and index_reduce, put_ and scatter_reduce(prod) raise."* The new
`assume_unique=True` overload **does not apply to Chemprop** — our aggregation indices are
inherently non-unique (every atom has 2–4 incoming bonds), and the PR says *"for non-unique indices
results won't be deterministic even under deterministic mode."* Inductor also keeps these as eager
fallbacks in deterministic mode, so compiling buys nothing there.
**→ Today, deterministic Chemprop pays a cub radix sort. §2.5's Hashimoto `Bᵀ` gather-sum is the
only way out, because it has no duplicate-index writes at all.**
(Aside: `_scatter_via_index_put` *also* checks `index.stride(i) != 0` to avoid "creating large
intermediate tensors" — so **`.repeat` penalises the deterministic path too.**)

**The fp16/bf16 trap, with the best writeup anywhere** — FluxML/NNlib.jl#854 (closed 2026-10-24),
**RTX 5090, measured**, `src = randn(T, 1024, 20352)`, `r` = columns colliding on one output column:

| r | fp32 | fp16 | bf16 |
|---|---:|---:|---:|
| 0 | 0.106 ms | 0.184 | 0.183 |
| 1000 | 0.109 | **8.68** | **8.42** |
| 4000 | 0.125 | **31.4** | **27.8** |

Per-instruction breakdown for r=4000 bf16: `atom.acq_rel...bf16` 22.9 ms; `atom.relaxed` 5.0 ms;
`red.global.add.noftz.bf16` 8.6 ms; **two adjacent elements via `red.global.add.noftz.bf16x2`
0.054 ms**; one element + a zero partner via `bf16x2` **0.121 ms** (≈ fp32's 0.125 ms).

> *"The paired instruction is 200–400× faster than the single one, even with one element per thread,
> and as fast as Float32. **PyTorch's `index_add_` and `scatter_add_` do the latter for this reason
> (`fastSpecializedAtomicAdd` in `ATen/native/cuda/KernelUtils.cuh`). They add the value and a zero
> to the aligned 32-bit word that holds the element.**"*

So PyTorch already does the right thing — **but only when the packed path is reachable, which
depends on alignment** (hence the `d_h % 8 == 0` requirement in §1 #2). That path has had real bugs
*this month*: #203089 "Fix fp16/bf16 packed atomic adds writing outside `self`" (merged 2026-11-02)
and #203243 "Pad packed fp16/bf16 atomic adds with -0.0" (merged 2026-11-03);
rusty1s/pytorch_scatter#269 "Possible bug in half-precision scatter" has been **open since 2022**.

**→ If Chemprop moves to bf16: (a) `d_h % 8 == 0`, (b) require PyTorch ≥ the Nov-2026 fixes,
(c) accumulate the aggregation in fp32 even under autocast. Note `autocast` does NOT promote
`scatter_add_`/`index_add_`** — they run in whatever dtype `H` happens to be. (No ecosystem report
of actual fp16 *overflow* in GNN sum aggregation; degrees are small. The concern is precision and
speed, not overflow — consistent with §6.)

### 7.8 Where the published hotspot breakdowns put the time

**PyG #4891 "[Roadmap] CPU Performance Optimization for PyG"** (Icelake Xeon):

| model + dataset | hotspots |
|---|---|
| GCN + Reddit (L=1, h=16) | `DataLoader` **83.49%**, `scatter_add_` 8.47% |
| SAGE + ogbn-products | `scatter_add_` 27.61%, `DataLoader` 25.70%, `aten::index` 20.26% |
| **GAT + CiteSeer** | `scatter_add_` **30.91%**, `scatter_max` 24.54%, `aten::mm` 10.34%, `index_select` 6.71% |
| GraphSAGE + mag240m | `DataLoader` **71.27%**, `spmm_mean` 14.91% |

**The CiteSeer row is the one at our magnitude** (3,327 nodes / 9,104 edges): scatter 31% +
gather 7% + `mm` 10%, i.e. **scatter beats `mm` 3:1 — at h=16.** But `mm` FLOPs grow as `d_h²` while
scatter traffic grows as `d_h`, so at `d_h=300–2048` the balance must shift hard toward the Linear.
**Nobody has measured that crossover. It is the single most useful thing the local benchmarking
could establish**, because it decides whether to invest in scatter kernels at all (§3.1 predicts
the GEMM wins decisively once the index bloat is gone).

And on fusing the gather into the reduction — **PyG Discussion #4728**, same GCN model, two formats:
`adj_t` (CSR) total **14.98 ms** (`torch_sparse::spmm_sum` 11.85 ms) vs `edge_index` (COO) total
**111.02 ms** (`aten::index_select` **60.89 ms = 54.8%** + `scatter_add_`) → **~7.4× end-to-end for
CSR SpMM over COO gather+scatter.** ⚠️ Single large graph, `cached=True`. But it is the strongest
signal that **never materialising `H[index]` is where the structural headroom is** — which is
exactly what §2.6(b)/(c) propose, and which Inductor will *not* do for you.

---

## 8. Apple Silicon / MPS / Metal

Reference points: **PyTorch 2.15.0 (2026-10-28)** / 2.16-dev, **MLX 0.32.4**, macOS 15+ required.

Of the five designs in §1, the ones that work on Apple Silicon are **#1 (Hashimoto, pure-PyTorch
bucketed form)**, **#2 (index fix)**, **#4 (numerics)** and **#5 (dense `bmm`)**. The megakernel
(#3) is CUDA-only; the Apple analogue is a hand-written Metal kernel (§8.3).

### 8.1 🛑 Three hard "no"s, settled with primary sources

**(a) There is no GPU SpMM anywhere in the Apple stack. The sparse-matmul reformulation is a dead
end on Apple Silicon — rule it out explicitly.**

The SDK headers on this machine settle it.
`MetalPerformanceShadersGraph.framework/Headers/MPSGraphSparseOps.h` contains **exactly two
things**: `MPSGraphSparseStorageType` (COO/CSC/CSR, macOS 12+) and
`MPSGraphCreateSparseOpDescriptor` + two `sparseTensorWith...` **constructors**. That is the entire
sparse API — **no sparse matmul, no SpMM, no SpMV, no sparse reduction.**
`MPSGraphMatrixMultiplicationOps.h` declares only dense `matrixMultiplication`, `HammingDistance`
and SDPA. The lower-level **MPSMatrix framework has zero `MPS*Sparse*` interfaces at all**.
Apple exposes SpMM **only on the CPU**, via Accelerate's Sparse BLAS (`SparseMultiply`,
`SparseMultiplyAdd`, CSC-oriented, multithreaded).

And on the PyTorch side: **`torch.sparse_csr_tensor(..., device="mps")` fails today** —
*"Could not run 'new_compressed_tensor' from the 'mps:0' device"* (tracking issue
pytorch#196192, open 2026-09-07, with a four-PR plan). So you cannot even *build* the sparse tensor
on the device you want to run it on. Published sparse-GEMM work for Apple Silicon is all
CPU/NEON (arXiv 2510.06957, ternary CSC + NEON, ~50% of peak).

> **Important distinction that saves the §2 proposal:** the **bucketed Hashimoto form is not a
> sparse matmul.** It is `index_select` + `add` — dense ops on dense tensors with an integer index.
> It needs **no** sparse support, runs on MPS today, and is the *reason* to prefer it over a
> `torch.sparse.mm` reformulation on this machine.

**(b) `torch.use_deterministic_algorithms(True)` + float scatter-add on MPS raises an error — there
is no slow-but-correct path.** From `aten/src/ATen/native/mps/operations/Indexing.mm` on main:
`index_put_` with `accumulate=True` on a float/complex dtype calls
`alertNotDeterministic("index_put_with_accumulate_mps")` (*"Metal atomic-add is non-associative for
floating/complex types"*), and **`index_add_` under determinism falls back to
`index_put_(accumulate=True)` — i.e. onto the op that raises.** Added in pytorch#195754
(merged 2026-09-05). `index_put_` with `accumulate=False` dispatches `index_put_serial_*` —
**one thread, fully serialised**. **→ There is no deterministic GPU segment-sum on MPS.** The only
routes are CPU, or a gather-sum formulation with no duplicate-index writes — which is exactly
§2.6(a).

**(c) fp8 and sub-8-bit buy storage, not speed, on every shipping Apple GPU.** Metal 4.1
(Xcode 27 / macOS 27) does add a tensor path (`MPP matmul2d` over `cooperative_tensor`) plus fp8,
fp4 and block-scaled MXFP4, and `MTLTensor` can carry a scale plane that *"supports the popular FP8
E8M0 block-wise scale factor format"*. But **only M5 / A19 ("Apple10") have GPU Neural
Accelerators**; M4 and earlier *"support the low-precision formats functionally but do not
accelerate them."* The measured numbers:

| chip | FP16 baseline | W8A16 fp8 | fp8×fp8 | INT8×INT8 |
|---|---|---|---|---|
| M5 (10-core GPU) | 14.2 TFLOPS | 15.95 (**1.12×**) | 16.35 (**1.15×**) | — |
| M5 Max (40-core) | 63.84 TFLOPS | 64.30 (**1.007×**) | 64.92 (**1.017×**) | **122 TOPS (≈1.9×)** |

*"FP8 has no compute advantage over FP16 on this chip."* ⚠️ Community benchmark, thermal throttling
affected the compute-bound rows — but it matches Apple's own framing. PyTorch 2.15 adds
`torch.float8_e4m3fn` on MPS with `F.scaled_mm` (tensorwise + rowwise) and makes **no
hardware-acceleration claim and reports no benchmarks** — it is framed purely as 50% memory
reduction. Rigel (arXiv 2606.12765) reverse-engineering M4 Max finds `matmul2d` executes **entirely
on shader cores with no dedicated matrix datapath**, accumulates in ≥fp32, and **fp8 (E4M3) is
emulated at 0.94× fp16 throughput despite reading half the bytes.**

**→ The only Apple-side low-precision wins are (i) INT8 matmul ≈2× on M5+, and (ii) narrower
storage of `H` to relieve bandwidth — which plain fp16/bf16 gives you today on every Mac.**
Consistent with Apple's own LLM data: M5 vs M4 is **up to 4× on time-to-first-token (compute-bound)
but only 19–27% on subsequent tokens (memory-bound)**, tracking the 153/120 GB/s bandwidth ratio
(+28%), not the compute ratio.

### 8.2 What *is* good on MPS (and the exact gotchas)

**These ops are all native Metal kernels now — no CPU fallback, no MPSGraph:**

| op | status | PR |
|---|---|---|
| `scatter_`/`scatter_add_`/`scatter_reduce_` (incl. `.two`) | native Metal, 5 kernels, MPSGraph dropped | #184057 (2026-05-17) |
| `index_add_`, `index_select` | native Metal; MPSGraph gather/scatter removed | #187109 (2026-06-15) |
| `segment_reduce` native MPS forward | **still open, not in any release** | #197014 |
| fp8 scatter kernel instantiation | merged | #196955 |

From the #187109 design notes (the single most useful document here):

- **`index_add` accumulates via `AtomicType<T>::atomic_add`.** `float`/`cfloat` have a **native
  Metal atomic add**; **`fp16`/`bf16` do not — they use a compare-and-swap loop that "collapses
  under index contention."** The kernel therefore **upcasts accumulation to fp32 only when
  `num_indices / size(dim) ≥ 16`.**
  > **Direct consequence for the D-MPNN: `2m/n ≈ 2.06`, far below 16, so an fp16/bf16
  > `index_add_`/`scatter_reduce_` takes the CAS-loop path, not the upcast path.** At contention ~2
  > the CAS should mostly succeed first try, so it is probably fine — but **you will not get a
  > clean 2× on the scatter from halving the dtype**, unlike a pure-streaming kernel.
- **`index_select` is a plain gather with no atomics**, and the contiguous case uses a **3-D-grid
  fast path that coarsens each slice to the widest power-of-two copy unit.** **This is the good
  path, and it is precisely the only primitive the bucketed Hashimoto form needs.**
- Index bounds are checked **on device** by a separate one-pass kernel (per-element
  branch-and-report *"cost ~20% on `index_add`"*), so expect ~one extra cheap launch per indexed op.

**bf16 on MPS: fully supported** (macOS 15+ is now the minimum for MPS at all, as of 2.15), and
**much faster than CPU bf16, which is the real story.** From #197014's own M4 Pro table, 4.2 M
elements / 1024 segments: **CPU bf16 17,963 µs vs MPS bf16 77 µs (233×)**, where fp32 is only 12.7×
— CPU bf16 is ~5.7× slower than CPU fp32 while MPS bf16 ≈ MPS fp32. **So if any part of the
pipeline falls back to CPU, bf16 is a trap.** bf16 is also the least-tested MPS dtype (recent bugs:
#200545, #199132, #195910); MPSInductor now explicitly **computes in fp32 after casting to
fp16/bf16** (#200559). **No fp64 ever** — MSL has no `double`, and `PYTORCH_ENABLE_MPS_FALLBACK=1`
doesn't help because the tensor can't be allocated on MPS at all.

**Large GEMM on MPS — the number that matters.** From pytorch#198182 (open, full methodology,
**M4 Max**, macOS 26.6.2, torch 2.15.0.dev20260921):

- **MPSGraph reaches 7.8–12.0 TFLOPS** from 512³ upwards.
- PyTorch's **native Metal `matmul`/`addmm` kernels run at 1.1–1.4 TFLOPS on essentially every
  shape** — a **6.0–9.8× deficit, median 7.9×** (16×16 threadgroup tiles, one output element per
  thread, no register blocking, no simdgroup matrices).
- *"Half precision gives the Metal kernel no speed-up; float16 and bfloat16 run at the same rate as
  float32, since inputs are converted to float in threadgroup memory."*
- MPSGraph timings vary up to **28% run-to-run**; the Metal kernel up to 11%. **Budget for that in
  any benchmark.**

**By default a `d_h × d_h` `Linear` with `d_h ≤ 2048` goes through MPSGraph — the fast path.**
`use_metal_mm` only diverts for integer `mm`, some complex cases, certain strided outputs, M1/M2
matmuls with `K > 32768` (where MPSGraph returns *wrong results* — #183535, #195163), and anything
with `PYTORCH_MPS_PREFER_METAL` set. **→ Do not set `PYTORCH_MPS_PREFER_METAL`. And if you write
your own fused kernel you are competing against 7.8–12 TFLOPS, not 1.3.**
Corroborating: ~13.3 TFLOPS fp32/fp16 `torch.matmul` on M4 Max 40-core; Rigel measures tiled fp16
`matmul2d` peaking at **14.8 TFLOP/s ≈ 46% of the scalar-ALU roofline**. fp16 on MLX/MPS gives only
**~2×** over fp32 vs CUDA's 5–6×.

**Memory bandwidth by chip** (peak theoretical; achieved on an irregular gather is a fraction):

| chip | BW | | chip | BW |
|---|---|---|---|---|
| M1 / M2 / M3 | 68 / 100 / 100 GB/s | | M4 | 120 GB/s |
| M1–M2 Pro | 200 GB/s | | M4 Pro | 273 GB/s |
| **M3 Pro** | **150 GB/s (25% regression vs M2 Pro)** | | M4 Max | **546 GB/s** (some 400 GB/s bins) |
| M1–M3 Max | 400 GB/s | | M5 / Pro / Max / Ultra | 153 / 307 / 614 / 1,228 GB/s |
| M1–M2 Ultra / M3 Ultra | 800 / 819 GB/s | | M4 Ultra | **never released** |

Caveats: the **Pro tier is the floor** (M3 Pro at 150 GB/s is 2.7× below M3 Max), so "works on a
MacBook Pro" is a much weaker claim than "works on an M*-Max"; and M4 Max binning means you cannot
assume 546 GB/s from the name.

**Unified memory has two consequences for this design.** (a) **Batch collation on CPU is nearly
free to hand to the GPU — there is no reason not to do the edge permutation / CSR construction on
the CPU at collate time**, which is exactly what §2.6(a) requires. (b) You are competing with the
browser, Xcode and the compositor for the same bandwidth.

**Correctness hygiene is not optional here.** Bugs in this exact op family are frequent and mostly
2026-dated: #195074 (`scatter_add_`/`index_add_`/`index_put_(accumulate)` wrote to the *wrong
elements* on a misaligned bool view), #163327 (`scatter_add_` incorrect with non-zero offset
slices), #189969 (`index_fill_` wrote outside the tensor view), #175191 (`scatter_` wrong for
non-contiguous src/index — MPS *and* CUDA), #176159, #161524. **Pin a version and add a CPU-vs-MPS
numeric equivalence test.** Also: non-contiguous ops on MPS cost >2× contiguous (#136003).

### 8.3 `torch.compile` on MPS — it exists, and there is a critical correctness warning

**MPSInductor shipped as a named highlight of PyTorch 2.8.0** (pytorch#157957, design #150121). The
widespread "Inductor isn't supported on MPS, it needs Triton-on-Metal" advice is **stale 2023-era
advice**. It **does not use Triton**: Inductor's Metal backend **codegens MSL directly and calls
`torch.mps._compile_shader`** (#143893), sidestepping the Triton-on-Metal problem entirely.
2.15 reports **MoE decode 5–6.5× faster than 2.14** with `torch.compile` on M3 Ultra.

> ### 🛑 **Read before trusting any `torch.compile`-on-MPS number in the local benchmark**
>
> **pytorch#201455: "[MPSInductor] Silently incorrect result: `index_add` after a reduction adds the
> reduced value once per reduction thread."** When MPSInductor **fuses a reduction with
> `index_add`/`scatter_add`**, `threadgroup_sum` leaves the reduced value in *every* thread and each
> then `atomic_add`s it — **the result is multiplied by the reduction size.** The reporter hit it on
> *"a model that averages 64 per-atom energy channels and then sums atoms into molecules with
> `index_add`: compiled energies were exactly 64× the eager ones."*
>
> **That is structurally a Chemprop `MeanAggregation` followed by a molecule-level sum.**
> Affected: **2.14.0, 2.14.1 and nightlies through 2026-10-18** (widths ≥32 on nightly, ≥8 on
> 2.14.0). **Not** eager, `backend="eager"` or `backend="aot_eager"`. Fixed by #201460, merged
> **2026-10-21 — seven days before the 2.15 release.** **Verify empirically whether the installed
> build has it; the reproducer in the issue is three lines.** A `torch.compile` speedup that is
> silently computing `64×` the right answer is worse than no speedup.

Other open MPSInductor bugs as of Nov 2026: #203248/#203943 (bool reductions), #203179/#202598
(int8/16 overflow differs from eager), **#202423 (silent truncation of kernels over 2³² elements)**,
#201441 (bounds check rejects negative indices from `arange` under dynamic shapes), #192096,
#169779. Recently fixed: #201377, #199642 (silently wrong recomputed `x²+y²` through `F.pad`),
#201114.

### 8.4 Custom Metal kernels — more mature than expected; this is the real lever

Three routes, increasing ceremony:

1. **`torch.mps.compile_shader` — Metal JIT from Python, one line. Public, in `torch.mps.__all__`.**
   ```python
   lib = torch.mps.compile_shader(
       "kernel void f(device float* out, constant float& v, uint i [[thread_position_in_grid]]) { out[i] = v; }")
   lib.f(x, 3.14)
   ```
   It runs `_embed_headers` against `torch/include`, so you can `#include <c10/metal/atomic.h>` and
   use the **same `AtomicType<T>::atomic_add` the in-tree kernels use**. `threads=[N,1,1]` is
   *total* threads; `group_size` is threads per threadgroup. Limitation: no `[[threadgroup(N)]]`
   parameters — declare threadgroup arrays in the body. There is an official PyTorch
   **"metal-kernel" agent skill** documenting the workflow. **This is the mechanism MPSInductor
   itself uses, and it is the right starting point.**
2. **`torch.mps.load_metallib`** — load a precompiled `.metallib`, *"useful for loading Metal
   libraries compiled ahead of time or generated by external tools (e.g. Triton, MetalASM)"*.
3. **`torch.utils.cpp_extension` with `.mm` (Objective-C++)** — the classic route; Apple's WWDC23
   session 10050 covers dispatching custom shaders on PyTorch's MPS stream.

**The recipe for a fast *and* deterministic message step on MPS:** sort/permute edges at collate
time (the adjacency is fixed per batch, collation is on CPU, unified memory makes the handoff
free), then run a **gather-sum kernel with one threadgroup per output row — no atomics,
deterministic, coalesced, and it reads the index once as `[2m]` int32 instead of `[2m, d_h]`
int64.** That is §2.6(b) specialised to Metal, and it simultaneously solves §8.1(b).

### 8.5 MLX

- **No `mx.segment_sum` and no `mx.scatter_add`.** Scatter-add is reached via
  **`mlx.core.array.at`**: `x = x.at[idx].add(y)`. (Note `x[idx] += y` applies **only one** update
  per duplicated index — a silent-wrong-answer trap.)
- **MLX's scatter is atomic-based exactly like MPS.** `mlx/backend/metal/kernels/indexing/
  scatter.h` calls `op.atomic_update(...)`; `atomic.h` shows **native Metal atomics only for
  `int`/`uint`/`ulong`/`float`** — fp16/bf16 go through an `mlx_atomic<T>{atomic<uint>}` CAS
  emulation. **→ MLX gives no algorithmic advantage over MPS for the segment-sum.** The easy wins
  (#709 "up to 10× faster scatter", #662, #683) all landed in 2024.
- **`mx.fast.metal_kernel`** is slightly *nicer* ergonomics than `compile_shader`:
  `atomic_outputs=True` makes outputs use Metal atomics, and with `init_value=0` you get
  `include_self=False` semantics for free; `template=[("T", mx.float32)]` for dtype generics;
  autograd via `@mx.custom_function` + `@kernel.vjp`; `compile_options.math_mode` ∈
  safe/relaxed/fast. (Known bug: #4603, wrong output rows on M3 Ultra / macOS 27 under
  `MathModeSafe`.)
- **Genuinely interesting MLX primitives for this workload:**
  - **`mx.gather_mm(a, b, lhs_indices, rhs_indices, *, sorted_indices=False)`** — *"performs a
    gather of the operands with the given indices followed by a (possibly batched) matrix
    multiplication... more efficient than explicitly applying a `take()` followed by a
    `matmul()`."* **This is the fused gather→GEMM shape §2.6(c) wants, it exists today on Apple
    GPUs, and the `sorted_indices` flag rewards pre-sorting the edge list.**
  - `mx.segmented_mm(a, b, segments)` — segments the *contraction* dimension, so **not** a
    drop-in segment-sum.
  - `mx.block_masked_mm(a, b, block_size=64)` — block-sparse GEMM, **only `block_size` 32 and 64**.
    `B` at 1.4 nnz/row is far too sparse to benefit.
  - `mx.quantized_matmul(..., group_size=64, bits=4)` — affine quantization at 2/3/4/5/6/8 bits
    (plus 1-bit on Metal), group sizes 32/64/128; mxfp8 and nvfp4 quantization landed in v0.30.3
    (Jan 2026) on Metal and CUDA. **But: `quantized_matmul` is 14–41% *slower* than
    dequantize + bf16 matmul at prefill sizes (M > 512) — MLX issue #4621. Our `M = 13,600` is
    firmly in that regime, so 4-bit quantizing `W_h` would likely be slower, not faster.**
- **`mx.compile` fuses element-wise and broadcast chains but does NOT fuse reductions with
  preceding ops** (#4355), and fusion **stops at a reshape**. So it will not fuse our
  gather→sum→GEMM chain.
- **`mlx-graphs`** (github.com/mlx-graphs/mlx-graphs) provides `scatter_add/max/mean/softmax`, a
  PyG-style `MessagePassing` base, and GCN/GAT/GATv2/GIN/GraphSAGE. **Claims "up to 10× speed
  improvement with respect to other frameworks on large datasets."** ⚠️ **v0.0.10, 226 stars, 21
  open issues, no published releases, last push 2026-10-19 — research-grade, and the 10× claim is
  self-reported with no independent reproduction found.**

### 8.6 Does anyone run GNNs fast on Apple Silicon? Honest answer: barely, and nobody has published a credible benchmark

- **PyG on MPS** is "supported" in the sense that examples were updated (pyg#7770, #7784, #7781),
  but real workloads hit gaps: **#6837 "Training TGN with MPS not supported on Mac M1 max" is still
  open from 2023**; #10453 (2025-09) is still improving MPS support. A field report describes the
  practical workaround as *"use CPU for message passing while using MPS for the decoder."*
- **`torch_scatter`/`torch_sparse` have no MPS backend at all** — the published wheel matrix is
  `cpu`, `cu126`, `cu128`, `cu129` only, and there are no arm64 wheels (must build from source,
  rusty1s/pytorch_scatter#241). **So `segment_coo`/`segment_csr` — the sorted, atomic-free kernels
  you would actually want — do not exist for Metal.** `torch_sparse.spmm` is CPU/CUDA only.
  **Good news: modern PyG doesn't need them** (§7.1), and neither does Chemprop.
- Closest prior art for a Metal sparse kernel: `mps-bitsandbytes` ships `spmm_coo`/`spmm_coo_int8`.
- **→ `mlx-graphs` is the only project that even claims fast GNNs on Apple Silicon, and the claim
  is unverified. That is a genuine gap — and an opportunity.**

---

## 9. Export constraints (ONNX / TorchScript)

Reference points: **ONNX 1.23/1.24 (opset 28/29), ONNX Runtime 1.31.x, torch 2.15.**

### 9.1 ⭐ The headline: `index_add_` and `scatter_reduce_` export to *different* ONNX ops, and only one is GPU-accelerated on Apple Silicon

From `onnxscript/function_libs/torch_lib/ops/core.py`, `aten_scatter_reduce`:

```python
reduce_mode = {"mean": "none", "sum": "add", "prod": "mul", "amin": "min", "amax": "max"}
if not include_self:
    cst = op.ConstantOfShape(op.Shape(src), value=value)        # value = 0 for add
    self = op.ScatterElements(self, index, cst, axis=dim, reduction=reduction_init)
result = op.ScatterElements(self, index, src, axis=dim, reduction=onnx_reduce)
```

- **`scatter_reduce_(reduce="sum")` → `ScatterElements(axis=0, reduction="add")`** (needs opset ≥ 16).
- **`include_self=False` costs a whole extra `ScatterElements` pass in the exported graph.** Since
  Chemprop's `self` is already `torch.zeros`, that is pure waste. **Dropping it removes one
  full-tensor scatter from every message-passing step of the ONNX graph** — a free, zero-risk graph
  size/latency win, on top of the eager-mode win in §1 #2.
- **`index_add_` has no direct translation** (`aten_index_add` in onnxscript literally
  `raise NotImplementedError()`) — **but it never reaches the ONNX layer**, because `aten.index_add`
  is in PyTorch's **core-ATen decomposition table** and `_index_add` rewrites it as
  `aten.index_put_(x1, (None,)*dim + (index,), tensor, accumulate=True)`, which *is* implemented and
  lowers through **`ScatterND`**. **So `index_add_` exports fine under `dynamo=True`, just via
  `ScatterND` instead of `ScatterElements` — and that turns out to be the better target.**
- ⚠️ **Standing exporter hazard: `reduce="mean"` maps to ONNX `"none"`** — i.e. it silently produces
  a **wrong graph** (overwrite instead of mean). The comment says *"'mean' doesn't support in ONNX
  1.14 definition"*, and **it is still mapped this way on main.** Chemprop already hit this and
  worked around it in commit `fa77f38e` / #1401. **Nothing prevents regression — add a test that
  compares exported vs eager `MeanAggregation` output, not just that export succeeds.**

**Opset facts** (from `torch/onnx/_constants.py` on main): `ONNX_MAX_OPSET = 23`,
`ONNX_DEFAULT_OPSET = 20`, `ONNX_TORCHSCRIPT_EXPORTER_MAX_OPSET = 20`, `ONNX_MIN_OPSET = 7`.
`ScatterND`: introduced opset 11, `reduction` added **16**, `max`/`min` **18**, still at version 18
in ONNX 1.24. `ScatterElements`: base 11/13, `reduction` **16**, `max`/`min` **18**. Torch's max
opset 23 ≈ ONNX 1.18, so torch is ~5 opsets behind the spec — **not a problem; everything needed
landed at 16/18.** Both specs permit duplicate indices for `add`/`mul`/`max`/`min` (and forbid them
for `none`), and **neither defines an accumulation order — so bit-exactness across runtimes is not
guaranteed.**

### 9.2 ONNX Runtime by execution provider

**CPU EP — good for this shape, but fp32-only.** `scatter.cc` is parallelised, with a revealing
comment: *"Work units identified by (outer_idx, inner_idx) are completely independent: they never
write to the same output element, even with reductions."* For an `axis=0` scatter on
`[n_nodes, d_h]`: `outer_size = 1`, `inner_size = d_h`, so you get **`d_h` = 300–2048 parallel work
units, no atomics, and deterministic** results (each column reduced serially in index order). That
is actually a very good implementation for our shape. **But `MLFloat16` and `BFloat16` with
`reduction="add"` raise `ORT_NOT_IMPLEMENTED`** — a hard error, not a slow path — for both
`ScatterElements` and `ScatterND`. **→ No fp16 and no bf16 segment-sum on the ORT CPU EP at all.**

**MLX EP (`onnxruntime-ep-mlx`) — the big find, and it changes the ONNX story on Apple Silicon.**
Under the **`onnxruntime` GitHub org**, created 2026-09-16, ~140 PRs in ~7 weeks, on PyPI and
crates.io. *"Run ONNX models on Apple Silicon GPUs with MLX and ONNX Runtime... a **plugin: no ONNX
Runtime fork is needed**. Supported parts of your model run in MLX; unsupported operators stay on
ONNX Runtime's CPU provider."* Requires macOS 14+, Python 3.11+, ORT 1.31+. A Rust ONNX→MLX
translator claims maximal convex partitions and compiles each into an MLX graph.

Now the critical detail, from its claim predicates in `rust/src/ops/shape.rs`:

| ONNX op | MLX EP claim |
|---|---|
| **`ScatterND`** | `reduction` ∈ {`none`, `add`}; `add` requires opset ≥ 16; data **fp32/fp16/bf16** → `mlx_scatter_add_single` |
| **`ScatterElements`** | **only `reduction="none"`** — source comment: *"add/mul/min/max reductions stay on CPU"* |
| `Gather`, `GatherElements`, `GatherND`, `Concat`, `Reshape`, `Transpose`, `Unsqueeze`, `Slice`, `Expand` | claimed |
| `MatMul`, `FusedMatMul`, `GroupedMatMul`, `MatMulNBits` | claimed |

> ### ⇒ The headline conclusion
>
> **PyTorch exports `scatter_reduce_(reduce="sum")` as `ScatterElements(reduction="add")` — exactly
> the one form the MLX EP refuses. So the D-MPNN's segment-sum silently falls back to the ORT CPU
> provider at every message-passing step, cutting the fused GPU partition in half at every depth —
> while `ScatterND(reduction="add")` *is* GPU-accelerated and even supports fp16/bf16, which the
> CPU EP does not.**
>
> Three exploits, increasing effort:
> 1. **Switch to `index_add_` with a 1-D index** → decomposes to `index_put_(accumulate=True)` →
>    **`ScatterND`**, which the MLX EP claims with `reduction="add"`. **Same single edit that fixes
>    the 223 MB index materialisation (§1 #2) and unlocks the PyTorch fast paths. One edit, three
>    independent wins — this is the highest-leverage change in the whole brief.**
> 2. Pass a **`custom_translation_table`** to `torch.onnx.export` mapping
>    `torch.ops.aten.scatter_reduce.two` → an ONNX Script function emitting
>    `ScatterND(reduction="add")`. ~20 lines.
> 3. Contribute `ScatterElements` reduction `add` to the MLX EP — its docs say adding an op is
>    *"one handler + one claim predicate + one registration line in a single
>    `rust/src/ops/<family>.rs` module, with zero `ep.rs` edits."*
>
> Honest caveats from its own README: coverage is *conditional* on attributes/dtypes/shapes;
> *"CPU fallback and small workloads can outweigh GPU gains"*; **one session per thread**; results
> **not bit-identical to CPU**. 7 weeks old, 5 stars. A promising research vehicle, not production
> infrastructure — but it is an official ORT-org effort and **the only credible ONNX-on-Apple-GPU
> path.** (onnxruntime#21271 "[Feature Request] MPS provider" has been **open since 2024-07-06**
> with 23 comments and no first-party MPS EP.)

**WebGPU EP** (Dawn → Metal on macOS): `ScatterElements` *with* reduction **is** implemented —
`i32`/`u32` use native WGSL atomics; **`f32` uses an `atomicCompareExchangeWeak` spin loop** (WGSL
has no float atomics); **`f16` uses a CAS loop on a `u32` packing *two* f16 values**, so adjacent
columns contend on the same word; **`bf16` is not reducible at all** (`ORT_THROW`). At our
contention (~2 updates/slot) f32 CAS retries should be low, so it might be workable — but the EP
has *"limited operator implementation coverage"*, reported latencies at **CPU parity or worse** on
M1 Pro (~80 ms vs ~70 ms CPU), and **#31627: "WebGPU EP crashes on macOS when running concurrent
multi-threaded inference"** (a race in Dawn's Metal backend). **Measure as a fallback, don't plan
on it.**

**CoreML EP: `ScatterND` is not in the supported-operator list** (nor `ScatterElements`). Combined
with CoreML's fixed-shape preference, **wrong EP for a dynamic-graph GNN.**

**Contrib ops: nothing to exploit.** The `com.microsoft` domain has `GatherND`, `FusedMatMul`,
`FusedGemm`, `SparseToDenseMatMul` — **but no scatter/segment reduction op.**
`onnxruntime-extensions` *does* have `ai.onnx.contrib::SegmentSum`, but the implementation is
**fp32-only, CPU-only, TF-style sorted segment_ids, and its docs entry is literally "TODO"**. It is
an API, not an accelerator.

### 9.3 Sparse matmul in ONNX: also a dead end

- ONNX has `SparseTensorProto` (since onnx#2019) with 2-D support, and ORT has
  **`com.microsoft.SparseToDenseMatMul`** (2-D COO or CSR A, N-D dense B). Type constraints:
  `sparse_tensor(float|double|int64|int32|uint64|uint32)` — **no fp16, no bf16.**
- **But the only kernel is `contrib_ops/cpu/math/sparse_dense_matmul.cc` — CPU EP only.** No CUDA,
  no CoreML, no WebGPU, no MLX. It is also behind a build define (`DISABLE_SPARSE_TENSORS`), so it
  is not guaranteed present in distributed wheels. And it is buggy: onnxruntime#31730 (sparse
  initializers cause a type error feeding `SparseToDenseMatMul`).
- **Combined with §8.1(a) (no GPU SpMM anywhere on Apple Silicon) and pytorch#196192 (no
  `sparse_csr` on MPS at all): the sparse-matmul reformulation should be ruled out explicitly
  rather than left as an open option.** Again: **the bucketed Hashimoto form is not a sparse matmul**
  — it is `Gather` + `Add` + `Concat`, which every EP supports.

### 9.4 TorchScript is dead — stop designing around it

- *"TorchScript is deprecated, please use `torch.export` instead."* Warnings added in **2.10**,
  raised to visible `FutureWarning`s in **2.14**. **Active removal has begun: pytorch#203777
  "[deprecation] Deprecate and remove `Serialization.TORCH_JIT`" (opened 2026-11-06), first line:
  *"We are starting to delete TorchScript from the codebase."***
- The legacy TorchScript ONNX exporter caps at opset 20 and the whole `symbolic_opset7..20` family
  is under "Deprecated APIs / deprecated:: 2.6".
- **The ONNX exporter switched to the `torch.export` path by default in 2.9** (`dynamo=True` is now
  the default).
- `torch.onnx.register_custom_op_symbolic()` is **deprecated since 2.6** — the replacement is the
  single `custom_translation_table=` parameter (valid only with `dynamo=True`), which covers
  unsupported ATen ops, `torch.library.custom_op` ops, **and overriding an existing translation**.
  `torch._custom_op` is deprecated with **removal targeted at 2.16** — use `torch.library.custom_op`.

> **Recommendation: retarget Chemprop's export story from `torch.jit.script` to
> `torch.export` + `torch.onnx.export(dynamo=True)`.** This also *lowers* the cost of a custom op —
> the `custom_translation_table` route is much easier than the TorchScript-symbolic era.

### 9.5 The architectural fork that must be called out

- **`torch.library.custom_op` survives `torch.export` and exports to ONNX**, given a
  `register_fake`/meta impl plus an ONNX Script translation. That is the documented, supported path.
- **`torch.compile` output does not export, and that's the right answer** — `torch.export` captures
  a Python-free graph for deployment; `torch.compile` is a guarded in-Python JIT. For an AOT
  artifact the path is `torch.export` → `aoti_compile_and_package` (AOTInductor), not ONNX.
- **Triton kernels do NOT serialize through `torch.export`** — pytorch#151746 (open): a Triton
  kernel wrapped in `torch.library.custom_op` fails on `torch.load` with
  `SerializeError: Unsupported target type for node Node(target='torch.ops.triton_kernel.add.default')`.
- **ORT custom ops are supported on CPU, CUDA and ROCM only. There is NO custom-op hook for the
  CoreML, WebGPU or MLX EPs.** So you cannot write a Metal segment-sum and plug it into ORT as a
  custom op — you would have to extend the MLX EP's Rust registry in-tree, or accept a CPU custom
  op.

> **⇒ The fastest Apple-Silicon path and the exportable path cannot be the same code.** Plan for two
> code paths from the start: a custom Metal/Triton kernel on the eager/compiled training path, and
> the **portable `index_add_` + `Gather`/`Add` formulation** on the exported path. Fortunately
> §2.6(a) *is* the portable formulation and is also ~2× better than today's — so the two paths need
> not diverge much.

### 9.6 fp8 / MX in ONNX: not available

- **ONNX spec:** fp8 (`FLOAT8E4M3FN`, `E4M3FNUZ`, `E5M2`, `E5M2FNUZ`) at **opset 19**; fp4
  (`FLOAT4E2M1`) at **opset 23** for `Quantize/DequantizeLinear`/`Cast`. `Quantize/DequantizeLinear`
  have a **`block_size` attribute** giving blocked scales. But I found **no evidence that ONNX
  defines the full OCP MX set** (`f6E2M3FN`, `f6E3M2FN`, or **`f8E8M0FNU` as a scale type**) as
  first-class element types. The MX spec pairs `f4E2M1FN` data with an `f8E8M0FNU` scale at block
  32 — ONNX has the data type but apparently **not the E8M0 scale type**. *Report as "partial / not
  confirmed."*
- **ORT 1.31 pins ONNX at 1.22.0 (opset 27)**: *"The initial ONNX 1.23 integration was reverted
  pending upstream fixes; this release does not ship that integration's new opset or FLOAT6
  support."* The explicit deferral of **FLOAT6** is the clearest signal that **MX-family support in
  ORT is not there yet.**
- **fp8 compute is GPU-only** (a TensorRT-EP-on-Hopper/Blackwell feature). The CPU EP's precision
  support is INT8 (VNNI/AMX), FP32 and BF16 — **FP8 is not listed.**
- **→ No fp8 and no MX path for ONNX inference on Apple Silicon.** (Contrast eager PyTorch MPS,
  which *does* have `float8_e4m3fn` + `F.scaled_mm` as of 2.15 — but §8.1(c): Metal's fp8 matmul is
  emulated at 0.94× fp16, so it buys storage, not speed.)

### 9.7 Risk classification of this brief's proposals

| proposal | `torch.export` | ONNX export | notes |
|---|---|---|---|
| **#2 `index_add_` + 1-D index** | ✅ | ✅ **better than today** | decomposes to `index_put_(accumulate=True)` → **`ScatterND`**, which the MLX EP GPU-accelerates and which supports fp16/bf16 where `ScatterElements` on ORT CPU hard-errors |
| **#2b drop `include_self=False`** | ✅ | ✅ | removes a redundant full `ScatterElements` from every depth step of the graph |
| **#1(a) Hashimoto bucketed pure-PyTorch** | ✅ | ✅ | lowers to `Gather`+`Add`+`Concat` — supported by **every** EP. **Lowest-risk real win.** |
| #1(b) custom CSR Triton/Metal kernel | ⚠️ custom_op only | ❌ | needs `torch.library.custom_op` + `register_fake` + a `custom_translation_table` entry **and** a pure-ATen decomposition. **Triton kernels do not serialize through `torch.export`** (#151746). **No ORT custom-op hook for MLX/CoreML/WebGPU EPs.** |
| #1(c) CUTLASS gather-GEMM epilogue | ❌ | ❌ | CUDA-only custom op |
| #3 megakernel | ❌ | ❌ | CUDA-only |
| #3b CUDA graphs / bucketed static shapes | ✅ | ✅ | a runtime wrapper, not a graph change — keep an uncaptured path |
| **#5 dense block-diagonal `bmm`** | ✅ | ✅ | `bmm`+`reshape`+`relu`; **fully export-safe on every EP** |
| bf16 message passing | ✅ | ⚠️ | **ORT CPU EP hard-errors on fp16/bf16 `ScatterElements`/`ScatterND` with `reduction="add"`.** Via `ScatterND` the MLX EP does support bf16. |
| fp8 / MX message passing | ⚠️ | ❌ | **ORT 1.31 pins ONNX 1.22 and explicitly defers FLOAT6; no MX element/scale types; fp8 compute is TensorRT-only.** |
| `torch.sparse.mm` reformulation | ❌ on MPS | ❌ practically | no `sparse_csr` on MPS at all; ORT's only SpMM is CPU-only, fp32-only, build-gated, buggy |

**Design guidance:** structure any custom kernel as a `torch.library.custom_op` with a
**`register_fake` meta kernel plus a pure-PyTorch decomposition**, and register an ONNX Script
translation via `custom_translation_table` (the modern replacement for the deprecated
`register_custom_op_symbolic`). Then eager/compiled uses the fast path and export traces the
decomposition. **But accept §9.5: the fastest path and the exportable path will not be the same
code — which is the strongest argument for making the *portable* formulation (§2.6a) the one you
optimise first.**

---

## 10. Falsifiable predictions for the local benchmarking agent

Stated with expected magnitudes so they can be confirmed or killed.

| # | prediction | expected magnitude | how to test |
|---|---|---|---|
| P1 | Replacing `scatter_reduce_` + `.repeat` index with `index_add_` + 1-D index reduces message-step **bytes moved** by **1.80×**, and this ratio is **independent of `d_h`** | 1.80× traffic; wall-clock win somewhere between 1.2× and 1.8× depending on how bandwidth-bound the machine is | compare `d_h=300` and `d_h=2048`; the *traffic* ratio should be identical, which is the fingerprint |
| **P2** 🛑 | **`.expand` will produce WRONG RESULTS on CPU** for Chemprop's exact shape (`strides (1,0)`, `self.size(1) > index.size(1)`) — pytorch#203078, open, reproduces on 2.9/2.15/2.16-dev. A `.expand` timing win may be a wrong-answer win | correctness, not speed | **compare `.expand` output against `.repeat` output elementwise before reporting any `.expand` timing.** Then use `index_add_` instead |
| P2b | Beyond bytes, `.repeat` disqualifies PyTorch's GNN fast paths: CPU `can_use_expanded_index_path` (→ COO→CSR + spmm-reduce) and CUDA `fast_scatter_kernel_eligible` (→ TMA/vectorized scatter). So the `index_add_` win should **exceed** the 1.80× byte ratio on CPU | > 1.80×, possibly much more on CPU (the PR that added the CPU path measured 6.3× on `scatter_add_` for SAGE/Reddit) | time `index_add_` vs `.repeat`-`scatter_reduce_` and check whether the ratio exceeds 1.80× |
| P2c | Dropping `include_self=False` saves a whole extra scatter traversal (`scatter_reduce_exclude_self_helper`) | ~1.3–1.5× on the scatter op alone (it is 2 passes → 1) | `torch.zeros(...).scatter_add_(...)` vs `scatter_reduce_(..., include_self=False)` |
| P2d | **bf16 at `d_h=300` fails the CUDA fast path**: row = 600 B, `600 % 16 = 8 ≠ 0`. Padding `d_h` 300→304 should produce a *discontinuous* speedup | step change at 304, not gradual | sweep `d_h` ∈ {296, 300, 304, 312} in bf16 |
| P3 | The explicit Hashimoto bucketed gather-sum gives a further **2.06× traffic reduction** over the `index_add_` form (**3.72× over current**) | 2.06× traffic; wall-clock 1.3–2.0× at `d_h=2048`, less at `d_h=300` where 4 kernel launches hurt | implement the §2.6(a) 4-bucket version; `nnz(B)/row` should measure **1.38–1.41** on any drug-like set |
| P4 | `nnz(B)/row ≈ 1.38` and row-nnz histogram ≈ {0:10.7%, 1:43.5%, 2:40.7%, 3:5.1%} on any drug-like dataset | within ±3% absolute | direct count from `edge_index` + degrees |
| P5 | The `W_h` GEMM is **compute-bound, not memory-bound** (AI 476 FLOP/byte at `d_h=2048`). The "memory-bound" signal is the index tensor + round-trips, not the math | isolating the GEMM should show it scaling with FLOPs, while the message step scales with bytes | time `H @ W_h.T` alone vs `message()` alone |
| P6 | **bf16 message passing with the current `Σ − H_rev` form will show materially worse error than bf16 with the Hashimoto form**, concentrated on degree-2 atoms | relative error on `M` up to ~1e-2 for the factored form vs ~1e-3 (plain rounding) for Hashimoto; fp32 difference negligible (~1e-6) | compute `M` both ways in bf16 against an fp64 reference; **stratify the error by `deg(src(e))`** — the signal is in degree 2 |
| P7 | Making the reduction fp32 while keeping GEMMs bf16 recovers essentially all of P6's loss | error back to pure-rounding level | bf16 GEMM + fp32 accumulate |
| P8 | Size-bucketed dense `bmm` costs **+2.6% FLOPs at `d_h=2048`** / +17.9% at `d_h=300` with `E_max≈47`; naive padding to batch-max inflates the `W_h` GEMM **3.43×** | as stated | measure `2m` distribution of your batches, then FLOP-count |
| P9 | On MPS, the dense `bmm` formulation (#5) beats every scatter-based formulation, despite more FLOPs | plausibly 1.5–3× if MPS scatter is as weak as expected | direct comparison on this machine |
| P10 | CUDA-graph capture over size-bucketed static shapes gives a large win at `d_h=300` | TorchMD-Net measured **2–10×** for molecules ≤300 atoms; expect the low end (2×) here since Chemprop batches more work per launch | not testable without CUDA; test the `torch.compile` + `mode="reduce-overhead"` analogue |
| P11 | `torch.compile` will recompile per batch unless shapes are bucketed; `dynamic=True` avoids recompiles but gives back most of the fusion win | ezyang measured **1.267× static+CUDAGraphs, 1.102× static, 1.021× dynamic** on PyG GCN. pytorch#94640 measured compiled **1.21× SLOWER** than eager when `dim_size` changed every batch | count recompiles with `TORCH_LOGS=recompiles`; it should equal the number of distinct `(n, 2m)` pairs, and blow the 64-entry cache |
| **P12** 🛑 | **`torch.compile` on MPS may be silently wrong** where a reduction fuses with `index_add`/`scatter_add` (pytorch#201455: result multiplied by the reduction size; reporter saw exactly 64×). Affected 2.14.0/2.14.1 and nightlies through 2026-10-18; fixed 2026-10-21, 7 days before 2.15 | correctness, not speed | run the 3-line reproducer in #201455 on the installed build **before** reporting any MPSInductor number; compare compiled-MPS vs eager-MPS vs CPU for `MeanAggregation` + molecule sum |
| P13 | On MPS, fp16/bf16 `index_add_` takes the **CAS-loop path, not the fp32-upcast path**, because the upcast triggers only at `num_indices/size(dim) ≥ 16` and ours is 2.06. So halving the dtype will **not** halve the scatter time | scatter speedup from bf16 ≪ 2× (possibly ~1×); the gather and GEMM will still benefit | time `index_add_` in fp32 vs bf16 on MPS at fixed `d_h` |
| P14 | On MPS, `Linear` goes through **MPSGraph (7.8–12 TFLOPS on M4 Max)**, not the native Metal kernel (1.1–1.4 TFLOPS). Setting `PYTORCH_MPS_PREFER_METAL` should make it **6–10× slower** | 6.0–9.8× regression, median 7.9× | toggle the env var — this also *verifies* which path you're on |
| P15 | On MPS, fp16/bf16 `matmul` gives **no speedup over fp32** if it lands on the native Metal kernel ("inputs are converted to float in threadgroup memory"), and only **~2×** via MPSGraph (vs CUDA's 5–6×) | ≤2× | time `H @ W_h.T` in fp32 vs bf16 on MPS |
| P16 | **MPS run-to-run variance is up to 28% for MPSGraph** and 11% for native Metal kernels | — | report medians over ≥20 reps with explicit IQR; a <30% difference on MPS is not a result |
| P17 | `torch.use_deterministic_algorithms(True)` + float `index_add_`/`scatter_add_` on **MPS raises `RuntimeError`**, not a slow path. The §2.6(a) gather-sum form is the only way to get determinism on-GPU | hard error vs clean run | try it both ways |
| P18 | `torch._segment_reduce(src, "sum", offsets=ptr)` already works on 2-D CUDA tensors (PyG uses it in production) and needs no new dependency — but with mean group size 2.06 the contention-avoidance mechanism has little to exploit | modest, ≤1.3× over `index_add_` (GeoT measured `segment_coo` ≈ 1.31× over `scatter_reduce` on Cora/Citeseer-magnitude graphs) | benchmark it at our shapes; **nobody has** |
| P19 | **Do not enable ROCm/inductor "partitioned scatter."** At low contention it is a measured **3.1× regression on H100** (0.549 → 1.724 ms); it shipped disabled by default for that reason | 3.1× slower | leave `partitioned_scatter_enabled=False` |
| P20 | A hand-written **Triton** segment-sum will lose to ATen at our size. lumen#225 measured Triton at **0.04–0.69× of torch on H100**, with torch's time **flat at ~21 µs from n=65 K to 1 M** (a launch-bound floor). But the §2.6(a/b) **gather-sum is not a segment reduction** (no atomics, fixed ≤3 loads/row) and may behave differently | Triton segment-sum ≪ 1×; Triton gather-sum plausibly > 1× | if you try Triton, try the gather-sum form, not the atomic scatter form |

---

## 11. Sources

**Fused GNN kernels**
- GE-SpMM (SC'20) — https://arxiv.org/pdf/2007.03179 · https://dai.sjtu.edu.cn/my_file/pdf/3dbfacbc-2c96-464f-baa8-293306662a63.pdf
- FusedMM (IPDPS'21) — https://arxiv.org/pdf/2011.06391 · https://arxiv.org/html/2011.06391
- FeatGraph (SC'20) — https://arxiv.org/pdf/2008.11359 · https://www.amazon.science/code-and-datasets/featgraph-sparse-kernels-for-gnns-based-on-tvm
- Graphiler (MLSys'22) — https://proceedings.mlsys.org/paper_files/paper/2022/file/a1126573153ad7e9f44ba80e99316482-Paper.pdf · https://gnnsys.github.io/papers/GNNSys21_paper_10.pdf
- TLPGNN (TOPC'24 / HPDC'22) — https://dl.acm.org/doi/10.1145/3644712 · https://yuede.github.io/files/22_hpdc_TLPGNN.pdf
- DF-GNN (NeurIPS'24) — https://arxiv.org/abs/2411.16127 · https://proceedings.mlr.press/v269/liu25a.html
- GeoT (segment reduction, public code) — https://arxiv.org/pdf/2404.03019 · https://arxiv.org/html/2404.03019v2 · https://github.com/fishmingyu/GeoT
- MaxK-GNN (ASPLOS'24) — https://arxiv.org/html/2312.08656v4 · https://github.com/harveyp123/maxk-gnn
- GNNBench (dgNN negative result) — https://arxiv.org/pdf/2404.04118
- GNN acceleration survey — https://arxiv.org/pdf/2306.14052
- IO-aware GNN layers — https://arxiv.org/pdf/2605.31500 · https://github.com/yandex-research/On-Efficient-Scaling-Of-GNNs
- Architectural implications of GNNs — https://arxiv.org/pdf/2009.00804
- DGL kernel-fusion blog — https://www.dgl.ai/blog/2019/05/04/kernel.html
- DGL maintenance / cuGraph-DGL removal — https://docs.dgl.ai/api/python/dgl.ops.html · https://docs.nvidia.com/cugraph/26.08/

**Non-backtracking / Hashimoto / line graph**
- NBA-GNN, "Non-backtracking Graph Neural Networks" — https://arxiv.org/pdf/2310.07430 · https://github.com/seonghyun26/nba-gnn
- LGNN, "Supervised Community Detection with Line Graph Neural Networks" — https://arxiv.org/pdf/1705.08415
- Non-backtracking matrix `B = ST − τ`, spectral properties — https://www.emergentmind.com/topics/non-backtracking-matrix · https://www.researchgate.net/publication/346014016_Spectral_properties_of_the_non-backtracking_matrix_of_a_graph
- Non-backtracking PageRank / Hashimoto matrices — https://link.springer.com/article/10.1007/s10915-019-00981-8 · https://eprints.maths.manchester.ac.uk/2666/1/plain.pdf
- Nonbacktracking expansion of finite graphs — https://arxiv.org/pdf/1702.05380
- Statistical physics of inference (BP ↔ non-backtracking diffusion on `L(G)`) — https://arxiv.org/pdf/1511.02476
- Non-backtracking cycles / length spectrum — https://eliassi.org/papers/appliednetsci19_nbd.pdf

**Sparse formats**
- cuSPARSE docs (Blocked-ELL, SELL) — https://docs.nvidia.com/cuda/cusparse/
- SELL-C-σ implementation — https://github.com/kylebeggs/EllSparse.jl
- Blocked-ELL vs CSR performance cliff; SparseDitto — https://arxiv.org/html/2608.05033
- DTC-SpMM / sparse tensor cores — https://dl.acm.org/doi/10.1145/3712285.3759849 · https://research.csc.ncsu.edu/picture/publications/papers/ppopp25.pdf
- Efficient quantized sparse matrix ops on tensor cores — https://arxiv.org/pdf/2209.06979

**Triton / CUTLASS / megakernels / grouped GEMM**
- CUTLASS repo + changelog — https://github.com/NVIDIA/cutlass · https://docs.nvidia.com/cutlass/4.2.1/CHANGELOG.html
- CUTLASS gather/scatter GEMM fusion (since 2.9.0) — https://github.com/NVIDIA/cutlass/issues/675 · https://www.mintlify.com/NVIDIA/cutlass/examples/cpp/fused-operations
- Epilogue Visitor Trees — https://research.colfax-intl.com/epilogue_visitor_tree/
- CUTLASS 4.x CuTe DSL (Python), blockscaled grouped GEMM — https://github.com/NVIDIA/cutlass/discussions/2649 · https://github.com/NVIDIA/cutlass/discussions/3072 · https://pypi.org/project/nvidia-cutlass/
- Blackwell hardware block-scaling tutorial — https://research.colfax-intl.com/cutlass-tutorial-hardware-supported-block-scaling-with-nvidia-blackwell-gpus/
- Hopper GEMM kernel development — https://research.colfax-intl.com/wp-content/uploads/2023/12/colfax-gemm-kernels-hopper.pdf
- "Weights That Stay" persistent megakernel (same projection reused across serial steps) — https://arxiv.org/html/2611.00506
- Ada-MK adaptive megakernel — https://arxiv.org/html/2605.11581v1
- AutoMegaKernel — https://arxiv.org/html/2606.09682v1 · https://github.com/RightNow-AI/AutoMegaKernel
- Cohere North Mini megakernel — https://cohere.com/blog/megakernels
- AMD MI355X megakernel — https://rocm.blogs.amd.com/software-tools-optimization/megakernel-intro/README.html
- ScatterMoE / fused scatter-gather grouped GEMM in Triton — https://github.com/NVIDIA/Megatron-LM/issues/4167
- pyg-lib `grouped_matmul`/`segment_matmul` (CUTLASS backend) — https://pyg-lib.readthedocs.io/en/latest/modules/ops.html · https://github.com/pyg-team/pyg-lib/releases
- FASTEN segmented matmul for heterogeneous GNNs — https://dl.acm.org/doi/fullHtml/10.1145/3650200.3656593
- Triton `atomic_add` contention / `sem="relaxed"` — https://medium.com/@michael.diggin/implementing-a-split-k-matrix-multiplication-kernel-in-triton-7ad93fe4a54c

**Small-graph / launch-overhead / molecular precedent**
- TorchMD-Net 2.0 — https://arxiv.org/pdf/2402.17660 · https://arxiv.org/html/2402.17660v2
- Reducing CUDA kernels in mini-batch HGNN training — https://arxiv.org/html/2408.08490v1
- PyGraph: compiler support for CUDA graphs in PyTorch — https://arxiv.org/pdf/2503.19779
- GraCE (OSDI'26) — https://www.usenix.org/system/files/osdi26-ghosh.pdf
- NVIDIA "Handling Dynamic Patterns" CUDA-graph guide — https://docs.nvidia.com/dl-cuda-graph/latest/torch-cuda-graph/handling-dynamic-patterns.html
- Optimize PyG with torch.compile — https://docs.nvidia.com/sdgm/research/pyg-compile-guide
- Metadata-driven host overheads in sampling-based GNN training — https://arxiv.org/pdf/2605.29346

**Numerics / determinism**
- PyTorch reproducibility notes (`index_add_`/`scatter_add_` use `atomicAdd`) — https://docs.pytorch.org/docs/2.15/notes/randomness.html
- `torch.use_deterministic_algorithms` — https://docs.pytorch.org/docs/2.15/generated/torch.use_deterministic_algorithms.html
- Forum: why `index_add_`/`scatter_add_` are nondeterministic on CUDA — https://discuss.pytorch.org/t/why-does-index-add-and-scatter-add-induce-non-deterministic-behavior-on-the-cuda-backend/45544
- NVIDIA tensor-core training (bf16 in / fp32 accumulate) — https://nvlabs.github.io/eccv2020-mixed-precision-tutorial/files/dusan_stosic-training-neural-networks-with-tensor-cores.pdf
- Split-K bf16 partial-sum cancellation bug (**up to 40% error on Ampere**) — https://github.com/modular/modular/issues/7165
- hipBLASLt long-K fp16 split-K accuracy/speed issue — https://github.com/ROCm/rocm-libraries/issues/14993
- Float non-associativity and reproducibility — https://arxiv.org/pdf/2408.05148
- Hawkeye: reproducing GPU-level non-determinism — https://arxiv.org/pdf/2603.20421
- Probabilistic error analysis: pairwise summation under stochastic rounding — https://arxiv.org/pdf/2603.24161
- Kahan / compensated summation background — https://arxiv.org/pdf/1312.3300 · https://forums.developer.nvidia.com/t/atomicadd-kahan-summation/37155

**Apple Silicon / MLX**
- MLX custom Metal kernels — https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html · https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fast.metal_kernel.html
- WWDC25 "Get started with MLX for Apple silicon" — https://developer.apple.com/videos/play/wwdc2025/315/
- MLX custom-kernel correctness issue on M3 Ultra — https://github.com/ml-explore/mlx/issues/4603
- Apple Silicon ANE vs GPU compute — https://stonesjtu.github.io/2026/09/02/apple-silicon-ai-compute-gpu-vs-ane/

**Chemprop**
- Chemprop v2 (≈2× faster, ≈3× less memory than v1; multi-GPU scaling) — https://chemrxiv.org/doi/10.26434/chemrxiv-2025-4p1nr · https://chemrxiv.org/doi/pdf/10.26434/chemrxiv-2025-4p1nr
- Chemprop v1 JCIM paper — https://pubs.acs.org/doi/pdf/10.1021/acs.jcim.3c01250
- "What drives performance in molecular MPNNs? An operator-level factorial benchmark" — https://arxiv.org/pdf/2605.30195 *(could not extract text; flagged for follow-up — directly relevant to whether the `−H_rev` term earns its cost)*
- "Optimal message passing for molecular prediction is simple, attentive" — https://arxiv.org/pdf/2509.10871 *(same; text not extractable)*

**Ecosystem: `pytorch_scatter` / PyG / scatter pathologies (§7)**
- `torch_scatter/scatter.py` (`scatter_sum` is a `scatter_add_` wrapper) — https://github.com/rusty1s/pytorch_scatter/blob/master/torch_scatter/scatter.py
- PyG #5120 "Move to PyTorch `scatter_reduce` implementation" — https://github.com/pyg-team/pytorch_geometric/pull/5120
- PyG `utils/_scatter.py` (what it dispatches to today) — https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/utils/_scatter.py
- PyG `utils/_segment.py` (native `torch._segment_reduce` fallback) — https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/utils/_segment.py
- PyG #10716 "Deprecate `torch-{scatter,sparse,...}` releases on PyPI" — https://github.com/pyg-team/pytorch_geometric/issues/10716
- rusty1s/pytorch_scatter #402 (`segment_csr` vs `segment_coo` mechanism), #509 (no PyPI wheels), #513, #515, #269 (half-precision scatter bug), #241 (Apple Silicon build) — https://github.com/rusty1s/pytorch_scatter/issues/402 · /509 · /513 · /515 · /269 · /241
- PyG #7596 (`index_*` vs `scatter_*` on TPU) — https://github.com/pyg-team/pytorch_geometric/issues/7596
- PyG #4891 "[Roadmap] CPU Performance Optimization for PyG" — https://github.com/pyg-team/pytorch_geometric/issues/4891
- PyG Discussion #4728 (CSR `adj_t` 14.98 ms vs COO `edge_index` 111.02 ms) — https://github.com/pyg-team/pytorch_geometric/discussions/4728
- pytorch #82703 (created `can_use_expanded_index_path`; 6.3× on SAGE/Reddit) — https://github.com/pytorch/pytorch/pull/82703
- pytorch #88729 (`index_add` → `scatter_add` for inner dim) — https://github.com/pytorch/pytorch/pull/88729
- **pytorch #203078 (`.expand`'d index → silent wrong answer, OPEN)** — https://github.com/pytorch/pytorch/issues/203078
- pytorch #200940 (`index_add_` 2–8× slower in 2.14 on 16–64 B rows; TMA crossover) — https://github.com/pytorch/pytorch/issues/200940
- pytorch #182818 (CuTeDSL scatter_add override for the expanded-1D-index pattern, unmerged) — https://github.com/pytorch/pytorch/pull/182818
- pytorch #194078, #168073 (partitioned scatter; 3.1× regression at low contention), #203900 (256 MiB work floor), #203975 — https://github.com/pytorch/pytorch/pull/194078 · /168073 · /203900 · /203975
- pytorch #183812 (claimed CPU `scatter_reduce` regression — **not reproduced**) — https://github.com/pytorch/pytorch/issues/183812
- pytorch #203118 (`assume_unique` overloads; deterministic routing) — https://github.com/pytorch/pytorch/pull/203118
- pytorch #70873 (earlier `unique_indices` attempt) — https://github.com/pytorch/pytorch/pull/70873
- pytorch #113232 (Inductor GNN support, by GeoT's author) — https://github.com/pytorch/pytorch/issues/113232
- PyTorch Sparse(GNN) Compiler RFC — https://dev-discuss.pytorch.org/t/pytorch-sparse-gnn-compiler-rfc/1644
- pytorch #94640 (dynamo + GNN dynamic batches; compiled 1.21× slower; ezyang's 1.267/1.102/1.021) — https://github.com/pytorch/pytorch/issues/94640
- PyG "Compiled Graph Neural Networks" — https://pytorch-geometric.readthedocs.io/en/latest/advanced/compile.html
- lumen #225 (Triton `scatter_add` 0.04–0.69× of torch, H100) — https://github.com/trainstation-project/lumen/issues/225
- triton-lang/triton #891; pytorch #192999, #179833 (Inductor `tl.atomic_add` fixes) — https://github.com/triton-lang/triton/issues/891
- **FluxML/NNlib.jl #854 (fp16/bf16 paired-atomic trap, RTX 5090, 200–400×)** — https://github.com/FluxML/NNlib.jl/issues/854
- pytorch #203089, #203243 (fp16/bf16 packed atomic-add fixes, Nov 2026) — https://github.com/pytorch/pytorch/pull/203089 · /203243
- pyg-lib #51, #146, #161, #213 (launch-overhead 5.8×), #271, #272, #333 — https://github.com/pyg-team/pyg-lib/pull/213
- **Chemprop Discussion #1019 (maintainer on GPU utilisation)** — https://github.com/chemprop/chemprop/discussions/1019
- Chemprop #457, #477 (PyG/DGL refactor rationale), #565, #1234 — https://github.com/chemprop/chemprop/issues/477
- **itakigawa/pyg_chemprop (15 s → 7 s/epoch, A100)** — https://github.com/itakigawa/pyg_chemprop
- NVIDIA cuik-molmaker (2.2×/2.9× via featurisation) — https://github.com/NVIDIA-BioNeMo/cuik-molmaker · https://arxiv.org/abs/2510.12719

**Apple Silicon / MPS / MLX (§8)**
- pytorch MPS indexing/scatter PRs: #184057, #187109, #182249, #141948, #196955, #198689, #195754, #196203 — https://github.com/pytorch/pytorch/pull/187109 · /184057
- `aten/src/ATen/native/mps/operations/Indexing.mm` — https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/mps/operations/Indexing.mm
- **pytorch #198182 (native Metal matmul 6–10× slower than MPSGraph, M4 Max)** — https://github.com/pytorch/pytorch/issues/198182
- pytorch #197014 (native MPS `segment_reduce` forward, still open; M4 Pro bf16 table) — https://github.com/pytorch/pytorch/pull/197014
- pytorch #196192 (tracking: sparse CSR/CSC/BSR on MPS — unsupported) — https://github.com/pytorch/pytorch/issues/196192
- **pytorch #201455 / #201460 (MPSInductor silently N× wrong on reduction+`index_add`)** — https://github.com/pytorch/pytorch/issues/201455
- pytorch #157957 (MPSInductor in 2.8.0), #150121 (design), #143893 (Metal compiler) — https://github.com/pytorch/pytorch/issues/157957
- MPS correctness bugs: #195074, #163327, #189969, #175191, #176159, #161524, #136003, #77764 — https://github.com/pytorch/pytorch/issues/77764
- MPS notes / macOS 15 minimum — https://github.com/pytorch/pytorch/blob/main/docs/source/notes/mps.md
- PyTorch 2.15 MPS release notes (fp8 `float8_e4m3fn` + `F.scaled_mm`) — https://huggingface.co/blog/mps-pytorch-2-15
- `torch/mps/__init__.py` (`compile_shader`, `load_metallib`) — https://github.com/pytorch/pytorch/blob/main/torch/mps/__init__.py
- Apple WWDC25 "Get started with MLX" / WWDC26 330 "Optimize custom ML operations with Metal tensors" — https://developer.apple.com/videos/play/wwdc2025/315/ · https://developer.apple.com/videos/play/wwdc2026/330/
- Apple WWDC23 10050 (custom shaders on PyTorch's MPS stream) — https://developer.apple.com/videos/play/wwdc2023/10050/
- MLX custom Metal kernels — https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html
- MLX `gather_mm`, `segmented_mm`, `block_masked_mm`, `quantized_matmul`, `array.at` — https://ml-explore.github.io/mlx/build/html/python/ops.html
- MLX scatter/atomics kernels — https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/kernels/indexing/scatter.h · .../atomic.h
- MLX issues #4621 (4-bit slower at M>512), #4355 (reductions not fused), #4681, #4603, #1439, #3480 — https://github.com/ml-explore/mlx/issues/4621
- mlx-graphs — https://github.com/mlx-graphs/mlx-graphs · https://mlx-graphs.github.io/mlx-graphs/
- Apple M5 LLM study (4× TTFT compute-bound vs 19–27% decode memory-bound) — https://machinelearning.apple.com/research/exploring-llms-mlx-m5
- Rigel: reverse-engineering Metal 4.1 on M4 Max — https://arxiv.org/abs/2606.12765
- fp8 Metal 4.1 microbenchmark (M5 / M5 Max) — https://github.com/Anemll/fp8-mlp-metal41-bench
- Apple Silicon memory bandwidth table — https://llmcheck.net/blog/apple-silicon-memory-bandwidth-llm/
- Apple Metal PyTorch page — https://developer.apple.com/metal/pytorch/
- PyG MPS issues #6837, #10453, #7770 — https://github.com/pyg-team/pytorch_geometric/issues/6837
- mps-bitsandbytes (`spmm_coo` for Metal) — https://github.com/mpsops/mps-bitsandbytes
- Sparse ternary GEMM on Apple Silicon (CPU/NEON) — https://arxiv.org/abs/2510.06957

**ONNX / TorchScript (§9)**
- onnxscript torchlib `core.py` (`aten_scatter_reduce`, `aten_index_put`) — https://github.com/microsoft/onnxscript/blob/main/onnxscript/function_libs/torch_lib/ops/core.py
- `torch/onnx/_constants.py` (opset limits) — https://github.com/pytorch/pytorch/blob/main/torch/onnx/_constants.py
- `torch/_decomp/decompositions.py` (`_index_add` → `index_put_`) — https://github.com/pytorch/pytorch/blob/main/torch/_decomp/decompositions.py
- ONNX `ScatterND` / `ScatterElements` specs — https://onnx.ai/onnx/operators/onnx__ScatterND.html · https://onnx.ai/onnx/operators/onnx__ScatterElements.html
- ONNX versioning — https://github.com/onnx/onnx/blob/main/docs/Versioning.md
- ORT CPU `scatter.cc` / `scatter_nd.cc` — https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/providers/cpu/tensor/scatter.cc
- **`onnxruntime-ep-mlx`** — https://github.com/onnxruntime/onnxruntime-ep-mlx · docs/DESIGN.md · docs/OP_ARCHITECTURE.md · docs/SCATTER_ND.md · `rust/src/ops/shape.rs` · PR #95
- onnxruntime #21271 "[Feature Request] MPS provider" (open since 2024) — https://github.com/microsoft/onnxruntime/issues/21271
- ORT WebGPU EP + `scatter_elements.cc`; #22077, #31627 — https://onnxruntime.ai/docs/execution-providers/WebGPU-ExecutionProvider.html
- ORT CoreML EP op list — https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html
- ORT contrib operators (no scatter/segment op) — https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md
- onnxruntime-extensions custom ops (`SegmentSum`, fp32/CPU-only, docs "TODO") — https://github.com/microsoft/onnxruntime-extensions/blob/main/docs/custom_ops.md
- `torch.onnx` docs / `custom_translation_table` tutorial — https://docs.pytorch.org/docs/main/onnx_export.html · https://docs.pytorch.org/tutorials/beginner/onnx/onnx_registry_tutorial.html
- C++/custom-op tutorial (`register_fake`) — https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html
- ORT add-custom-op — https://onnxruntime.ai/docs/reference/operators/add-custom-op.html
- **pytorch #151746 (Triton kernels don't serialize through `torch.export`)** — https://github.com/pytorch/pytorch/issues/151746
- **pytorch #203777 ("We are starting to delete TorchScript from the codebase")** — https://github.com/pytorch/pytorch/issues/203777
- onnxruntime #31730 (`SparseToDenseMatMul` sparse-initializer bug), #5578, #20297, #21148 — https://github.com/microsoft/onnxruntime/issues/31730
- ORT 1.31 release notes (ONNX pinned at 1.22; FLOAT6 deferred) — https://github.com/microsoft/onnxruntime/releases/tag/v1.31.0

**Low precision (§5)**
- OCP Microscaling (MX) Specification v1.1.0 — https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
- Transformer Engine support matrix — https://nvidia.github.io/TransformerEngine/support_matrix.html
- Colfax: hardware block-scaling on Blackwell; NVFP4 on sm_12x; optimizing NVFP4 on RTX PRO 6000 — https://research.colfax-intl.com/cutlass-tutorial-hardware-supported-block-scaling-with-nvidia-blackwell-gpus/ · https://research.colfax-intl.com/cutlass-tutorial-nvfp4-blockscaled-gemm-on-nvidia-rtx-pro-blackwell-gpus-sm12x/
- CUTLASS tcgen05 programming guide (FP32 accumulator fixed) — https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/guides/mma/tcgen05_programming.html
- **DeepSeek-V3 (FP8 accumulate ≈14 bits, ~2% error)** — https://arxiv.org/html/2412.19437v1
- **Insights into DeepSeek-V3 (FP22 accumulator, 1s8e13m)** — https://arxiv.org/html/2505.09343v1
- Hand-written MXFP8 GEMM on B200 — https://danielvegamyhre.github.io/2026/03/29/mxfp8-gemm.html
- Cursor kernels blog (MXFP8 2,750 vs FP8 3,300 vs BF16 1,500; quant = 40–76% of matmul) — https://cursor.com/blog/kernels
- torchao inference workflows (MXFP8 1.81/1.97; NVFP4 2.98/3.82; prefill 1.739× vs decode 1.058×) — https://docs.pytorch.org/ao/main/workflows/inference.html
- Blackwell microbenchmarks (B200 vs H200) — https://arxiv.org/html/2512.02189v1
- MR-GPTQ (FP4 PTQ; MXFP4 power-of-two scale error; NVFP4 group size) — https://arxiv.org/abs/2509.23202
- RTX Blackwell FP32-accumulate half-rate table — https://www.airealist.ai/p/nvidia-geforce-half-rate
- AMD MI350/CDNA4 MX support — https://www.amd.com/en/products/accelerators/instinct/mi350.html
- Intel Gaudi 3 white paper (FP8 = BF16 rate) — https://cdrdv2-public.intel.com/817486/gaudi-3-ai-accelerator-white-paper.pdf
- CUDA `atomicAdd` supported types — https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html
- **jax-ml/jax #42211 (`segment_sum` in bf16: 8192×1.25e-3 → 0.5 instead of 10.24)** — https://github.com/jax-ml/jax/issues/42211
- ROCm/aiter #5976 (bf16 split-K partials: ~6% of outputs ≥2 ULP off) — https://github.com/ROCm/aiter/issues/5976
- Megatron-LM #6199 (bf16 gradient reduction instability at 384 GPUs) — https://github.com/NVIDIA/Megatron-LM/issues/6199
- **Degree-Quant (ICLR'21), incl. the ZINC molecular table** — https://arxiv.org/pdf/2008.05000 · https://openreview.net/forum?id=NSBrFgJAHg
- SGQuant — https://arxiv.org/pdf/2007.05100
- EXACT (ICLR'22) — https://openreview.net/forum?id=vkaMaq95_rX
- MixQ-GNN (Theorem 1: integer aggregation with fused dequant) — https://arxiv.org/html/2505.09361v1
- QGTC (PPoPP'22, GNN aggregation on 1-bit tensor cores, uint32 accumulate) — https://arxiv.org/pdf/2111.09547
- **Quantized GNNs for molecular property prediction, J. Cheminform. 17:81 (2025)** — https://pmc.ncbi.nlm.nih.gov/articles/PMC12108020/ · https://link.springer.com/article/10.1186/s13321-025-00989-3
- **Graph-filtering noise-propagation analysis (shared shift operator)** — https://arxiv.org/html/2506.01404v1
- **LoopQ: Quantization for Looped Language Models** — https://arxiv.org/abs/2605.16343
- Quantizing Recursive Reasoning Models — https://arxiv.org/abs/2607.16237
- Quantization Degradation: a Signal–Noise Perspective — https://arxiv.org/abs/2608.08188
- Quantization Error Propagation — https://arxiv.org/abs/2504.09629
- Why Gated DeltaNet Survives 4-Bit Quantization — https://arxiv.org/abs/2609.04098
- SmoothQuant / AWQ / KVQuant (the opposite-direction consensus) — https://arxiv.org/pdf/2306.00978 · https://www.stat.berkeley.edu/~mmahoney/pubs/neurips-2024-kvquant.pdf
- Low-bit RNN background — https://arxiv.org/pdf/1802.02615 · https://arxiv.org/abs/1611.10176
- GPU cache sizes (H100 L2 50 MB, B200 126 MB, GB203 65 MB measured) — https://chipsandcheese.com/p/nvidias-h100-funny-l2-and-tons-of-bandwidth · https://chipsandcheese.com/p/nvidias-b200-keeping-the-cuda-juggernaut

**Molecular MPNN architecture (does the `−H_rev` term earn its cost?)**
- "What drives performance in molecular MPNNs? An operator-level factorial benchmark" (84 configs × 10 MoleculeNet datasets; *"performance variation is associated primarily with message construction rather than update complexity"*; update operators show **no** significant effect) — https://arxiv.org/abs/2605.30195
- "Optimal message passing for molecular prediction is simple, attentive and spatial" — https://arxiv.org/abs/2509.10871
- Analyzing Learned Molecular Representations (the original D-MPNN) — https://arxiv.org/pdf/1904.01561
- On the Scalability of GNNs for Molecular Graphs (NeurIPS'24) — https://arxiv.org/abs/2404.11568

**Reproduction scripts for the measured graph statistics in this brief**
- `nb_stats.py`, `designs.py` (this directory) — RDKit over `chemprop/tests/data/*.csv`.
  Regenerate with `uv run --with rdkit --with numpy python nb_stats.py` from the chemprop repo root.

---
