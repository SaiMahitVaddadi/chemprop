# Caching and quantization for D-MPNN / CheMeleon inference

An empirical study of how much of a Chemprop D-MPNN forward pass is redundant recomputation,
whether it can be cached *exactly*, and how far the weights and the cache can be quantized
before accuracy degrades.

Everything here was measured, not estimated. Where a number is arithmetic rather than a
wall-clock measurement, it says so.

## Setup

Measurements were taken on Apple Silicon (macOS), PyTorch 2.11, RDKit 2025.09.3, no CUDA.
The encoder is the real pretrained **CheMeleon** checkpoint (`d_v=72, d_e=14, d_h=2048,
depth=6, dropout=0.0`, 8.71M params), used frozen.

`scripts/cp_loader.py` loads Chemprop's real atom/bond/molgraph featurizers standalone via
`importlib`, stubbing `descriptastorus` and `cuik_molmaker`. Output is verified bit-identical
to `SimpleMoleculeMolGraphFeaturizer`. This exists so the study runs without a full Chemprop
install; it is not a reimplementation.

Derived encoder embeddings (~134 MB) are gitignored. Regenerate with `scripts/prep.py`.

## Protocol

- **Datasets**: 11 MoleculeNet tasks prepared; the weight/cache sweep completed on 5
  regression tasks — ESOL/delaney logS (1127), FreeSolv ΔG_hyd (639), Lipophilicity logD
  (2200), BACE pIC50 (1513), QM9-12 DFT properties (496). Molecule sizes span 5.5–27.1
  heavy atoms.
- **Splits**: 6 random + 4 Bemis-Murcko scaffold, 80/20.
- **Head**: ridge (λ=100, chosen once on fp32 by mean rank across all 11 datasets), fitted on
  **fp32 train** embeddings and evaluated on **quantized test** embeddings — i.e. the
  deployment case, not a retrained-for-quantization case.
- Deltas are **paired per split**, which buys 3–5× over comparing means.

### The noise floor — read this before any delta below

fp32 split-to-split CV is **4–12% on random splits** and 8–56% including scaffold splits.
Median paired-delta standard error is 1.07–2.27% per dataset.

> **A paired delta below ~2.3% is not distinguishable from split noise.**

Scaffold splits roughly *double* fp32 RMSE — a larger effect than any 8-bit quantization
scheme in this study.

## Findings

### 1. The input graph is ~99.8% redundant

Across 57,549 atoms there are only **102 distinct atom feature vectors**, and across 123,220
directed bonds only **17 distinct bond feature vectors**.

But caching feature vectors keyed on RDKit properties only yields **1.50×**, because **55% of
featurization time is reading properties out of RDKit in Python**, which no cache avoids. The
win comes from caching *type-ID arrays* per molecule and rehydrating with a NumPy fancy-index:

| variant | µs/mol | speedup |
|---|---|---|
| `SimpleMoleculeMolGraphFeaturizer` | 102.2 | 1.00× |
| RDKit property reads alone (floor) | 55.7 | 1.84× |
| type-table cache, keys computed live | 68.3 | 1.50× |
| **type-ID gather, IDs precomputed** | **4.4** | **23.4×** |

Storage: **217 B/mol** versus 7,530 B/mol for a cached `MolGraph` — **35× smaller** — and
verified bit-identical to the featurizer's output.

Incidental: `chemprop/featurizers/molgraph/molecule.py` allocates `E` with no dtype, so bond
features are **float64** while atom features are float32. Fixing the dtypes is 1.33× cache
memory for a one-line change.

### 2. A D-MPNN's hidden states can be cached exactly

Because the bond message excludes the reverse edge, `H_e^(t)` is a pure function of the
**non-backtracking walk tree of depth t** rooted at edge `e`. That is a Weisfeiler-Lehman
object, so a WL hash over Chemprop's own feature vectors is an exact cache key — no subgraph
isomorphism needed. Verified: edges sharing a radius-*r* key have identical hidden states to
≤7e-7 (float32 roundoff), so unique-key counts are the exact minimal cache size.

Two engineering details carry the result:

- **Seed the WL keys from input-graph type IDs** rather than hashing float feature rows:
  94 → **11.8 µs/mol** (8×). The type-ID cache from §1 pays for itself twice.
- **Schedule**: probe all layers, find the deepest fully-cached layer, skip everything beneath
  it, and never run the scatter when a layer is a full hit. Without this, a 0%-miss forward
  still cost 60% of a full forward.

Pooled max hit rate by radius across 11 datasets: 99.9% (r=0), 97.2%, 87.9%, 77.1%, 67.9%,
60.6% (r=5). **Reuse is governed by receptive-field radius, not layer count.**

Implied upper-bound speedup on a depth-6 encoder's message-passing layers:

| regime | range (11 datasets) | median |
|---|---|---|
| cold | 3.13× – 8.22× | 3.76× |
| warm, random split | 3.02× – 11.43× | 4.05× |
| warm, **scaffold split** | **1.65× – 9.00×** | **2.89×** |

**Strongly chemistry-dependent.** A congeneric series is the best case and a diverse library
the worst: BACE reaches 9–11×, QM9 bottoms out at 1.65×. On a synthetic 400-analog enumeration
(one scaffold × 20×20 R-groups) the whole library produced only **1,205 distinct radius-6
environments**, with marginal hit rate reaching 100% by roughly the 50th analog — a
combinatorial enumeration has O(R) environments for O(R²) molecules. End-to-end measured
speedups there were 2.97× / 6.30× / 7.89× for increasing model size, with outputs exact to
float32 roundoff.

Caveat: atom-environment hit rate at depth 6 is only 12–54%, so the final readout layer caches
far worse than the message-passing layers.

### 3. KV-cache intuitions do not transfer

An LLM KV cache works because causal masking makes keys/values final once computed; reuse is
*intra-sequence*. A D-MPNN forward has no autoregressive axis — messages flow *inward*, so a
scaffold's representation depends on its substituents and nothing can be frozen. Measured: a
single R-group swap dirties **11.8–59.1% of atoms at radius 1 and 47.1–90.9% at radius 5**. So
"prefix-cache the scaffold, recompute the R-group" does not work on drug-sized molecules.

What works is **recurrence across a library**, not locality within a parent-child pair. The
right analogy is cross-request prefix sharing in a long-lived serving process, not per-sequence
KV reuse.

A cache is only valid for fixed weights, so this is an inference / frozen-encoder technique.
Note `DropoutEstimator` activates dropout at inference, making hidden states stochastic — the
cache must be bypassed there. Ensembles need one cache per checkpoint, but share the §1
input-graph cache.

### 4. The cache tolerates 4 bits. The weights do not.

Matched-bit damage, cache versus weights: **235× at 8 bits, 37× at 6 bits, 187× at 4 bits,
224× at 2 bits.** Cache error is injected once and then averaged twice (neighbour summation,
then aggregation over atoms), so it behaves like independent noise; weight error is systematic
across every use. The activations are also ~6.6× the bytes of `W_h`, so the big tensor is the
cheap one to compress.

Weight quantization, paired % vs fp32 (+ = worse), 5 datasets × 10 splits:

| config | bits | mean | median | worst | resolved-worse |
|---|---|---|---|---|---|
| bf16 / fp16 | 16 | +0.00 | +0.00 | +0.02 | 0–1/5 |
| **MXFP8-e4m3** | 8.25 | **+0.50** | +0.47 | +1.13 | 1/5 |
| **int8-perchan** | 8 | **+0.47** | +0.14 | +2.20 | 1/5 |
| MXFP8-e5m2 | 8.25 | +8.65 | +4.91 | +21.95 | 4/5 |
| **MXFP6-e2m3** | 6.25 | **+1.95** | +1.24 | +5.86 | 3/5 |
| int6-g32 | 6.25 | +2.25 | +1.25 | +7.26 | 2/5 |
| MXFP6-e3m2 | 6.25 | +8.80 | +4.83 | +22.31 | 4/5 |
| NVFP4 | 4.5 | +27.41 | +16.41 | +51.13 | 5/5 |
| MXFP4-e2m1 | 4.25 | +38.53 | +23.37 | +104.26 | 5/5 |
| int4-g32 | 4.25 | +46.97 | +35.10 | +121.66 | 5/5 |
| int2-g32 | 2.25 | +3037 | +2631 | — | 5/5 |

**Mantissa bits dominate exponent range.** e4m3 (+0.50%) beats e5m2 (+8.65%) at 8 bits;
e2m3 (+1.95%) beats e3m2 (+8.80%) at 6 bits — 5–18× in both pairs. CheMeleon's weights are
heavy-tailed (kurtosis 6–12.7) but narrow in dynamic range, so exponent bits are wasted.

≤2-bit is not "degraded", it is **dead**: int1/int2/ternary give R² ≤ 0 on every dataset
(Lipophilicity int2 reaches R² = −1233), i.e. worse than predicting the training mean. There
is no normalization layer anywhere in the message-passing loop to absorb per-layer bias.

**Recipe: weights MXFP6-e2m3, environment cache int4 group-32** (+1.95% and +0.25%). If 8 bits
is acceptable, MXFP8-e4m3 or int8-per-channel are free. Avoid e5m2 and e3m2. Do not use 4-bit
weights — the cheapest 4-bit configuration still costs +13.3%.

Hardware caveat: all MX formats are block-32/UE8M0, so `d_h=300` violates `K%32` and is
excluded from every hardware MX path; MX acceleration is Blackwell/CDNA4 only, and fp8 measures
1.007–1.15× vs fp16 on Apple Silicon. The portable recommendation is bf16/fp16 activations with
fp32 accumulators.

### 5. Two one-line kernel fixes, and one that silently corrupts

`index_torch = edge_index[1].unsqueeze(1).repeat(1, H.shape[1])` materialises an `E × d_h`
**int64** tensor every layer — 223 MB at CheMeleon dimensions, 2× the byte volume of the
features it indexes. More importantly, `.repeat` fails PyTorch's stride-0 test and therefore
**disqualifies the fast GNN scatter paths on both CPU and CUDA**.

| | CPU `d_h=300 d=3` | CPU `d_h=2048 d=6` | MPS `d_h=2048 d=6` |
|---|---|---|---|
| `.repeat` (current) | 91.7 ms | 1279.9 ms | 690.0 ms |
| `.expand` | 61.1 ms (1.50×) | **1459.9 ms (0.88× — slower)** | 730.6 ms (0.94×) |
| **`index_add_`** | **42.5 ms (2.16×)** | **1094.1 ms (1.17×)** | **459.6 ms (1.50×)** |

**`.expand` is the wrong fix on two counts**: it regresses the large model on CPU, and
pytorch#203078 (open) is a *silent wrong-answer* bug for exactly this shape class — index with
strides `(1,0)` where `self.size(1) > index.size(1)`. Use `index_add_` with a 1-D index, which
is bitwise identical to current output on CPU (max diff exactly 0.0 at both sizes).

Separately, `scatter_reduce_(..., include_self=False)` costs **50.3 ms vs 24.7 ms** for
`include_self=True` at `d_h=2048`, because it performs an entire extra scatter traversal
writing zeros into a buffer `torch.zeros()` already zeroed — mathematically identical here.

Export angle: `index_add_` lowers to ONNX `ScatterND`, which ORT's MLX execution provider
accelerates on Apple Silicon, whereas `scatter_reduce_` lowers to `ScatterElements(reduction=
"add")` — the one form that EP refuses, causing a silent CPU fallback at every depth step.

### 6. Chemprop already computes the Hashimoto matrix, in the wrong factorization

`message()` computes `M = B·H` where `B = S Tᵀ − τ` is the Hashimoto non-backtracking
edge-adjacency matrix. Chemprop evaluates the factored form `S(TᵀH) − τH`, which is correct for
high-degree graphs and wrong for molecules. Measured on 6,253 molecules: mean atom degree
**2.09**, **nnz(B)/row = 1.384** (B is sparser than a gather; 10.7% of rows empty). Explicit
`B` needs **2.06× less traffic** than the `index_add_` form, and the crossover is
`mean(d²) < 4·mean(d) + 2` → **degree < 4.45**. Molecules sit at 2.09; Reddit misses by three
orders of magnitude, which is why no GNN-kernel paper proposes it.

It is algebraically exact, atomics-free, and export-safe. It also structurally removes a
numerical hazard: for a degree-2 atom, `M = (H_f + H_rev) − H_rev` is catastrophic
cancellation — harmless in fp32 (~1e-6) but **~4% error in bf16**, affecting the 43.5% of
directed bonds that are degree-2.

**This is the one headline claim in the study with no wall-clock measurement behind it.** The
traffic ratio and `nnz(B)` are measured; the speedup is arithmetic.

### 7. The depth experiment that did not work

Hypothesis: if depth 3 matched depth 6, FLOPs would halve *and* the cache would move from the
60.6% regime (r=5) to 87.9% (r=2). Measured, by truncating the depth-6 checkpoint:

| dataset | noise CV | depth 3 | depth 4 | depth 5 |
|---|---|---|---|---|
| ESOL | 3.97% | **+24.4%** | +14.9% | +2.5% (unresolved) |
| FreeSolv | 7.10% | **+23.7%** | +20.3% | +4.6% (unresolved) |
| Lipophilicity | 7.10% | **+15.0%** | +8.0% | +4.3% |
| QM9 | 12.14% | **+30.0%** | +16.3% | +7.1% |

Depth 3 costs 2.1–6.1 noise widths on 4/4 datasets. The obvious alternative explanation was
readout mismatch — `W_o` was trained to read depth-5 edge states. Testing that required three
formulations, because the readout is 2120→2048 while ESOL/FreeSolv have 902/511 training
molecules, so a naive refit is unidentifiable. The tell: the first two formulations showed a
gap *at depth 6*, where the map is exactly linear and recovery must be perfect.

The identifiable version distils `A_k → Z_6` on 8,000 molecules pooled from other datasets
with evaluation molecules removed (label-free target, so no leakage). Sanity check passes:
depth-6 reconstruction 0.004–0.007, distilled RMSE matches true depth-6. Result: at depth 5 the
representation **is** largely recoverable (recon 0.068–0.084); at depth 3 it is **not**
(0.151–0.242), and distilling *worsens* RMSE rather than closing the gap.

**So the depth-3 cost is genuine information loss, not readout mismatch.** Depth 5 is near-free
but barely moves the cache (60.6% → 67.9%). The 87.9% regime is closed off absent a full
retrain.

Consistent with the literature: molecular GNN depth is the weakest of the three scaling axes
(Spearman −0.11 to 0.55 for depth vs 0.69–0.82 for width, Sypetkowski et al. NeurIPS 2024),
and graph transformers use equal or *more* layers on molecules than Chemprop's default 3.

## Not validated

- **Every classification verdict is INCONCLUSIVE.** The sweep finished on regression only.
- No trained (MLP) head; ridge only. Encoder-side deltas are head-independent by construction,
  but the absolute numbers are not.
- No dataset above 2,200 molecules.
- No CUDA measurements at all.
- The Hashimoto speedup (§6) is unbenchmarked.
- The depth/cache correlation (Spearman −0.800, mechanism: molecule size) is **n=4**,
  p ≥ 0.33 — a consistent sign with a plausible confounder, not a measured relationship.

## Corrections made during the study

Recorded because several initial findings did not survive more data:

- **"4-bit is survivable at +20.7%"** → +38.5% across 5 datasets. The single-dataset result was
  optimistic.
- **"NVFP4 underperforms MXFP4"** → refuted; NVFP4 is better on 4/5. The original was an ESOL
  artifact.
- **"MX float beats int for weights"** → refuted (int wins 5/5 at 6 bits). It holds strongly
  for *activations*.
- **"`W_i` is sensitive because `H_0` is re-injected every layer, so error amplifies 6×"** →
  mechanism refuted. Jacobian amplification is **flat in depth** (0.58–0.69); `W_i`'s edge is a
  constant ~1.6–1.8×. The real driver is weight tying.
- **"Protecting `W_i` is 10× effective"** → 2.6× median. And uniform MXFP6-e2m3 beats the
  mixed-precision recipe entirely.
- **"Cosine similarity is a useless diagnostic"** → it *ranks* well (Spearman −0.941); it
  cannot threshold, since the whole usable range is cos ∈ [0.996, 1.000].
- **"Depth reduction and caching are in tension"** → wrong, from a cherry-picked pair. The
  correlation is negative, so they align — but depth 3 still costs real accuracy.
- **"`.expand` is the one-line fix for the index bloat"** → regresses the large model on CPU
  and is a silent wrong-answer hazard.
- A bug in `scripts/qlib.py`'s predecessor: `fq()` used the generic `(2−2^−m)·2^emax` maximum,
  giving e4m3 max = 480. OCP reserves the all-ones mantissa at emax for NaN, so the spec max is
  **448**. Affected ~0.5% of weights and made MXFP8-e4m3 slightly optimistic.

## File map

```
scripts/     measurement and analysis code; cp_loader.py is the standalone featurizer loader
results/     per-(dataset, config, split, task) records, tidy CSVs, printed tables
             every JSON carries its caveat in a top-level `caveat` field
briefs/      kernel-research-brief.md (kernel/ecosystem survey, 212 cited sources)
             message-passing-research-brief.md (architecture / traversal survey)
data/        MoleculeNet CSVs (DeepChem S3)
cache/       gitignored; regenerate with scripts/prep.py
```
