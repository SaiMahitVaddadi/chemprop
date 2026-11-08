# The cache tolerates 4 bits. The weights don't.

*Part 5 of a series on caching and quantizing D-MPNN inference.*

Having built a cache for a D-MPNN's hidden states, the obvious next question is how small you can
make it. Memory is what limits how many environments stay resident, and resident environments are
what drive the hit rate.

The answer turned out to be lopsided in a useful direction, and getting to it required throwing
out most of what I concluded from my first pass.

## Start with the noise floor

My first version of this study used one dataset, one split, and one seed. Every number in it was
wrong by enough to matter, and the reason is that **molecular property prediction is noisier than
the effects you're trying to measure.**

Across 5 MoleculeNet regression tasks × 10 splits each (6 random + 4 Bemis-Murcko scaffold), the
fp32 baseline's own split-to-split coefficient of variation:

| dataset | fp32 random-split CV |
|---|---|
| BACE pIC50 | 2.98% |
| ESOL logS | 3.97% |
| FreeSolv ΔG | 7.10% |
| Lipophilicity logD | 7.10% |
| QM9 (12 targets) | 12.14% |

Including scaffold splits it runs 8–56%. Median paired-delta standard error is 1.07–2.27% per
dataset, which gives the number that makes everything else readable:

> **A paired delta below ~2.3% is not distinguishable from split noise.**

Pairing deltas per split rather than comparing means buys 3–5× in resolution and is the only
reason any of this is measurable at all. For scale: **switching from random to scaffold splits
roughly doubles fp32 RMSE** — a larger effect than any 8-bit quantization scheme in this study.

Protocol throughout: frozen CheMeleon encoder, mean aggregation, ridge head (λ=100, chosen once on
fp32 by mean rank across all 11 datasets) fitted on **fp32** train embeddings and evaluated on
**quantized** test embeddings. That's the deployment case — you don't get to retrain the head for
your quantization scheme.

## The asymmetry

| bit width | cache damage | weight damage | ratio |
|---|---|---|---|
| 8-bit int | +0.00% | +0.61% | **235×** |
| 6-bit int | +0.08% | +2.25% | **37×** |
| 4-bit int | **+0.25%** | **+46.97%** | **187×** |
| 2-bit int | +13.3% | +3037% | **224×** |

At matched bit width the cache is one to two orders of magnitude cheaper to compress than the
weights. Two reasons, both structural:

**Cache error is injected once and then averaged twice.** A quantized hidden state enters the
neighbour summation, then the mean-aggregation over ~13 atoms. Independent rounding noise averages
down. Weight error is *systematic* — the same perturbed matrix is applied to every edge, at every
layer, so errors correlate rather than cancel.

**And the cache is the big tensor anyway.** The activations are ~6.6× the bytes of `W_h`. `W_h`
has a reuse factor of 13,600 in a typical batch, so its DRAM traffic is essentially nil and it
fits in L2 from an A100 onward. Compressing weights saves memory you weren't short of; compressing
the cache buys resident environments, which buys hit rate.

So: **int4 group-32 for the cache**, +0.25% mean with a +0.39% worst case, for a 7.5× reduction —
0.008–0.183 GiB instead of 0.06–1.38 GiB across the 11 datasets.

## The weights, in full

| config | bits | mean | median | worst | resolved-worse |
|---|---|---|---|---|---|
| bf16 / fp16 | 16 | +0.00 | +0.00 | +0.02 | 0–1/5 |
| **int8 per-channel** | 8 | **+0.47** | +0.14 | +2.20 | 1/5 |
| **MXFP8-e4m3** | 8.25 | **+0.50** | +0.47 | +1.13 | 1/5 |
| int8 group-32 | 8.25 | +0.61 | +0.25 | +2.29 | 1/5 |
| MXFP8-e5m2 | 8.25 | +8.65 | +4.91 | +21.95 | 4/5 |
| **MXFP6-e2m3** | 6.25 | **+1.95** | +1.24 | +5.86 | 3/5 |
| int6 group-32 | 6.25 | +2.25 | +1.25 | +7.26 | 2/5 |
| MXFP6-e3m2 | 6.25 | +8.80 | +4.83 | +22.31 | 4/5 |
| int6 per-channel | 6 | +5.18 | +4.43 | +10.41 | 5/5 |
| NVFP4 | 4.5 | +27.41 | +16.41 | +51.13 | 5/5 |
| MXFP4-e2m1 | 4.25 | +38.53 | +23.37 | +104.26 | 5/5 |
| int4 group-32 | 4.25 | +46.97 | +35.10 | +121.66 | 5/5 |
| int4 per-channel | 4 | +114.86 | +87.22 | +228.17 | 5/5 |
| int3 group-32 | 3.25 | +246 | +168 | — | 5/5 |
| ternary group-32 | 1.6 | +263 | +281 | — | 5/5 |
| int1 group-32 | 1.25 | +1865 | +1135 | — | 5/5 |
| int2 group-32 | 2.25 | +3038 | +2631 | — | 5/5 |

**8 bits is free. 4 bits breaks. ≤2 bits is dead** — and dead rather than degraded:
int1, int2 and ternary give **R² ≤ 0 on every dataset** (Lipophilicity int2 reaches R² = −1233),
meaning worse than predicting the training mean. There is no normalization layer anywhere in the
message-passing loop to absorb per-layer bias, which is plausibly why a D-MPNN tolerates this so
much worse than a transformer, where LayerNorm rescales after every block.

## Mantissa bits dominate exponent range

The cleanest mechanism in the study falls out of two format pairs at matched bit width:

| | 3 mantissa bits | 2 mantissa bits | ratio |
|---|---|---|---|
| 8-bit | e4m3: **+0.50%** | e5m2: +8.65% | 17× |
| 6-bit | e2m3: **+1.95%** | e3m2: +8.80% | 4.5× |

Spending a bit on exponent range instead of mantissa precision costs 4.5–17×. CheMeleon's weights
are heavy-tailed (kurtosis 6–12.7) but *narrow in dynamic range* — so the exponent bits are
wasted on range that isn't used, while the mantissa bits are doing all the work.

That single observation replaces a more complicated recommendation. My earlier recipe was mixed
precision: protect `W_i` at 8 bits, run everything else at MXFP6-e3m2. Adding a fifth dataset
broke it (+3.80% mean, +12.06% worst on BACE). **Plain uniform MXFP6-e2m3 is both better and
simpler.**

## Four things I had to retract

- **"4-bit is survivable at +20.7%"** → +38.5% across five datasets. The single-dataset number was
  optimistic by nearly 2×.
- **"MX float beats int for weights"** → refuted; int wins **5/5** at 6 bits. It holds strongly for
  *activations*, where int4 at +0.25% beats MXFP4 at +37.27% — a 150× gap. Match the format to the
  distribution, and weights and activations have different distributions.
- **"NVFP4 underperforms MXFP4"** → refuted; NVFP4 is better on 4/5 for weights and **5/5** for the
  cache (+0.74% vs +37.27%). My original was an ESOL artifact.
- **"`W_i` is sensitive because `H_0` is re-injected every layer, so its error amplifies 6×"** →
  the sensitivity is real (`W_i` is worst on 4/5; +16.19% vs `W_h`'s +4.26% under MXFP4) but my
  mechanism was wrong. Measured Jacobian amplification per unit weight error across depths 1–5 is
  **flat**: 0.69, 0.58, 0.62, 0.61, 0.66. No growth with depth. `W_i`'s edge is a constant
  ~1.6–1.8×, and the real driver is weight tying — one `W_h` reused across all iterations, so its
  error compounds in a way a per-layer bound captures and my residual story did not. Protecting
  `W_i` buys 2.6×, not the 10× I claimed.

One more, which I'd call useful rather than embarrassing: I said cosine similarity was a useless
diagnostic because int1 weights kept cos = 0.844 while RMSE rose 11×. It is in fact an excellent
*ranker* — Spearman(cos, damage) = **−0.941** across 76 configurations. What it cannot do is give
absolute scale, because the entire usable range compresses into cos ∈ [0.996, 1.000], where 0.9964
means +40.6% and 0.9996 means +3.9%. **Rank with it; never threshold on it.**

## Independent corroboration

Two results from groups with no connection to this work, which is the strongest evidence here
since everything else is my own measurement:

- **J. Cheminformatics 2025** measures W4A4 on a GCN with ESOL at **+25.4% RMSE**. I measured +28%
  on CheMeleon — different architecture, different group, same conclusion.
- **Degree-Quant**'s ZINC table shows INT8 is free or *better* on molecular graphs (GIN 0.414 →
  0.357), matching the 8-bit verdict.

## The hardware catch

The accuracy table above says MXFP6-e2m3. The hardware says you may not be able to use it.

All MX formats are **block-32 with a UE8M0 shared scale**, so `d_h=300` violates `K % 32` and is
categorically excluded from every hardware MX path. Padding 300 → 384 wastes 28% to buy 2×, which
is a wash. MX acceleration is Blackwell and CDNA4 only; Hopper has FP8 per-tensor only and
accumulates in FP22. And on Apple Silicon, fp8 measures **1.007–1.15× versus fp16** — no compute
advantage at all; only INT8 gets ~2×, on M5 and later.

So the portable recommendation is **bf16/fp16 activations with fp32 accumulators** — free on
accuracy, 2× ceiling, works everywhere, no alignment constraints. Treat the MX results as what
they are: an accuracy characterisation that becomes actionable when you're on silicon that
implements the format and a `d_h` divisible by 32.

Related, from the kernel work in Part 4: low precision does nothing for the scatter on MPS, which
takes a compare-and-swap path regardless of dtype. **Low precision only starts paying once the
scatter is gone.**

## Recipe

**Weights MXFP6-e2m3 (or bf16 for portability); environment cache int4 group-32.**
+1.95% and +0.25%. If 8 bits is acceptable, int8 per-channel or MXFP8-e4m3 are free. Avoid e5m2
and e3m2. **Do not use 4-bit weights** — the cheapest 4-bit configuration tested still costs
+13.3%.

## Not validated

Stated plainly because the gaps are substantial:

- **Every classification verdict is inconclusive.** The sweep completed on regression only; BACE,
  BBBP, ClinTox, SIDER, Tox21 and HIV are encoded but unswept.
- No trained MLP head — ridge only. Encoder-side deltas are head-independent by construction, but
  the absolute numbers are not.
- Nothing above 2,200 molecules.
- A bug in my own quantizer, found and corrected mid-study: `fq()` used the generic
  `(2 − 2^−m)·2^emax` maximum, giving e4m3 max = 480. The OCP spec reserves the all-ones mantissa
  at emax for NaN, so the true max is **448**. It affected ~0.5% of weights and made MXFP8-e4m3
  slightly optimistic (+0.50% versus +0.71% spec-correct). Both are reported.

## Reproducing

```bash
python scripts/run_sweep.py results_main.jsonl   # the full sweep
python scripts/hypotheses.py                     # verdict tables + noise floor
```
