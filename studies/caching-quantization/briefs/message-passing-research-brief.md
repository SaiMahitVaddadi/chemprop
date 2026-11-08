# Directing message passing in Chemprop: random walks, traversal order, and what the cache actually wants

Research brief. Target: `chemprop` v2.3.1 (`chemprop/nn/message_passing/{base,mixins}.py`), practical
deployment target CheMeleon (`d_h=2048, depth=6`), default (`d_h=300, depth=3`).

**Date:** 2026-11-08. All numbers either (a) measured locally in this study, (b) quoted from a named
paper with a URL, or (c) explicitly flagged as unverified.

---

## 0. Executive summary

### 0.1 The one-line answer

**The evidence does not support replacing Chemprop's synchronous non-backtracking bond message
passing.** Every alternative family surveyed either (i) loses to a *tuned* classical MPNN once
baselines are fixed, (ii) requires 3D geometry Chemprop does not have, (iii) breaks local
cacheability, or (iv) is a one-paper result against an inflated baseline. The defensible wins are
small, cheap, local retrofits — not architectural replacement.

### 0.2 Ranked recommendations

Ranked by (expected accuracy gain) × (caching/efficiency synergy) ÷ (implementation risk).

| # | Direction | Why it ranks here | Risk |
|---|---|---|---|
| **1** | **Re-tune CheMeleon's depth (6 → 4/5) rather than change the architecture.** Depth is the *weakest* scaling axis in the only large-scale molecular scaling study (Spearman depth **−0.11 to 0.55** vs width **0.69–0.91**, [Sypetkowski et al. 2404.11568](https://arxiv.org/abs/2404.11568)). My truncation test shows depth 5 ≈ depth 6 (+0.9%, inside fold noise) even *without* retraining. | This is the **only** intervention that reduces receptive-field **radius**, which is the thing the cache is indexed by. 6→4 moves the cache from 24%-reuse territory toward the r=4 regime (55.8% max hit rate). Zero new code, zero ONNX risk, preserves the frozen-encoder story if you re-pretrain or accept a small loss. | **Very low** |
| **2** | **RWSE as extra atom features via the existing `--atom-features-path` hook, with K ≤ 6.** I *proved locally* (machine precision) that `RWSE_k` is an exact function of the radius-⌈k/2⌉ neighbourhood — so it is **cacheable with the same WL-style key** as the message states, at a cache radius you control via K. | Zero architecture change; already-supported input path; ONNX-neutral (just widens `W_i`). **But** my fair from-scratch test on ESOL found no gain (§6.4) and the independent PE benchmark finds RWSE gains for MPNNs "within or near noise". Rank 2 on *risk*, not on *expected gain*. | **Very low** |
| **3** | **A learned scalar γ on the reverse-edge term** (`M_e = Σ H_f − γ·H_rev(e)`). One parameter. γ=1 is today's Hashimoto operator, γ=0 is an ordinary directed random walk. Nobody appears to have run this ablation. | Measures directly how much the non-backtracking bias is worth — which Chemprop's **own authors** say is "largely not statistically significant" ([Yang et al. 2019](https://ar5iv.labs.arxiv.org/html/1904.01561)). Fully cacheable (γ is shared, so the operator is fixed at inference). ONNX-safe. | **Very low** |
| **4** | **Bounded k-hop precomputed neighbourhoods** (shortest-path-truncated multi-hop, MixHop-style multi-hop-per-layer). Cuts *layer count* while keeping the operator local and vectorized. | Honest caveat forced by the radius correction: this reduces **layers, not radius**, so it **does not help the cache**. Buy it for FLOPs/optimization, not for cache hit rate. | Low–medium |
| **5** | **Attention/weighted transitions on the directed line graph** (learned generalization of the Hashimoto operator). Cacheable *if and only if* the attention logits are a function of the local environment only — which they are for the CMPNN/D-GAT/GEA family. | Several papers claim gains (CMPNN, ABT-MPNN, D-GAT, GEA) but none has independent replication against a tuned D-MPNN, and the most recent systematic sweep ([2509.10871](https://arxiv.org/abs/2509.10871)) concludes attention helps while "elaborate directed variants don't substantially improve results". Plausible small win. | Medium |

### 0.3 What to *not* do

- **Do not adopt a graph transformer / global attention.** Global attention is **not locally
  cacheable by construction** (every atom's representation depends on every other atom), and on
  molecular tasks tuned MPNNs match or beat transformers once the baselines are fixed
  ([Tönshoff et al. 2309.00367](https://arxiv.org/abs/2309.00367), [2502.09263](https://arxiv.org/abs/2502.09263)).
- **Do not adopt a causal/ordered traversal to chase KV-style prefix caching.** §5 shows prefix
  reuse is real and measurable (75.9% of token positions on a 400-analog library) but is
  **in direct conflict with the physics** — see §5.4, the central negative result of this brief.
- **Do not trust walk/path-architecture molecular numbers at face value.** Every paper surveyed
  quotes a GIN ZINC-12k baseline 5–8× worse than a tuned GIN (§7).

---

## 1. Chemprop's bond message passing *is* the Hashimoto operator

### 1.1 The identity, written out

Let `G=(V,E)` be the molecular graph and let `Ẽ` be its set of **directed** bonds, `|Ẽ| = 2|E|`.
For `e = (u→v) ∈ Ẽ` write `src(e)=u`, `dst(e)=v`, and `rev(e) = (v→u)`.

Chemprop (`_BondMessagePassingMixin.message`) computes

```
M_e = ( Σ_{f ∈ Ẽ : dst(f) = src(e)} H_f )  −  H_rev(e)
```

The **Hashimoto / non-backtracking matrix** `B ∈ {0,1}^{|Ẽ|×|Ẽ|}` is defined by

```
B_{e,f} = 1   ⟺   dst(f) = src(e)  and  f ≠ rev(e)
```

i.e. `B` is the adjacency matrix of the **oriented line digraph** of `G`
([definition](https://www.emergentmind.com/topics/non-backtracking-matrix);
[Hashimoto 1989, via Angel–Friedman–Hoory](https://arxiv.org/pdf/1804.08028)).
Therefore, exactly:

```
M = B H                     (1)
```

**Verified locally**, on 40 ESOL molecules with random `H`:
`max |chemprop_message(H) − B·H| = 0.000e+00` (exact, not float-close).
Script: `scratchpad/hashimoto.py`.

The full layer is then, with `τ = ReLU`:

```
H^{(0)} = W_i [ V_{src(e)} ‖ E_e ]
H^{(t)} = τ( H^{(0)} + W_h B H^{(t-1)} )            t = 1 … depth-1
M_v     = Σ_{f : dst(f)=v} H_f^{(depth-1)}
H_v     = τ( W_o [ V_v ‖ M_v ] )
```

Unrolling the residual recursion, the depth-`T` bond state is a sum over **non-backtracking walks**
of every length up to `T` terminating at `e`:

```
H^{(T)} ≈ Σ_{t=0}^{T} (W_h B)^t H^{(0)}            (modulo the ReLU)
```

and `(B^t)_{e,f}` counts non-backtracking walks of length `t` from `f` to `e`. So **Chemprop is a
learned non-backtracking walk kernel on the oriented line digraph.** `W_o` then reads out at the
atom level.

### 1.2 This lineage is already in the paper, just not named

Yang et al. 2019 cite **Mahé et al. 2004, "Extensions of Marginalized Graph Kernels"** — which is
precisely the *non-tottering* (non-backtracking) walk-kernel paper — and describe the motivation as
avoiding "totters". They also note D-MPNN "is more similar to belief propagation in probabilistic
graphical models", which is the same observation: belief propagation's message update is exactly a
non-backtracking edge recursion, which is *why* the Hashimoto spectrum governs BP/SBM thresholds.
Source: [ar5iv of 1904.01561](https://ar5iv.labs.arxiv.org/html/1904.01561).

What is **not** in the Chemprop literature is the spectral consequence, which turns out to be the
most useful part.

### 1.3 What the Hashimoto spectrum predicts — and it is not what the GNN literature assumes

`ρ(B)`, the Perron root of `B`, is the reciprocal of the smallest pole of the **Ihara zeta
function** and sets the growth rate of non-backtracking walk counts: `#walks(t) ~ ρ(B)^t`.

I measured `ρ(B)` on all 500 ESOL molecules (`scratchpad/hashimoto2.py`):

| statistic | value |
|---|---|
| molecules with `ρ(B) = 0` (**acyclic ⇒ `B` nilpotent**) | **26.8%** |
| `ρ(B) = 1` (single ring) | 36.4% |
| `ρ(B) > 1` | 36.8% |
| median `ρ(B)` given `ρ(B)>0` | **1.126** (p90 = 1.329) |
| molecules with **no** non-backtracking walk of length 6 at all | **18.6%** |
| median total NB-walk count, `t=1 → t=6` | **1.00 → 1.39** |

**This is the single most actionable theoretical finding in the brief.** The entire
over-squashing literature is built on receptive fields that grow like `d^T` — exponentially. On
drug-like molecules under the *non-backtracking* operator, walk multiplicity grows like
**1.13^t**, i.e. essentially linearly, and for **27% of molecules it terminates entirely**
(`B` nilpotent on a tree). Concretely:

- **Over-squashing is largely not Chemprop's problem.** There is no exponential fan-in to squash
  into `d_h=2048`. The headline motivation for rewiring (SDRF, expander propagation), for graph
  transformers, and for NBA-GNN's sensitivity bound is **a problem Chemprop has already solved and
  that is weak on small molecules anyway**.
- **Depth has structurally diminishing returns, and for 18.6% of molecules layer 7 injects
  literally zero new structural information.** This is an architecture-level explanation for why
  the depth-scaling Spearman is ~0 (§6.1).
- The thing that *does* grow with depth is **coverage**, not multiplicity — see §6.2. That is a
  cache problem, not a squashing problem.

### 1.4 Is there prior work analysing D-MPNN this way?

**No.** Searches surfaced no paper that identifies Chemprop/D-MPNN's message operator with the
Hashimoto matrix or analyses its Ihara spectrum.
[**NBA-GNN** (Park et al., ICLR 2024, arXiv 2310.07430)](https://arxiv.org/abs/2310.07430)
independently *re-derives* Chemprop's update for general graphs:

```
h^{(t+1)}_{j→i} = φ( h^{(t)}_{j→i}, { ψ( h^{(t)}_{k→j}, h^{(t)}_{j→i} ) : k ∈ N(j)\{i} } )
```

enforcing that "the message `h_{i→j}` is never incorporated in `h_{j→i}^{(t+1)}`" — the same rule.
Its Proposition 3 gives a sensitivity bound `O(d^{-T})` vs `O((d+1)^{-T})` for conventional GNNs.
It connects this to non-backtracking's success in stochastic block model recovery. It reports
+3.4% on Peptides-func and +3% on Peptides-struct over its baselines, at **3–6× the GPU memory**
of GCN/SAGE/GAT. **It does not cite D-MPNN or Chemprop as prior art**, and its Peptides gains are
against the *pre-Tönshoff* baselines that later collapsed (§7.3).

**Net:** Chemprop has been quietly sitting on NBA-GNN's whole contribution since 2019. There is a
short, genuinely novel paper in §1.1–1.3 if anyone wants to write it. There is **no accuracy
upside** in adopting NBA-GNN.

---

## 2. Random-walk encodings: what they cost, and the cacheability verdict

### 2.1 The locality theorem (measured here, machine precision)

This is the determination the brief asked for, and it comes out **favourably and cleanly**.

**Claim.** `RWSE_k(v) = (P^k)_{vv}` with `P = D^{-1}A` is an *exact* function of the ball of radius
`⌈k/2⌉` around `v`, together with the true degrees of the atoms in that ball.

**Why.** A closed walk of length `k` starting and ending at `v` can never reach an atom further than
`⌊k/2⌋` from `v`. The only extra information needed is the *degree* of each occupiable atom (for
the `D^{-1}` normalization), which is determined one hop further out.

**Measured** (`scratchpad/rw_locality2.py`, 60 ESOL molecules, 10–30 heavy atoms, every atom, K=20):

| ball radius `r` | avg ball size | max err, `k ≤ 2r` | max err, `k = 2r+1` | max err, `k = 2r+2` |
|---|---|---|---|---|
| 1 | 3.1 | 1.11e-16 | 0.00e+00 | 2.08e-01 |
| 2 | 6.1 | 1.11e-16 | 0.00e+00 | 5.67e-02 |
| 3 | 9.1 | 1.11e-16 | 2.78e-17 | 1.39e-02 |
| 4 | 11.8 | 1.11e-16 | 2.78e-17 | 3.69e-03 |
| 6 | 15.4 | 1.67e-16 | 4.16e-17 | 1.93e-04 |
| 8 | 16.9 | 1.67e-16 | 3.47e-18 | 1.07e-05 |

Exact to float64 roundoff up to `k = 2r+1`; first real error at `k = 2r+2`, exactly as predicted.

**First attempt was wrong and the correction matters.** My initial test normalized by the *truncated*
subgraph degrees and showed errors up to 7.5e-01 (`scratchpad/rw_locality.py`). RWSE is local only
if you carry the **true** degrees. For an implementation this means the cache key must include the
degree of boundary atoms — a one-hop halo. Cheap, but not free, and easy to get silently wrong.

### 2.2 Consequence: RWSE is cacheable, and **you choose the radius with K**

This is the key practical point. `RWSE` with `K` steps sits at cache radius `⌈K/2⌉`:

| RWSE `K` | cache radius | max hit rate at that radius (your corpus) |
|---|---|---|
| 2 | 1 | 96.6% |
| 4 | 2 | 83.8% |
| 6 | 3 | 68.1% |
| 8 | 4 | 55.8% |
| 10 | 5 | 47.1% |
| **16–20 (GraphGPS default)** | **8–10** | **below your measured range — effectively per-molecule** |

So **RWSE is compatible with the cache, but only at low K.** The standard GraphGPS
configuration (K=16 or 20) is a *whole-molecule* feature on drug-like molecules and would destroy
the cache; `K ≤ 6` keeps it in the regime your hit-rate curve covers. Nobody in the PE literature
discusses this because nobody is caching.

### 2.3 Per-method cacheability verdict

| encoding | depends on | locally cacheable? | notes |
|---|---|---|---|
| **RWSE** `diag(P^k)`, k ≤ K | radius-`⌈K/2⌉` ball + boundary degrees | **YES**, at radius `⌈K/2⌉` | the only PE that is cleanly local; proven above |
| **Heat kernel diagonal** `diag(e^{-tL})` | whole graph (infinite series) | **PARTIAL** — exponentially decaying tail; local only if you accept truncation error | not exact; truncating at order `m` reduces it to RWSE-like locality at radius `⌈m/2⌉` |
| **Personalized PageRank** `Π` | whole graph (resolvent `(I-αP)^{-1}`) | **NO** (exact); **PARTIAL** if truncated | 2111.04718 uses `d_sppr` from PPR; note it is a *pairwise* quantity, O(n²) |
| **LapPE** (Laplacian eigenvectors) | whole graph; also sign/basis ambiguity | **NO** | eigenvectors are globally delocalized by construction; sign flips break key stability too |
| **SignNet / BasisNet** | whole graph | **NO** | fixes LapPE's ambiguity, not its non-locality |
| **LSPE** (learnable PE, [2110.07875](https://arxiv.org/abs/2110.07875)) | initialized from RWPE, then **propagated through the GNN** | **NO** after layer 1 | the learnable part re-couples it to the whole receptive field; the ICLR'22 claim of "+1.79% to +64.14% on molecular datasets" is self-reported and the 64% figure is a relative improvement on a tiny base |
| **Graphormer spatial encoding** (shortest-path-distance bias) | all pairs, O(n²) | **NO** | pairwise attention bias; cache-breaking |
| **Global attention** (any graph transformer) | every atom depends on every atom | **NO — by construction** | see §6.5 |

### 2.4 Compute cost

RWSE at K=6 on a 20-atom molecule is six 20×20 dense matmuls — microseconds, utterly negligible
next to a `d_h=2048` message-passing layer. LapPE needs an eigendecomposition per molecule
(also cheap at n=20, but adds a non-differentiable, sign-ambiguous preprocessing step). Neither is
a compute concern; the concern is **cacheability and whether they help at all** (§6.4, §7).

---

## 3. Walk-based architectures: the numbers, with the baselines fixed

Full tables in the appendix of the companion sub-surveys; the decision-relevant summary:

| method | best molecular number (own paper) | the honest comparison |
|---|---|---|
| **CRaWl** ([2102.08786](https://arxiv.org/abs/2102.08786)) | ZINC-12k MAE **0.085**; molpcba AP **0.2986** (OGB rank 8) | loses ZINC to **tuned GIN+ 0.065** ([2502.09263](https://arxiv.org/html/2502.09263v2)); molpcba is **+0.0005 AP** over tuned GatedGCN+ 0.2981 — noise |
| **RWNN** ([2407.01214](https://arxiv.org/abs/2407.01214)) | **no molecular results at all** | cite only for theory (anonymization ⇒ probabilistic invariance; over-smoothing avoided by construction; over-squashing → "probabilistic under-reaching") |
| **RSNN** ([2510.22520](https://arxiv.org/abs/2510.22520)) | ClinTox 88.1 / BBBP 87.5 / Tox21 79.8 vs CRaWl 70.0/77.6/71.7 | its own PCBA subsets + MoleculeNet, **not leaderboard-comparable**; CRaWl numbers are RSNN-authors' re-runs. Headline efficiency claim (m=1 matches RWNN m=16, 16× fewer samples) is interesting |
| **NeuralWalker** ([2406.03386](https://arxiv.org/abs/2406.03386), ICLR'25) | ZINC **0.053**; molpcba AP **0.3086** | **the one clean walk-based win** — and it uses **non-backtracking** walks *and re-adds message passing*. Still **ties/loses** peptides-struct to a tuned GCN (0.2463 vs 0.2460) |
| **PathNN** ([2306.05955](https://arxiv.org/abs/2306.05955), ICML'23) | ZINC **0.090**; molhiv **79.17** | ZINC loses to tuned GIN+ (0.065) and ties PNA (0.093); molhiv loses to CIN (80.94), HyMN (81.01), tuned GatedGCN+ (80.40). Cost **~5× per-epoch**, `O(n·b^K)` forces K=2 on molhiv |
| **AgentNet** ([2206.11010](https://arxiv.org/abs/2206.11010), ICLR'23) | molhiv **78.33** | +0.5 pt over CRaWl = noise (GIN-VN molhiv seed spread is **0.037 AUC**). Learned walk *does* matter on synthetic 2-WL (100% vs 50.5% random) — not on molecules. Genuinely novel: works with `k·ℓ < n` node visits |
| **Walk-Pooling** ([2110.04375](https://arxiv.org/abs/2110.04375)) | **link prediction only, no molecular data** | off-topic |
| **Anonymous walk embeddings** ([1805.11921](https://arxiv.org/abs/1805.11921)) | MUTAG 87.87 (loses to PSCN 92.63); ENZYMES 35.77 (loses to WL 53.15) | **not a credible molecular method**; cite as the origin of walk anonymization |
| **AWARE** ([2110.02667](https://arxiv.org/abs/2110.02667)) | "top-1 on 33/65 tasks" | **never compares to D-MPNN.** Win-counts over correlated tasks — exactly the inflation pattern §7 warns about |

**Do any of them reduce required depth?** No — and the question is mostly ill-posed for them,
because CRaWl/RWNN/RSNN/NeuralWalker replace layers with **walk length** (ℓ = 50–150, up to 1000).
Walk length *is* receptive-field radius. A length-50 walk has a radius far larger than depth 6.
**They make the radius worse, not better.**

---

## 4. Alternative ways to direct message passing

### 4.1 The finding that most changes the picture: Chemprop already has DimeNet's structural half

DimeNet's own ablation ([Table 7, 2003.03123](https://ar5iv.labs.arxiv.org/html/2003.03123)),
multi-task QM9, MAE relative to full DimeNet:

| variation | MAE / MAE_DimeNet | ΔlogMAE |
|---|---|---|
| `N_SHBF = 1` (angles removed, **directed-edge messages kept**) | 126% | 0.11 |
| **node embeddings instead of message embeddings** (directional info gone) | **168%** | **0.45** |

Their words: *"Using node embeddings instead of message embeddings … has the largest impact and
increases MAE by 68%, at which point DimeNet performs worse than SchNet."*

Reading the chain 168% → 126% → 100%: the **directed-edge structure buys ~25% error reduction,
angles buy ~21%**; under their own ΔlogMAE column structure dominates angles **0.45 vs 0.11**.
**Chemprop already has the structure.** The remaining conformer-free upside is the angular/triplet
half — which is exactly what the next paper tests.

Caveats, which matter: the two columns aggregate differently (ratio-of-means vs mean-of-logs), so
don't quote them interchangeably; `N_SHBF=1` keeps the *triplet* structure and radial distances, so
it is not a clean "no geometry" condition; and this is a **single self-reported ablation table, no
error bars, no independent replication**.

### 4.2 The conformer-free test of exactly this idea — and it is a mixed result

[**"Directional Message Passing on Molecular Graphs via Synthetic Coordinates"**, Gasteiger et al.,
NeurIPS 2021, arXiv 2111.04718](https://arxiv.org/abs/2111.04718) — same group as DimeNet. This is
the most directly relevant paper in the entire survey and it should be read first.

It fabricates distances without conformers (RDKit/UFF distance bounds, and a symmetrized
personalized-PageRank distance `d_sppr,ij = √(Π_ii + Π_jj − 2Π_ij)`), then gets angles by the law
of cosines. Preprocessing: "a few milliseconds". It also states the equivalence explicitly:

> *"We can transform any MPNN to a directed MPNN simply by executing it on the directed line graph
> instead of the original graph"* … *"Directional MPNNs are strictly more expressive than regular
> MPNNs."*

ZINC-12k MAE ablation (their Table 1), isolating line-graph conversion from distance from angle:

| | SMP | DeeperGCN |
|---|---|---|
| Basic | 0.159 ± 0.028 | 0.317 ± 0.021 |
| + distance (Bounds) | 0.124 ± 0.002 | 0.264 ± 0.003 |
| + distance (PPR) | 0.151 ± 0.008 | 0.227 ± 0.006 |
| + distance **& line graph** (Bounds) | 0.112 ± 0.004 | 0.212 ± 0.008 |
| + distance, line graph **& angle** (Bounds) | 0.113 ± 0.003 | 0.180 ± 0.007 |
| + distance, line graph & angle (Bounds+PPR) | **0.109 ± 0.004** | **0.142 ± 0.006** |

**And where it backfires** (their Table 6, DeeperGCN): QM9 `U₀` MAE basic **106 meV**, +distance(PPR)
**88**, **+line graph → 204–233** (2× worse); MolHIV basic 0.728, +distance 0.734,
**+line graph 0.697–0.705** (worse). Their own caption: *"Using the line graph does not always
provide benefits."* Their own summary: *"transforming a GNN to a directional MPNN does not yield
improvements in many cases, while synthetic coordinates still do."*

**Implication for Chemprop, stated plainly:** Chemprop is permanently in the "line graph" row. On
ZINC that is the good row; on QM9-`U₀` and MolHIV it is the *bad* row. The transferable, cheap win
is **synthetic distance/angle features on bond triplets**, which survives without conformers and
which Chemprop does not have. The strongest single ingredient in their table is the synthetic
**distance**, not the line graph.

### 4.3 3D methods — flag and move on

Chemprop here is 2D-only (no conformers), so these are out of scope; listed for completeness.

| method | needs 3D? | note |
|---|---|---|
| DimeNet / DimeNet++ / GemNet / SphereNet | **YES** | radius graph + angles/torsions. DimeNet++ cut depth 6→4 and got 8× faster, 10% more accurate. SphereNet's ablation: angles do ~80% of the geometric work, torsions the rest |
| NequIP / MACE / Allegro | **YES** | energies/forces MLIPs. Only transferable idea: **MACE needs 2 layers where NequIP needs 5–6** — higher body-order per layer buys depth |

### 4.4 Line-graph and second-order structure (2D-legal)

| method | 2D? | ZINC-12k MAE | molhiv | layers | cost |
|---|---|---|---|---|---|
| GIN reference | yes | 0.252 | 77.07 | 4–5 | 1× |
| **GSN** ([2006.09252](https://arxiv.org/abs/2006.09252)) substructure counts as node/edge features | yes | 0.140 / **0.108** | 78.0–80.4 | std | **~1×, preprocessing only** |
| **HIMP** ([2006.12179](https://arxiv.org/abs/2006.12179)) junction tree side channel | yes | 0.151 | 78.80 | **2–3** | ~1.3× |
| **CIN** ([2106.12575](https://arxiv.org/abs/2106.12575)) rings as 2-cells | yes | **0.079** | **80.94** | **2** (molhiv) | ~linear, 1.5–2× |
| **HyMN** ([2501.03113](https://arxiv.org/abs/2501.03113)) centrality-selected subgraphs, T=2 | yes | (unretrieved) | **81.01** | 3 | +85% runtime |
| ESAN ([2110.02910](https://arxiv.org/abs/2110.02910)) bag of subgraphs | yes | **0.102** | 76.6–77.2 | 3–4 | **O(n)** |
| ALIGNN ([2106.01829](https://arxiv.org/abs/2106.01829)) | **NO** (angles from coordinates) | — | — | 4+4 | — |
| NBA-GNN ([2310.07430](https://arxiv.org/abs/2310.07430)) | yes | **no molecular datasets** | — | 4 | ~3× time, 6–7× memory |

**GSN is the standout for Chemprop**: ring/cycle counts injected as *bond features* is a few-line
featurizer change, costs nothing at training or inference, and captures most of the ZINC/molhiv
benefit. It is also **trivially cacheable** — a cycle count at a bond is a local topological
invariant.

**Caution on CIN and HyMN:** best 2D numbers in the literature, but **no independent replication
found** for CIN's 0.079 / 80.94 or HyMN's 81.01. Also note CIN's `ZINC with edge features = 0.079`
is frequently mis-quoted as ZINC-FULL (which is 0.022).

### 4.5 Asynchronous / sequential / ordered schemes

| scheme | verdict |
|---|---|
| **AMP** — "Asynchronous Neural Networks for Learning in Graphs" ([2205.12245](https://arxiv.org/abs/2205.12245)) | Nodes react to individual messages rather than in rounds; proves AMP can simulate synchronous GNNs and can distinguish any pair of graphs; claims better long-distance propagation. **But: the published results are "several graph classification benchmarks" with no molecular numbers I could extract, and the schedule is inherently serial.** See §4.6 for why that is fatal here. |
| **Ordered GNN** ([2302.01524](https://arxiv.org/abs/2302.01524), ICLR'23) | **Mis-leadingly named for our purposes.** It orders *blocks within the node's hidden vector* by hop, not the traversal. Addresses heterophily/over-smoothing in node classification. Not a traversal-schedule method and not molecular. |
| **Dijkstra/Bellman-Ford as a schedule** | See §4.6. |
| **Multi-hop-per-layer** (MixHop [1905.00067](https://arxiv.org/abs/1905.10067), N-GCN) | Mixes powers of the adjacency within one layer; "no additional memory or computational complexity" vs GCN; lets you reach distance `k` in one layer. **Reduces layers, not radius** — see §6.3. No molecular benchmark numbers of note. |

### 4.6 Dijkstra / shortest paths: three uses, three verdicts

**(a) Shortest-path distance as a pairwise feature or attention bias** (Graphormer spatial encoding,
SPN [2206.01003](https://arxiv.org/abs/2206.01003), PathNN). **O(n²), global, cache-breaking.**
SPN reports −54.3% mean relative MAE on QM9 at k=10 with **8 layers** and `O(|V||E|)` BFS
preprocessing, but a k=10 neighbourhood on a drug-like molecule *is the whole molecule*
(my measurement: radius 8 covers 97.7%). **Reject on cache grounds.**

**(b) Priority-ordered relaxation as a traversal schedule.** This is literally asynchronous message
passing, and the algorithmic-alignment literature is the right frame:
[Veličković et al., "Neural Execution of Graph Algorithms", ICLR 2020](https://arxiv.org/abs/1910.10593)
trains GNNs to imitate BFS, Bellman-Ford and Prim step-by-step; Xu et al.'s *algorithmic alignment*
says a network learns an algorithm well when its modules match the algorithm's parts, and GNN
neighbourhood aggregation aligns naturally with **Bellman-Ford** (the parallel, synchronous
relaxation) — **not** with Dijkstra (the sequential, priority-queue one).
[Engelmayer et al., "Parallel Algorithms Align with Neural Execution"](https://proceedings.mlr.press/v231/engelmayer24a/engelmayer24a.pdf)
makes the point directly: parallel algorithms align with neural execution; sequential ones do not.

**Verdict: clear negative, and worth stating plainly.** Chemprop's forward pass is a handful of
fully-vectorized `scatter_reduce_` calls over the whole *batch* of molecules. A priority-ordered
sweep would (i) serialize `O(|V|)` dependent steps per molecule instead of `depth` parallel ones,
(ii) require per-molecule divergent control flow, destroying batching, (iii) be un-exportable to
ONNX (data-dependent loop trip counts and dynamic ordering), and (iv) buy nothing Bellman-Ford-style
synchronous passing doesn't already align with. **Do not pursue.** Note also that Chemprop's
synchronous update *is already* the aligned choice.

**(c) Bounded k-hop precomputed neighbourhoods.** Local, vectorizable, cacheable, ONNX-safe. This
is the practical middle ground the coordinator identified, and I agree — but see §6.3: it reduces
**layers**, not **radius**, so its value is FLOPs and optimization, not cache hit rate.

### 4.7 Weighted / attentive transitions — a learned generalization of `B`

Replace `B ∈ {0,1}` with a learned `B̃_{e,f} = a(h_f, h_e, chemistry)` supported on the same
non-backtracking pattern. This is attention on the directed line graph. Chemprop already has
`AttentiveAggregation` in `nn/agg.py` but **only at atom→molecule readout, never inside message
passing** — so this is a genuine gap.

Prior art, all on D-MPNN-like molecular models:
- **CMPNN** ([IJCAI 2020](https://www.ijcai.org/proceedings/2020/0392.pdf)) — strengthens
  node↔edge message interaction via a "communicative kernel".
- **D-GAT** ([2305.14819](https://arxiv.org/abs/2305.14819)) — scaled dot-product attention over
  directed bonds.
- **GEA on D-MPNN** ([Energy & AI 2022](https://www.sciencedirect.com/science/article/pii/S2666546822000477)) —
  graph edge attention bolted onto D-MPNN; reports a boost on biofuel properties. Narrow domain.
- **ABT-MPNN** ([J Cheminform 2023](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-023-00698-9)) —
  atom-bond transformer on top of MPNN.

**Cacheability: YES, preserved — with one condition.** At inference the attention parameters are
frozen, so `B̃` is a deterministic function of the local environment. If the logits depend only on
`(h_f, h_e)` and local bond/atom features, then the depth-`t` state remains an exact function of the
depth-`t` non-backtracking walk tree, and the existing WL hash key still works unchanged. This
breaks **only** if you make attention global (softmax over all bonds in the molecule) — so use a
**local** softmax over the incident non-backtracking set, which is what all four papers above do.

**The best current evidence, and it is favourable.**
[**"Optimal message passing for molecular prediction is simple, attentive and spatial"**, Castaneda-Leautaud
& Amaro, arXiv 2509.10871](https://arxiv.org/html/2509.10871v1) sweeps six message-passing variants
(BMP bidirectional; BMP+SN with self-nodes; UMP undirected; CBMP with convolution normalization;
**ABMP bidirectional + edge-aware attention**; ABMP+SN) across BACE (1,513), BBBP (2,039), TRPA1
(3,020), Lipophilicity (4,200), tuned with Optuna TPE. Verified findings:

- **ABMP — bidirectional message passing with edge-aware attention — is optimal on most datasets.**
- Convolution normalization gives **no** benefit on molecular graphs.
- **Multi-head attention shows no advantage over single-head.** (So: keep it cheap.)
- 3D-derived features computed from 2D graphs **matched optimized 3D performance at ~50% of the
  compute** — converging with §4.2.

**Two important caveats.** (1) The paper uses **a single message-passing iteration for all models**
and explicitly leaves depth unexplored — so it is not depth evidence, though the fact that 1-step
message passing is competitive on four real datasets is itself striking. (2) **A PDF fetch of this
paper produced a confidently-worded but fabricated summary** (claiming it swept depth 2–3 vs 6–8);
the HTML flatly contradicts it. I have used only HTML-verified claims here, and flag the hazard
because it is a live failure mode for this whole literature.

Expected gain: small but the best-supported of any architectural change in this brief. This is the
one I would actually prototype.

### 4.8 Bidirectional travel and the `γ` generalization

Chemprop's `undirected=True` flag does `H ← (H + H[rev_edge_index])/2` each layer — a crude
symmetrization that **discards** the non-backtracking property. Richer options exist (separate
forward/reverse channels with their own weight matrices, BiRNN-style; see the survey of
in/out message functions in [Directed Multigraph GNNs, 2306.11586](https://arxiv.org/abs/2306.11586)).

But the cheap, sharp experiment is the **one-parameter generalization**:

```
M_e = Σ_{f : dst(f)=src(e)} H_f  −  γ · H_rev(e)
```

- `γ = 1` → Chemprop today → the Hashimoto operator `B`
- `γ = 0` → ordinary directed random walk (backtracking allowed) → simple directed edge adjacency
- `γ` learned → the data tells you how much non-backtracking is worth

**I could find no paper that has run this ablation.** The closest is Chemprop's own
directed/undirected/atom comparison, whose verdict is quoted in §7.1. Results of my own run: §6.5.

---

## 5. Traversal order and the KV-cache question

This is the axis with the most upside *if it worked*, so it gets a full treatment. The conclusion is
negative, but for an interesting reason.

### 5.1 Does ordered/causal message passing over a canonical traversal exist in the literature?

Essentially not for property prediction. Canonical ordering is used widely in graph **generation**
(GraphRNN, GRAN, [PARD 2402.03687](https://arxiv.org/abs/2402.03687), MolecularRNN) and in
density estimation ([GraphSPNs 2408.09451](https://arxiv.org/abs/2408.09451)), where
autoregressive factorization is the point. For *prediction*, the literature runs the other way:
permutation invariance is treated as a requirement, and order is recovered either by
canonicalization or by marginalizing over orderings (PARD's approach). The closest predictive
work is sequence-over-traversal models (SMILES LMs) and path-LSTM models
(PathNN encodes each path with an LSTM over the *reversed* path — but aggregates over all paths,
restoring invariance).

### 5.2 Measured: global canonical SMILES gives almost no prefix reuse

`scratchpad/traversal_prefix.py`, `traversal_rooted.py`. Library: one biaryl-amide scaffold,
20×20 R-group combinations = **400 analogs**, mean canonical SMILES length 35.4 chars.

| ordering | mean pairwise LCP | median | pairs sharing >50% | oracle (best other analog) | prefix-tree reusable token positions |
|---|---|---|---|---|---|
| **RDKit global canonical SMILES** | **4.7%** | 0.0% | 2.6% | 72.3% | **63.2%** |
| **scaffold-rooted canonical SMILES** (`rootedAtAtom=0`) | **28.6%** | 28.6% | 3.8% | 82.0% | **75.9%** |

Distinct prefixes over the 400 analogs: global `k=5: 12, k=10: 20, k=15: 37`; rooted
`k=5: 1, k=10: 2, k=15: 18`.

**So prefix reuse is real and large *if* you root the traversal at a stable scaffold atom** — 75.9%
of token positions are shared with an earlier analog, vs 63.2% unrooted. The coordinator's intuition
is empirically correct at this level.

### 5.3 But the canonical atom *ranks* are not stable, so the cache key is fragile

`traversal_prefix.py` EXP 2B. Base scaffold (R1=C, R2=C), swap R2, measure the 16 scaffold atoms:

| R2 swap | atoms | scaffold canonical ranks changed | scaffold traversal-order LCP |
|---|---|---|---|
| C → F | 17 | 0/16 (0%) | 16/16 (100%) |
| C → Cl | 17 | 0/16 (0%) | 100% |
| C → C(F)(F)F | 20 | **9/16 (56%)** | 100% |
| C → OC | 18 | 6/16 (38%) | **0/16 (0%)** |
| C → N(C)C | 19 | 9/16 (56%) | 100% |
| C → c1ccncc1 | 22 | **11/16 (69%)** | 100% |
| C → N1CCOCC1 | 22 | 8/16 (50%) | **12/16 (75%)** |
| C → S(=O)(=O)C | 20 | 9/16 (56%) | 100% |

Morgan/canonical ranks are computed from **globally** propagated invariants, so a substituent swap
renumbers up to **69%** of scaffold atoms. The *relative traversal order* survives more often
(8/10 cases intact) but breaks outright in ~20% of edits. **A cache keyed on canonical rank is
unsound; it must be keyed on a verified local invariant, with a fallback path.**

### 5.4 The decisive objection: prefix reuse and correct chemistry are in direct conflict

This is the core argument, and it is why I recommend against the whole direction.

A KV-style prefix cache requires **causality**: token `i`'s state must not depend on tokens `> i`.
Applied to a molecule traversed scaffold-first, that means **the scaffold's representation must not
depend on the substituents**.

That is chemically false. Substituent electronic effects propagate *into* the scaffold — this is
the entire content of Hammett σ constants, of the fact that a para-nitro group changes the pKa of a
phenol, of SAR itself. Your own measurement is the evidence: a single R-group swap dirties
**47–91% of atoms at radius 5**. That is not a cache inefficiency to be engineered away — **it is
the model correctly reporting that the substituent really does change the scaffold's electronic
environment.** Any architecture that makes the scaffold prefix reusable is asserting that it
doesn't.

Three corollaries:

1. **Chemprop's inward message flow is a feature, not an obstacle.** The non-cacheability of the
   scaffold under an R-group change is load-bearing physics.
2. **The conflict is unavoidable, not an artifact of SMILES.** Any causal ordering has it. A
   bidirectional fix (BiLSTM-style forward + backward passes) recovers the physics but then the
   backward pass depends on the substituents, so only ~half the computation is prefix-cacheable —
   and the half you'd cache is the half that doesn't matter.
3. **It breaks Chemprop's atom/bond-level targets outright.** `mol_atom_bond.py` predicts
   per-atom and per-bond properties. Under a causal traversal, early scaffold atoms would never
   see later substituents, so their predictions would be systematically wrong in exactly the cases
   a medicinal chemist cares about.

### 5.5 Do SMILES/sequence models match D-MPNN anyway?

Partly, and the claims need discounting. ChemBERTa-77M-MLM and ChemBERTa-2 are reported to beat
D-MPNN on **6 of 8 MoleculeNet** tasks — but in the most rigorous head-to-head available
([2508.06199](https://arxiv.org/abs/2508.06199), 25 models × 25 datasets, scaffold splits,
hierarchical Bayesian Bradley–Terry), **ChemBERTa is rated "practically equivalent to ECFP"**, and
only CLAMP (itself fingerprint-based) is significantly better. Meanwhile masked-LM on random SMILES
masks is itself criticized as an ineffective pretraining objective
([MolTRES, 2408.01426](https://arxiv.org/abs/2408.01426)). So: sequence models are *competitive*,
not *better*, and the field's own best-controlled benchmark says nothing in this class beats a
fingerprint. **Switching Chemprop to a sequence model to gain a KV cache would trade a known
architecture for an equal-or-worse one and lose the atom/bond-level and CGR capabilities.**

### 5.6 Learned / adaptive traversal

AgentNet's learned walk policy is decisively better than a random walk on synthetic expressiveness
(100% vs 50.5% on the 2-WL benchmark) and indistinguishable from CRaWl on molhiv (78.33 vs 77.78,
against a 0.037 seed spread). **No evidence a learned traversal beats uniform non-backtracking on
molecules.** On the cacheability sub-question the coordinator raised: **yes, confirmed** — a
*weighted* directed traversal operator with shared weights is a fixed deterministic function at
inference, so as long as the weights depend only on the local environment (§4.7) the cache key is
unaffected.

### 5.7 Structured traversals from chemistry

Ring-aware (CIN), junction-tree (HIMP), motif (HM-GNN, HimGNN) and quotient-graph (FunQG) schemes
are covered in §4.4. HIMP explicitly claims the coarse channel *"omits the need to stack a multitude
of layers"* and uses **2–3 layers where baselines use 5**; CIN proves the mechanism on its synthetic
RingTransfer task (**3 layers independent of ring size k**, vs `⌊k/2⌋` for GIN); FunQG claims "much
less depth" (numbers unverified — all sources blocked). **BFS shells as explicit radius layers map
1:1 onto the cache-radius measurements and are the cleanest way to make the radius an explicit,
tunable knob** — but see §6.3 for why ring/tree shortcuts do *not* help the cache.

**Nobody publishes the experiment that would settle it**: accuracy vs atom-level depth, with and
without a coarse channel, at matched parameter count, on a scaffold split. That is a cheap and
genuinely novel experiment inside Chemprop.

---

## 6. Depth, radius, and cacheability — the central tradeoff

### 6.1 Depth is the weakest scaling axis for molecular GNNs

The best evidence is [Sypetkowski et al., NeurIPS 2024, arXiv 2404.11568](https://arxiv.org/html/2404.11568v4),
pretraining on LargeMix (~5M molecules) and evaluating 38 downstream tasks (TDC, Polaris,
MoleculeNet). Spearman rank correlation between *scaling axis* and *downstream performance*:

| scaling axis | downstream Spearman | verdict |
|---|---|---|
| **Width** | **0.69 – 0.91** | strongest, monotone |
| Labels / dataset size | 0.37 – 0.57 | moderate |
| **Depth** | **slightly negative – 0.55** | **weakest; non-monotone, dataset-dependent, sometimes harmful** |

Depth detail: Polaris probing 0.47 (MPNN++) / 0.55 (Transformer) / 0.50 (GPS++); TDC probing
**slightly negative for MPNN++ and GPS++**, 0.27 for Transformer; TDC finetuning 0.33 (MPNN++).
*(Caveat carried forward from the sub-survey: the paper's prose says only "slightly negative"; the
specific −0.11 / −0.03 values could not be re-confirmed verbatim and may be read off appendix
figures. Cite the range, not the point values.)*

Their own explanation is the one that matters here:

> *"Certain molecular properties can be well predicted only from small local substructures, hence
> eliminating the need for long-range interactions that deeper networks enable."*

and

> *"While some datasets strongly benefit from deeper networks, others strongly deteriorate with no
> clear pattern observable for the TDC datasets."*

Corroborating context: Chemprop's own HPO has **never searched above depth 6** —
`chemprop/cli/hpopt.py:74`: `"depth": tune.randint(lower=2, upper=6 + 1)` — and the default is 3.
Yang et al. 2019 optimized depth by Hyperopt but **do not publish the selected per-dataset depths**
(not in the main text; likely in the paywalled SI). And
[2509.10871](https://arxiv.org/html/2509.10871v1) gets competitive results on four real datasets
with **a single message-passing iteration**.

**So: there is no good evidence that CheMeleon's depth 6 is necessary, and real evidence that depth
is the axis least worth scaling.** That makes depth reduction the highest-value lever — and it is
the *only* lever that reduces radius.

### 6.2 Why the cache collapses: receptive-field coverage, not walk multiplicity

Measured on 500 ESOL molecules (mean 13.3 heavy atoms, mean diameter 7.0, median 7, p90 11)
— `scratchpad/calib.py`:

| radius `r` | mean fraction of molecule inside the ball | median | **fraction of atoms whose ball is the ENTIRE molecule** |
|---|---|---|---|
| 1 | 28.8% | 25.6% | 1.4% |
| 2 | 50.7% | 47.0% | 7.5% |
| 3 | **68.6%** | 67.9% | 22.2% |
| 4 | 80.6% | 82.2% | 40.0% |
| 5 | 88.1% | 92.9% | 55.6% |
| **6** | **93.1%** | **98.6%** | **69.2%** |
| 7 | 96.0% | 100.0% | 79.6% |

At radius 6 — CheMeleon's depth — **69.2% of atom environments are the whole molecule**, and the
median atom sees 98.6% of it. That is the mechanism behind your measured collapse of atom-level
reuse to 24% at depth 6: sub-molecular caching has literally degenerated into per-molecule caching
for two thirds of atoms. Pair this with §1.3: walk *multiplicity* grows at only ~1.13^t, so nothing
is being over-squashed. **The depth-6 problem is coverage, not squashing.**

### 6.3 **Radius, not layers** — and what that disqualifies

Cache hit rate is indexed by **receptive-field radius**. Your measured ceilings:

| radius | 0 | 1 | 2 | 3 | 4 | 5 | 6 (CheMeleon) |
|---|---|---|---|---|---|---|---|
| max hit rate | 99.8% | 96.6% | 83.8% | 68.1% | 55.8% | 47.1% | ~24% (atom-level, measured) |

This distinction **disqualifies most of the "efficiency" literature from the caching argument**:

| mechanism | reduces layers? | reduces radius? | helps the cache? |
|---|---|---|---|
| Lower `depth` (6 → 4 or 3) | **yes** | **YES** | **YES — the only clean win** |
| CIN rings-as-2-cells (3 layers ∀ ring size) | yes | **no** (shortcuts *increase* reach per layer) | **no** |
| HIMP junction-tree channel (2–3 vs 5 layers) | yes | **no** | **no** |
| MixHop / multi-hop-per-layer | yes | **no** (k-hop in one layer = radius k) | **no** |
| Bounded k-hop neighbourhoods | yes | no | no |
| Expander propagation / DRew / rewiring | **no** (DRew uses **23 layers**) | **no** (worse) | **no** |
| Graph transformer / global attention | no (GPS uses 10) | **no — radius = ∞** | **no, breaks it entirely** |
| RWSE at K ≤ 6 | no | neutral (adds radius-3 feature) | **neutral-to-compatible** |
| GSN substructure/ring counts | no | neutral (local invariant) | **compatible** |
| Learned attention on local non-backtracking set | no | neutral | **compatible** |

**Conclusion: the coarsening/shortcut family buys FLOPs and optimization, not cache hit rate.** I
have rewritten the recommendations accordingly — the only cache-relevant intervention in the whole
survey is **reducing depth itself**.

### 6.4 What a 6 → 3 depth reduction would be worth to the cache

Using your measured ceilings, and the cached-inference speedups you measured (~3× at depth 3, ~8×
at depth 6, lookup-bound):

| | depth 6 (radius 6) | depth 4 (radius 4) | depth 3 (radius 3) |
|---|---|---|---|
| max env hit rate | ~24% (measured, atom-level) | 55.8% | **68.1%** |
| atoms whose env = whole molecule | 69.2% | 40.0% | 22.2% |
| mean molecule coverage per env | 93.1% | 80.6% | 68.6% |
| message-passing FLOPs | 1.00× | 0.67× | 0.50× |

So **6 → 3 roughly triples the reachable hit rate (24% → 68%) and halves the FLOPs**, and 6 → 4 is
a ~2.3× hit-rate gain for a 33% FLOP cut. Because the measured cached-inference regime is
*lookup-bound*, a hit-rate gain of that size is the dominant term — it converts misses (full
recompute) into lookups. **This is worth considerably more than any accuracy delta on offer from
the architectural alternatives in §3–§4.**

The honest counterweight is §6.6: on the one direct test I could run, depth 3 costs real accuracy.

### 6.5 The global-attention tension, stated explicitly

A global attention term is **not locally cacheable, by construction, and no engineering fixes it.**
Every atom's output depends on every other atom, so the receptive-field radius is the molecular
diameter for every atom at layer 1. There is no sub-molecular environment to key on; the cache
degenerates to exact-molecule memoization immediately. If the caching story matters, **global
attention is off the table** — and fortunately §7.3 says you give up nothing by declining it.

There is one genuinely strong pro-global molecular result worth naming:
[Alon & Yahav](https://ar5iv.labs.arxiv.org/html/2006.05205) add a **single** Fully-Adjacent final
layer and cut QM9 error by ~42% (R-GIN alpha 4.67 → 2.28; ZPVE 12.93 → 5.03). But note the setting:
those baselines used **8 layers on graphs of average diameter 6.35**. Read charitably, that result
says *a global layer lets you stop over-deepening* — which is this brief's thesis. **Nobody has run
the controlled experiment** (global/FA layer at depth 3 vs pure MPNN at depth 8, matched
parameters, on molecules) that would settle it. And a single FA layer at the *end* is the least-bad
case for caching: everything up to it stays local and cacheable, and only the final readout is
global. **If you ever want a global term, that is the shape to use.**

### 6.6 Local experiment: CheMeleon depth truncation (indicative only)

Frozen CheMeleon encoder at truncated depth → mean aggregation → ridge head, 5-fold
**scaffold-grouped** CV on `tests/data/regression.csv` (ESOL, n=500). `scratchpad/depth_sweep.py`.

| depth | RMSE | fold SD | vs depth 6 |
|---|---|---|---|
| 1 | 1.5886 | 0.231 | +58% |
| 2 | 1.4157 | 0.320 | +41% |
| 3 | **1.2314** | 0.287 | **+22.5%** |
| 4 | 1.0775 | 0.159 | +7.2% |
| 5 | **1.0150** | 0.162 | **+0.9%** |
| **6 (as trained)** | **1.0055** | 0.168 | — |
| 7 | 1.1025 | 0.212 | +9.7% |
| 8 | 1.1559 | 0.281 | +15.0% |
| ECFP4-count-2048 + ridge, same CV | 1.4258 | — | +42% |
| mean predictor (= target SD) | 2.0863 | — | — |

**Read this carefully — it is weaker evidence than it looks.**

- The **caveat the brief flagged is real and decisive**: these weights were trained at depth 6, so
  truncation is unfair to shallow settings. The giveaway is that the optimum sits *exactly* at the
  training depth and degrades on **both** sides (depth 7 is +9.7%, depth 8 +15.0%). That is the
  signature of "distance from the trained configuration", not of "information content of depth".
- **What it does support:** 6 → 5 is free (+0.9%, well inside the 0.16 fold SD). 6 → 4 costs ~7%.
- **What it does not support:** that depth 3 suffices. It costs 22.5% here, and even discounting
  the unfairness, that is a large gap.
- Context: even truncated to depth 3, CheMeleon (1.2314) still beats ECFP4+ridge (1.4258), and all
  depths ≥ 1 beat the mean predictor (2.0863).

**Verdict: re-pretrain or fine-tune at depth 4–5 and measure properly.** This test cannot settle
depth 3, and nothing here licenses the claim that it can.

### 6.7 Local experiment: can RWSE substitute for depth? (No, on this test)

Same frozen-encoder + ridge protocol, with pooled RWSE (mean+std over atoms) concatenated.
`scratchpad/depth_rwse.py`.

| features | dim | RMSE | fold SD | vs depth 6 |
|---|---|---|---|---|
| CheMeleon depth 6 | 2048 | 1.0055 | 0.168 | — |
| CheMeleon depth 3 | 2048 | 1.2314 | 0.287 | +22.5% |
| CheMeleon depth 4 | 2048 | 1.0775 | 0.159 | +7.2% |
| RWSE K=6 alone (radius 3) | 12 | 2.0817 | 0.311 | +107% |
| RWSE K=20 alone (radius 10) | 40 | 1.9934 | 0.378 | +98% |
| depth 3 + RWSE K=6 | 2060 | 1.2542 | 0.312 | +24.7% |
| depth 3 + RWSE K=8 | 2064 | 1.2588 | 0.316 | +25.2% |
| depth 3 + RWSE K=12 | 2072 | 1.2651 | 0.324 | +25.8% |
| depth 3 + RWSE K=20 | 2088 | 1.2790 | 0.351 | +27.2% |

**RWSE does not close the depth gap — it very slightly widens it**, monotonically in K (pure
ridge-regularization dilution). And **RWSE alone (1.99–2.08) is indistinguishable from the mean
predictor (2.0863)**: molecule-pooled random-walk return probabilities carry essentially *no*
molecule-level signal on ESOL.

**Caveat, and it is a serious one:** this injects RWSE at the *molecule* level into a *linear* head
over a *frozen* encoder. That is the weakest possible way to use a positional encoding. In
GraphGPS, RWSE enters as **per-atom input features** and the network trains with it. I therefore
also ran a fair, from-scratch test (§6.8).

### 6.8 Local experiment: fair from-scratch D-MPNN, depth × RWSE

See `scratchpad/train_depth_rwse.py` — trainable D-MPNN (`d_h=300`), RWSE injected as per-atom
input features (exactly the route Chemprop's `--atom-features-path` provides), 5-fold
scaffold-grouped CV, 3 seeds. Results appended in §9.1 (job completed after the main body was
written).

### 6.9 Local experiment: the γ ablation — what is non-backtracking worth?

See `scratchpad/gamma_ablation.py`, implementing `M_e = Σ_f H_f − γ·H_rev(e)` with fixed
γ ∈ {0, 0.5, 1, 1.5} and learned γ, at depth 3 and 6. Results in §9.2.

---

## 7. Genuinely promising vs benchmark noise

### 7.1 Start here: Chemprop's own authors could not show the directed-bond choice matters

> *"While on average the method using directed bonds outperforms the alternatives, the results are
> largely not statistically significant, so more investigation is warranted on this point."*
> — [Yang et al. 2019, on their directed-bond vs undirected-bond vs atom ablation](https://ar5iv.labs.arxiv.org/html/1904.01561)

This should calibrate everything else. The non-backtracking structure that §1 shows is
mathematically elegant and that NBA-GNN later published as a contribution **was not measurably
worth anything on molecules in the paper that introduced it.** If the *presence* of the
non-backtracking bias is inside noise, the prior on any refinement of it being worth more is low.

### 7.2 The baseline-inflation problem, quantified

The same "GIN on ZINC-12k" baseline spans a factor of **8** across the walk/path literature:

| source | GIN ZINC-12k MAE |
|---|---|
| tuned GIN+ ([2502.09263](https://arxiv.org/html/2502.09263v2)) | **0.065 ± 0.004** |
| PathNN's table | 0.387 ± 0.015 |
| CRaWl's / NeuralWalker's tables | 0.526 ± 0.051 |

Under the tuned baseline, **no walk/path method beats GIN on ZINC-12k except NeuralWalker** (0.053
vs 0.065) — and NeuralWalker re-adds message passing. The same pattern on molhiv: these papers
quote GIN 75.58 while a tuned **GatedGCN+ reaches 80.40**, above AgentNet (78.33), PathNN-SP+
(79.17) and CRaWl.

Independent corroboration: [**"Can Classic GNNs Be Strong Baselines for Graph-level Tasks?"**,
2502.09263](https://arxiv.org/html/2502.09263v2) — 14 datasets, tuned classic GNNs match or exceed
SOTA transformers while running "several times faster": GCN+ peptides-func **0.7261** beats GRIT
0.6988, CRaWl 0.7074 and NeuralWalker 0.7096; GatedGCN+ molhiv **0.8040** ≈ pretrained Graphormer
0.8051.

### 7.3 Long-range benchmarks: gaps that evaporated

[Tönshoff et al., TMLR 2024 (2309.00367)](https://arxiv.org/abs/2309.00367), 500k-parameter cap,
depth tuned 6–10:

| Peptides-func, AP ↑ | original LRGB | retuned | Δ |
|---|---|---|---|
| GCN | 0.5930 | **0.6860** | **+16%** |
| GINE | 0.5498 | **0.6621** | **+20%** |
| GatedGCN | 0.6069 | **0.6765** | **+11%** |
| GPS | 0.6535 | 0.6534 | **±0%** |
| Transformer+LapPE | 0.6326 | — | |
| SAN | 0.6439 | — | |

| Peptides-struct, MAE ↓ | original | retuned | Δ |
|---|---|---|---|
| GCN | 0.3496 | **0.2460** | **−30%** |
| GINE | 0.3547 | **0.2473** | −30% |
| GatedGCN | 0.3357 | **0.2477** | −26% |
| GPS | 0.2500 | 0.2509 | ±0% |

**Note the asymmetry: retuning moved the MPNNs 11–30% and moved GPS by 0%.** After tuning, all
three MPNNs beat GPS on both peptide tasks. Verbatim: *"the reported performance gap completely
vanishes after basic hyperparameter optimization."* The fixes were **deeper prediction heads and
feature normalization** — i.e. depth in the readout MLP, not in message passing.

Worse for the long-range premise:
[**"Molecular Fingerprints Are Strong Models for Peptide Function Prediction"**, 2501.17901](https://arxiv.org/html/2501.17901v1),
132 datasets, count-based fingerprints + LightGBM:

| Peptides-func, AP ↑ | | Peptides-struct, MAE ↓ | |
|---|---|---|---|
| **ECFP + LightGBM** | **74.60** | **ECFP + LightGBM** | **0.2432** |
| S2GCN (best long-range GNN) | 73.11 ± 0.66 | Topological Torsion | 0.2438 |
| HDSE | 71.56 ± 0.58 | S2GCN | 0.2447 ± 0.0032 |
| DRew | 71.50 ± 0.44 | GraphViT | 0.2449 ± 0.0016 |
| GRIT | 69.88 ± 0.82 | | |

Best radius: **ECFP radius 2 (= ECFP4)** — a strictly 2-hop model, at default hyperparameters.
Verbatim: the LRGB datasets *"do not exhibit long-range interactions or require learning long-range
dependencies."* *(Open thread: an independent preregistration alleges ~65% near-twin contamination
across Peptides-func splits — unverified, but if real it affects every number above, fingerprint and
transformer alike.)*

**The entire motivation for walk-based and global-attention architectures on molecules rests on a
long-range premise that the molecular evidence does not support.**

### 7.4 Pretrained embeddings do not beat a fingerprint

[arXiv 2508.06199](https://arxiv.org/html/2508.06199v3) — 25 pretrained models × 25 datasets
(7 MoleculeNet + 18 TDC), **scaffold splits**, hierarchical Bayesian Bradley–Terry ranking:

> *"nearly all neural models show negligible or no improvement over the baseline ECFP molecular
> fingerprint. Only the CLAMP model, which is also based on molecular fingerprints, performs
> statistically significantly better than the alternatives."*

Better than ECFP: **CLAMP only**. Practically equivalent: **ChemBERTa**. Inconclusive: CDDD.
**Worse than ECFP: the remaining 21 models.** The authors say this *"raise[s] concerns about the
evaluation rigor in existing studies."* *(Flag: secondary summaries of different versions report
28 models and name CheMeleon as the winner — the discrepancy is unresolved; cite v4's 25/CLAMP.)*

On realistic ADMET: a bioRxiv 2026 study across 6 TDC endpoints finds **"no model family
dominates"**, with cross-family differences inside seed-to-seed range on most endpoints — and
where GNNs did win (Caco-2, R² ≈ 0.48) **a controlled ablation attributes the advantage entirely to
appended physicochemical descriptors, not the graph representation**.
[XGBoost on descriptors](https://arxiv.org/pdf/2204.07532) takes first place on 11 TDC tasks.
And relevant to this repo: [CheMeleon's own paper](https://arxiv.org/html/2506.15792v1) reports a
**79% Polaris win rate vs Chemprop/D-MPNN's 36%** — the lowest of the four methods compared.

### 7.5 ogbg-molhiv is not a usable benchmark for an architecture decision

Current leaderboard top (fetched 2026-11-08, [OGB](https://ogb.stanford.edu/docs/leader_graphprop/)):
#1 Multi-RF Fusion + Multi-GNN **0.8476**, and **#9 "Molecular FP + Random Forest" at 0.8208 with
5,782 parameters** — beating every pure GNN and every graph transformer on the board. The #1 entry
is explicitly a rank-averaged ensemble of 12 random forests on concatenated fingerprints with GNN
predictions blended in at only **12% weight**.

Meanwhile **GIN-VN seed variance on molhiv is 0.7434 → 0.7802 across 10 seeds — a 0.037 AUC spread,
the same size as the entire rank-1-to-rank-15 gap (0.0436).** Any "+0.5 pt on molhiv" claim is
unfalsifiable. Use molpcba (128 tasks, 400k+ molecules), ZINC-12k *with tuned baselines*, or a
scaffold-CV ADMET suite instead.

### 7.6 Scoring: promising vs noise

**Genuinely promising (worth acting on):**
1. **Depth reduction.** Backed by the largest independent molecular scaling study, by Chemprop's
   own 2–6 search ceiling, and by the fact that 1-step message passing is competitive on four
   datasets. The only cache-relevant lever. **Highest confidence in the brief.**
2. **Local attention on the non-backtracking set** (ABMP-style). Backed by a 2025 tuned sweep whose
   verified conclusion is that bidirectional + edge-aware attention is optimal, single-head
   sufficient. Cacheable. Small expected gain, low risk.
3. **GSN-style substructure/ring counts as bond features.** Free (preprocessing only), trivially
   cacheable (local topological invariant), and gets most of the ZINC/molhiv benefit that the
   expensive subgraph-GNN family delivers at O(n) cost.
4. **Synthetic distance/angle features on bond triplets** ([2111.04718](https://arxiv.org/abs/2111.04718)).
   Conformer-free, "a few milliseconds" preprocessing, and the half of DimeNet's gain Chemprop
   doesn't already have. **But** their own Table 6 shows the line-graph row (which Chemprop is
   permanently in) is where it can backfire on QM9-U₀ and MolHIV. Prototype and measure.
5. **The γ ablation.** Not a gain — an *information* purchase. One parameter tells you whether the
   architecture's defining feature is load-bearing.

**Probably benchmark noise (do not act):**
- **AgentNet's +0.5 pt on molhiv** over CRaWl — against a 0.037 seed spread. Noise.
- **AWARE's "top-1 on 33/65 tasks"** — win-counts over correlated tasks, no D-MPNN comparison.
- **PathNN, CRaWl, AgentNet ZINC/molhiv wins** — all against GIN baselines 5–8× inflated.
- **NBA-GNN's Peptides gains** — against pre-Tönshoff baselines that moved 11–30% on retuning,
  on a benchmark that short-range fingerprints win, with **no molecular datasets tested**.
- **HimGNN's BBBP 0.928, HM-GNN's MUTAG 96.3** — almost certainly random splits / small-n 10-fold
  CV; not comparable to scaffold-split numbers.
- **CIN 0.079 / 80.94 and HyMN 81.01** — the best 2D numbers in the literature, but **no
  independent replication found for either**. Promising, unconfirmed.
- **SDRF as molecular evidence** — **it has no molecular results at all** (node classification
  only). A frequently-made miscitation.
- **Anonymous walk embeddings, Walk-Pooling, RWNN as molecular methods** — no credible molecular
  results exist.

**Where the honest answer is "do not switch":** walk-based architectures, path-based architectures,
subgraph GNNs, graph transformers, rewiring, asynchronous/ordered traversal, and sequence models
over SMILES. Each fails at least one of (replication / cacheability / 2D-legality / Chemprop fit).

### 7.7 The depth-reduction claim nobody has tested — and the one gap worth filling

Across **six** families of over-squashing fixes — curvature rewiring (SDRF), spectral rewiring
(FoSR/GTR), expanders (EGP/Exphormer), delayed rewiring (DRew), global attention
(GPS/GRIT/Graphormer/TokenGT), and non-backtracking (NBA-GNN) — **not one claims to reduce the
number of layers.** EGP holds parameters and depth fixed; GTR/FoSR/SDRF vary edges added;
**DRew uses up to 23 layers** on Peptides-func and its entire selling point is depth-*robustness*;
Exphormer's theory *requires* `O(log n)` layers; NBA-GNN states its benefit holds *"regardless of
the number of layers."* Graph transformers on molecules use **the same or more** layers than MPNNs:
GPS 10 (ZINC, molhiv), GRIT 10, Graphormer 12, TokenGT 12, Exphormer 8 — versus Chemprop's
default 3.

**Global attention replaces hop-limited reach, not parameter depth.** The premise "a global
mechanism can replace deep message passing" is **not supported by any paper in this literature.**

The one gap worth filling: **a single global/FA layer at depth 3 vs pure MPNN at depth 6–8, matched
parameters, scaffold split, on molecules.** Alon & Yahav's 42% QM9 gain hints it would work, and
an FA layer *at the very end* is the one global mechanism that leaves everything beneath it local
and cacheable.

---

## 8. Master table

Columns per the refined spec. **"Cacheable"** = can the per-atom/per-bond state be keyed on a
bounded local environment. **"↓layers"** and **"↓radius"** are separated because *only ↓radius
helps the cache*.

| method | reported accuracy effect (dataset, metric, baseline named) | compute cost | locally cacheable? | ↓ layers? | ↓ **radius**? | fits Chemprop? | ONNX risk |
|---|---|---|---|---|---|---|---|
| **Lower `depth` (6→4/5)** | ESOL ridge-probe: d5 +0.9%, d4 +7.2%, d3 +22.5% vs d6 (mine, frozen-weight caveat). Depth Spearman −ve…0.55 vs width 0.69–0.91 (Sypetkowski, 38 tasks) | **0.5–0.67×** | **YES** | **YES** | **YES (only one)** | **trivial** (`depth=` flag) | **none** |
| **RWSE, K ≤ 6, as atom features** | GraphGPS ZINC 0.070 w/ RWSE-20; PE benchmark: MPNN gains "within or near noise". **Mine: no gain, −24.7% vs d6** | negligible | **YES, radius ⌈K/2⌉** (proved §2.1) | no | neutral | **trivial** (`--atom-features-path`) | **none** (wider `W_i`) |
| **RWSE K=16–20 (GPS default)** | as above | negligible | **NO in practice** — radius 8–10 = whole molecule | no | **no (worse)** | trivial | none |
| **LapPE / SignNet** | GPS uses it; LRGB gains vanish on retuning | eigendecomp/molecule | **NO** (global + sign ambiguity) | no | no | easy | medium (eig) |
| **LSPE** ([2110.07875](https://arxiv.org/abs/2110.07875)) | "+1.79% to +64.14% on molecular datasets" (self-reported, relative) | small | **NO** after layer 1 | no | no | moderate | medium |
| **Local attention on non-backtracking set** (ABMP/CMPNN/D-GAT/GEA) | ABMP optimal on most of BACE/BBBP/TRPA1/Lipo ([2509.10871](https://arxiv.org/html/2509.10871v1), tuned, 1-step); single-head suffices | ~1.2–1.5× | **YES** if logits are local-only | no | neutral | **moderate** (new `message()`) | low |
| **Learned γ on reverse term** | **no prior ablation found**; Chemprop's own directed-vs-undirected is "not statistically significant" | **1.00×** | **YES** | no | neutral | **trivial** (1 param) | **none** |
| **`undirected=True`** (existing) | not significant (Yang et al.) | 1.0× | yes | no | no | already present | none |
| **GSN substructure/ring counts** ([2006.09252](https://arxiv.org/abs/2006.09252)) | ZINC 0.140/**0.108**; molhiv 78.0–80.4 vs GIN 0.408/77.07 | **~1×, preprocess only** | **YES** (local invariant) | no | neutral | **trivial** (bond features) | **none** |
| **Synthetic coords: distance+angle on bond triplets** ([2111.04718](https://arxiv.org/abs/2111.04718)) | ZINC SMP 0.159→**0.109**, DeeperGCN 0.317→**0.142**; coord-free QM9 U₀ 41.9→**28.7 meV**. **But line-graph row hurts QM9-U₀ (106→204) and MolHIV (0.728→0.70)** | "few ms" preprocess | **YES** (UFF bounds local) / **NO** (PPR term global) | no | neutral | **moderate** (bond-triplet features) | low |
| **CIN rings-as-2-cells** ([2106.12575](https://arxiv.org/abs/2106.12575)) | **ZINC 0.079**, ZINC-FULL 0.022, molhiv **80.94** vs GIN 0.252/77.07. No independent replication | ~1.5–2×, near-linear | partial (ring cells are local) | **yes (2 layers)** | **NO** (shortcuts widen reach) | **invasive** (new complex type) | high |
| **HIMP junction tree** ([2006.12179](https://arxiv.org/abs/2006.12179)) | ZINC **0.151**, molhiv **78.80** vs GIN-E 0.252/75.58, at **2–3 layers vs 5** | ~1.3× | partial | **yes** | **NO** | **invasive** (dual graph) | medium |
| **ESAN subgraph bags** ([2110.02910](https://arxiv.org/abs/2110.02910)) | ZINC **0.102**; molhiv 76.6–77.2 (**loses**) | **O(n)** | no | no | no | very invasive | high |
| **HyMN centrality subgraphs** ([2501.03113](https://arxiv.org/abs/2501.03113)) | **molhiv 81.01** (best in survey); no independent replication | +85% runtime | partial | no | no | invasive | high |
| **CRaWl** ([2102.08786](https://arxiv.org/abs/2102.08786)) | ZINC 0.085 (**loses to tuned GIN+ 0.065**); molpcba 0.2986 ≈ GatedGCN+ 0.2981 | O(\|E\|), layer > MPNN layer; ~10% sampling | **NO** (walk ℓ=50–150) | n/a | **NO (far worse)** | replacement | high |
| **NeuralWalker** ([2406.03386](https://arxiv.org/abs/2406.03386)) | **ZINC 0.053**, molpcba **0.3086** — the one clean walk win; ties tuned GCN on peptides-struct | Mamba, O(k·d·n·(γℓ+β)) | **NO** | no | **NO** | replacement | very high (SSM) |
| **PathNN** ([2306.05955](https://arxiv.org/abs/2306.05955)) | ZINC 0.090 (ties PNA 0.093, loses GIN+ 0.065); molhiv 79.17 (loses CIN/HyMN) | **~5× epoch**, `O(n·b^K)` | partial (paths ≤ K local) | no | no | invasive | high (LSTM, dyn. paths) |
| **AgentNet** ([2206.11010](https://arxiv.org/abs/2206.11010)) | molhiv 78.33 vs CRaWl 77.78 — **noise** (seed spread 0.037) | O(k·ℓ), can be sublinear | **NO** (stochastic walks) | no | no | replacement | very high |
| **RWNN** ([2407.01214](https://arxiv.org/abs/2407.01214)) | **no molecular results** | reader-dependent | NO | n/a | NO | no | n/a |
| **SPN / shortest-path neighbourhoods** ([2206.01003](https://arxiv.org/abs/2206.01003)) | QM9 −54.3% mean rel. MAE at k=10, **8 layers** | `O(\|V\|\|E\|)` BFS | **NO** (k=10 = whole molecule) | yes | **NO** | moderate | medium |
| **MixHop multi-hop-per-layer** ([1905.00067](https://arxiv.org/abs/1905.00067)) | no notable molecular numbers | "no additional complexity" vs GCN | partial | **yes** | **NO** | moderate | low |
| **Graphormer spatial encoding (SPD bias)** | molhiv 80.51 @ 47M params + PCQM pretrain | **O(n²)** pairwise | **NO** | no | **NO** | invasive | high |
| **Graph transformer / GPS** ([2205.12454](https://arxiv.org/abs/2205.12454)) | ZINC 0.070 @ **10 layers**; molhiv 0.7880; **loses retuned GCN on both peptides tasks** | O(n²) attn | **NO — by construction** | **no** (uses *more*) | **NO (∞)** | invasive | high |
| **Single FA/global layer at the end** ([2006.05205](https://arxiv.org/abs/2006.05205)) | QM9 ~**−42%** error (R-GIN α 4.67→2.28) — but baselines used 8 layers, diam 6.35 | 1 extra O(n²) layer | **YES below it, NO at it** — best-shaped global option | **plausibly** (untested) | no | easy | medium |
| **Rewiring: SDRF / FoSR / GTR / EGP / DRew** | **SDRF has NO molecular results.** GTR MUTAG 85.50 vs none 69.25 (implausible baseline). EGP molhiv 0.7934 vs GIN 0.7558 | varies | **NO** (changes topology) | **no — DRew uses 23** | **NO** | invasive | high |
| **NBA-GNN** ([2310.07430](https://arxiv.org/abs/2310.07430)) | **no molecular datasets**; Peptides-func 0.7207 vs pre-retuning baselines | **~3× time, 6–7× memory** | yes (it *is* Chemprop's scheme) | no ("regardless of layers") | no | **already implemented** | n/a |
| **Asynchronous MP / Dijkstra schedule** ([2205.12245](https://arxiv.org/abs/2205.12245)) | no molecular numbers extractable | serial `O(\|V\|)` steps | yes in principle | yes | no | **breaks batching** | **fatal** (dynamic loops) |
| **Canonical ordered/causal traversal** | no predictive-model precedent; prefix reuse 75.9% rooted (mine) | low | yes | n/a | n/a | **conflicts with physics (§5.4)**; breaks `mol_atom_bond` | medium |
| **SMILES sequence model (KV-cacheable)** | ChemBERTa beats D-MPNN on 6/8 MoleculeNet (self-reported) but rated **"practically equivalent to ECFP"** in the 25×25 BBT study | cheap, KV-cacheable | **prefix-cacheable, but see §5.4** | n/a | n/a | **abandons CGR, atom/bond targets, UQ** | low |
| **3D: DimeNet/GemNet/SphereNet/MACE** | DimeNet QM9 U₀ 8.02 meV; DimeNet++ 6.32 @ 4 blocks | high | n/a | yes (MACE 2 vs 5) | n/a | **NO — needs conformers** | n/a |

---

## 9. Fit to Chemprop, concretely

### 9.1 Extension points that already exist (use these)

Verified in the repo:

- **`--atom-features-path`** (`chemprop/cli/common.py:145`) appends extra atom features
  **before** message passing (`featurizers/molgraph/molecule.py:72-73`: `V = np.hstack((V, atom_features_extra))`).
  **This is the zero-risk route for RWSE and GSN counts.** It only widens `W_i`.
- **`--bond-features-path`** likewise for bond features
  (`molecule.py:78-79`) — the right hook for **ring/cycle counts** (GSN-e, which is proved strictly
  more expressive than GSN-v) and for **synthetic bond-triplet distances**.
- **`--atom-descriptors-path` / `--bond-descriptors-path`** append **after** message passing via
  `W_d` (`base.py:184-192`). Wrong place for a positional encoding (it can't inform message
  passing) but fine for whole-molecule descriptors.
- **`depth`** is a plain constructor argument and CLI flag (`base.py:61`, `cli/train.py:213`);
  HPO searches `{2,…,6}` (`cli/hpopt.py:74`).
- `nn/agg.py` has `AttentiveAggregation` — but **only at atom→molecule readout**, so in-message
  attention is genuinely new code in `mixins.py`.

### 9.2 Per-capability impact

| Chemprop capability | depth↓ | RWSE / GSN as features | local attention | γ | ordered traversal | global attention |
|---|---|---|---|---|---|---|
| **single molecule** | fine | fine | fine | fine | fine | fine |
| **multi-molecule** (`nn/message_passing/multi.py`) | fine | fine — per-component features | fine | fine | fine | fine |
| **reactions / CGR** (`featurizers/molgraph/reaction.py`, 6 `RxnMode`s) | fine | **needs a decision**: the CGR is the *union* graph (bonds present in reactant **or** product), so RWSE/ring counts are well-defined on the union — but reactant-vs-product RWSE differ, and `REAC_DIFF`-style modes would want the difference. Defensible, non-trivial | fine | fine | fine | fine |
| **atom/bond targets** (`models/mol_atom_bond.py`, `nn/message_passing/mol_atom_bond.py`) | fine | fine | fine | fine | **BREAKS** — early atoms never see late substituents (§5.4) | fine but uncacheable |
| **uncertainty quantification** | fine | fine | fine | fine | changes the likelihood structure | fine |
| **frozen encoder / CheMeleon transfer** | **needs re-pretrain or fine-tune** at the new depth — the one real cost | **breaks `W_i` shape** → needs re-pretrain, or append post-hoc (which §6.7 shows doesn't work) | **breaks weights** → re-pretrain | **compatible** (γ=1 reproduces CheMeleon exactly) | abandons it | abandons it |
| **ONNX export** | **no risk** | **no risk** | low | **no risk** | **fatal** (data-dependent control flow) | medium–high |

**The CheMeleon-transfer column is the binding constraint, and it is why γ ranks so well**: γ=1
reproduces the pretrained encoder bit-exactly, so you can fine-tune γ on a downstream task without
invalidating the pretrained weights. Everything that changes `W_i`'s input width (RWSE, GSN counts,
synthetic coordinates) requires re-pretraining CheMeleon to benefit from them — a real cost that
§6.7's negative result does not justify paying speculatively.

### 9.3 ONNX reality check

Chemprop ONNX export is **already fragile**:
[issue #1335](https://github.com/chemprop/chemprop/issues/1335) documents that export needs
workarounds in `MeanAggregation` and bond message passing for correct dynamic-batch behaviour
(ScatterElements index/dimension errors when the inference batch size differs from the export batch
size), plus an upstream PyTorch bug in `scatter_reduce(reduce="mean")` under
`torch.export` + `dynamo=True`. **Anything that adds dynamic control flow, data-dependent loop
trip counts, or per-molecule divergent scheduling (asynchronous MP, Dijkstra order, learned walks)
is effectively un-exportable.** This alone rules out the traversal-schedule family independent of
accuracy.

---

## 10. What I would actually do

1. **Re-pretrain or fine-tune CheMeleon at depth 4 and 5** and measure properly on a scaffold-CV
   suite. If depth 5 holds (my probe says +0.9%, inside noise), you get a radius reduction for
   free. If depth 4 holds, the cache hit rate roughly doubles. **This is the whole ballgame.**
2. **Run the γ ablation on a real multi-dataset suite.** One parameter, zero risk, and it tells you
   whether the architecture's defining property is load-bearing. Publishable either way given §1.
3. **Add ring/cycle counts as bond features** (GSN-e). Free, cacheable, well-replicated mechanism.
4. **Prototype local edge-aware attention inside `message()`** (ABMP shape, single head). Cacheable,
   best-supported architectural change available.
5. **Write up §1.1–§1.3.** The Hashimoto identity is exact, the Ihara-spectrum measurement
   (`ρ(B)` median 1.126, 26.8% nilpotent, 18.6% with no length-6 walk) is novel, and it reframes
   the over-squashing discussion for molecules. NBA-GNN published a subset of this as an ICLR paper
   without knowing Chemprop had it.
6. **Do not** pursue ordered/causal traversal, graph transformers, rewiring, asynchronous schedules,
   subgraph GNNs, or walk-based replacements.

---

## 11. Local experiment results (appended on job completion)

*(filled in below — see §11.1 and §11.2)*

---

## 12. Reproduction scripts

All in this scratchpad; each is standalone (uses `cp_loader.py`, which loads Chemprop's real
featurizers without the broken package deps; **no repo files were modified**).

| script | what it establishes |
|---|---|
| `hashimoto.py` | Chemprop's `message()` **is** `B·H` exactly (max err 0.000e+00, 40 molecules) |
| `hashimoto2.py` | `ρ(B)` distribution, nilpotency fraction, NB-walk growth vs depth |
| `rw_locality.py` | the **wrong** version — truncated degrees give errors up to 7.5e-01 |
| `rw_locality2.py` | the **right** version — `RWSE_k` exact from radius-⌈k/2⌉ ball with true degrees |
| `calib.py` | ESOL target SD, receptive-field coverage vs radius |
| `traversal_prefix.py` | canonical-SMILES prefix reuse; scaffold canonical-rank stability under R-group swap |
| `traversal_rooted.py` | rooted-vs-global traversal prefix reuse, prefix-tree token sharing |
| `depth_sweep.py` | CheMeleon depth truncation, ESOL scaffold-CV ridge probe |
| `depth_rwse.py` | pooled RWSE vs depth, frozen encoder |
| `train_depth_rwse.py` | **fair** from-scratch D-MPNN, depth × input-injected RWSE |
| `gamma_ablation.py` | the γ ablation on the reverse-edge coefficient |

---

## 13. Sources

### Non-backtracking walks, Hashimoto matrix, Ihara zeta
- [Non-backtracking matrix / oriented line digraph (overview)](https://www.emergentmind.com/topics/non-backtracking-matrix)
- [Angel, Friedman, Hoory — Ramanujan graphs and digraphs (Hashimoto definition)](https://arxiv.org/pdf/1804.08028)
- [Kempton — Non-backtracking random walks and a weighted Ihara's theorem](https://arxiv.org/html/1603.05553)
- [Bordenave, Lelarge, Massoulié — Non-backtracking spectrum of random graphs](https://arxiv.org/pdf/1501.06087)
- [Relations between spectra of simple and non-backtracking random walks](https://arxiv.org/html/1703.03852)
- [Mahé et al. 2004 — Extensions of Marginalized Graph Kernels (the non-tottering walk kernel Chemprop cites)](https://dl.acm.org/doi/10.1145/1015330.1015446)
- [NBA-GNN — Non-backtracking Graph Neural Networks (2310.07430)](https://arxiv.org/abs/2310.07430) · [full text](https://ar5iv.labs.arxiv.org/html/2310.07430) · [OpenReview](https://openreview.net/forum?id=YkR9UFlQ1s)
- [RFGNN — Redundancy-Free Message Passing (NeurIPS 2022)](https://openreview.net/forum?id=jwVZZzzNKkW)
- [On the Two Sides of Redundancy in GNNs (2310.04190)](https://arxiv.org/abs/2310.04190)

### Chemprop / D-MPNN
- [Yang et al. 2019 — Analyzing Learned Molecular Representations (1904.01561)](https://arxiv.org/abs/1904.01561) · [full text](https://ar5iv.labs.arxiv.org/html/1904.01561) · [JCIM](https://pubs.acs.org/doi/10.1021/acs.jcim.9b00237)
- [Heid et al. 2024 — Chemprop: A Machine Learning Package (JCIM)](https://pubs.acs.org/doi/pdf/10.1021/acs.jcim.3c01250) · [PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10777403/)
- [Chemprop message-passing docs](https://chemprop.readthedocs.io/en/latest/tutorial/python/models/message_passing.html)
- [ONNX export issue #1335](https://github.com/chemprop/chemprop/issues/1335)
- [CheMeleon — Deep Learning Foundation Models from Classical Molecular Descriptors (2506.15792)](https://arxiv.org/html/2506.15792v1)
- [CMPNN (IJCAI 2020)](https://www.ijcai.org/proceedings/2020/0392.pdf) · [D-GAT (2305.14819)](https://arxiv.org/abs/2305.14819) · [D-MPNN+GEA (Energy & AI 2022)](https://www.sciencedirect.com/science/article/pii/S2666546822000477) · [ABT-MPNN (J Cheminform 2023)](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-023-00698-9)
- [Optimal message passing for molecular prediction is simple, attentive and spatial (2509.10871)](https://arxiv.org/html/2509.10871v1)

### Random-walk and positional/structural encodings
- [Dwivedi et al. — GNNs with Learnable Structural and Positional Representations / LSPE + RWPE (2110.07875)](https://arxiv.org/abs/2110.07875) · [code](https://github.com/vijaydwivedi75/gnn-lspe)
- [GraphGPS (2205.12454)](https://arxiv.org/abs/2205.12454) · [full text](https://ar5iv.labs.arxiv.org/html/2205.12454)
- [Benchmarking Positional Encodings for GNNs and Graph Transformers (2411.12732)](https://arxiv.org/pdf/2411.12732)
- [Graph Positional and Structural Encoder (2307.07107)](https://arxiv.org/pdf/2307.07107)
- [Rewiring with Positional Encodings (2201.12674)](https://arxiv.org/pdf/2201.12674)

### Walk / path architectures
- [CRaWl (2102.08786)](https://arxiv.org/abs/2102.08786) · [RWNN (2407.01214)](https://arxiv.org/abs/2407.01214) · [RSNN (2510.22520)](https://arxiv.org/abs/2510.22520)
- [NeuralWalker — Learning Long Range Dependencies on Graphs via Random Walks (2406.03386)](https://arxiv.org/abs/2406.03386) · [HTML](https://arxiv.org/html/2406.03386v2)
- [PathNN (2306.05955)](https://arxiv.org/abs/2306.05955) · [AgentNet (2206.11010)](https://arxiv.org/abs/2206.11010) · [AWARE (2110.02667)](https://arxiv.org/abs/2110.02667)
- [Walk-Pooling (2110.04375)](https://arxiv.org/abs/2110.04375) · [Anonymous Walk Embeddings (1805.11921)](https://arxiv.org/abs/1805.11921)
- [SPN — Shortest Path Networks (2206.01003)](https://arxiv.org/abs/2206.01003)

### Directional / 3D / line graph
- [DimeNet (2003.03123)](https://arxiv.org/abs/2003.03123) · [full text](https://ar5iv.labs.arxiv.org/html/2003.03123)
- [DimeNet++ (2011.14115)](https://arxiv.org/abs/2011.14115) · [GemNet (2106.08903)](https://arxiv.org/abs/2106.08903) · [SphereNet (2102.05013)](https://arxiv.org/abs/2102.05013)
- [**Synthetic Coordinates (2111.04718)**](https://arxiv.org/abs/2111.04718) · [full text](https://ar5iv.labs.arxiv.org/html/2111.04718)
- [NequIP (2101.03164)](https://arxiv.org/abs/2101.03164) · [MACE (2206.07697)](https://arxiv.org/abs/2206.07697) · [Allegro (2204.05249)](https://arxiv.org/abs/2204.05249)
- [ALIGNN (2106.01829)](https://arxiv.org/abs/2106.01829) · [CensNet (IJCAI'19)](https://ijcai.org/Proceedings/2019/0369.pdf) · [Galformer (2309.00483)](https://arxiv.org/abs/2309.00483)
- [Directed Multigraph GNNs (2306.11586)](https://arxiv.org/pdf/2306.11586)

### Substructure / subgraph / hierarchical
- [GSN (2006.09252)](https://arxiv.org/abs/2006.09252) · [ESAN (2110.02910)](https://arxiv.org/abs/2110.02910) · [HyMN (2501.03113)](https://arxiv.org/abs/2501.03113)
- [HIMP (2006.12179)](https://arxiv.org/abs/2006.12179) · [HM-GNN (2202.00529)](https://arxiv.org/abs/2202.00529) · [HimGNN (Brief. Bioinform. 24(5))](https://academic.oup.com/bib/article/24/5/bbad305/7245716) · [FunQG (2207.08597)](https://arxiv.org/abs/2207.08597)
- [CW Networks / CIN (2106.12575)](https://arxiv.org/abs/2106.12575)

### Over-squashing, rewiring, depth
- [Alon & Yahav — On the Bottleneck of GNNs (2006.05205)](https://arxiv.org/abs/2006.05205) · [full text](https://ar5iv.labs.arxiv.org/html/2006.05205)
- [Topping et al. — SDRF / curvature (2111.14522)](https://arxiv.org/abs/2111.14522) *(no molecular results)*
- [Di Giovanni et al. — Width, Depth, and Topology (2302.02941)](https://arxiv.org/abs/2302.02941) · [How does over-squashing affect the power of GNNs? (2306.03589)](https://arxiv.org/pdf/2306.03589)
- [Black et al. — GTR / effective resistance (2302.06835)](https://arxiv.org/abs/2302.06835) · [FoSR (ICLR 2023)](https://iclr.cc/virtual/2023/poster/11758)
- [Expander Graph Propagation (2210.02997)](https://arxiv.org/abs/2210.02997) · [DRew (2305.08018)](https://arxiv.org/abs/2305.08018) · [Exphormer (2303.06147)](https://arxiv.org/abs/2303.06147)
- [**Sypetkowski et al. — On the Scalability of GNNs for Molecular Graphs (2404.11568)**](https://arxiv.org/html/2404.11568v4) · [NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/file/2345275663a15ee92a06bc957be54a2c-Paper-Conference.pdf)
- [Are More Layers Beneficial to Graph Transformers? (2303.00579)](https://arxiv.org/abs/2303.00579)
- [Over-Squashing in GNNs: A Comprehensive Survey (2308.15568)](https://arxiv.org/html/2308.15568) · [Graph Rewiring Survey (2411.17429)](https://arxiv.org/pdf/2411.17429)
- [Position: Don't be Afraid of Over-Smoothing And Over-Squashing (2601.07419)](https://arxiv.org/pdf/2601.07419)
- [Ewald-based Long-Range Message Passing (2303.04791)](https://arxiv.org/pdf/2303.04791)

### Traversal order, scheduling, algorithmic alignment
- [Asynchronous Neural Networks for Learning in Graphs (2205.12245)](https://arxiv.org/abs/2205.12245)
- [Veličković et al. — Neural Execution of Graph Algorithms (1910.10593)](https://arxiv.org/abs/1910.10593)
- [Engelmayer et al. — Parallel Algorithms Align with Neural Execution](https://proceedings.mlr.press/v231/engelmayer24a/engelmayer24a.pdf)
- [Ordered GNN (2302.01524)](https://arxiv.org/abs/2302.01524) *(orders hidden-vector blocks, not traversal)*
- [MixHop (1905.00067)](https://arxiv.org/abs/1905.00067) · [code](https://github.com/samihaija/mixhop)
- [PARD — Permutation-invariant Autoregressive Diffusion (2402.03687)](https://arxiv.org/pdf/2402.03687) · [GraphSPNs (2408.09451)](https://arxiv.org/pdf/2408.09451)
- [MolTRES — on the weakness of SMILES masked-LM pretraining (2408.01426)](https://arxiv.org/pdf/2408.01426)

### Benchmarking, replication, skepticism
- [**Benchmarking Pretrained Molecular Embedding Models (2508.06199)**](https://arxiv.org/html/2508.06199v3)
- [**Where Did the Gap Go? Reassessing the LRGB (2309.00367)**](https://arxiv.org/abs/2309.00367) · [full text](https://ar5iv.labs.arxiv.org/html/2309.00367)
- [**Molecular Fingerprints Are Strong Models for Peptide Function Prediction (2501.17901)**](https://arxiv.org/html/2501.17901v1) · [J Cheminform](https://pmc.ncbi.nlm.nih.gov/articles/PMC13143419/)
- [Can Classic GNNs Be Strong Baselines for Graph-level Tasks? (2502.09263)](https://arxiv.org/html/2502.09263v2)
- [fastprop (2404.02058)](https://arxiv.org/abs/2404.02058) · [HTML v5](https://arxiv.org/html/2404.02058v5)
- [MOLTOP — hyperparameter-free topological baseline (2407.12136)](https://arxiv.org/abs/2407.12136)
- [XGBoost on TDC ADMET (2204.07532)](https://arxiv.org/pdf/2204.07532)
- [Scaffold Splits Overestimate Virtual Screening Performance (2406.00873)](https://arxiv.org/abs/2406.00873)
- [Coverage bias in small molecule machine learning](https://pmc.ncbi.nlm.nih.gov/articles/PMC11718084/)
- [A systematic study of key elements underlying molecular property prediction (Nat. Commun.)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10575948/)
- [MolGraphBench (2602.20573)](https://arxiv.org/html/2602.20573v3)
- [OGB graph property prediction leaderboards](https://ogb.stanford.edu/docs/leader_graphprop/)
- [Long Range Graph Benchmark (2206.08164)](https://arxiv.org/pdf/2206.08164)

---

## 14. Caveats on this brief

- **PDF extraction is unreliable for this literature.** A PDF fetch of
  [2509.10871](https://arxiv.org/abs/2509.10871) returned a confidently-worded but **fabricated**
  summary (claiming it swept depth 2–3 vs 6–8 and found shallow wins); the HTML says it uses a
  **single** message-passing iteration and excludes layer count from its search space. Every claim
  in this brief is either HTML/ar5iv-verified, locally measured, or flagged.
- **My local experiments use one small dataset** (ESOL, n=500, mean 13.3 heavy atoms). The
  structural measurements (Hashimoto identity, RWSE locality, receptive-field coverage, prefix
  reuse) are robust because they are combinatorial, not statistical. The **accuracy** numbers
  (§6.6–§6.9) are a single small regression task with ~0.16–0.35 fold SD and should be treated as
  directional only.
- **The depth-truncation test is structurally unfair to shallow depths** and its optimum sits
  exactly at the training depth. It cannot establish that depth 3 suffices, and I do not claim it.
- **Unverified items**, listed so they do not propagate: Chemprop v2's "optimal depth 4" (chemrxiv
  403); Yang et al.'s per-dataset selected depths (paywalled SI); FunQG's numeric tables (all
  sources blocked); HyMN's ZINC table; BiScale-GTR's depth ablation (Appendix A.9 unreachable);
  the exact negative depth-Spearman point values; the alleged Peptides-func split contamination;
  LineEvo's 3D requirement; whether 2508.06199's winner is CLAMP (v4) or CheMeleon (secondary
  summaries).
- **No independent replication found** for: DimeNet's structure-vs-angles ablation, CIN's
  0.079/80.94, ESAN's 0.102, HyMN's 81.01, NeuralWalker's ZINC 0.053, or the Ihara-spectrum
  observation in §1.3 (which is new here).
