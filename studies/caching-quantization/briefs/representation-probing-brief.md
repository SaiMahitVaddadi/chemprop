# What does a frozen CheMeleon / D-MPNN representation actually encode?

Literature survey plus hands-on probing of the cached 2048-d CheMeleon embeddings from the
caching/quantization study. Everything measured here reuses the frozen fp32 encoder outputs
already on disk (11 MoleculeNet datasets, 13,226 unique molecules after dedup), so the whole
experimental section is minutes of linear algebra.

- Scripts: `studies/caching-quantization/experiments/rep_lib.py`, `rep1_intrinsic_dim.py`,
  `rep2_probes.py`, `rep3_descriptor_equiv.py`, `rep4_depth_probes.py`, `rep5_scaffold.py`,
  `rep6_env_cache_rank.py`
- Results: `studies/caching-quantization/results/results_rep{1..6}_*.json` (each carries a
  top-level `caveat` list)

Protocol, fixed once and never retuned: ridge in float64 on z-scored features with a penalised
intercept; λ = 100 for downstream task metrics (the value `scripts/prep.py` already chose on
fp32 embeddings) and for descriptor probes; the dual (kernel) solver is verified against
`qlib.Head` to 1.2e-5 on esol/tox21/sider. The companion study's **±2.3% paired-delta
split-noise floor** is treated as the significance threshold throughout.

---

## Part 1 — Literature survey

### 1.1 Probing work on molecular GNNs: thin, recent, and mostly negative

The canonical study is **Wang, Kaddour, Liu, Tang, Lasenby, Liu, "Evaluating Self-Supervised
Learning for Molecular Graph Embeddings" (MolGraphEval), NeurIPS 2023 D&B**
(<https://arxiv.org/abs/2206.08005>): 90,918 probe models over 1,875 pretrained GINs, 9 GSSL
objectives, scaffold 80/10/10. Numbers that matter:

- Frozen-embedding linear probe, mean ROC-AUC over 8 MoleculeNet sets: **random GIN 59.60**,
  ContextPred 60.36, GROVER 63.62, GraphCL 65.04, JOAO 65.05. After fine-tuning, random GIN
  reaches 66.16 — **fine-tuning a random GIN beats every frozen pretrained embedding.**
- Substructure counts fed to classical ML: linear regression 59.91, RF 61.95, XGBoost 62.31 —
  i.e. "which substructures are present" is worth roughly what a random GNN embedding is worth.
- Substructure-probe quality correlates with downstream property performance at Spearman
  **0.77–0.92 (all p < 1%)**. This is the strongest published evidence that what molecular GSSL
  buys is substructure awareness.
- A random GIN is *best or second-best* on 4 of 6 node/pair-level topology probes (node degree
  MSE 0.001 vs 0.009–0.199) and *worst* on graph-level ones (diameter 177.9 vs GROVER 78.4).
  The authors trace the random-net advantage to GIN init + sum aggregation.
- Split choice moves probe scores ~2%; probe init seed moves them <0.01%.

Two 2025–26 preprints are methodologically the closest templates for what we did here. **Varshini
A S, Ranasinghe, Tam, "Interpreting GFlowNets for Drug Discovery: What probes can and cannot
show," arXiv:2511.19264** reports linear-probe R² 0.996 for molecular weight and 0.996 for
heavy-atom count — and then an architecture-matched *random-init* control that scores 0.581 vs
0.586 trained on drug-likeness (**+0.005**), with the largest gain anywhere being +0.036. Against
a leakage-free RDKit+ECFP4 baseline the embedding never exceeds descriptors by more than +0.036
and loses on 4 of 11 targets. Their own conclusion: linear decodability of physicochemical
properties "reflects the graph architecture and atom featurization rather than learned structure."
**Elabid et al., arXiv:2605.06303** does the residualisation step a molecular probing study needs:
raw R² HeavyAtomCount 0.975 → **0.655 after removing the string-length/size confound**, cLogP
0.797 → 0.686, TPSA 0.806 → 0.686. Consistently hard to decode linearly: NumSpiroAtoms,
NumBridgeheadAtoms, HBD, QED.

**Steier, arXiv:2603.03155** (3D atomistic models, not 2D MPNNs) contains the single most
important methodological warning: on a purely compositional target, after *provable* linear
erasure of composition, ridge correctly returns R² ≈ −0.003 while gradient-boosted trees return
**R² = 0.68–0.95**. Any residualisation-based probing study that uses nonlinear probes will fool
itself. **Pinto, arXiv:2506.06443** finds frozen *intermediate*-layer embeddings beat final-layer
by 5.4% on average and up to 28.6% across 5 encoders / 22 tasks — directly relevant to "what gets
destroyed with depth."

### 1.2 Representational similarity (CKA) against fingerprints: essentially one paper

**Welsch, Hirte, Kirchmair, "Deciphering Molecular Embeddings with Centered Kernel Alignment,"
JCIM 2024, 64, 7303–7312** (DOI 10.1021/acs.jcim.4c00837) builds a random-forest-kernel CKA
(standard CKA assumes rotation invariance, which is false for RFs) and uses it to explain why
ECFP radius beyond 2 bonds barely changes RF performance — the internal representations are
near-identical. The specific GNN-vs-fingerprint CKA values could not be extracted (ACS/PMC
blocked automated fetch); do not quote them without the PDF.

**What does not exist:** no published CKA or CCA/SVCCA comparison of a D-MPNN/Chemprop
representation against ECFP and Mordred head-to-head. Praski et al. (below) explicitly note they
did not measure embedding–fingerprint similarity. The widely repeated "MolBERT correlates
strongly with Mordred" claim could not be sourced to a primary result; treat it as
unsubstantiated. **This is a genuine gap and the most publishable part of our own work.**

### 1.3 Intrinsic dimensionality: the weakest-supported sub-question

A circulating claim that "the effective rank of a 256-dimensional chemical fingerprint library is
3.54," attributed to Kirchoff et al., traces to a GitHub issue; the string "effective rank" does
not appear in that paper. **Do not cite 3.54.** What Kirchoff et al. (arXiv:2402.07970) do show is
indirectly relevant: an 8- or 16-dim learned embedding matches or beats 256-dim fingerprints on
similarity search (SmallSA-16 AUROC 0.692 vs ECFC-256 0.494).

MolGraphEval measures dimensional collapse directly: the singular-value spectrum of a random GIN
is smaller "by multiple magnitudes" than any pretrained one, and spectrum magnitude correlates
with downstream performance (rank correlation 0.806 node / 0.927 graph). But uniformity is *not*
a reliable indicator — InfoGraph has the most uniform embeddings and ranks 7th of 9.
**Roth & Liebig, "Rank Collapse Causes Over-Smoothing and Over-Correlation in GNNs," LoG 2023**
give the theory: with depth, node representations collapse into a low-dimensional invariant
subspace determined by the aggregation spectrum alone — independent of learned weights and input
features — for *all* aggregation functions. No paper measures intrinsic dimension (TwoNN/MLE) of
molecular embeddings; the one ID-evaluation paper found (IdEst, arXiv:2606.03338) is vision-only.

### 1.4 "GNNs just rediscover descriptors"

**Burns & Green, "Generalizable, fast, and accurate DeepQSPR with fastprop," J. Cheminform. 17
(2025)** (DOI 10.1186/s13321-025-01013-4; arXiv:2404.02058): ~1,600 Mordred descriptors + a plain
FFNN. Headline comparisons vs Chemprop — QM9 scaffold MAE 0.0060 vs 0.00814; flash point RMSE
13.0 vs 21.2 (p = 0.021); PAH R² 0.97 vs 0.59 (p = 0.0012); ARA accuracy 88.2% vs 82.4%; YSI MAE
25.0 vs 28.9; Pgp AUROC 0.90 vs 0.89. Runtimes 42 s vs 7 m 02 s (YSI), 30 s vs 5 m 44 s (flash).
Two things people misquote: a *linear* model on the same descriptors gets only 0.0095 MAE on QM9,
so the win is **descriptors + nonlinearity, not descriptors alone**; and fastprop loses badly
where atomic features matter (QuantumScents 0.651 vs Chemprop 0.85–0.88) and has no
connectivity-based stereocenter descriptors. fastprop's claim is the weaker, better-supported
"descriptors + DL match GNNs on most real datasets at 10× less compute" — **not** "GNNs
rediscover descriptors." The stronger claim is not tested by that paper.

**Burns, Zalte, Abreu, Sieg, Feldmann, Mathea, Green, "Deep Learning Foundation Models from
Classical Molecular Descriptors" (CheMeleon), arXiv:2506.15792**: a ~10M-parameter Chemprop
D-MPNN pretrained on 1M PubChem molecules regressing onto **1,613 Mordred descriptors**. Polaris
win rate 75% (21 wins) vs MiniMol 71%, RF 68%, fastprop 36%, Chemprop 32%; MoleculeACE 97% (29
wins) vs RF 50%. **What the authors claim it learns** is "rich and highly transferable molecular
representations"; the evidence offered is t-SNE plots and win rates. There is no probing study, no
descriptor-decoding analysis, and no ablation isolating which descriptor families drive transfer.
Since the pretraining target *is* descriptors, "the representation encodes descriptors" is assumed
by construction and never measured. That gap is exactly what rep2/rep3 below address.

Worth noting structurally: Chemprop itself concatenates molecule-level RDKit/descriptastorus
descriptors (`X_d`) to the learned fingerprint before the predictor head
(`chemprop/models/model.py`, `chemprop/featurizers/molecule.py`). The reference MPNN
implementation provides a slot for global descriptors — the field does not believe message passing
alone recovers them.

### 1.5 The skeptical literature on molecular pretraining

**Praski, Adamczyk, Czech, "Benchmarking Pretrained Molecular Embedding Models," arXiv:2508.06199
(2025)** is the most useful skeptical paper: 28 models (24 neural + 4 fingerprint/descriptor) on
25 classification datasets plus 11 external sets with 5-fold scaffold CV, scored with Bayesian
Bradley–Terry and a region of practical equivalence. **Every neural model except CheMeleon fails
to beat ECFP-count.** Mean AUROC: CheMeleon 81.17%, R-MAT 80.91%, MolBERT 80.51%, ChemBERTa
80.01%, **ECFP 79.95%**. CheMeleon > ECFP with posterior probability > 99.4% on the core
benchmark, but on external scaffold-CV the evidence is **inconclusive** (+0.65% to +0.74%, all 89%
HDIs include zero, P(practically equivalent) 66–68%). Pattern: the top performers all used
multitask regression onto Mordred/RDKit descriptors; the worst used generic masked reconstruction
or pure contrastive objectives.

**Deng et al., "A systematic study of key elements underlying molecular property prediction,"
Nature Communications 14, 6395 (2023)** (the published version of "Taking a Respite from
Representation Learning"; note the venue — Nat. Commun., not Nat. Mach. Intell.): 62,820 models
trained. RF on RDKit2D descriptors beats MolBERT and GROVER on BACE, BBBP, ESOL and Lipop
(p < 0.05). Diagnosed causes: GROVER only helps below ~1k data points; representation-learning
models show markedly higher metric variance, negatively correlated with mean performance; simple
descriptors are near-perfectly correlated with some targets (MolLogP vs ESOL ≈ −1).

**Sun, Dai, Yu, "Does GNN Pretraining Help Molecular Representation?" NeurIPS 2022**: with rich
features and a balanced scaffold split (no-pretrain baseline 78.0% AUROC), gains are Node
Prediction +0.122 ± 1.542, ContextPred +0.126 ± 1.234, Motif +0.128 ± 1.310, Contrastive
−0.578 ± 1.684 — **every gain an order of magnitude smaller than its own error bar**. Supervised
pretraining does help (+0.712 to +1.638) but erodes as features get richer and splits get better.
MolGraphEval explicitly rebuts this, arguing the conclusion rests on fixed pretraining
hyperparameters. Both are right about different things: MolGraphEval's own numbers show the gain
concentrates on tiny datasets (BBBP +12.3% vs MUV −0.3%, and BBBP has 2% of MUV's molecules).
**Present this as live, not settled.**

Also: **Tossou et al., JCIM 2024, 64(3), 697–711** (real-world covariate shift drops performance
up to 60% and calibration up to 40%); **van Tilborg, Alenicheva, Grisoni, JCIM 2022, 62,
5938–5951** (MoleculeACE — descriptor-based ML outperforms deep learning around activity cliffs);
and the benchmark-integrity bucket: **Guo et al., arXiv:2406.00873** (scaffold splits overestimate
virtual screening; UMAP-cluster splits are much harder), **Schuh, Daniluk, Sieber, Chem. Sci. 2026,
17(36), 17455–17467** (mean test-to-train nearest-neighbour ECFP Tanimoto: Polaris 0.494, TDC
0.559, MoleculeNet 0.589, DTI 0.779; cross-split contamination up to 4.95%), **Kretschmer et al.,
Nat. Commun. 16, 554 (2025)** (coverage bias).

### 1.6 Depth and information: the paper you want does not exist

**Alon & Yahav, ICLR 2021** give over-squashing (exponentially growing receptive field compressed
into a fixed vector; GCN/GIN more susceptible than GAT/GGNN; molecules especially exposed because
chain topologies create bottlenecks). **Roth & Liebig, LoG 2023** give rank collapse with depth.
The empirical consensus in the MPNN literature is 2–3 message-passing steps (Chemprop's default
depth is 3), with hierarchical designs added to enlarge the receptive field *without* stacking
layers. There is **no dedicated study of what information appears or dies at each message-passing
depth on molecules** — which is exactly what rep4 measures. Information-bottleneck work in
chemistry (GIB and its molecular variants) uses IB as a *training objective*, not as a layer-wise
analysis; do not cite it as evidence about layer content.

### 1.7 Scaffold and substructure

Substructure identity *is* recoverable — the firmest positive result in the survey (MolGraphEval,
§1.1). But **Bemis–Murcko scaffold identity decodability from embeddings appears unstudied.** The
nearest result is the GFlowNet paper's finding that probe R² barely changes under a
scaffold-disjoint split (max drop −0.057), which says the probes aren't exploiting scaffold
leakage, not that scaffold is decodable. One paper does attribute scaffold-split degradation to
memorization with numbers: **Zheng, Guo, Wang, Liu, arXiv:2607.10729 (2026)** introduce a
label-free structural-frontier split and report median error inflation 87.0% over a matched
scaffold control, 82.8% even with a 304k-parameter message-passing network, and a BBB ranking
*inversion* (frontier AUROC 0.409 ± 0.005). Striking but single-paper and very recent.

### 1.8 Probe-capacity methodology

**Hewitt & Liang, "Designing and Interpreting Probes with Control Tasks," EMNLP-IJCNLP 2019**
(<https://aclanthology.org/D19-1275/>) is the load-bearing citation. A *control task* has the same
input and output space as the real task, with outputs sampled independently at random and fixed
per input type, so it can only be learned by memorisation; **selectivity = real-task score −
control-task score**. The standard companions — Pimentel et al., "Information-Theoretic Probing
for Linguistic Structure," ACL 2020; Voita & Titov, "Information-Theoretic Probing with Minimum
Description Length," EMNLP 2020; Belinkov, "Probing Classifiers: Promises, Shortcomings, and
Advances," Computational Linguistics 2022 — argue respectively that probing accuracy under a
sufficiently powerful probe is a property of the *task*, not the representation; that MDL/codelength
is the capacity-robust alternative to accuracy; and that probe results without controls are
uninterpretable. (These four are standard references; only Hewitt & Liang was re-verified
in this session's survey pass, which was truncated before its section 9.)

### 1.9 What we adopted from all of this

1. A **random-weight CheMeleon of identical architecture** as the primary control — not a
   Gaussian-noise control. The GFlowNet paper and MolGraphEval both show random message passing is
   a strong featurizer; the only interesting quantity is the *lift over it*.
2. A **permuted-label control task** and reported selectivity.
3. **Fixed probe capacity** (2048 features, λ = 100) across every representation, plus a
   lower-capacity PCA-64 variant, so "decodable" cannot be a capacity artifact.
4. **Linear probes only.** Steier's erasure result is decisive: nonlinear probes resurrect
   information that was provably removed.
5. **Size as an explicit confound.** We report CKA with size (log n, n, n²) residualised out, and
   a sum-aggregation variant of the same encoder.
6. The **2.3% paired-delta noise floor** from the companion study as the significance bar.

---

## Part 2 — What we measured

### 2.1 Intrinsic dimensionality: the variance spectrum is tiny, the *useful* rank is not

`results_rep1_intrinsic_dim.json`

Pooled over 13,226 molecules, the centered covariance of the 2048-d embedding has
**participation ratio 3.30**, entropy effective rank **8.11**, with the top single direction
carrying **51.8%** of variance and the top 10 carrying **89.8%**. n90 = 11, n95 = 28, n99 = 227.
Per dataset the participation ratio ranges 1.51 (hiv) to 7.03 (qm9). There are **zero dead
dimensions and zero near-constant dimensions** — nothing is unused, the variance is just extremely
anisotropic. On the scale-free (z-scored / correlation) spectrum the picture is milder but still
low: pooled n95 = 48, n99 = 320.

Then the part that matters. Project to rank *k* on a basis fitted on **train molecules only**,
reconstruct to 2048-d, and run the identical ridge head:

| k | mean paired delta over 11 datasets | worst dataset |
|---|---|---|
| 64 | +12.5% | qm9 +37.0% |
| 128 | +5.4% | qm9 +18.4% |
| 256 | +1.0% | bace_reg +4.7% |
| 512 | +0.1% | bace_reg +1.3% |

Smallest rank inside the ±2.3% noise band, per dataset: tox21 16, clintox 32, sider 64, bbbp 128,
freesolv/qm9/bace_cls/hiv 256, esol/lipo/bace_reg **512**.

**So the embedding is NOT usefully low-rank.** 90% of the variance lives in 11 directions, but you
need ~256–512 directions to keep downstream accuracy. The task signal is in the low-variance tail.
Worst-case k = 512 is only a 4.0× memory win, against int4 group-32's measured 7.5× at +0.25%
RMSE. **PCA truncation loses to int4 on both axes.**

### 2.2 Linear probes: the decodable things are decodable from a random encoder too

`results_rep2_probes.json` — 37 RDKit targets, 13,226 molecules, 5 random 80/20 splits, λ = 100.

Functional-group presence is saturated: carboxylic acid, amide, sulfonamide, halogen, basic amine,
nitro, ester, phenol, any-sulfur all probe at **AUROC 0.989–1.000**. The permuted-label control
sits at 0.48–0.55, so the probes are sound. But the **random-weight CheMeleon scores 0.993–1.000
on the same targets** — lift from pretraining is **+0.000 to +0.005**. Functional-group
decodability is an architecture-and-atom-featurization fact, not a learned-chemistry fact. This
independently reproduces the GFlowNet paper's result on a D-MPNN.

Where pretraining does lift things (CheMeleon R² − random-encoder R²):

| target | CheMeleon R² | random enc | lift | ECFP4 | sum-agg CheMeleon |
|---|---|---|---|---|---|
| n_bridgehead | 0.653 | 0.361 | **+0.292** | 0.756 | 0.875 |
| max_ring_size | 0.743 | 0.526 | +0.218 | 0.697 | 0.828 |
| n_valence_e | 0.694 | 0.492 | +0.202 | 1.000 | 1.000 |
| n_rot_bonds | 0.711 | 0.510 | +0.202 | 0.995 | 0.998 |
| graph_diameter | 0.778 | 0.578 | +0.201 | 0.925 | 0.951 |
| heavy_atoms | 0.694 | 0.495 | +0.200 | 1.000 | 1.000 |
| mol_wt | 0.679 | 0.483 | +0.196 | 0.996 | 0.999 |
| tpsa | 0.680 | 0.486 | +0.194 | 0.996 | 0.999 |
| qed | 0.654 | 0.503 | +0.151 | 0.477 | 0.541 |
| frac_csp3 | 0.972 | 0.980 | −0.008 | 0.738 | 0.862 |
| log_bertz_ct | 0.961 | 0.913 | +0.048 | 0.561 | 0.821 |

Two findings here, the second one new:

**(a) Molecular weight is only R² = 0.679 from CheMeleon, versus 0.996 from ECFP4 counts.** This
is the opposite of the published pattern (MolGraphEval, GFlowNet probes, Elabid all report
MW R² ≈ 0.95–1.00). On a 1,127-molecule ESOL-only corpus we also got 0.962; it collapses to 0.679
on the diverse 13k pooled corpus.

**(b) The cause is MeanAggregation, and it is exactly recoverable.** The cached `Z` is a mean over
heavy atoms, so `Z × n_heavy` is the sum-aggregated embedding the same encoder would have produced
— no re-encoding needed. Under sum aggregation: mol_wt 0.679 → **0.999**, heavy_atoms 0.694 →
**1.000**, tpsa 0.680 → **0.999**, n_hetero 0.657 → **1.000**, n_hba 0.646 → 0.995,
n_ring_systems 0.793 → 0.990, graph_diameter 0.778 → 0.951. Intensive properties go the other way
(frac_csp3 0.972 → 0.862, qed 0.654 → 0.541). **The encoder encodes extensive quantities
essentially perfectly; the mean readout divides them out.** Published "MW is perfectly decodable"
results are reports about sum/add-pooled architectures, and they do not transfer to a mean-pooled
D-MPNN.

Capacity controls: 2048 iid Gaussian features give R² −0.07 to −0.27 and AUROC 0.48–0.55 (the
pure-overfitting floor is harmless at n = 10.6k train); the PCA-64 restriction of CheMeleon loses
0.02–0.42 R² depending on target, so the full 2048-d capacity is doing real work (consistent with
rep1).

Genuinely hard for every linear probe, including sum-aggregation and ECFP4: **QED** (0.654 /
0.541 / 0.477) and **Balaban J** (0.676 / 0.677 / 0.358). Both are nonlinear functions of
topology; QED is a product of eight desirability functions. Useful as a stated negative.

### 2.3 Is CheMeleon just re-deriving descriptors? Partly — and ECFP4 is not the right rival

`results_rep3_descriptor_equiv.json`

**Representational similarity**, pooled, 13,226 molecules. Raw CKA is useless here (raw RDKit 2D
collapses to 2 PCA dimensions at 99% variance because BertzCT and MolWt dwarf everything), so
z-scored CKA is primary and we also report CKA with size residualised out:

| pair | CKA_z | CKA size-free | SVCCA-64 |
|---|---|---|---|
| chemeleon ~ rdkit2d | 0.189 | 0.167 | 0.819 |
| chemeleon ~ ecfp4_counts | 0.079 | 0.051 | 0.799 |
| chemeleon ~ randenc | 0.265 | 0.234 | 0.910 |
| **chemeleon_sumagg ~ rdkit2d** | **0.884** | **0.326** | 0.750 |
| chemeleon_sumagg ~ ecfp4_counts | 0.421 | 0.193 | 0.800 |
| ecfp4_counts ~ ecfp4_bits | 0.818 | 0.860 | 0.986 |
| ecfp4_counts ~ rdkit2d | 0.533 | 0.375 | 0.797 |

The sum-aggregated CheMeleon embedding is **highly similar to RDKit 2D descriptors (CKA_z 0.884)**
— but residualising molecular size out drops it to 0.326, so most of that similarity is one shared
size axis. The mean-pooled embedding (what the cache stores and what the deployed model uses) is
*not* descriptor-like by CKA: 0.189 to RDKit 2D, which is lower than its similarity to its own
random-weight initialisation (0.265). SVCCA-64 is uninformative — it sits at 0.55–0.99 for every
pair of high-dimensional representations and does not discriminate; do not use it.

**Downstream head-to-head**, identical ridge protocol, λ chosen once per representation on 2 random
splits of the 8 single-task datasets, then frozen; evaluated on all 10 cached splits (6 random + 4
scaffold); regression aggregated as RMSE / test-target-std per task (QM9 carries 12 tasks in
different units). Positive delta = worse than CheMeleon:

| representation | dims | mean Δ | median Δ | datasets worse / better (±2.3%) |
|---|---|---|---|---|
| chemeleon | 2048 | 0.00% | 0.00% | — |
| randenc (same arch, random weights) | 2048 | **+7.99%** | +3.97% | 6 / 1 |
| ecfp4 counts | 2048 | **+13.59%** | +4.60% | 6 / 0 |
| **rdkit2d** | ~200 | **−2.19%** | **−0.19%** | 4 / 3 |
| chemeleon + ecfp4 | 4096 | −1.27% | −1.37% | 2 / 4 |
| chemeleon + rdkit2d | ~2250 | −5.26% | −1.97% | 1 / 5 |
| ecfp4 + rdkit2d | ~2250 | −3.38% | −1.47% | 3 / 3 |
| **chemeleon + ecfp4 + rdkit2d** | ~4300 | **−5.44%** | −2.53% | 1 / 6 |

Read this carefully:

- **Pretraining is real.** The same architecture with random weights is 8.0% worse on average and
  loses on 6 of 11 datasets (esol +18.8%, freesolv +23.3%, qm9 +22.7%, lipo +9.5%, bace_reg +9.7%).
  CKA(chemeleon, randenc) = 0.265 agrees: pretraining moved the representation a long way.
- **ECFP4 counts are clearly worse** (+13.6% mean, worse on 6 of 11, catastrophic on the small
  regression sets: esol +29%, freesolv +50%, qm9 +54%). Note this contradicts Praski et al.'s
  conclusion that ECFP-count matches most neural embeddings — but Praski used classification-only
  benchmarks with a tuned nonlinear head, while we use AUROC/RMSE with a linear ridge head. On our
  four classification datasets ECFP4 is indeed within noise (bace_cls −0.0%, bbbp +0.6%, sider
  −0.4%); the gap is entirely on regression.
- **~200 RDKit 2D descriptors are statistically indistinguishable from the 2048-d pretrained
  encoder**: mean −2.19%, median −0.19%, 4 datasets worse and 3 better, and the per-dataset split
  noise (delta sd) is 1.3–14.3%. The mean is dominated by one outlier, **QM9 at −40.0%** (496
  molecules, 12 quantum targets — descriptors are far better there). Excluding QM9 the mean is
  +1.6%, i.e. a dead heat.
- **They are complementary, not redundant.** Concatenating all three is better than any single
  representation on 6 of 11 datasets and worse on 1 (−5.44% mean; esol −9.2%, lipo −10.5%, qm9
  −24.3%). If CheMeleon were simply re-deriving descriptors, concatenation could not help this
  consistently.

Honest limit: **mordred and descriptastorus are not installed**, so "descriptors" here is RDKit's
own ~200 usable 2-D descriptors, not the 1,613 Mordred features CheMeleon was pretrained on and
fastprop uses. Both the CKA and the head-to-head therefore **understate** how descriptor-like
CheMeleon is. The real fastprop-style comparison would need Mordred.

### 2.4 Depth: the local/global hypothesis is refuted; ring topology is what depth buys

`results_rep4_depth_probes.json` — the pooled corpus re-encoded with the same pretrained weights
at depths 1–6. Depth 6 is **bit-identical** to the cached Z0 (max abs diff 0.00e+00), which
validates the harness.

The hypothesis was that local properties saturate early and global ones keep improving. **That is
wrong.** Mean fraction of the depth-6 probe score already reached at depth 3, and mean gain from
depth 3 to 6:

| group | frac of final at d3 | gain d3 → d6 | n |
|---|---|---|---|
| LOCAL (MW, atom counts, HBD, cLogP, TPSA, ...) | 0.762 | +0.163 | 15 |
| GLOBAL / topological (rings, diameter, Bertz, ...) | 0.791 | +0.154 | 15 |
| functional groups (9 SMARTS flags) | **0.996** | **+0.002** | 9 |

Local and global gain *equally* from depth. The only thing that saturates is functional-group
presence — and it saturates at **depth 1** (AUROC 0.954–1.000 at d1; d6 − d1 ≤ +0.042).

What actually keeps improving from depth 3 to depth 6, ranked: **n_bridgehead +0.318**
(0.335 → 0.653), **max_ring_size +0.287** (0.457 → 0.743), **n_spiro +0.245**, n_valence_e +0.231,
heavy_atoms +0.229, mol_wt +0.226, graph_diameter +0.214. What does not: frac_csp3 −0.005,
fsp3_scaffold +0.009, log_bertz_ct +0.052, aromatic_rings +0.082.

This is a clean, novel, mechanistic complement to the companion study's finding that truncating a
depth-6 checkpoint to depth 3 costs +15–30% RMSE and is genuine information loss. **Depth 4–6 buys
ring-system topology (bridgeheads, spiro centres, maximum ring size, graph diameter) and
molecular size — not chemistry.** And the reason size appears in that list is 2.2: under mean
aggregation, extensive quantities can only be recovered through long-range propagation, so
*mean pooling turns size into a global property*. These two results explain each other.

### 2.5 Scaffold is highly decodable — but mostly for architectural reasons

`results_rep5_scaffold.json` — 5,855 distinct Bemis–Murcko scaffolds in the pooled corpus
(76.2% singletons); 63 scaffolds with ≥15 members, covering 2,871 molecules (21.7%). Molecule-wise
80/20 split, 5 repeats. Majority-class baseline 0.3857; uniform chance 1/63 = 0.0159.

| representation | top-1 | permuted-label control | 1-NN scaffold agreement | 1-NN control |
|---|---|---|---|---|
| chemeleon | **0.9676** | 0.3310 | **0.9000** | 0.1460 |
| ecfp4 counts | 0.9314 | 0.1976 | 0.8676 | 0.1425 |
| chemeleon sum-agg | 0.9317 | 0.3645 | 0.9000 | 0.1460 |
| randenc | 0.8976 | 0.3714 | 0.8861 | 0.1470 |
| rdkit2d | 0.8631 | 0.3746 | 0.8704 | 0.1554 |
| gauss2048 | 0.1944 | 0.2188 | 0.1526 | 0.1666 |

Scaffold identity is **96.8% linearly decodable** over 63 classes, and 90% recoverable by raw
cosine 1-NN with no fitted probe at all — so it is not a probe-capacity artifact. It is a genuine
fact about the geometry: molecules cluster by scaffold in CheMeleon space. The frozen encoder is,
among other things, a scaffold detector, which is a mechanical partial explanation for why
scaffold splits roughly double RMSE in the companion study — a held-out scaffold is a held-out
region of the embedding, not just a held-out label.

But the **random-weight encoder already reaches 89.8% / 88.6%**. Pretraining adds ~7 points of
top-1 and ~1.4 points of 1-NN. Most scaffold separability is architectural. Depth sweep:
top-1 climbs 0.754 (d1) → 0.915 (d3) → 0.968 (d6), 1-NN 0.861 → 0.873 → 0.900 — consistent with
2.4 (ring topology accumulates with depth).

Note the first result in this table appears to be unpublished: the survey found **no** direct study
of Bemis–Murcko scaffold decodability from molecular GNN embeddings.

### 2.6 The caching payoff: low-rank loses to int4 by ~50×, and sparsity is the real opening

`results_rep6_env_cache_rank.json` — this measures the object the environment cache actually
stores: the per-edge hidden state H, 2048-d, at every message-passing layer. 20,000 sampled edges
per layer, 4 random splits, projection applied through the same `cache_q` hook the quantization
runs use.

**The per-layer rank grows monotonically with depth** (esol; lipo and bace_reg are the same shape):

| layer | participation ratio | entropy eff. rank | n95 | n99 | zeros | nnz/edge | alive dims |
|---|---|---|---|---|---|---|---|
| 0 | 7.32 | 13.6 | 23 | 45 | 94.8% | 107 | 1153 |
| 1 | 7.80 | 20.3 | 56 | 135 | 98.9% | 24 | 852 |
| 2 | 15.40 | 44.1 | 119 | 271 | 98.3% | 35 | 1201 |
| 3 | 24.08 | 76.5 | 222 | 498 | 97.9% | 43 | 1541 |
| 4 | 36.39 | 132.3 | 455 | 948 | 97.4% | 53 | 1850 |
| 5 | **57.91** | **225.1** | **708** | **1232** | 96.8% | 65 | 1892 |

The first layer is nearly rank-20; the last needs 708 directions for 95% of variance. **The cache
gets less compressible the deeper you go** — which is the opposite of the rank-collapse story
(Roth & Liebig) and worth saying out loud: in this 6-layer D-MPNN with a residual `H_0` re-injected
every layer, rank *grows* with depth.

End-to-end, low-rank projection applied after every layer, against the int4 reference run through
the identical hook:

| cache format | B/edge | esol Δ | lipo Δ | bace_reg Δ |
|---|---|---|---|---|
| rank-64 | 256 | +33.0% | +23.7% | +36.1% |
| rank-128 | 512 | +21.9% | +17.4% | +20.0% |
| rank-256 | 1024 | **+12.2%** | **+15.5%** | **+10.9%** |
| rank-512 | 2048 | +3.2% | +4.2% | +3.2% |
| rank-1024 | 4096 | +2.7% | +2.0% | −0.5% |
| **int4 group-32** | **1088** | **+0.22%** | **+0.54%** | **+0.09%** |

At matched memory (1024 vs 1088 B/edge) low-rank costs **+10.9% to +15.5% RMSE while int4 costs
+0.09% to +0.54%** — a 20–60× worse accuracy/byte trade. And the low-rank basis here was fitted on
edges from the *whole* dataset including test molecules, so this is an optimistic bound. **Low-rank
projection is a dead end for this cache. Question settled; do not build it.**

The same measurement hands over a better idea. H is the output of a ReLU and is **94.6–99.0%
zeros**, with a mean of 24–112 non-zeros per edge out of 2048 and 741–1892 dimensions ever non-zero
at a given layer. A bitmask-plus-int4 layout costs an estimated **144–330 B/edge** (esol) against int4
group-32's 1088 — a further **3.3–7.6×** on top of the quantization win, and the zero pattern is
stored *exactly*, so the only error is in the quantized non-zeros. That is an estimate of a layout,
not a benchmarked kernel: it ignores alignment, padding, and the gather/scatter cost that routinely
eats sparse-format wins on real hardware. It is the obvious next experiment for the caching track.

---

## Part 3 — What this means for the caching work

1. **The low-rank-storage question is answered, negatively, twice.** The molecule embedding needs
   rank 256–512 of 2048 to stay inside the noise band (4.0× worst case), and the per-edge
   environment cache needs rank ≥512 (a memory *loss*). int4 group-32 beats both by an order of
   magnitude in accuracy per byte. This is a clean negative result that closes a line of work
   cheaply, and it has a crisp explanation: 90% of the variance is in 11 directions but the task
   signal lives in the low-variance tail, so any variance-optimal basis throws away exactly the
   wrong thing.
2. **Quantization and sparsity are the orthogonal axes worth pursuing, not rank.** 95–99% of the
   environment cache is exact ReLU zeros, and only 852–1892 of the 2048 dimensions are ever
   non-zero at a given layer. A bitmask+int4 layout is the next thing to prototype and benchmark;
   the layout estimate says 3.3–7.6x beyond int4, but only a real kernel will say whether the gather cost
   eats it.
3. **Cache-size budgeting should be per-layer, not uniform.** Layer 0 needs 23 directions for 95% of its variance; layer 5 needs 708. A uniform cache format over-provisions the early layers and
   under-provisions the late ones. Mixed per-layer precision (early layers cheaper) is a concrete,
   untested design.
4. **The depth-truncation result now has a mechanism.** Truncating depth 6 → 3 costs +15–30% RMSE
   because depths 4–6 are where bridgehead/spiro/max-ring-size/diameter information and (because
   of mean pooling) molecular size get assembled. Functional-group information is already complete
   at depth 1, so nothing chemical is lost — the loss is topological. That makes the companion
   finding publishable rather than just measured.
5. **The scaffold-split penalty is partly an encoder property, not only a label-distribution
   property.** The frozen encoder decodes scaffold at 96.8% top-1 and clusters by scaffold tightly
   enough for cosine 1-NN to recover it 90% of the time. A held-out scaffold is out-of-distribution
   in the embedding, which is why a frozen-encoder-plus-ridge pipeline degrades so sharply.
6. **The int4 result is safe from a rank confound.** One worry would have been that int4 only looks
   cheap because the embedding is effectively low-rank and quantization noise falls in the null
   space. The spectrum says the opposite — there is no null space (zero dead dims, 1354 eigenvalues
   above 1e-6 of the leading one) and the useful rank is 256–512. int4's +0.25% is a real
   robustness property of the representation, not an artifact of degeneracy.

---

## Part 4 — Which findings are strong enough to build a post around

**Tier 1 — strong, numerically decisive, and (as far as the survey found) novel.**

1. **"Mean pooling is why your GNN can't count atoms."** mol_wt R² 0.679 mean-pooled → 0.999
   sum-pooled, heavy_atoms 0.694 → 1.000, TPSA 0.680 → 0.999, with the sum embedding reconstructed
   exactly as `Z × n_heavy` so there is no re-encoding and no new confound. It reconciles our
   numbers with the published MW ≈ 1.00 results (all sum/add-pooled), and it feeds directly into
   the depth result. Cheap, surprising, and checkable by any reader in ten lines.
2. **"Low-rank is not the compression you want: 11 directions hold 90% of the variance and none of
   the signal."** PR 3.30 / n90 = 11, versus rank 256–512 needed downstream; at matched bytes
   low-rank costs +10.9–15.5% against int4's +0.09–0.54% on the per-edge cache. Clean negative,
   directly actionable, and ties the probing work to the quantization work.
3. **"Depth 4–6 buys ring topology, not chemistry."** Functional groups saturate at depth 1
   (frac-of-final 0.996); bridgeheads +0.318 and max ring size +0.287 from depth 3 → 6; local and
   global targets gain equally (+0.163 vs +0.154), refuting the obvious hypothesis. This is the
   mechanism behind the companion study's +15–30% truncation cost, and the survey found no existing
   depth-wise molecular probing study.

**Tier 2 — solid, worth a post, but needs the caveats in the headline.**

4. **"CheMeleon versus 200 RDKit descriptors is a dead heat — and they're complementary."**
   Mean −2.19%, median −0.19%, 4 worse / 3 better, concatenation better on 6 of 11. The
   complementarity is the real finding (if CheMeleon were re-deriving descriptors, concatenation
   could not help). Caveats that must be stated: RDKit 2D, not the 1,613 Mordred features CheMeleon
   was actually pretrained on, so the test understates descriptor-equivalence; the mean is
   dominated by QM9 (−40%, n = 496); and a linear ridge head is not a trained Chemprop.
5. **"Scaffold is 97% decodable — and 90% of that is the architecture."** Strong numbers with two
   independent metrics and a probe-free 1-NN check, plus the honest deflation that a random-weight
   encoder reaches 89.8%. Pairs naturally with the scaffold-split-penalty result from the
   companion study.
6. **"What the random encoder already knows."** Functional-group AUROC 0.993–1.000 from random
   weights (lift ≤ +0.005), but +8.0% downstream penalty and CKA 0.265 — so pretraining matters
   for tasks and not for functional-group probes. This replicates the GFlowNet-probes result on a
   D-MPNN, which is useful but is a replication, not a discovery.

**Tier 3 — too weak, too confounded, or not finished. Do not build a post on these.**

7. **CKA numbers taken at face value.** CKA_z(sum-agg CheMeleon, RDKit 2D) = 0.884 looks like a
   headline until size residualisation drops it to 0.326. Raw CKA is worse than useless here
   (RDKit 2D collapses to 2 PCA dims). SVCCA-64 sits at 0.55–0.99 for every pair and discriminates
   nothing. Report CKA as a supporting diagnostic with the size-free column, never as a headline.
8. **"ECFP4 is much worse than CheMeleon" (+13.6%).** Real in our setup but contradicts Praski
   et al.'s 28-model Bayesian benchmark, and the contradiction is explainable by our linear head
   and our regression-heavy dataset mix (on our four classification sets ECFP4 is within noise).
   Would need a nonlinear head to be worth asserting.
9. **The sparse-cache byte counts (144–330 B/edge).** A layout estimate with no kernel behind it.
   Promising enough to run next; not publishable until benchmarked.
10. **Per-dataset spectra.** Each dataset has ≤2,200 molecules, so at most min(n−1, 2048)
    eigenvalues are non-zero by construction and the tail is sample-limited. Only the pooled
    spectrum (n = 13,226) should be quoted.
11. **Anything claiming information is *absent*.** Every probe here is linear by design (Steier's
    erasure result makes nonlinear probes untrustworthy for this purpose). Low R² means "not
    linearly decodable at this capacity," never "not encoded." The two honest negatives — QED and
    Balaban J, hard for every representation including ECFP4 and sum-aggregation — are the only
    absence-flavoured claims we can make, and they are about the targets more than the encoder.

---

## Caveats that apply to everything above

- Linear probes only. "Not decodable" always means "not linearly decodable at 2048 features with
  λ = 100."
- The probe corpus pools 11 datasets with very different chemistry, which inflates target variance
  and therefore R² relative to a within-dataset probe.
- The `rdkit2d` column in rep2 is **circular** for most targets — MolWt, TPSA, MolLogP, MolMR,
  LabuteASA, NumHDonors, RingCount and FractionCSP3 *are* columns of the RDKit descriptor vector,
  so that probe reads the answer off its own input. It is a sanity check, not evidence. The honest
  descriptor comparison is rep3's downstream head-to-head.
- A ridge head on a frozen representation is not a trained Chemprop. These results say which
  representation is more linearly useful, not which model wins end to end.
- mordred and descriptastorus are not installed; "descriptors" means RDKit's native ~200 usable
  2-D descriptors throughout.
- Depth truncation applies the pretrained `W_o` to an off-distribution hidden state at d < 6. The
  probes refit at each depth, which removes the mismatch for the probe but not for `W_o` inside
  the encoder.
- rep6's low-rank basis is fitted with test molecules included — deliberately optimistic, since an
  optimistic bound is sufficient to rule the method out.
- The survey's own sections 8 (partially) and 9 were lost to output truncation; the four probing
  methodology citations other than Hewitt & Liang are standard references that were not
  re-verified in this session.
