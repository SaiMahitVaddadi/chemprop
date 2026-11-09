# What else is worth writing about Chemprop — opportunity brief

**Purpose.** Candidate posts beyond the drafted eight. Each is scored on four axes: is the central
claim *true*, is it *actionable*, is it *novel*, and is it *testable on a laptop with ≤2,200-molecule
datasets*. The **Pruned** section is as load-bearing as the ranked one — eight leads that went into
this brief did not survive checking, including two I had half-drafted before the evidence
contradicted them, and one where the honest answer is the exact reverse of the common belief.

**Scope.** Excludes the drafted eight posts' territory: caching, quantization, scatter kernels,
depth, the Hashimoto operator, RWSE, non-backtracking walks.

**Sources, kept separate on purpose.**

- **Code read** (Chemprop `41057042`, v2.3.1 line): `chemprop/uncertainty/{estimator,calibrator,evaluator}.py`,
  `chemprop/nn/{metrics,predictors,agg}.py`, `chemprop/data/{splitting,datasets,dataloader}.py`,
  `chemprop/cli/{train,predict,hpopt,common}.py`, `chemprop/featurizers/molgraph/reaction.py`,
  `chemprop/models/mol_atom_bond.py`; `astartes/{main.py, samplers/extrapolation/scaffold.py}` and
  `deepchem/splits/splitters.py` upstream; `origin/master` (v1) for `args.py`, `data/scaffold.py`,
  `scripts/find_similar_mols.py`.
- **Issue tracker**: all 687 issues + 682 PRs (state=all), 52 Discussions, 34 releases.
- **Literature**: five independent sweeps — uncertainty quantification and the evidential critique;
  benchmarking/evaluation validity; descriptors vs learned representations and pretraining
  skepticism; applicability domain and OOD. All citations below were verified against a fetched
  primary source; items that could not be verified are flagged inline.

**Premise corrections, up front.** Three things assumed when this brief was commissioned are wrong
and were verified by registry inspection plus `git log --all -S` across all branches:

- **Chemprop has 11 registered uncertainty estimators, not ~12** — `none`, `mve`, `ensemble`,
  `classification`, `evidential-total`, `evidential-epistemic`, `evidential-aleatoric`, `dropout`,
  `classification-dirichlet`, `multiclass-dirichlet`, `quantile-regression`. `spectra-roundrobin` is
  commented out.
- **There is no Venn-ABERS calibrator and no CRPS evaluator, in v1 or v2, ever** — zero hits for
  `venn`, `abers`, `crps` across all branches. The calibrators are `zscaling`, `zelikman-interval`,
  `mve-weighting`, `conformal-regression`, `platt`, `isotonic`, plus `conformal-multilabel` /
  `-multiclass` / `-adaptive` and `isotonic-multiclass`. v1 additionally had `tscaling`, dropped in v2.
- **"Nearest-neighbour distance to the training set beats learned UQ" is not established, and the
  stronger literature goes the other way.** See **P2** — this is the sharpest reversal in the brief.

**Measurement provenance.** Numbers marked *(measured here)* were taken locally while writing this
brief, on `studies/caching-quantization/data/`, RDKit 2025.09.3, macOS. `chemprop` is **not
importable** in this environment (`descriptastorus` missing) and `astartes` is not installed, so the
split comparisons use faithful re-implementations read out of upstream source. That is flagged
everywhere it matters and is the single blocking verification step before §1 is published.

---

## Ranked candidates

| # | candidate | central claim | compute | audience | novelty | verdict |
|---|---|---|---|---|---|---|
| **1** | **Chemprop's uncertainty stack never sees a molecule** | it cannot express the method its own benchmark paper recommends | small | practitioner | **high** | **write first** |
| **2** | **Three scaffold splits, 3–17% shared test set** | "scaffold split" is not a protocol, and MoleculeNet never asked for it | **none** (done) | practitioner / methods | high | **write** |
| **3** | **Your `--v-kl` is discarded, and the default isn't the documented one** | three disagreeing defaults for the knob that decides what evidential means | small | practitioner / methods | high | **write** |
| **4** | **Your improvement is smaller than your split noise** | the field's deltas sit inside this study's measured noise floor | **tiny** | practitioner / methods | high | **write** |
| **5** | **The Ray bug that narrowed the search space for two years** | `qrandint(q=1)` degenerates; depth 6 was unreachable | **none** | practitioner / systems | high | **write** |
| **6** | **Is CheMeleon more than a well-conditioned Mordred?** | nobody has measured the embedding against its own pretraining target | medium | methods | **high** | **write** |
| **7** | **Four flags that changed meaning between versions** | docs were wrong about all four; no confirmed regression | **none** | practitioner | medium-high | **write** |
| **8** | **Your multicomponent model is not permutation-invariant** | concatenation makes CSV column order part of the model | small | practitioner | medium-high | **write** |
| **9** | **Reactions are featurized in O(N²)** | CGR builds edges by all-pairs scan | small | systems | high | **gate first** |
| **10** | **The dataset size Chemprop says it can't handle** | 217 B/mol against the 130 GB wontfix | medium | systems | medium | **append to post 1** |
| 11 | Five papers that solved activity cliffs | none independently replicated; the benchmark may be measuring little | medium | methods | medium | **maybe** |
| 12 | Mayr says yes, Jiang says no, both are right | the contradiction resolves on dataset scale | **none** | methods | medium | **maybe** |

**Grouping.** 1 and 3 are both "Chemprop's uncertainty subsystem" and share one compute dependency
(trained UQ heads); they can be a two- or four-part mini-series. 2, 4, 5 and 7 are all "you cannot
compare two numbers" from four different directions (cross-toolkit, statistical, hyperparameter,
cross-version). 6 is the only candidate that is primarily a *methods* contribution.

---

## 1. "Chemprop ships eleven uncertainty estimators, and none of them is the one its own benchmark paper recommends"

**Central claim.** Chemprop's uncertainty subsystem is *architecturally* incapable of expressing the
thing twenty years of applicability-domain research converged on. Every estimator, calibrator and
evaluator is a function of the model's own outputs only; none of them can see a molecule, a
fingerprint, or the training set. On top of that, three of the four regression calibrators cannot
change the rank ordering of uncertainty, the output column has no units and nothing checks them, and
the conformal guarantee is void by construction on the very splits Chemprop recommends. The post is
one architectural observation with four verified consequences.

### The architectural fact

Every signature in `chemprop/uncertainty/calibrator.py` is `fit(self, preds, uncs, targets, mask)`
and `apply(self, uncs)`. `UncertaintyEstimator.__call__(self, dataloader, models, trainer)` likewise
never receives the training distribution. `rdkit.DataStructs` is **not imported anywhere** in
`chemprop/`. A grep across `chemprop/`, `docs/`, `examples/`, `tests/` for `tanimoto`,
`applicability`, `mahalanobis`, `leverage`, `nearest`, `neighbor`, `ood`, `novelty`, `energy_score`,
`max_softmax` returns **zero hits**. The only chemical-similarity computation in the entire package
is inside the `KENNARD_STONE` and `KMEANS` splitters (`astartes` with
`hopts=dict(metric="jaccard"), fingerprint="morgan_fingerprint"`), and the distance matrix is never
exposed at prediction time.

**And v1 had this and v2 dropped it.** `origin/master:scripts/find_similar_mols.py` still exists on
the v1.7.1 line and not on `main`. Docstring: *"Script to find the molecules in the training set
which are most similar to each molecule in the test set."* It offered
`distance_measure: Literal['embedding', 'morgan', 'tanimoto']` — `embedding` via
`model_fingerprint(..., fingerprint_type='last_FFN')` with cosine, `morgan` via Morgan + Jaccard,
`tanimoto` via `Chem.RDKFingerprint` + `DataStructs.FingerprintSimilarity`. Added in `cc60cd62`
(2019-02-14), patched by PRs #30 and #144. It lived in `scripts/`, outside `find_packages()`, so it
was never importable even in v1 — and it was never ported. So the same release line that grew from
two uncertainty methods to twelve lost its only distance-to-training-set tool.

**No one has asked for it back.** `gh api search/issues` on `repo:chemprop/chemprop` returns 0 for
`"applicability domain"`, `"domain of applicability"`, `OOD`, `"distance to training"`,
`mahalanobis`, `"novelty detection"`; 0 code hits for `applicability` / `tanimoto` /
`nearest_neighbor`; 0 discussions. The nearest maintainer scope statement is issue #253 (clustering):
*"Is there something inherent to `chemprop` about clustering that necessitates its addition to the
package? It seems like it's not hard to do this on your own right now using the fingerprint
output"* (@davidegraff); *"a little too far from the core activity of chemprop"* (@cjmcgill). That is
adjacent evidence about clustering, not AD, but it is the stated posture.

### Why that matters: what the literature actually converged on

**The single most under-appreciated consensus in the literature: a learned secondary model on neural
features beats the neural UQ heads.** Four independent studies, four domains, the same structural
answer — Hirschfeld et al. 2020 → **MPNN RF**; Tran et al. (MLST 2020) → CFGP; Wollschläger, Gao,
Charpentier, Ketata & Günnemann, *"Uncertainty Estimation for Molecules: Desiderata and Methods,"*
**ICML 2023** ([arXiv:2306.14916](https://arxiv.org/abs/2306.14916)) → a localized neural kernel (a GP
on GNN features), with *"up to 2.5 and 2.1 times lower errors in terms of AUC-ROC score than dropout
or evidential regression-based methods"*; and Parrondo-Pizarro et al. 2026 → error models combining
distance-to-data with model variance. **Chemprop ships none of it.** The sharpest way to say it:
*Chemprop's own benchmark paper recommends a method Chemprop does not implement.*

The second consensus is **not** "use distance" and **not** "use the model's variance" — it is
**combine both in an error model**, and that has been reached twice independently, 13 years apart:

- Sheridan, *"Using random forest to model the domain applicability of another random forest model,"*
  **JCIM 53(11), 2837–2850 (2013)**, [10.1021/ci400482e](https://doi.org/10.1021/ci400482e). 10
  datasets, 7 DA metrics, RF fitted to unsigned error.
- Parrondo-Pizarro, Lanini & Rodríguez-Pérez, *"Uncertainty Quantification in Molecular Machine
  Learning for Property Predictions under Data Shifts,"* **JCIM (2026)**,
  [10.1021/acs.jcim.5c02381](https://doi.org/10.1021/acs.jcim.5c02381) (Novartis, in-house + public
  ADME): *"data-based UQ metrics (e.g., chemical distance), and model-based UQ metrics (e.g.,
  predicted value and variance) may capture complementary aspects of uncertainty. Their combination
  through error models … yielded higher-quality uncertainty estimates … showing robustness under
  various degrees and types of data shift."*

Chemprop cannot express an error model, because its calibrators take `uncs` and nothing else.

And the field's own verdict on what Chemprop does ship: Lanini et al., *"UNIQUE: A Framework for
Uncertainty Quantification Benchmarking,"* **JCIM 64 (2024)**,
[10.1021/acs.jcim.4c01578](https://doi.org/10.1021/acs.jcim.4c01578) — *"there is no UQ strategy that
consistently provides robust estimates about model's applicability on new samples. Depending on the
dataset, prediction task, and algorithm, accurate uncertainty estimations might be unfeasible to
obtain."* Chemprop's own maintainers agree: @davidegraff, issue **#349**, explaining why most of the
Hirschfeld et al. estimators were never ported to v2 — *"all of those uncertainty metrics have some
type of flaw and none of them are truly trustworthy."* That is the epigraph.

### Consequence A — the calibrators cannot change which molecules you trust *(proof, zero compute)*

Three of the four regression calibrators are **strictly monotone scalar functions of the
per-molecule uncertainty, fitted per task**, so the `spearman` evaluator — the one that measures
whether uncertainty is useful for triage — is **provably invariant** under all of them:

- `ZScalingCalibrator.apply`: `return uncs * self.scalings**2`
- `ZelikmanCalibrator.apply`: `return uncs * self.scalings**2`
- `RegressionConformalCalibrator.apply`: `return uncs + self.qhats`

*(measured here*, 200 synthetic points, variance in [0.01, 3]):

```
raw                  spearman(unc,|err|) = 0.411217
zscaling u*s^2       spearman(unc,|err|) = 0.411217
zelikman u*s^2       spearman(unc,|err|) = 0.411217
conformal u+qhat     spearman(unc,|err|) = 0.411217
```

Identical to six decimals, as monotonicity requires. Only `MVEWeightingCalibrator` can reorder, by
weighting *across ensemble members'* variance heads — and it needs an MVE ensemble, the most
expensive configuration Chemprop offers. Calibration fixes the *size* of your error bars, never
*which* predictions you distrust. `examples/uncertainty.ipynb` lists the calibrators and says nothing
about this.

Corollary with teeth: the conformal calibrator's own docstring concedes that applied to a
non-quantile model, *"both `t_{α/2}(x)` and `t_{1−α/2}(x)` are the same as `ŷ`. Then the interval
would be the same for every data point."* So `--uncertainty-method ensemble --calibration-method
conformal-regression` emits a **constant-width interval for every molecule** — correct marginal
coverage, zero discriminative power — with no warning.

Minor, one sentence: `ZScalingCalibrator.fit` minimises Gaussian NLL over a single scalar with
`scipy.optimize.fmin` (Nelder–Mead) when the exact minimiser is `s² = mean(e²/u)`.

### Consequence B — the uncertainty column has no units, and nothing checks them

- Units differ by estimator. `MVEEstimator`, `EnsembleEstimator`, `DropoutEstimator` and all three
  `Evidential*Estimator`s return a **variance**. `QuantileRegressionEstimator` returns a
  **half-interval** (`half_intervals.append(interval / 2)`, `estimator.py:~871`).
- Consumers assume a unit unconditionally. `NLLRegressionEvaluator.evaluate` computes
  `(2π·uncs).log()/2 + errors²/(2·uncs)` — a variance (`evaluator.py:62-64`).
  `RegressionConformalCalibrator.fit` names its input `half_interval_j` — a half-width
  (`calibrator.py:~276`).
- Nothing validates the pair. `process_predict_args` (`cli/predict.py:210-221`) checks file suffixes
  only; the calibrator is built by registry name (`cli/predict.py:437-441`) with no cross-check
  against `args.uncertainty_method`.

So `--uncertainty-method mve --calibration-method conformal-regression` conformalises a variance as a
half-width; `--uncertainty-method quantile --evaluation-methods nll-regression` evaluates a
half-width as a variance. Both run silently to completion.

**Independently observed, and unresolved.** Discussion **#1320**, *still unanswered*: *"throughout
the codebase, 'uncertainty' generally refers to the variance, and it is converted to the standard
deviation when needed, e.g. in `ZelikmanCalibrator` … the uncertainty should also be converted to a
standard deviation in `RegressionConformalCalibrator`. However, from the code, this conversion does
not appear to occur."* A maintainer asked another maintainer to look; no reply. If that is right,
**every conformal regression interval Chemprop emits is calibrated on variance where it should use
σ** — the same defect reached independently from the source here. Precedent: **#1228** found the
unscale transform applied **twice** to quantile bounds, and that v2 reports the **full**
interquantile range where v1 reported **half** (an undocumented factor of two in what the word means
across versions); **#356** found ENCE missing a square root relative to Scalia et al.

**A still-live ENCE bug** *(found here, unreported)*. `ExpectedNormalizedErrorEvaluator.evaluate`
(`evaluator.py:~160-175`) masks **multiplicatively** — `masked_uncs = uncs * mask` — then sorts and
bins **all n rows**, including masked-out ones carrying `unc = 0, error = 0`. Those sort to the front,
so the low-uncertainty bins fill with zeros, `root_mean_vars → 0`, and `|RMV − RMSE| / RMV → 0/0`. On
sparse multi-task sets (Tox21, SIDER, MUV, ClinTox) ENCE is nan or meaningless. The tell is the
contrast: `NLLRegressionEvaluator` and `SpearmanEvaluator` in the same file mask by boolean indexing
(`preds[:, j][mask_j]`), which is correct. One-line fix; file upstream regardless.

**And a units problem one level up: the main calibration metric is parametric where the calibrator
is not.** `CalibrationAreaEvaluator` builds its bins as `torch.special.erfinv(bins/num_bins)*√2` and
compares `|pred − target|` against `z·σ` (`evaluator.py:98-100`) — i.e. it assumes **Gaussian**
errors. Chemprop's `conformal-regression` calibrator is **distribution-free** by construction. So
evaluating a conformal model with `miscalibration_area` smuggles the Gaussian assumption back in
through the metric, and Rasmussen et al. report real chemical error distributions that violate it.

**Three citation bugs in the docs, free to report.** `platt`, `isotonic` and `isotonic-multiclass`
all cite **Guo et al. 2017** (should be Platt 1999 and Zadrozny & Elkan 2002); `conformal-regression`
cites **Angelopoulos & Bates** — a tutorial — for what is actually **conformalized quantile
regression**, i.e. Romano, Patterson & Candès, **NeurIPS 2019**
([arXiv:1905.03222](https://arxiv.org/abs/1905.03222)); and ENCE is attributed to **Scalia** in the
docs but **Levi** in the code. One credit where it is due: `zscaling` genuinely *is* Levi et al.'s
σ-scaling, confirmed from source, and because it is the `b = 0` special case of Hirschfeld's
two-parameter `σ̂² = a·U(x) + b` it **cannot collapse to constant variance** the way Hirschfeld
observed with `a ≈ 0`.

And the mirror image, issue **#1081**: where boolean-mask indexing *was* used, `targets[mask]`
flattened an `(n, t)` tensor to `(k,)`, so a per-task `R2Score(multioutput="raw_values")` silently
degraded to one pooled scalar broadcast t times — *"test/r2 [0.9262351989746094, 0.9262351989746094,
0.9262351989746094]"*. Maintainer: *"Looks like what we actually want is a `MaskedTensor`, but that
isn't available yet."* So Chemprop has two masking idioms, each correct where the other is wrong, and
the uncertainty evaluators use the wrong one. That framing is the section's spine.

### Consequence C — the conformal guarantee is void on the splits Chemprop recommends

All four conformal variants are **split (inductive)** conformal fitted on a held-out `--cal-path`
with `--conformal-alpha` (default 0.1). There is no Mondrian / group-conditional / stratified
variant; `conformal-adaptive` means the APS score function, not stratification. Nothing stratifies by
similarity, cluster or distance.

Split conformal gives **marginal**, not conditional, coverage, and its validity rests on
exchangeability — Barber, Candès, Ramdas & Tibshirani, *"Conformal prediction beyond
exchangeability,"* **Annals of Statistics 51(2), 816–845 (2023)**,
[arXiv:2202.13415](https://arxiv.org/abs/2202.13415): *"Its validity relies on the assumptions of
exchangeability of the data … However, exchangeability is often violated when predictive models are
deployed in practice."* A scaffold split and a time split are *constructed* to make
train/calibration and test non-exchangeable — that is their entire purpose. So if your calibration
set is drawn i.i.d. with training (the standard setup, and what Chemprop implements), **the nominal
coverage does not transfer to a scaffold test set by construction.** Marginal coverage of 0.90 is
compatible with 0.99 on the near 90% and 0.15 on the far 10%, which is exactly the decision a
practitioner is making when they ask about one novel molecule. The empirical echo: Ovadia et al.,
*"Can You Trust Your Model's Uncertainty?,"* **NeurIPS 2019**,
[arXiv:1906.02530](https://arxiv.org/abs/1906.02530) — *"traditional post-hoc calibration does indeed
fall short."*

The sentence to put in the post verbatim: **Chemprop will hand you a nominally-90% interval for a
molecule arbitrarily far from anything it was trained on, because no part of the uncertainty stack is
conditioned on the input distribution.**

### Consequence D — ensemble variance is biased low by 1/M, and it is the wrong variance

`EnsembleEstimator` computes `torch.var(stacked_preds, dim=0, correction=0)` (`estimator.py:~239`) —
the **biased** variance, dividing by M not M−1. At the smallest legal ensemble (M = 2, enforced at
`estimator.py:~205`) that is **2× too small**; at M = 5, 20% too small. Separately, an ensemble's
prediction spread estimates only the *epistemic* component and contains no aleatoric term, while
`NLLRegressionEvaluator` consumes it as the full predictive variance. Two independent defects, both
over-confident. This also explains a specific user experience: ensemble uncertainties that look far
too small until you recalibrate and then look fine — because `ZScalingCalibrator`'s single scalar
absorbs the 1/M bias *and* the missing aleatoric term into one number, which is Consequence A's
thesis from the other side.

### Consequence E — MVE's failure mode is the opposite of "the variance head collapses"

Worth its own section because the intuitive guess is wrong. The pathology is a *mean*-fitting
failure driven by the variance head, and three independent groups diagnosed it:

- **Seitzer, Tavakoli, Antić & Martius, *"On the Pitfalls of Heteroscedastic Uncertainty Estimation
  with Probabilistic Neural Networks,"* ICLR 2022** ([arXiv:2203.09168](https://arxiv.org/abs/2203.09168)) —
  the Gaussian-NLL mean gradient scales the residual by `1/σ²`, so a region the network has not fitted
  yet gets high predicted σ², its gradient is down-weighted, and **it never gets fitted**. "Rich get
  richer." Formally equivalent to *"modifying the training distribution to one where data points are
  sampled proportionally to 1/σ²"* — hard examples are effectively deleted from the training set.
  Networks converge to *"very poor but stable parameter estimates,"* failing to fit the mean **despite
  sufficient capacity**. Their fix, **β-NLL**, multiplies the per-point NLL by a **detached** `σ^(2β)`;
  β = 1 recovers the MSE mean-gradient exactly, β = 0.5 is empirically best. **This is the paper Heid
  et al. 2024 cite (ref 67) when warning Chemprop users about MVE.**
- **Skafte Detlefsen, Jørgensen & Hauberg, *"Reliable training and estimation of variance networks,"*
  NeurIPS 2019** ([arXiv:1906.03260](https://arxiv.org/abs/1906.03260)) — the root cause is that *"the
  MLE of σ² does not exist based on a single observation, unless the mean μ is known"*, so in
  mini-batch SGD isolated points have no local peers and the variance is not identifiable. Symptoms:
  variance **underestimated even when the mean is accurate**, and variance **does not grow outside the
  data support** — the model is arbitrarily confident exactly where it is worst. Fixes: locality-aware
  mini-batching, and mean-variance *split* training (alternate μ|σ and σ|μ), the principled version of
  "warm up the mean head first."
- **Stirn et al., *"Faithful Heteroscedastic Regression with Neural Networks,"* AISTATS 2023**
  ([arXiv:2212.09184](https://arxiv.org/abs/2212.09184)) — *"mean estimates from existing
  heteroscedastic solutions can be significantly less accurate than those from an equivalent
  homoscedastic model."* **One concrete recommendation for the post: never adopt MVE without checking
  its mean accuracy against a plain MSE model.**

**And Chemprop is in exactly the configuration all three diagnose as pathological.** `MVELoss`
(`nn/metrics.py`) is plain Nix & Weigend 1994 heteroscedastic NLL with **no β-NLL and no detached σ
weighting**; `MveFFN` (`nn/predictors.py:173-187`) applies `var = F.softplus(var)` with **no epsilon
floor and no clamp**; and `chemprop/schedulers.py` has a **learning-rate** warmup only — **no
variance-head warm-up and no two-stage mean/variance training**.

Empirical confirmation from inside Chemprop's own benchmark: Hirschfeld et al. had to **exclude FFN
MVE from the NLL figure and MPNN MVE from the calibrated-NLL figure** for extreme values, found MVE
had *"by far the most variation across random splits,"* and found its most-confident Lipophilicity
predictions had **higher** than average error. So in the only systematic Chemprop UQ benchmark,
MVE-in-Chemprop is the **highest-variance** estimator.

### Four shipped defaults worth a paragraph each *(all verified from source; none reported in the literature)*

- `--uncertainty-method` defaults to **`"none"`** and `--calibration-method` to **None**
  (`cli/predict.py:117,123`). The out-of-the-box path is uncalibrated.
- `--dropout-sampling-size` defaults to **10** MC samples, against the 30–100 typical in the
  literature.
- **The official tutorial uses the validation set as the calibration set.**
  `examples/uncertainty.ipynb`: *"use the validation set from the training as the calibration set as
  an example."* That set was already consumed by early stopping and checkpoint selection, so it is
  **not fresh**, and both σ-scaling and conformal fitted on it are **optimistically calibrated**. Note
  the honest scope: the CLI requires an explicit `--cal-path` with no default, so this is a tutorial
  problem, not a CLI default problem.
- `--uncertainty-dropout-p` 0.1; `--calibration-interval-percentile` 95; `--conformal-alpha` 0.1.

And the direct search result: **no paper reports that Chemprop's shipped UQ defaults are bad.** The
λ split-brain (candidate 3), the missing epsilon floor on the MVE variance head, the 10-sample MC
dropout default and the tutorial's val-as-calibration-set are all source findings, not literature
claims. Present them as such — that is the post's actual contribution.

### The literature the post should quote, because it is worse than the code

Hirschfeld, Swanson, Yang, Barzilay & Coley, *"Uncertainty Quantification Using Neural Networks for
Molecular Property Prediction,"* **JCIM 60(8), 3770–3780 (2020)**,
[10.1021/acs.jcim.0c00502](https://doi.org/10.1021/acs.jcim.0c00502) — the Chemprop group's own UQ
benchmark, on ESOL, FreeSolv, Lipophilicity, QM7, CLogP, random and scaffold splits:

- *"no method was consistently strong across all datasets … Occasionally, a method will even exhibit
  a negative Spearman correlation."*
- *"The MPNN Ensemble **consistently underestimates error**; this tendency was observed in the MPNN
  Bootstrap and MPNN Snapshot Ensembling as well."* (Confirms Consequence D empirically.)
- The vivid one: *"the top 25% of predictions identified by the MPNN MVE on CLogP had an RMSE that
  was 75% less than the RMSE on the entire test set, while **the top 25% of predictions using the
  MPNN MVE on lipo had higher RMSE than the overall test set**."*
- On scaffold splits, *"the first six MPNN methods have miscalibration areas several times higher
  for the scaffold split than for any random split."* (Confirms Consequence C empirically.)
- Final recommendation: *"we recommend the **MPNN RF** as a reasonable first approach"*, with
  *"FP RF representing a strong baseline"* — i.e. a plain fingerprint + random forest matched the
  graph network's UQ. **Neither MPNN RF nor FP RF is among Chemprop v2's twelve estimators.**

Also: Scalia, Grambow, Pernici, Li & Green, *"Evaluating Scalable Uncertainty Estimation Methods for
Deep Learning-Based Molecular Property Prediction,"* **JCIM 60(6), 2697–2717 (2020)**,
[10.1021/acs.jcim.9b00975](https://doi.org/10.1021/acs.jcim.9b00975) — *"Deep Ensembles and
bootstrapping consistently outperform MC-dropout … highlights the challenge posed by out-of-domain
uncertainty."* (So MC dropout, which Chemprop ships, is the one the literature ranks lowest. Note
also Chemprop's `DropoutEstimator` lets you set `--uncertainty-dropout-p` on a model **trained
without dropout**, which is not MC dropout in Gal & Ghahramani's sense at all; the docstring admits
the training-time probability is *"prefered but not possible if the model was not trained with
dropout."*)

### Experiment

**One run unblocks everything.** Train small MVE / evidential / 5-member-ensemble / MC-dropout heads
on frozen CheMeleon embeddings for the study's 5 regression datasets × the existing 6 random + 4
scaffold splits. This closes post 5's standing gap ("no trained MLP head"). Then:

1. Report **spearman(unc, |err|)** per estimator per split type. Prediction worth pre-registering:
   all estimators land near 0.3–0.4 and no calibrator moves it (Consequence A).
2. The units audit table, plus two mismatched invocations emitting confidently wrong numbers
   (Consequence B). Settle Discussion #1320 numerically: does conformal coverage come out at 1 − α
   when fed a variance vs a σ?
3. **The headline figure.** Bin test molecules by nearest-neighbour ECFP4 Tanimoto distance to train
   in 0.1-wide shells and plot **realized conformal coverage per shell** against the nominal 1 − α,
   next to the single marginal number. Then implement **distance-Mondrian conformal** — stratify the
   calibration set by the same shells, take per-shell quantiles — and report restored coverage and
   the width you pay. Compare against honest abstention via a risk-coverage curve.
4. Nearly free add-on: ensemble NLL at `correction=0` vs `correction=1` vs the proper decomposition
   (mean of per-member variances + variance of per-member means).

Budget: tens of minutes for the heads (ridge/small MLP over 2048-d frozen embeddings), minutes for
the fingerprints and shells. Laptop, no GPU.

**Pre-registered thresholds, so the result is falsifiable.** Liu, Wang, Glover, Feasel & Wallqvist,
*"Is an Applicability Domain Needed for QSAR Models Based on Deep Neural Networks?,"* **JCIM 59(1),
117–126 (2019)**, [10.1021/acs.jcim.8b00348](https://doi.org/10.1021/acs.jcim.8b00348), over 21
datasets: *"all three methods show highly reliable predictions for compounds within a Tanimoto
distance of 0.4 to any training sample"* and *"for test compounds with a Tanimoto distance of at
least 0.7 from any training sample, none of the models gave acceptable predictions."*

**Novelty.** The architectural observation and the four code consequences are new. The
distance-conditional conformal coverage figure for molecules was not located in the literature —
Barber et al. prove the gap exists in general; nobody appears to have measured it on molecules. A
Mondrian conformal variant with a *distance* taxonomy is also unlocated; Mondrian exists in
cheminformatics only with a **class-label** taxonomy (Sun et al., **JCIM 57 (2017)**,
[10.1021/acs.jcim.7b00159](https://doi.org/10.1021/acs.jcim.7b00159)). **Re-check "distance-conditional
conformal" and "Mondrian applicability domain" on Google Scholar and OpenReview before publishing** —
the sweep that found nothing was targeted, not exhaustive. And note the honest fallback: the general
fact that monotone recalibration preserves ranking is folklore; what is new is that Chemprop's entire
regression calibrator menu is in that class.

**Either outcome is publishable.** If per-shell coverage collapses, you have the figure every
Chemprop user should see plus a working fix and a credible PR. If it holds, that is a **surprising
negative result** — it would mean MoleculeNet's scaffold splits do not push far enough to break
exchangeability, which is independent confirmation of candidate 2's thesis from a different
measurement.

**Design warning.** Keep the **activity-cliff confound** explicit: a distance-based metric cannot
flag cliff errors by construction, since a cliff is a near-identical pair with very different
activity — inside the domain by every distance metric and badly predicted. Report the fraction of
large errors that are "far from training" vs "near but cliff-like"; that decomposition appears to be
unquantified, and MoleculeACE supplies cliff labels free on bioactivity targets.

---

## 2. "Three implementations of 'scaffold split' can share as little as 3% of their test set"

**Central claim.** "Bemis–Murcko scaffold split" names a *family* of protocols that put almost
disjoint molecules in the test set; Chemprop v2's member of that family is as different from itself
across seeds as it is from DeepChem's; and on the two datasets where the field most relies on it,
MoleculeNet never recommended it in the first place.

All numbers in this section are reproducible with
`uv run scripts/split_audit.py`; raw output in `results/split_audit.txt`.

### (a) Test sets barely overlap *(measured here)*

Jaccard overlap of the 10% test sets:

| dataset | n | astartes seed 0 vs 1 | astartes vs Chemprop v1 `balanced` | astartes vs DeepChem | v1 vs DeepChem |
|---|---|---|---|---|---|
| FreeSolv | 642 | 0.174 | 0.075 | 0.290 | 0.182 |
| ESOL | 1128 | 0.109 | 0.119 | 0.166 | 0.135 |
| BACE | 1513 | 0.034 | 0.031 | 0.086 | 0.041 |
| BBBP | 2039 | **0.012** | 0.028 | 0.025 | 0.076 |
| Lipophilicity | 4200 | 0.066 | 0.049 | 0.054 | 0.063 |
| HIV | 41120 | 0.028 | 0.055 | 0.058 | 0.063 |

A Chemprop v2 "scaffold split" number and a DeepChem-pipeline one share **2.5–29%** of their test
molecules, and on BBBP the figure is **1.2%** between two *seeds of the same implementation*. So
Chemprop v2's scaffold split is **not deterministic** and differs from itself about as much as it
differs from DeepChem. Note the trend worth remarking on: overlap *falls* as the dataset grows and
scaffolds get rarer, so the problem is worse on the datasets people care more about.

### (b) Acyclic molecules are structurally excluded from the test set *(measured here)*

Every acyclic molecule has the *same* Bemis–Murcko scaffold — the empty string — so they form one
cluster, and no splitter breaks a cluster. astartes even warns about it (`NoMatchingScaffold`) and
nothing acts on the warning. The result, on every dataset and every implementation tested:

| dataset | n | acyclic | empty-scaffold cluster | acyclic% train (scaffold) | acyclic% test (scaffold) | acyclic% test (random) |
|---|---|---|---|---|---|---|
| FreeSolv | 642 | 320 (49.8%) | 320 | 62.1–62.4% | **0.0%** | 53.8% |
| ESOL | 1128 | 317 (28.1%) | 317 | 35.1% | **0.0%** | 18.6% |
| BACE | 1513 | 5 (0.3%) | 5 | 0.0–0.4% | 0.0% | 0.0% |
| BBBP | 2039 | 99 (4.9%) | 99 | 6.1% | **0.0%** | 4.9% |
| Lipophilicity | 4200 | 5 (0.1%) | 5 | 0.1% | 0.0% | 0.2% |
| HIV | 41120 | 1582 (3.8%) | 1582 | 4.8% | **0.0%** | 4.1% |

This is a property of the *scaffold definition*, not of any implementation — all three agree, and
the matched random-split column shows the test set is only acyclic-free under the scaffold protocol.
It is also **not merely a size artefact**: on HIV the acyclic cluster is 1,582 molecules against a
4,112-molecule test set, so it would fit comfortably, and it still ends up in train in all three
implementations because the test set fills first from the 14,299 singletons. On FreeSolv the effect
is severe: half the dataset is acyclic, all of it is in train, and the "FreeSolv scaffold split"
number the field quotes is measured on a test set drawn from a 322-molecule cyclic subpopulation.
Train and test differ in ring count *by construction*.

### (c) MoleculeNet never asked for scaffold splits on these datasets

The datum that makes (b) sting. From MoleculeNet's own dataset table (Wu et al., **Chem. Sci. 9,
513–530, 2018**, [10.1039/c7sc02664a](https://doi.org/10.1039/c7sc02664a)), the prescribed splits are
**Scaffold for HIV, BACE and BBBP only**; **Random** for ESOL, FreeSolv, Lipophilicity, SIDER,
ClinTox, Tox21, PCBA, MUV and ToxCast; **Time** only for PDBbind. The near-universal use of scaffold
splits on the regression sets is **community convention, not the benchmark's recommendation** — and
the community adopted it on exactly the two datasets (ESOL, FreeSolv) where the acyclic exclusion is
worst.

### (d) Put every splitter on the distance axis *(measured here)*

Leakage audit using the definitions from Schuh et al. 2026 and Walters 2024 so the numbers are
directly comparable to theirs — ECFP4 (Morgan r=2, 2048 bits), per test molecule against the training
set:

| dataset | split | near (max Tc ≥ 0.9) | test scaffold also in train | mean top-5 Tc | mean max Tc |
|---|---|---|---|---|---|
| ESOL | random 80/10/10 | 0.071 | **0.805** | 0.454 | 0.571 |
| | astartes (cp v2) | 0.054 | 0.000 | 0.359 | 0.471 |
| | chemprop v1 balanced | 0.026 | 0.000 | 0.345 | 0.435 |
| | DeepChem | 0.018 | 0.000 | 0.316 | 0.380 |
| FreeSolv | random | 0.108 | **0.985** | 0.420 | 0.530 |
| | astartes (cp v2) | 0.062 | 0.000 | 0.272 | 0.343 |
| | chemprop v1 balanced | 0.015 | 0.000 | 0.307 | 0.402 |
| | DeepChem | 0.015 | 0.000 | 0.214 | 0.265 |
| BACE | random | 0.099 | 0.645 | 0.717 | 0.795 |
| | astartes (cp v2) | 0.026 | 0.000 | 0.657 | 0.723 |
| | chemprop v1 balanced | 0.053 | 0.000 | 0.637 | 0.710 |
| | DeepChem | 0.013 | 0.000 | 0.503 | 0.567 |
| BBBP | random | 0.088 | 0.603 | 0.458 | 0.583 |
| | astartes (cp v2) | 0.000 | 0.000 | 0.341 | 0.404 |
| | chemprop v1 balanced | 0.005 | 0.000 | 0.386 | 0.468 |
| | DeepChem | 0.000 | 0.000 | 0.352 | 0.421 |
| Lipophilicity | random | 0.074 | 0.526 | 0.529 | 0.648 |
| | astartes (cp v2) | 0.036 | 0.000 | 0.466 | 0.561 |
| | chemprop v1 balanced | 0.019 | 0.000 | 0.468 | 0.558 |
| | DeepChem | 0.050 | 0.000 | 0.446 | 0.538 |
| HIV | random | 0.042 | 0.620 | 0.508 | 0.614 |
| | astartes (cp v2) | 0.023 | 0.000 | 0.457 | 0.542 |
| | chemprop v1 balanced | 0.026 | 0.000 | 0.453 | 0.538 |
| | DeepChem | 0.007 | 0.000 | 0.384 | 0.443 |

Two things fall out. First, **all three scaffold implementations leave substantial near-duplicate
leakage** despite achieving exactly zero scaffold sharing — 1.8–5.4% of ESOL's test set and 1.5–6.2%
of FreeSolv's still have a ≥0.9-Tanimoto training neighbour, because *different scaffolds can be
highly similar*. That is exactly Guo, Hernandez-Hernandez & Ballester's mechanism — *"molecules with
different chemical scaffolds are often similar, which hence introduces unrealistically high
similarities between training molecules and test molecules following a scaffold split"*
([arXiv:2406.00873](https://arxiv.org/abs/2406.00873), ICANN 2024,
[10.1007/978-3-031-72359-9_5](https://doi.org/10.1007/978-3-031-72359-9_5)) — measured here on the
exact datasets and splitters Chemprop users run, and the clean refutation of "scaffold split prevents
leakage."

Second, **the three implementations differ in difficulty by up to 4×, with no consistent ordering** —
and the absence of an ordering is the finding, because it means you cannot even apply a correction
factor. On ESOL astartes leaks 3× DeepChem (0.054 vs 0.018); on BACE v1 leaks 4× DeepChem (0.053 vs
0.013); on Lipophilicity **DeepChem is the leakiest** (0.050 vs astartes' 0.036 and v1's 0.019); on
HIV DeepChem is the strictest by 3×. *(An earlier draft of this brief claimed Chemprop v2's splitter
was systematically the easiest of the three. Lipophilicity and HIV refute that; the claim is
"different and non-monotone," not "easier.")*

### (e) The duplicate-leakage table, with honest attribution *(measured here)*

| dataset | n | unparsable | unique canonical | dup groups | dup rows | conflicting labels | label spread |
|---|---|---|---|---|---|---|---|
| ESOL | 1128 | 0 | 1117 | 11 | 22 | 6 | max **1.03**, median 0.15 log units |
| FreeSolv | 642 | 0 | 642 | 0 | 0 | 0 | — |
| Lipophilicity | 4200 | 0 | 4200 | 0 | 0 | 0 | — |
| BACE | 1513 | 0 | 1513 | 0 | 0 | 0 | — |
| BBBP | 2050 | **11** | 1975 | 60 | 124 | **10** | binary, i.e. contradictory |
| HIV | 41127 | 7 | 41120 | 0 | 0 | 0 | — |

**Attribution matters here.** The BBBP row **replicates Walters 2023** almost exactly — he reports 11
SMILES with uncharged tetravalent nitrogen that RDKit cannot parse, 59 duplicate structures and 10
with contradictory labels ([*"We Need Better Benchmarks for Machine Learning in Drug
Discovery"*](http://practicalcheminformatics.blogspot.com/2023/08/we-need-better-benchmarks-for-machine.html),
Practical Cheminformatics, Aug 2023). Present it as replication, not discovery. The **ESOL** row —
six molecules whose duplicate rows disagree on log solubility by up to **1.03 log units**, the same
order as the RMSE the field reports on ESOL scaffold splits — was not located in that source and may
be new. And the actionable part: Chemprop's default split is `random`, which places a 2-member
duplicate group across train/test with probability 2(0.8)(0.2) = 0.32; Chemprop *ships*
`random_with_repeated_smiles` to prevent exactly this and does not default to it. Note the nuance the
post must carry: the popular "MoleculeNet is full of duplicates" complaint is **true of ESOL and
BBBP and false of FreeSolv, Lipophilicity, BACE and HIV**.

### Mechanism, verified from source

The implementations differ in cluster *ordering*, not direction:

- **DeepChem `ScaffoldSplitter`** (`deepchem/splits/splitters.py`): `MurckoScaffoldSmiles(includeChirality=False)`,
  clusters sorted **descending** by size (ties by first index), greedy fill train → valid → test, **no
  shuffling**. Its `seed` argument is accepted and never referenced — fully deterministic.
- **Chemprop v1 `scaffold_split(balanced=True)`** (`origin/master:chemprop/data/scaffold.py`):
  clusters larger than `val_size/2` or `test_size/2` go to a "big" bucket, the rest to "small", each
  shuffled with `Random(seed)`, concatenated big-then-small, greedy fill **train** first. (With
  `balanced=False` it sorts descending and the seed does nothing.)
- **Chemprop v2** delegates to astartes (`data/splitting.py:110-122`, `sampler="scaffold"`).
  `astartes._extrapolative_sampling` sorts clusters **ascending**, shuffles every cluster of size
  ≤ `min(n_test, n_val)` using `random_state`, then fills **test** first, **val** second, remainder to
  train. On an 80/10/10 ESOL split that threshold is **112** — above almost every scaffold cluster —
  so Chemprop v2's scaffold split is effectively a *random partition of scaffold groups*, not a
  deterministic hardest-case split. That is why reseeding moves ~90% of the test set, and it is the
  mechanism behind the existing study's "scaffold splits roughly double fp32 RMSE, 8–56% CV."
- **A fourth implementation worth naming**, because it inverts the assignment: scikit-fingerprints'
  `scaffold_train_test_split` calls `Chem.RemoveStereochemistry()` first, sorts **ascending**
  (`scaffold_sets.sort(key=len)`) and assigns the **smallest** sets to test; and its `use_csk=True`
  toggle is a *different scaffold definition* (cyclic skeleton, all atoms → carbon), not a different
  split.

**`SCAFFOLD_BALANCED` is not v1's balanced split.** This was flagged as the best unverified lead in
the issue sweep and is now settled: the enum says *balanced*, the sampler argument is `"scaffold"`,
and astartes' size-ordered-plus-shuffle-small procedure is not v1's big/small bucket procedure. One
name, two algorithms. All four implementations also default to discarding chirality, so enantiomer
pairs always co-locate — which matters given Walters' finding that **71% of BACE molecules have at
least one undefined stereocentre**.

### Related tracker material

Issue **#645**: the maintainer enumerated 12 split configurations and **8 of 12 were impossible or
silently lossy**; with `--separate-test-path` the test fraction of `--data-path` is thrown away and
you cannot set test size to 0. The shipped code contains the comment *"flip val and test size if test
size is 0 (**to bypass astartes check**)"* — a workaround against its own splitting dependency. And v2
**deleted cross-validation** (`CV`, `CV_NO_VAL` removed in #994, v2.1) for `--num-replicates`; 44
issues touch splits/replicates/CV, mostly users asking for CV. That substitution matters because
replicates *resample* the split while CV *partitions* it — different variance structure, same name in
users' heads.

### Prior art to cite, not duplicate

- **Walters, *"Some Thoughts on Splitting Chemical Datasets"*** (Practical Cheminformatics,
  2024-11-16) is the nearest adjacent work and belongs in the first paragraph. Biogen solubility,
  5×5-fold CV: random R² ≈ 0.70, scaffold ≈ 0.66, Butina ≈ 0.61, UMAP ≈ 0.56; mean Tanimoto to the 5
  nearest training neighbours 0.42 / 0.39 / 0.36 / 0.31. He flags the definition's brittleness
  (a single-atom-difference pair with Tanimoto 0.66 and *different* scaffolds), prefers Butina over
  UMAP, and argues for time splits. He does **not** compare implementations of the same named split,
  measure test-set overlap between splitters, or surface the empty-scaffold case.
- **Schuh, Daniluk & Sieber, *"Auditing widely used biomolecular benchmarks reveals systematic data
  inconsistencies,"* Chem. Sci. 17(36), 17455–17467 (2026)**,
  [10.1039/d6sc01799a](https://doi.org/10.1039/d6sc01799a) — the strongest quantitative audit. 51
  configurations across Polaris, TDC, MoleculeNet and DTI. Across 28 single-task datasets only
  **59.2% of test rows were audit-clean**: 31.5% shared a Murcko scaffold with training, 7.1% were
  near-train analogues. Noise-injection effects on ROC AUC: label conflicts −0.317, random noise
  −0.329, activity cliffs −0.023, **contamination +0.177 (contamination inflates scores)**. The
  counterfactual leaderboard is the headline: the leader's rank-1 probability was stable at 0.82 on
  clean test sets but **dropped below 0.5 in 15 of 28 datasets** when test sets were enriched for
  near-training analogues, and conflict-enriched panels gave mean **Kendall τ = 0.33**. Zero label
  conflicts in Polaris — third-party validation of its curation claim.
- **Shchur, Mumme, Bojchevski & Günnemann, *"Pitfalls of Graph Neural Network Evaluation,"*
  [arXiv:1811.05868](https://arxiv.org/abs/1811.05868) (2018)** — the canonical general result:
  *"considering different splits of the data leads to dramatically different rankings of models."*
- Alternatives the post should point at rather than relitigate: **Steshin, *Lo-Hi*, NeurIPS 2023**
  ([arXiv:2310.06399](https://arxiv.org/abs/2310.06399)); **Tossou et al., *MOOD*, JCIM 64(3),
  697–711 (2024)** ([10.1021/acs.jcim.3c01774](https://doi.org/10.1021/acs.jcim.3c01774)) — shift as
  the *distribution of test-to-train distances*, with deployment shift costing *"up to 60%"* accuracy
  and *"up to 40%"* calibration; **Joeres, Blumenthal & Kalinina, *DataSAIL*, Nat. Commun. 16, 3337
  (2025)** ([10.1038/s41467-025-58606-8](https://doi.org/10.1038/s41467-025-58606-8)); and the one
  that solves the public-data problem for time splits, **Landrum et al., *SIMPD*, J. Cheminform. 15
  (2023)** ([10.1186/s13321-023-00787-9](https://doi.org/10.1186/s13321-023-00787-9)), 99 public
  ChEMBL-derived datasets with objectives fitted to >130 Novartis lead-optimisation projects.
- **The calibration anchor for how pessimistic clustered splits are**: Sheridan, *"Time-Split
  Cross-Validation as a Method for Estimating the Goodness of Prospective Prediction,"* **JCIM 53(4),
  783–790 (2013)**, [10.1021/ci400084k](https://doi.org/10.1021/ci400084k) — *"time-split selection
  gives an R² that is more like that of true prospective prediction than the R² from random selection
  (too optimistic) or from our analog of leave-class-out selection (too pessimistic)."* So the
  ordering is random > time-split ≈ prospective > leave-class-out, and scaffold split sits somewhere
  in between **with no calibration against prospective truth at all**.

### The confirmed gap

The benchmarking sweep found **no published, controlled study that holds the model and dataset fixed
and varies only which library's `scaffold_split` you call.** Everything published varies the split
*family* (scaffold vs Butina vs UMAP vs time). That is this candidate's claim to novelty, and it is
narrow and defensible.

### Experiment

1. **Verify against the installed packages. Blocking.** Install `astartes` and `deepchem`, re-derive
   the overlap matrix, the acyclic table and the leakage audit from the real code paths. The present
   numbers are faithful re-implementations and should be treated as predictions until then. Minutes.
2. **Price the acyclic exclusion.** On ESOL and FreeSolv, compare scaffold-split error against two
   matched controls: a random split restricted to the cyclic subpopulation, and a scaffold split with
   the acyclic cluster *redistributed*. That separates "scaffold shift is hard" from "the test set has
   no acyclic molecules." Frozen CheMeleon + the study's ridge head (λ=100) over the existing 6+4
   splits, so the existing noise floor applies directly. Minutes.
3. **The standardization-straddling sub-experiment, near-free and the sharpest single result.**
   `MurckoScaffoldSmiles` operates on the SMILES as given, so Walters' documented BBBP β-lactams —
   the same compound as protonated acid, anionic carboxylate and salt — can yield *different*
   scaffold strings and land on **opposite sides** of the split. Measure, per dataset and splitter,
   how many pairs that are identical after `rdMolStandardize` (largest fragment + neutralize +
   canonicalize) straddle the train/test boundary, then re-run after standardizing and report the
   score change. That is leakage of literally the same compound, caused by non-standardized input, in
   the protocol whose entire job is to prevent leakage. Seconds of RDKit per dataset, and not located
   in the literature.
4. **Optional, the expensive version**: scrape 15–25 published "scaffold split" numbers for these
   datasets from papers that do not name the implementation, compute every pairwise delta, and report
   the fraction falling inside the measured implementation-only spread. See candidate 4 — that is
   really the same experiment and should be run once.

**Things to pre-empt:** fix all hyperparameters and say so loudly, or you reproduce Walters'
acknowledged limitation; report per-dataset, never averaged (FreeSolv at 642 molecules dominates any
mean); the deterministic arms are single points, so bootstrap the test set for their intervals and
label the two kinds of interval differently; and you may claim only **non-identifiability** — that
papers did not say which splitter they used — never that they used the wrong one. Non-identifiability
is both true and damning enough.

**Audience** practitioner first, methods second. **Novelty: high.** **Risk:** state the claim as
*"different, with near-disjoint test sets"*, never *"opposite"* — see **P1**.

---

## 3. "Your `--v-kl` is discarded, and the default you get isn't the one in `--help`"

**Central claim.** The hyperparameter the critical evidential-regression literature identifies as
determining whether "epistemic uncertainty" means anything — the regulariser coefficient λ — is, in
Chemprop v2.3.1, (i) **silently thrown away** in the standard invocation, (ii) set to a value
different from its documented CLI default, and (iii) **absent from the hyperparameter search space**.
Three independent failures of one knob.

**Verified chain** for `chemprop train -t regression-evidential --v-kl 0.05` (no `-l`):

1. `cli/train.py:1579` — `if args.loss_function is not None:` … `else: criterion = None`. With no
   `-l`, the block that reads `v_kl=args.v_kl`, `eps=args.eps`, `alpha=args.alpha` (lines 1581-1588)
   **never executes**. Your `--v-kl 0.05` is discarded.
2. `nn/predictors.py:136` — `self.criterion = criterion or Factory.build(self._T_default_criterion,
   task_weights=task_weights, threshold=threshold)`.
3. `nn/predictors.py:194` — `EvidentialFFN._T_default_criterion = EvidentialLoss`.
4. `nn/metrics.py:236` — `def __init__(self, task_weights=1.0, v_kl: float = 0.2, eps: float = 1e-8)`.

You get **λ = 0.2**, while `cli/train.py:463-469` documents `--v-kl` with `default=0.0`. The
documented default, the class default, and the value you actually get are three different numbers in
the same program. The only trace is `cli/train.py:1615-1618`, which logs the criterion *class* and
says nothing about the parameters it dropped.

The identical bug class was **fixed for `--task-weights`** (issue #1164 → PR #1170, v2.1.2;
`task_weights=args.task_weights` is passed separately at `train.py:1605`, outside the `if`) and left
in place for `v_kl`, `eps` and `alpha`. It is open as issue **#1416**; see also **#1233**, where the
maintainer filed it under *"'weird, sorta bugs' that can be worked around relatively easily"* and the
reporter's own suggested fix was *"update the documentation / online help."* The same applies to
`--quantile-loss-alpha` with the quantile loss.

**Not searchable either.** `DEFAULT_SEARCH_SPACE` (`cli/hpopt.py:33-54`, Ray branch 69-92) has no
`v_kl`, and `SEARCH_PARAM_KEYWORDS_MAP["all"] = list(DEFAULT_SEARCH_SPACE.keys())`, so
`--search-parameter-keywords all` does not tune it. **No loss-function hyperparameter is searchable.**

**Both branches, stated precisely, because the asymmetry is the finding.** With `-l evidential` the
CLI value wins and you get **λ = 0** (the documented default) and `--v-kl` *is* honoured. Without
`-l` — the natural invocation, since `-t regression-evidential` already implies the loss — you get
**λ = 0.2** and `--v-kl` is silently dropped. So the flag works only when it is redundant.

**And λ = 0 is, theoretically, the single worst choice.** At λ = 0 the objective is the pure
*overparameterised* NIG negative log likelihood, which has a data-independent minimising path at
**ν → 0** — and ν is the **denominator** of the epistemic uncertainty, `u²_ep = E[σ²]/ν`. So
Chemprop's documented CLI default removes the only thing constraining the quantity it then reports as
epistemic uncertainty. Soleimany et al.'s own λ-ablation includes λ = 0 and shows it is systematically
overconfident.

**There are five λ "defaults" in circulation, spanning two orders of magnitude** *(all verified from
source; not reported anywhere)*:

| where | λ | evidence |
|---|---|---|
| Amini's reference repo (`aamini/evidential-deep-learning`, `hello_world.py`) | **0.01** | `EvidentialRegression(true, pred, coeff=1e-2)` |
| Soleimany's Chemprop fork CLI (`aamini/chemprop@confidence-evidential`) | **1.0** | `--regularizer_coeff` `default=1.0` |
| Soleimany's paper configs and README | **0.2** | `reg_coefs: [0.2, …]` in every `configs/reproducibility/*.json` |
| Chemprop **v1** CLI | **0** | `evidential_regularization: float = 0` |
| Chemprop **v2** Python API | **0.2** | `nn/metrics.py:236` |
| Chemprop **v2** CLI | **0.0** | `cli/train.py:463-469` |

**v1 knew it mattered and said so.** `origin/master:chemprop/args.py:474` docstring: *"The default
value recommended by Soleimany et al. (2021) is 0.2. Optimal value is dataset-dependent; it is
recommended that users test different values to find the best value for their model."* v2's CLI help
repeats that sentence verbatim. Soleimany et al. themselves swept **nine** values
{0, 0.001, 0.01, 0.1, 0.2, 0.4, 0.6, 1.0, 2.0} on FreeSolv / Lipophilicity / Delaney / QM7 and used
**λ = 1.0** for the classification arm.

### What the critique actually says — and three corrections to the obvious framing

**Amini, Schwarting, Soleimany & Rus, *"Deep Evidential Regression,"* NeurIPS 33 (2020)**
([arXiv:1910.02600](https://arxiv.org/abs/1910.02600)) is the method; **Soleimany, Amini, Goldman,
Rus, Bhatia & Coley, ACS Cent. Sci. 7(8), 1356–1367 (2021)**
([10.1021/acscentsci.1c00546](https://doi.org/10.1021/acscentsci.1c00546)) is the chemistry-side
promoter. Both are cited in Chemprop's own docstrings. The critical literature is:

**Meinert, Gawlikowski & Lavin, *"The Unreasonable Effectiveness of Deep Evidential Regression,"*
AAAI 2023, 37(8):9134–9142** ([arXiv:2205.10060](https://arxiv.org/abs/2205.10060)). The objection is
**non-identifiability by overparameterisation of the marginal likelihood**, and three details matter:

- It lives in **(ν, β)**, not (ν, α). Verbatim: *"log L^NIG_i does not depend on ν_i and thus L_i(ω)
  is **minimized, independent of the data**, by sending ν_i → 0 … This ambiguity stays unresolved due
  to a **missing additional constraint for β_i** in the regularizer."* α is if anything
  *over*-constrained and gets stuck near 2.
- The objection is **not** mis-specification. They accept the Bayesian marginal and show that
  integrating out μ and σ² is a **lossy projection**: the four NIG hyperparameters are not recoverable
  from the Student-t. Identifiability from the loss, not a wrong likelihood.
- **The encoded quantity is not the residual.** It is *point-wise convergence speed*, dominated by the
  earliest epochs: *"residuals of the very first epochs predominantly define how small ν_i will
  eventually become since the magnitude of its gradient gradually decreases the better the model
  gets."* Their headline: *"the point-wise convergence speed of the model is used as a proxy for the
  epistemic uncertainty, which arguably stretches the canonical definition … the numerical values of
  the epistemic uncertainty cannot be interpreted as canonical Bayesian or Frequentist uncertainty
  estimations and only relative changes are conclusive."* **One sentence for the post: DER's
  "epistemic uncertainty" is a fossil record of early-training convergence speed, not a posterior
  width.** *(An earlier draft of this brief said "a learned function of the training residual." That
  is close but wrong, and the correct version is more interesting.)*

Two further findings worth using: the **aleatoric** estimate is also quantitatively wrong (synthetic
cubic with true σ = 3 gives *"a peaking structure with a peak height of roughly 0.7"* — off by ~4×
and the wrong shape); and re-running Amini's own released code on NYU Depth v2 shows *"the estimation
of the epistemic uncertainty in the prior art … is nearly equivalent to fitting the width of the
t-distribution … **missing the main disentanglement goal**."* And the title is not ironic in the way
it reads: the "effectiveness" account is **optimisation stability** — *"Defining the total evidence by
two, rather than a single parameter, keeps the optimizer sufficiently busy … thus reducing the
dependency to the random initialization"* — i.e. Amini's four-parameter head is accidentally a
regularised version of a degenerate objective. **The honest, awkward core: their own principled fix
loses to the thing it fixes, and they concede *"we cannot fundamentally fix the underlying issue and
deliver what was promised."***

**The formal results.** Bengs, Hüllermeier & Waegeman, *"Pitfalls of Epistemic Uncertainty
Quantification through Loss Minimisation,"* **NeurIPS 2022**
([arXiv:2203.06102](https://arxiv.org/abs/2203.06102)) is a **squeeze, not a blanket impossibility**:
Thm 1 — for any inner loss convex in the prediction, the empirical minimiser is Dirac at every sample
size, so **no epistemic uncertainty is ever reported**; Thm 2 — KL-regularised with λ above a
threshold never converges to the truth; Thm 3 — below a threshold, Dirac again. Their *"On
Second-Order Scoring Rules for Epistemic Uncertainty Quantification,"* **ICML 2023, PMLR
202:2078–2091** ([arXiv:2301.12736](https://arxiv.org/abs/2301.12736)) contains the clean
DER-specific result: **Proposition 5.3** — *"The deep evidential regression loss function L^DER_2 …
fulfils the conditions in Theorem 5.2 (ii)"*, i.e. **Amini's exact loss is formally proven not to
induce a proper second-order scoring rule.** ⚠️ **Do not upgrade this to a blanket impossibility** —
the authors insist: *"Although our results do not completely exclude existence, they do so for a
large class of second-order loss functions."* Their diagnosis is the cleanest conceptual statement in
the literature: a second-order prediction is scored against a zero-order observation, *"one would
rather need observations of realizations of the first-order distribution … However, such data cannot
exist even in principle … learning on the second-order (epistemic) level is **necessarily Bayesian
and requires a prior**."*

**Two independent replications — this is the strong ground.**

- **Jürgens, Meinert, Bengs, Hüllermeier & Waegeman, *"Is Epistemic Uncertainty Faithfully
  Represented by Evidential Deep Learning Methods?"* ICML 2024, PMLR 235:22624–22642**
  ([arXiv:2402.09056](https://arxiv.org/abs/2402.09056)). **Thm 3.2:** for Dirichlet and NIG families
  the joint predictive map is **not injective**, so the risk minimiser is not unique and *"a wide
  spectrum of [epistemic uncertainty] values is covered."* Meinert's non-identifiability generalised
  into a theorem covering both DER and Sensoy-style EDL. The regulariser is reframed as an
  **"uncertainty budget"**: *"the estimated epistemic uncertainty for all training data points is
  summed up, and this sum is fixed to a certain level determined by λ."* Conclusion: *"**The resulting
  measures of epistemic uncertainty cannot be interpreted in a quantitative manner.**"* And the
  empirical corroboration this brief's experiment should port to molecules: *"For a different run of
  the optimization algorithm **on the same training data**, we observed that the results may look very
  different"*, with *"Using different orders of magnitude for λ does not seem to significantly change
  that pattern."*
- **Shen, Ryu, Ghosh, Bu, Sattigeri, Das & Wornell, *"Are Uncertainty Quantification Capabilities of
  Evidential Deep Learning a Mirage?"* NeurIPS 2024**
  ([arXiv:2402.06160](https://arxiv.org/abs/2402.06160)) — independent group, independent mechanism,
  same verdict: learned distributional uncertainty *"does not vanish with an increasing number of
  observed samples"*, and *"EDL methods can be better interpreted as an EBM-based OOD detector."*
  Headline: *"even when EDL methods are empirically effective on downstream tasks, this occurs
  **despite** their poor uncertainty quantification capabilities."* Directly on λ: UQ is *"sensitive to
  hyperparameter λ, while generally benefit from small λ"*, **and the aleatoric estimate also varies
  with λ — which it should not, since data noise is a property of the data.**

The two converge on the same positive account — Jürgens hypothesise the OOD score *"could be
interpreted as density estimates in feature space"*, Shen arrive at *"EBM-based OOD detector"* — which
is the most defensible explanation of why DER works at all, and implies you could get the same thing
more cheaply and honestly with an explicit feature-space density. Earlier and independent:
**Oh & Shin, *"Improving Evidential Deep Learning via Multi-task Learning,"* AAAI 2022**
([arXiv:2112.09368](https://arxiv.org/abs/2112.09368)) found the same ν → 0 degeneracy a year before
Meinert — *"the model can circumvent to achieve lower NLL loss by increasing the predictive
uncertainty instead of achieving a higher predictive accuracy"*, with Thm 1 proving
`lim_{ν→0+} ∂L^NLL/∂γ = 0` — and their benchmarks are **drug-target affinity (Davis/KIBA, OOD on
BindingDB)**, i.e. molecular. **No rebuttal from Amini et al. exists**; the citation graph of
arXiv:2205.10060 contains no reply, comment or defence.

**On molecular data specifically**, three distinct negative results: Liu, Mo & Cheng, *"Uncertainty
Qualification for Deep Learning-Based Elementary Reaction Property Prediction,"* **JCIM 2024**
([10.1021/acs.jcim.4c01358](https://doi.org/10.1021/acs.jcim.4c01358)) — *"The deep ensemble model
outperforms others in accuracy and shows the highest reliability in estimating prediction uncertainty
across all elementary reaction property data sets"*, against MC-dropout and evidential, the cleanest
"evidential loses on chemistry data" result found; Wollschläger et al. ICML 2023 (above), ⚠️ on
molecular force fields rather than QSAR; and Garain et al., ChemRxiv 2025
([10.26434/chemrxiv-2025-fx019-v2](https://doi.org/10.26434/chemrxiv-2025-fx019-v2), ⚠️ preprint) —
*"DER produced uncertainty estimates where data noise and model uncertainty were not cleanly
separated"* on QM9/WS22, independently corroborating Meinert's disentanglement failure on molecules.
For honesty, two **contrary** results exist, both in materials rather than molecules: Gruich et al.,
**MLST 4:025019 (2023)** ([10.1088/2632-2153/accace](https://doi.org/10.1088/2632-2153/accace)), where
evidential wins on OC20 but *"recalibration of model uncertainties is shown to be essential"*; and
their 2026 follow-up (arXiv:2607.11091, abstract only).

**The fair verdict to write:** evidential is *a cheap heuristic with no replicated quality advantage
over an ensemble, whose aleatoric/epistemic split should not be trusted, and whose λ needs per-dataset
tuning.* Not a failed replication in the strong sense — a method whose headline claims were never
independently reproduced as *superiority*. Its pitch was never "better than an ensemble"; it was
"≈ ensemble quality at 1/N cost", and Soleimany's own active-learning result concedes the ensemble is
better.

**The NaN story, same code path, worth a section.** `EvidentialFFN.forward` produces
`v = softplus(·)`, `alpha = softplus(·) + 1`, `beta = softplus(·)` with **no floor on any of them**
(`nn/predictors.py:196-205`). The loss takes `0.5·log(π/v)`; `EvidentialTotalEstimator` takes
`(1 + 1/v)·(β/(α − 1))`, i.e. divides by `softplus(raw_alpha)`, which underflows to exactly 0 in fp32
below about −104. Issue **#1265** is this from a user: loss went 1.5 → NaN at epoch 0 and wrote a NaN
checkpoint that could not predict. Maintainer: *"In cases where the predicted `v` is too small (which
most likely happens at the very beginning of model training), it can lead to numerical instability …
I can reproduce this result on any dataset if I set the learning rate very high."* Closed **"for
now"** after the reporter said neither `--grad-clip` nor a lower LR worked. The sting: v1-era PRs
**#370/#371 "Clamp evidential `v` parameter"** exist — the fix was known in v1 and not carried into
v2. `MveFFN` has the same unfloored `softplus` and `MVELoss` the same `(mean − targets)²/(2·var)`.

### Experiment: a seed-replication identifiability audit

**The gap.** Jürgens et al. *prove* the NIG → Student-t map is non-injective so the derived epistemic
uncertainty *"can take almost any value"*, and corroborate it empirically — **on synthetic and vision
data only**. Meinert et al. independently predict the mechanism: ν is set by early-epoch convergence
speed. **Nobody has tested either on molecules**, and nobody has measured what Chemprop's λ = 0 CLI
default does to any of it.

**Design.** 4 datasets already in the tree (FreeSolv 642, ESOL 1128, BACE 1513, BBBP 2039 — exactly
the low-N set both Soleimany and Hirschfeld used) × λ ∈ {0, 0.01, 0.1, 0.2, 1.0} × 10 seeds, training
`regression-evidential` with the **split held fixed** across seeds so the only variation is optimiser
randomness. Repeat under random and scaffold splits. Cache per-molecule `(γ, ν, α, β)` on test, plus
per-molecule training residuals logged at epochs 1, 2, 3, 5, 10 and at convergence. Matched MVE and
5-member ensemble control arms.

**Five measurements, each a falsifiable prediction from a named theorem.**

1. **The identifiability signature.** Across seeds on identical data, compare the coefficient of
   variation of γ against that of ν and β, and the across-seed **Spearman agreement** of per-molecule
   `β/(ν(α−1))`. Theory predicts γ agrees tightly while ν and β are near-arbitrary and the epistemic
   *ranking* is unstable. **The first direct test of Jürgens Thm 3.2 on molecular data.** If γ is
   stable and the epistemic ranking is not, that one scatter plot is the whole post.
2. **The convergence-speed fossil.** Correlate final test-time `1/ν` against the per-molecule
   residual at epochs 1–3 versus at convergence. Meinert predicts the early correlation ≫ the final
   one. A clean one-number test of a named mechanism, never measured on molecules.
3. **The λ split-brain, quantified.** Spearman / ENCE / miscalibration area / NLL at λ = 0 (v2 CLI
   default) vs 0.2 (v2 API default) vs 0.01 (Amini's repo) vs 1.0 (Soleimany's fork) — i.e. measuring
   how much the shipped estimator depends on which door the user walked through. Combined with (1),
   test whether λ damps the seed instability; Jürgens report it does not, on other data.
4. **The unfixability demonstration.** Apply `zscaling`, `zelikman-interval` and
   `conformal-regression` to the cached uncertainties and show the `spearman` column is **bit-identical
   before and after**, then show `mve-weighting` is the only one that moves it. Provable from the code,
   so this is a *demonstration* rather than a discovery — but the composition is the payoff: *the
   seed-to-seed ranking instability from (1) is, by construction, unfixable by the calibrators the
   package ships.*
5. **Conformal coverage × calibration-fold provenance.** Reusing the same checkpoints, measure
   empirical coverage against nominal 1 − α with the calibration fold drawn from **training scaffolds**
   versus **novel scaffolds**, under random and scaffold splits. This is candidate 1's Consequence C,
   and it has never been done for Chemprop/D-MPNN.

**Cost.** Chemprop's uncertainty module is decoupled from training (estimator / calibrator / evaluator
all operate on cached predictions), so **one training run per (dataset, λ, seed) serves all five
measurements.** 4 × 5 × 10 = 200 runs on ≤2,039 molecules; minutes per run on MPS; overnight on a
laptop, and the analysis re-runs from cached tensors in seconds. Also log **how often the run NaNs**
per λ — free, and the practitioner payload.

**Two framing cautions.** Report all four datasets **separately, never averaged** — Hirschfeld's own
warning is *"if one were to benchmark UQ methods on a single dataset, one could draw completely
incorrect conclusions"*, and the general-ML version (Mucsányi et al.) finds *"Nine out of thirteen
tasks exhibit substantially different rankings between CIFAR-10 and ImageNet."* And Rasmussen et al.
show Spearman and NLL *"bear little information in themselves"* without simulated reference values —
so either include their simulated-reference construction, or report Spearman only as an across-seed
*agreement* statistic (measurement 1), where it is self-referential and the objection does not bite.

**Audience** practitioner (pick a λ, or don't use evidential) and methods. **Novelty:** the
Chemprop-specific chain (five defaults, discarded flag, unsearchable) is new and fully verified, and
measurements 1–2 are the first molecular tests of two named published predictions. The statistical
critique itself is **not** new and must be sourced as prior work. One thing **not** to say: there is
**no theorem** that a single forward pass provably cannot separate aleatoric from epistemic
uncertainty — the nearest real thing is Meinert's one-equation-two-unknowns argument, explicitly a
heuristic, and the empirical disentanglement failures hit ensembles, MC dropout, Laplace and SWAG just
as hard.

---

## 4. "Your reported improvement is smaller than your split noise"

**Central claim.** This study already measured the thing the molecular-ML replication literature
keeps asking for and never computes: how large split-to-split variation is relative to the deltas
papers report. Published MoleculeNet improvements on ≤2,200-molecule datasets are, in a large
fraction of cases, smaller than the spread a single fixed model produces by changing only the split.

**The asset.** The study's existing noise floor: fp32 split-to-split **CV 4–12% on random splits**,
**8–56% including scaffold splits**, median paired-delta SE **1.07–2.27%** per dataset, and the
threshold *"a paired delta below ~2.3% is not distinguishable from split noise."* Candidate 2 adds a
second axis: Chemprop v2's own scaffold split changes **89–97% of its test set** on a reseed.

**The confirmed gap.** The benchmarking sweep looked specifically and found **no paper that
quantifies seed-to-seed or split-to-split standard deviation and compares it directly against the
magnitude of published method-to-method deltas on MoleculeNet.** An arXiv abstract search for
`MoleculeNet AND "standard deviation"` returned zero results. Two leads in that direction ("On the
Importance of Statistical Rigor", Nikolentzos) did not resolve to verifiable papers and should not be
cited.

**What exists, and must be positioned against.**

- **Schuh et al. 2026** (candidate 2) does the closest thing — but varies *test-set composition*, not
  the split protocol or the seed. Its counterfactual leaderboard (rank-1 probability 0.82 → below 0.5
  in 15/28 datasets; τ = 0.33) is the number to beat.
- **Tönshoff, Ritzert, Rosenbluth & Grohe, *"Where Did the Gap Go? Reassessing the Long-Range Graph
  Benchmark,"* [arXiv:2309.00367](https://arxiv.org/abs/2309.00367) (LoG 2023)** is the template. The
  reported MPGNN-vs-Graph-Transformer gap on LRGB *"is overestimated due to suboptimal hyperparameter
  choices"* and **"completely vanishes after basic hyperparameter optimization"** — and they also
  found LRGB had a broken feature-normalisation step and a flawed link-prediction metric. *(The LRGB
  original, Dwivedi et al. NeurIPS 2022, was not verified in this sweep; confirm the arXiv ID before
  citing.)*
- **Errica, Podda, Bacciu & Micheli, *"A Fair Comparison of Graph Neural Networks for Graph
  Classification,"* ICLR 2020** ([arXiv:1912.09893](https://arxiv.org/abs/1912.09893)) — >47,000
  experiments under one framework; against structure-agnostic baselines, *"on some datasets,
  structural information has not been exploited yet."*
- **Robinson, Glen & Lee, *"Validating the validation: reanalyzing a large-scale comparison of deep
  learning and machine learning models for bioactivity prediction,"* JCAMD 34, 717–730 (2020)**,
  [10.1007/s10822-019-00274-0](https://doi.org/10.1007/s10822-019-00274-0) — the exact move this post
  makes, on Mayr et al.'s ChEMBL comparison. This is the methodological precedent to cite.
- **Ash et al., *"Practically Significant Method Comparison Protocols for Machine Learning in Small
  Molecule Drug Discovery,"* JCIM (2025)**,
  [10.1021/acs.jcim.5c01609](https://doi.org/10.1021/acs.jcim.5c01609) — the Polaris guideline paper,
  with a repeated-CV simulation and dynamic-range analysis; code at
  `github.com/polaris-hub/polaris-method-comparison`. And **Walters, *"Even More Thoughts on ML Method
  Comparisons"*** (Mar 2025) for the concrete practice: 5×5-fold CV, 8 method combinations, Tukey HSD
  with multiple-comparison-adjusted CIs, colour-coded best / statistically-equivalent / significantly
  worse, as a direct attack on *"the dreaded bold table."*
- **Ebner et al., *"Measuring AI Progress in Drug Discovery: A Reproducible Leaderboard for the Tox21
  Challenge,"* [arXiv:2511.14744](https://arxiv.org/abs/2511.14744) / JCIM 66(18) (2026)**,
  [10.1021/acs.jcim.6c01148](https://doi.org/10.1021/acs.jcim.6c01148) — 2015's DeepTox still tops
  Tox21, most pairwise comparisons among the top five *"inconclusive"*, and the dataset *"was altered
  and labels were imputed or manufactured"* on absorption into MoleculeNet, *"resulting in a loss of
  comparability across studies."*

**Experiment.** Three steps, all cheap.

1. **Collect published numbers.** 15–25 reported ROC-AUC / RMSE values for ESOL, FreeSolv, BACE,
   BBBP, SIDER, ClinTox from papers that say "scaffold split" without naming the implementation
   (MoleculeNet, Chemprop/Yang, GROVER, MolCLR, MoLFormer, Attentive FP, ChemBERTa, GEM, KPGT,
   CheMeleon). Record what each says about splits, seeds and repeats. The *provenance table alone* is
   publishable; expect most entries to report a single number or a mean over 3 seeds.
2. **Measure the three noise sources separately** on the same datasets with one fixed model (frozen
   CheMeleon + ridge, which the study already has): seed noise at fixed split, split noise within one
   implementation (astartes reseeded ×10), and **implementation noise across splitters** (candidate 2's
   arms). That decomposition is what nobody has published.
3. **Report the decision-flip rate**: the fraction of published pairwise deltas falling inside each
   noise band. Prediction, from Walters' 0.05 R² scaffold-vs-Butina gap and Schuh's τ = 0.33: most
   published deltas on BBBP / BACE / ClinTox are inside the band.

**Compute:** ridge over 2048 features × ≤2,200 rows × ~10 splitter arms × 10 seeds × 6 datasets is
seconds to minutes. The expensive optional arm is a Chemprop D-MPNN at default config — ~1–3 min per
CPU fit, ~210 fits, one overnight run, no GPU.

**One framing addition that is underused in this whole literature:** normalize every gap by the
dataset's *own* measurement noise (FreeSolv's ~0.6 kcal/mol experimental uncertainty; Landrum &
Riniker's finding that *~45% of IC50 values for identical molecules differ by >0.3 log* even under
optimal conditions, **JCIM 64, 1560–1567, 2024**,
[10.1021/acs.jcim.4c00049](https://doi.org/10.1021/acs.jcim.4c00049)) as well as by split σ. That
converts "beats SOTA by 0.01 AUC" into "0.2σ", which is the Robinson-et-al move and the single most
honest thing available.

**Audience** practitioner and methods. **Novelty: high** — the gap is confirmed, the asset already
exists, and the implementation-noise axis is new. **Risk:** the post must claim non-identifiability
and effect-size comparison, never that a specific paper is wrong.

---

## 5. "The Ray bug that narrowed Chemprop's search space for two years"

**Central claim.** `chemprop hpopt` searched a narrower and differently-shaped space than its
documentation described, for a reason nobody would guess: Ray Tune short-circuits quantization when
`q == 1`, silently turning `qrandint` into `randint` (whose upper bound is exclusive) and `quniform`
into a float sampler. Chemprop v2 could therefore **never try `depth = 6`** — the v1 default region —
until v2.2.0, and `aggregation_norm` was sampled as a float like `146.0`.

**This is substantially issue #1207**, and the post must credit @KnathanM in the first paragraph. He
sampled every distribution in v1 (Hyperopt) and v2 (Ray Tune) and plotted the PDFs:

- **`depth` lost its maximum.** v1 `hp.quniform(2, 6, q=1)`; v2 `tune.qrandint(2, 6, q=1)`. Ray's docs
  *say* `qrandint` includes the upper bound; it doesn't at `q=1`, because Ray short-circuits at
  `q == 1` ([`sample.py#L552-553`](https://github.com/ray-project/ray/blob/a9f8dc8/python/ray/tune/search/sample.py#L552-L553)),
  degenerating to `randint`. @shihchengli argued the opposite *from Ray's own documentation* and was
  corrected — the best possible illustration of why this was undetectable.
- **`aggregation_norm` stopped being an integer**, same cause. Visible in real user output:
  `aggregation-norm = 146.0` (#1168), `dropout = 0.35000000000000003` (#1081).
- **`quniform` is not uniform.** `y = round(x/q)·q` gives min and max *half* the interior mass.
  @KnathanM: *"in #571 it is claimed that the probability of dropout being 0 in v1 was 1/9, when in
  fact it is 1/16."* A documented v1 search space was misdescribed by the maintainers for years.
- Fixed by PR **#1230**, v2.2.0: *"The hyperparameter search space has been updated to include 6
  message passing steps as an option… This option was included in chemprop v1 and was **accidentally
  excluded** during our reimplementation in v2."*

**What is still true today**, verified here against `cli/hpopt.py:69-92` — this is the new material:

- `depth`: `tune.randint(2, 7)` → **2–6 only**. Not 1, not 7+.
- `ffn_num_layers` and every `*_ffn_num_layers` / `*_constrainer_ffn_num_layers`:
  `tune.randint(1, 3)` → **1 or 2 only**.
- `dropout`: `tune.choice([0.0]*8 + list(arange(0.05, 0.45, 0.05)))` — a deliberate **50% prior mass
  on exactly zero dropout**. Defensible, undocumented.
- **Absent entirely:** weight decay, number of epochs, early-stopping patience, every loss-function
  hyperparameter (candidate 3), and the featurizer mode.
- `aggregation_norm` is in the `all` space unconditionally, so it is sampled even when `aggregation`
  is `mean` or `sum`, where it does nothing — a dimension wasted in roughly two-thirds of trials.
- **hpopt tunes on one split.** Issue #1168: *"the hpopt only uses the first split to find optimal
  hyperparameters (we'd like to change this in the future)."* Against this study's measured 4–12%
  split CV on random splits and 8–56% including scaffold splits, **a single-split objective is
  selecting on noise of the same size as the effects it searches over.** That is the most actionable
  sentence in the candidate, and it connects straight to candidate 4.
- **Ray ignores the limits you set.** Issue **#1402** (OPEN):
  `--raytune-max-concurrent-trials 1` → *"Trial status: 5 RUNNING | 1 PENDING"* → OOM kills.
  Maintainer: *"We are also having this problem (and have been since March… It seems like Ray is
  ignoring the system resource specification."*
- Objective-selection bugs as a set: **#1254** hpopt ignored `--tracking-metric` and selected the
  *worst* model (*"the best config was based on obviously the largest loss of `val/loss`"*);
  **#1287**, maintainer-filed — *"If the tracking metric is `val_loss`, then the monitor mode should
  always be `min`. But right now it looks at the first metric in the metric list"* — i.e. the
  optimisation direction could silently invert with `--metrics R2`; open 11 months, fixed v2.3.0;
  **#912** `init_lr`/`final_lr` computed from the *default* `max_lr` rather than the trial's, so
  `best_config.toml` could emit `init_lr > max_lr`, a configuration no trial ever ran. And **#88**
  (v1): hpopt used the test set.
- **The maintainers want out.** **#1349** (OPEN) is Jackson Burns proposing to remove Ray: *"This has
  caused a huge number of problems… `ray` ignores temp dir settings… crashes on Windows with
  gibberish output files… doesn't play well with AWS… My personal experience is that `ray` is
  difficult to control, i.e. it will use GPUs and CPUs that are not assigned to it."* hpopt/Ray is
  the **largest single issue theme in the repository at 56 issues**, the shipped HPO tutorial notebook
  is currently broken by Ray (#1394, OPEN), and the de facto recommendation is the user-contributed
  Optuna notebook (PR #1353, v2.2.4).

**Experiment.** Essentially none — verification plus synthesis. Do run the three-line check that
`tune.qrandint(2, 6, 1)` never returns 6, rather than quoting it. Optional and more expensive: show
that a knob *outside* the space (weight decay, or epochs) moves the metric more than the in-space
knobs do; a frozen-embedding proxy with a ridge/MLP head can stand in for a real hpopt run.

**Audience** practitioner, and anyone who has quoted a Chemprop hpopt result. **Novelty: high as a
post, medium as a finding** — #1207 is a maintainer issue thread with plots, not a blog post, and the
Ray `q == 1` root cause is unknown outside it.

---

## 6. "Is CheMeleon anything more than a well-conditioned compression of Mordred?"

**Central claim.** CheMeleon is a D-MPNN pretrained to regress 1,613 Mordred descriptors. Nobody —
including its authors — has measured the information relationship between the resulting embedding and
its own pretraining target. If the embedding's advantage lives entirely in the Mordred-predictive
subspace, then descriptor-target pretraining is **conditioning and denoising, not new chemistry** —
and that single sentence reconciles four separate strands of the literature.

**The model.** Burns, Zalte, Abreu, Sieg, Feldmann, Mathea & Green, *"Deep Learning Foundation Models
for Low-Data Regimes from Classical Molecular Descriptors,"* **JCIM 66(17), 10692–10701 (2026)**,
[10.1021/acs.jcim.6c01546](https://doi.org/10.1021/acs.jcim.6c01546) (preprint
[arXiv:2506.15792](https://arxiv.org/abs/2506.15792); weights
[Zenodo 10.5281/zenodo.15426600](https://doi.org/10.5281/zenodo.15426600)). D-MPNN, `d_h = 2048`,
depth 6, mean aggregation, 8.7M message-passing / 12.9M total parameters, pretrained on **1M random
PubChem molecules** to regress **1,613 Mordred descriptors** with **85% of targets randomly masked**
each step. *(Note: the title changed twice between preprint versions; cite the published one. Esther
Heid is not an author.)*

**Reported results** — 58 benchmarks, 28 Polaris/TDC + 30 MoleculeACE, win = best or statistically
indistinguishable under Tukey HSD over 5-fold CV:

| model | Polaris (28) | MoleculeACE (30) |
|---|---|---|
| **CheMeleon** | **75%** (21) | **97%** (29) |
| minimol | 71% (20) | 40% (12) |
| RF (Morgan counts 2048/r2 + RDKit desc.) | 68% (19) | 50% (15) |
| MoLFormer | 39% (11) | 17% (5) |
| fastprop (Mordred + FNN) | 36% (10) | 17% (5) |
| Chemprop (random init, same architecture) | 32% (9) | **0%** |
| MolCLR | 14% (4) | **0%** |

**Three holes the paper leaves open, in its own words.**

1. **No hyperparameter optimization for anybody**: *"we do not perform hyperparameter optimization
   for any model on any benchmark dataset."* This cuts both ways — CheMeleon is also untuned — but it
   is explicitly an out-of-the-box comparison, and Sun et al.'s finding that hyperparameters dominate
   pretraining choice on small data is unaddressed.
2. **No frozen-embedding / linear-probe results.** Evaluation is end-to-end fine-tuning. The only
   frozen evidence is a kNN probe on 20 ToxCast endpoints.
3. **The mechanism claim is an argument, not a measurement.** The paper argues descriptors are
   *"deterministic, low-noise, and computationally inexpensive"* against experimental labels that are
   *"sparse and noisy"* and SSL proxy tasks that *"may not directly align with the physicochemical
   properties relevant to downstream applications"*, and says the model *"learns to construct a
   representation that encodes the dense information of descriptors… within a flexible,
   differentiable neural network architecture."* **It never tests whether the embedding contains
   information beyond Mordred itself.**

Also note the tension the authors themselves flag: *"the apparent difficulty of this benchmark for
all tested models suggests that foundation modeling, while able to improve the absolute performance,
does not reliably discern the presence of activity cliffs"* — alongside the 97%/100% MoleculeACE win
rate, which is a *relative* statistic under Tukey HSD, not evidence that cliffs are solved. And
Uni-Mol, GROVER and MolE are not in the comparison.

**Two independent evaluations exist, and they are more sober.**

- **Ebner et al. 2026** (candidate 4): CheMeleon was submitted externally to the reproducible Tox21
  leaderboard and ranks **4th** by average test AUC — behind DeepTox (2015), the CLAMP encoder, and
  SNN, ahead of Random Forest, with most pairwise comparisons among the top five *"inconclusive."*
  This is the single best sanity check on CheMeleon in the literature.
- **Praski, Adamczyk & Czech, *"Benchmarking Pretrained Molecular Embedding Models For Molecular
  Representation Learning,"* [arXiv:2508.06199](https://arxiv.org/abs/2508.06199)** (preprint, v5
  2026-11-03): 28 models × 25 classification datasets, frozen embeddings, **identical hyperparameter
  grids for every representation**, hierarchical Bayesian Bradley–Terry with a ROPE. *"All neural
  models except CheMeleon failed to outperform the classical ECFP count fingerprint; CheMeleon was
  the only model rated as better. However, further testing on external validation datasets, based on
  scaffold cross-validation and a Bayesian hierarchical correlated t-test, revealed **no definite
  advantage of either embedding method**."* So **CheMeleon's advantage evaporates under scaffold
  cross-validation** — in the most statistically careful comparison available.

**What is already published, so don't re-find it.** The obvious experiment — frozen CheMeleon + a
simple head vs Mordred + a simple head on small datasets — **exists twice**: Praski et al. above
(no CheMeleon in the earlier version, ECFP rather than Mordred baseline, classification only), and
**Ben Hicham, Rittig, Grohe & Mitsos, *"Tabular foundation models for in-context prediction of
molecular properties,"* [arXiv:2604.16123](https://arxiv.org/abs/2604.16123) (2026)**, which ran
frozen CheMeleon vs Mordred vs RDKit2D vs Morgan on **58 small datasets** (28 Polaris + 30
MoleculeACE, ~100–6,000 molecules) with TabPFN-2.6 and TabICL heads: TabPFN+CheMeleon **86.2%**,
TabPFN+Mordred 67.2%, TabPFN+RDKit2D 56.9%, fine-tuned CheMeleon 41.4%, TabPFN+Morgan 22.4%, and
100% on MoleculeACE. Gaps in that paper: no ridge/linear head, and *"we do not perform task-specific
hyperparameter optimization for any of the classical tabular ML models."*

### The experiment that is actually unclaimed

On ~6 datasets ≤2,200 molecules (FreeSolv 642, ESOL 1128, SIDER 1427, ClinTox 1478, BACE 1513, BBBP
2039; optionally MoleculeACE assays for cliff structure), 20 scaffold splits each. The study already
has most of the machinery in `experiments/{rep_lib,rep1_intrinsic_dim,rep2_probes,residual_flow}.py`
— frozen-CheMeleon linear probes, a random-weight-encoder floor, a Gaussian-feature capacity floor,
Hewitt–Liang control tasks, intrinsic dimensionality.

1. **Recoverability.** Ridge from frozen Z (2048-d) → each of the **1,613 real Mordred descriptors**.
   Report the R² distribution against the existing floors (random-weight CheMeleon of identical
   architecture, Gaussian-2048, train-mean, PCA-restricted probe). **Install real Mordred for this** —
   `rep2_probes.py` currently admits it uses RDKit's ~210 descriptors as a proxy, and since Mordred
   *is* the pretraining target, the proxy understates exactly the quantity in question
   (`uv add mordredcommunity`; the original `mordred` is unmaintained).
2. **Subspace split.** Let `P` project Z onto the span of the Mordred-predictive directions, fitted on
   an **external** PubChem-like corpus so the split never sees downstream labels. Decompose each task
   three ways: ridge on `PZ`, ridge on `(I−P)Z`, ridge on raw Mordred.
3. **The decisive statistic.** If ridge on `PZ` ≈ ridge on `Z` > ridge on Mordred, then CheMeleon's
   win is **a better-conditioned basis for the same information**, not new chemistry — and that
   explains simultaneously why descriptor-target pretraining works (Burns 2026), why SSL pretraining
   doesn't (Praski 2025, Sun 2022), and why RDKit features regularize small-data D-MPNNs (Yang 2019).
   If `(I−P)Z` carries the win, that is a positive finding: message passing learned something Mordred
   cannot express, and the next question is what.
4. **The HPO-budget curve** — fills the hole all three papers leave. Plot the CheMeleon-minus-Mordred
   gap against **matched** HPO trials (0, 1, 10, 50, 200 per arm; ridge λ then LightGBM) on the same
   splits. Sun, Dai & Yu, *"Does GNN Pretraining Help Molecular Representation?,"* **NeurIPS 2022**
   ([arXiv:2207.06010](https://arxiv.org/abs/2207.06010)) found *"hyper-parameters could have larger
   impacts on accuracy of downstream tasks than the choice of pretraining tasks, especially when the
   scales of downstream tasks are small."* If the gap shrinks monotonically toward zero, the low-data
   foundation-model advantage is partly a tuning artifact. If not, you have strengthened CheMeleon on
   a dimension its authors left open — which is also publishable and more generous.
5. **Leakage check, nearly free.** CheMeleon was pretrained on 1M random PubChem molecules. Report
   exact-InChIKey overlap with each test set. Nobody reports this.
6. **Tie-in to the series.** `residual_flow.py` already exists; regress the target on Mordred and ask
   how much of the *residual* Z explains, and symmetrically. Paired with the Jacobian/source-term
   analysis already written, *"the constant source term `H_0` is why the embedding keeps
   descriptor-recoverable information at depth 6"* is a mechanism claim this study is uniquely placed
   to make.

**Confounds to pre-register:** Z is 2048-d and Mordred ~1,613-d after NaN/constant filtering, so
standardize both and hold ridge capacity fixed (`rep2_probes.py` already does); Mordred NaN/inf
handling must be identical in probe and baseline or the Mordred arm is silently weaker; fit `Z →
Mordred` on an external corpus only; and ridge is a *linear* probe, so low R² means "not linearly
decodable," not "absent."

**Feasibility:** Mordred over ~10k SMILES with multiprocessing is tens of minutes on CPU; one frozen
forward pass of the depth-6 `d_h=2048` D-MPNN over 10k molecules is minutes on MPS (chunked encoders
already in `scripts/qlib.py`); ridge over 2048 features × 2k rows × 20 splits × 6 datasets × 30 λ is
seconds. A weekend, no GPU.

**Audience** methods. **Novelty: high** — this is the question the pretraining literature set up and
nobody asked. **Risk:** it is a probing result, so the conclusion is about linear decodability; say
so.

---

## 7. "Four Chemprop flags that changed meaning between versions, and the docs were wrong about all four"

**Central claim.** Comparing a v1 number to a v2 number, or a v2.0 number to a v2.2 number, requires
knowing four undocumented or mis-documented semantic changes. Each was confirmed by a maintainer, and
in three of four the authoritative documentation was wrong at the time.

1. **`--ffn-num-layers` changed meaning** (#1183, open 15 months, resolved docs-only in #1368). v1:
   number of `Linear` layers. v2: number of **hidden** layers, i.e. `n_layers + 1` Linears. To
   reproduce a v1 `--ffn_num_layers 2` architecture you pass **1**. Maintainer: *"our documentation
   doesn't mention this. Our v1 to v2 transition spreadsheet says that we changed the number of
   default FFN layers from 2 to 1 and that we did not change `--ffn_num_layers`. While the default
   value… did change from 2 to 1, because we also changed `--ffn_num_layers` to give one more layer
   than before, the default didn't actually change. What did change is our hpopt search space."* A
   user walked straight into it in-thread. @davidegraff's rename proposal was declined.
2. **The "Noam" scheduler was never Noam** (#968/#969, 12 comments). *"the actual definition of the
   Noam scheduler does not match up with the v1 implementation… it's originally defined as a square
   root decay rather than exponential decay… this seems to be an implementation error with the v1
   scheduler."* @shihchengli: *"Do we want to save this incorrect implementation under another name
   and still keep it in our codebase, since we have used it for a long time and it seems to work well
   in our projects?"* — which is what happened. v2.0.5: *"The current scheduler does work well though,
   so it was decided to not change the definition. Instead the scheduler was renamed and
   refactored."* Also unresolved: what `d_model` means for a GNN with different message and FFN widths.
3. **`--num-folds` meant a 33/33/33 split** (#1168). v2.0.4 `--num-folds 3` → a **33/33/33** split;
   v2.1 `--num-replicates 3` → **3 × 80/10/10**. Maintainer: *"Our documentation for <2.1 was
   incorrect about `--num-folds`. It said 'k-fold cross-validation can be run by specifying
   `--num-folds <k>`'… which makes it sound like specifying `--num-folds` was enough. But actually
   that flag does nothing if you don't specify `--split cv`."* Then v2.1 deleted the `cv` split types
   (#994).
4. **Quantile "uncertainty" changed by a factor of two** (#1228). v1 reported half the interquantile
   range; v2 reports the full range. Undocumented. (Candidate 1's territory; cite once.)

**And three more that are release-note-only**, which is where the post earns its keep:

- **Batch norm was disabled by default in v2.1.0** (#1058). v1 had it on.
- **`val_loss` was redefined in v2.1.0** to be computed the same way as `train_loss` (#1020) —
  silently changing what every checkpoint callback and every hpopt objective monitors.
- **The default featurization changed mid-2.x.** v2.2.0: *"Stereochemical information (R/S and
  cis/trans) is included in the default featurization. If a model is trained on molecules that do not
  include this stereochemical info, some of the model weights will not be updated. This could cause
  erroneous predictions at inference time if molecules are used with stereochemical info."*
  `--ignore-stereo` is opt-in (#1196, #1216).

**The quietest high-impact one of all, #1097 (v2.1.1):** *"we fixed a bug where the transforms for
scaling extra features/descriptors were **turned off during validation**. This caused models trained
with these extra inputs to **not report accurate metrics during training**, which is a problem if the
'best' model is selected instead of the last model as is done in hyperparameter optimization."* So
for everyone using `--descriptors-path` or `--atom-features-path`, every validation metric → every
checkpoint selection → every hpopt result from v2.0 through v2.1.0 was computed on unscaled inputs.
A second fix in the same family landed in v2.2.2 (#1309, *"Use raw extra features/descriptors when
scaling"*). There is **no CHANGELOG.md**; release notes are the changelog.

**The honest headline, and the restraint is the credibility.** Across 687 issues there is **no
confirmed case of v2 being less accurate than v1.** The only direct complaint (#1168) resolved as
user error, and the error is item 3. Issue #426 is a genuine measured regression — but v1.5 → v1.6
(atom maps silently dropped in reaction mode; `is_keeping_atom_map()` always returned False so
`reac_id_to_prod_id` came back `{}`, degrading the CGR; reporter's val MAE **10.49 (v1.6) vs 6.11
(v1.5)** at 16 epochs; fixed v1.6.1), not v1 → v2. So the thesis is **"nobody reported v2 being
worse, but here are seven silent changes that make a v1→v2 comparison meaningless"**, not "v2 is
worse."

**Experiment.** None required. One cheap addition worth making: build the v1-equivalent v2
architecture (`--ffn-num-layers 1`) and the naive translation (`--ffn-num-layers 2`) and report the
measured difference against the noise floor, so the post ends with a number rather than a warning.

**Audience** practitioner. **Novelty: medium-high** — every item is public in a closed issue, none is
assembled anywhere, and the v1→v2 migration guide is **a public Google Sheet that #1183 proves was
wrong and nothing in CI validates**.

---

## 8. "Your multicomponent model is not permutation-invariant, and the column order is part of the model"

**Central claim.** `MulticomponentMessagePassing` **concatenates** per-component embeddings, so for a
solvation or reaction-condition model `solute,solvent` and `solvent,solute` are different inputs
producing different predictions. The CSV column order is a modelling choice, it is not documented as
one, and there is no way to ask for invariance.

**Evidence.** Issue **#806**, open **2.5 years**, 13 comments, a substantial theory discussion
(invariance vs *anti*symmetry — the ΔProperty / DeepDelta case wants antisymmetry), with
@davidegraff sketching a `Fusion` module to replace the `torch.cat` and @kevingreenman noting *"I'm
trying to think of whether there's a way for us to satisfy some of the 'guarantees' from approaches
like DeepDelta… that currently require significant data augmentation."* **Closed October 2026 with
nothing shipped.** Fifteen issues touch multicomponent. Supporting evidence that the subsystem is
under-tested: **#1098** (v2.1.1) *"Fix batch size calculation for multicomponent"* — batch sizes were
simply wrong for an entire release; **#1054** — the `v1_to_v2` converter silently produced a
**half-size** model file (2.3 MB → 1.2 MB) for 2-SMILES-column models, then `KeyError: 'blocks'` at
predict, and after a community fix still had the wrong input dimension (`3748x86 and 147x300`) because
the v1 and v2 featurizers differ; and **#575** / **#506**, both `[TODO]: add tests for multicomponent`,
**filed after the code shipped**.

**Experiment.** Train one multicomponent regression model (a solvation set, or any two-column set)
and predict the test set twice with the component columns swapped. Report the distribution of
|Δprediction| against the study's established noise floor. Both outcomes publish: a large spread
means column order is a silent hyperparameter nobody tunes; a small spread means the model *learned*
approximate invariance from data, which is more interesting and raises the question of what capacity
that cost. The cleanest variant uses `--mpn-shared`, where the two components share weights so the
concatenation is the only asymmetry — making the measurement a direct read on the concatenation.
Cheap: one small model, two prediction passes.

**Audience** practitioner, especially solvation, co-crystal and reaction-condition work.
**Novelty: medium-high** — the behaviour is discussed by maintainers in #806; the *measurement* of how
much it matters is absent, and the issue closing with nothing shipped means a user arriving today
gets no guidance.

---

## 9. "Chemprop featurizes a reaction in O(N²)"

**Central claim.** `CondensedGraphOfReactionFeaturizer.__call__` builds the CGR edge list by scanning
**every atom pair**, while the molecule featurizer iterates bonds. Reaction featurization is
quadratic in combined atom count where molecular featurization is linear.

**Evidence.** `chemprop/featurizers/molgraph/reaction.py:128-137`:

```python
for u in range(n_atoms_tot):
    for v in range(u + 1, n_atoms_tot):
        b_reac, b_prod = self._get_bonds(reac, pdt, r2p_idx_map, pdt_idxs, n_atoms_reac, u, v)
        if b_reac is None and b_prod is None:
            continue
```

Compare `molgraph/molecule.py`, which iterates `mol.GetBonds()`. The correct construction for a CGR
is the *union* of reactant and product bonds mapped through `r2p_idx_map` — O(E_r + E_p).

Pair-scan cost alone *(measured here*; `_get_bonds` does strictly more, so these are lower bounds):

| combined atoms N | pairs | pair-scan |
|---|---|---|
| 40 | 780 | 0.16 ms |
| 80 | 3,160 | 0.55 ms |
| 150 | 11,175 | 1.87 ms |
| 300 | 44,850 | 7.88 ms |

Calibration: post 1 measured *whole-molecule* featurization at 102 µs/mol. So a 40-atom CGR spends
more time in the pair scan alone than a drug-sized molecule spends on its entire featurization, and a
150-atom CGR — ordinary for an enzymatic or organometallic reaction — about 20×.

**Gate this before benchmarking.** Determine whether anyone trains on reactions large enough for this
to bite. Published Chemprop reaction work is mostly small organic reactions at N ≈ 30, only 11 issues
touch reactions/CGR, and v2.3.1 shipped cuik-molmaker for reactions with a claimed 3× speedup (#1365)
which may already dominate this path. **If realistic N is ~30, this is a one-paragraph upstream PR,
not a post.** If it is written: (a) benchmark the real featurizer against a bond-union rewrite,
verifying bit-identical `MolGraph` output using post 1's verification protocol; (b) sweep N. Honest
framing: an asymptotic defect whose practical importance is unestablished — fine said that way, bad
said any other way.

**Audience** systems. **Novelty: high** — nobody has looked.

---

## 10. "The dataset size Chemprop says it can't handle" — append to post 1

**Central claim.** Issue **#858** was closed **wontfix** after a user OOM'd a 10M-molecule dataset at
256 GB; maintainers measured ~130 GB for 5M *unfeaturized* datapoints and declared it out of scope:
*"Chemprop will greedily load your entire dataset at the beginning of training… This is a design
decision that massively simplifies the backend work."* Post 1's type-ID representation is 217 B/mol
against 7,530 B/mol for a cached `MolGraph` — **35×** — which changes the arithmetic on exactly this
case.

**The arithmetic, and its honest limit.** At 10M molecules: 2.2 GB of type-ID arrays versus 75 GB of
`MolGraph`s. But #858's 130 GB is for **unfeaturized** datapoints, i.e. retained RDKit `Mol` objects,
which the type-ID cache does **not** address. So the correct claim is narrower: the type-ID
representation removes the `MolGraph` half, and you would still need lazy `Mol` construction — itself
an open maintainer TODO, **#1256** *"Reduce memory usage by forming RDKit mol object only when
needed"*. Together the two plausibly move the ceiling an order of magnitude; separately, neither
does. **State it that way or the post overclaims.**

**Supporting material.** PR **#903** (which made CLI caching default-on) has Jackson Burns' numbers —
*"in my tests with 1M datapoints, the cached Molgraphs only took on the order of 10 GB of memory. I
figure if someone is trying to train on 1M datapoints, they probably have that much memory to
spare"* — and a still-unfixed leak noted in the same PR: *"after setting the cache `dataset.cache =
True`, trying to clear the cache `dataset.cache = False` only reduced the memory by 1G instead of
10G, so we may not be clearing the cache correctly. This can be looked into later."* The competing
answer that must be benchmarked against rather than ignored: cuik-molmaker (v2.2.1) claims *"1.6×
training, 2.4× inference, memory reduced by ~80%"*, and in v2.3.0 it became a **mandatory**
dependency (#1375).

**Experiment.** Peak RSS for `MolGraph` cache vs type-ID cache vs cuik-molmaker at 10k / 100k / 1M
molecules, plus the leak check (`cache = True` then `False`, measure RSS). The 1M point is feasible on
a laptop; the 10M claim must be extrapolated and labelled as extrapolation. An hour or two.

**Audience** systems. **Novelty: medium** — post 1's number applied to a specific wontfix'd issue,
which is a payoff rather than a new finding. **Best shape: a short follow-up appended to post 1.**

---

## 11–12. Lower-confidence literature candidates

**11. "Five papers solved activity cliffs in two years. None has been replicated."**
The anchor is van Tilborg, Alenicheva & Grisoni, *"Exposing the Limitations of Molecular Machine
Learning with Activity Cliffs,"* **JCIM 62(23), 5938–5951 (2022)**,
[10.1021/acs.jcim.2c01073](https://doi.org/10.1021/acs.jcim.2c01073) — 24 approaches, 30 targets,
*"while all methods struggled in the presence of activity cliffs, machine learning approaches based
on molecular descriptors outperformed more complex deep learning methods"* (MoleculeACE). It has one
independent replication with an important qualifier — Dablander, Hanser, Lambiotte & Morris,
*"Exploring QSAR models for activity-cliff prediction,"* **J. Cheminform. 15, 47 (2023)**,
[10.1186/s13321-023-00708-w](https://doi.org/10.1186/s13321-023-00708-w), which confirms the failure
but finds a *"substantial increase in AC-sensitivity when the actual activity of one compound is
given"*, i.e. the failure is largely of *absolute potency* prediction, not of cliff *detection* given
an anchor. Since then: **Cui et al.** (Nat. Commun. 17, 9376, 2026,
[10.1038/s41467-026-75713-2](https://doi.org/10.1038/s41467-026-75713-2)), **Cheng et al.** MaskMol
(BMC Biology 23, 279, 2025, [10.1186/s12915-025-02389-3](https://doi.org/10.1186/s12915-025-02389-3),
claiming to beat 25 SOTA methods), **Yu et al.** ACtriplet (J. Pharm. Anal. 15(8), 101317, 2025),
**Wan et al.** (Nat. Commun. 16, 413, 2025,
[10.1038/s41467-024-55082-4](https://doi.org/10.1038/s41467-024-55082-4)), **Gu et al.** GGAP-CPI
(JCIM 65(19), 9910–9926, 2025) — five papers in two years, each on self-chosen comparisons, **none
independently replicated**. That is precisely the pattern Errica / Shchur / Tönshoff warn about. Two
dissents sharpen it further: **Schuh et al. 2026** found injected activity cliffs produced only
**−0.023 ROC AUC**, by far the smallest of four perturbations, so cliffs may matter *less* to
benchmark unreliability than contamination or label conflicts; and **López-Pérez &
Miranda-Quintana** argue the **SALI formulation itself is mathematically flawed** (*iCliff*, JCIM
65(11), 5801–5810, 2025, [10.1021/acs.jcim.5c00506](https://doi.org/10.1021/acs.jcim.5c00506)) and,
separately, that *"as far as generalization, random splitting was the better-performing data
splitting alternative overall"* (Mol. Inf. 44(1), e202400054, 2025,
[10.1002/minf.202400054](https://doi.org/10.1002/minf.202400054)) — which cuts against the
Guo/Walters/Lo-Hi consensus and should be engaged, not ignored. **Verdict: a good skeptical post, but
the cheap version is a literature post with no new measurement, and the measured version needs
re-running five papers' experiments. Lower priority than 1–8.** The one cheap original piece inside
it is the error decomposition from candidate 1's design warning: what fraction of a model's large
errors are cliff-driven versus distance-driven, which appears to be unquantified.

**12. "Mayr says yes, Jiang says no, and both are right."**
The cleanest contradiction in the field. **Mayr, Klambauer, Unterthiner, Steijaert, Wegner,
Ceulemans, Clevert & Hochreiter, Chem. Sci. 9, 5441–5451 (2018)**,
[10.1039/c8sc00148k](https://doi.org/10.1039/c8sc00148k), using **nested cluster-cross-validation**
designed to defeat compound-series and HP-selection bias, found *"deep learning methods significantly
outperform all competing methods"* on ChEMBL-scale multi-task data. **Jiang, Wu, Hsieh, Chen, Liao,
Wang, Shen, Cao, Wu & Hou, J. Cheminform. 13, 12 (2021)**,
[10.1186/s13321-020-00479-8](https://doi.org/10.1186/s13321-020-00479-8), 11 datasets × 50 runs,
found *"the descriptor-based models outperform the graph-based models in terms of prediction accuracy
and computational efficiency"*, with descriptor models taking **24 of 33** top-3 positions and graph
models winning only *"on a fraction of larger or multi-task datasets"* (Tox21, ToxCast). The
reconciliation is **scale and task structure, not method**: Mayr used hundreds of thousands of
compounds across >1,000 assays; Jiang used mostly sub-5,000-molecule single-task sets. **Deng, Yang,
Wang, Ojima, Samaras & Wang, Nat. Commun. 14, 6395 (2023)**,
[10.1038/s41467-023-41948-6](https://doi.org/10.1038/s41467-023-41948-6), supplies the explicit
ladder from **62,820 models**: classical ML on fixed representations wins below ~1,000 molecules;
SMILES RNNs show advantages around ~6,000; MolBERT improves above ~10,000; representation learning
reaches parity around ~100,000+. Their thesis sentence is quotable on its own: *"a model cannot save
an unqualified dataset which cannot remedy an improper evaluation for an ambiguous chemical space
generalization claim."* So the honest framing is **not "GNNs don't work" but "MoleculeNet is in the
regime where GNNs cannot be shown to work."** The benchmarking sweep found **no paper that states
this reconciliation explicitly**, and the ladder is a **single-paper** claim (Deng 2023), directionally
consistent with Jiang 2021 and Yang 2019. Counterweights to include: **Sypetkowski et al.,
*"On the Scalability of GNNs for Molecular Graphs,"* NeurIPS 2024**
([arXiv:2404.11568](https://arxiv.org/abs/2404.11568)) — MolGPS beats prior SOTA on **26 of 38** tasks,
but needed ~5M molecules × hundreds of millions of labels of *supervised multi-task* pretraining; and
**Yang et al. 2019**'s own account of the `rdkit_2d_normalized` trick, which is the mechanism in
miniature: *"The results appear to be highly data set-dependent. Some data sets, such as QM9 and
ESOL, show marked improvement with the addition of features, while other data sets, such as PCBA and
HIV, actually show worse performance with the features"*, because *"the features may help to
regularize a representation derived from a small data set: because the features are derived from more
general chemical knowledge, they implicitly provide the model some understanding of a larger chemical
domain."* **Verdict: zero compute, strong thesis, but it is an argument rather than a finding, and it
is only as good as the writing. Best as a framing section inside candidate 6, which measures the same
mechanism.**

---

## Pruned

Investigated and rejected, or substantially reduced. Eight entries; the first three are reversals of
things this brief initially believed.

**P1. "Chemprop's scaffold split and DeepChem's put the same molecules on opposite sides." — NOT
TRUE.** I expected astartes' "fill test first" to invert DeepChem's "fill train first." It doesn't:
astartes sorts **ascending** and fills test first, DeepChem sorts **descending** and fills train
first, and those compose to the **same** direction — rare scaffolds to test, large scaffolds to train
— in all three implementations. Measured acyclic fractions and achieved split sizes agree closely
across all three (§2). The surviving claim is about *overlap, reproducibility and difficulty*, not
direction. This correction belongs **inside** candidate 2; it is the kind of reversal the series
README says to keep.

**P2. "Nearest-neighbour similarity to the training set beats learned UQ." — BACKWARDS, and this is
the sharpest reversal in the brief.** The head-to-head literature exists, is ~18 years old, and
mostly goes the other way:

- Tetko et al., **JCIM 48(9), 1733–1746 (2008)**,
  [10.1021/ci800151m](https://doi.org/10.1021/ci800151m) — six distance-to-model types including
  Tanimoto and leverage, 12 QSAR models: *"the distances to model based on standard deviation of
  predicted toxicity calculated from the ensemble of models afforded the best results."*
- Sushko et al. (30 authors), **JCIM 50(12), 2094–2111 (2010)**,
  [10.1021/ci100253r](https://doi.org/10.1021/ci100253r) — 30 models: *"the DMs based on an ensemble
  (consensus) model provide systematically better performance than other DMs."*
- Sheridan, **JCIM 53(11), 2837–2850 (2013)** — the most damaging result, from the author who
  originated the distance view: *"it appears that it is possible to construct a useful error model
  using only two metrics (TREE_SD and PREDICTED). These do not require calculating
  similarities/distances between the molecules being predicted and the molecules used to build the
  activity model."*
- Toplak et al., **JCIM 54(2), 431–441 (2014)**,
  [10.1021/ci4006595](https://doi.org/10.1021/ci4006595) — 20 datasets, 10 reliability methods:
  alternatives *"can outperform standard reliability scores that rely only on similarity to compounds
  in the training set."*

**Distance wins in exactly one regime, and it is well documented.** Liu & Wallqvist, **JCIM 59(1),
181–189 (2019)**, [10.1021/acs.jcim.8b00597](https://doi.org/10.1021/acs.jcim.8b00597): ensemble
variance is **non-monotonic in distance** — it rises, then *falls* for genuinely out-of-domain
molecules — *"This explains why ensemble variance fails to identify large prediction errors"* — and
their similarity metric wins there, while *"for within-domain compounds, the two metrics performed
similarly, with ensemble variance marginally but consistently outperforming"*. Note that is a
**single paper**, RF + descriptors, melting point + 4 datasets; the non-monotone turn-down has never
been shown for an MVE head or a learned representation on public data, which is why it appears as an
add-on experiment under candidate 1. Also note the regime-dependence result — Sheridan, **JCIM 55(6),
1098–1107 (2015)**, [10.1021/acs.jcim.5b00110](https://doi.org/10.1021/acs.jcim.5b00110):
`SIMILARITYNEAREST1` matters more on *less diverse* training sets, with the operational rule *"when
the mean pairwise similarity … within a QSAR training set is less than 0.5, one can use only TREE_SD,
PREDICTED, but otherwise one should use TREE_SD, PREDICTED, SIMILARITYNEAREST1"* — **single paper,
proprietary Merck data, random forest only, never replicated on public data or for neural models.**
The second, independent sweep reached the same place from the UQ side and supplied the defensible
wording: *"Fingerprint- and latent-distance baselines are competitive with learned UQ for
rank-ordering errors on molecules, and have been reported to beat specific learned baselines on
specific datasets (Janet 2019; Musielewicz et al. 2024 for GNN interatomic potentials). But the one
systematic head-to-head on molecular property regression (Hirschfeld, JCIM 2020) found no method
consistently superior, did not find distance methods on top, and recommended a learned method. The
stronger claim is not established."* Note also Hirschfeld's structural reason why distance baselines
*should* sometimes fail: *"Distance metrics implicitly assume that there is a relatively smooth
mapping between the input representation and the output value, so … this assumption is clearly
violated for some molecular property prediction tasks due to the known presence of structure-activity
cliffs."* **Disposition:** the premise of the original ask is wrong; downgrade it to a conjecture
wherever it appears. The honest post is candidate 1's "combine, don't choose", and the cheap
replication targets are the Liu & Wallqvist non-monotone turn-down and Sheridan's 0.5 diversity
threshold.

**P2b. "GOOD and DrugOOD are uncertainty benchmarks." — NO.** Verified by grepping the full texts:
**`"calibrat"` returns 0 hits in both.** Gui, Li, Wang & Ji, *GOOD*, **NeurIPS 2022 D&B**
([arXiv:2206.08452](https://arxiv.org/abs/2206.08452)) and Ji et al., *DrugOOD*
([arXiv:2201.09637](https://arxiv.org/abs/2201.09637)) measure **OOD generalisation**, i.e. accuracy
under shift, not UQ — DrugOOD's "uncertainty" hits concern *measurement* uncertainty during curation.
The same goes for Antoniuk et al., *BOOM*, **NeurIPS 2025 D&B**
([arXiv:2505.01912](https://arxiv.org/abs/2505.01912)), which is nonetheless worth citing for its own
finding across 150+ model-task combinations: *"no existing model achieves strong generalization across
all tasks: even the top-performing model exhibited an average OOD error 3× higher than
in-distribution. Current chemical foundation models do not show strong OOD extrapolation"*, with the
mechanism that poor-OOD models *"overwhelmingly produce an S-shaped parity plot … capable of
clustering OOD samples together but unable to extend the prediction region beyond the training
data"*, and *"ID performance is not necessarily correlated with OOD performance."* **Use these for
"models don't generalise OOD," never for "UQ doesn't detect OOD."** The gap is itself notable: all
three major molecular OOD benchmarks measure accuracy under shift and **none measures calibration
under shift** — which is the hole candidate 1's headline figure falls into.

**P3. "GNN latent-space distance is a reliable applicability-domain metric." — UNSUPPORTED, not
refuted.** Janet, Duan, Yang, Nandy & Kulik, **Chem. Sci. 10, 7913–7922 (2019)**,
[10.1039/c9sc02298h](https://doi.org/10.1039/c9sc02298h) is the source: *"The calibrated performance
of this approach exceeds widely used uncertainty metrics."* Hirschfeld et al. circumscribe it
precisely: *"Janet et al. found the calibrated Latent Distance UQ to significantly outperform
uncalibrated ensembling on the two datasets they tested using a FFN with an application-informed
feature vector"* — two datasets, a feature-engineered FFN rather than a graph net, against
*uncalibrated* ensembling — and in their own broader benchmark latent distance is not top-ranked. The
prior should be skeptical on theoretical grounds too: Liu et al., SNGP, **NeurIPS 2020**
([arXiv:2006.10108](https://arxiv.org/abs/2006.10108)) identify *"input distance awareness … as a
necessary condition for a DNN to achieve high-quality uncertainty estimation"*, and van Amersfoort et
al., **ICML 2020** ([arXiv:2003.02037](https://arxiv.org/abs/2003.02037)) show feature collapse needs
an explicit gradient penalty *"enforcing detectability of changes in the input"* — a trained encoder
has no incentive to keep OOD inputs far away in latent space. But no paper was located that
*demonstrates* latent distance failing on molecules, so the status is **unsupported, not refuted**,
and "measure it on CheMeleon embeddings vs ECFP Tanimoto" is a legitimate sub-experiment under
candidate 1 rather than its own post.

**P4. "Bounding box, convex hull and leverage/Williams plots are useful AD methods." — NO SUPPORTING
EVIDENCE as error predictors.** Sahigara et al. (Molecules 2012) show convex hull and KDE mainly
**reject the most compounds**, which improves retained-set statistics trivially — you get the same
effect by rejecting at random proportionally — and is not evidence of error discrimination. Leverage
is widely recommended in regulatory QSAR and was among the six distances that *lost* to ensemble SD
in Tetko 2008. Nothing to write; include only as a one-line "don't bother" if candidate 1 needs a
baseline table.

**P5. "`datasets.py:220` defaults `cache=False`, so featurization repeats every epoch." — NOT TRUE
FOR CLI USERS,** who are almost everyone. `MoleculeDataset.__post_init__` does set `self.cache =
False` (`data/datasets.py:220`), but `cli/train.py:2337-2346` does `if not args.no_cache:
train_dset.cache = True; val_dset.cache = True`. The CLI caches train and val by default with
`--no-cache` as the opt-out; the test set is deliberately uncached; `hpopt` inherits via
`train_model`; `predict` is single-pass. **What survives is narrower and still worth one section:** the
Python API defaults to on-the-fly featurization, `num_workers` defaults to 0
(`cli/common.py:35-39`) so that re-featurization is single-threaded, and **of the 24 notebooks in
`examples/` only 2 mention `cache`** (`convert_v1_to_v2.ipynb`, `extra_features_descriptors.ipynb`) —
`training.ipynb`, the canonical entry point, is not one of them. So anyone following the Python
tutorial re-featurizes the whole training set every epoch while the CLI user on the same data does it
once, and nobody has filed it. A verified, original, *small* finding. It is also the **second example
of candidate 3's thesis** — the Python API and the CLI disagree about defaults in ways that change
both your runtime and your results — and the **third** is issue **#1382** (OPEN), where
`CuikmolmakerDataset` silently ignores `_reorder_atoms` / `_keep_h` / `_add_h` because it featurizes
straight from the SMILES string (*"This is a silent bug. Any atom-level data supposed to be aligned to
the atom mapping gets paired with the wrong graph node… with no error raised"*, with a three-line
repro printing `MoleculeDataset: [0.12011 0.15999 0.12011]` vs `CuikmolmakerDataset: [0.12011 0.12011
0.15999]`) — and the CLI guards that combination at `cli/common.py:247-251` while the Python API does
not. **Three instances of the same asymmetry is a post; one is a footnote.** Consider folding P5 +
#1382 + #1383 (`make_mol(reorder_atoms=True)` silently scrambled atom order, fixed v2.3.1) + #382
(atom-mapped SMILES yielding non-identical scaffolds → *"data contamination during scaffold splits"*)
into candidate 3 as its second half, or splitting them out as **"the CLI guards it, the API
doesn't."**

**P6. "`AttentiveAggregation` has a softmax overflow hazard." — TRUE BUT UNREACHABLE.** Verified
*(measured here)*: with `self.W(H).exp()` and no max subtraction (`nn/agg.py:130`), output goes nan
once the max logit exceeds ~88.7 in fp32 — finite at max logit 16.3, nan at 97.6 and 130.1; in fp16
the threshold is ~11.1. No `logsumexp`, no `clamp`, no per-graph max subtraction, and zero issues or
PRs mention numerical stability for this path. *But:* `AttentiveAggregation` carries **no
`@AggregationRegistry.register(...)` decorator** — unlike `MeanAggregation`, `SumAggregation`,
`NormAggregation` — so `--aggregation attentive` does not exist and it is absent from
`AggregationRegistry.keys()` in the search space; #1207 notes *"We could add our attention
aggregation to the search space by adding it to the aggregation registry"*; it was already broken for
checkpoint reload (#1120, fixed #1133); and Chemprop's CLI has **no mixed-precision flag**, so the
dangerous fp16 threshold needs a hand-built `Trainer(precision="16-mixed")`. **Disposition:** an
upstream issue (register it or drop it from `__all__`; subtract the per-graph max) plus at most two
sentences elsewhere. The tempting framing — "the one aggregator the CLI can't reach is the one that
isn't overflow-safe" — is cute, but exposure is near zero and a post about an unreachable code path is
a post about nothing.

**P7. "ONNX export is fragile." — TRUE, WELL-DOCUMENTED BY THE MAINTAINERS, AND BELONGS IN POST 4.**
`grep -ri onnx chemprop/` returns nothing outside this study's briefs — Chemprop has no export path,
so #1335 is about users exporting it themselves. The issue content is excellent: a reporter's 2×2
ablation showing (i) un-patched export fails with `ScatterElements: indices element out of data
bounds, idx=10 must be within [-10,9]` because message passing allocated scatter outputs with
`len(bmg.V)`, a **Python int** that `torch.export` specialised to the export example's atom count
(fixed with the symbolic `bmg.V.shape[0]`, PR #1379, v2.3.1, in **four** places); and (ii)
`scatter_reduce(reduce="mean")` is **silently numerically wrong** under ONNX — `max_abs_diff =
0.107093`, `mean_abs_diff = 0.043006` on a regression head, not a crash — because ONNX has no mean
scatter_reduce (upstream [pytorch#175738](https://github.com/pytorch/pytorch/issues/175738)). The
decision arc is the best part: a maintainer first refused the sum/count rewrite on performance
grounds (*"I don't think we can upstream `reduce='mean' -> reduce='sum' / count` for performance
reasons"*), then it was benchmarked — *"the speed difference isn't much. If anything the two-sum
approach may actually be slightly faster"* — reversing the decision (PR #1401, merged 2026-10-21;
verified at `nn/agg.py:66-85`). And **PR #1421 is still OPEN**, so `--batch-norm` models still cannot
export and every user hand-rolls `register_pytree_node` for `BatchMolGraph`. **Disposition:** this is
the *resolution* of post 4's own open gap (the unverified ORT-MLX `ScatterND` vs `ScatterElements`
claim), and the 0.107 silent-wrongness finding strengthens post 4's existing "two more
silent-wrong-answer bugs" section, which already has the right frame. **Appendix or revision to post
4, not a new post.**

**P8. "Chemprop v2 training is non-deterministic even at a fixed seed." — TRUE, AND ALREADY POST 2's
TERRITORY.** Issue **#755** is the origin story for the coupling post 2 documents: a reporter ran the
identical ESOL command twice with `--data-seed 42 --pytorch-seed 42` and got test/mse **0.477** vs
**0.560** — a 17% swing, larger than most published architecture deltas — because `pl.Trainer` never
got `deterministic=True`. @KnathanM's framing: *"there are three sources of randomness… 1. The data
split 2. The model parameter initialization 3. The steps taken by `pl.Trainer`. We currently have CLI
flags to control 1 and 2."* The fix (`cli/train.py:1824-1831, 1964`) couples determinism to
`--pytorch-seed`, and **`--data-seed` defaults to 0 while `--pytorch-seed` defaults to None**
(`train.py:583-593`), so out of the box you get a reproducible *split* with non-reproducible *weights*
— the worst combination for benchmarking, because it looks reproducible. `hpopt.py:418` does the same.
Post 2 already says all of this, including that `torch.use_deterministic_algorithms(True)` silently
does nothing on MPS. **Disposition:** add #755's 0.477-vs-0.560 numbers and the flag-default asymmetry
to post 2 as the historical citation it currently lacks. Not a new post.

**P9. Claims that research shows are NOT TRUE, and must not be repeated.**

- **"fastprop shows descriptors beat Chemprop."** Overstated as usually quoted. Burns & Green,
  **J. Cheminform. 17, 73 (2025)**,
  [10.1186/s13321-025-01013-4](https://doi.org/10.1186/s13321-025-01013-4) — mordred + 2-layer MLP —
  beats Chemprop on 6 of 8 datasets but beats the *best published* model on only **2 of 8** (flash
  point, PAH). Critically, **the tuning budget is not matched and the paper says so**: fastprop metrics
  are *"reported after model optimization using the `fastprop train ... --optimize` option"* while
  re-run Chemprop used *"the default settings… except that the number of epochs is increased"*. So the
  defensible statement is "**tuned** mordred-MLP beats **default-hyperparameter** Chemprop on 6 of 8
  small datasets," which is a much weaker claim. The authors also declare their failure modes:
  *"fastprop is fundamentally limited by the types of chemicals which can be uniquely described by the
  mordred package"* and *"incapable of incorporating atomic features"* (hence QuantumScents, AUROC
  0.651 vs Chemprop's 0.88), and it loses QM9 to Uni-Mol on 3D and Fubrain to DeepDelta. Single paper,
  7 citations, **no independent replication located**.
- **"Do we really need graph neural networks for molecular property prediction?"** — **this paper
  could not be found.** Not in Crossref, not on arXiv, not in the citation graph of any related work.
  It appears to be a conflation of MOLTOP, the peptide-fingerprints paper, and/or Jiang et al. 2021.
  **Do not cite it.** If the rhetorical title is wanted, the honest attribution is Jiang et al. 2021.
- **GROVER's "more than 6% on average over SOTA"** (Rong et al., NeurIPS 2020,
  [arXiv:2007.02835](https://arxiv.org/abs/2007.02835)) — **contradicted independently twice**: Deng
  et al. 2023 found RF on fixed representations beats GROVER on BACE, BBBP, ESOL and Lipophilicity,
  and Praski et al. 2025 rated GROVER statistically *worse* than ECFP.
- **Hu et al.'s "up to 9.4% absolute improvements in ROC-AUC"** (ICLR 2020,
  [arXiv:1905.12265](https://arxiv.org/abs/1905.12265)) is routinely quoted as a typical gain. It is a
  best-task number, and the same paper's own warning is under-quoted: *"naive strategies, which
  pre-train GNNs at the level of either entire graphs or individual nodes, give limited improvement
  and can even lead to negative transfer on many downstream tasks."* **Negative transfer was reported
  in the founding paper of the field.**
- **"Frozen CheMeleon + simple head vs Mordred + simple head on small datasets" — already published
  twice** (Praski et al. 2025; Ben Hicham et al. 2026, with TabPFN+CheMeleon at 86.2% vs
  TabPFN+Mordred at 67.2% across 58 datasets). Do not frame a post around re-finding it. See
  candidate 6 for what is actually unclaimed.

**P10. "Atom/bond-level and multi-component targets are under-documented." — MOSTLY NOT TRUE.**
`docs/source/tutorial/cli/mol_atom_bond.rst` covers input format, atom-map reordering
(`--reorder-atoms`), the (n_atoms, n_atoms) bond-target matrix form, and constrained prediction via
`--constraints-path` / `--constraints-to-targets`; `examples/` ships `mol_atom_bond.ipynb`,
`constrained_mol_atom_bond.ipynb` and four multicomponent/reaction notebooks. The genuine footgun is
narrow and belongs in P5's cluster: per-atom targets are **positional**, so the same molecule written
by a different canonicaliser silently misaligns the target vector unless atom maps and
`--reorder-atoms` are used. Residuals worth a footnote: MAB is the newest and thinnest subsystem (5
issues, v2.2.0); **#1356** found *"bond fingerprints, whose final dimension is twice the expected
hidden dimension"*; **#1351 is OPEN** — constrained MAB training *hard-aborts the interpreter* on
Apple Silicon when Lightning auto-selects MPS (`'mps.slice' op failed: length value 12 does not fit
within the dimension size (4)`), passing on CPU and not even surfacing as a pytest failure, which is
directly relevant to this study's environment. What *is* genuinely missing is **`chemprop interpret`**:
`docs/source/tutorial/cli/interpret.rst` is an "under construction" stub with the entire v1 MCTS
documentation commented out, and there is no `interpret` entry in `chemprop/cli/` (the subcommands
are `train`, `predict`, `convert`, `fingerprint`, `hpopt`). The capability survives only as
`examples/interpreting_monte_carlo_tree_search.ipynb` and
`examples/interpreting_with_myerson_values.ipynb`. A "what v2 dropped" post is possible but is a
changelog, not a finding — unless it is merged with candidate 1's "v1 had `find_similar_mols.py` and
v2 doesn't", in which case it becomes "**three capabilities v2 lost**", which is a post.

**P11. "Answer the maintainers' own FAQ."** Discussion **#1042** is a maintainer-seeded list of **14
FAQ questions with no answers posted** — *"How does the model handle multi-task learning?"*, *"How
does chemprop handle missing data in the SMILES strings or target columns?"*, *"Can the user define
custom evaluation metrics?"*, *"Is there built-in support for cross-validation or leave-one-out
validation methods?"*, *"What are the best practices for the uses of the checkpoint file?"*, *"Are
there guidelines for choosing optimal hyperparameters…?"*, *"Can one feed an MPNN into a
gradient-boosted tree method?"* — with the only non-maintainer reply being an unrelated crash report.
It is the maintainers' own enumeration of what their docs don't cover. **Disposition:** a
documentation contribution, best offered upstream as a PR rather than a blog post. Noted so it isn't
lost. Related doc-rot worth one sentence wherever tutorials come up: **#798 `CI doesn't complain when
notebooks fail`** (zero comments) is why the shipped tutorials rot silently, and **#1393** records that
they produce near-constant predictions — *"The models in the tutorials are trained for very few
epochs and on very small amounts of data, mostly so that they run quickly on our testing suite."*

**P12. "Environment-dependent 9.5× slowdown."** Issue **#565** — the single most-reacted issue in the
repo at 3 reactions, which is itself a datum about engagement signal — reports the same 5-epoch ESOL
job taking **17 s** on a py3.9 environment and **2m42s** on a py3.11 one at `num_workers 0`. Striking,
and adjacent to post 4's measurement-hygiene section. But it is a single unreproduced report across
two unpinned environments, closed; a post needs the reproduction first, and that is
dependency-archaeology rather than an experiment. **One sentence in post 4 as a cautionary citation.**

---

## Open gaps in this brief

1. **§2's split numbers are re-implementations, not the installed packages.** Blocking for candidate 2.
   `scripts/split_audit.py` reproduces every number in §2 and documents exactly which upstream source
   each re-implementation was read from; raw output in `results/split_audit.txt`.
2. **Two things the Chemprop authors might say officially could not be retrieved.** The **SI of the
   Chemprop software paper** (Heid et al.,
   [10.1021/acs.jcim.3c01250](https://doi.org/10.1021/acs.jcim.3c01250)) contains its UQ benchmark, and
   the **Chemprop v2 paper** (Graff et al.,
   [10.1021/acs.jcim.5c02332](https://doi.org/10.1021/acs.jcim.5c02332)) has no PMC record yet; ACS
   blocks automated fetch. These are the only places the authors might state a shipped-default
   recommendation, so **get them before claiming anything about what Chemprop officially recommends**.
3. **Novelty claims rest on targeted searches, not exhaustive sweeps.** Specifically re-check
   "distance-conditional conformal" and "Mondrian applicability domain" (candidate 1) and
   "scaffold split implementation variance" (candidate 2) on Google Scholar and OpenReview, which the
   sweeps could not cover. All five literature agents exhausted the session's WebSearch budget and
   finished via Crossref / Europe PMC / NCBI / arXiv APIs — more reliable for citation accuracy, weaker
   for open-ended discovery. Absence of a paper here is **weak** evidence.
4. **Leads that could not be verified and must not be cited on this brief's authority:** the LRGB
   original's arXiv ID (commonly given as 2206.08164); arXiv:2502.03982 on temporal shift in
   pharmaceutical data; ChemBERTa-3's exact wording about MoLFormer's scaffold splits differing from
   DeepChem's (search snippet only; the ChemRxiv PDF 403'd); Meinert & Lavin's `Φ = ν + 2α` correction
   (arXiv:2104.06135) appears never to have been published beyond preprint; Cortés-Ciriano, Firth,
   Bender & Watson, JCIM 2018 ([10.1021/acs.jcim.8b00376](https://doi.org/10.1021/acs.jcim.8b00376)) —
   paywalled, cite as reported-in-review and do not quote numbers; and no clean primary record was
   found for the Platt 1999 chapter.
5. **ACS Publications returns HTTP 403 to automated fetches.** Every ACS abstract quoted here came
   from PubMed, which carries the publisher-deposited abstract verbatim. Three full texts were read
   from extracted PDF (Liu 2019 "Dissecting"; Hirschfeld 2020 preprint; Meinert 2023), so quotes from
   those include body text.
6. **A latent disagreement in the evidential literature worth knowing about.** Meinert et al. note
   that the published regulariser's `Φ = 2ν + α` arguably **miscounts the NIG pseudo-observations**,
   and that `Φ = ν + 2α` *"would actually be better motivated"* and is already used in some
   follow-ups. So implementations silently diverge on the regulariser itself, not just its coefficient.
   Chemprop uses the published `(2v + alpha)` form (`nn/metrics.py:254`), which is the right call for
   fidelity to Amini — but it means "evidential" is not one method across packages either.
