# Chemprop ships eleven uncertainty estimators, and none of them can see a molecule

*Part 9 of a series on Chemprop internals.*

Chemprop has an unusually complete uncertainty stack: **11 estimators, 10 calibrators, 9
evaluators.** MVE, deep ensembles, three flavours of evidential regression, MC dropout, quantile
regression, two Dirichlet variants. You can calibrate with z-scaling, Zelikman intervals,
MVE-weighting, conformal regression, Platt scaling or isotonic regression, then evaluate with NLL,
miscalibration area, ENCE, Spearman or conformal coverage.

Then you go looking for the thing the literature actually recommends, and it isn't there — and the
reason it isn't there is architectural rather than an oversight.

## The interface

Every calibrator in `chemprop/uncertainty/calibrator.py` has one of two signatures:

```python
def fit(self, preds: Tensor, uncs: Tensor, targets: Tensor, mask: Tensor) -> Self
def fit(self, uncs: Tensor, targets: Tensor, mask: Tensor) -> Self
def apply(self, uncs: Tensor) -> Tensor
```

Predictions, uncertainties, targets, a mask. **No molecule. No dataset. No training set.** The
evaluators are the same. And it isn't that the chemistry is reached indirectly either:

- `rdkit` is imported **nowhere** in `chemprop/uncertainty/`
- `DataStructs` — RDKit's fingerprint-similarity module — is imported **nowhere in `chemprop/` at
  all**

So the entire UQ stack is a pure post-processor over numeric columns. It can rescale a variance; it
cannot ask *how far is this molecule from anything I was trained on.*

That matters because of what four independent studies converged on. Hirschfeld et al. found a
random forest on MPNN features beat the network's own UQ heads. Tran et al. got the same from a
convolution-fed Gaussian process. Wollschläger et al. (ICML 2023) formalised it as a GP on GNN
features. Parrondo-Pizarro et al. (JCIM 2026) landed on error models over learned features. Four
groups, four methods, one shape: **a learned secondary model over the representation beats the
uncertainty head.** Chemprop ships none of it, and the interface above is why — a secondary model
needs the features and the training set, and `fit(preds, uncs, targets, mask)` has neither.

There's also no Venn-ABERS calibrator and no CRPS evaluator, ever, in any version.

## Three things I verified in the source

**1. The `spearman` evaluator is mathematically invariant to two of the four regression
calibrators.**

`ZScalingCalibrator.apply` and `ZelikmanCalibrator.apply` are both, verbatim:

```python
return uncs * self.scalings**2
```

A per-task multiplication by a non-negative scalar. And `SpearmanEvaluator` computes a *rank*
correlation between uncertainties and absolute errors. Rank correlation is invariant under any
strictly increasing transform, so it cannot move. Measured, over four orders of magnitude of
scaling:

| scalings² | Spearman | Δ from uncalibrated |
|---|---|---|
| 0.01 | 0.968254630085 | +0.000e+00 |
| 1.00 | 0.968254630085 | +0.000e+00 |
| 3.70 | 0.968254630085 | +0.000e+00 |
| 10000.00 | 0.968254630085 | +0.000e+00 |

Identical to twelve decimal places. So `--calibration-method zscaling --evaluation-methods
spearman` is a no-op, silently. It's not a bug — both halves are correct in isolation — but the CLI
will happily let you spend a calibration set proving nothing, and nothing warns you.

**2. The ENCE evaluator mishandles masked labels, and another evaluator in the same file gets it
right.**

`ExpectedNormalizedErrorEvaluator.evaluate` begins:

```python
masked_preds = preds * mask
masked_targets = targets * mask
masked_uncs = uncs * mask
errors = torch.abs(masked_preds - masked_targets)
sort_idx = torch.argsort(masked_uncs, dim=0)
```

Multiplying by a boolean mask **zeroes masked entries rather than removing them**:

```
uncs        = [5. 3. 9. 1.]
mask        = [T  F  T  F]
uncs * mask = [5. 0. 9. 0.]      <- masked rows become 0.0
argsort     = [1 3 0 2]          <- and sort into the lowest bins
```

ENCE then chunks the sorted array into 100 bins and computes `|RMV − RMSE| / RMV` per bin. Every
masked row lands in the low-uncertainty bins carrying a fake zero error, so the bottom bins are
filled with absent data, `RMV → 0`, and the normalisation divides by approximately zero.

For a dense single-task set this never fires. For **sparse multi-task data it dominates** — Tox21
and SIDER are mostly missing labels, which is the regime ENCE is most useful in. And the
`SpearmanEvaluator` eleven lines away does the masking correctly, with boolean indexing:

```python
uncs_j = uncs[:, j][mask_j]
```

Two evaluators, one file, two different masking conventions, one of them wrong.

**3. Ensemble variance is biased low, at five call sites.**

```python
vars = torch.var(stacked_preds, dim=0, correction=0)
```

`correction=0` is the population variance, dividing by *M* rather than *M − 1*. For a 5-model
ensemble that understates the variance by 20%; for the 2-model ensembles people actually run under
compute pressure, by 50%. Ensemble variance is already known to under-estimate — Kahle & Zipoli
measured deep ensembles underestimating uncertainty "by an order of magnitude" far from data,
because members "deviate in a very similar fashion, most probably due to common biases" — so a
further systematic factor in the same direction is the wrong direction to be sloppy in.

## What the literature says the heads can't do anyway

Even fixed, the architectural problem stands, because the failure modes are not calibration
failures.

**MVE's real pathology is a training failure, not a scaling one.** Seitzer et al.'s "rich get
richer" effect: a mean-variance network down-weights high-error points through the variance term,
so it stops fitting exactly the regions it is uncertain about. The standard mitigations are β-NLL,
a variance floor, or warming up the mean head before the variance head. Chemprop has **none of the
three**. And Hirschfeld's datum is the one to lead with: MPNN MVE's **most confident 25% of
Lipophilicity predictions had a higher RMSE than the entire test set.** The ranking was worse than
useless — it was inverted.

**The classic OOD scores don't work on molecules.** MSP, Energy and ODIN score 45–57 AUROC on
BACE/BBBP/HIV while the same backbone reaches 79–97% in-distribution accuracy (SGOOD, CIKM 2024).
On BACE all three are *below* 50 — anti-correlated. There's a theorem for it: ReLU networks produce
almost always high-confidence predictions far from the training data (Hein et al., CVPR 2019), and
a D-MPNN is a ReLU network.

**And conformal prediction doesn't answer the question people ask it.** Marginal coverage is
explicitly a population-average guarantee. Barber, Candès, Ramdas & Tibshirani (2021) proved exact
distribution-free *conditional* coverage is **impossible** without distributional assumptions. So
"is this particular prediction trustworthy" is not a question marginal conformal answers, and no
assumption-free method can. Empirically it shows up exactly as the theory predicts: under temporal
shift, Morger et al. measured aggregated conformal prediction at a nominal 0.80 validity delivering
**0.81 in cross-validation and 0.56 on the temporal holdout** — 0.50 on the active class, and 0.47
on liver toxicity with **0.00** active-class validity. A descriptor-space applicability check would
have raised no alarm; their UMAP plots of the two sets "seem to cover a similar area."

One smaller thing with real blast radius: the uncertainty column's **units differ by estimator**.
Some return variance, some return a half-interval. Nothing validates which you have, and nothing
stops you feeding a half-interval to a calibrator expecting variance. It was raised in Discussion
#1320 and never answered.

## What I'd actually build

The interface is the fix, not the estimators. `fit(preds, uncs, targets, mask)` needs to become
something that also receives the featurized training set and the model's representation, because
every method with a positive track record needs one or both.

Concretely, and in order of how cheap it is:

1. **An input-space distance baseline.** ECFP4 nearest-neighbour Tanimoto to the training set as an
   uncertainty column. Remarkably, **none of the four graph-OOD benchmarks — GOOD-D, UB-GOLD,
   PGR-MOOD, SGOOD — contains a fingerprint or Tanimoto baseline at all.** The obvious control has
   never been run, and MOOD's authors concluded "the input space distance is a good surrogate for
   the internal model representation distance," i.e. the latent space adds nothing over it.
2. **A secondary error model** over the frozen representation — the thing four papers agree on.
3. **Mondrian (class- or distance-conditional) conformal prediction**, since Vovk gave the recipe
   for conditional validity in 2013 and the stack only implements the marginal version.

Two honest caveats on my own enthusiasm. "Distance beats learned UQ" is **not** the field's
consensus — Tetko, Sushko, Sheridan and Toplak all find ensemble variance beating distance on
diverse data, with distance winning only in the far tail. The consensus is *combine them in an
error model*, which is recommendation 2, not 1. And a distance-based signal **cannot** flag activity
cliffs by construction: a cliff compound is near the training set and still mispredicted, and cliff
prevalence runs 7–52% depending on target (van Tilborg et al., JCIM 2022). Any honest evaluation has
to split errors into "far from training" versus "near but cliff-like," which nobody appears to have
reported.

## Verification notes

The code claims here — 11 estimators, the calibrator signatures, the absent RDKit import, the
Spearman invariance to twelve decimals, the ENCE mask multiply, `correction=0` at five sites — I
read and ran against Chemprop 2.3.1 myself. The literature claims are cited above and several
come from single papers; the SGOOD table in particular is one table in one CIKM paper, and is
cheap to re-run if you want to lean on it.
