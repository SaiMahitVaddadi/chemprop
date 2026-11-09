# Series outline

Eleven posts, alternating between practitioner-facing and systems-facing register.

| # | post | track | state |
|---|---|---|---|
| 1 | [Your featurizer computes 102 distinct atom vectors 57,000 times](01-featurizer-redundancy.md) | practitioner | complete |
| 2 | [A D-MPNN is a non-backtracking walk, so you can cache it exactly](02-exact-caching.md) | systems | complete |
| 3 | [Why KV-cache intuitions don't transfer to graph networks](03-kv-cache-intuitions.md) | systems | complete |
| 4 | [Two one-line fixes worth 4.6×, and one that looks right and isn't](04-kernel-fixes.md) | practitioner | complete |
| 5 | [The cache tolerates 4 bits. The weights don't.](05-quantization-asymmetry.md) | practitioner | regression only |
| 6 | [The depth experiment that didn't work](06-depth-experiment.md) | both | complete |
| 7 | [Chemprop is already computing the Hashimoto matrix, in the wrong factorization](07-hashimoto.md) | systems | complete, not recommended |
| 8 | [The one feature that attacks radius](08-rwse-radius.md) | both | complete |
| 9 | [Chemprop ships eleven uncertainty estimators, and none can see a molecule](09-uncertainty-blind.md) | practitioner | complete |
| 10 | [Half of FreeSolv shares one "scaffold", and it is the empty string](10-scaffold-split.md) | practitioner | §"not one thing" unverified |
| 11 | [Mean pooling is why your GNN can't count atoms](11-mean-pooling.md) | both | complete |

## Narrative arc

1 and 2 are the setup: redundancy exists, and in the message-passing loop it is exactly
exploitable. 3 is the conceptual turn — why the obvious analogy fails, what replaces it, and the γ
sweep showing Chemprop's reverse-term coefficient is a sharp optimum. 4 is the practical payoff and
the shortest path to a real speedup. 5 is the compression result. 6 is the negative result and the
series' spine. 7 is the deepest technical finding and the one that does *not* get recommended. 8
resolves the radius-versus-depth tension the whole series circles, and closes by noting that the
architecture investigation mostly validated Chemprop rather than improving it.

## Open gaps, by post

- **2** — the cache is a prototype in a standalone harness verified bitwise-identical to Chemprop's
  math, not integrated into `chemprop train`. Hit rates come from random and scaffold splits only,
  and scaffold split is a weak shift (median Tanimoto 0.577 → 0.610, vs 0.705+ for clustering), so
  2.89× is a mild-shift case rather than a worst case. The applicability-domain side effect is a
  hypothesis, not a result. The determinism section's sampler findings use
  faithful standalone copies of Chemprop's samplers (Chemprop isn't importable in the measurement
  env), so re-verify against the real classes before filing anything upstream.
- **3** — no canonical-order model was trained, so the traversal idea is argued down (physics,
  key instability, batching) rather than measured down. γ at 0.5 and 2.0 at depth 6 unrun.
- **4** — the ORT-MLX `ScatterND` vs `ScatterElements` execution-provider claim is unverified; no
  `onnxruntime` in the measurement environment.
- **5** — every classification verdict is inconclusive; no trained MLP head; nothing above 2,200
  molecules.
- **6** — the truncation and from-scratch results disagree in sign; the decisive experiment
  (re-pretraining CheMeleon at depth 3–4) is unrun.
- **7** — `torch.compile` on the bucketed kernel is blocked by an Inductor Metal codegen bug.
- **9** — code claims verified against 2.3.1 directly; the literature claims are cited and several
  rest on single papers (the SGOOD OOD table in particular is one table in one CIKM paper).
- **10** — the scaffold statistics are measured with RDKit directly. The cross-implementation
  Jaccard numbers (0.012–0.29) come from re-implementations read out of upstream source and are
  NOT verified; `chemprop`, `astartes` and `deepchem` all need installing before that section is
  publishable.
- **11** — linear probes on frozen embeddings only; a probe failing shows information is not
  linearly available, not that it is absent. Descriptor baseline uses RDKit's ~200 native 2-D
  descriptors, not the 1,613 Mordred features CheMeleon was pretrained on, which understates
  descriptor equivalence. The frozen-CheMeleon-vs-descriptors comparison is already published
  twice (Praski arXiv:2508.06199; Ben Hicham arXiv:2604.16123); the mean-pooling and depth-
  decomposition results are not.
- **8** — RWSE is measured on one dataset from scratch; using it with CheMeleon needs re-pretraining
  because it changes `W_i`'s width.

## Things to keep when editing

Three habits carry most of the credibility and are easy to edit out by accident:

- **The noise floor comes before any delta.** Post 5 is unreadable without it, and it is the reason
  the first version of that study was wrong.
- **Corrections stay in.** Nine findings did not survive more data, four of them mine. Post 6 is
  built around two of my own reversals, and post 7 around an accounting error I made. These are
  load-bearing, not throat-clearing.
- **Measured vs reasoned stays separated.** Where a number is arithmetic rather than wall-clock,
  the posts say so — post 7's traffic claim is the clearest case, since counting `torch.cat`
  dropped it from 2.06× to 1.14×.
- **Post 6 does not say depth reduction is "dead."** It said that in an earlier draft, before the
  from-scratch numbers came in with the opposite sign. The two tests bracket the question; the
  honest word is *unresolved*.
