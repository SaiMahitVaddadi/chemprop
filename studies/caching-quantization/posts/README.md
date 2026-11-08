# Series outline

Seven posts, alternating between practitioner-facing and systems-facing register.

| # | post | track | state |
|---|---|---|---|
| 1 | [Your featurizer computes 102 distinct atom vectors 57,000 times](01-featurizer-redundancy.md) | practitioner | complete |
| 2 | [A D-MPNN is a non-backtracking walk, so you can cache it exactly](02-exact-caching.md) | systems | complete |
| 3 | [Why KV-cache intuitions don't transfer to graph networks](03-kv-cache-intuitions.md) | systems | complete |
| 4 | [Two one-line fixes worth 4.6×, and one that looks right and isn't](04-kernel-fixes.md) | practitioner | complete |
| 5 | [The cache tolerates 4 bits. The weights don't.](05-quantization-asymmetry.md) | practitioner | regression only |
| 6 | [The depth experiment that didn't work](06-depth-experiment.md) | both | complete |
| 7 | [Chemprop is already computing the Hashimoto matrix, in the wrong factorization](07-hashimoto.md) | systems | complete, not recommended |

## Narrative arc

1 and 2 are the setup: redundancy exists, and in the message-passing loop it is exactly
exploitable. 3 is the conceptual turn — why the obvious analogy fails and what replaces it. 4 is
the practical payoff and the shortest path to a real speedup. 5 is the compression result. 6 is the
negative result and the series' spine. 7 is the deepest technical finding and the one that does
*not* get recommended, which is the honest ending.

## Open gaps, by post

- **2** — the cache is a prototype in a standalone harness verified bitwise-identical to Chemprop's
  math, not integrated into `chemprop train`.
- **3** — the traversal direction (canonical ordered message passing, learned γ) is unexecuted.
  `experiments/gamma_ablation.py` is written and unrun.
- **4** — the ORT-MLX `ScatterND` vs `ScatterElements` execution-provider claim is unverified; no
  `onnxruntime` in the measurement environment.
- **5** — every classification verdict is inconclusive; no trained MLP head; nothing above 2,200
  molecules.
- **7** — `torch.compile` on the bucketed kernel is blocked by an Inductor Metal codegen bug.

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
