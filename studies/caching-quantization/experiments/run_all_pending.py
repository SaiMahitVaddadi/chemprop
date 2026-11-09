"""Run every pending experiment in one go, with skip-if-done and a dry-run.

    python experiments/run_all_pending.py --dry-run     # show the plan + cost
    python experiments/run_all_pending.py               # run everything missing
    python experiments/run_all_pending.py --quick       # smoke test, minutes
    python experiments/run_all_pending.py --only hashimoto cat
    python experiments/run_all_pending.py --force       # re-run completed stages

Each stage writes one results JSON; a stage whose output already exists is skipped
unless --force.  Stages run sequentially in the FOREGROUND - no background jobs, so
the whole thing can be launched once and watched.
"""
import argparse, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"

# name -> (script, output file, rough cost, what it unblocks)
STAGES = [
    ("hashimoto", "hashimoto_bench.py", "results_hashimoto.json", "~5 min",
     "post 7 - the only headline claim with no wall-clock behind it"),
    ("cat", "cat_cost.py", "results_cat_cost.json", "~2 min",
     "post 4 - corrects the brief's traffic accounting"),
    ("bmm", "bmm_bucketed.py", "results_bmm_bucketed.json", "~10 min",
     "post 4/7 - the MPS escape hatch (P8/P9)"),
    ("determinism", "determinism_probe.py", "results_determinism.json", "~3 min",
     "post 2 - separates WL-key imprecision from scatter nondeterminism"),
    ("determinism-data", "determinism_data.py", "results_determinism_data.json", "~2 min",
     "post 2 - dataloader/sampler/scaler determinism (the axis that moves seed variance)"),
    ("residual-flow", "residual_flow.py", "results_residual_flow.json", "~15 min",
     "new post - does source re-injection prevent information loss (flow analysis)"),
    ("bond-agg", "bond_aggregation.py", "results_bond_aggregation.json", "~5 min",
     "post 11 - bond-level readout family; pair with rep2 probes to quantify the loss"),
    ("pooling", "pooling_probes.py", "results_pooling_probes.json", "~8 min",
     "post 11 - collision test and moment sufficiency"),
    ("anchor", "anchor_study.py", "results_anchor_study.json", "~3 min",
     "post 3 - is there a message-passing starting point, and is anchoring stable"),
    ("gamma", "gamma_ablation.py", "results_gamma_ablation.json", "~40 min",
     "novel: how much is non-backtracking worth"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args()

    plan = [s for s in STAGES if a.only is None or s[0] in a.only]
    print(f"{'stage':<16}{'script':<24}{'cost':<9}{'status':<10}unblocks")
    todo = []
    for name, script, out, cost, why in plan:
        done = (RESULTS / out).exists()
        missing_script = not (HERE / script).exists()
        status = "MISSING" if missing_script else ("done" if done and not a.force else "run")
        if status == "run":
            todo.append((name, script, out))
        print(f"{name:<16}{script:<24}{cost:<9}{status:<10}{why}")

    if a.dry_run:
        print(f"\n{len(todo)} stage(s) would run. Scripts marked MISSING need writing first.")
        return

    for name, script, out in todo:
        cmd = [sys.executable, str(HERE / script)] + (["--quick"] if a.quick else [])
        print(f"\n{'='*70}\n[{name}] {' '.join(cmd)}\n{'='*70}")
        t0 = time.perf_counter()
        rc = subprocess.call(cmd, cwd=str(HERE.parent))
        print(f"[{name}] exit {rc} in {time.perf_counter()-t0:.0f}s")
        if rc != 0:
            print(f"[{name}] FAILED - continuing with remaining stages")
    print("\nall requested stages attempted")


if __name__ == "__main__":
    main()
