"""Foreground, chunked gamma ablation. Usage: gamma_run.py <depth> <cfg_idx...> <nseeds>"""
import sys, time
sys.path.insert(0,"/private/tmp/claude-502/-Users-svaddadi-Documents-GitHub-chemprop/54c934b6-e3f8-4460-b930-d5d089b68966/scratchpad")
import numpy as np
exec(open("gamma_ablation.py").read().split('print("ESOL,')[0])
CFG=[("gamma=0.0  ordinary RW (backtracking)",0.0,False),
     ("gamma=0.5  half-suppressed",0.5,False),
     ("gamma=1.0  chemprop / Hashimoto B",1.0,False),
     ("gamma learned (init 1.0)",1.0,True)]
depth=int(sys.argv[1]); idxs=[int(x) for x in sys.argv[2].split(",")]; ns=int(sys.argv[3])
for i in idxs:
    lbl,gm,lr=CFG[i]; t0=time.time()
    rs=[run(depth,gm,lr,s) for s in range(ns)]
    print(f"depth={depth} | {lbl:<36} RMSE {np.mean([r[0] for r in rs]):.4f} "
          f"sd {np.std([r[0] for r in rs]):.4f}  final_gamma {np.mean([r[1] for r in rs]):.3f} "
          f"({time.time()-t0:.0f}s, {ns} seeds)", flush=True)
