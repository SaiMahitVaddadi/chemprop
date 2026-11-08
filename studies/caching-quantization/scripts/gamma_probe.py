import sys,time
sys.path.insert(0,"/private/tmp/claude-502/-Users-svaddadi-Documents-GitHub-chemprop/54c934b6-e3f8-4460-b930-d5d089b68966/scratchpad")
t0=time.time()
exec(open("gamma_ablation.py").read().split('print("ESOL,')[0])
print(f"setup {time.time()-t0:.1f}s")
t0=time.time(); r=run(3,1.0,False,0); print(f"one depth-3 5-fold run: {time.time()-t0:.1f}s -> RMSE {r[0]:.4f}")
t0=time.time(); r=run(6,1.0,False,0); print(f"one depth-6 5-fold run: {time.time()-t0:.1f}s -> RMSE {r[0]:.4f}")
