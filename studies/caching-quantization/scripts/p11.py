"""P11: torch.compile recompile count vs distinct (n,2m) pairs; eager-vs-compiled numerics."""
import sys, os, io, re, contextlib
sys.path.insert(0,'/private/tmp/claude-502/-Users-svaddadi-Documents-GitHub-chemprop/54c934b6-e3f8-4460-b930-d5d089b68966/scratchpad')
from common import *
from hash_mp import *
from mp_opt import MPOpt
from robust import race, show
import torch._dynamo as dyn
torch.set_grad_enabled(False)
dev=sys.argv[1]; d_h=int(sys.argv[2]); depth=int(sys.argv[3])
mode=sys.argv[4] if len(sys.argv)>4 else "default"
# ---- several real, variable-shaped batches ----
sizes=[40,60,80,40]
B=[]
for s in sizes:
    b=build_batch(n=s); B.append(b)
shapes=[(b['n_atom'],b['n_edge']) for b in B]
uniq=sorted(set(shapes))
print(f"{len(B)} real batches, shapes (n,2m)={shapes}")
print(f"distinct (n,2m) pairs = {len(uniq)}  <- P11 predicts this equals the recompile count")
sd=MP(d_h,depth).state_dict()
m=MP(d_h,depth,mode="index_add").eval(); m.load_state_dict(sd); m=m.to(dev)
ref=[m(b['V'].to(dev),b['E'].to(dev),b['ei'].to(dev),b['rev'].to(dev)).cpu() for b in B]

kw=dict(default={}, reduce_overhead=dict(mode="reduce-overhead"), dynamic=dict(dynamic=True))[mode]
dyn.reset(); dyn.utils.counters.clear()
cm=torch.compile(m, **kw)
import logging
torch._logging.set_logs(recompiles=True)
log=io.StringIO(); h=logging.StreamHandler(log); h.setLevel(logging.DEBUG)
lg=logging.getLogger("torch._dynamo.guards"); lg.addHandler(h); lg.propagate=False
outs=[]
for b in B:
    outs.append(cm(b['V'].to(dev),b['E'].to(dev),b['ei'].to(dev),b['rev'].to(dev)).cpu())
txt=log.getvalue()
nrec=len(re.findall(r"Recompiling function", txt))
cnt=dyn.utils.counters
ug=cnt.get('stats',{}).get('unique_graphs',None)
print(f"\nmode={mode}: 'Recompiling function' log events = {nrec}; unique_graphs counter = {ug}; stats={dict(cnt.get('stats',{}))}")
print(f"  (first compile is not a 'recompile', so total graph compiles = {nrec+1})")
print(f"{'batch':>6}{'max|compiled-eager|':>22}{'rel':>12}")
bad=False
for i,(o,r) in enumerate(zip(outs,ref)):
    e=float((o-r).abs().max()); rel=e/float(r.abs().max())
    if rel>1e-4: bad=True
    print(f"{i:>6}{e:>22.3e}{rel:>12.3e}")
print("  NUMERICS: "+("MISMATCH vs eager -- speedup is void" if bad else "OK (matches eager to fp32 rounding)"))
# timing on one fixed shape
b=B[0]; a=(b['V'].to(dev),b['E'].to(dev),b['ei'].to(dev),b['rev'].to(dev))
sync=torch.mps.synchronize if dev=='mps' else None
show(race({"eager":lambda:m(*a), f"compile/{mode}":lambda:cm(*a)},rounds=9,sync=sync))
