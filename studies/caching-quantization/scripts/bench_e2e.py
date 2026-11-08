import sys; sys.path.insert(0,'.')
from common import *
from mp_opt import MPOpt
import resource

mode=sys.argv[1]; d_h=int(sys.argv[2]); depth=int(sys.argv[3]); dev=sys.argv[4]
b=build_batch(); V,E,ei,rev=b['V'],b['E'],b['ei'],b['rev']
nV,nE=b['n_atom'],b['n_edge']
torch.manual_seed(0)
cls = MP if mode in ("repeat","expand","index_add") else MPOpt
m=cls(d_h,depth,mode=mode).eval()
if d_h==2048: m.load_state_dict(load_chemeleon()['state_dict'])
if mode=="fused_pad":
    maxdeg=int(torch.bincount(ei[1],minlength=nV).max())
    padidx=torch.full((nV,maxdeg),nE,dtype=torch.long); cnt=torch.zeros(nV,dtype=torch.long)
    for f,d in enumerate(ei[1].tolist()):
        padidx[d,cnt[d]]=f; cnt[d]+=1
    m.set_pad(padidx.to(dev))
m=m.to(dev)
Vd,Ed,eid,revd=V.to(dev),E.to(dev),ei.to(dev),rev.to(dev)
sync=torch.mps.synchronize if dev=='mps' else None
if dev=='mps': torch.mps.empty_cache()
with torch.no_grad():
    out=m(Vd,Ed,eid,revd)
    if sync: sync()
    mn,md=timeit(lambda: m(Vd,Ed,eid,revd),sync=sync)
rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20
mpsmem=torch.mps.driver_allocated_memory()/2**20 if dev=='mps' else 0
F=2*nE*86*d_h+(depth-1)*2*nE*d_h*d_h+2*nV*(72+d_h)*d_h
import numpy as np
np.save(f"/tmp/out_{mode}_{d_h}_{dev}.npy", out.float().cpu().numpy())
print(f"{mode:<12}{dev:<5}{mn*1e3:>9.1f}{md*1e3:>9.1f}{F/mn/1e9:>9.1f}{rss:>10.0f}{mpsmem:>10.0f}")
