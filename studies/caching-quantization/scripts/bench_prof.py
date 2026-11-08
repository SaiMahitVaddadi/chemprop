import sys; sys.path.insert(0,'.')
from common import *
from torch.profiler import profile, ProfilerActivity, record_function

b=build_batch(); V,E,ei,rev=b['V'],b['E'],b['ei'],b['rev']
nV,nE=b['n_atom'],b['n_edge']

def prof(m, args, dev, tag):
    acts=[ProfilerActivity.CPU]
    with torch.no_grad():
        for _ in range(3): m(*args)
        if dev=='mps': torch.mps.synchronize()
        with profile(activities=acts, profile_memory=True, record_shapes=False) as p:
            for _ in range(3): m(*args)
            if dev=='mps': torch.mps.synchronize()
    ka = p.key_averages()
    print(f"\n### {tag} ({dev}) ###")
    sk = 'self_cpu_time_total'
    rows=sorted(ka, key=lambda e: -getattr(e,sk,0))
    tot=sum(max(getattr(e,sk,0),0) for e in rows)
    print(f"{'op':<34}{'self ms':>9}{'%':>7}{'alloc MB':>11}")
    for e in rows[:11]:
        t=getattr(e,sk,0)/1e3/3
        if t<=0.002: continue
        print(f"{e.key:<34}{t:>9.2f}{getattr(e,sk,0)/max(tot,1)*100:>7.1f}{e.cpu_memory_usage/3/2**20:>11.1f}")
    print(f"{'TOTAL':<34}{tot/1e3/3:>9.2f}")

for d_h,depth,lab in [(300,3,'d_h=300 depth=3'),(2048,6,'d_h=2048 depth=6')]:
    base=MP(d_h,depth).eval()
    if d_h==2048: base.load_state_dict(load_chemeleon()['state_dict'])
    for dev in ['cpu']:
        args=(V.to(dev),E.to(dev),ei.to(dev),rev.to(dev))
        for mode in ('repeat','index_add'):
            m=MP(d_h,depth,mode=mode).eval(); m.load_state_dict(base.state_dict()); m=m.to(dev)
            prof(m,args,dev,f"{lab} / {mode}")
