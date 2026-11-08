import sys; sys.path.insert(0, '.')
from common import *
import resource, gc

b = build_batch(); V,E,ei,rev = b['V'],b['E'],b['ei'],b['rev']
nV,nE = b['n_atom'], b['n_edge']

def flops(d_h, depth, d_v=72, d_e=14):
    f  = 2*nE*(d_v+d_e)*d_h          # W_i
    f += (depth-1)*2*nE*d_h*d_h      # W_h
    f += 2*nV*(d_v+d_h)*d_h          # W_o
    return f

cfgs = [("chemprop default d_h=300 depth=3", 300, 3),
        ("CheMeleon      d_h=2048 depth=6", 2048, 6)]

for label, d_h, depth in cfgs:
    base = MP(d_h, depth, mode="repeat").eval()
    if d_h == 2048:
        sd = load_chemeleon()['state_dict']
        base.load_state_dict(sd)
    F = flops(d_h, depth)
    print(f"\n########## {label} | {F/1e9:.2f} GFLOP/batch, {F/1e9/b['n_mol']*1e3:.2f} MFLOP/mol ##########")
    for dev in (['cpu','mps'] if torch.backends.mps.is_available() else ['cpu']):
        sync = torch.mps.synchronize if dev=='mps' else None
        Vd,Ed,eid,revd = V.to(dev),E.to(dev),ei.to(dev),rev.to(dev)
        print(f"-- {dev} --")
        print(f"{'mode':<14}{'ms':>9}{'GFLOP/s':>10}{'us/mol':>9}{'speedup':>9}{'max|diff| vs repeat':>22}")
        t0 = None; ref = None
        for mode in ("repeat","expand","index_add"):
            m = MP(d_h, depth, mode=mode).eval(); m.load_state_dict(base.state_dict()); m = m.to(dev)
            with torch.no_grad():
                out = m(Vd,Ed,eid,revd)
                if sync: sync()
                mn, md = timeit(lambda: m(Vd,Ed,eid,revd), sync=sync)
            if ref is None: ref, t0 = out.float().cpu(), mn
            d = (out.float().cpu()-ref).abs().max().item()
            print(f"{mode:<14}{mn*1e3:>9.1f}{F/mn/1e9:>10.1f}{mn/b['n_mol']*1e6:>9.0f}{t0/mn:>9.2f}{d:>22.3e}")
