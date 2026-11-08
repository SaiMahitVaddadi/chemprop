"""Micro-benchmark: the scatter index construction + reduction, isolated."""
import sys; sys.path.insert(0, '.')
from common import *

b = build_batch()
V, E, ei, rev = b['V'], b['E'], b['ei'], b['rev']
nV, nE = b['n_atom'], b['n_edge']
print(f"batch: {b['n_mol']} mols, {nV} atoms, {nE} directed bonds")
print(f"degree stats: mean in-degree = {nE/nV:.3f}")

def run(dev, dtype):
    eid = ei.to(dev); nv = nV
    dst = eid[1]
    out = {}
    sync = torch.mps.synchronize if dev == 'mps' else None
    for d_h in (300, 2048):
        H = torch.randn(nE, d_h, dtype=dtype, device=dev)
        # 1. index build only
        f_rep_idx = lambda: dst.unsqueeze(1).repeat(1, d_h)
        f_exp_idx = lambda: dst.unsqueeze(1).expand(-1, d_h)
        # 2. full scatter
        def f_repeat():
            idx = dst.unsqueeze(1).repeat(1, d_h)
            return torch.zeros(nv, d_h, dtype=dtype, device=dev).scatter_reduce_(
                0, idx, H, reduce="sum", include_self=False)
        def f_expand():
            idx = dst.unsqueeze(1).expand(-1, d_h)
            return torch.zeros(nv, d_h, dtype=dtype, device=dev).scatter_reduce_(
                0, idx, H, reduce="sum", include_self=False)
        def f_expand_incl():
            idx = dst.unsqueeze(1).expand(-1, d_h)
            return torch.zeros(nv, d_h, dtype=dtype, device=dev).scatter_reduce_(
                0, idx, H, reduce="sum", include_self=True)
        def f_index_add():
            return torch.zeros(nv, d_h, dtype=dtype, device=dev).index_add_(0, dst, H)
        def f_scatter_add():
            idx = dst.unsqueeze(1).expand(-1, d_h)
            return torch.zeros(nv, d_h, dtype=dtype, device=dev).scatter_add_(0, idx, H)
        res = {}
        for nm, fn in [("idx .repeat", f_rep_idx), ("idx .expand", f_exp_idx),
                       ("scatter_reduce(.repeat)", f_repeat),
                       ("scatter_reduce(.expand)", f_expand),
                       ("scatter_reduce(.expand,incl_self)", f_expand_incl),
                       ("scatter_add_(.expand)", f_scatter_add),
                       ("index_add_", f_index_add)]:
            mn, md = timeit(fn, sync=sync)
            res[nm] = mn * 1e3
        # equivalence
        a = f_repeat(); c = f_expand(); e2 = f_index_add(); f2 = f_scatter_add()
        eq = (torch.equal(a, c), torch.equal(a, e2), torch.equal(a, f2))
        out[d_h] = (res, eq)
    return out

for dev in (['cpu', 'mps'] if torch.backends.mps.is_available() else ['cpu']):
    for dtype in ([torch.float32] if dev == 'cpu' else [torch.float32]):
        o = run(dev, dtype)
        print(f"\n=== {dev} / {dtype} ===")
        names = list(o[300][0].keys())
        print(f"{'op':<36}{'d_h=300 ms':>13}{'d_h=2048 ms':>14}")
        for n in names:
            print(f"{n:<36}{o[300][0][n]:>13.3f}{o[2048][0][n]:>14.3f}")
        for d_h in (300, 2048):
            print(f"  d_h={d_h} bitwise-equal(repeat vs expand, vs index_add_, vs scatter_add_): {o[d_h][1]}")
