"""P5: is the W_h GEMM compute-bound and FLOP-scaling, while message() is byte-scaling?"""
import sys; sys.path.insert(0,'/private/tmp/claude-502/-Users-svaddadi-Documents-GitHub-chemprop/54c934b6-e3f8-4460-b930-d5d089b68966/scratchpad')
from common import *
from hash_mp import *
from robust import race
torch.set_grad_enabled(False)
dev = sys.argv[1]
b = build_batch(); ei,rev = b['ei'],b['rev']; nV,nE = b['n_atom'],b['n_edge']
bk = build_buckets(ei,rev,nV); bkd = bk_to(bk,dev)
src,dst,rv = ei[0].to(dev),ei[1].to(dev),rev.to(dev)
sync = torch.mps.synchronize if dev=='mps' else None
print(f"{dev}: nE={nE} nV={nV} nnz={bk['nnz']}")
res={}
for d_h in (300,600,1024,2048):
    H = torch.randn(nE,d_h,device=dev); Hp=H[bk['perm'].to(dev)].contiguous()
    W = torch.randn(d_h,d_h,device=dev); Wt=W.t().contiguous()
    H0 = torch.randn(nE,d_h,device=dev)
    def f_gemm(): return H@Wt
    def f_gemm_fused(): return torch.addmm(H0,H,Wt).relu_()
    def f_cur():
        idx=dst.unsqueeze(1).repeat(1,d_h)
        N=torch.zeros(nV,d_h,device=dev).scatter_reduce_(0,idx,H,reduce="sum",include_self=False)
        return N[src]-H[rv]
    def f_ia():
        N=torch.zeros(nV,d_h,device=dev).index_add_(0,dst,H)
        return torch.index_select(N,0,src)-torch.index_select(H,0,rv)
    def f_hash(): return hash_message(Hp,bkd)
    r = race({"gemm":f_gemm,"gemm+bias+relu":f_gemm_fused,"msg_current":f_cur,
              "msg_index_add":f_ia,"msg_hashimoto":f_hash}, rounds=11, sync=sync)
    res[d_h]=r
    gf = 2*nE*d_h*d_h/1e9
    # message bytes (fp32), analytic
    v=4*d_h  # bytes per d_h-vector
    B_cur=(8*nE+2*nV)*v; B_ia=(4*nE+2*nV)*v; B_h=(bk['nnz']+nE)*v
    print(f"\n d_h={d_h}:  W_h GEMM {gf:.2f} GFLOP ; msg bytes cur {B_cur/1e6:.1f} MB  ia {B_ia/1e6:.1f} MB  hash {B_h/1e6:.1f} MB")
    print(f"   {'op':<18}{'min ms':>9}{'GFLOP/s':>10}{'GB/s':>9}")
    print(f"   {'gemm':<18}{r['gemm'][0]*1e3:>9.2f}{gf/r['gemm'][0]:>10.1f}{'':>9}")
    print(f"   {'gemm+bias+relu':<18}{r['gemm+bias+relu'][0]*1e3:>9.2f}{gf/r['gemm+bias+relu'][0]:>10.1f}{'':>9}")
    for nm,BB in (("msg_current",B_cur),("msg_index_add",B_ia),("msg_hashimoto",B_h)):
        print(f"   {nm:<18}{r[nm][0]*1e3:>9.2f}{'':>10}{BB/r[nm][0]/1e9:>9.1f}")
print("\n=== SCALING from d_h=300 -> 2048 (ratio 6.827) ===")
print(f" predicted if FLOP-bound (d_h^2): 46.6x ;  if byte-bound (d_h^1): 6.83x")
for nm in ("gemm","gemm+bias+relu","msg_current","msg_index_add","msg_hashimoto"):
    print(f"  {nm:<18} measured time ratio {res[2048][nm][0]/res[300][nm][0]:>7.2f}x")
