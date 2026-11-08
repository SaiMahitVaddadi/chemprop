import sys; sys.path.insert(0,'/private/tmp/claude-502/-Users-svaddadi-Documents-GitHub-chemprop/54c934b6-e3f8-4460-b930-d5d089b68966/scratchpad')
from common import *
from hash_mp import *
from robust import race, show
torch.set_grad_enabled(False)
d_h = int(sys.argv[1]); depth = int(sys.argv[2]); dev = sys.argv[3]
b = build_batch(); V,E,ei,rev = b['V'],b['E'],b['ei'],b['rev']
nV,nE = b['n_atom'], b['n_edge']
bk = build_buckets(ei, rev, nV)
sync = torch.mps.synchronize if dev=='mps' else None

# ---- ISOLATED message step (the thing P3 is about) ----
H = torch.randn(nE, d_h, device=dev)
Hp = H[bk['perm'].to(dev)].contiguous()
bkd = bk_to(bk, dev)
src,dst,rv = ei[0].to(dev), ei[1].to(dev), rev.to(dev)
def f_cur():
    idx = dst.unsqueeze(1).repeat(1,d_h)
    N = torch.zeros(nV,d_h,device=dev).scatter_reduce_(0,idx,H,reduce="sum",include_self=False)
    return N[src] - H[rv]
def f_ia():
    N = torch.zeros(nV,d_h,device=dev).index_add_(0,dst,H)
    return torch.index_select(N,0,src) - torch.index_select(H,0,rv)
def f_hash(): return hash_message(Hp, bkd)
print(f"=== ISOLATED message(), {dev}, d_h={d_h} ===")
show(race({"current(.repeat)":f_cur,"index_add_":f_ia,"hashimoto(4-bucket)":f_hash}, rounds=11, sync=sync))

# ---- FULL forward ----
ref = MP(d_h,depth,mode="repeat").eval()
if d_h==2048: ref.load_state_dict(load_chemeleon()['state_dict'])
sd = ref.state_dict()
from mp_opt import MPOpt
mr = MP(d_h,depth,mode="repeat").eval();  mr.load_state_dict(sd);  mr=mr.to(dev)
mi = MP(d_h,depth,mode="index_add").eval(); mi.load_state_dict(sd); mi=mi.to(dev)
mo = MPOpt(d_h,depth,mode="index_add").eval(); mo.load_state_dict(sd); mo=mo.to(dev)
mh = MPHash(d_h,depth).eval(); mh.load_state_dict(sd); mh=mh.to(dev)
Vd,Ed = V.to(dev),E.to(dev); Ep = E[bk['perm']].to(dev)
fns = {"current(.repeat)": lambda: mr(Vd,Ed,src.unsqueeze(0).expand(1,-1) if False else torch.stack([src,dst]),rv),
       "index_add_":       lambda: mi(Vd,Ed,torch.stack([src,dst]),rv),
       "index_add+select":  lambda: mo(Vd,Ed,torch.stack([src,dst]),rv),
       "hashimoto":        lambda: mh(Vd,Ep,bkd)}
eid = torch.stack([src,dst])
fns = {"current(.repeat)": lambda: mr(Vd,Ed,eid,rv),
       "index_add_":       lambda: mi(Vd,Ed,eid,rv),
       "index_add+select": lambda: mo(Vd,Ed,eid,rv),
       "hashimoto":        lambda: mh(Vd,Ep,bkd)}
print(f"\n=== FULL forward, {dev}, d_h={d_h} depth={depth} ===")
show(race(fns, rounds=9, sync=sync))
R = mr(Vd,Ed,eid,rv).float().cpu()
for n,f in fns.items():
    o=f().float().cpu()
    print(f"  {n:<22} max|diff vs current| {(o-R).abs().max():.3e}  bitwise {torch.equal(o,R)}")
