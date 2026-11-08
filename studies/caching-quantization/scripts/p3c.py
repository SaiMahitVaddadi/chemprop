import sys; sys.path.insert(0,'/private/tmp/claude-502/-Users-svaddadi-Documents-GitHub-chemprop/54c934b6-e3f8-4460-b930-d5d089b68966/scratchpad')
from common import *
from hash_mp import *
from mp_opt import MPOpt
from robust import race, show
torch.set_grad_enabled(False)
d_h=int(sys.argv[1]); depth=int(sys.argv[2]); dev=sys.argv[3]; rounds=int(sys.argv[4]) if len(sys.argv)>4 else 15
b=build_batch(); V,E,ei,rev=b['V'],b['E'],b['ei'],b['rev']; nV,nE=b['n_atom'],b['n_edge']
bk=build_buckets(ei,rev,nV); bkd=bk_to(bk,dev)
sync=torch.mps.synchronize if dev=='mps' else None
sd=MP(d_h,depth).state_dict()
if d_h==2048: sd=load_chemeleon()['state_dict']
mk=lambda C,**k:(lambda m:(m.load_state_dict(sd),m.eval().to(dev))[1])(C(d_h,depth,**k))
mr=mk(MP,mode="repeat"); mo=mk(MPOpt,mode="index_add"); mh=mk(MPHash); mho=mk(MPHashOut)
eid=torch.stack([ei[0],ei[1]]).to(dev); rv=rev.to(dev); Vd,Ed=V.to(dev),E.to(dev); Ep=E[bk['perm']].to(dev)
fns={"current(.repeat)":lambda:mr(Vd,Ed,eid,rv),
     "index_add_+select":lambda:mo(Vd,Ed,eid,rv),
     "hashimoto(cat)":lambda:mh(Vd,Ep,bkd),
     "hashimoto(out=)":lambda:mho(Vd,Ep,bkd)}
print(f"=== FULL forward {dev} d_h={d_h} depth={depth} rounds={rounds} ===")
show(race(fns,rounds=rounds,sync=sync))
R=mr(Vd,Ed,eid,rv).cpu().double()
# fp64 reference forward
W_i,W_h,W_o,bo=sd['W_i.weight'].double(),sd['W_h.weight'].double(),sd['W_o.weight'].double(),sd['W_o.bias'].double()
src,dst=ei[0],ei[1]
H0=torch.cat([V[src],E],1).double()@W_i.t(); H=torch.relu(H0)
for _ in range(1,depth):
    H=torch.relu(H0+fact_message(H,src,dst,rev,nV)@W_h.t())
Mn=torch.zeros(nV,d_h,dtype=torch.float64).index_add_(0,dst,H)
REF=torch.cat((V.double(),Mn),1)@W_o.t()+bo
print(f"  {'variant':<22}{'max|err| vs fp64':>18}{'rel':>12}")
sc=REF.abs().max()
for n,f in fns.items():
    o=f().cpu().double(); e=(o-REF).abs().max()
    print(f"  {n:<22}{e:>18.3e}{e/sc:>12.3e}")
