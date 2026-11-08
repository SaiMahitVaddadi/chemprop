"""Quantify the torch.cat traffic the brief's accounting omits."""
import sys; sys.path.insert(0,'/private/tmp/claude-502/-Users-svaddadi-Documents-GitHub-chemprop/54c934b6-e3f8-4460-b930-d5d089b68966/scratchpad')
from common import *
from hash_mp import *
from robust import race, show
torch.set_grad_enabled(False)
dev=sys.argv[1]; d_h=int(sys.argv[2])
b=build_batch(); V,E,ei,rev=b['V'],b['E'],b['ei'],b['rev']; nV,nE=b['n_atom'],b['n_edge']
bk=build_buckets(ei,rev,nV); bkd=bk_to(bk,dev); sync=torch.mps.synchronize if dev=='mps' else None
src,dst,rv=ei[0].to(dev),ei[1].to(dev),rev.to(dev)
H=torch.randn(nE,d_h,device=dev); Hp=H[bk['perm'].to(dev)].contiguous()
M=torch.empty(nE,d_h,device=dev); SC=torch.empty(nE,d_h,device=dev)
Vd=V.to(dev); Ed=E.to(dev); Mn=torch.randn(nV,d_h,device=dev)
parts=[Hp[:bk['counts'][0][1]].clone()]+[Hp[:c].clone() for r,c in bk['counts'][1:] if c]
fns={"hash: cat(parts)":lambda:hash_message(Hp,bkd),
     "hash: view.copy_/add_":lambda:hash_message_copy(Hp,bkd,M),
     "bare torch.cat(4 parts)":lambda:torch.cat(parts,0),
     "initialize cat([V[src],E])":lambda:torch.cat([torch.index_select(Vd,0,src),Ed],1),
     "finalize cat([V,M_n])":lambda:torch.cat((Vd,Mn),1)}
if dev=='cpu': fns["hash: index_select(out=)"]=lambda:hash_message_out(Hp,bkd,M,SC)
print(f"=== {dev} d_h={d_h} ; nnz={bk['nnz']} 2m={nE} n={nV} ===")
show(race(fns,rounds=13,sync=sync))
v=4*d_h
print(f"\n  one (2m x d_h) fp32 tensor = {nE*v/1e6:.1f} MB ; one (n x d_h) = {nV*v/1e6:.1f} MB")
print(f"  modelled message traffic (d_h-vectors): hashimoto-ideal {bk['nnz']+nE} | "
      f"+cat {bk['nnz']+3*nE} | index_add {4*nE+2*nV} | current {8*nE+2*nV}")
print(f"  => cat inflates the Hashimoto message step {(bk['nnz']+3*nE)/(bk['nnz']+nE):.2f}x in traffic;")
print(f"     Hashimoto-vs-index_add traffic ratio falls from {(4*nE+2*nV)/(bk['nnz']+nE):.2f}x "
      f"to {(4*nE+2*nV)/(bk['nnz']+3*nE):.2f}x once the cat is counted.")
