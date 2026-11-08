import sys, time
sys.path.insert(0, "/private/tmp/claude-502/-Users-svaddadi-Documents-GitHub-chemprop/54c934b6-e3f8-4460-b930-d5d089b68966/scratchpad")
import numpy as np, torch
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")
from cp_loader import SimpleMoleculeMolGraphFeaturizer
torch.manual_seed(0)

smis = [l.split(",")[0].strip() for l in open("tests/data/regression.csv").readlines()[1:]]
mols = [m for m in (Chem.MolFromSmiles(s) for s in smis) if m is not None]
N = len(mols); feat = SimpleMoleculeMolGraphFeaturizer()
mgs = [feat(m) for m in mols]
V = torch.from_numpy(np.concatenate([g.V for g in mgs])).float()
E = torch.from_numpy(np.concatenate([g.E for g in mgs])).float()
oa = oe = 0; eis=[]; revs=[]
for g in mgs:
    eis.append(g.edge_index+oa); revs.append(g.rev_edge_index+oe)
    oa += g.V.shape[0]; oe += g.edge_index.shape[1]
ei = torch.from_numpy(np.hstack(eis)).long(); rev = torch.from_numpy(np.concatenate(revs)).long()

def timeit(fn, reps=3):
    b=1e9
    for _ in range(reps):
        t=time.perf_counter(); fn(); b=min(b,time.perf_counter()-t)
    return b

class MP(torch.nn.Module):
    def __init__(s, d_h, depth):
        super().__init__()
        s.W_i=torch.nn.Linear(86,d_h,bias=False); s.W_h=torch.nn.Linear(d_h,d_h,bias=False)
        s.W_o=torch.nn.Linear(72+d_h,d_h); s.depth=depth
    def forward(s, V, E, ei, rev):
        H_0 = s.W_i(torch.cat([V[ei[0]], E], dim=1)); H = torch.relu(H_0)
        for _ in range(1, s.depth):
            idx = ei[1].unsqueeze(1).expand(-1,H.shape[1])
            M = torch.zeros(V.shape[0],H.shape[1],dtype=H.dtype).scatter_reduce_(
                0,idx,H,reduce="sum",include_self=False)[ei[0]] - H[rev]
            H = torch.relu(H_0 + s.W_h(M))
        idx = ei[1].unsqueeze(1).expand(-1,H.shape[1])
        M = torch.zeros(V.shape[0],H.shape[1],dtype=H.dtype).scatter_reduce_(
            0,idx,H,reduce="sum",include_self=False)
        return s.W_o(torch.cat((V,M),dim=1))

print(f"{N} molecules | atoms={V.shape[0]} directed bonds={ei.shape[1]}")
print("\n=== cost of message passing vs. cost of the cache lookup that would replace it ===")
print(f"{'config':<34}{'ms/batch':>10}{'us/mol':>9}{'GFLOP/mol':>11}")
WL_US, GATHER_US_PER_LAYER = 94.0, 15.4   # measured in bench.py
for name,d_h,depth in [("chemprop default d_h=300 depth=3",300,3),
                       ("chemprop d_h=600 depth=5",600,5),
                       ("CheMeleon d_h=2048 depth=6",2048,6)]:
    m = MP(d_h,depth).eval()
    with torch.no_grad():
        m(V,E,ei,rev); t = timeit(lambda: m(V,E,ei,rev))
    fl = 2*(ei.shape[1]*86*d_h + (depth-1)*ei.shape[1]*d_h*d_h + V.shape[0]*(72+d_h)*d_h)
    overhead = WL_US + GATHER_US_PER_LAYER*depth
    saved = 0.75*t/N*1e6
    print(f"{name:<34}{t*1e3:>10.1f}{t/N*1e6:>9.1f}{fl/N/1e9:>11.3f}"
          f"   cache: pay {overhead:.0f}us, save ~{saved:.0f}us -> "
          f"{'WIN %.1fx'%(t/N*1e6/(t/N*1e6-saved+overhead)) if saved>overhead else 'LOSS'}")

# ---------------- quantization ----------------
print("\n=== quantization of the encoder (CheMeleon dims, depth 6) ===")
m = MP(2048,6).eval()
with torch.no_grad():
    ref = m(V,E,ei,rev)
    t32 = timeit(lambda: m(V,E,ei,rev))
print(f"{'mode':<30}{'ms':>8}{'speedup':>9}{'rel L2 err':>12}{'max abs':>10}")
print(f"{'fp32 baseline':<30}{t32*1e3:>8.1f}{1.0:>9.2f}{0.0:>12.1e}{0.0:>10.1e}")

qm = torch.ao.quantization.quantize_dynamic(m, {torch.nn.Linear}, dtype=torch.qint8).eval()
with torch.no_grad():
    o = qm(V,E,ei,rev); tq = timeit(lambda: qm(V,E,ei,rev))
err = (o-ref).norm()/ref.norm()
print(f"{'int8 dynamic (all Linear)':<30}{tq*1e3:>8.1f}{t32/tq:>9.2f}{err:>12.2e}{(o-ref).abs().max():>10.2e}")

for dt,nm in [(torch.bfloat16,"bf16"),(torch.float16,"fp16")]:
    md = MP(2048,6).eval(); md.load_state_dict(m.state_dict()); md = md.to(dt)
    with torch.no_grad():
        o = md(V.to(dt),E.to(dt),ei,rev); td = timeit(lambda: md(V.to(dt),E.to(dt),ei,rev))
    o = o.float(); err=(o-ref).norm()/ref.norm()
    print(f"{nm+' weights+activations':<30}{td*1e3:>8.1f}{t32/td:>9.2f}{err:>12.2e}{(o-ref).abs().max():>10.2e}")

# ---------------- quantizing the CACHE instead of the compute ----------------
print("\n=== quantizing the cached vectors (error enters once, not compounded) ===")
with torch.no_grad():
    H_0 = m.W_i(torch.cat([V[ei[0]],E],dim=1)); H=torch.relu(H_0)
    for _ in range(1,6):
        idx=ei[1].unsqueeze(1).expand(-1,H.shape[1])
        Msg=torch.zeros(V.shape[0],H.shape[1]).scatter_reduce_(0,idx,H,reduce="sum",include_self=False)[ei[0]]-H[rev]
        H=torch.relu(H_0+m.W_h(Msg))
    idx=ei[1].unsqueeze(1).expand(-1,H.shape[1])
    M=torch.zeros(V.shape[0],H.shape[1]).scatter_reduce_(0,idx,H,reduce="sum",include_self=False)
print(f"{'cache dtype':<22}{'bytes/env':>11}{'1M envs':>10}{'rel L2 err on output':>22}")
for nm,bits in [("fp32",32),("fp16",16),("int8 per-row",8),("int4 per-row",4)]:
    if nm=="fp32": Mq=M.clone()
    elif nm=="fp16": Mq=M.half().float()
    else:
        n=2**(bits-1)-1
        sc=M.abs().amax(1,keepdim=True)/n
        Mq=torch.round(M/sc).clamp(-n-1,n)*sc
    with torch.no_grad():
        o=m.W_o(torch.cat((V,Mq),dim=1))
    print(f"{nm:<22}{2048*bits//8:>11}{1e6*2048*bits/8/2**30:>9.1f}G"
          f"{((o-ref).norm()/ref.norm()):>22.2e}")
