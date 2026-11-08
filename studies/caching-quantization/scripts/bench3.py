import sys, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/private/tmp/claude-502/-Users-svaddadi-Documents-GitHub-chemprop/54c934b6-e3f8-4460-b930-d5d089b68966/scratchpad")
import numpy as np, torch
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")
from cp_loader import SimpleMoleculeMolGraphFeaturizer
torch.manual_seed(0)
torch.backends.quantized.engine = "qnnpack"

smis=[l.split(",")[0].strip() for l in open("tests/data/regression.csv").readlines()[1:]]
mols=[m for m in (Chem.MolFromSmiles(s) for s in smis) if m is not None]
N=len(mols); feat=SimpleMoleculeMolGraphFeaturizer(); mgs=[feat(m) for m in mols]
V=torch.from_numpy(np.concatenate([g.V for g in mgs])).float()
E=torch.from_numpy(np.concatenate([g.E for g in mgs])).float()
oa=oe=0; eis=[]; revs=[]
for g in mgs:
    eis.append(g.edge_index+oa); revs.append(g.rev_edge_index+oe)
    oa+=g.V.shape[0]; oe+=g.edge_index.shape[1]
ei=torch.from_numpy(np.hstack(eis)).long(); rev=torch.from_numpy(np.concatenate(revs)).long()

def timeit(fn,reps=3,sync=None):
    b=1e9
    for _ in range(reps):
        t=time.perf_counter(); fn()
        if sync: sync()
        b=min(b,time.perf_counter()-t)
    return b

class MP(torch.nn.Module):
    def __init__(s,d_h,depth):
        super().__init__()
        s.W_i=torch.nn.Linear(86,d_h,bias=False); s.W_h=torch.nn.Linear(d_h,d_h,bias=False)
        s.W_o=torch.nn.Linear(72+d_h,d_h); s.depth=depth
    def forward(s,V,E,ei,rev):
        H_0=s.W_i(torch.cat([V[ei[0]],E],dim=1)); H=torch.relu(H_0)
        for _ in range(1,s.depth):
            idx=ei[1].unsqueeze(1).expand(-1,H.shape[1])
            M=torch.zeros(V.shape[0],H.shape[1],dtype=H.dtype,device=H.device).scatter_reduce_(
                0,idx,H,reduce="sum",include_self=False)[ei[0]]-H[rev]
            H=torch.relu(H_0+s.W_h(M))
        idx=ei[1].unsqueeze(1).expand(-1,H.shape[1])
        M=torch.zeros(V.shape[0],H.shape[1],dtype=H.dtype,device=H.device).scatter_reduce_(
            0,idx,H,reduce="sum",include_self=False)
        return s.W_o(torch.cat((V,M),dim=1))

m=MP(2048,6).eval()
with torch.no_grad():
    ref=m(V,E,ei,rev); t32=timeit(lambda:m(V,E,ei,rev))
print(f"CheMeleon dims (d_h=2048, depth=6), {N} molecules in one batch")
print(f"\n{'mode':<34}{'ms':>8}{'speedup':>9}{'rel L2 err':>12}{'max abs err':>13}")
print(f"{'fp32 CPU baseline':<34}{t32*1e3:>8.1f}{1.0:>9.2f}{0.0:>12.1e}{0.0:>13.1e}")

def report(nm,t,o):
    o=o.float().cpu()
    print(f"{nm:<34}{t*1e3:>8.1f}{t32/t:>9.2f}{((o-ref).norm()/ref.norm()):>12.2e}"
          f"{(o-ref).abs().max():>13.2e}")

# int8 dynamic (qnnpack)
try:
    qm=torch.ao.quantization.quantize_dynamic(m,{torch.nn.Linear},dtype=torch.qint8).eval()
    with torch.no_grad():
        o=qm(V,E,ei,rev); report("int8 dynamic, all Linear (qnnpack)",timeit(lambda:qm(V,E,ei,rev)),o)
except Exception as e:
    print(f"{'int8 dynamic':<34}  FAILED: {type(e).__name__}: {str(e)[:60]}")

# int8 dynamic on W_h only (the depth-repeated GEMM)
try:
    qm2=MP(2048,6).eval(); qm2.load_state_dict(m.state_dict())
    qm2.W_h=torch.ao.nn.quantized.dynamic.Linear.from_float(qm2.W_h)
    with torch.no_grad():
        o=qm2(V,E,ei,rev); report("int8 dynamic, W_h only",timeit(lambda:qm2(V,E,ei,rev)),o)
except Exception as e:
    print(f"{'int8 dynamic W_h only':<34}  FAILED: {type(e).__name__}: {str(e)[:60]}")

# low precision
for dt,nm in [(torch.bfloat16,"bf16 CPU"),(torch.float16,"fp16 CPU")]:
    md=MP(2048,6).eval(); md.load_state_dict(m.state_dict()); md=md.to(dt)
    with torch.no_grad():
        o=md(V.to(dt),E.to(dt),ei,rev); report(nm,timeit(lambda:md(V.to(dt),E.to(dt),ei,rev)),o)

# MPS
if torch.backends.mps.is_available():
    sync=torch.mps.synchronize
    for dt,nm in [(torch.float32,"fp32 MPS"),(torch.float16,"fp16 MPS")]:
        md=MP(2048,6).eval(); md.load_state_dict(m.state_dict()); md=md.to("mps").to(dt)
        Vm,Em,eim,revm=V.to("mps").to(dt),E.to("mps").to(dt),ei.to("mps"),rev.to("mps")
        with torch.no_grad():
            o=md(Vm,Em,eim,revm); sync()
            report(nm,timeit(lambda:md(Vm,Em,eim,revm),sync=sync),o)

# weight-only fake-quant error (portable accuracy result, no kernel needed)
print(f"\n--- weight-only quantization error (simulated, per-output-channel) ---")
print(f"{'weight bits':<34}{'rel L2 err':>12}{'max abs err':>13}")
for bits in (8,6,4):
    md=MP(2048,6).eval(); md.load_state_dict(m.state_dict())
    with torch.no_grad():
        for W in (md.W_i,md.W_h,md.W_o):
            n=2**(bits-1)-1
            sc=W.weight.abs().amax(1,keepdim=True)/n
            W.weight.copy_(torch.round(W.weight/sc).clamp(-n-1,n)*sc)
        o=md(V,E,ei,rev)
    print(f"{'int'+str(bits)+' weights':<34}{((o-ref).norm()/ref.norm()):>12.2e}"
          f"{(o-ref).abs().max():>13.2e}")

# quantizing the CACHE rather than the compute
print(f"\n--- quantizing the CACHED vectors (error enters once at W_o, not compounded) ---")
with torch.no_grad():
    H_0=m.W_i(torch.cat([V[ei[0]],E],dim=1)); H=torch.relu(H_0)
    for _ in range(1,6):
        idx=ei[1].unsqueeze(1).expand(-1,H.shape[1])
        Msg=torch.zeros(V.shape[0],H.shape[1]).scatter_reduce_(0,idx,H,reduce="sum",include_self=False)[ei[0]]-H[rev]
        H=torch.relu(H_0+m.W_h(Msg))
    idx=ei[1].unsqueeze(1).expand(-1,H.shape[1])
    M=torch.zeros(V.shape[0],H.shape[1]).scatter_reduce_(0,idx,H,reduce="sum",include_self=False)
print(f"{'cache dtype':<22}{'B/env':>8}{'1M envs':>10}{'rel L2 err':>13}{'max abs err':>13}")
for nm,bits in [("fp32",32),("fp16",16),("int8 per-row",8),("int4 per-row",4)]:
    if nm=="fp32": Mq=M.clone()
    elif nm=="fp16": Mq=M.half().float()
    else:
        n=2**(bits-1)-1; sc=M.abs().amax(1,keepdim=True).clamp(min=1e-12)/n
        Mq=torch.round(M/sc).clamp(-n-1,n)*sc
    with torch.no_grad(): o=m.W_o(torch.cat((V,Mq),dim=1))
    print(f"{nm:<22}{2048*bits//8:>8}{1e6*2048*bits/8/2**30:>9.1f}G"
          f"{((o-ref).norm()/ref.norm()):>13.2e}{(o-ref).abs().max():>13.2e}")
