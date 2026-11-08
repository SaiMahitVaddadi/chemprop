"""Precision sweep on REAL pretrained CheMeleon weights, with a real task metric.

Workflow mirrors how CheMeleon is actually deployed: frozen encoder -> mean agg
-> head fit on fp32 train embeddings -> quantized encoder at inference.
"""
import sys
sys.path.insert(0,"/private/tmp/chemprop-quant-study")
import numpy as np, torch
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")
from cp_loader import SimpleMoleculeMolGraphFeaturizer
torch.set_grad_enabled(False)

CKPT="/Users/svaddadi/Library/Caches/molfeat/chemeleon/chemeleon_mp.pt"
ck=torch.load(CKPT,weights_only=True,map_location="cpu")
hp=ck["hyper_parameters"]; sd=ck["state_dict"]
D_H, DEPTH = hp["d_h"], hp["depth"]
W_i, W_h = sd["W_i.weight"].clone(), sd["W_h.weight"].clone()
W_o, b_o = sd["W_o.weight"].clone(), sd["W_o.bias"].clone()
print(f"CheMeleon: d_h={D_H} depth={DEPTH} | real pretrained weights, "
      f"{sum(v.numel() for v in sd.values())/1e6:.2f}M params")
for n,W in [("W_i",W_i),("W_h",W_h),("W_o",W_o)]:
    a=W.abs()
    print(f"  {n:4s} {tuple(W.shape)!s:14s} std={W.std():.4f} amax={a.max():.3f} "
          f"p99.9={a.flatten().quantile(0.999):.3f} kurtosis={((W-W.mean())**4).mean()/W.var()**2:.1f}")

# ---------------- data ----------------
rows=[l.strip().split(",") for l in open("tests/data/regression.csv").readlines()[1:]]
pairs=[(s,float(y)) for s,y in rows if Chem.MolFromSmiles(s) is not None]
feat=SimpleMoleculeMolGraphFeaturizer()
mgs=[feat(Chem.MolFromSmiles(s)) for s,_ in pairs]
y=torch.tensor([v for _,v in pairs]).float()
V=torch.from_numpy(np.concatenate([g.V for g in mgs])).float()
E=torch.from_numpy(np.concatenate([g.E for g in mgs]).astype(np.float32))
oa=oe=0; eis=[];revs=[];bt=[]
for i,g in enumerate(mgs):
    eis.append(g.edge_index+oa); revs.append(g.rev_edge_index+oe); bt+=[i]*g.V.shape[0]
    oa+=g.V.shape[0]; oe+=g.edge_index.shape[1]
ei=torch.from_numpy(np.hstack(eis)).long(); rev=torch.from_numpy(np.concatenate(revs)).long()
bt=torch.tensor(bt).long(); NM=len(pairs)
print(f"\nESOL: {NM} molecules, {V.shape[0]} atoms, target=logSolubility")

def encode(Wi,Wh,Wo,bo,Vq=None):
    Vx = V if Vq is None else Vq
    H_0=torch.nn.functional.linear(torch.cat([Vx[ei[0]],E],1),Wi); H=torch.relu(H_0)
    for _ in range(1,DEPTH):
        idx=ei[1].unsqueeze(1).expand(-1,D_H)
        M=torch.zeros(Vx.shape[0],D_H).scatter_reduce_(0,idx,H,reduce="sum",
                        include_self=False)[ei[0]]-H[rev]
        H=torch.relu(H_0+torch.nn.functional.linear(M,Wh))
    idx=ei[1].unsqueeze(1).expand(-1,D_H)
    M=torch.zeros(Vx.shape[0],D_H).scatter_reduce_(0,idx,H,reduce="sum",include_self=False)
    Hv=torch.nn.functional.linear(torch.cat((Vx,M),1),Wo,bo)
    cnt=torch.zeros(NM).index_add_(0,bt,torch.ones(bt.shape[0])).clamp(min=1)
    return torch.zeros(NM,D_H).index_add_(0,bt,Hv)/cnt.unsqueeze(1)   # MeanAggregation

# ---------------- quantizers ----------------
def fq(x,emax,m,emin):
    """round-to-nearest in a float format with `m` mantissa bits, exponent in [emin,emax]"""
    e=torch.floor(torch.log2(x.abs().clamp(min=1e-30))).clamp(emin,emax)
    step=torch.exp2(e-m)
    maxv=(2-2.0**(-m))*2.0**emax
    return (torch.round(x/step)*step).clamp(-maxv,maxv)

FP={"e4m3":(8,3,-6),"e5m2":(15,2,-14),"e3m2":(4,2,-2),"e2m3":(2,3,0),"e2m1":(2,1,0)}

def mx(x,fmt,blk=32):
    """OCP microscaling: per-block shared power-of-two (E8M0) scale + element format"""
    emax,m,emin=FP[fmt]
    s=x.shape; xf=x.reshape(-1,s[-1])
    pad=(-xf.shape[1])%blk
    if pad: xf=torch.cat([xf,torch.zeros(xf.shape[0],pad)],1)
    b=xf.reshape(xf.shape[0],-1,blk)
    amax=b.abs().amax(-1,keepdim=True)
    X=torch.exp2(torch.floor(torch.log2(amax.clamp(min=1e-30)))-emax)
    X=X.clamp(min=2.0**-127)
    q=fq(b/X,emax,m,emin)*X
    q=q.reshape(xf.shape[0],-1)[:,:s[-1]]
    return q.reshape(s)

def nvfp4(x,blk=16):
    """E2M1 elements with an E4M3 (not power-of-two) per-16 block scale"""
    emax,m,emin=FP["e2m1"]; s=x.shape; xf=x.reshape(-1,s[-1])
    pad=(-xf.shape[1])%blk
    if pad: xf=torch.cat([xf,torch.zeros(xf.shape[0],pad)],1)
    b=xf.reshape(xf.shape[0],-1,blk)
    X=fq(b.abs().amax(-1,keepdim=True)/6.0,*FP["e4m3"]).clamp(min=1e-30)
    q=fq((b/X).clamp(-6,6),emax,m,emin)*X
    return q.reshape(xf.shape[0],-1)[:,:s[-1]].reshape(s)

def intq(x,bits,group=None,ternary=False):
    s=x.shape
    xf=x.reshape(-1,s[-1]) if group is None else None
    if group is None:
        a=x.abs().amax(-1,keepdim=True).clamp(min=1e-30)      # per-output-channel
        blocks=x
    else:
        xf=x.reshape(-1,s[-1]); pad=(-xf.shape[1])%group
        if pad: xf=torch.cat([xf,torch.zeros(xf.shape[0],pad)],1)
        blocks=xf.reshape(xf.shape[0],-1,group)
        a=blocks.abs().amax(-1,keepdim=True).clamp(min=1e-30)
    if bits==1:
        q=torch.sign(blocks)*blocks.abs().mean(-1,keepdim=True)   # XNOR-style binarization
    elif ternary:
        t=0.7*blocks.abs().mean(-1,keepdim=True)
        q=torch.sign(blocks)*(blocks.abs()>t)*blocks.abs().mean(-1,keepdim=True)
    else:
        n=2**(bits-1)-1; sc=a/n
        q=torch.round(blocks/sc).clamp(-n-1,n)*sc
    if group is None: return q.reshape(s)
    return q.reshape(xf.shape[0],-1)[:,:s[-1]].reshape(s)

# ---------------- baseline + ridge head ----------------
Z0=encode(W_i,W_h,W_o,b_o)
g=torch.Generator().manual_seed(0); perm=torch.randperm(NM,generator=g)
tr,te=perm[:350],perm[350:]
mu,sg=Z0[tr].mean(0),Z0[tr].std(0).clamp(min=1e-6)
def fit(Z,lam=1.0):
    X=torch.cat([(Z[tr]-mu)/sg,torch.ones(len(tr),1)],1)
    A=X.T@X+lam*torch.eye(X.shape[1]); return torch.linalg.solve(A,X.T@y[tr])
w=fit(Z0)
def metrics(Z):
    X=torch.cat([(Z[te]-mu)/sg,torch.ones(len(te),1)],1)
    p=X@w; e=p-y[te]
    return float((e**2).mean().sqrt()), float(e.abs().mean())
r0,m0=metrics(Z0)
print(f"\nfp32 baseline (frozen CheMeleon + ridge head): test RMSE={r0:.4f} MAE={m0:.4f}")
print(f"target std = {float(y[te].std()):.3f}  (so RMSE {r0:.3f} is a real signal)\n")

def report(name,bits,Wi,Wh,Wo,bo):
    Z=encode(Wi,Wh,Wo,bo)
    r,mm=metrics(Z)
    cos=float(torch.nn.functional.cosine_similarity(Z,Z0,dim=1).mean())
    rel=float((Z-Z0).norm()/Z0.norm())
    print(f"{name:<26}{bits:>6}{r:>9.4f}{(r-r0)/r0*100:>+8.1f}%{mm:>8.4f}"
          f"{rel:>10.2e}{cos:>8.5f}")

print("--- WEIGHT quantization (all three matrices; activations stay fp32) ---")
print(f"{'format':<26}{'bits':>6}{'RMSE':>9}{'dRMSE':>9}{'MAE':>8}{'rel L2':>10}{'cos':>8}")
report("fp32 (reference)",32,W_i,W_h,W_o,b_o)
for nm,fn,bits in [("bf16",lambda W:fq(W,127,7,-126),16),
                   ("fp16",lambda W:fq(W,15,10,-14),16),
                   ("MXFP8 e4m3",lambda W:mx(W,"e4m3"),8.25),
                   ("MXFP8 e5m2",lambda W:mx(W,"e5m2"),8.25),
                   ("MXFP6 e3m2",lambda W:mx(W,"e3m2"),6.25),
                   ("MXFP6 e2m3",lambda W:mx(W,"e2m3"),6.25),
                   ("MXFP4 e2m1",lambda W:mx(W,"e2m1"),4.25),
                   ("NVFP4 (e2m1/e4m3 blk16)",lambda W:nvfp4(W),4.5),
                   ("int8 per-channel",lambda W:intq(W,8),8),
                   ("int8 group-32",lambda W:intq(W,8,32),8.25),
                   ("int6 group-32",lambda W:intq(W,6,32),6.25),
                   ("int4 per-channel",lambda W:intq(W,4),4),
                   ("int4 group-32",lambda W:intq(W,4,32),4.25),
                   ("int3 group-32",lambda W:intq(W,3,32),3.25),
                   ("int2 group-32",lambda W:intq(W,2,32),2.25),
                   ("ternary group-32",lambda W:intq(W,2,32,True),1.6),
                   ("int1 binary group-32",lambda W:intq(W,1,32),1.25)]:
    report(nm,bits,fn(W_i),fn(W_h),fn(W_o),b_o)

print("\n--- which matrix is sensitive? (quantize ONE to MXFP4, rest fp32) ---")
print(f"{'quantized':<26}{'bits':>6}{'RMSE':>9}{'dRMSE':>9}{'MAE':>8}{'rel L2':>10}{'cos':>8}")
f4=lambda W:mx(W,"e2m1")
report("W_i only (1x applied)",4.25,f4(W_i),W_h,W_o,b_o)
report("W_h only (5x applied)",4.25,W_i,f4(W_h),W_o,b_o)
report("W_o only (1x applied)",4.25,W_i,W_h,f4(W_o),b_o)
print("\n--- mixed: W_h protected at 8 bits, W_i/W_o pushed low ---")
print(f"{'config':<26}{'bits':>6}{'RMSE':>9}{'dRMSE':>9}{'MAE':>8}{'rel L2':>10}{'cos':>8}")
report("W_h MXFP8, W_i/W_o MXFP4",5.0,f4(W_i),mx(W_h,"e4m3"),f4(W_o),b_o)
report("W_h MXFP6, W_i/W_o MXFP4",4.6,f4(W_i),mx(W_h,"e3m2"),f4(W_o),b_o)
report("W_h int4 g32, W_i/W_o MXFP8",4.5,mx(W_i,"e4m3"),intq(W_h,4,32),mx(W_o,"e4m3"),b_o)

print("\n--- CACHE/ACTIVATION quantization (weights fp32; error enters once) ---")
print(f"{'format':<26}{'bits':>6}{'RMSE':>9}{'dRMSE':>9}{'MAE':>8}{'rel L2':>10}{'cos':>8}")
for nm,fn,bits in [("fp16 cache",lambda Z:fq(Z,15,10,-14),16),
                   ("bf16 cache",lambda Z:fq(Z,127,7,-126),16),
                   ("MXFP8 e4m3 cache",lambda Z:mx(Z,"e4m3"),8.25),
                   ("MXFP6 e3m2 cache",lambda Z:mx(Z,"e3m2"),6.25),
                   ("MXFP4 e2m1 cache",lambda Z:mx(Z,"e2m1"),4.25),
                   ("NVFP4 cache",lambda Z:nvfp4(Z),4.5),
                   ("int8 per-row cache",lambda Z:intq(Z,8),8),
                   ("int4 per-row cache",lambda Z:intq(Z,4),4),
                   ("int4 group-32 cache",lambda Z:intq(Z,4,32),4.25),
                   ("int2 group-32 cache",lambda Z:intq(Z,2,32),2.25)]:
    Zq=fn(Z0); r,mm=metrics(Zq)
    cos=float(torch.nn.functional.cosine_similarity(Zq,Z0,dim=1).mean())
    rel=float((Zq-Z0).norm()/Z0.norm())
    print(f"{nm:<26}{bits:>6}{r:>9.4f}{(r-r0)/r0*100:>+8.1f}%{mm:>8.4f}{rel:>10.2e}{cos:>8.5f}")

print("\n--- CORRECTED mixed precision: protect W_i (it re-enters as a residual every layer) ---")
print(f"{'config':<34}{'bits':>6}{'RMSE':>9}{'dRMSE':>9}{'MAE':>8}{'cos':>9}")
def rep2(name,bits,Wi,Wh,Wo):
    Z=encode(Wi,Wh,Wo,b_o); r,mm=metrics(Z)
    cos=float(torch.nn.functional.cosine_similarity(Z,Z0,dim=1).mean())
    print(f"{name:<34}{bits:>6}{r:>9.4f}{(r-r0)/r0*100:>+8.1f}%{mm:>8.4f}{cos:>9.5f}")
e8=lambda W:mx(W,"e4m3"); e6=lambda W:mx(W,"e3m2"); e4=lambda W:mx(W,"e2m1")
rep2("all MXFP4 (from above)",4.25,e4(W_i),e4(W_h),e4(W_o))
rep2("W_i MXFP8, W_h/W_o MXFP4",4.3,e8(W_i),e4(W_h),e4(W_o))
rep2("W_i bf16,  W_h/W_o MXFP4",4.3,fq(W_i,127,7,-126),e4(W_h),e4(W_o))
rep2("W_i bf16,  W_h/W_o int4 g32",4.3,fq(W_i,127,7,-126),intq(W_h,4,32),intq(W_o,4,32))
rep2("W_i MXFP8, W_h/W_o MXFP6",6.3,e8(W_i),e6(W_h),e6(W_o))
rep2("W_i+W_o MXFP8, W_h MXFP4",4.4,e8(W_i),e4(W_h),e8(W_o))

print("\n--- ENVIRONMENT cache quantization: quantize per-edge H at EVERY layer ---")
print("    (this is the cache from the analog-series design, not the final embedding)")
print(f"{'cache format':<34}{'bits':>6}{'RMSE':>9}{'dRMSE':>9}{'MAE':>8}{'cos':>9}")
def encode_qcache(qf):
    H_0=torch.nn.functional.linear(torch.cat([V[ei[0]],E],1),W_i); H=torch.relu(H_0)
    for _ in range(1,DEPTH):
        idx=ei[1].unsqueeze(1).expand(-1,D_H)
        M=torch.zeros(V.shape[0],D_H).scatter_reduce_(0,idx,H,reduce="sum",
                        include_self=False)[ei[0]]-H[rev]
        H=torch.relu(H_0+torch.nn.functional.linear(M,W_h))
        H=qf(H)                       # store/retrieve through the quantized cache
    idx=ei[1].unsqueeze(1).expand(-1,D_H)
    M=torch.zeros(V.shape[0],D_H).scatter_reduce_(0,idx,H,reduce="sum",include_self=False)
    Hv=torch.nn.functional.linear(torch.cat((V,M),1),W_o,b_o)
    cnt=torch.zeros(NM).index_add_(0,bt,torch.ones(bt.shape[0])).clamp(min=1)
    return torch.zeros(NM,D_H).index_add_(0,bt,Hv)/cnt.unsqueeze(1)
for nm,fn,bits in [("fp16",lambda H:fq(H,15,10,-14),16),
                   ("bf16",lambda H:fq(H,127,7,-126),16),
                   ("MXFP8 e4m3",lambda H:mx(H,"e4m3"),8.25),
                   ("int8 per-row",lambda H:intq(H,8),8),
                   ("int8 group-32",lambda H:intq(H,8,32),8.25),
                   ("MXFP6 e3m2",lambda H:mx(H,"e3m2"),6.25),
                   ("int6 group-32",lambda H:intq(H,6,32),6.25),
                   ("MXFP4 e2m1",lambda H:mx(H,"e2m1"),4.25),
                   ("int4 group-32",lambda H:intq(H,4,32),4.25),
                   ("int2 group-32",lambda H:intq(H,2,32),2.25)]:
    Z=encode_qcache(fn); r,mm=metrics(Z)
    cos=float(torch.nn.functional.cosine_similarity(Z,Z0,dim=1).mean())
    print(f"{nm:<34}{bits:>6}{r:>9.4f}{(r-r0)/r0*100:>+8.1f}%{mm:>8.4f}{cos:>9.5f}")
mem=lambda b: 1e6*2048*b/8/2**30
print(f"\n  memory for a 1M-entry environment cache @ d_h=2048:")
for nm,b in [("fp32",32),("fp16",16),("int8",8.25),("MXFP6",6.25),("int4",4.25)]:
    print(f"    {nm:<8}{mem(b):>6.2f} GiB")
