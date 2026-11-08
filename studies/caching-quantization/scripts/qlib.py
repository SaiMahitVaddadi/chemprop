"""Shared library for the multi-dataset quantization study on CheMeleon.

Contents
  - quantizers (lifted verbatim from quant.py, with a verification suite)
  - dataset loaders (real MoleculeNet CSVs + local chemprop test CSVs)
  - chunked exact CheMeleon encoder (memory-bounded; molecules are disjoint so
    chunking is exact)
  - random + Bemis-Murcko scaffold splits
  - ridge head fit on fp32 train embeddings, metrics for regression/classification
    with NaN masking
"""
import gzip
import os
import sys

import numpy as np
import torch

SCRATCH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRATCH)
REPO = "/Users/svaddadi/Documents/GitHub/chemprop"
DATA = os.path.join(SCRATCH, "data")
CKPT = "/Users/svaddadi/Library/Caches/molfeat/chemeleon/chemeleon_mp.pt"

torch.set_grad_enabled(False)

# --------------------------------------------------------------------------- #
# quantizers (identical math to quant.py)
# --------------------------------------------------------------------------- #
FP = {"e4m3": (8, 3, -6), "e5m2": (15, 2, -14), "e3m2": (4, 2, -2),
      "e2m3": (2, 3, 0), "e2m1": (2, 1, 0)}
# OCP reserves the all-ones mantissa at emax for NaN in e4m3, so its true max is
# 448, not the (2-2^-3)*2^8 = 480 that the generic formula in quant.py produces.
# Only ~0.5% of CheMeleon weights sit in that top code, and MXFP8 is near-lossless
# either way, but we carry the spec-correct variant to prove it does not matter.
FMAX_SPEC = {"e4m3": 448.0}


def fq(x, emax, m, emin, maxv=None):
    """Round-to-nearest into a float format with `m` mantissa bits."""
    e = torch.floor(torch.log2(x.abs().clamp(min=1e-30))).clamp(emin, emax)
    step = torch.exp2(e - m)
    if maxv is None:
        maxv = (2 - 2.0 ** (-m)) * 2.0 ** emax
    return (torch.round(x / step) * step).clamp(-maxv, maxv)


def mx(x, fmt, blk=32, spec_max=False):
    """OCP microscaling: per-block shared power-of-two (E8M0) scale + element fmt."""
    emax, m, emin = FP[fmt]
    mv = FMAX_SPEC.get(fmt) if spec_max else None
    s = x.shape
    xf = x.reshape(-1, s[-1])
    pad = (-xf.shape[1]) % blk
    if pad:
        xf = torch.cat([xf, torch.zeros(xf.shape[0], pad)], 1)
    b = xf.reshape(xf.shape[0], -1, blk)
    amax = b.abs().amax(-1, keepdim=True)
    X = torch.exp2(torch.floor(torch.log2(amax.clamp(min=1e-30))) - emax)
    X = X.clamp(min=2.0 ** -127)
    q = fq(b / X, emax, m, emin, mv) * X
    q = q.reshape(xf.shape[0], -1)[:, :s[-1]]
    return q.reshape(s)


def nvfp4(x, blk=16):
    """E2M1 elements with an E4M3 (not power-of-two) per-16 block scale."""
    emax, m, emin = FP["e2m1"]
    s = x.shape
    xf = x.reshape(-1, s[-1])
    pad = (-xf.shape[1]) % blk
    if pad:
        xf = torch.cat([xf, torch.zeros(xf.shape[0], pad)], 1)
    b = xf.reshape(xf.shape[0], -1, blk)
    X = fq(b.abs().amax(-1, keepdim=True) / 6.0, *FP["e4m3"]).clamp(min=1e-30)
    q = fq((b / X).clamp(-6, 6), emax, m, emin) * X
    return q.reshape(xf.shape[0], -1)[:, :s[-1]].reshape(s)


def intq(x, bits, group=None, ternary=False):
    s = x.shape
    if group is None:
        a = x.abs().amax(-1, keepdim=True).clamp(min=1e-30)   # per-output-channel
        blocks = x
        xf = None
    else:
        xf = x.reshape(-1, s[-1])
        pad = (-xf.shape[1]) % group
        if pad:
            xf = torch.cat([xf, torch.zeros(xf.shape[0], pad)], 1)
        blocks = xf.reshape(xf.shape[0], -1, group)
        a = blocks.abs().amax(-1, keepdim=True).clamp(min=1e-30)
    if bits == 1:
        q = torch.sign(blocks) * blocks.abs().mean(-1, keepdim=True)
    elif ternary:
        t = 0.7 * blocks.abs().mean(-1, keepdim=True)
        q = torch.sign(blocks) * (blocks.abs() > t) * blocks.abs().mean(-1, keepdim=True)
    else:
        n = 2 ** (bits - 1) - 1
        sc = a / n
        q = torch.round(blocks / sc).clamp(-n - 1, n) * sc
    if group is None:
        return q.reshape(s)
    return q.reshape(xf.shape[0], -1)[:, :s[-1]].reshape(s)


BF16 = lambda W: fq(W, 127, 7, -126)
FP16 = lambda W: fq(W, 15, 10, -14)


def verify_quantizers():
    """Spot-check the quantizers against the OCP spec. Returns list of (name, ok, detail)."""
    out = []
    # 1. e2m1 representable set
    g = torch.linspace(-8, 8, 20001).reshape(1, -1)
    vals = sorted(set(torch.unique(fq(g, *FP["e2m1"])).tolist()))
    want = sorted({0.0} | {s * v for v in (0.5, 1, 1.5, 2, 3, 4, 6) for s in (1, -1)})
    out.append(("e2m1 value set", np.allclose(vals, want), f"{len(vals)} vals max={max(vals)}"))
    # 2. e4m3 max value = 448, e5m2 max = 57344
    out.append(("e4m3 amax=448 (spec_max=True)",
                abs(float(fq(torch.tensor([[1e9]]), *FP["e4m3"], 448.0)) - 448) < 1e-6,
                "quant.py default is 480 (generic formula); see KNOWN DEVIATION"))
    out.append(("e5m2 amax=57344", abs(float(fq(torch.tensor([[1e9]]), *FP["e5m2"])) - 57344) < 1e-6, ""))
    # 3. MX block scale is a power of two and exactly one per 32 elements
    x = torch.randn(7, 96) * torch.tensor([1.0, 1e-3, 1e3]).repeat_interleave(32)
    q = mx(x, "e2m1")
    ok = True
    for r in range(7):
        for b in range(3):
            blk = x[r, 32 * b:32 * (b + 1)]
            qb = q[r, 32 * b:32 * (b + 1)]
            X = 2.0 ** (np.floor(np.log2(float(blk.abs().max()))) - 2)
            nz = qb[qb != 0] / X
            # every nonzero element must be an exact e2m1 code
            if not np.allclose(sorted(set(nz.abs().tolist())),
                               sorted(set(np.round(nz.abs().numpy() * 2) / 2)), atol=1e-6):
                ok = False
            if float((qb / X).abs().max()) > 6.0 + 1e-6:
                ok = False
    out.append(("MX e2m1 block scale/codes", ok, "pow2 scale, codes in e2m1 grid, |v|<=6"))
    # 4. relative error bounds: RTN into m mantissa bits => rel err <= 2^-(m+1)
    for fmt, (emax, m, emin) in FP.items():
        # must stay in the NORMAL range of the format: e2m1/e2m3 have emin=0, so
        # anything below 1.0 is subnormal and the 2^-(m+1) bound does not apply.
        x = torch.rand(500, 64) + 1.0
        q = fq(x, emax, m, emin)
        rel = float(((q - x).abs() / x).max())
        out.append((f"fq {fmt} relerr<=2^-{m+1}", rel <= 2.0 ** -(m + 1) + 1e-6, f"{rel:.2e}"))
    # 5. MXFP8-e4m3 should be near-lossless relative to fp32 at 8 bits
    W = torch.randn(512, 2048)
    for fmt, tol in [("e4m3", 0.04), ("e3m2", 0.12), ("e2m1", 0.35)]:
        r = float((mx(W, fmt) - W).norm() / W.norm())
        out.append((f"mx {fmt} relL2", r < tol, f"{r:.4f}"))
    # 6. NVFP4 should beat MXFP4 on a Gaussian (finer scale granularity)
    r_nv = float((nvfp4(W) - W).norm() / W.norm())
    r_mx = float((mx(W, "e2m1") - W).norm() / W.norm())
    out.append(("NVFP4 relL2 < MXFP4 (gaussian)", r_nv < r_mx, f"nv={r_nv:.4f} mx={r_mx:.4f}"))
    # 7. int8 per-channel exactness of scale/clamp
    W2 = torch.randn(64, 256)
    q = intq(W2, 8)
    a = W2.abs().amax(-1, keepdim=True)
    sc = a / 127
    out.append(("int8 per-chan on grid",
                bool(((q / sc - torch.round(q / sc)).abs().max() < 1e-4)), ""))
    out.append(("int8 relL2 < 0.01", float((q - W2).norm() / W2.norm()) < 0.01,
                f"{float((q-W2).norm()/W2.norm()):.4f}"))
    # 8. group-32 int must beat per-channel int at the same bit width
    for b in (4, 8):
        rg = float((intq(W2, b, 32) - W2).norm() / W2.norm())
        rc = float((intq(W2, b) - W2).norm() / W2.norm())
        out.append((f"int{b} g32 < int{b} per-chan", rg < rc, f"g32={rg:.4f} pc={rc:.4f}"))
    return out


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #
def load_chemeleon():
    ck = torch.load(CKPT, weights_only=True, map_location="cpu")
    hp, sd = ck["hyper_parameters"], ck["state_dict"]
    return (sd["W_i.weight"].clone(), sd["W_h.weight"].clone(),
            sd["W_o.weight"].clone(), sd["W_o.bias"].clone(), hp["d_h"], hp["depth"])


# --------------------------------------------------------------------------- #
# featurization / chunked graph batches
# --------------------------------------------------------------------------- #
_FEAT = None


def featurizer():
    global _FEAT
    if _FEAT is None:
        from cp_loader import SimpleMoleculeMolGraphFeaturizer
        _FEAT = SimpleMoleculeMolGraphFeaturizer()
    return _FEAT


def build_chunks(smis, chunk=400):
    """Featurize and pack into graph-batch chunks. Returns list of tuples."""
    from rdkit import Chem
    feat = featurizer()
    mgs = [feat(Chem.MolFromSmiles(s)) for s in smis]
    chunks = []
    for st in range(0, len(mgs), chunk):
        sub = mgs[st:st + chunk]
        V = np.concatenate([g.V for g in sub]).astype(np.float32)
        E = np.concatenate([g.E for g in sub]).astype(np.float32)
        eis, revs, bt = [], [], []
        oa = oe = 0
        for i, g in enumerate(sub):
            eis.append(g.edge_index + oa)
            revs.append(g.rev_edge_index + oe)
            bt += [i] * g.V.shape[0]
            oa += g.V.shape[0]
            oe += g.edge_index.shape[1]
        chunks.append((torch.from_numpy(V), torch.from_numpy(E),
                       torch.from_numpy(np.hstack(eis)).long(),
                       torch.from_numpy(np.concatenate(revs)).long(),
                       torch.tensor(bt).long(), len(sub)))
    return chunks


def encode(chunks, Wi, Wh, Wo, bo, depth=6, d_h=2048, cache_q=None, Vq=None):
    """Exact CheMeleon BondMessagePassing + MeanAggregation, chunked.

    cache_q: optional fn applied to the per-edge hidden state H after every
             message-passing layer (simulates a low-precision environment cache).
    Vq:      optional fn applied to the atom feature matrix.
    """
    outs = []
    for V, E, ei, rev, bt, nm in chunks:
        if Vq is not None:
            V = Vq(V)
        n_v = V.shape[0]
        H_0 = torch.nn.functional.linear(torch.cat([V[ei[0]], E], 1), Wi)
        H = torch.relu(H_0)
        if cache_q is not None:
            H = cache_q(H)
        idx = ei[1].unsqueeze(1).expand(-1, d_h)
        for _ in range(1, depth):
            M = torch.zeros(n_v, d_h).scatter_reduce_(
                0, idx, H, reduce="sum", include_self=False)[ei[0]] - H[rev]
            H = torch.relu(H_0 + torch.nn.functional.linear(M, Wh))
            if cache_q is not None:
                H = cache_q(H)
        M = torch.zeros(n_v, d_h).scatter_reduce_(0, idx, H, reduce="sum", include_self=False)
        Hv = torch.nn.functional.linear(torch.cat((V, M), 1), Wo, bo)
        cnt = torch.zeros(nm).index_add_(0, bt, torch.ones(bt.shape[0])).clamp(min=1)
        outs.append(torch.zeros(nm, d_h).index_add_(0, bt, Hv) / cnt.unsqueeze(1))
    return torch.cat(outs)


# --------------------------------------------------------------------------- #
# datasets
# --------------------------------------------------------------------------- #
def _read(path):
    import pandas as pd
    if path.endswith(".gz"):
        with gzip.open(path, "rt") as fh:
            return pd.read_csv(fh)
    return pd.read_csv(path)


def _clean(smis, Y, cap=None, seed=12345):
    """Drop unparseable / bond-less molecules, optionally subsample."""
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    keep, out = [], []
    for i, s in enumerate(smis):
        m = Chem.MolFromSmiles(s)
        if m is None or m.GetNumBonds() == 0:
            continue
        keep.append(i)
        out.append(Chem.MolToSmiles(m))
    Y = np.asarray(Y, dtype=np.float64)[keep]
    if cap is not None and len(out) > cap:
        rng = np.random.default_rng(seed)
        sel = np.sort(rng.choice(len(out), cap, replace=False))
        out = [out[i] for i in sel]
        Y = Y[sel]
    return out, Y


def load_dataset(name, cap=2200):
    """Returns dict(name, smiles, Y[n,T], task_type, tasks)."""
    if name == "esol":
        df = _read(f"{DATA}/delaney-processed.csv")
        s, Y = _clean(df["smiles"].tolist(),
                      df[["measured log solubility in mols per litre"]].values, cap)
        return dict(name=name, smiles=s, Y=Y, task_type="reg", tasks=["logS"],
                    desc="ESOL/Delaney aqueous solubility")
    if name == "freesolv":
        df = _read(f"{DATA}/SAMPL.csv")
        s, Y = _clean(df["smiles"].tolist(), df[["expt"]].values, cap)
        return dict(name=name, smiles=s, Y=Y, task_type="reg", tasks=["dGhyd"],
                    desc="FreeSolv hydration free energy (kcal/mol)")
    if name == "lipo":
        df = _read(f"{DATA}/Lipophilicity.csv")
        s, Y = _clean(df["smiles"].tolist(), df[["exp"]].values, cap)
        return dict(name=name, smiles=s, Y=Y, task_type="reg", tasks=["logD"],
                    desc="Lipophilicity logD7.4 (ChEMBL)")
    if name == "bace_reg":
        df = _read(f"{DATA}/bace.csv")
        s, Y = _clean(df["mol"].tolist(), df[["pIC50"]].values, cap)
        return dict(name=name, smiles=s, Y=Y, task_type="reg", tasks=["pIC50"],
                    desc="BACE-1 binding affinity pIC50")
    if name == "qm9":
        df = _read(f"{REPO}/tests/data/regression/mol_multitask.csv")
        tasks = [c for c in df.columns if c != "smiles"]
        s, Y = _clean(df["smiles"].tolist(), df[tasks].values, cap)
        return dict(name=name, smiles=s, Y=Y, task_type="reg", tasks=tasks,
                    desc="QM9 subset, 12 DFT properties")
    if name == "bace_cls":
        df = _read(f"{DATA}/bace.csv")
        s, Y = _clean(df["mol"].tolist(), df[["Class"]].values, cap)
        return dict(name=name, smiles=s, Y=Y, task_type="cls", tasks=["active"],
                    desc="BACE-1 binary activity")
    if name == "bbbp":
        df = _read(f"{DATA}/BBBP.csv")
        s, Y = _clean(df["smiles"].tolist(), df[["p_np"]].values, cap)
        return dict(name=name, smiles=s, Y=Y, task_type="cls", tasks=["bbb"],
                    desc="Blood-brain barrier penetration")
    if name == "clintox":
        df = _read(f"{DATA}/clintox.csv.gz")
        s, Y = _clean(df["smiles"].tolist(), df[["FDA_APPROVED", "CT_TOX"]].values, cap)
        return dict(name=name, smiles=s, Y=Y, task_type="cls",
                    tasks=["FDA_APPROVED", "CT_TOX"], desc="ClinTox, 2 tasks")
    if name == "sider":
        df = _read(f"{DATA}/sider.csv.gz")
        tasks = [c for c in df.columns if c != "smiles"]
        s, Y = _clean(df["smiles"].tolist(), df[tasks].values, cap)
        return dict(name=name, smiles=s, Y=Y, task_type="cls", tasks=tasks,
                    desc="SIDER 27 side-effect tasks")
    if name == "tox21":
        df = _read(f"{DATA}/tox21.csv.gz")
        tasks = [c for c in df.columns if c not in ("smiles", "mol_id")]
        s, Y = _clean(df["smiles"].tolist(), df[tasks].values, cap)
        return dict(name=name, smiles=s, Y=Y, task_type="cls", tasks=tasks,
                    desc="Tox21 12 assays (NaN labels)")
    if name == "hiv":
        df = _read(f"{DATA}/HIV.csv")
        s, Y = _clean(df["smiles"].tolist(), df[["HIV_active"]].values, cap)
        return dict(name=name, smiles=s, Y=Y, task_type="cls", tasks=["HIV_active"],
                    desc="HIV replication inhibition")
    raise KeyError(name)


# --------------------------------------------------------------------------- #
# splits
# --------------------------------------------------------------------------- #
def random_split(n, seed, frac=0.8):
    rng = np.random.default_rng(seed)
    p = rng.permutation(n)
    k = int(round(frac * n))
    return p[:k], p[k:]


def scaffold_groups(smis):
    from rdkit.Chem.Scaffolds import MurckoScaffold
    groups = {}
    for i, s in enumerate(smis):
        try:
            sc = MurckoScaffold.MurckoScaffoldSmiles(smiles=s, includeChirality=False)
        except Exception:
            sc = ""
        groups.setdefault(sc, []).append(i)
    return list(groups.values())


def scaffold_split(groups, n, seed=None, frac=0.8):
    """seed=None -> deterministic largest-scaffold-first (classic Bemis-Murcko).
    seed=int   -> seeded random ordering of whole scaffold sets (scaffold-balanced)."""
    gs = list(groups)
    if seed is None:
        gs.sort(key=lambda g: (-len(g), g[0]))
    else:
        rng = np.random.default_rng(seed)
        rng.shuffle(gs)
    cut = int(round(frac * n))
    tr, te = [], []
    for g in gs:
        if len(tr) + len(g) <= cut:
            tr += g
        else:
            te += g
    return np.array(sorted(tr)), np.array(sorted(te))


def make_splits(smis, n_random=6, n_scaffold=4, frac=0.8):
    n = len(smis)
    out = [(f"rand{s}", *random_split(n, 1000 + s, frac)) for s in range(n_random)]
    gs = scaffold_groups(smis)
    out.append(("scafdet", *scaffold_split(gs, n, None, frac)))
    for s in range(n_scaffold - 1):
        out.append((f"scaf{s}", *scaffold_split(gs, n, 2000 + s, frac)))
    return out, len(gs)


# --------------------------------------------------------------------------- #
# ridge head (fit on fp32 train embeddings) + metrics
# --------------------------------------------------------------------------- #
class Head:
    """Per-split ridge head, fitted ONCE on fp32 train embeddings.

    Deployment-realistic: the head never sees the quantized encoder.
    Per-task NaN masking is handled by refitting only on valid rows of that task.
    """

    def __init__(self, Z0, Y, tr, lam=10.0):
        self.tr = tr
        Ztr = Z0[tr]
        self.mu = Ztr.mean(0)
        self.sg = Ztr.std(0).clamp(min=1e-6)
        Xt = (Ztr - self.mu) / self.sg
        Xt = torch.cat([Xt, torch.ones(len(tr), 1)], 1)
        d = Xt.shape[1]
        T = Y.shape[1]
        self.W = torch.zeros(d, T, dtype=torch.float64)
        self.ymu = np.zeros(T)
        self.ysd = np.ones(T)
        I = torch.eye(d, dtype=torch.float64)
        Xt64 = Xt.double()
        full = ~np.isnan(Y[tr]).any(0)          # tasks with no NaN -> one shared solve
        if full.any():
            A = Xt64.T @ Xt64 + lam * I
            L = torch.linalg.cholesky(A)
            cols = np.where(full)[0]
            Yf = Y[tr][:, cols]
            self.ymu[cols] = Yf.mean(0)
            self.ysd[cols] = np.maximum(Yf.std(0), 1e-9)
            Yn = torch.from_numpy((Yf - self.ymu[cols]) / self.ysd[cols])
            self.W[:, cols] = torch.cholesky_solve(Xt64.T @ Yn, L)
        for t in np.where(~full)[0]:
            m = ~np.isnan(Y[tr][:, t])
            if m.sum() < 10:
                self.ysd[t] = np.nan
                continue
            Xm = Xt64[torch.from_numpy(m)]
            yv = Y[tr][m, t]
            self.ymu[t] = yv.mean()
            self.ysd[t] = max(yv.std(), 1e-9)
            A = Xm.T @ Xm + lam * I
            yn = torch.from_numpy((yv - self.ymu[t]) / self.ysd[t])
            self.W[:, t] = torch.linalg.solve(A, Xm.T @ yn)

    def predict(self, Z, idx):
        X = (Z[idx] - self.mu) / self.sg
        X = torch.cat([X, torch.ones(len(idx), 1)], 1).double()
        P = (X @ self.W).numpy()
        return P * self.ysd[None, :] + self.ymu[None, :]


def eval_metrics(P, Y, te, task_type):
    """Per-task metrics with NaN masking. Returns dict of arrays over tasks."""
    from sklearn.metrics import average_precision_score, roc_auc_score
    Yt = Y[te]
    T = Y.shape[1]
    res = {}
    if task_type == "reg":
        rmse, mae, r2 = np.full(T, np.nan), np.full(T, np.nan), np.full(T, np.nan)
        for t in range(T):
            m = ~np.isnan(Yt[:, t])
            if m.sum() < 5:
                continue
            e = P[m, t] - Yt[m, t]
            rmse[t] = np.sqrt((e ** 2).mean())
            mae[t] = np.abs(e).mean()
            v = Yt[m, t].var()
            r2[t] = 1 - (e ** 2).mean() / v if v > 0 else np.nan
        res = dict(rmse=rmse, mae=mae, r2=r2)
    else:
        au, ap = np.full(T, np.nan), np.full(T, np.nan)
        for t in range(T):
            m = ~np.isnan(Yt[:, t])
            y = Yt[m, t]
            if m.sum() < 10 or len(np.unique(y)) < 2:
                continue
            au[t] = roc_auc_score(y, P[m, t])
            ap[t] = average_precision_score(y, P[m, t])
        res = dict(auroc=au, ap=ap)
    return res
