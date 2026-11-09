"""Shared helpers for the representation-probing study on frozen CheMeleon embeddings.

Everything here operates on the cached fp32 encoder outputs produced by
scripts/prep.py (2048-d, mean-aggregated, one vector per molecule).

Protocol decisions (fixed once, documented so they are not re-tuned per result):
  * ridge probes solved in float64 on z-scored features with an intercept;
  * LAM_DOWNSTREAM = 100 is the penalty the companion quantization study chose on
    fp32 embeddings (cache/lambda.json) -- reused unchanged for every downstream
    task metric so the rank-k curve is comparable to the int4 numbers;
  * LAM_PROBE = 100 likewise for descriptor probes, with a one-off scan reported
    in results so the choice is auditable;
  * every probe gets three controls: predict-train-mean, a frozen random-weight
    CheMeleon encoder of the same 2048-d shape, and a permuted-label control task
    (Hewitt & Liang selectivity).
"""
import json
import os
import pickle
import sys

import numpy as np
import torch

SCRIPTS = "/Users/svaddadi/Documents/GitHub/chemprop/studies/caching-quantization/scripts"
STUDY = "/Users/svaddadi/Documents/GitHub/chemprop/studies/caching-quantization"
RESULTS = os.path.join(STUDY, "results")
CACHE = "/Users/svaddadi/Documents/chemprop-cache-study/cache"
if not os.path.isdir(CACHE):
    CACHE = "/private/tmp/chemprop-quant-study/cache"
sys.path.insert(0, SCRIPTS)

torch.set_grad_enabled(False)

LAM_DOWNSTREAM = 100.0
LAM_PROBE = 100.0

DATASETS = ["esol", "freesolv", "lipo", "bace_reg", "qm9", "bace_cls", "bbbp",
            "clintox", "sider", "tox21", "hiv"]
REG = ["esol", "freesolv", "lipo", "bace_reg", "qm9"]


# --------------------------------------------------------------------------- #
# cached data
# --------------------------------------------------------------------------- #
def load(name):
    with open(f"{CACHE}/{name}.pkl", "rb") as fh:
        D = pickle.load(fh)
    D["Z"] = np.load(f"{CACHE}/{name}_Z0.npy")
    D["name"] = name
    return D


def meta():
    with open(f"{CACHE}/meta.json") as fh:
        return json.load(fh)


def pooled(names=None, dedup=True):
    """Concatenate every cached dataset into one probe corpus (dedup by SMILES)."""
    names = names or DATASETS
    smis, Zs, src = [], [], []
    seen = {}
    for nm in names:
        D = load(nm)
        for i, s in enumerate(D["smiles"]):
            if dedup and s in seen:
                continue
            seen[s] = True
            smis.append(s)
            Zs.append(D["Z"][i])
            src.append(nm)
    return smis, np.stack(Zs), src


# --------------------------------------------------------------------------- #
# RDKit interpretable targets
# --------------------------------------------------------------------------- #
# SMARTS for functional-group indicator flags
SMARTS = {
    "fg_carboxylic_acid": "[CX3](=O)[OX2H1]",
    "fg_amide": "[NX3][CX3](=[OX1])",
    "fg_sulfonamide": "[#16X4]([NX3])(=[OX1])(=[OX1])",
    "fg_halogen": "[F,Cl,Br,I]",
    "fg_basic_amine": "[NX3;H2,H1,H0;!$(N[#6]=[O,N,S]);!$(N=*);!$(N#*);!$([N+]);!$(n)]",
    "fg_nitro": "[$([NX3](=O)=O),$([NX3+](=O)[O-])]",
    "fg_ester": "[CX3](=O)[OX2H0][#6]",
    "fg_phenol": "[OX2H][c]",
    "fg_sulfur_any": "[#16]",
}

# (name, kind) -- kind 'reg' uses a ridge probe + R2, 'cls' a logistic probe + AUROC
TARGETS = [
    # --- local / count-like (should be easy, low depth) ---
    ("mol_wt", "reg"), ("heavy_atoms", "reg"), ("n_hetero", "reg"),
    ("n_hbd", "reg"), ("n_hba", "reg"), ("frac_csp3", "reg"),
    ("formal_charge", "reg"), ("n_valence_e", "reg"),
    # --- bulk physchem (sums of atom/fragment contributions) ---
    ("clogp", "reg"), ("tpsa", "reg"), ("mol_mr", "reg"), ("labute_asa", "reg"),
    # --- topological / global ---
    ("n_rot_bonds", "reg"), ("ring_count", "reg"), ("aromatic_rings", "reg"),
    ("aliphatic_rings", "reg"), ("n_ring_systems", "reg"), ("max_ring_size", "reg"),
    ("n_bridgehead", "reg"), ("n_spiro", "reg"),
    ("graph_diameter", "reg"), ("log_bertz_ct", "reg"), ("balaban_j", "reg"),
    ("log_kappa3", "reg"), ("chi1v", "reg"), ("n_stereocenters", "reg"),
    ("qed", "reg"), ("fsp3_scaffold", "reg"),
] + [(k, "cls") for k in SMARTS]

# which targets we call "local" vs "global" for the depth hypothesis
LOCAL = {"mol_wt", "heavy_atoms", "n_hetero", "n_hbd", "n_hba", "frac_csp3",
         "formal_charge", "n_valence_e", "clogp", "tpsa", "mol_mr", "labute_asa",
         "chi1v", "fg_halogen", "fg_sulfur_any"}
GLOBAL = {"n_rot_bonds", "ring_count", "aromatic_rings", "aliphatic_rings",
          "n_ring_systems", "max_ring_size", "n_bridgehead", "n_spiro",
          "graph_diameter", "log_bertz_ct", "balaban_j", "log_kappa3", "qed",
          "fsp3_scaffold", "n_stereocenters"}


def rdkit_targets(smis):
    """Compute the interpretable probe targets. Returns (dict name->array, mask)."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Crippen, Descriptors, GraphDescriptors, rdMolDescriptors
    from rdkit.Chem import QED
    from rdkit.Chem.Scaffolds import MurckoScaffold
    RDLogger.DisableLog("rdApp.*")

    pats = {k: Chem.MolFromSmarts(v) for k, v in SMARTS.items()}
    out = {n: [] for n, _ in TARGETS}
    ok = np.ones(len(smis), bool)
    for i, s in enumerate(smis):
        m = Chem.MolFromSmiles(s)
        if m is None:
            ok[i] = False
            for n, _ in TARGETS:
                out[n].append(np.nan)
            continue
        ri = m.GetRingInfo()
        rings = ri.AtomRings()
        # fused ring systems = connected components of the ring-ring sharing graph
        nsys = 0
        if rings:
            parent = list(range(len(rings)))

            def find(a):
                while parent[a] != a:
                    parent[a] = parent[parent[a]]
                    a = parent[a]
                return a
            for a in range(len(rings)):
                for b in range(a + 1, len(rings)):
                    if set(rings[a]) & set(rings[b]):
                        ra, rb = find(a), find(b)
                        if ra != rb:
                            parent[ra] = rb
            nsys = len({find(a) for a in range(len(rings))})
        dm = Chem.GetDistanceMatrix(m)
        fin = dm[np.isfinite(dm) & (dm < 1e6)]
        diam = float(fin.max()) if fin.size else 0.0
        try:
            scaf = MurckoScaffold.GetScaffoldForMol(m)
            fsp3s = rdMolDescriptors.CalcFractionCSP3(scaf) if scaf.GetNumAtoms() else 0.0
        except Exception:
            fsp3s = 0.0
        try:
            nst = len(Chem.FindMolChiralCenters(m, includeUnassigned=True, useLegacyImplementation=False))
        except Exception:
            nst = 0
        vals = {
            "mol_wt": Descriptors.MolWt(m),
            "heavy_atoms": m.GetNumHeavyAtoms(),
            "n_hetero": rdMolDescriptors.CalcNumHeteroatoms(m),
            "n_hbd": rdMolDescriptors.CalcNumHBD(m),
            "n_hba": rdMolDescriptors.CalcNumHBA(m),
            "frac_csp3": rdMolDescriptors.CalcFractionCSP3(m),
            "formal_charge": Chem.GetFormalCharge(m),
            "n_valence_e": Descriptors.NumValenceElectrons(m),
            "clogp": Crippen.MolLogP(m),
            "tpsa": rdMolDescriptors.CalcTPSA(m),
            "mol_mr": Crippen.MolMR(m),
            "labute_asa": rdMolDescriptors.CalcLabuteASA(m),
            "n_rot_bonds": rdMolDescriptors.CalcNumRotatableBonds(m),
            "ring_count": ri.NumRings(),
            "aromatic_rings": rdMolDescriptors.CalcNumAromaticRings(m),
            "aliphatic_rings": rdMolDescriptors.CalcNumAliphaticRings(m),
            "n_ring_systems": nsys,
            "max_ring_size": max((len(r) for r in rings), default=0),
            "n_bridgehead": rdMolDescriptors.CalcNumBridgeheadAtoms(m),
            "n_spiro": rdMolDescriptors.CalcNumSpiroAtoms(m),
            "graph_diameter": diam,
            "log_bertz_ct": float(np.log1p(max(GraphDescriptors.BertzCT(m), 0.0))),
            "balaban_j": GraphDescriptors.BalabanJ(m),
            "log_kappa3": float(np.log1p(max(rdMolDescriptors.CalcKappa3(m), 0.0))),
            "chi1v": rdMolDescriptors.CalcChi1v(m),
            "n_stereocenters": nst,
            "qed": QED.qed(m),
            "fsp3_scaffold": fsp3s,
        }
        for k, p in pats.items():
            vals[k] = 1.0 if m.HasSubstructMatch(p) else 0.0
        for n, _ in TARGETS:
            v = vals[n]
            out[n].append(float(v) if np.isfinite(v) else np.nan)
    return {k: np.asarray(v, float) for k, v in out.items()}, ok


# --------------------------------------------------------------------------- #
# classical representations
# --------------------------------------------------------------------------- #
def ecfp4(smis, nbits=2048, counts=True):
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=nbits)
    X = np.zeros((len(smis), nbits), np.float32)
    for i, s in enumerate(smis):
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        fp = gen.GetCountFingerprintAsNumPy(m) if counts else gen.GetFingerprintAsNumPy(m)
        X[i] = fp.astype(np.float32)
    return X


_RD_NAMES = None


def rdkit2d(smis):
    """RDKit's own native 2D descriptor list (~210). mordred/descriptastorus absent."""
    global _RD_NAMES
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    names = [n for n, _ in Descriptors._descList]
    _RD_NAMES = names
    calc = Descriptors.CalcMolDescriptors
    X = np.full((len(smis), len(names)), np.nan, np.float64)
    for i, s in enumerate(smis):
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        try:
            d = calc(m, missingVal=np.nan, silent=True)
        except Exception:
            continue
        X[i] = [d.get(n, np.nan) for n in names]
    X[~np.isfinite(X)] = np.nan
    # column-median impute, then drop constant and numerically unusable columns.
    # RDKit's Ipc overflows float64 on large molecules (it is a product of
    # characteristic-polynomial coefficients), so it is dropped outright.
    med = np.nanmedian(X, 0)
    med[~np.isfinite(med)] = 0.0
    X = np.where(np.isfinite(X), X, med[None, :])
    keep = (X.std(0) > 0) & (np.abs(X).max(0) < 1e12)
    keep &= np.array([n != "Ipc" for n in names])
    return np.ascontiguousarray(X[:, keep]), [n for n, k in zip(names, keep) if k]


def rand_encoder_Z(smis, seed=0, d_h=2048, depth=6, chunk=400):
    """Frozen RANDOM-weight CheMeleon of identical architecture -- the probe-capacity
    control. A random GNN is a non-trivial featurizer, so this is the honest floor."""
    import qlib
    Wi, Wh, Wo, bo, D_H, DEPTH = qlib.load_chemeleon()
    g = torch.Generator().manual_seed(seed)

    def like(W):
        s = 1.0 / np.sqrt(W.shape[1])
        return (torch.rand(W.shape, generator=g) * 2 - 1) * s
    chunks = qlib.build_chunks(smis, chunk)
    return qlib.encode(chunks, like(Wi), like(Wh), like(Wo), torch.zeros_like(bo),
                       depth, d_h).numpy()


# --------------------------------------------------------------------------- #
# probes
# --------------------------------------------------------------------------- #
def _zscale(X):
    """Return (mu, sg) where constant columns get sg=inf so they z-score to 0.
    Needed because ECFP count columns are often all-zero on a given split."""
    mu, sd = X.mean(0), X.std(0)
    sg = np.where(sd > 1e-9, sd, np.inf)
    return mu, sg


def ridge_multi(X, Y, tr, te, lam):
    """Ridge from X to MANY targets at once.

    The Gram matrix and its Cholesky depend only on (X, tr, lam), so they are
    shared across every target -- this is what makes 35 probes x 5 splits x 5
    representations tractable. Returns predictions on `te`, shape (len(te), T).
    """
    X = X.astype(np.float64)
    Xtr = X[tr]
    mu, sg = _zscale(Xtr)
    A = np.concatenate([(Xtr - mu) / sg, np.ones((len(tr), 1))], 1)
    B = np.concatenate([(X[te] - mu) / sg, np.ones((len(te), 1))], 1)
    G = A.T @ A
    G.flat[:: G.shape[0] + 1] += lam
    L = np.linalg.cholesky(G)
    Yt = Y[tr].astype(np.float64)
    ym = Yt.mean(0)
    ys = np.maximum(Yt.std(0), 1e-12)
    rhs = A.T @ ((Yt - ym) / ys)
    W = np.linalg.solve(L.T, np.linalg.solve(L, rhs))
    return (B @ W) * ys + ym, ym


def ridge_dual_multi(X, Y, tr, te, lam):
    """Ridge from X to many targets, solved in the DUAL (n x n) form.

    Mathematically identical to ridge_multi (same z-scoring, same penalised
    intercept column) but costs O(n^3) instead of O(d^3), which is what makes
    4096-d concatenated representations affordable at n<=2200. Verified against
    qlib.Head / ridge_multi in rep3.
    """
    X = X.astype(np.float64)
    mu, sg = _zscale(X[tr])
    A = np.concatenate([(X[tr] - mu) / sg, np.ones((len(tr), 1))], 1)
    B = np.concatenate([(X[te] - mu) / sg, np.ones((len(te), 1))], 1)
    Yt = Y[tr].astype(np.float64)
    ym, ys = Yt.mean(0), np.maximum(Yt.std(0), 1e-12)
    K = A @ A.T
    K.flat[:: K.shape[0] + 1] += lam
    alpha = np.linalg.solve(K, (Yt - ym) / ys)
    return (B @ (A.T @ alpha)) * ys + ym


class DualRidge:
    """Dual (kernel) ridge with the train Gram and cross term built ONCE.

    G = A A^T and C = B A^T depend only on (X, split), so a lambda sweep and a
    NaN-masked multi-task fit reuse them. A NaN-masked task t uses the exact
    submatrices G[ix_(m,m)] and C[:, m]. Verified against qlib.Head to ~1e-5
    (the residual is the float32 cache precision, not the solver).
    """

    def __init__(self, X, tr, te, clip=None):
        X = X.astype(np.float64)
        mu, sg = _zscale(X[tr])
        A = (X[tr] - mu) / sg
        B = (X[te] - mu) / sg
        if clip is not None:
            # Uniform guard against test-time extrapolation. Raw RDKit descriptors
            # have unbounded ranges (one freesolv test molecule otherwise drives the
            # ridge to RMSE 10), so every representation gets the same clip. For
            # CheMeleon it almost never binds; the effect is reported in rep3.
            A = np.clip(A, -clip, clip)
            B = np.clip(B, -clip, clip)
        A = np.concatenate([A, np.ones((len(tr), 1))], 1)
        B = np.concatenate([B, np.ones((len(te), 1))], 1)
        self.G = A @ A.T
        self.C = B @ A.T
        self.n_te = len(te)

    def predict(self, Y, tr, lam):
        T = Y.shape[1]
        P = np.full((self.n_te, T), np.nan)
        full = ~np.isnan(Y[tr]).any(0)
        if full.any():
            cols = np.where(full)[0]
            Yt = Y[tr][:, cols]
            ym, ys = Yt.mean(0), np.maximum(Yt.std(0), 1e-12)
            K = self.G.copy()
            K.flat[:: K.shape[0] + 1] += lam
            P[:, cols] = self.C @ np.linalg.solve(K, (Yt - ym) / ys) * ys + ym
        for t in np.where(~full)[0]:
            m = np.where(~np.isnan(Y[tr, t]))[0]
            if len(m) < 10:
                continue
            yv = Y[tr[m], t]
            ym, ys = yv.mean(), max(yv.std(), 1e-12)
            K = self.G[np.ix_(m, m)].copy()
            K.flat[:: K.shape[0] + 1] += lam
            P[:, t] = self.C[:, m] @ np.linalg.solve(K, (yv - ym) / ys) * ys + ym
        return P


def score_split(P, Y, te, task_type):
    """Primary metric per task from a prediction matrix (RMSE reg / AUROC cls)."""
    from sklearn.metrics import roc_auc_score
    T = Y.shape[1]
    vals = np.full(T, np.nan)
    for t in range(T):
        m = ~np.isnan(Y[te, t])
        if m.sum() < 5 or np.isnan(P[m, t]).any():
            continue
        if task_type == "reg":
            vals[t] = float(np.sqrt(((P[m, t] - Y[te, t][m]) ** 2).mean()))
        else:
            if len(np.unique(Y[te, t][m])) < 2:
                continue
            vals[t] = float(roc_auc_score(Y[te, t][m], P[m, t]))
    return vals


def fit_eval(X, Y, tr, te, task_type, lam, return_per_task=False):
    """Convenience wrapper: one split, one lambda, primary metric."""
    v = score_split(DualRidge(X, tr, te).predict(Y, tr, lam), Y, te, task_type)
    if return_per_task:
        return v
    return float(np.nanmean(v)) if np.isfinite(v).any() else np.nan


def ridge_fit(X, y, lam):
    """float64 ridge on z-scored X with intercept; y is z-scored internally."""
    mu, sg = _zscale(X)
    Xs = np.concatenate([(X - mu) / sg, np.ones((len(X), 1))], 1)
    ym, ys = y.mean(), max(y.std(), 1e-12)
    A = Xs.T @ Xs + lam * np.eye(Xs.shape[1])
    w = np.linalg.solve(A, Xs.T @ ((y - ym) / ys))
    return dict(mu=mu, sg=sg, w=w, ym=ym, ys=ys)


def ridge_pred(f, X):
    Xs = np.concatenate([(X - f["mu"]) / f["sg"], np.ones((len(X), 1))], 1)
    return Xs @ f["w"] * f["ys"] + f["ym"]


def probe_reg(X, y, tr, te, lam=LAM_PROBE):
    f = ridge_fit(X[tr].astype(np.float64), y[tr].astype(np.float64), lam)
    p = ridge_pred(f, X[te].astype(np.float64))
    e = p - y[te]
    v = y[te].var()
    r2 = 1 - (e ** 2).mean() / v if v > 0 else np.nan
    # baseline: predict the TRAIN mean
    e0 = y[tr].mean() - y[te]
    r2_mean = 1 - (e0 ** 2).mean() / v if v > 0 else np.nan
    return float(r2), float(r2_mean), float(np.sqrt((e ** 2).mean()))


def probe_cls(X, y, tr, te, lam=LAM_PROBE):
    """Linear probe via ridge on the 0/1 label; AUROC of the linear score.
    Keeps one solver / one penalty across every probe (comparability > calibration)."""
    from sklearn.metrics import roc_auc_score
    if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
        return np.nan, 0.5, np.nan
    f = ridge_fit(X[tr].astype(np.float64), y[tr].astype(np.float64), lam)
    p = ridge_pred(f, X[te].astype(np.float64))
    return float(roc_auc_score(y[te], p)), 0.5, float(y[tr].mean())


def splits_k(n, k=5, seed=7, frac=0.8):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(k):
        p = rng.permutation(n)
        c = int(round(frac * n))
        out.append((p[:c], p[c:]))
    return out


# --------------------------------------------------------------------------- #
# representation similarity
# --------------------------------------------------------------------------- #
def _center(X):
    X = X.astype(np.float64)
    return X - X.mean(0, keepdims=True)


def linear_cka(X, Y, standardize=False):
    """Linear CKA (Kornblith et al. 2019), computed through the feature-space form
    ||Y^T X||_F^2 / (||X^T X||_F ||Y^T Y||_F) -- exact, O(n d^2)."""
    X, Y = _center(X), _center(Y)
    if standardize:
        X = X / np.maximum(X.std(0, keepdims=True), 1e-12)
        Y = Y / np.maximum(Y.std(0, keepdims=True), 1e-12)
    num = np.linalg.norm(Y.T @ X, "fro") ** 2
    dx = np.linalg.norm(X.T @ X, "fro")
    dy = np.linalg.norm(Y.T @ Y, "fro")
    return float(num / (dx * dy)) if dx > 0 and dy > 0 else np.nan


def pca_reduce(X, var=0.99, kmax=256):
    """PCA-truncate to `var` explained variance (capped at kmax). Returns scores."""
    A = _center(X)
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    cs = np.cumsum(S ** 2) / np.sum(S ** 2)
    kk = min(int(np.searchsorted(cs, var) + 1), kmax, A.shape[0] - 1)
    return U[:, :kk] * S[:kk]


def svcca_from_reduced(A, B, k=64):
    qa = np.linalg.qr(A)[0]
    qb = np.linalg.qr(B)[0]
    s = np.linalg.svd(qa.T @ qb, compute_uv=False)
    return float(np.mean(s[:min(k, len(s))]))


def svcca(X, Y, k=64, var=0.99):
    """SVCCA: PCA-truncate both views, then mean of the top-k canonical correlations."""
    def red(A):
        A = _center(A)
        U, S, Vt = np.linalg.svd(A, full_matrices=False)
        cs = np.cumsum(S ** 2) / np.sum(S ** 2)
        kk = int(np.searchsorted(cs, var) + 1)
        kk = min(kk, A.shape[0] - 1, k * 4)
        return U[:, :kk] * S[:kk]
    A, B = red(X), red(Y)
    qa = np.linalg.qr(A)[0]
    qb = np.linalg.qr(B)[0]
    s = np.linalg.svd(qa.T @ qb, compute_uv=False)
    kk = min(k, len(s))
    return float(np.mean(s[:kk])), int(A.shape[1]), int(B.shape[1])


# --------------------------------------------------------------------------- #
# spectrum / effective rank
# --------------------------------------------------------------------------- #
def spectrum(Z):
    """Eigenvalues of the centered covariance, descending."""
    A = _center(Z)
    s = np.linalg.svd(A, compute_uv=False)
    lam = s ** 2 / max(len(A) - 1, 1)
    return lam


def rank_stats(lam, Z=None):
    lam = np.clip(lam, 0, None)
    tot = lam.sum()
    p = lam / tot if tot > 0 else lam
    nz = p[p > 0]
    ent = float(-(nz * np.log(nz)).sum())
    cs = np.cumsum(p)
    out = dict(
        n_dim=int(len(lam)),
        participation_ratio=float(tot ** 2 / (lam ** 2).sum()) if tot > 0 else 0.0,
        entropy_eff_rank=float(np.exp(ent)),
        stable_rank=float(tot / lam[0]) if lam[0] > 0 else 0.0,
        n90=int(np.searchsorted(cs, 0.90) + 1),
        n95=int(np.searchsorted(cs, 0.95) + 1),
        n99=int(np.searchsorted(cs, 0.99) + 1),
        top1_var_frac=float(p[0]),
        top10_var_frac=float(p[:10].sum()),
        cond_1e6=int((lam > lam[0] * 1e-6).sum()),
    )
    if Z is not None:
        sd = Z.std(0)
        out["n_dead_1e-6"] = int((sd < 1e-6).sum())
        out["n_nearconst_1pct"] = int((sd < 0.01 * np.median(sd)).sum())
        out["median_dim_std"] = float(np.median(sd))
    return out


def pca_basis(Ztr, kmax):
    A = _center(Ztr)
    mu = Ztr.astype(np.float64).mean(0)
    _, S, Vt = np.linalg.svd(A, full_matrices=False)
    return mu, Vt[:kmax], S


def rank_k_recon(Z, mu, V, k):
    A = Z.astype(np.float64) - mu
    return (A @ V[:k].T) @ V[:k] + mu


SCRATCH = "/private/tmp/chemprop-rep-study"


def cached(key, fn):
    """Memoise an expensive array build (RDKit targets, random encoder, ECFP, ...)
    to /private/tmp so experiments 2-6 do not each pay for it."""
    os.makedirs(SCRATCH, exist_ok=True)
    path = f"{SCRATCH}/{key}.npy"
    if os.path.exists(path):
        return np.load(path, allow_pickle=True)
    a = fn()
    np.save(path, a, allow_pickle=True)
    return a


def cached_obj(key, fn):
    """Same as cached() but for arbitrary Python objects (pickle)."""
    os.makedirs(SCRATCH, exist_ok=True)
    path = f"{SCRATCH}/{key}.pkl"
    if os.path.exists(path):
        with open(path, "rb") as fh:
            return pickle.load(fh)
    o = fn()
    with open(path, "wb") as fh:
        pickle.dump(o, fh)
    return o


def jdump(obj, path):
    def conv(o):
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.bool_,)):
            return bool(o)
        raise TypeError(str(type(o)))
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=1, default=conv)
    print(f"wrote {path}")


def score_target(kind, yte, p, ytr=None):
    """Unified probe score: R2 for reg, AUROC for cls. Also returns Spearman rho
    for reg, which is robust to the heavy tails some RDKit descriptors have."""
    if kind == "reg":
        v = float(np.var(yte))
        if v <= 0:
            return np.nan, np.nan
        r2 = float(1 - ((p - yte) ** 2).mean() / v)
        from scipy.stats import spearmanr
        try:
            rho = float(spearmanr(p, yte).statistic)
        except Exception:
            rho = np.nan
        return r2, rho
    from sklearn.metrics import roc_auc_score
    if len(np.unique(yte)) < 2:
        return np.nan, np.nan
    a = float(roc_auc_score(yte, p))
    return a, a
