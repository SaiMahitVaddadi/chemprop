"""Shared data + model harness for chemprop message-passing kernel study."""
import sys, time, warnings, os
warnings.filterwarnings("ignore")
SP = "/private/tmp/claude-502/-Users-svaddadi-Documents-GitHub-chemprop/54c934b6-e3f8-4460-b930-d5d089b68966/scratchpad"
sys.path.insert(0, SP)
import numpy as np, torch
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")
from cp_loader import SimpleMoleculeMolGraphFeaturizer

REPO = "/Users/svaddadi/Documents/GitHub/chemprop"
CHEMELEON = "/Users/svaddadi/Library/Caches/molfeat/chemeleon/chemeleon_mp.pt"


def build_batch(n=500, csv=f"{REPO}/tests/data/regression.csv"):
    smis = [l.split(",")[0].strip() for l in open(csv).readlines()[1:]]
    mols = [m for m in (Chem.MolFromSmiles(s) for s in smis) if m is not None][:n]
    feat = SimpleMoleculeMolGraphFeaturizer()
    mgs = [feat(m) for m in mols]
    V = torch.from_numpy(np.concatenate([g.V for g in mgs])).float()
    E = torch.from_numpy(np.concatenate([g.E for g in mgs])).float()
    oa = oe = 0; eis = []; revs = []; batch = []
    for i, g in enumerate(mgs):
        eis.append(g.edge_index + oa); revs.append(g.rev_edge_index + oe)
        batch += [i] * g.V.shape[0]
        oa += g.V.shape[0]; oe += g.edge_index.shape[1]
    ei = torch.from_numpy(np.hstack(eis)).long()
    rev = torch.from_numpy(np.concatenate(revs)).long()
    return dict(V=V, E=E, ei=ei, rev=rev, batch=torch.tensor(batch).long(),
                n_mol=len(mgs), n_atom=V.shape[0], n_edge=ei.shape[1])


def timeit(fn, reps=7, warmup=2, sync=None):
    for _ in range(warmup):
        fn()
    if sync: sync()
    ts = []
    for _ in range(reps):
        t = time.perf_counter(); fn()
        if sync: sync()
        ts.append(time.perf_counter() - t)
    return min(ts), float(np.median(ts))


class MP(torch.nn.Module):
    """Verbatim-equivalent reimplementation of chemprop BondMessagePassing (no deps)."""
    def __init__(self, d_h=300, depth=3, d_v=72, d_e=14, mode="repeat"):
        super().__init__()
        self.W_i = torch.nn.Linear(d_v + d_e, d_h, bias=False)
        self.W_h = torch.nn.Linear(d_h, d_h, bias=False)
        self.W_o = torch.nn.Linear(d_v + d_h, d_h)
        self.depth = depth; self.mode = mode

    def _scatter(self, H, ei, nV):
        m = self.mode
        if m == "repeat":
            idx = ei[1].unsqueeze(1).repeat(1, H.shape[1])
            return torch.zeros(nV, H.shape[1], dtype=H.dtype, device=H.device).scatter_reduce_(
                0, idx, H, reduce="sum", include_self=False)
        if m == "expand":
            idx = ei[1].unsqueeze(1).expand(-1, H.shape[1])
            return torch.zeros(nV, H.shape[1], dtype=H.dtype, device=H.device).scatter_reduce_(
                0, idx, H, reduce="sum", include_self=False)
        if m == "index_add":
            return torch.zeros(nV, H.shape[1], dtype=H.dtype, device=H.device).index_add_(0, ei[1], H)
        raise ValueError(m)

    def forward(self, V, E, ei, rev):
        nV = V.shape[0]
        H_0 = self.W_i(torch.cat([V[ei[0]], E], dim=1))
        H = torch.relu(H_0)
        for _ in range(1, self.depth):
            M = self._scatter(H, ei, nV)[ei[0]] - H[rev]
            H = torch.relu(H_0 + self.W_h(M))
        M = self._scatter(H, ei, nV)
        return self.W_o(torch.cat((V, M), dim=1))


def load_chemeleon():
    sd = torch.load(CHEMELEON, weights_only=True)
    return sd
