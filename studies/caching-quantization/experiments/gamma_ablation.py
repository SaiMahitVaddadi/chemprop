"""How much is non-backtracking actually worth?  A one-parameter generalisation.

Chemprop hard-codes a coefficient of 1 on the reverse term:

    M_e = sum_{f: dest(f)=src(e)} H_f  -  1 * H_rev(e)

gamma = 1 gives the non-backtracking walk (Chemprop / the Hashimoto operator);
gamma = 0 gives an ordinary random walk (plain sum over incident edges).  Making
gamma a learnable scalar measures empirically how much the non-backtracking
constraint is worth.  Yang et al. (2019) motivated it by analogy to avoiding
"totters" and nobody appears to have ablated it since.

Three arms, trained from scratch (not CheMeleon - that is frozen at gamma=1):
  fixed_1    gamma pinned to 1.0   (Chemprop today)
  fixed_0    gamma pinned to 0.0   (ordinary walk)
  learned    one gamma per depth iteration, initialised at 1.0

Reports the learned gamma values alongside test RMSE, so "what did it converge to"
and "did it matter" are both visible.

Run: python experiments/gamma_ablation.py [--quick]
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent / "."))
from hashimoto_bench import load_smiles  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from cp_loader import SimpleMoleculeMolGraphFeaturizer  # noqa: E402
from rdkit import Chem, RDLogger  # noqa: E402
from rdkit.Chem.Scaffolds import MurckoScaffold  # noqa: E402

RDLogger.DisableLog("rdApp.*")
HERE = Path(__file__).resolve().parent.parent
OUT = HERE / "results"

DATASETS = {
    "esol": ("delaney-processed.csv", "smiles", "measured log solubility in mols per litre"),
    "freesolv": ("SAMPL.csv", "smiles", "expt"),
}


def read_dataset(name):
    fn, scol, ycol = DATASETS[name]
    import csv
    rows = list(csv.DictReader((HERE / "data" / fn).open()))
    out = []
    for r in rows:
        s = (r.get(scol) or "").strip()
        try:
            y = float(r[ycol])
        except (KeyError, TypeError, ValueError):
            continue
        if s and Chem.MolFromSmiles(s) is not None:
            out.append((s, y))
    return out


class Batch:
    """Flat batched graph with the pieces the message step needs."""

    def __init__(self, graphs, ys):
        self.V = torch.from_numpy(np.concatenate([g.V for g in graphs])).float()
        self.E = torch.from_numpy(np.concatenate([g.E for g in graphs]).astype(np.float32))
        eis, revs, bt, oa, oe = [], [], [], 0, 0
        for i, g in enumerate(graphs):
            eis.append(g.edge_index + oa)
            revs.append(g.rev_edge_index + oe)
            bt += [i] * g.V.shape[0]
            oa += g.V.shape[0]
            oe += g.edge_index.shape[1]
        self.ei = torch.from_numpy(np.hstack(eis)).long()
        self.rev = torch.from_numpy(np.concatenate(revs)).long()
        self.batch = torch.tensor(bt).long()
        self.y = torch.tensor(ys).float()
        self.n_mol = len(graphs)


class GammaMPNN(nn.Module):
    def __init__(self, d_v, d_e, d_h=300, depth=3, mode="fixed_1", dropout=0.0):
        super().__init__()
        self.W_i = nn.Linear(d_v + d_e, d_h, bias=False)
        self.W_h = nn.Linear(d_h, d_h, bias=False)
        self.W_o = nn.Linear(d_v + d_h, d_h)
        self.ffn = nn.Sequential(nn.Linear(d_h, d_h), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(d_h, 1))
        self.depth, self.d_h, self.mode = depth, d_h, mode
        init = {"fixed_1": 1.0, "fixed_0": 0.0, "learned": 1.0}[mode]
        g = torch.full((max(depth - 1, 1),), init)
        self.gamma = nn.Parameter(g) if mode == "learned" else nn.Parameter(g, requires_grad=False)

    def forward(self, b):
        H_0 = self.W_i(torch.cat([self.V_sel(b), b.E], dim=1))
        H = torch.relu(H_0)
        for t in range(1, self.depth):
            nodesum = torch.zeros(b.V.shape[0], self.d_h, device=H.device).index_add_(0, b.ei[1], H)
            M = nodesum.index_select(0, b.ei[0]) - self.gamma[t - 1] * H.index_select(0, b.rev)
            H = torch.relu(H_0 + self.W_h(M))
        nodesum = torch.zeros(b.V.shape[0], self.d_h, device=H.device).index_add_(0, b.ei[1], H)
        Hv = torch.relu(self.W_o(torch.cat((b.V, nodesum), dim=1)))
        cnt = torch.zeros(b.n_mol, device=H.device).index_add_(
            0, b.batch, torch.ones(b.batch.shape[0], device=H.device)).clamp(min=1)
        Z = torch.zeros(b.n_mol, self.d_h, device=H.device).index_add_(0, b.batch, Hv) / cnt[:, None]
        return self.ffn(Z).squeeze(-1)

    @staticmethod
    def V_sel(b):
        return b.V.index_select(0, b.ei[0])


def scaffold_split(smis, frac=0.8, seed=0):
    groups = {}
    for i, s in enumerate(smis):
        try:
            k = MurckoScaffold.MurckoScaffoldSmiles(smiles=s, includeChirality=False)
        except Exception:
            k = s
        groups.setdefault(k, []).append(i)
    sets = sorted(groups.values(), key=len, reverse=True)
    rng = np.random.default_rng(seed)
    if seed:
        rng.shuffle(sets)
    n_tr = int(frac * len(smis))
    tr, te = [], []
    for s in sets:
        (tr if len(tr) + len(s) <= n_tr else te).extend(s)
    return np.array(tr), np.array(te)


def run_split(data, graphs, tr, te, mode, epochs, seed, device="cpu"):
    torch.manual_seed(seed)
    ys = np.array([y for _, y in data], dtype=np.float32)
    mu, sd = ys[tr].mean(), ys[tr].std() + 1e-8
    model = GammaMPNN(graphs[0].V.shape[1], graphs[0].E.shape[1], mode=mode).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    rng = np.random.default_rng(seed)
    bs = 50
    for ep in range(epochs):
        model.train()
        order = rng.permutation(tr)
        for i in range(0, len(order), bs):
            sl = order[i : i + bs]
            b = Batch([graphs[j] for j in sl], ((ys[sl] - mu) / sd).tolist())
            loss = nn.functional.mse_loss(model(b), b.y)
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(te), 100):
            sl = te[i : i + 100]
            b = Batch([graphs[j] for j in sl], ((ys[sl] - mu) / sd).tolist())
            preds.append(model(b))
    p = torch.cat(preds).numpy() * sd + mu
    rmse = float(np.sqrt(((p - ys[te]) ** 2).mean()))
    gam = model.gamma.detach().cpu().numpy().tolist()
    return rmse, gam


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--splits", type=int, default=5)
    ap.add_argument("--datasets", nargs="*", default=["esol", "freesolv"])
    a = ap.parse_args()
    epochs = 5 if a.quick else a.epochs
    n_splits = 2 if a.quick else a.splits

    feat = SimpleMoleculeMolGraphFeaturizer()
    res = {"epochs": epochs, "n_splits": n_splits, "runs": []}
    for ds in a.datasets:
        data = read_dataset(ds)
        if a.quick:
            data = data[:300]
        graphs, keep = [], []
        for i, (s, _) in enumerate(data):
            g = feat(Chem.MolFromSmiles(s))
            if g.edge_index.shape[1]:
                graphs.append(g)
                keep.append(i)
        data = [data[i] for i in keep]
        smis = [s for s, _ in data]
        print(f"\n=== {ds}: {len(data)} molecules, {epochs} epochs, {n_splits} splits ===")
        for mode in ("fixed_1", "fixed_0", "learned"):
            rmses, gammas = [], []
            for sp in range(n_splits):
                tr, te = scaffold_split(smis, seed=sp)
                t0 = time.perf_counter()
                r, g = run_split(data, graphs, tr, te, mode, epochs, seed=sp)
                rmses.append(r)
                gammas.append(g)
                print(f"  {mode:8s} split {sp} RMSE {r:.4f}  gamma {[round(x,3) for x in g]}"
                      f"  ({time.perf_counter()-t0:.0f}s)")
            res["runs"].append({
                "dataset": ds, "mode": mode,
                "rmse_mean": float(np.mean(rmses)), "rmse_std": float(np.std(rmses)),
                "rmse_per_split": rmses, "gamma_per_split": gammas,
                "gamma_mean": np.mean(np.array(gammas), axis=0).tolist(),
            })

    res["caveat"] = (
        "Trained from scratch at d_h=300, depth=3 on scaffold splits - NOT comparable to the "
        "CheMeleon numbers elsewhere in this study, which use a frozen pretrained depth-6 "
        "encoder with a ridge head. These models are small and lightly tuned, so treat the "
        "arms as a controlled comparison between gamma settings, not as absolute accuracy. "
        "The noise floor here is the rmse_std across splits; differences inside it mean nothing."
    )
    OUT.mkdir(exist_ok=True)
    (OUT / "results_gamma_ablation.json").write_text(json.dumps(res, indent=2))
    print(f"\nwrote {OUT / 'results_gamma_ablation.json'}")


if __name__ == "__main__":
    main()
