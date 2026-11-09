"""Bond-level readout: the aggregation family Chemprop's standard MPNN cannot reach.

Chemprop's `MPNN.fingerprint` is:

    H_v = self.message_passing(bmg, V_d)     # atom states only
    H   = self.agg(H_v, bmg.batch)           # atoms -> molecule

The per-edge hidden states H_e are consumed *inside* `_MessagePassingBase.forward`
(scattered into atoms) and never returned. So every readout operates on atoms, even
though message passing computes its states on directed bonds.

This is not a missing capability in the library -- `mol_atom_bond.py:229` already has
`edge_finalize(H, E, E_d)` with its own learned `W_eo = nn.Linear(d_e + d_h, d_h)`, and
line 304 returns `(H_v, H_e)`. The plain MPNN simply has no path to it.

What is lost by summing edges into atoms, precisely:
  * per-edge identity -- which bond contributed what
  * any non-linear edge statistic: variance across the bonds at an atom, max, moments
  * the directed pairing -- H_{u->v} and H_{v->u} are different vectors, and the atom
    sum at v receives only the inbound half
Total "mass" is preserved (each directed edge lands in exactly one atom sum); the
*distribution* over bonds is not.

Implements the full family for edge-level pooling so the loss can be measured rather
than argued: sum / mean / norm(const) / L2-row-norm / max / min / std / var / moments
3-5, PNA degree scaling, attention pooling (numerically safe), learned gated pooling,
and bond-type-conditioned pooling.

Run: python experiments/bond_aggregation.py [--quick]
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hashimoto_bench import build_batch, load_smiles, msg_index_add  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from cp_loader import SimpleMoleculeMolGraphFeaturizer  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "results"
CKPT = Path.home() / "Library/Caches/molfeat/chemeleon/chemeleon_mp.pt"
EPS = 1e-8


# ----------------------------------------------------------- scatter-free reductions
def _seg_sum(H, idx, n):
    return torch.zeros(n, H.shape[1], dtype=H.dtype, device=H.device).index_add_(0, idx, H)


def _seg_count(idx, n, dtype, device):
    return torch.zeros(n, 1, dtype=dtype, device=device).index_add_(
        0, idx, torch.ones(idx.shape[0], 1, dtype=dtype, device=device)
    ).clamp(min=1.0)


def _seg_max(H, idx, n):
    out = torch.full((n, H.shape[1]), float("-inf"), dtype=H.dtype, device=H.device)
    return out.index_reduce_(0, idx, H, reduce="amax", include_self=True).nan_to_num(
        neginf=0.0)


def _seg_min(H, idx, n):
    out = torch.full((n, H.shape[1]), float("inf"), dtype=H.dtype, device=H.device)
    return out.index_reduce_(0, idx, H, reduce="amin", include_self=True).nan_to_num(
        posinf=0.0)


def _seg_moment(H, idx, n, k):
    """Central k-th moment per segment. k=2 -> variance."""
    c = _seg_count(idx, n, H.dtype, H.device)
    mu = _seg_sum(H, idx, n) / c
    d = H - mu.index_select(0, idx)
    return _seg_sum(d.pow(k), idx, n) / c


# --------------------------------------------------------------------- the family
class BondAggregation(nn.Module):
    r"""Pool directed-bond hidden states to one vector per molecule.

    mode:
      sum          sum_e H_e                       extensive
      mean         sum_e H_e / n_e                 intensive
      norm         sum_e H_e / c                    extensive, rescaled (chemprop's "norm")
      l2row        sum_e H_e / ||.||_2              intensive, magnitude-free (EGAT's "norm")
      max / min    elementwise extremum             intensive, order statistic
      std / var    spread across bonds              NOT recoverable from an atom sum
      moment3/4/5  skew- and kurtosis-like          NOT recoverable from an atom sum
      pna          [mean, max, min, std] x degree scalers  (Corso et al. NeurIPS 2020)
      attention    softmax(W H) weighted sum        learned, max-subtracted for stability
      gated        sigmoid(W H) * V H, summed       learned soft selection (GGNN-style)
      bondtype     separate pool per bond order, concatenated
    """

    MODES = ("sum", "mean", "norm", "l2row", "max", "min", "std", "var",
             "moment3", "moment4", "moment5", "pna", "attention", "gated", "bondtype")

    def __init__(self, mode: str, d_h: int, norm_const: float = 100.0,
                 n_bond_types: int = 5, scalers=(-1.0, 0.0, 1.0), delta: float = 1.0):
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"unknown mode {mode!r}; options {self.MODES}")
        self.mode, self.d_h, self.norm_const = mode, d_h, norm_const
        self.n_bond_types, self.scalers, self.delta = n_bond_types, scalers, delta
        if mode == "attention":
            self.score = nn.Linear(d_h, 1)
        elif mode == "gated":
            self.gate = nn.Linear(d_h, 1)
            self.value = nn.Linear(d_h, d_h)

    @property
    def out_dim(self) -> int:
        if self.mode == "pna":
            return self.d_h * 4 * len(self.scalers)
        if self.mode == "bondtype":
            return self.d_h * self.n_bond_types
        return self.d_h

    def forward(self, H: Tensor, e_batch: Tensor, n_mol: int,
                bond_type: Tensor | None = None) -> Tensor:
        m, n = self.mode, n_mol
        if m == "sum":
            return _seg_sum(H, e_batch, n)
        if m == "mean":
            return _seg_sum(H, e_batch, n) / _seg_count(e_batch, n, H.dtype, H.device)
        if m == "norm":
            return _seg_sum(H, e_batch, n) / self.norm_const
        if m == "l2row":
            s = _seg_sum(H, e_batch, n)
            return s / s.norm(dim=1, keepdim=True).clamp(min=EPS)
        if m == "max":
            return _seg_max(H, e_batch, n)
        if m == "min":
            return _seg_min(H, e_batch, n)
        if m == "var":
            return _seg_moment(H, e_batch, n, 2)
        if m == "std":
            return (_seg_moment(H, e_batch, n, 2) + EPS).sqrt()
        if m.startswith("moment"):
            return _seg_moment(H, e_batch, n, int(m[-1]))
        if m == "pna":
            c = _seg_count(e_batch, n, H.dtype, H.device)
            stats = torch.cat([
                _seg_sum(H, e_batch, n) / c,
                _seg_max(H, e_batch, n),
                _seg_min(H, e_batch, n),
                (_seg_moment(H, e_batch, n, 2) + EPS).sqrt(),
            ], dim=1)
            # PNA degree scaler S(d, a) = (log(d+1)/delta)^a
            base = (torch.log(c + 1.0) / self.delta).clamp(min=EPS)
            return torch.cat([stats * base.pow(a) for a in self.scalers], dim=1)
        if m == "attention":
            s = self.score(H)
            # subtract the per-molecule max before exp -- chemprop's AttentiveAggregation
            # omits this and calls .exp() directly, which can overflow to inf -> nan
            mx = _seg_max(s, e_batch, n).index_select(0, e_batch)
            w = (s - mx).exp()
            z = _seg_sum(w, e_batch, n).clamp(min=EPS)
            return _seg_sum(w * H, e_batch, n) / z
        if m == "gated":
            return _seg_sum(torch.sigmoid(self.gate(H)) * self.value(H), e_batch, n)
        if m == "bondtype":
            if bond_type is None:
                raise ValueError("mode='bondtype' needs bond_type")
            outs = []
            for t in range(self.n_bond_types):
                sel = (bond_type == t).to(H.dtype).unsqueeze(1)
                outs.append(_seg_sum(H * sel, e_batch, n))
            return torch.cat(outs, dim=1)
        raise AssertionError(m)


class EnvironmentAggregation(nn.Module):
    r"""Two-level readout: pool to per-node ENVIRONMENTS first, then to the molecule.

    Mirrors the EGAT `AggregatorFunctions` design (h_env / f_env):

      h_env_v = AGG_{u in N(v)} h_u          node features over v's neighbourhood
      f_env_v = AGG_{e incident to v} e_e    edge features over v's incident bonds
      z       = GLOBAL(h_env) , GLOBAL(f_env)
      out     = MIX(z_node, z_edge)          concat | linear | bilinear

    Two properties make this the odd one out among readouts:

    1. `f_env` keeps bond information at NODE granularity -- a middle path between
       Chemprop (scatter edges into atoms, then discard) and a global bond pool.
    2. It operates at a chosen RADIUS, which is the same granularity the environment
       cache is keyed on. Every other readout sits downstream of the cache and is
       therefore free; an environment readout is cacheable IN THE SAME TABLE, so a
       cache hit delivers the readout too. Measured hit rate by radius (pooled, 11
       datasets): r=0 99.9%, r=1 97.2%, r=2 87.9%, r=3 77.1%, r=4 67.9%, r=5 60.6%.
       So readout radius is a cache-budget knob with a known price.

    `self_include=True` adds h_v back to its own environment (EGAT's
    `h_env += x`), which is the same source-reinjection structure as Chemprop's
    `H_0 + W_h M` -- see experiments/residual_flow.py.
    """

    MODES = ("sum", "mean", "max", "min", "std", "norm", "l2row", "attention")
    MIXES = ("concat", "linear", "bilinear")

    def __init__(self, mode: str, d_node: int, d_edge: int, mix: str = "concat",
                 global_mode: str = "sum", self_include: bool = True,
                 norm_const: float = 100.0):
        super().__init__()
        if mode not in self.MODES or mix not in self.MIXES:
            raise ValueError(f"mode {mode!r} / mix {mix!r}")
        self.mode, self.mix, self.global_mode = mode, mix, global_mode
        self.self_include, self.norm_const = self_include, norm_const
        self.d_node, self.d_edge = d_node, d_edge
        if mode == "attention":
            self.score_n = nn.Linear(d_node, 1)
            self.score_e = nn.Linear(d_edge, 1)
        if mix == "linear":
            self.mixer = nn.Linear(d_node + d_edge, d_node + d_edge)
        elif mix == "bilinear":
            # captures multiplicative node x edge interactions that concat+linear cannot
            self.mixer = nn.Bilinear(d_node, d_edge, d_node + d_edge)

    @property
    def out_dim(self) -> int:
        return self.d_node + self.d_edge

    def _pool(self, X, idx, n, score=None):
        m = self.mode
        if m == "sum":
            return _seg_sum(X, idx, n)
        if m == "mean":
            return _seg_sum(X, idx, n) / _seg_count(idx, n, X.dtype, X.device)
        if m == "max":
            return _seg_max(X, idx, n)
        if m == "min":
            return _seg_min(X, idx, n)
        if m == "std":
            return (_seg_moment(X, idx, n, 2) + EPS).sqrt()
        if m == "norm":
            return _seg_sum(X, idx, n) / self.norm_const
        if m == "l2row":
            t = _seg_sum(X, idx, n)
            return t / t.norm(dim=1, keepdim=True).clamp(min=EPS)
        if m == "attention":
            sc = score(X)
            mx = _seg_max(sc, idx, n).index_select(0, idx)   # max-subtracted, unlike
            w = (sc - mx).exp()                              # EGAT's and chemprop's .exp()
            return _seg_sum(w * X, idx, n) / _seg_sum(w, idx, n).clamp(min=EPS)
        raise AssertionError(m)

    def forward(self, H_v, H_e, ei, a_batch, e_batch, n_mol):
        n_v = H_v.shape[0]
        # h_env: neighbour node features into each node (via inbound edges)
        h_env = self._pool(H_v.index_select(0, ei[0]), ei[1], n_v,
                           getattr(self, "score_n", None))
        if self.self_include:
            h_env = h_env + H_v
        # f_env: incident edge features into each node -- the bond info chemprop drops
        f_env = self._pool(H_e, ei[1], n_v, getattr(self, "score_e", None))
        g = (lambda X: _seg_sum(X, a_batch, n_mol)) if self.global_mode == "sum" else (
            lambda X: _seg_sum(X, a_batch, n_mol)
            / _seg_count(a_batch, n_mol, X.dtype, X.device))
        zn, ze = g(h_env), g(f_env)
        if self.mix == "concat":
            return torch.cat([zn, ze], dim=1)
        if self.mix == "linear":
            return self.mixer(torch.cat([zn, ze], dim=1))
        return self.mixer(zn, ze)


class BidirectionalEnvironment(nn.Module):
    r"""Pool the forward flow and the reverse flow separately, then combine.

    Chemprop keeps only the INBOUND half: M_v = sum_{f: dest(f)=v} H_f. But H_{u->v}
    and H_{v->u} are different vectors, so each atom has two distinct aggregates:

        in_v  = AGG_{e: dest(e)=v} H_e      messages arriving at v
        out_v = AGG_{e: src(e)=v}  H_e      messages leaving v

    combine:
      concat   [in, out]              keeps both, 2x width
      sum      in + out               symmetric, direction-blind
      diff     in - out               the DIVERGENCE of the message flow at v
      bilinear B(in, out)             multiplicative interaction

    `diff` is the interesting one. In the flow reading of the update from
    experiments/residual_flow.py -- advection-diffusion with a source term -- in_v - out_v
    is the net flux at v, i.e. the discrete divergence of the message field. A node where
    it is large is a source or a sink of representation. Chemprop cannot express it,
    because it only ever forms `in`.
    """

    COMBINES = ("concat", "sum", "diff", "bilinear")

    def __init__(self, d_h: int, combine: str = "concat", mode: str = "sum"):
        super().__init__()
        if combine not in self.COMBINES:
            raise ValueError(combine)
        self.combine, self.mode, self.d_h = combine, mode, d_h
        if combine == "bilinear":
            self.mixer = nn.Bilinear(d_h, d_h, d_h)

    @property
    def out_dim(self) -> int:
        return self.d_h * 2 if self.combine == "concat" else self.d_h

    def flows(self, H_e, ei, n_v):
        f = _seg_sum if self.mode == "sum" else _seg_max
        return f(H_e, ei[1], n_v), f(H_e, ei[0], n_v)      # inbound, outbound

    def forward(self, H_e, ei, n_v):
        i_v, o_v = self.flows(H_e, ei, n_v)
        if self.combine == "concat":
            return torch.cat([i_v, o_v], dim=1)
        if self.combine == "sum":
            return i_v + o_v
        if self.combine == "diff":
            return i_v - o_v
        return self.mixer(i_v, o_v)


def origin_sensitivity(H_e, ei, V, n_v, n_bins: int = 8, seed: int = 0):
    """Does it matter WHERE a message came from, or only that it arrived?

    Chemprop's atom sum is origin-blind: sum_e H_e discards which neighbour sent each
    message. Three measurements, no training required:

      divergence_ratio   ||in - out|| / ||in + out||, per atom. How much of the flow at
                         an atom is directional. Chemprop forms only `in`, so a large
                         value means it is discarding a real asymmetry.
      origin_tag_gain    participation ratio of origin-TAGGED pooling (separate bucket
                         per source-atom bin, concatenated) minus that of the blind sum.
                         Positive means origin carried information the sum collapsed.
      shuffle_delta      relative change in the atom sums under a GLOBAL permutation of
                         the edge->atom assignment (not degree-preserving: atoms change
                         degree too, so this is an upper bound on the origin effect).
                         If origin were irrelevant, reassigning which message landed
                         where would barely move the aggregate.
    """
    g = torch.Generator().manual_seed(seed)
    inb = _seg_sum(H_e, ei[1], n_v)
    outb = _seg_sum(H_e, ei[0], n_v)
    div = ((inb - outb).norm(dim=1) / (inb + outb).norm(dim=1).clamp(min=EPS))

    def pr(X):
        sv = torch.linalg.svdvals(X.float() - X.float().mean(0, keepdim=True))
        return float((sv.sum() ** 2) / (sv.pow(2).sum() + EPS))

    # origin tag = coarse bin of the source atom's feature vector
    src_key = (V.index_select(0, ei[0]) @ torch.randn(V.shape[1], 1, generator=g)).squeeze(1)
    bins = torch.bucketize(src_key, src_key.quantile(
        torch.linspace(0, 1, n_bins + 1)[1:-1]))
    tagged = torch.cat([_seg_sum(H_e * (bins == b).to(H_e.dtype).unsqueeze(1), ei[1], n_v)
                        for b in range(n_bins)], dim=1)

    # degree-preserving shuffle of which edge lands at which atom
    dst = ei[1]
    perm = torch.randperm(dst.shape[0], generator=g)
    shuffled = _seg_sum(H_e, dst[perm], n_v)

    return {
        "divergence_ratio_mean": float(div.mean()),
        "divergence_ratio_median": float(div.median()),
        "divergence_ratio_p90": float(div.quantile(0.9)),
        "pr_blind_sum": pr(inb),
        "pr_origin_tagged": pr(tagged),
        "origin_tag_gain": pr(tagged) - pr(inb),
        "shuffle_relative_delta": float((shuffled - inb).norm() / inb.norm().clamp(min=EPS)),
        "n_origin_bins": n_bins,
    }


def atom_vs_bond_readout(V, E, ei, rev, W_i, W_h, depth, a_batch, e_batch, n_mol):
    """Return (H_v atom states, H_e final edge states). Chemprop keeps only the first."""
    H_0 = torch.nn.functional.linear(torch.cat([V[ei[0]], E], 1), W_i)
    H = torch.relu(H_0)
    for _ in range(1, depth):
        H = torch.relu(H_0 + torch.nn.functional.linear(
            msg_index_add(H, ei, rev, V.shape[0]), W_h))
    M_v = _seg_sum(H, ei[1], V.shape[0])          # what chemprop scatters into atoms
    return M_v, H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()

    feat = SimpleMoleculeMolGraphFeaturizer()
    smis = load_smiles(limit=100 if a.quick else 400)
    b = build_batch(smis, feat)
    V, E, ei, rev = b["V"], b["E"], b["ei"], b["rev"]
    n_v, n_e = V.shape[0], ei.shape[1]

    if CKPT.exists() and not a.quick:
        sd = torch.load(CKPT, weights_only=True, map_location="cpu")["state_dict"]
        W_i, W_h, d_h, depth = sd["W_i.weight"], sd["W_h.weight"], 2048, 6
    else:
        d_h, depth = 300, 3
        torch.manual_seed(0)
        W_i, W_h = torch.randn(d_h, 86) * 0.05, torch.randn(d_h, d_h) / d_h**0.5

    # per-molecule segment ids for atoms and for directed bonds, from the same
    # graphs build_batch used
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    gs = [feat(Chem.MolFromSmiles(s)) for s in smis]
    gs = [g for g in gs if g.edge_index.shape[1] > 0]
    a_batch = torch.tensor([i for i, g in enumerate(gs) for _ in range(g.V.shape[0])])
    e_batch = torch.tensor([i for i, g in enumerate(gs) for _ in range(g.edge_index.shape[1])])
    n_mol = len(gs)
    assert a_batch.shape[0] == n_v and e_batch.shape[0] == n_e

    with torch.no_grad():
        M_v, H_e = atom_vs_bond_readout(V, E, ei, rev, W_i, W_h, depth,
                                        a_batch, e_batch, n_mol)

    res = {"n_mol": n_mol, "n_atom": int(n_v), "n_dir_bond": int(n_e), "d_h": int(d_h),
           "depth": int(depth), "modes": {}}
    print(f"{n_mol} molecules | {n_v} atoms | {n_e} directed bonds | d_h={d_h} depth={depth}\n")
    print(f"{'mode':12}{'out_dim':>9}{'mean |z|':>11}{'rank':>7}  note")
    for mode in BondAggregation.MODES:
        agg = BondAggregation(mode, d_h)
        with torch.no_grad():
            bt = torch.randint(0, 5, (n_e,)) if mode == "bondtype" else None
            Z = agg(H_e, e_batch, n_mol, bond_type=bt)
            s = torch.linalg.svdvals(Z.float() - Z.float().mean(0, keepdim=True))
            pr = float((s.sum() ** 2) / (s.pow(2).sum() + EPS))
        res["modes"][mode] = {"out_dim": int(agg.out_dim),
                              "mean_abs": float(Z.abs().mean()),
                              "participation_ratio": pr}
        print(f"{mode:12}{agg.out_dim:>9}{float(Z.abs().mean()):>11.4f}{pr:>7.1f}")

    # environment-level readout: radius-1 h_env / f_env, three mixing modes
    print(f"\n{'env mode':12}{'mix':10}{'out_dim':>9}{'mean |z|':>11}{'rank':>7}")
    res["env_modes"] = {}
    for mode in EnvironmentAggregation.MODES:
        for mix in EnvironmentAggregation.MIXES:
            agg = EnvironmentAggregation(mode, d_h, d_h, mix=mix)
            with torch.no_grad():
                Z = agg(M_v, H_e, ei, a_batch, e_batch, n_mol)
                sv = torch.linalg.svdvals(Z.float() - Z.float().mean(0, keepdim=True))
                pr = float((sv.sum() ** 2) / (sv.pow(2).sum() + EPS))
            res["env_modes"][f"{mode}|{mix}"] = {
                "out_dim": int(agg.out_dim), "mean_abs": float(Z.abs().mean()),
                "participation_ratio": pr}
            if mix == "concat":
                print(f"{mode:12}{mix:10}{agg.out_dim:>9}{float(Z.abs().mean()):>11.4f}{pr:>7.1f}")

    # bidirectional flow and message origin
    print(f"\n{'bidirectional':16}{'out_dim':>9}{'mean |z|':>11}{'rank':>7}")
    res["bidirectional"] = {}
    for comb in BidirectionalEnvironment.COMBINES:
        agg = BidirectionalEnvironment(d_h, combine=comb)
        with torch.no_grad():
            Z = agg(H_e, ei, n_v)
            Zm = _seg_sum(Z, a_batch, n_mol)
            sv = torch.linalg.svdvals(Zm.float() - Zm.float().mean(0, keepdim=True))
            pr = float((sv.sum() ** 2) / (sv.pow(2).sum() + EPS))
        res["bidirectional"][comb] = {"out_dim": int(agg.out_dim),
                                      "mean_abs": float(Zm.abs().mean()),
                                      "participation_ratio": pr}
        print(f"{comb:16}{agg.out_dim:>9}{float(Zm.abs().mean()):>11.4f}{pr:>7.1f}")

    with torch.no_grad():
        res["origin"] = origin_sensitivity(H_e, ei, V, n_v)
    o = res["origin"]
    print(f"\nmessage origin:")
    print(f"  divergence ||in-out||/||in+out||  mean {o['divergence_ratio_mean']:.4f}  "
          f"median {o['divergence_ratio_median']:.4f}  p90 {o['divergence_ratio_p90']:.4f}")
    print(f"  participation ratio  blind sum {o['pr_blind_sum']:.1f} -> "
          f"origin-tagged {o['pr_origin_tagged']:.1f}  (gain {o['origin_tag_gain']:+.1f})")
    print(f"  degree-blind shuffle moves the atom sums by {o['shuffle_relative_delta']:.3f} "
          f"(relative)")

    res["caveat"] = [
        "GLOBAL DIVERGENCE IS IDENTICALLY ZERO and that is a theorem, not a measurement "
        "artifact: every directed edge is counted once inbound (at its destination) and "
        "once outbound (at its source), so sum_v (in_v - out_v) = sum_e H_e - sum_e H_e = 0. "
        "The `diff` combine is therefore useless as a MOLECULE-level readout and is only "
        "meaningful per atom -- relevant to mol_atom_bond.py's atom-level targets.",
        "shuffle_relative_delta uses a global permutation, so atoms change degree as well "
        "as origin; it bounds the origin effect from above rather than isolating it. A "
        "degree-preserving shuffle is the tighter control and is not implemented.",
        "This measures the SHAPE and conditioning of each bond readout, not downstream "
        "accuracy. Pair it with the rep2 probe suite to measure what bond readout decodes "
        "that the atom sum cannot -- that is the number that quantifies the information loss.",
        "bondtype mode uses RANDOM bond-type ids here as a shape check; wire it to real "
        "bond orders from the featurizer before reading anything into it.",
        "Attention and gated modes have untrained random weights, so their values are "
        "arbitrary; only out_dim and conditioning are meaningful.",
        "The attention mode subtracts a per-molecule max before exp. Chemprop's "
        "AttentiveAggregation does not (it calls .exp() directly), which is an overflow "
        "hazard -- that difference is deliberate here.",
        "CheMeleon's weights were trained with an atom-only readout, so these edge states "
        "are not what a bond-readout model would learn. A from-scratch comparison is the "
        "companion experiment and is not run here.",
    ]
    OUT.mkdir(exist_ok=True)
    (OUT / "results_bond_aggregation.json").write_text(json.dumps(res, indent=2))
    print(f"\nwrote {OUT / 'results_bond_aggregation.json'}")


if __name__ == "__main__":
    main()
