"""Does Chemprop's source re-injection prevent information loss? A flow analysis.

Chemprop's update is NOT a standard residual connection:

    H^(t) = tau( H_0 + W_h M^(t) )          # H_0, not H^(t-1)

The same H_0 is re-injected at EVERY layer. That is a *constant source term*, not a
skip connection. Written as a flow on the graph it is forward-Euler on an
advection-diffusion equation with recharge:

    dH/dt = -(I - W_h B) H + S,    S = H_0,    B = Hashimoto non-backtracking operator

which is the routing equation of a catchment with continuous rainfall. The hydrology
dictionary, used throughout:

    H_0            rainfall / recharge (constant forcing)
    W_h B          channel routing operator (advection)
    graph Laplacian L  dispersion / mixing  -> Dirichlet energy is the mixing measure
    tau = ReLU     storage with a threshold (infiltration excess / spillway)
    depth t        time step
    dH/dx_u        the unit hydrograph: impulse response of the catchment

THE CLAIM TO PROVE: the source term is what prevents information loss, and it does so
for a reason that is exactly statable.

  (a) FIXED POINT. If ||W_h B|| < 1 the map is a contraction.
      Without a source:  H* = tau(W_h B H*)   has the unique fixed point H* = 0.
      With a source:      H* = tau(H_0 + W_h B H*)  has a unique fixed point that is a
      continuous, injective-on-its-range function of H_0.
      So in the deep limit the no-source network forgets its input *exactly*, and the
      source network does not. Depth cannot destroy what is re-injected.

  (b) MARKOV / DATA PROCESSING. H_0 -> H^(1) -> ... -> H^(T) is a Markov chain only
      without re-injection, and the data processing inequality then forces
      I(X; H^(t)) to be non-increasing. Re-injection makes H^(t) depend on H_0
      DIRECTLY, not only through H^(t-1), so the sequence is not a Markov chain and
      the DPI does not apply. The usual information-decay argument is unavailable.

  (c) JACOBIAN. dH^(T)/dH_0 has one direct path (re-injection at layer T) plus T-1
      indirect ones. Without the source only the indirect product survives and decays
      like ||W_h B||^T; with it there is always an O(1) term.

This script measures (a), (c), and three consequences, for three arms:

    source    H^(t) = tau(H_0      + W_h M)    # Chemprop
    residual  H^(t) = tau(H^(t-1)  + W_h M)    # a true residual
    none      H^(t) = tau(           W_h M)    # no shortcut

Run: python experiments/residual_flow.py [--quick]
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hashimoto_bench import build_batch, build_buckets, load_smiles, msg_index_add  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from cp_loader import SimpleMoleculeMolGraphFeaturizer  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "results"
CKPT = Path.home() / "Library/Caches/molfeat/chemeleon/chemeleon_mp.pt"
ARMS = ("source", "residual", "none")


def step(H, H_0, W_h, ei, rev, n_v, arm):
    M = msg_index_add(H, ei, rev, n_v)
    Z = torch.nn.functional.linear(M, W_h)
    if arm == "source":
        return torch.relu(H_0 + Z)
    if arm == "residual":
        return torch.relu(H + Z)
    return torch.relu(Z)


def effective_rank(X, eps=1e-12):
    """exp(entropy of normalised singular values) -- the participation-style rank."""
    s = torch.linalg.svdvals(X.float() - X.float().mean(0, keepdim=True))
    p = (s / s.sum().clamp(min=eps)).clamp(min=eps)
    return float(torch.exp(-(p * p.log()).sum()))


def dirichlet_energy(H, ei):
    """sum over directed edges of ||H_src - H_dst||^2 == the graph mixing / dispersion term."""
    d = H.index_select(0, ei[0]) - H.index_select(0, ei[1])
    return float((d * d).sum() / H.shape[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--depth", type=int, default=12, help="probe past the trained depth")
    ap.add_argument("--d-h", type=int, default=None)
    ap.add_argument("--probe-dim", type=int, default=48,
                    help="random projection width for the Jacobian estimate")
    a = ap.parse_args()
    depth = 6 if a.quick else a.depth

    feat = SimpleMoleculeMolGraphFeaturizer()
    b = build_batch(load_smiles(limit=60 if a.quick else 300), feat)
    V, E, ei, rev = b["V"], b["E"], b["ei"], b["rev"]
    n_v, n_e = V.shape[0], ei.shape[1]

    if CKPT.exists() and a.d_h is None:
        sd = torch.load(CKPT, weights_only=True, map_location="cpu")["state_dict"]
        W_i, W_h = sd["W_i.weight"], sd["W_h.weight"]
        d_h, tag = W_h.shape[0], "chemeleon"
    else:
        d_h = a.d_h or 300
        torch.manual_seed(0)
        W_i = torch.randn(d_h, 86) * 0.05
        W_h = torch.randn(d_h, d_h) / d_h**0.5
        tag = f"random_init_d{d_h}"

    H_0 = torch.nn.functional.linear(torch.cat([V[ei[0]], E], 1), W_i)
    res = {"weights": tag, "d_h": int(d_h), "depth_probed": depth,
           "batch": {"n_mol": b["n_mol"], "n_atom": int(n_v), "n_dir_bond": int(n_e)},
           "operator": {}, "per_arm": {}}

    # ---- (a) is the routing operator a contraction? ||W_h B|| via power iteration on
    #      the composed linear map, which is what actually governs the fixed point.
    torch.manual_seed(1)
    v = torch.randn(n_e, 1)
    v /= v.norm()
    nb = 0.0
    with torch.no_grad():
        for _ in range(200):
            w = msg_index_add(v, ei, rev, n_v)          # single column: B alone
            nb = float(w.norm())
            if nb < 1e-20:
                break
            v = w / nb
    n_wh = float(torch.linalg.matrix_norm(W_h, ord=2))
    deg = np.bincount(ei[1].numpy(), minlength=n_v)
    res["operator"] = {
        "spectral_norm_B": nb,
        "spectral_norm_Wh": n_wh,
        "spectral_norm_product": nb * n_wh,
        "is_contraction": bool(nb * n_wh < 1.0),
        "mean_degree": float(deg.mean()),
        "note": ("H -> W_h(BH) is Kronecker-structured, so its spectral norm is "
                 "||B||_2 * ||W_h||_2 exactly; B is power-iterated on a single feature "
                 "column. An earlier version broadcast one column across all d_h, which "
                 "made W_h sum d_h identical columns and inflated the estimate ~20x. "
                 "product < 1 is SUFFICIENT (ReLU is 1-Lipschitz) but not necessary for "
                 "the no-source map to collapse to 0."),
    }
    print(f"{tag} d_h={d_h} | ||B||={nb:.4f} ||W_h||={n_wh:.4f} "
          f"product={nb*n_wh:.4f} ({'contraction' if nb*n_wh < 1 else 'NOT a contraction'})")

    # ---- per-arm trajectories
    print(f"\n{'arm':10}{'t':>3}{'eff.rank':>10}{'dirichlet':>12}{'|H|':>10}"
          f"{'J_sigma_max':>13}{'decode R2':>11}")
    JAC_AT = sorted({1, 2, 3, 6, depth})
    P = torch.randn(d_h, a.probe_dim) / d_h**0.5          # fixed random readout for probing
    # held-out edge split + a low-dim target, so the probe is overdetermined
    torch.manual_seed(7)
    perm = torch.randperm(n_e)
    tr_idx, te_idx = perm[: int(0.7 * n_e)], perm[int(0.7 * n_e):]
    Y_tgt = (H_0.float() @ torch.randn(d_h, 16) / d_h**0.5).contiguous()
    LAM = 1.0
    if len(tr_idx) < d_h + 1:
        print(f"  WARNING probe underdetermined: {len(tr_idx)} train edges < d_h={d_h}; "
              f"use more molecules for a meaningful decode R2")
    for arm in ARMS:
        traj = {"eff_rank": [], "dirichlet": [], "norm": [], "jac_sigma_max": [],
                "decode_r2_of_H0": []}
        H = torch.relu(H_0).clone()
        for t in range(1, depth + 1):
            with torch.no_grad():
                H = step(H, H_0, W_h, ei, rev, n_v, arm)
                traj["eff_rank"].append(effective_rank(H))
                traj["dirichlet"].append(dirichlet_energy(H, ei))
                traj["norm"].append(float(H.norm() / H.shape[0] ** 0.5))
                traj.setdefault("layer_gain", []).append(
                    traj["norm"][-1] / traj["norm"][-2] if len(traj["norm"]) > 1 else float("nan"))

            # ---- (c) Jacobian dH^(t)/dH_0, estimated on a random subspace.
            # Only at checkpoints: each one re-runs the forward from scratch, so doing
            # every t makes the script O(depth^2) in autograd.
            if t not in JAC_AT:
                traj["jac_sigma_max"].append(float("nan"))
            else:
                h0 = H_0.clone().requires_grad_(True)
                Hk = torch.relu(h0)
                for _ in range(t):
                    Hk = step(Hk, h0, W_h, ei, rev, n_v, arm)
                cols = []
                for j in range(min(8, a.probe_dim)):
                    g = torch.autograd.grad((Hk @ P[:, j]).sum(), h0, retain_graph=True)[0]
                    cols.append(g.reshape(-1))
                J = torch.stack(cols)
                traj["jac_sigma_max"].append(float(torch.linalg.svdvals(J)[0]))

            # ---- information proxy: linear-probe recovery of a low-dim projection of
            #      H_0 from H^(t), fitted on train edges and scored on held-out edges.
            with torch.no_grad():
                Xtr, Xte = H[tr_idx].float(), H[te_idx].float()
                Ytr, Yte = Y_tgt[tr_idx], Y_tgt[te_idx]
                Xtr = torch.cat([Xtr, torch.ones(Xtr.shape[0], 1)], 1)
                Xte = torch.cat([Xte, torch.ones(Xte.shape[0], 1)], 1)
                A = Xtr.T @ Xtr + LAM * torch.eye(Xtr.shape[1])
                Wp = torch.linalg.solve(A, Xtr.T @ Ytr)
                pred = Xte @ Wp
                ss_res = ((pred - Yte) ** 2).sum()
                ss_tot = ((Yte - Ytr.mean(0)) ** 2).sum()
                traj["decode_r2_of_H0"].append(float(1 - ss_res / ss_tot))
            if t in (1, 2, 3, 6, depth):
                print(f"{arm:10}{t:>3}{traj['eff_rank'][-1]:>10.2f}"
                      f"{traj['dirichlet'][-1]:>12.3e}{traj['norm'][-1]:>10.3f}"
                      f"{traj['jac_sigma_max'][-1]:>13.3e}"
                      f"{traj['decode_r2_of_H0'][-1]:>11.4f}")

        # ---- (a) fixed point: iterate far past depth and test H_0 dependence
        with torch.no_grad():
            Hf = torch.relu(H_0).clone()
            for _ in range(200 if not a.quick else 50):
                Hf = step(Hf, H_0, W_h, ei, rev, n_v, arm)
            # perturb H_0 and see whether the fixed point moves
            H0b = H_0 + 0.1 * torch.randn_like(H_0)
            Hg = torch.relu(H0b).clone()
            for _ in range(200 if not a.quick else 50):
                Hg = step(Hg, H0b, W_h, ei, rev, n_v, arm)
            traj["fixed_point"] = {
                "norm": float(Hf.norm() / Hf.shape[0] ** 0.5),
                "collapsed_to_zero": bool(Hf.abs().max() < 1e-6),
                "eff_rank": effective_rank(Hf) if Hf.abs().max() > 1e-12 else 0.0,
                "sensitivity_to_H0": float((Hf - Hg).norm() / (H_0 - H0b).norm().clamp(min=1e-12)),
            }
        res["per_arm"][arm] = traj
        fp = traj["fixed_point"]
        print(f"{arm:10}  fixed point: |H*|={fp['norm']:.4f} "
              f"collapsed={fp['collapsed_to_zero']} "
              f"dH*/dH_0={fp['sensitivity_to_H0']:.4f}")

    res["predictions"] = {
        "none": ("effective rank and ||H|| decay toward 0; decode_r2_of_H0 falls toward 0; "
                 "jac_sigma_max decays geometrically like ||W_h B||^t; fixed point collapses"),
        "source": ("decode_r2_of_H0 stays high at all depths; jac_sigma_max bounded below; "
                   "fixed point nonzero and sensitive to H_0"),
        "residual": ("intermediate: no collapse, but H_0 information is diluted by "
                     "accumulated routing rather than refreshed"),
    }
    res["caveat"] = (
        "The Jacobian is estimated on a random 8-dimensional readout subspace, so "
        "jac_sigma_max is a lower bound on the true leading singular value and comparable "
        "ACROSS arms at fixed t, not as an absolute. decode_r2_of_H0 is a linear-probe lower "
        "bound on mutual information, not an MI estimate. ||W_h B|| is a power-iteration "
        "estimate restricted to a rank-1 feature direction. The 'residual' and 'none' arms "
        "use CheMeleon weights that were trained WITH the source term, so this measures the "
        "flow geometry of those weights under a modified update, not how such a model would "
        "train. A from-scratch three-arm training run is the companion experiment."
    )
    OUT.mkdir(exist_ok=True)
    (OUT / "results_residual_flow.json").write_text(json.dumps(res, indent=2))
    print(f"\nwrote {OUT / 'results_residual_flow.json'}")


if __name__ == "__main__":
    main()
