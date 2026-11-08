"""Direct test of the proposed mechanism for hypothesis 3.

Claim: W_i is hypersensitive because H_0 = W_i([V||E]) is re-injected as a residual
at EVERY one of the `depth` layers, so its error is amplified ~depth times, while
W_h error is attenuated by relu + neighbour summation.

Three independent probes:
  1. depth scan 1..8 -- does W_i sensitivity scale with depth and W_h's not?
  2. RESIDUAL ABLATION -- rerun with H = relu(W_h M) instead of relu(H_0 + W_h M).
     If the mechanism is right, W_i hypersensitivity must largely disappear.
  3. NOISE INJECTION -- add Gaussian noise of a FIXED relative size (1e-3) to one
     matrix at a time. This measures the Jacobian amplification alone, removing the
     confound that MXFP4 may simply produce a larger error on W_i's weights than on
     W_h's. Reported as amplification = relL2(output) / relL2(weight).
"""
import json
import os
import pickle

import numpy as np
import torch

import qlib

CACHE = os.path.join(qlib.SCRATCH, "cache")
LAM = json.load(open(f"{CACHE}/lambda.json"))["lam"]
e4 = lambda W: qlib.mx(W, "e2m1")


def encode(chunks, Wi, Wh, Wo, bo, depth, d_h=2048, residual=True):
    outs = []
    for V, E, ei, rev, bt, nm in chunks:
        n_v = V.shape[0]
        H_0 = torch.nn.functional.linear(torch.cat([V[ei[0]], E], 1), Wi)
        H = torch.relu(H_0)
        idx = ei[1].unsqueeze(1).expand(-1, d_h)
        for _ in range(1, depth):
            M = torch.zeros(n_v, d_h).scatter_reduce_(
                0, idx, H, reduce="sum", include_self=False)[ei[0]] - H[rev]
            Z = torch.nn.functional.linear(M, Wh)
            H = torch.relu(H_0 + Z) if residual else torch.relu(Z)
        M = torch.zeros(n_v, d_h).scatter_reduce_(0, idx, H, reduce="sum", include_self=False)
        Hv = torch.nn.functional.linear(torch.cat((V, M), 1), Wo, bo)
        cnt = torch.zeros(nm).index_add_(0, bt, torch.ones(bt.shape[0])).clamp(min=1)
        outs.append(torch.zeros(nm, d_h).index_add_(0, bt, Hv) / cnt.unsqueeze(1))
    return torch.cat(outs)


def noisy(W, rel, gen):
    n = torch.randn(W.shape, generator=gen)
    return W + n * (rel * W.norm() / n.norm())


def main():
    W_i, W_h, W_o, b_o, d_h, _ = qlib.load_chemeleon()
    W = dict(Wi=W_i, Wh=W_h, Wo=W_o)
    recs = []

    # weight-space error of MXFP4 per matrix (the confound to control for)
    print("=== weight-space MXFP4 error per matrix (controls for 'W_i is just harder to quantize') ===")
    wrel = {}
    for n, M in W.items():
        wrel[n] = float((e4(M) - M).norm() / M.norm())
        k = float(((M - M.mean()) ** 4).mean() / M.var() ** 2)
        print(f"  {n}: shape={tuple(M.shape)} params={M.numel()/1e6:.3f}M "
              f"relL2(MXFP4)={wrel[n]:.4f} kurtosis={k:.1f}")
    print()

    DS = ["esol", "bbbp"]
    data = {}
    for name in DS:
        with open(f"{CACHE}/{name}.pkl", "rb") as fh:
            D = pickle.load(fh)
        D["chunks"] = qlib.build_chunks(D["smiles"], 400)
        data[name] = D

    for residual in (True, False):
        tag = "residual(real)" if residual else "no-residual(ablated)"
        print(f"=== {tag}: MXFP4 on ONE matrix + fixed-size noise injection ===")
        print(f"{'depth':>5} {'ds':<6} {'relL2_out: Wi':>13}{'Wh':>9}{'Wo':>9}"
              f" | {'amplification(noise 1e-3): Wi':>29}{'Wh':>9}{'Wo':>9}"
              f" | {'dPrimary%: Wi':>13}{'Wh':>9}{'Wo':>9}")
        for depth in range(1, 9):
            for name in DS:
                D = data[name]
                ch, Y, tt, tasks = D["chunks"], D["Y"], D["task_type"], D["tasks"]
                Z0 = encode(ch, W_i, W_h, W_o, b_o, depth, d_h, residual)
                heads = [(sn, te, qlib.Head(Z0, Y, tr, LAM)) for sn, tr, te in D["splits"]]
                key = "rmse" if tt == "reg" else "auroc"
                base = {}
                for sn, te, h in heads:
                    v = qlib.eval_metrics(h.predict(Z0, te), Y, te, tt)[key]
                    # bbbp's deterministic scaffold split yields a single-class test
                    # set (no AUROC); such splits are skipped, not imputed.
                    base[sn] = np.nanmean(v) if np.isfinite(v).any() else np.nan
                rel_out, amp, dmet = {}, {}, {}
                gen = torch.Generator().manual_seed(7)
                for mn in ("Wi", "Wh", "Wo"):
                    q = {k: v for k, v in W.items()}
                    q[mn] = e4(W[mn])
                    Zq = encode(ch, q["Wi"], q["Wh"], q["Wo"], b_o, depth, d_h, residual)
                    rel_out[mn] = float((Zq - Z0).norm() / Z0.norm())
                    d = []
                    for sn, te, h in heads:
                        if not np.isfinite(base[sn]):
                            continue
                        vv = qlib.eval_metrics(h.predict(Zq, te), Y, te, tt)[key]
                        v = np.nanmean(vv)
                        sign = 1.0 if tt == "reg" else -1.0   # report as % degradation
                        d.append(sign * (v - base[sn]) / abs(base[sn]) * 100)
                    dmet[mn] = float(np.mean(d))
                    # fixed-relative-size noise -> pure Jacobian amplification
                    qn = {k: v for k, v in W.items()}
                    qn[mn] = noisy(W[mn], 1e-3, gen)
                    Zn = encode(ch, qn["Wi"], qn["Wh"], qn["Wo"], b_o, depth, d_h, residual)
                    amp[mn] = float((Zn - Z0).norm() / Z0.norm()) / 1e-3
                    recs.append(dict(residual=residual, depth=depth, dataset=name,
                                     matrix=mn, rel_out_mxfp4=rel_out[mn],
                                     amp_noise=amp[mn], d_primary_pct=dmet[mn],
                                     w_relL2_mxfp4=wrel[mn], metric=key,
                                     base=float(np.nanmean(list(base.values())))))
                print(f"{depth:>5} {name:<6} "
                      f"{rel_out['Wi']:>13.4f}{rel_out['Wh']:>9.4f}{rel_out['Wo']:>9.4f}"
                      f" | {amp['Wi']:>29.2f}{amp['Wh']:>9.2f}{amp['Wo']:>9.2f}"
                      f" | {dmet['Wi']:>+13.1f}{dmet['Wh']:>+9.1f}{dmet['Wo']:>+9.1f}",
                      flush=True)
        print()
    with open(f"{qlib.SCRATCH}/results_depth.json", "w") as fh:
        json.dump(recs, fh, indent=1)


if __name__ == "__main__":
    main()
