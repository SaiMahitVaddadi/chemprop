"""Explicit Hashimoto operator, SELL-C-sigma / bucketed pure-PyTorch form (brief 2.6(a)).

build_buckets() runs ONCE per batch (BatchMolGraph construction). The hot loop is
`hash_message()`: one index_select per bucket, r-1 adds, one cat.  No scatter, no atomics.
"""
import torch


def build_buckets(ei, rev, nV):
    """ei: (2, 2m) long, rev: (2m,) long.  Returns dict with edge permutation (bucket order)
    and per-bucket gather indices expressed in PERMUTED edge coordinates."""
    src, dst = ei[0], ei[1]
    nE = src.numel()
    deg = torch.bincount(src, minlength=nV)                 # atom degree
    r_e = deg[src] - 1                                      # row-nnz of B for each directed bond
    perm = torch.argsort(r_e, stable=True)
    inv = torch.empty_like(perm); inv[perm] = torch.arange(nE, dtype=perm.dtype)

    # INC[u, :deg(u)] = directed bonds f with dst(f)==u
    maxd = int(deg.max())
    order = torch.argsort(dst, stable=True)
    slot = torch.cat([torch.arange(int(d)) for d in deg.tolist()]) if nE else torch.empty(0, dtype=torch.long)
    INC = torch.zeros(nV, maxd, dtype=torch.long)
    INC[dst[order], slot] = order

    buckets = []            # list of (r, idx[n_r, r]) in permuted coords, rows contiguous
    r_sorted = r_e[perm]
    for r in range(0, maxd):
        sel = (r_sorted == r).nonzero(as_tuple=True)[0]      # permuted row ids (contiguous block)
        if sel.numel() == 0:
            buckets.append((r, None)); continue
        if r == 0:
            buckets.append((r, torch.empty(sel.numel(), 0, dtype=torch.long))); continue
        e_old = perm[sel]                                    # original edge ids
        d = r + 1
        cand = INC[src[e_old], :d]                           # (n, d) incoming bonds at src
        mask = cand != rev[e_old].unsqueeze(1)               # drop exactly the reverse bond
        cols = cand[mask].view(-1, r)
        buckets.append((r, inv[cols].contiguous()))
    counts = [(r, (0 if t is None else t.shape[0])) for r, t in buckets]
    return dict(perm=perm, inv=inv, buckets=buckets, counts=counts,
                src_p=src[perm].contiguous(), dst_p=dst[perm].contiguous(),
                rev_p=inv[rev[perm]].contiguous(), nnz=int(r_e.sum()), nE=nE, nV=nV)


def hash_message(H, bk):
    """M = B H, explicit.  H must be in bucket order."""
    parts = []
    for r, idx in bk["buckets"]:
        if idx is None:
            continue
        if r == 0:
            parts.append(H.new_zeros(idx.shape[0], H.shape[1]))
        elif r == 1:
            parts.append(torch.index_select(H, 0, idx[:, 0]))
        else:
            acc = torch.index_select(H, 0, idx[:, 0])
            for k in range(1, r):
                acc = acc + torch.index_select(H, 0, idx[:, k])
            parts.append(acc)
    return parts[0] if len(parts) == 1 else torch.cat(parts, 0)


def fact_message(H, src, dst, rev, nV):
    """Current chemprop semantics, index_add_ form (bitwise == scatter_reduce on CPU)."""
    N = torch.zeros(nV, H.shape[1], dtype=H.dtype, device=H.device).index_add_(0, dst, H)
    return torch.index_select(N, 0, src) - torch.index_select(H, 0, rev)


class MPHash(torch.nn.Module):
    """Full BondMessagePassing forward using the explicit bucketed Hashimoto operator."""
    def __init__(self, d_h=300, depth=3, d_v=72, d_e=14):
        super().__init__()
        self.W_i = torch.nn.Linear(d_v + d_e, d_h, bias=False)
        self.W_h = torch.nn.Linear(d_h, d_h, bias=False)
        self.W_o = torch.nn.Linear(d_v + d_h, d_h)
        self.depth = depth

    def forward(self, V, E_p, bk):
        H_0 = self.W_i(torch.cat([torch.index_select(V, 0, bk["src_p"]), E_p], dim=1))
        H = torch.relu(H_0)
        Wt = self.W_h.weight.t()
        for _ in range(1, self.depth):
            M = hash_message(H, bk)
            H = torch.addmm(H_0, M, Wt).relu_()
        M = torch.zeros(V.shape[0], H.shape[1], dtype=H.dtype,
                        device=H.device).index_add_(0, bk["dst_p"], H)
        return self.W_o(torch.cat((V, M), dim=1))


def bk_to(bk, dev):
    o = dict(bk)
    for k in ("perm", "inv", "src_p", "dst_p", "rev_p"):
        o[k] = bk[k].to(dev)
    o["buckets"] = [(r, None if t is None else t.to(dev)) for r, t in bk["buckets"]]
    return o


def hash_message_out(H, bk, M, scratch):
    """Same as hash_message but writes into preallocated M (bucket order => contiguous slices),
    avoiding the torch.cat, which is an extra full (2m x d_h) write + read the brief does not count."""
    lo = 0
    for r, idx in bk["buckets"]:
        if idx is None:
            continue
        n = idx.shape[0]; sl = M[lo:lo + n]; lo += n
        if r == 0:
            sl.zero_()
        else:
            torch.index_select(H, 0, idx[:, 0], out=sl)
            for k in range(1, r):
                s = scratch[:n]
                torch.index_select(H, 0, idx[:, k], out=s)
                sl.add_(s)
    return M


class MPHashOut(torch.nn.Module):
    def __init__(self, d_h=300, depth=3, d_v=72, d_e=14):
        super().__init__()
        self.W_i = torch.nn.Linear(d_v + d_e, d_h, bias=False)
        self.W_h = torch.nn.Linear(d_h, d_h, bias=False)
        self.W_o = torch.nn.Linear(d_v + d_h, d_h)
        self.depth = depth; self.d_h = d_h

    def forward(self, V, E_p, bk):
        H_0 = self.W_i(torch.cat([torch.index_select(V, 0, bk["src_p"]), E_p], dim=1))
        H = torch.relu(H_0)
        nE = H.shape[0]
        M = H.new_empty(nE, self.d_h)
        mx = max((0 if t is None else t.shape[0]) for r, t in bk["buckets"])
        scratch = H.new_empty(mx, self.d_h)
        Wt = self.W_h.weight.t()
        for _ in range(1, self.depth):
            hash_message_out(H, bk, M, scratch)
            H = torch.addmm(H_0, M, Wt).relu_()
        Mn = torch.zeros(V.shape[0], self.d_h, dtype=H.dtype,
                         device=H.device).index_add_(0, bk["dst_p"], H)
        return self.W_o(torch.cat((V, Mn), dim=1))


def hash_message_copy(H, bk, M):
    """MPS-SAFE version of hash_message_out.  Does NOT use index_select(out=<offset view>),
    which is SILENTLY WRONG on MPS in torch 2.11 (storage_offset ignored).  Uses
    view.copy_/view.add_, which are correct on both devices."""
    lo = 0
    for r, idx in bk["buckets"]:
        if idx is None:
            continue
        n = idx.shape[0]; sl = M[lo:lo + n]; lo += n
        if r == 0:
            sl.zero_()
        else:
            sl.copy_(torch.index_select(H, 0, idx[:, 0]))
            for k in range(1, r):
                sl.add_(torch.index_select(H, 0, idx[:, k]))
    return M


class MPHashSafe(torch.nn.Module):
    """Recommended form: bucketed Hashimoto, no cat in the hot loop, MPS-safe."""
    def __init__(self, d_h=300, depth=3, d_v=72, d_e=14):
        super().__init__()
        self.W_i = torch.nn.Linear(d_v + d_e, d_h, bias=False)
        self.W_h = torch.nn.Linear(d_h, d_h, bias=False)
        self.W_o = torch.nn.Linear(d_v + d_h, d_h)
        self.depth = depth; self.d_h = d_h

    def forward(self, V, E_p, bk):
        H_0 = self.W_i(torch.cat([torch.index_select(V, 0, bk["src_p"]), E_p], dim=1))
        H = torch.relu(H_0)
        M = H.new_empty(H.shape[0], self.d_h)
        Wt = self.W_h.weight.t()
        for _ in range(1, self.depth):
            hash_message_copy(H, bk, M)
            H = torch.addmm(H_0, M, Wt).relu_()
        Mn = torch.zeros(V.shape[0], self.d_h, dtype=H.dtype,
                         device=H.device).index_add_(0, bk["dst_p"], H)
        return self.W_o(torch.cat((V, Mn), dim=1))
