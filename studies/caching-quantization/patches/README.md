# Patches

## `0001-index_add-and-dtype.diff`

Two independent changes, both behaviour-preserving.

**1. `.repeat`'d int64 scatter index → `index_add_` (9 sites, 5 files).**

Chemprop builds its scatter index as `batch.unsqueeze(1).repeat(1, H.shape[1])`, which
materialises a full `N × d_h` **int64** tensor — 223 MB at CheMeleon dimensions, twice the byte
volume of the features it indexes, allocated and written on every call. More importantly,
`.repeat` fails PyTorch's stride-0 eligibility test and therefore disqualifies the fast GNN
scatter paths on both CPU (`can_use_expanded_index_path`, which routes to a COO→CSR +
spmm-reduce) and CUDA (`fast_scatter_kernel_eligible`, which routes to TMA/vectorized scatter).

`scatter_reduce_(..., include_self=False)` is also replaced, because into a freshly allocated
`torch.zeros` buffer it is mathematically identical to `index_add_` while performing an extra
full scatter traversal writing the identity.

Measured: **2.16× on the default model on CPU**, bitwise identical (max abs diff exactly 0.0 at
both `d_h=300` and `d_h=2048`). On MPS, 1.50× at CheMeleon dimensions, but note MPS scatter is
not bitwise reproducible in *any* variant.

Sites: `nn/message_passing/mixins.py` (×2), `nn/message_passing/base.py`,
`nn/message_passing/mol_atom_bond.py`, `nn/agg.py` (×3 — Mean, Sum, Attentive),
`nn/ffn.py` (×2 — constrained prediction).

**Do not "fix" this with `.expand` instead.** It regresses the large model on CPU (0.88×), and
pytorch#203078 (open) is a silent wrong-answer bug for exactly this shape class: index with
strides `(1,0)` where `self.size(1) > index.size(1)`.

**2. `E` dtype in `featurizers/molgraph/molecule.py`.**

`np.empty((2 * n_bonds, self.bond_fdim))` has no dtype, so bond features are **float64** while
atom features are `np.single`. `collate.py` casts to float32 anyway, so the only effect is 2×
the bytes for every cached `MolGraph` and a wasted conversion. 1.33× cache memory for one word.

### Applying

```bash
git apply studies/caching-quantization/patches/0001-index_add-and-dtype.diff
pytest tests/ -q
```

### Not included, deliberately

`AttentiveAggregation.forward` computes `self.W(H).exp()` with **no max subtraction** before
normalising — a textbook softmax overflow hazard (`exp` of a large positive logit gives `inf`,
then `inf/inf` gives `nan`). Fixing it changes numerics rather than preserving them, so it
belongs in its own change with its own tests, not in a performance patch.
