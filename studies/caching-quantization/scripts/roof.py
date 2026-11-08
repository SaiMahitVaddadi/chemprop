# Roofline / traffic model for Chemprop D-MPNN message step, from the measured graph stats
nA, nB = 120735, 252646     # over 6253 molecules (chemprop tests/data)
nmol = 6253
# scale to the reference batch of 500 drug-like molecules quoted in the task
atoms_per_mol, bonds_per_mol = nA/nmol, nB/nmol
for (n, m, label) in [(6600, 13600, "task's quoted batch (13 atoms/mol)"),
                      (int(500*atoms_per_mol), int(500*bonds_per_mol), "500 mols at chemprop-test size")]:
    print(f"\n=== {label}: n_atoms={n} n_dirbonds={m} ===")
    nnzB_per_row = 1.384
    nnzB = int(nnzB_per_row*m)
    for dh in (300, 2048):
        f4 = 4
        H  = m*dh*f4
        N  = n*dh*f4
        idx64 = m*dh*8           # the .repeat'd int64 index tensor
        # current factored path, as written in mixins.py
        cur = (H          # read H for scatter
             + idx64      # write index_torch
             + idx64      # read index_torch inside scatter_reduce_
             + N          # write node-sum buffer (zeros + scatter)
             + N          # zero-fill the buffer
             + H          # read node sums via gather (m reads of a dh vector)
             + H          # read H[rev_edge_index]
             + H)         # write M
        # same but with index_add_ (1-D index, no .repeat)
        fix = H + N + N + H + H + H
        # explicit Hashimoto CSR / bucketed-gather
        hashi = nnzB*dh*f4 + H
        gemm_traffic = H + H + dh*dh*f4
        gemm_flops   = 2*m*dh*dh
        print(f" d_h={dh:5d}  H={H/1e6:7.1f}MB  idx64={idx64/1e6:7.1f}MB  W_h={dh*dh*4/1e6:5.1f}MB")
        print(f"   message traffic: current={cur/1e6:8.1f}MB  index_add_fix={fix/1e6:8.1f}MB  hashimoto={hashi/1e6:8.1f}MB")
        print(f"   -> index fix alone: {cur/fix:.2f}x ; hashimoto on top of fix: {fix/hashi:.2f}x ; total {cur/hashi:.2f}x")
        print(f"   W_h GEMM: {gemm_flops/1e9:7.1f} GFLOP, traffic {gemm_traffic/1e6:6.1f}MB, AI={gemm_flops/gemm_traffic:7.1f} FLOP/byte")
        print(f"   message-step AI (hashimoto) = {(nnzB*dh)/(hashi):.3f} FLOP/byte  <-- pure bandwidth")
        print(f"   ratio message-traffic/GEMM-traffic: current={cur/gemm_traffic:.2f}  hashimoto={hashi/gemm_traffic:.2f}")
