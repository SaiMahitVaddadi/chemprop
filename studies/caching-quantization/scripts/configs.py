"""The quantization configurations swept in this study.

Each config is (group, name, nominal_bits, kind, spec):
  kind "w"     -> spec = (fn_Wi, fn_Wh, fn_Wo); weights quantized, activations fp32
  kind "cache" -> spec = fn applied to the per-edge hidden state H after EVERY
                  message-passing layer (a low-precision environment cache)
Nominal bits include the block-scale overhead (e.g. MXFP4 = 4 + 8/32 = 4.25).
"""
from qlib import BF16, FP16, intq, mx, nvfp4

I = lambda W: W                                    # identity / fp32
e8 = lambda W: mx(W, "e4m3")
e8s = lambda W: mx(W, "e4m3", spec_max=True)
e8b = lambda W: mx(W, "e5m2")
e6 = lambda W: mx(W, "e3m2")
e6b = lambda W: mx(W, "e2m3")
e4 = lambda W: mx(W, "e2m1")
nv = nvfp4
i8c = lambda W: intq(W, 8)
i8g = lambda W: intq(W, 8, 32)
i6c = lambda W: intq(W, 6)
i6g = lambda W: intq(W, 6, 32)
i4c = lambda W: intq(W, 4)
i4g = lambda W: intq(W, 4, 32)
i3g = lambda W: intq(W, 3, 32)
i2g = lambda W: intq(W, 2, 32)
i1g = lambda W: intq(W, 1, 32)
tern = lambda W: intq(W, 2, 32, True)

# ---------------- (A) uniform weight quantization ---------------- #
A = [
    ("fp32", 32.0, (I, I, I)),
    ("bf16", 16.0, (BF16, BF16, BF16)),
    ("fp16", 16.0, (FP16, FP16, FP16)),
    ("MXFP8-e4m3", 8.25, (e8, e8, e8)),
    ("MXFP8-e4m3-spec448", 8.25, (e8s, e8s, e8s)),
    ("MXFP8-e5m2", 8.25, (e8b, e8b, e8b)),
    ("MXFP6-e3m2", 6.25, (e6, e6, e6)),
    ("MXFP6-e2m3", 6.25, (e6b, e6b, e6b)),
    ("MXFP4-e2m1", 4.25, (e4, e4, e4)),
    ("NVFP4", 4.50, (nv, nv, nv)),
    ("int8-perchan", 8.00, (i8c, i8c, i8c)),
    ("int8-g32", 8.25, (i8g, i8g, i8g)),
    ("int6-perchan", 6.00, (i6c, i6c, i6c)),
    ("int6-g32", 6.25, (i6g, i6g, i6g)),
    ("int4-perchan", 4.00, (i4c, i4c, i4c)),
    ("int4-g32", 4.25, (i4g, i4g, i4g)),
    ("int3-g32", 3.25, (i3g, i3g, i3g)),
    ("int2-g32", 2.25, (i2g, i2g, i2g)),
    ("int1-g32", 1.25, (i1g, i1g, i1g)),
    ("ternary-g32", 1.60, (tern, tern, tern)),
]

# ---------------- (B) per-matrix sensitivity + mixed precision ---------------- #
B = [
    # which matrix carries the error? (quantize exactly one)
    ("only-Wi@MXFP4", 4.25, (e4, I, I)),
    ("only-Wh@MXFP4", 4.25, (I, e4, I)),
    ("only-Wo@MXFP4", 4.25, (I, I, e4)),
    ("only-Wi@int4g32", 4.25, (i4g, I, I)),
    ("only-Wh@int4g32", 4.25, (I, i4g, I)),
    ("only-Wo@int4g32", 4.25, (I, I, i4g)),
    ("only-Wi@int2g32", 2.25, (i2g, I, I)),
    ("only-Wh@int2g32", 2.25, (I, i2g, I)),
    ("only-Wo@int2g32", 2.25, (I, I, i2g)),
    # protect W_i
    ("Wi@MXFP8_rest@MXFP4", 4.33, (e8, e4, e4)),
    ("Wi@bf16_rest@MXFP4", 4.49, (BF16, e4, e4)),
    ("Wi@MXFP8_rest@MXFP6", 6.29, (e8, e6, e6)),
    ("Wi@MXFP8_rest@int4g32", 4.33, (e8, i4g, i4g)),
    ("Wi@MXFP8_rest@int3g32", 3.35, (e8, i3g, i3g)),
    ("Wi@MXFP8_rest@int2g32", 2.37, (e8, i2g, i2g)),
    ("Wi+Wo@MXFP8_Wh@MXFP4", 6.18, (e8, e4, e8)),
    # protect W_h instead (the prior study's original, wrong, guess)
    ("Wh@MXFP8_rest@MXFP4", 6.17, (e4, e8, e4)),
    ("Wh@MXFP6_rest@MXFP4", 5.21, (e4, e6, e4)),
]

# ---------------- (C) environment-cache / activation quantization ---------------- #
C = [
    ("cache-fp16", 16.0, FP16),
    ("cache-bf16", 16.0, BF16),
    ("cache-MXFP8-e4m3", 8.25, e8),
    ("cache-MXFP6-e3m2", 6.25, e6),
    ("cache-MXFP4-e2m1", 4.25, e4),
    ("cache-NVFP4", 4.50, nv),
    ("cache-int8-perrow", 8.00, i8c),
    ("cache-int8-g32", 8.25, i8g),
    ("cache-int6-g32", 6.25, i6g),
    ("cache-int4-perrow", 4.00, i4c),
    ("cache-int4-g32", 4.25, i4g),
    ("cache-int2-g32", 2.25, i2g),
]

ALL = ([("A", n, b, "w", s) for n, b, s in A]
       + [("B", n, b, "w", s) for n, b, s in B]
       + [("C", n, b, "cache", s) for n, b, s in C])
