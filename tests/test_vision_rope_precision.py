"""
Regression test for the Qwen2/Qwen2.5-VL vision-tower RoPE table construction
in exllamav2/vlm/vision_tower.py.

Bug: the position/frequency outer product used to build self.rope_cos /
self.rope_sin for the "qwen2"/"qwen2.5" vision_model_type branch was computed
entirely in float16 (inv_freq cast to .half() *before* the outer product with
the position index tensor `s`, which was itself built directly in float16).
float16 has only ~3 decimal digits of precision, so for position indices in
the hundreds-to-low-thousands range (well within the default
vision_max_size=16384 / spatial_patch_size=14 -> max_seqlen=1171 config used
by real Qwen2-VL/Qwen2.5-VL checkpoints) the RoPE phase angles lose enough
precision to produce a visibly wrong cos/sin table -- max abs error >0.33 in
this test, far above float16 rounding noise.

The sibling "pixtral" branch in the same __init__ does this correctly: it
keeps the position-index / frequency outer product in float32 and only casts
to half at the very end (`.cos().half()` / `.sin().half()`). The fix makes
the qwen2/qwen2.5 branch follow the same convention.

This result then feeds exllamav2/vlm/processor/qwen2.py:position_embeddings,
whose cos/sin tensors are consumed directly by the raw CUDA extension op
`ext_c.rope_(...)` in exllamav2/attn.py -- there is no Python-level dtype
promotion on that path, so the float16-outer-product error propagates
unmodified into the vision attention RoPE application.

This test does not require model weights or a GPU; it only exercises the
tensor-math construction that vision_tower.py performs.
"""
import math
import torch


def build_qwen2_vision_rope(dim: int, max_seqlen: int, rope_theta: float, keep_fp32: bool):
    """Mirrors exllamav2/vlm/vision_tower.py's qwen2/qwen2.5 RoPE table build."""
    if keep_fp32:
        # Fixed (post-fix) construction: stay in float32 through the outer
        # product, matching the pixtral branch's own convention; cast once.
        inv_freq = 1.0 / (rope_theta ** (torch.arange(0, dim, 2).float() / dim))
        s = torch.arange(max_seqlen, dtype=torch.float32)
        inv_freq = torch.outer(s, inv_freq)
    else:
        # Pre-fix (buggy) construction.
        inv_freq = 1.0 / (rope_theta ** (torch.arange(0, dim, 2).float() / dim)).half()
        s = torch.arange(max_seqlen, dtype=inv_freq.dtype).half()
        inv_freq = torch.outer(s, inv_freq)

    rope_cos = inv_freq.cos().half()
    rope_sin = inv_freq.sin().half()
    return rope_cos, rope_sin


def test_qwen2_vision_rope_precision():
    # Matches a real Qwen2-VL / Qwen2.5-VL vision_config:
    # vision_head_dim=80 -> dim=40, vision_max_size=16384, spatial_patch_size=14.
    dim = 40
    rope_theta = 10000.0
    max_seqlen = int(math.ceil(16384 / 14))  # 1171, well under float16's 2048 exact-int limit

    buggy_cos, buggy_sin = build_qwen2_vision_rope(dim, max_seqlen, rope_theta, keep_fp32=False)
    fixed_cos, fixed_sin = build_qwen2_vision_rope(dim, max_seqlen, rope_theta, keep_fp32=True)

    # Reference computed independently in float64 and only cast down once.
    ref_inv_freq = 1.0 / (rope_theta ** (torch.arange(0, dim, 2, dtype=torch.float64) / dim))
    ref_s = torch.arange(max_seqlen, dtype=torch.float64)
    ref_outer = torch.outer(ref_s, ref_inv_freq)
    ref_cos = ref_outer.cos().half()
    ref_sin = ref_outer.sin().half()

    buggy_err = (buggy_cos.float() - ref_cos.float()).abs().max().item()
    fixed_err = (fixed_cos.float() - ref_cos.float()).abs().max().item()

    # Pre-fix: the all-half-precision outer product diverges substantially
    # from the float64 reference even within the default config's max_seqlen.
    assert buggy_err > 0.1, (
        f"expected the pre-fix construction to show a large precision error, got {buggy_err}"
    )
    # Post-fix: staying in float32 through the outer product and casting once
    # at the end should match ordinary float16-rounding noise only.
    assert fixed_err < 1e-2, (
        f"fixed construction should closely match the float64 reference, got {fixed_err}"
    )

    print(f"buggy max abs cos error vs fp64 reference: {buggy_err:.4f}")
    print(f"fixed max abs cos error vs fp64 reference: {fixed_err:.4f}")


if __name__ == "__main__":
    test_qwen2_vision_rope_precision()
    print("OK")
