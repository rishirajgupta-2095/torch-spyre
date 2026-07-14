# Isolation diagnostic for spyre.scaled_bmm (FP8 batched matmul).
# Run on a Spyre-enabled machine:  python scaled_bmm_smoke.py
#
# Goal: pin down WHERE batched FP8 goes wrong by testing each piece alone,
# with two known-good controls (2D FP8 matmul, and fp16 batched matmul).
import torch
import torch_spyre  # noqa: F401  registers "spyre" device + custom ops + lowerings

DEVICE = "spyre"
FP8 = torch.float8_e4m3fn


def report(name, out, ref):
    out = out.float().cpu()
    ref = ref.float().cpu()
    diff = (out - ref).abs()
    tol = 0.1 + 0.1 * ref.abs()
    n_bad = (diff > tol).sum().item()
    print(
        f"[{name}] shape={tuple(out.shape)} ok={n_bad == 0} "
        f"max_diff={diff.max().item():.3f} n_over_tol={n_bad}/{diff.numel()} "
        f"ref_mean={ref.abs().mean().item():.3f}"
    )
    print(f"        out[:4]={[round(v, 2) for v in out.flatten()[:4].tolist()]}  "
          f"ref[:4]={[round(v, 2) for v in ref.flatten()[:4].tolist()]}")


def compile_run(fn, *tensors):
    torch._dynamo.reset()
    c = torch.compile(fn, dynamic=False)
    return c(*[t.to(DEVICE) for t in tensors])


# ---------------------------------------------------------------------------
# Control A: 2D FP8 _scaled_mm (the only path with existing tests). Should pass.
# ---------------------------------------------------------------------------
def ctrl_scaled_mm():
    a = torch.rand(64, 128, dtype=torch.float16)
    b = torch.rand(128, 64, dtype=torch.float16)
    sa = torch.tensor(1.0, dtype=torch.float16)
    sb = torch.tensor(1.0, dtype=torch.float16)

    def fn(a, b, sa, sb):
        qa = torch.ops.spyre.quantize_fp8_with_scale(a, sa)
        qb = torch.ops.spyre.quantize_weight_fp8_with_scale(b, sb)
        return torch.ops.aten._scaled_mm(qa, qb, sa, sb, bias=None,
                                         out_dtype=torch.float16)

    qa = a.clamp(-448, 448).to(FP8).to(torch.float16)
    qb = b.clamp(-448, 448).to(FP8).to(torch.float16)
    report("A: 2D scaled_mm (control)", compile_run(fn, a, b, sa, sb), qa @ qb)


# ---------------------------------------------------------------------------
# Control D: fp16 batched matmul (torch.bmm), 3D. Tests batch handling WITHOUT
# FP8. If this is correct, the batch-loop codegen itself is fine.
# ---------------------------------------------------------------------------
def ctrl_fp16_bmm():
    a = torch.rand(4, 64, 128, dtype=torch.float16)
    b = torch.rand(4, 128, 64, dtype=torch.float16)
    report("D: 3D fp16 bmm (control)", compile_run(torch.bmm, a, b), a @ b)


# ---------------------------------------------------------------------------
# Test B: quantize ops alone on a 3D input (dequantize back to fp16). Isolates
# whether the FP8 quantize ops mishandle a batched tensor.
# ---------------------------------------------------------------------------
def test_quant_only():
    a = torch.rand(4, 64, 128, dtype=torch.float16)
    s = torch.tensor(1.0, dtype=torch.float16)

    def fn(a, s):
        qa = torch.ops.spyre.quantize_fp8_with_scale(a, s)
        return torch.ops.spyre.dequantize_fp8_with_scale(qa, s)

    ref = a.clamp(-448, 448).to(FP8).to(torch.float16)
    report("B: 3D quantize-only", compile_run(fn, a, s), ref)


# ---------------------------------------------------------------------------
# Test C: scaled_bmm alone, fed inputs that are ALREADY fp8 (quantized on CPU).
# No quantize op in the graph -> isolates the scaled_bmm lowering/codegen.
# ---------------------------------------------------------------------------
def test_bmm_only(name, a_shape, b_shape):
    a = torch.rand(a_shape, dtype=torch.float16).clamp(-448, 448).to(FP8)
    b = torch.rand(b_shape, dtype=torch.float16).clamp(-448, 448).to(FP8)
    sa = torch.tensor(1.0, dtype=torch.float16)
    sb = torch.tensor(1.0, dtype=torch.float16)

    def fn(a, b, sa, sb):
        return torch.ops.spyre.scaled_bmm(a, b, sa, sb, out_dtype=torch.float16)

    ref = a.float() @ b.float()
    report(f"C: scaled_bmm-only {name}", compile_run(fn, a, b, sa, sb), ref)


def guard(fn, *args):
    try:
        fn(*args)
    except Exception as e:  # noqa: BLE001
        msg = str(e).splitlines()[0] if str(e) else type(e).__name__
        print(f"[{getattr(fn, '__name__', fn)}{args}] FAILED: {type(e).__name__}: {msg}")


if __name__ == "__main__":
    guard(ctrl_scaled_mm)
    guard(ctrl_fp16_bmm)
    guard(test_quant_only)
    guard(test_bmm_only, "3D_B1", (1, 64, 128), (1, 128, 64))
    guard(test_bmm_only, "3D_B4", (4, 64, 128), (4, 128, 64))
    guard(test_bmm_only, "4D", (1, 32, 64, 128), (1, 32, 128, 64))
