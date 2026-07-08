# FP8 Batched Matrix Multiplication (3D × 2D)

This document describes how `torch-spyre` supports `aten._scaled_mm` with a
3D activation tensor paired with a 2D weight tensor — a shape pattern that
appears in quantized LLM projection layers (e.g. Granite).

## Background

PyTorch's `aten._scaled_mm` is the FP8 scaled matrix multiplication op used
by quantized transformer models. Its canonical form is:

```
_scaled_mm(mat1, mat2, scale_a, scale_b, bias, out_dtype) -> Tensor
```

In practice, LLM inference passes activations with a batch dimension:

```
mat1: [batch, seq_len, hidden]   # 3D — activations
mat2: [hidden, hidden]           # 2D — pre-quantized weight
```

Upstream PyTorch enforces that both inputs are 2D.
torch._scaled_mm only accepts 2D tensors ([M, K] × [K, N]). This is hard-enforced in PyTorch's C++ backend with a TORCH_CHECK on dim() == 2. all examples show shapes like (16, 16), (M, K), (K, N), never 3D.

On Spyre
the hardware FP8 kernel (`BATCH_MATMUL_FP8_OP`) also requires 2D inputs, but
this is a hardware layout constraint, not a mathematical one — the batched
case is expressible as a 2D operation via flattening.

## The problem: two separate rejection points

A naive attempt to compile `_scaled_mm([B,M,K], [K,N])` on Spyre hits two
distinct failures at different pipeline stages:

### Stage 1 — Dynamo tracing (FakeTensor shape propagation)

During `torch.compile`, Dynamo traces the user function symbolically. For
every op it encounters, it calls `get_fake_value` which runs the op through
PyTorch's **Meta dispatch key** to compute output shapes. The Meta kernel for
`_scaled_mm` is `meta_scaled_mm` in `torch/_meta_registrations.py`, which
calls `_check_scaled_mm_sizes`:

```python
# torch/_meta_registrations.py
def _check_scaled_mm_sizes(self, mat2, ...):
    torch._check(
        self.dim() == 2 and mat2.dim() == 2,   # ← fails for 3D mat1
        lambda: f"Inputs must be 2D ..."
    )
```

This fails before any FX graph is produced. No Inductor pass can help here —
they all run after tracing completes.

### Stage 2 — Hardware kernel (post-lowering)

Even if tracing succeeded, `lower_scaled_mm` in `lowering.py` would emit a
`BATCH_MATMUL_FP8_OP` with 3D ranges, which the Spyre kernel rejects at
runtime because it only operates on 2D buffers.

Both stages must be addressed.

## Solution: two-layer fix

### Layer 1 — unblock Dynamo (`_monkey_patch.py`)

`meta_scaled_mm` resolves `_check_scaled_mm_sizes` as a module global at call
time. Replacing that global after import is sufficient — the dispatcher
registration does not need to change.

`_patch_scaled_mm_meta_for_3d()` (called from `_patch_tensor_for_spyre()` at
import time) installs a wrapper:

```python
def _patched_check_scaled_mm_sizes(self, mat2, scale_a, scale_b,
                                    bias=None, scale_result=None,
                                    out_dtype=None, use_fast_accum=False):
    if self.dim() == 3 and mat2.dim() == 2:
        B, M, K = self.shape
        K2, N = mat2.shape
        torch._check(K == K2, ...)                         # validate contraction dim
        dtype = out_dtype if out_dtype is not None else self.dtype
        return self.new_empty((B, M, N), dtype=dtype)      # correct meta output shape
    return _orig_check(self, mat2, ...)                    # 2D and all else unchanged
```

This returns the correct output shape `[B, M, N]` to Dynamo so tracing
succeeds. No actual computation or data movement occurs here — it is pure
shape bookkeeping.

### Layer 2 — make it executable (`temp_passes.py` + `passes.py`)

After tracing, the FX graph contains a `_scaled_mm` node with a 3D first
input. `flatten_3d_scaled_mm_pass` (registered first in `CustomPostPasses`)
rewrites it to a hardware-compatible 2D form:

```
BEFORE:
  mat1 [B, M, K] ──┐
  mat2 [K, N]    ───┤── _scaled_mm ──→ [B, M, N]
  scale_a, ...  ────┘

AFTER:
  mat1 [B, M, K] ── view ──→ [B*M, K] ──┐
  mat2 [K, N]    ─────────────────────────┤── _scaled_mm ──→ [B*M, N] ── view ──→ [B, M, N]
  scale_a, ...  ──────────────────────────┘
```

The scale tensors are per-tensor scalars and pass through unchanged. The
2D `_scaled_mm` node then follows the existing `lower_scaled_mm` 2D path and
emits a valid `BATCH_MATMUL_FP8_OP`.

The `meta["val"]` field is set explicitly on every inserted node so all
subsequent passes (layout propagation, work division, padding) see correct
shapes.

### Pipeline flow

```
import torch_spyre
  └── _patch_tensor_for_spyre()
        └── _patch_scaled_mm_meta_for_3d()   ← installs wrapper

torch.compile(model)(activations_3d, weight_2d, ...)
  │
  ├── [Dynamo tracing]
  │     _scaled_mm seen → FakeTensor → Meta dispatch
  │       └── _check_scaled_mm_sizes (patched)
  │             └── returns new_empty([B,M,N])    ← tracing succeeds ✓
  │
  └── [Inductor — CustomPostPasses]
        └── flatten_3d_scaled_mm_pass
              └── view([B,M,K]→[B*M,K])
                   + _scaled_mm(2D,2D)            ← 2D lowering ✓
                   + view([B*M,N]→[B,M,N])        ← hardware kernel ✓
```

## Comparison with upstream PyTorch

Upstream PyTorch does not support 3D `_scaled_mm`. The 2D restriction is
baked into the op's contract for CUDA (cuBLAS constraint). Spyre relaxes this
transparently: user model code calls `aten._scaled_mm` normally; the backend
handles the shape adaptation internally.

| | Upstream PyTorch | torch-spyre |
|---|---|---|
| `_scaled_mm(2D, 2D)` | supported | supported, identical behaviour |
| `_scaled_mm(3D, 2D)` | `RuntimeError: Inputs must be 2D` | supported via graph rewrite |
| Shape inference | `_check_scaled_mm_sizes` enforces 2D | wrapper returns `[B,M,N]` for 3D×2D |
| Kernel call | GPU cuBLAS (2D) | Spyre FP8 kernel (2D), flattened |

## Prior art

The graph-pass pattern (`view → op → view`) is well established in this
codebase. `_unflatten_mm_to_bmm` in `temp_passes.py` uses the same structure
to adapt `mm` for the `bmm` hardware path. `flatten_3d_scaled_mm_pass`
deliberately mirrors it.

The module-global replacement pattern for Meta kernels is used by other
out-of-tree PyTorch backends (`torch_xla`, `torch-mlir`) that need to extend
aten op behaviour for hardware that handles shapes differently from CUDA.
The official in-tree mechanism for new ops is `@torch.library.register_fake`;
for existing aten ops in out-of-tree backends the module-global replacement
is the established alternative.

## Before / after example

```python
import torch
import torch_spyre

BATCH, SEQ, HIDDEN = 2, 128, 4096   # typical Granite projection shapes

def fp8_proj(activations, weight, scale_a, scale_w):
    q_a = torch.ops.spyre.quantize_fp8_with_scale(activations, scale_a)
    q_w = torch.ops.spyre.quantize_weight_fp8_with_scale(weight, scale_w)
    return torch.ops.aten._scaled_mm(
        q_a, q_w, scale_a, scale_w, bias=None, out_dtype=torch.float16
    )

activations = torch.randn(BATCH, SEQ, HIDDEN, dtype=torch.float16, device="spyre")
weight      = torch.randn(HIDDEN, HIDDEN,      dtype=torch.float16, device="spyre")
scale_a     = torch.tensor([1.0], dtype=torch.float16, device="spyre")
scale_w     = torch.tensor([1.0], dtype=torch.float16, device="spyre")

# Before this fix: raises RuntimeError during torch.compile
# After this fix:
out = torch.compile(fp8_proj)(activations, weight, scale_a, scale_w)
print(tuple(out.shape))   # (2, 128, 4096)
print(out.dtype)           # torch.float16
```

## Alternative: `torch.ops.spyre.scaled_mm` custom op

A complementary approach is a Spyre-specific custom op that accepts N-D
inputs natively:

```python
torch.ops.spyre.scaled_mm(activations_3d, weight_2d, scale_a, scale_w,
                           out_dtype=torch.float16)
```

**Advantages over the current approach:**

- Uses `@torch.library.register_fake` — the official shape-inference API.
  No monkey-patching of PyTorch internals.
- N-D generalizes trivially: `output_shape = (*self.shape[:-1], mat2.shape[1])`.
- Shape handling lives in the lowering, not in a separate FX pass.
- Decoupled from the internal structure of `_meta_registrations.py`.

**Limitation:**

Existing models (Granite, HuggingFace FP8 paths) call `aten._scaled_mm`
directly via PyTorch's quantization stack. A `spyre.scaled_mm` op requires
either a model-side change or a decomposition that routes
`aten._scaled_mm(3D, ...)` → `spyre.scaled_mm(3D, ...)` inside Inductor.
That decomposition runs post-grad, which still requires the Layer 1 patch to
let Dynamo accept the 3D input first. The current approach therefore remains
necessary for transparent `aten._scaled_mm` compatibility.

**Recommendation:** keep the current approach for backward-compatible model
support; add `torch.ops.spyre.scaled_mm` as a clean forward-looking public
API for new Spyre-specific code. The two are complementary, not mutually
exclusive.

## Files changed

| File | Change |
|---|---|
| `torch_spyre/_monkey_patch.py` | `_patch_scaled_mm_meta_for_3d()` — wraps `_check_scaled_mm_sizes` |
| `torch_spyre/_inductor/temp_passes.py` | `flatten_3d_scaled_mm_pass` — FX graph rewrite |
| `torch_spyre/_inductor/passes.py` | registers pass first in `CustomPostPasses` |
| `tests/inductor/test_inductor_ops.py` | 3D numerical correctness params |
| `tests/inductor/test_inductor_fx_passes.py` | 3D graph-structure params + output shape assertion |

## Follow-ups

- Remove the now-dead 3D branch in `lower_scaled_mm` (or replace with
  `Unsupported`) so nothing silently relies on the broken 3D-kernel path.
- Implement `torch.ops.spyre.scaled_mm` as the clean public API for N-D FP8
  matmul on Spyre.
- Generalize `flatten_3d_scaled_mm_pass` to N-D (`shape[:-1]`) when a model
  requires it.
