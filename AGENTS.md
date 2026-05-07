# AGENTS.md

Guidelines for agents (and humans) contributing to this crate.

---

## What This Crate Is

A from-scratch tensor + automatic differentiation library, built for learning. It is not intended for production. The goal is a clean, readable implementation that makes the math legible — not a performant one.

---

## Architecture

### Module Layout

```
src/
  lib.rs          — crate root; re-exports public surface; houses chain-rule notes
  shape.rs        — TensorShape: dimension semantics and shape-compatibility logic
  tensor.rs       — Tensor trait + TensorOutput + error types
  tensor/
    cpu.rs        — CPUTensor: naive CPU backend (row-major, no SIMD, no CUDA)
```

New backends (CUDA, SIMD, etc.) belong under `tensor/` as sibling modules to `cpu.rs`. Each backend is a concrete type implementing the `Tensor` trait — no backend-specific logic should leak into the trait itself.

### Key Types

| Type | Role |
|---|---|
| `TensorShape` | Encodes dimension layout. Treats the last 1–2 dims as the matrix/vector shape; leading dims are batch dims. |
| `Tensor` | Backend-agnostic trait. Defines the op surface every backend must support. |
| `TensorOutput<T>` | A `Result`-like enum returned from fallible multi-var ops. Exists so operator overloading chains stay ergonomic without fighting Rust's orphan rules. |
| `CPUTensor` | The reference backend. Flat row-major `Vec<f32>`, no acceleration. |

---

## Shape Semantics

- The last two dimensions are always the "core" shape (height × width for matrices, or just length for vectors).
- Every dimension before that is a batch dimension.
- Broadcasting is **left-to-right only**: a batched left operand may broadcast over an unbatched right operand, never the reverse. This is an intentional simplification — bidirectional NumPy-style broadcasting is out of scope.
- Only 1D (vector) and 2D (matrix) core shapes are supported. Higher-rank contraction is not a goal.

---

## Gradient Calculation Plan

The end goal is reverse-mode automatic differentiation via the chain rule:

```
f(x)  = a(b(c(d(e(x)))))
f'(x) = a'(b(c(d(e(x))))) · b'(c(d(e(x)))) · c'(d(e(x))) · d'(e(x)) · e'(x)
```

Each composed operation contributes its local derivative; the full gradient is the product of all local derivatives along the path from output back to input.

To support this, the planned approach is:

1. **Computation graph** — every op records its input tensors and the op type that produced the output. This forms a DAG.
2. **Backward pass** — calling `.backward()` on an output tensor walks the DAG in reverse topological order, accumulating gradients via the chain rule at each node.
3. **Local derivative per op** — each operation (`relu`, `mul`, `add`, etc.) must be paired with its backward rule (e.g., ReLU's backward is a mask: `grad * (x > 0)`).
4. **Gradient accumulation** — nodes with multiple outgoing edges (used more than once) accumulate gradients before propagating further backward.

The `Tensor` trait will eventually need backward-facing counterparts for each op, or a separate `grad_fn` mechanism attached to the output. The exact shape of this API is still open — defer to the code when it exists.

---

## Operations

Operations on `Tensor` are split into two categories:

- **Single-var ops** (`relu`, `softmax`, `neg`): element-wise, always shape-preserving, infallible. They return `Self` directly.
- **Multi-var ops** (`add`, `sub`, `mul`): require shape compatibility checks. They return `TensorOutput<Self>` to surface shape errors without panicking.

`sub` is implemented as `add(neg(other))` — keep that delegation; don't duplicate arithmetic.

---

## Error Handling

- `TensorOutput<T>` is the return type for fallible ops, not `Result`. Treat it like one.
- `TensorOpError` carries the operation that failed *and* the specific error variant. Both fields must be populated — don't throw away the operation context when re-using another op's error (see `sub` mutating the error's `operation` field after delegating to `add`).
- Don't add new error variants speculatively. Add them when the failure case actually exists.

---

## Code Standards

### Documentation

Write intent, not mechanics. Anyone reading the code can see *what* it does. Comments and doc strings should explain *why* — the constraint being enforced, the tradeoff being made, the invariant being maintained.

**Good:**
```rust
// Broadcasting is only supported from left to right.
// Right side may only have 2 dimensions.
right_batch_dims.is_empty()
```

**Avoid:**
```rust
// Returns true if right_batch_dims is empty
right_batch_dims.is_empty()
```

Doc comments on public items should be a single sentence where possible. If a function has a meaningful precondition or an invariant it relies on, state it.

### Style

- Keep operations lean. Single-purpose functions with clear names beat multi-purpose ones with flags.
- If a safety assumption comes from a previous check (e.g., `matches_mul_size_op` guarantees the operands are matrices), name that check in the comment at the use site so readers know why indexing `[0]` and `[1]` can't panic.
- Prefer flat data with computed index arithmetic over nested `Vec<Vec<...>>`. All tensor storage is flat row-major.
- `allow(dead_code)` is acceptable on internal enums during development, but should not accumulate indefinitely.

### Testing

- Tests for shape logic belong in `shape.rs`.
- Tests for op correctness belong in `tensor/cpu.rs` (or the relevant backend file).
- Cover the boundary: zero-dim batches, 1D vectors, mismatched shapes, and broadcasting edge cases.
