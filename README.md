# 6502-TRANSFORMER

A tiny transformer with a **full forward + backward pass**, written in
**assembly only** — for two vintage CPUs that were never meant to run one.

- **6502** — the MOS 6502 (NES, C64, Apple II).
- **AGC** — the Apollo Guidance Computer (lunar descent guidance).

Same model, same task, two architectures. Ternary weights everywhere.
No Python. No C. No libraries. Just assembly.

## The model

- d_model = 4, seq_len = 2, vocab = 4, 1 layer, 1 attention head, FFN = 4.
- All weights **ternary** — values in {-1, 0, 1}.
- Task: tokens `[0, 1]` → predict token `2`. Loss = MSE on the last
  token's logits vs a one-hot target.
- Training: SGD with **latent weights**. Each weight keeps a higher-precision
  latent value; the update is `latent -= grad >> 2` (lr = 1/4), clamped,
  then re-quantized to {-1, 0, 1} with a dead zone (|latent| ≤ 16 → 0).
  The dead zone is what produces sparsity.
- Deterministic. Fixed tiny dataset. Fixed initial weights (in the source).

## The forward pass (both CPUs)

1. **Embedding** — lookup: `X[t] = Emb[token_t]`.
2. **Q / K / V** — each is a ternary dot product (the user's kernel):
   `Q = Wq · X`, `K = Wk · X`, `V = Wv · X`.
3. **Attention scores** — `S[i][j] = Q[i] · K[j]` (general dot product).
4. **Softmax** — max-subtract, exp lookup table, normalize.
5. **Attention output** — `O[i] = Σ_j softmax[i][j] · V[j]` (weighted sum).
6. **FFN + residual** — `H1 = ReLU(W1 · O)`, `H = O + W2 · H1`.
7. **Logits** — `Logits = Wo · H` (ternary dot product).
8. **Loss** — MSE vs the one-hot target.

## The backward pass (both CPUs)

Gradients flow back through every layer:

- `dLogits = 2 · (Logits − Target)` (last token only).
- `dWo` (outer product) → `dH` (transpose dot via gathered columns).
- `dW2` → `dH1` (transpose dot) → **ReLU mask** (gradient zeroed where
  `H1 ≤ 0`) → `dW1` → `dO` (residual + FFN path).
- Attention backward: `dAttnW` and `dV` (weighted sums), the
  **softmax Jacobian**, then `dQ` and `dK`.
- `dWq`, `dWk`, `dWv` (outer products with `X`), `dX` (transpose dots
  from all three projections), `dEmb` (accumulated per token).
- **SGD update** — latent, clamp, re-quantize, store (all weight matrices).

Only the last token contributes to the loss, so backprop starts at
`Logits[1]`. Cross-token gradients still flow: `O[1]` uses `V[0]` and
`V[1]`, and `S[1][*]` uses `K[0]` and `K[1]`.

## What is omitted (honest gaps)

- **LayerNorm is omitted** on both CPUs. The residual + ReLU carry the
  signal. LayerNorm (mean, variance, sqrt, divide) is expensive on both
  architectures and is a documented gap, not a faked one.
- **AGC embedding is frozen** — `G_X` (embedding gradients) is computed,
  but the AGC `EMB` table is not updated by SGD. The 6502 embedding
  *is* updated. This asymmetry is documented in `agc/backward.asm`.
- **Fixed-point approximations** — softmax uses an exp table; attention
  rescaling uses shifts (`>>8` for the attention output, `>>4` for score
  gradients). Gradient *directions* are exact; magnitudes are approximate.
  Correct for a tiny demo transformer trained by SGD.
- **Loss may saturate** (16-bit accumulator) for large logits. In practice
  logits stay small.

## Status

**Written, not executed.** There is no 6502 emulator and no AGC emulator
on the build VM. The code is written carefully, cross-checked by reading
(label and memory-map checks), and structured to assemble — but it has
not been assembled or run. Do not claim it runs until it has been
assembled and executed and the loss decreases across training iterations.

## Repo map

| Path | What it is |
|------|-----------|
| `6502/ternary_dot.asm` | **The 6502 kernel.** User-supplied ternary dot product, verbatim. |
| `6502/math.asm` | MUL8, DOT8, ReLU, quantize, clamp, softmax (exp table), gather-column, ternary-call wrapper. |
| `6502/forward.asm` | 6502 forward pass. |
| `6502/backward.asm` | 6502 backward pass + SGD update. |
| `6502/main.asm` | 6502 entry, INIT (ROM→RAM weights, latent = ternary×64), data, 8-iteration training loop. |
| `6502/memmap.asm` | 6502 memory map (activations, gradients, counters). |
| `agc/ternary_squib.asm` | **The AGC kernel.** User-supplied ternary squib, verbatim. |
| `agc/math.asm` | AGC helpers (MUL8, DIV, exp lookup, ReLU, clamp, quantize). |
| `agc/forward.asm` | AGC forward pass. |
| `agc/backward.asm` | AGC backward pass + SGD update. |
| `agc/main.agc` | AGC entry, memory map, data tables, training loop (3 iterations). |

## Building

**6502** — ca65 syntax (cc65 toolchain). Assemble and link with your
config (the code uses `.segment` / `.org`; provide a linker config that
defines `CODE` at `$8000` and `VECTORS` at `$FFFA`), then run on a 6502
emulator or hardware. Watch `LOSS` (`$0304`, 16-bit) — it should decrease
across the 8 training iterations.

**AGC** — AGC assembly (yaAGC syntax). Assemble with yaAGC and run on an
AGC emulator. The training loop runs 3 iterations (forward → backward).

## Why ternary?

The 6502 has no multiply. The AGC has no multiply. A ternary weight
(-1, 0, 1) turns a dot product into **add / subtract / skip**:

- `W == 0` → skip the element entirely (sparsity = speed).
- `W == 1` → add `X`.
- `W == -1` → subtract `X`.

No multiply instruction needed. That's the whole trick — and it's why
a transformer fits on CPUs that were never meant to run one.

## License

SNAPKITTY OPAQUE SOURCE LICENSE v1.0. See `LICENSE`.
