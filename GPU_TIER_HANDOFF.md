# GPU tier stage 2 — validation handoff

This branch (`gpu-tier-overhaul`) reworks the CUDA backend from a synchronous
correctness-first stub into a usable acceleration tier. It was written and
compile-checked in an environment **without a GPU and without the model**, so
nothing here has been executed on real hardware yet. This document is the
instruction set for a Claude instance (or a human) running on the target
machine to build, validate, benchmark, and report.

**Target machine assumptions:** RTX 5090 (32 GB), 192 GB DDR5, fast NVMe with
the ~370 GB GLM-5.2 int4 container, Linux, CUDA toolkit ≥ 12.x installed.

## What changed

| File | Change |
|---|---|
| `c/backend_cuda.cu` | Rewritten. Warp-per-row matvec kernel (vectorized `uint32`/`float4` weight loads, warp-shuffle reduction) for S<4; tiled batch kernel (weight row fetched once, reused for 8 positions) for S≥4; one non-blocking stream per device; pinned-host + device bump arenas; three entry points (see below). Scalar fallback kept bit-compatible for odd shapes. |
| `c/backend_cuda.h` | New API: `coli_cuda_matmul_stream` (synchronous, weights streamed through persistent scratch — for prefill batches), `coli_cuda_ffn_enqueue` (async full expert FFN on the device stream), `coli_cuda_sync`. |
| `c/glm.c` | `moe()`: VRAM-pinned experts are now **enqueued asynchronously** and the CPU computes the remaining experts while the GPU works (previously the GPU serialized into the critical path); results collected at an end-of-block sync. `matmul_qt()`: batches of ≥ `CUDA_PREFILL_ROWS` rows on non-resident tensors stream weights over PCIe once and run on the GPU — this accelerates prefill (dense projections, shared experts, and batch-union routed experts all pass through this seam). `repin_pass()` drains the stream before freeing device weights. New env knobs + stats. |
| `c/tests/test_backend_cuda.cu` | Extended: tile kernel vs CPU reference on all 4 formats, streamed matmul, scalar-fallback shapes, async FFN (out-of-order row gather, double enqueue + single sync, arena reuse across syncs, dimension-mismatch rejection, leak check). References are computed in-test; inputs are exact binary fractions so any summation order must match. |

New environment knobs (all existing ones unchanged):

- `CUDA_PREFILL=0` — disable the batch weight-streaming path (default on when `COLI_CUDA=1`).
- `CUDA_PREFILL_ROWS=n` — minimum batch rows before weights are streamed through PCIe (default 16).

## What you (local Claude) need to do

Work through the stages **in order** — each gates the next. If a stage fails,
stop, capture the exact output, and report (see "Reporting" below) rather than
pushing on to benchmarks built on a broken base.

### Stage 0 — environment sanity

```bash
nvidia-smi                       # GPU visible, driver OK
nvcc --version                   # toolkit present (CUDA_HOME=/usr/local/cuda assumed by the Makefile)
gcc --version                    # OpenMP-capable gcc
```

### Stage 1 — build + unit correctness (minutes)

```bash
cd c
make clean && make               # CPU-only build must stay green
make cuda-test CUDA=1            # kernel/API correctness ON YOUR GPU
make CUDA=1                      # full engine with the backend
```

`cuda-test` must print `cuda backend: q8/q4/q2/f32 matvec+tile+stream+ffn ok`.
This is the single most important gate: it exercises every new kernel and the
async FFN path against CPU references. A mismatch here is a real bug — check
first whether it's the vectorized loads (re-run mentally against `weight_at`
semantics: int4 low-nibble-first, int2 2-bit LSB-first) and report the failing
format + shape.

### Stage 2 — architecture correctness on the tiny oracle (minutes)

The repo ships a token-exactness harness against a tiny random model with the
real GLM architecture:

```bash
pip install torch transformers safetensors huggingface_hub
python tools/make_glm_oracle.py                    # generates ./glm_tiny
SNAP=./glm_tiny TF=1 ./glm 64 16 16                # expect "32/32 posizioni"
# now the same, forcing every GPU path on:
COLI_CUDA=1 COLI_GPU=0 CUDA_DENSE=1 CUDA_PREFILL_ROWS=1 SNAP=./glm_tiny TF=1 ./glm 64 16 16
```

Both runs must report 32/32. Note: the GPU computes f32×dequant while the CPU
idot path quantizes activations to int8, so logits can differ in the last ulps
— teacher-forcing compares argmax, which must still match. If the second run
drops below 32/32, bisect with `CUDA_PREFILL=0` (isolates the streaming path)
and `CUDA_DENSE=0` (isolates the resident dense path) and report which knob
flips it.

### Stage 3 — fixture A/B benchmark, no full model needed (tens of minutes)

```bash
python tools/make_glm_bench_model.py --output /nvme/colibri-bench-medium --device cuda
python tools/benchmark_cuda_fixture.py --model /nvme/colibri-bench-medium --gpu 0
```

This replays fixed tokens through the 313M fixture and compares CPU streaming
vs CUDA tiers. On this branch the CUDA hot-expert configuration should now
beat its own pre-branch numbers (the old backend measured neutral-to-negative;
if you have `main` checked out too, run the fixture there once for a direct
before/after).

### Stage 4 — real model (hours; the actual deliverable)

All runs: `COLI_MODEL=/nvme/glm52_i4`, int8 MTP head installed, same prompt
set. Capture the full per-turn stats lines.

```bash
# A. CPU baseline (warm cache, learned pin — run twice, report 2nd)
./coli chat --ngen 32 --temp 0

# B. + async VRAM expert tier (the 5090 holds dense projections + hot experts)
COLI_CUDA=1 COLI_GPU=0 CUDA_EXPERT_GB=24 PIN=stats.txt PIN_GB=140 ./coli chat --ngen 32 --temp 0

# C. + GPU prefill: paste/pipe a LONG prompt (≥ 2000 tokens) and time prefill
COLI_CUDA=1 COLI_GPU=0 CUDA_EXPERT_GB=24 ./coli run --ngen 8 < long_prompt.txt
CUDA_PREFILL=0 COLI_CUDA=1 COLI_GPU=0 CUDA_EXPERT_GB=24 ./coli run --ngen 8 < long_prompt.txt   # control
```

Measure and record for each: tok/s, expert hit-rate, `CUDA expert tier: … 
chiamate servite da VRAM`, `[CUDA] streamed batch matmul` count, RSS, and for
C the wall-clock prefill segment specifically.

**Expected outcomes (honest):**
- B vs A decode: the VRAM tier previously *serialized* into the token path;
  now it overlaps with CPU expert compute. On a warm 192 GB cache, expect a
  modest but real gain — anywhere from +10% to +50% tok/s depending on how
  many hits land on the ~1,200 experts a 24 GB tier holds. If B ≤ A, the
  overlap isn't engaging: check the stats line shows nonzero VRAM-served calls.
- C vs control: this is the headline. Prefill matmuls of the dense parts,
  shared experts, and batch-union routed experts move to the GPU. Expect a
  multiple (2–10×) on the matmul-bound share of prefill; total prefill also
  contains disk I/O, which this branch does not touch.
- Decode with `CUDA_PREFILL_ROWS` at default must NOT regress vs A when
  `CUDA_EXPERT_GB=0` (the streaming path never triggers at S=1).

### Stage 5 — report

Open one summary (GitHub issue on the fork, or a markdown file next to this
one) containing: hardware, driver/toolkit versions, every stage's pass/fail,
the A/B/C numbers with their stats lines, and any `[CUDA]` stderr warnings
seen. If everything passes and B/C show wins, this branch is ready to be
offered upstream (JustVugg/colibri) — the author explicitly requests exactly
these measurements in the README.

## Known limitations (do not "fix" during validation — note them)

1. One stream per device: within a single device, weight H2D copies and
   kernels serialize; double-buffered copy/compute overlap is a follow-up.
2. Streamed weights transfer from pageable memory (the expert cache), not
   pinned staging — effective PCIe bandwidth is below peak.
3. The GPU FFN accumulates in f32 over dequantized weights, matching the CPU
   non-idot path, not the idot path — last-ulp logit differences are expected.
4. On VRAM exhaustion the streaming path disables itself after 8 consecutive
   failures and logs once; the run continues on CPU. That's by design.
