# goinfer caller-side decode analysis (read-only)

A read-only review of how `goinfer/gpu` drives this binding per token, to find
caller-side performance levers. **goinfer was not modified.** Numbers for wrapper
primitives come from this repo's [wrapper benchmarks](wrapper-perf.md); the
v22→v29 GPU findings from the [dependent-dispatch campaign](v29-dependent-dispatch-regression.md).

## 1. The production path is already well-optimized — don't touch it

`gpu.DecodeRunner` (the resident path, used by `residentDecoder` via
`BuildResident`) already does everything a caller-side optimization pass would
recommend:

- **Resident preallocation.** All scratch storage buffers, uniform buffers, and
  bind groups are built **once** at model load. Per token, `Run()` only
  `WriteBuffer`s the input embedding + the pos-dependent uniforms, re-records the
  fixed dispatch plan, and does **one Submit + one Poll**.
- **Op fusion.** `rmsQuant` (RMSNorm→quantize) and `swigluQuant` (SwiGLU→quantize)
  collapse links on the serial decode spine — fewer dispatches, fewer barriers.
- **One compute pass for the whole token** (KV append is a compute kernel, so no
  pass break is forced) — minimal pass/encoder overhead.
- **Batched `RunN`** records K runners into one command buffer to amortize the
  per-token encode + fence over K (speculative/batched decode).

Why this matters, quantified by the wrapper benchmarks: the *superseded*
`DecodeTokenFused`/`DecodeToken` path recreates per dispatch a storage buffer
(`CreateBuffer`), a uniform (`CreateBufferInit` = alloc + map + memcpy + unmap),
and a bind group (`CreateBindGroup` ≈ 830 ns) — and releases them all at token
end. At ~22 dispatches/layer × ~15 layers ≈ **~330 of each per token**, that is
roughly:

| per-token churn (staged path) | rough CPU |
|-------------------------------|-----------|
| ~330 × `CreateBindGroup`       | ~0.27 ms  |
| ~330 × `CreateBufferInit` (uniform; map+memcpy+unmap) | ~0.5 ms |
| ~330 × `CreateBuffer` (scratch) | ~0.3 ms  |
| ~1000 × `Release` + GC of ~1000 Go allocs | ~0.1 ms+ |
| **total, on the critical path before the one Submit** | **~1 ms+/token** |

The resident runner eliminates essentially all of it (per-token allocations ≈ 0;
recording is alloc-free at ~80 ns/dispatch). This is the right design and matches
the binding's strengths (the hot path — `SetPipeline`/`SetBindGroup`/`Dispatch` —
is allocation-free and one CGO call each).

## 2. The one real remaining caller-side lever: the eligibility gate

`Architecture.decodeRunnerEligible()` (decoder/residency.go) returns **false** —
so `BuildResident` returns `(nil, false)` and the model **falls back to the staged
path** that pays the full ~1 ms+/token churn above — for:

- hybrid / non-uniform-forward families: **gemma4, llama4, qwen3.5, granite, nemotron**
- **MoE** that isn't `moeResidentEligible()`
- `NonGatedMLP`, `LearnedPosEmbed`, `OutBias`, `NormPlacement != NormPre2`,
  `FinalLogitSoftcap != 0`, `AttnLogitSoftcap != 0` (e.g. Gemma-2-class softcap)
- additionally, **any f32 projection** forces the staged path (`BuildResident`
  comment: "or a projection is f32").

For these architectures, decode pays the per-token object churn that
`DecodeRunner` was specifically built to remove — on top of the GPU time, on the
critical path. **Extending resident-runner eligibility to these archs (or giving
the staged fallback the same resident-preallocation treatment) is the
highest-value remaining caller-side work.** It is zero benefit to already-eligible
models (GQA/MLA dense) and a ~1 ms+/token CPU win for the ineligible ones.

Suggested order (most modern coverage per unit effort):
1. **Softcap / non-gated-MLP / OutBias** models (Gemma-2-class): these are
   ordinary dense forwards blocked only by a scalar/elementwise feature — cheapest
   to make eligible.
2. **MoE residency** (`moeResidentEligible`) coverage for Mixtral/Qwen-MoE.
3. **f32-projection** models: quantize-on-load (the runner already does W8A8
   upload for eligible models) so they stop forcing the staged path.
4. Hybrid families (gemma4/llama4/qwen3.5) are genuinely harder (non-uniform
   forward) — lowest priority.

## 3. GPU side (for already-resident models): the lever is dispatch count

For eligible models the per-token CPU is already minimal, so the residual cost is
GPU. This repo's campaign established that the v22→v29 +1.9 ms is **not** a
wgpu-hal barrier change, **not** pipeline-state switching, and **not** naga
codegen (all flat/negligible v22→v29). It tracks **dispatch/fence count and
real-data memory traffic**. goinfer's own `gpu-assessment.md` §0.5 measures the
same shape (~74 µs/fence; per-glue-point overhead × layers ≈ ms/token). Since the
runner is already one-submit, the GPU-side levers are:

- **More spine fusion.** `rmsQuant`/`swigluQuant` already cut links; the ~330-dispatch
  spine still has fuseable adjacents (e.g. residual-add into the preceding GEMV
  store — `gemvAdd` already does this for down-proj; apply wherever a tiny
  elementwise immediately follows a GEMV).
- **Batched `RunN`** wherever >1 token is available (speculative decode, prompt
  reprocessing) — amortizes the per-token fence (~74 µs) and re-encode over K.
- **No wgpu version change helps** — staying on v29 (for its features) costs
  nothing extra on barriers/switches/codegen; do not pin an older wgpu expecting a
  decode win.

## 4. Recommended next measurement (I couldn't run it — needs model weights)

`DecodeRunner` already ships `TWrite`/`TEncode`/`TSync` instrumentation. Run a
resident decode on the RTX 2070 SUPER box and report the per-token split:

- If `TSync` (GPU) dominates and `TEncode` is small → the resident path is
  CPU-optimal; all remaining wins are GPU-side (§3) or the eligibility gate (§2).
- If `TEncode` (re-record) is non-trivial → consider caching more of the encode
  (note: WebGPU command buffers are single-use, so the re-record itself is
  unavoidable; the win would be fewer steps via §3 fusion).

This decomposition turns "decode is slow" into a number that says exactly which of
§2 / §3 to fund.
