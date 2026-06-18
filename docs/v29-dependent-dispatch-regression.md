# v29 dependent-dispatch chain: regression investigation

**Verdict (TL;DR):** On a serially-dependent compute-dispatch chain, the
per-barrier and per-dispatch costs are **flat from the v22-era lib (cogentcore)
through v29** — there is **no wgpu-native regression** in dependent dispatches.
This holds not just for one barrier per dispatch but **across 1–8 barriers per
dispatch** (the [barriers-per-dispatch sweep](#follow-up-done-barriers-per-dispatch-sweep)
generalizes the chain to goinfer-like multi-buffer kernels): per-barrier cost
*scales with the number of storage-buffer transitions per dispatch* but is
identical on every version at every count (v29 within ~2% of v22 at B=8). The
synthetic chain does **not** reproduce goinfer's +1.9 ms/token on any axis it
can express. Per STEP-1's stop condition, the cost is therefore **not a generic
wgpu-hal barrier change** — it is **specific to goinfer's setup** (and likely not
barrier-bound at all). There is **no version "sweet-spot" pin to recover** and
**no upstream bug to file**.

**Update — pipeline-state switching also ruled out.** The barrier sweep used one
re-dispatched pipeline; real glue cycles many distinct pipelines (rope, rmsnorm,
quant, swiglu, residual…), so each dispatch pays a pipeline+bind-group switch. A
second sweep ([pipeline-state-switch](#follow-up-2-done-pipeline-state-switching))
cycled 1–8 distinct pipelines through the same dependent chain on the same GPU.
Device-side per-dispatch time is **flat (~1.50 µs) at every N on every version,
v22→v29** — the per-switch cost is ≈0 and does not regress.

**Update 2 — naga codegen also ruled out; investigation closed.** The last cheap
suspect was naga's SPIR-V codegen for the real glue kernels. An A/B
([naga-codegen](#follow-up-3-done-naga-codegen-ab-spir-v-passthrough)) ran
naga-22 vs naga-29 SPIR-V of real goinfer kernels (rmsnorm/qknorm/residual/swiglu)
on the **same** v29 runtime via passthrough. naga 29 does emit +250 words of
bounds-check code on the dynamic-loop kernels, but the **runtime delta is at the
noise floor (≤+0.2 µs, ~5%, only on 2 kernels)** and the shipped v29 runtime path
(naga 29.0.1) is actually the **fastest** arm. ≤4% of goinfer's gap. **All cheap,
isolable axes — barriers, pipeline switches, codegen — are flat or negligible
v22→v29.** The residual +1.9 ms is real but glue-specific and not cheaply
isolable on the binding side (real memory traffic / resource tracking / dispatch
counts the synthetic benches strip out). **Decision unchanged:** stay on
cogentcore for decode; v29 only for its features; the lever is cutting glue
dispatch count (decode fusion in goinfer, out of scope here).

---

## What was measured

A chain of **K serially-dependent** compute dispatches, each reading the previous
dispatch's output buffer and writing the next (ping-pong A→B→A…), forcing a
storage-buffer read-after-write **barrier between every dispatch**. Work per
dispatch is deliberately tiny (`out[i] = in[i] + 1`, n=256) so the chain is
**latency-bound on barriers/launches**, mirroring decode glue.

A **control** runs the same K dispatches but **independent** (each writes its own
buffer, no hazard), so the GPU may overlap them. Subtracting isolates cost:

```
dependent_gpu   ≈ K × (dispatch + barrier)
independent_gpu ≈ K × dispatch          (barriers elided / overlapped)
per_barrier     = (dependent_gpu − independent_gpu) / K
```

Device-side time via timestamp-query; reported as the median of 30 runs.

**Hardware:** NVIDIA GeForce RTX 2070 SUPER, Vulkan, driver 595.58.03, Nobara
(Fedora 43), Linux 7.0.5 — the same discrete-GPU box on which goinfer's
+3.2 ms/token (+1.9 ms GPU) was measured.

Benches (committed in this repo):
- [`cmd/chainbench`](../cmd/chainbench) — uses this binding (v29).
- [`cmd/chainprobe`](../cmd/chainprobe) — self-contained raw-cgo, version-portable
  (links v25/v27/v29 unchanged via two `-D` flags); also the minimal upstream repro.
- [`bench/cogentbase`](../bench/cogentbase) — same chain through
  `cogentcore/webgpu` (the bundled pre-futures **v22-era** lib = known-fast baseline).
- [`scripts/bisect.sh`](../scripts/bisect.sh) — swaps headers+lib per release and runs the probe.

Reproduce:

```sh
# v25/v27/v29 bisect on the Vulkan box:
GO=$HOME/sdk/go/bin/go BACKEND=vulkan KS=100,200,400 bash scripts/bisect.sh
# v22-era baseline:
cd bench/cogentbase && go build . && ./cogentbase -backend vulkan -ksweep 100,200,400 -csv
```

## Per-version results (Vulkan, RTX 2070 SUPER, n=256, K=400)

| wgpu-native              | ABI era      | per-dispatch (µs) | **per-barrier (µs)** |
|--------------------------|--------------|-------------------|----------------------|
| cogentcore v0.23.0       | v22-era      | 0.163             | **1.333**            |
| v25.0.2.2 (`0x19000202`) | new-ABI      | 0.154             | **1.341**            |
| v27.0.4.1 (`0x1b000401`) | new-ABI      | 0.155             | **1.374**            |
| v29.0.0.0 (`0x1d000000`) | new-ABI      | 0.155             | **1.353**            |

Per-barrier is constant to within run-to-run noise (~±0.03 µs) across **four**
lib generations spanning the v22→v24 wgpu-core/hal rewrite *and* every new-ABI
release. The independent control overlaps cleanly on Vulkan (~0.16 µs/dispatch,
≈8× cheaper than the dependent chain), confirming the bench correctly separates
the two costs. (On Metal, by contrast, wgpu-hal serializes dispatches in a
compute encoder regardless, so the control does not overlap and per-barrier ≈ 0
— Metal cannot isolate this cost. Decode is Vulkan, so this is moot.)

## Why this is conclusive

- The bench **does** reproduce a real, repeatable per-barrier cost on Vulkan
  (~1.33 µs), so the methodology is sound — it is not failing to see *any* cost.
- That cost **did not change** from the known-fast v22-era baseline to v29.
- goinfer's regression is **+1.9 ms GPU over ~535 dispatches ≈ +3.5 µs *extra*
  per dispatch, v29 vs v22**. Our generic chain shows **0 µs** delta v22→v29.
  The missing ~3.5 µs/dispatch must come from per-dispatch surface this trivial
  kernel does **not** exercise.

In short: it is **not** a change in wgpu-hal's per-dispatch barrier emission for
a single storage-buffer RAW hazard. Both branches of STEP 2's hypothesis
("regression entered v22→v25" and "entered v25→v29") are **refuted** for the
generic case.

## What this rules out (and in)

Ruled **out** as the cause:
- A more-conservative single-RAW-barrier cost in newer wgpu-hal.
- A per-dispatch launch-overhead regression.
- An instance-flag / validation default (runs were Empty-flags, validation off).
- A submit/poll-granularity effect on the barrier itself.

- **More barriers per dispatch** — tested directly in the
  [sweep below](#follow-up-done-barriers-per-dispatch-sweep) (1–8 storage-buffer
  barriers/dispatch). Per-barrier scales with count but stays flat across
  versions, so this is **also ruled out** as a *version* cause.

Ruled **in** (remaining, goinfer-specific) — the +3.5 µs/dispatch must track
something none of these benches express. Candidates, now that barrier cost
(single- and multi-buffer) is eliminated:
1. **The chain isn't actually barrier-bound** in goinfer the way modeled (e.g.
   real per-dispatch ALU/memory work dominates, and that path changed).
2. **Larger working sets / buffer sizes** forcing real cache/memory traffic
   (these buffers are 1 KB).
3. **Dispatch count / submit structure / CPU-side encode** differences between
   the two measured goinfer configurations.
4. **Pipeline/shader specialization** differences (naga codegen) per kernel.

## Mitigation

- **No version pin helps.** Because v25/v27/v29 are identical to the v22 baseline
  on this axis, there is no new-ABI release that is "fast" here — staying on v29
  (which has dot4I8Packed + the v29 features) costs nothing extra on generic
  dependent dispatches. Do not downgrade expecting a barrier win.
- **No generic upstream issue to file.** The minimal repro (`cmd/chainprobe`)
  shows flat cost across versions, so there is nothing for gfx-rs/wgpu to fix at
  the generic-barrier level. An issue would only be warranted *after* the
  follow-up below localizes a version-dependent, shape-specific cost.
- **The real lever is goinfer's chain structure**, not the wgpu version: reducing
  the number of read↔write storage-buffer transitions per dispatch (fewer
  storage buffers, batching independent decode steps so the GPU can overlap
  them, or fusing dispatches) attacks the ~1.33 µs × (barriers/dispatch) term
  directly. Note this repo's scope is the binding; goinfer is explicitly not to
  be modified here — this is guidance for that work, not a change.

## Follow-up (done): barriers-per-dispatch sweep

The trivial chain above forces exactly **one** barrier per dispatch; goinfer's
decode kernels bind several read-write storage buffers, so the worry was that a
v29 state-tracker change might cost more *per touched buffer*, invisible to a
2-binding chain. The benches gained a `-bindings B` knob (each dispatch
increments **B** read-write storage buffers in place → **B barriers/dispatch**),
and the v22→v29 bisect was re-run at B = 1, 2, 4, 8 on the same Vulkan GPU
(K=400, n=256, 30 runs).

**per-barrier (µs) — by barriers/dispatch × version:**

| barriers/dispatch | cogent (v22-era) | v25.0.2.2 | v27.0.4.1 | v29.0.0.0 |
|-------------------|------------------|-----------|-----------|-----------|
| 1                 | 1.330            | 1.349     | 1.351     | 1.332     |
| 2                 | 1.493            | 1.537     | 1.513     | 1.514     |
| 4                 | 1.818            | 1.871     | 1.871     | 1.874     |
| 8                 | 2.478            | 2.538     | 2.538     | 2.537     |

Two facts:
1. **Per-barrier scales with B** — ≈ a 1.17 µs fixed pipeline-barrier floor plus
   ≈0.16 µs per additional bound storage-buffer transition. So a kernel that
   forces more read↔write storage transitions per dispatch *does* cost
   proportionally more. The absolute cost is real and is a function of the chain
   shape.
2. **It is still flat across versions at every B.** The v22→v29 gap never opens:
   v29 is within ~2% of the v22-era baseline even at 8 barriers/dispatch. There
   is **no version-dependent component** to barrier cost, single- or multi-buffer.

This **rules out the leading remaining hypothesis** (a v29 per-buffer-transition
regression) and makes the verdict definitive: wgpu-native's barrier behavior did
not regress v22→v29 on any axis this bench can express. goinfer's +1.9 ms must
come from something **outside generic compute-pass barriers** — e.g. its chain
not actually being barrier-bound the way modeled, larger buffers driving real
memory traffic, a difference in dispatch count or submit structure, or a
CPU-side/encode cost — and is therefore not addressable by a wgpu version change.

The one actionable lever this *does* surface for that separate goinfer work:
because per-barrier scales ≈0.16 µs × (storage transitions/dispatch), cutting the
number of read↔write storage buffers per decode dispatch (or fusing/batching to
let independent steps overlap) reduces the cost directly — on every wgpu version
equally. (Out of scope for this repo; goinfer is not modified here.)

Reproduce: `BINDINGS=8 GO=$HOME/sdk/go/bin/go BACKEND=vulkan KS=400 bash scripts/bisect.sh`
and `cd bench/cogentbase && ./cogentbase -ksweep 400 -bindings 8 -csv`.

## Follow-up 2 (done): pipeline-state switching

The barrier sweep above re-dispatches **one** pipeline. goinfer's real decode
glue — ~338 tiny serially-dependent dispatches/token — instead cycles **many
distinct pipelines** (rope, rmsnorm, quant, swiglu, residual, …), so each
dispatch incurs a `SetPipeline(different)` + `SetBindGroup(different)`. That
pipeline+bind-group **state switch** is the one axis the flat synthetic chain
never exercised. This sweep isolates it.

`-distinct N` cycles **N genuinely distinct compute pipelines** (each a separate
shader module + pipeline object — verified by checking the handles differ; a
distinct baked constant prevents wgpu from deduping them), each with its own bind
group, across the K dependent dispatches. Everything else is held identical to
the `N=1` control: same tiny per-dispatch work, same serial dependency (1
read-write storage barrier/dispatch), same K=400. The only new variable is the
per-dispatch pipeline+bindgroup switch. Measured with **device-side timestamp
queries** (not Submit+Poll wall clock).

**switch cost = per-dispatch GPU time at N − at N=1** (cancels the
barrier/dispatch baseline, leaving the pipeline+bindgroup-switch cost).

**device-side per-dispatch GPU time (µs) — N pipelines × version (K=400, 50 runs):**

| N (distinct pipelines) | cogent (v22-era) | v25.0.2.2 | v27.0.4.1 | v29.0.0.0 |
|------------------------|------------------|-----------|-----------|-----------|
| 1 (baseline)           | 1.495            | 1.503     | 1.502     | 1.507     |
| 2                      | 1.495            | 1.504     | 1.504     | 1.502     |
| 4                      | 1.495            | 1.504     | 1.504     | 1.503     |
| 8                      | 1.495            | 1.505     | 1.504     | 1.504     |

**switch cost (µs):**

| N | cogent (v22) | v25    | v27    | v29    |
|---|--------------|--------|--------|--------|
| 2 | 0.000        | +0.000 | +0.002 | −0.005 |
| 4 | 0.000        | +0.000 | +0.002 | −0.004 |
| 8 | 0.000        | +0.001 | +0.002 | −0.003 |

**Verdict: branch 6 — pipeline-state switching is NOT the cause.** On the GPU
timeline, cycling 8 distinct pipelines costs the same per dispatch as
re-dispatching one (~1.50 µs), and this is true on every generation from the
v22-era baseline through v29. The per-switch cost is ≈0 and the **v22→v29 gap
never opens** (it is within ±0.005 µs — noise — at every N). On NVIDIA Vulkan a
pipeline+descriptor-set bind is essentially free on the device timeline and was
not made more expensive in any wgpu-native generation.

**Magnitude check vs goinfer.** Matching goinfer's +1.9 ms over ~338
switches/token would require a v29-vs-v22 switch-cost delta of ≈ 1.9 ms / 338 ≈
**5.6 µs/switch**. Measured delta: **≈0 µs/switch** (v29 is if anything a hair
faster than v22). Pipeline switching accounts for ~**0%** of the regression —
three-plus orders of magnitude short.

(Aside, CPU side: wall-clock per-iteration rose ~0.1 ms going N=1→N≥2 on v29 —
real but tiny CPU encode cost of distinct `SetPipeline`/`SetBindGroup` calls, of
the same order on all versions, and not the device-side GPU regression goinfer
measured with timestamps. This is exactly why the experiment uses device
timestamps, not Submit+Poll wall clock — the earlier tooling mistake.)

### Recommended next probe (not built here): SPIR-V passthrough A/B

Barriers (1–8/dispatch) and pipeline switching (1–8 distinct) are both flat
v22→v29. The one thing all these synthetic benches share is a **trivial kernel**
(`out[i]=in[i]+1`). The remaining suspect is **naga codegen of the *real* glue
kernels** — if v29's naga emits worse SPIR-V (more instructions, worse register
allocation, lost vectorization, extra bounds checks) for goinfer's actual
rope/rmsnorm/quant/swiglu shaders, only those kernels would show it.

Proposed isolation, decisive and small:
1. Take one real glue kernel's WGSL from goinfer (read-only; do not modify
   goinfer).
2. Compile it to SPIR-V **once** with each version's naga (`naga in.wgsl out.spv`,
   matching the pinned wgpu-native commits), and `diff` the disassembly.
3. A/B the two SPIR-V blobs on the same v29 runtime via
   `ShaderSourceSPIRV`/SpirvShaderPassthrough (bypasses naga at run time), in the
   same dependent chain, device timestamps.
   - If the v22-naga SPIR-V runs faster than the v29-naga SPIR-V → **naga codegen
     regression**, now with a minimal filable upstream repro (the two .spv + the
     timing).
   - If both run identically → the cost is not in the kernel binary either, and
     attention turns to dispatch/submit *count* or buffer sizes in the real glue.

This is the highest-value next step; it is intentionally **not** built here.

## Follow-up 3 (done): naga codegen A/B (SPIR-V passthrough)

Barriers and pipeline switches are flat; the last cheap suspect is **naga
codegen of the real kernels**. To isolate the compiler from the runtime, real
goinfer glue WGSL (copied verbatim as read-only fixtures under
[`bench/nagaprobe/fixtures/`](../bench/nagaprobe/fixtures); goinfer untouched)
was compiled to SPIR-V offline by **naga-cli 22.0.0** and **29.0.x**, and both
blobs were run on the **same v29 runtime** via the standard `WGPUShaderSourceSPIRV`
passthrough path (`cmd/nagaprobe`) — the runtime's own naga never runs, so the
driver gets exactly the bytes each naga version emitted. Same dependent chain as
chainprobe (K=400, one reused rw storage buffer ⇒ 1 barrier/dispatch), device
timestamps, 60 runs. Arms: (a) naga 22 SPIR-V, (b) naga 29 SPIR-V, (c) the WGSL
compiled by the v29 runtime's **own** naga 29.0.1 (the config goinfer actually uses).

wgpu-native v22.1.0.5 (cogentcore's lib) pins naga 22.1.0; v29.0.0.0 pins naga 29.0.1.

**SPIR-V size (naga 22 → naga 29):**

| fixture  | shape                    | 22 → 29 bytes | Δwords |
|----------|--------------------------|---------------|--------|
| rmsnorm  | workgroup reduce + loops | 3500 → 4500   | **+250** |
| qknorm   | workgroup reduce + loops | 3668 → 4668   | **+250** |
| residual | trivial elementwise      | 1200 → 1200   | 0      |
| swiglu   | elementwise + exp        | 1436 → 1436   | 0      |

naga 29 emits **+250 words only on the two kernels with dynamically-bound,
array-indexed loops** (rmsnorm/qknorm) and **nothing extra** on the loop-free
elementwise kernels — i.e. it adds per-access bounds/robustness code in the loop
bodies. So there *is* a real naga 22→29 codegen change, and it lands exactly
where expected.

**device-side per-dispatch GPU time (µs), K=400 (representative; reduce kernels are noisy at ~±0.15 µs):**

| fixture  | (a) naga 22 spv | (b) naga 29 spv | (c) v29 runtime (naga 29.0.1) |
|----------|-----------------|-----------------|-------------------------------|
| rmsnorm  | 3.41            | 3.57            | **3.31**                      |
| qknorm   | 3.43            | 3.28–3.66       | **2.81–3.38**                 |
| residual | 1.371           | 1.369           | 1.369                         |
| swiglu   | 1.400           | 1.396           | 1.507                         |

**Verdict: branch 6 — naga codegen is NOT the cause.**
- The naga-22-vs-29 *runtime* delta is **at the noise floor**: ~+0.15 µs (≈5%) on
  rmsnorm, inconsistent in sign on qknorm across runs, and **zero** on the
  elementwise controls. The +250-word bounds-check bloat barely moves the clock
  on this GPU.
- Crucially, arm (c) — the v29 runtime's own naga 29.0.1, which is **what goinfer
  actually runs** — is the **fastest** arm, faster than naga-22 passthrough. The
  runtime configures naga to emit lean code (device robustness handles bounds),
  so the passthrough bloat doesn't even appear in the shipped path. There is no
  regression in the configuration goinfer uses.

**Magnitude vs goinfer.** Matching +1.9 ms over ~338 dispatches needs ≈ **5.6 µs
per dispatch**. The worst-case naga effect measured is **≤ +0.2 µs** on *two* of
the kernel shapes (and ≈0 in the shipped runtime path). Applying the worst case
to *all* 338 dispatches (wildly generous — most glue is elementwise/flat) gives
≈ +0.07 ms, **< 4 %** of the gap. naga codegen cannot explain the regression.

### Decision: close the binding-side investigation

All cheap, isolable suspects are now exhausted and flat:

| axis                         | result (v22 → v29)                  |
|------------------------------|-------------------------------------|
| per-barrier cost (1–8/disp)  | flat (~1.33–2.5 µs, ±2%)            |
| pipeline-state switch        | flat (~0 µs/switch)                 |
| naga codegen (real kernels)  | ≤5% on 2 reduce kernels; 0 shipped  |

The residual +1.9 ms is therefore **real but not cheaply isolable on the binding
side** — it must involve effects the synthetic benches deliberately strip out:
actual large-buffer **memory traffic** (these run on ~1 KB; real decode moves MB
of weights/activations/KV per token), **resource-tracking/validation overhead
with the real number of live resources**, or the real **dispatch/submit count and
sizes**. None of these is a wgpu-version bug to file or a flag/pin to flip.

**Decision unchanged:** stay on cogentcore for decode; use this v29 binding only
for what needs v29 (dot4I8Packed / timestamp control / subgroup info), not for a
glue speedup. The real lever remains **cutting the number of glue dispatches per
token** — the decode-fusion work in goinfer — which attacks the cost directly and
is independent of wgpu version. (goinfer is out of scope for this repo and was
not modified.)

Repro: `bash scripts/naga-spv.sh` (compile blobs) then the `nagaprobe` A/B loop
in that script; blobs are committed under `bench/nagaprobe/spv/`.

## Appendix: raw CSV

Original barrier sweep — `csv,verHex,backend,mode,K,n,gpu_ms,wall_ms,per_dispatch_us,per_barrier_us`:

```
csv,0xcoge0023,vulkan,isolated,100,256,0.1328,0.0000,0.2083,1.3277
csv,0xcoge0023,vulkan,isolated,200,256,0.2663,0.0000,0.1779,1.3315
csv,0xcoge0023,vulkan,isolated,400,256,0.5330,0.0000,0.1629,1.3326
csv,0x19000202,Vulkan,isolated,100,256,0.1356,0.0000,0.1853,1.3565
csv,0x19000202,Vulkan,isolated,200,256,0.2720,0.0000,0.1643,1.3600
csv,0x19000202,Vulkan,isolated,400,256,0.5363,0.0000,0.1538,1.3406
csv,0x1b000401,Vulkan,isolated,100,256,0.1368,0.0000,0.1859,1.3683
csv,0x1b000401,Vulkan,isolated,200,256,0.2745,0.0000,0.1658,1.3723
csv,0x1b000401,Vulkan,isolated,400,256,0.5496,0.0000,0.1550,1.3739
csv,0x1d000000,Vulkan,isolated,100,256,0.1300,0.0000,0.1545,1.3000
csv,0x1d000000,Vulkan,isolated,200,256,0.2600,0.0000,0.1545,1.3000
csv,0x1d000000,Vulkan,isolated,400,256,0.5413,0.0000,0.1545,1.3533
```

Pipeline-state-switch sweep — `csv,verHex,backend,mode,K,n,distinct,gpu_ms,wall_ms,per_dispatch_us,per_barrier_us` (dependent rows; per_dispatch_us is the device-side switch metric):

```
csv,0x1d000000,Vulkan,dependent,400,256,1,0.6028,0.9554,1.5070,0.0000
csv,0x1b000401,Vulkan,dependent,400,256,1,0.6009,0.9449,1.5022,0.0000
csv,0x19000202,Vulkan,dependent,400,256,1,0.6014,0.9451,1.5034,0.0000
csv,0xcoge0023,vulkan,dependent,400,256,1,0.5980,1.5133,1.4950,0.0000
csv,0x1d000000,Vulkan,dependent,400,256,2,0.6007,1.0775,1.5018,0.0000
csv,0x1b000401,Vulkan,dependent,400,256,2,0.6016,1.0611,1.5041,0.0000
csv,0x19000202,Vulkan,dependent,400,256,2,0.6014,1.0833,1.5035,0.0000
csv,0xcoge0023,vulkan,dependent,400,256,2,0.5980,1.4863,1.4950,0.0000
csv,0x1d000000,Vulkan,dependent,400,256,4,0.6014,1.0761,1.5034,0.0000
csv,0x1b000401,Vulkan,dependent,400,256,4,0.6015,1.0597,1.5038,0.0000
csv,0x19000202,Vulkan,dependent,400,256,4,0.6015,1.0853,1.5038,0.0000
csv,0xcoge0023,vulkan,dependent,400,256,4,0.5980,1.1229,1.4950,0.0000
csv,0x1d000000,Vulkan,dependent,400,256,8,0.6018,1.0860,1.5044,0.0000
csv,0x1b000401,Vulkan,dependent,400,256,8,0.6014,1.0629,1.5035,0.0000
csv,0x19000202,Vulkan,dependent,400,256,8,0.6020,1.4894,1.5051,0.0000
csv,0xcoge0023,vulkan,dependent,400,256,8,0.5980,1.6408,1.4950,0.0000
```

naga codegen A/B — `csv,verHex,fixture,label,K,gpu_ms,per_dispatch_us` (all on the v29 runtime; label = SPIR-V source):

```
csv,0x1d000000,rmsnorm,naga22,400,1.3625,3.4062
csv,0x1d000000,rmsnorm,naga29,400,1.4237,3.5593
csv,0x1d000000,rmsnorm,wgsl29rt,400,1.3228,3.3071
csv,0x1d000000,qknorm,naga22,400,1.3842,3.4604
csv,0x1d000000,qknorm,naga29,400,1.4657,3.6642
csv,0x1d000000,qknorm,wgsl29rt,400,1.3524,3.3810
csv,0x1d000000,residual,naga22,400,0.5484,1.3711
csv,0x1d000000,residual,naga29,400,0.5475,1.3686
csv,0x1d000000,residual,wgsl29rt,400,0.5476,1.3691
csv,0x1d000000,swiglu,naga22,400,0.5599,1.3998
csv,0x1d000000,swiglu,naga29,400,0.5586,1.3964
csv,0x1d000000,swiglu,wgsl29rt,400,0.6028,1.5069
```
