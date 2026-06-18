# v29 dependent-dispatch chain: regression investigation

**Verdict (TL;DR):** On a generic serially-dependent compute-dispatch chain, the
per-barrier and per-dispatch costs are **flat from the v22-era lib (cogentcore)
through v29** — there is **no wgpu-native regression** in generic dependent
dispatches. The synthetic chain does **not** reproduce goinfer's +1.9 ms/token.
Per the investigation's STEP-1 stop condition, this means the cost is **specific
to goinfer's kernel/binding shapes**, not to a change in how wgpu-hal emits
barriers between dispatches. There is therefore **no version "sweet-spot" pin to
recover** and **no generic upstream bug to file**; the next move is to widen the
synthetic per-dispatch surface to match goinfer's kernels (see
[§Recommended follow-up](#recommended-follow-up)).

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

Ruled **in** (remaining, goinfer-specific) — the +3.5 µs/dispatch must track
something the trivial kernel lacks. Most likely, in rough order:
1. **More barriers per dispatch.** goinfer's decode kernels bind several
   read-write storage buffers; if v29's state tracker emits a transition per
   touched buffer, a kernel with N storage buffers pays ≈N× the single-barrier
   cost, and any per-buffer-transition change is multiplied by N — invisible to a
   2-binding chain.
2. **Larger working sets / buffer sizes** forcing real cache/memory traffic per
   barrier rather than the 1 KB here.
3. **Bind-group / pipeline rebinding** cost differences per dispatch.
4. **Buffer usage-state churn** (e.g. uniform/storage/copy transitions) absent here.

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

## Recommended follow-up

One targeted experiment converts "goinfer-specific" from inference to a named
cause: extend the bench with a `-bindings B` knob that gives each dispatch **B**
read-write storage buffers (B barriers/dispatch), then re-run the
v22→v29 bisect at B = 1, 2, 4, 8.

- If per-barrier stays flat across versions at every B → the cost scales with
  goinfer's barrier *count*, but wgpu's *per-barrier* cost still never regressed;
  the fix is purely in goinfer's chain shape.
- If the v22→v29 gap **opens up** as B grows → that is the regression, now with a
  minimal multi-buffer repro suitable to file upstream. (Draft issue title:
  *"compute-pass: per-dispatch barrier cost scales with bound storage-buffer
  count on Vulkan in vNN, not in v22"* — fill in once measured.)

This is the single highest-value next step and is cheap (one bench edit + one
`bisect.sh` run on the box).

## Appendix: raw CSV

`csv,verHex,backend,mode,K,n,gpu_ms,wall_ms,per_dispatch_us,per_barrier_us`

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
