# Handoff: `RecordSteps` batched dispatch recording (for goinfer)

A ready-to-paste brief for the goinfer team/agent. `RecordSteps` is an additive,
non-breaking fast path in this binding (`github.com/townsendmerino/wgpu`) for
collapsing the per-dispatch CGO crossings in a hot recorder. goinfer's `gpu/`
package currently imports `cogentcore/webgpu`, which does **not** have this API —
it applies after (or as part of) the mechanical import swap to this binding.

---

**Subject: New opt-in fast path in `github.com/townsendmerino/wgpu` — `RecordSteps` (batched dispatch recording)**

The v29 wgpu drop-in binding (`github.com/townsendmerino/wgpu`, the mechanical
import-swap replacement for the `cogentcore/webgpu` subset `gpu/` uses) now has an
**additive, non-breaking** API for collapsing the per-dispatch CGO crossings in a
hot recorder.

**What's new**

```go
type ComputeStep struct {
    Pipeline  *ComputePipeline
    BindGroup *BindGroup
    X, Y, Z   uint32
}
func (p *ComputePassEncoder) RecordSteps(steps []ComputeStep)
```

- Records the whole batch in **one Go→C crossing** (the
  `SetPipeline`/`SetBindGroup`/`DispatchWorkgroups` loop runs in C) instead of
  ~3 crossings/dispatch.
- Binds each step at **group 0, no dynamic offsets**; skips a redundant
  `SetPipeline` when consecutive steps share a pipeline.
- The per-call methods are unchanged — this is purely additive.

**Measured:** ~5× faster *recording* (85.8 → 16.7 ns/dispatch, record-only).
**Absolute: ~0.04 ms/token** for a ~535-dispatch chain. It only shrinks the encode
(`TEncode`) phase, which is small next to the GPU fence (`TSync`) — so this is a
clean low-risk micro-win, **not** a decode-latency mover. Prioritize accordingly
(it's ~1% of the record phase; do it opportunistically).

**Where it maps in goinfer:** `gpu/decoderunner.go`'s `record()` is already the
exact shape —

```go
for _, s := range r.steps {                 // runStep{pl, bg, gx, gy}
    pass.SetPipeline(s.pl)
    pass.SetBindGroup(0, s.bg, nil)
    pass.DispatchWorkgroups(s.gx, s.gy, 1)
}
```

`runStep` is 1:1 with `ComputeStep`. Adoption (only after `gpu/` is on
`townsendmerino/wgpu` — `RecordSteps` is **not** in `cogentcore/webgpu`):

1. Build the `[]wgpu.ComputeStep` **once** alongside `r.steps` (it's fixed per
   runner) and store it on the runner — no per-token Go-struct rebuild:
   ```go
   r.recordSteps = make([]wgpu.ComputeStep, len(r.steps))
   for i, s := range r.steps {
       r.recordSteps[i] = wgpu.ComputeStep{Pipeline: s.pl, BindGroup: s.bg, X: s.gx, Y: s.gy, Z: 1}
   }
   ```
2. `record()` collapses to:
   ```go
   func (r *DecodeRunner) record(pass *wgpu.ComputePassEncoder) { pass.RecordSteps(r.recordSteps) }
   ```
   (`RecordSteps` re-marshals the slice into 3 C arrays per call — a few
   allocs/token, negligible; the plan itself is built once.)
3. In `runBatch`, call `pass.RecordSteps(runner.recordSteps)` per runner in row
   order — this preserves the cross-runner ordering you depend on for the shared
   KV cache.

**Constraints / verification**

- Group-0 bindings, no dynamic offsets only (matches `record()` today). If a step
  ever needs a non-zero group index or dynamic offset, keep the per-call methods
  for that step.
- `RecordSteps` is bit-identical to the unrolled loop (asserted by a parity test
  in the binding). **Re-run the `DecodeRunner` parity tests after switching —
  output must be unchanged.**
- Confirm with the existing `TWrite`/`TEncode`/`TSync` instrumentation: `TEncode`
  should drop, `TSync` unchanged.

Reference: `command.go` (`RecordSteps`), `recordsteps_test.go` (parity test +
benchmark), `docs/wrapper-perf.md`.
