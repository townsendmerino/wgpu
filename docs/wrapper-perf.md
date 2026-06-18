# Wrapper performance campaign — findings

**TL;DR:** The Go/CGO wrapper is already lean and is **not** where time goes. The
per-dispatch hot path is allocation-free and one CGO call per op; data transfer
is zero-copy; the only non-trivial per-call cost (`CreateBindGroup`, ~830 ns) is
dominated by wgpu-core's C-side validation, not the Go wrapper. This matches the
[dependent-dispatch investigation](v29-dependent-dispatch-regression.md), which
independently measured wrapper overhead at ~+0.1 ms vs. +1.9 ms of GPU execution
for the decode workload. **There is no meaningful binding-side speedup to make.**
The real levers are caller-side (reuse bind groups; cut glue dispatch count) and
are out of this repo's scope.

## Measured (RTX-class CPU benchmarks; `wrapper_bench_test.go`)

`go test -run '^$' -bench Wrapper -benchmem`

| operation                          | cost            | Go allocs            |
|------------------------------------|-----------------|----------------------|
| per-dispatch record (SetBindGroup+Dispatch, in a reused pass) | ~80–240 ns | **0 / dispatch** (≈3–4 / pass) |
| `CreateBindGroup` + `Release`      | ~830 ns         | ~1 wrapper alloc (the returned handle) |
| `Submit` + blocking `Poll`         | ms-scale        | 3 (GPU-bound, not wrapper) |

Per token (~535 dispatches), with bind groups **reused**: recording is ~0.04 ms
of CPU and a handful of allocations total. If a caller instead **rebuilds** a
bind group per dispatch, that's ~535 × 830 ns ≈ 0.44 ms CPU + ~535 allocs/token —
the single largest avoidable CPU cost, and it lives in the caller, not here.

## Triage of the suggested optimizations

| suggestion | verdict |
|---|---|
| Zero-copy `ToBytes`/`FromBytes` | **Already done** — [bytes.go](../bytes.go) is an `unsafe.Slice` reinterpret (no copy); `GetMappedRange` aliases C memory. `CreateBufferInit`'s single `memcpy` into the mapped range is unavoidable for an init-from-data path. |
| Batch `Submit` to take a slice | **Already done** — `Queue.Submit(...*CommandBuffer)` builds one C array, one CGO call. |
| Reduce polling frequency / add a sleep | **Reject** — the busy-wait is only in one-time adapter/device *setup*; `MapAsync` completes inside a blocking `Poll(true)`. A sleep would *raise* latency on the latency-bound decode chain. |
| Cache string/label conversions | **Reject** (not worth it) — labels are set once at creation; the conversion never recurs on a hot path. |
| Lighten `clearLastError`/`takeLastError` per call | **Reject** (micro + risky) — two uncontended mutex ops per `Create*` (not per dispatch); a few tens of ns against ~830 ns of C-side work, and the mutex guards correct cross-thread error capture. Not worth weakening. |
| Minimize CGO crossings per dispatch | **Real but low-value** — a batched compute-pass recorder (loop in C) would collapse ~2 crossings/dispatch into one, saving ~0.04 ms/token. It diverges from the cogentcore drop-in surface and would require caller changes to use. Not implemented; available on request. |
| Expose async futures for latency hiding | **Out of scope** — would break the synchronous drop-in contract; decode is GPU-bound, so there's little CPU latency to hide. |

## Where performance actually lives

1. **GPU command execution** — the decode chain is GPU-bound (the +1.9 ms is on
   the device timeline). The lever is **fewer glue dispatches per token** (decode
   fusion in goinfer), independent of this binding.
2. **Caller-side bind-group reuse** — hold and reuse `*BindGroup` across dispatches
   rather than recreating; the binding already supports this (just keep the handle).

Both are caller-side. The binding itself has no hot-path inefficiency to fix.
