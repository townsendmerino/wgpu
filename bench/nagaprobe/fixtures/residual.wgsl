// Fixture copied verbatim from goinfer/gpu/layer.go (residualShaderWGSL).
// Trivial elementwise x += y — the control: minimal control flow, so naga
// codegen should be near-identical across versions.
struct P { n: u32, _a: u32, _b: u32, _c: u32 };
@group(0) @binding(0) var<storage, read_write> x: array<f32>;  // x += y
@group(0) @binding(1) var<storage, read>       y: array<f32>;
@group(0) @binding(2) var<uniform>             p: P;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p.n) { return; }
    x[i] = x[i] + y[i];
}
