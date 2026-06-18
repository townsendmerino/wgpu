// Fixture copied verbatim from goinfer/gpu/layer.go (swigluShaderWGSL).
// Elementwise SwiGLU: silu(gate)*up. Transcendental (exp) elementwise glue.
struct P { n: u32, _a: u32, _b: u32, _c: u32 };
@group(0) @binding(0) var<storage, read>       gate: array<f32>;
@group(0) @binding(1) var<storage, read>       up:   array<f32>;
@group(0) @binding(2) var<storage, read_write> dst:  array<f32>;
@group(0) @binding(3) var<uniform>             p:    P;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p.n) { return; }
    let g = gate[i];
    dst[i] = (g / (1.0 + exp(-g))) * up[i];   // silu(gate)·up
}
