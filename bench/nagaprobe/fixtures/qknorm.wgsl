// Fixture copied verbatim from goinfer/gpu/qknorm.go (qkNormWGSL).
// Per-head QK RMS-norm: one workgroup per head, 64-lane tree-reduce over headDim,
// then rescale in place. Dynamic-bound loops + workgroup barriers — a prime
// candidate for naga robustness/bounds-check codegen differences.
struct P { heads: u32, hd: u32, eps: f32, addone: u32 };
@group(0) @binding(0) var<storage, read_write> vec:    array<f32>;  // [heads*hd], normalized in place
@group(0) @binding(1) var<storage, read>       weight: array<f32>;  // [hd]
@group(0) @binding(2) var<uniform>             p:      P;
var<workgroup> sh: array<f32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let h = wid.x;
    if (h >= p.heads) { return; }
    let t = lid.x;
    let base = h * p.hd;
    var ss: f32 = 0.0;
    for (var i: u32 = t; i < p.hd; i = i + 64u) { let v = vec[base + i]; ss = ss + v * v; }
    sh[t] = ss;
    workgroupBarrier();
    var stride: u32 = 32u;
    loop {
        if (stride == 0u) { break; }
        if (t < stride) { sh[t] = sh[t] + sh[t + stride]; }
        workgroupBarrier();
        stride = stride / 2u;
    }
    let inv = 1.0 / sqrt(sh[0] / f32(p.hd) + p.eps);
    for (var i: u32 = t; i < p.hd; i = i + 64u) {
        var w = weight[i];
        if (p.addone == 1u) { w = w + 1.0; }
        vec[base + i] = vec[base + i] * inv * w;
    }
}
