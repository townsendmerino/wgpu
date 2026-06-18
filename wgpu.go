//go:build cgo

package wgpu

/*
#include <stdlib.h>
#include "./lib/wgpu.h"
*/
import "C"

import "unsafe"

// cBool / goBool convert between Go bool and WGPUBool.
func cBool(b bool) C.WGPUBool {
	if b {
		return 1
	}
	return 0
}

func goBool(b C.WGPUBool) bool { return b != 0 }

// newStringView allocates a C copy of s and wraps it in a WGPUStringView.
// The caller must release it with freeStringView once the C call returns.
// An empty string yields the null view {NULL, 0}.
func newStringView(s string) C.WGPUStringView {
	if len(s) == 0 {
		return C.WGPUStringView{data: nil, length: 0}
	}
	return C.WGPUStringView{
		data:   C.CString(s),
		length: C.size_t(len(s)),
	}
}

func freeStringView(sv C.WGPUStringView) {
	if sv.data != nil {
		C.free(unsafe.Pointer(sv.data))
	}
}

// goStringView copies an (output) WGPUStringView into a Go string.
func goStringView(sv C.WGPUStringView) string {
	if sv.data == nil || sv.length == 0 || sv.length == C.size_t(^uintptr(0)) {
		return ""
	}
	return C.GoStringN(sv.data, C.int(sv.length))
}

// errOr returns msg if non-empty, else fallback.
func errOr(msg, fallback string) string {
	if msg != "" {
		return msg
	}
	return fallback
}

// ---- opaque handle types with trivial Release (mirrors cogentcore/wgpu.go) ----

type (
	BindGroup       struct{ ref C.WGPUBindGroup }
	BindGroupLayout struct{ ref C.WGPUBindGroupLayout }
	CommandBuffer   struct{ ref C.WGPUCommandBuffer }
	PipelineLayout  struct{ ref C.WGPUPipelineLayout }
	QuerySet        struct{ ref C.WGPUQuerySet }
	ShaderModule    struct{ ref C.WGPUShaderModule }
)

func (p *BindGroup) Release()       { C.wgpuBindGroupRelease(p.ref) }
func (p *BindGroupLayout) Release() { C.wgpuBindGroupLayoutRelease(p.ref) }
func (p *CommandBuffer) Release()   { C.wgpuCommandBufferRelease(p.ref) }
func (p *PipelineLayout) Release()  { C.wgpuPipelineLayoutRelease(p.ref) }
func (p *QuerySet) Release()        { C.wgpuQuerySetRelease(p.ref) }
func (p *ShaderModule) Release()    { C.wgpuShaderModuleRelease(p.ref) }

// GetCount returns the number of queries in the set.
func (p *QuerySet) GetCount() uint32 { return uint32(C.wgpuQuerySetGetCount(p.ref)) }
