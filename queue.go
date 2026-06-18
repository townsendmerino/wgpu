//go:build cgo

package wgpu

/*
#include <stdlib.h>
#include <stdint.h>
#include "./lib/wgpu.h"
*/
import "C"

import "unsafe"

// SubmissionIndex identifies a queue submission (for fenced Poll).
type SubmissionIndex uint64

// Queue mirrors cogentcore/webgpu.Queue.
type Queue struct {
	ref       C.WGPUQueue
	deviceRef C.WGPUDevice
}

// Submit submits command buffers and returns the submission index (usable with
// Device.Poll for fenced waits).
func (p *Queue) Submit(commands ...*CommandBuffer) SubmissionIndex {
	n := len(commands)
	if n == 0 {
		return SubmissionIndex(C.wgpuQueueSubmitForIndex(p.ref, 0, nil))
	}
	arr := C.malloc(C.size_t(n) * C.size_t(unsafe.Sizeof(C.WGPUCommandBuffer(nil))))
	defer C.free(arr)
	s := unsafe.Slice((*C.WGPUCommandBuffer)(arr), n)
	for i, c := range commands {
		s[i] = c.ref
	}
	return SubmissionIndex(C.wgpuQueueSubmitForIndex(p.ref, C.size_t(n), (*C.WGPUCommandBuffer)(arr)))
}

// WriteBuffer writes data into a buffer at bufferOffset via the queue.
func (p *Queue) WriteBuffer(buffer *Buffer, bufferOffset uint64, data []byte) error {
	if len(data) == 0 {
		C.wgpuQueueWriteBuffer(p.ref, buffer.ref, C.uint64_t(bufferOffset), nil, 0)
		return nil
	}
	C.wgpuQueueWriteBuffer(p.ref, buffer.ref, C.uint64_t(bufferOffset),
		unsafe.Pointer(&data[0]), C.size_t(len(data)))
	return nil
}

// GetTimestampPeriod returns the number of nanoseconds per timestamp tick
// (v29 extra; multiply resolved timestamp deltas by this to get nanoseconds).
func (p *Queue) GetTimestampPeriod() float32 {
	return float32(C.wgpuQueueGetTimestampPeriod(p.ref))
}

func (p *Queue) Release() { C.wgpuQueueRelease(p.ref) }
