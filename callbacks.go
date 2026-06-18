//go:build cgo

package wgpu

// //export Go callbacks invoked by the C trampolines in trampolines.go. The
// preamble here contains only declarations (stdlib headers + the vendored wgpu
// header is intentionally NOT included), satisfying cgo's //export rule.

/*
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
*/
import "C"

import (
	"fmt"
	"os"
	"runtime/cgo"
	"sync"
	"unsafe"
)

// BufferMapCallback is invoked when an async buffer map completes.
type BufferMapCallback func(status BufferMapAsyncStatus)

//export cgoBufferMapCallbackGo
func cgoBufferMapCallbackGo(status C.uint32_t, ud1 unsafe.Pointer) {
	// ud1 is a malloc'd cell holding a cgo.Handle value (see Buffer.MapAsync).
	h := cgo.Handle(*(*C.uintptr_t)(ud1))
	C.free(ud1)
	defer h.Delete()
	if cb, ok := h.Value().(BufferMapCallback); ok && cb != nil {
		cb(BufferMapAsyncStatus(uint32(status)))
	}
}

// ---- last-error capture (uncaptured device errors) -------------------------
//
// v29's pop-error-scope is async; rather than thread futures through every
// call, we set a device uncaptured-error callback that stashes the most recent
// validation/OOM message. wgpu-core reports validation errors synchronously
// during the offending call, so reading lastError right after a NULL-returning
// Create* call yields the real reason. Always logged to stderr too, so nothing
// is ever silent.

var (
	lastErrMu  sync.Mutex
	lastErrMsg string
)

//export cgoUncapturedErrorGo
func cgoUncapturedErrorGo(errType C.uint32_t, msg *C.char, length C.size_t) {
	m := ""
	if msg != nil && length > 0 {
		m = C.GoStringN(msg, C.int(length))
	}
	lastErrMu.Lock()
	lastErrMsg = m
	lastErrMu.Unlock()
	fmt.Fprintf(os.Stderr, "[wgpu] uncaptured error (type %d): %s\n", uint32(errType), m)
}

//export cgoLogCallbackGo
func cgoLogCallbackGo(level C.uint32_t, msg *C.char, length C.size_t) {
	if msg == nil || length == 0 {
		return
	}
	fmt.Fprintf(os.Stderr, "[wgpu][log %d] %s\n", uint32(level), C.GoStringN(msg, C.int(length)))
}

func clearLastError() {
	lastErrMu.Lock()
	lastErrMsg = ""
	lastErrMu.Unlock()
}

func takeLastError() string {
	lastErrMu.Lock()
	defer lastErrMu.Unlock()
	s := lastErrMsg
	lastErrMsg = ""
	return s
}
