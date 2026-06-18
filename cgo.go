//go:build cgo

package wgpu

/*
// ---- static link against the vendored libwgpu_native.a, per platform ----
// Layout mirrors cogentcore: lib/<goos>/<goarch>/libwgpu_native.a

// Linux
#cgo linux,amd64 LDFLAGS: -L${SRCDIR}/lib/linux/amd64 -lwgpu_native
#cgo linux,arm64 LDFLAGS: -L${SRCDIR}/lib/linux/arm64 -lwgpu_native
#cgo linux LDFLAGS: -lm -ldl -lpthread

// Darwin (Metal/QuartzCore/Foundation are required by wgpu-native's Metal backend)
#cgo darwin,amd64 LDFLAGS: -L${SRCDIR}/lib/darwin/amd64 -lwgpu_native
#cgo darwin,arm64 LDFLAGS: -L${SRCDIR}/lib/darwin/arm64 -lwgpu_native
#cgo darwin LDFLAGS: -framework QuartzCore -framework Metal -framework Foundation

// Windows (GNU/mingw static lib)
#cgo windows,amd64 LDFLAGS: -L${SRCDIR}/lib/windows/amd64 -lwgpu_native
#cgo windows LDFLAGS: -lopengl32 -lgdi32 -ld3dcompiler_47 -lws2_32 -luserenv -lbcrypt -lntdll -lole32 -loleaut32 -lpropsys -lruntimeobject

#include "./lib/wgpu.h"

// defined in trampolines.go; forwards to the Go side.
extern void cgoLogCallback(WGPULogLevel level, WGPUStringView message, void *userdata);
*/
import "C"

// LogLevel mirrors WGPULogLevel.
type LogLevel uint32

const (
	LogLevelOff   LogLevel = C.WGPULogLevel_Off
	LogLevelError LogLevel = C.WGPULogLevel_Error
	LogLevelWarn  LogLevel = C.WGPULogLevel_Warn
	LogLevelInfo  LogLevel = C.WGPULogLevel_Info
	LogLevelDebug LogLevel = C.WGPULogLevel_Debug
	LogLevelTrace LogLevel = C.WGPULogLevel_Trace
)

func init() {
	C.wgpuSetLogCallback(C.WGPULogCallback(C.cgoLogCallback), nil)
}

// SetLogLevel controls wgpu-native's internal log verbosity (default: silent).
func SetLogLevel(level LogLevel) { C.wgpuSetLogLevel(C.WGPULogLevel(level)) }

// GetVersion returns the linked wgpu-native version as a packed integer.
func GetVersion() uint32 { return uint32(C.wgpuGetVersion()) }
