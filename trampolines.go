//go:build cgo

package wgpu

// This file holds C trampoline *definitions* (non-static, external linkage) that
// forward wgpu-native callbacks to the Go side. It must NOT contain any //export
// directives — those live in callbacks.go, which only declares the Go functions
// these trampolines call. Forwarding primitives (not WGPU structs by value)
// keeps the //export boundary simple and ABI-safe.

/*
#include <stdint.h>
#include <stddef.h>
#include "./lib/wgpu.h"

// implemented in callbacks.go via //export
extern void cgoBufferMapCallbackGo(uint32_t status, void *ud1);
extern void cgoUncapturedErrorGo(uint32_t errType, char const *msg, size_t len);
extern void cgoLogCallbackGo(uint32_t level, char const *msg, size_t len);

// Buffer map completion — fires inside wgpuDevicePoll / wgpuInstanceProcessEvents.
void cgoBufferMapCB(WGPUMapAsyncStatus status, WGPUStringView message, void *ud1, void *ud2) {
    (void)message; (void)ud2;
    cgoBufferMapCallbackGo((uint32_t)status, ud1);
}

// Uncaptured device errors — may fire at any time; we only stash the message.
void cgoUncapturedErrorCB(WGPUDevice const *device, WGPUErrorType type, WGPUStringView message, void *ud1, void *ud2) {
    (void)device; (void)ud1; (void)ud2;
    cgoUncapturedErrorGo((uint32_t)type, message.data, message.length);
}

// wgpu-native internal logging.
void cgoLogCallback(WGPULogLevel level, WGPUStringView message, void *userdata) {
    (void)userdata;
    cgoLogCallbackGo((uint32_t)level, message.data, message.length);
}
*/
import "C"
