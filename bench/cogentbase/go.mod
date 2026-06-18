// Separate module: the v22-era baseline bench depends on cogentcore/webgpu,
// which we deliberately keep OUT of the main binding's go.mod. This measures
// the "known-fast" baseline goinfer compared v29 against, on the same GPU.
module cogentbase

go 1.26

require github.com/cogentcore/webgpu v0.23.0
