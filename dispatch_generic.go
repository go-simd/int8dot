//go:build !amd64 && !ppc64le && !s390x && !riscv64 && !loong64

package int8dot

// On architectures without a SIMD kernel, the dot products are the scalar
// reference itself.
//
// arm64 is intentionally included here: the vector integer-multiply mnemonics
// this package needs (VSMULL/VUMULL/VSMLAL/VUMLAL/VSXTL/VMUL) are not exposed by
// the Go assembler until Go 1.27 — Go 1.26 (current stable) only has VPMULL
// (polynomial multiply), which is unusable for integer arithmetic. A ready NEON
// kernel can be wired in once 1.27 ships; until then arm64 uses this scalar
// path. See README "Architecture support".

func dot(a, b []int8) int32             { return dotRef(a, b) }
func dotUint8(a, b []uint8) uint32      { return dotUint8Ref(a, b) }
func dotU8S8(a []uint8, b []int8) int32 { return dotU8S8Ref(a, b) }
