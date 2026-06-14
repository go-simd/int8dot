//go:build !amd64 && !ppc64le && !s390x && !riscv64 && !loong64 && !(arm64 && go1.27)

package int8dot

// On architectures without a SIMD kernel, the dot products are the scalar
// reference itself.
//
// arm64 is included here ONLY on stable Go: the vector integer-multiply
// mnemonics this package needs (VSMULL/VUMULL/VSXTL/VUXTL) are not exposed by the
// Go assembler until Go 1.27 — Go 1.26 (current stable) only has VPMULL
// (polynomial multiply), which is unusable for integer arithmetic. On Go 1.27+
// the NEON kernel (dispatch_arm64.go / kernels_arm64.s) takes over, so the
// `!(arm64 && go1.27)` term drops arm64 from this scalar path there. See README
// "Architecture support".

func dot(a, b []int8) int32             { return dotRef(a, b) }
func dotUint8(a, b []uint8) uint32      { return dotUint8Ref(a, b) }
func dotU8S8(a []uint8, b []int8) int32 { return dotU8S8Ref(a, b) }
