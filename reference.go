package int8dot

// This file holds the scalar reference for every dot product. It is always
// compiled (no build tag): it is the oracle the differential tests and fuzzers
// compare the SIMD kernels against, and it is also the kernel the generic
// (non-SIMD) build calls directly.
//
// Integer multiply-accumulate is associative and exact, so — unlike the
// floating-point sibling package — the reference needs no lane-blocked
// reduction order: the SIMD kernels are required to be bit-for-bit identical to
// these straight-line loops on every architecture.

// dotRef returns Σ aᵢbᵢ for signed bytes, accumulated in int32. Each product is
// widened to int32 before the add so the multiply itself cannot overflow.
func dotRef(a, b []int8) int32 {
	var acc int32
	for i := range a {
		acc += int32(a[i]) * int32(b[i])
	}
	return acc
}

// dotUint8Ref returns Σ aᵢbᵢ for unsigned bytes, accumulated in uint32.
func dotUint8Ref(a, b []uint8) uint32 {
	var acc uint32
	for i := range a {
		acc += uint32(a[i]) * uint32(b[i])
	}
	return acc
}

// dotU8S8Ref returns Σ uint8(aᵢ)·bᵢ, accumulated in int32: the first operand is
// unsigned, the second signed.
func dotU8S8Ref(a []uint8, b []int8) int32 {
	var acc int32
	for i := range a {
		acc += int32(a[i]) * int32(b[i])
	}
	return acc
}
