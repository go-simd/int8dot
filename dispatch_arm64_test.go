//go:build arm64 && go1.27

package int8dot

import (
	"math/rand"
	"testing"
)

// TestDispatchARM64 drives all three dot products down the arm64 NEON kernels
// directly and checks them bit-for-bit against the scalar reference. NEON is the
// arm64 baseline, so — unlike the runtime-gated amd64/riscv64 dispatchers — there
// is no feature flag to toggle: the kernel is the only SIMD path and is always
// taken on a Go 1.27 arm64 build, so calling it across every size (the vector
// body, the < 16 scalar tail, and the boundaries between them) is what reaches
// 100% coverage of kernels_arm64.s and dispatch_arm64.go. Integer MAC is exact,
// so equality is required, not tolerance.
func TestDispatchARM64(t *testing.T) {
	r := rand.New(rand.NewSource(99))
	for _, n := range sizes {
		as, bs := randI8(n, r), randI8(n, r)
		if got, want := dotNEON(as, bs), dotRef(as, bs); got != want {
			t.Fatalf("dotNEON n=%d: %d != %d", n, got, want)
		}
		au, bu := randU8(n, r), randU8(n, r)
		if got, want := dotUint8NEON(au, bu), dotUint8Ref(au, bu); got != want {
			t.Fatalf("dotUint8NEON n=%d: %d != %d", n, got, want)
		}
		bi := randI8(n, r)
		if got, want := dotU8S8NEON(au, bi), dotU8S8Ref(au, bi); got != want {
			t.Fatalf("dotU8S8NEON n=%d: %d != %d", n, got, want)
		}
	}
}

// TestExtremesARM64 stresses the NEON widening path with worst-case-magnitude
// inputs (−128/127 signed, 255 unsigned) over many blocks, confirming no 16-bit
// product overflow leaks through the int32 accumulation.
func TestExtremesARM64(t *testing.T) {
	const n = 1024
	maxI8 := make([]int8, n)
	minI8 := make([]int8, n)
	maxU8 := make([]uint8, n)
	for i := range maxI8 {
		maxI8[i] = 127
		minI8[i] = -128
		maxU8[i] = 255
	}
	if got, want := dotNEON(minI8, maxI8), dotRef(minI8, maxI8); got != want {
		t.Errorf("dotNEON extremes = %d, want %d", got, want)
	}
	if got, want := dotUint8NEON(maxU8, maxU8), dotUint8Ref(maxU8, maxU8); got != want {
		t.Errorf("dotUint8NEON extremes = %d, want %d", got, want)
	}
	if got, want := dotU8S8NEON(maxU8, minI8), dotU8S8Ref(maxU8, minI8); got != want {
		t.Errorf("dotU8S8NEON extremes = %d, want %d", got, want)
	}
}
