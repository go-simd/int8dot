package int8dot

import (
	"math/rand"
	"testing"
)

// sizes exercises the vector body, the scalar tail and the boundaries between
// them (the SIMD stride is at most 16 elements on every architecture).
var sizes = []int{0, 1, 2, 3, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 100, 127, 128, 255, 256, 1000}

func randI8(n int, r *rand.Rand) []int8 {
	s := make([]int8, n)
	for i := range s {
		s[i] = int8(r.Intn(256) - 128)
	}
	return s
}

func randU8(n int, r *rand.Rand) []uint8 {
	s := make([]uint8, n)
	for i := range s {
		s[i] = uint8(r.Intn(256))
	}
	return s
}

func TestDot(t *testing.T) {
	r := rand.New(rand.NewSource(1))
	for _, n := range sizes {
		a, b := randI8(n, r), randI8(n, r)
		got, want := Dot(a, b), dotRef(a, b)
		if got != want {
			t.Errorf("Dot(n=%d) = %d, want %d", n, got, want)
		}
	}
}

func TestDotUint8(t *testing.T) {
	r := rand.New(rand.NewSource(2))
	for _, n := range sizes {
		a, b := randU8(n, r), randU8(n, r)
		got, want := DotUint8(a, b), dotUint8Ref(a, b)
		if got != want {
			t.Errorf("DotUint8(n=%d) = %d, want %d", n, got, want)
		}
	}
}

func TestDotU8S8(t *testing.T) {
	r := rand.New(rand.NewSource(3))
	for _, n := range sizes {
		a, b := randU8(n, r), randI8(n, r)
		got, want := DotU8S8(a, b), dotU8S8Ref(a, b)
		if got != want {
			t.Errorf("DotU8S8(n=%d) = %d, want %d", n, got, want)
		}
	}
}

// TestExtremes checks the worst-case-magnitude inputs that stress the widening
// path (−128/127 for signed, 255 for unsigned) at a size large enough to fill
// many accumulator cycles, confirming no 16-bit product overflow leaks.
func TestExtremes(t *testing.T) {
	const n = 1024
	maxI8 := make([]int8, n)
	minI8 := make([]int8, n)
	maxU8 := make([]uint8, n)
	for i := range maxI8 {
		maxI8[i] = 127
		minI8[i] = -128
		maxU8[i] = 255
	}
	if got, want := Dot(minI8, maxI8), dotRef(minI8, maxI8); got != want {
		t.Errorf("Dot extremes = %d, want %d", got, want)
	}
	if got, want := DotUint8(maxU8, maxU8), dotUint8Ref(maxU8, maxU8); got != want {
		t.Errorf("DotUint8 extremes = %d, want %d", got, want)
	}
	if got, want := DotU8S8(maxU8, minI8), dotU8S8Ref(maxU8, minI8); got != want {
		t.Errorf("DotU8S8 extremes = %d, want %d", got, want)
	}
}

func TestPanics(t *testing.T) {
	mustPanic := func(name string, f func()) {
		t.Helper()
		defer func() {
			if recover() == nil {
				t.Errorf("%s: expected panic on length mismatch", name)
			}
		}()
		f()
	}
	mustPanic("Dot", func() { Dot(make([]int8, 3), make([]int8, 4)) })
	mustPanic("DotUint8", func() { DotUint8(make([]uint8, 3), make([]uint8, 4)) })
	mustPanic("DotU8S8", func() { DotU8S8(make([]uint8, 3), make([]int8, 4)) })
}

func FuzzDot(f *testing.F) {
	f.Add([]byte{1, 2, 3}, []byte{4, 5, 6})
	f.Add([]byte{}, []byte{})
	f.Add(make([]byte, 64), make([]byte, 64))
	f.Fuzz(func(t *testing.T, ab, bb []byte) {
		n := min(len(ab), len(bb))
		ab, bb = ab[:n], bb[:n]

		// signed
		as, bs := make([]int8, n), make([]int8, n)
		// unsigned
		au, bu := make([]uint8, n), make([]uint8, n)
		for i := 0; i < n; i++ {
			as[i], bs[i] = int8(ab[i]), int8(bb[i])
			au[i], bu[i] = ab[i], bb[i]
		}
		if got, want := Dot(as, bs), dotRef(as, bs); got != want {
			t.Errorf("Dot mismatch n=%d: %d != %d", n, got, want)
		}
		if got, want := DotUint8(au, bu), dotUint8Ref(au, bu); got != want {
			t.Errorf("DotUint8 mismatch n=%d: %d != %d", n, got, want)
		}
		if got, want := DotU8S8(au, bs), dotU8S8Ref(au, bs); got != want {
			t.Errorf("DotU8S8 mismatch n=%d: %d != %d", n, got, want)
		}
	})
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
