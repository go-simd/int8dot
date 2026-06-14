//go:build amd64

package int8dot

import (
	"math/rand"
	"testing"
)

// TestDispatchAMD64 drives all three dot products down BOTH amd64 paths — the
// AVX2 kernel and the scalar-reference fallback — by toggling hasAVX2, restoring
// it with defer. The AVX2 branch is only forced on when the CPU actually has
// AVX2 (the VPMADDWD/VPMOVSXBW instructions would #UD otherwise); the scalar
// fallback is always safe. The native amd64 CI runner has AVX2, so both branches
// are covered there, making this the authoritative 100%-coverage gate for the
// amd64 dispatchers. Results are bit-exact integers, so equality is required.
func TestDispatchAMD64(t *testing.T) {
	saved := hasAVX2
	defer func() { hasAVX2 = saved }()

	r := rand.New(rand.NewSource(99))
	check := func(label string) {
		for _, n := range sizes {
			as, bs := randI8(n, r), randI8(n, r)
			if got, want := dot(as, bs), dotRef(as, bs); got != want {
				t.Fatalf("%s dot n=%d: %d != %d", label, n, got, want)
			}
			au, bu := randU8(n, r), randU8(n, r)
			if got, want := dotUint8(au, bu), dotUint8Ref(au, bu); got != want {
				t.Fatalf("%s dotUint8 n=%d: %d != %d", label, n, got, want)
			}
			bi := randI8(n, r)
			if got, want := dotU8S8(au, bi), dotU8S8Ref(au, bi); got != want {
				t.Fatalf("%s dotU8S8 n=%d: %d != %d", label, n, got, want)
			}
		}
	}

	hasAVX2 = false // scalar fallback: always safe
	check("scalar")
	if saved {
		hasAVX2 = true
		check("avx2")
	} else {
		t.Log("CPU lacks AVX2; kernel branch not exercised on this host")
	}
}
