//go:build riscv64

package int8dot

import (
	"math/rand"
	"testing"
)

// TestDispatchRISCV64 drives all three dot products down BOTH riscv64 paths —
// the RVV kernel and the scalar-reference fallback — by toggling hasV, restoring
// it with defer. The RVV branch is only forced on when the CPU actually has the
// vector extension (the V* instructions would trap otherwise); the scalar
// fallback is always safe. The qemu runner (rv64,v=true) has RVV, so both
// branches are covered there. Results are bit-exact integers.
func TestDispatchRISCV64(t *testing.T) {
	saved := hasV
	defer func() { hasV = saved }()

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

	hasV = false // scalar fallback: always safe
	check("scalar")
	if saved {
		hasV = true
		check("rvv")
	} else {
		t.Log("CPU lacks RVV; kernel branch not exercised on this host")
	}
}
