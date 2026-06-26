//go:build ppc64le

package int8dot

import (
	"math/rand"
	"testing"

	"golang.org/x/sys/cpu"
)

// TestDispatchPPC64LE drives all three dot products down BOTH ppc64le paths —
// the VMX kernel and the scalar-reference fallback — by toggling hasVSX,
// restoring it with defer. The fallback (hasVSX=false) is always safe. The
// kernel branch emits ISA-3.0 (POWER9) instructions (e.g. LXVB16X) that SIGILL
// on POWER8, so it is forced on only when the host is genuinely POWER9+. Under
// the QEMU power9 CI target IsPOWER9 is true, so both branches are covered
// there. Results are bit-exact integers, so equality is required.
func TestDispatchPPC64LE(t *testing.T) {
	saved := hasVSX
	defer func() { hasVSX = saved }()

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

	// Scalar fallback: always safe.
	hasVSX = false
	check("fallback")

	// Kernel: ISA-3.0 (POWER9) instructions SIGILL on POWER8, so only force the
	// VMX branch on a genuine POWER9+ host (true under QEMU power9 CI).
	if !cpu.PPC64.IsPOWER9 {
		t.Log("pre-POWER9 host; VMX kernel branch not exercised")
		return
	}
	hasVSX = true
	check("kernel")
}
