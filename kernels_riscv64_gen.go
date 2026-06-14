//go:build ignore

// Command gen produces kernels_riscv64.s with go-asmgen: RVV (vector extension)
// widening multiply-accumulate kernels for the int8/uint8/u8×s8 dot products.
//
// RVV is length-agnostic: VSETVLI sets the active vector length for the current
// element width, so the kernels need no separate scalar tail — the final short
// chunk runs through the same loop with a smaller VL. Each chunk:
//
//   - VSETVLI … E8 grants VL bytes; VLE8V loads VL bytes of a and b.
//   - VWMULVV widens-and-multiplies 8-bit × 8-bit into 16-bit products
//     (VWMULUVV unsigned, VWMULSUVV signed×unsigned for u8×s8 — note RVV's
//     widening multiply is signed(vs2)×unsigned(vs1), so the *unsigned* operand
//     is passed first).
//   - VSETVLI … E16 (same VL) then VWREDSUMVS reduces the VL 16-bit products
//     into a single 32-bit lane, seeded with the running sum carried in V8, and
//     VMVXS reads it back to the GPR X9. The widening reduction keeps the sum in
//     32 bits, so it is exact (no 16-bit product overflow, no saturation).
//
// All three operations get a SIMD kernel; integer MAC is exact, so each is
// bit-for-bit equal to the scalar reference.
//
// The V extension is detected at runtime (the dispatcher gates on
// cpu.RISCV64.HasV); without it the scalar reference is used.
//
// Run: go run kernels_riscv64_gen.go
package main

import (
	"fmt"
	"os"

	"github.com/go-asmgen/asmgen/abi"
	"github.com/go-asmgen/asmgen/emit"
	"github.com/go-asmgen/asmgen/riscv64"
)

func dotSig() abi.Signature {
	return abi.LayoutArgs(
		[]abi.Arg{abi.Slice("a"), abi.Slice("b")},
		[]abi.Arg{abi.Scalar("ret", abi.Int32)},
	)
}

func dotSigU() abi.Signature {
	return abi.LayoutArgs(
		[]abi.Arg{abi.Slice("a"), abi.Slice("b")},
		[]abi.Arg{abi.Scalar("ret", abi.Uint32)},
	)
}

func main() {
	f := emit.NewFile("riscv64")
	// Reduction width note: VWMULVV/VWMULSUVV products fit in signed int16, so
	// the signed widening reduction VWREDSUMVS is exact. VWMULUVV products are
	// uint16 (up to 65025 > 32767), so the unsigned widening reduction
	// VWREDSUMUVS must be used or the high products would be sign-extended wrong.
	f.Add(dotKernel("dotRVV", "VWMULVV", "VWREDSUMVS", dotSig()).Func())
	f.Add(dotKernel("dotUint8RVV", "VWMULUVV", "VWREDSUMUVS", dotSigU()).Func())
	f.Add(dotKernel("dotU8S8RVV", "VWMULSUVV", "VWREDSUMVS", dotSig()).Func())
	if err := os.WriteFile("kernels_riscv64.s", []byte(f.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote kernels_riscv64.s")
}

// dotKernel emits a length-agnostic widening-MAC kernel. mul is the widening
// byte multiply. When swap is true (u8×s8 via VWMULSUVV) the unsigned operand a
// must be the first multiplier source, so a→V2(vs2 signed slot is actually the
// .SU form's unsigned-first) — handled by emitting the a/b registers in the
// order the .SU mnemonic expects (unsigned, signed).
//
// Register plan: X10 a_base, X11 a_len (remaining count), X12 b_base; X13 VL;
// X14 byte-stride scratch; X9 running 32-bit sum. V1 a chunk, V2 b chunk, V4
// 16-bit products, V8 reduction seed/result.
func dotKernel(name, mul, wredsum string, sig abi.Signature) *riscv64.Builder {
	b := riscv64.NewFunc(name, sig, 0)
	b.LoadArg("a_base", "X10").LoadArg("a_len", "X11").LoadArg("b_base", "X12").
		Raw("MOV $0, X9"). // running 32-bit sum (in a GPR)
		Label("loop").
		Raw("BEQZ X11, done").
		Raw("VSETVLI X11, E8, M1, TA, MA, X14"). // X14 = vl <= X11 (element/byte count)
		Raw("VLE8V (X10), V1").
		Raw("VLE8V (X12), V2")
	// VWMUL* : vd = vs2 * vs1 (widened). For VWMULSUVV the signed operand is
	// vs2 and the unsigned operand is vs1, so a (unsigned) -> vs1 = V1, b
	// (signed) -> vs2 = V2 gives signed(b)*unsigned(a). For the symmetric
	// VWMULVV / VWMULUVV the order is irrelevant.
	b.Raw(mul+" V1, V2, V4"). // V4 (E16, occupies V4:V5) = a*b widened
		// Reduce this chunk's VL 16-bit products into V8[0] with a zero seed,
		// then add the 32-bit partial to the GPR running sum. AVL = X14 (the
		// element count granted for the byte load) so the E16 grant covers
		// exactly those products. Carrying the sum in a GPR (rather than across
		// vector iterations) keeps it correct independent of the vector tail.
		// Zero the 32-bit reduction seed in V8[0] (the widening reduction reads
		// its scalar seed at 2*SEW = E32, so it must be zeroed at E32 width, not
		// E16).
		Raw("VSETIVLI $1, E32, M1, TA, MA, X13").
		Raw("VMVSX X0, V8"). // V8[0] = 0 (32-bit)
		Raw("VSETVLI X14, E16, M2, TA, MA, X13").
		Raw(wredsum+" V8, V4, V8"). // V8[0] = seed(V8[0]) + Σ V4 (widening 16->32)
		// Read the 32-bit reduction result (switch to E32 so VMVXS reads 32 bits).
		Raw("VSETIVLI $1, E32, M1, TA, MA, X13").
		Raw("VMVXS V8, X13").     // X13 = chunk partial (32-bit)
		Raw("ADDW X13, X9, X9").  // running sum += partial
		Raw("ADD X14, X10, X10"). // advance a by vl bytes
		Raw("ADD X14, X12, X12"). // advance b by vl bytes
		Raw("SUB X14, X11, X11"). // remaining -= vl
		Raw("JMP loop").
		Label("done").StoreRet("X9", "ret").Ret()
	return b
}
