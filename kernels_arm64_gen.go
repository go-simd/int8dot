//go:build ignore

// Command gen produces kernels_arm64.s with go-asmgen: NEON widening
// multiply-accumulate kernels for the int8 / uint8 / u8×s8 dot products.
//
// # Why widening multiply, not SDOT/UDOT
//
// AArch64 has dedicated 4-lane byte dot-product-accumulate instructions
// (SDOT/UDOT, and the mixed-sign USDOT) that would be the natural one-instruction
// kernel for these contractions. As of the Go 1.27 development tree, however, the
// Go arm64 assembler only exposes those as the *SVE* forms (ZSDOT / ZUDOT /
// ZUSDOT, operating on the scalable Z-registers with a predicate) — there is no
// NEON / Advanced-SIMD SDOT/UDOT mnemonic in cmd/internal/obj/arm64. The SVE
// forms have a vector-length-agnostic programming model that go-asmgen's Raw
// emitter does not model, and — decisively — Apple Silicon (darwin/arm64, where
// this package is developed and natively tested) does not implement SVE at all,
// so an SVE kernel could not be validated on the development host. We therefore
// take the NEON widening-multiply path, exactly like go-simd/base32's and
// go-simd/adler32's arm64 ports.
//
// The integer NEON multiplies used here (VSMULL/VUMULL and the VSXTL/VUXTL
// widening) are only assemblable on Go 1.27+ — the released Go arm64 assembler
// exposes only the polynomial VPMULL — so the generated kernel is guarded
// //go:build arm64 && go1.27 and the stable build falls back to the scalar
// reference (see dispatch_generic.go).
//
// # Algorithm (per 16-byte block)
//
// For each 16-element block of a and b:
//
//   - Widen the 16 a-bytes and 16 b-bytes to two H8 vectors of 16-bit values
//     each: VSXTL/VSXTL2 (sign-extend) for the signed operand, VUXTL/VUXTL2
//     (zero-extend) for the unsigned operand. After widening, every value fits in
//     int16, so the subsequent halfword multiply is exact.
//   - Multiply the low 8 and high 8 widened lanes with the H4→S4 widening
//     multiply (VSMULL/VSMULL2 signed, VUMULL/VUMULL2 unsigned), producing 16
//     int32/uint32 products. Each byte×byte product fits in 16 bits, so widening
//     to 32 bits loses nothing.
//   - Accumulate the four S4 product vectors into a running S4 accumulator with
//     VADD; the accumulation is 32-bit, so it is exact (no overflow within the
//     saturation bound documented in the package).
//
// After the vector loop the 4-lane accumulator is reduced to a scalar with VADDV,
// and a straight-line scalar tail handles the final < 16 elements. Integer MAC is
// exact and order-independent, so every kernel is bit-for-bit identical to the
// scalar reference.
//
// DotU8S8 multiplies an unsigned a by a signed b: a is zero-extended (VUXTL), b
// is sign-extended (VSXTL), and the signed halfword multiply (VSMULL) then yields
// the correct signed product, since both extended operands are exact int16 and
// the product fits in int32.
//
// NEON is the arm64 baseline, so there is no runtime feature gate; the gotip
// (go1.27) CI job and the native darwin/arm64 differential tests + FuzzDot are
// the proof.
//
// Run: go run kernels_arm64_gen.go
package main

import (
	"fmt"
	"os"
	"strings"

	"github.com/go-asmgen/asmgen/abi"
	"github.com/go-asmgen/asmgen/arm64"
	"github.com/go-asmgen/asmgen/emit"
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
	f := emit.NewFile("arm64")
	// extA/extB: the byte->halfword widen for a / b ("VSXTL" sign-extend signed,
	// "VUXTL" zero-extend unsigned). mul: the H4->S4 widening multiply (VSMULL
	// signed, VUMULL unsigned). loadA/loadB: the scalar-tail byte load (MOVB
	// signed, MOVBU unsigned). For DotU8S8, a is unsigned and b is signed.
	f.Add(dotKernel("dotNEON", "VSXTL", "VSXTL", "VSMULL", "MOVB", "MOVB", dotSig()).Func())
	f.Add(dotKernel("dotUint8NEON", "VUXTL", "VUXTL", "VUMULL", "MOVBU", "MOVBU", dotSigU()).Func())
	f.Add(dotKernel("dotU8S8NEON", "VUXTL", "VSXTL", "VSMULL", "MOVBU", "MOVB", dotSig()).Func())

	// The integer NEON multiplies (VSMULL/VUMULL) and widening (VSXTL/VUXTL) are
	// only assemblable on Go 1.27+, matching the //go:build go1.27 Go file; emit
	// writes a bare "arm64", which we narrow here.
	out := strings.Replace(f.String(), "//go:build arm64\n", "//go:build arm64 && go1.27\n", 1)
	if err := os.WriteFile("kernels_arm64.s", []byte(out), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote kernels_arm64.s")
}

// dotKernel emits a NEON widening-MAC kernel.
//
// extA/extB are the byte->halfword widen mnemonics for a and b (VSXTL for the
// signed operand, VUXTL for the unsigned one); each is emitted in both its low
// (".B8 -> .H8") and high (extName+"2", ".B16 -> .H8") forms. mul is the H4->S4
// widening multiply (VSMULL signed / VUMULL unsigned), emitted as mul and mul+"2".
// loadA/loadB are the scalar-tail byte loads (MOVB signed, MOVBU unsigned).
//
// Register plan: R0 a_base, R1 a_len (element count), R2 b_base; R3 i (loop
// index, also the i+16 bound test); R4 scalar accumulator; R5/R6 scalar scratch.
// V0 a bytes, V1 b bytes; V2/V3 widened a halves, V4/V5 widened b halves;
// V16..V19 the four S4 product vectors; V20 the S4 accumulator.
func dotKernel(name, extA, extB, mul, loadA, loadB string, sig abi.Signature) *arm64.Builder {
	b := arm64.NewFunc(name, sig, 0)
	b.LoadArg("a_base", "R0").LoadArg("a_len", "R1").LoadArg("b_base", "R2").
		Raw("MOVD $0, R4"). // scalar accumulator
		Raw("VEOR V20.B16, V20.B16, V20.B16"). // S4 accumulator = 0
		Raw("MOVD $0, R3"). // i = 0
		Label("vloop").
		Raw("ADD $16, R3, R5").Raw("CMP R1, R5").Raw("BGT vtail"). // i+16 > len -> tail
		Raw("VLD1 (R0), [V0.B16]"). // 16 a-bytes
		Raw("VLD1 (R2), [V1.B16]"). // 16 b-bytes
		// Widen bytes -> 16-bit halves (low 8 into .H8, high 8 into .H8).
		Raw(extA + " V0.B8, V2.H8").
		Raw(extA + "2 V0.B16, V3.H8").
		Raw(extB + " V1.B8, V4.H8").
		Raw(extB + "2 V1.B16, V5.H8").
		// Widening halfword multiply -> 32-bit products (4 lanes each).
		Raw(mul + " V2.H4, V4.H4, V16.S4").
		Raw(mul + "2 V2.H8, V4.H8, V17.S4").
		Raw(mul + " V3.H4, V5.H4, V18.S4").
		Raw(mul + "2 V3.H8, V5.H8, V19.S4").
		// Accumulate the four product vectors into V20 (32-bit, exact).
		Raw("VADD V16.S4, V20.S4, V20.S4").
		Raw("VADD V17.S4, V20.S4, V20.S4").
		Raw("VADD V18.S4, V20.S4, V20.S4").
		Raw("VADD V19.S4, V20.S4, V20.S4").
		Raw("ADD $16, R0").Raw("ADD $16, R2").
		Raw("ADD $16, R3, R3").Raw("B vloop").
		Label("vtail").
		// Reduce the 4-lane accumulator to a scalar and seed R4.
		Raw("VADDV V20.S4, V21").Raw("VMOV V21.S[0], R5").
		Raw("ADD R5, R4, R4").
		Label("sloop").
		Raw("CMP R1, R3").Raw("BGE done"). // i >= len -> done
		Raw(loadA + " (R0), R5").
		Raw(loadB + " (R2), R6").
		Raw("MUL R6, R5, R5").
		Raw("ADD R5, R4, R4").
		Raw("ADD $1, R0").Raw("ADD $1, R2").
		Raw("ADD $1, R3, R3").Raw("B sloop").
		Label("done").
		// StoreRet writes the low 32 bits of R4 (MOVW) to the int32/uint32 result.
		StoreRet("R4", "ret").Ret()
	return b
}
