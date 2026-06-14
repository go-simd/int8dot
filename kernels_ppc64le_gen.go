//go:build ignore

// Command gen produces kernels_ppc64le.s with go-asmgen: VMX (Power vector)
// even/odd widening multiply-accumulate kernels for the int8 and uint8 dot
// products.
//
// Power has no single byte multiply-accumulate (no VPDPBUSD/SDOT analogue) and,
// in the released Go assembler, no byte-multiply-sum either (VMSUMMBM/VMSUMUBM
// are absent — verified against cmd/internal/obj/ppc64/anames.go; only
// VMSUMUDM, the doubleword form, exists). The kernels therefore use the
// even/odd widening multiplies that ARE available:
//
//   - VMULESB/VMULOSB multiply the even / odd signed bytes of two vectors into
//     signed 16-bit products (VMULEUB/VMULOUB for the unsigned kernel). Sixteen
//     bytes give 8 even + 8 odd halfword products.
//   - Those 16-bit products are widened to 32 bits and summed with the same
//     even/odd idiom one level up: VMULESH/VMULOSH (VMULEUH/VMULOUH) multiply
//     each halfword product by 1 (a VSPLTISH $1 ones vector) into a 32-bit
//     word, and VADDUWM accumulates into four running word vectors. No overflow
//     can occur — each byte product fits in 16 bits and the accumulation is
//     32-bit — so the result is exact.
//
// The four word accumulators (V8..V11) are summed (VADDUWM), the 4-lane vector
// is spilled to a 16-byte stack scratch (STXVW4X) and the four words are added
// in GPRs; a scalar MULLW tail finishes the < 16 remainder.
//
// DotU8S8 (unsigned×signed) has no mixed-sign byte multiply on Power and is left
// to the scalar reference (see dispatch_ppc64le.go).
//
// VSX/VMX is the POWER8 baseline, so there is no runtime feature gate; the
// qemu-ppc64le differential tests and fuzzers are the proof. Lane order does not
// matter: every op is an element-wise reduction with a commutative horizontal
// sum. Byte loads use LXVB16X (the byte-indexed VSX load) so the 16 bytes land
// in element order regardless of endianness.
//
// Note the VSX↔VMX register aliasing: LXVB16X targets a VSR (VS32+k == Vk);
// the kernels load into VS32/VS33 (== V0/V1) and do the VMX arithmetic on V0/V1.
//
// Run: go run kernels_ppc64le_gen.go
package main

import (
	"fmt"
	"os"

	"github.com/go-asmgen/asmgen/abi"
	"github.com/go-asmgen/asmgen/emit"
	"github.com/go-asmgen/asmgen/ppc64"
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
	f := emit.NewFile("ppc64le")
	f.Add(dotKernel("dotVMX", "VMULESB", "VMULOSB", "VMULESH", "VMULOSH", "MOVB", dotSig()).Func())
	f.Add(dotKernel("dotUint8VMX", "VMULEUB", "VMULOUB", "VMULEUH", "VMULOUH", "MOVBZ", dotSigU()).Func())
	if err := os.WriteFile("kernels_ppc64le.s", []byte(f.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote kernels_ppc64le.s")
}

// foldWords sums V8..V11 (each 4×int32) into V8, spills it to the 16-byte stack
// scratch at 32(R1) and adds the four words into R9.
func foldWords(b *ppc64.Builder) {
	b.Raw("VADDUWM V9, V8, V8").
		Raw("VADDUWM V10, V8, V8").
		Raw("VADDUWM V11, V8, V8").
		Raw("MOVD $32, R12").
		Raw("STXVW4X VS40, (R1)(R12)"). // VS40 == V8
		Raw("MOVWZ 32(R1), R9").
		Raw("MOVWZ 36(R1), R10").Raw("ADD R10, R9").
		Raw("MOVWZ 40(R1), R10").Raw("ADD R10, R9").
		Raw("MOVWZ 44(R1), R10").Raw("ADD R10, R9")
}

// dotKernel emits an even/odd widening MAC kernel. mulEB/mulOB are the byte
// even/odd multiplies; mulEH/mulOH widen the halfword products to words; load is
// the scalar-tail byte load (MOVB signed / MOVBZ unsigned).
//
// Register plan: R3 a_base, R4 a_len, R5 b_base; R6 i; R7 i+16; R8 byte off;
// R9 scalar accumulator; R10/R11 scratch. V0/V1 inputs, V2/V3 halfword products,
// V8..V11 word accumulators, V12 halfword-ones.
func dotKernel(name, mulEB, mulOB, mulEH, mulOH, load string, sig abi.Signature) *ppc64.Builder {
	b := ppc64.NewFunc(name, sig, 64)
	b.LoadArg("a_base", "R3").LoadArg("a_len", "R4").LoadArg("b_base", "R5").
		Raw("VSPLTISW $0, V8").Raw("VSPLTISW $0, V9").
		Raw("VSPLTISW $0, V10").Raw("VSPLTISW $0, V11").
		Raw("VSPLTISH $1, V12"). // halfword ones for the widen step
		Raw("MOVD $0, R6").
		Label("vloop").
		Raw("ADD $16, R6, R7").Raw("CMP R7, R4").Raw("BGT vtail").
		Raw("MOVD R6, R8").
		Raw("LXVB16X (R3)(R8), VS32"). // V0 = 16 a-bytes (element order)
		Raw("LXVB16X (R5)(R8), VS33"). // V1 = 16 b-bytes
		Raw(mulEB + " V0, V1, V2").    // 8 even halfword products
		Raw(mulOB + " V0, V1, V3").    // 8 odd  halfword products
		// widen even-product halfwords -> words, accumulate
		Raw(mulEH + " V2, V12, V4").Raw("VADDUWM V4, V8, V8").
		Raw(mulOH + " V2, V12, V5").Raw("VADDUWM V5, V9, V9").
		Raw(mulEH + " V3, V12, V6").Raw("VADDUWM V6, V10, V10").
		Raw(mulOH + " V3, V12, V7").Raw("VADDUWM V7, V11, V11").
		Raw("ADD $16, R6, R6").Raw("BR vloop").
		Label("vtail")
	foldWords(b)
	b.Label("sloop").
		Raw("CMP R6, R4").Raw("BGE done").
		Raw("MOVD R6, R8").
		Raw(load+" (R3)(R8), R10").
		Raw(load+" (R5)(R8), R11").
		Raw("MULLW R11, R10").
		Raw("ADD R10, R9").
		Raw("ADD $1, R6, R6").Raw("BR sloop").
		Label("done").StoreRet("R9", "ret").Ret()
	return b
}
