//go:build ignore

// Command gen produces kernels_loong64.s with go-asmgen: LSX (128-bit vector)
// even/odd widening multiply-accumulate kernels for the int8 and uint8 dot
// products.
//
// LoongArch has no byte multiply-accumulate, so the kernels use the LSX even/odd
// widening multiplies (all real mnemonics in the released assembler — no WORD
// encodings needed, unlike the floating-point LSX kernels):
//
//   - VMULWEVHB/VMULWODHB multiply the even / odd signed bytes of two vectors
//     into signed 16-bit products (the …U forms VMULWEVHBU/VMULWODHBU for the
//     unsigned kernel). Sixteen bytes give 8 even + 8 odd halfword products.
//   - Those 16-bit products are widened to 32-bit words with the even/odd
//     widening *add* against a zero vector — VADDWEVWH/VADDWODWH (…U for
//     unsigned) widen even/odd halfwords to words — and VADDW accumulates the
//     words into four running word vectors. No overflow can occur (each byte
//     product fits in 16 bits, the accumulation is 32-bit), so the result is
//     exact.
//
// The four word accumulators (V16..V19) are summed (VADDW), spilled to a 16-byte
// stack scratch (VMOVQ) and the four words are added in GPRs; a scalar MULW tail
// finishes the < 16 remainder.
//
// DotU8S8 (unsigned×signed) has no mixed-sign byte multiply on LoongArch and is
// left to the scalar reference (see dispatch_loong64.go).
//
// LSX is the LA464 baseline, so there is no runtime feature gate; the
// qemu-loong64 differential tests and fuzzers are the proof. Lane order does not
// matter: every op is an element-wise reduction with a commutative horizontal
// sum.
//
// Run: go run kernels_loong64_gen.go
package main

import (
	"fmt"
	"os"

	"github.com/go-asmgen/asmgen/abi"
	"github.com/go-asmgen/asmgen/emit"
	"github.com/go-asmgen/asmgen/loong64"
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
	f := emit.NewFile("loong64")
	f.Add(dotKernel("dotLSX", "VMULWEVHB", "VMULWODHB", "VADDWEVWH", "VADDWODWH", "MOVB", dotSig()).Func())
	f.Add(dotUint8Kernel().Func())
	if err := os.WriteFile("kernels_loong64.s", []byte(f.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote kernels_loong64.s")
}

// foldWords sums V16..V19 (each 4×int32) into V16, spills it to the stack
// scratch at SP+32 (R3 is SP) and adds the four words into R9.
func foldWords(b *loong64.Builder) {
	b.Raw("VADDW V17, V16, V16").
		Raw("VADDW V18, V16, V16").
		Raw("VADDW V19, V16, V16").
		Raw("MOVV $32, R11").Raw("ADDV R3, R11, R11").
		Raw("VMOVQ V16, (R11)").
		Raw("MOVW (R11), R9").
		Raw("MOVW 4(R11), R10").Raw("ADDV R10, R9, R9").
		Raw("MOVW 8(R11), R10").Raw("ADDV R10, R9, R9").
		Raw("MOVW 12(R11), R10").Raw("ADDV R10, R9, R9")
}

// dotKernel emits an even/odd widening MAC kernel. mulEB/mulOB are the byte
// even/odd widening multiplies; addEH/addOH widen the halfword products to words
// (added against a zero vector); load is the scalar-tail byte load.
//
// Register plan: R4 a_base, R5 a_len, R6 b_base; R7 i; R8 i+16; R10/R11 addr;
// R9 scalar accumulator; R3 is SP. V0/V1 inputs, V2/V3 halfword products,
// V4..V7 widened words, V14 zero, V16..V19 word accumulators.
func dotKernel(name, mulEB, mulOB, addEH, addOH, load string, sig abi.Signature) *loong64.Builder {
	b := loong64.NewFunc(name, sig, 48)
	b.LoadArg("a_base", "R4").LoadArg("a_len", "R5").LoadArg("b_base", "R6").
		Raw("VXORV V16, V16, V16").Raw("VXORV V17, V17, V17").
		Raw("VXORV V18, V18, V18").Raw("VXORV V19, V19, V19").
		Raw("VXORV V14, V14, V14"). // zero vector for the widening adds
		Raw("MOVV $0, R7").
		Label("vloop").
		Raw("ADDV $16, R7, R8").Raw("BLT R5, R8, vtail").
		Raw("ADDV R4, R7, R10").Raw("VMOVQ (R10), V0"). // 16 a-bytes
		Raw("ADDV R6, R7, R11").Raw("VMOVQ (R11), V1"). // 16 b-bytes
		Raw(mulEB + " V0, V1, V2").                     // 8 even halfword products
		Raw(mulOB + " V0, V1, V3").                     // 8 odd  halfword products
		// widen halfword products to words against the zero vector, accumulate
		Raw(addEH + " V2, V14, V4").Raw("VADDW V4, V16, V16").
		Raw(addOH + " V2, V14, V5").Raw("VADDW V5, V17, V17").
		Raw(addEH + " V3, V14, V6").Raw("VADDW V6, V18, V18").
		Raw(addOH + " V3, V14, V7").Raw("VADDW V7, V19, V19").
		Raw("ADDV $16, R7, R7").Raw("JMP vloop").
		Label("vtail")
	foldWords(b)
	b.Label("sloop").
		Raw("BGE R7, R5, done").
		Raw("ADDV R4, R7, R10").Raw(load+" (R10), R11").
		Raw("ADDV R6, R7, R12").Raw(load+" (R12), R13").
		Raw("MULW R13, R11, R11").
		Raw("ADDV R11, R9, R9").
		Raw("ADDV $1, R7, R7").Raw("JMP sloop").
		Label("done").StoreRet("R9", "ret").Ret()
	return b
}

// dotUint8Kernel is the unsigned u8×u8 kernel. It mirrors dotKernel but cannot
// use the unsigned widening *add* VADDWEV.W.HU (mis-emulated by qemu-la464 — it
// corrupts the high lanes), so it widens the unsigned halfword products to words
// with the unsigned widening multiply-add VMADDWEV.W.HU / VMADDWOD.W.HU against
// a halfword-ones vector: acc += zext(hw_product) × 1. The ones vector is built
// on the stack (a doubleword of 0x0001000100010001 stored twice, VMOVQ-loaded)
// since LSX has no splat-immediate the released assembler exposes.
func dotUint8Kernel() *loong64.Builder {
	b := loong64.NewFunc("dotUint8LSX", dotSigU(), 48)
	b.LoadArg("a_base", "R4").LoadArg("a_len", "R5").LoadArg("b_base", "R6").
		Raw("VXORV V16, V16, V16").Raw("VXORV V17, V17, V17").
		Raw("VXORV V18, V18, V18").Raw("VXORV V19, V19, V19").
		// build halfword-ones vector V14 via the stack scratch at SP+16
		Raw("MOVV $0x0001000100010001, R14").
		Raw("MOVV $16, R11").Raw("ADDV R3, R11, R11").
		Raw("MOVV R14, (R11)").Raw("MOVV R14, 8(R11)").
		Raw("VMOVQ (R11), V14"). // V14 = sixteen halfword 1s
		Raw("MOVV $0, R7").
		Label("vloop").
		Raw("ADDV $16, R7, R8").Raw("BLT R5, R8, vtail").
		Raw("ADDV R4, R7, R10").Raw("VMOVQ (R10), V0").
		Raw("ADDV R6, R7, R11").Raw("VMOVQ (R11), V1").
		Raw("VMULWEVHBU V0, V1, V2"). // 8 even unsigned halfword products
		Raw("VMULWODHBU V0, V1, V3"). // 8 odd  unsigned halfword products
		// widen+accumulate: acc += zext(even/odd halfword of products) × 1
		Raw("VMADDWEVWHU V2, V14, V16").
		Raw("VMADDWODWHU V2, V14, V17").
		Raw("VMADDWEVWHU V3, V14, V18").
		Raw("VMADDWODWHU V3, V14, V19").
		Raw("ADDV $16, R7, R7").Raw("JMP vloop").
		Label("vtail")
	foldWords(b)
	b.Label("sloop").
		Raw("BGE R7, R5, done").
		Raw("ADDV R4, R7, R10").Raw("MOVBU (R10), R11").
		Raw("ADDV R6, R7, R12").Raw("MOVBU (R12), R13").
		Raw("MULW R13, R11, R11").
		Raw("ADDV R11, R9, R9").
		Raw("ADDV $1, R7, R7").Raw("JMP sloop").
		Label("done").StoreRet("R9", "ret").Ret()
	return b
}
