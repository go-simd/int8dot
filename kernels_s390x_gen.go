//go:build ignore

// Command gen produces kernels_s390x.s with go-asmgen: z/Architecture
// vector-facility even/odd widening multiply-accumulate kernels for the int8 and
// uint8 dot products.
//
// z has no byte multiply-accumulate (no VPDPBUSD/SDOT analogue), so the kernels
// use the even/odd multiplies the vector facility provides:
//
//   - VMEB/VMOB multiply the even / odd signed bytes of two vectors into signed
//     16-bit products (VMLEB/VMLOB for the unsigned kernel). Sixteen bytes give
//     8 even + 8 odd halfword products.
//   - Those halfword products are widened to 32 bits and summed one level up
//     with VMEH/VMOH (VMLEH/VMLOH unsigned): each halfword product is multiplied
//     by 1 (a VREPIH-style ones vector built with VZERO+VLEIH, here a constant
//     loaded via the GPR path) into a 32-bit word, then VAF accumulates into
//     four running word vectors. No overflow is possible — each byte product
//     fits in 16 bits and the accumulation is 32-bit — so the result is exact.
//
// The four word accumulators (V16..V19) are summed (VAF) and the four 32-bit
// lanes are extracted with VLGVF and added in GPRs; a scalar MULLW tail finishes
// the < 16 remainder.
//
// Big-endian note: s390x is big-endian, but every operation here is an
// element-wise reduction followed by a commutative horizontal sum of the four
// word lanes (VLGVF $0..$3 are all pulled and added), so lane numbering does not
// affect the result. The even/odd multiplies consume the same 16 bytes in the
// same order on both operands, so the products pair correctly regardless of
// endianness. The qemu-s390x (big-endian) differential tests and fuzzers are
// the proof.
//
// DotU8S8 (unsigned×signed) has no mixed-sign byte multiply on z and is left to
// the scalar reference (see dispatch_s390x.go).
//
// The vector facility is the z13 baseline, so there is no runtime feature gate.
//
// Run: go run kernels_s390x_gen.go
package main

import (
	"fmt"
	"os"

	"github.com/go-asmgen/asmgen/abi"
	"github.com/go-asmgen/asmgen/emit"
	"github.com/go-asmgen/asmgen/s390x"
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
	f := emit.NewFile("s390x")
	f.Add(dotKernel("dotVX", "VMEB", "VMOB", "VMEH", "VMOH", "MOVB", dotSig()).Func())
	f.Add(dotKernel("dotUint8VX", "VMLEB", "VMLOB", "VMLEH", "VMLOH", "MOVBZ", dotSigU()).Func())
	if err := os.WriteFile("kernels_s390x.s", []byte(f.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote kernels_s390x.s")
}

// foldWords sums V16..V19 (each 4×int32) into V16 and adds the four lanes into
// R9.
func foldWords(b *s390x.Builder) {
	b.Raw("VAF V17, V16, V16").
		Raw("VAF V18, V16, V16").
		Raw("VAF V19, V16, V16").
		Raw("VLGVF $0, V16, R9").
		Raw("VLGVF $1, V16, R10").Raw("ADD R10, R9").
		Raw("VLGVF $2, V16, R10").Raw("ADD R10, R9").
		Raw("VLGVF $3, V16, R10").Raw("ADD R10, R9")
}

// dotKernel emits an even/odd widening MAC kernel. mulEB/mulOB multiply bytes
// even/odd into halfwords; mulEH/mulOH widen the halfword products to words;
// load is the scalar-tail byte load (MOVB signed / MOVBZ unsigned).
//
// Register plan: R1 a_base, R2 a_len, R3 b_base; R4 i; R5 i+16; R6 off; R9
// scalar accumulator; R10 scratch. V1/V2 inputs, V3/V4 halfword products, V5
// halfword-ones, V16..V19 word accumulators.
func dotKernel(name, mulEB, mulOB, mulEH, mulOH, load string, sig abi.Signature) *s390x.Builder {
	b := s390x.NewFunc(name, sig, 0)
	b.LoadArg("a_base", "R1").LoadArg("a_len", "R2").LoadArg("b_base", "R3").
		Raw("VZERO V16").Raw("VZERO V17").Raw("VZERO V18").Raw("VZERO V19").
		Raw("VREPIH $1, V5"). // halfword ones for the widen step
		Raw("MOVD $0, R4").
		Label("vloop").
		Raw("ADD $16, R4, R5").Raw("CMPBGT R5, R2, vtail").
		Raw("MOVD R4, R6").
		Raw("VL (R1)(R6*1), V1"). // 16 a-bytes
		Raw("VL (R3)(R6*1), V2"). // 16 b-bytes
		Raw(mulEB + " V1, V2, V3").
		Raw(mulOB + " V1, V2, V4").
		Raw(mulEH + " V3, V5, V6").Raw("VAF V6, V16, V16").
		Raw(mulOH + " V3, V5, V7").Raw("VAF V7, V17, V17").
		Raw(mulEH + " V4, V5, V20").Raw("VAF V20, V18, V18").
		Raw(mulOH + " V4, V5, V21").Raw("VAF V21, V19, V19").
		Raw("ADD $16, R4, R4").Raw("BR vloop").
		Label("vtail")
	foldWords(b)
	b.Label("sloop").
		Raw("CMPBGE R4, R2, done").
		Raw("MOVD R4, R6").
		Raw(load+" (R1)(R6*1), R10").
		Raw(load+" (R3)(R6*1), R11").
		Raw("MULLW R11, R10").
		Raw("ADD R10, R9").
		Raw("ADD $1, R4, R4").Raw("BR sloop").
		Label("done").StoreRet("R9", "ret").Ret()
	return b
}
