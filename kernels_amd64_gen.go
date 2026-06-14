//go:build ignore

// Command gen produces kernels_amd64.s with go-asmgen: AVX2 integer
// multiply-accumulate kernels for the int8/uint8/u8×s8 dot products.
//
// Each 16-byte chunk is sign- or zero-extended byte→word with VPMOVSXBW/
// VPMOVZXBW (16 int16/uint16 in a YMM), then VPMADDWD multiplies the two word
// vectors pairwise and adds adjacent products into 32-bit lanes —
// int16×int16→int32 with NO saturation, so the result is exact (each byte
// product fits in 16 bits and the pairwise sum of two fits in 32). Products
// accumulate into a YMM via VPADDD; two independent accumulators (Y0, Y6) over a
// 32-byte stride give instruction-level parallelism.
//
// For the mixed u8×s8 path VPMADDWD's int16 operands are exact: a zero-extended
// uint8 (0..255) and a sign-extended int8 are both representable as int16 and
// their product fits in int16's range, which the wider int32 pairwise add
// absorbs.
//
// The YMM accumulator is folded to a scalar with VEXTRACTI128 + VPADDD + two
// VPHADDD + VMOVD, and a scalar tail finishes the < stride remainder. Integer
// MAC is exact, so every kernel is bit-for-bit equal to the scalar reference.
//
// AVX-512-VNNI (VPDPBUSD) note: VPDPBUSD is the native u8×s8 multiply-accumulate
// and would be the ideal DotU8S8 kernel, but it cannot be exercised in this
// project's validation environment — qemu's TCG does not implement any AVX-512,
// so neither the differential tests nor the coverage gate can cover that path on
// CI. Rather than ship an unvalidated, uncovered kernel, amd64 uses the AVX2
// path (validated bit-exact on a native x86-64 VM) for all three operations;
// adding a VPDPBUSD fast path is a clean follow-up once a VNNI-capable,
// test-covered runner is available. See README "Architecture support".
//
// Run: go run kernels_amd64_gen.go
package main

import (
	"fmt"
	"os"

	"github.com/go-asmgen/asmgen/abi"
	"github.com/go-asmgen/asmgen/amd64"
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
	f := emit.NewFile("amd64")
	f.Add(avx2Kernel("dotAVX2", "VPMOVSXBW", "VPMOVSXBW", "MOVBQSX", "MOVBQSX", dotSig()).Func())
	f.Add(avx2Kernel("dotUint8AVX2", "VPMOVZXBW", "VPMOVZXBW", "MOVBQZX", "MOVBQZX", dotSigU()).Func())
	f.Add(avx2Kernel("dotU8S8AVX2", "VPMOVZXBW", "VPMOVSXBW", "MOVBQZX", "MOVBQSX", dotSig()).Func())
	if err := os.WriteFile("kernels_amd64.s", []byte(f.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote kernels_amd64.s")
}

// foldY0 reduces the two 8×int32 YMM accumulators (Y0, Y6) to the int32 in SI
// (the scalar accumulator; AX/BX still hold the a/b base pointers for the tail).
func foldY0(b *amd64.Builder) {
	b.Raw("VPADDD Y6, Y0, Y0"). // combine the two accumulators
					Raw("VEXTRACTI128 $1, Y0, X1"). // high 128 (4 lanes)
					Raw("VPADDD X1, X0, X0").       // 4 lanes
					Raw("VPHADDD X0, X0, X0").      // -> 2 partial sums
					Raw("VPHADDD X0, X0, X0").      // low lane = total
					Raw("VMOVD X0, SI").
					Raw("VZEROUPPER")
}

// avx2Kernel emits the byte→word→int32 VPMADDWD kernel. extA/extB are the
// byte→word extends for a and b (VPMOVSXBW signed / VPMOVZXBW unsigned); loadA/
// loadB are the matching scalar-tail loads.
func avx2Kernel(name, extA, extB, loadA, loadB string, sig abi.Signature) *amd64.Builder {
	b := amd64.NewFunc(name, sig, 0)
	b.LoadArg("a_base", "AX").LoadArg("a_len", "CX").LoadArg("b_base", "BX").
		Raw("VPXOR Y0, Y0, Y0"). // accumulator 0
		Raw("VPXOR Y6, Y6, Y6"). // accumulator 1
		Raw("XORQ DI, DI").      // i = 0
		Label("vloop").          // stride 32
		Raw("LEAQ 32(DI), R8").Raw("CMPQ R8, CX").Raw("JGT vtail").
		// first 16 bytes -> Y0
		Raw(extA + " (AX)(DI*1), Y1"). // 16 a-words
		Raw(extB + " (BX)(DI*1), Y2"). // 16 b-words
		Raw("VPMADDWD Y2, Y1, Y1").    // 8 int32 pair-sums
		Raw("VPADDD Y1, Y0, Y0").
		// next 16 bytes -> Y6
		Raw(extA + " 16(AX)(DI*1), Y3").
		Raw(extB + " 16(BX)(DI*1), Y4").
		Raw("VPMADDWD Y4, Y3, Y3").
		Raw("VPADDD Y3, Y6, Y6").
		Raw("ADDQ $32, DI").Raw("JMP vloop").
		Label("vtail")
	foldY0(b)
	b.Label("sloop").
		Raw("CMPQ DI, CX").Raw("JGE done").
		Raw(loadA+" (AX)(DI*1), R9").
		Raw(loadB+" (BX)(DI*1), R10").
		Raw("IMULQ R10, R9").
		Raw("ADDQ R9, SI").
		Raw("ADDQ $1, DI").Raw("JMP sloop").
		Label("done").StoreRet("SI", "ret").Ret()
	return b
}
