package int8dot

import (
	"math/rand"
	"testing"
)

// Benchmarks compare the SIMD kernel against the scalar reference at a typical
// embedding dimension. Run e.g. `go test -bench . -benchmem`.

var benchSizes = []int{64, 256, 768, 4096}

func benchData(n int) ([]int8, []int8, []uint8, []uint8) {
	r := rand.New(rand.NewSource(7))
	return randI8(n, r), randI8(n, r), randU8(n, r), randU8(n, r)
}

func BenchmarkDot(b *testing.B) {
	for _, n := range benchSizes {
		a, c, _, _ := benchData(n)
		b.Run(name(n), func(b *testing.B) {
			b.SetBytes(int64(n))
			for i := 0; i < b.N; i++ {
				sink32 = Dot(a, c)
			}
		})
	}
}

func BenchmarkDotScalar(b *testing.B) {
	for _, n := range benchSizes {
		a, c, _, _ := benchData(n)
		b.Run(name(n), func(b *testing.B) {
			b.SetBytes(int64(n))
			for i := 0; i < b.N; i++ {
				sink32 = dotRef(a, c)
			}
		})
	}
}

func BenchmarkDotUint8(b *testing.B) {
	for _, n := range benchSizes {
		_, _, u, v := benchData(n)
		b.Run(name(n), func(b *testing.B) {
			b.SetBytes(int64(n))
			for i := 0; i < b.N; i++ {
				sinkU32 = DotUint8(u, v)
			}
		})
	}
}

func BenchmarkDotU8S8(b *testing.B) {
	for _, n := range benchSizes {
		_, c, u, _ := benchData(n)
		b.Run(name(n), func(b *testing.B) {
			b.SetBytes(int64(n))
			for i := 0; i < b.N; i++ {
				sink32 = DotU8S8(u, c)
			}
		})
	}
}

var (
	sink32  int32
	sinkU32 uint32
)

func name(n int) string {
	switch n {
	case 64:
		return "64"
	case 256:
		return "256"
	case 768:
		return "768"
	default:
		return "4096"
	}
}
