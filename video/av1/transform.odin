package av1

// Normative 1-D and 2-D inverse transforms (section 7.13) and dequantization
// helpers (section 7.12). Vectorized across lanes: lane l of T[k] is sample k
// of line l, processing 4 or 8 rows/columns simultaneously.

import common "../../common"
import "base:intrinsics"
import "core:simd"

Tx_1D :: enum u8 {
	DCT,
	ADST,
	FLIPADST,
	IDTX,
}

// {row_1d, col_1d} for each of the 16 2-D transform types (7.13.3).
@(rodata)
TX_1D_TYPES := [TX_TYPES][2]Tx_1D{
	DCT_DCT           = {.DCT, .DCT},
	ADST_DCT          = {.DCT, .ADST},
	DCT_ADST          = {.ADST, .DCT},
	ADST_ADST         = {.ADST, .ADST},
	FLIPADST_DCT      = {.DCT, .FLIPADST},
	DCT_FLIPADST      = {.FLIPADST, .DCT},
	FLIPADST_FLIPADST = {.FLIPADST, .FLIPADST},
	ADST_FLIPADST     = {.FLIPADST, .ADST},
	FLIPADST_ADST     = {.ADST, .FLIPADST},
	IDTX              = {.IDTX, .IDTX},
	V_DCT             = {.IDTX, .DCT},
	H_DCT             = {.DCT, .IDTX},
	V_ADST            = {.IDTX, .ADST},
	H_ADST            = {.ADST, .IDTX},
	V_FLIPADST        = {.IDTX, .FLIPADST},
	H_FLIPADST        = {.FLIPADST, .IDTX},
}

SINPI_1_9 :: 1321
SINPI_2_9 :: 2482
SINPI_3_9 :: 3344
SINPI_4_9 :: 3803

round2 :: #force_inline proc "contextless" (x: i64, n: uint) -> i32 {
	if n == 0 do return i32(x)
	return i32((x + (i64(1) << (n - 1))) >> n)
}

round2_signed :: #force_inline proc "contextless" (x: i64, n: uint) -> i32 {
	return x >= 0 ? round2(x, n) : -round2(-x, n)
}

ITX_MAX_LANES :: 16 when intrinsics.has_target_feature("avx512f") else (8 when intrinsics.has_target_feature("avx2") else 4)

round2_lanes :: #force_inline proc "contextless" (x: $V/#simd[$L]$E, n: uint) -> V {
	if n == 0 do return x
	sh := n & 15
	half := V(E(1) << ((n - 1) & 15))
	// `simd.shr_masked` masks the shift count to the bit-width minus 1 instead of
	// emitting Odin's default `simd.shr` saturation bounds check (`vpminud + vpcmpgtud + vpandnq`),
	// compiling to a single vector arithmetic right shift (`vpsrad` / `vpsravd`).
	when E == i32 {
		return simd.shr_masked(x + half, (#simd[L]u32)(sh))
	} else {
		return simd.shr_masked(x + half, (#simd[L]u64)(sh))
	}
}

clamp_lanes :: #force_inline proc "contextless" (x: $V/#simd[$L]$E, r: uint, $R16: bool) -> V {
	// When `$R16` is known at compile time (`r == 16` for all 8-bit/10-bit column transforms
	// and 8-bit row transforms), clamping bounds become compile-time immediates `[-32768, 32767]`
	// instead of computing `1 << clamp(r - 1, 15, 23)` and splatting runtime vectors on every call.
	when R16 && E == i32 {
		return simd.clamp(x, V(-32768), V(32767))
	} else {
		hi := E(1) << clamp(r - 1, 15, 23)
		return simd.clamp(x, V(-hi), V(hi - 1))
	}
}

cos128 :: #force_inline proc "contextless" (angle: int) -> i64 {
	// Because `COS128_LOOKUP` is `@(rodata)` and `cos128` is `#force_inline`, LLVM
	// constant-folds this entire function to an immediate whenever `angle` is constant.
	a := angle & 255
	switch {
	case a <= 64:
		return i64(COS128_LOOKUP[a])
	case a <= 128:
		return -i64(COS128_LOOKUP[128 - a])
	case a <= 192:
		return -i64(COS128_LOOKUP[a - 128])
	}
	return i64(COS128_LOOKUP[256 - a])
}

sin128 :: #force_inline proc "contextless" (angle: int) -> i64 {
	return cos128(angle - 64)
}

brev :: #force_inline proc "contextless" (bits: uint, x: int) -> int {
	t := 0
	for i in 0 ..< bits do t |= ((x >> i) & 1) << (bits - 1 - i)
	return t
}

// 7.13.2.1: Butterfly rotation B(a, b, angle, flip, r).
//
// Performance note:
// When `$R16` is true (`r == 16`), inputs `t[a]` and `t[b]` are guaranteed by the AV1 spec
// to fit in signed 16-bit `[-32768, 32767]` and `cos128`/`sin128` fit in `[-4096, 4096]`.
// Packing `(t[a], t[b])` into interleaved `i16` pairs (`#simd[2*L]i16`) and widening to
// `#simd[2*L]i32` inside `simd.sums_of_n(ab * cs, 2)` matches LLVM's pairwise multiply-add
// pattern (`sext(v2Ni16) * sext(v2Ni16)` + pairwise horizontal add), lowering directly to a
// single `vpdpwssd` / `vpmaddwd` (or NEON `smull + saddlp`) instruction with a `.rodata`
// broadcast operand (`DWORD BCST [rip+...]`) instead of 2 multiplies + 1 add/sub + 2 shifts.
@(private)
B :: #force_inline proc "contextless" (t: ^[$M]$V/#simd[$L]$E, a, b, angle: int, flip: bool, $R16: bool) {
	when R16 && E == i32 {
		V2I16 :: #simd[2 * L]i16
		V2I32 :: #simd[2 * L]i32
		VU32  :: #simd[L]u32
		c := i32(cos128(angle))
		s := i32(sin128(angle))
		ua := transmute(VU32)t[a] & VU32(0xFFFF)
		ub := simd.shl_masked(transmute(VU32)t[b], VU32(16))
		ab := V2I32(transmute(V2I16)(ua | ub))
		cs_x := V2I32(transmute(V2I16)VU32((u32(c) & 0xFFFF) | (u32(-s) << 16)))
		cs_y := V2I32(transmute(V2I16)VU32((u32(s) & 0xFFFF) | (u32(c) << 16)))
		half := V(1 << 11)
		sh := VU32(12)
		x := simd.shr_masked(half + simd.sums_of_n(ab * cs_x, 2), sh)
		y := simd.shr_masked(half + simd.sums_of_n(ab * cs_y, 2), sh)
		if flip {
			t[a], t[b] = y, x
		} else {
			t[a], t[b] = x, y
		}
	} else {
		c, s := V(E(cos128(angle))), V(E(sin128(angle)))
		x := t[a] * c - t[b] * s
		y := t[a] * s + t[b] * c
		t[a], t[b] = round2_lanes(x, 12), round2_lanes(y, 12)
		if flip do t[a], t[b] = t[b], t[a]
	}
}

// 7.13.2.1: Hadamard rotation H(a, b, flip, r) with clamping to r bits.
@(private)
H :: #force_inline proc "contextless" (t: ^[$M]$V, a, b: int, flip: bool, r: uint, $R16: bool) {
	a, b := a, b
	if flip do a, b = b, a
	x, y := t[a], t[b]
	t[a] = clamp_lanes(x + y, r, R16)
	t[b] = clamp_lanes(x - y, r, R16)
}

// 7.13.2.2 - 7.13.2.3: 1-D Inverse DCT of length 1 << n (n in 2..6),
// ordered by sub-block (0..3, 4..7, 8..15, 16..31, 32..63) to minimize live vector registers
// and skip all-zero high-frequency sub-blocks when `max_in` is small.
@(private)
idct_1d :: #force_inline proc "contextless" (t: ^[$M]$V, $n: uint, r: uint, max_in := 32, $R16: bool) {
	#unroll for i in 0 ..< (1 << n) {
		if j := brev(n, i); i < j {
			t[i], t[j] = t[j], t[i]
		}
	}

	// Stage 0..3 (length 4).
	#unroll for i in 0 ..< 2 do B(t, 2 * i, 2 * i + 1, 32 + 16 * i, i == 0, R16)
	#unroll for i in 0 ..< 2 do H(t, i, 3 - i, false, r, R16)

	// Stage 4..7 -> 0..7 (length 8).
	when n >= 3 {
		when n >= 5 {
			if max_in <= (1 << (n - 3)) {
				#unroll for i in 0 ..< 4 {
					t[7 - i] = t[i]
				}
			} else {
				#unroll for i in 0 ..< 2 do B(t, 4 + i, 7 - i, 56 - 32 * i, false, R16)
				#unroll for i in 0 ..< 2 do H(t, 4 + 2 * i, 5 + 2 * i, i != 0, r, R16)
				B(t, 6, 5, 32, true, R16)
				#unroll for i in 0 ..< 4 do H(t, i, 7 - i, false, r, R16)
			}
		} else {
			#unroll for i in 0 ..< 2 do B(t, 4 + i, 7 - i, 56 - 32 * i, false, R16)
			#unroll for i in 0 ..< 2 do H(t, 4 + 2 * i, 5 + 2 * i, i != 0, r, R16)
			B(t, 6, 5, 32, true, R16)
			#unroll for i in 0 ..< 4 do H(t, i, 7 - i, false, r, R16)
		}
	}

	// Stage 8..15 -> 0..15 (length 16).
	when n >= 4 {
		when n == 6 {
			if max_in <= 4 {
				#unroll for i in 0 ..< 8 {
					t[15 - i] = t[i]
				}
			} else {
				#unroll for i in 0 ..< 4 do B(t, 8 + i, 15 - i, 12 + (brev(2, 3 - i) << 4), false, R16)
				#unroll for i in 0 ..< 4 do H(t, 8 + 2 * i, 9 + 2 * i, (i & 1) != 0, r, R16)
				#unroll for i in 0 ..< 2 do B(t, 14 - i, 9 + i, 48 + 64 * i, true, R16)
				#unroll for i in 0 ..< 2 do #unroll for j in 0 ..< 2 {
					H(t, 8 + 4 * i + j, 11 + 4 * i - j, i != 0, r, R16)
				}
				#unroll for i in 0 ..< 2 do B(t, 13 - i, 10 + i, 32, true, R16)
				#unroll for i in 0 ..< 8 do H(t, i, 15 - i, false, r, R16)
			}
		} else {
			#unroll for i in 0 ..< 4 do B(t, 8 + i, 15 - i, 12 + (brev(2, 3 - i) << 4), false, R16)
			#unroll for i in 0 ..< 4 do H(t, 8 + 2 * i, 9 + 2 * i, (i & 1) != 0, r, R16)
			#unroll for i in 0 ..< 2 do B(t, 14 - i, 9 + i, 48 + 64 * i, true, R16)
			#unroll for i in 0 ..< 2 do #unroll for j in 0 ..< 2 {
				H(t, 8 + 4 * i + j, 11 + 4 * i - j, i != 0, r, R16)
			}
			#unroll for i in 0 ..< 2 do B(t, 13 - i, 10 + i, 32, true, R16)
			#unroll for i in 0 ..< 8 do H(t, i, 15 - i, false, r, R16)
		}
	}

	// Stage 16..31 -> 0..31 (length 32).
	when n >= 5 {
		#unroll for i in 0 ..< 8 do B(t, 16 + i, 31 - i, 6 + (brev(3, 7 - i) << 3), false, R16)
		#unroll for i in 0 ..< 8 do H(t, 16 + 2 * i, 17 + 2 * i, (i & 1) != 0, r, R16)
		#unroll for i in 0 ..< 2 do #unroll for j in 0 ..< 2 {
			B(t, 30 - 4 * i - j, 17 + 4 * i + j, 24 + (j << 6) + ((1 - i) << 5), true, R16)
		}
		#unroll for i in 0 ..< 4 do #unroll for j in 0 ..< 2 {
			H(t, 16 + 4 * i + j, 19 + 4 * i - j, (i & 1) != 0, r, R16)
		}
		#unroll for i in 0 ..< 4 do B(t, 29 - i, 18 + i, 48 + (i >> 1) * 64, true, R16)
		#unroll for i in 0 ..< 2 do #unroll for j in 0 ..< 4 {
			H(t, 16 + i * 8 + j, 23 + i * 8 - j, i != 0, r, R16)
		}
		#unroll for i in 0 ..< 4 do B(t, 27 - i, 20 + i, 32, true, R16)
		#unroll for i in 0 ..< 16 do H(t, i, 31 - i, false, r, R16)
	}

	// Stage 32..63 -> 0..63 (length 64).
	when n == 6 {
		#unroll for i in 0 ..< 16 do B(t, 32 + i, 63 - i, 63 - 4 * brev(4, i), false, R16)
		#unroll for i in 0 ..< 16 do H(t, 32 + i * 2, 33 + i * 2, (i & 1) != 0, r, R16)
		#unroll for i in 0 ..< 4 do #unroll for j in 0 ..< 2 {
			B(t, 62 - i * 4 - j, 33 + i * 4 + j, 60 - 16 * brev(2, i) + 64 * j, true, R16)
		}
		#unroll for i in 0 ..< 8 do #unroll for j in 0 ..< 2 {
			H(t, 32 + i * 4 + j, 35 + i * 4 - j, (i & 1) != 0, r, R16)
		}
		#unroll for i in 0 ..< 2 do #unroll for j in 0 ..< 4 {
			B(t, 61 - i * 8 - j, 34 + i * 8 + j, 56 - i * 32 + (j >> 1) * 64, true, R16)
		}
		#unroll for i in 0 ..< 4 do #unroll for j in 0 ..< 4 {
			H(t, 32 + 8 * i + j, 39 + 8 * i - j, (i & 1) != 0, r, R16)
		}
		#unroll for i in 0 ..< 8 do B(t, 59 - i, 36 + i, i < 4 ? 48 : 112, true, R16)
		#unroll for i in 0 ..< 8 {
			H(t, 32 + i, 47 - i, false, r, R16)
			H(t, 48 + i, 63 - i, true, r, R16)
		}
		#unroll for i in 0 ..< 8 do B(t, 55 - i, 40 + i, 32, true, R16)
		#unroll for i in 0 ..< 32 do H(t, i, 63 - i, false, r, R16)
	}
}

// 7.13.2.4: Inverse ADST input array permutation.
@(private)
adst_input_permute :: #force_inline proc "contextless" (t: ^[$M]$V, $n: uint) {
	N0 :: 1 << n
	c := t^
	#unroll for i in 0 ..< N0 {
		idx := (i & 1) != 0 ? (i - 1) : (N0 - i - 1)
		t[i] = c[idx]
	}
}

// 7.13.2.5: Inverse ADST output array permutation.
@(private)
adst_output_permute :: #force_inline proc "contextless" (t: ^[$M]$V, $n: uint) {
	N0 :: 1 << n
	c := t^
	#unroll for i in 0 ..< N0 {
		a := (i >> 3) & 1
		b := ((i >> 2) & 1) ~ ((i >> 3) & 1)
		cc := ((i >> 1) & 1) ~ ((i >> 2) & 1)
		d := (i & 1) ~ ((i >> 1) & 1)
		idx := ((d << 3) | (cc << 2) | (b << 1) | a) >> (4 - n)
		t[i] = (i & 1) != 0 ? -c[idx] : c[idx]
	}
}

// 7.13.2.6: Inverse ADST4.
@(private)
iadst4 :: #force_inline proc "contextless" (t: ^[$M]$V) {
	s0 := V(SINPI_1_9) * t[0]
	s1 := V(SINPI_2_9) * t[0]
	s2 := V(SINPI_3_9) * t[1]
	s3 := V(SINPI_4_9) * t[2]
	s4 := V(SINPI_1_9) * t[2]
	s5 := V(SINPI_2_9) * t[3]
	s6 := V(SINPI_4_9) * t[3]
	b7 := (t[0] - t[2]) + t[3]
	s0 = s0 + s3 + s5
	s1 = s1 - s4 - s6
	s3 = s2
	s2 = V(SINPI_3_9) * b7
	x0 := s0 + s3
	x1 := s1 + s3
	x2 := s2
	x3 := s0 + s1 - s3
	t[0] = round2_lanes(x0, 12)
	t[1] = round2_lanes(x1, 12)
	t[2] = round2_lanes(x2, 12)
	t[3] = round2_lanes(x3, 12)
}

// 7.13.2.7: Inverse ADST8.
@(private)
iadst8 :: #force_inline proc "contextless" (t: ^[$M]$V, r: uint, $R16: bool) {
	adst_input_permute(t, 3)
	#unroll for i in 0 ..< 4 do B(t, 2 * i, 2 * i + 1, 60 - 16 * i, true, R16)
	#unroll for i in 0 ..< 4 do H(t, i, 4 + i, false, r, R16)
	#unroll for i in 0 ..< 2 do B(t, 4 + 3 * i, 5 + i, 48 - 32 * i, true, R16)
	#unroll for j in 0 ..< 2 do #unroll for i in 0 ..< 2 do H(t, 4 * j + i, 2 + 4 * j + i, false, r, R16)
	#unroll for i in 0 ..< 2 do B(t, 2 + 4 * i, 3 + 4 * i, 32, true, R16)
	adst_output_permute(t, 3)
}

// 7.13.2.8: Inverse ADST16.
@(private)
iadst16 :: #force_inline proc "contextless" (t: ^[$M]$V, r: uint, $R16: bool) {
	adst_input_permute(t, 4)
	#unroll for i in 0 ..< 8 do B(t, 2 * i, 2 * i + 1, 62 - 8 * i, true, R16)
	#unroll for i in 0 ..< 8 do H(t, i, 8 + i, false, r, R16)
	#unroll for i in 0 ..< 2 {
		B(t, 8 + 2 * i, 9 + 2 * i, 56 - 32 * i, true, R16)
		B(t, 13 + 2 * i, 12 + 2 * i, 8 + 32 * i, true, R16)
	}
	#unroll for j in 0 ..< 2 do #unroll for i in 0 ..< 4 do H(t, 8 * j + i, 4 + 8 * j + i, false, r, R16)
	#unroll for j in 0 ..< 2 do #unroll for i in 0 ..< 2 do B(t, 4 + 8 * j + 3 * i, 5 + 8 * j + i, 48 - 32 * i, true, R16)
	#unroll for j in 0 ..< 4 do #unroll for i in 0 ..< 2 do H(t, 4 * j + i, 2 + 4 * j + i, false, r, R16)
	#unroll for i in 0 ..< 4 do B(t, 2 + 4 * i, 3 + 4 * i, 32, true, R16)
	adst_output_permute(t, 4)
}

// 7.13.2.11 - 7.13.2.15: Inverse identity transform of length 1 << n (n in 2..5).
@(private)
iidtx_1d :: #force_inline proc "contextless" (t: ^[$M]$V, $n: uint) {
	N0 :: 1 << n
	when n == 2 {
		#unroll for i in 0 ..< N0 do t[i] = round2_lanes(t[i] * V(5793), 12)
	} else when n == 3 {
		#unroll for i in 0 ..< N0 do t[i] = t[i] * V(2)
	} else when n == 4 {
		#unroll for i in 0 ..< N0 do t[i] = round2_lanes(t[i] * V(11586), 12)
	} else when n == 5 {
		#unroll for i in 0 ..< N0 do t[i] = t[i] * V(4)
	}
}

@(private)
itx_1d_impl :: #force_inline proc "contextless" (t: ^[$M]$V, $n: uint, kind: Tx_1D, r: uint, max_in: int = 32, $R16: bool) {
	N0 :: 1 << n
	when n >= 5 {
		when n == 5 {
			if kind == .IDTX {
				iidtx_1d(t, 5)
				return
			}
		}
		idct_1d(t, n, r, max_in, R16)
	} else {
		switch kind {
		case .DCT:
			idct_1d(t, n, r, max_in, R16)
		case .ADST, .FLIPADST:
			when n == 2 {
				iadst4(t)
			} else when n == 3 {
				iadst8(t, r, R16)
			} else {
				iadst16(t, r, R16)
			}
			if kind == .FLIPADST {
				#unroll for i in 0 ..< (N0 >> 1) {
					t[i], t[N0 - 1 - i] = t[N0 - 1 - i], t[i]
				}
			}
		case .IDTX:
			iidtx_1d(t, n)
		}
	}
}

// Dispatches a 1-D transform of length 1 << n on lanes.
@(private)
itx_1d :: proc "contextless" (t: ^[$M]$V, $n: uint, kind: Tx_1D, r: uint, max_in: int = 32) {
	E :: intrinsics.type_elem_type(V)
	when E == i32 {
		if r == 16 {
			itx_1d_impl(t, n, kind, 16, max_in, true)
			return
		}
	}
	itx_1d_impl(t, n, kind, r, max_in, false)
}

// 7.13.2.10: 1-D Inverse Walsh-Hadamard transform (length 4).
@(private)
iwht4 :: #force_inline proc "contextless" (t: ^[4]i32, shift: uint) {
	a := t[0] >> shift
	c := t[1] >> shift
	d := t[2] >> shift
	b := t[3] >> shift
	a += c
	d -= b
	e := (a - d) >> 1
	b = e - b
	c = e - c
	a -= b
	d += c
	t[0], t[1], t[2], t[3] = a, b, c, d
}

// 2-D lossless 4x4 WHT in place (7.13.3).
inverse_wht4x4 :: proc "contextless" (block: []i32, bit_depth := 8) #no_bounds_check {
	col_clamp := uint(max(bit_depth + 6, 16))
	hi := i32(1) << (col_clamp - 1)
	for i in 0 ..< 4 {
		row := [4]i32{block[i * 4], block[i * 4 + 1], block[i * 4 + 2], block[i * 4 + 3]}
		iwht4(&row, 2)
		for j in 0 ..< 4 do block[i * 4 + j] = clamp(row[j], -hi, hi - 1)
	}
	for j in 0 ..< 4 {
		col := [4]i32{block[j], block[4 + j], block[8 + j], block[12 + j]}
		iwht4(&col, 0)
		for i in 0 ..< 4 do block[i * 4 + j] = col[i]
	}
}

// Constant residual produced by a DC-only DCT_DCT block of size tx_size (7.13.3).
dc_only_residual :: proc "contextless" (c0: i32, tx_size: int, bit_depth := 8) -> i32 #no_bounds_check {
	log2_w := int(TX_WIDTH_LOG2[tx_size])
	log2_h := int(TX_HEIGHT_LOG2[tx_size])
	row_shift := uint(TRANSFORM_ROW_SHIFT[tx_size])
	col_clamp := uint(max(bit_depth + 6, 16))
	hi := i32(1) << (col_clamp - 1)
	t0 := c0
	if abs(log2_w - log2_h) == 1 do t0 = round2(i64(t0) * 2896, 12)
	t0 = round2(i64(t0) * 2896, 12)
	r0 := clamp(round2(i64(t0), row_shift), -hi, hi - 1)
	c1 := round2(i64(r0) * 2896, 12)
	return round2(i64(c1), 4)
}

// 7.13.3: 2-D inverse transform of `block` (w * h raster order, where coefficients
// outside the top-left min(w,32) x min(h,32) region are zero) in place.
inverse_transform_2d :: proc(block: []i32, tx_size, tx_type: int, lossless: bool, eob := 1024, bit_depth := 8, max_r := 31, max_c := 31) #no_bounds_check {
	if eob <= 0 do return
	if lossless {
		inverse_wht4x4(block, bit_depth)
		return
	}
	w := int(TX_WIDTH[tx_size])
	h := int(TX_HEIGHT[tx_size])
	if eob == 1 && tx_type == DCT_DCT {
		dc := dc_only_residual(block[0], tx_size, bit_depth)
		v := (#simd[4]i32)(dc)
		n := w * h
		for idx := 0; idx < n; idx += 4 {
			intrinsics.unaligned_store((^#simd[4]i32)(&block[idx]), v)
		}
		return
	}
	src_tmp: [64 * 64]i32 = ---
	copy(src_tmp[:w * h], block[:w * h])
	if bit_depth > 10 {
		itx_2d_lanes(block, src_tmp[:w * h], false, tx_size, tx_type, bit_depth, max_r, max_c, i64)
	} else {
		itx_2d_lanes(block, src_tmp[:w * h], false, tx_size, tx_type, bit_depth, max_r, max_c, i32)
	}
}

// 7.13.3: 2-D inverse transform from `src` (`dequant_buf`) into `dst` (`itx_buf`),
// zeroing only the active `active_h x active_w` entries of `src` as they are read.
inverse_transform_2d_clear :: proc "contextless" (dst, src: []i32, tx_size, tx_type, bit_depth, max_r, max_c: int) #no_bounds_check {
	if bit_depth > 10 {
		itx_2d_lanes(dst, src, true, tx_size, tx_type, bit_depth, max_r, max_c, i64)
	} else {
		itx_2d_lanes(dst, src, true, tx_size, tx_type, bit_depth, max_r, max_c, i32)
	}
}

// Transposes 4×4 sub-matrices in-lane across all 128-bit (4-lane) sub-vectors of `m`
// simultaneously (`4` independent 4×4 transposes for `L == 16` in `zmm`, or `2` for `L == 8`
// in `ymm`) in 8 `vshufps` / `vpunpck*` instructions, avoiding extracting/inserting 128-bit
// lanes before and after transposing.
@(private)
transpose4_lanes :: #force_inline proc "contextless" (m: [4]$V/#simd[$L]$E) -> [4]V {
	when L == 16 {
		t0 := simd.shuffle(m[0], m[1], 0, 16, 1, 17, 4, 20, 5, 21, 8, 24, 9, 25, 12, 28, 13, 29)
		t1 := simd.shuffle(m[0], m[1], 2, 18, 3, 19, 6, 22, 7, 23, 10, 26, 11, 27, 14, 30, 15, 31)
		t2 := simd.shuffle(m[2], m[3], 0, 16, 1, 17, 4, 20, 5, 21, 8, 24, 9, 25, 12, 28, 13, 29)
		t3 := simd.shuffle(m[2], m[3], 2, 18, 3, 19, 6, 22, 7, 23, 10, 26, 11, 27, 14, 30, 15, 31)
		return {
			simd.shuffle(t0, t2, 0, 1, 16, 17, 4, 5, 20, 21, 8, 9, 24, 25, 12, 13, 28, 29),
			simd.shuffle(t0, t2, 2, 3, 18, 19, 6, 7, 22, 23, 10, 11, 26, 27, 14, 15, 30, 31),
			simd.shuffle(t1, t3, 0, 1, 16, 17, 4, 5, 20, 21, 8, 9, 24, 25, 12, 13, 28, 29),
			simd.shuffle(t1, t3, 2, 3, 18, 19, 6, 7, 22, 23, 10, 11, 26, 27, 14, 15, 30, 31),
		}
	} else when L == 8 {
		t0 := simd.shuffle(m[0], m[1], 0, 8, 1, 9, 4, 12, 5, 13)
		t1 := simd.shuffle(m[0], m[1], 2, 10, 3, 11, 6, 14, 7, 15)
		t2 := simd.shuffle(m[2], m[3], 0, 8, 1, 9, 4, 12, 5, 13)
		t3 := simd.shuffle(m[2], m[3], 2, 10, 3, 11, 6, 14, 7, 15)
		return {
			simd.shuffle(t0, t2, 0, 1, 8, 9, 4, 5, 12, 13),
			simd.shuffle(t0, t2, 2, 3, 10, 11, 6, 7, 14, 15),
			simd.shuffle(t1, t3, 0, 1, 8, 9, 4, 5, 12, 13),
			simd.shuffle(t1, t3, 2, 3, 10, 11, 6, 7, 14, 15),
		}
	} else {
		return common.transpose(m)
	}
}

@(private)
itx_row_group :: proc "contextless" (dst, src: []i32, $log2_w: uint, $MAX_IN: int, $E: typeid, $L: int, y0, w, active_w: int, rect_scale: bool, row_kind: Tx_1D, row_shift, row_clamp, col_clamp: uint) #no_bounds_check {
	W0 :: 1 << log2_w
	V :: #simd[L]E
	V4 :: #simd[4]E
	W4 :: #simd[4]i32
	t: [W0]V = ---
	w_zero := W4(0)
	#unroll for b in 0 ..< MAX_IN / 4 {
		jb := b * 4
		if MAX_IN <= 8 || jb < active_w {
			tile: [4]V = ---
			when L == 16 {
				#unroll for l in 0 ..< 4 {
					p0 := (^W4)(&src[(y0 + l) * w + jb])
					p1 := (^W4)(&src[(y0 + 4 + l) * w + jb])
					p2 := (^W4)(&src[(y0 + 8 + l) * w + jb])
					p3 := (^W4)(&src[(y0 + 12 + l) * w + jb])
					q0 := V4(intrinsics.unaligned_load(p0))
					q1 := V4(intrinsics.unaligned_load(p1))
					q2 := V4(intrinsics.unaligned_load(p2))
					q3 := V4(intrinsics.unaligned_load(p3))
					intrinsics.unaligned_store(p0, w_zero)
					intrinsics.unaligned_store(p1, w_zero)
					intrinsics.unaligned_store(p2, w_zero)
					intrinsics.unaligned_store(p3, w_zero)
					v01 := simd.shuffle(q0, q1, 0, 1, 2, 3, 4, 5, 6, 7)
					v23 := simd.shuffle(q2, q3, 0, 1, 2, 3, 4, 5, 6, 7)
					tile[l] = simd.shuffle(v01, v23, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)
				}
			} else when L == 8 {
				#unroll for l in 0 ..< 4 {
					p0 := (^W4)(&src[(y0 + l) * w + jb])
					p1 := (^W4)(&src[(y0 + 4 + l) * w + jb])
					lo := V4(intrinsics.unaligned_load(p0))
					hi := V4(intrinsics.unaligned_load(p1))
					intrinsics.unaligned_store(p0, w_zero)
					intrinsics.unaligned_store(p1, w_zero)
					tile[l] = simd.shuffle(lo, hi, 0, 1, 2, 3, 4, 5, 6, 7)
				}
			} else {
				#unroll for l in 0 ..< 4 {
					p0 := (^W4)(&src[(y0 + l) * w + jb])
					tile[l] = V4(intrinsics.unaligned_load(p0))
					intrinsics.unaligned_store(p0, w_zero)
				}
			}
			tile = transpose4_lanes(tile)
			#unroll for k in 0 ..< 4 {
				v := tile[k]
				if rect_scale do v = round2_lanes(v * V(2896), 12)
				t[jb + k] = v
			}
		} else {
			#unroll for k in 0 ..< 4 {
				t[jb + k] = 0
			}
		}
	}
	when W0 > MAX_IN {
		#unroll for j in MAX_IN ..< W0 do t[j] = 0
	}
	itx_1d(&t, log2_w, row_kind, row_clamp, MAX_IN)
	row_half := V(row_shift == 0 ? 0 : E(1) << ((row_shift - 1) & 15))
	when E == i32 {
		row_sh_v := (#simd[L]u32)(row_shift & 15)
	} else {
		row_sh_v := (#simd[L]u64)(row_shift & 15)
	}
	#unroll for b in 0 ..< W0 / 4 {
		jb := b * 4
		tile: [4]V = ---
		#unroll for k in 0 ..< 4 {
			tile[k] = clamp_lanes(simd.shr_masked(t[jb + k] + row_half, row_sh_v), col_clamp, E == i32)
		}
		tile = transpose4_lanes(tile)
		when L == 16 {
			#unroll for l in 0 ..< 4 {
				v := tile[l]
				q0 := simd.shuffle(v, v, 0, 1, 2, 3)
				q1 := simd.shuffle(v, v, 4, 5, 6, 7)
				q2 := simd.shuffle(v, v, 8, 9, 10, 11)
				q3 := simd.shuffle(v, v, 12, 13, 14, 15)
				intrinsics.unaligned_store((^W4)(&dst[(y0 + l) * w + jb]), W4(q0))
				intrinsics.unaligned_store((^W4)(&dst[(y0 + 4 + l) * w + jb]), W4(q1))
				intrinsics.unaligned_store((^W4)(&dst[(y0 + 8 + l) * w + jb]), W4(q2))
				intrinsics.unaligned_store((^W4)(&dst[(y0 + 12 + l) * w + jb]), W4(q3))
			}
		} else when L == 8 {
			#unroll for l in 0 ..< 4 {
				v := tile[l]
				lo := simd.shuffle(v, v, 0, 1, 2, 3)
				hi := simd.shuffle(v, v, 4, 5, 6, 7)
				intrinsics.unaligned_store((^W4)(&dst[(y0 + l) * w + jb]), W4(lo))
				intrinsics.unaligned_store((^W4)(&dst[(y0 + 4 + l) * w + jb]), W4(hi))
			}
		} else {
			#unroll for l in 0 ..< 4 {
				intrinsics.unaligned_store((^W4)(&dst[(y0 + l) * w + jb]), W4(tile[l]))
			}
		}
	}
}

@(private)
itx_col_group :: proc "contextless" (block: []i32, $log2_h: uint, $MAX_IN: int, $E: typeid, $L: int, x0, w, active_h: int, col_kind: Tx_1D, col_clamp: uint) #no_bounds_check {
	H0 :: 1 << log2_h
	V :: #simd[L]E
	WL :: #simd[L]i32
	t: [H0]V = ---
	#unroll for b in 0 ..< MAX_IN / 4 {
		ib := b * 4
		if MAX_IN <= 8 || ib < active_h {
			#unroll for k in 0 ..< 4 {
				t[ib + k] = V(intrinsics.unaligned_load((^WL)(&block[(ib + k) * w + x0])))
			}
		} else {
			#unroll for k in 0 ..< 4 {
				t[ib + k] = 0
			}
		}
	}
	when H0 > MAX_IN {
		#unroll for i in MAX_IN ..< H0 do t[i] = 0
	}
	when E == i32 {
		itx_1d_impl(&t, log2_h, col_kind, 16, MAX_IN, true)
	} else {
		itx_1d_impl(&t, log2_h, col_kind, col_clamp, MAX_IN, false)
	}
	#unroll for i in 0 ..< H0 {
		out := round2_lanes(t[i], 4)
		intrinsics.unaligned_store((^WL)(&block[i * w + x0]), WL(out))
	}
}

@(private)
itx_row_dispatch :: #force_inline proc "contextless" (dst, src: []i32, $E: typeid, $L: int, log2_w, y0, w, active_w: int, rect_scale: bool, row_kind: Tx_1D, row_shift, row_clamp, col_clamp: uint) #no_bounds_check {
	if active_w == 4 {
		switch log2_w {
		case 2: itx_row_group(dst, src, 2, 4, E, L, y0, w, 4, rect_scale, row_kind, row_shift, row_clamp, col_clamp)
		case 3: itx_row_group(dst, src, 3, 4, E, L, y0, w, 4, rect_scale, row_kind, row_shift, row_clamp, col_clamp)
		case 4: itx_row_group(dst, src, 4, 4, E, L, y0, w, 4, rect_scale, row_kind, row_shift, row_clamp, col_clamp)
		case 5: itx_row_group(dst, src, 5, 4, E, L, y0, w, 4, rect_scale, row_kind, row_shift, row_clamp, col_clamp)
		case 6: itx_row_group(dst, src, 6, 4, E, L, y0, w, 4, rect_scale, row_kind, row_shift, row_clamp, col_clamp)
		}
	} else {
		switch log2_w {
		case 3: itx_row_group(dst, src, 3, 8, E, L, y0, w, 8, rect_scale, row_kind, row_shift, row_clamp, col_clamp)
		case 4: itx_row_group(dst, src, 4, 16, E, L, y0, w, active_w, rect_scale, row_kind, row_shift, row_clamp, col_clamp)
		case 5: itx_row_group(dst, src, 5, 32, E, L, y0, w, active_w, rect_scale, row_kind, row_shift, row_clamp, col_clamp)
		case 6: itx_row_group(dst, src, 6, 32, E, L, y0, w, active_w, rect_scale, row_kind, row_shift, row_clamp, col_clamp)
		}
	}
}

@(private)
itx_col_dispatch :: #force_inline proc "contextless" (block: []i32, $E: typeid, $L: int, log2_h, w, recon_w, active_h: int, col_kind: Tx_1D, col_clamp: uint) #no_bounds_check {
	if active_h == 4 {
		for x0 := 0; x0 < recon_w; x0 += L {
			switch log2_h {
			case 2: itx_col_group(block, 2, 4, E, L, x0, w, 4, col_kind, col_clamp)
			case 3: itx_col_group(block, 3, 4, E, L, x0, w, 4, col_kind, col_clamp)
			case 4: itx_col_group(block, 4, 4, E, L, x0, w, 4, col_kind, col_clamp)
			case 5: itx_col_group(block, 5, 4, E, L, x0, w, 4, col_kind, col_clamp)
			case 6: itx_col_group(block, 6, 4, E, L, x0, w, 4, col_kind, col_clamp)
			}
		}
	} else {
		for x0 := 0; x0 < recon_w; x0 += L {
			switch log2_h {
			case 3: itx_col_group(block, 3, 8, E, L, x0, w, 8, col_kind, col_clamp)
			case 4: itx_col_group(block, 4, 16, E, L, x0, w, active_h, col_kind, col_clamp)
			case 5: itx_col_group(block, 5, 32, E, L, x0, w, active_h, col_kind, col_clamp)
			case 6: itx_col_group(block, 6, 32, E, L, x0, w, active_h, col_kind, col_clamp)
			}
		}
	}
}

@(private)
itx_2d_lanes :: proc "contextless" (dst, src: []i32, clear_src: bool, tx_size, tx_type, bit_depth, max_r, max_c: int, $E: typeid) #no_bounds_check {
	log2_w := int(TX_WIDTH_LOG2[tx_size])
	log2_h := int(TX_HEIGHT_LOG2[tx_size])
	w := 1 << uint(log2_w)
	h := 1 << uint(log2_h)
	active_w := min(min(w, 32), (max_c + 4) & ~int(3))
	active_h := min(min(h, 32), (max_r + 4) & ~int(3))
	rect_scale := abs(log2_w - log2_h) == 1
	row_shift := uint(TRANSFORM_ROW_SHIFT[tx_size])
	row_clamp := uint(bit_depth + 8)
	col_clamp := uint(16) when E == i32 else uint(max(bit_depth + 6, 16))
	kinds := TX_1D_TYPES[tx_type]
	row_kind, col_kind := kinds[0], kinds[1]

	y0 := 0
	when ITX_MAX_LANES >= 16 && E == i32 {
		for ; y0 + 16 <= active_h; y0 += 16 {
			itx_row_dispatch(dst, src, E, 16, log2_w, y0, w, active_w, rect_scale, row_kind, row_shift, row_clamp, col_clamp)
		}
	}
	when ITX_MAX_LANES >= 8 && E == i32 {
		for ; y0 + 8 <= active_h; y0 += 8 {
			itx_row_dispatch(dst, src, E, 8, log2_w, y0, w, active_w, rect_scale, row_kind, row_shift, row_clamp, col_clamp)
		}
	}
	for ; y0 < active_h; y0 += 4 {
		itx_row_dispatch(dst, src, E, 4, log2_w, y0, w, active_w, rect_scale, row_kind, row_shift, row_clamp, col_clamp)
	}
	recon_w := (clear_src && row_kind == .IDTX) ? (active_w <= 8 ? active_w : (active_w + 15) & ~int(15)) : w
	if clear_src {
		V4 :: #simd[4]E
		W4 :: #simd[4]i32
		if max_r == 0 && col_kind == .DCT {
			for x0 := 0; x0 < recon_w; x0 += 4 {
				p := (^W4)(&dst[x0])
				v := V4(intrinsics.unaligned_load(p))
				v = round2_lanes(clamp_lanes(round2_lanes(v * V4(2896), 12), col_clamp, E == i32), 4)
				intrinsics.unaligned_store(p, W4(v))
			}
			return
		}
		if col_kind == .IDTX {
			rows := min(max_r + 1, h)
			switch log2_h {
			case 2:
				for i in 0 ..< rows {
					for x0 := 0; x0 < recon_w; x0 += 4 {
						p := (^W4)(&dst[i * w + x0])
						v := V4(intrinsics.unaligned_load(p))
						v = round2_lanes(round2_lanes(v * V4(5793), 12), 4)
						intrinsics.unaligned_store(p, W4(v))
					}
				}
			case 3:
				for i in 0 ..< rows {
					for x0 := 0; x0 < recon_w; x0 += 4 {
						p := (^W4)(&dst[i * w + x0])
						v := V4(intrinsics.unaligned_load(p))
						v = round2_lanes(v * V4(2), 4)
						intrinsics.unaligned_store(p, W4(v))
					}
				}
			case 4:
				for i in 0 ..< rows {
					for x0 := 0; x0 < recon_w; x0 += 4 {
						p := (^W4)(&dst[i * w + x0])
						v := V4(intrinsics.unaligned_load(p))
						v = round2_lanes(round2_lanes(v * V4(11586), 12), 4)
						intrinsics.unaligned_store(p, W4(v))
					}
				}
			case 5:
				for i in 0 ..< rows {
					for x0 := 0; x0 < recon_w; x0 += 4 {
						p := (^W4)(&dst[i * w + x0])
						v := V4(intrinsics.unaligned_load(p))
						v = round2_lanes(v * V4(4), 4)
						intrinsics.unaligned_store(p, W4(v))
					}
				}
			}
			return
		}
	}
	when ITX_MAX_LANES >= 16 && E == i32 {
		if recon_w >= 16 {
			itx_col_dispatch(dst, E, 16, log2_h, w, recon_w, active_h, col_kind, col_clamp)
			return
		}
	}
	when ITX_MAX_LANES >= 8 && E == i32 {
		if recon_w >= 8 {
			itx_col_dispatch(dst, E, 8, log2_h, w, recon_w, active_h, col_kind, col_clamp)
			return
		}
	}
	itx_col_dispatch(dst, E, 4, log2_h, w, recon_w, active_h, col_kind, col_clamp)
}

// ---- Quantizer Lookup & Dequantization Helpers (7.12.2, 7.12.3) -------------

// 7.12.2: DC and AC quantizer step sizes for (qindex, delta_q, bit_depth).
dc_quant :: #force_inline proc "contextless" (qindex: int, delta_q: i8, bit_depth := 8) -> i32 #no_bounds_check {
	bd_idx := (bit_depth - 8) >> 1
	return i32(DC_QLOOKUP[bd_idx][clamp(qindex + int(delta_q), 0, 255)])
}

ac_quant :: #force_inline proc "contextless" (qindex: int, delta_q: i8, bit_depth := 8) -> i32 #no_bounds_check {
	bd_idx := (bit_depth - 8) >> 1
	return i32(AC_QLOOKUP[bd_idx][clamp(qindex + int(delta_q), 0, 255)])
}

// 7.12.3: dqDenom shift (0 for 1, 1 for 2, 2 for 4) for tx_size.
@(rodata)
TX_DQ_SHIFT := [TX_SIZES_ALL]u8{
	TX_4X4   = 0,
	TX_8X8   = 0,
	TX_16X16 = 0,
	TX_32X32 = 1,
	TX_64X64 = 2,
	TX_4X8   = 0,
	TX_8X4   = 0,
	TX_8X16  = 0,
	TX_16X8  = 0,
	TX_16X32 = 1,
	TX_32X16 = 1,
	TX_32X64 = 2,
	TX_64X32 = 2,
	TX_4X16  = 0,
	TX_16X4  = 0,
	TX_8X32  = 0,
	TX_32X8  = 0,
	TX_16X64 = 1,
	TX_64X16 = 1,
}

// 7.12.3: dequantizes a quantized coefficient at position `pos` (in the min(w,32) x min(h,32)
// coefficient grid) into the `Dequant[i][j]` value at raster index `(pos / tw) * w + (pos % tw)`.
dequant_coeff :: #force_inline proc "contextless" (quant: i32, pos, tx_size, tx_type, plane: int, dc_q, ac_q: i32, qm_level: u8, bit_depth := 8) -> i32 #no_bounds_check {
	q := pos == 0 ? dc_q : ac_q
	if qm_level < 15 && tx_type < IDTX {
		qm := i32(QUANTIZER_MATRIX[qm_level][int(plane > 0)][int(QM_OFFSET[tx_size]) + pos])
		q = round2(i64(q * qm), 5)
	}
	dq := i64(quant) * i64(q)
	sign := i32(dq < 0 ? -1 : 1)
	abs_dq := i32(abs(dq) & 0xFFFFFF) >> uint(TX_DQ_SHIFT[tx_size])
	dq2 := sign * abs_dq
	limit := i32(1) << uint(7 + bit_depth)
	return clamp(dq2, -limit, limit - 1)
}

// ---- Transform Size, Set, Class & Scan Helpers (5.11.36 - 5.11.48, 8.3.2) ---

// 5.11.36: finds the transform size matching dimensions (w, h).
find_tx_size :: #force_inline proc "contextless" (w, h: int) -> int #no_bounds_check {
	for tx_sz in 0 ..< TX_SIZES_ALL {
		if int(TX_WIDTH[tx_sz]) == w && int(TX_HEIGHT[tx_sz]) == h {
			return tx_sz
		}
	}
	return TX_4X4
}

// 5.11.37: returns the transform size for `plane` (where `tx_sz` is the luma transform size).
get_uv_tx_size :: #force_inline proc "contextless" (mi_size, plane, tx_sz: int, lossless: bool, sub_x, sub_y: u8) -> int #no_bounds_check {
	if plane == 0 do return tx_sz
	if lossless do return TX_4X4
	res_size := get_plane_residual_size(mi_size, plane, uint(sub_x), uint(sub_y))
	uv_tx := int(MAX_TX_SIZE_RECT[res_size])
	if TX_WIDTH[uv_tx] == 64 || TX_HEIGHT[uv_tx] == 64 {
		if TX_WIDTH[uv_tx] == 16 do return TX_16X32
		if TX_HEIGHT[uv_tx] == 16 do return TX_32X16
		return TX_32X32
	}
	return uv_tx
}

// 5.11.48: returns the transform set for `tx_sz`.
get_tx_set :: #force_inline proc "contextless" (tx_sz: int, is_inter, reduced_tx_set: bool) -> int #no_bounds_check {
	tx_sz_sqr := int(TX_SIZE_SQR[tx_sz])
	tx_sz_sqr_up := int(TX_SIZE_SQR_UP[tx_sz])
	if tx_sz_sqr_up > TX_32X32 do return TX_SET_DCTONLY
	if is_inter {
		if reduced_tx_set || tx_sz_sqr_up == TX_32X32 do return TX_SET_INTER_3
		if tx_sz_sqr == TX_16X16 do return TX_SET_INTER_2
		return TX_SET_INTER_1
	} else {
		if tx_sz_sqr_up == TX_32X32 do return TX_SET_DCTONLY
		if reduced_tx_set || tx_sz_sqr == TX_16X16 do return TX_SET_INTRA_2
		return TX_SET_INTRA_1
	}
}

// 8.3.2: returns the transform class (TX_CLASS_2D, TX_CLASS_HORIZ, TX_CLASS_VERT).
get_tx_class :: #force_inline proc "contextless" (tx_type: int) -> int {
	switch tx_type {
	case V_DCT, V_ADST, V_FLIPADST:
		return TX_CLASS_VERT
	case H_DCT, H_ADST, H_FLIPADST:
		return TX_CLASS_HORIZ
	case:
		return TX_CLASS_2D
	}
}

@(private)
get_mrow_scan :: #force_inline proc "contextless" (tx_sz: int) -> []u16 {
	switch tx_sz {
	case TX_4X4:   return MROW_SCAN_4X4[:]
	case TX_4X8:   return MROW_SCAN_4X8[:]
	case TX_8X4:   return MROW_SCAN_8X4[:]
	case TX_8X8:   return MROW_SCAN_8X8[:]
	case TX_8X16:  return MROW_SCAN_8X16[:]
	case TX_16X8:  return MROW_SCAN_16X8[:]
	case TX_16X16: return MROW_SCAN_16X16[:]
	case TX_4X16:  return MROW_SCAN_4X16[:]
	case:          return MROW_SCAN_16X4[:]
	}
}

@(private)
get_mcol_scan :: #force_inline proc "contextless" (tx_sz: int) -> []u16 {
	switch tx_sz {
	case TX_4X4:   return MCOL_SCAN_4X4[:]
	case TX_4X8:   return MCOL_SCAN_4X8[:]
	case TX_8X4:   return MCOL_SCAN_8X4[:]
	case TX_8X8:   return MCOL_SCAN_8X8[:]
	case TX_8X16:  return MCOL_SCAN_8X16[:]
	case TX_16X8:  return MCOL_SCAN_16X8[:]
	case TX_16X16: return MCOL_SCAN_16X16[:]
	case TX_4X16:  return MCOL_SCAN_4X16[:]
	case:          return MCOL_SCAN_16X4[:]
	}
}

@(private)
get_default_scan :: #force_inline proc "contextless" (tx_sz: int) -> []u16 {
	switch tx_sz {
	case TX_4X4:   return DEFAULT_SCAN_4X4[:]
	case TX_4X8:   return DEFAULT_SCAN_4X8[:]
	case TX_8X4:   return DEFAULT_SCAN_8X4[:]
	case TX_8X8:   return DEFAULT_SCAN_8X8[:]
	case TX_8X16:  return DEFAULT_SCAN_8X16[:]
	case TX_16X8:  return DEFAULT_SCAN_16X8[:]
	case TX_16X16: return DEFAULT_SCAN_16X16[:]
	case TX_16X32: return DEFAULT_SCAN_16X32[:]
	case TX_32X16: return DEFAULT_SCAN_32X16[:]
	case TX_4X16:  return DEFAULT_SCAN_4X16[:]
	case TX_16X4:  return DEFAULT_SCAN_16X4[:]
	case TX_8X32:  return DEFAULT_SCAN_8X32[:]
	case TX_32X8:  return DEFAULT_SCAN_32X8[:]
	case:          return DEFAULT_SCAN_32X32[:]
	}
}

// 5.11.41: returns the scan order table for `(tx_sz, plane_tx_type)`.
get_scan :: proc "contextless" (tx_sz, plane_tx_type: int) -> []u16 #no_bounds_check {
	if tx_sz == TX_16X64 do return DEFAULT_SCAN_16X32[:]
	if tx_sz == TX_64X16 do return DEFAULT_SCAN_32X16[:]
	if TX_SIZE_SQR_UP[tx_sz] == TX_64X64 do return DEFAULT_SCAN_32X32[:]
	if plane_tx_type == IDTX do return get_default_scan(tx_sz)
	switch plane_tx_type {
	case V_DCT, V_ADST, V_FLIPADST:
		return get_mrow_scan(tx_sz)
	case H_DCT, H_ADST, H_FLIPADST:
		return get_mcol_scan(tx_sz)
	case:
		return get_default_scan(tx_sz)
	}
}
