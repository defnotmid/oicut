package av1

// Separable fixed-point butterfly analysis transforms (exact transposes of the
// normative synthesis transforms in transform.odin) and quantization helpers.

import common "../../common"
import "base:intrinsics"

// Transpose of B(t, a, b, angle, flip): non-flipped butterfly negates angle;
// flipped butterfly is symmetric.
@(private)
BT :: #force_inline proc "contextless" (t: ^[$M]$V, a, b, angle: int, flip: bool) {
	B(t, a, b, flip ? angle : -angle, flip, false)
}

// Transpose of H(t, a, b, flip, r) without intermediate clamping.
@(private)
HT :: #force_inline proc "contextless" (t: ^[$M]$V, a, b: int, flip: bool) {
	a, b := a, b
	if flip do a, b = b, a
	x, y := t[a], t[b]
	t[a] = x + y
	t[b] = x - y
}

// 1-D Forward DCT of length 1 << n (n in 2..6): exact transpose of idct_1d.
@(private)
fdct_1d :: #force_inline proc "contextless" (t: ^[$M]$V, $n: uint) {
	when n == 6 {
		#unroll for i in 0 ..< 32 do HT(t, i, 63 - i, false)
		#unroll for i in 0 ..< 8 do BT(t, 55 - i, 40 + i, 32, true)
	}
	when n >= 5 {
		#unroll for i in 0 ..< 16 do HT(t, i, 31 - i, false)
	}
	when n == 6 {
		#unroll for i in 0 ..< 8 {
			HT(t, 32 + i, 47 - i, false)
			HT(t, 48 + i, 63 - i, true)
		}
	}
	when n >= 5 {
		#unroll for i in 0 ..< 4 do BT(t, 27 - i, 20 + i, 32, true)
	}
	when n >= 4 {
		#unroll for i in 0 ..< 8 do HT(t, i, 15 - i, false)
	}
	when n == 6 {
		#unroll for i in 0 ..< 8 do BT(t, 59 - i, 36 + i, i < 4 ? 48 : 112, true)
	}
	when n >= 5 {
		#unroll for i in 0 ..< 2 do #unroll for j in 0 ..< 4 {
			HT(t, 16 + i * 8 + j, 23 + i * 8 - j, i != 0)
		}
	}
	when n >= 4 {
		#unroll for i in 0 ..< 2 do BT(t, 13 - i, 10 + i, 32, true)
	}
	when n >= 3 {
		#unroll for i in 0 ..< 4 do HT(t, i, 7 - i, false)
	}
	when n == 6 {
		#unroll for i in 0 ..< 4 do #unroll for j in 0 ..< 4 {
			HT(t, 32 + 8 * i + j, 39 + 8 * i - j, (i & 1) != 0)
		}
	}
	when n >= 5 {
		#unroll for i in 0 ..< 4 do BT(t, 29 - i, 18 + i, 48 + (i >> 1) * 64, true)
	}
	when n >= 4 {
		#unroll for i in 0 ..< 2 do #unroll for j in 0 ..< 2 {
			HT(t, 8 + 4 * i + j, 11 + 4 * i - j, i != 0)
		}
	}
	when n >= 3 {
		BT(t, 6, 5, 32, true)
	}
	#unroll for i in 0 ..< 2 do HT(t, i, 3 - i, false)
	when n == 6 {
		#unroll for i in 0 ..< 2 do #unroll for j in 0 ..< 4 {
			BT(t, 61 - i * 8 - j, 34 + i * 8 + j, 56 - i * 32 + (j >> 1) * 64, true)
		}
	}
	when n >= 5 {
		#unroll for i in 0 ..< 4 do #unroll for j in 0 ..< 2 {
			HT(t, 16 + 4 * i + j, 19 + 4 * i - j, (i & 1) != 0)
		}
	}
	when n >= 4 {
		#unroll for i in 0 ..< 2 do BT(t, 14 - i, 9 + i, 48 + 64 * i, true)
	}
	when n >= 3 {
		#unroll for i in 0 ..< 2 do HT(t, 4 + 2 * i, 5 + 2 * i, i != 0)
	}
	#unroll for i in 0 ..< 2 do BT(t, 2 * i, 2 * i + 1, 32 + 16 * i, i == 0)
	when n == 6 {
		#unroll for i in 0 ..< 8 do #unroll for j in 0 ..< 2 {
			HT(t, 32 + i * 4 + j, 35 + i * 4 - j, (i & 1) != 0)
		}
	}
	when n >= 5 {
		#unroll for i in 0 ..< 2 do #unroll for j in 0 ..< 2 {
			BT(t, 30 - 4 * i - j, 17 + 4 * i + j, 24 + (j << 6) + ((1 - i) << 5), true)
		}
	}
	when n >= 4 {
		#unroll for i in 0 ..< 4 do HT(t, 8 + 2 * i, 9 + 2 * i, (i & 1) != 0)
	}
	when n >= 3 {
		#unroll for i in 0 ..< 2 do BT(t, 4 + i, 7 - i, 56 - 32 * i, false)
	}
	when n == 6 {
		#unroll for i in 0 ..< 4 do #unroll for j in 0 ..< 2 {
			BT(t, 62 - i * 4 - j, 33 + i * 4 + j, 60 - 16 * brev(2, i) + 64 * j, true)
		}
	}
	when n >= 5 {
		#unroll for i in 0 ..< 8 do HT(t, 16 + 2 * i, 17 + 2 * i, (i & 1) != 0)
	}
	when n >= 4 {
		#unroll for i in 0 ..< 4 do BT(t, 8 + i, 15 - i, 12 + (brev(2, 3 - i) << 4), false)
	}
	when n == 6 {
		#unroll for i in 0 ..< 16 do HT(t, 32 + i * 2, 33 + i * 2, (i & 1) != 0)
	}
	when n >= 5 {
		#unroll for i in 0 ..< 8 do BT(t, 16 + i, 31 - i, 6 + (brev(3, 7 - i) << 3), false)
	}
	when n == 6 {
		#unroll for i in 0 ..< 16 do BT(t, 32 + i, 63 - i, 63 - 4 * brev(4, i), false)
	}
	c := t^
	#unroll for i in 0 ..< (1 << n) do t[i] = c[brev(n, i)]
}

// Transpose of adst_input_permute.
@(private)
adst_input_permute_t :: #force_inline proc "contextless" (t: ^[$M]$V, $n: uint) {
	N0 :: 1 << n
	c := t^
	#unroll for i in 0 ..< N0 {
		idx := (i & 1) != 0 ? (i - 1) : (N0 - i - 1)
		t[idx] = c[i]
	}
}

// Transpose of adst_output_permute.
@(private)
adst_output_permute_t :: #force_inline proc "contextless" (t: ^[$M]$V, $n: uint) {
	N0 :: 1 << n
	c := t^
	#unroll for i in 0 ..< N0 {
		a := (i >> 3) & 1
		b := ((i >> 2) & 1) ~ ((i >> 3) & 1)
		cc := ((i >> 1) & 1) ~ ((i >> 2) & 1)
		d := (i & 1) ~ ((i >> 1) & 1)
		idx := ((d << 3) | (cc << 2) | (b << 1) | a) >> (4 - n)
		t[idx] = (i & 1) != 0 ? -c[i] : c[i]
	}
}

// Forward ADST4: exact transpose of iadst4.
@(private)
fadst4 :: #force_inline proc "contextless" (t: ^[$M]$V) {
	s0 := V(SINPI_1_9) * t[0]
	s1 := V(SINPI_4_9) * t[0]
	s2 := V(SINPI_2_9) * t[1]
	s3 := V(SINPI_1_9) * t[1]
	s4 := V(SINPI_3_9) * t[2]
	s5 := V(SINPI_4_9) * t[3]
	s6 := V(SINPI_2_9) * t[3]
	s7 := V(SINPI_3_9) * (t[0] + t[1] - t[3])
	x0 := s0 + s2 + s5
	x1 := s7
	x2 := s1 - s3 + s6
	x3 := s4
	t[0] = round2_lanes(x0 + x3, 12)
	t[1] = round2_lanes(x1, 12)
	t[2] = round2_lanes(x2 - x3, 12)
	t[3] = round2_lanes(x2 - x0 + x3, 12)
}

// Forward ADST8: exact transpose of iadst8.
@(private)
fadst8 :: #force_inline proc "contextless" (t: ^[$M]$V) {
	adst_output_permute_t(t, 3)
	#unroll for i in 0 ..< 2 do BT(t, 2 + 4 * i, 3 + 4 * i, 32, true)
	#unroll for j in 0 ..< 2 do #unroll for i in 0 ..< 2 do HT(t, 4 * j + i, 2 + 4 * j + i, false)
	#unroll for i in 0 ..< 2 do BT(t, 4 + 3 * i, 5 + i, 48 - 32 * i, true)
	#unroll for i in 0 ..< 4 do HT(t, i, 4 + i, false)
	#unroll for i in 0 ..< 4 do BT(t, 2 * i, 2 * i + 1, 60 - 16 * i, true)
	adst_input_permute_t(t, 3)
}

// Forward ADST16: exact transpose of iadst16.
@(private)
fadst16 :: #force_inline proc "contextless" (t: ^[$M]$V) {
	adst_output_permute_t(t, 4)
	#unroll for i in 0 ..< 4 do BT(t, 2 + 4 * i, 3 + 4 * i, 32, true)
	#unroll for j in 0 ..< 4 do #unroll for i in 0 ..< 2 do HT(t, 4 * j + i, 2 + 4 * j + i, false)
	#unroll for j in 0 ..< 2 do #unroll for i in 0 ..< 2 do BT(t, 4 + 8 * j + 3 * i, 5 + 8 * j + i, 48 - 32 * i, true)
	#unroll for j in 0 ..< 2 do #unroll for i in 0 ..< 4 do HT(t, 8 * j + i, 4 + 8 * j + i, false)
	#unroll for i in 0 ..< 2 {
		BT(t, 8 + 2 * i, 9 + 2 * i, 56 - 32 * i, true)
		BT(t, 13 + 2 * i, 12 + 2 * i, 8 + 32 * i, true)
	}
	#unroll for i in 0 ..< 8 do HT(t, i, 8 + i, false)
	#unroll for i in 0 ..< 8 do BT(t, 2 * i, 2 * i + 1, 62 - 8 * i, true)
	adst_input_permute_t(t, 4)
}

// Dispatches a 1-D forward transform of length 1 << n on lanes.
@(private)
ftx_1d :: #force_inline proc "contextless" (t: ^[$M]$V, $n: uint, kind: Tx_1D) {
	N0 :: 1 << n
	when n >= 5 {
		when n == 5 {
			if kind == .IDTX {
				iidtx_1d(t, 5)
				return
			}
		}
		fdct_1d(t, n)
	} else {
		switch kind {
		case .DCT:
			fdct_1d(t, n)
		case .ADST:
			when n == 2 {
				fadst4(t)
			} else when n == 3 {
				fadst8(t)
			} else {
				fadst16(t)
			}
		case .FLIPADST:
			#unroll for i in 0 ..< (N0 >> 1) {
				t[i], t[N0 - 1 - i] = t[N0 - 1 - i], t[i]
			}
			when n == 2 {
				fadst4(t)
			} else when n == 3 {
				fadst8(t)
			} else {
				fadst16(t)
			}
		case .IDTX:
			iidtx_1d(t, n)
		}
	}
}

// 1-D Forward Walsh-Hadamard transform (length 4): exact inverse of iwht4.
@(private)
fwht4 :: #force_inline proc "contextless" (v: ^[4]i32) {
	a, d := v[0] + v[1], v[3] - v[2]
	e := (a - d) >> 1
	b, c := e - v[1], e - v[2]
	v^ = {a - c, c, d + b, b}
}

// 2-D lossless 4x4 forward WHT in place, producing coefficients scaled by 4
// (matching DC_QLOOKUP/AC_QLOOKUP step size 4 at qindex=0).
forward_wht4x4 :: proc "contextless" (block: []i32) #no_bounds_check {
	for j in 0 ..< 4 {
		col := [4]i32{block[j], block[4 + j], block[8 + j], block[12 + j]}
		fwht4(&col)
		for i in 0 ..< 4 do block[i * 4 + j] = col[i]
	}
	for i in 0 ..< 4 {
		row := [4]i32{block[i * 4], block[i * 4 + 1], block[i * 4 + 2], block[i * 4 + 3]}
		fwht4(&row)
		for j in 0 ..< 4 do block[i * 4 + j] = row[j] * 4
	}
}

// 2-D forward transform of `block` (w * h raster order) in place.
// Outputs coefficients at the `Dequant` scale consumed directly by `inverse_transform_2d`
// (and zeroed outside the top-left min(w, 32) x min(h, 32) region).
forward_transform_2d :: proc(block: []i32, tx_size, tx_type: int, lossless: bool, bit_depth := 8) #no_bounds_check {
	if lossless {
		forward_wht4x4(block)
		return
	}
	if bit_depth > 8 {
		ftx_2d_lanes(block, tx_size, tx_type, i64)
	} else {
		ftx_2d_lanes(block, tx_size, tx_type, i32)
	}
}

@(private)
ftx_col_group :: #force_inline proc "contextless" (block: []i32, $log2_h: uint, $E: typeid, x0, w, th: int, col_kind: Tx_1D, row_shift: uint) #no_bounds_check {
	H0 :: 1 << log2_h
	V :: #simd[4]E
	W4 :: #simd[4]i32
	t: [H0]V = ---
	for i in 0 ..< H0 {
		t[i] = V(intrinsics.unaligned_load((^W4)(&block[i * w + x0]))) * V(8)
	}
	ftx_1d(&t, log2_h, col_kind)
	for i in 0 ..< th {
		out := round2_lanes(t[i], row_shift)
		intrinsics.unaligned_store((^W4)(&block[i * w + x0]), W4(out))
	}
	for i in th ..< H0 {
		intrinsics.unaligned_store((^W4)(&block[i * w + x0]), W4(0))
	}
}

@(private)
ftx_row_group :: #force_inline proc "contextless" (block: []i32, $log2_w: uint, $E: typeid, y0, w: int, rect_scale: bool, row_kind: Tx_1D, out_shift: uint) #no_bounds_check {
	W0 :: 1 << log2_w
	V :: #simd[4]E
	W4 :: #simd[4]i32
	t: [W0]V = ---
	for jb := 0; jb < W0; jb += 4 {
		tile: [4]V = ---
		#unroll for l in 0 ..< 4 {
			tile[l] = V(intrinsics.unaligned_load((^W4)(&block[(y0 + l) * w + jb])))
		}
		tile = common.transpose(tile)
		#unroll for k in 0 ..< 4 do t[jb + k] = tile[k]
	}
	ftx_1d(&t, log2_w, row_kind)
	TW :: min(W0, 32)
	for jb := 0; jb < TW; jb += 4 {
		tile: [4]V = ---
		#unroll for k in 0 ..< 4 {
			v := t[jb + k]
			if rect_scale do v = round2_lanes(v * V(2896), 12)
			tile[k] = round2_lanes(v, out_shift)
		}
		tile = common.transpose(tile)
		#unroll for l in 0 ..< 4 {
			intrinsics.unaligned_store((^W4)(&block[(y0 + l) * w + jb]), W4(tile[l]))
		}
	}
	when W0 > 32 {
		for jb := 32; jb < W0; jb += 4 {
			#unroll for l in 0 ..< 4 {
				intrinsics.unaligned_store((^W4)(&block[(y0 + l) * w + jb]), W4(0))
			}
		}
	}
}

@(private)
ftx_2d_lanes :: proc "contextless" (block: []i32, tx_size, tx_type: int, $E: typeid) #no_bounds_check {
	log2_w := int(TX_WIDTH_LOG2[tx_size])
	log2_h := int(TX_HEIGHT_LOG2[tx_size])
	w := 1 << uint(log2_w)
	h := 1 << uint(log2_h)
	th := min(h, 32)
	rect_scale := abs(log2_w - log2_h) == 1
	row_shift := uint(TRANSFORM_ROW_SHIFT[tx_size])
	out_shift := uint(1 + 2 * TX_DQ_SHIFT[tx_size])
	kinds := TX_1D_TYPES[tx_type]
	row_kind, col_kind := kinds[0], kinds[1]

	for x0 := 0; x0 < w; x0 += 4 {
		switch log2_h {
		case 2: ftx_col_group(block, 2, E, x0, w, th, col_kind, row_shift)
		case 3: ftx_col_group(block, 3, E, x0, w, th, col_kind, row_shift)
		case 4: ftx_col_group(block, 4, E, x0, w, th, col_kind, row_shift)
		case 5: ftx_col_group(block, 5, E, x0, w, th, col_kind, row_shift)
		case 6: ftx_col_group(block, 6, E, x0, w, th, col_kind, row_shift)
		}
	}
	for y0 := 0; y0 < th; y0 += 4 {
		switch log2_w {
		case 2: ftx_row_group(block, 2, E, y0, w, rect_scale, row_kind, out_shift)
		case 3: ftx_row_group(block, 3, E, y0, w, rect_scale, row_kind, out_shift)
		case 4: ftx_row_group(block, 4, E, y0, w, rect_scale, row_kind, out_shift)
		case 5: ftx_row_group(block, 5, E, y0, w, rect_scale, row_kind, out_shift)
		case 6: ftx_row_group(block, 6, E, y0, w, rect_scale, row_kind, out_shift)
		}
	}
}

// Quantizes a forward-transformed coefficient `coeff` (in the `Dequant` domain)
// at `pos` (in the min(w,32) x min(h,32) grid) into a quantized level `quant`
// and its reconstructed `dequant` value.
quantize_coeff :: #force_inline proc "contextless" (coeff: i32, pos, tx_size, tx_type, plane: int, dc_q, ac_q: i32, qm_level: u8, bit_depth := 8, bias: i32 = -1) -> (quant, dequant: i32) #no_bounds_check {
	if coeff == 0 do return 0, 0
	q := pos == 0 ? dc_q : ac_q
	if qm_level < 15 && tx_type < IDTX {
		qm := i32(QUANTIZER_MATRIX[qm_level][int(plane > 0)][int(QM_OFFSET[tx_size]) + pos])
		q = round2(i64(q * qm), 5)
	}
	if q <= 0 do return 0, 0
	scaled := i64(abs(coeff)) << uint(TX_DQ_SHIFT[tx_size])
	b := bias >= 0 ? i64(bias) : i64(q >> 1)
	level := i32((scaled + b) / i64(q))
	if level == 0 do return 0, 0
	level = min(level, (1 << 18) - 1)
	q_signed := coeff < 0 ? -level : level
	return q_signed, dequant_coeff(q_signed, pos, tx_size, tx_type, plane, dc_q, ac_q, qm_level, bit_depth)
}
