package av1

// Normative inter prediction (section 7.11.3): motion vector scaling, 8-tap /
// 4-tap / bilinear subpel convolution (with SIMD fast paths for unscaled
// motion), local/global affine warped motion, OBMC blending, and compound
// averaging / distance-weighted / wedge / diff-weighted / inter-intra blending.

import "base:intrinsics"
@(require) import "base:runtime"
import "core:simd"

Ref_Plane :: struct($T: typeid) {
	data:                   []T,
	stride:                 int,
	last_x, last_y:         int, // bottom-right valid sample in this plane
	ref_width, ref_height:  int, // luma reference dimensions
	upscaled_width:         int, // luma upscaled width
}

// 7.11.3.2: Derives (inter_round0, inter_round1, inter_post_round).
inter_rounding_vars :: #force_inline proc "contextless" (is_compound: bool, bit_depth := 8) -> (r0, r1, post: uint) {
	r0 = bit_depth == 12 ? 5 : 3
	r1 = is_compound ? 7 : (bit_depth == 12 ? 9 : 11)
	post = uint(2 * FILTER_BITS) - (r0 + r1)
	return
}

// 7.11.3.3: Scales a luma motion vector `mv` = {row, col} (in 1/8-luma-pel units)
// for block top-left `(x, y)` in `plane` to 1/1024-pel `(start_x, start_y, step_x, step_y)`.
scale_mv_for_block :: proc "contextless" (
	x, y: int,
	mv: [2]i16,
	sub_x, sub_y: uint,
	frame_w, frame_h: int,
	ref_upscaled_w, ref_h: int,
) -> (start_x, start_y, step_x, step_y: int) {
	x_scale := ((ref_upscaled_w << REF_SCALE_SHIFT) + (frame_w / 2)) / frame_w
	y_scale := ((ref_h << REF_SCALE_SHIFT) + (frame_h / 2)) / frame_h
	half_sample :: 1 << (SUBPEL_BITS - 1)
	orig_x := (x << SUBPEL_BITS) + ((2 * int(mv[1])) >> sub_x) + half_sample
	orig_y := (y << SUBPEL_BITS) + ((2 * int(mv[0])) >> sub_y) + half_sample
	base_x := i64(orig_x) * i64(x_scale) - i64(half_sample << REF_SCALE_SHIFT)
	base_y := i64(orig_y) * i64(y_scale) - i64(half_sample << REF_SCALE_SHIFT)
	off :: (1 << (SCALE_SUBPEL_BITS - SUBPEL_BITS)) / 2
	shift_pos :: uint(REF_SCALE_SHIFT + SUBPEL_BITS - SCALE_SUBPEL_BITS)
	shift_step :: uint(REF_SCALE_SHIFT - SCALE_SUBPEL_BITS)
	start_x = int(round2_signed(base_x, shift_pos)) + off
	start_y = int(round2_signed(base_y, shift_pos)) + off
	step_x = int(round2_signed(i64(x_scale), shift_step))
	step_y = int(round2_signed(i64(y_scale), shift_step))
	return
}

// Maps an 8-tap filter index to its 4-tap small-block variant when dim <= 4 (7.11.3.4).
@(private)
effective_interp_filter :: #force_inline proc "contextless" (f, dim: int) -> int {
	if dim <= 4 {
		if f == EIGHTTAP_REGULAR || f == EIGHTTAP_SHARP do return 4
		if f == EIGHTTAP_SMOOTH do return 5
	}
	return f
}

@(private)
edge_copy :: proc "contextless" (dst: []$T, dst_stride: int, ref: ^Ref_Plane(T), x0, y0, cols, rows: int) #no_bounds_check {
	width := ref.last_x + 1
	lo := clamp(-x0, 0, cols)
	hi := clamp(width - x0, lo, cols)
	for r in 0 ..< rows {
		line := ref.data[clamp(y0 + r, 0, ref.last_y) * ref.stride:]
		out := dst[r * dst_stride:][:cols]
		for &v in out[:lo] do v = line[0]
		copy(out[lo:hi], line[x0 + lo:x0 + hi])
		for &v in out[hi:] do v = line[ref.last_x]
	}
}

// 7.11.3.4: 2-D subpel convolution into an unclipped `i32` prediction buffer `pred`
// (stride `pred_stride`), using `(inter_round0, inter_round1)` for either single
// (`is_compound == false`) or compound (`is_compound == true`) prediction.
convolve_2d_pred :: proc(
	pred: []i32,
	pred_stride: int,
	ref: ^Ref_Plane($T),
	start_x, start_y, step_x, step_y: int,
	w, h: int,
	filter_x, filter_y: int,
	is_compound: bool,
	bit_depth := 8,
) #no_bounds_check {
	r0, r1, _ := inter_rounding_vars(is_compound, bit_depth)
	fx_kind := effective_interp_filter(filter_x, w)
	fy_kind := effective_interp_filter(filter_y, h)
	kx_table := &SUBPEL_FILTERS[fx_kind]
	ky_table := &SUBPEL_FILTERS[fy_kind]

	// Fast path for unscaled 1:1 reference sampling (step_x == 1024 && step_y == 1024).
	if step_x == (1 << SCALE_SUBPEL_BITS) && step_y == (1 << SCALE_SUBPEL_BITS) {
		fx := (start_x >> 6) & SUBPEL_MASK
		fy := (start_y >> 6) & SUBPEL_MASK
		x0 := (start_x >> SCALE_SUBPEL_BITS) - 3
		y0 := (start_y >> SCALE_SUBPEL_BITS) - 3
		cols := w + 7
		rows := h + 7
		if x0 < 0 || y0 < 0 || x0 + cols - 1 > ref.last_x || y0 + rows - 1 > ref.last_y {
			convolve_unscaled_edge_i32(raw_data(pred), pred_stride, ref, x0, y0, cols, rows, w, h, fx, fy, &kx_table[fx], &ky_table[fy], r0, r1)
			return
		}
		src := raw_data(ref.data)
		src_stride := ref.stride
		origin := y0 * src_stride + x0
		convolve_unscaled_dispatch_i32(raw_data(pred), pred_stride, src[origin:], src_stride, w, h, fx, fy, &kx_table[fx], &ky_table[fy], r0, r1)
		return
	}

	convolve_scaled_i32(pred, pred_stride, ref, start_x, start_y, step_x, step_y, w, h, kx_table, ky_table, r0, r1)
}

@(private)
convolve_scaled_i32 :: #force_no_inline proc "contextless" (
	pred: []i32,
	pred_stride: int,
	ref: ^Ref_Plane($T),
	start_x, start_y, step_x, step_y, w, h: int,
	kx_table, ky_table: ^[16][8]i16,
	r0, r1: uint,
) #no_bounds_check {
	// General scaled convolution path (7.11.3.4).
	inter_h := (((h - 1) * step_y + (1 << SCALE_SUBPEL_BITS) - 1) >> SCALE_SUBPEL_BITS) + 8
	y0 := (start_y >> SCALE_SUBPEL_BITS) - 3
	intermediate: [272 * 128]i32 = ---
	for r in 0 ..< inter_h {
		ry := clamp(y0 + r, 0, ref.last_y)
		line := ref.data[ry * ref.stride:]
		for c in 0 ..< w {
			p := start_x + step_x * c
			kx := &kx_table[(p >> 6) & SUBPEL_MASK]
			ix := (p >> SCALE_SUBPEL_BITS) - 3
			s := i32(0)
			if ix >= 0 && ix + 7 <= ref.last_x {
				#unroll for t in 0 ..< 8 do s += i32(kx[t]) * i32(line[ix + t])
			} else {
				#unroll for t in 0 ..< 8 do s += i32(kx[t]) * i32(line[clamp(ix + t, 0, ref.last_x)])
			}
			intermediate[r * 128 + c] = round2(i64(s), r0)
		}
	}
	y_frac := start_y & ((1 << SCALE_SUBPEL_BITS) - 1)
	for r in 0 ..< h {
		p := y_frac + step_y * r
		ky := &ky_table[(p >> 6) & SUBPEL_MASK]
		base := (p >> SCALE_SUBPEL_BITS) * 128
		out_row := pred[r * pred_stride:]
		for c in 0 ..< w {
			s := i32(0)
			#unroll for t in 0 ..< 8 do s += i32(ky[t]) * intermediate[base + t * 128 + c]
			out_row[c] = round2(i64(s), r1)
		}
	}
}

@(private)
convolve_unscaled_edge_i32 :: #force_no_inline proc "contextless" (
	dst: [^]i32,
	dst_stride: int,
	ref: ^Ref_Plane($T),
	x0, y0, cols, rows, w, h, fx, fy: int,
	kx, ky: ^[8]i16,
	r0, r1: uint,
) #no_bounds_check {
	UNSCALED_STRIDE :: 136
	local: [UNSCALED_STRIDE * UNSCALED_STRIDE]T = ---
	edge_copy(local[:], UNSCALED_STRIDE, ref, x0, y0, cols, rows)
	convolve_unscaled_dispatch_i32(dst, dst_stride, raw_data(local[:]), UNSCALED_STRIDE, w, h, fx, fy, kx, ky, r0, r1)
}

@(private)
convolve_unscaled_dispatch_i32 :: #force_inline proc "contextless" (
	dst: [^]i32,
	dst_stride: int,
	src: [^]$T,
	src_stride: int,
	w, h, fx, fy: int,
	kx, ky: ^[8]i16,
	r0, r1: uint,
) #no_bounds_check {
	LANES :: (16 when intrinsics.has_target_feature("avx2") else 8) when T == u8 else max(8, runtime.NATIVE_SIMD_BIT_WIDTH / 32)
	switch {
	case w >= LANES:
		convolve_unscaled_lanes_i32(dst, dst_stride, src, src_stride, w, h, fx, fy, kx, ky, r0, r1, LANES)
	case w >= 8:
		convolve_unscaled_lanes_i32(dst, dst_stride, src, src_stride, w, h, fx, fy, kx, ky, r0, r1, 8)
	case w >= 4:
		convolve_unscaled_lanes_i32(dst, dst_stride, src, src_stride, w, h, fx, fy, kx, ky, r0, r1, 4)
	case:
		convolve_unscaled_lanes_i32(dst, dst_stride, src, src_stride, w, h, fx, fy, kx, ky, r0, r1, 2)
	}
}

// Packs two `#simd[N]i16` row vectors `(a, b)` into interleaved 16-bit pairs
// `[a0, b0, a1, b1, ...]` stored as `#simd[2*N]i16`.
//
// Performance note:
// Returning `#simd[2*N]i16` (1× 512-bit `zmm` for `N == 16`, or 1× 256-bit `ymm` for `N == 8`)
// instead of pre-widening to `#simd[2*N]i32` (which would occupy 2× `zmm` registers) keeps
// sliding vertical filter pairs (`p01, p23, p45, p67`) in single vector registers across loop
// iterations and allows LLVM to pattern-match the widening multiply + `simd.sums_of_n(..., 2)`
// inside `madd_i16_pairs` directly into `vpdpwssd` / `vpmaddwd` (`vpermt2w` + `vpdpwssd`).
@(private)
pack_i16_pairs_2 :: #force_inline proc "contextless" (a, b: #simd[2]i16) -> #simd[4]i16 {
	ua := (#simd[2]u32)(transmute(#simd[2]u16)a)
	ub := simd.shl_masked((#simd[2]u32)(transmute(#simd[2]u16)b), (#simd[2]u32)(16))
	return transmute(#simd[4]i16)(ua | ub)
}

@(private)
pack_i16_pairs_4 :: #force_inline proc "contextless" (a, b: #simd[4]i16) -> #simd[8]i16 {
	ua := (#simd[4]u32)(transmute(#simd[4]u16)a)
	ub := simd.shl_masked((#simd[4]u32)(transmute(#simd[4]u16)b), (#simd[4]u32)(16))
	return transmute(#simd[8]i16)(ua | ub)
}

@(private)
pack_i16_pairs_8 :: #force_inline proc "contextless" (a, b: #simd[8]i16) -> #simd[16]i16 {
	ua := (#simd[8]u32)(transmute(#simd[8]u16)a)
	ub := simd.shl_masked((#simd[8]u32)(transmute(#simd[8]u16)b), (#simd[8]u32)(16))
	return transmute(#simd[16]i16)(ua | ub)
}

@(private)
pack_i16_pairs_16 :: #force_inline proc "contextless" (a, b: #simd[16]i16) -> #simd[32]i16 {
	ua := (#simd[16]u32)(transmute(#simd[16]u16)a)
	ub := simd.shl_masked((#simd[16]u32)(transmute(#simd[16]u16)b), (#simd[16]u32)(16))
	return transmute(#simd[32]i16)(ua | ub)
}

@(private)
pack_i16_pairs :: proc{pack_i16_pairs_2, pack_i16_pairs_4, pack_i16_pairs_8, pack_i16_pairs_16}

@(private)
splat_i16_pair :: #force_inline proc "contextless" (ka, kb: i16, $N: int) -> #simd[2 * N]i16 {
	U32 :: #simd[N]u32
	I16_2 :: #simd[2 * N]i16
	u := u32(u16(ka)) | (u32(u16(kb)) << 16)
	return transmute(I16_2)U32(u)
}

// Pairwise 16-bit signed multiply-accumulate: computes `a[i]*ka + b[i]*kb` into `i32` lanes.
// Widening `#simd[2*N]i16 -> #simd[2*N]i32` immediately before `simd.sums_of_n(..., 2)`
// enables LLVM to lower the entire operation to a single `vpdpwssd` / `vpmaddwd` instruction.
@(private)
madd_i16_pairs_4 :: #force_inline proc "contextless" (ab, k: #simd[4]i16) -> #simd[2]i32 {
	return simd.sums_of_n((#simd[4]i32)(ab) * (#simd[4]i32)(k), 2)
}

@(private)
madd_i16_pairs_8 :: #force_inline proc "contextless" (ab, k: #simd[8]i16) -> #simd[4]i32 {
	return simd.sums_of_n((#simd[8]i32)(ab) * (#simd[8]i32)(k), 2)
}

@(private)
madd_i16_pairs_16 :: #force_inline proc "contextless" (ab, k: #simd[16]i16) -> #simd[8]i32 {
	return simd.sums_of_n((#simd[16]i32)(ab) * (#simd[16]i32)(k), 2)
}

@(private)
madd_i16_pairs_32 :: #force_inline proc "contextless" (ab, k: #simd[32]i16) -> #simd[16]i32 {
	return simd.sums_of_n((#simd[32]i32)(ab) * (#simd[32]i32)(k), 2)
}

@(private)
madd_i16_pairs :: proc{madd_i16_pairs_4, madd_i16_pairs_8, madd_i16_pairs_16, madd_i16_pairs_32}

@(private)
convolve_unscaled_lanes_i32 :: proc "contextless" (
	dst: [^]i32,
	dst_stride: int,
	src: [^]$T,
	src_stride: int,
	w, h, fx, fy: int,
	kx, ky: ^[8]i16,
	r0, r1: uint,
	$N: int,
) #no_bounds_check {
	V :: #simd[N]i32
	U :: #simd[N]u32
	S :: #simd[N]T
	I16 :: #simd[N]i16
	HW :: #simd[N]i16 when T == u8 else #simd[N]i32
	HU :: #simd[N]u16 when T == u8 else #simd[N]u32

	if fx == 0 && fy == 0 {
		sh := U(u32((14 - (r0 + r1)) & 15))
		base := 3 * src_stride + 3
		for r in 0 ..< h {
			for c := 0; c < w; c += N {
				sv := V(intrinsics.unaligned_load((^S)(&src[base + r * src_stride + c])))
				intrinsics.unaligned_store((^V)(&dst[r * dst_stride + c]), simd.shl_masked(sv, sh))
			}
		}
		return
	}

	half0: HW
	shift0: HU
	when T == u8 {
		half0 = HW(2)
		shift0 = HU(2)
	} else {
		half0 = HW(i32(1) << ((r0 - 1) & 7))
		shift0 = HU(u32(r0 & 7))
	}
	half1 := V(i32(1) << ((r1 - 1) & 15))
	shift1 := U(u32(r1 & 15))
	four_tap_x := N <= 4 || ((kx[0] | kx[1] | kx[6] | kx[7]) == 0)
	six_tap_x := (kx[0] | kx[7]) == 0
	four_tap_y := (ky[0] | ky[1] | ky[6] | ky[7]) == 0
	six_tap_y := (ky[0] | ky[7]) == 0

	if fy == 0 {
		kvx: [8]HW = ---
		when T == u8 {
			#unroll for t in 0 ..< 8 do kvx[t] = HW(kx[t] >> 1)
		} else {
			#unroll for t in 0 ..< 8 do kvx[t] = HW(i32(kx[t]))
		}
		sh7 := U(7)
		base := 3 * src_stride
		if four_tap_x {
			for r in 0 ..< h {
				for c := 0; c < w; c += N {
					at := base + r * src_stride + c
					acc := half0
					#unroll for t in 2 ..< 6 {
						acc += kvx[t] * HW(intrinsics.unaligned_load((^S)(&src[at + t])))
					}
					m := V(simd.shr_masked(acc, shift0))
					out := r1 == 7 ? m : simd.shr_masked(simd.shl_masked(m, sh7) + half1, shift1)
					intrinsics.unaligned_store((^V)(&dst[r * dst_stride + c]), out)
				}
			}
		} else if six_tap_x {
			for r in 0 ..< h {
				for c := 0; c < w; c += N {
					at := base + r * src_stride + c
					acc := half0
					#unroll for t in 1 ..< 7 {
						acc += kvx[t] * HW(intrinsics.unaligned_load((^S)(&src[at + t])))
					}
					m := V(simd.shr_masked(acc, shift0))
					out := r1 == 7 ? m : simd.shr_masked(simd.shl_masked(m, sh7) + half1, shift1)
					intrinsics.unaligned_store((^V)(&dst[r * dst_stride + c]), out)
				}
			}
		} else {
			for r in 0 ..< h {
				for c := 0; c < w; c += N {
					at := base + r * src_stride + c
					acc := half0
					#unroll for t in 0 ..< 8 {
						acc += kvx[t] * HW(intrinsics.unaligned_load((^S)(&src[at + t])))
					}
					m := V(simd.shr_masked(acc, shift0))
					out := r1 == 7 ? m : simd.shr_masked(simd.shl_masked(m, sh7) + half1, shift1)
					intrinsics.unaligned_store((^V)(&dst[r * dst_stride + c]), out)
				}
			}
		}
		return
	}

	if fx == 0 {
		when T == u8 {
			kvy16: [8]HW = ---
			#unroll for t in 0 ..< 8 do kvy16[t] = HW(ky[t] >> 1)
			v_half := r1 == 7 ? HW(2) : HW(32)
			v_shift := r1 == 7 ? HU(2) : HU(6)
			if four_tap_y {
				for r in 0 ..< h {
					for c := 0; c < w; c += N {
						at := r * src_stride + 3 + c
						acc := v_half
						#unroll for t in 2 ..< 6 {
							acc += kvy16[t] * HW(intrinsics.unaligned_load((^S)(&src[at + t * src_stride])))
						}
						intrinsics.unaligned_store((^V)(&dst[r * dst_stride + c]), V(simd.shr_masked(acc, v_shift)))
					}
				}
			} else if six_tap_y {
				for r in 0 ..< h {
					for c := 0; c < w; c += N {
						at := r * src_stride + 3 + c
						acc := v_half
						#unroll for t in 1 ..< 7 {
							acc += kvy16[t] * HW(intrinsics.unaligned_load((^S)(&src[at + t * src_stride])))
						}
						intrinsics.unaligned_store((^V)(&dst[r * dst_stride + c]), V(simd.shr_masked(acc, v_shift)))
					}
				}
			} else {
				for r in 0 ..< h {
					for c := 0; c < w; c += N {
						at := r * src_stride + 3 + c
						acc := v_half
						#unroll for t in 0 ..< 8 {
							acc += kvy16[t] * HW(intrinsics.unaligned_load((^S)(&src[at + t * src_stride])))
						}
						intrinsics.unaligned_store((^V)(&dst[r * dst_stride + c]), V(simd.shr_masked(acc, v_shift)))
					}
				}
			}
			return
		} else {
			kvy: [8]V = ---
			pre_sh := uint((7 - r0) & 7)
			#unroll for t in 0 ..< 8 do kvy[t] = V(i32(ky[t]) << pre_sh)
			if four_tap_y {
				for r in 0 ..< h {
					for c := 0; c < w; c += N {
						at := r * src_stride + 3 + c
						acc := half1
						#unroll for t in 2 ..< 6 {
							acc += kvy[t] * V(intrinsics.unaligned_load((^S)(&src[at + t * src_stride])))
						}
						intrinsics.unaligned_store((^V)(&dst[r * dst_stride + c]), simd.shr_masked(acc, shift1))
					}
				}
			} else if six_tap_y {
				for r in 0 ..< h {
					for c := 0; c < w; c += N {
						at := r * src_stride + 3 + c
						acc := half1
						#unroll for t in 1 ..< 7 {
							acc += kvy[t] * V(intrinsics.unaligned_load((^S)(&src[at + t * src_stride])))
						}
						intrinsics.unaligned_store((^V)(&dst[r * dst_stride + c]), simd.shr_masked(acc, shift1))
					}
				}
			} else {
				for r in 0 ..< h {
					for c := 0; c < w; c += N {
						at := r * src_stride + 3 + c
						acc := half1
						#unroll for t in 0 ..< 8 {
							acc += kvy[t] * V(intrinsics.unaligned_load((^S)(&src[at + t * src_stride])))
						}
						intrinsics.unaligned_store((^V)(&dst[r * dst_stride + c]), simd.shr_masked(acc, shift1))
					}
				}
			}
			return
		}
	}

	HW2 :: #simd[2 * N]i16
	HU2 :: #simd[2 * N]u16
	kvx: [8]HW = ---
	kvx2: [8]HW2 = ---
	#unroll for t in 0 ..< 8 {
		when T == u8 {
			kvx[t] = HW(kx[t] >> 1)
			kvx2[t] = HW2(kx[t] >> 1)
		} else {
			kvx[t] = HW(i32(kx[t]))
			kvx2[t] = HW2(0)
		}
	}
	_ = kvx2
	half0_2 :: HW2(2)
	shift0_2 :: HU2(2)
	kvy01 := splat_i16_pair(ky[0], ky[1], N)
	kvy12 := splat_i16_pair(ky[1], ky[2], N)
	kvy23 := splat_i16_pair(ky[2], ky[3], N)
	kvy34 := splat_i16_pair(ky[3], ky[4], N)
	kvy45 := splat_i16_pair(ky[4], ky[5], N)
	kvy56 := splat_i16_pair(ky[5], ky[6], N)
	kvy67 := splat_i16_pair(ky[6], ky[7], N)

	hrow8 :: #force_inline proc "contextless" (src: [^]T, at: int, kvx: ^[8]HW, half0: HW, shift0: HU) -> I16 #no_bounds_check {
		acc := half0
		#unroll for t in 0 ..< 8 {
			acc += kvx[t] * HW(intrinsics.unaligned_load((^S)(&src[at + t])))
		}
		return I16(simd.shr_masked(acc, shift0))
	}
	hrow6 :: #force_inline proc "contextless" (src: [^]T, at: int, kvx: ^[8]HW, half0: HW, shift0: HU) -> I16 #no_bounds_check {
		acc := half0
		#unroll for t in 1 ..< 7 {
			acc += kvx[t] * HW(intrinsics.unaligned_load((^S)(&src[at + t])))
		}
		return I16(simd.shr_masked(acc, shift0))
	}
	hrow4 :: #force_inline proc "contextless" (src: [^]T, at: int, kvx: ^[8]HW, half0: HW, shift0: HU) -> I16 #no_bounds_check {
		acc := half0
		#unroll for t in 2 ..< 6 {
			acc += kvx[t] * HW(intrinsics.unaligned_load((^S)(&src[at + t])))
		}
		return I16(simd.shr_masked(acc, shift0))
	}
	// Performance note:
	// For 8-bit (`T == u8, N == 16`), a single row of 16 `i16` accumulators only occupies a
	// 256-bit `ymm` register. Concatenating two rows `(at0, at1)` into `#simd[32]u8` and widening
	// to `#simd[32]i16` (`512-bit zmm`) computes both horizontal rows simultaneously in a single
	// 512-bit `zmm` pass (`vinserti128 + vpmovzxbw zmm + vpmullw zmm + vpaddw zmm`), cutting
	// horizontal convolution instructions in half.
	hrow8_pair :: #force_inline proc "contextless" (src: [^]T, at0, at1: int, kvx: ^[8]HW, kvx2: ^[8]HW2, half0: HW, shift0: HU) -> (I16, I16) #no_bounds_check {
		when T == u8 && N == 16 {
			_ = kvx
			_ = half0
			_ = shift0
			acc := half0_2
			#unroll for t in 0 ..< 8 {
				s0 := intrinsics.unaligned_load((^S)(&src[at0 + t]))
				s1 := intrinsics.unaligned_load((^S)(&src[at1 + t]))
				s01 := simd.shuffle(s0, s1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31)
				acc += kvx2[t] * HW2(s01)
			}
			res := simd.shr_masked(acc, shift0_2)
			return simd.shuffle(res, res, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15),
				simd.shuffle(res, res, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31)
		} else {
			_ = kvx2
			return hrow8(src, at0, kvx, half0, shift0), hrow8(src, at1, kvx, half0, shift0)
		}
	}
	hrow6_pair :: #force_inline proc "contextless" (src: [^]T, at0, at1: int, kvx: ^[8]HW, kvx2: ^[8]HW2, half0: HW, shift0: HU) -> (I16, I16) #no_bounds_check {
		when T == u8 && N == 16 {
			_ = kvx
			_ = half0
			_ = shift0
			acc := half0_2
			#unroll for t in 1 ..< 7 {
				s0 := intrinsics.unaligned_load((^S)(&src[at0 + t]))
				s1 := intrinsics.unaligned_load((^S)(&src[at1 + t]))
				s01 := simd.shuffle(s0, s1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31)
				acc += kvx2[t] * HW2(s01)
			}
			res := simd.shr_masked(acc, shift0_2)
			return simd.shuffle(res, res, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15),
				simd.shuffle(res, res, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31)
		} else {
			_ = kvx2
			return hrow6(src, at0, kvx, half0, shift0), hrow6(src, at1, kvx, half0, shift0)
		}
	}
	hrow4_pair :: #force_inline proc "contextless" (src: [^]T, at0, at1: int, kvx: ^[8]HW, kvx2: ^[8]HW2, half0: HW, shift0: HU) -> (I16, I16) #no_bounds_check {
		when T == u8 && N == 16 {
			_ = kvx
			_ = half0
			_ = shift0
			acc := half0_2
			#unroll for t in 2 ..< 6 {
				s0 := intrinsics.unaligned_load((^S)(&src[at0 + t]))
				s1 := intrinsics.unaligned_load((^S)(&src[at1 + t]))
				s01 := simd.shuffle(s0, s1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31)
				acc += kvx2[t] * HW2(s01)
			}
			res := simd.shr_masked(acc, shift0_2)
			return simd.shuffle(res, res, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15),
				simd.shuffle(res, res, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31)
		} else {
			_ = kvx2
			return hrow4(src, at0, kvx, half0, shift0), hrow4(src, at1, kvx, half0, shift0)
		}
	}
	if four_tap_y {
		if four_tap_x {
			for c := 0; c < w; c += N {
				m2, m3 := hrow4_pair(src, 2 * src_stride + c, 3 * src_stride + c, &kvx, &kvx2, half0, shift0)
				m4 := hrow4(src, 4 * src_stride + c, &kvx, half0, shift0)
				p23 := pack_i16_pairs(m2, m3)
				p34 := pack_i16_pairs(m3, m4)
				for r := 0; r < h; r += 2 {
					at5 := (r + 5) * src_stride + c
					m5, m6 := hrow4_pair(src, at5, at5 + src_stride, &kvx, &kvx2, half0, shift0)
					p45 := pack_i16_pairs(m4, m5)
					acc0 := half1 + madd_i16_pairs(p23, kvy23) + madd_i16_pairs(p45, kvy45)
					intrinsics.unaligned_store((^V)(&dst[r * dst_stride + c]), simd.shr_masked(acc0, shift1))
					p56 := pack_i16_pairs(m5, m6)
					acc1 := half1 + madd_i16_pairs(p34, kvy23) + madd_i16_pairs(p56, kvy45)
					intrinsics.unaligned_store((^V)(&dst[(r + 1) * dst_stride + c]), simd.shr_masked(acc1, shift1))
					m4 = m6
					p23, p34 = p45, p56
				}
			}
		} else if six_tap_x {
			for c := 0; c < w; c += N {
				m2, m3 := hrow6_pair(src, 2 * src_stride + c, 3 * src_stride + c, &kvx, &kvx2, half0, shift0)
				m4 := hrow6(src, 4 * src_stride + c, &kvx, half0, shift0)
				p23 := pack_i16_pairs(m2, m3)
				p34 := pack_i16_pairs(m3, m4)
				for r := 0; r < h; r += 2 {
					at5 := (r + 5) * src_stride + c
					m5, m6 := hrow6_pair(src, at5, at5 + src_stride, &kvx, &kvx2, half0, shift0)
					p45 := pack_i16_pairs(m4, m5)
					acc0 := half1 + madd_i16_pairs(p23, kvy23) + madd_i16_pairs(p45, kvy45)
					intrinsics.unaligned_store((^V)(&dst[r * dst_stride + c]), simd.shr_masked(acc0, shift1))
					p56 := pack_i16_pairs(m5, m6)
					acc1 := half1 + madd_i16_pairs(p34, kvy23) + madd_i16_pairs(p56, kvy45)
					intrinsics.unaligned_store((^V)(&dst[(r + 1) * dst_stride + c]), simd.shr_masked(acc1, shift1))
					m4 = m6
					p23, p34 = p45, p56
				}
			}
		} else {
			for c := 0; c < w; c += N {
				m2, m3 := hrow8_pair(src, 2 * src_stride + c, 3 * src_stride + c, &kvx, &kvx2, half0, shift0)
				m4 := hrow8(src, 4 * src_stride + c, &kvx, half0, shift0)
				p23 := pack_i16_pairs(m2, m3)
				p34 := pack_i16_pairs(m3, m4)
				for r := 0; r < h; r += 2 {
					at5 := (r + 5) * src_stride + c
					m5, m6 := hrow8_pair(src, at5, at5 + src_stride, &kvx, &kvx2, half0, shift0)
					p45 := pack_i16_pairs(m4, m5)
					acc0 := half1 + madd_i16_pairs(p23, kvy23) + madd_i16_pairs(p45, kvy45)
					intrinsics.unaligned_store((^V)(&dst[r * dst_stride + c]), simd.shr_masked(acc0, shift1))
					p56 := pack_i16_pairs(m5, m6)
					acc1 := half1 + madd_i16_pairs(p34, kvy23) + madd_i16_pairs(p56, kvy45)
					intrinsics.unaligned_store((^V)(&dst[(r + 1) * dst_stride + c]), simd.shr_masked(acc1, shift1))
					m4 = m6
					p23, p34 = p45, p56
				}
			}
		}
		return
	}

	if six_tap_y {
		if four_tap_x {
			for c := 0; c < w; c += N {
				m1, m2 := hrow4_pair(src, 1 * src_stride + c, 2 * src_stride + c, &kvx, &kvx2, half0, shift0)
				m3, m4 := hrow4_pair(src, 3 * src_stride + c, 4 * src_stride + c, &kvx, &kvx2, half0, shift0)
				m5 := hrow4(src, 5 * src_stride + c, &kvx, half0, shift0)
				p12 := pack_i16_pairs(m1, m2)
				p23 := pack_i16_pairs(m2, m3)
				p34 := pack_i16_pairs(m3, m4)
				p45 := pack_i16_pairs(m4, m5)
				for r := 0; r < h; r += 2 {
					at6 := (r + 6) * src_stride + c
					m6, m7 := hrow4_pair(src, at6, at6 + src_stride, &kvx, &kvx2, half0, shift0)
					p56 := pack_i16_pairs(m5, m6)
					acc0 := half1 + madd_i16_pairs(p12, kvy12) + madd_i16_pairs(p34, kvy34) + madd_i16_pairs(p56, kvy56)
					intrinsics.unaligned_store((^V)(&dst[r * dst_stride + c]), simd.shr_masked(acc0, shift1))
					p67 := pack_i16_pairs(m6, m7)
					acc1 := half1 + madd_i16_pairs(p23, kvy12) + madd_i16_pairs(p45, kvy34) + madd_i16_pairs(p67, kvy56)
					intrinsics.unaligned_store((^V)(&dst[(r + 1) * dst_stride + c]), simd.shr_masked(acc1, shift1))
					m5 = m7
					p12, p34 = p34, p56
					p23, p45 = p45, p67
				}
			}
		} else if six_tap_x {
			for c := 0; c < w; c += N {
				m1, m2 := hrow6_pair(src, 1 * src_stride + c, 2 * src_stride + c, &kvx, &kvx2, half0, shift0)
				m3, m4 := hrow6_pair(src, 3 * src_stride + c, 4 * src_stride + c, &kvx, &kvx2, half0, shift0)
				m5 := hrow6(src, 5 * src_stride + c, &kvx, half0, shift0)
				p12 := pack_i16_pairs(m1, m2)
				p23 := pack_i16_pairs(m2, m3)
				p34 := pack_i16_pairs(m3, m4)
				p45 := pack_i16_pairs(m4, m5)
				for r := 0; r < h; r += 2 {
					at6 := (r + 6) * src_stride + c
					m6, m7 := hrow6_pair(src, at6, at6 + src_stride, &kvx, &kvx2, half0, shift0)
					p56 := pack_i16_pairs(m5, m6)
					acc0 := half1 + madd_i16_pairs(p12, kvy12) + madd_i16_pairs(p34, kvy34) + madd_i16_pairs(p56, kvy56)
					intrinsics.unaligned_store((^V)(&dst[r * dst_stride + c]), simd.shr_masked(acc0, shift1))
					p67 := pack_i16_pairs(m6, m7)
					acc1 := half1 + madd_i16_pairs(p23, kvy12) + madd_i16_pairs(p45, kvy34) + madd_i16_pairs(p67, kvy56)
					intrinsics.unaligned_store((^V)(&dst[(r + 1) * dst_stride + c]), simd.shr_masked(acc1, shift1))
					m5 = m7
					p12, p34 = p34, p56
					p23, p45 = p45, p67
				}
			}
		} else {
			for c := 0; c < w; c += N {
				m1, m2 := hrow8_pair(src, 1 * src_stride + c, 2 * src_stride + c, &kvx, &kvx2, half0, shift0)
				m3, m4 := hrow8_pair(src, 3 * src_stride + c, 4 * src_stride + c, &kvx, &kvx2, half0, shift0)
				m5 := hrow8(src, 5 * src_stride + c, &kvx, half0, shift0)
				p12 := pack_i16_pairs(m1, m2)
				p23 := pack_i16_pairs(m2, m3)
				p34 := pack_i16_pairs(m3, m4)
				p45 := pack_i16_pairs(m4, m5)
				for r := 0; r < h; r += 2 {
					at6 := (r + 6) * src_stride + c
					m6, m7 := hrow8_pair(src, at6, at6 + src_stride, &kvx, &kvx2, half0, shift0)
					p56 := pack_i16_pairs(m5, m6)
					acc0 := half1 + madd_i16_pairs(p12, kvy12) + madd_i16_pairs(p34, kvy34) + madd_i16_pairs(p56, kvy56)
					intrinsics.unaligned_store((^V)(&dst[r * dst_stride + c]), simd.shr_masked(acc0, shift1))
					p67 := pack_i16_pairs(m6, m7)
					acc1 := half1 + madd_i16_pairs(p23, kvy12) + madd_i16_pairs(p45, kvy34) + madd_i16_pairs(p67, kvy56)
					intrinsics.unaligned_store((^V)(&dst[(r + 1) * dst_stride + c]), simd.shr_masked(acc1, shift1))
					m5 = m7
					p12, p34 = p34, p56
					p23, p45 = p45, p67
				}
			}
		}
		return
	}

	if four_tap_x {
		for c := 0; c < w; c += N {
			m0, m1 := hrow4_pair(src, 0 * src_stride + c, 1 * src_stride + c, &kvx, &kvx2, half0, shift0)
			m2, m3 := hrow4_pair(src, 2 * src_stride + c, 3 * src_stride + c, &kvx, &kvx2, half0, shift0)
			m4, m5 := hrow4_pair(src, 4 * src_stride + c, 5 * src_stride + c, &kvx, &kvx2, half0, shift0)
			m6 := hrow4(src, 6 * src_stride + c, &kvx, half0, shift0)
			p01 := pack_i16_pairs(m0, m1)
			p12 := pack_i16_pairs(m1, m2)
			p23 := pack_i16_pairs(m2, m3)
			p34 := pack_i16_pairs(m3, m4)
			p45 := pack_i16_pairs(m4, m5)
			p56 := pack_i16_pairs(m5, m6)
			for r := 0; r < h; r += 2 {
				at7 := (r + 7) * src_stride + c
				m7, m8 := hrow4_pair(src, at7, at7 + src_stride, &kvx, &kvx2, half0, shift0)
				p67 := pack_i16_pairs(m6, m7)
				acc0 := half1 + madd_i16_pairs(p01, kvy01) + madd_i16_pairs(p23, kvy23) +
					madd_i16_pairs(p45, kvy45) + madd_i16_pairs(p67, kvy67)
				intrinsics.unaligned_store((^V)(&dst[r * dst_stride + c]), simd.shr_masked(acc0, shift1))
				p78 := pack_i16_pairs(m7, m8)
				acc1 := half1 + madd_i16_pairs(p12, kvy01) + madd_i16_pairs(p34, kvy23) +
					madd_i16_pairs(p56, kvy45) + madd_i16_pairs(p78, kvy67)
				intrinsics.unaligned_store((^V)(&dst[(r + 1) * dst_stride + c]), simd.shr_masked(acc1, shift1))
				m6 = m8
				p01, p23, p45 = p23, p45, p67
				p12, p34, p56 = p34, p56, p78
			}
		}
		return
	}

	if six_tap_x {
		for c := 0; c < w; c += N {
			m0, m1 := hrow6_pair(src, 0 * src_stride + c, 1 * src_stride + c, &kvx, &kvx2, half0, shift0)
			m2, m3 := hrow6_pair(src, 2 * src_stride + c, 3 * src_stride + c, &kvx, &kvx2, half0, shift0)
			m4, m5 := hrow6_pair(src, 4 * src_stride + c, 5 * src_stride + c, &kvx, &kvx2, half0, shift0)
			m6 := hrow6(src, 6 * src_stride + c, &kvx, half0, shift0)
			p01 := pack_i16_pairs(m0, m1)
			p12 := pack_i16_pairs(m1, m2)
			p23 := pack_i16_pairs(m2, m3)
			p34 := pack_i16_pairs(m3, m4)
			p45 := pack_i16_pairs(m4, m5)
			p56 := pack_i16_pairs(m5, m6)
			for r := 0; r < h; r += 2 {
				at7 := (r + 7) * src_stride + c
				m7, m8 := hrow6_pair(src, at7, at7 + src_stride, &kvx, &kvx2, half0, shift0)
				p67 := pack_i16_pairs(m6, m7)
				acc0 := half1 + madd_i16_pairs(p01, kvy01) + madd_i16_pairs(p23, kvy23) +
					madd_i16_pairs(p45, kvy45) + madd_i16_pairs(p67, kvy67)
				intrinsics.unaligned_store((^V)(&dst[r * dst_stride + c]), simd.shr_masked(acc0, shift1))
				p78 := pack_i16_pairs(m7, m8)
				acc1 := half1 + madd_i16_pairs(p12, kvy01) + madd_i16_pairs(p34, kvy23) +
					madd_i16_pairs(p56, kvy45) + madd_i16_pairs(p78, kvy67)
				intrinsics.unaligned_store((^V)(&dst[(r + 1) * dst_stride + c]), simd.shr_masked(acc1, shift1))
				m6 = m8
				p01, p23, p45 = p23, p45, p67
				p12, p34, p56 = p34, p56, p78
			}
		}
		return
	}

	for c := 0; c < w; c += N {
		m0, m1 := hrow8_pair(src, 0 * src_stride + c, 1 * src_stride + c, &kvx, &kvx2, half0, shift0)
		m2, m3 := hrow8_pair(src, 2 * src_stride + c, 3 * src_stride + c, &kvx, &kvx2, half0, shift0)
		m4, m5 := hrow8_pair(src, 4 * src_stride + c, 5 * src_stride + c, &kvx, &kvx2, half0, shift0)
		m6 := hrow8(src, 6 * src_stride + c, &kvx, half0, shift0)
		p01 := pack_i16_pairs(m0, m1)
		p12 := pack_i16_pairs(m1, m2)
		p23 := pack_i16_pairs(m2, m3)
		p34 := pack_i16_pairs(m3, m4)
		p45 := pack_i16_pairs(m4, m5)
		p56 := pack_i16_pairs(m5, m6)
		for r := 0; r < h; r += 2 {
			at7 := (r + 7) * src_stride + c
			m7, m8 := hrow8_pair(src, at7, at7 + src_stride, &kvx, &kvx2, half0, shift0)
			p67 := pack_i16_pairs(m6, m7)
			acc0 := half1 + madd_i16_pairs(p01, kvy01) + madd_i16_pairs(p23, kvy23) +
				madd_i16_pairs(p45, kvy45) + madd_i16_pairs(p67, kvy67)
			intrinsics.unaligned_store((^V)(&dst[r * dst_stride + c]), simd.shr_masked(acc0, shift1))
			p78 := pack_i16_pairs(m7, m8)
			acc1 := half1 + madd_i16_pairs(p12, kvy01) + madd_i16_pairs(p34, kvy23) +
				madd_i16_pairs(p56, kvy45) + madd_i16_pairs(p78, kvy67)
			intrinsics.unaligned_store((^V)(&dst[(r + 1) * dst_stride + c]), simd.shr_masked(acc1, shift1))
			m6 = m8
			p01, p23, p45 = p23, p45, p67
			p12, p34, p56 = p34, p56, p78
		}
	}
}

@(private)
predict_unscaled_edge_single :: #force_no_inline proc "contextless" (
	dst: [^]$T,
	dst_stride: int,
	ref: ^Ref_Plane(T),
	x0, y0, cols, rows, w, h, fx, fy: int,
	kx, ky: ^[8]i16,
	r0, r1: uint,
	max_val: i32,
) #no_bounds_check {
	UNSCALED_STRIDE :: 136
	local: [UNSCALED_STRIDE * UNSCALED_STRIDE]T = ---
	edge_copy(local[:], UNSCALED_STRIDE, ref, x0, y0, cols, rows)
	predict_unscaled_dispatch_single(dst, dst_stride, raw_data(local[:]), UNSCALED_STRIDE, w, h, fx, fy, kx, ky, r0, r1, max_val)
}

@(private)
predict_unscaled_dispatch_single :: #force_inline proc "contextless" (
	dst: [^]$T,
	dst_stride: int,
	src: [^]T,
	src_stride: int,
	w, h, fx, fy: int,
	kx, ky: ^[8]i16,
	r0, r1: uint,
	max_val: i32,
) #no_bounds_check {
	LANES :: (16 when intrinsics.has_target_feature("avx2") else 8) when T == u8 else max(8, runtime.NATIVE_SIMD_BIT_WIDTH / 32)
	switch {
	case w >= LANES:
		predict_unscaled_lanes_single(dst, dst_stride, src, src_stride, w, h, fx, fy, kx, ky, r0, r1, max_val, LANES)
	case w >= 8:
		predict_unscaled_lanes_single(dst, dst_stride, src, src_stride, w, h, fx, fy, kx, ky, r0, r1, max_val, 8)
	case w >= 4:
		predict_unscaled_lanes_single(dst, dst_stride, src, src_stride, w, h, fx, fy, kx, ky, r0, r1, max_val, 4)
	case:
		predict_unscaled_lanes_single(dst, dst_stride, src, src_stride, w, h, fx, fy, kx, ky, r0, r1, max_val, 2)
	}
}

@(private)
predict_unscaled_lanes_single :: proc "contextless" (
	dst: [^]$T,
	dst_stride: int,
	src: [^]T,
	src_stride: int,
	w, h, fx, fy: int,
	kx, ky: ^[8]i16,
	r0, r1: uint,
	max_val: i32,
	$N: int,
) #no_bounds_check {
	V :: #simd[N]i32
	U :: #simd[N]u32
	S :: #simd[N]T
	I16 :: #simd[N]i16
	HW :: #simd[N]i16 when T == u8 else #simd[N]i32
	HU :: #simd[N]u16 when T == u8 else #simd[N]u32

	if fx == 0 && fy == 0 {
		base := 3 * src_stride + 3
		for r in 0 ..< h {
			for c := 0; c < w; c += N {
				sv := intrinsics.unaligned_load((^S)(&src[base + r * src_stride + c]))
				intrinsics.unaligned_store((^S)(&dst[r * dst_stride + c]), sv)
			}
		}
		return
	}

	zero := V(0)
	hi := V(max_val)
	half0: HW
	shift0: HU
	when T == u8 {
		half0 = HW(2)
		shift0 = HU(2)
	} else {
		half0 = HW(i32(1) << ((r0 - 1) & 7))
		shift0 = HU(u32(r0 & 7))
	}
	half1 := V(i32(1) << ((r1 - 1) & 15))
	shift1 := U(u32(r1 & 15))
	four_tap_x := N <= 4 || ((kx[0] | kx[1] | kx[6] | kx[7]) == 0)
	six_tap_x := (kx[0] | kx[7]) == 0
	four_tap_y := (ky[0] | ky[1] | ky[6] | ky[7]) == 0
	six_tap_y := (ky[0] | ky[7]) == 0

	if fy == 0 {
		kvx: [8]HW = ---
		base := 3 * src_stride
		when T == u8 {
			#unroll for t in 0 ..< 8 do kvx[t] = HW(kx[t] >> 1)
			zero16 := HW(0)
			hi16 := HW(255)
			half1_16 := HW(8)
			sh1_16 := HU(4)
			if four_tap_x {
				for r in 0 ..< h {
					for c := 0; c < w; c += N {
						at := base + r * src_stride + c
						acc := half0
						#unroll for t in 2 ..< 6 {
							acc += kvx[t] * HW(intrinsics.unaligned_load((^S)(&src[at + t])))
						}
						m := simd.shr_masked(acc, shift0)
						out := simd.clamp(simd.shr_masked(m + half1_16, sh1_16), zero16, hi16)
						intrinsics.unaligned_store((^S)(&dst[r * dst_stride + c]), S(out))
					}
				}
			} else if six_tap_x {
				for r in 0 ..< h {
					for c := 0; c < w; c += N {
						at := base + r * src_stride + c
						acc := half0
						#unroll for t in 1 ..< 7 {
							acc += kvx[t] * HW(intrinsics.unaligned_load((^S)(&src[at + t])))
						}
						m := simd.shr_masked(acc, shift0)
						out := simd.clamp(simd.shr_masked(m + half1_16, sh1_16), zero16, hi16)
						intrinsics.unaligned_store((^S)(&dst[r * dst_stride + c]), S(out))
					}
				}
			} else {
				for r in 0 ..< h {
					for c := 0; c < w; c += N {
						at := base + r * src_stride + c
						acc := half0
						#unroll for t in 0 ..< 8 {
							acc += kvx[t] * HW(intrinsics.unaligned_load((^S)(&src[at + t])))
						}
						m := simd.shr_masked(acc, shift0)
						out := simd.clamp(simd.shr_masked(m + half1_16, sh1_16), zero16, hi16)
						intrinsics.unaligned_store((^S)(&dst[r * dst_stride + c]), S(out))
					}
				}
			}
			return
		} else {
			#unroll for t in 0 ..< 8 do kvx[t] = HW(i32(kx[t]))
			sh7 := U(7)
			if four_tap_x {
				for r in 0 ..< h {
					for c := 0; c < w; c += N {
						at := base + r * src_stride + c
						acc := half0
						#unroll for t in 2 ..< 6 {
							acc += kvx[t] * HW(intrinsics.unaligned_load((^S)(&src[at + t])))
						}
						m := simd.shr_masked(acc, shift0)
						out := simd.clamp(simd.shr_masked(simd.shl_masked(m, sh7) + half1, shift1), zero, hi)
						intrinsics.unaligned_store((^S)(&dst[r * dst_stride + c]), S(out))
					}
				}
			} else if six_tap_x {
				for r in 0 ..< h {
					for c := 0; c < w; c += N {
						at := base + r * src_stride + c
						acc := half0
						#unroll for t in 1 ..< 7 {
							acc += kvx[t] * HW(intrinsics.unaligned_load((^S)(&src[at + t])))
						}
						m := simd.shr_masked(acc, shift0)
						out := simd.clamp(simd.shr_masked(simd.shl_masked(m, sh7) + half1, shift1), zero, hi)
						intrinsics.unaligned_store((^S)(&dst[r * dst_stride + c]), S(out))
					}
				}
			} else {
				for r in 0 ..< h {
					for c := 0; c < w; c += N {
						at := base + r * src_stride + c
						acc := half0
						#unroll for t in 0 ..< 8 {
							acc += kvx[t] * HW(intrinsics.unaligned_load((^S)(&src[at + t])))
						}
						m := simd.shr_masked(acc, shift0)
						out := simd.clamp(simd.shr_masked(simd.shl_masked(m, sh7) + half1, shift1), zero, hi)
						intrinsics.unaligned_store((^S)(&dst[r * dst_stride + c]), S(out))
					}
				}
			}
			return
		}
	}

	if fx == 0 {
		when T == u8 {
			kvy16: [8]HW = ---
			#unroll for t in 0 ..< 8 do kvy16[t] = HW(ky[t] >> 1)
			zero16 := HW(0)
			hi16 := HW(255)
			half1_16 := HW(32)
			sh1_16 := HU(6)
			if four_tap_y {
				for r in 0 ..< h {
					for c := 0; c < w; c += N {
						at := r * src_stride + 3 + c
						acc := half1_16
						#unroll for t in 2 ..< 6 {
							acc += kvy16[t] * HW(intrinsics.unaligned_load((^S)(&src[at + t * src_stride])))
						}
						out := simd.clamp(simd.shr_masked(acc, sh1_16), zero16, hi16)
						intrinsics.unaligned_store((^S)(&dst[r * dst_stride + c]), S(out))
					}
				}
			} else if six_tap_y {
				for r in 0 ..< h {
					for c := 0; c < w; c += N {
						at := r * src_stride + 3 + c
						acc := half1_16
						#unroll for t in 1 ..< 7 {
							acc += kvy16[t] * HW(intrinsics.unaligned_load((^S)(&src[at + t * src_stride])))
						}
						out := simd.clamp(simd.shr_masked(acc, sh1_16), zero16, hi16)
						intrinsics.unaligned_store((^S)(&dst[r * dst_stride + c]), S(out))
					}
				}
			} else {
				for r in 0 ..< h {
					for c := 0; c < w; c += N {
						at := r * src_stride + 3 + c
						acc := half1_16
						#unroll for t in 0 ..< 8 {
							acc += kvy16[t] * HW(intrinsics.unaligned_load((^S)(&src[at + t * src_stride])))
						}
						out := simd.clamp(simd.shr_masked(acc, sh1_16), zero16, hi16)
						intrinsics.unaligned_store((^S)(&dst[r * dst_stride + c]), S(out))
					}
				}
			}
			return
		} else {
			kvy: [8]V = ---
			pre_sh := uint((7 - r0) & 7)
			#unroll for t in 0 ..< 8 do kvy[t] = V(i32(ky[t]) << pre_sh)
			if four_tap_y {
				for r in 0 ..< h {
					for c := 0; c < w; c += N {
						at := r * src_stride + 3 + c
						acc := half1
						#unroll for t in 2 ..< 6 {
							acc += kvy[t] * V(intrinsics.unaligned_load((^S)(&src[at + t * src_stride])))
						}
						out := simd.clamp(simd.shr_masked(acc, shift1), zero, hi)
						intrinsics.unaligned_store((^S)(&dst[r * dst_stride + c]), S(out))
					}
				}
			} else if six_tap_y {
				for r in 0 ..< h {
					for c := 0; c < w; c += N {
						at := r * src_stride + 3 + c
						acc := half1
						#unroll for t in 1 ..< 7 {
							acc += kvy[t] * V(intrinsics.unaligned_load((^S)(&src[at + t * src_stride])))
						}
						out := simd.clamp(simd.shr_masked(acc, shift1), zero, hi)
						intrinsics.unaligned_store((^S)(&dst[r * dst_stride + c]), S(out))
					}
				}
			} else {
				for r in 0 ..< h {
					for c := 0; c < w; c += N {
						at := r * src_stride + 3 + c
						acc := half1
						#unroll for t in 0 ..< 8 {
							acc += kvy[t] * V(intrinsics.unaligned_load((^S)(&src[at + t * src_stride])))
						}
						out := simd.clamp(simd.shr_masked(acc, shift1), zero, hi)
						intrinsics.unaligned_store((^S)(&dst[r * dst_stride + c]), S(out))
					}
				}
			}
			return
		}
	}

	HW2 :: #simd[2 * N]i16
	HU2 :: #simd[2 * N]u16
	kvx: [8]HW = ---
	kvx2: [8]HW2 = ---
	#unroll for t in 0 ..< 8 {
		when T == u8 {
			kvx[t] = HW(kx[t] >> 1)
			kvx2[t] = HW2(kx[t] >> 1)
		} else {
			kvx[t] = HW(i32(kx[t]))
			kvx2[t] = HW2(0)
		}
	}
	_ = kvx2
	half0_2 :: HW2(2)
	shift0_2 :: HU2(2)
	kvy01 := splat_i16_pair(ky[0], ky[1], N)
	kvy12 := splat_i16_pair(ky[1], ky[2], N)
	kvy23 := splat_i16_pair(ky[2], ky[3], N)
	kvy34 := splat_i16_pair(ky[3], ky[4], N)
	kvy45 := splat_i16_pair(ky[4], ky[5], N)
	kvy56 := splat_i16_pair(ky[5], ky[6], N)
	kvy67 := splat_i16_pair(ky[6], ky[7], N)

	hrow8 :: #force_inline proc "contextless" (src: [^]T, at: int, kvx: ^[8]HW, half0: HW, shift0: HU) -> I16 #no_bounds_check {
		acc := half0
		#unroll for t in 0 ..< 8 {
			acc += kvx[t] * HW(intrinsics.unaligned_load((^S)(&src[at + t])))
		}
		return I16(simd.shr_masked(acc, shift0))
	}
	hrow6 :: #force_inline proc "contextless" (src: [^]T, at: int, kvx: ^[8]HW, half0: HW, shift0: HU) -> I16 #no_bounds_check {
		acc := half0
		#unroll for t in 1 ..< 7 {
			acc += kvx[t] * HW(intrinsics.unaligned_load((^S)(&src[at + t])))
		}
		return I16(simd.shr_masked(acc, shift0))
	}
	hrow4 :: #force_inline proc "contextless" (src: [^]T, at: int, kvx: ^[8]HW, half0: HW, shift0: HU) -> I16 #no_bounds_check {
		acc := half0
		#unroll for t in 2 ..< 6 {
			acc += kvx[t] * HW(intrinsics.unaligned_load((^S)(&src[at + t])))
		}
		return I16(simd.shr_masked(acc, shift0))
	}
	hrow8_pair :: #force_inline proc "contextless" (src: [^]T, at0, at1: int, kvx: ^[8]HW, kvx2: ^[8]HW2, half0: HW, shift0: HU) -> (I16, I16) #no_bounds_check {
		when T == u8 && N == 16 {
			_ = kvx
			_ = half0
			_ = shift0
			acc := half0_2
			#unroll for t in 0 ..< 8 {
				s0 := intrinsics.unaligned_load((^S)(&src[at0 + t]))
				s1 := intrinsics.unaligned_load((^S)(&src[at1 + t]))
				s01 := simd.shuffle(s0, s1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31)
				acc += kvx2[t] * HW2(s01)
			}
			res := simd.shr_masked(acc, shift0_2)
			return simd.shuffle(res, res, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15),
				simd.shuffle(res, res, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31)
		} else {
			_ = kvx2
			return hrow8(src, at0, kvx, half0, shift0), hrow8(src, at1, kvx, half0, shift0)
		}
	}
	hrow6_pair :: #force_inline proc "contextless" (src: [^]T, at0, at1: int, kvx: ^[8]HW, kvx2: ^[8]HW2, half0: HW, shift0: HU) -> (I16, I16) #no_bounds_check {
		when T == u8 && N == 16 {
			_ = kvx
			_ = half0
			_ = shift0
			acc := half0_2
			#unroll for t in 1 ..< 7 {
				s0 := intrinsics.unaligned_load((^S)(&src[at0 + t]))
				s1 := intrinsics.unaligned_load((^S)(&src[at1 + t]))
				s01 := simd.shuffle(s0, s1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31)
				acc += kvx2[t] * HW2(s01)
			}
			res := simd.shr_masked(acc, shift0_2)
			return simd.shuffle(res, res, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15),
				simd.shuffle(res, res, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31)
		} else {
			_ = kvx2
			return hrow6(src, at0, kvx, half0, shift0), hrow6(src, at1, kvx, half0, shift0)
		}
	}
	hrow4_pair :: #force_inline proc "contextless" (src: [^]T, at0, at1: int, kvx: ^[8]HW, kvx2: ^[8]HW2, half0: HW, shift0: HU) -> (I16, I16) #no_bounds_check {
		when T == u8 && N == 16 {
			_ = kvx
			_ = half0
			_ = shift0
			acc := half0_2
			#unroll for t in 2 ..< 6 {
				s0 := intrinsics.unaligned_load((^S)(&src[at0 + t]))
				s1 := intrinsics.unaligned_load((^S)(&src[at1 + t]))
				s01 := simd.shuffle(s0, s1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31)
				acc += kvx2[t] * HW2(s01)
			}
			res := simd.shr_masked(acc, shift0_2)
			return simd.shuffle(res, res, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15),
				simd.shuffle(res, res, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31)
		} else {
			_ = kvx2
			return hrow4(src, at0, kvx, half0, shift0), hrow4(src, at1, kvx, half0, shift0)
		}
	}
	if four_tap_y {
		if four_tap_x {
			for c := 0; c < w; c += N {
				m2, m3 := hrow4_pair(src, 2 * src_stride + c, 3 * src_stride + c, &kvx, &kvx2, half0, shift0)
				m4 := hrow4(src, 4 * src_stride + c, &kvx, half0, shift0)
				p23 := pack_i16_pairs(m2, m3)
				p34 := pack_i16_pairs(m3, m4)
				for r := 0; r < h; r += 2 {
					at5 := (r + 5) * src_stride + c
					m5, m6 := hrow4_pair(src, at5, at5 + src_stride, &kvx, &kvx2, half0, shift0)
					p45 := pack_i16_pairs(m4, m5)
					acc0 := half1 + madd_i16_pairs(p23, kvy23) + madd_i16_pairs(p45, kvy45)
					out0 := simd.clamp(simd.shr_masked(acc0, shift1), zero, hi)
					intrinsics.unaligned_store((^S)(&dst[r * dst_stride + c]), S(out0))
					p56 := pack_i16_pairs(m5, m6)
					acc1 := half1 + madd_i16_pairs(p34, kvy23) + madd_i16_pairs(p56, kvy45)
					out1 := simd.clamp(simd.shr_masked(acc1, shift1), zero, hi)
					intrinsics.unaligned_store((^S)(&dst[(r + 1) * dst_stride + c]), S(out1))
					m4 = m6
					p23, p34 = p45, p56
				}
			}
		} else if six_tap_x {
			for c := 0; c < w; c += N {
				m2, m3 := hrow6_pair(src, 2 * src_stride + c, 3 * src_stride + c, &kvx, &kvx2, half0, shift0)
				m4 := hrow6(src, 4 * src_stride + c, &kvx, half0, shift0)
				p23 := pack_i16_pairs(m2, m3)
				p34 := pack_i16_pairs(m3, m4)
				for r := 0; r < h; r += 2 {
					at5 := (r + 5) * src_stride + c
					m5, m6 := hrow6_pair(src, at5, at5 + src_stride, &kvx, &kvx2, half0, shift0)
					p45 := pack_i16_pairs(m4, m5)
					acc0 := half1 + madd_i16_pairs(p23, kvy23) + madd_i16_pairs(p45, kvy45)
					out0 := simd.clamp(simd.shr_masked(acc0, shift1), zero, hi)
					intrinsics.unaligned_store((^S)(&dst[r * dst_stride + c]), S(out0))
					p56 := pack_i16_pairs(m5, m6)
					acc1 := half1 + madd_i16_pairs(p34, kvy23) + madd_i16_pairs(p56, kvy45)
					out1 := simd.clamp(simd.shr_masked(acc1, shift1), zero, hi)
					intrinsics.unaligned_store((^S)(&dst[(r + 1) * dst_stride + c]), S(out1))
					m4 = m6
					p23, p34 = p45, p56
				}
			}
		} else {
			for c := 0; c < w; c += N {
				m2, m3 := hrow8_pair(src, 2 * src_stride + c, 3 * src_stride + c, &kvx, &kvx2, half0, shift0)
				m4 := hrow8(src, 4 * src_stride + c, &kvx, half0, shift0)
				p23 := pack_i16_pairs(m2, m3)
				p34 := pack_i16_pairs(m3, m4)
				for r := 0; r < h; r += 2 {
					at5 := (r + 5) * src_stride + c
					m5, m6 := hrow8_pair(src, at5, at5 + src_stride, &kvx, &kvx2, half0, shift0)
					p45 := pack_i16_pairs(m4, m5)
					acc0 := half1 + madd_i16_pairs(p23, kvy23) + madd_i16_pairs(p45, kvy45)
					out0 := simd.clamp(simd.shr_masked(acc0, shift1), zero, hi)
					intrinsics.unaligned_store((^S)(&dst[r * dst_stride + c]), S(out0))
					p56 := pack_i16_pairs(m5, m6)
					acc1 := half1 + madd_i16_pairs(p34, kvy23) + madd_i16_pairs(p56, kvy45)
					out1 := simd.clamp(simd.shr_masked(acc1, shift1), zero, hi)
					intrinsics.unaligned_store((^S)(&dst[(r + 1) * dst_stride + c]), S(out1))
					m4 = m6
					p23, p34 = p45, p56
				}
			}
		}
		return
	}

	if six_tap_y {
		if four_tap_x {
			for c := 0; c < w; c += N {
				m1, m2 := hrow4_pair(src, 1 * src_stride + c, 2 * src_stride + c, &kvx, &kvx2, half0, shift0)
				m3, m4 := hrow4_pair(src, 3 * src_stride + c, 4 * src_stride + c, &kvx, &kvx2, half0, shift0)
				m5 := hrow4(src, 5 * src_stride + c, &kvx, half0, shift0)
				p12 := pack_i16_pairs(m1, m2)
				p23 := pack_i16_pairs(m2, m3)
				p34 := pack_i16_pairs(m3, m4)
				p45 := pack_i16_pairs(m4, m5)
				for r := 0; r < h; r += 2 {
					at6 := (r + 6) * src_stride + c
					m6, m7 := hrow4_pair(src, at6, at6 + src_stride, &kvx, &kvx2, half0, shift0)
					p56 := pack_i16_pairs(m5, m6)
					acc0 := half1 + madd_i16_pairs(p12, kvy12) + madd_i16_pairs(p34, kvy34) + madd_i16_pairs(p56, kvy56)
					out0 := simd.clamp(simd.shr_masked(acc0, shift1), zero, hi)
					intrinsics.unaligned_store((^S)(&dst[r * dst_stride + c]), S(out0))
					p67 := pack_i16_pairs(m6, m7)
					acc1 := half1 + madd_i16_pairs(p23, kvy12) + madd_i16_pairs(p45, kvy34) + madd_i16_pairs(p67, kvy56)
					out1 := simd.clamp(simd.shr_masked(acc1, shift1), zero, hi)
					intrinsics.unaligned_store((^S)(&dst[(r + 1) * dst_stride + c]), S(out1))
					m5 = m7
					p12, p34 = p34, p56
					p23, p45 = p45, p67
				}
			}
		} else if six_tap_x {
			for c := 0; c < w; c += N {
				m1, m2 := hrow6_pair(src, 1 * src_stride + c, 2 * src_stride + c, &kvx, &kvx2, half0, shift0)
				m3, m4 := hrow6_pair(src, 3 * src_stride + c, 4 * src_stride + c, &kvx, &kvx2, half0, shift0)
				m5 := hrow6(src, 5 * src_stride + c, &kvx, half0, shift0)
				p12 := pack_i16_pairs(m1, m2)
				p23 := pack_i16_pairs(m2, m3)
				p34 := pack_i16_pairs(m3, m4)
				p45 := pack_i16_pairs(m4, m5)
				for r := 0; r < h; r += 2 {
					at6 := (r + 6) * src_stride + c
					m6, m7 := hrow6_pair(src, at6, at6 + src_stride, &kvx, &kvx2, half0, shift0)
					p56 := pack_i16_pairs(m5, m6)
					acc0 := half1 + madd_i16_pairs(p12, kvy12) + madd_i16_pairs(p34, kvy34) + madd_i16_pairs(p56, kvy56)
					out0 := simd.clamp(simd.shr_masked(acc0, shift1), zero, hi)
					intrinsics.unaligned_store((^S)(&dst[r * dst_stride + c]), S(out0))
					p67 := pack_i16_pairs(m6, m7)
					acc1 := half1 + madd_i16_pairs(p23, kvy12) + madd_i16_pairs(p45, kvy34) + madd_i16_pairs(p67, kvy56)
					out1 := simd.clamp(simd.shr_masked(acc1, shift1), zero, hi)
					intrinsics.unaligned_store((^S)(&dst[(r + 1) * dst_stride + c]), S(out1))
					m5 = m7
					p12, p34 = p34, p56
					p23, p45 = p45, p67
				}
			}
		} else {
			for c := 0; c < w; c += N {
				m1, m2 := hrow8_pair(src, 1 * src_stride + c, 2 * src_stride + c, &kvx, &kvx2, half0, shift0)
				m3, m4 := hrow8_pair(src, 3 * src_stride + c, 4 * src_stride + c, &kvx, &kvx2, half0, shift0)
				m5 := hrow8(src, 5 * src_stride + c, &kvx, half0, shift0)
				p12 := pack_i16_pairs(m1, m2)
				p23 := pack_i16_pairs(m2, m3)
				p34 := pack_i16_pairs(m3, m4)
				p45 := pack_i16_pairs(m4, m5)
				for r := 0; r < h; r += 2 {
					at6 := (r + 6) * src_stride + c
					m6, m7 := hrow8_pair(src, at6, at6 + src_stride, &kvx, &kvx2, half0, shift0)
					p56 := pack_i16_pairs(m5, m6)
					acc0 := half1 + madd_i16_pairs(p12, kvy12) + madd_i16_pairs(p34, kvy34) + madd_i16_pairs(p56, kvy56)
					out0 := simd.clamp(simd.shr_masked(acc0, shift1), zero, hi)
					intrinsics.unaligned_store((^S)(&dst[r * dst_stride + c]), S(out0))
					p67 := pack_i16_pairs(m6, m7)
					acc1 := half1 + madd_i16_pairs(p23, kvy12) + madd_i16_pairs(p45, kvy34) + madd_i16_pairs(p67, kvy56)
					out1 := simd.clamp(simd.shr_masked(acc1, shift1), zero, hi)
					intrinsics.unaligned_store((^S)(&dst[(r + 1) * dst_stride + c]), S(out1))
					m5 = m7
					p12, p34 = p34, p56
					p23, p45 = p45, p67
				}
			}
		}
		return
	}

	if four_tap_x {
		for c := 0; c < w; c += N {
			m0, m1 := hrow4_pair(src, 0 * src_stride + c, 1 * src_stride + c, &kvx, &kvx2, half0, shift0)
			m2, m3 := hrow4_pair(src, 2 * src_stride + c, 3 * src_stride + c, &kvx, &kvx2, half0, shift0)
			m4, m5 := hrow4_pair(src, 4 * src_stride + c, 5 * src_stride + c, &kvx, &kvx2, half0, shift0)
			m6 := hrow4(src, 6 * src_stride + c, &kvx, half0, shift0)
			p01 := pack_i16_pairs(m0, m1)
			p12 := pack_i16_pairs(m1, m2)
			p23 := pack_i16_pairs(m2, m3)
			p34 := pack_i16_pairs(m3, m4)
			p45 := pack_i16_pairs(m4, m5)
			p56 := pack_i16_pairs(m5, m6)
			for r := 0; r < h; r += 2 {
				at7 := (r + 7) * src_stride + c
				m7, m8 := hrow4_pair(src, at7, at7 + src_stride, &kvx, &kvx2, half0, shift0)
				p67 := pack_i16_pairs(m6, m7)
				acc0 := half1 + madd_i16_pairs(p01, kvy01) + madd_i16_pairs(p23, kvy23) +
					madd_i16_pairs(p45, kvy45) + madd_i16_pairs(p67, kvy67)
				out0 := simd.clamp(simd.shr_masked(acc0, shift1), zero, hi)
				intrinsics.unaligned_store((^S)(&dst[r * dst_stride + c]), S(out0))
				p78 := pack_i16_pairs(m7, m8)
				acc1 := half1 + madd_i16_pairs(p12, kvy01) + madd_i16_pairs(p34, kvy23) +
					madd_i16_pairs(p56, kvy45) + madd_i16_pairs(p78, kvy67)
				out1 := simd.clamp(simd.shr_masked(acc1, shift1), zero, hi)
				intrinsics.unaligned_store((^S)(&dst[(r + 1) * dst_stride + c]), S(out1))
				m6 = m8
				p01, p23, p45 = p23, p45, p67
				p12, p34, p56 = p34, p56, p78
			}
		}
		return
	}

	if six_tap_x {
		for c := 0; c < w; c += N {
			m0, m1 := hrow6_pair(src, 0 * src_stride + c, 1 * src_stride + c, &kvx, &kvx2, half0, shift0)
			m2, m3 := hrow6_pair(src, 2 * src_stride + c, 3 * src_stride + c, &kvx, &kvx2, half0, shift0)
			m4, m5 := hrow6_pair(src, 4 * src_stride + c, 5 * src_stride + c, &kvx, &kvx2, half0, shift0)
			m6 := hrow6(src, 6 * src_stride + c, &kvx, half0, shift0)
			p01 := pack_i16_pairs(m0, m1)
			p12 := pack_i16_pairs(m1, m2)
			p23 := pack_i16_pairs(m2, m3)
			p34 := pack_i16_pairs(m3, m4)
			p45 := pack_i16_pairs(m4, m5)
			p56 := pack_i16_pairs(m5, m6)
			for r := 0; r < h; r += 2 {
				at7 := (r + 7) * src_stride + c
				m7, m8 := hrow6_pair(src, at7, at7 + src_stride, &kvx, &kvx2, half0, shift0)
				p67 := pack_i16_pairs(m6, m7)
				acc0 := half1 + madd_i16_pairs(p01, kvy01) + madd_i16_pairs(p23, kvy23) +
					madd_i16_pairs(p45, kvy45) + madd_i16_pairs(p67, kvy67)
				out0 := simd.clamp(simd.shr_masked(acc0, shift1), zero, hi)
				intrinsics.unaligned_store((^S)(&dst[r * dst_stride + c]), S(out0))
				p78 := pack_i16_pairs(m7, m8)
				acc1 := half1 + madd_i16_pairs(p12, kvy01) + madd_i16_pairs(p34, kvy23) +
					madd_i16_pairs(p56, kvy45) + madd_i16_pairs(p78, kvy67)
				out1 := simd.clamp(simd.shr_masked(acc1, shift1), zero, hi)
				intrinsics.unaligned_store((^S)(&dst[(r + 1) * dst_stride + c]), S(out1))
				m6 = m8
				p01, p23, p45 = p23, p45, p67
				p12, p34, p56 = p34, p56, p78
			}
		}
		return
	}

	for c := 0; c < w; c += N {
		m0, m1 := hrow8_pair(src, 0 * src_stride + c, 1 * src_stride + c, &kvx, &kvx2, half0, shift0)
		m2, m3 := hrow8_pair(src, 2 * src_stride + c, 3 * src_stride + c, &kvx, &kvx2, half0, shift0)
		m4, m5 := hrow8_pair(src, 4 * src_stride + c, 5 * src_stride + c, &kvx, &kvx2, half0, shift0)
		m6 := hrow8(src, 6 * src_stride + c, &kvx, half0, shift0)
		p01 := pack_i16_pairs(m0, m1)
		p12 := pack_i16_pairs(m1, m2)
		p23 := pack_i16_pairs(m2, m3)
		p34 := pack_i16_pairs(m3, m4)
		p45 := pack_i16_pairs(m4, m5)
		p56 := pack_i16_pairs(m5, m6)
		for r := 0; r < h; r += 2 {
			at7 := (r + 7) * src_stride + c
			m7, m8 := hrow8_pair(src, at7, at7 + src_stride, &kvx, &kvx2, half0, shift0)
			p67 := pack_i16_pairs(m6, m7)
			acc0 := half1 + madd_i16_pairs(p01, kvy01) + madd_i16_pairs(p23, kvy23) +
				madd_i16_pairs(p45, kvy45) + madd_i16_pairs(p67, kvy67)
			out0 := simd.clamp(simd.shr_masked(acc0, shift1), zero, hi)
			intrinsics.unaligned_store((^S)(&dst[r * dst_stride + c]), S(out0))
			p78 := pack_i16_pairs(m7, m8)
			acc1 := half1 + madd_i16_pairs(p12, kvy01) + madd_i16_pairs(p34, kvy23) +
				madd_i16_pairs(p56, kvy45) + madd_i16_pairs(p78, kvy67)
			out1 := simd.clamp(simd.shr_masked(acc1, shift1), zero, hi)
			intrinsics.unaligned_store((^S)(&dst[(r + 1) * dst_stride + c]), S(out1))
			m6 = m8
			p01, p23, p45 = p23, p45, p67
			p12, p34, p56 = p34, p56, p78
		}
	}
}

// Direct single-reference inter prediction into `dst[y * dst_stride + x:]`
// when `!is_compound && !is_interintra`.
predict_inter_single :: proc(
	dst: []$T,
	dst_stride: int,
	x, y, w, h: int,
	ref: ^Ref_Plane(T),
	start_x, start_y, step_x, step_y: int,
	filter_x, filter_y: int,
	bit_depth := 8,
) #no_bounds_check {
	max_val := (i32(1) << uint(bit_depth)) - 1
	if step_x == (1 << SCALE_SUBPEL_BITS) && step_y == (1 << SCALE_SUBPEL_BITS) {
		r0, r1, _ := inter_rounding_vars(false, bit_depth)
		fx_kind := effective_interp_filter(filter_x, w)
		fy_kind := effective_interp_filter(filter_y, h)
		kx_table := &SUBPEL_FILTERS[fx_kind]
		ky_table := &SUBPEL_FILTERS[fy_kind]
		fx := (start_x >> 6) & SUBPEL_MASK
		fy := (start_y >> 6) & SUBPEL_MASK
		x0 := (start_x >> SCALE_SUBPEL_BITS) - 3
		y0 := (start_y >> SCALE_SUBPEL_BITS) - 3
		cols := w + 7
		rows := h + 7
		dst_ptr := raw_data(dst[y * dst_stride + x:])
		if x0 < 0 || y0 < 0 || x0 + cols - 1 > ref.last_x || y0 + rows - 1 > ref.last_y {
			predict_unscaled_edge_single(dst_ptr, dst_stride, ref, x0, y0, cols, rows, w, h, fx, fy, &kx_table[fx], &ky_table[fy], r0, r1, max_val)
			return
		}
		src := raw_data(ref.data)
		src_stride := ref.stride
		origin := y0 * src_stride + x0
		predict_unscaled_dispatch_single(dst_ptr, dst_stride, src[origin:], src_stride, w, h, fx, fy, &kx_table[fx], &ky_table[fy], r0, r1, max_val)
		return
	}

	predict_scaled_single(dst, dst_stride, x, y, w, h, ref, start_x, start_y, step_x, step_y, filter_x, filter_y, bit_depth, max_val)
}

@(private)
predict_scaled_single :: #force_no_inline proc(
	dst: []$T,
	dst_stride: int,
	x, y, w, h: int,
	ref: ^Ref_Plane(T),
	start_x, start_y, step_x, step_y: int,
	filter_x, filter_y: int,
	bit_depth: int,
	max_val: i32,
) #no_bounds_check {
	pred: [128 * 128]i32 = ---
	convolve_2d_pred(pred[:], w, ref, start_x, start_y, step_x, step_y, w, h, filter_x, filter_y, false, bit_depth)
	for r in 0 ..< h {
		out_row := dst[(y + r) * dst_stride + x:]
		for c in 0 ..< w {
			out_row[c] = T(clamp(pred[r * w + c], 0, max_val))
		}
	}
}

// ---- Warped Motion (7.11.3.5 - 7.11.3.8) ------------------------------------

// 7.11.3.7: Approximate reciprocal lookup `div_factor / (1 << div_shift) ~= 1 / d`.
resolve_divisor :: proc "contextless" (d: i64) -> (div_shift: uint, div_factor: i64) #no_bounds_check {
	ad := abs(d)
	if ad == 0 do return 0, 0
	n := uint(63 - intrinsics.count_leading_zeros(u64(ad)))
	e := ad - (i64(1) << n)
	f: i64
	if n > DIV_LUT_BITS {
		f = i64(round2(e, n - DIV_LUT_BITS))
	} else {
		f = e << (DIV_LUT_BITS - n)
	}
	div_shift = n + DIV_LUT_PREC_BITS
	lut := i64(DIV_LUT[clamp(int(f), 0, DIV_LUT_NUM - 1)])
	div_factor = d < 0 ? -lut : lut
	return
}

// 7.11.3.6: Derives shear parameters (alpha, beta, gamma, delta) and validity
// from `warp_params` = [0..5].
setup_shear :: proc "contextless" (warp_params: [6]i32) -> (valid: bool, alpha, beta, gamma, delta: i32) {
	if warp_params[2] == 0 do return false, 0, 0, 0, 0
	alpha0 := clamp(warp_params[2] - (1 << WARPEDMODEL_PREC_BITS), -32768, 32767)
	beta0 := clamp(warp_params[3], -32768, 32767)
	div_shift, div_factor := resolve_divisor(i64(warp_params[2]))
	v := i64(warp_params[4]) << WARPEDMODEL_PREC_BITS
	gamma0 := clamp(round2_signed(v * div_factor, div_shift), -32768, 32767)
	w := i64(warp_params[3]) * i64(warp_params[4])
	delta0 := clamp(warp_params[5] - round2_signed(w * div_factor, div_shift) - (1 << WARPEDMODEL_PREC_BITS), -32768, 32767)

	alpha = round2_signed(i64(alpha0), WARP_PARAM_REDUCE_BITS) << WARP_PARAM_REDUCE_BITS
	beta = round2_signed(i64(beta0), WARP_PARAM_REDUCE_BITS) << WARP_PARAM_REDUCE_BITS
	gamma = round2_signed(i64(gamma0), WARP_PARAM_REDUCE_BITS) << WARP_PARAM_REDUCE_BITS
	delta = round2_signed(i64(delta0), WARP_PARAM_REDUCE_BITS) << WARP_PARAM_REDUCE_BITS

	if 4 * abs(int(alpha)) + 7 * abs(int(beta)) >= (1 << WARPEDMODEL_PREC_BITS) do return false, alpha, beta, gamma, delta
	if 4 * abs(int(gamma)) + 4 * abs(int(delta)) >= (1 << WARPEDMODEL_PREC_BITS) do return false, alpha, beta, gamma, delta
	return true, alpha, beta, gamma, delta
}

// 7.11.3.8: Least-squares local warp estimation from `cand_list` samples.
estimate_local_warp :: #force_no_inline proc "contextless" (mi_size, mi_row, mi_col: int, mv: [2]i16, cand_list: [][4]i32) -> (valid: bool, params: [6]i32) #no_bounds_check {
	a00, a01, a11: i64
	bx0, bx1, by0, by1: i64
	w4 := int(NUM_4X4_BLOCKS_WIDE[mi_size])
	h4 := int(NUM_4X4_BLOCKS_HIGH[mi_size])
	mid_y := i64(mi_row * 4 + h4 * 2 - 1)
	mid_x := i64(mi_col * 4 + w4 * 2 - 1)
	suy := mid_y * 8
	sux := mid_x * 8
	duy := suy + i64(mv[0])
	dux := sux + i64(mv[1])

	ls_product :: #force_inline proc "contextless" (a, b: i64) -> i64 {
		return ((a * b) >> 2) + (a + b)
	}

	for cand in cand_list {
		sy := i64(cand[0]) - suy
		sx := i64(cand[1]) - sux
		dy := i64(cand[2]) - duy
		dx := i64(cand[3]) - dux
		if abs(sx - dx) < LS_MV_MAX && abs(sy - dy) < LS_MV_MAX {
			a00 += ls_product(sx, sx) + 8
			a01 += ls_product(sx, sy) + 4
			a11 += ls_product(sy, sy) + 8
			bx0 += ls_product(sx, dx) + 8
			bx1 += ls_product(sy, dx) + 4
			by0 += ls_product(sx, dy) + 4
			by1 += ls_product(sy, dy) + 8
		}
	}

	det := a00 * a11 - a01 * a01
	if det == 0 do return false, params

	div_shift_u, div_factor := resolve_divisor(det)
	div_shift := int(div_shift_u) - WARPEDMODEL_PREC_BITS
	if div_shift < 0 {
		div_factor <<= uint(-div_shift)
		div_shift = 0
	}
	ds := uint(div_shift)

	nondiag :: #force_inline proc "contextless" (v, df: i64, s: uint) -> i32 {
		return clamp(round2_signed(v * df, s), -WARPEDMODEL_NONDIAGAFFINE_CLAMP + 1, WARPEDMODEL_NONDIAGAFFINE_CLAMP - 1)
	}
	diag :: #force_inline proc "contextless" (v, df: i64, s: uint) -> i32 {
		return clamp(round2_signed(v * df, s), (1 << WARPEDMODEL_PREC_BITS) - WARPEDMODEL_NONDIAGAFFINE_CLAMP + 1, (1 << WARPEDMODEL_PREC_BITS) + WARPEDMODEL_NONDIAGAFFINE_CLAMP - 1)
	}

	params[2] = diag(a11 * bx0 - a01 * bx1, div_factor, ds)
	params[3] = nondiag(-a01 * bx0 + a00 * bx1, div_factor, ds)
	params[4] = nondiag(a11 * by0 - a01 * by1, div_factor, ds)
	params[5] = diag(-a01 * by0 + a00 * by1, div_factor, ds)

	vx := i64(mv[1]) * (1 << (WARPEDMODEL_PREC_BITS - 3)) - (mid_x * i64(params[2] - (1 << WARPEDMODEL_PREC_BITS)) + mid_y * i64(params[3]))
	vy := i64(mv[0]) * (1 << (WARPEDMODEL_PREC_BITS - 3)) - (mid_x * i64(params[4]) + mid_y * i64(params[5] - (1 << WARPEDMODEL_PREC_BITS)))
	params[0] = i32(clamp(vx, -WARPEDMODEL_TRANS_CLAMP, WARPEDMODEL_TRANS_CLAMP - 1))
	params[1] = i32(clamp(vy, -WARPEDMODEL_TRANS_CLAMP, WARPEDMODEL_TRANS_CLAMP - 1))
	return true, params
}

// 7.11.3.5: Warps a `w x h` block at `(x, y)` in `plane` (in 8x8 sub-blocks)
// into `pred` (stride `pred_stride`).
warp_plane_pred :: #force_no_inline proc(
	pred: []i32,
	pred_stride: int,
	ref: ^Ref_Plane($T),
	x, y, w, h: int,
	sub_x, sub_y: uint,
	warp_params: [6]i32,
	is_compound: bool,
	bit_depth := 8,
) #no_bounds_check {
	valid, alpha, beta, gamma, delta := setup_shear(warp_params)
	if !valid do return
	r0, r1, _ := inter_rounding_vars(is_compound, bit_depth)

	h8 := (h + 7) >> 3
	w8 := (w + 7) >> 3
	for i8 in 0 ..< h8 {
		for j8 in 0 ..< w8 {
			src_x := i64((x + j8 * 8 + 4) << sub_x)
			src_y := i64((y + i8 * 8 + 4) << sub_y)
			dst_x := i64(warp_params[2]) * src_x + i64(warp_params[3]) * src_y + i64(warp_params[0])
			dst_y := i64(warp_params[4]) * src_x + i64(warp_params[5]) * src_y + i64(warp_params[1])

			x4 := dst_x >> sub_x
			y4 := dst_y >> sub_y
			ix4 := int(x4 >> WARPEDMODEL_PREC_BITS)
			sx4 := int(x4 & ((1 << WARPEDMODEL_PREC_BITS) - 1))
			iy4 := int(y4 >> WARPEDMODEL_PREC_BITS)
			sy4 := int(y4 & ((1 << WARPEDMODEL_PREC_BITS) - 1))

			intermediate: [15][8]i32 = ---
			for i1 := -7; i1 < 8; i1 += 1 {
				ry := clamp(iy4 + i1, 0, ref.last_y)
				line := ref.data[ry * ref.stride:]
				for i2 := -4; i2 < 4; i2 += 1 {
					sx := sx4 + int(alpha) * i2 + int(beta) * i1
					offs := clamp(int(round2(i64(sx), WARPEDDIFF_PREC_BITS)) + WARPEDPIXEL_PREC_SHIFTS, 0, len(WARPED_FILTERS) - 1)
					wf := WARPED_FILTERS[offs]
					s := i32(0)
					#unroll for i3 in 0 ..< 8 {
						cx := clamp(ix4 + i2 - 3 + i3, 0, ref.last_x)
						s += i32(wf[i3]) * i32(line[cx])
					}
					intermediate[i1 + 7][i2 + 4] = round2(i64(s), r0)
				}
			}

			i1_end := min(4, h - i8 * 8 - 4)
			i2_end := min(4, w - j8 * 8 - 4)
			for i1 := -4; i1 < i1_end; i1 += 1 {
				out_row := pred[(i8 * 8 + i1 + 4) * pred_stride:]
				for i2 := -4; i2 < i2_end; i2 += 1 {
					sy := sy4 + int(gamma) * i2 + int(delta) * i1
					offs := clamp(int(round2(i64(sy), WARPEDDIFF_PREC_BITS)) + WARPEDPIXEL_PREC_SHIFTS, 0, len(WARPED_FILTERS) - 1)
					wf := WARPED_FILTERS[offs]
					s := i32(0)
					#unroll for i3 in 0 ..< 8 {
						s += i32(wf[i3]) * intermediate[i1 + i3 + 4][i2 + 4]
					}
					out_row[j8 * 8 + i2 + 4] = round2(i64(s), r1)
				}
			}
		}
	}
}

// ---- Compound Masks & Blending (7.11.3.10 - 7.11.3.15) ----------------------

// 7.11.3.11: Evaluates `MasterMask[dir][y][x]` for `dir in 0..5` and `x, y in 0..63`.
@(private)
wedge_master_mask :: proc "contextless" (dir, y, x: int) -> u8 #no_bounds_check {
	switch dir {
	case WEDGE_VERTICAL:
		return WEDGE_MASTER_VERTICAL[x]
	case WEDGE_HORIZONTAL:
		return WEDGE_MASTER_VERTICAL[y]
	case WEDGE_OBLIQUE63:
		shift := MASK_MASTER_SIZE / 4 - (y >> 1)
		if (y & 1) == 0 {
			return WEDGE_MASTER_OBLIQUE_EVEN[clamp(x - shift, 0, MASK_MASTER_SIZE - 1)]
		}
		return WEDGE_MASTER_OBLIQUE_ODD[clamp(x - shift + 1, 0, MASK_MASTER_SIZE - 1)]
	case WEDGE_OBLIQUE27:
		return wedge_master_mask(WEDGE_OBLIQUE63, x, y)
	case WEDGE_OBLIQUE117:
		return 64 - wedge_master_mask(WEDGE_OBLIQUE63, y, MASK_MASTER_SIZE - 1 - x)
	case: // WEDGE_OBLIQUE153
		return 64 - wedge_master_mask(WEDGE_OBLIQUE63, x, MASK_MASTER_SIZE - 1 - y)
	}
}

// 7.11.3.11: Builds the `w x h` luma wedge mask (`mask` stride `mask_stride`).
build_wedge_mask :: proc "contextless" (mask: []u8, mask_stride: int, bsize, wedge_sign, wedge_index: int) #no_bounds_check {
	w4 := int(NUM_4X4_BLOCKS_WIDE[bsize])
	h4 := int(NUM_4X4_BLOCKS_HIGH[bsize])
	w := w4 * 4
	h := h4 * 4
	shape := h4 > w4 ? 0 : (h4 < w4 ? 1 : 2)
	entry := WEDGE_CODEBOOK[shape][wedge_index]
	dir := int(entry[0])
	xoff := MASK_MASTER_SIZE / 2 - ((int(entry[1]) * w) >> 3)
	yoff := MASK_MASTER_SIZE / 2 - ((int(entry[2]) * h) >> 3)

	sum := 0
	for i in 0 ..< w do sum += int(wedge_master_mask(dir, yoff, xoff + i))
	for i in 1 ..< h do sum += int(wedge_master_mask(dir, yoff + i, xoff))
	avg := (sum + (w + h - 1) / 2) / (w + h - 1)
	flip_sign := int(avg < 32)
	invert := wedge_sign != flip_sign

	for i in 0 ..< h {
		row := mask[i * mask_stride:]
		for j in 0 ..< w {
			m := wedge_master_mask(dir, yoff + i, xoff + j)
			row[j] = invert ? (64 - m) : m
		}
	}
}

// 7.11.3.12: Builds the `w x h` difference-weighted mask (`mask` stride `mask_stride`).
build_diffwtd_mask :: proc "contextless" (
	mask: []u8,
	mask_stride: int,
	pred0, pred1: []i32,
	pred_stride: int,
	w, h, mask_type: int,
	bit_depth := 8,
) #no_bounds_check {
	V4 :: #simd[4]i32
	U4 :: #simd[4]u32
	B4 :: #simd[4]u8
	_, _, post := inter_rounding_vars(true, bit_depth)
	shift := (uint(bit_depth - 8) + post) & 15
	half := V4(i32(1) << (shift - 1))
	sv := U4(u32(shift))
	c38 := V4(38)
	c64 := V4(64)
	zero := V4(0)
	u4 := U4(4)
	for i in 0 ..< h {
		mrow := raw_data(mask[i * mask_stride:])
		p0 := raw_data(pred0[i * pred_stride:])
		p1 := raw_data(pred1[i * pred_stride:])
		for j := 0; j < w; j += 4 {
			v0 := intrinsics.unaligned_load((^V4)(&p0[j]))
			v1 := intrinsics.unaligned_load((^V4)(&p1[j]))
			diff := simd.shr_masked(simd.abs(v0 - v1) + half, sv)
			m := simd.clamp(c38 + simd.shr_masked(diff, u4), zero, c64)
			res := mask_type != 0 ? (c64 - m) : m
			intrinsics.unaligned_store((^B4)(&mrow[j]), B4(res))
		}
	}
}

// 7.11.3.13: Builds the `w x h` inter-intra mode variant mask (`mask` stride `mask_stride`).
build_interintra_mask :: proc "contextless" (mask: []u8, mask_stride: int, w, h, interintra_mode: int) #no_bounds_check {
	size_scale := MAX_SB_SIZE / max(h, w)
	for i in 0 ..< h {
		mrow := mask[i * mask_stride:]
		for j in 0 ..< w {
			switch interintra_mode {
			case II_V_PRED:
				mrow[j] = II_WEIGHTS_1D[i * size_scale]
			case II_H_PRED:
				mrow[j] = II_WEIGHTS_1D[j * size_scale]
			case II_SMOOTH_PRED:
				mrow[j] = II_WEIGHTS_1D[min(i, j) * size_scale]
			case:
				mrow[j] = 32
			}
		}
	}
}

// 7.11.3.15: Derives (fwd_weight, bck_weight) from relative frame distances
// `dist0 = abs(rel_dist(ref0, cur))` and `dist1 = abs(rel_dist(ref1, cur))`.
distance_weights :: proc "contextless" (dist0, dist1: int) -> (fwd_weight, bck_weight: i32) #no_bounds_check {
	d0 := clamp(dist1, 0, MAX_FRAME_DISTANCE)
	d1 := clamp(dist0, 0, MAX_FRAME_DISTANCE)
	order := int(d0 <= d1)
	if d0 == 0 || d1 == 0 {
		return i32(QUANT_DIST_LOOKUP[3][order]), i32(QUANT_DIST_LOOKUP[3][1 - order])
	}
	idx := 0
	for idx < 3 {
		c0 := int(QUANT_DIST_WEIGHT[idx][order])
		c1 := int(QUANT_DIST_WEIGHT[idx][1 - order])
		if order != 0 {
			if d0 * c0 > d1 * c1 do break
		} else {
			if d0 * c0 < d1 * c1 do break
		}
		idx += 1
	}
	return i32(QUANT_DIST_LOOKUP[idx][order]), i32(QUANT_DIST_LOOKUP[idx][1 - order])
}

// 7.11.3.1: Blends two compound predictions (`pred0`, `pred1`) using `COMPOUND_AVERAGE`.
blend_compound_average :: proc "contextless" (
	dst: []$T,
	dst_stride, x, y, w, h: int,
	pred0, pred1: []i32,
	pred_stride: int,
	bit_depth := 8,
) #no_bounds_check {
	_, _, post := inter_rounding_vars(true, bit_depth)
	shift := (uint(1 + post)) & 15
	max_val := (i32(1) << uint(bit_depth)) - 1

	when intrinsics.has_target_feature("avx512f") {
		if w >= 16 {
			V16 :: #simd[16]i32
			VT16 :: #simd[16]T
			half := V16(i32(1) << (shift - 1))
			sv := (#simd[16]u32)(shift)
			zero := V16(0)
			hi := V16(max_val)
			for r in 0 ..< h {
				out_row := raw_data(dst[(y + r) * dst_stride + x:])
				p0 := raw_data(pred0[r * pred_stride:])
				p1 := raw_data(pred1[r * pred_stride:])
				for c := 0; c < w; c += 16 {
					v0 := intrinsics.unaligned_load((^V16)(&p0[c]))
					v1 := intrinsics.unaligned_load((^V16)(&p1[c]))
					res := simd.clamp(simd.shr_masked(v0 + v1 + half, sv), zero, hi)
					intrinsics.unaligned_store((^VT16)(&out_row[c]), VT16(res))
				}
			}
			return
		}
	}

	if w >= 8 {
		V8 :: #simd[8]i32
		VT8 :: #simd[8]T
		half := V8(i32(1) << (shift - 1))
		sv := (#simd[8]u32)(shift)
		zero := V8(0)
		hi := V8(max_val)
		for r in 0 ..< h {
			out_row := raw_data(dst[(y + r) * dst_stride + x:])
			p0 := raw_data(pred0[r * pred_stride:])
			p1 := raw_data(pred1[r * pred_stride:])
			for c := 0; c < w; c += 8 {
				v0 := intrinsics.unaligned_load((^V8)(&p0[c]))
				v1 := intrinsics.unaligned_load((^V8)(&p1[c]))
				res := simd.clamp(simd.shr_masked(v0 + v1 + half, sv), zero, hi)
				intrinsics.unaligned_store((^VT8)(&out_row[c]), VT8(res))
			}
		}
		return
	}

	V4 :: #simd[4]i32
	VT4 :: #simd[4]T
	half := V4(i32(1) << (shift - 1))
	sv := (#simd[4]u32)(shift)
	zero := V4(0)
	hi := V4(max_val)

	for r in 0 ..< h {
		out_row := raw_data(dst[(y + r) * dst_stride + x:])
		p0 := raw_data(pred0[r * pred_stride:])
		p1 := raw_data(pred1[r * pred_stride:])
		for c := 0; c < w; c += 4 {
			v0 := intrinsics.unaligned_load((^V4)(&p0[c]))
			v1 := intrinsics.unaligned_load((^V4)(&p1[c]))
			res := simd.clamp(simd.shr_masked(v0 + v1 + half, sv), zero, hi)
			intrinsics.unaligned_store((^VT4)(&out_row[c]), VT4(res))
		}
	}
}

// 7.11.3.1: Blends two compound predictions (`pred0`, `pred1`) using `COMPOUND_DISTANCE`.
blend_compound_distance :: proc "contextless" (
	dst: []$T,
	dst_stride, x, y, w, h: int,
	pred0, pred1: []i32,
	pred_stride: int,
	fwd_weight, bck_weight: i32,
	bit_depth := 8,
) #no_bounds_check {
	_, _, post := inter_rounding_vars(true, bit_depth)
	shift := (uint(4 + post)) & 15
	max_val := (i32(1) << uint(bit_depth)) - 1

	when intrinsics.has_target_feature("avx512f") {
		if w >= 16 {
			V16 :: #simd[16]i32
			VT16 :: #simd[16]T
			half := V16(i32(1) << (shift - 1))
			sv := (#simd[16]u32)(shift)
			fw := V16(fwd_weight)
			bw := V16(bck_weight)
			zero := V16(0)
			hi := V16(max_val)
			for r in 0 ..< h {
				out_row := raw_data(dst[(y + r) * dst_stride + x:])
				p0 := raw_data(pred0[r * pred_stride:])
				p1 := raw_data(pred1[r * pred_stride:])
				for c := 0; c < w; c += 16 {
					v0 := intrinsics.unaligned_load((^V16)(&p0[c]))
					v1 := intrinsics.unaligned_load((^V16)(&p1[c]))
					res := simd.clamp(simd.shr_masked(fw * v0 + bw * v1 + half, sv), zero, hi)
					intrinsics.unaligned_store((^VT16)(&out_row[c]), VT16(res))
				}
			}
			return
		}
	}

	if w >= 8 {
		V8 :: #simd[8]i32
		VT8 :: #simd[8]T
		half := V8(i32(1) << (shift - 1))
		sv := (#simd[8]u32)(shift)
		fw := V8(fwd_weight)
		bw := V8(bck_weight)
		zero := V8(0)
		hi := V8(max_val)
		for r in 0 ..< h {
			out_row := raw_data(dst[(y + r) * dst_stride + x:])
			p0 := raw_data(pred0[r * pred_stride:])
			p1 := raw_data(pred1[r * pred_stride:])
			for c := 0; c < w; c += 8 {
				v0 := intrinsics.unaligned_load((^V8)(&p0[c]))
				v1 := intrinsics.unaligned_load((^V8)(&p1[c]))
				res := simd.clamp(simd.shr_masked(fw * v0 + bw * v1 + half, sv), zero, hi)
				intrinsics.unaligned_store((^VT8)(&out_row[c]), VT8(res))
			}
		}
		return
	}

	V4 :: #simd[4]i32
	VT4 :: #simd[4]T
	half := V4(i32(1) << (shift - 1))
	sv := (#simd[4]u32)(shift)
	fw := V4(fwd_weight)
	bw := V4(bck_weight)
	zero := V4(0)
	hi := V4(max_val)

	for r in 0 ..< h {
		out_row := raw_data(dst[(y + r) * dst_stride + x:])
		p0 := raw_data(pred0[r * pred_stride:])
		p1 := raw_data(pred1[r * pred_stride:])
		for c := 0; c < w; c += 4 {
			v0 := intrinsics.unaligned_load((^V4)(&p0[c]))
			v1 := intrinsics.unaligned_load((^V4)(&p1[c]))
			res := simd.clamp(simd.shr_masked(fw * v0 + bw * v1 + half, sv), zero, hi)
			intrinsics.unaligned_store((^VT4)(&out_row[c]), VT4(res))
		}
	}
}

// 7.11.3.14: Mask blend process for compound or inter-intra predictions.
// For `interintra == true`, `dst` already holds the intra prediction `pred1`,
// and `pred0` holds the single-reference inter prediction.
blend_mask :: proc "contextless" (
	dst: []$T,
	dst_stride, dst_x, dst_y, w, h: int,
	pred0, pred1: []i32,
	pred_stride: int,
	mask: []u8,
	mask_stride: int,
	sub_x, sub_y: uint,
	interintra, wedge_interintra: bool,
	bit_depth := 8,
) #no_bounds_check {
	_, _, post := inter_rounding_vars(!interintra, bit_depth)
	max_val := (i32(1) << uint(bit_depth)) - 1
	direct_mask := (sub_x == 0 && sub_y == 0) || (interintra && !wedge_interintra)

	if !interintra && direct_mask && (w & 3) == 0 {
		sh := (uint(6 + post)) & 15
		when intrinsics.has_target_feature("avx512f") {
			if w >= 16 {
				V16 :: #simd[16]i32
				U16 :: #simd[16]u32
				B16 :: #simd[16]u8
				VT16 :: #simd[16]T
				half := V16(i32(1) << (sh - 1))
				sv := U16(u32(sh))
				c64 := V16(64)
				zero := V16(0)
				hi := V16(max_val)
				for r in 0 ..< h {
					out_row := raw_data(dst[(dst_y + r) * dst_stride + dst_x:])
					p0 := raw_data(pred0[r * pred_stride:])
					p1 := raw_data(pred1[r * pred_stride:])
					mrow := raw_data(mask[r * mask_stride:])
					for c := 0; c < w; c += 16 {
						m := V16(intrinsics.unaligned_load((^B16)(&mrow[c])))
						v0 := intrinsics.unaligned_load((^V16)(&p0[c]))
						v1 := intrinsics.unaligned_load((^V16)(&p1[c]))
						res := simd.clamp(simd.shr_masked(m * v0 + (c64 - m) * v1 + half, sv), zero, hi)
						intrinsics.unaligned_store((^VT16)(&out_row[c]), VT16(res))
					}
				}
				return
			}
		}
		if w >= 8 {
			V8 :: #simd[8]i32
			U8 :: #simd[8]u32
			B8 :: #simd[8]u8
			VT8 :: #simd[8]T
			half := V8(i32(1) << (sh - 1))
			sv := U8(u32(sh))
			c64 := V8(64)
			zero := V8(0)
			hi := V8(max_val)
			for r in 0 ..< h {
				out_row := raw_data(dst[(dst_y + r) * dst_stride + dst_x:])
				p0 := raw_data(pred0[r * pred_stride:])
				p1 := raw_data(pred1[r * pred_stride:])
				mrow := raw_data(mask[r * mask_stride:])
				for c := 0; c < w; c += 8 {
					m := V8(intrinsics.unaligned_load((^B8)(&mrow[c])))
					v0 := intrinsics.unaligned_load((^V8)(&p0[c]))
					v1 := intrinsics.unaligned_load((^V8)(&p1[c]))
					res := simd.clamp(simd.shr_masked(m * v0 + (c64 - m) * v1 + half, sv), zero, hi)
					intrinsics.unaligned_store((^VT8)(&out_row[c]), VT8(res))
				}
			}
			return
		}
		V4 :: #simd[4]i32
		U4 :: #simd[4]u32
		B4 :: #simd[4]u8
		VT4 :: #simd[4]T
		half := V4(i32(1) << (sh - 1))
		sv := U4(u32(sh))
		c64 := V4(64)
		zero := V4(0)
		hi := V4(max_val)
		for r in 0 ..< h {
			out_row := raw_data(dst[(dst_y + r) * dst_stride + dst_x:])
			p0 := raw_data(pred0[r * pred_stride:])
			p1 := raw_data(pred1[r * pred_stride:])
			mrow := raw_data(mask[r * mask_stride:])
			for c := 0; c < w; c += 4 {
				m := V4(intrinsics.unaligned_load((^B4)(&mrow[c])))
				v0 := intrinsics.unaligned_load((^V4)(&p0[c]))
				v1 := intrinsics.unaligned_load((^V4)(&p1[c]))
				res := simd.clamp(simd.shr_masked(m * v0 + (c64 - m) * v1 + half, sv), zero, hi)
				intrinsics.unaligned_store((^VT4)(&out_row[c]), VT4(res))
			}
		}
		return
	}

	sh_comp := (uint(6 + post)) & 15
	half_comp := i32(1) << (sh_comp - 1)
	sh_ii := post & 15
	half_ii := sh_ii > 0 ? (i32(1) << (sh_ii - 1)) : i32(0)

	for r in 0 ..< h {
		out_row := dst[(dst_y + r) * dst_stride + dst_x:]
		p0 := pred0[r * pred_stride:]
		for c in 0 ..< w {
			m: i32
			if direct_mask {
				m = i32(mask[r * mask_stride + c])
			} else if sub_x != 0 && sub_y == 0 {
				m = (i32(mask[r * mask_stride + 2 * c]) + i32(mask[r * mask_stride + 2 * c + 1]) + 1) >> 1
			} else if sub_x == 0 && sub_y != 0 {
				m = (i32(mask[(2 * r) * mask_stride + c]) + i32(mask[(2 * r + 1) * mask_stride + c]) + 1) >> 1
			} else {
				m = (i32(mask[(2 * r) * mask_stride + 2 * c]) +
				     i32(mask[(2 * r) * mask_stride + 2 * c + 1]) +
				     i32(mask[(2 * r + 1) * mask_stride + 2 * c]) +
				     i32(mask[(2 * r + 1) * mask_stride + 2 * c + 1]) + 2) >> 2
			}
			if interintra {
				v0 := clamp((p0[c] + half_ii) >> sh_ii, 0, max_val)
				v1 := i32(out_row[c])
				out_row[c] = T((m * v1 + (64 - m) * v0 + 32) >> 6)
			} else {
				v0 := p0[c]
				v1 := pred1[r * pred_stride + c]
				out_row[c] = T(clamp((m * v0 + (64 - m) * v1 + half_comp) >> sh_comp, 0, max_val))
			}
		}
	}
}

// 7.11.3.9 - 7.11.3.10: Returns the OBMC blending mask for `length` (2, 4, 8, 16, 32).
obmc_mask_for_length :: #force_inline proc "contextless" (length: int) -> []u8 {
	switch length {
	case 2:
		return OBMC_MASK_2[:]
	case 4:
		return OBMC_MASK_4[:]
	case 8:
		return OBMC_MASK_8[:]
	case 16:
		return OBMC_MASK_16[:]
	case:
		return OBMC_MASK_32[:]
	}
}

// 7.11.3.10: Overlap blending process for OBMC (`pass == 0` for above, `pass == 1` for left).
blend_obmc :: proc "contextless" (
	dst: []$T,
	dst_stride, pred_x, pred_y, pred_w, pred_h, pass: int,
	obmc_pred: []i32,
	obmc_stride: int,
	bit_depth := 8,
) #no_bounds_check {
	max_val := (i32(1) << uint(bit_depth)) - 1
	mask := obmc_mask_for_length(pass == 0 ? pred_h : pred_w)
	for i in 0 ..< pred_h {
		out_row := dst[(pred_y + i) * dst_stride + pred_x:]
		orow := obmc_pred[i * obmc_stride:]
		for j in 0 ..< pred_w {
			m := i32(pass == 0 ? mask[i] : mask[j])
			cur := i32(out_row[j])
			ov := clamp(orow[j], 0, max_val)
			out_row[j] = T((m * cur + (64 - m) * ov + 32) >> 6)
		}
	}
}
