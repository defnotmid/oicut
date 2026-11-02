package av1

// Constrained Directional Enhancement Filter (CDEF, section 7.15) shared by
// the AV1 decoder and encoder.

import "base:intrinsics"
import "core:simd"

_ :: simd

@(private)
floor_log2_u32 :: #force_inline proc "contextless" (v: u32) -> int {
	if v == 0 do return 0
	return int(31 - intrinsics.count_leading_zeros(v))
}

// Section 7.15.2: CDEF direction process on an 8x8 luma block at 4x4 coordinates (r, c).
// Uses 128-bit `#simd[8]i16` register shuffles and Exploits the palindrome symmetry
// of `v_w_diag` and `v_w_alt` to compute all 8 directional projections in registers.
cdef_direction :: proc "contextless" (
	luma: []$T,
	luma_stride: int,
	r, c: int,
	bit_depth := 8,
) -> (y_dir: int, var: int) #no_bounds_check {
	V8  :: #simd[8]i16
	U8  :: #simd[8]u16
	VT  :: #simd[8]T
	I8  :: #simd[8]i32

	load_row :: #force_inline proc "contextless" (ptr: [^]T, off: int, shift: U8, bias: V8) -> V8 #no_bounds_check {
		when T == u8 {
			_ = shift
			return V8(intrinsics.unaligned_load((^VT)(&ptr[off]))) - bias
		} else {
			raw := V8(intrinsics.unaligned_load((^VT)(&ptr[off])))
			return simd.shr(raw, shift) - bias
		}
	}

	shift_add_8 :: #force_inline proc "contextless" (
		a0, a1, a2, a3, a4, a5, a6, a7: V8,
		v_w_lo, v_w_hi: I8,
	) -> i32 {
		z := V8(0)
		lo := a0 +
			simd.shuffle(z, a1, 7, 8, 9, 10, 11, 12, 13, 14) +
			simd.shuffle(z, a2, 6, 7, 8, 9, 10, 11, 12, 13) +
			simd.shuffle(z, a3, 5, 6, 7, 8, 9, 10, 11, 12) +
			simd.shuffle(z, a4, 4, 5, 6, 7, 8, 9, 10, 11) +
			simd.shuffle(z, a5, 3, 4, 5, 6, 7, 8, 9, 10) +
			simd.shuffle(z, a6, 2, 3, 4, 5, 6, 7, 8, 9) +
			simd.shuffle(z, a7, 1, 2, 3, 4, 5, 6, 7, 8)
		hi := simd.shuffle(a1, z, 7, 8, 9, 10, 11, 12, 13, 14) +
			simd.shuffle(a2, z, 6, 7, 8, 9, 10, 11, 12, 13) +
			simd.shuffle(a3, z, 5, 6, 7, 8, 9, 10, 11, 12) +
			simd.shuffle(a4, z, 4, 5, 6, 7, 8, 9, 10, 11) +
			simd.shuffle(a5, z, 3, 4, 5, 6, 7, 8, 9, 10) +
			simd.shuffle(a6, z, 2, 3, 4, 5, 6, 7, 8, 9) +
			simd.shuffle(a7, z, 1, 2, 3, 4, 5, 6, 7, 8)
		lo32 := I8(lo)
		hi32 := I8(hi)
		return simd.reduce_add_ordered(lo32 * lo32 * v_w_lo + hi32 * hi32 * v_w_hi)
	}

	shift_add_4 :: #force_inline proc "contextless" (
		b0, b1, b2, b3: V8,
		v_w_lo, v_w_hi: I8,
	) -> i32 {
		z := V8(0)
		lo := b0 +
			simd.shuffle(z, b1, 7, 8, 9, 10, 11, 12, 13, 14) +
			simd.shuffle(z, b2, 6, 7, 8, 9, 10, 11, 12, 13) +
			simd.shuffle(z, b3, 5, 6, 7, 8, 9, 10, 11, 12)
		hi := simd.shuffle(b1, z, 7, 8, 9, 10, 11, 12, 13, 14) +
			simd.shuffle(b2, z, 6, 7, 8, 9, 10, 11, 12, 13) +
			simd.shuffle(b3, z, 5, 6, 7, 8, 9, 10, 11, 12)
		lo32 := I8(lo)
		hi32 := I8(hi)
		return simd.reduce_add_ordered(lo32 * lo32 * v_w_lo + hi32 * hi32 * v_w_hi)
	}

	x0 := c << MI_SIZE_LOG2
	y0 := r << MI_SIZE_LOG2
	shift := U8(u16(bit_depth - 8) & 7)
	bias  := V8(128)
	ptr   := raw_data(luma)
	base  := y0 * luma_stride + x0

	r0 := load_row(ptr, base + 0 * luma_stride, shift, bias)
	r1 := load_row(ptr, base + 1 * luma_stride, shift, bias)
	r2 := load_row(ptr, base + 2 * luma_stride, shift, bias)
	r3 := load_row(ptr, base + 3 * luma_stride, shift, bias)
	r4 := load_row(ptr, base + 4 * luma_stride, shift, bias)
	r5 := load_row(ptr, base + 5 * luma_stride, shift, bias)
	r6 := load_row(ptr, base + 6 * luma_stride, shift, bias)
	r7 := load_row(ptr, base + 7 * luma_stride, shift, bias)

	v_w_diag_lo := I8{840, 420, 280, 210, 168, 140, 120, 105}
	v_w_diag_hi := I8{120, 140, 168, 210, 280, 420, 840, 0}
	v_w_alt_lo  := I8{420, 210, 140, 105, 105, 105, 105, 105}
	v_w_alt_hi  := I8{140, 210, 420, 0, 0, 0, 0, 0}

	cost: [8]i32 = ---
	cost[0] = shift_add_8(r0, r1, r2, r3, r4, r5, r6, r7, v_w_diag_lo, v_w_diag_hi)
	cost[4] = shift_add_8(r7, r6, r5, r4, r3, r2, r1, r0, v_w_diag_lo, v_w_diag_hi)

	// Row-pair sums q0..q3 for hv6, alt7, alt5.
	q0 := r0 + r1
	q1 := r2 + r3
	q2 := r4 + r5
	q3 := r6 + r7
	v_hv6 := I8((q0 + q1) + (q2 + q3))
	cost[6] = simd.reduce_add_ordered(v_hv6 * v_hv6) * 105
	cost[7] = shift_add_4(q0, q1, q2, q3, v_w_alt_lo, v_w_alt_hi)
	cost[5] = shift_add_4(q3, q2, q1, q0, v_w_alt_lo, v_w_alt_hi)

	// Column-pair sums c0..c3 for hv2, alt1, alt3.
	p01 := simd.shuffle(r0, r1, 0, 2, 4, 6, 8, 10, 12, 14) + simd.shuffle(r0, r1, 1, 3, 5, 7, 9, 11, 13, 15)
	p23 := simd.shuffle(r2, r3, 0, 2, 4, 6, 8, 10, 12, 14) + simd.shuffle(r2, r3, 1, 3, 5, 7, 9, 11, 13, 15)
	p45 := simd.shuffle(r4, r5, 0, 2, 4, 6, 8, 10, 12, 14) + simd.shuffle(r4, r5, 1, 3, 5, 7, 9, 11, 13, 15)
	p67 := simd.shuffle(r6, r7, 0, 2, 4, 6, 8, 10, 12, 14) + simd.shuffle(r6, r7, 1, 3, 5, 7, 9, 11, 13, 15)

	t0 := simd.shuffle(p01, p23, 0, 4, 8, 12, 1, 5, 9, 13)
	t1 := simd.shuffle(p45, p67, 0, 4, 8, 12, 1, 5, 9, 13)
	t2 := simd.shuffle(p01, p23, 2, 6, 10, 14, 3, 7, 11, 15)
	t3 := simd.shuffle(p45, p67, 2, 6, 10, 14, 3, 7, 11, 15)

	c0 := simd.shuffle(t0, t1, 0, 1, 2, 3, 8, 9, 10, 11)
	c1 := simd.shuffle(t0, t1, 4, 5, 6, 7, 12, 13, 14, 15)
	c2 := simd.shuffle(t2, t3, 0, 1, 2, 3, 8, 9, 10, 11)
	c3 := simd.shuffle(t2, t3, 4, 5, 6, 7, 12, 13, 14, 15)

	v_hv2 := I8((c0 + c1) + (c2 + c3))
	cost[2] = simd.reduce_add_ordered(v_hv2 * v_hv2) * 105
	cost[1] = shift_add_4(c0, c1, c2, c3, v_w_alt_lo, v_w_alt_hi)
	cost[3] = shift_add_4(c3, c2, c1, c0, v_w_alt_lo, v_w_alt_hi)

	// Track `opp_cost` (`cost[(y_dir + 4) & 7]`) alongside `best_cost` inside the unrolled loop
	// so `cost: [8]i32` is only ever indexed by compile-time constants and stays 100% in registers
	// without spilling to the stack for a runtime-indexed reload.
	best_cost := cost[0]
	opp_cost := cost[4]
	y_dir = 0
	#unroll for i in 1 ..< 8 {
		if cost[i] > best_cost {
			best_cost = cost[i]
			opp_cost = cost[(i + 4) & 7]
			y_dir = i
		}
	}
	var = int((best_cost - opp_cost) >> 10)
	return
}

@(private)
CDEF_VERY_LARGE :: i16(0x4000)

@(private)
CDEF_SIMD_LANES :: 16 when intrinsics.has_target_feature("avx2") else 8

@(private)
cdef_constrain_simd :: #force_inline proc "contextless" (
	diff: $V/#simd[$W]i16,
	thresh_u: #simd[W]u16,
	damp: u16,
) -> V {
	U :: #simd[W]u16
	ad := U(simd.abs(diff))
	val := V(simd.min(simd.saturating_sub(thresh_u, simd.shr(ad, U(damp))), ad))
	return simd.select(simd.lanes_lt(diff, V(0)), -val, val)
}

@(private)
cdef_load_lanes :: #force_inline proc "contextless" (ptr: [^]$T, stride: int, $W: int, $L: int) -> #simd[L]i16 #no_bounds_check {
	when L == 32 && W == 8 {
		_ = stride
		r01 := (#simd[32]i16)(intrinsics.unaligned_load((^#simd[32]T)(ptr)))
		r23 := (#simd[32]i16)(intrinsics.unaligned_load((^#simd[32]T)(&ptr[24])))
		return simd.shuffle(
			r01, r23,
			0, 1, 2, 3, 4, 5, 6, 7,
			40, 41, 42, 43, 44, 45, 46, 47,
			16, 17, 18, 19, 20, 21, 22, 23,
			56, 57, 58, 59, 60, 61, 62, 63,
		)
	} else when L == 16 && W == 8 {
		r0 := (#simd[8]i16)(intrinsics.unaligned_load((^#simd[8]T)(ptr)))
		r1 := (#simd[8]i16)(intrinsics.unaligned_load((^#simd[8]T)(&ptr[stride])))
		return simd.shuffle(r0, r1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)
	} else when L == 16 && W == 4 {
		_ = stride
		r01 := (#simd[16]i16)(intrinsics.unaligned_load((^#simd[16]T)(ptr)))
		r23 := (#simd[16]i16)(intrinsics.unaligned_load((^#simd[16]T)(&ptr[12])))
		return simd.shuffle(r01, r23, 0, 1, 2, 3, 20, 21, 22, 23, 8, 9, 10, 11, 28, 29, 30, 31)
	} else when L == 8 && W == 8 {
		return (#simd[8]i16)(intrinsics.unaligned_load((^#simd[8]T)(ptr)))
	} else {
		r0 := (#simd[4]i16)(intrinsics.unaligned_load((^#simd[4]T)(ptr)))
		r1 := (#simd[4]i16)(intrinsics.unaligned_load((^#simd[4]T)(&ptr[stride])))
		return simd.shuffle(r0, r1, 0, 1, 2, 3, 4, 5, 6, 7)
	}
}

@(private)
cdef_store_lanes :: #force_inline proc "contextless" (ptr: [^]$T, stride: int, v: #simd[$L]i16, $W: int) #no_bounds_check {
	vt := (#simd[L]T)(v)
	when L == 32 && W == 8 {
		r0 := simd.shuffle(vt, vt, 0, 1, 2, 3, 4, 5, 6, 7)
		r1 := simd.shuffle(vt, vt, 16, 17, 18, 19, 20, 21, 22, 23)
		r2 := simd.shuffle(vt, vt, 8, 9, 10, 11, 12, 13, 14, 15)
		r3 := simd.shuffle(vt, vt, 24, 25, 26, 27, 28, 29, 30, 31)
		intrinsics.unaligned_store((^#simd[8]T)(ptr), r0)
		intrinsics.unaligned_store((^#simd[8]T)(&ptr[stride]), r1)
		intrinsics.unaligned_store((^#simd[8]T)(&ptr[stride * 2]), r2)
		intrinsics.unaligned_store((^#simd[8]T)(&ptr[stride * 3]), r3)
	} else when L == 16 && W == 8 {
		r0 := simd.shuffle(vt, vt, 0, 1, 2, 3, 4, 5, 6, 7)
		r1 := simd.shuffle(vt, vt, 8, 9, 10, 11, 12, 13, 14, 15)
		intrinsics.unaligned_store((^#simd[8]T)(ptr), r0)
		intrinsics.unaligned_store((^#simd[8]T)(&ptr[stride]), r1)
	} else when L == 16 && W == 4 {
		r0 := simd.shuffle(vt, vt, 0, 1, 2, 3)
		r1 := simd.shuffle(vt, vt, 8, 9, 10, 11)
		r2 := simd.shuffle(vt, vt, 4, 5, 6, 7)
		r3 := simd.shuffle(vt, vt, 12, 13, 14, 15)
		intrinsics.unaligned_store((^#simd[4]T)(ptr), r0)
		intrinsics.unaligned_store((^#simd[4]T)(&ptr[stride]), r1)
		intrinsics.unaligned_store((^#simd[4]T)(&ptr[stride * 2]), r2)
		intrinsics.unaligned_store((^#simd[4]T)(&ptr[stride * 3]), r3)
	} else when L == 8 && W == 8 {
		intrinsics.unaligned_store((^#simd[8]T)(ptr), vt)
	} else {
		r0 := simd.shuffle(vt, vt, 0, 1, 2, 3)
		r1 := simd.shuffle(vt, vt, 4, 5, 6, 7)
		intrinsics.unaligned_store((^#simd[4]T)(ptr), r0)
		intrinsics.unaligned_store((^#simd[4]T)(&ptr[stride]), r1)
	}
}

@(private)
cdef_load_tmp_row :: #force_inline proc "contextless" (dst: [^]i16, src: [^]$T, $W: int) #no_bounds_check {
	when W == 8 {
		v01 := (#simd[16]i16)(intrinsics.unaligned_load((^#simd[16]T)(src)))
		intrinsics.unaligned_store((^#simd[16]i16)(dst), v01)
	} else {
		v0 := (#simd[8]i16)(intrinsics.unaligned_load((^#simd[8]T)(src)))
		intrinsics.unaligned_store((^#simd[8]i16)(dst), v0)
	}
}

@(private)
cdef_load_tmp_row_patch :: #force_inline proc "contextless" (
	dst: [^]i16,
	src: [^]$T,
	left_bak: ^[2]i16,
	left_mod: bool,
	in_place: bool,
	$W: int,
) #no_bounds_check {
	// Performance note:
	// Loading a full 16-lane (`W == 8`) or 8-lane (`W == 4`) vector, patching `left_bak` in-register,
	// and issuing a single 32-byte (`ymm`) or 16-byte (`xmm`) store ensures every subsequent per-row
	// load in `cdef_load_lanes` is strictly contained within a single store buffer entry (0-cycle STLF).
	when W == 8 {
		v01 := (#simd[16]i16)(intrinsics.unaligned_load((^#simd[16]T)(src)))
		if left_mod {
			v01 = simd.replace(v01, 0, left_bak[0])
			v01 = simd.replace(v01, 1, left_bak[1])
		}
		if in_place {
			left_bak[0] = simd.extract(v01, 8)
			left_bak[1] = simd.extract(v01, 9)
		}
		intrinsics.unaligned_store((^#simd[16]i16)(dst), v01)
	} else {
		v0 := (#simd[8]i16)(intrinsics.unaligned_load((^#simd[8]T)(src)))
		if left_mod {
			v0 = simd.replace(v0, 0, left_bak[0])
			v0 = simd.replace(v0, 1, left_bak[1])
		}
		if in_place {
			left_bak[0] = simd.extract(v0, 4)
			left_bak[1] = simd.extract(v0, 5)
		}
		intrinsics.unaligned_store((^#simd[8]i16)(dst), v0)
	}
}

@(private)
cdef_filter_tmp_simd :: #force_inline proc "contextless" (
	dst_plane: []$T,
	dst_stride: int,
	tmp: ^[13 * 16]i16,
	x0, y0, h: int,
	pri_str, sec_str: int,
	pri_damp, sec_damp: uint,
	pt0_u8, pt1_u8: u8,
	dir: int,
	$W: int,
	$H: int,
	$IS_BORDER: bool,
) #no_bounds_check {
	L :: (32 when W == 8 && intrinsics.has_target_feature("avx512bw") else CDEF_SIMD_LANES)
	V :: #simd[L]i16
	U :: #simd[L]u16
	TMP_STRIDE :: 8 when W == 4 else 16
	STEP :: L / W
	_ = h

	pri_thresh_u := U(u16(pri_str))
	sec_thresh_u := U(u16(sec_str))
	pri_damp_u := u16(pri_damp & 15)
	sec_damp_u := u16(sec_damp & 15)
	dir_p2 := (dir + 2) & 7
	dir_m2 := (dir - 2) & 7
	src_ptr := ([^]i16)(tmp)
	dst_ptr := raw_data(dst_plane)
	dst_row := dst_ptr[y0 * dst_stride + x0:]

	off_pri0 := int(CDEF_DIRECTIONS[dir][0][0]) * TMP_STRIDE + int(CDEF_DIRECTIONS[dir][0][1])
	off_pri1 := int(CDEF_DIRECTIONS[dir][1][0]) * TMP_STRIDE + int(CDEF_DIRECTIONS[dir][1][1])
	off_s0_0 := int(CDEF_DIRECTIONS[dir_m2][0][0]) * TMP_STRIDE + int(CDEF_DIRECTIONS[dir_m2][0][1])
	off_s0_1 := int(CDEF_DIRECTIONS[dir_m2][1][0]) * TMP_STRIDE + int(CDEF_DIRECTIONS[dir_m2][1][1])
	off_s1_0 := int(CDEF_DIRECTIONS[dir_p2][0][0]) * TMP_STRIDE + int(CDEF_DIRECTIONS[dir_p2][0][1])
	off_s1_1 := int(CDEF_DIRECTIONS[dir_p2][1][0]) * TMP_STRIDE + int(CDEF_DIRECTIONS[dir_p2][1][1])
	pt0 := V(i16(pt0_u8))
	pt1 := V(i16(pt1_u8))

	if sec_str == 0 {
		#unroll for step_idx in 0 ..< H / STEP {
			i := step_idx * STEP
			row_off := (i + 2) * TMP_STRIDE + 2
			x      := cdef_load_lanes(&src_ptr[row_off], TMP_STRIDE, W, L)
			p0_neg := cdef_load_lanes(&src_ptr[row_off - off_pri0], TMP_STRIDE, W, L)
			p0_pos := cdef_load_lanes(&src_ptr[row_off + off_pri0], TMP_STRIDE, W, L)
			p1_neg := cdef_load_lanes(&src_ptr[row_off - off_pri1], TMP_STRIDE, W, L)
			p1_pos := cdef_load_lanes(&src_ptr[row_off + off_pri1], TMP_STRIDE, W, L)

			sum := pt0 * (cdef_constrain_simd(p0_neg - x, pri_thresh_u, pri_damp_u) +
				cdef_constrain_simd(p0_pos - x, pri_thresh_u, pri_damp_u)) +
				pt1 * (cdef_constrain_simd(p1_neg - x, pri_thresh_u, pri_damp_u) +
				cdef_constrain_simd(p1_pos - x, pri_thresh_u, pri_damp_u))

			res := x + simd.shr(sum + V(8) + simd.shr(sum, U(15)), U(4))
			cdef_store_lanes(&dst_row[i * dst_stride], dst_stride, res, W)
		}
		return
	}

	if pri_str == 0 {
		#unroll for step_idx in 0 ..< H / STEP {
			i := step_idx * STEP
			row_off := (i + 2) * TMP_STRIDE + 2
			x       := cdef_load_lanes(&src_ptr[row_off], TMP_STRIDE, W, L)
			s00_neg := cdef_load_lanes(&src_ptr[row_off - off_s0_0], TMP_STRIDE, W, L)
			s00_pos := cdef_load_lanes(&src_ptr[row_off + off_s0_0], TMP_STRIDE, W, L)
			s10_neg := cdef_load_lanes(&src_ptr[row_off - off_s1_0], TMP_STRIDE, W, L)
			s10_pos := cdef_load_lanes(&src_ptr[row_off + off_s1_0], TMP_STRIDE, W, L)
			s01_neg := cdef_load_lanes(&src_ptr[row_off - off_s0_1], TMP_STRIDE, W, L)
			s01_pos := cdef_load_lanes(&src_ptr[row_off + off_s0_1], TMP_STRIDE, W, L)
			s11_neg := cdef_load_lanes(&src_ptr[row_off - off_s1_1], TMP_STRIDE, W, L)
			s11_pos := cdef_load_lanes(&src_ptr[row_off + off_s1_1], TMP_STRIDE, W, L)

			sum0 := cdef_constrain_simd(s00_neg - x, sec_thresh_u, sec_damp_u) +
				cdef_constrain_simd(s00_pos - x, sec_thresh_u, sec_damp_u) +
				cdef_constrain_simd(s10_neg - x, sec_thresh_u, sec_damp_u) +
				cdef_constrain_simd(s10_pos - x, sec_thresh_u, sec_damp_u)
			sum1 := cdef_constrain_simd(s01_neg - x, sec_thresh_u, sec_damp_u) +
				cdef_constrain_simd(s01_pos - x, sec_thresh_u, sec_damp_u) +
				cdef_constrain_simd(s11_neg - x, sec_thresh_u, sec_damp_u) +
				cdef_constrain_simd(s11_pos - x, sec_thresh_u, sec_damp_u)
			sum := sum0 + sum0 + sum1

			res := x + simd.shr(sum + V(8) + simd.shr(sum, U(15)), U(4))
			cdef_store_lanes(&dst_row[i * dst_stride], dst_stride, res, W)
		}
		return
	}

	max_mask := V(0x3FFF)
	#unroll for step_idx in 0 ..< H / STEP {
		i := step_idx * STEP
		row_off := (i + 2) * TMP_STRIDE + 2
		x := cdef_load_lanes(&src_ptr[row_off], TMP_STRIDE, W, L)

		p0_neg := cdef_load_lanes(&src_ptr[row_off - off_pri0], TMP_STRIDE, W, L)
		p0_pos := cdef_load_lanes(&src_ptr[row_off + off_pri0], TMP_STRIDE, W, L)
		p1_neg := cdef_load_lanes(&src_ptr[row_off - off_pri1], TMP_STRIDE, W, L)
		p1_pos := cdef_load_lanes(&src_ptr[row_off + off_pri1], TMP_STRIDE, W, L)
		max_v: V
		when IS_BORDER {
			max_v = simd.max(x, simd.max(simd.max(p0_neg & max_mask, p0_pos & max_mask), simd.max(p1_neg & max_mask, p1_pos & max_mask)))
		} else {
			_ = max_mask
			max_v = simd.max(x, simd.max(simd.max(p0_neg, p0_pos), simd.max(p1_neg, p1_pos)))
		}
		min_v := simd.min(x, simd.min(simd.min(p0_neg, p0_pos), simd.min(p1_neg, p1_pos)))
		sum := pt0 * (cdef_constrain_simd(p0_neg - x, pri_thresh_u, pri_damp_u) +
			cdef_constrain_simd(p0_pos - x, pri_thresh_u, pri_damp_u)) +
			pt1 * (cdef_constrain_simd(p1_neg - x, pri_thresh_u, pri_damp_u) +
			cdef_constrain_simd(p1_pos - x, pri_thresh_u, pri_damp_u))

		s00_neg := cdef_load_lanes(&src_ptr[row_off - off_s0_0], TMP_STRIDE, W, L)
		s00_pos := cdef_load_lanes(&src_ptr[row_off + off_s0_0], TMP_STRIDE, W, L)
		s10_neg := cdef_load_lanes(&src_ptr[row_off - off_s1_0], TMP_STRIDE, W, L)
		s10_pos := cdef_load_lanes(&src_ptr[row_off + off_s1_0], TMP_STRIDE, W, L)
		when IS_BORDER {
			max_v = simd.max(max_v, simd.max(simd.max(s00_neg & max_mask, s00_pos & max_mask), simd.max(s10_neg & max_mask, s10_pos & max_mask)))
		} else {
			max_v = simd.max(max_v, simd.max(simd.max(s00_neg, s00_pos), simd.max(s10_neg, s10_pos)))
		}
		min_v = simd.min(min_v, simd.min(simd.min(s00_neg, s00_pos), simd.min(s10_neg, s10_pos)))
		sum0 := cdef_constrain_simd(s00_neg - x, sec_thresh_u, sec_damp_u) +
			cdef_constrain_simd(s00_pos - x, sec_thresh_u, sec_damp_u) +
			cdef_constrain_simd(s10_neg - x, sec_thresh_u, sec_damp_u) +
			cdef_constrain_simd(s10_pos - x, sec_thresh_u, sec_damp_u)

		s01_neg := cdef_load_lanes(&src_ptr[row_off - off_s0_1], TMP_STRIDE, W, L)
		s01_pos := cdef_load_lanes(&src_ptr[row_off + off_s0_1], TMP_STRIDE, W, L)
		s11_neg := cdef_load_lanes(&src_ptr[row_off - off_s1_1], TMP_STRIDE, W, L)
		s11_pos := cdef_load_lanes(&src_ptr[row_off + off_s1_1], TMP_STRIDE, W, L)
		when IS_BORDER {
			max_v = simd.max(max_v, simd.max(simd.max(s01_neg & max_mask, s01_pos & max_mask), simd.max(s11_neg & max_mask, s11_pos & max_mask)))
		} else {
			max_v = simd.max(max_v, simd.max(simd.max(s01_neg, s01_pos), simd.max(s11_neg, s11_pos)))
		}
		min_v = simd.min(min_v, simd.min(simd.min(s01_neg, s01_pos), simd.min(s11_neg, s11_pos)))
		sum1 := cdef_constrain_simd(s01_neg - x, sec_thresh_u, sec_damp_u) +
			cdef_constrain_simd(s01_pos - x, sec_thresh_u, sec_damp_u) +
			cdef_constrain_simd(s11_neg - x, sec_thresh_u, sec_damp_u) +
			cdef_constrain_simd(s11_pos - x, sec_thresh_u, sec_damp_u)

		sum += sum0 + sum0 + sum1
		res := simd.clamp(x + simd.shr(sum + V(8) + simd.shr(sum, U(15)), U(4)), min_v, max_v)
		cdef_store_lanes(&dst_row[i * dst_stride], dst_stride, res, W)
	}
}

@(private)
cdef_interior_load_tmp :: #force_inline proc "contextless" (
	tmp: ^[13 * 16]i16,
	src_ptr: [^]$T,
	src_stride: int,
	top_row0, top_row1: [^]T,
	left_bak: ^[8][2]i16,
	left_mod: bool,
	in_place: bool,
	x0, y0: int,
	$W: int,
	$H: int,
) #no_bounds_check {
	TMP_STRIDE :: 8 when W == 4 else 16
	cdef_load_tmp_row(&tmp[0 * TMP_STRIDE], &top_row0[x0 - 2], W)
	cdef_load_tmp_row(&tmp[1 * TMP_STRIDE], &top_row1[x0 - 2], W)
	src_row := src_ptr[y0 * src_stride + x0 - 2:]
	#unroll for dy in 0 ..< H {
		cdef_load_tmp_row_patch(&tmp[(dy + 2) * TMP_STRIDE], &src_row[dy * src_stride], &left_bak[dy], left_mod, in_place, W)
	}
	#unroll for dy in H ..< H + 2 {
		cdef_load_tmp_row(&tmp[(dy + 2) * TMP_STRIDE], &src_row[dy * src_stride], W)
	}
}

@(private)
cdef_interior_load_and_filter :: #force_inline proc "contextless" (
	dst_plane: []$T,
	dst_stride: int,
	src_ptr: [^]T,
	src_stride: int,
	top_row0, top_row1: [^]T,
	left_bak: ^[8][2]i16,
	left_mod: bool,
	in_place: bool,
	x0, y0: int,
	pri_str, sec_str: int,
	pri_damp, sec_damp: uint,
	pri_taps: [2]u8,
	dir: int,
	$W: int,
	$H: int,
) #no_bounds_check {
	tmp: [13 * 16]i16 = ---
	cdef_interior_load_tmp(&tmp, src_ptr, src_stride, top_row0, top_row1, left_bak, left_mod, in_place, x0, y0, W, H)
	cdef_filter_tmp_simd(dst_plane, dst_stride, &tmp, x0, y0, H, pri_str, sec_str, pri_damp, sec_damp, pri_taps[0], pri_taps[1], dir, W, H, false)
}

@(private)
cdef_load_uv_4x4 :: #force_inline proc "contextless" (ptr: [^]i16) -> #simd[32]i16 #no_bounds_check {
	r01 := intrinsics.unaligned_load((^#simd[32]i16)(ptr))
	r23 := intrinsics.unaligned_load((^#simd[32]i16)(&ptr[32]))
	return simd.shuffle(
		r01, r23,
		0, 1, 2, 3, 32, 33, 34, 35, 16, 17, 18, 19, 48, 49, 50, 51,
		8, 9, 10, 11, 40, 41, 42, 43, 24, 25, 26, 27, 56, 57, 58, 59,
	)
}

@(private)
cdef_interior_uv_4x4_load_tmp :: #force_inline proc "contextless" (
	tmp: ^[9 * 16]i16,
	src_u, src_v: [^]$T,
	src_stride_u, src_stride_v: int,
	top0_u, top1_u, top0_v, top1_v: [^]T,
	left_bak_u, left_bak_v: ^[8][2]i16,
	left_mod: bool,
	in_place: bool,
	x0, y0: int,
) #no_bounds_check {
	TMP_STRIDE :: 16
	cdef_load_tmp_row(&tmp[0 * TMP_STRIDE + 0], &top0_u[x0 - 2], 4)
	cdef_load_tmp_row(&tmp[0 * TMP_STRIDE + 8], &top0_v[x0 - 2], 4)
	cdef_load_tmp_row(&tmp[1 * TMP_STRIDE + 0], &top1_u[x0 - 2], 4)
	cdef_load_tmp_row(&tmp[1 * TMP_STRIDE + 8], &top1_v[x0 - 2], 4)

	row_u := src_u[y0 * src_stride_u + x0 - 2:]
	row_v := src_v[y0 * src_stride_v + x0 - 2:]
	#unroll for dy in 0 ..< 4 {
		cdef_load_tmp_row_patch(&tmp[(dy + 2) * TMP_STRIDE + 0], &row_u[dy * src_stride_u], &left_bak_u[dy], left_mod, in_place, 4)
		cdef_load_tmp_row_patch(&tmp[(dy + 2) * TMP_STRIDE + 8], &row_v[dy * src_stride_v], &left_bak_v[dy], left_mod, in_place, 4)
	}
	#unroll for dy in 4 ..< 6 {
		cdef_load_tmp_row(&tmp[(dy + 2) * TMP_STRIDE + 0], &row_u[dy * src_stride_u], 4)
		cdef_load_tmp_row(&tmp[(dy + 2) * TMP_STRIDE + 8], &row_v[dy * src_stride_v], 4)
	}
}

@(private)
cdef_interior_uv_4x4_filter_tmp :: #force_inline proc "contextless" (
	dst_u, dst_v: []$T,
	dst_stride_u, dst_stride_v: int,
	tmp: ^[9 * 16]i16,
	x0, y0: int,
	pri_str, sec_str: int,
	pri_damp, sec_damp: uint,
	pri_taps: [2]u8,
	dir: int,
) #no_bounds_check {
	V :: #simd[32]i16
	U :: #simd[32]u16
	TMP_STRIDE :: 16

	pri_thresh_u := U(u16(pri_str))
	sec_thresh_u := U(u16(sec_str))
	pri_damp_u := u16(pri_damp & 15)
	sec_damp_u := u16(sec_damp & 15)
	dir_p2 := (dir + 2) & 7
	dir_m2 := (dir - 2) & 7

	off_pri0 := int(CDEF_DIRECTIONS[dir][0][0]) * TMP_STRIDE + int(CDEF_DIRECTIONS[dir][0][1])
	off_pri1 := int(CDEF_DIRECTIONS[dir][1][0]) * TMP_STRIDE + int(CDEF_DIRECTIONS[dir][1][1])
	off_s0_0 := int(CDEF_DIRECTIONS[dir_m2][0][0]) * TMP_STRIDE + int(CDEF_DIRECTIONS[dir_m2][0][1])
	off_s0_1 := int(CDEF_DIRECTIONS[dir_m2][1][0]) * TMP_STRIDE + int(CDEF_DIRECTIONS[dir_m2][1][1])
	off_s1_0 := int(CDEF_DIRECTIONS[dir_p2][0][0]) * TMP_STRIDE + int(CDEF_DIRECTIONS[dir_p2][0][1])
	off_s1_1 := int(CDEF_DIRECTIONS[dir_p2][1][0]) * TMP_STRIDE + int(CDEF_DIRECTIONS[dir_p2][1][1])

	base := ([^]i16)(&tmp[2 * TMP_STRIDE + 2])
	x := cdef_load_uv_4x4(base)
	pt0 := V(i16(pri_taps[0]))
	pt1 := V(i16(pri_taps[1]))

	res: V
	if sec_str == 0 {
		p0_pos := cdef_load_uv_4x4(&base[off_pri0])
		p0_neg := cdef_load_uv_4x4(&base[-off_pri0])
		p1_pos := cdef_load_uv_4x4(&base[off_pri1])
		p1_neg := cdef_load_uv_4x4(&base[-off_pri1])
		sum := pt0 * (cdef_constrain_simd(p0_pos - x, pri_thresh_u, pri_damp_u) +
			cdef_constrain_simd(p0_neg - x, pri_thresh_u, pri_damp_u)) +
			pt1 * (cdef_constrain_simd(p1_pos - x, pri_thresh_u, pri_damp_u) +
			cdef_constrain_simd(p1_neg - x, pri_thresh_u, pri_damp_u))
		res = x + simd.shr_masked(sum + V(8) + simd.shr_masked(sum, U(15)), U(4))
	} else if pri_str == 0 {
		s00_pos := cdef_load_uv_4x4(&base[off_s0_0])
		s00_neg := cdef_load_uv_4x4(&base[-off_s0_0])
		s01_pos := cdef_load_uv_4x4(&base[off_s0_1])
		s01_neg := cdef_load_uv_4x4(&base[-off_s0_1])
		s10_pos := cdef_load_uv_4x4(&base[off_s1_0])
		s10_neg := cdef_load_uv_4x4(&base[-off_s1_0])
		s11_pos := cdef_load_uv_4x4(&base[off_s1_1])
		s11_neg := cdef_load_uv_4x4(&base[-off_s1_1])
		sum0 := cdef_constrain_simd(s00_pos - x, sec_thresh_u, sec_damp_u) +
			cdef_constrain_simd(s00_neg - x, sec_thresh_u, sec_damp_u) +
			cdef_constrain_simd(s10_pos - x, sec_thresh_u, sec_damp_u) +
			cdef_constrain_simd(s10_neg - x, sec_thresh_u, sec_damp_u)
		sum1 := cdef_constrain_simd(s01_pos - x, sec_thresh_u, sec_damp_u) +
			cdef_constrain_simd(s01_neg - x, sec_thresh_u, sec_damp_u) +
			cdef_constrain_simd(s11_neg - x, sec_thresh_u, sec_damp_u) +
			cdef_constrain_simd(s11_pos - x, sec_thresh_u, sec_damp_u)
		sum := sum0 + sum0 + sum1
		res = x + simd.shr_masked(sum + V(8) + simd.shr_masked(sum, U(15)), U(4))
	} else {
		p0_pos := cdef_load_uv_4x4(&base[off_pri0])
		p0_neg := cdef_load_uv_4x4(&base[-off_pri0])
		p1_pos := cdef_load_uv_4x4(&base[off_pri1])
		p1_neg := cdef_load_uv_4x4(&base[-off_pri1])
		s00_pos := cdef_load_uv_4x4(&base[off_s0_0])
		s00_neg := cdef_load_uv_4x4(&base[-off_s0_0])
		s01_pos := cdef_load_uv_4x4(&base[off_s0_1])
		s01_neg := cdef_load_uv_4x4(&base[-off_s0_1])
		s10_pos := cdef_load_uv_4x4(&base[off_s1_0])
		s10_neg := cdef_load_uv_4x4(&base[-off_s1_0])
		s11_pos := cdef_load_uv_4x4(&base[off_s1_1])
		s11_neg := cdef_load_uv_4x4(&base[-off_s1_1])

		min_v := simd.min(
			simd.min(simd.min(simd.min(x, p0_pos), simd.min(p0_neg, p1_pos)), simd.min(simd.min(p1_neg, s00_pos), simd.min(s00_neg, s01_pos))),
			simd.min(simd.min(simd.min(s01_neg, s10_pos), simd.min(s10_neg, s11_pos)), s11_neg),
		)
		max_v := simd.max(
			simd.max(simd.max(simd.max(x, p0_pos), simd.max(p0_neg, p1_pos)), simd.max(simd.max(p1_neg, s00_pos), simd.max(s00_neg, s01_pos))),
			simd.max(simd.max(simd.max(s01_neg, s10_pos), simd.max(s10_neg, s11_pos)), s11_neg),
		)

		sum := pt0 * (cdef_constrain_simd(p0_pos - x, pri_thresh_u, pri_damp_u) +
			cdef_constrain_simd(p0_neg - x, pri_thresh_u, pri_damp_u)) +
			pt1 * (cdef_constrain_simd(p1_pos - x, pri_thresh_u, pri_damp_u) +
			cdef_constrain_simd(p1_neg - x, pri_thresh_u, pri_damp_u))

		sum0 := cdef_constrain_simd(s00_pos - x, sec_thresh_u, sec_damp_u) +
			cdef_constrain_simd(s00_neg - x, sec_thresh_u, sec_damp_u) +
			cdef_constrain_simd(s10_pos - x, sec_thresh_u, sec_damp_u) +
			cdef_constrain_simd(s10_neg - x, sec_thresh_u, sec_damp_u)
		sum1 := cdef_constrain_simd(s01_pos - x, sec_thresh_u, sec_damp_u) +
			cdef_constrain_simd(s01_neg - x, sec_thresh_u, sec_damp_u) +
			cdef_constrain_simd(s11_neg - x, sec_thresh_u, sec_damp_u) +
			cdef_constrain_simd(s11_pos - x, sec_thresh_u, sec_damp_u)

		sum += sum0 + sum0 + sum1
		res = simd.clamp(x + simd.shr_masked(sum + V(8) + simd.shr_masked(sum, U(15)), U(4)), min_v, max_v)
	}
	vt := (#simd[32]T)(res)

	out_u := raw_data(dst_u)[y0 * dst_stride_u + x0:]
	out_v := raw_data(dst_v)[y0 * dst_stride_v + x0:]
	intrinsics.unaligned_store((^#simd[4]T)(out_u), simd.shuffle(vt, vt, 0, 1, 2, 3))
	intrinsics.unaligned_store((^#simd[4]T)(&out_u[dst_stride_u]), simd.shuffle(vt, vt, 8, 9, 10, 11))
	intrinsics.unaligned_store((^#simd[4]T)(&out_u[dst_stride_u * 2]), simd.shuffle(vt, vt, 4, 5, 6, 7))
	intrinsics.unaligned_store((^#simd[4]T)(&out_u[dst_stride_u * 3]), simd.shuffle(vt, vt, 12, 13, 14, 15))
	intrinsics.unaligned_store((^#simd[4]T)(out_v), simd.shuffle(vt, vt, 16, 17, 18, 19))
	intrinsics.unaligned_store((^#simd[4]T)(&out_v[dst_stride_v]), simd.shuffle(vt, vt, 24, 25, 26, 27))
	intrinsics.unaligned_store((^#simd[4]T)(&out_v[dst_stride_v * 2]), simd.shuffle(vt, vt, 20, 21, 22, 23))
	intrinsics.unaligned_store((^#simd[4]T)(&out_v[dst_stride_v * 3]), simd.shuffle(vt, vt, 28, 29, 30, 31))
}

@(private)
cdef_filter_block_border :: #force_no_inline proc "contextless" (
	dst_plane: []$T,
	dst_stride: int,
	src_plane: []T,
	src_stride: int,
	top_row0, top_row1: [^]T,
	left_bak: ^[8][2]i16,
	left_mod: bool,
	in_place: bool,
	x0, y0, w, h, max_x, max_y: int,
	pri_str, sec_str: int,
	pri_damp, sec_damp: uint,
	pri_taps: [2]u8,
	dir: int,
) #no_bounds_check {
	tmp_stride := w == 4 ? 8 : 16
	tmp: [13 * 16]i16 = ---
	v_large := (#simd[16]i16)(CDEF_VERY_LARGE)
	#unroll for r_idx in 0 ..< 13 {
		intrinsics.unaligned_store((^#simd[16]i16)(&tmp[r_idx * 16]), v_large)
	}
	iy_start := max(-2, -y0)
	iy_end   := min(h + 2, max_y - y0)
	ix_start := max(-2, -x0)
	ix_end   := min(w + 2, max_x - x0)
	for dy in iy_start ..< iy_end {
		tmp_row := tmp[(dy + 2) * tmp_stride + 2:]
		if dy < 0 {
			t_row := dy == -2 ? top_row0 : top_row1
			for dx in ix_start ..< ix_end {
				tmp_row[dx] = i16(t_row[x0 + dx])
			}
		} else {
			src_row := src_plane[(y0 + dy) * src_stride + x0:]
			for dx in ix_start ..< ix_end {
				tmp_row[dx] = i16(src_row[dx])
			}
			if left_mod && dy < h {
				if ix_start <= -2 do tmp_row[-2] = left_bak[dy][0]
				if ix_start <= -1 do tmp_row[-1] = left_bak[dy][1]
			}
		}
	}
	if in_place {
		for dy in 0 ..< h {
			left_bak[dy][0] = tmp[(dy + 2) * tmp_stride + w]
			left_bak[dy][1] = tmp[(dy + 2) * tmp_stride + w + 1]
		}
	}
	if w == 8 {
		cdef_filter_tmp_simd(dst_plane, dst_stride, &tmp, x0, y0, h, pri_str, sec_str, pri_damp, sec_damp, pri_taps[0], pri_taps[1], dir, 8, 8, true)
	} else if h == 4 {
		cdef_filter_tmp_simd(dst_plane, dst_stride, &tmp, x0, y0, h, pri_str, sec_str, pri_damp, sec_damp, pri_taps[0], pri_taps[1], dir, 4, 4, true)
	} else {
		cdef_filter_tmp_simd(dst_plane, dst_stride, &tmp, x0, y0, h, pri_str, sec_str, pri_damp, sec_damp, pri_taps[0], pri_taps[1], dir, 4, 8, true)
	}
}

@(private)
cdef_copy_block :: #force_inline proc "contextless" (
	dst_plane: []$T,
	dst_stride: int,
	src_plane: []T,
	src_stride: int,
	x0, y0, w, h: int,
) #no_bounds_check {
	dst_ptr := raw_data(dst_plane)
	src_ptr := raw_data(src_plane)
	if w == 8 {
		for i in 0 ..< h {
			v := intrinsics.unaligned_load((^#simd[8]T)(&src_ptr[(y0 + i) * src_stride + x0]))
			intrinsics.unaligned_store((^#simd[8]T)(&dst_ptr[(y0 + i) * dst_stride + x0]), v)
		}
	} else {
		for i in 0 ..< h {
			v := intrinsics.unaligned_load((^#simd[4]T)(&src_ptr[(y0 + i) * src_stride + x0]))
			intrinsics.unaligned_store((^#simd[4]T)(&dst_ptr[(y0 + i) * dst_stride + x0]), v)
		}
	}
}

@(private)
cdef_filter_block_with_lines :: proc "contextless" (
	dst_plane: []$T,
	dst_stride: int,
	src_plane: []T,
	src_stride: int,
	top_row0, top_row1: [^]T,
	left_bak: ^[8][2]i16,
	left_mod: bool,
	in_place: bool,
	mi_rows, mi_cols: int,
	r, c: int,
	sub_x, sub_y: uint,
	pri_str, sec_str, damping, dir: int,
	bit_depth := 8,
) -> (modified: bool) #no_bounds_check {
	x0 := (c * MI_SIZE) >> sub_x
	y0 := (r * MI_SIZE) >> sub_y
	w := 8 >> sub_x
	h := 8 >> sub_y
	if pri_str == 0 && sec_str == 0 {
		if !in_place {
			cdef_copy_block(dst_plane, dst_stride, src_plane, src_stride, x0, y0, w, h)
		}
		return false
	}

	coeff_shift := uint(bit_depth - 8)
	tap_set := (pri_str >> coeff_shift) & 1
	pri_taps := CDEF_PRI_TAPS[tap_set]
	pri_damp := pri_str > 0 ? uint(max(0, damping - floor_log2_u32(u32(pri_str)))) : 0
	sec_damp := sec_str > 0 ? uint(max(0, damping - floor_log2_u32(u32(sec_str)))) : 0

	max_y := (mi_rows * MI_SIZE) >> sub_y
	max_x := (mi_cols * MI_SIZE) >> sub_x
	src_ptr := raw_data(src_plane)

	if y0 >= 2 && y0 + h + 2 <= max_y && x0 >= 2 && x0 + w + 2 <= max_x {
		if w == 8 {
			cdef_interior_load_and_filter(dst_plane, dst_stride, src_ptr, src_stride, top_row0, top_row1, left_bak, left_mod, in_place, x0, y0, pri_str, sec_str, pri_damp, sec_damp, pri_taps, dir, 8, 8)
		} else if h == 4 {
			cdef_interior_load_and_filter(dst_plane, dst_stride, src_ptr, src_stride, top_row0, top_row1, left_bak, left_mod, in_place, x0, y0, pri_str, sec_str, pri_damp, sec_damp, pri_taps, dir, 4, 4)
		} else {
			cdef_interior_load_and_filter(dst_plane, dst_stride, src_ptr, src_stride, top_row0, top_row1, left_bak, left_mod, in_place, x0, y0, pri_str, sec_str, pri_damp, sec_damp, pri_taps, dir, 4, 8)
		}
		return true
	}

	cdef_filter_block_border(dst_plane, dst_stride, src_plane, src_stride, top_row0, top_row1, left_bak, left_mod, in_place, x0, y0, w, h, max_x, max_y, pri_str, sec_str, pri_damp, sec_damp, pri_taps, dir)
	return true
}

// Section 7.15.3: CDEF filter process for one 8x8 (or subsampled chroma) block at (r, c).
cdef_filter_block :: proc "contextless" (
	dst_plane: []$T,
	dst_stride: int,
	src_plane: []T,
	src_stride: int,
	mi_rows, mi_cols: int,
	r, c: int,
	sub_x, sub_y: uint,
	pri_str, sec_str, damping, dir: int,
	bit_depth := 8,
) #no_bounds_check {
	y0 := (r * MI_SIZE) >> sub_y
	src_ptr := raw_data(src_plane)
	top_row0 := y0 >= 2 ? &src_ptr[(y0 - 2) * src_stride] : nil
	top_row1 := y0 >= 1 ? &src_ptr[(y0 - 1) * src_stride] : nil
	dummy_left: [8][2]i16 = ---
	cdef_filter_block_with_lines(
		dst_plane, dst_stride,
		src_plane, src_stride,
		top_row0, top_row1,
		&dummy_left, false, false,
		mi_rows, mi_cols,
		r, c, sub_x, sub_y,
		pri_str, sec_str, damping, dir,
		bit_depth,
	)
}

@(private)
CDEF_MAX_LINE_W :: 4096

// Section 7.15: Runs CDEF across the entire frame, writing deringed samples into `cdef_planes`
// from `curr_planes` (supports in-place execution when `cdef_planes` aliases `curr_planes`).
// `cdef_idx_grid` is indexed by 64x64 block `(r >> 4) * sb64_cols + (c >> 4)` (value -1 for skip).
// `skips` is indexed by 4x4 block `r * mi_cols + c`.
cdef_frame :: proc(
	cdef_planes: [3][]$T,
	cdef_strides: [3]int,
	curr_planes: [3][]T,
	curr_strides: [3]int,
	cdef_idx_grid: []i8,
	sb64_cols: int,
	skips: []bool,
	h: ^Frame_Header,
	enable_cdef: bool,
	num_planes: int,
	sub_x, sub_y: uint,
	bit_depth := 8,
) #no_bounds_check {
	mi_rows := h.mi_rows
	mi_cols := h.mi_cols
	coeff_shift := uint(bit_depth - 8)
	base_damping := int(h.cdef_damping_minus_3) + 3 + int(coeff_shift)
	uv_damping := base_damping - 1

	skips_ptr := raw_data(skips)
	cdef_enabled := enable_cdef && !h.coded_lossless && !h.allow_intrabc
	in_place := raw_data(cdef_planes[0]) == raw_data(curr_planes[0]) && (mi_cols * MI_SIZE) <= CDEF_MAX_LINE_W
	if !cdef_enabled && in_place do return

	Preset :: struct {
		y_pri, y_sec:         int,
		y_sec_damp:           uint,
		uv_pri, uv_sec:       int,
		uv_pri_damp:          uint,
		uv_sec_damp:          uint,
		uv_pri_taps:          [2]u8,
	}
	presets: [8]Preset = ---
	num_presets := 1 << uint(h.cdef_bits & 3)
	for i in 0 ..< num_presets {
		yp := int(h.cdef_y_pri_strength[i]) << coeff_shift
		ys := int(h.cdef_y_sec_strength[i]) << coeff_shift
		up := num_planes > 1 ? (int(h.cdef_uv_pri_strength[i]) << coeff_shift) : 0
		us := num_planes > 1 ? (int(h.cdef_uv_sec_strength[i]) << coeff_shift) : 0
		presets[i] = Preset{
			y_pri       = yp,
			y_sec       = ys,
			y_sec_damp  = ys > 0 ? uint(max(0, base_damping - floor_log2_u32(u32(ys)))) : 0,
			uv_pri      = up,
			uv_sec      = us,
			uv_pri_damp = up > 0 ? uint(max(0, uv_damping - floor_log2_u32(u32(up)))) : 0,
			uv_sec_damp = us > 0 ? uint(max(0, uv_damping - floor_log2_u32(u32(us)))) : 0,
			uv_pri_taps = CDEF_PRI_TAPS[(up >> coeff_shift) & 1],
		}
	}

	max_y_luma := mi_rows * MI_SIZE
	max_x_luma := mi_cols * MI_SIZE
	max_y_uv := max_y_luma >> sub_y
	max_x_uv := max_x_luma >> sub_x
	w_uv := 8 >> sub_x
	h_uv := 8 >> sub_y

	top_buf: [3][2][2][CDEF_MAX_LINE_W]T = ---
	left_baks: [3][8][2]i16 = ---
	prev_row_modified := false

	for r := 0; r < mi_rows; r += 2 {
		r_sb := (r >> 4) * sb64_cols
		row_has_cdef := false
		if cdef_enabled {
			for sb_c in 0 ..< sb64_cols {
				if cdef_idx_grid[r_sb + sb_c] >= 0 {
					row_has_cdef = true
					break
				}
			}
		}
		if !row_has_cdef && in_place {
			prev_row_modified = false
			continue
		}

		row0_off := r * mi_cols
		row1_off := min(r + 1, mi_rows - 1) * mi_cols
		cur_slot := (r >> 1) & 1
		prev_slot := cur_slot ~ 1

		if in_place && r + 2 < mi_rows {
			for p in 0 ..< num_planes {
				sx := p == 0 ? uint(0) : sub_x
				sy := p == 0 ? uint(0) : sub_y
				plane_w := (mi_cols * MI_SIZE) >> sx
				y_bot := ((r + 2) * MI_SIZE) >> sy
				stride := curr_strides[p]
				copy(top_buf[p][cur_slot][0][:plane_w], curr_planes[p][(y_bot - 2) * stride:][:plane_w])
				copy(top_buf[p][cur_slot][1][:plane_w], curr_planes[p][(y_bot - 1) * stride:][:plane_w])
			}
		}

		use_top_buf := in_place && prev_row_modified
		y0_luma := r * MI_SIZE
		luma_ptr := raw_data(curr_planes[0])
		top0_y := use_top_buf ? &top_buf[0][prev_slot][0][0] : (y0_luma >= 2 ? &luma_ptr[(y0_luma - 2) * curr_strides[0]] : nil)
		top1_y := use_top_buf ? &top_buf[0][prev_slot][1][0] : (y0_luma >= 1 ? &luma_ptr[(y0_luma - 1) * curr_strides[0]] : nil)
		y0_uv := (r * MI_SIZE) >> sub_y
		row_interior_y := y0_luma >= 2 && y0_luma + 10 <= max_y_luma
		row_interior_uv := y0_uv >= 2 && y0_uv + h_uv + 2 <= max_y_uv

		left_mod: [3]bool
		for c := 0; c < mi_cols; c += 2 {
			idx := !cdef_enabled ? -1 : int(cdef_idx_grid[r_sb + (c >> 4)])
			if idx < 0 && in_place && (c & 15) == 0 {
				left_mod = {}
				c += 14
				continue
			}
			skip := idx < 0 || (c + 1 < mi_cols ? ((intrinsics.unaligned_load((^u16)(&skips_ptr[row0_off + c])) &
				intrinsics.unaligned_load((^u16)(&skips_ptr[row1_off + c]))) == 0x0101) : (skips_ptr[row0_off + c] && skips_ptr[row1_off + c]))
			if skip {
				if in_place {
					left_mod = {}
				} else {
					for p in 0 ..< num_planes {
						sx := p == 0 ? uint(0) : sub_x
						sy := p == 0 ? uint(0) : sub_y
						start_y := (r * MI_SIZE) >> sy
						start_x := (c * MI_SIZE) >> sx
						w := (MI_SIZE * 2) >> sx
						h_blk := (MI_SIZE * 2) >> sy
						cdef_copy_block(cdef_planes[p], cdef_strides[p], curr_planes[p], curr_strides[p], start_x, start_y, w, h_blk)
					}
				}
				continue
			}

			pr := presets[idx & 7]
			y_pri := pr.y_pri
			y_sec := pr.y_sec
			uv_pri := pr.uv_pri
			uv_sec := pr.uv_sec

			x0_luma := c * MI_SIZE
			x0_uv := x0_luma >> sub_x
			interior_y := row_interior_y && x0_luma >= 2 && x0_luma + 10 <= max_x_luma
			interior_uv := num_planes > 1 && row_interior_uv && x0_uv >= 2 && x0_uv + w_uv + 2 <= max_x_uv

			// Performance note:
			// Populate `tmp_y` and `tmp_uv` BEFORE running `cdef_direction` (~45 cycles of pure ALU work).
			// This allows the CPU's Store Buffer to drain the 16B/32B `tmp` stores to L1D cache during
			// `cdef_direction`, eliminating the ~25-cycle Store-to-Load Forwarding (STLF) stall when
			// `cdef_filter_tmp_simd` / `cdef_interior_uv_4x4_filter_tmp` issue 64-byte `zmm` loads.
			tmp_y: [13 * 16]i16 = ---
			if interior_y && (y_pri != 0 || y_sec != 0) {
				cdef_interior_load_tmp(&tmp_y, luma_ptr, curr_strides[0], top0_y, top1_y, &left_baks[0], left_mod[0], in_place, x0_luma, y0_luma, 8, 8)
			}
			tmp_uv: [9 * 16]i16 = ---
			_ = tmp_uv
			when intrinsics.has_target_feature("avx512bw") {
				if interior_uv && w_uv == 4 && h_uv == 4 && (uv_pri != 0 || uv_sec != 0) {
					u_ptr := raw_data(curr_planes[1])
					v_ptr := raw_data(curr_planes[2])
					top0_u := use_top_buf ? &top_buf[1][prev_slot][0][0] : &u_ptr[(y0_uv - 2) * curr_strides[1]]
					top1_u := use_top_buf ? &top_buf[1][prev_slot][1][0] : &u_ptr[(y0_uv - 1) * curr_strides[1]]
					top0_v := use_top_buf ? &top_buf[2][prev_slot][0][0] : &v_ptr[(y0_uv - 2) * curr_strides[2]]
					top1_v := use_top_buf ? &top_buf[2][prev_slot][1][0] : &v_ptr[(y0_uv - 1) * curr_strides[2]]
					cdef_interior_uv_4x4_load_tmp(
						&tmp_uv, u_ptr, v_ptr, curr_strides[1], curr_strides[2],
						top0_u, top1_u, top0_v, top1_v,
						&left_baks[1], &left_baks[2], left_mod[1], in_place, x0_uv, y0_uv,
					)
				}
			}

			y_dir, var: int
			if y_pri > 0 || uv_pri > 0 {
				y_dir, var = cdef_direction(curr_planes[0], curr_strides[0], r, c, bit_depth)
			}
			dir := y_pri == 0 ? 0 : y_dir
			if y_pri > 0 {
				var_str := (var >> 6) != 0 ? min(floor_log2_u32(u32(var >> 6)), 12) : 0
				y_pri = var != 0 ? ((y_pri * (4 + var_str) + 8) >> 4) : 0
			}

			if y_pri == 0 && y_sec == 0 {
				if !in_place {
					cdef_copy_block(cdef_planes[0], cdef_strides[0], curr_planes[0], curr_strides[0], x0_luma, y0_luma, 8, 8)
				}
				left_mod[0] = false
			} else {
				y_pri_taps := CDEF_PRI_TAPS[(y_pri >> coeff_shift) & 1]
				y_pri_damp := y_pri > 0 ? uint(max(0, base_damping - floor_log2_u32(u32(y_pri)))) : 0
				if interior_y {
					cdef_filter_tmp_simd(
						cdef_planes[0], cdef_strides[0], &tmp_y,
						x0_luma, y0_luma, 8,
						y_pri, y_sec, y_pri_damp, pr.y_sec_damp, y_pri_taps[0], y_pri_taps[1], dir,
						8, 8, false,
					)
				} else {
					cdef_filter_block_border(
						cdef_planes[0], cdef_strides[0],
						curr_planes[0], curr_strides[0],
						top0_y, top1_y,
						&left_baks[0], left_mod[0], in_place,
						x0_luma, y0_luma, 8, 8, max_x_luma, max_y_luma,
						y_pri, y_sec, y_pri_damp, pr.y_sec_damp, y_pri_taps, dir,
					)
				}
				left_mod[0] = in_place
			}

			if num_planes == 1 do continue
			if uv_pri == 0 && uv_sec == 0 {
				if !in_place {
					for p in 1 ..= 2 {
						cdef_copy_block(cdef_planes[p], cdef_strides[p], curr_planes[p], curr_strides[p], x0_uv, y0_uv, w_uv, h_uv)
					}
				}
				left_mod[1] = false
				left_mod[2] = false
				continue
			}

			uv_dir := uv_pri == 0 ? 0 : int(CDEF_UV_DIR[sub_x][sub_y][y_dir])

			when intrinsics.has_target_feature("avx512bw") {
				if interior_uv && w_uv == 4 && h_uv == 4 {
					cdef_interior_uv_4x4_filter_tmp(
						cdef_planes[1], cdef_planes[2],
						cdef_strides[1], cdef_strides[2],
						&tmp_uv, x0_uv, y0_uv,
						uv_pri, uv_sec, pr.uv_pri_damp, pr.uv_sec_damp, pr.uv_pri_taps, uv_dir,
					)
					left_mod[1] = in_place
					left_mod[2] = in_place
					continue
				}
			}

			for p in 1 ..= 2 {
				uv_ptr := raw_data(curr_planes[p])
				top0_uv := use_top_buf ? &top_buf[p][prev_slot][0][0] : (y0_uv >= 2 ? &uv_ptr[(y0_uv - 2) * curr_strides[p]] : nil)
				top1_uv := use_top_buf ? &top_buf[p][prev_slot][1][0] : (y0_uv >= 1 ? &uv_ptr[(y0_uv - 1) * curr_strides[p]] : nil)
				if interior_uv {
					if w_uv == 4 && h_uv == 4 {
						cdef_interior_load_and_filter(
							cdef_planes[p], cdef_strides[p],
							uv_ptr, curr_strides[p],
							top0_uv, top1_uv,
							&left_baks[p], left_mod[p], in_place,
							x0_uv, y0_uv,
							uv_pri, uv_sec, pr.uv_pri_damp, pr.uv_sec_damp, pr.uv_pri_taps, uv_dir,
							4, 4,
						)
					} else if w_uv == 8 {
						cdef_interior_load_and_filter(
							cdef_planes[p], cdef_strides[p],
							uv_ptr, curr_strides[p],
							top0_uv, top1_uv,
							&left_baks[p], left_mod[p], in_place,
							x0_uv, y0_uv,
							uv_pri, uv_sec, pr.uv_pri_damp, pr.uv_sec_damp, pr.uv_pri_taps, uv_dir,
							8, 8,
						)
					} else {
						cdef_interior_load_and_filter(
							cdef_planes[p], cdef_strides[p],
							uv_ptr, curr_strides[p],
							top0_uv, top1_uv,
							&left_baks[p], left_mod[p], in_place,
							x0_uv, y0_uv,
							uv_pri, uv_sec, pr.uv_pri_damp, pr.uv_sec_damp, pr.uv_pri_taps, uv_dir,
							4, 8,
						)
					}
				} else {
					cdef_filter_block_border(
						cdef_planes[p], cdef_strides[p],
						curr_planes[p], curr_strides[p],
						top0_uv, top1_uv,
						&left_baks[p], left_mod[p], in_place,
						x0_uv, y0_uv, w_uv, h_uv, max_x_uv, max_y_uv,
						uv_pri, uv_sec, pr.uv_pri_damp, pr.uv_sec_damp, pr.uv_pri_taps, uv_dir,
					)
				}
				left_mod[p] = in_place
			}
		}
		prev_row_modified = true
	}
}
