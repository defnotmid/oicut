package av1

// Superres Upscaling (section 7.16) and Loop Restoration (Wiener + Self-Guided
// SGRPROJ, section 7.17) shared by the AV1 decoder and encoder.
// Uses an #soa slice for loop restoration units per plane.

import "base:intrinsics"
import "base:runtime"
import "core:simd"

@(private)
LR_SIMD_LANES :: max(8, runtime.NATIVE_SIMD_BIT_WIDTH / 32)

LR_Unit :: struct {
	restoration_type: u8,       // RESTORE_NONE, RESTORE_WIENER, RESTORE_SGRPROJ
	sgr_set:          u8,       // 0..15
	sgr_xqd:          [2]i8,    // [w0, w1]
	wiener:           [2][3]i8, // [0]=vfilter[0..2], [1]=hfilter[0..2]
}

// Section 5.11.57: count_units_in_frame.
count_units_in_frame :: #force_inline proc "contextless" (unit_size, frame_size: int) -> int {
	return max((frame_size + (unit_size >> 1)) / unit_size, 1)
}

// Section 7.16: Superres horizontal upscaling for a single plane.
upscale_plane :: proc "contextless" (
	dst: []$T,
	dst_stride: int,
	src: []T,
	src_stride: int,
	frame_width, upscaled_width, frame_height, mi_cols: int,
	sub_x, sub_y: uint,
	bit_depth := 8,
) #no_bounds_check {
	downscaled_w := int(round2(i64(frame_width), sub_x))
	upscaled_w := int(round2(i64(upscaled_width), sub_x))
	plane_h := int(round2(i64(frame_height), sub_y))
	max_val := (i32(1) << uint(bit_depth)) - 1

	step_x := ((downscaled_w << SUPERRES_SCALE_BITS) + (upscaled_w / 2)) / upscaled_w
	err := (upscaled_w * step_x) - (downscaled_w << SUPERRES_SCALE_BITS)
	initial_subpel_x := ((-((upscaled_w - downscaled_w) << (SUPERRES_SCALE_BITS - 1)) + upscaled_w / 2) / upscaled_w +
		(1 << (SUPERRES_EXTRA_BITS - 1)) - err / 2) & SUPERRES_SCALE_MASK

	mi_w := mi_cols >> sub_x
	max_x := mi_w * MI_SIZE - 1

	for y in 0 ..< plane_h {
		src_row := src[y * src_stride:]
		dst_row := dst[y * dst_stride:]
		for x in 0 ..< upscaled_w {
			src_x := -(1 << SUPERRES_SCALE_BITS) + initial_subpel_x + x * step_x
			src_x_px := src_x >> SUPERRES_SCALE_BITS
			src_x_subpel := (src_x & SUPERRES_SCALE_MASK) >> SUPERRES_EXTRA_BITS
			taps := UPSCALE_FILTER[src_x_subpel]
			sum := i32(0)
			#unroll for k in 0 ..< SUPERRES_FILTER_TAPS {
				sample_x := clamp(src_x_px + (k - SUPERRES_FILTER_OFFSET), 0, max_x)
				sum += i32(src_row[sample_x]) * i32(taps[k])
			}
			dst_row[x] = T(clamp(round2(i64(sum), FILTER_BITS), 0, max_val))
		}
	}
}

// Section 7.16: Upscaling process across all planes when `h.use_superres` is true.
upscale_frame :: proc "contextless" (
	dst_planes: [3][]$T,
	dst_strides: [3]int,
	src_planes: [3][]T,
	src_strides: [3]int,
	h: ^Frame_Header,
	num_planes: int,
	sub_x, sub_y: uint,
	bit_depth := 8,
) #no_bounds_check {
	for p in 0 ..< num_planes {
		sx := p == 0 ? uint(0) : sub_x
		sy := p == 0 ? uint(0) : sub_y
		if !h.use_superres {
			plane_w := int(round2(i64(h.upscaled_width), sx))
			plane_h := int(round2(i64(h.frame_height), sy))
			for y in 0 ..< plane_h {
				copy(dst_planes[p][y * dst_strides[p]:][:plane_w], src_planes[p][y * src_strides[p]:][:plane_w])
			}
		} else {
			upscale_plane(
				dst_planes[p], dst_strides[p],
				src_planes[p], src_strides[p],
				h.frame_width, h.upscaled_width, h.frame_height, h.mi_cols,
				sx, sy,
				bit_depth,
			)
		}
	}
}

// Section 7.17.5: Wiener coefficient process (expands 3 symmetric taps to 7 taps with DC gain 128).
wiener_coeffs :: #force_inline proc "contextless" (coeff: [3]i8) -> (filter: [7]i32) {
	filter[3] = 128
	#unroll for i in 0 ..< 3 {
		c := i32(coeff[i])
		filter[i] = c
		filter[6 - i] = c
		filter[3] -= 2 * c
	}
	return
}

@(private)
LR_MAX_W :: 384
@(private)
LR_MAX_H :: 64

// Fills a padded source strip `strip[0 ..< h + 6][0 ..< w + 6]` where `strip[r][c]`
// corresponds to `(x + c - 3, y + r - 3)` using section 7.17.6 rules.
@(private)
lr_fill_source_strip :: proc "contextless" (
	strip: ^[LR_MAX_H + 6][LR_MAX_W + 16]i32,
	cdef_plane: []$T,
	cdef_stride: int,
	curr_plane: []T,
	curr_stride: int,
	x, y, w, h: int,
	plane_end_x, plane_end_y: int,
	stripe_start_y, stripe_end_y: int,
) #no_bounds_check {
	N  :: LR_SIMD_LANES
	VN :: #simd[N]i32
	SN :: #simd[N]T
	cols := w + 6
	for r in 0 ..< h + 6 {
		sy := clamp(y + r - 3, 0, plane_end_y)
		src_row: []T
		if sy < stripe_start_y {
			sy = clamp(max(stripe_start_y - 2, sy), 0, plane_end_y)
			src_row = curr_plane[sy * curr_stride:]
		} else if sy > stripe_end_y {
			sy = clamp(min(stripe_end_y + 2, sy), 0, plane_end_y)
			src_row = curr_plane[sy * curr_stride:]
		} else {
			src_row = cdef_plane[sy * cdef_stride:]
		}
		dst_row := &strip[r]
		if x >= 3 && x + w + 2 <= plane_end_x {
			src_ptr := raw_data(src_row[x - 3:])
			c := 0
			for ; c + N <= cols; c += N {
				v := VN(intrinsics.unaligned_load((^SN)(&src_ptr[c])))
				intrinsics.unaligned_store((^VN)(&dst_row[c]), v)
			}
			for ; c < cols; c += 1 {
				dst_row[c] = i32(src_ptr[c])
			}
		} else {
			for c in 0 ..< cols {
				sx := clamp(x + c - 3, 0, plane_end_x)
				dst_row[c] = i32(src_row[sx])
			}
		}
	}
}

// Section 7.17.4: Wiener filter process for a block at `(x, y)` of size `w x h`.
wiener_filter_block :: proc "contextless" (
	lr_plane: []$T,
	lr_stride: int,
	cdef_plane: []T,
	cdef_stride: int,
	curr_plane: []T,
	curr_stride: int,
	x, y, w, h: int,
	plane_end_x, plane_end_y: int,
	stripe_start_y, stripe_end_y: int,
	v_Tap3, h_Tap3: [3]i8,
	bit_depth := 8,
) #no_bounds_check {
	N    :: LR_SIMD_LANES
	VN   :: #simd[N]i32
	UN   :: #simd[N]u32
	I16N :: #simd[N]i16
	VTN  :: #simd[N]T
	V8   :: #simd[8]i32
	U8   :: #simd[8]u32
	I16_8 :: #simd[8]i16
	VT8  :: #simd[8]T

	r0, r1, _ := inter_rounding_vars(false, bit_depth)
	r0_sh := r0 & 7
	r1_sh := r1 & 15
	half0_s := i32(1) << (r0_sh - 1)
	half1_s := i32(1) << (r1_sh - 1)
	vfilter := wiener_coeffs(v_Tap3)
	hfilter := wiener_coeffs(h_Tap3)
	offset := i32(1) << (uint(bit_depth) + uint(FILTER_BITS) - r0_sh - 1)
	limit := (i32(1) << (uint(bit_depth) + 1 + uint(FILTER_BITS) - r0_sh)) - 1
	min_inter := -offset
	max_inter := limit - offset
	max_val := (i32(1) << uint(bit_depth)) - 1

	// Horizontal Wiener filter outputs are clamped to `[min_inter, max_inter]`, which fits
	// within signed 16-bit `i16` for both 8-bit (`[-16384, 16383]`) and 10-bit (`[-32768, 32767]`).
	// Storing `intermediate` as `i16` instead of `i32` halves the stack footprint (56 KB vs 112 KB)
	// and memory bandwidth, and allows the 7-tap vertical pass to load `i16` rows directly into
	// `pack_i16_pairs` + `madd_i16_pairs` (`vpdpwssd` / `vpmaddwd`).
	intermediate: [LR_MAX_H + 6][LR_MAX_W + 16]i16 = ---
	row_buf: [LR_MAX_W + 16]T = ---
	is_interior_x := x >= 3 && x + w + 2 <= plane_end_x
	cols := w + 6

	hf01_8 := splat_i16_pair(i16(hfilter[0]), i16(hfilter[1]), 8)
	hf23_8 := splat_i16_pair(i16(hfilter[2]), i16(hfilter[3]), 8)
	half0_v := V8(half0_s)
	sh0_v := U8(u32(r0_sh))
	min_inter_v := V8(min_inter)
	max_inter_v := V8(max_inter)
	when N > 8 {
		hf01_n := splat_i16_pair(i16(hfilter[0]), i16(hfilter[1]), N)
		hf23_n := splat_i16_pair(i16(hfilter[2]), i16(hfilter[3]), N)
		half0_n := VN(half0_s)
		sh0_n := UN(u32(r0_sh))
		min_inter_n := VN(min_inter)
		max_inter_n := VN(max_inter)
	}

	for r in 0 ..< h + 6 {
		sy := clamp(y + r - 3, 0, plane_end_y)
		src_row: []T
		if sy < stripe_start_y {
			sy = clamp(max(stripe_start_y - 2, sy), 0, plane_end_y)
			src_row = curr_plane[sy * curr_stride:]
		} else if sy > stripe_end_y {
			sy = clamp(min(stripe_end_y + 2, sy), 0, plane_end_y)
			src_row = curr_plane[sy * curr_stride:]
		} else {
			src_row = cdef_plane[sy * cdef_stride:]
		}
		src_ptr: [^]T
		if is_interior_x {
			src_ptr = raw_data(src_row[x - 3:])
		} else {
			for c in 0 ..< cols {
				row_buf[c] = src_row[clamp(x + c - 3, 0, plane_end_x)]
			}
			src_ptr = raw_data(row_buf[:])
		}
		d_row := &intermediate[r]
		c := 0
		when N > 8 {
			for ; c + N <= w; c += N {
				t0 := I16N(intrinsics.unaligned_load((^VTN)(&src_ptr[c + 0]))) + I16N(intrinsics.unaligned_load((^VTN)(&src_ptr[c + 6])))
				t1 := I16N(intrinsics.unaligned_load((^VTN)(&src_ptr[c + 1]))) + I16N(intrinsics.unaligned_load((^VTN)(&src_ptr[c + 5])))
				t2 := I16N(intrinsics.unaligned_load((^VTN)(&src_ptr[c + 2]))) + I16N(intrinsics.unaligned_load((^VTN)(&src_ptr[c + 4])))
				t3 := I16N(intrinsics.unaligned_load((^VTN)(&src_ptr[c + 3])))
				s := half0_n + madd_i16_pairs(pack_i16_pairs(t0, t1), hf01_n) + madd_i16_pairs(pack_i16_pairs(t2, t3), hf23_n)
				v := I16N(simd.clamp(simd.shr_masked(s, sh0_n), min_inter_n, max_inter_n))
				intrinsics.unaligned_store((^I16N)(&d_row[c]), v)
			}
		}
		for ; c + 8 <= w; c += 8 {
			t0 := I16_8(intrinsics.unaligned_load((^VT8)(&src_ptr[c + 0]))) + I16_8(intrinsics.unaligned_load((^VT8)(&src_ptr[c + 6])))
			t1 := I16_8(intrinsics.unaligned_load((^VT8)(&src_ptr[c + 1]))) + I16_8(intrinsics.unaligned_load((^VT8)(&src_ptr[c + 5])))
			t2 := I16_8(intrinsics.unaligned_load((^VT8)(&src_ptr[c + 2]))) + I16_8(intrinsics.unaligned_load((^VT8)(&src_ptr[c + 4])))
			t3 := I16_8(intrinsics.unaligned_load((^VT8)(&src_ptr[c + 3])))
			s := half0_v + madd_i16_pairs(pack_i16_pairs(t0, t1), hf01_8) + madd_i16_pairs(pack_i16_pairs(t2, t3), hf23_8)
			v := I16_8(simd.clamp(simd.shr_masked(s, sh0_v), min_inter_v, max_inter_v))
			intrinsics.unaligned_store((^I16_8)(&d_row[c]), v)
		}
		for ; c < w; c += 1 {
			s := hfilter[0] * (i32(src_ptr[c + 0]) + i32(src_ptr[c + 6])) +
				hfilter[1] * (i32(src_ptr[c + 1]) + i32(src_ptr[c + 5])) +
				hfilter[2] * (i32(src_ptr[c + 2]) + i32(src_ptr[c + 4])) +
				hfilter[3] * i32(src_ptr[c + 3])
			v := (s + half0_s) >> r0_sh
			d_row[c] = i16(clamp(v, min_inter, max_inter))
		}
	}

	half1_v := V8(half1_s)
	sh1_v := U8(u32(r1_sh))
	zero_v := V8(0)
	max_val_v := V8(max_val)
	when T == u8 {
		vf01_8 := splat_i16_pair(i16(vfilter[0]), i16(vfilter[1]), 8)
		vf23_8 := splat_i16_pair(i16(vfilter[2]), i16(vfilter[3]), 8)
	} else {
		vf00_8 := splat_i16_pair(i16(vfilter[0]), i16(vfilter[0]), 8)
		vf11_8 := splat_i16_pair(i16(vfilter[1]), i16(vfilter[1]), 8)
		vf22_8 := splat_i16_pair(i16(vfilter[2]), i16(vfilter[2]), 8)
		vf3_8  := V8(vfilter[3])
	}
	when N > 8 {
		half1_n := VN(half1_s)
		sh1_n := UN(u32(r1_sh))
		zero_n := VN(0)
		max_val_n := VN(max_val)
		when T == u8 {
			vf01_n := splat_i16_pair(i16(vfilter[0]), i16(vfilter[1]), N)
			vf23_n := splat_i16_pair(i16(vfilter[2]), i16(vfilter[3]), N)
		} else {
			vf00_n := splat_i16_pair(i16(vfilter[0]), i16(vfilter[0]), N)
			vf11_n := splat_i16_pair(i16(vfilter[1]), i16(vfilter[1]), N)
			vf22_n := splat_i16_pair(i16(vfilter[2]), i16(vfilter[2]), N)
			vf3_n  := VN(vfilter[3])
		}
	}

	for r in 0 ..< h {
		out_row := raw_data(lr_plane[(y + r) * lr_stride + x:])
		r0_ptr := &intermediate[r + 0]
		r1_ptr := &intermediate[r + 1]
		r2_ptr := &intermediate[r + 2]
		r3_ptr := &intermediate[r + 3]
		r4_ptr := &intermediate[r + 4]
		r5_ptr := &intermediate[r + 5]
		r6_ptr := &intermediate[r + 6]
		c := 0
		when N > 8 {
			for ; c + N <= w; c += N {
				m0 := intrinsics.unaligned_load((^I16N)(&r0_ptr[c]))
				m1 := intrinsics.unaligned_load((^I16N)(&r1_ptr[c]))
				m2 := intrinsics.unaligned_load((^I16N)(&r2_ptr[c]))
				m3 := intrinsics.unaligned_load((^I16N)(&r3_ptr[c]))
				m4 := intrinsics.unaligned_load((^I16N)(&r4_ptr[c]))
				m5 := intrinsics.unaligned_load((^I16N)(&r5_ptr[c]))
				m6 := intrinsics.unaligned_load((^I16N)(&r6_ptr[c]))
				when T == u8 {
					s := half1_n + madd_i16_pairs(pack_i16_pairs(m0 + m6, m1 + m5), vf01_n) +
						madd_i16_pairs(pack_i16_pairs(m2 + m4, m3), vf23_n)
				} else {
					s := half1_n + madd_i16_pairs(pack_i16_pairs(m0, m6), vf00_n) +
						madd_i16_pairs(pack_i16_pairs(m1, m5), vf11_n) +
						madd_i16_pairs(pack_i16_pairs(m2, m4), vf22_n) + vf3_n * VN(m3)
				}
				v := simd.clamp(simd.shr_masked(s, sh1_n), zero_n, max_val_n)
				intrinsics.unaligned_store((^VTN)(&out_row[c]), VTN(v))
			}
		}
		for ; c + 8 <= w; c += 8 {
			m0 := intrinsics.unaligned_load((^I16_8)(&r0_ptr[c]))
			m1 := intrinsics.unaligned_load((^I16_8)(&r1_ptr[c]))
			m2 := intrinsics.unaligned_load((^I16_8)(&r2_ptr[c]))
			m3 := intrinsics.unaligned_load((^I16_8)(&r3_ptr[c]))
			m4 := intrinsics.unaligned_load((^I16_8)(&r4_ptr[c]))
			m5 := intrinsics.unaligned_load((^I16_8)(&r5_ptr[c]))
			m6 := intrinsics.unaligned_load((^I16_8)(&r6_ptr[c]))
			when T == u8 {
				s := half1_v + madd_i16_pairs(pack_i16_pairs(m0 + m6, m1 + m5), vf01_8) +
					madd_i16_pairs(pack_i16_pairs(m2 + m4, m3), vf23_8)
			} else {
				s := half1_v + madd_i16_pairs(pack_i16_pairs(m0, m6), vf00_8) +
					madd_i16_pairs(pack_i16_pairs(m1, m5), vf11_8) +
					madd_i16_pairs(pack_i16_pairs(m2, m4), vf22_8) + vf3_8 * V8(m3)
			}
			v := simd.clamp(simd.shr_masked(s, sh1_v), zero_v, max_val_v)
			intrinsics.unaligned_store((^VT8)(&out_row[c]), VT8(v))
		}
		for ; c < w; c += 1 {
			s := vfilter[0] * (i32(r0_ptr[c]) + i32(r6_ptr[c])) +
				vfilter[1] * (i32(r1_ptr[c]) + i32(r5_ptr[c])) +
				vfilter[2] * (i32(r2_ptr[c]) + i32(r4_ptr[c])) +
				vfilter[3] * i32(r3_ptr[c])
			v := (s + half1_s) >> r1_sh
			out_row[c] = T(clamp(v, 0, max_val))
		}
	}
}

// Precomputed `((z << SGRPROJ_SGR_BITS) + (z / 2)) / (z + 1)` table for `z in 0..=255`
// (Section 7.17.3). Eliminates the per-pixel 64-bit integer division (`idiv`) in `box_pass`.
@(private, rodata)
SGR_X_BY_XPLUS1 := [256]u16{
	1, 128, 171, 192, 205, 213, 219, 224, 228, 230, 233, 235, 236, 238, 239, 240,
	241, 242, 243, 243, 244, 244, 245, 245, 246, 246, 247, 247, 247, 247, 248, 248,
	248, 248, 249, 249, 249, 249, 249, 250, 250, 250, 250, 250, 250, 250, 251, 251,
	251, 251, 251, 251, 251, 251, 251, 251, 252, 252, 252, 252, 252, 252, 252, 252,
	252, 252, 252, 252, 252, 252, 252, 252, 252, 253, 253, 253, 253, 253, 253, 253,
	253, 253, 253, 253, 253, 253, 253, 253, 253, 253, 253, 253, 253, 253, 253, 253,
	253, 253, 253, 253, 253, 253, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254,
	254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254,
	254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254,
	254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254,
	254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 255, 255, 255, 255, 255, 255,
	255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
	255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
	255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
	255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
	255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 256,
}

// Section 7.17.2 & 7.17.3: Self-guided filter and box filter processes for a block at `(x, y)` of size `w x h`.
sgrproj_filter_block :: proc "contextless" (
	lr_plane: []$T,
	lr_stride: int,
	cdef_plane: []T,
	cdef_stride: int,
	curr_plane: []T,
	curr_stride: int,
	x, y, w, h: int,
	plane_end_x, plane_end_y: int,
	stripe_start_y, stripe_end_y: int,
	set: u8,
	xqd: [2]i8,
	bit_depth := 8,
) #no_bounds_check {
	N  :: LR_SIMD_LANES
	V8 :: #simd[N]i32
	U8 :: #simd[N]u32
	S8 :: #simd[N]T
	flt0: [LR_MAX_H][LR_MAX_W]i32 = ---
	flt1: [LR_MAX_H][LR_MAX_W]i32 = ---

	params := SGR_PARAMS[set]
	r0 := int(params[0])
	r1 := int(params[2])

	strip: [LR_MAX_H + 6][LR_MAX_W + 16]i32 = ---
	sq_strip: [LR_MAX_H + 6][LR_MAX_W + 16]i32 = ---
	if r0 != 0 || r1 != 0 {
		lr_fill_source_strip(
			&strip,
			cdef_plane, cdef_stride,
			curr_plane, curr_stride,
			x, y, w, h,
			plane_end_x, plane_end_y,
			stripe_start_y, stripe_end_y,
		)
		cols := w + 6
		for r in 0 ..< h + 6 {
			s_row := &strip[r]
			q_row := &sq_strip[r]
			c := 0
			for ; c + N <= cols; c += N {
				v := intrinsics.unaligned_load((^V8)(&s_row[c]))
				intrinsics.unaligned_store((^V8)(&q_row[c]), v * v)
			}
			for ; c < cols; c += 1 {
				v := s_row[c]
				q_row[c] = v * v
			}
		}
	}

	box_pass :: proc "contextless" (
		flt: ^[LR_MAX_H][LR_MAX_W]i32,
		strip: ^[LR_MAX_H + 6][LR_MAX_W + 16]i32,
		sq_strip: ^[LR_MAX_H + 6][LR_MAX_W + 16]i32,
		cdef_plane: []T,
		cdef_stride: int,
		x, y, w, h: int,
		r, eps, pass, bit_depth: int,
	) #no_bounds_check {
		if r == 0 do return
		// A and B cover i in -1 ..< h + 1 (at index i + 1 in 0 ..< h + 2) and
		// j in -1 ..< w + 1 (at index j + 1 in 0 ..< w + 2).
		A: [LR_MAX_H + 2][LR_MAX_W + 2]i32 = ---
		B: [LR_MAX_H + 2][LR_MAX_W + 2]i32 = ---
		col_a: [LR_MAX_W + 16]i32 = ---
		col_b: [LR_MAX_W + 16]i32 = ---

		n := i32((2 * r + 1) * (2 * r + 1))
		n2e := int(n) * int(n) * eps
		s := i32(((1 << SGRPROJ_MTABLE_BITS) + n2e / 2) / n2e)
		// Clamping `p` to `max_p` before multiplying by `s` prevents 32-bit signed overflow
		// while preserving 100% bit-exactness since any `p >= max_p` produces `z >= 255`.
		max_p := (i32(255) << SGRPROJ_MTABLE_BITS) / s + 1
		one_over_n := i32(((1 << SGRPROJ_RECIP_BITS) + (int(n) / 2)) / int(n))
		bd_minus_8 := uint(bit_depth - 8)
		a_sh := (2 * bd_minus_8) & 7
		d_sh := bd_minus_8 & 7
		a_half := a_sh == 0 ? i32(0) : (i32(1) << (a_sh - 1))
		d_half := d_sh == 0 ? i32(0) : (i32(1) << (d_sh - 1))
		diam := 2 * r + 1
		cols_needed := w + 2 + 2 * r
		col_start := 2 - r
		col_end := col_start + cols_needed

		v_n := V8(n)
		v_s := V8(s)
		v_max_p := V8(max_p)
		v_zero := V8(0)
		v_255 := V8(255)
		v_256 := V8(1 << SGRPROJ_SGR_BITS)
		v_one_over_n := V8(one_over_n)
		v_a_half := V8(a_half)
		v_d_half := V8(d_half)
		v_a_sh := U8(u32(a_sh))
		v_d_sh := U8(u32(d_sh))
		v_m_half := V8(1 << (SGRPROJ_MTABLE_BITS - 1))
		v_m_sh := U8(SGRPROJ_MTABLE_BITS)
		v_r_half := V8(1 << (SGRPROJ_RECIP_BITS - 1))
		v_r_sh := U8(SGRPROJ_RECIP_BITS)

		for i := -1; i < h + 1; i += 1 {
			if pass == 0 && (i & 1) == 0 do continue // even rows of A/B have weight 0 in pass 0
			base_r := i + 3 - r
			if diam == 5 {
				q0, q1, q2, q3, q4 := &sq_strip[base_r], &sq_strip[base_r + 1], &sq_strip[base_r + 2], &sq_strip[base_r + 3], &sq_strip[base_r + 4]
				s0, s1, s2, s3, s4 := &strip[base_r], &strip[base_r + 1], &strip[base_r + 2], &strip[base_r + 3], &strip[base_r + 4]
				c := col_start
				for ; c + N <= col_end; c += N {
					va := intrinsics.unaligned_load((^V8)(&q0[c])) + intrinsics.unaligned_load((^V8)(&q1[c])) +
						intrinsics.unaligned_load((^V8)(&q2[c])) + intrinsics.unaligned_load((^V8)(&q3[c])) +
						intrinsics.unaligned_load((^V8)(&q4[c]))
					vb := intrinsics.unaligned_load((^V8)(&s0[c])) + intrinsics.unaligned_load((^V8)(&s1[c])) +
						intrinsics.unaligned_load((^V8)(&s2[c])) + intrinsics.unaligned_load((^V8)(&s3[c])) +
						intrinsics.unaligned_load((^V8)(&s4[c]))
					intrinsics.unaligned_store((^V8)(&col_a[c]), va)
					intrinsics.unaligned_store((^V8)(&col_b[c]), vb)
				}
				for ; c < col_end; c += 1 {
					col_a[c] = q0[c] + q1[c] + q2[c] + q3[c] + q4[c]
					col_b[c] = s0[c] + s1[c] + s2[c] + s3[c] + s4[c]
				}
			} else {
				q0, q1, q2 := &sq_strip[base_r], &sq_strip[base_r + 1], &sq_strip[base_r + 2]
				s0, s1, s2 := &strip[base_r], &strip[base_r + 1], &strip[base_r + 2]
				c := col_start
				for ; c + N <= col_end; c += N {
					va := intrinsics.unaligned_load((^V8)(&q0[c])) + intrinsics.unaligned_load((^V8)(&q1[c])) + intrinsics.unaligned_load((^V8)(&q2[c]))
					vb := intrinsics.unaligned_load((^V8)(&s0[c])) + intrinsics.unaligned_load((^V8)(&s1[c])) + intrinsics.unaligned_load((^V8)(&s2[c]))
					intrinsics.unaligned_store((^V8)(&col_a[c]), va)
					intrinsics.unaligned_store((^V8)(&col_b[c]), vb)
				}
				for ; c < col_end; c += 1 {
					col_a[c] = q0[c] + q1[c] + q2[c]
					col_b[c] = s0[c] + s1[c] + s2[c]
				}
			}

			a_row := &A[i + 1]
			b_row := &B[i + 1]
			idx := 0
			for ; idx + N <= w + 2; idx += N {
				at := col_start + idx
				va_win: V8
				vb_win: V8
				if diam == 5 {
					va_win = intrinsics.unaligned_load((^V8)(&col_a[at])) + intrinsics.unaligned_load((^V8)(&col_a[at + 1])) +
						intrinsics.unaligned_load((^V8)(&col_a[at + 2])) + intrinsics.unaligned_load((^V8)(&col_a[at + 3])) +
						intrinsics.unaligned_load((^V8)(&col_a[at + 4]))
					vb_win = intrinsics.unaligned_load((^V8)(&col_b[at])) + intrinsics.unaligned_load((^V8)(&col_b[at + 1])) +
						intrinsics.unaligned_load((^V8)(&col_b[at + 2])) + intrinsics.unaligned_load((^V8)(&col_b[at + 3])) +
						intrinsics.unaligned_load((^V8)(&col_b[at + 4]))
				} else {
					va_win = intrinsics.unaligned_load((^V8)(&col_a[at])) + intrinsics.unaligned_load((^V8)(&col_a[at + 1])) +
						intrinsics.unaligned_load((^V8)(&col_a[at + 2]))
					vb_win = intrinsics.unaligned_load((^V8)(&col_b[at])) + intrinsics.unaligned_load((^V8)(&col_b[at + 1])) +
						intrinsics.unaligned_load((^V8)(&col_b[at + 2]))
				}
				va_r := simd.shr_masked(va_win + v_a_half, v_a_sh)
				vd_r := simd.shr_masked(vb_win + v_d_half, v_d_sh)
				vp := simd.clamp(va_r * v_n - vd_r * vd_r, v_zero, v_max_p)
				vz := simd.min(simd.shr_masked(vp * v_s + v_m_half, v_m_sh), v_255)
				za := transmute([N]i32)vz
				a2_arr: [N]i32 = ---
				#unroll for l in 0 ..< N {
					a2_arr[l] = i32(SGR_X_BY_XPLUS1[za[l]])
				}
				va2 := transmute(V8)a2_arr
				vb2 := simd.shr_masked((v_256 - va2) * vb_win * v_one_over_n + v_r_half, v_r_sh)
				intrinsics.unaligned_store((^V8)(&a_row[idx]), va2)
				intrinsics.unaligned_store((^V8)(&b_row[idx]), vb2)
			}
			for ; idx < w + 2; idx += 1 {
				at := col_start + idx
				a_win := col_a[at] + col_a[at + 1] + col_a[at + 2]
				b_win := col_b[at] + col_b[at + 1] + col_b[at + 2]
				if diam == 5 {
					a_win += col_a[at + 3] + col_a[at + 4]
					b_win += col_b[at + 3] + col_b[at + 4]
				}
				a_r := (a_win + a_half) >> a_sh
				d_r := (b_win + d_half) >> d_sh
				p := clamp(a_r * n - d_r * d_r, 0, max_p)
				z := min(int((p * s + (1 << (SGRPROJ_MTABLE_BITS - 1))) >> SGRPROJ_MTABLE_BITS), 255)
				a2 := i32(SGR_X_BY_XPLUS1[z])
				b2 := ((1 << SGRPROJ_SGR_BITS) - a2) * b_win * one_over_n
				a_row[idx] = a2
				b_row[idx] = (b2 + (1 << (SGRPROJ_RECIP_BITS - 1))) >> SGRPROJ_RECIP_BITS
			}
		}

		sh_even :: uint(SGRPROJ_SGR_BITS) + 5 - uint(SGRPROJ_RST_BITS)
		half_even :: i32(1) << (sh_even - 1)
		sh_odd :: uint(SGRPROJ_SGR_BITS) + 4 - uint(SGRPROJ_RST_BITS)
		half_odd :: i32(1) << (sh_odd - 1)
		v_five := V8(5)
		v_six := V8(6)
		v_three := V8(3)
		v_four := V8(4)
		v_half_even := V8(half_even)
		v_sh_even := U8(sh_even)
		v_half_odd := V8(half_odd)
		v_sh_odd := U8(sh_odd)

		if pass == 0 {
			for i in 0 ..< h {
				cdef_row := cdef_plane[(y + i) * cdef_stride + x:]
				f_row := &flt[i]
				if (i & 1) == 0 {
					// Even i: i-1 and i+1 are odd rows (rows i and i+2 in A/B), row i+1 has weight 0.
					a_up, a_dn := &A[i], &A[i + 2]
					b_up, b_dn := &B[i], &B[i + 2]
					j := 0
					for ; j + N <= w; j += N {
						a_sum := v_five * (intrinsics.unaligned_load((^V8)(&a_up[j])) + intrinsics.unaligned_load((^V8)(&a_up[j + 2])) +
							intrinsics.unaligned_load((^V8)(&a_dn[j])) + intrinsics.unaligned_load((^V8)(&a_dn[j + 2]))) +
							v_six * (intrinsics.unaligned_load((^V8)(&a_up[j + 1])) + intrinsics.unaligned_load((^V8)(&a_dn[j + 1])))
						b_sum := v_five * (intrinsics.unaligned_load((^V8)(&b_up[j])) + intrinsics.unaligned_load((^V8)(&b_up[j + 2])) +
							intrinsics.unaligned_load((^V8)(&b_dn[j])) + intrinsics.unaligned_load((^V8)(&b_dn[j + 2]))) +
							v_six * (intrinsics.unaligned_load((^V8)(&b_up[j + 1])) + intrinsics.unaligned_load((^V8)(&b_dn[j + 1])))
						cv := V8(intrinsics.unaligned_load((^S8)(&cdef_row[j])))
						v := simd.shr_masked(a_sum * cv + b_sum + v_half_even, v_sh_even)
						intrinsics.unaligned_store((^V8)(&f_row[j]), v)
					}
					for ; j < w; j += 1 {
						a_sum := 5 * (a_up[j] + a_up[j + 2] + a_dn[j] + a_dn[j + 2]) + 6 * (a_up[j + 1] + a_dn[j + 1])
						b_sum := 5 * (b_up[j] + b_up[j + 2] + b_dn[j] + b_dn[j + 2]) + 6 * (b_up[j + 1] + b_dn[j + 1])
						v := a_sum * i32(cdef_row[j]) + b_sum
						f_row[j] = (v + half_even) >> sh_even
					}
				} else {
					// Odd i: only row i (index i+1 in A/B) is odd; i-1 and i+1 have weight 0.
					a_cur := &A[i + 1]
					b_cur := &B[i + 1]
					j := 0
					for ; j + N <= w; j += N {
						a_sum := v_five * (intrinsics.unaligned_load((^V8)(&a_cur[j])) + intrinsics.unaligned_load((^V8)(&a_cur[j + 2]))) +
							v_six * intrinsics.unaligned_load((^V8)(&a_cur[j + 1]))
						b_sum := v_five * (intrinsics.unaligned_load((^V8)(&b_cur[j])) + intrinsics.unaligned_load((^V8)(&b_cur[j + 2]))) +
							v_six * intrinsics.unaligned_load((^V8)(&b_cur[j + 1]))
						cv := V8(intrinsics.unaligned_load((^S8)(&cdef_row[j])))
						v := simd.shr_masked(a_sum * cv + b_sum + v_half_odd, v_sh_odd)
						intrinsics.unaligned_store((^V8)(&f_row[j]), v)
					}
					for ; j < w; j += 1 {
						a_sum := 5 * (a_cur[j] + a_cur[j + 2]) + 6 * a_cur[j + 1]
						b_sum := 5 * (b_cur[j] + b_cur[j + 2]) + 6 * b_cur[j + 1]
						v := a_sum * i32(cdef_row[j]) + b_sum
						f_row[j] = (v + half_odd) >> sh_odd
					}
				}
			}
		} else {
			for i in 0 ..< h {
				cdef_row := cdef_plane[(y + i) * cdef_stride + x:]
				f_row := &flt[i]
				a_up, a_md, a_dn := &A[i], &A[i + 1], &A[i + 2]
				b_up, b_md, b_dn := &B[i], &B[i + 1], &B[i + 2]
				j := 0
				for ; j + N <= w; j += N {
					a_sum := v_three * (intrinsics.unaligned_load((^V8)(&a_up[j])) + intrinsics.unaligned_load((^V8)(&a_up[j + 2])) +
						intrinsics.unaligned_load((^V8)(&a_dn[j])) + intrinsics.unaligned_load((^V8)(&a_dn[j + 2]))) +
						v_four * (intrinsics.unaligned_load((^V8)(&a_up[j + 1])) + intrinsics.unaligned_load((^V8)(&a_md[j])) +
						intrinsics.unaligned_load((^V8)(&a_md[j + 1])) + intrinsics.unaligned_load((^V8)(&a_md[j + 2])) +
						intrinsics.unaligned_load((^V8)(&a_dn[j + 1])))
					b_sum := v_three * (intrinsics.unaligned_load((^V8)(&b_up[j])) + intrinsics.unaligned_load((^V8)(&b_up[j + 2])) +
						intrinsics.unaligned_load((^V8)(&b_dn[j])) + intrinsics.unaligned_load((^V8)(&b_dn[j + 2]))) +
						v_four * (intrinsics.unaligned_load((^V8)(&b_up[j + 1])) + intrinsics.unaligned_load((^V8)(&b_md[j])) +
						intrinsics.unaligned_load((^V8)(&b_md[j + 1])) + intrinsics.unaligned_load((^V8)(&b_md[j + 2])) +
						intrinsics.unaligned_load((^V8)(&b_dn[j + 1])))
					cv := V8(intrinsics.unaligned_load((^S8)(&cdef_row[j])))
					v := simd.shr_masked(a_sum * cv + b_sum + v_half_even, v_sh_even)
					intrinsics.unaligned_store((^V8)(&f_row[j]), v)
				}
				for ; j < w; j += 1 {
					a_sum := 3 * (a_up[j] + a_up[j + 2] + a_dn[j] + a_dn[j + 2]) +
						4 * (a_up[j + 1] + a_md[j] + a_md[j + 1] + a_md[j + 2] + a_dn[j + 1])
					b_sum := 3 * (b_up[j] + b_up[j + 2] + b_dn[j] + b_dn[j + 2]) +
						4 * (b_up[j + 1] + b_md[j] + b_md[j + 1] + b_md[j + 2] + b_dn[j + 1])
					v := a_sum * i32(cdef_row[j]) + b_sum
					f_row[j] = (v + half_even) >> sh_even
				}
			}
		}
	}

	if r0 != 0 {
		box_pass(
			&flt0, &strip, &sq_strip,
			cdef_plane, cdef_stride,
			x, y, w, h,
			r0, int(params[1]), 0, bit_depth,
		)
	}
	if r1 != 0 {
		box_pass(
			&flt1, &strip, &sq_strip,
			cdef_plane, cdef_stride,
			x, y, w, h,
			r1, int(params[3]), 1, bit_depth,
		)
	}

	w0 := i32(xqd[0])
	w1 := i32(xqd[1])
	w2 := (i32(1) << SGRPROJ_PRJ_BITS) - w0 - w1
	max_val := (i32(1) << uint(bit_depth)) - 1
	sh_final :: uint(SGRPROJ_RST_BITS + SGRPROJ_PRJ_BITS)
	half_final :: i32(1) << (sh_final - 1)
	v_w0 := V8(w0)
	v_w1 := V8(w1)
	v_w2 := V8(w2)
	v_half_final := V8(half_final)
	v_sh_final := U8(sh_final)
	v_sh_rst := U8(SGRPROJ_RST_BITS)
	v_zero := V8(0)
	v_max := V8(max_val)

	for i in 0 ..< h {
		cdef_row := cdef_plane[(y + i) * cdef_stride + x:]
		out_row := lr_plane[(y + i) * lr_stride + x:]
		f0_row := &flt0[i]
		f1_row := &flt1[i]
		j := 0
		for ; j + N <= w; j += N {
			u := simd.shl_masked(V8(intrinsics.unaligned_load((^S8)(&cdef_row[j]))), v_sh_rst)
			f0 := r0 != 0 ? intrinsics.unaligned_load((^V8)(&f0_row[j])) : u
			f1 := r1 != 0 ? intrinsics.unaligned_load((^V8)(&f1_row[j])) : u
			v := v_w1 * u + v_w0 * f0 + v_w2 * f1
			s := simd.clamp(simd.shr_masked(v + v_half_final, v_sh_final), v_zero, v_max)
			intrinsics.unaligned_store((^S8)(&out_row[j]), S8(s))
		}
		for ; j < w; j += 1 {
			u := i32(cdef_row[j]) << SGRPROJ_RST_BITS
			v := w1 * u
			v += w0 * (r0 != 0 ? f0_row[j] : u)
			v += w2 * (r1 != 0 ? f1_row[j] : u)
			s := (v + half_final) >> sh_final
			out_row[j] = T(clamp(s, 0, max_val))
		}
	}
}

// Section 7.17: Loop restoration process across all planes.
//
// Performance note:
// Supports in-place execution (`raw_data(lr_planes[p]) == raw_data(cdef_planes[p])`).
// When running in-place, `RESTORE_NONE` planes and `RESTORE_NONE` units become zero-cost no-ops
// (avoiding full-frame `3.1 MB` allocations and `memcpy`s at 1080p+), and active `RESTORE_WIENER` /
// `RESTORE_SGRPROJ` units preserve the 3 unfiltered CDEF columns `[x + bw - 3 ..< x + bw]` needed
// by their right-hand neighbor in a tiny 192-byte stack buffer (`left_bak: [LR_MAX_H][3]T`).
loop_restore_frame :: proc(
	lr_planes: [3][]$T,
	lr_strides: [3]int,
	cdef_planes: [3][]T,
	cdef_strides: [3]int,
	curr_planes: [3][]T,
	curr_strides: [3]int,
	units: [3]#soa[]LR_Unit,
	h: ^Frame_Header,
	num_planes: int,
	sub_x, sub_y: uint,
	bit_depth := 8,
) #no_bounds_check {
	if !h.uses_lr || h.allow_intrabc {
		for p in 0 ..< num_planes {
			if raw_data(lr_planes[p]) == raw_data(cdef_planes[p]) do continue
			sx := p == 0 ? uint(0) : sub_x
			sy := p == 0 ? uint(0) : sub_y
			plane_w := int(round2(i64(h.upscaled_width), sx))
			plane_h := int(round2(i64(h.frame_height), sy))
			for y in 0 ..< plane_h {
				copy(lr_planes[p][y * lr_strides[p]:][:plane_w], cdef_planes[p][y * cdef_strides[p]:][:plane_w])
			}
		}
		return
	}

	for p in 0 ..< num_planes {
		in_place := raw_data(lr_planes[p]) == raw_data(cdef_planes[p])
		sx := p == 0 ? uint(0) : sub_x
		sy := p == 0 ? uint(0) : sub_y
		plane_w := int(round2(i64(h.upscaled_width), sx))
		plane_h := int(round2(i64(h.frame_height), sy))
		if h.frame_restoration_type[p] == RESTORE_NONE {
			if !in_place {
				for y in 0 ..< plane_h {
					copy(lr_planes[p][y * lr_strides[p]:][:plane_w], cdef_planes[p][y * cdef_strides[p]:][:plane_w])
				}
			}
			continue
		}

		plane_end_x := plane_w - 1
		plane_end_y := plane_h - 1
		unit_size := int(h.loop_restoration_size[p])
		unit_rows := count_units_in_frame(unit_size, plane_end_y + 1)
		unit_cols := count_units_in_frame(unit_size, plane_end_x + 1)
		p_units := units[p]

		// Copy any trailing right/bottom margin not covered by MI_SIZE steps if dimensions are odd.
		covered_w := ((h.upscaled_width >> MI_SIZE_LOG2) * MI_SIZE) >> sx
		covered_h := ((h.frame_height >> MI_SIZE_LOG2) * MI_SIZE) >> sy
		if !in_place && (covered_w < plane_w || covered_h < plane_h) {
			for y in 0 ..< plane_h {
				copy(lr_planes[p][y * lr_strides[p]:][:plane_w], cdef_planes[p][y * cdef_strides[p]:][:plane_w])
			}
		}

		left_bak: [LR_MAX_H][3]T = ---
		next_bak: [LR_MAX_H][3]T = ---
		saved_lr: [LR_MAX_H][3]T = ---

		for luma_y := 0; luma_y < h.frame_height; {
			row := luma_y >> MI_SIZE_LOG2
			stripe_num := (luma_y + 8) / 64
			stripe_start_y := (-8 + stripe_num * 64) >> sy
			stripe_end_y := stripe_start_y + (64 >> sy) - 1
			unit_row := min(unit_rows - 1, ((row * MI_SIZE + 8) >> sy) / unit_size)
			y := (row * MI_SIZE) >> sy

			end_luma_y := luma_y + MI_SIZE
			for end_luma_y < h.frame_height {
				next_row := end_luma_y >> MI_SIZE_LOG2
				next_stripe := (end_luma_y + 8) / 64
				next_unit_row := min(unit_rows - 1, ((next_row * MI_SIZE + 8) >> sy) / unit_size)
				next_y := (next_row * MI_SIZE) >> sy
				if next_stripe != stripe_num || next_unit_row != unit_row || next_y - y >= LR_MAX_H do break
				end_luma_y += MI_SIZE
			}
			max_span_y := ((end_luma_y >> MI_SIZE_LOG2) * MI_SIZE) >> sy
			bh := min(max_span_y - y, plane_end_y - y + 1)
			luma_y = end_luma_y
			if bh <= 0 do continue

			left_mod := false
			// Process horizontal spans belonging to each unit_col along this stripe.
			for luma_x := 0; luma_x < h.upscaled_width; {
				col := luma_x >> MI_SIZE_LOG2
				x := (col * MI_SIZE) >> sx
				unit_col := min(unit_cols - 1, x / unit_size)
				// Advance luma_x while within the same restoration unit.
				end_luma_x := luma_x + MI_SIZE
				for end_luma_x < h.upscaled_width {
					next_col := end_luma_x >> MI_SIZE_LOG2
					next_x := (next_col * MI_SIZE) >> sx
					if min(unit_cols - 1, next_x / unit_size) != unit_col || next_x - x >= 256 do break
					end_luma_x += MI_SIZE
				}
				max_span_x := ((end_luma_x >> MI_SIZE_LOG2) * MI_SIZE) >> sx
				bw := min(max_span_x - x, plane_end_x - x + 1)
				luma_x = end_luma_x
				if bw <= 0 do continue

				u_idx := unit_row * unit_cols + unit_col
				rtype := p_units[u_idx].restoration_type
				if rtype == RESTORE_NONE {
					if !in_place {
						for ry in 0 ..< bh {
							copy(
								lr_planes[p][(y + ry) * lr_strides[p] + x:][:bw],
								cdef_planes[p][(y + ry) * cdef_strides[p] + x:][:bw],
							)
						}
					}
					left_mod = false
					continue
				}

				save_right := in_place && (x + bw < plane_w) && bw >= 3
				if save_right {
					rx := x + bw - 3
					for ry in 0 ..< bh {
						row_p := cdef_planes[p][(y + ry) * cdef_strides[p] + rx:]
						next_bak[ry] = {row_p[0], row_p[1], row_p[2]}
					}
				}
				if left_mod {
					lx := x - 3
					for ry in 0 ..< bh {
						row_p := cdef_planes[p][(y + ry) * cdef_strides[p] + lx:]
						saved_lr[ry] = {row_p[0], row_p[1], row_p[2]}
						row_p[0] = left_bak[ry][0]
						row_p[1] = left_bak[ry][1]
						row_p[2] = left_bak[ry][2]
					}
				}

				if rtype == RESTORE_WIENER {
					wiener_filter_block(
						lr_planes[p], lr_strides[p],
						cdef_planes[p], cdef_strides[p],
						curr_planes[p], curr_strides[p],
						x, y, bw, bh,
						plane_end_x, plane_end_y,
						stripe_start_y, stripe_end_y,
						p_units[u_idx].wiener[0], p_units[u_idx].wiener[1],
						bit_depth,
					)
				} else if rtype == RESTORE_SGRPROJ {
					sgrproj_filter_block(
						lr_planes[p], lr_strides[p],
						cdef_planes[p], cdef_strides[p],
						curr_planes[p], curr_strides[p],
						x, y, bw, bh,
						plane_end_x, plane_end_y,
						stripe_start_y, stripe_end_y,
						p_units[u_idx].sgr_set, p_units[u_idx].sgr_xqd,
						bit_depth,
					)
				}

				if left_mod {
					lx := x - 3
					for ry in 0 ..< bh {
						row_p := lr_planes[p][(y + ry) * lr_strides[p] + lx:]
						row_p[0] = saved_lr[ry][0]
						row_p[1] = saved_lr[ry][1]
						row_p[2] = saved_lr[ry][2]
					}
				}
				if save_right {
					for ry in 0 ..< bh {
						left_bak[ry] = next_bak[ry]
					}
					left_mod = true
				} else {
					left_mod = false
				}
			}
		}
	}
}
