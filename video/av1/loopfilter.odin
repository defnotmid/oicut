package av1

// Deblocking Loop Filter (section 7.14) shared by the AV1 decoder and encoder.
// Uses an #soa per-4x4 cell grid and 4-lane #simd[4]i32 vectorization across
// each 4-sample edge segment.

import "base:intrinsics"
import "core:simd"

_ :: intrinsics
_ :: simd

// Per-4x4 cell loop filter parameters stored in an #soa slice.
LF_Cell :: struct {
	levels:     [4]u8, // [0]=Y vert, [1]=Y horz, [2]=U, [3]=V (section 7.14.5)
	tx_size_y:  u8,    // LoopfilterTxSizes[0][row][col]
	tx_size_uv: u8,    // LoopfilterTxSizes[1..2][row >> sub_y][col >> sub_x]
	mi_size:    u8,    // MiSizes[row][col]
	skip_inter: bool,  // Skips[row][col] && RefFrames[row][col][0] > INTRA_FRAME
}

LF_Limits :: [MAX_LOOP_FILTER + 1]struct {
	limit, blimit, thresh: i32,
}

// Precomputes (limit, blimit, thresh) for all filter levels 0..63 (section 7.14.4).
compute_lf_limits :: proc "contextless" (sharpness: u8) -> (limits: LF_Limits) {
	shift: uint = sharpness > 4 ? 2 : (sharpness > 0 ? 1 : 0)
	max_lim := i32(9 - int(sharpness))
	for lvl in 0 ..= MAX_LOOP_FILTER {
		lv := i32(lvl)
		lim := sharpness > 0 ? clamp(lv >> shift, 1, max_lim) : max(1, lv >> shift)
		limits[lvl] = {
			limit  = lim,
			blimit = 2 * (lv + 2) + lim,
			thresh = lv >> 4,
		}
	}
	return
}

// Section 7.14.5: Adaptive filter strength selection process for a single (plane, pass) index
// `lvl_idx = (plane == 0) ? pass : (plane + 1)` in 0..3.
compute_lf_level :: proc "contextless" (
	h: ^Frame_Header,
	segment_id: u8,
	ref_frame0: i8,
	y_mode: u8,
	delta_lf: [FRAME_LF_COUNT]i8,
	lvl_idx: int,
) -> u8 {
	base_lvl := int(h.loop_filter_level[lvl_idx])
	if base_lvl == 0 do return 0
	dlf := int(h.delta_lf_multi ? delta_lf[lvl_idx] : delta_lf[0])
	lvl_seg := clamp(base_lvl + dlf, 0, MAX_LOOP_FILTER)

	feature := uint(SEG_LVL_ALT_LF_Y_V + lvl_idx)
	if h.segmentation_enabled && (h.feature_enabled[segment_id] & (1 << feature)) != 0 {
		lvl_seg = clamp(lvl_seg + int(h.feature_data[segment_id][feature]), 0, MAX_LOOP_FILTER)
	}

	if h.loop_filter_delta_enabled {
		n_shift := uint(lvl_seg >> 5)
		ref := max(int(INTRA_FRAME), int(ref_frame0))
		if ref == INTRA_FRAME {
			lvl_seg += int(h.loop_filter_ref_deltas[INTRA_FRAME]) << n_shift
		} else {
			mode_type := (y_mode >= NEARESTMV && y_mode != GLOBALMV && y_mode != GLOBAL_GLOBALMV) ? 1 : 0
			lvl_seg += (int(h.loop_filter_ref_deltas[ref]) << n_shift) + (int(h.loop_filter_mode_deltas[mode_type]) << n_shift)
		}
		lvl_seg = clamp(lvl_seg, 0, MAX_LOOP_FILTER)
	}
	return u8(lvl_seg)
}

// Computes all 4 directional/plane filter levels for a block.
compute_lf_levels :: proc "contextless" (
	h: ^Frame_Header,
	segment_id: u8,
	ref_frame0: i8,
	y_mode: u8,
	delta_lf: [FRAME_LF_COUNT]i8,
) -> (levels: [4]u8) {
	#unroll for i in 0 ..< 4 {
		levels[i] = compute_lf_level(h, segment_id, ref_frame0, y_mode, delta_lf, i)
	}
	return
}

// Section 5.11.38: get_plane_residual_size.
get_plane_residual_size :: #force_inline proc "contextless" (subsize, plane: int, sub_x, sub_y: uint) -> int {
	sx := plane > 0 ? sub_x : 0
	sy := plane > 0 ? sub_y : 0
	return int(SUBSAMPLED_SIZE[subsize][sx][sy])
}

// Section 7.14.1: Applies the deblocking loop filter in place across all active planes.
loop_filter_frame :: proc(
	planes: [3][]$T,
	strides: [3]int,
	cells: #soa[]LF_Cell,
	h: ^Frame_Header,
	num_planes: int,
	sub_x, sub_y: uint,
	bit_depth := 8,
) #no_bounds_check {
	if h.loop_filter_level[0] == 0 && h.loop_filter_level[1] == 0 do return
	limits := compute_lf_limits(h.loop_filter_sharpness)
	mi_cols := h.mi_cols
	row_limit := min(h.mi_rows, (h.frame_height + MI_SIZE - 1) >> MI_SIZE_LOG2)
	col_limit := min(h.mi_cols, (h.frame_width + MI_SIZE - 1) >> MI_SIZE_LOG2)

	lf_levels := cells.levels
	lf_mi_sizes := cells.mi_size
	lf_skip_inter := cells.skip_inter

	for plane in 0 ..< num_planes {
		if plane > 0 && h.loop_filter_level[1 + plane] == 0 do continue
		sx := plane == 0 ? uint(0) : sub_x
		sy := plane == 0 ? uint(0) : sub_y
		row_step := 1 << sy
		col_step := 1 << sx
		p_slice := planes[plane]
		stride := strides[plane]
		lf_tx := plane == 0 ? cells.tx_size_y : cells.tx_size_uv
		max_fsize := plane == 0 ? 16 : 8

		// Pass 0: Vertical edges (dx = 1, dy = 0).
		if plane > 0 || h.loop_filter_level[0] != 0 {
			lvl_idx := plane == 0 ? 0 : (plane + 1)
			for row := 0; row < row_limit; row += row_step {
				yp := (row * MI_SIZE) >> sy
				row_base := (row | int(sy)) * mi_cols
				for col := 0; col < col_limit; {
					xp := (col * MI_SIZE) >> sx
					curr_idx := row_base + (col | int(sx))

					tx_sz := int(lf_tx[curr_idx])
					tx_dim := int(TX_WIDTH[tx_sz])
					step_col := ((tx_dim - (xp & (tx_dim - 1))) >> 2) << sx
					if col > 0 && (xp & (tx_dim - 1)) == 0 {
						prev_idx := curr_idx - col_step
						lvl := lf_levels[curr_idx][lvl_idx]
						if lvl == 0 {
							lvl = lf_levels[prev_idx][lvl_idx]
						}
						if lvl != 0 {
							skip_edge := false
							if lf_skip_inter[curr_idx] {
								mi_size := int(lf_mi_sizes[curr_idx])
								plane_size := int(SUBSAMPLED_SIZE[mi_size][sx][sy])
								blk_dim := int(BLOCK_WIDTH[plane_size])
								if (xp & (blk_dim - 1)) != 0 {
									skip_edge = true
									step_col = ((blk_dim - (xp & (blk_dim - 1))) >> 2) << sx
								}
							}
							if !skip_edge {
								prev_tx_dim := int(TX_WIDTH[lf_tx[prev_idx]])
								filter_size := min(max_fsize, min(prev_tx_dim, tx_dim))
								lim := limits[lvl]
								if bit_depth <= 10 {
									filter_edge_4_vert(p_slice, stride, xp, yp, plane, filter_size, lim.limit, lim.blimit, lim.thresh, bit_depth, i16)
								} else {
									filter_edge_4_vert(p_slice, stride, xp, yp, plane, filter_size, lim.limit, lim.blimit, lim.thresh, bit_depth, i32)
								}
							}
						}
					}
					col += step_col
				}
			}
		}

		// Pass 1: Horizontal edges (dx = 0, dy = 1).
		if plane > 0 || h.loop_filter_level[1] != 0 {
			lvl_idx := plane == 0 ? 1 : (plane + 1)
			prev_row_off := mi_cols << sy
			aligned_col_limit := (col_limit + col_step - 1) & ~(col_step - 1)
			max_xp := (aligned_col_limit * MI_SIZE) >> sx
			for row := row_step; row < row_limit; row += row_step {
				yp := (row * MI_SIZE) >> sy
				row_base := (row | int(sy)) * mi_cols
				for col := 0; col < col_limit; {
					xp := (col * MI_SIZE) >> sx
					curr_idx := row_base + (col | int(sx))

					tx_sz := int(lf_tx[curr_idx])
					tx_w := int(TX_WIDTH[tx_sz])
					rem_cur_w := tx_w - (xp & (tx_w - 1))
					tx_dim := int(TX_HEIGHT[tx_sz])
					if (yp & (tx_dim - 1)) != 0 {
						col += max(col_step, (rem_cur_w >> 2) << sx)
						continue
					}

					if lf_skip_inter[curr_idx] {
						mi_size := int(lf_mi_sizes[curr_idx])
						plane_size := int(SUBSAMPLED_SIZE[mi_size][sx][sy])
						blk_dim := int(BLOCK_HEIGHT[plane_size])
						if (yp & (blk_dim - 1)) != 0 {
							blk_w := int(BLOCK_WIDTH[plane_size])
							rem_blk_w := blk_w - (xp & (blk_w - 1))
							col += max(col_step, (rem_blk_w >> 2) << sx)
							continue
						}
					}

					prev_idx := curr_idx - prev_row_off
					prev_tx_sz := int(lf_tx[prev_idx])
					prev_tx_w := int(TX_WIDTH[prev_tx_sz])
					rem_prev_w := prev_tx_w - (xp & (prev_tx_w - 1))
					span_w := min(min(rem_cur_w, rem_prev_w), max_xp - xp)

					lvl := lf_levels[curr_idx][lvl_idx]
					if lvl == 0 {
						lvl = lf_levels[prev_idx][lvl_idx]
						if lvl == 0 {
							col += max(col_step, (span_w >> 2) << sx)
							continue
						}
					}

					prev_tx_dim := int(TX_HEIGHT[prev_tx_sz])
					filter_size := min(max_fsize, min(prev_tx_dim, tx_dim))
					lim := limits[lvl]
					dx := 0
					if bit_depth <= 10 {
						when intrinsics.has_target_feature("avx2") {
							for ; dx + 16 <= span_w; dx += 16 {
								filter_edge_horz(p_slice, stride, xp + dx, yp, plane, filter_size, lim.limit, lim.blimit, lim.thresh, bit_depth, 16, i16)
							}
						}
						for ; dx + 8 <= span_w; dx += 8 {
							filter_edge_horz(p_slice, stride, xp + dx, yp, plane, filter_size, lim.limit, lim.blimit, lim.thresh, bit_depth, 8, i16)
						}
						for ; dx < span_w; dx += 4 {
							filter_edge_horz(p_slice, stride, xp + dx, yp, plane, filter_size, lim.limit, lim.blimit, lim.thresh, bit_depth, 4, i16)
						}
					} else {
						when intrinsics.has_target_feature("avx2") {
							for ; dx + 8 <= span_w; dx += 8 {
								filter_edge_horz(p_slice, stride, xp + dx, yp, plane, filter_size, lim.limit, lim.blimit, lim.thresh, bit_depth, 8, i32)
							}
						}
						for ; dx < span_w; dx += 4 {
							filter_edge_horz(p_slice, stride, xp + dx, yp, plane, filter_size, lim.limit, lim.blimit, lim.thresh, bit_depth, 4, i32)
						}
					}
					col += max(col_step, (span_w >> 2) << sx)
				}
			}
		}
	}
}

@(private)
lf_transpose_4x4 :: #force_inline proc "contextless" (r0, r1, r2, r3: #simd[4]$E) -> (c0, c1, c2, c3: #simd[4]E) {
	t0 := simd.shuffle(r0, r1, 0, 4, 1, 5)
	t1 := simd.shuffle(r2, r3, 0, 4, 1, 5)
	t2 := simd.shuffle(r0, r1, 2, 6, 3, 7)
	t3 := simd.shuffle(r2, r3, 2, 6, 3, 7)
	c0 = simd.shuffle(t0, t1, 0, 1, 4, 5)
	c1 = simd.shuffle(t0, t1, 2, 3, 6, 7)
	c2 = simd.shuffle(t2, t3, 0, 1, 4, 5)
	c3 = simd.shuffle(t2, t3, 2, 3, 6, 7)
	return
}

// Section 7.14.6.3: L-lane SIMD narrow 4-tap filter.
@(private)
lf_narrow_simd :: #force_inline proc "contextless" (
	p1, p0, q0, q1: #simd[$L]$E,
	dp10, dq10, thresh_bd: #simd[L]E,
	filter_mask: $M,
	bit_depth: int,
) -> (op1, op0, oq0, oq1: #simd[L]E) {
	V :: #simd[L]E
	U :: #simd[L]u16 when E == i16 else #simd[L]u32

	bd_shift := uint(bit_depth - 8)
	hev_mask := simd.lanes_gt(dp10, thresh_bd) | simd.lanes_gt(dq10, thresh_bd)

	offset := V(E(0x80) << bd_shift)
	clamp_lo := V(-(E(1) << uint(bit_depth - 1)))
	clamp_hi := V((E(1) << uint(bit_depth - 1)) - 1)

	ps1 := p1 - offset
	ps0 := p0 - offset
	qs0 := q0 - offset
	qs1 := q1 - offset

	flt_hev := simd.clamp(ps1 - qs1, clamp_lo, clamp_hi)
	flt := simd.select(hev_mask, flt_hev, V(0))
	flt = simd.clamp(flt + V(3) * (qs0 - ps0), clamp_lo, clamp_hi)
	flt1 := simd.shr(simd.clamp(flt + V(4), clamp_lo, clamp_hi), U(3))
	flt2 := simd.shr(simd.clamp(flt + V(3), clamp_lo, clamp_hi), U(3))

	n_oq0 := simd.clamp(qs0 - flt1, clamp_lo, clamp_hi) + offset
	n_op0 := simd.clamp(ps0 + flt2, clamp_lo, clamp_hi) + offset
	flt_half := simd.shr(flt1 + V(1), U(1))
	n_oq1 := simd.select(hev_mask, q1, simd.clamp(qs1 - flt_half, clamp_lo, clamp_hi) + offset)
	n_op1 := simd.select(hev_mask, p1, simd.clamp(ps1 + flt_half, clamp_lo, clamp_hi) + offset)

	op1 = simd.select(filter_mask, n_op1, p1)
	op0 = simd.select(filter_mask, n_op0, p0)
	oq0 = simd.select(filter_mask, n_oq0, q0)
	oq1 = simd.select(filter_mask, n_oq1, q1)
	return
}

// Section 7.14.6.4: L-lane SIMD 14-tap luma wide filter using sliding-window sum.
@(private)
lf_wide16_simd :: #force_inline proc "contextless" (
	p6, p5, p4, p3, p2, p1, p0: #simd[$L]$E,
	q0, q1, q2, q3, q4, q5, q6: #simd[L]E,
	op2_in, op1_in, op0_in, oq0_in, oq1_in, oq2_in: #simd[L]E,
	use_wide16: $M,
) -> (
	op5, op4, op3, op2, op1, op0: #simd[L]E,
	oq0, oq1, oq2, oq3, oq4, oq5: #simd[L]E,
) {
	V :: #simd[L]E
	U :: #simd[L]u16 when E == i16 else #simd[L]u32

	// Sliding window W_i = 8 + sum_{j=-6..6} s[clamp(i+j, -7, 6)] for i = -6..5.
	w := V(8) + p6 * V(6) + p5 + p4 + p3 + p2 + p1 + p0 + q0
	op5 = simd.select(use_wide16, simd.shr(w + p6 + p5 + p4, U(4)), p5)

	w = w - p6 + q1
	op4 = simd.select(use_wide16, simd.shr(w + p5 + p4 + p3, U(4)), p4)

	w = w - p6 + q2
	op3 = simd.select(use_wide16, simd.shr(w + p4 + p3 + p2, U(4)), p3)

	w = w - p6 + q3
	op2 = simd.select(use_wide16, simd.shr(w + p3 + p2 + p1, U(4)), op2_in)

	w = w - p6 + q4
	op1 = simd.select(use_wide16, simd.shr(w + p2 + p1 + p0, U(4)), op1_in)

	w = w - p6 + q5
	op0 = simd.select(use_wide16, simd.shr(w + p1 + p0 + q0, U(4)), op0_in)

	w = w - p6 + q6
	oq0 = simd.select(use_wide16, simd.shr(w + p0 + q0 + q1, U(4)), oq0_in)

	w = w - p5 + q6
	oq1 = simd.select(use_wide16, simd.shr(w + q0 + q1 + q2, U(4)), oq1_in)

	w = w - p4 + q6
	oq2 = simd.select(use_wide16, simd.shr(w + q1 + q2 + q3, U(4)), oq2_in)

	w = w - p3 + q6
	oq3 = simd.select(use_wide16, simd.shr(w + q2 + q3 + q4, U(4)), q3)

	w = w - p2 + q6
	oq4 = simd.select(use_wide16, simd.shr(w + q3 + q4 + q5, U(4)), q4)

	w = w - p1 + q6
	oq5 = simd.select(use_wide16, simd.shr(w + q4 + q5 + q6, U(4)), q5)
	return
}

// L-lane SIMD horizontal boundary filter (pass == 1, dx = 0, dy = 1: L contiguous columns at row yp).
@(private)
filter_edge_horz :: proc(
	plane_buf: []$T,
	stride, xp, yp, plane, filter_size: int,
	limit, blimit, thresh: i32,
	bit_depth: int,
	$L: int,
	$E: typeid,
) #no_bounds_check {
	V :: #simd[L]E
	U :: #simd[L]u16 when E == i16 else #simd[L]u32
	VT :: #simd[L]T

	load_row :: #force_inline proc "contextless" (buf: []T, idx: int) -> V #no_bounds_check {
		return V(intrinsics.unaligned_load((^VT)(&buf[idx])))
	}
	store_row :: #force_inline proc "contextless" (buf: []T, idx: int, v: V) #no_bounds_check {
		intrinsics.unaligned_store((^VT)(&buf[idx]), VT(v))
	}

	bd_shift := uint(bit_depth - 8)
	limit_bd := V(E(limit << bd_shift))
	blimit_bd := V(E(blimit << bd_shift))

	base := yp * stride + xp
	p1 := load_row(plane_buf, base - 2 * stride)
	p0 := load_row(plane_buf, base - 1 * stride)
	q0 := load_row(plane_buf, base + 0 * stride)
	q1 := load_row(plane_buf, base + 1 * stride)

	dp10 := simd.abs(p1 - p0)
	dq10 := simd.abs(q1 - q0)
	f_bad := simd.lanes_gt(dp10, limit_bd) |
		simd.lanes_gt(dq10, limit_bd) |
		simd.lanes_gt(simd.abs(p0 - q0) * V(2) + simd.shr(simd.abs(p1 - q1), U(1)), blimit_bd)
	if simd.reduce_or(~f_bad) == 0 do return

	thresh_bd := V(E(thresh << bd_shift))
	if filter_size == 4 {
		filter_mask := ~f_bad
		op1, op0, oq0, oq1 := lf_narrow_simd(p1, p0, q0, q1, dp10, dq10, thresh_bd, filter_mask, bit_depth)
		store_row(plane_buf, base - 2 * stride, op1)
		store_row(plane_buf, base - 1 * stride, op0)
		store_row(plane_buf, base + 0 * stride, oq0)
		store_row(plane_buf, base + 1 * stride, oq1)
		return
	}

	p2 := load_row(plane_buf, base - 3 * stride)
	q2 := load_row(plane_buf, base + 2 * stride)
	f_bad |= simd.lanes_gt(simd.abs(p2 - p1), limit_bd) | simd.lanes_gt(simd.abs(q2 - q1), limit_bd)

	p3, q3: V
	if plane == 0 {
		p3 = load_row(plane_buf, base - 4 * stride)
		q3 = load_row(plane_buf, base + 3 * stride)
		f_bad |= simd.lanes_gt(simd.abs(p3 - p2), limit_bd) | simd.lanes_gt(simd.abs(q3 - q2), limit_bd)
	}
	filter_mask := ~f_bad
	if simd.reduce_or(filter_mask) == 0 do return

	op1, op0, oq0, oq1 := lf_narrow_simd(p1, p0, q0, q1, dp10, dq10, thresh_bd, filter_mask, bit_depth)

	flat_thresh := V(E(1) << bd_shift)
	flat_bad := simd.lanes_gt(dp10, flat_thresh) |
		simd.lanes_gt(dq10, flat_thresh) |
		simd.lanes_gt(simd.abs(p2 - p0), flat_thresh) |
		simd.lanes_gt(simd.abs(q2 - q0), flat_thresh)
	if plane == 0 {
		flat_bad |= simd.lanes_gt(simd.abs(p3 - p0), flat_thresh) | simd.lanes_gt(simd.abs(q3 - q0), flat_thresh)
	}
	use_wide8 := filter_mask & (~flat_bad)
	if simd.reduce_or(use_wide8) == 0 {
		store_row(plane_buf, base - 2 * stride, op1)
		store_row(plane_buf, base - 1 * stride, op0)
		store_row(plane_buf, base + 0 * stride, oq0)
		store_row(plane_buf, base + 1 * stride, oq1)
		return
	}

	if plane != 0 {
		w_op1 := simd.shr(p2 * V(3) + p1 * V(2) + p0 * V(2) + q0 + V(4), U(3))
		w_op0 := simd.shr(p2 + p1 * V(2) + p0 * V(2) + q0 * V(2) + q1 + V(4), U(3))
		w_oq0 := simd.shr(p1 + p0 * V(2) + q0 * V(2) + q1 * V(2) + q2 + V(4), U(3))
		w_oq1 := simd.shr(p0 + q0 * V(2) + q1 * V(2) + q2 * V(3) + V(4), U(3))
		store_row(plane_buf, base - 2 * stride, simd.select(use_wide8, w_op1, op1))
		store_row(plane_buf, base - 1 * stride, simd.select(use_wide8, w_op0, op0))
		store_row(plane_buf, base + 0 * stride, simd.select(use_wide8, w_oq0, oq0))
		store_row(plane_buf, base + 1 * stride, simd.select(use_wide8, w_oq1, oq1))
		return
	}

	mid4 := p1 + p0 + q0 + q1 + V(4)
	w8_op2 := simd.shr(p3 * V(3) + p2 * V(2) + p1 + p0 + q0 + V(4), U(3))
	w8_op1 := simd.shr(p3 * V(2) + p2 + p1 + mid4, U(3))
	w8_op0 := simd.shr(p3 + p2 + p0 + q2 + mid4, U(3))
	w8_oq0 := simd.shr(p2 + q0 + q2 + q3 + mid4, U(3))
	w8_oq1 := simd.shr(q1 + q2 + q3 * V(2) + mid4, U(3))
	w8_oq2 := simd.shr(p0 + q0 + q1 + q2 * V(2) + q3 * V(3) + V(4), U(3))

	op2 := simd.select(use_wide8, w8_op2, p2)
	op1 = simd.select(use_wide8, w8_op1, op1)
	op0 = simd.select(use_wide8, w8_op0, op0)
	oq0 = simd.select(use_wide8, w8_oq0, oq0)
	oq1 = simd.select(use_wide8, w8_oq1, oq1)
	oq2 := simd.select(use_wide8, w8_oq2, q2)

	if filter_size >= 16 {
		p6 := load_row(plane_buf, base - 7 * stride)
		p5 := load_row(plane_buf, base - 6 * stride)
		p4 := load_row(plane_buf, base - 5 * stride)
		q4 := load_row(plane_buf, base + 4 * stride)
		q5 := load_row(plane_buf, base + 5 * stride)
		q6 := load_row(plane_buf, base + 6 * stride)

		flat2_bad := simd.lanes_gt(simd.abs(p6 - p0), flat_thresh) |
			simd.lanes_gt(simd.abs(q6 - q0), flat_thresh) |
			simd.lanes_gt(simd.abs(p5 - p0), flat_thresh) |
			simd.lanes_gt(simd.abs(q5 - q0), flat_thresh) |
			simd.lanes_gt(simd.abs(p4 - p0), flat_thresh) |
			simd.lanes_gt(simd.abs(q4 - q0), flat_thresh)
		use_wide16 := use_wide8 & (~flat2_bad)
		if simd.reduce_or(use_wide16) != 0 {
			op5, op4, op3, oq3, oq4, oq5: V
			op5, op4, op3, op2, op1, op0, oq0, oq1, oq2, oq3, oq4, oq5 = lf_wide16_simd(
				p6, p5, p4, p3, p2, p1, p0,
				q0, q1, q2, q3, q4, q5, q6,
				op2, op1, op0, oq0, oq1, oq2,
				use_wide16,
			)
			store_row(plane_buf, base - 6 * stride, op5)
			store_row(plane_buf, base - 5 * stride, op4)
			store_row(plane_buf, base - 4 * stride, op3)
			store_row(plane_buf, base - 3 * stride, op2)
			store_row(plane_buf, base - 2 * stride, op1)
			store_row(plane_buf, base - 1 * stride, op0)
			store_row(plane_buf, base + 0 * stride, oq0)
			store_row(plane_buf, base + 1 * stride, oq1)
			store_row(plane_buf, base + 2 * stride, oq2)
			store_row(plane_buf, base + 3 * stride, oq3)
			store_row(plane_buf, base + 4 * stride, oq4)
			store_row(plane_buf, base + 5 * stride, oq5)
			return
		}
	}

	store_row(plane_buf, base - 3 * stride, op2)
	store_row(plane_buf, base - 2 * stride, op1)
	store_row(plane_buf, base - 1 * stride, op0)
	store_row(plane_buf, base + 0 * stride, oq0)
	store_row(plane_buf, base + 1 * stride, oq1)
	store_row(plane_buf, base + 2 * stride, oq2)
}

// 4-lane SIMD vertical boundary filter (pass == 0, dx = 1, dy = 0: 4 rows at column xp).
@(private)
filter_edge_4_vert :: proc(
	plane_buf: []$T,
	stride, xp, yp, plane, filter_size: int,
	limit, blimit, thresh: i32,
	bit_depth: int,
	$E: typeid,
) #no_bounds_check {
	V4 :: #simd[4]E
	U4 :: #simd[4]u16 when E == i16 else #simd[4]u32
	VT :: #simd[4]T

	load_row :: #force_inline proc "contextless" (buf: []T, idx: int) -> V4 #no_bounds_check {
		return V4(intrinsics.unaligned_load((^VT)(&buf[idx])))
	}
	store_row :: #force_inline proc "contextless" (buf: []T, idx: int, v: V4) #no_bounds_check {
		intrinsics.unaligned_store((^VT)(&buf[idx]), VT(v))
	}
	gather_col :: #force_inline proc "contextless" (buf: []T, base, stride, dx: int) -> V4 #no_bounds_check {
		return V4{
			E(buf[base + 0 * stride + dx]),
			E(buf[base + 1 * stride + dx]),
			E(buf[base + 2 * stride + dx]),
			E(buf[base + 3 * stride + dx]),
		}
	}
	scatter_col :: #force_inline proc "contextless" (buf: []T, base, stride, dx: int, v: V4) #no_bounds_check {
		arr := transmute([4]E)v
		buf[base + 0 * stride + dx] = T(arr[0])
		buf[base + 1 * stride + dx] = T(arr[1])
		buf[base + 2 * stride + dx] = T(arr[2])
		buf[base + 3 * stride + dx] = T(arr[3])
	}
	store_4x4 :: #force_inline proc "contextless" (buf: []T, base, stride, dx: int, c0, c1, c2, c3: V4) #no_bounds_check {
		r0, r1, r2, r3 := lf_transpose_4x4(c0, c1, c2, c3)
		store_row(buf, base + 0 * stride + dx, r0)
		store_row(buf, base + 1 * stride + dx, r1)
		store_row(buf, base + 2 * stride + dx, r2)
		store_row(buf, base + 3 * stride + dx, r3)
	}

	bd_shift := uint(bit_depth - 8)
	limit_bd := V4(E(limit << bd_shift))
	blimit_bd := V4(E(blimit << bd_shift))

	base := yp * stride + xp
	p1, p0, q0, q1 := lf_transpose_4x4(
		load_row(plane_buf, base + 0 * stride - 2),
		load_row(plane_buf, base + 1 * stride - 2),
		load_row(plane_buf, base + 2 * stride - 2),
		load_row(plane_buf, base + 3 * stride - 2),
	)

	dp10 := simd.abs(p1 - p0)
	dq10 := simd.abs(q1 - q0)
	f_bad := simd.lanes_gt(dp10, limit_bd) |
		simd.lanes_gt(dq10, limit_bd) |
		simd.lanes_gt(simd.abs(p0 - q0) * V4(2) + simd.shr(simd.abs(p1 - q1), U4(1)), blimit_bd)
	if simd.reduce_or(~f_bad) == 0 do return

	thresh_bd := V4(E(thresh << bd_shift))
	if filter_size == 4 {
		filter_mask := ~f_bad
		op1, op0, oq0, oq1 := lf_narrow_simd(p1, p0, q0, q1, dp10, dq10, thresh_bd, filter_mask, bit_depth)
		store_4x4(plane_buf, base, stride, -2, op1, op0, oq0, oq1)
		return
	}

	p2 := gather_col(plane_buf, base, stride, -3)
	q2 := gather_col(plane_buf, base, stride, 2)
	f_bad |= simd.lanes_gt(simd.abs(p2 - p1), limit_bd) | simd.lanes_gt(simd.abs(q2 - q1), limit_bd)

	p3, q3: V4
	if plane == 0 {
		p3 = gather_col(plane_buf, base, stride, -4)
		q3 = gather_col(plane_buf, base, stride, 3)
		f_bad |= simd.lanes_gt(simd.abs(p3 - p2), limit_bd) | simd.lanes_gt(simd.abs(q3 - q2), limit_bd)
	}
	filter_mask := ~f_bad
	if simd.reduce_or(filter_mask) == 0 do return

	op1, op0, oq0, oq1 := lf_narrow_simd(p1, p0, q0, q1, dp10, dq10, thresh_bd, filter_mask, bit_depth)

	flat_thresh := V4(E(1) << bd_shift)
	flat_bad := simd.lanes_gt(dp10, flat_thresh) |
		simd.lanes_gt(dq10, flat_thresh) |
		simd.lanes_gt(simd.abs(p2 - p0), flat_thresh) |
		simd.lanes_gt(simd.abs(q2 - q0), flat_thresh)
	if plane == 0 {
		flat_bad |= simd.lanes_gt(simd.abs(p3 - p0), flat_thresh) | simd.lanes_gt(simd.abs(q3 - q0), flat_thresh)
	}
	use_wide8 := filter_mask & (~flat_bad)
	if simd.reduce_or(use_wide8) == 0 {
		store_4x4(plane_buf, base, stride, -2, op1, op0, oq0, oq1)
		return
	}

	if plane != 0 {
		w_op1 := simd.shr(p2 * V4(3) + p1 * V4(2) + p0 * V4(2) + q0 + V4(4), U4(3))
		w_op0 := simd.shr(p2 + p1 * V4(2) + p0 * V4(2) + q0 * V4(2) + q1 + V4(4), U4(3))
		w_oq0 := simd.shr(p1 + p0 * V4(2) + q0 * V4(2) + q1 * V4(2) + q2 + V4(4), U4(3))
		w_oq1 := simd.shr(p0 + q0 * V4(2) + q1 * V4(2) + q2 * V4(3) + V4(4), U4(3))
		store_4x4(
			plane_buf, base, stride, -2,
			simd.select(use_wide8, w_op1, op1),
			simd.select(use_wide8, w_op0, op0),
			simd.select(use_wide8, w_oq0, oq0),
			simd.select(use_wide8, w_oq1, oq1),
		)
		return
	}

	mid4 := p1 + p0 + q0 + q1 + V4(4)
	w8_op2 := simd.shr(p3 * V4(3) + p2 * V4(2) + p1 + p0 + q0 + V4(4), U4(3))
	w8_op1 := simd.shr(p3 * V4(2) + p2 + p1 + mid4, U4(3))
	w8_op0 := simd.shr(p3 + p2 + p0 + q2 + mid4, U4(3))
	w8_oq0 := simd.shr(p2 + q0 + q2 + q3 + mid4, U4(3))
	w8_oq1 := simd.shr(q1 + q2 + q3 * V4(2) + mid4, U4(3))
	w8_oq2 := simd.shr(p0 + q0 + q1 + q2 * V4(2) + q3 * V4(3) + V4(4), U4(3))

	op2 := simd.select(use_wide8, w8_op2, p2)
	op1 = simd.select(use_wide8, w8_op1, op1)
	op0 = simd.select(use_wide8, w8_op0, op0)
	oq0 = simd.select(use_wide8, w8_oq0, oq0)
	oq1 = simd.select(use_wide8, w8_oq1, oq1)
	oq2 := simd.select(use_wide8, w8_oq2, q2)

	if filter_size >= 16 {
		p6, p5, p4, _ := lf_transpose_4x4(
			load_row(plane_buf, base + 0 * stride - 7),
			load_row(plane_buf, base + 1 * stride - 7),
			load_row(plane_buf, base + 2 * stride - 7),
			load_row(plane_buf, base + 3 * stride - 7),
		)
		_, q4, q5, q6 := lf_transpose_4x4(
			load_row(plane_buf, base + 0 * stride + 3),
			load_row(plane_buf, base + 1 * stride + 3),
			load_row(plane_buf, base + 2 * stride + 3),
			load_row(plane_buf, base + 3 * stride + 3),
		)

		flat2_bad := simd.lanes_gt(simd.abs(p6 - p0), flat_thresh) |
			simd.lanes_gt(simd.abs(q6 - q0), flat_thresh) |
			simd.lanes_gt(simd.abs(p5 - p0), flat_thresh) |
			simd.lanes_gt(simd.abs(q5 - q0), flat_thresh) |
			simd.lanes_gt(simd.abs(p4 - p0), flat_thresh) |
			simd.lanes_gt(simd.abs(q4 - q0), flat_thresh)
		use_wide16 := use_wide8 & (~flat2_bad)
		if simd.reduce_or(use_wide16) != 0 {
			op5, op4, op3, oq3, oq4, oq5: V4
			op5, op4, op3, op2, op1, op0, oq0, oq1, oq2, oq3, oq4, oq5 = lf_wide16_simd(
				p6, p5, p4, p3, p2, p1, p0,
				q0, q1, q2, q3, q4, q5, q6,
				op2, op1, op0, oq0, oq1, oq2,
				use_wide16,
			)
			store_4x4(plane_buf, base, stride, -6, op5, op4, op3, op2)
			store_4x4(plane_buf, base, stride, -2, op1, op0, oq0, oq1)
			store_4x4(plane_buf, base, stride, 2, oq2, oq3, oq4, oq5)
			return
		}
	}

	scatter_col(plane_buf, base, stride, -3, op2)
	store_4x4(plane_buf, base, stride, -2, op1, op0, oq0, oq1)
	scatter_col(plane_buf, base, stride, 2, oq2)
}
