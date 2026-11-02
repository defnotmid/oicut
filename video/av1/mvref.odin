package av1

import "base:intrinsics"
import "core:simd"

// Motion Field Estimation (section 7.9), Motion Vector Prediction (section 7.10),
// and Motion Field MV Storage (section 7.19) shared by the AV1 decoder and encoder.
// Uses Structure-of-Arrays (#soa) grids for cache-dense neighborhood and temporal scans.

Mv :: [2]i16 // [0] = row (y), [1] = col (x) in 1/8 luma samples

INVALID_MV :: Mv{-32768, -32768}

// Per-4x4 block mode/motion cell stored in an #soa slice across the frame.
// Before decoding a frame/tile, ref_frames is initialized to {NONE, NONE} so
// ref_frames[0] != NONE indicates the cell has been decoded in the current frame.
MI_Cell :: struct {
	mvs:        [2]Mv,
	ref_frames: [2]i8,
	mi_size:    u8,
	y_mode:     u8,
	is_inter:   bool,
}

// Frame-wide 4x4 mode info grid with tile bounds for the active tile.
MI_Grid :: struct {
	cells:        #soa[]MI_Cell,
	mi_rows:      int,
	mi_cols:      int,
	mi_row_start: int,
	mi_row_end:   int,
	mi_col_start: int,
	mi_col_end:   int,
}

// Per-8x8 cell saved for a reference frame (sections 7.19 and 7.20; only odd
// 4x4 coordinates row = 2*y8 + 1, col = 2*x8 + 1 are ever read by section 7.9.2).
Saved_MV_Cell :: struct {
	mv:        Mv,
	ref_frame: i8,
}

Saved_Ref_MVs :: struct {
	cells:       #soa[]Saved_MV_Cell, // length (mi_rows >> 1) * (mi_cols >> 1)
	mi_rows:     int,
	mi_cols:     int,
	frame_type:  u8,
	order_hint:  u8,
	order_hints: [TOTAL_REFS_PER_FRAME]u8,
}

is_inside_tile :: #force_inline proc "contextless" (g: ^MI_Grid, r, c: int) -> bool {
	return c >= g.mi_col_start && c < g.mi_col_end && r >= g.mi_row_start && r < g.mi_row_end
}

// Section 5.11.53: clamp_mv_row.
clamp_mv_row :: #force_inline proc "contextless" (mvec: i32, mi_row, mi_rows, bh4, border: int) -> i16 {
	mb_to_top := -i32((mi_row * MI_SIZE) * 8)
	mb_to_bottom := i32(((mi_rows - bh4 - mi_row) * MI_SIZE) * 8)
	return i16(clamp(mvec, mb_to_top - i32(border), mb_to_bottom + i32(border)))
}

// Section 5.11.54: clamp_mv_col.
clamp_mv_col :: #force_inline proc "contextless" (mvec: i32, mi_col, mi_cols, bw4, border: int) -> i16 {
	mb_to_left := -i32((mi_col * MI_SIZE) * 8)
	mb_to_right := i32(((mi_cols - bw4 - mi_col) * MI_SIZE) * 8)
	return i16(clamp(mvec, mb_to_left - i32(border), mb_to_right + i32(border)))
}

Mode_Set :: bit_set[0 ..< 32; u32]
Ref_Frame_Set :: bit_set[-1 ..< TOTAL_REFS_PER_FRAME]

NEWMV_MODES :: Mode_Set{NEWMV, NEW_NEWMV, NEAR_NEWMV, NEW_NEARMV, NEAREST_NEWMV, NEW_NEARESTMV}
NEARMV_MODES :: Mode_Set{NEARMV, NEAR_NEARMV, NEAR_NEWMV, NEW_NEARMV}

// Section 7.10.2.9: has_newmv.
has_newmv :: #force_inline proc "contextless" (mode: int) -> bool {
	return mode in NEWMV_MODES
}

// Section 5.11.23: has_nearmv.
has_nearmv :: #force_inline proc "contextless" (mode: int) -> bool {
	return mode in NEARMV_MODES
}

// Section 5.11.30: get_mode.
get_inter_comp_mode :: #force_inline proc "contextless" (y_mode, ref_list: int) -> int {
	if ref_list == 0 {
		if y_mode < NEAREST_NEARESTMV do return y_mode
		if y_mode == NEW_NEWMV || y_mode == NEW_NEARESTMV || y_mode == NEW_NEARMV do return NEWMV
		if y_mode == NEAREST_NEARESTMV || y_mode == NEAREST_NEWMV do return NEARESTMV
		if y_mode == NEAR_NEARMV || y_mode == NEAR_NEWMV do return NEARMV
		return GLOBALMV
	}
	if y_mode == NEW_NEWMV || y_mode == NEAREST_NEWMV || y_mode == NEAR_NEWMV do return NEWMV
	if y_mode == NEAREST_NEARESTMV || y_mode == NEW_NEARESTMV do return NEARESTMV
	if y_mode == NEAR_NEARMV || y_mode == NEW_NEARMV do return NEARMV
	return GLOBALMV
}

// Section 7.10.2.10: Lower precision process.
lower_mv_precision :: #force_inline proc "contextless" (mv: ^Mv, allow_high_precision_mv, force_integer_mv: bool) {
	if allow_high_precision_mv do return
	v0 := mv[0]
	v1 := mv[1]
	s0 := (v0 >> 15) & 1
	s1 := (v1 >> 15) & 1
	if force_integer_mv {
		mv[0] = (v0 + 3 + s0) & ~i16(7)
		mv[1] = (v1 + 3 + s1) & ~i16(7)
	} else {
		mv[0] = (v0 + s0) & ~i16(1)
		mv[1] = (v1 + s1) & ~i16(1)
	}
}

@(private)
round2_signed_14 :: #force_inline proc "contextless" (x: i64) -> i32 {
	return i32((x + 8192 + (x >> 63)) >> 14)
}

@(private)
project_mv_mult :: #force_inline proc "contextless" (mv: Mv, mult: i64) -> Mv {
	limit :: i32((1 << 14) - 1)
	s0 := round2_signed_14(i64(mv[0]) * mult)
	s1 := round2_signed_14(i64(mv[1]) * mult)
	return Mv{
		i16(clamp(s0, -limit, limit)),
		i16(clamp(s1, -limit, limit)),
	}
}

// Section 7.9.3: Get MV projection process.
get_mv_projection :: #force_inline proc "contextless" (mv: Mv, numerator, denominator: int) -> Mv {
	clipped_den := min(MAX_FRAME_DISTANCE, max(0, denominator))
	clipped_num := clamp(numerator, -MAX_FRAME_DISTANCE, MAX_FRAME_DISTANCE)
	mult := i64(clipped_num) * i64(DIV_MULT[clipped_den])
	return project_mv_mult(mv, mult)
}

// Section 7.9.4: Get block position process.
get_block_position :: #force_inline proc "contextless" (x8, y8, dst_sign: int, proj_mv: Mv, mi_rows, mi_cols: int) -> (pos_x8, pos_y8: int, valid: bool) {
	project :: #force_inline proc "contextless" (v8: int, delta: i32, sign, max8, max_off8: int) -> (int, bool) {
		base8 := v8 & ~int(7)
		offset8 := int((delta + ((delta >> 31) & 63)) >> 6)
		res := v8 + sign * offset8
		ok := uint(res) < uint(max8) && res >= base8 - max_off8 && res < base8 + 8 + max_off8
		return res, ok
	}
	ok_y, ok_x: bool
	pos_y8, ok_y = project(y8, i32(proj_mv[0]), dst_sign, mi_rows >> 1, MAX_OFFSET_HEIGHT)
	pos_x8, ok_x = project(x8, i32(proj_mv[1]), dst_sign, mi_cols >> 1, MAX_OFFSET_WIDTH)
	valid = ok_y && ok_x
	return
}

// Section 7.9.2: Projection process.
motion_field_projection :: proc "contextless" (
	motion_field_mvs: ^[TOTAL_REFS_PER_FRAME][]Mv,
	saved_refs: ^[NUM_REF_FRAMES]Saved_Ref_MVs,
	ref_frame_idx: [REFS_PER_FRAME]i8,
	order_hints: [TOTAL_REFS_PER_FRAME]u8,
	cur_order_hint: u8,
	enable_order_hint: bool,
	order_hint_bits: u8,
	mi_rows, mi_cols: int,
	src: int,
	dst_sign: int,
) -> bool #no_bounds_check {
	src_idx := int(ref_frame_idx[src - LAST_FRAME])
	if src_idx < 0 || src_idx >= NUM_REF_FRAMES do return false
	sref := &saved_refs[src_idx]
	if sref.mi_rows != mi_rows || sref.mi_cols != mi_cols || sref.frame_type == INTRA_ONLY_FRAME || sref.frame_type == KEY_FRAME {
		return false
	}
	w8 := mi_cols >> 1
	h8 := mi_rows >> 1
	ref_to_cur := get_relative_dist(int(order_hints[src]), int(cur_order_hint), enable_order_hint, order_hint_bits)
	if abs(ref_to_cur) > MAX_FRAME_DISTANCE do return true

	ref_to_dst: [TOTAL_REFS_PER_FRAME]int = ---
	ref_offsets: [TOTAL_REFS_PER_FRAME]int = ---
	cur_mult: [TOTAL_REFS_PER_FRAME]i64 = ---
	dst_mult_v: [TOTAL_REFS_PER_FRAME]#simd[8]i32 = ---
	src_ref_valid: Ref_Frame_Set
	clipped_cur_num := i64(clamp(ref_to_cur * dst_sign, -MAX_FRAME_DISTANCE, MAX_FRAME_DISTANCE))

	for dst in LAST_FRAME ..= ALTREF_FRAME {
		ref_to_dst[dst] = clamp(get_relative_dist(int(cur_order_hint), int(order_hints[dst]), enable_order_hint, order_hint_bits), -MAX_FRAME_DISTANCE, MAX_FRAME_DISTANCE)
		ref_offsets[dst] = get_relative_dist(int(order_hints[src]), int(sref.order_hints[dst]), enable_order_hint, order_hint_bits)
	}
	for sr in LAST_FRAME ..= ALTREF_FRAME {
		ro := ref_offsets[sr]
		if ro > 0 && ro <= MAX_FRAME_DISTANCE {
			src_ref_valid += {sr}
			dm := i64(DIV_MULT[ro])
			cur_mult[sr] = clipped_cur_num * dm
			dm32 := i32(DIV_MULT[ro])
			dm_arr: [8]i32
			for dst in LAST_FRAME ..= ALTREF_FRAME {
				dm_arr[dst] = i32(ref_to_dst[dst]) * dm32
			}
			dst_mult_v[sr] = simd.from_array(dm_arr)
		}
	}

	src_refs := sref.cells.ref_frame
	src_mvs := sref.cells.mv
	mf_ptrs: [TOTAL_REFS_PER_FRAME][^]Mv = ---
	#unroll for dst in LAST_FRAME ..= ALTREF_FRAME {
		mf_ptrs[dst] = raw_data(motion_field_mvs[dst])
	}

	v_zero := (#simd[8]i32)(0)
	v_8191 := (#simd[8]i32)(8191)
	v_8192 := (#simd[8]i32)(8192)
	v_14 := (#simd[8]u32)(14)
	limit :: i32((1 << 14) - 1)
	v_neg_lim := (#simd[8]i32)(-limit)
	v_pos_lim := (#simd[8]i32)(limit)

	for y8 in 0 ..< h8 {
		row_base := y8 * w8
		y_sb_align := y8 & ~int(7)
		y_proj_end := min(y_sb_align + 8, h8)
		for x8 := 0; x8 < w8; {
			cell_idx := row_base + x8
			src_ref := int(src_refs[cell_idx])
			if !(src_ref in src_ref_valid) {
				x8 += 1
				continue
			}
			mv := src_mvs[cell_idx]
			proj_mv := project_mv_mult(mv, cur_mult[src_ref])
			off_y8 := int((i32(proj_mv[0]) + ((i32(proj_mv[0]) >> 31) & 63)) >> 6)
			pos_y8 := y8 + dst_sign * off_y8
			if pos_y8 >= y_sb_align && pos_y8 < y_proj_end {
				off_x8 := int((i32(proj_mv[1]) + ((i32(proj_mv[1]) >> 31) & 63)) >> 6)
				pos_x8 := x8 + dst_sign * off_x8
				dm_v := dst_mult_v[src_ref]
				p0 := (#simd[8]i32)(i32(mv[0])) * dm_v
				p1 := (#simd[8]i32)(i32(mv[1])) * dm_v
				b0 := simd.select(simd.lanes_lt(p0, v_zero), v_8191, v_8192)
				b1 := simd.select(simd.lanes_lt(p1, v_zero), v_8191, v_8192)
				s0: [8]i16 = ---
				s1: [8]i16 = ---
				intrinsics.unaligned_store((^#simd[8]i16)(&s0[0]), (#simd[8]i16)(simd.clamp(simd.shr(p0 + b0, v_14), v_neg_lim, v_pos_lim)))
				intrinsics.unaligned_store((^#simd[8]i16)(&s1[0]), (#simd[8]i16)(simd.clamp(simd.shr(p1 + b1, v_14), v_neg_lim, v_pos_lim)))
				dst_row_base := pos_y8 * w8
				for {
					x_sb_align := x8 & ~int(7)
					if uint(pos_x8) < uint(w8) && pos_x8 >= x_sb_align - MAX_OFFSET_WIDTH && pos_x8 < x_sb_align + 8 + MAX_OFFSET_WIDTH {
						dst_idx := dst_row_base + pos_x8
						#unroll for dst in LAST_FRAME ..= ALTREF_FRAME {
							mf_ptrs[dst][dst_idx] = Mv{s0[dst], s1[dst]}
						}
					}
					x8 += 1
					if x8 >= w8 do break
					next_idx := row_base + x8
					if int(src_refs[next_idx]) != src_ref || src_mvs[next_idx] != mv do break
					pos_x8 += 1
				}
			} else {
				for {
					x8 += 1
					if x8 >= w8 do break
					next_idx := row_base + x8
					if int(src_refs[next_idx]) != src_ref || src_mvs[next_idx] != mv do break
				}
			}
		}
	}
	return true
}

// Section 7.9.1: Motion field estimation process.
// motion_field_mvs[LAST_FRAME..ALTREF_FRAME] must each have length >= (mi_rows >> 1) * (mi_cols >> 1).
motion_field_estimation :: proc "contextless" (
	motion_field_mvs: ^[TOTAL_REFS_PER_FRAME][]Mv,
	saved_refs: ^[NUM_REF_FRAMES]Saved_Ref_MVs,
	ref_frame_idx: [REFS_PER_FRAME]i8,
	order_hints: [TOTAL_REFS_PER_FRAME]u8,
	cur_order_hint: u8,
	enable_order_hint: bool,
	order_hint_bits: u8,
	mi_rows, mi_cols: int,
) #no_bounds_check {
	w8 := mi_cols >> 1
	h8 := mi_rows >> 1
	n8 := w8 * h8
	for ref in LAST_FRAME ..= ALTREF_FRAME {
		field := motion_field_mvs[ref][:n8]
		for &mv in field do mv = INVALID_MV
	}

	last_idx := int(ref_frame_idx[0])
	if last_idx >= 0 && last_idx < NUM_REF_FRAMES {
		cur_gold_hint := order_hints[GOLDEN_FRAME]
		last_alt_hint := saved_refs[last_idx].order_hints[ALTREF_FRAME]
		if last_alt_hint != cur_gold_hint {
			motion_field_projection(
				motion_field_mvs, saved_refs, ref_frame_idx, order_hints,
				cur_order_hint, enable_order_hint, order_hint_bits, mi_rows, mi_cols,
				LAST_FRAME, -1,
			)
		}
	}

	ref_stamp := MFMV_STACK_SIZE - 2

	if get_relative_dist(int(order_hints[BWDREF_FRAME]), int(cur_order_hint), enable_order_hint, order_hint_bits) > 0 {
		if motion_field_projection(
			motion_field_mvs, saved_refs, ref_frame_idx, order_hints,
			cur_order_hint, enable_order_hint, order_hint_bits, mi_rows, mi_cols,
			BWDREF_FRAME, 1,
		) {
			ref_stamp -= 1
		}
	}

	if get_relative_dist(int(order_hints[ALTREF2_FRAME]), int(cur_order_hint), enable_order_hint, order_hint_bits) > 0 {
		if motion_field_projection(
			motion_field_mvs, saved_refs, ref_frame_idx, order_hints,
			cur_order_hint, enable_order_hint, order_hint_bits, mi_rows, mi_cols,
			ALTREF2_FRAME, 1,
		) {
			ref_stamp -= 1
		}
	}

	if ref_stamp >= 0 && get_relative_dist(int(order_hints[ALTREF_FRAME]), int(cur_order_hint), enable_order_hint, order_hint_bits) > 0 {
		if motion_field_projection(
			motion_field_mvs, saved_refs, ref_frame_idx, order_hints,
			cur_order_hint, enable_order_hint, order_hint_bits, mi_rows, mi_cols,
			ALTREF_FRAME, 1,
		) {
			ref_stamp -= 1
		}
	}

	if ref_stamp >= 0 {
		motion_field_projection(
			motion_field_mvs, saved_refs, ref_frame_idx, order_hints,
			cur_order_hint, enable_order_hint, order_hint_bits, mi_rows, mi_cols,
			LAST2_FRAME, -1,
		)
	}
}

// Section 7.19: Motion field motion vector storage process.
// Extracts odd-coordinate 4x4 cells (row = 2*y8 + 1, col = 2*x8 + 1) into dst.
store_motion_field_mvs :: proc "contextless" (
	dst: ^Saved_Ref_MVs,
	grid: ^MI_Grid,
	ref_frame_idx: [REFS_PER_FRAME]i8,
	ref_order_hint: [NUM_REF_FRAMES]u8,
	cur_order_hint: u8,
	enable_order_hint: bool,
	order_hint_bits: u8,
) #no_bounds_check {
	w8 := grid.mi_cols >> 1
	h8 := grid.mi_rows >> 1
	ref_ok: Ref_Frame_Set
	for r in LAST_FRAME ..= ALTREF_FRAME {
		ref_idx := int(ref_frame_idx[r - LAST_FRAME])
		if ref_idx >= 0 && ref_idx < NUM_REF_FRAMES {
			dist := get_relative_dist(int(ref_order_hint[ref_idx]), int(cur_order_hint), enable_order_hint, order_hint_bits)
			if dist < 0 do ref_ok += {r}
		}
	}

	mi_refs := grid.cells.ref_frames
	mi_mvs := grid.cells.mvs
	mi_sizes := grid.cells.mi_size
	dst_refs := dst.cells.ref_frame
	dst_mvs := dst.cells.mv
	mi_cols := grid.mi_cols

	for y8 in 0 ..< h8 {
		row_base := (2 * y8 + 1) * mi_cols
		out_base := y8 * w8
		for x8 := 0; x8 < w8; {
			cell_idx := row_base + (2 * x8 + 1)
			bw8 := min(w8 - x8, int(NUM_4X4_BLOCKS_WIDE[mi_sizes[cell_idx]] + 1) >> 1)
			mf_ref := i8(NONE)
			mf_mv := Mv{0, 0}
			rfs := mi_refs[cell_idx]
			mvs := mi_mvs[cell_idx]
			#unroll for list in 0 ..< 2 {
				r := int(rfs[list])
				if r in ref_ok {
					mv := mvs[list]
					if (abs(int(mv[0])) | abs(int(mv[1]))) <= REFMVS_LIMIT {
						mf_ref = i8(r)
						mf_mv = mv
					}
				}
			}
			for n in 0 ..< bw8 {
				out_idx := out_base + x8 + n
				dst_refs[out_idx] = mf_ref
				dst_mvs[out_idx] = mf_mv
			}
			x8 += bw8
		}
	}
}

// Section 7.10.2.1: Setup global MV process.
setup_global_mv :: #force_inline proc "contextless" (
	ref: int,
	mi_row, mi_col, mi_size: int,
	gm_type: [TOTAL_REFS_PER_FRAME]u8,
	gm_params: [TOTAL_REFS_PER_FRAME][6]i32,
	allow_high_precision_mv, force_integer_mv: bool,
) -> Mv {
	if ref <= INTRA_FRAME || gm_type[ref] == IDENTITY {
		return Mv{0, 0}
	}
	typ := gm_type[ref]
	mv: Mv
	if typ == TRANSLATION {
		mv[0] = i16(gm_params[ref][0] >> (WARPEDMODEL_PREC_BITS - 3))
		mv[1] = i16(gm_params[ref][1] >> (WARPEDMODEL_PREC_BITS - 3))
	} else {
		bw := int(BLOCK_WIDTH[mi_size])
		bh := int(BLOCK_HEIGHT[mi_size])
		x := i32(mi_col * MI_SIZE + bw / 2 - 1)
		y := i32(mi_row * MI_SIZE + bh / 2 - 1)
		xc := i64(gm_params[ref][2] - (1 << WARPEDMODEL_PREC_BITS)) * i64(x) + i64(gm_params[ref][3]) * i64(y) + i64(gm_params[ref][0])
		yc := i64(gm_params[ref][4]) * i64(x) + i64(gm_params[ref][5] - (1 << WARPEDMODEL_PREC_BITS)) * i64(y) + i64(gm_params[ref][1])
		if allow_high_precision_mv {
			mv[0] = i16(round2_signed(yc, WARPEDMODEL_PREC_BITS - 3))
			mv[1] = i16(round2_signed(xc, WARPEDMODEL_PREC_BITS - 3))
		} else {
			mv[0] = i16(round2_signed(yc, WARPEDMODEL_PREC_BITS - 2) * 2)
			mv[1] = i16(round2_signed(xc, WARPEDMODEL_PREC_BITS - 2) * 2)
		}
	}
	lower_mv_precision(&mv, allow_high_precision_mv, force_integer_mv)
	return mv
}

// Output of find_mv_stack (section 7.10.2).
MV_Stack :: struct {
	ref_stack_mv:    [MAX_REF_MV_STACK_SIZE][2]Mv,
	weight_stack:    [MAX_REF_MV_STACK_SIZE]int,
	drl_ctx_stack:   [MAX_REF_MV_STACK_SIZE]int,
	global_mvs:      [2]Mv,
	gm_override:     [2]bool,
	num_mv_found:    int,
	new_mv_count:    int,
	found_match:     bool,
	close_matches:   int,
	total_matches:   int,
	new_mv_context:  int,
	ref_mv_context:  int,
	zero_mv_context: int,
}

// Section 7.10.2: Find MV stack process.
find_mv_stack :: proc "contextless" (
	st: ^MV_Stack,
	grid: ^MI_Grid,
	mi_row, mi_col, mi_size: int,
	ref_frame: [2]i8,
	is_compound: bool,
	allow_high_precision_mv, force_integer_mv: bool,
	use_ref_frame_mvs: bool,
	motion_field_mvs: ^[TOTAL_REFS_PER_FRAME][]Mv,
	gm_type: [TOTAL_REFS_PER_FRAME]u8,
	gm_params: [TOTAL_REFS_PER_FRAME][6]i32,
	ref_frame_sign_bias: [TOTAL_REFS_PER_FRAME]u8,
) #no_bounds_check {
	bw4 := int(NUM_4X4_BLOCKS_WIDE[mi_size])
	bh4 := int(NUM_4X4_BLOCKS_HIGH[mi_size])

	st.num_mv_found = 0
	st.new_mv_count = 0
	rf0 := int(ref_frame[0])
	st.global_mvs[0] = setup_global_mv(rf0, mi_row, mi_col, mi_size, gm_type, gm_params, allow_high_precision_mv, force_integer_mv)
	st.gm_override[0] = rf0 > INTRA_FRAME && gm_type[rf0] > TRANSLATION
	if is_compound {
		rf1 := int(ref_frame[1])
		st.global_mvs[1] = setup_global_mv(rf1, mi_row, mi_col, mi_size, gm_type, gm_params, allow_high_precision_mv, force_integer_mv)
		st.gm_override[1] = rf1 > INTRA_FRAME && gm_type[rf1] > TRANSLATION
	} else {
		st.global_mvs[1] = {0, 0}
		st.gm_override[1] = false
	}

	has_top := mi_row > grid.mi_row_start
	has_left := mi_col > grid.mi_col_start

	found_above_match := 0
	if has_top {
		st.found_match = false
		mv_scan_row(st, grid, mi_row, mi_col, bw4, -1, ref_frame, is_compound)
		found_above_match = int(st.found_match)
	}

	found_left_match := 0
	if has_left {
		st.found_match = false
		mv_scan_col(st, grid, mi_row, mi_col, bh4, -1, ref_frame, is_compound)
		found_left_match = int(st.found_match)
	}

	if has_top && max(bw4, bh4) <= 16 {
		st.found_match = false
		mv_scan_point(st, grid, mi_row, mi_col, -1, bw4, ref_frame, is_compound)
		if st.found_match do found_above_match = 1
	}

	st.close_matches = found_above_match + found_left_match
	num_nearest := st.num_mv_found
	num_new := st.new_mv_count

	for idx in 0 ..< num_nearest {
		st.weight_stack[idx] += REF_CAT_LEVEL
	}

	st.zero_mv_context = 0
	if use_ref_frame_mvs && motion_field_mvs != nil {
		mv_temporal_scan(st, grid, mi_row, mi_col, bw4, bh4, ref_frame, is_compound, allow_high_precision_mv, force_integer_mv, motion_field_mvs)
	}

	if has_top {
		if has_left {
			st.found_match = false
			mv_scan_point(st, grid, mi_row, mi_col, -1, -1, ref_frame, is_compound)
			if st.found_match do found_above_match = 1
		}

		st.found_match = false
		mv_scan_row(st, grid, mi_row, mi_col, bw4, -3, ref_frame, is_compound)
		if st.found_match do found_above_match = 1
	}

	if has_left {
		st.found_match = false
		mv_scan_col(st, grid, mi_row, mi_col, bh4, -3, ref_frame, is_compound)
		if st.found_match do found_left_match = 1
	}

	if has_top && bh4 > 1 {
		st.found_match = false
		mv_scan_row(st, grid, mi_row, mi_col, bw4, -5, ref_frame, is_compound)
		if st.found_match do found_above_match = 1
	}

	if has_left && bw4 > 1 {
		st.found_match = false
		mv_scan_col(st, grid, mi_row, mi_col, bh4, -5, ref_frame, is_compound)
		if st.found_match do found_left_match = 1
	}

	st.total_matches = found_above_match + found_left_match

	mv_sort_stack(st, 0, num_nearest, is_compound)
	mv_sort_stack(st, num_nearest, st.num_mv_found, is_compound)

	if st.num_mv_found < 2 {
		mv_extra_search(st, grid, mi_row, mi_col, mi_size, ref_frame, is_compound, allow_high_precision_mv, force_integer_mv, ref_frame_sign_bias)
	}

	mv_context_and_clamping(st, grid, mi_row, mi_col, mi_size, is_compound, num_new)
}

// Section 7.10.2.2: Scan row process.
@(private)
mv_scan_row :: #force_inline proc "contextless" (
	st: ^MV_Stack,
	grid: ^MI_Grid,
	mi_row, mi_col, bw4, delta_row_in: int,
	ref_frame: [2]i8,
	is_compound: bool,
) #no_bounds_check {
	end4 := min(min(bw4, grid.mi_cols - mi_col), 16)
	delta_row := delta_row_in
	delta_col := 0
	use_step16 := bw4 >= 16
	if abs(delta_row_in) > 1 {
		delta_row += mi_row & 1
		delta_col = 1 - (mi_col & 1)
	}
	mv_row := mi_row + delta_row
	if mv_row < grid.mi_row_start do return
	base_col := mi_col + delta_col
	limit4 := min(end4, grid.mi_col_end - base_col)
	mi_sizes := grid.cells.mi_size
	row_base := mv_row * grid.mi_cols + base_col
	for i := 0; i < limit4; {
		cell_idx := row_base + i
		cand_size := int(mi_sizes[cell_idx])
		length := min(bw4, int(NUM_4X4_BLOCKS_WIDE[cand_size]))
		if abs(delta_row_in) > 1 do length = max(2, length)
		if use_step16 do length = max(4, length)
		add_ref_mv_candidate(st, grid, cell_idx, cand_size, ref_frame, is_compound, length * 2)
		i += length
	}
}

// Section 7.10.2.3: Scan col process.
@(private)
mv_scan_col :: #force_inline proc "contextless" (
	st: ^MV_Stack,
	grid: ^MI_Grid,
	mi_row, mi_col, bh4, delta_col_in: int,
	ref_frame: [2]i8,
	is_compound: bool,
) #no_bounds_check {
	end4 := min(min(bh4, grid.mi_rows - mi_row), 16)
	delta_row := 0
	delta_col := delta_col_in
	use_step16 := bh4 >= 16
	if abs(delta_col_in) > 1 {
		delta_row = 1 - (mi_row & 1)
		delta_col += mi_col & 1
	}
	mv_col := mi_col + delta_col
	if mv_col < grid.mi_col_start do return
	base_row := mi_row + delta_row
	limit4 := min(end4, grid.mi_row_end - base_row)
	mi_sizes := grid.cells.mi_size
	stride := grid.mi_cols
	cell_idx := base_row * stride + mv_col
	for i := 0; i < limit4; {
		cand_size := int(mi_sizes[cell_idx])
		length := min(bh4, int(NUM_4X4_BLOCKS_HIGH[cand_size]))
		if abs(delta_col_in) > 1 do length = max(2, length)
		if use_step16 do length = max(4, length)
		add_ref_mv_candidate(st, grid, cell_idx, cand_size, ref_frame, is_compound, length * 2)
		i += length
		cell_idx += length * stride
	}
}

// Section 7.10.2.4: Scan point process.
@(private)
mv_scan_point :: #force_inline proc "contextless" (
	st: ^MV_Stack,
	grid: ^MI_Grid,
	mi_row, mi_col, delta_row, delta_col: int,
	ref_frame: [2]i8,
	is_compound: bool,
) #no_bounds_check {
	mv_row := mi_row + delta_row
	mv_col := mi_col + delta_col
	if !is_inside_tile(grid, mv_row, mv_col) do return
	cell_idx := mv_row * grid.mi_cols + mv_col
	add_ref_mv_candidate(st, grid, cell_idx, int(grid.cells.mi_size[cell_idx]), ref_frame, is_compound, 4)
}

// Section 7.10.2.5 & 7.10.2.6: Temporal scan and temporal sample processes.
@(private)
mv_temporal_scan :: #force_inline proc "contextless" (
	st: ^MV_Stack,
	grid: ^MI_Grid,
	mi_row, mi_col, bw4, bh4: int,
	ref_frame: [2]i8,
	is_compound: bool,
	allow_high_precision_mv, force_integer_mv: bool,
	motion_field_mvs: ^[TOTAL_REFS_PER_FRAME][]Mv,
) #no_bounds_check {
	step_w4 := bw4 >= 16 ? 4 : 2
	step_h4 := bh4 >= 16 ? 4 : 2

	for delta_row := 0; delta_row < min(bh4, 16); delta_row += step_h4 {
		for delta_col := 0; delta_col < min(bw4, 16); delta_col += step_w4 {
			add_tpl_ref_mv(st, grid, mi_row, mi_col, delta_row, delta_col, ref_frame, is_compound, allow_high_precision_mv, force_integer_mv, motion_field_mvs)
		}
	}

	allow_extension := bh4 >= int(NUM_4X4_BLOCKS_HIGH[BLOCK_8X8]) &&
		bh4 < int(NUM_4X4_BLOCKS_HIGH[BLOCK_64X64]) &&
		bw4 >= int(NUM_4X4_BLOCKS_WIDE[BLOCK_8X8]) &&
		bw4 < int(NUM_4X4_BLOCKS_WIDE[BLOCK_64X64])
	if allow_extension {
		tpl_sample_pos := [3][2]int{
			{bh4, -2},
			{bh4, bw4},
			{bh4 - 2, bw4},
		}
		for pos in tpl_sample_pos {
			dr, dc := pos[0], pos[1]
			row := (mi_row & 15) + dr
			col := (mi_col & 15) + dc
			if row >= 0 && row < 16 && col >= 0 && col < 16 {
				add_tpl_ref_mv(st, grid, mi_row, mi_col, dr, dc, ref_frame, is_compound, allow_high_precision_mv, force_integer_mv, motion_field_mvs)
			}
		}
	}
}

@(private)
add_tpl_ref_mv :: #force_inline proc "contextless" (
	st: ^MV_Stack,
	grid: ^MI_Grid,
	mi_row, mi_col, delta_row, delta_col: int,
	ref_frame: [2]i8,
	is_compound: bool,
	allow_high_precision_mv, force_integer_mv: bool,
	motion_field_mvs: ^[TOTAL_REFS_PER_FRAME][]Mv,
) #no_bounds_check {
	mv_row := (mi_row + delta_row) | 1
	mv_col := (mi_col + delta_col) | 1
	if !is_inside_tile(grid, mv_row, mv_col) do return
	x8 := mv_col >> 1
	y8 := mv_row >> 1
	w8 := grid.mi_cols >> 1
	idx8 := y8 * w8 + x8

	if delta_row == 0 && delta_col == 0 {
		st.zero_mv_context = 1
	}

	if !is_compound {
		rf0 := int(ref_frame[0])
		if rf0 <= INTRA_FRAME do return
		cand_mv := motion_field_mvs[rf0][idx8]
		if cand_mv[0] == -32768 do return
		lower_mv_precision(&cand_mv, allow_high_precision_mv, force_integer_mv)
		if delta_row == 0 && delta_col == 0 {
			st.zero_mv_context = int((abs(int(cand_mv[0]) - int(st.global_mvs[0][0])) | abs(int(cand_mv[1]) - int(st.global_mvs[0][1]))) >= 16)
		}
		cand_u32 := transmute(u32)cand_mv
		for idx in 0 ..< st.num_mv_found {
			if cand_u32 == transmute(u32)st.ref_stack_mv[idx][0] {
				st.weight_stack[idx] += 2
				return
			}
		}
		if st.num_mv_found < MAX_REF_MV_STACK_SIZE {
			st.ref_stack_mv[st.num_mv_found][0] = cand_mv
			st.weight_stack[st.num_mv_found] = 2
			st.num_mv_found += 1
		}
	} else {
		rf0 := int(ref_frame[0])
		rf1 := int(ref_frame[1])
		if rf0 <= INTRA_FRAME || rf1 <= INTRA_FRAME do return
		cand_mv0 := motion_field_mvs[rf0][idx8]
		if cand_mv0[0] == -32768 do return
		cand_mv1 := motion_field_mvs[rf1][idx8]
		if cand_mv1[0] == -32768 do return
		lower_mv_precision(&cand_mv0, allow_high_precision_mv, force_integer_mv)
		lower_mv_precision(&cand_mv1, allow_high_precision_mv, force_integer_mv)
		if delta_row == 0 && delta_col == 0 {
			st.zero_mv_context = int((abs(int(cand_mv0[0]) - int(st.global_mvs[0][0])) |
				abs(int(cand_mv0[1]) - int(st.global_mvs[0][1])) |
				abs(int(cand_mv1[0]) - int(st.global_mvs[1][0])) |
				abs(int(cand_mv1[1]) - int(st.global_mvs[1][1]))) >= 16)
		}
		cand_pair := [2]Mv{cand_mv0, cand_mv1}
		cand_u64 := transmute(u64)cand_pair
		for idx in 0 ..< st.num_mv_found {
			if cand_u64 == transmute(u64)st.ref_stack_mv[idx] {
				st.weight_stack[idx] += 2
				return
			}
		}
		if st.num_mv_found < MAX_REF_MV_STACK_SIZE {
			st.ref_stack_mv[st.num_mv_found] = cand_pair
			st.weight_stack[st.num_mv_found] = 2
			st.num_mv_found += 1
		}
	}
}

// Section 7.10.2.7: Add reference motion vector process.
@(private)
add_ref_mv_candidate :: #force_inline proc "contextless" (
	st: ^MV_Stack,
	grid: ^MI_Grid,
	cell_idx, cand_size: int,
	ref_frame: [2]i8,
	is_compound: bool,
	weight: int,
) #no_bounds_check {
	cand_refs := grid.cells.ref_frames[cell_idx]
	if !is_compound {
		rf0 := ref_frame[0]
		if rf0 == INTRA_FRAME && !grid.cells.is_inter[cell_idx] do return
		if cand_refs[0] == rf0 {
			search_stack_single(st, grid, cell_idx, cand_size, 0, weight)
		} else if cand_refs[1] == rf0 {
			search_stack_single(st, grid, cell_idx, cand_size, 1, weight)
		}
	} else {
		if transmute(u16)cand_refs == transmute(u16)ref_frame {
			search_stack_compound(st, grid, cell_idx, weight)
		}
	}
}

// Section 7.10.2.8: Search stack process.
@(private)
search_stack_single :: #force_inline proc "contextless" (
	st: ^MV_Stack,
	grid: ^MI_Grid,
	cell_idx, cand_size, cand_list, weight: int,
) #no_bounds_check {
	cand_mode := int(grid.cells.y_mode[cell_idx])
	cand_mv: Mv
	if st.gm_override[0] && (cand_mode == GLOBALMV || cand_mode == GLOBAL_GLOBALMV) && min(BLOCK_WIDTH[cand_size], BLOCK_HEIGHT[cand_size]) >= 8 {
		cand_mv = st.global_mvs[0]
	} else {
		cand_mv = grid.cells.mvs[cell_idx][cand_list]
	}
	if has_newmv(cand_mode) {
		st.new_mv_count += 1
	}
	st.found_match = true
	cand_u32 := transmute(u32)cand_mv
	for idx in 0 ..< st.num_mv_found {
		if cand_u32 == transmute(u32)st.ref_stack_mv[idx][0] {
			st.weight_stack[idx] += weight
			return
		}
	}
	if st.num_mv_found < MAX_REF_MV_STACK_SIZE {
		st.ref_stack_mv[st.num_mv_found][0] = cand_mv
		st.weight_stack[st.num_mv_found] = weight
		st.num_mv_found += 1
	}
}

// Section 7.10.2.9: Compound search stack process.
@(private)
search_stack_compound :: #force_inline proc "contextless" (
	st: ^MV_Stack,
	grid: ^MI_Grid,
	cell_idx, weight: int,
) #no_bounds_check {
	cand_mvs := grid.cells.mvs[cell_idx]
	cand_mode := int(grid.cells.y_mode[cell_idx])
	if cand_mode == GLOBAL_GLOBALMV {
		if st.gm_override[0] do cand_mvs[0] = st.global_mvs[0]
		if st.gm_override[1] do cand_mvs[1] = st.global_mvs[1]
	}
	st.found_match = true
	if has_newmv(cand_mode) {
		st.new_mv_count += 1
	}
	cand_u64 := transmute(u64)cand_mvs
	for idx in 0 ..< st.num_mv_found {
		if cand_u64 == transmute(u64)st.ref_stack_mv[idx] {
			st.weight_stack[idx] += weight
			return
		}
	}
	if st.num_mv_found < MAX_REF_MV_STACK_SIZE {
		st.ref_stack_mv[st.num_mv_found] = cand_mvs
		st.weight_stack[st.num_mv_found] = weight
		st.num_mv_found += 1
	}
}

// Section 7.10.2.11: Sorting process (stable bubble sort by weight descending).
@(private)
mv_sort_stack :: #force_inline proc "contextless" (st: ^MV_Stack, start, end_in: int, is_compound: bool) #no_bounds_check {
	end := end_in
	for end > start {
		new_end := start
		for idx := start + 1; idx < end; idx += 1 {
			if st.weight_stack[idx - 1] < st.weight_stack[idx] {
				st.weight_stack[idx - 1], st.weight_stack[idx] = st.weight_stack[idx], st.weight_stack[idx - 1]
				st.ref_stack_mv[idx - 1][0], st.ref_stack_mv[idx][0] = st.ref_stack_mv[idx][0], st.ref_stack_mv[idx - 1][0]
				if is_compound {
					st.ref_stack_mv[idx - 1][1], st.ref_stack_mv[idx][1] = st.ref_stack_mv[idx][1], st.ref_stack_mv[idx - 1][1]
				}
				new_end = idx
			}
		}
		end = new_end
	}
}

// Section 7.10.2.12 & 7.10.2.13: Extra search and add extra MV candidate processes.
@(private)
mv_extra_search :: proc "contextless" (
	st: ^MV_Stack,
	grid: ^MI_Grid,
	mi_row, mi_col, mi_size: int,
	ref_frame: [2]i8,
	is_compound: bool,
	allow_high_precision_mv, force_integer_mv: bool,
	ref_frame_sign_bias: [TOTAL_REFS_PER_FRAME]u8,
) #no_bounds_check {
	ref_id_mvs: [2][2]Mv
	ref_diff_mvs: [2][2]Mv
	ref_id_count: [2]int
	ref_diff_count: [2]int

	w4 := min(16, min(int(NUM_4X4_BLOCKS_WIDE[mi_size]), grid.mi_cols - mi_col))
	h4 := min(16, min(int(NUM_4X4_BLOCKS_HIGH[mi_size]), grid.mi_rows - mi_row))
	num4x4 := min(w4, h4)

	mi_refs := grid.cells.ref_frames
	mi_mvs := grid.cells.mvs
	mi_sizes := grid.cells.mi_size

	for pass in 0 ..< 2 {
		idx := 0
		for idx < num4x4 && st.num_mv_found < 2 {
			mv_row := pass == 0 ? mi_row - 1 : mi_row + idx
			mv_col := pass == 0 ? mi_col + idx : mi_col - 1
			if !is_inside_tile(grid, mv_row, mv_col) do break
			cell_idx := mv_row * grid.mi_cols + mv_col
			cand_refs := mi_refs[cell_idx]
			cand_mvs := mi_mvs[cell_idx]

			if is_compound {
				for cand_list in 0 ..< 2 {
					cand_ref := cand_refs[cand_list]
					if cand_ref > INTRA_FRAME {
						for list in 0 ..< 2 {
							cand_mv := cand_mvs[cand_list]
							if cand_ref == ref_frame[list] && ref_id_count[list] < 2 {
								lower_mv_precision(&cand_mv, allow_high_precision_mv, force_integer_mv)
								ref_id_mvs[list][ref_id_count[list]] = cand_mv
								ref_id_count[list] += 1
							} else if ref_diff_count[list] < 2 {
								if ref_frame_sign_bias[cand_ref] != ref_frame_sign_bias[ref_frame[list]] {
									cand_mv[0] = -cand_mv[0]
									cand_mv[1] = -cand_mv[1]
								}
								lower_mv_precision(&cand_mv, allow_high_precision_mv, force_integer_mv)
								ref_diff_mvs[list][ref_diff_count[list]] = cand_mv
								ref_diff_count[list] += 1
							}
						}
					}
				}
			} else {
				for cand_list in 0 ..< 2 {
					cand_ref := cand_refs[cand_list]
					if cand_ref > INTRA_FRAME {
						cand_mv := cand_mvs[cand_list]
						if ref_frame_sign_bias[cand_ref] != ref_frame_sign_bias[ref_frame[0]] {
							cand_mv[0] = -cand_mv[0]
							cand_mv[1] = -cand_mv[1]
						}
						lower_mv_precision(&cand_mv, allow_high_precision_mv, force_integer_mv)
						found := false
						for k in 0 ..< st.num_mv_found {
							if cand_mv == st.ref_stack_mv[k][0] {
								found = true
								break
							}
						}
						if !found && st.num_mv_found < MAX_REF_MV_STACK_SIZE {
							st.ref_stack_mv[st.num_mv_found][0] = cand_mv
							st.weight_stack[st.num_mv_found] = 2
							st.num_mv_found += 1
						}
					}
				}
			}

			cand_size := mi_sizes[cell_idx]
			if pass == 0 {
				idx += int(NUM_4X4_BLOCKS_WIDE[cand_size])
			} else {
				idx += int(NUM_4X4_BLOCKS_HIGH[cand_size])
			}
		}
	}

	if is_compound {
		combined_mvs: [2][2]Mv
		for list in 0 ..< 2 {
			comp_count := 0
			for idx in 0 ..< ref_id_count[list] {
				combined_mvs[comp_count][list] = ref_id_mvs[list][idx]
				comp_count += 1
			}
			for idx := 0; idx < ref_diff_count[list] && comp_count < 2; idx += 1 {
				combined_mvs[comp_count][list] = ref_diff_mvs[list][idx]
				comp_count += 1
			}
			for comp_count < 2 {
				combined_mvs[comp_count][list] = st.global_mvs[list]
				comp_count += 1
			}
		}
		if st.num_mv_found == 1 {
			if combined_mvs[0][0] == st.ref_stack_mv[0][0] && combined_mvs[0][1] == st.ref_stack_mv[0][1] {
				st.ref_stack_mv[st.num_mv_found][0] = combined_mvs[1][0]
				st.ref_stack_mv[st.num_mv_found][1] = combined_mvs[1][1]
			} else {
				st.ref_stack_mv[st.num_mv_found][0] = combined_mvs[0][0]
				st.ref_stack_mv[st.num_mv_found][1] = combined_mvs[0][1]
			}
			st.weight_stack[st.num_mv_found] = 2
			st.num_mv_found += 1
		} else {
			for idx in 0 ..< 2 {
				st.ref_stack_mv[st.num_mv_found][0] = combined_mvs[idx][0]
				st.ref_stack_mv[st.num_mv_found][1] = combined_mvs[idx][1]
				st.weight_stack[st.num_mv_found] = 2
				st.num_mv_found += 1
			}
		}
	} else {
		for idx in st.num_mv_found ..< 2 {
			st.ref_stack_mv[idx][0] = st.global_mvs[0]
		}
	}
}

// Section 7.10.2.14: Context and clamping process.
@(private)
mv_context_and_clamping :: #force_inline proc "contextless" (
	st: ^MV_Stack,
	grid: ^MI_Grid,
	mi_row, mi_col, mi_size: int,
	is_compound: bool,
	num_new: int,
) #no_bounds_check {
	bw := int(BLOCK_WIDTH[mi_size])
	bh := int(BLOCK_HEIGHT[mi_size])
	bw4 := int(NUM_4X4_BLOCKS_WIDE[mi_size])
	bh4 := int(NUM_4X4_BLOCKS_HIGH[mi_size])
	num_lists := is_compound ? 2 : 1

	for idx in 0 ..< st.num_mv_found {
		z := 0
		if idx + 1 < st.num_mv_found {
			w0 := st.weight_stack[idx]
			w1 := st.weight_stack[idx + 1]
			if w0 >= REF_CAT_LEVEL {
				if w1 < REF_CAT_LEVEL do z = 1
			} else {
				z = 2
			}
		}
		st.drl_ctx_stack[idx] = z
	}

	for list in 0 ..< num_lists {
		for idx in 0 ..< st.num_mv_found {
			ref_mv := st.ref_stack_mv[idx][list]
			ref_mv[0] = clamp_mv_row(i32(ref_mv[0]), mi_row, grid.mi_rows, bh4, MV_BORDER + bh * 8)
			ref_mv[1] = clamp_mv_col(i32(ref_mv[1]), mi_col, grid.mi_cols, bw4, MV_BORDER + bw * 8)
			st.ref_stack_mv[idx][list] = ref_mv
		}
	}

	if st.close_matches == 0 {
		st.new_mv_context = min(st.total_matches, 1)
		st.ref_mv_context = st.total_matches
	} else if st.close_matches == 1 {
		st.new_mv_context = 3 - min(num_new, 1)
		st.ref_mv_context = 2 + st.total_matches
	} else {
		st.new_mv_context = 5 - min(num_new, 1)
		st.ref_mv_context = 5
	}
}

// Section 7.10.3: Has overlappable candidates process.
has_overlappable_candidates :: proc "contextless" (
	grid: ^MI_Grid,
	mi_row, mi_col, mi_size: int,
	avail_u, avail_l: bool,
) -> bool #no_bounds_check {
	mi_refs := grid.cells.ref_frames
	if avail_u {
		w4 := int(NUM_4X4_BLOCKS_WIDE[mi_size])
		row_base := (mi_row - 1) * grid.mi_cols
		for x4 := mi_col; x4 < min(grid.mi_cols, mi_col + w4); x4 += 2 {
			if mi_refs[row_base + (x4 | 1)][0] > INTRA_FRAME {
				return true
			}
		}
	}
	if avail_l {
		h4 := int(NUM_4X4_BLOCKS_HIGH[mi_size])
		for y4 := mi_row; y4 < min(grid.mi_rows, mi_row + h4); y4 += 2 {
			if mi_refs[(y4 | 1) * grid.mi_cols + (mi_col - 1)][0] > INTRA_FRAME {
				return true
			}
		}
	}
	return false
}

// Section 7.10.4: Find warp samples process.
find_warp_samples :: proc "contextless" (
	cand_list: ^[LEAST_SQUARES_SAMPLES_MAX][4]i32,
	grid: ^MI_Grid,
	mi_row, mi_col, mi_size: int,
	avail_u, avail_l: bool,
	ref_frame0: i8,
	mv0: Mv,
) -> (num_samples: int) #no_bounds_check {
	num_samples_scanned := 0
	w4 := int(NUM_4X4_BLOCKS_WIDE[mi_size])
	h4 := int(NUM_4X4_BLOCKS_HIGH[mi_size])
	threshold := i32(clamp(int(max(BLOCK_WIDTH[mi_size], BLOCK_HEIGHT[mi_size])), 16, 112))

	add_sample :: proc "contextless" (
		cand_list: ^[LEAST_SQUARES_SAMPLES_MAX][4]i32,
		num_samples, num_samples_scanned: ^int,
		grid: ^MI_Grid,
		mi_row, mi_col, delta_row, delta_col: int,
		ref_frame0: i8,
		mv0: Mv,
		threshold: i32,
	) #no_bounds_check {
		if num_samples_scanned^ >= LEAST_SQUARES_SAMPLES_MAX do return
		mv_row := mi_row + delta_row
		mv_col := mi_col + delta_col
		if !is_inside_tile(grid, mv_row, mv_col) do return
		cell_idx := mv_row * grid.mi_cols + mv_col
		rfs := grid.cells[cell_idx].ref_frames
		if rfs[0] == NONE || rfs[0] != ref_frame0 || rfs[1] != NONE do return

		cand_sz := int(grid.cells[cell_idx].mi_size)
		cand_w4 := int(NUM_4X4_BLOCKS_WIDE[cand_sz])
		cand_h4 := int(NUM_4X4_BLOCKS_HIGH[cand_sz])
		cand_row := mv_row & ~(cand_h4 - 1)
		cand_col := mv_col & ~(cand_w4 - 1)
		mid_y := i32(cand_row * 4 + cand_h4 * 2 - 1)
		mid_x := i32(cand_col * 4 + cand_w4 * 2 - 1)

		cand_mv := grid.cells[cand_row * grid.mi_cols + cand_col].mvs[0]
		mv_diff := abs(i32(cand_mv[0]) - i32(mv0[0])) + abs(i32(cand_mv[1]) - i32(mv0[1]))
		valid := mv_diff <= threshold

		cand := [4]i32{
			mid_y * 8,
			mid_x * 8,
			mid_y * 8 + i32(cand_mv[0]),
			mid_x * 8 + i32(cand_mv[1]),
		}
		num_samples_scanned^ += 1
		if !valid && num_samples_scanned^ > 1 do return
		cand_list[num_samples^] = cand
		if valid do num_samples^ += 1
	}

	do_top_left := true
	do_top_right := true

	if avail_u {
		src_size := int(grid.cells[(mi_row - 1) * grid.mi_cols + mi_col].mi_size)
		src_w := int(NUM_4X4_BLOCKS_WIDE[src_size])
		if w4 <= src_w {
			col_offset := -(mi_col & (src_w - 1))
			if col_offset < 0 do do_top_left = false
			if col_offset + src_w > w4 do do_top_right = false
			add_sample(cand_list, &num_samples, &num_samples_scanned, grid, mi_row, mi_col, -1, 0, ref_frame0, mv0, threshold)
		} else {
			for i := 0; i < min(w4, grid.mi_cols - mi_col); {
				sz := int(grid.cells[(mi_row - 1) * grid.mi_cols + (mi_col + i)].mi_size)
				sw := int(NUM_4X4_BLOCKS_WIDE[sz])
				mi_step := min(w4, sw)
				add_sample(cand_list, &num_samples, &num_samples_scanned, grid, mi_row, mi_col, -1, i, ref_frame0, mv0, threshold)
				i += mi_step
			}
		}
	}

	if avail_l {
		src_size := int(grid.cells[mi_row * grid.mi_cols + (mi_col - 1)].mi_size)
		src_h := int(NUM_4X4_BLOCKS_HIGH[src_size])
		if h4 <= src_h {
			row_offset := -(mi_row & (src_h - 1))
			if row_offset < 0 do do_top_left = false
			add_sample(cand_list, &num_samples, &num_samples_scanned, grid, mi_row, mi_col, 0, -1, ref_frame0, mv0, threshold)
		} else {
			for i := 0; i < min(h4, grid.mi_rows - mi_row); {
				sz := int(grid.cells[(mi_row + i) * grid.mi_cols + (mi_col - 1)].mi_size)
				sh := int(NUM_4X4_BLOCKS_HIGH[sz])
				mi_step := min(h4, sh)
				add_sample(cand_list, &num_samples, &num_samples_scanned, grid, mi_row, mi_col, i, -1, ref_frame0, mv0, threshold)
				i += mi_step
			}
		}
	}

	if do_top_left {
		add_sample(cand_list, &num_samples, &num_samples_scanned, grid, mi_row, mi_col, -1, -1, ref_frame0, mv0, threshold)
	}
	if do_top_right && max(w4, h4) <= 16 {
		add_sample(cand_list, &num_samples, &num_samples_scanned, grid, mi_row, mi_col, -1, w4, ref_frame0, mv0, threshold)
	}
	if num_samples == 0 && num_samples_scanned > 0 {
		num_samples = 1
	}
	return num_samples
}
