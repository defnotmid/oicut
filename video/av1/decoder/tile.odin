package av1_decoder

// Tile, superblock, partition, mode info, motion vector, palette, prediction,
// transform tree, coefficient decoding, and reconstruction (AV1 specification
// sections 5.11, 6.10, 7.11, 7.12, 7.13, and 8.3).

import "base:intrinsics"
import "core:simd"
import av1 "../"

@(private)
Block_Decoded_Row :: bit_set[-1 ..< 63]

@(private)
Tile_State :: struct {
	d:            ^Decoder,
	h:            ^av1.Frame_Header,
	s:            ^av1.Sequence_Header,
	cdfs:         ^av1.CDF_Context,
	sym:          av1.Symbol_Decoder,
	grid:         av1.MI_Grid,
	tile_num:     int,
	mi_row_start: int,
	mi_row_end:   int,
	mi_col_start: int,
	mi_col_end:   int,

	// Tile-level delta-Q, delta-LF & loop restoration reference state.
	current_qindex: int,
	delta_lf:       [av1.FRAME_LF_COUNT]i8,
	read_deltas:    bool,
	seg_dc_q:       [3][av1.MAX_SEGMENTS]i32,
	seg_ac_q:       [3][av1.MAX_SEGMENTS]i32,
	ref_sgr_xqd:    [3][2]i8,
	ref_lr_wiener:  [3][2][av1.WIENER_COEFFS]i8,
	sign_bias_u8:   [av1.TOTAL_REFS_PER_FRAME]u8,
	mf_slices:      [av1.TOTAL_REFS_PER_FRAME][]av1.Mv,

	// Tile-local above/left contexts (indexed by plane 4x4 coordinates).
	// Packed as `(dc_category << 6) | cul_level` where `cul_level in 0..63` and `dc_category in 0..2`.
	above_coeff_ctx:    [3][2048 + 32]u8,
	left_coeff_ctx:     [3][64 + 16]u8,
	above_seg_pred_ctx: [2048]u8,
	left_seg_pred_ctx:  [64]u8,
	tx_type_grid:       [16 * 16]u8,

	// Superblock BlockDecoded flags: [plane][y + 1] bit_set[-1 ..< 63; u64] for y in -1 ..= 32.
	block_decoded: [3][36]Block_Decoded_Row,

	// Current block state (set by `decode_block`).
	mi_row, mi_col, mi_size: int,
	bw4, bh4:                int,
	has_chroma:              bool,
	avail_u, avail_l:        bool,
	avail_u_chroma:          bool,
	avail_l_chroma:          bool,
	segment_id:              u8,
	lossless:                bool,
	skip_mode:               u8,
	skip:                    bool,
	is_inter:                bool,
	use_intrabc:             bool,
	ref_frame:               [2]i8,
	y_mode, uv_mode:         u8,
	angle_delta_y:           i8,
	angle_delta_uv:          i8,
	use_filter_intra:        bool,
	filter_intra_mode:       u8,
	palette_size_y:          u8,
	palette_size_uv:         u8,
	palette_colors_y:        [av1.PALETTE_COLORS]u16,
	palette_colors_u:        [av1.PALETTE_COLORS]u16,
	palette_colors_v:        [av1.PALETTE_COLORS]u16,
	color_map_y:             [64 * 64]u8,
	color_map_uv:            [64 * 64]u8,
	cfl_alpha_u:             i16,
	cfl_alpha_v:             i16,
	max_luma_w, max_luma_h:  int,
	tx_size:                 u8,
	plane_tx_type:           u8,
	mv:                      [2]av1.Mv,
	pred_mv:                 [2]av1.Mv,
	ref_mv_idx:              u8,
	mv_stack:                av1.MV_Stack,
	interp_filter:           [2]u8,
	interintra:              bool,
	interintra_mode:         u8,
	wedge_interintra:        bool,
	wedge_index:             u8,
	wedge_sign:              u8,
	motion_mode:             u8,
	comp_group_idx:          u8,
	compound_idx:            u8,
	compound_type:           u8,
	mask_type:               u8,
	num_samples:             int,
	cand_list:               [av1.LEAST_SQUARES_SAMPLES_MAX][4]i32,
	local_warp_valid:        bool,
	local_warp_params:       [6]i32,
	mask_buf:                [128 * 128]u8,
	preds:                   [2][128 * 128]i32,
	levels_buf:              [36 * 64]u8,
	nz_list:                 [1024]u32,
	dequant_buf:             [64 * 64]i32,
	itx_buf:                 [64 * 64]i32,
}

// Section 5.11.2: Decode tile syntax.
@(private)
decode_tile :: proc(
	d: ^Decoder,
	h: ^av1.Frame_Header,
	tile_num: int,
	data: []u8,
	cdfs: ^av1.CDF_Context,
) -> Error {
	t := &d.tile_states[tile_num]

	s := &d.parser.sequence
	tile_row := tile_num / h.tile_info.tile_cols
	tile_col := tile_num % h.tile_info.tile_cols

	t.d = d
	t.h = h
	t.s = s
	t.cdfs = cdfs
	t.tile_num = tile_num
	t.mi_row_start = int(h.tile_info.mi_row_starts[tile_row])
	t.mi_row_end = int(h.tile_info.mi_row_starts[tile_row + 1])
	t.mi_col_start = int(h.tile_info.mi_col_starts[tile_col])
	t.mi_col_end = int(h.tile_info.mi_col_starts[tile_col + 1])
	t.current_qindex = int(h.base_q_idx)
	t.delta_lf = {}
	update_tile_quants(t)

	for p in 0 ..< 3 {
		t.ref_sgr_xqd[p] = av1.SGRPROJ_XQD_MID
		for pass in 0 ..< 2 {
			t.ref_lr_wiener[p][pass] = av1.WIENER_TAPS_MID
		}
	}
	for i in 0 ..< av1.TOTAL_REFS_PER_FRAME {
		t.sign_bias_u8[i] = u8(h.ref_frame_sign_bias[i])
	}
	if h.use_ref_frame_mvs {
		n8 := (h.mi_rows >> 1) * (h.mi_cols >> 1)
		for ref in av1.LAST_FRAME ..= av1.ALTREF_FRAME {
			t.mf_slices[ref] = d.motion_field_mvs[ref][:n8]
		}
	}

	cells := h.mi_rows * h.mi_cols
	t.grid = av1.MI_Grid{
		cells        = d.mi_cells[:cells],
		mi_rows      = h.mi_rows,
		mi_cols      = h.mi_cols,
		mi_row_start = t.mi_row_start,
		mi_row_end   = t.mi_row_end,
		mi_col_start = t.mi_col_start,
		mi_col_end   = t.mi_col_end,
	}

	av1.init_symbol(&t.sym, data, h.disable_cdf_update)
	clear_above_context(t)

	sb_size := s.use_128x128_superblock ? av1.BLOCK_128X128 : av1.BLOCK_64X64
	sb_size4 := int(av1.NUM_4X4_BLOCKS_WIDE[sb_size])

	for r := t.mi_row_start; r < t.mi_row_end; r += sb_size4 {
		clear_left_context(t)
		for c := t.mi_col_start; c < t.mi_col_end; c += sb_size4 {
			t.read_deltas = h.delta_q_present
			clear_cdef(t, r, c)
			clear_block_decoded_flags(t, r, c, sb_size4)
			read_lr(t, r, c, sb_size)
			decode_partition(t, r, c, sb_size) or_return
		}
	}

	if !av1.exit_symbol(&t.sym) {
		return .Invalid_Data
	}
	return .None
}

// Section 5.11.3: Clear above context function.
@(private)
clear_above_context :: proc(t: ^Tile_State) #no_bounds_check {
	w4 := min(len(t.above_seg_pred_ctx), t.mi_col_end - t.mi_col_start)
	w4_pad := w4 + 16
	for p in 0 ..< 3 {
		for &v in t.above_coeff_ctx[p][:w4_pad] do v = 0
	}
	for &v in t.above_seg_pred_ctx[:w4] do v = 0
}

// Section 5.11.4: Clear left context function.
@(private)
clear_left_context :: proc(t: ^Tile_State) {
	t.left_coeff_ctx = {}
	t.left_seg_pred_ctx = {}
}

// Section 5.11.55: Clear CDEF function.
@(private)
clear_cdef :: proc(t: ^Tile_State, r, c: int) {
	sb64_cols := (t.h.mi_cols + 15) >> 4
	r64 := r >> 4
	c64 := c >> 4
	t.d.cdef_idx[r64 * sb64_cols + c64] = -1
	if t.s.use_128x128_superblock {
		cdef_size4 := int(av1.NUM_4X4_BLOCKS_WIDE[av1.BLOCK_64X64])
		if c + cdef_size4 < t.h.mi_cols {
			t.d.cdef_idx[r64 * sb64_cols + (c64 + 1)] = -1
		}
		if r + cdef_size4 < t.h.mi_rows {
			t.d.cdef_idx[(r64 + 1) * sb64_cols + c64] = -1
			if c + cdef_size4 < t.h.mi_cols {
				t.d.cdef_idx[(r64 + 1) * sb64_cols + (c64 + 1)] = -1
			}
		}
	}
}

// Section 5.11.5: Clear block decoded flags function.
@(private)
clear_block_decoded_flags :: proc(t: ^Tile_State, r, c, sb_size4: int) #no_bounds_check {
	num_planes := t.s.color.mono_chrome ? 1 : 3
	for plane in 0 ..< num_planes {
		sub_x := plane > 0 ? uint(t.s.color.subsampling_x) : 0
		sub_y := plane > 0 ? uint(t.s.color.subsampling_y) : 0
		sb_w4 := (t.mi_col_end - c) >> sub_x
		sb_h4 := (t.mi_row_end - r) >> sub_y
		lim_y := sb_size4 >> sub_y
		lim_x := sb_size4 >> sub_x
		top_bits := min(sb_w4, lim_x + 1) + 1
		t.block_decoded[plane][0] = transmute(Block_Decoded_Row)((u64(1) << uint(top_bits)) - 1)
		active_h := min(sb_h4, lim_y)
		for y in 0 ..< active_h {
			t.block_decoded[plane][y + 1] = {-1}
		}
		for y in active_h ..= lim_y {
			t.block_decoded[plane][y + 1] = {}
		}
	}
}

// Section 5.11.57: Read loop restoration syntax.
@(private)
read_lr :: proc(t: ^Tile_State, r, c, b_size: int) {
	if t.h.allow_intrabc do return
	w := int(av1.NUM_4X4_BLOCKS_WIDE[b_size])
	h := int(av1.NUM_4X4_BLOCKS_HIGH[b_size])
	num_planes := t.s.color.mono_chrome ? 1 : 3
	for plane in 0 ..< num_planes {
		if t.h.frame_restoration_type[plane] == av1.RESTORE_NONE do continue
		sub_x := plane > 0 ? uint(t.s.color.subsampling_x) : 0
		sub_y := plane > 0 ? uint(t.s.color.subsampling_y) : 0
		unit_size := int(t.h.loop_restoration_size[plane])
		unit_rows := t.d.lr_unit_rows[plane]
		unit_cols := t.d.lr_unit_cols[plane]
		if unit_rows == 0 || unit_cols == 0 do continue

		unit_row_start := (r * (av1.MI_SIZE >> sub_y) + unit_size - 1) / unit_size
		unit_row_end := min(unit_rows, ((r + h) * (av1.MI_SIZE >> sub_y) + unit_size - 1) / unit_size)
		numerator: int
		denom: int
		if t.h.use_superres {
			numerator = (av1.MI_SIZE >> sub_x) * (int(t.h.coded_denom) + av1.SUPERRES_DENOM_MIN)
			denom = unit_size * av1.SUPERRES_NUM
		} else {
			numerator = av1.MI_SIZE >> sub_x
			denom = unit_size
		}
		unit_col_start := (c * numerator + denom - 1) / denom
		unit_col_end := min(unit_cols, ((c + w) * numerator + denom - 1) / denom)
		for unit_row in unit_row_start ..< unit_row_end {
			for unit_col in unit_col_start ..< unit_col_end {
				read_lr_unit(t, plane, unit_row, unit_col)
			}
		}
	}
}

// Section 5.11.58: Read loop restoration unit syntax.
@(private)
read_lr_unit :: proc(t: ^Tile_State, plane, unit_row, unit_col: int) {
	fr_type := t.h.frame_restoration_type[plane]
	restoration_type := u8(av1.RESTORE_NONE)
	switch fr_type {
	case av1.RESTORE_WIENER:
		use_wiener := av1.read_symbol(&t.sym, t.cdfs.use_wiener_cdf[:])
		restoration_type = use_wiener != 0 ? av1.RESTORE_WIENER : av1.RESTORE_NONE
	case av1.RESTORE_SGRPROJ:
		use_sgrproj := av1.read_symbol(&t.sym, t.cdfs.use_sgrproj_cdf[:])
		restoration_type = use_sgrproj != 0 ? av1.RESTORE_SGRPROJ : av1.RESTORE_NONE
	case av1.RESTORE_SWITCHABLE:
		rt := av1.read_symbol(&t.sym, t.cdfs.restoration_type_cdf[:])
		restoration_type = u8(rt)
	}

	ucols := t.d.lr_unit_cols[plane]
	u_idx := unit_row * ucols + unit_col
	unit := av1.LR_Unit{restoration_type = restoration_type}

	if restoration_type == av1.RESTORE_WIENER {
		for pass in 0 ..< 2 {
			first_coeff := 0
			if plane > 0 {
				first_coeff = 1
				unit.wiener[pass][0] = 0
			}
			for j in first_coeff ..< 3 {
				min_v := int(av1.WIENER_TAPS_MIN[j])
				max_v := int(av1.WIENER_TAPS_MAX[j])
				k := int(av1.WIENER_TAPS_K[j])
				v := av1.read_signed_subexp_with_ref_bool(&t.sym, min_v, max_v + 1, k, i32(t.ref_lr_wiener[plane][pass][j]))
				unit.wiener[pass][j] = i8(v)
				t.ref_lr_wiener[plane][pass][j] = i8(v)
			}
		}
	} else if restoration_type == av1.RESTORE_SGRPROJ {
		lr_sgr_set := u8(av1.read_literal(&t.sym, av1.SGRPROJ_PARAMS_BITS))
		unit.sgr_set = lr_sgr_set
		for i in 0 ..< 2 {
			radius := av1.SGR_PARAMS[lr_sgr_set][i * 2]
			min_v := int(av1.SGRPROJ_XQD_MIN[i])
			max_v := int(av1.SGRPROJ_XQD_MAX[i])
			v: i32
			if radius != 0 {
				v = av1.read_signed_subexp_with_ref_bool(&t.sym, min_v, max_v + 1, av1.SGRPROJ_PRJ_SUBEXP_K, i32(t.ref_sgr_xqd[plane][i]))
			} else {
				v = 0
				if i == 1 {
					v = clamp(i32(1 << av1.SGRPROJ_PRJ_BITS) - i32(t.ref_sgr_xqd[plane][0]), i32(min_v), i32(max_v))
				}
			}
			unit.sgr_xqd[i] = i8(v)
			t.ref_sgr_xqd[plane][i] = i8(v)
		}
	}
	t.d.lr_units[plane][u_idx] = unit
}

// Section 5.11.6 & 8.3.2: Decode partition syntax.
@(private)
decode_partition :: proc(t: ^Tile_State, r, c, b_size: int) -> Error {
	if r >= t.h.mi_rows || c >= t.h.mi_cols do return .None

	avail_u := av1.is_inside_tile(&t.grid, r - 1, c)
	avail_l := av1.is_inside_tile(&t.grid, r, c - 1)
	num4x4 := int(av1.NUM_4X4_BLOCKS_WIDE[b_size])
	half_block4x4 := num4x4 >> 1
	quarter_block4x4 := half_block4x4 >> 1
	has_rows := (r + half_block4x4) < t.h.mi_rows
	has_cols := (c + half_block4x4) < t.h.mi_cols

	bsl := int(av1.MI_WIDTH_LOG2[b_size])
	above := avail_u && int(av1.MI_WIDTH_LOG2[t.grid.cells[(r - 1) * t.grid.mi_cols + c].mi_size]) < bsl
	left := avail_l && int(av1.MI_HEIGHT_LOG2[t.grid.cells[r * t.grid.mi_cols + (c - 1)].mi_size]) < bsl
	ctx := int(left) * 2 + int(above)

	partition_cdf :: proc(cdfs: ^av1.CDF_Context, bsl, ctx: int) -> []u16 {
		switch bsl {
		case 1: return cdfs.partition_w8_cdf[ctx][:]
		case 2: return cdfs.partition_w16_cdf[ctx][:]
		case 3: return cdfs.partition_w32_cdf[ctx][:]
		case 4: return cdfs.partition_w64_cdf[ctx][:]
		case:   return cdfs.partition_w128_cdf[ctx][:]
		}
	}

	partition: int
	if b_size < av1.BLOCK_8X8 {
		partition = av1.PARTITION_NONE
	} else if has_rows && has_cols {
		partition = av1.read_symbol(&t.sym, partition_cdf(t.cdfs, bsl, ctx))
	} else if has_cols {
		pcdf := partition_cdf(t.cdfs, bsl, ctx)
		psum := (pcdf[av1.PARTITION_VERT] - pcdf[av1.PARTITION_VERT - 1]) +
			(pcdf[av1.PARTITION_SPLIT] - pcdf[av1.PARTITION_SPLIT - 1]) +
			(pcdf[av1.PARTITION_HORZ_A] - pcdf[av1.PARTITION_HORZ_A - 1]) +
			(pcdf[av1.PARTITION_VERT_A] - pcdf[av1.PARTITION_VERT_A - 1]) +
			(pcdf[av1.PARTITION_VERT_B] - pcdf[av1.PARTITION_VERT_B - 1])
		if b_size != av1.BLOCK_128X128 {
			psum += pcdf[av1.PARTITION_VERT_4] - pcdf[av1.PARTITION_VERT_4 - 1]
		}
		cdf := [3]u16{u16(32768 - int(psum)), 32768, 0}
		split_or_horz := av1.read_symbol(&t.sym, cdf[:])
		partition = split_or_horz != 0 ? av1.PARTITION_SPLIT : av1.PARTITION_HORZ
	} else if has_rows {
		pcdf := partition_cdf(t.cdfs, bsl, ctx)
		psum := (pcdf[av1.PARTITION_HORZ] - pcdf[av1.PARTITION_HORZ - 1]) +
			(pcdf[av1.PARTITION_SPLIT] - pcdf[av1.PARTITION_SPLIT - 1]) +
			(pcdf[av1.PARTITION_HORZ_A] - pcdf[av1.PARTITION_HORZ_A - 1]) +
			(pcdf[av1.PARTITION_HORZ_B] - pcdf[av1.PARTITION_HORZ_B - 1]) +
			(pcdf[av1.PARTITION_VERT_A] - pcdf[av1.PARTITION_VERT_A - 1])
		if b_size != av1.BLOCK_128X128 {
			psum += pcdf[av1.PARTITION_HORZ_4] - pcdf[av1.PARTITION_HORZ_4 - 1]
		}
		cdf := [3]u16{u16(32768 - int(psum)), 32768, 0}
		split_or_vert := av1.read_symbol(&t.sym, cdf[:])
		partition = split_or_vert != 0 ? av1.PARTITION_SPLIT : av1.PARTITION_VERT
	} else {
		partition = av1.PARTITION_SPLIT
	}

	sub_size := int(av1.PARTITION_SUBSIZE[partition][b_size])
	split_size := int(av1.PARTITION_SUBSIZE[av1.PARTITION_SPLIT][b_size])
	if sub_size == av1.BLOCK_INVALID {
		return .Invalid_Data
	}

	switch partition {
	case av1.PARTITION_NONE:
		decode_block(t, r, c, sub_size) or_return
	case av1.PARTITION_HORZ:
		decode_block(t, r, c, sub_size) or_return
		if has_rows {
			decode_block(t, r + half_block4x4, c, sub_size) or_return
		}
	case av1.PARTITION_VERT:
		decode_block(t, r, c, sub_size) or_return
		if has_cols {
			decode_block(t, r, c + half_block4x4, sub_size) or_return
		}
	case av1.PARTITION_SPLIT:
		decode_partition(t, r, c, sub_size) or_return
		decode_partition(t, r, c + half_block4x4, sub_size) or_return
		decode_partition(t, r + half_block4x4, c, sub_size) or_return
		decode_partition(t, r + half_block4x4, c + half_block4x4, sub_size) or_return
	case av1.PARTITION_HORZ_A:
		decode_block(t, r, c, split_size) or_return
		decode_block(t, r, c + half_block4x4, split_size) or_return
		decode_block(t, r + half_block4x4, c, sub_size) or_return
	case av1.PARTITION_HORZ_B:
		decode_block(t, r, c, sub_size) or_return
		decode_block(t, r + half_block4x4, c, split_size) or_return
		decode_block(t, r + half_block4x4, c + half_block4x4, split_size) or_return
	case av1.PARTITION_VERT_A:
		decode_block(t, r, c, split_size) or_return
		decode_block(t, r + half_block4x4, c, split_size) or_return
		decode_block(t, r, c + half_block4x4, sub_size) or_return
	case av1.PARTITION_VERT_B:
		decode_block(t, r, c, sub_size) or_return
		decode_block(t, r, c + half_block4x4, split_size) or_return
		decode_block(t, r + half_block4x4, c + half_block4x4, split_size) or_return
	case av1.PARTITION_HORZ_4:
		decode_block(t, r + quarter_block4x4 * 0, c, sub_size) or_return
		decode_block(t, r + quarter_block4x4 * 1, c, sub_size) or_return
		decode_block(t, r + quarter_block4x4 * 2, c, sub_size) or_return
		if r + quarter_block4x4 * 3 < t.h.mi_rows {
			decode_block(t, r + quarter_block4x4 * 3, c, sub_size) or_return
		}
	case av1.PARTITION_VERT_4:
		decode_block(t, r, c + quarter_block4x4 * 0, sub_size) or_return
		decode_block(t, r, c + quarter_block4x4 * 1, sub_size) or_return
		decode_block(t, r, c + quarter_block4x4 * 2, sub_size) or_return
		if c + quarter_block4x4 * 3 < t.h.mi_cols {
			decode_block(t, r, c + quarter_block4x4 * 3, sub_size) or_return
		}
	}
	return .None
}

// Section 5.11.7: Decode block syntax.
@(private)
decode_block :: proc(t: ^Tile_State, r, c, sub_size: int) -> Error {
	t.mi_row = r
	t.mi_col = c
	t.mi_size = sub_size
	t.bw4 = int(av1.NUM_4X4_BLOCKS_WIDE[sub_size])
	t.bh4 = int(av1.NUM_4X4_BLOCKS_HIGH[sub_size])

	sub_x := int(t.s.color.subsampling_x)
	sub_y := int(t.s.color.subsampling_y)
	if t.bh4 == 1 && sub_y != 0 && (r & 1) == 0 {
		t.has_chroma = false
	} else if t.bw4 == 1 && sub_x != 0 && (c & 1) == 0 {
		t.has_chroma = false
	} else {
		t.has_chroma = !t.s.color.mono_chrome
	}

	t.avail_u = av1.is_inside_tile(&t.grid, r - 1, c)
	t.avail_l = av1.is_inside_tile(&t.grid, r, c - 1)
	t.avail_u_chroma = t.avail_u
	t.avail_l_chroma = t.avail_l
	if t.has_chroma {
		if sub_y != 0 && t.bh4 == 1 {
			t.avail_u_chroma = av1.is_inside_tile(&t.grid, r - 2, c)
		}
		if sub_x != 0 && t.bw4 == 1 {
			t.avail_l_chroma = av1.is_inside_tile(&t.grid, r, c - 2)
		}
	}

	mode_info(t)
	if t.palette_size_y > 0 || t.palette_size_uv > 0 {
		palette_tokens(t)
	}
	var_tx := read_block_tx_size(t)
	if t.skip {
		reset_block_context(t)
	}
	is_compound := t.ref_frame[1] > av1.INTRA_FRAME
	populate_block_grids(t, is_compound, var_tx)

	compute_prediction(t)
	residual(t) or_return
	return .None
}

@(private)
fill_row_u8 :: #force_inline proc "contextless" (dst: [^]u8, n: int, val: u8) #no_bounds_check {
	if n <= 8 {
		if n < 4 {
			if n <= 0 do return
			dst[0] = val
			dst[n >> 1] = val
			dst[n - 1] = val
			return
		}
		v4 := u32(val) * 0x01010101
		intrinsics.unaligned_store((^u32)(dst), v4)
		intrinsics.unaligned_store((^u32)(&dst[n - 4]), v4)
		return
	}
	if n <= 16 {
		v8 := u64(val) * 0x0101010101010101
		intrinsics.unaligned_store((^u64)(dst), v8)
		intrinsics.unaligned_store((^u64)(&dst[n - 8]), v8)
		return
	}
	v16 := (#simd[16]u8)(val)
	intrinsics.unaligned_store((^#simd[16]u8)(dst), v16)
	intrinsics.unaligned_store((^#simd[16]u8)(&dst[n - 16]), v16)
}

@(private)
fill_row_u16 :: #force_inline proc "contextless" (dst: [^]u16, n: int, val: u16) #no_bounds_check {
	if n <= 8 {
		if n < 4 {
			if n <= 0 do return
			dst[0] = val
			dst[n >> 1] = val
			dst[n - 1] = val
			return
		}
		v4 := u64(val) * 0x0001000100010001
		intrinsics.unaligned_store((^u64)(dst), v4)
		intrinsics.unaligned_store((^u64)(&dst[n - 4]), v4)
		return
	}
	if n <= 16 {
		v8 := (#simd[8]u16)(val)
		intrinsics.unaligned_store((^#simd[8]u16)(dst), v8)
		intrinsics.unaligned_store((^#simd[8]u16)(&dst[n - 8]), v8)
		return
	}
	v16 := (#simd[16]u16)(val)
	intrinsics.unaligned_store((^#simd[16]u16)(dst), v16)
	intrinsics.unaligned_store((^#simd[16]u16)(&dst[n - 16]), v16)
}

@(private)
fill_row_u32 :: #force_inline proc "contextless" (dst: [^]u32, n: int, val: u32) #no_bounds_check {
	switch n {
	case 1:
		dst[0] = val
	case 2:
		intrinsics.unaligned_store((^u64)(dst), u64(val) | (u64(val) << 32))
	case 4:
		intrinsics.unaligned_store((^#simd[4]u32)(dst), (#simd[4]u32)(val))
	case 8:
		intrinsics.unaligned_store((^#simd[8]u32)(dst), (#simd[8]u32)(val))
	case 16:
		v := (#simd[8]u32)(val)
		intrinsics.unaligned_store((^#simd[8]u32)(dst), v)
		intrinsics.unaligned_store((^#simd[8]u32)(&dst[8]), v)
	case 32:
		v := (#simd[8]u32)(val)
		intrinsics.unaligned_store((^#simd[8]u32)(dst), v)
		intrinsics.unaligned_store((^#simd[8]u32)(&dst[8]), v)
		intrinsics.unaligned_store((^#simd[8]u32)(&dst[16]), v)
		intrinsics.unaligned_store((^#simd[8]u32)(&dst[24]), v)
	case:
		for i in 0 ..< n {
			dst[i] = val
		}
	}
}

@(private)
fill_row_u64 :: #force_inline proc "contextless" (dst: [^]u64, n: int, val: u64) #no_bounds_check {
	switch n {
	case 1:
		dst[0] = val
	case 2:
		intrinsics.unaligned_store((^#simd[2]u64)(dst), (#simd[2]u64)(val))
	case 4:
		intrinsics.unaligned_store((^#simd[4]u64)(dst), (#simd[4]u64)(val))
	case 8:
		v := (#simd[4]u64)(val)
		intrinsics.unaligned_store((^#simd[4]u64)(dst), v)
		intrinsics.unaligned_store((^#simd[4]u64)(&dst[4]), v)
	case 16:
		v := (#simd[4]u64)(val)
		intrinsics.unaligned_store((^#simd[4]u64)(dst), v)
		intrinsics.unaligned_store((^#simd[4]u64)(&dst[4]), v)
		intrinsics.unaligned_store((^#simd[4]u64)(&dst[8]), v)
		intrinsics.unaligned_store((^#simd[4]u64)(&dst[12]), v)
	case 32:
		v := (#simd[4]u64)(val)
		intrinsics.unaligned_store((^#simd[4]u64)(dst), v)
		intrinsics.unaligned_store((^#simd[4]u64)(&dst[4]), v)
		intrinsics.unaligned_store((^#simd[4]u64)(&dst[8]), v)
		intrinsics.unaligned_store((^#simd[4]u64)(&dst[12]), v)
		intrinsics.unaligned_store((^#simd[4]u64)(&dst[16]), v)
		intrinsics.unaligned_store((^#simd[4]u64)(&dst[20]), v)
		intrinsics.unaligned_store((^#simd[4]u64)(&dst[24]), v)
		intrinsics.unaligned_store((^#simd[4]u64)(&dst[28]), v)
	case:
		for i in 0 ..< n {
			dst[i] = val
		}
	}
}

@(private)
fill_boundary_u8 :: #force_inline proc "contextless" (
	grid: [^]u8,
	mi_row, mi_col, mi_cols, end_y, end_x: int,
	val: u8,
) #no_bounds_check {
	if end_y <= 0 || end_x <= 0 do return
	bot_base := (mi_row + end_y - 1) * mi_cols + mi_col
	fill_row_u8(&grid[bot_base], end_x, val)
	right_col := mi_col + end_x - 1
	for y in 0 ..< end_y - 1 {
		grid[(mi_row + y) * mi_cols + right_col] = val
	}
}

@(private)
fill_boundary_u16 :: #force_inline proc "contextless" (
	grid: [^]u16,
	mi_row, mi_col, mi_cols, end_y, end_x: int,
	val: u16,
) #no_bounds_check {
	if end_y <= 0 || end_x <= 0 do return
	bot_base := (mi_row + end_y - 1) * mi_cols + mi_col
	fill_row_u16(&grid[bot_base], end_x, val)
	right_col := mi_col + end_x - 1
	for y in 0 ..< end_y - 1 {
		grid[(mi_row + y) * mi_cols + right_col] = val
	}
}

@(private)
populate_block_grids :: proc(t: ^Tile_State, is_compound, var_tx: bool) #no_bounds_check {
	d := t.d
	h := t.h
	mvs := t.mv
	if !is_compound {
		mvs[1] = {0, 0}
	}
	mvs_u64 := transmute(u64)mvs
	refs_u16 := transmute(u16)t.ref_frame
	mi_sz_u8 := u8(t.mi_size)
	y_mode := t.y_mode
	is_inter := t.is_inter
	is_inter_u8 := u8(is_inter)
	skip := t.skip
	skip_u8 := u8(skip)
	tx_sz := t.tx_size

	end_y := min(t.bh4, h.mi_rows - t.mi_row)
	end_x := min(t.bw4, h.mi_cols - t.mi_col)
	mi_cols := h.mi_cols

	// 1. Mode info & tile-local syntax grids (`d.mi_cells`, `d.tile_cells`).
	mi_mvs := ([^]u64)(d.mi_cells.mvs)
	mi_refs := ([^]u16)(d.mi_cells.ref_frames)
	mi_sizes := d.mi_cells.mi_size
	mi_ymodes := d.mi_cells.y_mode
	mi_inters := ([^]u8)(d.mi_cells.is_inter)
	tc_skips := ([^]u8)(d.tile_cells.skip)
	tc_inter_tx := d.tile_cells.inter_tx_size

	if is_inter {
		tc_filters := ([^]u16)(d.tile_cells.interp_filters)
		flt_u16 := transmute(u16)t.interp_filter
		for y in 0 ..< end_y {
			row_base := (t.mi_row + y) * mi_cols + t.mi_col
			fill_row_u64(&mi_mvs[row_base], end_x, mvs_u64)
			fill_row_u16(&mi_refs[row_base], end_x, refs_u16)
			fill_row_u8(&mi_sizes[row_base], end_x, mi_sz_u8)
			fill_row_u8(&mi_ymodes[row_base], end_x, y_mode)
			fill_row_u8(&mi_inters[row_base], end_x, is_inter_u8)
			fill_row_u8(&tc_skips[row_base], end_x, skip_u8)
			if !var_tx do fill_row_u8(&tc_inter_tx[row_base], end_x, tx_sz)
			fill_row_u16(&tc_filters[row_base], end_x, flt_u16)
		}
		if is_compound {
			fill_boundary_u8(d.tile_cells.comp_group_idx, t.mi_row, t.mi_col, mi_cols, end_y, end_x, t.comp_group_idx)
			fill_boundary_u8(d.tile_cells.compound_idx, t.mi_row, t.mi_col, mi_cols, end_y, end_x, t.compound_idx)
		}
	} else {
		for y in 0 ..< end_y {
			row_base := (t.mi_row + y) * mi_cols + t.mi_col
			fill_row_u64(&mi_mvs[row_base], end_x, mvs_u64)
			fill_row_u16(&mi_refs[row_base], end_x, refs_u16)
			fill_row_u8(&mi_sizes[row_base], end_x, mi_sz_u8)
			fill_row_u8(&mi_ymodes[row_base], end_x, y_mode)
			fill_row_u8(&mi_inters[row_base], end_x, is_inter_u8)
			fill_row_u8(&tc_skips[row_base], end_x, skip_u8)
			fill_row_u8(&tc_inter_tx[row_base], end_x, tx_sz)
		}
		fill_boundary_u8(d.tile_cells.uv_mode, t.mi_row, t.mi_col, mi_cols, end_y, end_x, t.uv_mode)
	}

	if h.segmentation_enabled {
		seg_ids := raw_data(d.segment_ids)
		seg_id := t.segment_id
		for y in 0 ..< end_y {
			row_base := (t.mi_row + y) * mi_cols + t.mi_col
			fill_row_u8(&seg_ids[row_base], end_x, seg_id)
		}
	}
	if h.skip_mode_present {
		fill_boundary_u8(d.tile_cells.skip_mode, t.mi_row, t.mi_col, mi_cols, end_y, end_x, t.skip_mode)
	}
	if h.allow_screen_content_tools {
		tc_pal_sizes := ([^]u16)(d.tile_cells.palette_sizes)
		ps_u16 := transmute(u16)([2]u8{t.palette_size_y, t.palette_size_uv})
		fill_boundary_u16(tc_pal_sizes, t.mi_row, t.mi_col, mi_cols, end_y, end_x, ps_u16)
		if ps_u16 != 0 && end_y > 0 && end_x > 0 {
			tc_pal_colors := d.tile_cells.palette_colors
			pal_cols := [2][av1.PALETTE_COLORS]u16{t.palette_colors_y, t.palette_colors_u}
			bot_base := (t.mi_row + end_y - 1) * mi_cols + t.mi_col
			for x in 0 ..< end_x {
				tc_pal_colors[bot_base + x] = pal_cols
			}
			for y in 0 ..< end_y - 1 {
				tc_pal_colors[(t.mi_row + y) * mi_cols + t.mi_col + end_x - 1] = pal_cols
			}
		}
	}

	// 2. Deblocking loop filter grid (`d.lf_cells`; `tx_size_y` and `mi_size` alias `tile_cells.inter_tx_size` and `mi_cells.mi_size`).
	if !h.all_lossless && (h.loop_filter_level[0] != 0 || h.loop_filter_level[1] != 0) {
		levels_u32 := transmute(u32)av1.compute_lf_levels(h, t.segment_id, t.ref_frame[0], t.y_mode, t.delta_lf)
		mi_size_chunk := (t.bw4 > 16 || t.bh4 > 16) ? av1.BLOCK_64X64 : t.mi_size
		uv_tx := u8(av1.get_uv_tx_size(mi_size_chunk, 1, int(t.tx_size), t.lossless, t.s.color.subsampling_x, t.s.color.subsampling_y))
		skip_inter_u8 := u8(skip && is_inter)
		lf_levels := ([^]u32)(d.lf_cells.levels)
		lf_tx_uv := d.lf_cells.tx_size_uv
		lf_skip_inter := ([^]u8)(d.lf_cells.skip_inter)
		for y in 0 ..< end_y {
			row_base := (t.mi_row + y) * mi_cols + t.mi_col
			fill_row_u32(&lf_levels[row_base], end_x, levels_u32)
			fill_row_u8(&lf_tx_uv[row_base], end_x, uv_tx)
			fill_row_u8(&lf_skip_inter[row_base], end_x, skip_inter_u8)
		}
	}
}

// Section 5.11.33: Reset block context function.
@(private)
reset_block_context :: proc(t: ^Tile_State) #no_bounds_check {
	num_planes := 1 + (t.has_chroma ? 2 : 0)
	for plane in 0 ..< num_planes {
		sub_x := plane > 0 ? uint(t.s.color.subsampling_x) : 0
		sub_y := plane > 0 ? uint(t.s.color.subsampling_y) : 0
		tile_x4 := t.mi_col_start >> sub_x
		c_start := (t.mi_col >> sub_x) - tile_x4
		c_len := ((t.mi_col + t.bw4) >> sub_x) - (t.mi_col >> sub_x)
		r_start := (t.mi_row >> sub_y) & 63
		r_len := ((t.mi_row + t.bh4) >> sub_y) - (t.mi_row >> sub_y)
		fill_row_u8(&t.above_coeff_ctx[plane][c_start], c_len, 0)
		fill_row_u8(&t.left_coeff_ctx[plane][r_start], r_len, 0)
	}
}

// Section 5.11.8: Mode info syntax.
@(private)
mode_info :: proc(t: ^Tile_State) {
	t.interintra = false
	t.wedge_interintra = false
	t.use_filter_intra = false
	t.palette_size_y = 0
	t.palette_size_uv = 0
	t.angle_delta_y = 0
	t.angle_delta_uv = 0
	t.cfl_alpha_u = 0
	t.cfl_alpha_v = 0
	t.comp_group_idx = 0
	t.compound_idx = 1
	t.motion_mode = av1.SIMPLE
	t.compound_type = av1.COMPOUND_AVERAGE
	t.mv = {}

	if t.h.frame_is_intra {
		intra_frame_mode_info(t)
	} else {
		inter_frame_mode_info(t)
	}
}

// Section 5.11.9: Intra frame mode info syntax.
@(private)
intra_frame_mode_info :: #force_no_inline proc(t: ^Tile_State) {
	t.skip = false
	if t.d.seg_id_pre_skip {
		intra_segment_id(t)
	}
	t.skip_mode = 0
	read_skip(t)
	if !t.d.seg_id_pre_skip {
		intra_segment_id(t)
	}
	t.lossless = t.d.lossless_array[t.segment_id]
	read_cdef(t)
	read_delta_qindex(t)
	read_delta_lf(t)
	t.read_deltas = false
	t.ref_frame = {av1.INTRA_FRAME, av1.NONE}
	if t.h.allow_intrabc {
		use_intrabc := av1.read_symbol(&t.sym, t.cdfs.intrabc_cdf[:]) != 0
		t.use_intrabc = use_intrabc
	} else {
		t.use_intrabc = false
	}
	if t.use_intrabc {
		t.is_inter = true
		t.y_mode = av1.DC_PRED
		t.uv_mode = av1.DC_PRED
		t.motion_mode = av1.SIMPLE
		t.compound_type = av1.COMPOUND_AVERAGE
		t.palette_size_y = 0
		t.palette_size_uv = 0
		t.interp_filter = {av1.BILINEAR, av1.BILINEAR}
		av1.find_mv_stack(
			&t.mv_stack, &t.grid,
			t.mi_row, t.mi_col, t.mi_size,
			t.ref_frame, false,
			t.h.allow_high_precision_mv, t.h.force_integer_mv,
			false, nil,
			t.h.gm_type, t.h.gm_params, {},
		)
		assign_mv(t, false)
	} else {
		t.is_inter = false
		above_mode := av1.INTRA_MODE_CONTEXT[t.avail_u ? t.grid.cells[(t.mi_row - 1) * t.grid.mi_cols + t.mi_col].y_mode : av1.DC_PRED]
		left_mode := av1.INTRA_MODE_CONTEXT[t.avail_l ? t.grid.cells[t.mi_row * t.grid.mi_cols + (t.mi_col - 1)].y_mode : av1.DC_PRED]
		t.y_mode = u8(av1.read_symbol(&t.sym, t.cdfs.intra_frame_y_mode_cdf[above_mode][left_mode][:]))
		intra_angle_info_y(t)
		if t.has_chroma {
			cfl_allowed := is_cfl_allowed(t)
			if cfl_allowed {
				t.uv_mode = u8(av1.read_symbol(&t.sym, t.cdfs.uv_mode_cfl_allowed_cdf[t.y_mode][:]))
			} else {
				t.uv_mode = u8(av1.read_symbol(&t.sym, t.cdfs.uv_mode_cfl_not_allowed_cdf[t.y_mode][:]))
			}
			if t.uv_mode == av1.UV_CFL_PRED {
				read_cfl_alphas(t)
			} else {
				intra_angle_info_uv(t)
			}
		}
		if t.mi_size >= av1.BLOCK_8X8 &&
		   av1.BLOCK_WIDTH[t.mi_size] <= 64 &&
		   av1.BLOCK_HEIGHT[t.mi_size] <= 64 &&
		   t.h.allow_screen_content_tools {
			palette_mode_info(t)
		}
		filter_intra_mode_info(t)
	}
}

@(private)
is_cfl_allowed :: #force_inline proc "contextless" (t: ^Tile_State) -> bool {
	if t.lossless {
		return av1.get_plane_residual_size(t.mi_size, 1, uint(t.s.color.subsampling_x), uint(t.s.color.subsampling_y)) == av1.BLOCK_4X4
	}
	return max(av1.BLOCK_WIDTH[t.mi_size], av1.BLOCK_HEIGHT[t.mi_size]) <= 32
}

// Section 5.11.10: Intra segment ID syntax.
@(private)
intra_segment_id :: proc(t: ^Tile_State) {
	if t.h.segmentation_enabled {
		read_segment_id(t)
	} else {
		t.segment_id = 0
	}
	t.lossless = t.d.lossless_array[t.segment_id]
}

// Section 5.11.11: Read segment ID syntax.
@(private)
read_segment_id :: proc(t: ^Tile_State) {
	prev_ul := t.avail_u && t.avail_l ? int(t.d.segment_ids[(t.mi_row - 1) * t.h.mi_cols + (t.mi_col - 1)]) : -1
	prev_u := t.avail_u ? int(t.d.segment_ids[(t.mi_row - 1) * t.h.mi_cols + t.mi_col]) : -1
	prev_l := t.avail_l ? int(t.d.segment_ids[t.mi_row * t.h.mi_cols + (t.mi_col - 1)]) : -1

	pred: int
	if prev_u == -1 {
		pred = prev_l == -1 ? 0 : prev_l
	} else if prev_l == -1 {
		pred = prev_u
	} else {
		pred = prev_ul == prev_u ? prev_u : prev_l
	}

	if t.skip {
		t.segment_id = u8(pred)
		return
	}

	ctx := 0
	if prev_ul < 0 {
		ctx = 0
	} else if prev_ul == prev_u && prev_ul == prev_l {
		ctx = 2
	} else if prev_ul == prev_u || prev_ul == prev_l || prev_u == prev_l {
		ctx = 1
	}

	sym := av1.read_symbol(&t.sym, t.cdfs.segment_id_cdf[ctx][:])
	seg := neg_deinterleave(sym, pred, t.d.last_active_seg_id + 1)
	t.segment_id = u8(clamp(seg, 0, av1.MAX_SEGMENTS - 1))
}

@(private)
neg_deinterleave :: #force_inline proc "contextless" (diff, ref, mx: int) -> int {
	if ref == 0 do return diff
	if ref >= mx - 1 do return mx - diff - 1
	if 2 * ref < mx {
		if diff <= 2 * ref {
			if (diff & 1) != 0 do return ref + ((diff + 1) >> 1)
			return ref - (diff >> 1)
		}
		return diff
	} else {
		if diff <= 2 * (mx - ref - 1) {
			if (diff & 1) != 0 do return ref + ((diff + 1) >> 1)
			return ref - (diff >> 1)
		}
		return mx - (diff + 1)
	}
}

// Section 5.11.12: Skip mode syntax.
@(private)
read_skip_mode :: proc(t: ^Tile_State) {
	if (t.d.seg_id_pre_skip && seg_feature_active(t, av1.SEG_LVL_SKIP)) ||
	   (t.d.seg_id_pre_skip && seg_feature_active(t, av1.SEG_LVL_REF_FRAME)) ||
	   (t.d.seg_id_pre_skip && seg_feature_active(t, av1.SEG_LVL_GLOBALMV)) ||
	   !t.h.skip_mode_present ||
	   min(av1.BLOCK_WIDTH[t.mi_size], av1.BLOCK_HEIGHT[t.mi_size]) < 8 {
		t.skip_mode = 0
		return
	}
	ctx := 0
	if t.avail_u do ctx += int(t.d.tile_cells[(t.mi_row - 1) * t.h.mi_cols + t.mi_col].skip_mode)
	if t.avail_l do ctx += int(t.d.tile_cells[t.mi_row * t.h.mi_cols + (t.mi_col - 1)].skip_mode)
	t.skip_mode = u8(av1.read_symbol(&t.sym, t.cdfs.skip_mode_cdf[ctx][:]))
}

// Section 5.11.13: Skip syntax.
@(private)
read_skip :: proc(t: ^Tile_State) {
	if t.d.seg_id_pre_skip && seg_feature_active(t, av1.SEG_LVL_SKIP) {
		t.skip = true
	} else {
		ctx := 0
		if t.avail_u && t.d.tile_cells[(t.mi_row - 1) * t.h.mi_cols + t.mi_col].skip do ctx += 1
		if t.avail_l && t.d.tile_cells[t.mi_row * t.h.mi_cols + (t.mi_col - 1)].skip do ctx += 1
		t.skip = av1.read_symbol(&t.sym, t.cdfs.skip_cdf[ctx][:]) != 0
	}
}

// Section 5.11.14: Quantizer index delta syntax.
@(private)
update_tile_quants :: proc(t: ^Tile_State) #no_bounds_check {
	h := t.h
	bit_depth := int(t.s.color.bit_depth)
	max_seg := h.segmentation_enabled ? av1.MAX_SEGMENTS : 1
	for seg in 0 ..< max_seg {
		qindex := get_qindex(h, false, t.current_qindex, seg)
		t.seg_dc_q[0][seg] = av1.dc_quant(qindex, h.delta_q_y_dc, bit_depth)
		t.seg_ac_q[0][seg] = av1.ac_quant(qindex, 0, bit_depth)
		t.seg_dc_q[1][seg] = av1.dc_quant(qindex, h.delta_q_u_dc, bit_depth)
		t.seg_ac_q[1][seg] = av1.ac_quant(qindex, h.delta_q_u_ac, bit_depth)
		t.seg_dc_q[2][seg] = av1.dc_quant(qindex, h.delta_q_v_dc, bit_depth)
		t.seg_ac_q[2][seg] = av1.ac_quant(qindex, h.delta_q_v_ac, bit_depth)
	}
}

@(private)
read_delta_qindex :: proc(t: ^Tile_State) {
	sb_size := t.s.use_128x128_superblock ? av1.BLOCK_128X128 : av1.BLOCK_64X64
	if t.mi_size == sb_size && t.skip do return
	if t.read_deltas {
		delta_q_abs := av1.read_symbol(&t.sym, t.cdfs.delta_q_cdf[:])
		if delta_q_abs == av1.DELTA_Q_SMALL {
			delta_q_rem_bits := int(av1.read_literal(&t.sym, 3)) + 1
			delta_q_abs_bits := int(av1.read_literal(&t.sym, delta_q_rem_bits))
			delta_q_abs = delta_q_abs_bits + (1 << uint(delta_q_rem_bits)) + 1
		}
		if delta_q_abs != 0 {
			delta_q_sign := av1.read_bool(&t.sym) != 0
			reduced := delta_q_sign ? -delta_q_abs : delta_q_abs
			t.current_qindex = clamp(t.current_qindex + (reduced << uint(t.h.delta_q_res)), 1, 255)
			update_tile_quants(t)
		}
	}
}

// Section 5.11.15: Loop filter delta syntax.
@(private)
read_delta_lf :: proc(t: ^Tile_State) {
	sb_size := t.s.use_128x128_superblock ? av1.BLOCK_128X128 : av1.BLOCK_64X64
	if t.mi_size == sb_size && t.skip do return
	if t.read_deltas && t.h.delta_lf_present {
		frame_lf_count := 1
		if t.h.delta_lf_multi {
			frame_lf_count = !t.s.color.mono_chrome ? av1.FRAME_LF_COUNT : av1.FRAME_LF_COUNT - 2
		}
		for i in 0 ..< frame_lf_count {
			cdf := t.h.delta_lf_multi ? t.cdfs.delta_lf_multi_cdf[i][:] : t.cdfs.delta_lf_cdf[:]
			delta_lf_abs := av1.read_symbol(&t.sym, cdf)
			if delta_lf_abs == av1.DELTA_LF_SMALL {
				rem_bits := int(av1.read_literal(&t.sym, 3)) + 1
				abs_bits := int(av1.read_literal(&t.sym, rem_bits))
				delta_lf_abs = abs_bits + (1 << uint(rem_bits)) + 1
			}
			if delta_lf_abs != 0 {
				sign := av1.read_bool(&t.sym) != 0
				reduced := sign ? -delta_lf_abs : delta_lf_abs
				t.delta_lf[i] = i8(clamp(int(t.delta_lf[i]) + (reduced << uint(t.h.delta_lf_res)), -av1.MAX_LOOP_FILTER, av1.MAX_LOOP_FILTER))
			}
		}
	}
}

// Section 5.11.56: Read CDEF syntax.
@(private)
read_cdef :: proc(t: ^Tile_State) {
	if t.skip || t.h.coded_lossless || !t.s.enable_cdef || t.h.allow_intrabc {
		return
	}
	cdef_size4 := int(av1.NUM_4X4_BLOCKS_WIDE[av1.BLOCK_64X64])
	cdef_mask4 := ~(cdef_size4 - 1)
	r := t.mi_row & cdef_mask4
	c := t.mi_col & cdef_mask4
	sb64_cols := (t.h.mi_cols + 15) >> 4
	r64 := r >> 4
	c64 := c >> 4
	if t.d.cdef_idx[r64 * sb64_cols + c64] == -1 {
		val := i8(av1.read_literal(&t.sym, int(t.h.cdef_bits)))
		w4 := int(av1.NUM_4X4_BLOCKS_WIDE[t.mi_size])
		h4 := int(av1.NUM_4X4_BLOCKS_HIGH[t.mi_size])
		for y := r; y < t.mi_row + h4; y += cdef_size4 {
			for x := c; x < t.mi_col + w4; x += cdef_size4 {
				t.d.cdef_idx[(y >> 4) * sb64_cols + (x >> 4)] = val
			}
		}
	}
}

// Section 5.11.16: Segmentation feature active function.
@(private)
seg_feature_active :: #force_inline proc "contextless" (t: ^Tile_State, feature: int) -> bool {
	return t.h.segmentation_enabled && (t.h.feature_enabled[t.segment_id] & (u8(1) << uint(feature))) != 0
}

// Section 5.11.17 - 5.11.19: Read block TX size syntax.
@(private)
read_tx_size :: proc(t: ^Tile_State, allow_select: bool) {
	if t.lossless {
		t.tx_size = av1.TX_4X4
		return
	}
	max_rect_tx_size := int(av1.MAX_TX_SIZE_RECT[t.mi_size])
	max_tx_depth := int(av1.MAX_TX_DEPTH[t.mi_size])
	t.tx_size = u8(max_rect_tx_size)
	if t.mi_size > av1.BLOCK_4X4 && allow_select && t.h.tx_mode == av1.TX_MODE_SELECT {
		max_tx_w := int(av1.TX_WIDTH[max_rect_tx_size])
		max_tx_h := int(av1.TX_HEIGHT[max_rect_tx_size])
		above_w := 0
		if t.avail_u {
			u_idx := (t.mi_row - 1) * t.h.mi_cols + t.mi_col
			if t.grid.cells[u_idx].is_inter {
				above_w = int(av1.BLOCK_WIDTH[t.grid.cells[u_idx].mi_size])
			} else {
				above_w = get_above_tx_width(t, t.mi_row, t.mi_col)
			}
		}
		left_h := 0
		if t.avail_l {
			l_idx := t.mi_row * t.h.mi_cols + (t.mi_col - 1)
			if t.grid.cells[l_idx].is_inter {
				left_h = int(av1.BLOCK_HEIGHT[t.grid.cells[l_idx].mi_size])
			} else {
				left_h = get_left_tx_height(t, t.mi_row, t.mi_col)
			}
		}
		ctx := int(above_w >= max_tx_w) + int(left_h >= max_tx_h)
		tx_depth: int
		switch max_tx_depth {
		case 4: tx_depth = av1.read_symbol(&t.sym, t.cdfs.tx_64x64_cdf[ctx][:])
		case 3: tx_depth = av1.read_symbol(&t.sym, t.cdfs.tx_32x32_cdf[ctx][:])
		case 2: tx_depth = av1.read_symbol(&t.sym, t.cdfs.tx_16x16_cdf[ctx][:])
		case:   tx_depth = av1.read_symbol(&t.sym, t.cdfs.tx_8x8_cdf[ctx][:])
		}
		for _ in 0 ..< tx_depth {
			t.tx_size = av1.SPLIT_TX_SIZE[t.tx_size]
		}
	}
}

@(private)
get_above_tx_width :: #force_inline proc "contextless" (t: ^Tile_State, row, col: int) -> int {
	if row == t.mi_row {
		if !t.avail_u do return 64
		u_idx := (row - 1) * t.h.mi_cols + col
		if t.d.tile_cells[u_idx].skip && t.grid.cells[u_idx].is_inter {
			return int(av1.BLOCK_WIDTH[t.grid.cells[u_idx].mi_size])
		}
	}
	return int(av1.TX_WIDTH[t.d.tile_cells[(row - 1) * t.h.mi_cols + col].inter_tx_size])
}

@(private)
get_left_tx_height :: #force_inline proc "contextless" (t: ^Tile_State, row, col: int) -> int {
	if col == t.mi_col {
		if !t.avail_l do return 64
		l_idx := row * t.h.mi_cols + (col - 1)
		if t.d.tile_cells[l_idx].skip && t.grid.cells[l_idx].is_inter {
			return int(av1.BLOCK_HEIGHT[t.grid.cells[l_idx].mi_size])
		}
	}
	return int(av1.TX_HEIGHT[t.d.tile_cells[row * t.h.mi_cols + (col - 1)].inter_tx_size])
}

@(private)
read_block_tx_size :: proc(t: ^Tile_State) -> bool {
	bw4 := t.bw4
	bh4 := t.bh4
	if t.h.tx_mode == av1.TX_MODE_SELECT &&
	   t.mi_size > av1.BLOCK_4X4 &&
	   t.is_inter &&
	   !t.skip &&
	   !t.lossless {
		max_tx_sz := int(av1.MAX_TX_SIZE_RECT[t.mi_size])
		tx_w4 := int(av1.TX_WIDTH[max_tx_sz]) / av1.MI_SIZE
		tx_h4 := int(av1.TX_HEIGHT[max_tx_sz]) / av1.MI_SIZE
		for row := t.mi_row; row < t.mi_row + bh4; row += tx_h4 {
			for col := t.mi_col; col < t.mi_col + bw4; col += tx_w4 {
				read_var_tx_size(t, row, col, max_tx_sz, 0)
			}
		}
		return true
	}
	read_tx_size(t, !t.skip || !t.is_inter)
	return false
}

@(private)
read_var_tx_size :: proc(t: ^Tile_State, row, col, tx_sz, depth: int) #no_bounds_check {
	if row >= t.h.mi_rows || col >= t.h.mi_cols do return
	txfm_split := 0
	if tx_sz == av1.TX_4X4 || depth == av1.MAX_VARTX_DEPTH {
		txfm_split = 0
	} else {
		above := int(get_above_tx_width(t, row, col) < int(av1.TX_WIDTH[tx_sz]))
		left := int(get_left_tx_height(t, row, col) < int(av1.TX_HEIGHT[tx_sz]))
		sz := min(64, int(max(av1.BLOCK_WIDTH[t.mi_size], av1.BLOCK_HEIGHT[t.mi_size])))
		max_tx_sz := av1.find_tx_size(sz, sz)
		tx_sz_sqr_up := int(av1.TX_SIZE_SQR_UP[tx_sz])
		ctx := int(tx_sz_sqr_up != max_tx_sz) * 3 + (av1.TX_SIZES - 1 - max_tx_sz) * 6 + above + left
		txfm_split = av1.read_symbol(&t.sym, t.cdfs.txfm_split_cdf[ctx][:])
	}
	w4 := int(av1.TX_WIDTH[tx_sz]) / av1.MI_SIZE
	h4 := int(av1.TX_HEIGHT[tx_sz]) / av1.MI_SIZE
	if txfm_split != 0 {
		sub_tx_sz := int(av1.SPLIT_TX_SIZE[tx_sz])
		step_w := int(av1.TX_WIDTH[sub_tx_sz]) / av1.MI_SIZE
		step_h := int(av1.TX_HEIGHT[sub_tx_sz]) / av1.MI_SIZE
		for i := 0; i < h4; i += step_h {
			for j := 0; j < w4; j += step_w {
				read_var_tx_size(t, row + i, col + j, sub_tx_sz, depth + 1)
			}
		}
	} else {
		inter_tx := t.d.tile_cells.inter_tx_size
		tx_u8 := u8(tx_sz)
		end_i := min(h4, t.h.mi_rows - row)
		end_j := min(w4, t.h.mi_cols - col)
		for i in 0 ..< end_i {
			row_base := (row + i) * t.h.mi_cols + col
			fill_row_u8(&inter_tx[row_base], end_j, tx_u8)
		}
		t.tx_size = tx_u8
	}
}

// Section 5.11.20: Inter frame mode info syntax.
@(private)
inter_frame_mode_info :: proc(t: ^Tile_State) {
	t.use_intrabc = false
	left_ref_frame: [2]i8
	left_intra: bool
	left_single: bool
	if t.avail_l {
		l_idx := t.mi_row * t.h.mi_cols + (t.mi_col - 1)
		left_ref_frame = t.grid.cells[l_idx].ref_frames
		left_intra = left_ref_frame[0] <= av1.INTRA_FRAME
		left_single = left_ref_frame[1] <= av1.INTRA_FRAME
	} else {
		left_ref_frame = {av1.INTRA_FRAME, av1.NONE}
		left_intra = true
		left_single = true
	}

	above_ref_frame: [2]i8
	above_intra: bool
	above_single: bool
	if t.avail_u {
		u_idx := (t.mi_row - 1) * t.h.mi_cols + t.mi_col
		above_ref_frame = t.grid.cells[u_idx].ref_frames
		above_intra = above_ref_frame[0] <= av1.INTRA_FRAME
		above_single = above_ref_frame[1] <= av1.INTRA_FRAME
	} else {
		above_ref_frame = {av1.INTRA_FRAME, av1.NONE}
		above_intra = true
		above_single = true
	}

	t.skip = false
	inter_segment_id(t, true)
	read_skip_mode(t)
	if t.skip_mode != 0 {
		t.skip = true
	} else {
		read_skip(t)
	}
	if !t.d.seg_id_pre_skip {
		inter_segment_id(t, false)
	}
	t.lossless = t.d.lossless_array[t.segment_id]
	read_cdef(t)
	read_delta_qindex(t)
	read_delta_lf(t)
	t.read_deltas = false

	read_is_inter(t, above_intra, left_intra)
	if t.is_inter {
		inter_block_mode_info(t, above_ref_frame, left_ref_frame, above_intra, left_intra, above_single, left_single)
	} else {
		intra_block_mode_info(t)
	}
}

// Section 5.11.21: Inter segment ID syntax.
@(private)
inter_segment_id :: proc(t: ^Tile_State, pre_skip: bool) {
	if !t.h.segmentation_enabled {
		t.segment_id = 0
		return
	}
	predicted_segment_id := get_segment_id(t)
	if !t.h.segmentation_update_map {
		t.segment_id = predicted_segment_id
		return
	}
	if pre_skip {
		if !t.d.seg_id_pre_skip {
			t.segment_id = 0
			return
		}
	} else {
		if t.skip {
			for i in 0 ..< t.bw4 {
				t.above_seg_pred_ctx[t.mi_col - t.mi_col_start + i] = 0
			}
			for i in 0 ..< t.bh4 {
				t.left_seg_pred_ctx[(t.mi_row + i) & 63] = 0
			}
			read_segment_id(t)
			return
		}
	}
	if t.h.segmentation_temporal_update {
		ctx := int(t.left_seg_pred_ctx[t.mi_row & 63] + t.above_seg_pred_ctx[t.mi_col - t.mi_col_start])
		seg_id_predicted := u8(av1.read_symbol(&t.sym, t.cdfs.segment_id_predicted_cdf[ctx][:]))
		if seg_id_predicted != 0 {
			t.segment_id = predicted_segment_id
		} else {
			read_segment_id(t)
		}
		for i in 0 ..< t.bw4 {
			t.above_seg_pred_ctx[t.mi_col - t.mi_col_start + i] = seg_id_predicted
		}
		for i in 0 ..< t.bh4 {
			t.left_seg_pred_ctx[(t.mi_row + i) & 63] = seg_id_predicted
		}
	} else {
		read_segment_id(t)
	}
}

@(private)
get_segment_id :: proc(t: ^Tile_State) -> u8 {
	bw4 := min(t.h.mi_cols - t.mi_col, t.bw4)
	bh4 := min(t.h.mi_rows - t.mi_row, t.bh4)
	seg := u8(7)
	for y in 0 ..< bh4 {
		for x in 0 ..< bw4 {
			prev_seg := t.d.prev_segment_ids[(t.mi_row + y) * t.h.mi_cols + (t.mi_col + x)]
			seg = min(seg, prev_seg)
		}
	}
	return seg
}

// Section 5.11.22: Is inter syntax.
@(private)
read_is_inter :: proc(t: ^Tile_State, above_intra, left_intra: bool) {
	if t.skip_mode != 0 {
		t.is_inter = true
		return
	}
	if seg_feature_active(t, av1.SEG_LVL_REF_FRAME) {
		t.is_inter = t.h.feature_data[t.segment_id][av1.SEG_LVL_REF_FRAME] != av1.INTRA_FRAME
	} else if seg_feature_active(t, av1.SEG_LVL_GLOBALMV) {
		t.is_inter = true
	} else {
		ctx := 0
		if t.avail_u && t.avail_l {
			ctx = (left_intra && above_intra) ? 3 : int(left_intra || above_intra)
		} else if t.avail_u || t.avail_l {
			ctx = 2 * int(t.avail_u ? above_intra : left_intra)
		}
		t.is_inter = av1.read_symbol(&t.sym, t.cdfs.is_inter_cdf[ctx][:]) != 0
	}
}

// Section 5.11.23: Intra block mode info syntax.
@(private)
intra_block_mode_info :: #force_no_inline proc(t: ^Tile_State) {
	t.ref_frame = {av1.INTRA_FRAME, av1.NONE}
	ctx := av1.SIZE_GROUP[t.mi_size]
	t.y_mode = u8(av1.read_symbol(&t.sym, t.cdfs.y_mode_cdf[ctx][:]))
	intra_angle_info_y(t)
	if t.has_chroma {
		cfl_allowed := is_cfl_allowed(t)
		if cfl_allowed {
			t.uv_mode = u8(av1.read_symbol(&t.sym, t.cdfs.uv_mode_cfl_allowed_cdf[t.y_mode][:]))
		} else {
			t.uv_mode = u8(av1.read_symbol(&t.sym, t.cdfs.uv_mode_cfl_not_allowed_cdf[t.y_mode][:]))
		}
		if t.uv_mode == av1.UV_CFL_PRED {
			read_cfl_alphas(t)
		} else {
			intra_angle_info_uv(t)
		}
	}
	if t.mi_size >= av1.BLOCK_8X8 &&
	   av1.BLOCK_WIDTH[t.mi_size] <= 64 &&
	   av1.BLOCK_HEIGHT[t.mi_size] <= 64 &&
	   t.h.allow_screen_content_tools {
		palette_mode_info(t)
	}
	filter_intra_mode_info(t)
}

// Section 5.11.24: Inter block mode info syntax.
@(private)
inter_block_mode_info :: proc(
	t: ^Tile_State,
	above_ref_frame, left_ref_frame: [2]i8,
	above_intra, left_intra, above_single, left_single: bool,
) {
	t.palette_size_y = 0
	t.palette_size_uv = 0
	read_ref_frames(t, above_ref_frame, left_ref_frame, above_intra, left_intra, above_single, left_single)
	is_compound := t.ref_frame[1] > av1.INTRA_FRAME

	mf_ptr: ^[av1.TOTAL_REFS_PER_FRAME][]av1.Mv = t.h.use_ref_frame_mvs ? &t.mf_slices : nil
	av1.find_mv_stack(
		&t.mv_stack, &t.grid,
		t.mi_row, t.mi_col, t.mi_size,
		t.ref_frame, is_compound,
		t.h.allow_high_precision_mv, t.h.force_integer_mv,
		t.h.use_ref_frame_mvs, mf_ptr,
		t.h.gm_type, t.h.gm_params, t.sign_bias_u8,
	)

	if t.skip_mode != 0 {
		t.y_mode = av1.NEAREST_NEARESTMV
	} else if seg_feature_active(t, av1.SEG_LVL_SKIP) || seg_feature_active(t, av1.SEG_LVL_GLOBALMV) {
		t.y_mode = av1.GLOBALMV
	} else if is_compound {
		read_compound_mode(t)
	} else {
		new_mv := av1.read_symbol(&t.sym, t.cdfs.new_mv_cdf[t.mv_stack.new_mv_context][:])
		if new_mv == 0 {
			t.y_mode = av1.NEWMV
		} else {
			zero_mv := av1.read_symbol(&t.sym, t.cdfs.zero_mv_cdf[t.mv_stack.zero_mv_context][:])
			if zero_mv == 0 {
				t.y_mode = av1.GLOBALMV
			} else {
				ref_mv := av1.read_symbol(&t.sym, t.cdfs.ref_mv_cdf[t.mv_stack.ref_mv_context][:])
				t.y_mode = ref_mv == 0 ? av1.NEARESTMV : av1.NEARMV
			}
		}
	}

	t.ref_mv_idx = 0
	if t.y_mode == av1.NEWMV || t.y_mode == av1.NEW_NEWMV {
		for idx in 0 ..< 2 {
			if t.mv_stack.num_mv_found > idx + 1 {
				drl_mode := av1.read_symbol(&t.sym, t.cdfs.drl_mode_cdf[t.mv_stack.drl_ctx_stack[idx]][:])
				if drl_mode == 0 {
					t.ref_mv_idx = u8(idx)
					break
				}
				t.ref_mv_idx = u8(idx + 1)
			}
		}
	} else if av1.has_nearmv(int(t.y_mode)) {
		t.ref_mv_idx = 1
		for idx in 1 ..< 3 {
			if t.mv_stack.num_mv_found > idx + 1 {
				drl_mode := av1.read_symbol(&t.sym, t.cdfs.drl_mode_cdf[t.mv_stack.drl_ctx_stack[idx]][:])
				if drl_mode == 0 {
					t.ref_mv_idx = u8(idx)
					break
				}
				t.ref_mv_idx = u8(idx + 1)
			}
		}
	}

	assign_mv(t, is_compound)
	read_interintra_mode(t, is_compound)
	read_motion_mode(t, is_compound)
	read_compound_type(t, is_compound)

	if t.h.interpolation_filter == av1.SWITCHABLE {
		num_dirs := t.s.enable_dual_filter ? 2 : 1
		for dir in 0 ..< num_dirs {
			if needs_interp_filter(t) {
				ctx := get_interp_filter_ctx(t, dir, is_compound)
				t.interp_filter[dir] = u8(av1.read_symbol(&t.sym, t.cdfs.interp_filter_cdf[ctx][:]))
			} else {
				t.interp_filter[dir] = av1.EIGHTTAP_REGULAR
			}
		}
		if !t.s.enable_dual_filter {
			t.interp_filter[1] = t.interp_filter[0]
		}
	} else {
		t.interp_filter = {t.h.interpolation_filter, t.h.interpolation_filter}
	}
}

@(private)
get_interp_filter_ctx :: proc(t: ^Tile_State, dir: int, is_compound: bool) -> int {
	ctx := ((dir & 1) * 2 + int(is_compound)) * 4
	left_type := 3
	above_type := 3
	if t.avail_l {
		l_idx := t.mi_row * t.h.mi_cols + (t.mi_col - 1)
		rfs := t.grid.cells[l_idx].ref_frames
		if rfs[0] == t.ref_frame[0] || rfs[1] == t.ref_frame[0] {
			left_type = int(t.d.tile_cells[l_idx].interp_filters[dir])
		}
	}
	if t.avail_u {
		u_idx := (t.mi_row - 1) * t.h.mi_cols + t.mi_col
		rfs := t.grid.cells[u_idx].ref_frames
		if rfs[0] == t.ref_frame[0] || rfs[1] == t.ref_frame[0] {
			above_type = int(t.d.tile_cells[u_idx].interp_filters[dir])
		}
	}
	if left_type == above_type {
		ctx += left_type
	} else if left_type == 3 {
		ctx += above_type
	} else if above_type == 3 {
		ctx += left_type
	} else {
		ctx += 3
	}
	return ctx
}

// Section 5.11.24: needs_interp_filter.
@(private)
needs_interp_filter :: proc "contextless" (t: ^Tile_State) -> bool {
	large := min(av1.BLOCK_WIDTH[t.mi_size], av1.BLOCK_HEIGHT[t.mi_size]) >= 8
	if t.skip_mode != 0 || t.motion_mode == av1.LOCALWARP {
		return false
	}
	if large && t.y_mode == av1.GLOBALMV {
		return t.h.gm_type[t.ref_frame[0]] == av1.TRANSLATION
	}
	if large && t.y_mode == av1.GLOBAL_GLOBALMV {
		return t.h.gm_type[t.ref_frame[0]] == av1.TRANSLATION || t.h.gm_type[t.ref_frame[1]] == av1.TRANSLATION
	}
	return true
}

// Section 5.11.25: Reference frames syntax.
@(private)
read_ref_frames :: proc(
	t: ^Tile_State,
	above_ref_frame, left_ref_frame: [2]i8,
	above_intra, left_intra, above_single, left_single: bool,
) {
	if t.skip_mode != 0 {
		t.ref_frame[0] = i8(t.h.skip_mode_frame[0])
		t.ref_frame[1] = i8(t.h.skip_mode_frame[1])
		return
	}
	if seg_feature_active(t, av1.SEG_LVL_REF_FRAME) {
		t.ref_frame[0] = i8(t.h.feature_data[t.segment_id][av1.SEG_LVL_REF_FRAME])
		t.ref_frame[1] = av1.NONE
		return
	}
	if seg_feature_active(t, av1.SEG_LVL_SKIP) || seg_feature_active(t, av1.SEG_LVL_GLOBALMV) {
		t.ref_frame[0] = av1.LAST_FRAME
		t.ref_frame[1] = av1.NONE
		return
	}

	comp_mode := av1.SINGLE_REFERENCE
	if t.h.reference_select && min(av1.BLOCK_WIDTH[t.mi_size], av1.BLOCK_HEIGHT[t.mi_size]) >= 8 {
		ctx := get_comp_mode_ctx(t.avail_u, t.avail_l, above_ref_frame, left_ref_frame, above_intra, left_intra, above_single, left_single)
		comp_mode = av1.read_symbol(&t.sym, t.cdfs.comp_mode_cdf[ctx][:])
	}

	count_refs :: #force_inline proc "contextless" (
		avail_u, avail_l: bool,
		above_rf, left_rf: [2]i8,
		ft: i8,
	) -> int {
		c := 0
		if avail_u {
			if above_rf[0] == ft do c += 1
			if above_rf[1] == ft do c += 1
		}
		if avail_l {
			if left_rf[0] == ft do c += 1
			if left_rf[1] == ft do c += 1
		}
		return c
	}

	ref_count_ctx :: #force_inline proc "contextless" (c0, c1: int) -> int {
		return c0 < c1 ? 0 : (c0 == c1 ? 1 : 2)
	}

	c_last := count_refs(t.avail_u, t.avail_l, above_ref_frame, left_ref_frame, av1.LAST_FRAME)
	c_last2 := count_refs(t.avail_u, t.avail_l, above_ref_frame, left_ref_frame, av1.LAST2_FRAME)
	c_last3 := count_refs(t.avail_u, t.avail_l, above_ref_frame, left_ref_frame, av1.LAST3_FRAME)
	c_gold := count_refs(t.avail_u, t.avail_l, above_ref_frame, left_ref_frame, av1.GOLDEN_FRAME)
	c_bwd := count_refs(t.avail_u, t.avail_l, above_ref_frame, left_ref_frame, av1.BWDREF_FRAME)
	c_arf2 := count_refs(t.avail_u, t.avail_l, above_ref_frame, left_ref_frame, av1.ALTREF2_FRAME)
	c_arf := count_refs(t.avail_u, t.avail_l, above_ref_frame, left_ref_frame, av1.ALTREF_FRAME)

	ctx_last12_vs_last3gold := ref_count_ctx(c_last + c_last2, c_last3 + c_gold)
	ctx_last_vs_last2 := ref_count_ctx(c_last, c_last2)
	ctx_last3_vs_gold := ref_count_ctx(c_last3, c_gold)
	ctx_brfarf2_vs_arf := ref_count_ctx(c_bwd + c_arf2, c_arf)
	ctx_brf_vs_arf2 := ref_count_ctx(c_bwd, c_arf2)
	ctx_fwd_vs_bwd := ref_count_ctx(c_last + c_last2 + c_last3 + c_gold, c_bwd + c_arf2 + c_arf)
	ctx_last2_vs_last3gold := ref_count_ctx(c_last2, c_last3 + c_gold)

	if comp_mode == av1.COMPOUND_REFERENCE {
		ctx_crt := get_comp_ref_type_ctx(t.avail_u, t.avail_l, above_ref_frame, left_ref_frame, above_intra, left_intra, above_single, left_single)
		comp_ref_type := av1.read_symbol(&t.sym, t.cdfs.comp_ref_type_cdf[ctx_crt][:])
		if comp_ref_type == av1.UNIDIR_COMP_REFERENCE {
			uni_comp_ref := av1.read_symbol(&t.sym, t.cdfs.uni_comp_ref_cdf[ctx_fwd_vs_bwd][0][:])
			if uni_comp_ref != 0 {
				t.ref_frame[0] = av1.BWDREF_FRAME
				t.ref_frame[1] = av1.ALTREF_FRAME
			} else {
				uni_p1 := av1.read_symbol(&t.sym, t.cdfs.uni_comp_ref_cdf[ctx_last2_vs_last3gold][1][:])
				if uni_p1 != 0 {
					uni_p2 := av1.read_symbol(&t.sym, t.cdfs.uni_comp_ref_cdf[ctx_last3_vs_gold][2][:])
					t.ref_frame[0] = av1.LAST_FRAME
					t.ref_frame[1] = uni_p2 != 0 ? av1.GOLDEN_FRAME : av1.LAST3_FRAME
				} else {
					t.ref_frame[0] = av1.LAST_FRAME
					t.ref_frame[1] = av1.LAST2_FRAME
				}
			}
		} else {
			comp_ref := av1.read_symbol(&t.sym, t.cdfs.comp_ref_cdf[ctx_last12_vs_last3gold][0][:])
			if comp_ref == 0 {
				comp_ref_p1 := av1.read_symbol(&t.sym, t.cdfs.comp_ref_cdf[ctx_last_vs_last2][1][:])
				t.ref_frame[0] = comp_ref_p1 != 0 ? av1.LAST2_FRAME : av1.LAST_FRAME
			} else {
				comp_ref_p2 := av1.read_symbol(&t.sym, t.cdfs.comp_ref_cdf[ctx_last3_vs_gold][2][:])
				t.ref_frame[0] = comp_ref_p2 != 0 ? av1.GOLDEN_FRAME : av1.LAST3_FRAME
			}
			comp_bwdref := av1.read_symbol(&t.sym, t.cdfs.comp_bwd_ref_cdf[ctx_brfarf2_vs_arf][0][:])
			if comp_bwdref == 0 {
				comp_bwdref_p1 := av1.read_symbol(&t.sym, t.cdfs.comp_bwd_ref_cdf[ctx_brf_vs_arf2][1][:])
				t.ref_frame[1] = comp_bwdref_p1 != 0 ? av1.ALTREF2_FRAME : av1.BWDREF_FRAME
			} else {
				t.ref_frame[1] = av1.ALTREF_FRAME
			}
		}
	} else {
		p1 := av1.read_symbol(&t.sym, t.cdfs.single_ref_cdf[ctx_fwd_vs_bwd][0][:])
		if p1 != 0 {
			p2 := av1.read_symbol(&t.sym, t.cdfs.single_ref_cdf[ctx_brfarf2_vs_arf][1][:])
			if p2 == 0 {
				p6 := av1.read_symbol(&t.sym, t.cdfs.single_ref_cdf[ctx_brf_vs_arf2][5][:])
				t.ref_frame[0] = p6 != 0 ? av1.ALTREF2_FRAME : av1.BWDREF_FRAME
			} else {
				t.ref_frame[0] = av1.ALTREF_FRAME
			}
		} else {
			p3 := av1.read_symbol(&t.sym, t.cdfs.single_ref_cdf[ctx_last12_vs_last3gold][2][:])
			if p3 != 0 {
				p5 := av1.read_symbol(&t.sym, t.cdfs.single_ref_cdf[ctx_last3_vs_gold][4][:])
				t.ref_frame[0] = p5 != 0 ? av1.GOLDEN_FRAME : av1.LAST3_FRAME
			} else {
				p4 := av1.read_symbol(&t.sym, t.cdfs.single_ref_cdf[ctx_last_vs_last2][3][:])
				t.ref_frame[0] = p4 != 0 ? av1.LAST2_FRAME : av1.LAST_FRAME
			}
		}
		t.ref_frame[1] = av1.NONE
	}
}

@(private)
check_backward :: #force_inline proc "contextless" (rf: i8) -> int {
	return int(rf >= av1.BWDREF_FRAME && rf <= av1.ALTREF_FRAME)
}

@(private)
get_comp_mode_ctx :: proc "contextless" (
	avail_u, avail_l: bool,
	above_rf, left_rf: [2]i8,
	above_intra, left_intra, above_single, left_single: bool,
) -> int {
	if avail_u && avail_l {
		if above_single && left_single {
			return check_backward(above_rf[0]) ~ check_backward(left_rf[0])
		} else if above_single {
			return 2 + int(check_backward(above_rf[0]) != 0 || above_intra)
		} else if left_single {
			return 2 + int(check_backward(left_rf[0]) != 0 || left_intra)
		}
		return 4
	} else if avail_u {
		return above_single ? check_backward(above_rf[0]) : 3
	} else if avail_l {
		return left_single ? check_backward(left_rf[0]) : 3
	}
	return 1
}

@(private)
get_comp_ref_type_ctx :: proc "contextless" (
	avail_u, avail_l: bool,
	above_rf, left_rf: [2]i8,
	above_intra, left_intra, above_single, left_single: bool,
) -> int {
	is_samedir :: #force_inline proc "contextless" (r0, r1: i8) -> bool {
		return (r0 >= av1.BWDREF_FRAME) == (r1 >= av1.BWDREF_FRAME)
	}
	above0, above1 := above_rf[0], above_rf[1]
	left0, left1 := left_rf[0], left_rf[1]
	above_comp := avail_u && !above_intra && !above_single
	left_comp := avail_l && !left_intra && !left_single
	above_uni := above_comp && is_samedir(above0, above1)
	left_uni := left_comp && is_samedir(left0, left1)

	if avail_u && !above_intra && avail_l && !left_intra {
		samedir := int(is_samedir(above0, left0))
		if !above_comp && !left_comp {
			return 1 + 2 * samedir
		} else if !above_comp {
			return !left_uni ? 1 : 3 + samedir
		} else if !left_comp {
			return !above_uni ? 1 : 3 + samedir
		} else {
			if !above_uni && !left_uni do return 0
			if !above_uni || !left_uni do return 2
			return 3 + int((above0 == av1.BWDREF_FRAME) == (left0 == av1.BWDREF_FRAME))
		}
	} else if avail_u && avail_l {
		if above_comp do return 1 + 2 * int(above_uni)
		if left_comp do return 1 + 2 * int(left_uni)
		return 2
	} else if above_comp {
		return 4 * int(above_uni)
	} else if left_comp {
		return 4 * int(left_uni)
	}
	return 2
}

// Section 5.11.24: Compound mode syntax.
@(private)
read_compound_mode :: proc(t: ^Tile_State) {
	ctx := int(av1.COMPOUND_MODE_CTX_MAP[t.mv_stack.ref_mv_context >> 1][min(t.mv_stack.new_mv_context, av1.COMP_NEWMV_CTXS - 1)])
	comp_mode := av1.read_symbol(&t.sym, t.cdfs.compound_mode_cdf[ctx][:])
	t.y_mode = u8(av1.NEAREST_NEARESTMV + comp_mode)
}

// Section 5.11.26: Assign MV syntax.
@(private)
assign_mv :: proc(t: ^Tile_State, is_compound: bool) {
	num_lists := is_compound ? 2 : 1
	for i in 0 ..< num_lists {
		comp_mode := t.use_intrabc ? av1.NEWMV : av1.get_inter_comp_mode(int(t.y_mode), i)
		if t.use_intrabc {
			t.pred_mv[0] = t.mv_stack.ref_stack_mv[0][0]
			if t.pred_mv[0][0] == 0 && t.pred_mv[0][1] == 0 {
				t.pred_mv[0] = t.mv_stack.ref_stack_mv[1][0]
			}
			if t.pred_mv[0][0] == 0 && t.pred_mv[0][1] == 0 {
				sb_size := t.s.use_128x128_superblock ? av1.BLOCK_128X128 : av1.BLOCK_64X64
				sb_size4 := int(av1.NUM_4X4_BLOCKS_HIGH[sb_size])
				if t.mi_row - sb_size4 < t.mi_row_start {
					t.pred_mv[0][0] = 0
					t.pred_mv[0][1] = -i16((sb_size4 * av1.MI_SIZE + av1.INTRABC_DELAY_PIXELS) * 8)
				} else {
					t.pred_mv[0][0] = -i16(sb_size4 * av1.MI_SIZE * 8)
					t.pred_mv[0][1] = 0
				}
			}
		} else if comp_mode == av1.GLOBALMV {
			t.pred_mv[i] = t.mv_stack.global_mvs[i]
		} else {
			pos := comp_mode == av1.NEARESTMV ? 0 : int(t.ref_mv_idx)
			if comp_mode == av1.NEWMV && t.mv_stack.num_mv_found <= 1 {
				pos = 0
			}
			t.pred_mv[i] = t.mv_stack.ref_stack_mv[pos][i]
		}

		if comp_mode == av1.NEWMV {
			read_mv(t, i)
		} else {
			t.mv[i] = t.pred_mv[i]
		}
	}
}

// Section 5.11.27 & 5.11.28: Read MV and MV component syntax.
@(private)
read_mv :: proc(t: ^Tile_State, ref_list: int) {
	mv_ctx := t.use_intrabc ? av1.MV_INTRABC_CONTEXT : 0
	mv_joint := av1.read_symbol(&t.sym, t.cdfs.mv_joint_cdf[mv_ctx][:])
	diff_mv: av1.Mv
	if mv_joint == av1.MV_JOINT_HZVNZ || mv_joint == av1.MV_JOINT_HNZVNZ {
		diff_mv[0] = read_mv_component(t, mv_ctx, 0)
	}
	if mv_joint == av1.MV_JOINT_HNZVZ || mv_joint == av1.MV_JOINT_HNZVNZ {
		diff_mv[1] = read_mv_component(t, mv_ctx, 1)
	}
	t.mv[ref_list][0] = t.pred_mv[ref_list][0] + diff_mv[0]
	t.mv[ref_list][1] = t.pred_mv[ref_list][1] + diff_mv[1]
}

@(private)
read_mv_component :: proc(t: ^Tile_State, mv_ctx, comp: int) -> i16 {
	mv_sign := av1.read_symbol(&t.sym, t.cdfs.mv_sign_cdf[mv_ctx][comp][:])
	mv_class := av1.read_symbol(&t.sym, t.cdfs.mv_class_cdf[mv_ctx][comp][:])
	mag: int
	if mv_class == 0 {
		mv_class0_bit := av1.read_symbol(&t.sym, t.cdfs.mv_class0_bit_cdf[mv_ctx][comp][:])
		mv_class0_fr := t.h.force_integer_mv ? 3 : av1.read_symbol(&t.sym, t.cdfs.mv_class0_fr_cdf[mv_ctx][comp][mv_class0_bit][:])
		mv_class0_hp := t.h.allow_high_precision_mv ? av1.read_symbol(&t.sym, t.cdfs.mv_class0_hp_cdf[mv_ctx][comp][:]) : 1
		mag = ((mv_class0_bit << 3) | (mv_class0_fr << 1) | mv_class0_hp) + 1
	} else {
		bits := 0
		for i in 0 ..< mv_class {
			mv_bit := av1.read_symbol(&t.sym, t.cdfs.mv_bit_cdf[mv_ctx][comp][i][:])
			bits |= mv_bit << uint(i)
		}
		mag = av1.CLASS0_SIZE << uint(mv_class + 2)
		mv_fr := t.h.force_integer_mv ? 3 : av1.read_symbol(&t.sym, t.cdfs.mv_fr_cdf[mv_ctx][comp][:])
		mv_hp := t.h.allow_high_precision_mv ? av1.read_symbol(&t.sym, t.cdfs.mv_hp_cdf[mv_ctx][comp][:]) : 1
		mag += ((bits << 3) | (mv_fr << 1) | mv_hp) + 1
	}
	return i16(mv_sign != 0 ? -mag : mag)
}

// Section 5.11.28: Interintra mode syntax.
@(private)
read_interintra_mode :: proc(t: ^Tile_State, is_compound: bool) {
	t.interintra = false
	t.wedge_interintra = false
	if t.skip_mode == 0 &&
	   t.s.enable_interintra_compound &&
	   !is_compound &&
	   av1.BLOCK_WIDTH[t.mi_size] >= 8 &&
	   av1.BLOCK_WIDTH[t.mi_size] <= 32 &&
	   av1.BLOCK_HEIGHT[t.mi_size] >= 8 &&
	   av1.BLOCK_HEIGHT[t.mi_size] <= 32 {
		ctx := av1.SIZE_GROUP[t.mi_size] - 1
		interintra := av1.read_symbol(&t.sym, t.cdfs.inter_intra_cdf[ctx][:]) != 0
		t.interintra = interintra
		if interintra {
			t.interintra_mode = u8(av1.read_symbol(&t.sym, t.cdfs.inter_intra_mode_cdf[ctx][:]))
			t.ref_frame[1] = av1.INTRA_FRAME
			t.angle_delta_y = 0
			t.angle_delta_uv = 0
			t.use_filter_intra = false
			if av1.WEDGE_BITS[t.mi_size] > 0 {
				t.wedge_interintra = av1.read_symbol(&t.sym, t.cdfs.wedge_inter_intra_cdf[t.mi_size][:]) != 0
				if t.wedge_interintra {
					t.wedge_index = u8(av1.read_symbol(&t.sym, t.cdfs.wedge_index_cdf[t.mi_size][:]))
					t.wedge_sign = 0
				}
			}
		}
	}
}

// Section 5.11.27: Motion mode syntax.
@(private)
read_motion_mode :: proc(t: ^Tile_State, is_compound: bool) {
	if t.skip_mode != 0 {
		t.motion_mode = av1.SIMPLE
		return
	}
	if !t.h.is_motion_mode_switchable {
		t.motion_mode = av1.SIMPLE
		return
	}
	if min(av1.BLOCK_WIDTH[t.mi_size], av1.BLOCK_HEIGHT[t.mi_size]) < 8 {
		t.motion_mode = av1.SIMPLE
		return
	}
	if !t.h.force_integer_mv && (t.y_mode == av1.GLOBALMV || t.y_mode == av1.GLOBAL_GLOBALMV) {
		if t.h.gm_type[t.ref_frame[0]] > av1.TRANSLATION {
			t.motion_mode = av1.SIMPLE
			return
		}
	}
	if is_compound || t.ref_frame[1] == av1.INTRA_FRAME || !av1.has_overlappable_candidates(&t.grid, t.mi_row, t.mi_col, t.mi_size, t.avail_u, t.avail_l) {
		t.motion_mode = av1.SIMPLE
		return
	}

	t.num_samples = av1.find_warp_samples(
		&t.cand_list, &t.grid,
		t.mi_row, t.mi_col, t.mi_size,
		t.avail_u, t.avail_l,
		t.ref_frame[0], t.mv[0],
	)
	rf_slot := &t.d.ref_slots[t.h.ref_frame_idx[t.ref_frame[0] - av1.LAST_FRAME]]
	x_scale := ((rf_slot.upscaled_width << av1.REF_SCALE_SHIFT) + (t.h.frame_width / 2)) / t.h.frame_width
	y_scale := ((rf_slot.frame_height << av1.REF_SCALE_SHIFT) + (t.h.frame_height / 2)) / t.h.frame_height
	no_scale := 1 << av1.REF_SCALE_SHIFT
	is_scaled := x_scale != no_scale || y_scale != no_scale

	if t.h.force_integer_mv || t.num_samples == 0 || !t.h.allow_warped_motion || is_scaled {
		use_obmc := av1.read_symbol(&t.sym, t.cdfs.use_obmc_cdf[t.mi_size][:])
		t.motion_mode = use_obmc != 0 ? av1.OBMC : av1.SIMPLE
	} else {
		t.motion_mode = u8(av1.read_symbol(&t.sym, t.cdfs.motion_mode_cdf[t.mi_size][:]))
	}
}

// Section 5.11.29: Compound type syntax.
@(private)
read_compound_type :: proc(t: ^Tile_State, is_compound: bool) {
	t.comp_group_idx = 0
	t.compound_idx = 1
	if t.skip_mode != 0 {
		t.compound_type = av1.COMPOUND_AVERAGE
		return
	}
	if is_compound {
		n := av1.WEDGE_BITS[t.mi_size]
		if t.s.enable_masked_compound {
			ctx := get_comp_group_idx_ctx(t)
			t.comp_group_idx = u8(av1.read_symbol(&t.sym, t.cdfs.comp_group_idx_cdf[ctx][:]))
		}
		if t.comp_group_idx == 0 {
			if t.s.enable_jnt_comp {
				ctx := get_compound_idx_ctx(t)
				t.compound_idx = u8(av1.read_symbol(&t.sym, t.cdfs.compound_idx_cdf[ctx][:]))
				t.compound_type = t.compound_idx != 0 ? av1.COMPOUND_AVERAGE : av1.COMPOUND_DISTANCE
			} else {
				t.compound_type = av1.COMPOUND_AVERAGE
			}
		} else {
			if n == 0 {
				t.compound_type = av1.COMPOUND_DIFFWTD
			} else {
				t.compound_type = u8(av1.read_symbol(&t.sym, t.cdfs.compound_type_cdf[t.mi_size][:]))
			}
		}
		if t.compound_type == av1.COMPOUND_WEDGE {
			t.wedge_index = u8(av1.read_symbol(&t.sym, t.cdfs.wedge_index_cdf[t.mi_size][:]))
			t.wedge_sign = u8(av1.read_bool(&t.sym))
		} else if t.compound_type == av1.COMPOUND_DIFFWTD {
			t.mask_type = u8(av1.read_literal(&t.sym, 1))
		}
	} else if t.interintra {
		t.compound_type = t.wedge_interintra ? av1.COMPOUND_WEDGE : av1.COMPOUND_INTRA
	} else {
		t.compound_type = av1.COMPOUND_AVERAGE
	}
}

@(private)
get_comp_group_idx_ctx :: proc(t: ^Tile_State) -> int {
	ctx := 0
	if t.avail_u {
		u_idx := (t.mi_row - 1) * t.h.mi_cols + t.mi_col
		rfs := t.grid.cells[u_idx].ref_frames
		if rfs[1] > av1.INTRA_FRAME {
			ctx += int(t.d.tile_cells[u_idx].comp_group_idx)
		} else if rfs[0] == av1.ALTREF_FRAME {
			ctx += 3
		}
	}
	if t.avail_l {
		l_idx := t.mi_row * t.h.mi_cols + (t.mi_col - 1)
		rfs := t.grid.cells[l_idx].ref_frames
		if rfs[1] > av1.INTRA_FRAME {
			ctx += int(t.d.tile_cells[l_idx].comp_group_idx)
		} else if rfs[0] == av1.ALTREF_FRAME {
			ctx += 3
		}
	}
	return min(5, ctx)
}

@(private)
get_compound_idx_ctx :: proc(t: ^Tile_State) -> int {
	fwd := abs(av1.get_relative_dist(int(t.h.order_hints[t.ref_frame[0]]), int(t.h.order_hint), t.s.enable_order_hint, u8(t.s.order_hint_bits)))
	bck := abs(av1.get_relative_dist(int(t.h.order_hints[t.ref_frame[1]]), int(t.h.order_hint), t.s.enable_order_hint, u8(t.s.order_hint_bits)))
	ctx := fwd == bck ? 3 : 0
	if t.avail_u {
		u_idx := (t.mi_row - 1) * t.h.mi_cols + t.mi_col
		rfs := t.grid.cells[u_idx].ref_frames
		if rfs[1] > av1.INTRA_FRAME {
			ctx += int(t.d.tile_cells[u_idx].compound_idx)
		} else if rfs[0] == av1.ALTREF_FRAME {
			ctx += 1
		}
	}
	if t.avail_l {
		l_idx := t.mi_row * t.h.mi_cols + (t.mi_col - 1)
		rfs := t.grid.cells[l_idx].ref_frames
		if rfs[1] > av1.INTRA_FRAME {
			ctx += int(t.d.tile_cells[l_idx].compound_idx)
		} else if rfs[0] == av1.ALTREF_FRAME {
			ctx += 1
		}
	}
	return ctx
}

// Section 5.11.42 & 5.11.43: Intra angle info syntax.
@(private)
intra_angle_info_y :: proc(t: ^Tile_State) {
	t.angle_delta_y = 0
	if t.mi_size >= av1.BLOCK_8X8 && av1.is_directional_mode(int(t.y_mode)) {
		sym := av1.read_symbol(&t.sym, t.cdfs.angle_delta_cdf[t.y_mode - av1.V_PRED][:])
		t.angle_delta_y = i8(sym - av1.MAX_ANGLE_DELTA)
	}
}

@(private)
intra_angle_info_uv :: proc(t: ^Tile_State) {
	t.angle_delta_uv = 0
	if t.mi_size >= av1.BLOCK_8X8 && av1.is_directional_mode(int(t.uv_mode)) {
		sym := av1.read_symbol(&t.sym, t.cdfs.angle_delta_cdf[t.uv_mode - av1.V_PRED][:])
		t.angle_delta_uv = i8(sym - av1.MAX_ANGLE_DELTA)
	}
}

// Section 5.11.44: Read CFL alphas syntax.
@(private)
read_cfl_alphas :: proc(t: ^Tile_State) {
	cfl_alpha_signs := av1.read_symbol(&t.sym, t.cdfs.cfl_sign_cdf[:])
	sign_u := (cfl_alpha_signs + 1) / 3
	sign_v := (cfl_alpha_signs + 1) % 3
	if sign_u != 0 {
		ctx := (sign_u - 1) * 3 + sign_v
		cfl_alpha_u := 1 + av1.read_symbol(&t.sym, t.cdfs.cfl_alpha_cdf[ctx][:])
		t.cfl_alpha_u = i16(sign_u == 1 ? -cfl_alpha_u : cfl_alpha_u)
	} else {
		t.cfl_alpha_u = 0
	}
	if sign_v != 0 {
		ctx := (sign_v - 1) * 3 + sign_u
		cfl_alpha_v := 1 + av1.read_symbol(&t.sym, t.cdfs.cfl_alpha_cdf[ctx][:])
		t.cfl_alpha_v = i16(sign_v == 1 ? -cfl_alpha_v : cfl_alpha_v)
	} else {
		t.cfl_alpha_v = 0
	}
}

// Section 4.7: CeilLog2.
@(private)
ceil_log2 :: #force_inline proc "contextless" (x: int) -> int {
	if x < 2 do return 0
	i := 1
	p := 2
	for p < x {
		i += 1
		p <<= 1
	}
	return i
}

// Section 5.11.45 & 5.11.46: Palette mode info syntax & palette cache.
@(private)
palette_mode_info :: proc(t: ^Tile_State) {
	bsize_ctx := int(av1.MI_WIDTH_LOG2[t.mi_size]) + int(av1.MI_HEIGHT_LOG2[t.mi_size]) - 2
	bit_depth := int(t.s.color.bit_depth)
	if t.y_mode == av1.DC_PRED {
		ctx := 0
		if t.avail_u && t.d.tile_cells[(t.mi_row - 1) * t.h.mi_cols + t.mi_col].palette_sizes[0] > 0 do ctx += 1
		if t.avail_l && t.d.tile_cells[t.mi_row * t.h.mi_cols + (t.mi_col - 1)].palette_sizes[0] > 0 do ctx += 1
		has_palette_y := av1.read_symbol(&t.sym, t.cdfs.palette_y_mode_cdf[bsize_ctx][ctx][:]) != 0
		if has_palette_y {
			t.palette_size_y = u8(av1.read_symbol(&t.sym, t.cdfs.palette_y_size_cdf[bsize_ctx][:]) + 2)
			cache: [2 * av1.PALETTE_COLORS]u16
			cache_n := get_palette_cache(t, 0, &cache)
			idx := 0
			for i := 0; i < cache_n && idx < int(t.palette_size_y); i += 1 {
				if av1.read_bool(&t.sym) != 0 {
					t.palette_colors_y[idx] = cache[i]
					idx += 1
				}
			}
			if idx < int(t.palette_size_y) {
				t.palette_colors_y[idx] = u16(av1.read_literal(&t.sym, bit_depth))
				idx += 1
			}
			palette_bits := 0
			if idx < int(t.palette_size_y) {
				min_bits := bit_depth - 3
				palette_bits = min_bits + int(av1.read_literal(&t.sym, 2))
			}
			for idx < int(t.palette_size_y) {
				delta := int(av1.read_literal(&t.sym, palette_bits)) + 1
				next_color := min(int(t.palette_colors_y[idx - 1]) + delta, (1 << uint(bit_depth)) - 1)
				range_left := (1 << uint(bit_depth)) - next_color - 1
				t.palette_colors_y[idx] = u16(next_color)
				palette_bits = min(palette_bits, ceil_log2(range_left))
				idx += 1
			}
			// Sort palette_colors_y ascending (5.11.45).
			for i in 0 ..< int(t.palette_size_y) - 1 {
				for j in i + 1 ..< int(t.palette_size_y) {
					if t.palette_colors_y[i] > t.palette_colors_y[j] {
						t.palette_colors_y[i], t.palette_colors_y[j] = t.palette_colors_y[j], t.palette_colors_y[i]
					}
				}
			}
		}
	}

	if t.has_chroma && t.uv_mode == av1.DC_PRED {
		ctx := t.palette_size_y > 0 ? 1 : 0
		has_palette_uv := av1.read_symbol(&t.sym, t.cdfs.palette_uv_mode_cdf[ctx][:]) != 0
		if has_palette_uv {
			t.palette_size_uv = u8(av1.read_symbol(&t.sym, t.cdfs.palette_uv_size_cdf[bsize_ctx][:]) + 2)
			cache: [2 * av1.PALETTE_COLORS]u16
			cache_n := get_palette_cache(t, 1, &cache)
			idx := 0
			for i := 0; i < cache_n && idx < int(t.palette_size_uv); i += 1 {
				if av1.read_bool(&t.sym) != 0 {
					t.palette_colors_u[idx] = cache[i]
					idx += 1
				}
			}
			if idx < int(t.palette_size_uv) {
				t.palette_colors_u[idx] = u16(av1.read_literal(&t.sym, bit_depth))
				idx += 1
			}
			palette_bits := 0
			if idx < int(t.palette_size_uv) {
				min_bits := bit_depth - 3
				palette_bits = min_bits + int(av1.read_literal(&t.sym, 2))
			}
			for idx < int(t.palette_size_uv) {
				delta := int(av1.read_literal(&t.sym, palette_bits))
				next_color := min(int(t.palette_colors_u[idx - 1]) + delta, (1 << uint(bit_depth)) - 1)
				range_left := (1 << uint(bit_depth)) - next_color
				t.palette_colors_u[idx] = u16(next_color)
				palette_bits = min(palette_bits, ceil_log2(range_left))
				idx += 1
			}
			for i in 0 ..< int(t.palette_size_uv) - 1 {
				for j in i + 1 ..< int(t.palette_size_uv) {
					if t.palette_colors_u[i] > t.palette_colors_u[j] {
						t.palette_colors_u[i], t.palette_colors_u[j] = t.palette_colors_u[j], t.palette_colors_u[i]
					}
				}
			}
			if av1.read_bool(&t.sym) != 0 {
				min_bits := bit_depth - 4
				max_val := 1 << uint(bit_depth)
				palette_bits = min_bits + int(av1.read_literal(&t.sym, 2))
				t.palette_colors_v[0] = u16(av1.read_literal(&t.sym, bit_depth))
				for idx2 in 1 ..< int(t.palette_size_uv) {
					palette_delta_v := int(av1.read_literal(&t.sym, palette_bits))
					if palette_delta_v != 0 {
						if av1.read_bool(&t.sym) != 0 {
							palette_delta_v = -palette_delta_v
						}
					}
					val := int(t.palette_colors_v[idx2 - 1]) + palette_delta_v
					if val < 0 {
						val += max_val
					} else if val >= max_val {
						val -= max_val
					}
					t.palette_colors_v[idx2] = u16(clamp(val, 0, (1 << uint(bit_depth)) - 1))
				}
			} else {
				for idx2 in 0 ..< int(t.palette_size_uv) {
					t.palette_colors_v[idx2] = u16(av1.read_literal(&t.sym, bit_depth))
				}
			}
		}
	}
}

@(private)
get_palette_cache :: proc(t: ^Tile_State, plane: int, cache: ^[2 * av1.PALETTE_COLORS]u16) -> int {
	above_n := 0
	if ((t.mi_row * av1.MI_SIZE) & 63) != 0 && t.avail_u {
		above_n = int(t.d.tile_cells[(t.mi_row - 1) * t.h.mi_cols + t.mi_col].palette_sizes[plane])
	}
	left_n := 0
	if t.avail_l {
		left_n = int(t.d.tile_cells[t.mi_row * t.h.mi_cols + (t.mi_col - 1)].palette_sizes[plane])
	}
	above_idx := 0
	left_idx := 0
	n := 0
	for above_idx < above_n && left_idx < left_n {
		above_c := t.d.tile_cells[(t.mi_row - 1) * t.h.mi_cols + t.mi_col].palette_colors[plane][above_idx]
		left_c := t.d.tile_cells[t.mi_row * t.h.mi_cols + (t.mi_col - 1)].palette_colors[plane][left_idx]
		if left_c < above_c {
			if n == 0 || left_c != cache[n - 1] {
				cache[n] = left_c
				n += 1
			}
			left_idx += 1
		} else {
			if n == 0 || above_c != cache[n - 1] {
				cache[n] = above_c
				n += 1
			}
			above_idx += 1
			if left_c == above_c {
				left_idx += 1
			}
		}
	}
	for above_idx < above_n {
		val := t.d.tile_cells[(t.mi_row - 1) * t.h.mi_cols + t.mi_col].palette_colors[plane][above_idx]
		above_idx += 1
		if n == 0 || val != cache[n - 1] {
			cache[n] = val
			n += 1
		}
	}
	for left_idx < left_n {
		val := t.d.tile_cells[t.mi_row * t.h.mi_cols + (t.mi_col - 1)].palette_colors[plane][left_idx]
		left_idx += 1
		if n == 0 || val != cache[n - 1] {
			cache[n] = val
			n += 1
		}
	}
	return n
}

// Section 5.11.24: Filter intra mode info syntax.
@(private)
filter_intra_mode_info :: proc(t: ^Tile_State) {
	t.use_filter_intra = false
	if t.s.enable_filter_intra &&
	   t.y_mode == av1.DC_PRED &&
	   t.palette_size_y == 0 &&
	   max(av1.BLOCK_WIDTH[t.mi_size], av1.BLOCK_HEIGHT[t.mi_size]) <= 32 {
		t.use_filter_intra = av1.read_symbol(&t.sym, t.cdfs.filter_intra_cdf[t.mi_size][:]) != 0
		if t.use_filter_intra {
			t.filter_intra_mode = u8(av1.read_symbol(&t.sym, t.cdfs.filter_intra_mode_cdf[:]))
		}
	}
}

// Section 5.11.49 & 5.11.50: Palette tokens syntax & color context.
@(private)
palette_tokens :: #force_no_inline proc(t: ^Tile_State) {
	block_h := int(av1.BLOCK_HEIGHT[t.mi_size])
	block_w := int(av1.BLOCK_WIDTH[t.mi_size])
	onscreen_h := min(block_h, (t.h.mi_rows - t.mi_row) * av1.MI_SIZE)
	onscreen_w := min(block_w, (t.h.mi_cols - t.mi_col) * av1.MI_SIZE)

	if t.palette_size_y > 0 {
		t.color_map_y[0] = u8(av1.read_ns_bool(&t.sym, u32(t.palette_size_y)))
		color_order: [av1.PALETTE_COLORS]u8
		for i := 1; i < onscreen_h + onscreen_w - 1; i += 1 {
			for j := min(i, onscreen_w - 1); j >= max(0, i - onscreen_h + 1); j -= 1 {
				r := i - j
				c := j
				ctx := get_palette_color_context(&t.color_map_y, r, c, int(t.palette_size_y), &color_order)
				idx: int
				switch t.palette_size_y {
				case 2: idx = av1.read_symbol(&t.sym, t.cdfs.palette_size_2_y_color_cdf[ctx][:])
				case 3: idx = av1.read_symbol(&t.sym, t.cdfs.palette_size_3_y_color_cdf[ctx][:])
				case 4: idx = av1.read_symbol(&t.sym, t.cdfs.palette_size_4_y_color_cdf[ctx][:])
				case 5: idx = av1.read_symbol(&t.sym, t.cdfs.palette_size_5_y_color_cdf[ctx][:])
				case 6: idx = av1.read_symbol(&t.sym, t.cdfs.palette_size_6_y_color_cdf[ctx][:])
				case 7: idx = av1.read_symbol(&t.sym, t.cdfs.palette_size_7_y_color_cdf[ctx][:])
				case:   idx = av1.read_symbol(&t.sym, t.cdfs.palette_size_8_y_color_cdf[ctx][:])
				}
				t.color_map_y[r * 64 + c] = color_order[idx]
			}
		}
		for i in 0 ..< onscreen_h {
			for j in onscreen_w ..< block_w {
				t.color_map_y[i * 64 + j] = t.color_map_y[i * 64 + (onscreen_w - 1)]
			}
		}
		for i in onscreen_h ..< block_h {
			for j in 0 ..< block_w {
				t.color_map_y[i * 64 + j] = t.color_map_y[(onscreen_h - 1) * 64 + j]
			}
		}
	}

	if t.palette_size_uv > 0 {
		t.color_map_uv[0] = u8(av1.read_ns_bool(&t.sym, u32(t.palette_size_uv)))
		sub_x := uint(t.s.color.subsampling_x)
		sub_y := uint(t.s.color.subsampling_y)
		bh := block_h >> sub_y
		bw := block_w >> sub_x
		oh := onscreen_h >> sub_y
		ow := onscreen_w >> sub_x
		if bw < 4 {
			bw += 2
			ow += 2
		}
		if bh < 4 {
			bh += 2
			oh += 2
		}
		color_order: [av1.PALETTE_COLORS]u8
		for i := 1; i < oh + ow - 1; i += 1 {
			for j := min(i, ow - 1); j >= max(0, i - oh + 1); j -= 1 {
				r := i - j
				c := j
				ctx := get_palette_color_context(&t.color_map_uv, r, c, int(t.palette_size_uv), &color_order)
				idx: int
				switch t.palette_size_uv {
				case 2: idx = av1.read_symbol(&t.sym, t.cdfs.palette_size_2_uv_color_cdf[ctx][:])
				case 3: idx = av1.read_symbol(&t.sym, t.cdfs.palette_size_3_uv_color_cdf[ctx][:])
				case 4: idx = av1.read_symbol(&t.sym, t.cdfs.palette_size_4_uv_color_cdf[ctx][:])
				case 5: idx = av1.read_symbol(&t.sym, t.cdfs.palette_size_5_uv_color_cdf[ctx][:])
				case 6: idx = av1.read_symbol(&t.sym, t.cdfs.palette_size_6_uv_color_cdf[ctx][:])
				case 7: idx = av1.read_symbol(&t.sym, t.cdfs.palette_size_7_uv_color_cdf[ctx][:])
				case:   idx = av1.read_symbol(&t.sym, t.cdfs.palette_size_8_uv_color_cdf[ctx][:])
				}
				t.color_map_uv[r * 64 + c] = color_order[idx]
			}
		}
		for i in 0 ..< oh {
			for j in ow ..< bw {
				t.color_map_uv[i * 64 + j] = t.color_map_uv[i * 64 + (ow - 1)]
			}
		}
		for i in oh ..< bh {
			for j in 0 ..< bw {
				t.color_map_uv[i * 64 + j] = t.color_map_uv[(oh - 1) * 64 + j]
			}
		}
	}
}

@(private)
Palette_Color_Set :: bit_set[0 ..< av1.PALETTE_COLORS; u8]

@(private)
get_palette_color_context :: #force_inline proc "contextless" (
	color_map: ^[64 * 64]u8,
	r, c, n: int,
	color_order: ^[av1.PALETTE_COLORS]u8,
) -> int #no_bounds_check {
	used: Palette_Color_Set
	o_idx := 0
	ctx: int
	idx := r * 64 + c
	if c == 0 {
		ctx = 0
		t := color_map[idx - 64]
		color_order[0] = t
		used = {int(t)}
		o_idx = 1
	} else if r == 0 {
		ctx = 0
		l := color_map[idx - 1]
		color_order[0] = l
		used = {int(l)}
		o_idx = 1
	} else {
		l := color_map[idx - 1]
		t := color_map[idx - 64]
		tl := color_map[idx - 65]
		if t == l {
			if t == tl {
				ctx = 4
				color_order[0] = t
				used = {int(t)}
				o_idx = 1
			} else {
				ctx = 3
				color_order[0] = t
				color_order[1] = tl
				used = {int(t), int(tl)}
				o_idx = 2
			}
		} else if t == tl || l == tl {
			ctx = 2
			other := t == tl ? l : t
			color_order[0] = tl
			color_order[1] = other
			used = {int(tl), int(other)}
			o_idx = 2
		} else {
			ctx = 1
			color_order[0] = min(t, l)
			color_order[1] = max(t, l)
			color_order[2] = tl
			used = {int(t), int(l), int(tl)}
			o_idx = 3
		}
	}
	rem := (~transmute(u8)used) & ((u8(1) << uint(n)) - 1)
	for rem != 0 {
		bit := intrinsics.count_trailing_zeros(rem)
		color_order[o_idx] = bit
		o_idx += 1
		rem &= rem - 1
	}
	return ctx
}

@(private)
fb_stride :: #force_inline proc "contextless" (fb: ^Frame_Buffer, plane: int) -> int {
	return plane == 0 ? fb.y_stride : fb.uv_stride
}

@(private)
fb_plane_u8 :: #force_inline proc "contextless" (fb: ^Frame_Buffer, plane: int) -> []u8 {
	switch plane {
	case 0: return fb.y[:]
	case 1: return fb.u[:]
	case:   return fb.v[:]
	}
}

@(private)
fb_plane_u16 :: #force_inline proc "contextless" (fb: ^Frame_Buffer, plane: int) -> []u16 {
	switch plane {
	case 0: return fb.y16[:]
	case 1: return fb.u16[:]
	case:   return fb.v16[:]
	}
}

@(private)
ref_plane_u8 :: #force_inline proc "contextless" (fb: ^Frame_Buffer, plane: int, upscaled_w, frame_w, frame_h: int, sub_x, sub_y: uint) -> av1.Ref_Plane(u8) {
	pw := (upscaled_w + int(sub_x)) >> sub_x
	ph := (frame_h + int(sub_y)) >> sub_y
	return {
		data           = fb_plane_u8(fb, plane),
		stride         = fb_stride(fb, plane),
		last_x         = pw - 1,
		last_y         = ph - 1,
		ref_width      = frame_w,
		ref_height     = frame_h,
		upscaled_width = upscaled_w,
	}
}

@(private)
ref_plane_u16 :: #force_inline proc "contextless" (fb: ^Frame_Buffer, plane: int, upscaled_w, frame_w, frame_h: int, sub_x, sub_y: uint) -> av1.Ref_Plane(u16) {
	pw := (upscaled_w + int(sub_x)) >> sub_x
	ph := (frame_h + int(sub_y)) >> sub_y
	return {
		data           = fb_plane_u16(fb, plane),
		stride         = fb_stride(fb, plane),
		last_x         = pw - 1,
		last_y         = ph - 1,
		ref_width      = frame_w,
		ref_height     = frame_h,
		upscaled_width = upscaled_w,
	}
}

@(private)
clamp_mv_for_pred :: #force_inline proc "contextless" (
	mv: av1.Mv,
	mi_row, mi_col, mi_rows, mi_cols, bw4, bh4: int,
	use_intrabc: bool,
) -> av1.Mv {
	if use_intrabc do return mv
	return {
		av1.clamp_mv_row(i32(mv[0]), mi_row, mi_rows, bh4, av1.MV_BORDER + bh4 * 32),
		av1.clamp_mv_col(i32(mv[1]), mi_col, mi_cols, bw4, av1.MV_BORDER + bw4 * 32),
	}
}

// Section 7.11.2.8: Intra edge filter type from neighbor smooth modes.
@(private)
get_filter_type :: #force_inline proc "contextless" (t: ^Tile_State, plane: int) -> int #no_bounds_check {
	above_smooth := false
	left_smooth := false
	avail_u := plane == 0 ? t.avail_u : t.avail_u_chroma
	avail_l := plane == 0 ? t.avail_l : t.avail_l_chroma
	if avail_u {
		r := (t.mi_row & ~(plane > 0 ? int(t.s.color.subsampling_y) : 0)) - 1
		c := t.mi_col | (plane > 0 ? int(t.s.color.subsampling_x) : 0)
		idx := r * t.h.mi_cols + c
		if t.grid.cells[idx].ref_frames[0] == av1.INTRA_FRAME {
			m := plane == 0 ? t.grid.cells[idx].y_mode : t.d.tile_cells[idx].uv_mode
			above_smooth = av1.is_smooth_mode(m)
		}
	}
	if avail_l {
		r := t.mi_row | (plane > 0 ? int(t.s.color.subsampling_y) : 0)
		c := (t.mi_col & ~(plane > 0 ? int(t.s.color.subsampling_x) : 0)) - 1
		idx := r * t.h.mi_cols + c
		if t.grid.cells[idx].ref_frames[0] == av1.INTRA_FRAME {
			m := plane == 0 ? t.grid.cells[idx].y_mode : t.d.tile_cells[idx].uv_mode
			left_smooth = av1.is_smooth_mode(m)
		}
	}
	return int(above_smooth || left_smooth)
}

// Section 5.11.33: Compute prediction syntax & inter prediction pipeline.
@(private)
compute_prediction :: proc(t: ^Tile_State) #no_bounds_check {
	sb_mask := t.s.use_128x128_superblock ? 31 : 15
	sub_block_mi_row := t.mi_row & sb_mask
	sub_block_mi_col := t.mi_col & sb_mask
	num_planes := 1 + (t.has_chroma ? 2 : 0)
	bit_depth := int(t.s.color.bit_depth)

	for plane in 0 ..< num_planes {
		sub_x := plane > 0 ? uint(t.s.color.subsampling_x) : 0
		sub_y := plane > 0 ? uint(t.s.color.subsampling_y) : 0
		plane_sz := av1.get_plane_residual_size(t.mi_size, plane, sub_x, sub_y)
		num4x4_w := int(av1.NUM_4X4_BLOCKS_WIDE[plane_sz])
		num4x4_h := int(av1.NUM_4X4_BLOCKS_HIGH[plane_sz])
		base_x := (t.mi_col >> sub_x) * av1.MI_SIZE
		base_y := (t.mi_row >> sub_y) * av1.MI_SIZE
		cand_row := (t.mi_row >> sub_y) << sub_y
		cand_col := (t.mi_col >> sub_x) << sub_x

		is_inter_intra := t.is_inter && t.ref_frame[1] == av1.INTRA_FRAME
		if is_inter_intra {
			mode := u8(av1.SMOOTH_PRED)
			switch t.interintra_mode {
			case av1.II_DC_PRED:     mode = av1.DC_PRED
			case av1.II_V_PRED:      mode = av1.V_PRED
			case av1.II_H_PRED:      mode = av1.H_PRED
			case av1.II_SMOOTH_PRED: mode = av1.SMOOTH_PRED
			}
			cur_fb := t.d.cur
			stride := fb_stride(cur_fb, plane)
			avail_l := plane == 0 ? t.avail_l : t.avail_l_chroma
			avail_u := plane == 0 ? t.avail_u : t.avail_u_chroma
			bd_tr := ((sub_block_mi_col >> sub_x) + num4x4_w) in t.block_decoded[plane][sub_block_mi_row >> sub_y]
			bd_bl := ((sub_block_mi_col >> sub_x) - 1) in t.block_decoded[plane][(sub_block_mi_row >> sub_y) + num4x4_h + 1]
			max_x := ((t.h.mi_cols * av1.MI_SIZE) >> sub_x) - 1
			max_y := ((t.h.mi_rows * av1.MI_SIZE) >> sub_y) - 1
			tx_sz := av1.find_tx_size(num4x4_w * 4, num4x4_h * 4)
			edges := av1.Intra_Edges{
				have_left                = avail_l,
				have_above               = avail_u,
				have_above_right         = bd_tr,
				have_below_left          = bd_bl,
				max_x                    = max_x,
				max_y                    = max_y,
				enable_intra_edge_filter = t.s.enable_intra_edge_filter,
			}
			if bit_depth > 8 {
				av1.predict_intra(fb_plane_u16(cur_fb, plane), stride, base_x, base_y, tx_sz, int(mode), edges, bit_depth)
			} else {
				av1.predict_intra(fb_plane_u8(cur_fb, plane), stride, base_x, base_y, tx_sz, int(mode), edges, bit_depth)
			}
		}

		if t.is_inter {
			pred_w := int(av1.BLOCK_WIDTH[t.mi_size]) >> sub_x
			pred_h := int(av1.BLOCK_HEIGHT[t.mi_size]) >> sub_y
			some_use_intra := t.ref_frame[0] == av1.INTRA_FRAME
			if !some_use_intra && (cand_row != t.mi_row || cand_col != t.mi_col) {
				mi_refs := t.d.mi_cells.ref_frames
				for r in 0 ..< (num4x4_h << sub_y) {
					for c in 0 ..< (num4x4_w << sub_x) {
						if mi_refs[(cand_row + r) * t.h.mi_cols + (cand_col + c)][0] == av1.INTRA_FRAME {
							some_use_intra = true
						}
					}
				}
			}
			if some_use_intra {
				pred_w = num4x4_w * 4
				pred_h = num4x4_h * 4
				cand_row = t.mi_row
				cand_col = t.mi_col
			}
			r := 0
			for y := 0; y < num4x4_h * 4; y += pred_h {
				c := 0
				for x := 0; x < num4x4_w * 4; x += pred_w {
					predict_inter(t, plane, base_x + x, base_y + y, pred_w, pred_h, cand_row + r, cand_col + c)
					c += 1
				}
				r += 1
			}
		}
	}
}

// Section 7.11.3: Inter prediction process.
@(private)
predict_inter :: proc(
	t: ^Tile_State,
	plane, x, y, w, h, cand_row, cand_col: int,
) #no_bounds_check {
	d := t.d
	fh := t.h
	s := t.s
	cur_fb := d.cur
	cur_stride := fb_stride(cur_fb, plane)
	bit_depth := int(s.color.bit_depth)
	sub_x := plane > 0 ? uint(s.color.subsampling_x) : 0
	sub_y := plane > 0 ? uint(s.color.subsampling_y) : 0

	cand_idx := cand_row * fh.mi_cols + cand_col
	cand_refs := d.mi_cells.ref_frames[cand_idx]
	is_compound := cand_refs[1] > av1.INTRA_FRAME

	if plane == 0 {
		t.local_warp_valid = false
		if t.motion_mode == av1.LOCALWARP {
			valid_ls, ls_params := av1.estimate_local_warp(t.mi_size, t.mi_row, t.mi_col, t.mv[0], t.cand_list[:t.num_samples])
			if valid_ls {
				valid_sh, _, _, _, _ := av1.setup_shear(ls_params)
				if valid_sh {
					t.local_warp_valid = true
					t.local_warp_params = ls_params
				}
			}
		}
	}

	is_inter_intra := t.is_inter && t.ref_frame[1] == av1.INTRA_FRAME
	if !is_compound && !is_inter_intra {
		ref_frame := cand_refs[0]
		cand_y_mode := d.mi_cells.y_mode[cand_idx]
		use_warp := false
		warp_p: [6]i32
		if w >= 8 && h >= 8 && !fh.force_integer_mv {
			if t.motion_mode == av1.LOCALWARP && t.local_warp_valid {
				use_warp = true
				warp_p = t.local_warp_params
			} else if (cand_y_mode == av1.GLOBALMV || cand_y_mode == av1.GLOBAL_GLOBALMV) &&
			          fh.gm_type[ref_frame] > av1.TRANSLATION &&
			          !is_scaled_prev(fh, ref_frame, &d.ref_slots) {
				valid_sh, _, _, _, _ := av1.setup_shear(fh.gm_params[ref_frame])
				if valid_sh {
					use_warp = true
					warp_p = fh.gm_params[ref_frame]
				}
			}
		}

		ref_fb: ^Frame_Buffer
		ref_upscaled_w, ref_frame_w, ref_frame_h: int
		if t.use_intrabc {
			ref_fb = cur_fb
			ref_upscaled_w = fh.upscaled_width
			ref_frame_w = fh.frame_width
			ref_frame_h = fh.frame_height
		} else {
			slot := &d.ref_slots[fh.ref_frame_idx[ref_frame - av1.LAST_FRAME]]
			ref_fb = slot.buf
			ref_upscaled_w = slot.upscaled_width
			ref_frame_w = slot.frame_width
			ref_frame_h = slot.frame_height
		}

		if !use_warp {
			cand_mv0 := d.mi_cells.mvs[cand_idx][0]
			cand_filters := d.tile_cells.interp_filters[cand_idx]
			mv := clamp_mv_for_pred(
				cand_mv0,
				t.mi_row, t.mi_col,
				fh.mi_rows, fh.mi_cols,
				t.bw4, t.bh4,
				t.use_intrabc,
			)
			start_x, start_y, step_x, step_y := av1.scale_mv_for_block(
				x, y, mv, sub_x, sub_y,
				fh.frame_width, fh.frame_height,
				ref_upscaled_w, ref_frame_h,
			)
			if bit_depth > 8 {
				rp := ref_plane_u16(ref_fb, plane, ref_upscaled_w, ref_frame_w, ref_frame_h, sub_x, sub_y)
				av1.predict_inter_single(
					fb_plane_u16(cur_fb, plane), cur_stride,
					x, y, w, h, &rp,
					start_x, start_y, step_x, step_y,
					int(cand_filters[0]), int(cand_filters[1]),
					bit_depth,
				)
			} else {
				rp := ref_plane_u8(ref_fb, plane, ref_upscaled_w, ref_frame_w, ref_frame_h, sub_x, sub_y)
				av1.predict_inter_single(
					fb_plane_u8(cur_fb, plane), cur_stride,
					x, y, w, h, &rp,
					start_x, start_y, step_x, step_y,
					int(cand_filters[0]), int(cand_filters[1]),
					bit_depth,
				)
			}
			if t.motion_mode == av1.OBMC {
				predict_obmc(t, plane, x, y, w, h)
			}
			return
		}
	}

	// Pack `(cand_idx << 2) | plane` into the 2nd parameter so all 6 parameters fit in
	// SysV AMD64 integer argument registers (`%rdi, %rsi, %rdx, %rcx, %r8, %r9`) without
	// stack argument marshalling or 1-byte store -> 2-byte `movzwl` STLF stalls in `decode_block`.
	predict_inter_compound_or_warp(t, (cand_idx << 2) | plane, x, y, w, h)
}

@(private)
predict_inter_compound_or_warp :: #force_no_inline proc(
	t: ^Tile_State,
	plane_and_cand, x, y, w, h: int,
) #no_bounds_check {
	plane := plane_and_cand & 3
	cand_idx := plane_and_cand >> 2
	d := t.d
	fh := t.h
	s := t.s
	cur_fb := d.cur
	cur_stride := fb_stride(cur_fb, plane)
	bit_depth := int(s.color.bit_depth)
	sub_x := plane > 0 ? uint(s.color.subsampling_x) : 0
	sub_y := plane > 0 ? uint(s.color.subsampling_y) : 0

	cand_refs := d.mi_cells.ref_frames[cand_idx]
	cand_mvs := d.mi_cells.mvs[cand_idx]
	cand_y_mode := d.mi_cells.y_mode[cand_idx]
	cand_filters := d.tile_cells.interp_filters[cand_idx]
	is_compound := cand_refs[1] > av1.INTRA_FRAME
	is_inter_intra := t.is_inter && t.ref_frame[1] == av1.INTRA_FRAME

	preds := &t.preds
	num_refs := is_compound ? 2 : 1
	for ref_list in 0 ..< num_refs {
		ref_frame := cand_refs[ref_list]
		use_warp := false
		warp_p: [6]i32
		if w >= 8 && h >= 8 && !fh.force_integer_mv {
			if t.motion_mode == av1.LOCALWARP && t.local_warp_valid {
				use_warp = true
				warp_p = t.local_warp_params
			} else if (cand_y_mode == av1.GLOBALMV || cand_y_mode == av1.GLOBAL_GLOBALMV) &&
			          fh.gm_type[ref_frame] > av1.TRANSLATION &&
			          !is_scaled_prev(fh, ref_frame, &d.ref_slots) {
				valid_sh, _, _, _, _ := av1.setup_shear(fh.gm_params[ref_frame])
				if valid_sh {
					use_warp = true
					warp_p = fh.gm_params[ref_frame]
				}
			}
		}

		ref_fb: ^Frame_Buffer
		ref_upscaled_w, ref_frame_w, ref_frame_h: int
		if t.use_intrabc {
			ref_fb = cur_fb
			ref_upscaled_w = fh.upscaled_width
			ref_frame_w = fh.frame_width
			ref_frame_h = fh.frame_height
		} else {
			slot := &d.ref_slots[fh.ref_frame_idx[ref_frame - av1.LAST_FRAME]]
			ref_fb = slot.buf
			ref_upscaled_w = slot.upscaled_width
			ref_frame_w = slot.frame_width
			ref_frame_h = slot.frame_height
		}

		mv := clamp_mv_for_pred(
			cand_mvs[ref_list],
			t.mi_row, t.mi_col,
			fh.mi_rows, fh.mi_cols,
			t.bw4, t.bh4,
			t.use_intrabc,
		)

		if use_warp {
			if bit_depth > 8 {
				rp := ref_plane_u16(ref_fb, plane, ref_upscaled_w, ref_frame_w, ref_frame_h, sub_x, sub_y)
				av1.warp_plane_pred(preds[ref_list][:], w, &rp, x, y, w, h, sub_x, sub_y, warp_p, is_compound, bit_depth)
			} else {
				rp := ref_plane_u8(ref_fb, plane, ref_upscaled_w, ref_frame_w, ref_frame_h, sub_x, sub_y)
				av1.warp_plane_pred(preds[ref_list][:], w, &rp, x, y, w, h, sub_x, sub_y, warp_p, is_compound, bit_depth)
			}
		} else {
			start_x, start_y, step_x, step_y := av1.scale_mv_for_block(
				x, y, mv, sub_x, sub_y,
				fh.frame_width, fh.frame_height,
				ref_upscaled_w, ref_frame_h,
			)
			if bit_depth > 8 {
				rp := ref_plane_u16(ref_fb, plane, ref_upscaled_w, ref_frame_w, ref_frame_h, sub_x, sub_y)
				av1.convolve_2d_pred(
					preds[ref_list][:], w, &rp,
					start_x, start_y, step_x, step_y,
					w, h,
					int(cand_filters[0]), int(cand_filters[1]),
					is_compound, bit_depth,
				)
			} else {
				rp := ref_plane_u8(ref_fb, plane, ref_upscaled_w, ref_frame_w, ref_frame_h, sub_x, sub_y)
				av1.convolve_2d_pred(
					preds[ref_list][:], w, &rp,
					start_x, start_y, step_x, step_y,
					w, h,
					int(cand_filters[0]), int(cand_filters[1]),
					is_compound, bit_depth,
				)
			}
		}
	}

	if !is_compound && !is_inter_intra {
		max_val := i32((1 << uint(bit_depth)) - 1)
		if bit_depth > 8 {
			cur_plane := fb_plane_u16(cur_fb, plane)
			for r in 0 ..< h {
				for c in 0 ..< w {
					cur_plane[(y + r) * cur_stride + (x + c)] = u16(clamp(preds[0][r * w + c], 0, max_val))
				}
			}
		} else {
			cur_plane := fb_plane_u8(cur_fb, plane)
			for r in 0 ..< h {
				for c in 0 ..< w {
					cur_plane[(y + r) * cur_stride + (x + c)] = u8(clamp(preds[0][r * w + c], 0, max_val))
				}
			}
		}
		if t.motion_mode == av1.OBMC {
			predict_obmc(t, plane, x, y, w, h)
		}
		return
	}

	switch t.compound_type {
	case av1.COMPOUND_AVERAGE:
		if bit_depth > 8 {
			av1.blend_compound_average(fb_plane_u16(cur_fb, plane), cur_stride, x, y, w, h, preds[0][:], preds[1][:], w, bit_depth)
		} else {
			av1.blend_compound_average(fb_plane_u8(cur_fb, plane), cur_stride, x, y, w, h, preds[0][:], preds[1][:], w, bit_depth)
		}
	case av1.COMPOUND_DISTANCE:
		d0 := abs(av1.get_relative_dist(int(fh.order_hints[cand_refs[0]]), int(fh.order_hint), s.enable_order_hint, u8(s.order_hint_bits)))
		d1 := abs(av1.get_relative_dist(int(fh.order_hints[cand_refs[1]]), int(fh.order_hint), s.enable_order_hint, u8(s.order_hint_bits)))
		fwd_w, bck_w := av1.distance_weights(d0, d1)
		if bit_depth > 8 {
			av1.blend_compound_distance(fb_plane_u16(cur_fb, plane), cur_stride, x, y, w, h, preds[0][:], preds[1][:], w, fwd_w, bck_w, bit_depth)
		} else {
			av1.blend_compound_distance(fb_plane_u8(cur_fb, plane), cur_stride, x, y, w, h, preds[0][:], preds[1][:], w, fwd_w, bck_w, bit_depth)
		}
	case av1.COMPOUND_WEDGE:
		mask_stride := int(av1.BLOCK_WIDTH[t.mi_size])
		if plane == 0 {
			av1.build_wedge_mask(t.mask_buf[:], mask_stride, t.mi_size, int(t.wedge_sign), int(t.wedge_index))
		}
		if bit_depth > 8 {
			av1.blend_mask(
				fb_plane_u16(cur_fb, plane), cur_stride, x, y, w, h,
				preds[0][:], preds[1][:], w,
				t.mask_buf[:], mask_stride,
				sub_x, sub_y, is_inter_intra, is_inter_intra, bit_depth,
			)
		} else {
			av1.blend_mask(
				fb_plane_u8(cur_fb, plane), cur_stride, x, y, w, h,
				preds[0][:], preds[1][:], w,
				t.mask_buf[:], mask_stride,
				sub_x, sub_y, is_inter_intra, is_inter_intra, bit_depth,
			)
		}
	case av1.COMPOUND_INTRA:
		av1.build_interintra_mask(t.mask_buf[:], w, w, h, int(t.interintra_mode))
		if bit_depth > 8 {
			av1.blend_mask(
				fb_plane_u16(cur_fb, plane), cur_stride, x, y, w, h,
				preds[0][:], preds[1][:], w,
				t.mask_buf[:], w,
				sub_x, sub_y, true, false, bit_depth,
			)
		} else {
			av1.blend_mask(
				fb_plane_u8(cur_fb, plane), cur_stride, x, y, w, h,
				preds[0][:], preds[1][:], w,
				t.mask_buf[:], w,
				sub_x, sub_y, true, false, bit_depth,
			)
		}
	case av1.COMPOUND_DIFFWTD:
		mask_stride := int(av1.BLOCK_WIDTH[t.mi_size])
		if plane == 0 {
			av1.build_diffwtd_mask(
				t.mask_buf[:], mask_stride,
				preds[0][:], preds[1][:], w,
				w, h,
				int(t.mask_type), bit_depth,
			)
		}
		if bit_depth > 8 {
			av1.blend_mask(
				fb_plane_u16(cur_fb, plane), cur_stride, x, y, w, h,
				preds[0][:], preds[1][:], w,
				t.mask_buf[:], mask_stride,
				sub_x, sub_y, false, false, bit_depth,
			)
		} else {
			av1.blend_mask(
				fb_plane_u8(cur_fb, plane), cur_stride, x, y, w, h,
				preds[0][:], preds[1][:], w,
				t.mask_buf[:], mask_stride,
				sub_x, sub_y, false, false, bit_depth,
			)
		}
	}
}

@(private)
is_scaled_prev :: proc "contextless" (h: ^av1.Frame_Header, ref_frame: i8, slots: ^[av1.NUM_REF_FRAMES]Ref_Slot) -> bool {
	slot := &slots[h.ref_frame_idx[ref_frame - av1.LAST_FRAME]]
	return slot.upscaled_width != h.frame_width || slot.frame_height != h.frame_height
}

// Section 7.11.3.9: Overlapped block motion compensation (OBMC).
@(private)
predict_obmc :: proc(
	t: ^Tile_State,
	plane, x, y, w, h: int,
) #no_bounds_check {
	_ = x
	_ = y
	d := t.d
	fh := t.h
	s := t.s
	sub_x := plane > 0 ? uint(s.color.subsampling_x) : 0
	sub_y := plane > 0 ? uint(s.color.subsampling_y) : 0
	cur_fb := d.cur
	cur_stride := fb_stride(cur_fb, plane)
	bit_depth := int(s.color.bit_depth)

	obmc_buf := t.preds[0][:64 * 64]
	mi_sizes := d.mi_cells.mi_size
	mi_refs := d.mi_cells.ref_frames
	mi_mvs := d.mi_cells.mvs
	tc_filters := d.tile_cells.interp_filters

	// Above overlap (pass == 0).
	if t.avail_u && av1.get_plane_residual_size(t.mi_size, plane, sub_x, sub_y) >= av1.BLOCK_8X8 {
		n_limit := int(min(u8(4), av1.MI_WIDTH_LOG2[t.mi_size]))
		x4 := t.mi_col
		y4 := t.mi_row
		n_count := 0
		for n_count < n_limit && x4 < min(fh.mi_cols, t.mi_col + t.bw4) {
			cand_row := t.mi_row - 1
			cand_col := x4 | 1
			u_idx := cand_row * fh.mi_cols + cand_col
			u_size := mi_sizes[u_idx]
			u_ref0 := mi_refs[u_idx][0]
			step4 := clamp(int(av1.NUM_4X4_BLOCKS_WIDE[u_size]), 2, 16)
			if u_ref0 > av1.INTRA_FRAME {
				n_count += 1
				pred_w := min(w, (step4 * av1.MI_SIZE) >> sub_x)
				pred_h := min(h >> 1, 32 >> sub_y)
				pred_x := (x4 * 4) >> sub_x
				pred_y := (y4 * 4) >> sub_y
				slot := &d.ref_slots[fh.ref_frame_idx[u_ref0 - av1.LAST_FRAME]]
				ref_fb := slot.buf
				mv := mi_mvs[u_idx][0]
				u_flt := tc_filters[u_idx]
				sx, sy, stx, sty := av1.scale_mv_for_block(
					pred_x, pred_y, mv, sub_x, sub_y,
					fh.frame_width, fh.frame_height,
					slot.upscaled_width, slot.frame_height,
				)
				if bit_depth > 8 {
					rp := ref_plane_u16(ref_fb, plane, slot.upscaled_width, slot.frame_width, slot.frame_height, sub_x, sub_y)
					av1.convolve_2d_pred(
						obmc_buf, pred_w, &rp,
						sx, sy, stx, sty,
						pred_w, pred_h,
						int(u_flt[0]), int(u_flt[1]),
						false, bit_depth,
					)
					av1.blend_obmc(fb_plane_u16(cur_fb, plane), cur_stride, pred_x, pred_y, pred_w, pred_h, 0, obmc_buf, pred_w, bit_depth)
				} else {
					rp := ref_plane_u8(ref_fb, plane, slot.upscaled_width, slot.frame_width, slot.frame_height, sub_x, sub_y)
					av1.convolve_2d_pred(
						obmc_buf, pred_w, &rp,
						sx, sy, stx, sty,
						pred_w, pred_h,
						int(u_flt[0]), int(u_flt[1]),
						false, bit_depth,
					)
					av1.blend_obmc(fb_plane_u8(cur_fb, plane), cur_stride, pred_x, pred_y, pred_w, pred_h, 0, obmc_buf, pred_w, bit_depth)
				}
			}
			x4 += step4
		}
	}

	// Left overlap (pass == 1).
	if t.avail_l {
		n_limit := int(min(u8(4), av1.MI_HEIGHT_LOG2[t.mi_size]))
		x4 := t.mi_col
		y4 := t.mi_row
		n_count := 0
		for n_count < n_limit && y4 < min(fh.mi_rows, t.mi_row + t.bh4) {
			cand_col := t.mi_col - 1
			cand_row := y4 | 1
			l_idx := cand_row * fh.mi_cols + cand_col
			l_size := mi_sizes[l_idx]
			l_ref0 := mi_refs[l_idx][0]
			step4 := clamp(int(av1.NUM_4X4_BLOCKS_HIGH[l_size]), 2, 16)
			if l_ref0 > av1.INTRA_FRAME {
				n_count += 1
				pred_w := min(w >> 1, 32 >> sub_x)
				pred_h := min(h, (step4 * av1.MI_SIZE) >> sub_y)
				pred_x := (x4 * 4) >> sub_x
				pred_y := (y4 * 4) >> sub_y
				slot := &d.ref_slots[fh.ref_frame_idx[l_ref0 - av1.LAST_FRAME]]
				ref_fb := slot.buf
				mv := mi_mvs[l_idx][0]
				l_flt := tc_filters[l_idx]
				sx, sy, stx, sty := av1.scale_mv_for_block(
					pred_x, pred_y, mv, sub_x, sub_y,
					fh.frame_width, fh.frame_height,
					slot.upscaled_width, slot.frame_height,
				)
				if bit_depth > 8 {
					rp := ref_plane_u16(ref_fb, plane, slot.upscaled_width, slot.frame_width, slot.frame_height, sub_x, sub_y)
					av1.convolve_2d_pred(
						obmc_buf, pred_w, &rp,
						sx, sy, stx, sty,
						pred_w, pred_h,
						int(l_flt[0]), int(l_flt[1]),
						false, bit_depth,
					)
					av1.blend_obmc(fb_plane_u16(cur_fb, plane), cur_stride, pred_x, pred_y, pred_w, pred_h, 1, obmc_buf, pred_w, bit_depth)
				} else {
					rp := ref_plane_u8(ref_fb, plane, slot.upscaled_width, slot.frame_width, slot.frame_height, sub_x, sub_y)
					av1.convolve_2d_pred(
						obmc_buf, pred_w, &rp,
						sx, sy, stx, sty,
						pred_w, pred_h,
						int(l_flt[0]), int(l_flt[1]),
						false, bit_depth,
					)
					av1.blend_obmc(fb_plane_u8(cur_fb, plane), cur_stride, pred_x, pred_y, pred_w, pred_h, 1, obmc_buf, pred_w, bit_depth)
				}
			}
			y4 += step4
		}
	}
}

// Section 5.11.34: Residual syntax.
@(private)
residual :: proc(t: ^Tile_State) -> Error #no_bounds_check {
	sb_mask := t.s.use_128x128_superblock ? 31 : 15
	if t.is_inter && t.skip && t.mi_row + t.bh4 <= t.h.mi_rows && t.mi_col + t.bw4 <= t.h.mi_cols {
		sb_r := t.mi_row & sb_mask
		sb_c := t.mi_col & sb_mask
		y_bits := transmute(Block_Decoded_Row)(((u64(1) << uint(t.bw4)) - 1) << uint(sb_c + 1))
		for i in 0 ..< t.bh4 {
			t.block_decoded[0][sb_r + i + 1] += y_bits
		}
		if t.has_chroma {
			sub_x := uint(t.s.color.subsampling_x)
			sub_y := uint(t.s.color.subsampling_y)
			c_r := sb_r >> sub_y
			c_c := sb_c >> sub_x
			c_h4 := max(1, t.bh4 >> sub_y)
			c_w4 := max(1, t.bw4 >> sub_x)
			uv_bits := transmute(Block_Decoded_Row)(((u64(1) << uint(c_w4)) - 1) << uint(c_c + 1))
			for i in 0 ..< c_h4 {
				t.block_decoded[1][c_r + i + 1] += uv_bits
				t.block_decoded[2][c_r + i + 1] += uv_bits
			}
		}
		return .None
	}
	width_chunks := max(1, int(av1.BLOCK_WIDTH[t.mi_size]) >> 6)
	height_chunks := max(1, int(av1.BLOCK_HEIGHT[t.mi_size]) >> 6)
	mi_size_chunk := (width_chunks > 1 || height_chunks > 1) ? av1.BLOCK_64X64 : t.mi_size
	num_planes := 1 + (t.has_chroma ? 2 : 0)

	for chunk_y in 0 ..< height_chunks {
		for chunk_x in 0 ..< width_chunks {
			mi_row_chunk := t.mi_row + (chunk_y << 4)
			mi_col_chunk := t.mi_col + (chunk_x << 4)
			sub_block_mi_row := mi_row_chunk & sb_mask
			sub_block_mi_col := mi_col_chunk & sb_mask

			for plane in 0 ..< num_planes {
				sub_x := plane > 0 ? uint(t.s.color.subsampling_x) : 0
				sub_y := plane > 0 ? uint(t.s.color.subsampling_y) : 0
				tx_sz := av1.get_uv_tx_size(mi_size_chunk, plane, int(t.tx_size), t.lossless, t.s.color.subsampling_x, t.s.color.subsampling_y)
				step_x := int(av1.TX_WIDTH[tx_sz]) >> 2
				step_y := int(av1.TX_HEIGHT[tx_sz]) >> 2
				plane_sz := av1.get_plane_residual_size(mi_size_chunk, plane, sub_x, sub_y)
				num4x4_w := int(av1.NUM_4X4_BLOCKS_WIDE[plane_sz])
				num4x4_h := int(av1.NUM_4X4_BLOCKS_HIGH[plane_sz])
				base_x := (mi_col_chunk >> sub_x) * av1.MI_SIZE
				base_y := (mi_row_chunk >> sub_y) * av1.MI_SIZE

				if t.is_inter && !t.lossless && plane == 0 {
					transform_tree(t, base_x, base_y, num4x4_w * 4, num4x4_h * 4) or_return
				} else {
					base_x_block := (t.mi_col >> sub_x) * av1.MI_SIZE
					base_y_block := (t.mi_row >> sub_y) * av1.MI_SIZE
					for y := 0; y < num4x4_h; y += step_y {
						for x := 0; x < num4x4_w; x += step_x {
							transform_block(
								t, plane, base_x_block, base_y_block, tx_sz,
								x + ((chunk_x << 4) >> sub_x),
								y + ((chunk_y << 4) >> sub_y),
								sub_block_mi_row, sub_block_mi_col,
							) or_return
						}
					}
				}
			}
		}
	}
	return .None
}

// Section 5.11.35: Transform tree syntax.
@(private)
transform_tree :: proc(t: ^Tile_State, start_x, start_y, w, h: int) -> Error {
	max_x := t.h.mi_cols * av1.MI_SIZE
	max_y := t.h.mi_rows * av1.MI_SIZE
	if start_x >= max_x || start_y >= max_y {
		return .None
	}
	row := start_y >> av1.MI_SIZE_LOG2
	col := start_x >> av1.MI_SIZE_LOG2
	luma_tx_sz := int(t.d.tile_cells[row * t.h.mi_cols + col].inter_tx_size)
	luma_w := int(av1.TX_WIDTH[luma_tx_sz])
	luma_h := int(av1.TX_HEIGHT[luma_tx_sz])
	if w <= luma_w && h <= luma_h {
		tx_sz := av1.find_tx_size(w, h)
		sb_mask := t.s.use_128x128_superblock ? 31 : 15
		transform_block(t, 0, start_x, start_y, tx_sz, 0, 0, row & sb_mask, col & sb_mask) or_return
	} else {
		if w > h {
			transform_tree(t, start_x, start_y, w / 2, h) or_return
			transform_tree(t, start_x + w / 2, start_y, w / 2, h) or_return
		} else if w < h {
			transform_tree(t, start_x, start_y, w, h / 2) or_return
			transform_tree(t, start_x, start_y + h / 2, w, h / 2) or_return
		} else {
			transform_tree(t, start_x, start_y, w / 2, h / 2) or_return
			transform_tree(t, start_x + w / 2, start_y, w / 2, h / 2) or_return
			transform_tree(t, start_x, start_y + h / 2, w / 2, h / 2) or_return
			transform_tree(t, start_x + w / 2, start_y + h / 2, w / 2, h / 2) or_return
		}
	}
	return .None
}

// Section 5.11.35: Transform block syntax.
@(private)
transform_block :: proc(
	t: ^Tile_State,
	plane, base_x, base_y, tx_sz, x, y: int,
	sub_block_mi_row, sub_block_mi_col: int,
) -> Error #no_bounds_check {
	start_x := base_x + 4 * x
	start_y := base_y + 4 * y
	sub_x := plane > 0 ? uint(t.s.color.subsampling_x) : 0
	sub_y := plane > 0 ? uint(t.s.color.subsampling_y) : 0
	row := (start_y << sub_y) >> av1.MI_SIZE_LOG2
	col := (start_x << sub_x) >> av1.MI_SIZE_LOG2
	sb_mask := t.s.use_128x128_superblock ? 31 : 15
	sub_row := row & sb_mask
	sub_col := col & sb_mask
	step_x := int(av1.TX_WIDTH[tx_sz]) >> av1.MI_SIZE_LOG2
	step_y := int(av1.TX_HEIGHT[tx_sz]) >> av1.MI_SIZE_LOG2
	max_x := (t.h.mi_cols * av1.MI_SIZE) >> sub_x
	max_y := (t.h.mi_rows * av1.MI_SIZE) >> sub_y
	if start_x >= max_x || start_y >= max_y {
		return .None
	}

	cur_fb := t.d.cur
	stride := fb_stride(cur_fb, plane)
	bit_depth := int(t.s.color.bit_depth)

	if !t.is_inter {
		if (plane == 0 && t.palette_size_y > 0) || (plane > 0 && t.palette_size_uv > 0) {
			palette := plane == 0 ? t.palette_colors_y[:] : (plane == 1 ? t.palette_colors_u[:] : t.palette_colors_v[:])
			cmap := plane == 0 ? t.color_map_y[:] : t.color_map_uv[:]
			if bit_depth > 8 {
				av1.predict_palette(fb_plane_u16(cur_fb, plane), stride, start_x, start_y, x, y, tx_sz, palette, cmap, 64)
			} else {
				av1.predict_palette(fb_plane_u8(cur_fb, plane), stride, start_x, start_y, x, y, tx_sz, palette, cmap, 64)
			}
		} else {
			is_cfl := plane > 0 && t.uv_mode == av1.UV_CFL_PRED
			mode := plane == 0 ? t.y_mode : (is_cfl ? u8(av1.DC_PRED) : t.uv_mode)
			angle_delta := plane == 0 ? t.angle_delta_y : t.angle_delta_uv
			avail_l := (plane == 0 ? t.avail_l : t.avail_l_chroma) || x > 0
			avail_u := (plane == 0 ? t.avail_u : t.avail_u_chroma) || y > 0
			bd_tr := ((sub_col >> sub_x) + step_x) in t.block_decoded[plane][sub_row >> sub_y]
			bd_bl := ((sub_col >> sub_x) - 1) in t.block_decoded[plane][(sub_row >> sub_y) + step_y + 1]
			edges := av1.Intra_Edges{
				have_left                = avail_l,
				have_above               = avail_u,
				have_above_right         = bd_tr,
				have_below_left          = bd_bl,
				max_x                    = max_x - 1,
				max_y                    = max_y - 1,
				enable_intra_edge_filter = t.s.enable_intra_edge_filter,
				filter_type              = get_filter_type(t, plane),
				angle_delta              = int(angle_delta),
				use_filter_intra         = plane == 0 && t.use_filter_intra,
				filter_intra_mode        = int(t.filter_intra_mode),
			}
			if bit_depth > 8 {
				av1.predict_intra(fb_plane_u16(cur_fb, plane), stride, start_x, start_y, tx_sz, int(mode), edges, bit_depth)
			} else {
				av1.predict_intra(fb_plane_u8(cur_fb, plane), stride, start_x, start_y, tx_sz, int(mode), edges, bit_depth)
			}
			if is_cfl {
				alpha := int(plane == 1 ? t.cfl_alpha_u : t.cfl_alpha_v)
				if bit_depth > 8 {
					av1.predict_chroma_from_luma(
						fb_plane_u16(cur_fb, plane), stride,
						fb_plane_u16(cur_fb, 0), cur_fb.y_stride,
						start_x, start_y, tx_sz, alpha,
						sub_x, sub_y, t.max_luma_w, t.max_luma_h, bit_depth,
					)
				} else {
					av1.predict_chroma_from_luma(
						fb_plane_u8(cur_fb, plane), stride,
						fb_plane_u8(cur_fb, 0), cur_fb.y_stride,
						start_x, start_y, tx_sz, alpha,
						sub_x, sub_y, t.max_luma_w, t.max_luma_h, bit_depth,
					)
				}
			}
		}
		if plane == 0 {
			t.max_luma_w = start_x + step_x * 4
			t.max_luma_h = start_y + step_y * 4
		}
	}

	if !t.skip {
		eob, max_r, max_c, dc_dq0 := coeffs(t, plane, start_x, start_y, tx_sz)
		if eob > 0 {
			reconstruct(t, plane, start_x, start_y, tx_sz, eob, max_r, max_c, dc_dq0)
		}
	}

	bd_plane := &t.block_decoded[plane]
	base_r := (sub_row >> sub_y) + 1
	base_c := (sub_col >> sub_x) + 1
	row_bits := transmute(Block_Decoded_Row)(((u64(1) << uint(step_x)) - 1) << uint(base_c))
	for i in 0 ..< step_y {
		bd_plane[base_r + i] += row_bits
	}
	return .None
}

@(private)
read_golomb :: #force_no_inline proc "contextless" (sym: ^av1.Symbol_Decoder, q_in: i32) -> i32 {
	length := 0
	for {
		length += 1
		if av1.read_bool(sym) != 0 || length >= 20 {
			break
		}
	}
	rem := 1
	for i := length - 2; i >= 0; i -= 1 {
		rem = (rem << 1) | int(av1.read_bool(sym))
	}
	return q_in + i32(rem - 1)
}

@(private, rodata)
SKIP_CTX_3X3 := [3][3]u8{
	{1, 2, 3},
	{2, 4, 5},
	{3, 5, 6},
}

@(private)
load_ctx_u64_or :: #force_inline proc "contextless" (p: [^]u8, n: int) -> u64 #no_bounds_check {
	if n <= 8 {
		return intrinsics.unaligned_load((^u64)(p)) << uint((8 - n) << 3)
	}
	return intrinsics.unaligned_load((^u64)(p)) | intrinsics.unaligned_load((^u64)(&p[8]))
}

@(private)
dc_sign_delta :: #force_inline proc "contextless" (p: [^]u8, n: int) -> int #no_bounds_check {
	POS_MASK :: u64(0x8080808080808080)
	NEG_MASK :: u64(0x4040404040404040)
	if n <= 8 {
		v := intrinsics.unaligned_load((^u64)(p)) << uint((8 - n) << 3)
		return int(intrinsics.count_ones(v & POS_MASK)) - int(intrinsics.count_ones(v & NEG_MASK))
	}
	v0 := intrinsics.unaligned_load((^u64)(p))
	v1 := intrinsics.unaligned_load((^u64)(&p[8]))
	pos := intrinsics.count_ones(v0 & POS_MASK) + intrinsics.count_ones(v1 & POS_MASK)
	neg := intrinsics.count_ones(v0 & NEG_MASK) + intrinsics.count_ones(v1 & NEG_MASK)
	return int(pos) - int(neg)
}

// Sync only the 4 mutable state fields (`byte_pos, dif, rng, cnt`) between the register-promoted
// local `sym` in `coeffs` and `t.sym`. Whole-struct copies (`t.sym = sym`) would copy trailing
// struct padding after `disable_cdf_update: bool`, causing LLVM SROA to allocate `%xmm` registers
// and emit `insertq` / stack spills for padding bytes inside the hot `coeff_base_cdf` loop.
@(private)
sync_sym_to_tile :: #force_inline proc "contextless" (t: ^Tile_State, sym: ^av1.Symbol_Decoder) {
	t.sym.byte_pos = sym.byte_pos
	t.sym.dif = sym.dif
	t.sym.rng = sym.rng
	t.sym.cnt = sym.cnt
}

@(private)
sync_sym_from_tile :: #force_inline proc "contextless" (t: ^Tile_State, sym: ^av1.Symbol_Decoder) {
	sym.byte_pos = t.sym.byte_pos
	sym.dif = t.sym.dif
	sym.rng = t.sym.rng
	sym.cnt = t.sym.cnt
}

@(private)
coeff_br :: #force_inline proc "contextless" (t: ^Tile_State, sym: ^av1.Symbol_Decoder, cdf: [^]u16) -> int #no_bounds_check {
	sync_sym_to_tile(t, sym)
	res := av1.read_coeff_br(&t.sym, cdf)
	sync_sym_from_tile(t, sym)
	return res
}

// Section 5.11.39: Coefficients syntax & fused dequantization.
@(private)
coeffs :: #force_no_inline proc "contextless" (t: ^Tile_State, plane, start_x, start_y, tx_sz: int) -> (eob, max_r, max_c: int, dc_dq0: i32) #no_bounds_check {
	x4 := start_x >> 2
	y4 := start_y >> 2
	w_log2 := uint(av1.TX_WIDTH_LOG2[tx_sz]) & 7
	w := int(av1.TX_WIDTH[tx_sz])
	h := int(av1.TX_HEIGHT[tx_sz])
	w4 := w >> 2
	h4 := h >> 2
	tx_sz_ctx := (int(av1.TX_SIZE_SQR[tx_sz]) + int(av1.TX_SIZE_SQR_UP[tx_sz]) + 1) >> 1
	ptype := plane > 0 ? 1 : 0

	cul_level := 0
	dc_category := u8(0)

	sub_x := plane > 0 ? (uint(t.s.color.subsampling_x) & 1) : 0
	sub_y := plane > 0 ? (uint(t.s.color.subsampling_y) & 1) : 0
	max_x4 := t.h.mi_cols >> sub_x
	max_y4 := t.h.mi_rows >> sub_y
	tile_x4 := t.mi_col_start >> sub_x

	a_ptr := ([^]u8)(&t.above_coeff_ctx[plane][x4 - tile_x4])
	l_ptr := ([^]u8)(&t.left_coeff_ctx[plane][y4 & 63])

	// Compute all_zero context (8.3.2).
	bsize := av1.get_plane_residual_size(t.mi_size, plane, sub_x, sub_y)
	bw := int(av1.BLOCK_WIDTH[bsize])
	bh := int(av1.BLOCK_HEIGHT[bsize])

	skip_ctx := 0
	if plane == 0 {
		if bw != w || bh != h {
			va := load_ctx_u64_or(a_ptr, w4)
			vl := load_ctx_u64_or(l_ptr, h4)
			top_cls := int((va & 0x3F3F3F3F3F3F3F3F) != 0) + int((va & 0x3C3C3C3C3C3C3C3C) != 0)
			left_cls := int((vl & 0x3F3F3F3F3F3F3F3F) != 0) + int((vl & 0x3C3C3C3C3C3C3C3C) != 0)
			skip_ctx = int(SKIP_CTX_3X3[top_cls][left_cls])
		}
	} else {
		va := load_ctx_u64_or(a_ptr, w4)
		vl := load_ctx_u64_or(l_ptr, h4)
		skip_ctx = int(va != 0) + int(vl != 0) + 7 + (bw != w || bh != h ? 3 : 0)
	}

	all_zero := av1.read_symbol_bool(&t.sym, t.cdfs.txb_skip_cdf[tx_sz_ctx][skip_ctx][:]) != 0
	if all_zero {
		if plane == 0 && t.is_inter && t.has_chroma {
			tx_base := ((y4 & 15) << 4) | (x4 & 15)
			for j in 0 ..< h4 {
				fill_row_u8(&t.tx_type_grid[tx_base + (j << 4)], w4, av1.DCT_DCT)
			}
		}
	} else {
		if plane == 0 {
			t.plane_tx_type = transform_type(t, &t.sym, x4, y4, tx_sz)
		} else {
			t.plane_tx_type = compute_tx_type(t, plane, tx_sz, x4, y4)
		}
		tx_class := av1.get_tx_class(int(t.plane_tx_type))
		scan := av1.get_scan(tx_sz, int(t.plane_tx_type))
		eob_ctx := tx_class == av1.TX_CLASS_2D ? 0 : 1
		eob_multisize := min(int(w_log2), 5) + min(int(av1.TX_HEIGHT_LOG2[tx_sz]), 5) - 4
		eob_pt: int
		switch eob_multisize {
		case 0: eob_pt = av1.read_symbol(&t.sym, t.cdfs.eob_pt_16_cdf[ptype][eob_ctx][:]) + 1
		case 1: eob_pt = av1.read_symbol(&t.sym, t.cdfs.eob_pt_32_cdf[ptype][eob_ctx][:]) + 1
		case 2: eob_pt = av1.read_symbol(&t.sym, t.cdfs.eob_pt_64_cdf[ptype][eob_ctx][:]) + 1
		case 3: eob_pt = av1.read_symbol(&t.sym, t.cdfs.eob_pt_128_cdf[ptype][eob_ctx][:]) + 1
		case 4: eob_pt = av1.read_symbol(&t.sym, t.cdfs.eob_pt_256_cdf[ptype][eob_ctx][:]) + 1
		case 5: eob_pt = av1.read_symbol(&t.sym, t.cdfs.eob_pt_512_cdf[ptype][:]) + 1
		case:   eob_pt = av1.read_symbol(&t.sym, t.cdfs.eob_pt_1024_cdf[ptype][:]) + 1
		}
		sym := av1.Symbol_Decoder{
			data               = t.sym.data,
			byte_pos           = t.sym.byte_pos,
			dif                = t.sym.dif,
			rng                = t.sym.rng,
			cnt                = t.sym.cnt,
			disable_cdf_update = t.sym.disable_cdf_update,
		}
		if eob_pt < 2 {
			eob = eob_pt
		} else {
			eob_shift := eob_pt - 2
			eob = (1 << (uint(eob_shift) & 15)) + 1
			if eob_shift > 0 {
				eob_extra := av1.read_symbol_bool(&sym, t.cdfs.eob_extra_cdf[tx_sz_ctx][ptype][eob_pt - 3][:])
				if eob_extra != 0 {
					eob += 1 << (uint(eob_shift - 1) & 15)
				}
				for i in 1 ..< eob_shift {
					if av1.read_bool(&sym) != 0 {
						eob += 1 << (uint(eob_shift - 1 - i) & 15)
					}
				}
			}
		}

		bit_depth := int(t.s.color.bit_depth)
		dc_q := t.seg_dc_q[plane][t.segment_id]
		ac_q := t.seg_ac_q[plane][t.segment_id]
		qm_level := t.d.seg_qm_level[plane][t.segment_id]
		br_tx_ctx := min(tx_sz_ctx, av1.TX_32X32)

		if eob == 1 {
			level := av1.read_symbol_3(&sym, raw_data(t.cdfs.coeff_base_eob_cdf[tx_sz_ctx][ptype][0][:])) + 1
			if level > av1.NUM_BASE_LEVELS {
				level = coeff_br(t, &sym, raw_data(t.cdfs.coeff_br_cdf[br_tx_ctx][ptype][0][:]))
			}
			q0 := i32(level)
			dc_sign_sum := dc_sign_delta(a_ptr, w4) + dc_sign_delta(l_ptr, h4)
			dc_ctx := dc_sign_sum < 0 ? 1 : (dc_sign_sum > 0 ? 2 : 0)
			sign := av1.read_symbol_bool(&sym, t.cdfs.dc_sign_cdf[ptype][dc_ctx][:])
			if q0 > av1.NUM_BASE_LEVELS + av1.COEFF_BASE_RANGE {
				sync_sym_to_tile(t, &sym)
				q0 = read_golomb(&t.sym, q0)
				sync_sym_from_tile(t, &sym)
			}
			dc_category = sign != 0 ? 1 : 2
			q0 &= 0xFFFFF
			cul_level = min(63, int(q0))
			dc_dq0 = av1.dequant_coeff(
				sign != 0 ? -q0 : q0, 0, tx_sz, int(t.plane_tx_type), plane,
				dc_q, ac_q, qm_level, bit_depth,
			)
			if int(t.plane_tx_type) != av1.DCT_DCT || t.lossless {
				t.dequant_buf[0] = dc_dq0
			}
		} else {
			adj_tx_sz := int(av1.ADJUSTED_TX_SIZE[tx_sz])
			bwl := uint(av1.TX_WIDTH_LOG2[adj_tx_sz]) & 7
			bw_mask := (1 << bwl) - 1
			txh := int(av1.TX_HEIGHT[adj_tx_sz])
			nz_count := 1

			{
				c := eob - 1
				pos := int(scan[c])
				row := pos >> bwl
				col := pos & bw_mask
				lpos := (row << 6) | col
				base_eob_ctx := 1 + int(c > ((txh << bwl) >> 3)) + int(c > ((txh << bwl) >> 2))
				level := av1.read_symbol_3(&sym, raw_data(t.cdfs.coeff_base_eob_cdf[tx_sz_ctx][ptype][base_eob_ctx][:])) + 1
				level_tok := u8(level * 0x41)
				if level > av1.NUM_BASE_LEVELS {
					br_ctx: int
					if tx_class == av1.TX_CLASS_2D {
						br_ctx = ((row | col) < 2) ? 7 : 14
					} else if tx_class == av1.TX_CLASS_HORIZ {
						br_ctx = col == 0 ? 7 : 14
					} else {
						br_ctx = row == 0 ? 7 : 14
					}
					level = coeff_br(t, &sym, raw_data(t.cdfs.coeff_br_cdf[br_tx_ctx][ptype][br_ctx][:]))
					level_tok = u8(level + (3 << 6))
				}
				t.levels_buf[lpos] = level_tok
				t.nz_list[0] = (u32(level) << 16) | u32(lpos)
			}

			dc_base_ctx: int
			dc_m3: int
			base_cdfs := &t.cdfs.coeff_base_cdf[tx_sz_ctx][ptype]
			br_cdfs := &t.cdfs.coeff_br_cdf[br_tx_ctx][ptype]

			if tx_class == av1.TX_CLASS_2D {
				base_ctx_table := &av1.COEFF_BASE_CTX_OFFSET[tx_sz]
				for c := eob - 2; c >= 1; c -= 1 {
					pos := int(scan[c])
					row := pos >> bwl
					col := pos & bw_mask
					lpos := (row << 6) | col
					l := ([^]u8)(&t.levels_buf[lpos])
					m3 := int(l[1]) + int(l[64]) + int(l[65])
					m5 := m3 + int(l[2]) + int(l[128])
					ctx := (m5 > 512 ? 4 : (m5 + 64) >> 7) + int(base_ctx_table[min(row, 4)][min(col, 4)])
					level := av1.read_symbol_4(&sym, raw_data(base_cdfs[ctx][:]))
					if level != 0 {
						level_tok := u8(level * 0x41)
						if level > av1.NUM_BASE_LEVELS {
							m_br := m3 & 63
							br_ctx := (m_br > 12 ? 6 : (m_br + 1) >> 1) + (((row | col) < 2) ? 7 : 14)
							level = coeff_br(t, &sym, raw_data(br_cdfs[br_ctx][:]))
							level_tok = u8(level + (3 << 6))
						}
						t.levels_buf[lpos] = level_tok
						t.nz_list[nz_count] = (u32(level) << 16) | u32(lpos)
						nz_count += 1
					}
				}
				dc_base_ctx = 0
				dc_m3 = int(t.levels_buf[1]) + int(t.levels_buf[64]) + int(t.levels_buf[65])
			} else if tx_class == av1.TX_CLASS_HORIZ {
				for c := eob - 2; c >= 1; c -= 1 {
					pos := int(scan[c])
					row := pos >> bwl
					col := pos & bw_mask
					lpos := (row << 6) | col
					l := ([^]u8)(&t.levels_buf[lpos])
					m3 := int(l[1]) + int(l[64]) + int(l[2])
					m5 := m3 + int(l[3]) + int(l[4])
					ctx := (m5 > 512 ? 4 : (m5 + 64) >> 7) + int(av1.COEFF_BASE_POS_CTX_OFFSET[min(col, 2)])
					level := av1.read_symbol_4(&sym, raw_data(base_cdfs[ctx][:]))
					if level != 0 {
						level_tok := u8(level * 0x41)
						if level > av1.NUM_BASE_LEVELS {
							m_br := m3 & 63
							br_ctx := (m_br > 12 ? 6 : (m_br + 1) >> 1) + (col == 0 ? 7 : 14)
							level = coeff_br(t, &sym, raw_data(br_cdfs[br_ctx][:]))
							level_tok = u8(level + (3 << 6))
						}
						t.levels_buf[lpos] = level_tok
						t.nz_list[nz_count] = (u32(level) << 16) | u32(lpos)
						nz_count += 1
					}
				}
				dc_m3 = int(t.levels_buf[1]) + int(t.levels_buf[64]) + int(t.levels_buf[2])
				dc_m5 := dc_m3 + int(t.levels_buf[3]) + int(t.levels_buf[4])
				dc_base_ctx = (dc_m5 > 512 ? 4 : (dc_m5 + 64) >> 7) + int(av1.COEFF_BASE_POS_CTX_OFFSET[0])
			} else {
				for c := eob - 2; c >= 1; c -= 1 {
					pos := int(scan[c])
					row := pos >> bwl
					col := pos & bw_mask
					lpos := (row << 6) | col
					l := ([^]u8)(&t.levels_buf[lpos])
					m3 := int(l[1]) + int(l[64]) + int(l[128])
					m5 := m3 + int(l[192]) + int(l[256])
					ctx := (m5 > 512 ? 4 : (m5 + 64) >> 7) + int(av1.COEFF_BASE_POS_CTX_OFFSET[min(row, 2)])
					level := av1.read_symbol_4(&sym, raw_data(base_cdfs[ctx][:]))
					if level != 0 {
						level_tok := u8(level * 0x41)
						if level > av1.NUM_BASE_LEVELS {
							m_br := m3 & 63
							br_ctx := (m_br > 12 ? 6 : (m_br + 1) >> 1) + (row == 0 ? 7 : 14)
							level = coeff_br(t, &sym, raw_data(br_cdfs[br_ctx][:]))
							level_tok = u8(level + (3 << 6))
						}
						t.levels_buf[lpos] = level_tok
						t.nz_list[nz_count] = (u32(level) << 16) | u32(lpos)
						nz_count += 1
					}
				}
				dc_m3 = int(t.levels_buf[1]) + int(t.levels_buf[64]) + int(t.levels_buf[128])
				dc_m5 := dc_m3 + int(t.levels_buf[192]) + int(t.levels_buf[256])
				dc_base_ctx = (dc_m5 > 512 ? 4 : (dc_m5 + 64) >> 7) + int(av1.COEFF_BASE_POS_CTX_OFFSET[0])
			}

			dc_level := av1.read_symbol_4(&sym, raw_data(base_cdfs[dc_base_ctx][:]))
			if dc_level != 0 {
				if dc_level > av1.NUM_BASE_LEVELS {
					m_br := dc_m3 & 63
					br_ctx := m_br > 12 ? 6 : (m_br + 1) >> 1
					dc_level = coeff_br(t, &sym, raw_data(br_cdfs[br_ctx][:]))
				}
				q0 := i32(dc_level)
				dc_sign_sum := dc_sign_delta(a_ptr, w4) + dc_sign_delta(l_ptr, h4)
				dc_ctx := dc_sign_sum < 0 ? 1 : (dc_sign_sum > 0 ? 2 : 0)
				sign := av1.read_symbol_bool(&sym, t.cdfs.dc_sign_cdf[ptype][dc_ctx][:])
				if q0 > av1.NUM_BASE_LEVELS + av1.COEFF_BASE_RANGE {
					sync_sym_to_tile(t, &sym)
					q0 = read_golomb(&t.sym, q0)
					sync_sym_from_tile(t, &sym)
				}
				dc_category = sign != 0 ? 1 : 2
				q0 &= 0xFFFFF
				cul_level += int(q0)
				t.dequant_buf[0] = av1.dequant_coeff(
					sign != 0 ? -q0 : q0, 0, tx_sz, int(t.plane_tx_type), plane,
					dc_q, ac_q, qm_level, bit_depth,
				)
			}

			dq_shift := uint(av1.TX_DQ_SHIFT[tx_sz]) & 3
			dq_limit := i32(1) << (uint(7 + bit_depth) & 31)
			use_qm := qm_level < 15 && int(t.plane_tx_type) < av1.IDTX
			if !use_qm {
				for i := nz_count - 1; i >= 0; i -= 1 {
					entry := t.nz_list[i]
					lpos := int(entry & 0xFFFF)
					q := i32(entry >> 16)
					t.levels_buf[lpos] = 0
					row := lpos >> 6
					col := lpos & 63
					max_r = max(max_r, row)
					max_c = max(max_c, col)
					sign := av1.read_bool(&sym)
					abs_dq: i32
					if q > av1.NUM_BASE_LEVELS + av1.COEFF_BASE_RANGE {
						sync_sym_to_tile(t, &sym)
						q = read_golomb(&t.sym, q) & 0xFFFFF
						sync_sym_from_tile(t, &sym)
						abs_dq = i32(((u64(q) * u64(ac_q)) & 0xFFFFFF) >> dq_shift)
						abs_dq = min(abs_dq, dq_limit - 1 + i32(sign))
					} else {
						abs_dq = (q * ac_q) >> dq_shift
					}
					cul_level += int(q)
					t.dequant_buf[(row << w_log2) | col] = sign != 0 ? -abs_dq : abs_dq
				}
			} else {
				qm_base := int(av1.QM_OFFSET[tx_sz])
				qm_table := &av1.QUANTIZER_MATRIX[qm_level][ptype]
				for i := nz_count - 1; i >= 0; i -= 1 {
					entry := t.nz_list[i]
					lpos := int(entry & 0xFFFF)
					q := i32(entry >> 16)
					t.levels_buf[lpos] = 0
					row := lpos >> 6
					col := lpos & 63
					max_r = max(max_r, row)
					max_c = max(max_c, col)
					sign := av1.read_bool(&sym)
					if q > av1.NUM_BASE_LEVELS + av1.COEFF_BASE_RANGE {
						sync_sym_to_tile(t, &sym)
						q = read_golomb(&t.sym, q) & 0xFFFFF
						sync_sym_from_tile(t, &sym)
					}
					cul_level += int(q)
					pos := (row << bwl) | col
					qm := i32(qm_table[qm_base + pos])
					step_q := av1.round2(i64(ac_q * qm), 5)
					abs_dq := i32(((u64(q) * u64(step_q)) & 0xFFFFFF) >> dq_shift)
					abs_dq = min(abs_dq, dq_limit - 1 + i32(sign))
					t.dequant_buf[(row << w_log2) | col] = sign != 0 ? -abs_dq : abs_dq
				}
			}
			cul_level = min(63, cul_level)
		}
		sync_sym_to_tile(t, &sym)
	}

	ctx_byte := (dc_category << 6) | u8(cul_level)
	fill_row_u8(a_ptr, min(w4, max_x4 - x4), ctx_byte)
	fill_row_u8(l_ptr, min(h4, max_y4 - y4), ctx_byte)
	return
}

// Section 5.11.47: Transform type syntax.
@(private)
transform_type :: proc "contextless" (t: ^Tile_State, sym: ^av1.Symbol_Decoder, x4, y4, tx_sz: int) -> u8 #no_bounds_check {
	set := av1.get_tx_set(tx_sz, t.is_inter, t.h.reduced_tx_set)
	tx_type := u8(av1.DCT_DCT)
	q_idx := t.h.segmentation_enabled ? get_qindex(t.h, true, t.current_qindex, int(t.segment_id)) : int(t.h.base_q_idx)
	if set > 0 && q_idx > 0 {
		tx_sz_sqr := int(av1.TX_SIZE_SQR[tx_sz])
		if t.is_inter {
			switch set {
			case av1.TX_SET_INTER_1:
				s := av1.read_symbol(sym, t.cdfs.inter_tx_type_set1_cdf[tx_sz_sqr][:])
				tx_type = av1.TX_TYPE_INTER_INV_SET1[s]
			case av1.TX_SET_INTER_2:
				s := av1.read_symbol(sym, t.cdfs.inter_tx_type_set2_cdf[:])
				tx_type = av1.TX_TYPE_INTER_INV_SET2[s]
			case:
				s := av1.read_symbol(sym, t.cdfs.inter_tx_type_set3_cdf[tx_sz_sqr][:])
				tx_type = av1.TX_TYPE_INTER_INV_SET3[s]
			}
		} else {
			intra_dir := t.use_filter_intra ? av1.FILTER_INTRA_MODE_TO_INTRA_DIR[t.filter_intra_mode] : t.y_mode
			if set == av1.TX_SET_INTRA_1 {
				s := av1.read_symbol(sym, t.cdfs.intra_tx_type_set1_cdf[tx_sz_sqr][intra_dir][:])
				tx_type = av1.TX_TYPE_INTRA_INV_SET1[s]
			} else {
				s := av1.read_symbol(sym, t.cdfs.intra_tx_type_set2_cdf[tx_sz_sqr][intra_dir][:])
				tx_type = av1.TX_TYPE_INTRA_INV_SET2[s]
			}
		}
	}
	if t.is_inter && t.has_chroma {
		w4 := int(av1.TX_WIDTH[tx_sz]) >> 2
		h4 := int(av1.TX_HEIGHT[tx_sz]) >> 2
		tx_base := ((y4 & 15) << 4) | (x4 & 15)
		for j in 0 ..< h4 {
			fill_row_u8(&t.tx_type_grid[tx_base + (j << 4)], w4, tx_type)
		}
	}
	return tx_type
}

// Section 5.11.40: Compute transform type function.
@(private)
compute_tx_type :: proc "contextless" (t: ^Tile_State, plane, tx_sz, block_x, block_y: int) -> u8 #no_bounds_check {
	tx_sz_sqr_up := int(av1.TX_SIZE_SQR_UP[tx_sz])
	if t.lossless || tx_sz_sqr_up > av1.TX_32X32 {
		return av1.DCT_DCT
	}
	tx_set := av1.get_tx_set(tx_sz, t.is_inter, t.h.reduced_tx_set)
	if plane == 0 {
		return t.tx_type_grid[((block_y & 15) << 4) | (block_x & 15)]
	}
	if t.is_inter {
		x4 := max(t.mi_col, block_x << uint(t.s.color.subsampling_x))
		y4 := max(t.mi_row, block_y << uint(t.s.color.subsampling_y))
		tx_type := t.tx_type_grid[((y4 & 15) << 4) | (x4 & 15)]
		if av1.TX_TYPE_IN_SET_INTER[tx_set][tx_type] == 0 {
			return av1.DCT_DCT
		}
		return tx_type
	}
	tx_type := av1.MODE_TO_TXFM[t.uv_mode]
	if av1.TX_TYPE_IN_SET_INTRA[tx_set][tx_type] == 0 {
		return av1.DCT_DCT
	}
	return tx_type
}

// Section 7.13.3: Inverse transform reconstruction.
@(private)
reconstruct :: #force_no_inline proc(t: ^Tile_State, plane, x, y, tx_sz, eob, max_r, max_c: int, dc_dq0: i32) #no_bounds_check {
	w := int(av1.TX_WIDTH[tx_sz])
	h := int(av1.TX_HEIGHT[tx_sz])
	bit_depth := int(t.s.color.bit_depth)

	cur_fb := t.d.cur
	stride := fb_stride(cur_fb, plane)
	max_val := i32((1 << uint(bit_depth)) - 1)
	v_zero := (#simd[4]i32)(0)
	v_max := (#simd[4]i32)(max_val)

	// Fast path for DC-only DCT_DCT blocks: avoid touching dequant_buf or running 2-D IDCT.
	if eob == 1 && int(t.plane_tx_type) == av1.DCT_DCT && !t.lossless {
		dc := av1.dc_only_residual(dc_dq0, tx_sz, bit_depth)
		if dc == 0 do return
		if bit_depth > 8 {
			frame_buf := fb_plane_u16(cur_fb, plane)
			when intrinsics.has_target_feature("avx512f") {
				if w >= 16 {
					v16_zero := (#simd[16]i32)(0)
					v16_max := (#simd[16]i32)(max_val)
					v16_dc := (#simd[16]i32)(dc)
					for i in 0 ..< h {
						row_off := (y + i) * stride + x
						for j := 0; j < w; j += 16 {
							p := (^#simd[16]u16)(&frame_buf[row_off + j])
							cur := (#simd[16]i32)(intrinsics.unaligned_load(p))
							out := (#simd[16]u16)(simd.clamp(cur + v16_dc, v16_zero, v16_max))
							intrinsics.unaligned_store(p, out)
						}
					}
					return
				}
			}
			if w >= 8 {
				v8_zero := (#simd[8]i32)(0)
				v8_max := (#simd[8]i32)(max_val)
				v8_dc := (#simd[8]i32)(dc)
				for i in 0 ..< h {
					row_off := (y + i) * stride + x
					for j := 0; j < w; j += 8 {
						p := (^#simd[8]u16)(&frame_buf[row_off + j])
						cur := (#simd[8]i32)(intrinsics.unaligned_load(p))
						out := (#simd[8]u16)(simd.clamp(cur + v8_dc, v8_zero, v8_max))
						intrinsics.unaligned_store(p, out)
					}
				}
			} else {
				v_dc := (#simd[4]i32)(dc)
				for i in 0 ..< h {
					row_off := (y + i) * stride + x
					for j := 0; j < w; j += 4 {
						p := (^#simd[4]u16)(&frame_buf[row_off + j])
						cur := (#simd[4]i32)(intrinsics.unaligned_load(p))
						out := (#simd[4]u16)(simd.clamp(cur + v_dc, v_zero, v_max))
						intrinsics.unaligned_store(p, out)
					}
				}
			}
		} else {
			frame_buf := fb_plane_u8(cur_fb, plane)
			when intrinsics.has_target_feature("avx512f") {
				if w >= 16 {
					v16_zero := (#simd[16]i32)(0)
					v16_max := (#simd[16]i32)(max_val)
					v16_dc := (#simd[16]i32)(dc)
					for i in 0 ..< h {
						row_off := (y + i) * stride + x
						for j := 0; j < w; j += 16 {
							p := (^#simd[16]u8)(&frame_buf[row_off + j])
							cur := (#simd[16]i32)(intrinsics.unaligned_load(p))
							out := (#simd[16]u8)(simd.clamp(cur + v16_dc, v16_zero, v16_max))
							intrinsics.unaligned_store(p, out)
						}
					}
					return
				}
			}
			if w >= 8 {
				v8_zero := (#simd[8]i32)(0)
				v8_max := (#simd[8]i32)(max_val)
				v8_dc := (#simd[8]i32)(dc)
				for i in 0 ..< h {
					row_off := (y + i) * stride + x
					for j := 0; j < w; j += 8 {
						p := (^#simd[8]u8)(&frame_buf[row_off + j])
						cur := (#simd[8]i32)(intrinsics.unaligned_load(p))
						out := (#simd[8]u8)(simd.clamp(cur + v8_dc, v8_zero, v8_max))
						intrinsics.unaligned_store(p, out)
					}
				}
			} else {
				v_dc := (#simd[4]i32)(dc)
				for i in 0 ..< h {
					row_off := (y + i) * stride + x
					for j := 0; j < w; j += 4 {
						p := (^#simd[4]u8)(&frame_buf[row_off + j])
						cur := (#simd[4]i32)(intrinsics.unaligned_load(p))
						out := (#simd[4]u8)(simd.clamp(cur + v_dc, v_zero, v_max))
						intrinsics.unaligned_store(p, out)
					}
				}
			}
		}
		return
	}

	if t.lossless {
		av1.inverse_transform_2d(
			t.dequant_buf[:w * h],
			tx_sz, int(t.plane_tx_type),
			true, eob, bit_depth,
			max_r, max_c,
		)
		if bit_depth > 8 {
			frame_buf := fb_plane_u16(cur_fb, plane)
			for i in 0 ..< 4 {
				row_off := (y + i) * stride + x
				dq_p := (^#simd[4]i32)(&t.dequant_buf[i * 4])
				res := intrinsics.unaligned_load(dq_p)
				intrinsics.unaligned_store(dq_p, v_zero)
				p := (^#simd[4]u16)(&frame_buf[row_off])
				cur := (#simd[4]i32)(intrinsics.unaligned_load(p))
				out := (#simd[4]u16)(simd.clamp(cur + res, v_zero, v_max))
				intrinsics.unaligned_store(p, out)
			}
		} else {
			frame_buf := fb_plane_u8(cur_fb, plane)
			for i in 0 ..< 4 {
				row_off := (y + i) * stride + x
				dq_p := (^#simd[4]i32)(&t.dequant_buf[i * 4])
				res := intrinsics.unaligned_load(dq_p)
				intrinsics.unaligned_store(dq_p, v_zero)
				p := (^#simd[4]u8)(&frame_buf[row_off])
				cur := (#simd[4]i32)(intrinsics.unaligned_load(p))
				out := (#simd[4]u8)(simd.clamp(cur + res, v_zero, v_max))
				intrinsics.unaligned_store(p, out)
			}
		}
		return
	}

	tx_type := int(t.plane_tx_type)
	av1.inverse_transform_2d_clear(
		t.itx_buf[:w * h],
		t.dequant_buf[:w * h],
		tx_sz, tx_type, bit_depth,
		max_r, max_c,
	)

	kinds := av1.TX_1D_TYPES[tx_type]
	recon_w := kinds[0] == .IDTX ? min(min(w, 32), (max_c + 4) & ~int(3)) : w
	recon_h := kinds[1] == .IDTX ? min(max_r + 1, h) : h
	dq_stride := (max_r == 0 && kinds[1] == .DCT) ? 0 : w

	if bit_depth > 8 {
		frame_buf := fb_plane_u16(cur_fb, plane)
		when intrinsics.has_target_feature("avx512f") {
			if recon_w >= 16 {
				v16_zero := (#simd[16]i32)(0)
				v16_max := (#simd[16]i32)(max_val)
				v8_zero := (#simd[8]i32)(0)
				v8_max := (#simd[8]i32)(max_val)
				recon_w16 := recon_w & ~int(15)
				for i in 0 ..< recon_h {
					row_off := (y + i) * stride + x
					dq_off := i * dq_stride
					for j := 0; j < recon_w16; j += 16 {
						res := intrinsics.unaligned_load((^#simd[16]i32)(&t.itx_buf[dq_off + j]))
						p := (^#simd[16]u16)(&frame_buf[row_off + j])
						cur := (#simd[16]i32)(intrinsics.unaligned_load(p))
						out := (#simd[16]u16)(simd.clamp(cur + res, v16_zero, v16_max))
						intrinsics.unaligned_store(p, out)
					}
					j := recon_w16
					if j + 8 <= recon_w {
						res := intrinsics.unaligned_load((^#simd[8]i32)(&t.itx_buf[dq_off + j]))
						p := (^#simd[8]u16)(&frame_buf[row_off + j])
						cur := (#simd[8]i32)(intrinsics.unaligned_load(p))
						out := (#simd[8]u16)(simd.clamp(cur + res, v8_zero, v8_max))
						intrinsics.unaligned_store(p, out)
						j += 8
					}
					if j < recon_w {
						res := intrinsics.unaligned_load((^#simd[4]i32)(&t.itx_buf[dq_off + j]))
						p := (^#simd[4]u16)(&frame_buf[row_off + j])
						cur := (#simd[4]i32)(intrinsics.unaligned_load(p))
						out := (#simd[4]u16)(simd.clamp(cur + res, v_zero, v_max))
						intrinsics.unaligned_store(p, out)
					}
				}
				return
			}
		}
		if recon_w >= 8 {
			v8_zero := (#simd[8]i32)(0)
			v8_max := (#simd[8]i32)(max_val)
			recon_w8 := recon_w & ~int(7)
			for i in 0 ..< recon_h {
				row_off := (y + i) * stride + x
				dq_off := i * dq_stride
				for j := 0; j < recon_w8; j += 8 {
					res := intrinsics.unaligned_load((^#simd[8]i32)(&t.itx_buf[dq_off + j]))
					p := (^#simd[8]u16)(&frame_buf[row_off + j])
					cur := (#simd[8]i32)(intrinsics.unaligned_load(p))
					out := (#simd[8]u16)(simd.clamp(cur + res, v8_zero, v8_max))
					intrinsics.unaligned_store(p, out)
				}
				if recon_w8 < recon_w {
					res := intrinsics.unaligned_load((^#simd[4]i32)(&t.itx_buf[dq_off + recon_w8]))
					p := (^#simd[4]u16)(&frame_buf[row_off + recon_w8])
					cur := (#simd[4]i32)(intrinsics.unaligned_load(p))
					out := (#simd[4]u16)(simd.clamp(cur + res, v_zero, v_max))
					intrinsics.unaligned_store(p, out)
				}
			}
		} else {
			for i in 0 ..< recon_h {
				row_off := (y + i) * stride + x
				dq_off := i * dq_stride
				res := intrinsics.unaligned_load((^#simd[4]i32)(&t.itx_buf[dq_off]))
				p := (^#simd[4]u16)(&frame_buf[row_off])
				cur := (#simd[4]i32)(intrinsics.unaligned_load(p))
				out := (#simd[4]u16)(simd.clamp(cur + res, v_zero, v_max))
				intrinsics.unaligned_store(p, out)
			}
		}
	} else {
		frame_buf := fb_plane_u8(cur_fb, plane)
		when intrinsics.has_target_feature("avx512f") {
			if recon_w >= 16 {
				v16_zero := (#simd[16]i32)(0)
				v16_max := (#simd[16]i32)(max_val)
				v8_zero := (#simd[8]i32)(0)
				v8_max := (#simd[8]i32)(max_val)
				recon_w16 := recon_w & ~int(15)
				for i in 0 ..< recon_h {
					row_off := (y + i) * stride + x
					dq_off := i * dq_stride
					for j := 0; j < recon_w16; j += 16 {
						res := intrinsics.unaligned_load((^#simd[16]i32)(&t.itx_buf[dq_off + j]))
						p := (^#simd[16]u8)(&frame_buf[row_off + j])
						cur := (#simd[16]i32)(intrinsics.unaligned_load(p))
						out := (#simd[16]u8)(simd.clamp(cur + res, v16_zero, v16_max))
						intrinsics.unaligned_store(p, out)
					}
					j := recon_w16
					if j + 8 <= recon_w {
						res := intrinsics.unaligned_load((^#simd[8]i32)(&t.itx_buf[dq_off + j]))
						p := (^#simd[8]u8)(&frame_buf[row_off + j])
						cur := (#simd[8]i32)(intrinsics.unaligned_load(p))
						out := (#simd[8]u8)(simd.clamp(cur + res, v8_zero, v8_max))
						intrinsics.unaligned_store(p, out)
						j += 8
					}
					if j < recon_w {
						res := intrinsics.unaligned_load((^#simd[4]i32)(&t.itx_buf[dq_off + j]))
						p := (^#simd[4]u8)(&frame_buf[row_off + j])
						cur := (#simd[4]i32)(intrinsics.unaligned_load(p))
						out := (#simd[4]u8)(simd.clamp(cur + res, v_zero, v_max))
						intrinsics.unaligned_store(p, out)
					}
				}
				return
			}
		}
		if recon_w >= 8 {
			v8_zero := (#simd[8]i32)(0)
			v8_max := (#simd[8]i32)(max_val)
			recon_w8 := recon_w & ~int(7)
			for i in 0 ..< recon_h {
				row_off := (y + i) * stride + x
				dq_off := i * dq_stride
				for j := 0; j < recon_w8; j += 8 {
					res := intrinsics.unaligned_load((^#simd[8]i32)(&t.itx_buf[dq_off + j]))
					p := (^#simd[8]u8)(&frame_buf[row_off + j])
					cur := (#simd[8]i32)(intrinsics.unaligned_load(p))
					out := (#simd[8]u8)(simd.clamp(cur + res, v8_zero, v8_max))
					intrinsics.unaligned_store(p, out)
				}
				if recon_w8 < recon_w {
					res := intrinsics.unaligned_load((^#simd[4]i32)(&t.itx_buf[dq_off + recon_w8]))
					p := (^#simd[4]u8)(&frame_buf[row_off + recon_w8])
					cur := (#simd[4]i32)(intrinsics.unaligned_load(p))
					out := (#simd[4]u8)(simd.clamp(cur + res, v_zero, v_max))
					intrinsics.unaligned_store(p, out)
				}
			}
		} else {
			for i in 0 ..< recon_h {
				row_off := (y + i) * stride + x
				dq_off := i * dq_stride
				res := intrinsics.unaligned_load((^#simd[4]i32)(&t.itx_buf[dq_off]))
				p := (^#simd[4]u8)(&frame_buf[row_off])
				cur := (#simd[4]i32)(intrinsics.unaligned_load(p))
				out := (#simd[4]u8)(simd.clamp(cur + res, v_zero, v_max))
				intrinsics.unaligned_store(p, out)
			}
		}
	}
}

