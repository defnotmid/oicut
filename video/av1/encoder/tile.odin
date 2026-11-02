package av1_encoder

// Tile encoding, partition RDO, intra/inter mode selection, motion search,
// forward transform/quantization, in-loop reconstruction, and normative AV1
// symbol emission (AV1 Specification sections 5.11, 7.11, 7.12, 7.13, 8.3).

import "base:intrinsics"
import "core:simd"
import av1 ".."

@(private)
Tile :: struct {
	e:            ^Encoder,
	cdfs:         av1.CDF_Context,
	sym:          av1.Symbol_Writer,
	grid:         av1.MI_Grid,
	tile_num:     int,
	mi_row_start: int,
	mi_row_end:   int,
	mi_col_start: int,
	mi_col_end:   int,

	// Tile-local above/left contexts (indexed by plane 4x4 coordinates).
	above_level_ctx:    [3][2048]u8,
	above_dc_ctx:       [3][2048]u8,
	left_level_ctx:     [3][64]u8,
	left_dc_ctx:        [3][64]u8,
	above_seg_pred_ctx: [2048]u8,
	left_seg_pred_ctx:  [64]u8,

	// Superblock BlockDecoded flags: [plane][y + 1][x + 1] for y, x in -1 ..= 32.
	block_decoded: [3][36][36]bool,

	// Current coding block state.
	mi_row, mi_col, mi_size: int,
	bw4, bh4:                int,
	has_chroma:              bool,
	avail_u, avail_l:        bool,
	avail_u_chroma:          bool,
	avail_l_chroma:          bool,
	segment_id:              u8,
	pred_seg_id:             u8,
	lossless:                bool,
	skip_mode:               u8,
	skip:                    bool,
	is_inter:                bool,
	ref_frame:               [2]i8,
	y_mode, uv_mode:         u8,
	angle_delta_y:           i8,
	angle_delta_uv:          i8,
	cfl_alpha_u:             i16,
	cfl_alpha_v:             i16,
	max_luma_w, max_luma_h:  int,
	tx_size:                 u8,
	plane_tx_type:           u8,
	mv:                      [2]av1.Mv,
	pred_mv:                 [2]av1.Mv,
	last_mv:                 av1.Mv,
	ref_mv_idx:              u8,
	mv_stack:                av1.MV_Stack,
	interp_filter:           [2]u8,

	// Per-coding-block transform coefficient storage (up to 64x64 per plane).
	blk_coeffs:  [3][64 * 64]i32,
	blk_eobs:    [3][256]u16,
	quant_buf:   [1024]i32,
	fcoeffs_buf: [64 * 64]i32,
	dequant_buf: [64 * 64]i32,
}

@(private)
encode_tile :: proc(t: ^Tile) {
	if t.e.bit_depth > 8 {
		encode_tile_typed(t, u16)
	} else {
		encode_tile_typed(t, u8)
	}
}

@(private)
put_symbol :: #force_inline proc(w: ^av1.Symbol_Writer, symbol: int, cdf: []u16) {
	av1.write_symbol(w, cdf, symbol)
}

@(private)
encode_tile_typed :: proc(t: ^Tile, $T: typeid) {
	e := t.e
	h := &e.header
	cells := h.mi_rows * h.mi_cols
	t.grid = av1.MI_Grid{
		cells        = e.mi_cells[:cells],
		mi_rows      = h.mi_rows,
		mi_cols      = h.mi_cols,
		mi_row_start = t.mi_row_start,
		mi_row_end   = t.mi_row_end,
		mi_col_start = t.mi_col_start,
		mi_col_end   = t.mi_col_end,
	}

	av1.start_symbol_writer(&t.sym, h.disable_cdf_update)
	t.last_mv = {}
	clear_above_context(t)
	stage_source(t, T)

	sb_size :: av1.BLOCK_64X64
	sb_size4 :: 16
	for r := t.mi_row_start; r < t.mi_row_end; r += sb_size4 {
		clear_left_context(t)
		for c := t.mi_col_start; c < t.mi_col_end; c += sb_size4 {
			clear_cdef(t, r, c)
			clear_block_decoded_flags(t, r, c, sb_size4)
			encode_partition(t, r, c, sb_size, T)
		}
	}

	av1.finish_symbol_writer(&t.sym)
}

@(private)
clear_above_context :: proc(t: ^Tile) {
	t.above_level_ctx = {}
	t.above_dc_ctx = {}
	t.above_seg_pred_ctx = {}
}

@(private)
clear_left_context :: proc(t: ^Tile) {
	t.left_level_ctx = {}
	t.left_dc_ctx = {}
	t.left_seg_pred_ctx = {}
}

@(private)
clear_cdef :: proc(t: ^Tile, r, c: int) {
	t.e.cdef_idx[(r >> 4) * t.e.sb_cols + (c >> 4)] = -1
}

@(private)
clear_block_decoded_flags :: proc(t: ^Tile, r, c, sb_size4: int) {
	e := t.e
	num_planes := e.mono_chrome ? 1 : 3
	for plane in 0 ..< num_planes {
		sub_x := plane > 0 ? uint(e.ss_x) : 0
		sub_y := plane > 0 ? uint(e.ss_y) : 0
		sb_w4 := (t.mi_col_end - c) >> sub_x
		sb_h4 := (t.mi_row_end - r) >> sub_y
		lim_y := sb_size4 >> sub_y
		lim_x := sb_size4 >> sub_x
		for y := -1; y <= lim_y; y += 1 {
			for x := -1; x <= lim_x; x += 1 {
				val := false
				if y < 0 && x < sb_w4 {
					val = true
				} else if x < 0 && y < sb_h4 {
					val = true
				}
				t.block_decoded[plane][y + 1][x + 1] = val
			}
		}
		t.block_decoded[plane][lim_y + 1][0] = false
	}
}

@(private)
stage_source :: proc(t: ^Tile, $T: typeid) #no_bounds_check {
	e := t.e
	num_planes := e.mono_chrome ? 1 : 3
	for p in 0 ..< num_planes {
		sx := p > 0 ? uint(e.ss_x) : 0
		sy := p > 0 ? uint(e.ss_y) : 0
		w := (e.config.width + int(sx)) >> sx
		h := (e.config.height + int(sy)) >> sy
		aw := e.aligned_w[p]
		ah := e.aligned_h[p]
		in_s := e.input_strides[p]
		src_s := e.source_strides[p]
		inp := plane_t(e.input[p], T)
		src := plane_t(e.source[p], T)
		y0 := (t.mi_row_start * av1.MI_SIZE) >> sy
		y1 := min(ah, ((t.mi_row_end + 15) & ~int(15)) * av1.MI_SIZE >> sy)
		x0 := (t.mi_col_start * av1.MI_SIZE) >> sx
		x1 := min(aw, ((t.mi_col_end + 15) & ~int(15)) * av1.MI_SIZE >> sx)
		for y in y0 ..< y1 {
			iy := min(y, h - 1)
			in_row := inp[iy * in_s:]
			dst_row := src[y * src_s:]
			if x0 < w {
				cw := min(w, x1) - x0
				copy(dst_row[x0:x0 + cw], in_row[x0:x0 + cw])
			}
			last := in_row[w - 1]
			for x in max(w, x0) ..< x1 {
				dst_row[x] = last
			}
		}
	}
}

@(private)
partition_cdf :: proc(cdfs: ^av1.CDF_Context, bsl, ctx: int) -> []u16 {
	switch bsl {
	case 1: return cdfs.partition_w8_cdf[ctx][:]
	case 2: return cdfs.partition_w16_cdf[ctx][:]
	case 3: return cdfs.partition_w32_cdf[ctx][:]
	case 4: return cdfs.partition_w64_cdf[ctx][:]
	case:   return cdfs.partition_w128_cdf[ctx][:]
	}
}

@(private)
block_size_for_log2 :: proc "contextless" (log2: int) -> int {
	switch log2 {
	case 2: return av1.BLOCK_4X4
	case 3: return av1.BLOCK_8X8
	case 4: return av1.BLOCK_16X16
	case 5: return av1.BLOCK_32X32
	case:   return av1.BLOCK_64X64
	}
}

@(private)
should_split :: proc(t: ^Tile, r, c, b_size: int, $T: typeid) -> bool #no_bounds_check {
	e := t.e
	h := &e.header
	if b_size <= av1.BLOCK_4X4 do return false

	w4 := int(av1.NUM_4X4_BLOCKS_WIDE[b_size])
	h4 := int(av1.NUM_4X4_BLOCKS_HIGH[b_size])
	if r + h4 > h.mi_rows || c + w4 > h.mi_cols {
		return true
	}

	target_bsize := block_size_for_log2(e.config.block_log2)
	if b_size > target_bsize {
		return true
	}

	if h.segmentation_enabled && h.segmentation_update_map {
		seg0 := e.segment_ids[r * h.mi_cols + c]
		for dy in 0 ..< h4 {
			for dx in 0 ..< w4 {
				if e.segment_ids[(r + dy) * h.mi_cols + (c + dx)] != seg0 {
					return true
				}
			}
		}
	}

	if e.config.block_log2 < 6 || e.lossless || b_size <= av1.BLOCK_8X8 {
		return false
	}

	if e.config.ultrafast && b_size <= av1.BLOCK_16X16 {
		return false
	}

	px := c * av1.MI_SIZE
	py := r * av1.MI_SIZE
	pw := w4 * av1.MI_SIZE
	ph := h4 * av1.MI_SIZE
	hw := pw >> 1
	hh := ph >> 1
	s := e.source_strides[0]
	src := plane_t(e.source[0], T)[py * s + px:]

	if !e.keyframe {
		rs := e.strides[0]
		ref_plane := plane_t(e.reference[0], T)
		ref := ref_plane[py * rs + px:]
		s00 := block_sad(src, s, ref, rs, hw, hh)
		cand_mvs := [2]av1.Mv{t.last_mv, {0, 0}}
		num_cands := 1
		if h.use_ref_frame_mvs {
			tpl_mv := e.motion_field_mvs[av1.LAST_FRAME][(r >> 1) * (h.mi_cols >> 1) + (c >> 1)]
			if tpl_mv[0] != -32768 && tpl_mv != t.last_mv {
				cand_mvs[1] = tpl_mv
				num_cands = 2
			}
		}
		for ci in 0 ..< num_cands {
			dy := int(cand_mvs[ci][0]) >> 3
			dx := int(cand_mvs[ci][1]) >> 3
			if (dx != 0 || dy != 0) && px + dx >= 0 && py + dy >= 0 && px + dx + pw <= e.aligned_w[0] && py + dy + ph <= e.aligned_h[0] {
				mref := ref_plane[(py + dy) * rs + (px + dx):]
				ms00 := block_sad(src, s, mref, rs, hw, hh)
				if ms00 < s00 {
					ref = mref
					s00 = ms00
				}
			}
		}
		s01 := block_sad(src[hw:], s, ref[hw:], rs, hw, hh)
		s10 := block_sad(src[hh * s:], s, ref[hh * rs:], rs, hw, hh)
		s11 := block_sad(src[hh * s + hw:], s, ref[hh * rs + hw:], rs, hw, hh)
		total := s00 + s01 + s10 + s11
		if e.lossless do return total > 0
		min_s := min(min(s00, s01), min(s10, s11))
		max_s := max(max(s00, s01), max(s10, s11))
		q := u64(av1.ac_quant(e.q_idx, 0, e.bit_depth))
		area := u64(hw * hh)
		noise := (area * q) >> 5
		if b_size == av1.BLOCK_16X16 {
			return max_s > min_s * 2 + noise && max_s > noise * 3
		}
		if b_size == av1.BLOCK_64X64 {
			return max_s > min_s + (noise >> 2) || max_s > noise
		}
		return max_s > min_s + (noise >> 1) || max_s > noise * 2
	}

	p0 := src
	p1 := src[(hh - 2) * s + hw - 2:]
	v0 := block_sad(p0, s, src[2 * s:], s, hw, hh)
	v1 := block_sad(p1, s, src[hh * s + hw - 2:], s, hw, hh)
	h0 := block_sad(p0, s, src[2:], s, hw, hh)
	h1 := block_sad(p1, s, src[(hh - 2) * s + hw:], s, hw, hh)
	v_diff := v0 + v1
	h_diff := h0 + h1
	if b_size > av1.BLOCK_32X32 {
		p2 := src[hw - 2:]
		p3 := src[(hh - 2) * s:]
		v2 := block_sad(p2, s, src[2 * s + hw - 2:], s, hw, hh)
		v3 := block_sad(p3, s, src[hh * s:], s, hw, hh)
		h2 := block_sad(p2, s, src[hw:], s, hw, hh)
		h3 := block_sad(p3, s, src[(hh - 2) * s + 2:], s, hw, hh)
		v_diff += v2 + v3
		h_diff += h2 + h3
		q0, q1, q2, q3 := v0 + h0, v1 + h1, v2 + h2, v3 + h3
		max_q := max(max(q0, q1), max(q2, q3))
		min_q := min(min(q0, q1), min(q2, q3))
		return min(v_diff, h_diff) > 0 || max_q > min_q * 2
	}
	area := u64(hw * hh) << uint(e.bit_depth - 8)
	return min(v_diff, h_diff) > area / 2 || max(v0 + h0, v1 + h1) > min(v0 + h0, v1 + h1) * 3 + area
}

// Section 5.11.6: Encode partition syntax.
@(private)
encode_partition :: proc(t: ^Tile, r, c, b_size: int, $T: typeid) {
	h := &t.e.header
	if r >= h.mi_rows || c >= h.mi_cols do return

	avail_u := av1.is_inside_tile(&t.grid, r - 1, c)
	avail_l := av1.is_inside_tile(&t.grid, r, c - 1)
	num4x4 := int(av1.NUM_4X4_BLOCKS_WIDE[b_size])
	half_block4x4 := num4x4 >> 1
	has_rows := (r + half_block4x4) < h.mi_rows
	has_cols := (c + half_block4x4) < h.mi_cols

	bsl := int(av1.MI_WIDTH_LOG2[b_size])
	above := avail_u && int(av1.MI_WIDTH_LOG2[t.grid.cells[(r - 1) * t.grid.mi_cols + c].mi_size]) < bsl
	left := avail_l && int(av1.MI_HEIGHT_LOG2[t.grid.cells[r * t.grid.mi_cols + (c - 1)].mi_size]) < bsl
	ctx := int(left) * 2 + int(above)

	partition: int
	if b_size < av1.BLOCK_8X8 {
		partition = av1.PARTITION_NONE
	} else if has_rows && has_cols {
		split := should_split(t, r, c, b_size, T)
		partition = split ? av1.PARTITION_SPLIT : av1.PARTITION_NONE
		put_symbol(&t.sym, partition, partition_cdf(&t.cdfs, bsl, ctx))
	} else if has_cols {
		pcdf := partition_cdf(&t.cdfs, bsl, ctx)
		psum := (pcdf[av1.PARTITION_VERT] - pcdf[av1.PARTITION_VERT - 1]) +
			(pcdf[av1.PARTITION_SPLIT] - pcdf[av1.PARTITION_SPLIT - 1]) +
			(pcdf[av1.PARTITION_HORZ_A] - pcdf[av1.PARTITION_HORZ_A - 1]) +
			(pcdf[av1.PARTITION_VERT_A] - pcdf[av1.PARTITION_VERT_A - 1]) +
			(pcdf[av1.PARTITION_VERT_B] - pcdf[av1.PARTITION_VERT_B - 1])
		if b_size != av1.BLOCK_128X128 {
			psum += pcdf[av1.PARTITION_VERT_4] - pcdf[av1.PARTITION_VERT_4 - 1]
		}
		cdf := [3]u16{u16(32768 - int(psum)), 32768, 0}
		put_symbol(&t.sym, 1, cdf[:])
		partition = av1.PARTITION_SPLIT
	} else if has_rows {
		pcdf := partition_cdf(&t.cdfs, bsl, ctx)
		psum := (pcdf[av1.PARTITION_HORZ] - pcdf[av1.PARTITION_HORZ - 1]) +
			(pcdf[av1.PARTITION_SPLIT] - pcdf[av1.PARTITION_SPLIT - 1]) +
			(pcdf[av1.PARTITION_HORZ_A] - pcdf[av1.PARTITION_HORZ_A - 1]) +
			(pcdf[av1.PARTITION_HORZ_B] - pcdf[av1.PARTITION_HORZ_B - 1]) +
			(pcdf[av1.PARTITION_VERT_A] - pcdf[av1.PARTITION_VERT_A - 1])
		if b_size != av1.BLOCK_128X128 {
			psum += pcdf[av1.PARTITION_HORZ_4] - pcdf[av1.PARTITION_HORZ_4 - 1]
		}
		cdf := [3]u16{u16(32768 - int(psum)), 32768, 0}
		put_symbol(&t.sym, 1, cdf[:])
		partition = av1.PARTITION_SPLIT
	} else {
		partition = av1.PARTITION_SPLIT
	}

	sub_size := int(av1.PARTITION_SUBSIZE[partition][b_size])
	if partition == av1.PARTITION_SPLIT {
		encode_partition(t, r, c, sub_size, T)
		encode_partition(t, r, c + half_block4x4, sub_size, T)
		encode_partition(t, r + half_block4x4, c, sub_size, T)
		encode_partition(t, r + half_block4x4, c + half_block4x4, sub_size, T)
	} else {
		encode_block(t, r, c, sub_size, T)
	}
}

@(private)
seg_feature_active :: #force_inline proc "contextless" (t: ^Tile, feature: int) -> bool {
	h := &t.e.header
	return h.segmentation_enabled && (h.feature_enabled[t.segment_id] & (u8(1) << uint(feature))) != 0
}

@(private)
get_prev_segment_id :: proc(t: ^Tile) -> u8 #no_bounds_check {
	e := t.e
	h := &e.header
	bw4 := min(h.mi_cols - t.mi_col, t.bw4)
	bh4 := min(h.mi_rows - t.mi_row, t.bh4)
	seg := u8(7)
	for y in 0 ..< bh4 {
		for x in 0 ..< bw4 {
			prev_seg := e.prev_segment_ids[(t.mi_row + y) * h.mi_cols + (t.mi_col + x)]
			seg = min(seg, prev_seg)
		}
	}
	return seg
}

@(private)
get_spatial_seg_pred :: proc(t: ^Tile) -> (pred, ctx: int) #no_bounds_check {
	e := t.e
	h := &e.header
	prev_ul := t.avail_u && t.avail_l ? int(e.segment_ids[(t.mi_row - 1) * h.mi_cols + (t.mi_col - 1)]) : -1
	prev_u := t.avail_u ? int(e.segment_ids[(t.mi_row - 1) * h.mi_cols + t.mi_col]) : -1
	prev_l := t.avail_l ? int(e.segment_ids[t.mi_row * h.mi_cols + (t.mi_col - 1)]) : -1

	if prev_u == -1 {
		pred = prev_l == -1 ? 0 : prev_l
	} else if prev_l == -1 {
		pred = prev_u
	} else {
		pred = prev_ul == prev_u ? prev_u : prev_l
	}

	if prev_ul < 0 {
		ctx = 0
	} else if prev_ul == prev_u && prev_ul == prev_l {
		ctx = 2
	} else if prev_ul == prev_u || prev_ul == prev_l || prev_u == prev_l {
		ctx = 1
	} else {
		ctx = 0
	}
	return
}

@(private)
neg_interleave :: #force_inline proc "contextless" (val, ref, mx: int) -> int {
	if ref == 0 do return val
	if ref >= mx - 1 do return mx - val - 1
	diff := val - ref
	if 2 * ref < mx {
		if abs(diff) <= ref {
			return diff > 0 ? (2 * diff - 1) : (-2 * diff)
		}
		return val
	} else {
		if abs(diff) <= mx - ref - 1 {
			return diff > 0 ? (2 * diff - 1) : (-2 * diff)
		}
		return mx - (val + 1)
	}
}

@(private)
get_qindex :: proc "contextless" (h: ^av1.Frame_Header, segment_id: int) -> int {
	qindex := int(h.base_q_idx)
	if h.segmentation_enabled && (h.feature_enabled[segment_id] & (1 << av1.SEG_LVL_ALT_Q)) != 0 {
		qindex += int(h.feature_data[segment_id][av1.SEG_LVL_ALT_Q])
	}
	return clamp(qindex, 0, 255)
}

@(private)
is_cfl_allowed :: #force_inline proc "contextless" (t: ^Tile) -> bool {
	if t.lossless {
		return av1.get_plane_residual_size(t.mi_size, 1, uint(t.e.ss_x), uint(t.e.ss_y)) == av1.BLOCK_4X4
	}
	return max(av1.BLOCK_WIDTH[t.mi_size], av1.BLOCK_HEIGHT[t.mi_size]) <= 32
}

// Section 5.11.7: Encode coding block.
@(private)
encode_block :: proc(t: ^Tile, r, c, sub_size: int, $T: typeid) {
	e := t.e
	h := &e.header

	t.mi_row = r
	t.mi_col = c
	t.mi_size = sub_size
	t.bw4 = int(av1.NUM_4X4_BLOCKS_WIDE[sub_size])
	t.bh4 = int(av1.NUM_4X4_BLOCKS_HIGH[sub_size])

	sub_x := e.ss_x
	sub_y := e.ss_y
	if t.bh4 == 1 && sub_y != 0 && (r & 1) == 0 {
		t.has_chroma = false
	} else if t.bw4 == 1 && sub_x != 0 && (c & 1) == 0 {
		t.has_chroma = false
	} else {
		t.has_chroma = !e.mono_chrome
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

	// Determine segment_id and lossless status for this block.
	if !h.segmentation_enabled {
		t.segment_id = 0
		t.pred_seg_id = 0
	} else {
		t.pred_seg_id = get_prev_segment_id(t)
		if !h.segmentation_update_map && !h.frame_is_intra {
			t.segment_id = t.pred_seg_id
		} else {
			t.segment_id = clamp(e.segment_ids[r * h.mi_cols + c], 0, u8(e.last_active_seg_id))
		}
	}
	t.lossless = e.lossless_array[t.segment_id]
	t.skip_mode = 0
	t.angle_delta_y = 0
	t.angle_delta_uv = 0
	t.cfl_alpha_u = 0
	t.cfl_alpha_v = 0
	t.mv = {}
	t.pred_mv = {}
	t.ref_mv_idx = 0
	t.interp_filter = {h.interpolation_filter == av1.SWITCHABLE ? av1.EIGHTTAP_REGULAR : h.interpolation_filter, h.interpolation_filter == av1.SWITCHABLE ? av1.EIGHTTAP_REGULAR : h.interpolation_filter}

	seg_skip := seg_feature_active(t, av1.SEG_LVL_SKIP)
	seg_globalmv := seg_feature_active(t, av1.SEG_LVL_GLOBALMV)
	seg_has_ref := seg_feature_active(t, av1.SEG_LVL_REF_FRAME)
	seg_ref := seg_has_ref ? i8(h.feature_data[t.segment_id][av1.SEG_LVL_REF_FRAME]) : i8(-1)

	// Select default transform size prior to intra/inter mode search.
	if t.lossless {
		t.tx_size = av1.TX_4X4
	} else if t.mi_size == av1.BLOCK_64X64 {
		t.tx_size = av1.TX_32X32
	} else if t.mi_size == av1.BLOCK_8X8 && e.keyframe {
		t.tx_size = av1.TX_4X4
	} else {
		t.tx_size = u8(av1.MAX_TX_SIZE_RECT[t.mi_size])
	}

	// Select prediction mode (intra vs inter) and motion vectors.
	select_prediction_mode(t, seg_skip, seg_globalmv, seg_has_ref, seg_ref, T)

	if !t.lossless && !t.is_inter {
		if t.mi_size == av1.BLOCK_64X64 {
			t.tx_size = av1.TX_32X32
		} else if t.mi_size == av1.BLOCK_8X8 && e.keyframe {
			t.tx_size = av1.TX_4X4
		} else {
			t.tx_size = u8(av1.MAX_TX_SIZE_RECT[t.mi_size])
		}
	}

	// Populate grid cells prior to inter/intra prediction and transform.
	is_compound := t.ref_frame[1] > av1.INTRA_FRAME
	populate_block_grids(t, is_compound)

	// Compute inter prediction into `e.planes` if inter.
	if t.is_inter {
		compute_inter_prediction(t, T)
		if !t.lossless && t.mi_size > av1.BLOCK_4X4 {
			if t.mi_size == av1.BLOCK_64X64 {
				t.tx_size = av1.TX_32X32
			} else {
				px := c * av1.MI_SIZE
				py := r * av1.MI_SIZE
				hw := (t.bw4 * av1.MI_SIZE) >> 1
				hh := (t.bh4 * av1.MI_SIZE) >> 1
				ss := e.source_strides[0]
				ps := e.strides[0]
				src := plane_t(e.source[0], T)[py * ss + px:]
				prd := plane_t(e.planes[0], T)[py * ps + px:]
				s00 := block_sad(src, ss, prd, ps, hw, hh)
				s01 := block_sad(src[hw:], ss, prd[hw:], ps, hw, hh)
				s10 := block_sad(src[hh * ss:], ss, prd[hh * ps:], ps, hw, hh)
				s11 := block_sad(src[hh * ss + hw:], ss, prd[hh * ps + hw:], ps, hw, hh)
				min_s := min(min(s00, s01), min(s10, s11))
				max_s := max(max(s00, s01), max(s10, s11))
				q := u64(av1.ac_quant(e.q_idx, 0, e.bit_depth))
				noise := (u64(hw * hh) * q) >> 5
				max_tx := u8(av1.MAX_TX_SIZE_RECT[t.mi_size])
				t.tx_size = (max_s > min_s * 2 + noise) ? av1.SPLIT_TX_SIZE[max_tx] : max_tx
			}
			for y in 0 ..< t.bh4 {
				row := t.mi_row + y
				if row >= h.mi_rows do break
				row_base := row * h.mi_cols
				for x in 0 ..< t.bw4 {
					col := t.mi_col + x
					if col >= h.mi_cols do break
					idx := row_base + col
					e.tile_cells[idx].inter_tx_size = t.tx_size
					e.lf_cells[idx].tx_size_y = t.tx_size
				}
			}
		}
	}

	// Forward transform, quantize, and reconstruct in-loop into `e.planes`.
	any_nonzero := transform_and_reconstruct_block(t, seg_skip, T)

	// Determine whether `t.skip` can be signaled.
	spatial_pred, _ := get_spatial_seg_pred(t)
	can_skip_seg := !(h.segmentation_enabled && h.segmentation_update_map && !e.seg_id_pre_skip && int(t.segment_id) != spatial_pred)
	t.skip = seg_skip || (!any_nonzero && can_skip_seg)
	if t.skip && t.is_inter && !t.lossless {
		t.tx_size = u8(av1.MAX_TX_SIZE_RECT[t.mi_size])
	}

	// Emit block mode info, transform size, context updates, and residuals.
	write_mode_info(t)
	write_block_tx_size(t)
	if t.skip {
		reset_block_context(t)
		for y in 0 ..< t.bh4 {
			row := t.mi_row + y
			if row >= h.mi_rows do break
			row_base := row * h.mi_cols
			for x in 0 ..< t.bw4 {
				col := t.mi_col + x
				if col >= h.mi_cols do break
				idx := row_base + col
				e.tile_cells[idx].skip = true
				e.lf_cells[idx].skip_inter = t.is_inter
				e.lf_cells[idx].tx_size_y = t.tx_size
			}
		}
	} else {
		write_residual(t)
	}
}

@(private)
populate_block_grids :: proc(t: ^Tile, is_compound: bool) #no_bounds_check {
	e := t.e
	h := &e.header
	delta_lf: [av1.FRAME_LF_COUNT]i8
	levels := av1.compute_lf_levels(h, t.segment_id, t.ref_frame[0], t.y_mode, delta_lf)
	uv_tx := u8(av1.get_uv_tx_size(t.mi_size, 1, int(t.tx_size), t.lossless, u8(e.ss_x), u8(e.ss_y)))
	for y in 0 ..< t.bh4 {
		row := t.mi_row + y
		if row >= h.mi_rows do break
		row_base := row * h.mi_cols
		for x in 0 ..< t.bw4 {
			col := t.mi_col + x
			if col >= h.mi_cols do break
			idx := row_base + col
			mvs := t.mv
			if !is_compound {
				mvs[1] = {0, 0}
			}
			e.mi_cells[idx] = {
				mvs        = mvs,
				ref_frames = t.ref_frame,
				mi_size    = u8(t.mi_size),
				y_mode     = t.y_mode,
				is_inter   = t.is_inter,
			}
			e.segment_ids[idx] = t.segment_id
			e.tile_cells[idx].skip = false
			e.tile_cells[idx].skip_mode = t.skip_mode
			e.tile_cells[idx].uv_mode = t.uv_mode
			e.tile_cells[idx].inter_tx_size = t.tx_size
			e.tile_cells[idx].comp_group_idx = 0
			e.tile_cells[idx].compound_idx = 1
			e.tile_cells[idx].interp_filters = t.interp_filter
			e.lf_cells[idx].levels = levels
			e.lf_cells[idx].mi_size = u8(t.mi_size)
			e.lf_cells[idx].tx_size_y = t.tx_size
			e.lf_cells[idx].tx_size_uv = uv_tx
			e.lf_cells[idx].skip_inter = false
		}
	}
}

@(private)
reset_block_context :: proc(t: ^Tile) {
	e := t.e
	num_planes := 1 + (t.has_chroma ? 2 : 0)
	for plane in 0 ..< num_planes {
		sub_x := plane > 0 ? uint(e.ss_x) : 0
		sub_y := plane > 0 ? uint(e.ss_y) : 0
		tile_x4 := t.mi_col_start >> sub_x
		for i := t.mi_col >> sub_x; i < ((t.mi_col + t.bw4) >> sub_x); i += 1 {
			t.above_level_ctx[plane][i - tile_x4] = 0
			t.above_dc_ctx[plane][i - tile_x4] = 0
		}
		for i := t.mi_row >> sub_y; i < ((t.mi_row + t.bh4) >> sub_y); i += 1 {
			t.left_level_ctx[plane][i & 63] = 0
			t.left_dc_ctx[plane][i & 63] = 0
		}
	}
}

// Section 7.11.2.8: Intra edge filter type from neighbor smooth modes.
@(private)
get_filter_type :: #force_inline proc "contextless" (t: ^Tile, plane: int) -> int #no_bounds_check {
	e := t.e
	h := &e.header
	above_smooth := false
	left_smooth := false
	avail_u := plane == 0 ? t.avail_u : t.avail_u_chroma
	avail_l := plane == 0 ? t.avail_l : t.avail_l_chroma
	if avail_u {
		r := (t.mi_row & ~(plane > 0 ? e.ss_y : 0)) - 1
		c := t.mi_col | (plane > 0 ? e.ss_x : 0)
		idx := r * h.mi_cols + c
		if t.grid.cells[idx].ref_frames[0] == av1.INTRA_FRAME {
			m := plane == 0 ? t.grid.cells[idx].y_mode : e.tile_cells[idx].uv_mode
			above_smooth = av1.is_smooth_mode(m)
		}
	}
	if avail_l {
		r := t.mi_row | (plane > 0 ? e.ss_y : 0)
		c := (t.mi_col & ~(plane > 0 ? e.ss_x : 0)) - 1
		idx := r * h.mi_cols + c
		if t.grid.cells[idx].ref_frames[0] == av1.INTRA_FRAME {
			m := plane == 0 ? t.grid.cells[idx].y_mode : e.tile_cells[idx].uv_mode
			left_smooth = av1.is_smooth_mode(m)
		}
	}
	return int(above_smooth || left_smooth)
}

@(private)
block_sad :: proc "contextless" (a: []$T, a_stride: int, b: []T, b_stride: int, w, h: int) -> u64 #no_bounds_check {
	switch w {
	case 4:  return sad_rows(4, h, a, a_stride, b, b_stride)
	case 8:  return sad_rows(8, h, a, a_stride, b, b_stride)
	case 16: return sad_rows(16, h, a, a_stride, b, b_stride)
	case 32: return sad_rows(32, h, a, a_stride, b, b_stride)
	case 64: return sad_rows(64, h, a, a_stride, b, b_stride)
	}
	sad := u64(0)
	for r in 0 ..< h {
		ar := a[r * a_stride:]
		br := b[r * b_stride:]
		for c in 0 ..< w {
			sad += u64(abs(int(ar[c]) - int(br[c])))
		}
	}
	return sad
}

@(private)
sad_rows :: #force_inline proc "contextless" ($W: int, h: int, a: []$T, a_stride: int, b: []T, b_stride: int) -> u64 #no_bounds_check {
	when T == u8 {
		total: #simd[W]u16
		for row in 0 ..< h {
			x := intrinsics.unaligned_load((^#simd[W]u8)(&a[row * a_stride]))
			y := intrinsics.unaligned_load((^#simd[W]u8)(&b[row * b_stride]))
			total += (#simd[W]u16)(simd.max(x, y) - simd.min(x, y))
		}
		return u64(simd.reduce_add_ordered((#simd[W]u32)(total)))
	} else {
		L :: min(W, 8)
		total: #simd[L]u32
		for row in 0 ..< h {
			ra, rb := a[row * a_stride:], b[row * b_stride:]
			#unroll for i in 0 ..< W / L {
				col := i * L
				x := intrinsics.unaligned_load((^#simd[L]u16)(&ra[col]))
				y := intrinsics.unaligned_load((^#simd[L]u16)(&rb[col]))
				total += (#simd[L]u32)(simd.max(x, y) - simd.min(x, y))
			}
		}
		return u64(simd.reduce_add_ordered(total))
	}
}

@(private)
choose_intra_y :: proc(t: ^Tile, $T: typeid) -> (best_mode: u8, best_angle: i8, best_sad: u64) #no_bounds_check {
	e := t.e
	h := &e.header
	s := &e.seq
	px := t.mi_col * av1.MI_SIZE
	py := t.mi_row * av1.MI_SIZE
	bw := t.bw4 * 4
	bh := t.bh4 * 4
	tx_sz := int(t.tx_size)
	tw := int(av1.TX_WIDTH[tx_sz])
	th := int(av1.TX_HEIGHT[tx_sz])
	step_x := tw >> 2
	step_y := th >> 2
	max_x := h.mi_cols * av1.MI_SIZE - 1
	max_y := h.mi_rows * av1.MI_SIZE - 1
	sub_row := t.mi_row & 15
	sub_col := t.mi_col & 15
	filter_type := get_filter_type(t, 0)

	recon := plane_t(e.planes[0], T)
	r_stride := e.strides[0]
	src_plane := plane_t(e.source[0], T)
	s_stride := e.source_strides[0]

	y_cdf: []u16
	if h.frame_is_intra {
		above_mode := av1.INTRA_MODE_CONTEXT[t.avail_u ? t.grid.cells[(t.mi_row - 1) * t.grid.mi_cols + t.mi_col].y_mode : av1.DC_PRED]
		left_mode := av1.INTRA_MODE_CONTEXT[t.avail_l ? t.grid.cells[t.mi_row * t.grid.mi_cols + (t.mi_col - 1)].y_mode : av1.DC_PRED]
		y_cdf = t.cdfs.intra_frame_y_mode_cdf[above_mode][left_mode][:]
	} else {
		y_cdf = t.cdfs.y_mode_cdf[av1.SIZE_GROUP[t.mi_size]][:]
	}
	q := u64(max(4, int(av1.ac_quant(e.q_idx, 0, e.bit_depth))))

	fast_modes := [4]u8{av1.DC_PRED, av1.V_PRED, av1.H_PRED, av1.SMOOTH_PRED}
	num_modes := e.config.ultrafast ? 4 : av1.INTRA_MODES

	best_mode = av1.DC_PRED
	best_angle = 0
	best_sad = max(u64)
	best_cost := max(u64)

	for idx in 0 ..< num_modes {
		mode := e.config.ultrafast ? fast_modes[idx] : u8(idx)
		bits := u64(av1.symbol_cost(y_cdf, int(mode)))
		if t.mi_size >= av1.BLOCK_8X8 && av1.is_directional_mode(int(mode)) {
			bits += u64(av1.symbol_cost(t.cdfs.angle_delta_cdf[mode - av1.V_PRED][:], av1.MAX_ANGLE_DELTA))
		}
		rate_cost := (q * bits) >> 11
		sad := u64(0)
		if tw < bw || th < bh {
			for dy := 0; dy < bh; dy += th {
				for dx := 0; dx < bw; dx += tw {
					sx, sy := px + dx, py + dy
					if sx > max_x || sy > max_y do continue
					x4, y4 := dx >> 2, dy >> 2
					bd_tr := dy == 0 ? t.block_decoded[0][sub_row][sub_col + x4 + step_x + 1] : (dx + tw < bw)
					bd_bl := dx == 0 ? t.block_decoded[0][sub_row + y4 + step_y + 1][sub_col] : false
					edges := av1.Intra_Edges{
						have_left                = t.avail_l || dx > 0,
						have_above               = t.avail_u || dy > 0,
						have_above_right         = bd_tr,
						have_below_left          = bd_bl,
						max_x                    = max_x,
						max_y                    = max_y,
						enable_intra_edge_filter = s.enable_intra_edge_filter,
						filter_type              = filter_type,
						angle_delta              = 0,
					}
					av1.predict_intra(recon, r_stride, sx, sy, tx_sz, int(mode), edges, e.bit_depth)
					sad += block_sad(src_plane[sy * s_stride + sx:], s_stride, recon[sy * r_stride + sx:], r_stride, tw, th)
					for yy in 0 ..< th {
						copy(recon[(sy + yy) * r_stride + sx:][:tw], src_plane[(sy + yy) * s_stride + sx:][:tw])
					}
				}
			}
		} else {
			bd_tr := t.block_decoded[0][sub_row][sub_col + t.bw4 + 1]
			bd_bl := t.block_decoded[0][sub_row + t.bh4 + 1][sub_col]
			edges := av1.Intra_Edges{
				have_left                = t.avail_l,
				have_above               = t.avail_u,
				have_above_right         = bd_tr,
				have_below_left          = bd_bl,
				max_x                    = max_x,
				max_y                    = max_y,
				enable_intra_edge_filter = s.enable_intra_edge_filter,
				filter_type              = filter_type,
				angle_delta              = 0,
			}
			av1.predict_intra(recon, r_stride, px, py, tx_sz, int(mode), edges, e.bit_depth)
			sad = block_sad(src_plane[py * s_stride + px:], s_stride, recon[py * r_stride + px:], r_stride, tw, th)
		}
		cost := sad + rate_cost
		if cost < best_cost {
			best_cost = cost
			best_sad = sad
			best_mode = mode
		}
	}

	if !e.config.ultrafast && t.mi_size >= av1.BLOCK_8X8 && av1.is_directional_mode(int(best_mode)) && best_sad > (u64(bw * bh) << uint(e.bit_depth - 8)) {
		mode_bits := u64(av1.symbol_cost(y_cdf, int(best_mode)))
		acdf := t.cdfs.angle_delta_cdf[best_mode - av1.V_PRED][:]
		for delta in -av1.MAX_ANGLE_DELTA ..= av1.MAX_ANGLE_DELTA {
			if delta == 0 do continue
			bits := mode_bits + u64(av1.symbol_cost(acdf, delta + av1.MAX_ANGLE_DELTA))
			rate_cost := (q * bits) >> 11
			sad := u64(0)
			if tw < bw || th < bh {
				for dy := 0; dy < bh; dy += th {
					for dx := 0; dx < bw; dx += tw {
						sx, sy := px + dx, py + dy
						if sx > max_x || sy > max_y do continue
						x4, y4 := dx >> 2, dy >> 2
						bd_tr := dy == 0 ? t.block_decoded[0][sub_row][sub_col + x4 + step_x + 1] : (dx + tw < bw)
						bd_bl := dx == 0 ? t.block_decoded[0][sub_row + y4 + step_y + 1][sub_col] : false
						edges := av1.Intra_Edges{
							have_left                = t.avail_l || dx > 0,
							have_above               = t.avail_u || dy > 0,
							have_above_right         = bd_tr,
							have_below_left          = bd_bl,
							max_x                    = max_x,
							max_y                    = max_y,
							enable_intra_edge_filter = s.enable_intra_edge_filter,
							filter_type              = filter_type,
							angle_delta              = delta,
						}
						av1.predict_intra(recon, r_stride, sx, sy, tx_sz, int(best_mode), edges, e.bit_depth)
						sad += block_sad(src_plane[sy * s_stride + sx:], s_stride, recon[sy * r_stride + sx:], r_stride, tw, th)
						for yy in 0 ..< th {
							copy(recon[(sy + yy) * r_stride + sx:][:tw], src_plane[(sy + yy) * s_stride + sx:][:tw])
						}
					}
				}
			} else {
				bd_tr := t.block_decoded[0][sub_row][sub_col + t.bw4 + 1]
				bd_bl := t.block_decoded[0][sub_row + t.bh4 + 1][sub_col]
				edges := av1.Intra_Edges{
					have_left                = t.avail_l,
					have_above               = t.avail_u,
					have_above_right         = bd_tr,
					have_below_left          = bd_bl,
					max_x                    = max_x,
					max_y                    = max_y,
					enable_intra_edge_filter = s.enable_intra_edge_filter,
					filter_type              = filter_type,
					angle_delta              = delta,
				}
				av1.predict_intra(recon, r_stride, px, py, tx_sz, int(best_mode), edges, e.bit_depth)
				sad = block_sad(src_plane[py * s_stride + px:], s_stride, recon[py * r_stride + px:], r_stride, tw, th)
			}
			cost := sad + rate_cost
			if cost < best_cost {
				best_cost = cost
				best_sad = sad
				best_angle = i8(delta)
			}
		}
	}
	return
}

@(private)
choose_intra_uv :: proc(t: ^Tile, $T: typeid) #no_bounds_check {
	e := t.e
	h := &e.header
	s := &e.seq
	t.uv_mode = av1.DC_PRED
	t.angle_delta_uv = 0
	t.cfl_alpha_u = 0
	t.cfl_alpha_v = 0
	if e.config.ultrafast || t.lossless do return

	sub_x := uint(e.ss_x)
	sub_y := uint(e.ss_y)
	px := (t.mi_col >> sub_x) * av1.MI_SIZE
	py := (t.mi_row >> sub_y) * av1.MI_SIZE
	max_x := ((h.mi_cols * av1.MI_SIZE) >> sub_x) - 1
	max_y := ((h.mi_rows * av1.MI_SIZE) >> sub_y) - 1
	if px > max_x || py > max_y do return

	plane_sz := av1.get_plane_residual_size(t.mi_size, 1, sub_x, sub_y)
	nw4 := int(av1.NUM_4X4_BLOCKS_WIDE[plane_sz])
	nh4 := int(av1.NUM_4X4_BLOCKS_HIGH[plane_sz])
	bw := nw4 * 4
	bh := nh4 * 4
	tx_sz := av1.get_uv_tx_size(t.mi_size, 1, int(t.tx_size), t.lossless, u8(e.ss_x), u8(e.ss_y))
	tw := int(av1.TX_WIDTH[tx_sz])
	th := int(av1.TX_HEIGHT[tx_sz])
	step_x := tw >> 2
	step_y := th >> 2
	sub_row := (t.mi_row & 15) >> sub_y
	sub_col := (t.mi_col & 15) >> sub_x
	filter_type := get_filter_type(t, 1)

	u_recon := plane_t(e.planes[1], T)
	v_recon := plane_t(e.planes[2], T)
	r_stride := e.strides[1]
	u_src := plane_t(e.source[1], T)
	v_src := plane_t(e.source[2], T)
	s_stride := e.source_strides[1]

	cfl_ok := is_cfl_allowed(t)
	uv_cdf := cfl_ok ? t.cdfs.uv_mode_cfl_allowed_cdf[t.y_mode][:] : t.cdfs.uv_mode_cfl_not_allowed_cdf[t.y_mode][:]
	q := u64(max(4, int(av1.ac_quant(e.q_idx, h.delta_q_u_ac, e.bit_depth))))

	best_mode := u8(av1.DC_PRED)
	best_cost := max(u64)

	for m in 0 ..< av1.INTRA_MODES {
		mode := u8(m)
		bits := u64(av1.symbol_cost(uv_cdf, m))
		if t.mi_size >= av1.BLOCK_8X8 && av1.is_directional_mode(m) {
			bits += u64(av1.symbol_cost(t.cdfs.angle_delta_cdf[m - av1.V_PRED][:], av1.MAX_ANGLE_DELTA))
		}
		rate_cost := (q * bits) >> 11
		sad := u64(0)
		if tw < bw || th < bh {
			for dy := 0; dy < bh; dy += th {
				for dx := 0; dx < bw; dx += tw {
					sx, sy := px + dx, py + dy
					if sx > max_x || sy > max_y do continue
					x4, y4 := dx >> 2, dy >> 2
					bd_tr := dy == 0 ? t.block_decoded[1][sub_row][sub_col + x4 + step_x + 1] : (dx + tw < bw)
					bd_bl := dx == 0 ? t.block_decoded[1][sub_row + y4 + step_y + 1][sub_col] : false
					edges := av1.Intra_Edges{
						have_left                = t.avail_l_chroma || dx > 0,
						have_above               = t.avail_u_chroma || dy > 0,
						have_above_right         = bd_tr,
						have_below_left          = bd_bl,
						max_x                    = max_x,
						max_y                    = max_y,
						enable_intra_edge_filter = s.enable_intra_edge_filter,
						filter_type              = filter_type,
					}
					av1.predict_intra(u_recon, r_stride, sx, sy, tx_sz, m, edges, e.bit_depth)
					av1.predict_intra(v_recon, r_stride, sx, sy, tx_sz, m, edges, e.bit_depth)
					sad += block_sad(u_src[sy * s_stride + sx:], s_stride, u_recon[sy * r_stride + sx:], r_stride, tw, th) +
						block_sad(v_src[sy * s_stride + sx:], s_stride, v_recon[sy * r_stride + sx:], r_stride, tw, th)
					for yy in 0 ..< th {
						copy(u_recon[(sy + yy) * r_stride + sx:][:tw], u_src[(sy + yy) * s_stride + sx:][:tw])
						copy(v_recon[(sy + yy) * r_stride + sx:][:tw], v_src[(sy + yy) * s_stride + sx:][:tw])
					}
				}
			}
		} else {
			bd_tr := t.block_decoded[1][sub_row][sub_col + nw4 + 1]
			bd_bl := t.block_decoded[1][sub_row + nh4 + 1][sub_col]
			edges := av1.Intra_Edges{
				have_left                = t.avail_l_chroma,
				have_above               = t.avail_u_chroma,
				have_above_right         = bd_tr,
				have_below_left          = bd_bl,
				max_x                    = max_x,
				max_y                    = max_y,
				enable_intra_edge_filter = s.enable_intra_edge_filter,
				filter_type              = filter_type,
			}
			av1.predict_intra(u_recon, r_stride, px, py, tx_sz, m, edges, e.bit_depth)
			av1.predict_intra(v_recon, r_stride, px, py, tx_sz, m, edges, e.bit_depth)
			sad = block_sad(u_src[py * s_stride + px:], s_stride, u_recon[py * r_stride + px:], r_stride, tw, th) +
				block_sad(v_src[py * s_stride + px:], s_stride, v_recon[py * r_stride + px:], r_stride, tw, th)
		}
		cost := sad + rate_cost
		if cost < best_cost {
			best_cost = cost
			best_mode = mode
		}
	}

	if cfl_ok && tw == bw && th == bh {
		bd_tr := t.block_decoded[1][sub_row][sub_col + nw4 + 1]
		bd_bl := t.block_decoded[1][sub_row + nh4 + 1][sub_col]
		edges := av1.Intra_Edges{
			have_left                = t.avail_l_chroma,
			have_above               = t.avail_u_chroma,
			have_above_right         = bd_tr,
			have_below_left          = bd_bl,
			max_x                    = max_x,
			max_y                    = max_y,
			enable_intra_edge_filter = s.enable_intra_edge_filter,
			filter_type              = filter_type,
		}
		av1.predict_intra(u_recon, r_stride, px, py, tx_sz, av1.DC_PRED, edges, e.bit_depth)
		av1.predict_intra(v_recon, r_stride, px, py, tx_sz, av1.DC_PRED, edges, e.bit_depth)

		luma_plane := plane_t(e.planes[0], T)
		luma_stride := e.strides[0]
		log2_w := uint(av1.TX_WIDTH_LOG2[tx_sz])
		log2_h := uint(av1.TX_HEIGHT_LOG2[tx_sz])
		shift_q3 := uint(3 - sub_x - sub_y)
		l_q3: [32 * 32]i16 = ---
		luma_avg := i32(0)
		for i in 0 ..< th {
			luma_y := min((py + i) << sub_y, t.max_luma_h - (1 << sub_y))
			for j in 0 ..< tw {
				luma_x := min((px + j) << sub_x, t.max_luma_w - (1 << sub_x))
				sum_l := i32(0)
				for dy in 0 ..= int(sub_y) {
					lrow := luma_plane[(luma_y + dy) * luma_stride + luma_x:]
					for dx in 0 ..= int(sub_x) {
						sum_l += i32(lrow[dx])
					}
				}
				v := sum_l << shift_q3
				l_q3[i * tw + j] = i16(v)
				luma_avg += v
			}
		}
		luma_avg = av1.round2(i64(luma_avg), log2_w + log2_h)

		sum_ll := i64(0)
		sum_lu := i64(0)
		sum_lv := i64(0)
		for i in 0 ..< th {
			urow := u_recon[(py + i) * r_stride + px:]
			vrow := v_recon[(py + i) * r_stride + px:]
			us := u_src[(py + i) * s_stride + px:]
			vs := v_src[(py + i) * s_stride + px:]
			for j in 0 ..< tw {
				dl := i64(i32(l_q3[i * tw + j]) - luma_avg)
				du := i64(i32(us[j]) - i32(urow[j]))
				dv := i64(i32(vs[j]) - i32(vrow[j]))
				sum_ll += dl * dl
				sum_lu += dl * du
				sum_lv += dl * dv
			}
		}
		if sum_ll > 0 {
			au := int(clamp((sum_lu * 64 + (sum_lu >= 0 ? sum_ll / 2 : -sum_ll / 2)) / sum_ll, -16, 16))
			av := int(clamp((sum_lv * 64 + (sum_lv >= 0 ? sum_ll / 2 : -sum_ll / 2)) / sum_ll, -16, 16))
			if au != 0 || av != 0 {
				sign_u := au < 0 ? 1 : (au > 0 ? 2 : 0)
				sign_v := av < 0 ? 1 : (av > 0 ? 2 : 0)
				cfl_bits := u64(av1.symbol_cost(uv_cdf, av1.UV_CFL_PRED))
				cfl_bits += u64(av1.symbol_cost(t.cdfs.cfl_sign_cdf[:], sign_u * 3 + sign_v - 1))
				if sign_u != 0 {
					cfl_bits += u64(av1.symbol_cost(t.cdfs.cfl_alpha_cdf[(sign_u - 1) * 3 + sign_v][:], abs(au) - 1))
				}
				if sign_v != 0 {
					cfl_bits += u64(av1.symbol_cost(t.cdfs.cfl_alpha_cdf[(sign_v - 1) * 3 + sign_u][:], abs(av) - 1))
				}
				av1.predict_chroma_from_luma(u_recon, r_stride, luma_plane, luma_stride, px, py, tx_sz, au, sub_x, sub_y, t.max_luma_w, t.max_luma_h, e.bit_depth)
				av1.predict_chroma_from_luma(v_recon, r_stride, luma_plane, luma_stride, px, py, tx_sz, av, sub_x, sub_y, t.max_luma_w, t.max_luma_h, e.bit_depth)
				cfl_sad := block_sad(u_src[py * s_stride + px:], s_stride, u_recon[py * r_stride + px:], r_stride, tw, th) +
					block_sad(v_src[py * s_stride + px:], s_stride, v_recon[py * r_stride + px:], r_stride, tw, th)
				cfl_cost := cfl_sad + ((q * cfl_bits) >> 11)
				if cfl_cost < best_cost {
					best_cost = cfl_cost
					best_mode = av1.UV_CFL_PRED
					t.cfl_alpha_u = i16(au)
					t.cfl_alpha_v = i16(av)
				}
			}
		}
	}

	t.uv_mode = best_mode
	for y in 0 ..< t.bh4 {
		row := t.mi_row + y
		if row >= h.mi_rows do break
		for x in 0 ..< t.bw4 {
			col := t.mi_col + x
			if col >= h.mi_cols do break
			e.tile_cells[row * h.mi_cols + col].uv_mode = best_mode
		}
	}
}

@(private)
ref_plane_for_slot :: proc(e: ^Encoder, ref_frame: i8, plane: int, $T: typeid) -> av1.Ref_Plane(T) #no_bounds_check {
	slot := e.header.ref_frame_idx[ref_frame - av1.LAST_FRAME]
	buf := slot == 2 ? e.altref[plane] : (slot == 1 ? e.golden[plane] : e.reference[plane])
	sx := plane > 0 ? uint(e.ss_x) : 0
	sy := plane > 0 ? uint(e.ss_y) : 0
	pw := (e.config.width + int(sx)) >> sx
	ph := (e.config.height + int(sy)) >> sy
	return {
		data           = plane_t(buf, T),
		stride         = e.strides[plane],
		last_x         = pw - 1,
		last_y         = ph - 1,
		ref_width      = e.config.width,
		ref_height     = e.config.height,
		upscaled_width = e.config.width,
	}
}

@(private)
eval_inter_mv :: proc(
	t: ^Tile,
	rp: ^av1.Ref_Plane($T),
	mv: av1.Mv,
	filter: int,
	pred_buf: []T,
	src: []T,
	s_stride: int,
	px, py, w, bh: int,
) -> u64 #no_bounds_check {
	e := t.e
	h := &e.header
	cmv := av1.Mv{
		av1.clamp_mv_row(i32(mv[0]), t.mi_row, h.mi_rows, t.bh4, av1.MV_BORDER + t.bh4 * 32),
		av1.clamp_mv_col(i32(mv[1]), t.mi_col, h.mi_cols, t.bw4, av1.MV_BORDER + t.bw4 * 32),
	}
	if (cmv[0] & 7) == 0 && (cmv[1] & 7) == 0 && h.frame_width == h.upscaled_width {
		ix := px + int(cmv[1] >> 3)
		iy := py + int(cmv[0] >> 3)
		if ix >= 0 && iy >= 0 && ix + w - 1 <= rp.last_x && iy + bh - 1 <= rp.last_y {
			return block_sad(src, s_stride, rp.data[iy * rp.stride + ix:], rp.stride, w, bh)
		}
	}
	sx, sy, stx, sty := av1.scale_mv_for_block(px, py, cmv, 0, 0, h.frame_width, h.frame_height, h.upscaled_width, h.frame_height)
	av1.predict_inter_single(pred_buf, w, 0, 0, w, bh, rp, sx, sy, stx, sty, filter, filter, e.bit_depth)
	return block_sad(src, s_stride, pred_buf, w, w, bh)
}

@(private)
eval_compound_mv :: proc(
	t: ^Tile,
	rp0, rp1: ^av1.Ref_Plane($T),
	mv0, mv1: av1.Mv,
	filter: int,
	pred_buf: []T,
	preds: ^[2][64 * 64]i32,
	src: []T,
	s_stride: int,
	px, py, w, bh: int,
) -> u64 #no_bounds_check {
	e := t.e
	h := &e.header
	cmv0 := av1.Mv{
		av1.clamp_mv_row(i32(mv0[0]), t.mi_row, h.mi_rows, t.bh4, av1.MV_BORDER + t.bh4 * 32),
		av1.clamp_mv_col(i32(mv0[1]), t.mi_col, h.mi_cols, t.bw4, av1.MV_BORDER + t.bw4 * 32),
	}
	cmv1 := av1.Mv{
		av1.clamp_mv_row(i32(mv1[0]), t.mi_row, h.mi_rows, t.bh4, av1.MV_BORDER + t.bh4 * 32),
		av1.clamp_mv_col(i32(mv1[1]), t.mi_col, h.mi_cols, t.bw4, av1.MV_BORDER + t.bw4 * 32),
	}
	sx0, sy0, stx0, sty0 := av1.scale_mv_for_block(px, py, cmv0, 0, 0, h.frame_width, h.frame_height, h.upscaled_width, h.frame_height)
	sx1, sy1, stx1, sty1 := av1.scale_mv_for_block(px, py, cmv1, 0, 0, h.frame_width, h.frame_height, h.upscaled_width, h.frame_height)
	av1.convolve_2d_pred(preds[0][:], w, rp0, sx0, sy0, stx0, sty0, w, bh, filter, filter, true, e.bit_depth)
	av1.convolve_2d_pred(preds[1][:], w, rp1, sx1, sy1, stx1, sty1, w, bh, filter, filter, true, e.bit_depth)
	av1.blend_compound_average(pred_buf, w, 0, 0, w, bh, preds[0][:], preds[1][:], w, e.bit_depth)
	return block_sad(src, s_stride, pred_buf, w, w, bh)
}

@(private)
ref_frame_cost_256 :: proc(t: ^Tile, rf0, rf1: i8) -> u64 #no_bounds_check {
	h := &t.e.header
	left_ref_frame := [2]i8{av1.INTRA_FRAME, av1.NONE}
	left_intra := true
	left_single := true
	if t.avail_l {
		l_idx := t.mi_row * h.mi_cols + (t.mi_col - 1)
		left_ref_frame = t.grid.cells[l_idx].ref_frames
		left_intra = left_ref_frame[0] <= av1.INTRA_FRAME
		left_single = left_ref_frame[1] <= av1.INTRA_FRAME
	}
	above_ref_frame := [2]i8{av1.INTRA_FRAME, av1.NONE}
	above_intra := true
	above_single := true
	if t.avail_u {
		u_idx := (t.mi_row - 1) * h.mi_cols + t.mi_col
		above_ref_frame = t.grid.cells[u_idx].ref_frames
		above_intra = above_ref_frame[0] <= av1.INTRA_FRAME
		above_single = above_ref_frame[1] <= av1.INTRA_FRAME
	}

	is_compound := rf1 > av1.INTRA_FRAME
	bits := u64(0)
	if h.reference_select && min(av1.BLOCK_WIDTH[t.mi_size], av1.BLOCK_HEIGHT[t.mi_size]) >= 8 {
		ctx := get_comp_mode_ctx(t.avail_u, t.avail_l, above_ref_frame, left_ref_frame, above_intra, left_intra, above_single, left_single)
		bits += u64(av1.symbol_cost(t.cdfs.comp_mode_cdf[ctx][:], is_compound ? av1.COMPOUND_REFERENCE : av1.SINGLE_REFERENCE))
	}

	count_refs :: #force_inline proc "contextless" (avail_u, avail_l: bool, above_rf, left_rf: [2]i8, ft: i8) -> int {
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
	ctx_fwd_vs_bwd := ref_count_ctx(c_last + c_last2 + c_last3 + c_gold, c_bwd + c_arf2 + c_arf)
	ctx_last2_vs_last3gold := ref_count_ctx(c_last2, c_last3 + c_gold)

	if is_compound {
		ctx_crt := get_comp_ref_type_ctx(t.avail_u, t.avail_l, above_ref_frame, left_ref_frame, above_intra, left_intra, above_single, left_single)
		if rf1 == av1.GOLDEN_FRAME {
			bits += u64(av1.symbol_cost(t.cdfs.comp_ref_type_cdf[ctx_crt][:], av1.UNIDIR_COMP_REFERENCE))
			bits += u64(av1.symbol_cost(t.cdfs.uni_comp_ref_cdf[ctx_fwd_vs_bwd][0][:], 0))
			bits += u64(av1.symbol_cost(t.cdfs.uni_comp_ref_cdf[ctx_last2_vs_last3gold][1][:], 1))
			bits += u64(av1.symbol_cost(t.cdfs.uni_comp_ref_cdf[ctx_last3_vs_gold][2][:], 1))
		} else {
			bits += u64(av1.symbol_cost(t.cdfs.comp_ref_type_cdf[ctx_crt][:], av1.BIDIR_COMP_REFERENCE))
			if rf0 == av1.LAST_FRAME {
				bits += u64(av1.symbol_cost(t.cdfs.comp_ref_cdf[ctx_last12_vs_last3gold][0][:], 0))
				bits += u64(av1.symbol_cost(t.cdfs.comp_ref_cdf[ctx_last_vs_last2][1][:], 0))
			} else {
				bits += u64(av1.symbol_cost(t.cdfs.comp_ref_cdf[ctx_last12_vs_last3gold][0][:], 1))
				bits += u64(av1.symbol_cost(t.cdfs.comp_ref_cdf[ctx_last3_vs_gold][2][:], 1))
			}
			bits += u64(av1.symbol_cost(t.cdfs.comp_bwd_ref_cdf[ctx_brfarf2_vs_arf][0][:], 1))
		}
	} else {
		if rf0 == av1.ALTREF_FRAME {
			bits += u64(av1.symbol_cost(t.cdfs.single_ref_cdf[ctx_fwd_vs_bwd][0][:], 1))
			bits += u64(av1.symbol_cost(t.cdfs.single_ref_cdf[ctx_brfarf2_vs_arf][1][:], 1))
		} else if rf0 == av1.GOLDEN_FRAME {
			bits += u64(av1.symbol_cost(t.cdfs.single_ref_cdf[ctx_fwd_vs_bwd][0][:], 0))
			bits += u64(av1.symbol_cost(t.cdfs.single_ref_cdf[ctx_last12_vs_last3gold][2][:], 1))
			bits += u64(av1.symbol_cost(t.cdfs.single_ref_cdf[ctx_last3_vs_gold][4][:], 1))
		} else {
			bits += u64(av1.symbol_cost(t.cdfs.single_ref_cdf[ctx_fwd_vs_bwd][0][:], 0))
			bits += u64(av1.symbol_cost(t.cdfs.single_ref_cdf[ctx_last12_vs_last3gold][2][:], 0))
			bits += u64(av1.symbol_cost(t.cdfs.single_ref_cdf[ctx_last_vs_last2][3][:], 0))
		}
	}
	return bits
}

@(private)
mv_component_cost_256 :: #force_inline proc "contextless" (t: ^Tile, mv_ctx, comp: int, diff: i16) -> u64 #no_bounds_check {
	h := &t.e.header
	sign := diff < 0 ? 1 : 0
	v := int(abs(diff)) - 1
	hp := v & 1
	fr := (v >> 1) & 3
	offset := v >> 3

	cost := u64(av1.symbol_cost(t.cdfs.mv_sign_cdf[mv_ctx][comp][:], sign))
	if offset < av1.CLASS0_SIZE {
		cost += u64(av1.symbol_cost(t.cdfs.mv_class_cdf[mv_ctx][comp][:], 0))
		cost += u64(av1.symbol_cost(t.cdfs.mv_class0_bit_cdf[mv_ctx][comp][:], offset))
		if !h.force_integer_mv {
			cost += u64(av1.symbol_cost(t.cdfs.mv_class0_fr_cdf[mv_ctx][comp][offset][:], fr))
		}
		if h.allow_high_precision_mv {
			cost += u64(av1.symbol_cost(t.cdfs.mv_class0_hp_cdf[mv_ctx][comp][:], hp))
		}
	} else {
		mv_class := int(31 - intrinsics.count_leading_zeros(u32(offset)))
		if mv_class >= av1.MV_CLASSES {
			return 1 << 20
		}
		bits := offset - (av1.CLASS0_SIZE << uint(mv_class - 1))
		cost += u64(av1.symbol_cost(t.cdfs.mv_class_cdf[mv_ctx][comp][:], mv_class))
		for i in 0 ..< mv_class {
			cost += u64(av1.symbol_cost(t.cdfs.mv_bit_cdf[mv_ctx][comp][i][:], (bits >> uint(i)) & 1))
		}
		if !h.force_integer_mv {
			cost += u64(av1.symbol_cost(t.cdfs.mv_fr_cdf[mv_ctx][comp][:], fr))
		}
		if h.allow_high_precision_mv {
			cost += u64(av1.symbol_cost(t.cdfs.mv_hp_cdf[mv_ctx][comp][:], hp))
		}
	}
	return cost
}

@(private)
mv_cost_256 :: #force_inline proc "contextless" (t: ^Tile, mv, pred_mv: av1.Mv) -> u64 #no_bounds_check {
	mv_ctx := 0
	diff_r := mv[0] - pred_mv[0]
	diff_c := mv[1] - pred_mv[1]
	mv_joint := (diff_r != 0 ? 2 : 0) | (diff_c != 0 ? 1 : 0)
	cost := u64(av1.symbol_cost(t.cdfs.mv_joint_cdf[mv_ctx][:], mv_joint))
	if diff_r != 0 {
		cost += mv_component_cost_256(t, mv_ctx, 0, diff_r)
	}
	if diff_c != 0 {
		cost += mv_component_cost_256(t, mv_ctx, 1, diff_c)
	}
	return cost
}

@(private)
search_ref_motion :: proc(
	t: ^Tile,
	ref_frame: i8,
	do_full_search: bool,
	$T: typeid,
) -> (best_mode: u8, best_mv: av1.Mv, best_pred_mv: av1.Mv, best_filter: u8, best_cost: u64) #no_bounds_check {
	e := t.e
	h := &e.header
	sign_bias_u8: [av1.TOTAL_REFS_PER_FRAME]u8
	for i in 0 ..< av1.TOTAL_REFS_PER_FRAME {
		sign_bias_u8[i] = u8(h.ref_frame_sign_bias[i])
	}
	rf_pair := [2]i8{ref_frame, av1.NONE}
	t.ref_frame = rf_pair
	mf_ptr: ^[av1.TOTAL_REFS_PER_FRAME][]av1.Mv = h.use_ref_frame_mvs ? &e.motion_field_mvs : nil
	av1.find_mv_stack(
		&t.mv_stack, &t.grid,
		t.mi_row, t.mi_col, t.mi_size,
		rf_pair, false,
		h.allow_high_precision_mv, h.force_integer_mv,
		h.use_ref_frame_mvs, mf_ptr,
		h.gm_type, h.gm_params, sign_bias_u8,
	)

	px := t.mi_col * av1.MI_SIZE
	py := t.mi_row * av1.MI_SIZE
	w := t.bw4 * av1.MI_SIZE
	bh := t.bh4 * av1.MI_SIZE
	s_stride := e.source_strides[0]
	src := plane_t(e.source[0], T)[py * s_stride + px:]
	rp := ref_plane_for_slot(e, ref_frame, 0, T)

	pred_buf: [64 * 64]T = ---
	bd_shift := uint(e.bit_depth - 8)
	ac_q := int(av1.ac_quant(e.q_idx, 0, e.bit_depth))
	rate_q := u64(max(16 << bd_shift, ac_q))

	filt_ctx := get_interp_filter_ctx(t, 0, false)
	reg_filt_bits := h.interpolation_filter == av1.SWITCHABLE ? u64(av1.symbol_cost(t.cdfs.interp_filter_cdf[filt_ctx][:], av1.EIGHTTAP_REGULAR)) : u64(0)
	global_filt_bits := min(w, bh) >= 8 ? u64(0) : reg_filt_bits

	not_new_bits := u64(av1.symbol_cost(t.cdfs.new_mv_cdf[t.mv_stack.new_mv_context][:], 1))
	global_bits := not_new_bits + u64(av1.symbol_cost(t.cdfs.zero_mv_cdf[t.mv_stack.zero_mv_context][:], 0))
	not_zero_bits := not_new_bits + u64(av1.symbol_cost(t.cdfs.zero_mv_cdf[t.mv_stack.zero_mv_context][:], 1))
	nearest_bits := not_zero_bits + u64(av1.symbol_cost(t.cdfs.ref_mv_cdf[t.mv_stack.ref_mv_context][:], 0))
	near_bits := not_zero_bits + u64(av1.symbol_cost(t.cdfs.ref_mv_cdf[t.mv_stack.ref_mv_context][:], 1))
	if t.mv_stack.num_mv_found > 2 {
		near_bits += u64(av1.symbol_cost(t.cdfs.drl_mode_cdf[t.mv_stack.drl_ctx_stack[1]][:], 0))
	}
	new_base_bits := u64(av1.symbol_cost(t.cdfs.new_mv_cdf[t.mv_stack.new_mv_context][:], 0))
	if t.mv_stack.num_mv_found > 1 {
		new_base_bits += u64(av1.symbol_cost(t.cdfs.drl_mode_cdf[t.mv_stack.drl_ctx_stack[0]][:], 0))
	}

	nearest_mv := t.mv_stack.ref_stack_mv[0][0]
	near_mv := t.mv_stack.ref_stack_mv[1][0]
	global_mv := t.mv_stack.global_mvs[0]

	// 1. Evaluate GLOBALMV (zero MV).
	best_mode = av1.GLOBALMV
	best_mv = global_mv
	best_pred_mv = global_mv
	best_filter = av1.EIGHTTAP_REGULAR
	zero_sad := eval_inter_mv(t, &rp, global_mv, av1.EIGHTTAP_REGULAR, pred_buf[:], src, s_stride, px, py, w, bh)
	best_mode_bits := global_bits
	best_cost = zero_sad + ((rate_q * (global_bits + global_filt_bits)) >> 12)

	// 2. Evaluate NEARESTMV.
	nearest_sad := zero_sad
	if nearest_mv != global_mv {
		nearest_sad = eval_inter_mv(t, &rp, nearest_mv, av1.EIGHTTAP_REGULAR, pred_buf[:], src, s_stride, px, py, w, bh)
	}
	nearest_cost := nearest_sad + ((rate_q * (nearest_bits + reg_filt_bits)) >> 12)
	if nearest_cost < best_cost {
		best_cost = nearest_cost
		best_mode = av1.NEARESTMV
		best_mv = nearest_mv
		best_pred_mv = nearest_mv
		best_mode_bits = nearest_bits
	}

	// 3. Evaluate NEARMV (if found).
	near_sad := zero_sad
	has_near := t.mv_stack.num_mv_found > 1 && near_mv != nearest_mv && near_mv != global_mv
	if has_near {
		near_sad = eval_inter_mv(t, &rp, near_mv, av1.EIGHTTAP_REGULAR, pred_buf[:], src, s_stride, px, py, w, bh)
		c := near_sad + ((rate_q * (near_bits + reg_filt_bits)) >> 12)
		if c < best_cost {
			best_cost = c
			best_mode = av1.NEARMV
			best_mv = near_mv
			best_pred_mv = near_mv
			best_mode_bits = near_bits
		}
	}

	// Early exit if residual SAD is already tiny or full search is not requested.
	skip_sad := u64(w * bh) * u64(max(1 << bd_shift, ac_q >> 5))
	best_raw_sad := best_mode == av1.GLOBALMV ? zero_sad : (best_mode == av1.NEARESTMV ? nearest_sad : near_sad)
	if !do_full_search || best_raw_sad <= skip_sad {
		if ref_frame == av1.LAST_FRAME do t.last_mv = best_mv
		return
	}

	// 4. Multi-seed integer diamond search (for NEWMV with ref_mv_idx = 0).
	search_mv := global_mv
	search_sad := zero_sad
	search_cost := search_sad + ((rate_q * mv_cost_256(t, search_mv, nearest_mv)) >> 12)
	raw_seeds := [3]av1.Mv{nearest_mv, near_mv, t.last_mv}
	for raw in raw_seeds {
		cand := av1.Mv{i16(int(raw[0]) & ~int(7)), i16(int(raw[1]) & ~int(7))}
		if cand == global_mv || cand == search_mv do continue
		sad := cand == nearest_mv ? nearest_sad : (has_near && cand == near_mv ? near_sad : eval_inter_mv(t, &rp, cand, av1.EIGHTTAP_REGULAR, pred_buf[:], src, s_stride, px, py, w, bh))
		cost := sad + ((rate_q * mv_cost_256(t, cand, nearest_mv)) >> 12)
		if cost < search_cost {
			search_cost = cost
			search_sad = sad
			search_mv = cand
		}
	}
	base_r := int(search_mv[0])
	base_c := int(search_mv[1])
	max_range := e.config.search_range * 8

	DIAMOND :: [4][2]int{{-1, 0}, {1, 0}, {0, -1}, {0, 1}}
	for step := max_range; step >= 8; step >>= 1 {
		last_dir := -1
		for iter := 0; iter < 4; iter += 1 {
			cur_r := int(search_mv[0])
			cur_c := int(search_mv[1])
			best_dir := -1
			for d, dir in DIAMOND {
				if dir == (last_dir ~ 1) do continue
				cand_r := cur_r + d[0] * step
				cand_c := cur_c + d[1] * step
				if abs(cand_r - base_r) > max_range || abs(cand_c - base_c) > max_range do continue
				cand := av1.Mv{i16(cand_r), i16(cand_c)}
				sad := eval_inter_mv(t, &rp, cand, av1.EIGHTTAP_REGULAR, pred_buf[:], src, s_stride, px, py, w, bh)
				cost := sad + ((rate_q * mv_cost_256(t, cand, nearest_mv)) >> 12)
				if cost < search_cost {
					search_cost = cost
					search_sad = sad
					search_mv = cand
					best_dir = dir
				}
			}
			if best_dir < 0 do break
			last_dir = best_dir
		}
	}

	// Check if subpel predictors beat the integer search result before subpel refinement.
	if nearest_mv != search_mv {
		cost := nearest_sad + ((rate_q * mv_cost_256(t, nearest_mv, nearest_mv)) >> 12)
		if cost < search_cost {
			search_cost = cost
			search_sad = nearest_sad
			search_mv = nearest_mv
		}
	}
	if has_near && near_mv != search_mv {
		cost := near_sad + ((rate_q * mv_cost_256(t, near_mv, nearest_mv)) >> 12)
		if cost < search_cost {
			search_cost = cost
			search_sad = near_sad
			search_mv = near_mv
		}
	}

	// 5. Subpel refinement (half-pel = 4, quarter-pel = 2, eighth-pel = 1) with diagonal check.
	if !h.force_integer_mv && !e.config.ultrafast && search_sad > skip_sad && search_sad < best_cost + (best_cost >> 2) {
		min_step := h.allow_high_precision_mv ? 1 : 2
		for step := 4; step >= min_step; step >>= 1 {
			cur_r := int(search_mv[0])
			cur_c := int(search_mv[1])
			c_up := u64(max(u64))
			c_dn := u64(max(u64))
			c_lt := u64(max(u64))
			c_rt := u64(max(u64))
			for d, dir in DIAMOND {
				cand := av1.Mv{i16(cur_r + d[0] * step), i16(cur_c + d[1] * step)}
				sad := eval_inter_mv(t, &rp, cand, av1.EIGHTTAP_REGULAR, pred_buf[:], src, s_stride, px, py, w, bh)
				cost := sad + ((rate_q * mv_cost_256(t, cand, nearest_mv)) >> 12)
				if dir == 0 do c_up = cost
				if dir == 1 do c_dn = cost
				if dir == 2 do c_lt = cost
				if dir == 3 do c_rt = cost
				if cost < search_cost {
					search_cost = cost
					search_sad = sad
					search_mv = cand
				}
			}
			dr := c_up < c_dn ? -step : step
			dc := c_lt < c_rt ? -step : step
			diag := av1.Mv{i16(cur_r + dr), i16(cur_c + dc)}
			diag_sad := eval_inter_mv(t, &rp, diag, av1.EIGHTTAP_REGULAR, pred_buf[:], src, s_stride, px, py, w, bh)
			diag_cost := diag_sad + ((rate_q * mv_cost_256(t, diag, nearest_mv)) >> 12)
			if diag_cost < search_cost {
				search_cost = diag_cost
				search_sad = diag_sad
				search_mv = diag
			}
		}
	}

	// Check if NEWMV beats best_cost using exact CDF mode + MV bit cost.
	if search_mv != nearest_mv && search_mv != global_mv {
		new_mode_mv_bits := new_base_bits + mv_cost_256(t, search_mv, nearest_mv)
		new_total := search_sad + ((rate_q * (new_mode_mv_bits + reg_filt_bits)) >> 12)
		if new_total < best_cost {
			best_cost = new_total
			best_mode = av1.NEWMV
			best_mv = search_mv
			best_pred_mv = nearest_mv
			best_mode_bits = new_mode_mv_bits
		}
	}

	// 6. Switchable interpolation filter check on subpel MVs.
	if !e.config.ultrafast && h.interpolation_filter == av1.SWITCHABLE && ((best_mv[0] & 7) != 0 || (best_mv[1] & 7) != 0) {
		for f in u8(av1.EIGHTTAP_SMOOTH) ..= u8(av1.EIGHTTAP_SHARP) {
			f_bits := (best_mode == av1.GLOBALMV && min(w, bh) >= 8) ? u64(0) : u64(av1.symbol_cost(t.cdfs.interp_filter_cdf[filt_ctx][:], int(f)))
			sad := eval_inter_mv(t, &rp, best_mv, int(f), pred_buf[:], src, s_stride, px, py, w, bh)
			c := sad + ((rate_q * (best_mode_bits + f_bits)) >> 12)
			if c < best_cost {
				best_cost = c
				best_filter = f
			}
		}
	}
	if ref_frame == av1.LAST_FRAME do t.last_mv = best_mv
	return
}

@(private)
select_prediction_mode :: proc(
	t: ^Tile,
	seg_skip, seg_globalmv, seg_has_ref: bool,
	seg_ref: i8,
	$T: typeid,
) #no_bounds_check {
	e := t.e
	h := &e.header

	// Intra frame or segment forced to INTRA_FRAME.
	if h.frame_is_intra || (seg_has_ref && seg_ref == av1.INTRA_FRAME) {
		t.is_inter = false
		t.ref_frame = {av1.INTRA_FRAME, av1.NONE}
		t.y_mode, t.angle_delta_y, _ = choose_intra_y(t, T)
		t.uv_mode = av1.DC_PRED
		return
	}

	// Segment forced to SKIP or GLOBALMV.
	if seg_skip || seg_globalmv {
		t.is_inter = true
		rf := seg_has_ref ? seg_ref : i8(av1.LAST_FRAME)
		t.ref_frame = {rf, av1.NONE}
		t.y_mode = av1.GLOBALMV
		t.uv_mode = av1.DC_PRED
		sign_bias_u8: [av1.TOTAL_REFS_PER_FRAME]u8
		for i in 0 ..< av1.TOTAL_REFS_PER_FRAME {
			sign_bias_u8[i] = u8(h.ref_frame_sign_bias[i])
		}
		mf_ptr: ^[av1.TOTAL_REFS_PER_FRAME][]av1.Mv = h.use_ref_frame_mvs ? &e.motion_field_mvs : nil
		av1.find_mv_stack(
			&t.mv_stack, &t.grid,
			t.mi_row, t.mi_col, t.mi_size,
			t.ref_frame, false,
			h.allow_high_precision_mv, h.force_integer_mv,
			h.use_ref_frame_mvs, mf_ptr,
			h.gm_type, h.gm_params, sign_bias_u8,
		)
		t.pred_mv[0] = t.mv_stack.global_mvs[0]
		t.mv[0] = t.pred_mv[0]
		return
	}

	// Segment forced to a specific inter reference frame.
	if seg_has_ref && seg_ref > av1.INTRA_FRAME {
		t.is_inter = true
		t.ref_frame = {seg_ref, av1.NONE}
		mode, mv, pred_mv, filt, _ := search_ref_motion(t, seg_ref, true, T)
		t.y_mode = mode
		t.mv[0] = mv
		t.pred_mv[0] = pred_mv
		t.interp_filter = {filt, filt}
		t.uv_mode = av1.DC_PRED
		return
	}

	// General inter frame: evaluate LAST_FRAME, GOLDEN_FRAME / ALTREF_FRAME, Compound, and Intra.
	bd_shift := uint(e.bit_depth - 8)
	ac_q := int(av1.ac_quant(e.q_idx, 0, e.bit_depth))
	rate_q := u64(max(16 << bd_shift, ac_q))
	best_rf := [2]i8{av1.LAST_FRAME, av1.NONE}
	last_mode, last_mv, last_pred_mv, last_filt, last_cost := search_ref_motion(t, av1.LAST_FRAME, true, T)
	last_ref_cost := (rate_q * ref_frame_cost_256(t, av1.LAST_FRAME, av1.NONE)) >> 12
	best_mode := last_mode
	best_mv := [2]av1.Mv{last_mv, {}}
	best_pred_mv := [2]av1.Mv{last_pred_mv, {}}
	best_filt := last_filt
	best_cost := last_cost + last_ref_cost
	best_stack := t.mv_stack

	full_alt_thresh := u64(t.bw4 * t.bh4 * 8) << bd_shift
	if !e.config.ultrafast && (!e.last_keyframe || e.has_altref) && (t.lossless || last_cost > last_ref_cost) {
		alt_refs := [2]i8{av1.GOLDEN_FRAME, av1.ALTREF_FRAME}
		num_alt := 1 + int(e.has_altref)
		px := t.mi_col * av1.MI_SIZE
		py := t.mi_row * av1.MI_SIZE
		w := t.bw4 * av1.MI_SIZE
		bh := t.bh4 * av1.MI_SIZE
		s_stride := e.source_strides[0]
		src := plane_t(e.source[0], T)[py * s_stride + px:]
		rp_last := ref_plane_for_slot(e, av1.LAST_FRAME, 0, T)
		pred_buf: [64 * 64]T = ---
		comp_preds: [2][64 * 64]i32 = ---
		sign_bias_u8: [av1.TOTAL_REFS_PER_FRAME]u8
		for i in 0 ..< av1.TOTAL_REFS_PER_FRAME {
			sign_bias_u8[i] = u8(h.ref_frame_sign_bias[i])
		}

		for alt_rf, ri in alt_refs[:num_alt] {
			if ri == 0 && (e.last_keyframe || ((e.frames % u64(e.config.keyframe_interval)) & 7) == 1) && e.has_altref {
				continue
			}
			alt_ref_cost := (rate_q * ref_frame_cost_256(t, alt_rf, av1.NONE)) >> 12
			if best_cost <= alt_ref_cost do continue
			alt_mode, alt_mv, alt_pred_mv, alt_filt, alt_cost := search_ref_motion(t, alt_rf, t.lossless || last_cost > full_alt_thresh, T)
			if alt_cost + alt_ref_cost < best_cost {
				best_rf = {alt_rf, av1.NONE}
				best_mode = alt_mode
				best_mv = {alt_mv, {}}
				best_pred_mv = {alt_pred_mv, {}}
				best_filt = alt_filt
				best_cost = alt_cost + alt_ref_cost
				best_stack = t.mv_stack
			}

			// Evaluate Compound Reference (LAST_FRAME + alt_rf) when reference_select is enabled.
			if h.reference_select && min(w, bh) >= 8 {
				comp_rf := [2]i8{av1.LAST_FRAME, alt_rf}
				t.ref_frame = comp_rf
				mf_ptr: ^[av1.TOTAL_REFS_PER_FRAME][]av1.Mv = h.use_ref_frame_mvs ? &e.motion_field_mvs : nil
				av1.find_mv_stack(
					&t.mv_stack, &t.grid,
					t.mi_row, t.mi_col, t.mi_size,
					comp_rf, true,
					h.allow_high_precision_mv, h.force_integer_mv,
					h.use_ref_frame_mvs, mf_ptr,
					h.gm_type, h.gm_params, sign_bias_u8,
				)
				comp_ref_bits := ref_frame_cost_256(t, av1.LAST_FRAME, alt_rf)
				comp_ctx := int(av1.COMPOUND_MODE_CTX_MAP[t.mv_stack.ref_mv_context >> 1][min(t.mv_stack.new_mv_context, av1.COMP_NEWMV_CTXS - 1)])
				filt_ctx := get_interp_filter_ctx(t, 0, true)
				reg_filt_bits := h.interpolation_filter == av1.SWITCHABLE ? u64(av1.symbol_cost(t.cdfs.interp_filter_cdf[filt_ctx][:], av1.EIGHTTAP_REGULAR)) : u64(0)
				rp_alt := ref_plane_for_slot(e, alt_rf, 0, T)

				c_nearest := t.mv_stack.ref_stack_mv[0]
				c_near := t.mv_stack.ref_stack_mv[1]
				c_global := t.mv_stack.global_mvs

				Comp_Cand :: struct {
					mode: u8,
					mv0, mv1: av1.Mv,
					pred0, pred1: av1.Mv,
				}
				cands := [4]Comp_Cand{
					{av1.NEAREST_NEARESTMV, c_nearest[0], c_nearest[1], c_nearest[0], c_nearest[1]},
					{av1.NEAR_NEARMV, c_near[0], c_near[1], c_near[0], c_near[1]},
					{av1.GLOBAL_GLOBALMV, c_global[0], c_global[1], c_global[0], c_global[1]},
					{av1.NEW_NEWMV, last_mv, alt_mv, c_nearest[0], c_nearest[1]},
				}
				for cand, ci in cands {
					if ci == 1 && (t.mv_stack.num_mv_found <= 1 || (cand.mv0 == c_nearest[0] && cand.mv1 == c_nearest[1])) do continue
					if ci == 2 && cand.mv0 == c_nearest[0] && cand.mv1 == c_nearest[1] do continue
					if ci == 3 && cand.mv0 == c_nearest[0] && cand.mv1 == c_nearest[1] do continue

					mode_bits := comp_ref_bits + u64(av1.symbol_cost(t.cdfs.compound_mode_cdf[comp_ctx][:], int(cand.mode) - av1.NEAREST_NEARESTMV))
					if cand.mode == av1.NEW_NEWMV {
						if t.mv_stack.num_mv_found > 1 {
							mode_bits += u64(av1.symbol_cost(t.cdfs.drl_mode_cdf[t.mv_stack.drl_ctx_stack[0]][:], 0))
						}
						mode_bits += mv_cost_256(t, cand.mv0, cand.pred0) + mv_cost_256(t, cand.mv1, cand.pred1)
					} else if cand.mode == av1.NEAR_NEARMV {
						if t.mv_stack.num_mv_found > 2 {
							mode_bits += u64(av1.symbol_cost(t.cdfs.drl_mode_cdf[t.mv_stack.drl_ctx_stack[1]][:], 0))
						}
					}
					if cand.mode != av1.GLOBAL_GLOBALMV {
						mode_bits += reg_filt_bits
					}
					rate_cost := (rate_q * mode_bits) >> 12
					if best_cost <= rate_cost do continue
					c_sad := eval_compound_mv(t, &rp_last, &rp_alt, cand.mv0, cand.mv1, av1.EIGHTTAP_REGULAR, pred_buf[:], &comp_preds, src, s_stride, px, py, w, bh)
					if c_sad + rate_cost < best_cost {
						best_cost = c_sad + rate_cost
						best_rf = comp_rf
						best_mode = cand.mode
						best_mv = {cand.mv0, cand.mv1}
						best_pred_mv = {cand.pred0, cand.pred1}
						best_filt = av1.EIGHTTAP_REGULAR
						best_stack = t.mv_stack
					}
				}
			}
		}
	}

	intra_thresh := u64(t.bw4 * t.bh4 * 16) * u64(max(8 << bd_shift, ac_q >> 2))
	if !t.lossless && t.mi_size <= av1.BLOCK_32X32 && best_cost > intra_thresh {
		intra_mode, intra_angle, intra_sad := choose_intra_y(t, T)
		intra_pen := u64(max(8 << bd_shift, ac_q >> 1)) * 6
		if intra_sad + intra_pen < best_cost {
			t.is_inter = false
			t.ref_frame = {av1.INTRA_FRAME, av1.NONE}
			t.y_mode = intra_mode
			t.angle_delta_y = intra_angle
			t.uv_mode = av1.DC_PRED
			return
		}
	}

	t.is_inter = true
	t.ref_frame = best_rf
	t.y_mode = best_mode
	t.mv = best_mv
	t.pred_mv = best_pred_mv
	t.interp_filter = {best_filt, best_filt}
	t.uv_mode = av1.DC_PRED
	t.mv_stack = best_stack
}

@(private)
compute_inter_prediction :: proc(t: ^Tile, $T: typeid) #no_bounds_check {
	e := t.e
	h := &e.header
	num_planes := 1 + (t.has_chroma ? 2 : 0)
	preds: [2][64 * 64]i32 = ---

	for plane in 0 ..< num_planes {
		sub_x := plane > 0 ? uint(e.ss_x) : 0
		sub_y := plane > 0 ? uint(e.ss_y) : 0
		plane_sz := av1.get_plane_residual_size(t.mi_size, plane, sub_x, sub_y)
		num4x4_w := int(av1.NUM_4X4_BLOCKS_WIDE[plane_sz])
		num4x4_h := int(av1.NUM_4X4_BLOCKS_HIGH[plane_sz])
		base_x := (t.mi_col >> sub_x) * av1.MI_SIZE
		base_y := (t.mi_row >> sub_y) * av1.MI_SIZE
		cand_row := (t.mi_row >> sub_y) << sub_y
		cand_col := (t.mi_col >> sub_x) << sub_x

		pred_w := int(av1.BLOCK_WIDTH[t.mi_size]) >> sub_x
		pred_h := int(av1.BLOCK_HEIGHT[t.mi_size]) >> sub_y
		some_use_intra := false
		for r in 0 ..< (num4x4_h << sub_y) {
			for c in 0 ..< (num4x4_w << sub_x) {
				if t.grid.cells[(cand_row + r) * h.mi_cols + (cand_col + c)].ref_frames[0] == av1.INTRA_FRAME {
					some_use_intra = true
				}
			}
		}
		if some_use_intra {
			pred_w = num4x4_w * 4
			pred_h = num4x4_h * 4
			cand_row = t.mi_row
			cand_col = t.mi_col
		}

		dst_plane := plane_t(e.planes[plane], T)
		dst_stride := e.strides[plane]

		r := 0
		for y := 0; y < num4x4_h * 4; y += pred_h {
			c := 0
			for x := 0; x < num4x4_w * 4; x += pred_w {
				cand_idx := (cand_row + r) * h.mi_cols + (cand_col + c)
				cand_cell := e.mi_cells[cand_idx]
				cand_tc := e.tile_cells[cand_idx]
				cand_comp := cand_cell.ref_frames[1] > av1.INTRA_FRAME
				if !cand_comp {
					rf := cand_cell.ref_frames[0]
					rp := ref_plane_for_slot(e, rf, plane, T)
					cmv := av1.Mv{
						av1.clamp_mv_row(i32(cand_cell.mvs[0][0]), t.mi_row, h.mi_rows, t.bh4, av1.MV_BORDER + t.bh4 * 32),
						av1.clamp_mv_col(i32(cand_cell.mvs[0][1]), t.mi_col, h.mi_cols, t.bw4, av1.MV_BORDER + t.bw4 * 32),
					}
					sx, sy, stx, sty := av1.scale_mv_for_block(
						base_x + x, base_y + y, cmv, sub_x, sub_y,
						h.frame_width, h.frame_height, h.upscaled_width, h.frame_height,
					)
					av1.predict_inter_single(
						dst_plane, dst_stride,
						base_x + x, base_y + y, pred_w, pred_h, &rp,
						sx, sy, stx, sty,
						int(cand_tc.interp_filters[0]), int(cand_tc.interp_filters[1]),
						e.bit_depth,
					)
				} else {
					for ref_list in 0 ..< 2 {
						rf := cand_cell.ref_frames[ref_list]
						rp := ref_plane_for_slot(e, rf, plane, T)
						cmv := av1.Mv{
							av1.clamp_mv_row(i32(cand_cell.mvs[ref_list][0]), t.mi_row, h.mi_rows, t.bh4, av1.MV_BORDER + t.bh4 * 32),
							av1.clamp_mv_col(i32(cand_cell.mvs[ref_list][1]), t.mi_col, h.mi_cols, t.bw4, av1.MV_BORDER + t.bw4 * 32),
						}
						sx, sy, stx, sty := av1.scale_mv_for_block(
							base_x + x, base_y + y, cmv, sub_x, sub_y,
							h.frame_width, h.frame_height, h.upscaled_width, h.frame_height,
						)
						av1.convolve_2d_pred(
							preds[ref_list][:], pred_w, &rp,
							sx, sy, stx, sty,
							pred_w, pred_h,
							int(cand_tc.interp_filters[0]), int(cand_tc.interp_filters[1]),
							true, e.bit_depth,
						)
					}
					av1.blend_compound_average(dst_plane, dst_stride, base_x + x, base_y + y, pred_w, pred_h, preds[0][:], preds[1][:], pred_w, e.bit_depth)
				}
				c += 1
			}
			r += 1
		}
	}
}

@(private)
compute_tx_type :: proc "contextless" (t: ^Tile, plane, tx_sz, block_x, block_y: int) -> u8 #no_bounds_check {
	e := t.e
	h := &e.header
	tx_sz_sqr_up := int(av1.TX_SIZE_SQR_UP[tx_sz])
	if t.lossless || tx_sz_sqr_up > av1.TX_32X32 {
		return av1.DCT_DCT
	}
	tx_set := av1.get_tx_set(tx_sz, t.is_inter, h.reduced_tx_set)
	if plane == 0 {
		return e.tile_cells[block_y * h.mi_cols + block_x].tx_type
	}
	if t.is_inter {
		x4 := max(t.mi_col, block_x << uint(e.ss_x))
		y4 := max(t.mi_row, block_y << uint(e.ss_y))
		tx_type := e.tile_cells[y4 * h.mi_cols + x4].tx_type
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

@(private)
transform_and_reconstruct_block :: proc(t: ^Tile, seg_skip: bool, $T: typeid) -> (any_nonzero: bool) #no_bounds_check {
	V4 :: #simd[4]i32
	UT4 :: #simd[4]T
	V8 :: #simd[8]i32
	UT8 :: #simd[8]T

	e := t.e
	h := &e.header
	s := &e.seq
	num_planes := 1 + (t.has_chroma ? 2 : 0)
	bit_depth := e.bit_depth
	max_val := i32((1 << uint(bit_depth)) - 1)
	max_v4 := V4(max_val)
	max_v8 := V8(max_val)
	qindex := get_qindex(h, int(t.segment_id))

	for plane in 0 ..< num_planes {
		if plane == 1 && !t.is_inter && t.has_chroma {
			choose_intra_uv(t, T)
		}
		sub_x := plane > 0 ? uint(e.ss_x) : 0
		sub_y := plane > 0 ? uint(e.ss_y) : 0
		tx_sz := av1.get_uv_tx_size(t.mi_size, plane, int(t.tx_size), t.lossless, u8(e.ss_x), u8(e.ss_y))
		step_x := int(av1.TX_WIDTH[tx_sz]) >> 2
		step_y := int(av1.TX_HEIGHT[tx_sz]) >> 2
		w := int(av1.TX_WIDTH[tx_sz])
		bh := int(av1.TX_HEIGHT[tx_sz])
		adj_tx_sz := int(av1.ADJUSTED_TX_SIZE[tx_sz])
		tw := int(av1.TX_WIDTH[adj_tx_sz])
		th := int(av1.TX_HEIGHT[adj_tx_sz])
		bwl := uint(av1.TX_WIDTH_LOG2[adj_tx_sz])
		seg_eob := (tx_sz == av1.TX_16X64 || tx_sz == av1.TX_64X16) ? 512 : min(1024, w * bh)
		dq_shift := uint(av1.TX_DQ_SHIFT[tx_sz])

		plane_sz := av1.get_plane_residual_size(t.mi_size, plane, sub_x, sub_y)
		num4x4_w := int(av1.NUM_4X4_BLOCKS_WIDE[plane_sz])
		num4x4_h := int(av1.NUM_4X4_BLOCKS_HIGH[plane_sz])
		base_x := (t.mi_col >> sub_x) * av1.MI_SIZE
		base_y := (t.mi_row >> sub_y) * av1.MI_SIZE
		max_x := (h.mi_cols * av1.MI_SIZE) >> sub_x
		max_y := (h.mi_rows * av1.MI_SIZE) >> sub_y

		dc_delta := plane == 0 ? h.delta_q_y_dc : (plane == 1 ? h.delta_q_u_dc : h.delta_q_v_dc)
		ac_delta := plane == 0 ? i8(0) : (plane == 1 ? h.delta_q_u_ac : h.delta_q_v_ac)
		dc_q := av1.dc_quant(qindex, dc_delta, bit_depth)
		ac_q := av1.ac_quant(qindex, ac_delta, bit_depth)
		bias_num := i32(t.is_inter ? 2 : 3)
		dc_bias := t.lossless ? (dc_q >> 1) : ((dc_q * bias_num) >> 3)
		ac_bias := t.lossless ? (ac_q >> 1) : ((ac_q * bias_num) >> 3)
		dc_thresh := max(1, (dc_q - dc_bias + (i32(1) << dq_shift) - 1) >> dq_shift)
		ac_thresh := max(1, (ac_q - ac_bias + (i32(1) << dq_shift) - 1) >> dq_shift)
		ac_thresh_v4 := V4(ac_thresh)

		recon := plane_t(e.planes[plane], T)
		r_stride := e.strides[plane]
		src := plane_t(e.source[plane], T)
		s_stride := e.source_strides[plane]

		tb_idx := 0
		coeff_off := 0

		for y := 0; y < num4x4_h; y += step_y {
			for x := 0; x < num4x4_w; x += step_x {
				start_x := base_x + 4 * x
				start_y := base_y + 4 * y
				if start_x >= max_x || start_y >= max_y do continue

				row := (start_y << sub_y) >> av1.MI_SIZE_LOG2
				col := (start_x << sub_x) >> av1.MI_SIZE_LOG2
				sub_row := row & 15
				sub_col := col & 15
				x4 := start_x >> 2
				y4 := start_y >> 2

				if !t.is_inter {
					is_cfl := plane > 0 && t.uv_mode == av1.UV_CFL_PRED
					mode := plane == 0 ? t.y_mode : (is_cfl ? u8(av1.DC_PRED) : t.uv_mode)
					angle_delta := plane == 0 ? t.angle_delta_y : t.angle_delta_uv
					avail_l := (plane == 0 ? t.avail_l : t.avail_l_chroma) || x > 0
					avail_u := (plane == 0 ? t.avail_u : t.avail_u_chroma) || y > 0
					bd_tr := t.block_decoded[plane][(sub_row >> sub_y) - 1 + 1][(sub_col >> sub_x) + step_x + 1]
					bd_bl := t.block_decoded[plane][(sub_row >> sub_y) + step_y + 1][(sub_col >> sub_x) - 1 + 1]
					edges := av1.Intra_Edges{
						have_left                = avail_l,
						have_above               = avail_u,
						have_above_right         = bd_tr,
						have_below_left          = bd_bl,
						max_x                    = max_x - 1,
						max_y                    = max_y - 1,
						enable_intra_edge_filter = s.enable_intra_edge_filter,
						filter_type              = get_filter_type(t, plane),
						angle_delta              = int(angle_delta),
					}
					av1.predict_intra(recon, r_stride, start_x, start_y, tx_sz, int(mode), edges, bit_depth)
					if is_cfl {
						alpha := int(plane == 1 ? t.cfl_alpha_u : t.cfl_alpha_v)
						av1.predict_chroma_from_luma(
							recon, r_stride,
							plane_t(e.planes[0], T), e.strides[0],
							start_x, start_y, tx_sz, alpha,
							sub_x, sub_y, t.max_luma_w, t.max_luma_h, bit_depth,
						)
					}
					if plane == 0 {
						t.max_luma_w = start_x + step_x * 4
						t.max_luma_h = start_y + step_y * 4
					}
				}

				if plane == 0 {
					luma_tx_type := u8(av1.DCT_DCT)
					if !t.is_inter && !t.lossless && qindex > 0 {
						tx_set := av1.get_tx_set(tx_sz, false, h.reduced_tx_set)
						cand_tx := av1.MODE_TO_TXFM[t.y_mode]
						if av1.TX_TYPE_IN_SET_INTRA[tx_set][cand_tx] != 0 {
							luma_tx_type = cand_tx
						}
					}
					for j in 0 ..< step_y {
						for i in 0 ..< step_x {
							if y4 + j < h.mi_rows && x4 + i < h.mi_cols {
								e.tile_cells[(y4 + j) * h.mi_cols + (x4 + i)].tx_type = luma_tx_type
							}
						}
					}
				}

				tx_type := compute_tx_type(t, plane, tx_sz, x4, y4)
				scan := av1.get_scan(tx_sz, int(tx_type))
				q_slice := t.blk_coeffs[plane][coeff_off : coeff_off + seg_eob]

				eob := 0
				if !seg_skip {
					all_zero := false
					if w == 4 {
						any_diff: V4 = 0
						for r in 0 ..< bh {
							sr := src[(start_y + r) * s_stride + start_x:]
							pr := recon[(start_y + r) * r_stride + start_x:]
							sv := V4(intrinsics.unaligned_load((^UT4)(&sr[0])))
							pv := V4(intrinsics.unaligned_load((^UT4)(&pr[0])))
							diff := sv - pv
							any_diff |= diff
							intrinsics.unaligned_store((^V4)(&t.fcoeffs_buf[r * 4]), diff)
						}
						all_zero = simd.reduce_or(any_diff) == 0
					} else {
						any_diff: V8 = 0
						for r in 0 ..< bh {
							sr := src[(start_y + r) * s_stride + start_x:]
							pr := recon[(start_y + r) * r_stride + start_x:]
							br := t.fcoeffs_buf[r * w:]
							for c := 0; c < w; c += 8 {
								sv := V8(intrinsics.unaligned_load((^UT8)(&sr[c])))
								pv := V8(intrinsics.unaligned_load((^UT8)(&pr[c])))
								diff := sv - pv
								any_diff |= diff
								intrinsics.unaligned_store((^V8)(&br[c]), diff)
							}
						}
						all_zero = simd.reduce_or(any_diff) == 0
					}

					if !all_zero {
						av1.forward_transform_2d(t.fcoeffs_buf[:w * bh], tx_sz, int(tx_type), t.lossless, bit_depth)

						nz_count := 0
						max_r := 0
						max_c := 0
						for r in 0 ..< th {
							f_row := t.fcoeffs_buf[r * w:]
							dq_row := t.dequant_buf[r * w:]
							q_row := q_slice[r << bwl:]
							for col_pos := 0; col_pos < tw; col_pos += 4 {
								fv := intrinsics.unaligned_load((^V4)(&f_row[col_pos]))
								if (r > 0 || col_pos > 0) && simd.reduce_or(simd.lanes_ge(simd.abs(fv), ac_thresh_v4)) == 0 {
									intrinsics.unaligned_store((^V4)(&q_row[col_pos]), V4(0))
									intrinsics.unaligned_store((^V4)(&dq_row[col_pos]), V4(0))
									continue
								}
								for lane in 0 ..< 4 {
									c_col := col_pos + lane
									pos := (r << bwl) + c_col
									coeff := f_row[c_col]
									thresh := pos == 0 ? dc_thresh : ac_thresh
									if abs(coeff) < thresh {
										q_row[c_col] = 0
										dq_row[c_col] = 0
										continue
									}
									bias := pos == 0 ? dc_bias : ac_bias
									qv, dqv := av1.quantize_coeff(
										coeff, pos, tx_sz, int(tx_type), plane,
										dc_q, ac_q, 15, bit_depth, bias,
									)
									q_row[c_col] = qv
									dq_row[c_col] = dqv
									if qv != 0 {
										nz_count += 1
										max_r = max(max_r, r)
										max_c = max(max_c, c_col)
									}
								}
							}
						}

						if nz_count > 0 {
							rem := nz_count
							nz_scan: [64]u16 = ---
							found := 0
							for c in 0 ..< seg_eob {
								if q_slice[scan[c]] != 0 {
									if found < 64 {
										nz_scan[found] = u16(c)
									}
									found += 1
									eob = c + 1
									rem -= 1
									if rem == 0 do break
								}
							}
							if !t.lossless && found <= 64 && abs(q_slice[scan[eob - 1]]) == 1 {
								best_score := i64(0)
								best_found := 0
								best_eob := 0
								accu_score := i64(0)
								base_bits := i64(t.is_inter ? (plane == 0 ? 7 : 6) : 5)
								for i in 0 ..< found {
									curr_c := int(nz_scan[i])
									pos := int(scan[curr_c])
									prev_eob := i > 0 ? int(nz_scan[i - 1]) + 1 : 0
									zero_run := curr_c - prev_eob
									dq_idx := (pos >> bwl) * w + (pos & ((1 << bwl) - 1))
									av := i64(abs(t.fcoeffs_buf[dq_idx])) << dq_shift
									step := i64(pos == 0 ? dc_q : ac_q)
									aq := i64(abs(q_slice[pos]))
									dist_gain := aq * (2 * av - aq * step)
									eob_bits := i == 0 && curr_c > 0 ? i64(32 - intrinsics.count_leading_zeros(u32(curr_c))) : 0
									bits := (i == 0 ? base_bits : i64(4)) + eob_bits + i64((min(zero_run, 32) * 3) >> 2)
									if !t.is_inter && pos == 0 {
										bits = 2
									}
									rate_cost := (step * bits) >> 3
									accu_score += dist_gain - rate_cost
									if accu_score >= best_score {
										best_score = accu_score
										best_found = i + 1
										best_eob = curr_c + 1
									}
								}
								if best_found < found {
									for i in best_found ..< found {
										pos := int(scan[nz_scan[i]])
										dq_idx := (pos >> bwl) * w + (pos & ((1 << bwl) - 1))
										q_slice[pos] = 0
										t.dequant_buf[dq_idx] = 0
									}
									eob = best_eob
									if eob == 1 && scan[0] == 0 {
										max_r = 0
										max_c = 0
									}
								}
							}
							if eob > 0 {
								any_nonzero = true
								av1.inverse_transform_2d(
									t.dequant_buf[:w * bh],
									tx_sz, int(tx_type),
									t.lossless, eob, bit_depth, max_r, max_c,
								)
								if w == 4 {
									for r in 0 ..< bh {
										pr := recon[(start_y + r) * r_stride + start_x:]
										pv := V4(intrinsics.unaligned_load((^UT4)(&pr[0])))
										dv := intrinsics.unaligned_load((^V4)(&t.dequant_buf[r * 4]))
										cv := simd.clamp(pv + dv, V4(0), max_v4)
										intrinsics.unaligned_store((^UT4)(&pr[0]), UT4(cv))
									}
								} else {
									for r in 0 ..< bh {
										pr := recon[(start_y + r) * r_stride + start_x:]
										dr := t.dequant_buf[r * w:]
										for c := 0; c < w; c += 8 {
											pv := V8(intrinsics.unaligned_load((^UT8)(&pr[c])))
											dv := intrinsics.unaligned_load((^V8)(&dr[c]))
											cv := simd.clamp(pv + dv, V8(0), max_v8)
											intrinsics.unaligned_store((^UT8)(&pr[c]), UT8(cv))
										}
									}
								}
							}
						}
					}
				}

				if eob == 0 && plane == 0 {
					for j in 0 ..< step_y {
						for i in 0 ..< step_x {
							if y4 + j < h.mi_rows && x4 + i < h.mi_cols {
								e.tile_cells[(y4 + j) * h.mi_cols + (x4 + i)].tx_type = av1.DCT_DCT
							}
						}
					}
				}
				t.blk_eobs[plane][tb_idx] = u16(eob)
				tb_idx += 1
				coeff_off += seg_eob

				for i in 0 ..< step_y {
					for j in 0 ..< step_x {
						t.block_decoded[plane][(sub_row >> sub_y) + i + 1][(sub_col >> sub_x) + j + 1] = true
					}
				}
			}
		}
	}
	return
}

// Section 5.11.8: Emit mode info syntax.
@(private)
write_mode_info :: proc(t: ^Tile) {
	if t.e.header.frame_is_intra {
		write_intra_frame_mode_info(t)
	} else {
		write_inter_frame_mode_info(t)
	}
}

@(private)
write_intra_frame_mode_info :: proc(t: ^Tile) {
	e := t.e
	if e.seg_id_pre_skip {
		write_intra_segment_id(t, false)
	}
	write_skip(t)
	if !e.seg_id_pre_skip {
		write_intra_segment_id(t, t.skip)
	}
	write_cdef(t)

	above_mode := av1.INTRA_MODE_CONTEXT[t.avail_u ? t.grid.cells[(t.mi_row - 1) * t.grid.mi_cols + t.mi_col].y_mode : av1.DC_PRED]
	left_mode := av1.INTRA_MODE_CONTEXT[t.avail_l ? t.grid.cells[t.mi_row * t.grid.mi_cols + (t.mi_col - 1)].y_mode : av1.DC_PRED]
	put_symbol(&t.sym, int(t.y_mode), t.cdfs.intra_frame_y_mode_cdf[above_mode][left_mode][:])
	write_intra_angle_info_y(t)
	if t.has_chroma {
		if is_cfl_allowed(t) {
			put_symbol(&t.sym, int(t.uv_mode), t.cdfs.uv_mode_cfl_allowed_cdf[t.y_mode][:])
		} else {
			put_symbol(&t.sym, int(t.uv_mode), t.cdfs.uv_mode_cfl_not_allowed_cdf[t.y_mode][:])
		}
		if t.uv_mode == av1.UV_CFL_PRED {
			write_cfl_alphas(t)
		} else {
			write_intra_angle_info_uv(t)
		}
	}
}

@(private)
write_intra_segment_id :: proc(t: ^Tile, skip_active: bool) {
	if t.e.header.segmentation_enabled {
		write_segment_id(t, skip_active)
	}
}

@(private)
write_segment_id :: proc(t: ^Tile, skip_active: bool) {
	if skip_active do return
	pred, ctx := get_spatial_seg_pred(t)
	sym := neg_interleave(int(t.segment_id), pred, t.e.last_active_seg_id + 1)
	put_symbol(&t.sym, sym, t.cdfs.segment_id_cdf[ctx][:])
}

@(private)
write_skip :: proc(t: ^Tile) {
	if seg_feature_active(t, av1.SEG_LVL_SKIP) do return
	e := t.e
	h := &e.header
	ctx := 0
	if t.avail_u && e.tile_cells[(t.mi_row - 1) * h.mi_cols + t.mi_col].skip do ctx += 1
	if t.avail_l && e.tile_cells[t.mi_row * h.mi_cols + (t.mi_col - 1)].skip do ctx += 1
	put_symbol(&t.sym, int(t.skip), t.cdfs.skip_cdf[ctx][:])
}

@(private)
write_cdef :: proc(t: ^Tile) {
	e := t.e
	h := &e.header
	s := &e.seq
	if t.skip || h.coded_lossless || !s.enable_cdef || h.allow_intrabc {
		return
	}
	cdef_size4 := int(av1.NUM_4X4_BLOCKS_WIDE[av1.BLOCK_64X64])
	cdef_mask4 := ~(cdef_size4 - 1)
	r := t.mi_row & cdef_mask4
	c := t.mi_col & cdef_mask4
	sb64_cols := e.sb_cols
	r64 := r >> 4
	c64 := c >> 4
	if e.cdef_idx[r64 * sb64_cols + c64] == -1 {
		av1.write_literal(&t.sym, 0, int(h.cdef_bits))
		for y := r; y < t.mi_row + t.bh4; y += cdef_size4 {
			for x := c; x < t.mi_col + t.bw4; x += cdef_size4 {
				e.cdef_idx[(y >> 4) * sb64_cols + (x >> 4)] = 0
			}
		}
	}
}

@(private)
write_inter_frame_mode_info :: proc(t: ^Tile) {
	e := t.e
	h := &e.header

	left_ref_frame: [2]i8
	left_intra: bool
	left_single: bool
	if t.avail_l {
		l_idx := t.mi_row * h.mi_cols + (t.mi_col - 1)
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
		u_idx := (t.mi_row - 1) * h.mi_cols + t.mi_col
		above_ref_frame = t.grid.cells[u_idx].ref_frames
		above_intra = above_ref_frame[0] <= av1.INTRA_FRAME
		above_single = above_ref_frame[1] <= av1.INTRA_FRAME
	} else {
		above_ref_frame = {av1.INTRA_FRAME, av1.NONE}
		above_intra = true
		above_single = true
	}

	write_inter_segment_id(t, true)
	write_skip(t)
	if !e.seg_id_pre_skip {
		write_inter_segment_id(t, false)
	}
	write_cdef(t)

	write_is_inter(t, above_intra, left_intra)
	if t.is_inter {
		write_inter_block_mode_info(t, above_ref_frame, left_ref_frame, above_intra, left_intra, above_single, left_single)
	} else {
		write_intra_block_mode_info(t)
	}
}

@(private)
write_inter_segment_id :: proc(t: ^Tile, pre_skip: bool) {
	e := t.e
	h := &e.header
	if !h.segmentation_enabled || !h.segmentation_update_map {
		return
	}
	if pre_skip {
		if !e.seg_id_pre_skip do return
	} else {
		if t.skip {
			for i in 0 ..< t.bw4 {
				t.above_seg_pred_ctx[t.mi_col - t.mi_col_start + i] = 0
			}
			for i in 0 ..< t.bh4 {
				t.left_seg_pred_ctx[(t.mi_row + i) & 63] = 0
			}
			return
		}
	}
	if h.segmentation_temporal_update {
		ctx := int(t.left_seg_pred_ctx[t.mi_row & 63] + t.above_seg_pred_ctx[t.mi_col - t.mi_col_start])
		seg_id_predicted := u8(t.segment_id == t.pred_seg_id ? 1 : 0)
		put_symbol(&t.sym, int(seg_id_predicted), t.cdfs.segment_id_predicted_cdf[ctx][:])
		if seg_id_predicted == 0 {
			write_segment_id(t, false)
		}
		for i in 0 ..< t.bw4 {
			t.above_seg_pred_ctx[t.mi_col - t.mi_col_start + i] = seg_id_predicted
		}
		for i in 0 ..< t.bh4 {
			t.left_seg_pred_ctx[(t.mi_row + i) & 63] = seg_id_predicted
		}
	} else {
		write_segment_id(t, false)
	}
}

@(private)
write_is_inter :: proc(t: ^Tile, above_intra, left_intra: bool) {
	if t.skip_mode != 0 || seg_feature_active(t, av1.SEG_LVL_REF_FRAME) || seg_feature_active(t, av1.SEG_LVL_GLOBALMV) {
		return
	}
	ctx := 0
	if t.avail_u && t.avail_l {
		ctx = (left_intra && above_intra) ? 3 : int(left_intra || above_intra)
	} else if t.avail_u || t.avail_l {
		ctx = 2 * int(t.avail_u ? above_intra : left_intra)
	}
	put_symbol(&t.sym, int(t.is_inter), t.cdfs.is_inter_cdf[ctx][:])
}

@(private)
write_intra_block_mode_info :: proc(t: ^Tile) {
	ctx := av1.SIZE_GROUP[t.mi_size]
	put_symbol(&t.sym, int(t.y_mode), t.cdfs.y_mode_cdf[ctx][:])
	write_intra_angle_info_y(t)
	if t.has_chroma {
		if is_cfl_allowed(t) {
			put_symbol(&t.sym, int(t.uv_mode), t.cdfs.uv_mode_cfl_allowed_cdf[t.y_mode][:])
		} else {
			put_symbol(&t.sym, int(t.uv_mode), t.cdfs.uv_mode_cfl_not_allowed_cdf[t.y_mode][:])
		}
		if t.uv_mode == av1.UV_CFL_PRED {
			write_cfl_alphas(t)
		} else {
			write_intra_angle_info_uv(t)
		}
	}
}

@(private)
write_intra_angle_info_y :: proc(t: ^Tile) {
	if t.mi_size >= av1.BLOCK_8X8 && av1.is_directional_mode(int(t.y_mode)) {
		put_symbol(&t.sym, int(t.angle_delta_y) + av1.MAX_ANGLE_DELTA, t.cdfs.angle_delta_cdf[t.y_mode - av1.V_PRED][:])
	}
}

@(private)
write_intra_angle_info_uv :: proc(t: ^Tile) {
	if t.mi_size >= av1.BLOCK_8X8 && av1.is_directional_mode(int(t.uv_mode)) {
		put_symbol(&t.sym, int(t.angle_delta_uv) + av1.MAX_ANGLE_DELTA, t.cdfs.angle_delta_cdf[t.uv_mode - av1.V_PRED][:])
	}
}

@(private)
write_cfl_alphas :: proc(t: ^Tile) {
	sign_u := t.cfl_alpha_u < 0 ? 1 : (t.cfl_alpha_u > 0 ? 2 : 0)
	sign_v := t.cfl_alpha_v < 0 ? 1 : (t.cfl_alpha_v > 0 ? 2 : 0)
	cfl_alpha_signs := sign_u * 3 + sign_v - 1
	put_symbol(&t.sym, cfl_alpha_signs, t.cdfs.cfl_sign_cdf[:])
	if sign_u != 0 {
		ctx := (sign_u - 1) * 3 + sign_v
		put_symbol(&t.sym, int(abs(t.cfl_alpha_u)) - 1, t.cdfs.cfl_alpha_cdf[ctx][:])
	}
	if sign_v != 0 {
		ctx := (sign_v - 1) * 3 + sign_u
		put_symbol(&t.sym, int(abs(t.cfl_alpha_v)) - 1, t.cdfs.cfl_alpha_cdf[ctx][:])
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

@(private)
write_ref_frames :: proc(
	t: ^Tile,
	above_ref_frame, left_ref_frame: [2]i8,
	above_intra, left_intra, above_single, left_single: bool,
) {
	if t.skip_mode != 0 ||
	   seg_feature_active(t, av1.SEG_LVL_REF_FRAME) ||
	   seg_feature_active(t, av1.SEG_LVL_SKIP) ||
	   seg_feature_active(t, av1.SEG_LVL_GLOBALMV) {
		return
	}

	h := &t.e.header
	is_compound := t.ref_frame[1] > av1.INTRA_FRAME
	if h.reference_select && min(av1.BLOCK_WIDTH[t.mi_size], av1.BLOCK_HEIGHT[t.mi_size]) >= 8 {
		ctx := get_comp_mode_ctx(t.avail_u, t.avail_l, above_ref_frame, left_ref_frame, above_intra, left_intra, above_single, left_single)
		put_symbol(&t.sym, is_compound ? av1.COMPOUND_REFERENCE : av1.SINGLE_REFERENCE, t.cdfs.comp_mode_cdf[ctx][:])
	}

	count_refs :: #force_inline proc "contextless" (avail_u, avail_l: bool, above_rf, left_rf: [2]i8, ft: i8) -> int {
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
	ctx_fwd_vs_bwd := ref_count_ctx(c_last + c_last2 + c_last3 + c_gold, c_bwd + c_arf2 + c_arf)
	ctx_last2_vs_last3gold := ref_count_ctx(c_last2, c_last3 + c_gold)

	if is_compound {
		ctx_crt := get_comp_ref_type_ctx(t.avail_u, t.avail_l, above_ref_frame, left_ref_frame, above_intra, left_intra, above_single, left_single)
		if t.ref_frame[1] == av1.GOLDEN_FRAME {
			put_symbol(&t.sym, av1.UNIDIR_COMP_REFERENCE, t.cdfs.comp_ref_type_cdf[ctx_crt][:])
			put_symbol(&t.sym, 0, t.cdfs.uni_comp_ref_cdf[ctx_fwd_vs_bwd][0][:])
			put_symbol(&t.sym, 1, t.cdfs.uni_comp_ref_cdf[ctx_last2_vs_last3gold][1][:])
			put_symbol(&t.sym, 1, t.cdfs.uni_comp_ref_cdf[ctx_last3_vs_gold][2][:])
		} else {
			put_symbol(&t.sym, av1.BIDIR_COMP_REFERENCE, t.cdfs.comp_ref_type_cdf[ctx_crt][:])
			if t.ref_frame[0] == av1.LAST_FRAME {
				put_symbol(&t.sym, 0, t.cdfs.comp_ref_cdf[ctx_last12_vs_last3gold][0][:])
				put_symbol(&t.sym, 0, t.cdfs.comp_ref_cdf[ctx_last_vs_last2][1][:])
			} else {
				put_symbol(&t.sym, 1, t.cdfs.comp_ref_cdf[ctx_last12_vs_last3gold][0][:])
				put_symbol(&t.sym, 1, t.cdfs.comp_ref_cdf[ctx_last3_vs_gold][2][:])
			}
			put_symbol(&t.sym, 1, t.cdfs.comp_bwd_ref_cdf[ctx_brfarf2_vs_arf][0][:])
		}
	} else {
		if t.ref_frame[0] == av1.ALTREF_FRAME {
			put_symbol(&t.sym, 1, t.cdfs.single_ref_cdf[ctx_fwd_vs_bwd][0][:])
			put_symbol(&t.sym, 1, t.cdfs.single_ref_cdf[ctx_brfarf2_vs_arf][1][:])
		} else if t.ref_frame[0] == av1.GOLDEN_FRAME {
			put_symbol(&t.sym, 0, t.cdfs.single_ref_cdf[ctx_fwd_vs_bwd][0][:])
			put_symbol(&t.sym, 1, t.cdfs.single_ref_cdf[ctx_last12_vs_last3gold][2][:])
			put_symbol(&t.sym, 1, t.cdfs.single_ref_cdf[ctx_last3_vs_gold][4][:])
		} else {
			put_symbol(&t.sym, 0, t.cdfs.single_ref_cdf[ctx_fwd_vs_bwd][0][:])
			put_symbol(&t.sym, 0, t.cdfs.single_ref_cdf[ctx_last12_vs_last3gold][2][:])
			put_symbol(&t.sym, 0, t.cdfs.single_ref_cdf[ctx_last_vs_last2][3][:])
		}
	}
}

@(private)
get_interp_filter_ctx :: proc(t: ^Tile, dir: int, is_compound: bool) -> int {
	e := t.e
	h := &e.header
	ctx := ((dir & 1) * 2 + int(is_compound)) * 4
	left_type := 3
	above_type := 3
	if t.avail_l {
		l_idx := t.mi_row * h.mi_cols + (t.mi_col - 1)
		rfs := t.grid.cells[l_idx].ref_frames
		if rfs[0] == t.ref_frame[0] || rfs[1] == t.ref_frame[0] {
			left_type = int(e.tile_cells[l_idx].interp_filters[dir])
		}
	}
	if t.avail_u {
		u_idx := (t.mi_row - 1) * h.mi_cols + t.mi_col
		rfs := t.grid.cells[u_idx].ref_frames
		if rfs[0] == t.ref_frame[0] || rfs[1] == t.ref_frame[0] {
			above_type = int(e.tile_cells[u_idx].interp_filters[dir])
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

@(private)
needs_interp_filter :: proc "contextless" (t: ^Tile) -> bool {
	h := &t.e.header
	large := min(av1.BLOCK_WIDTH[t.mi_size], av1.BLOCK_HEIGHT[t.mi_size]) >= 8
	if t.skip_mode != 0 do return false
	if large && t.y_mode == av1.GLOBALMV {
		return h.gm_type[t.ref_frame[0]] == av1.TRANSLATION
	}
	if large && t.y_mode == av1.GLOBAL_GLOBALMV {
		return h.gm_type[t.ref_frame[0]] == av1.TRANSLATION || h.gm_type[t.ref_frame[1]] == av1.TRANSLATION
	}
	return true
}

@(private)
write_inter_block_mode_info :: proc(
	t: ^Tile,
	above_ref_frame, left_ref_frame: [2]i8,
	above_intra, left_intra, above_single, left_single: bool,
) {
	e := t.e
	h := &e.header
	s := &e.seq

	write_ref_frames(t, above_ref_frame, left_ref_frame, above_intra, left_intra, above_single, left_single)
	is_compound := t.ref_frame[1] > av1.INTRA_FRAME

	if t.skip_mode == 0 && !seg_feature_active(t, av1.SEG_LVL_SKIP) && !seg_feature_active(t, av1.SEG_LVL_GLOBALMV) {
		if is_compound {
			ctx := int(av1.COMPOUND_MODE_CTX_MAP[t.mv_stack.ref_mv_context >> 1][min(t.mv_stack.new_mv_context, av1.COMP_NEWMV_CTXS - 1)])
			put_symbol(&t.sym, int(t.y_mode) - av1.NEAREST_NEARESTMV, t.cdfs.compound_mode_cdf[ctx][:])
		} else {
			if t.y_mode == av1.NEWMV {
				put_symbol(&t.sym, 0, t.cdfs.new_mv_cdf[t.mv_stack.new_mv_context][:])
			} else {
				put_symbol(&t.sym, 1, t.cdfs.new_mv_cdf[t.mv_stack.new_mv_context][:])
				if t.y_mode == av1.GLOBALMV {
					put_symbol(&t.sym, 0, t.cdfs.zero_mv_cdf[t.mv_stack.zero_mv_context][:])
				} else {
					put_symbol(&t.sym, 1, t.cdfs.zero_mv_cdf[t.mv_stack.zero_mv_context][:])
					put_symbol(&t.sym, t.y_mode == av1.NEARESTMV ? 0 : 1, t.cdfs.ref_mv_cdf[t.mv_stack.ref_mv_context][:])
				}
			}
		}
	}

	if t.y_mode == av1.NEWMV || t.y_mode == av1.NEW_NEWMV {
		for idx in 0 ..< 2 {
			if t.mv_stack.num_mv_found > idx + 1 {
				drl := int(t.ref_mv_idx) > idx ? 1 : 0
				put_symbol(&t.sym, drl, t.cdfs.drl_mode_cdf[t.mv_stack.drl_ctx_stack[idx]][:])
				if drl == 0 do break
			}
		}
	} else if av1.has_nearmv(int(t.y_mode)) {
		for idx in 1 ..< 3 {
			if t.mv_stack.num_mv_found > idx + 1 {
				drl := int(t.ref_mv_idx) > idx ? 1 : 0
				put_symbol(&t.sym, drl, t.cdfs.drl_mode_cdf[t.mv_stack.drl_ctx_stack[idx]][:])
				if drl == 0 do break
			}
		}
	}

	num_lists := is_compound ? 2 : 1
	for i in 0 ..< num_lists {
		comp_mode := av1.get_inter_comp_mode(int(t.y_mode), i)
		if comp_mode == av1.NEWMV {
			write_mv(t, i)
		}
	}

	if h.interpolation_filter == av1.SWITCHABLE {
		num_dirs := s.enable_dual_filter ? 2 : 1
		for dir in 0 ..< num_dirs {
			if needs_interp_filter(t) {
				ctx := get_interp_filter_ctx(t, dir, is_compound)
				put_symbol(&t.sym, int(t.interp_filter[dir]), t.cdfs.interp_filter_cdf[ctx][:])
			}
		}
	}
}

@(private)
write_mv :: proc(t: ^Tile, ref_list: int) {
	mv_ctx := 0
	diff_r := t.mv[ref_list][0] - t.pred_mv[ref_list][0]
	diff_c := t.mv[ref_list][1] - t.pred_mv[ref_list][1]
	mv_joint := (diff_r != 0 ? 2 : 0) | (diff_c != 0 ? 1 : 0)
	put_symbol(&t.sym, mv_joint, t.cdfs.mv_joint_cdf[mv_ctx][:])
	if diff_r != 0 {
		write_mv_component(t, mv_ctx, 0, diff_r)
	}
	if diff_c != 0 {
		write_mv_component(t, mv_ctx, 1, diff_c)
	}
}

@(private)
write_mv_component :: proc(t: ^Tile, mv_ctx, comp: int, diff: i16) {
	h := &t.e.header
	sign := diff < 0 ? 1 : 0
	v := int(abs(diff)) - 1
	hp := v & 1
	fr := (v >> 1) & 3
	offset := v >> 3

	put_symbol(&t.sym, sign, t.cdfs.mv_sign_cdf[mv_ctx][comp][:])
	if offset < av1.CLASS0_SIZE {
		put_symbol(&t.sym, 0, t.cdfs.mv_class_cdf[mv_ctx][comp][:])
		put_symbol(&t.sym, offset, t.cdfs.mv_class0_bit_cdf[mv_ctx][comp][:])
		if !h.force_integer_mv {
			put_symbol(&t.sym, fr, t.cdfs.mv_class0_fr_cdf[mv_ctx][comp][offset][:])
		}
		if h.allow_high_precision_mv {
			put_symbol(&t.sym, hp, t.cdfs.mv_class0_hp_cdf[mv_ctx][comp][:])
		}
	} else {
		mv_class := int(31 - intrinsics.count_leading_zeros(u32(offset)))
		bits := offset - (av1.CLASS0_SIZE << uint(mv_class - 1))
		put_symbol(&t.sym, mv_class, t.cdfs.mv_class_cdf[mv_ctx][comp][:])
		for i in 0 ..< mv_class {
			put_symbol(&t.sym, (bits >> uint(i)) & 1, t.cdfs.mv_bit_cdf[mv_ctx][comp][i][:])
		}
		if !h.force_integer_mv {
			put_symbol(&t.sym, fr, t.cdfs.mv_fr_cdf[mv_ctx][comp][:])
		}
		if h.allow_high_precision_mv {
			put_symbol(&t.sym, hp, t.cdfs.mv_hp_cdf[mv_ctx][comp][:])
		}
	}
}

@(private)
get_above_tx_width :: #force_inline proc "contextless" (t: ^Tile, row, col: int) -> int {
	e := t.e
	h := &e.header
	if row == t.mi_row {
		if !t.avail_u do return 64
		u_idx := (row - 1) * h.mi_cols + col
		if e.tile_cells[u_idx].skip && t.grid.cells[u_idx].is_inter {
			return int(av1.BLOCK_WIDTH[t.grid.cells[u_idx].mi_size])
		}
	}
	return int(av1.TX_WIDTH[e.tile_cells[(row - 1) * h.mi_cols + col].inter_tx_size])
}

@(private)
get_left_tx_height :: #force_inline proc "contextless" (t: ^Tile, row, col: int) -> int {
	e := t.e
	h := &e.header
	if col == t.mi_col {
		if !t.avail_l do return 64
		l_idx := row * h.mi_cols + (col - 1)
		if e.tile_cells[l_idx].skip && t.grid.cells[l_idx].is_inter {
			return int(av1.BLOCK_HEIGHT[t.grid.cells[l_idx].mi_size])
		}
	}
	return int(av1.TX_HEIGHT[e.tile_cells[row * h.mi_cols + (col - 1)].inter_tx_size])
}

// Section 5.11.17 - 5.11.19: Emit block transform size syntax.
@(private)
write_block_tx_size :: proc(t: ^Tile) {
	e := t.e
	h := &e.header
	if h.tx_mode == av1.TX_MODE_SELECT &&
	   t.mi_size > av1.BLOCK_4X4 &&
	   t.is_inter &&
	   !t.skip &&
	   !t.lossless {
		max_tx_sz := int(av1.MAX_TX_SIZE_RECT[t.mi_size])
		tx_w4 := int(av1.TX_WIDTH[max_tx_sz]) / av1.MI_SIZE
		tx_h4 := int(av1.TX_HEIGHT[max_tx_sz]) / av1.MI_SIZE
		for row := t.mi_row; row < t.mi_row + t.bh4; row += tx_h4 {
			for col := t.mi_col; col < t.mi_col + t.bw4; col += tx_w4 {
				write_var_tx_size(t, row, col, max_tx_sz, 0)
			}
		}
	} else {
		write_tx_size(t, !t.skip || !t.is_inter)
		for row in t.mi_row ..< min(h.mi_rows, t.mi_row + t.bh4) {
			for col in t.mi_col ..< min(h.mi_cols, t.mi_col + t.bw4) {
				e.tile_cells[row * h.mi_cols + col].inter_tx_size = t.tx_size
			}
		}
	}
}

@(private)
write_tx_size :: proc(t: ^Tile, allow_select: bool) {
	e := t.e
	h := &e.header
	if t.lossless do return
	max_rect_tx_size := int(av1.MAX_TX_SIZE_RECT[t.mi_size])
	max_tx_depth := int(av1.MAX_TX_DEPTH[t.mi_size])
	if t.mi_size > av1.BLOCK_4X4 && allow_select && h.tx_mode == av1.TX_MODE_SELECT {
		max_tx_w := int(av1.TX_WIDTH[max_rect_tx_size])
		max_tx_h := int(av1.TX_HEIGHT[max_rect_tx_size])
		above_w := 0
		if t.avail_u {
			u_idx := (t.mi_row - 1) * h.mi_cols + t.mi_col
			if t.grid.cells[u_idx].is_inter {
				above_w = int(av1.BLOCK_WIDTH[t.grid.cells[u_idx].mi_size])
			} else {
				above_w = get_above_tx_width(t, t.mi_row, t.mi_col)
			}
		}
		left_h := 0
		if t.avail_l {
			l_idx := t.mi_row * h.mi_cols + (t.mi_col - 1)
			if t.grid.cells[l_idx].is_inter {
				left_h = int(av1.BLOCK_HEIGHT[t.grid.cells[l_idx].mi_size])
			} else {
				left_h = get_left_tx_height(t, t.mi_row, t.mi_col)
			}
		}
		ctx := int(above_w >= max_tx_w) + int(left_h >= max_tx_h)
		tx_depth := int(t.tx_size) == max_rect_tx_size ? 0 : 1
		switch max_tx_depth {
		case 4: put_symbol(&t.sym, tx_depth, t.cdfs.tx_64x64_cdf[ctx][:])
		case 3: put_symbol(&t.sym, tx_depth, t.cdfs.tx_32x32_cdf[ctx][:])
		case 2: put_symbol(&t.sym, tx_depth, t.cdfs.tx_16x16_cdf[ctx][:])
		case:   put_symbol(&t.sym, tx_depth, t.cdfs.tx_8x8_cdf[ctx][:])
		}
	}
}

@(private)
write_var_tx_size :: proc(t: ^Tile, row, col, tx_sz, depth: int) {
	e := t.e
	h := &e.header
	if row >= h.mi_rows || col >= h.mi_cols do return
	txfm_split := int(t.tx_size) != tx_sz ? 1 : 0
	if tx_sz != av1.TX_4X4 && depth < av1.MAX_VARTX_DEPTH {
		above := int(get_above_tx_width(t, row, col) < int(av1.TX_WIDTH[tx_sz]))
		left := int(get_left_tx_height(t, row, col) < int(av1.TX_HEIGHT[tx_sz]))
		sz := min(64, int(max(av1.BLOCK_WIDTH[t.mi_size], av1.BLOCK_HEIGHT[t.mi_size])))
		max_tx_sz := av1.find_tx_size(sz, sz)
		tx_sz_sqr_up := int(av1.TX_SIZE_SQR_UP[tx_sz])
		ctx := int(tx_sz_sqr_up != max_tx_sz) * 3 + (av1.TX_SIZES - 1 - max_tx_sz) * 6 + above + left
		put_symbol(&t.sym, txfm_split, t.cdfs.txfm_split_cdf[ctx][:])
	}
	w4 := int(av1.TX_WIDTH[tx_sz]) / av1.MI_SIZE
	h4 := int(av1.TX_HEIGHT[tx_sz]) / av1.MI_SIZE
	if txfm_split != 0 {
		sub_tx_sz := int(av1.SPLIT_TX_SIZE[tx_sz])
		step_w := int(av1.TX_WIDTH[sub_tx_sz]) / av1.MI_SIZE
		step_h := int(av1.TX_HEIGHT[sub_tx_sz]) / av1.MI_SIZE
		for i := 0; i < h4; i += step_h {
			for j := 0; j < w4; j += step_w {
				write_var_tx_size(t, row + i, col + j, sub_tx_sz, depth + 1)
			}
		}
	} else {
		for i in 0 ..< h4 {
			if row + i >= h.mi_rows do break
			for j in 0 ..< w4 {
				if col + j >= h.mi_cols do break
				e.tile_cells[(row + i) * h.mi_cols + (col + j)].inter_tx_size = u8(tx_sz)
			}
		}
	}
}

// Section 5.11.34: Emit residual syntax.
@(private)
write_residual :: proc(t: ^Tile) {
	e := t.e
	h := &e.header
	num_planes := 1 + (t.has_chroma ? 2 : 0)

	for plane in 0 ..< num_planes {
		sub_x := plane > 0 ? uint(e.ss_x) : 0
		sub_y := plane > 0 ? uint(e.ss_y) : 0
		tx_sz := av1.get_uv_tx_size(t.mi_size, plane, int(t.tx_size), t.lossless, u8(e.ss_x), u8(e.ss_y))
		step_x := int(av1.TX_WIDTH[tx_sz]) >> 2
		step_y := int(av1.TX_HEIGHT[tx_sz]) >> 2
		w := int(av1.TX_WIDTH[tx_sz])
		bh := int(av1.TX_HEIGHT[tx_sz])
		seg_eob := (tx_sz == av1.TX_16X64 || tx_sz == av1.TX_64X16) ? 512 : min(1024, w * bh)

		plane_sz := av1.get_plane_residual_size(t.mi_size, plane, sub_x, sub_y)
		num4x4_w := int(av1.NUM_4X4_BLOCKS_WIDE[plane_sz])
		num4x4_h := int(av1.NUM_4X4_BLOCKS_HIGH[plane_sz])
		base_x := (t.mi_col >> sub_x) * av1.MI_SIZE
		base_y := (t.mi_row >> sub_y) * av1.MI_SIZE
		max_x := (h.mi_cols * av1.MI_SIZE) >> sub_x
		max_y := (h.mi_rows * av1.MI_SIZE) >> sub_y

		tb_idx := 0
		coeff_off := 0
		for y := 0; y < num4x4_h; y += step_y {
			for x := 0; x < num4x4_w; x += step_x {
				start_x := base_x + 4 * x
				start_y := base_y + 4 * y
				if start_x >= max_x || start_y >= max_y do continue
				eob := int(t.blk_eobs[plane][tb_idx])
				q_slice := t.blk_coeffs[plane][coeff_off : coeff_off + seg_eob]
				write_coeffs(t, plane, start_x, start_y, tx_sz, q_slice, eob)
				tb_idx += 1
				coeff_off += seg_eob
			}
		}
	}
}

@(private)
tx_type_to_symbol :: proc "contextless" (set: int, is_inter: bool, tx_type: u8) -> int {
	if is_inter {
		switch set {
		case av1.TX_SET_INTER_1:
			for v, i in av1.TX_TYPE_INTER_INV_SET1 {
				if v == tx_type do return i
			}
		case av1.TX_SET_INTER_2:
			for v, i in av1.TX_TYPE_INTER_INV_SET2 {
				if v == tx_type do return i
			}
		case:
			for v, i in av1.TX_TYPE_INTER_INV_SET3 {
				if v == tx_type do return i
			}
		}
	} else {
		if set == av1.TX_SET_INTRA_1 {
			for v, i in av1.TX_TYPE_INTRA_INV_SET1 {
				if v == tx_type do return i
			}
		} else {
			for v, i in av1.TX_TYPE_INTRA_INV_SET2 {
				if v == tx_type do return i
			}
		}
	}
	return 0
}

@(private)
write_transform_type :: proc(t: ^Tile, x4, y4, tx_sz: int, tx_type: u8) #no_bounds_check {
	e := t.e
	h := &e.header
	set := av1.get_tx_set(tx_sz, t.is_inter, h.reduced_tx_set)
	q_idx := h.segmentation_enabled ? get_qindex(h, int(t.segment_id)) : int(h.base_q_idx)
	if set > 0 && q_idx > 0 {
		tx_sz_sqr := int(av1.TX_SIZE_SQR[tx_sz])
		sym := tx_type_to_symbol(set, t.is_inter, tx_type)
		if t.is_inter {
			switch set {
			case av1.TX_SET_INTER_1: put_symbol(&t.sym, sym, t.cdfs.inter_tx_type_set1_cdf[tx_sz_sqr][:])
			case av1.TX_SET_INTER_2: put_symbol(&t.sym, sym, t.cdfs.inter_tx_type_set2_cdf[:])
			case:                    put_symbol(&t.sym, sym, t.cdfs.inter_tx_type_set3_cdf[tx_sz_sqr][:])
			}
		} else {
			if set == av1.TX_SET_INTRA_1 {
				put_symbol(&t.sym, sym, t.cdfs.intra_tx_type_set1_cdf[tx_sz_sqr][t.y_mode][:])
			} else {
				put_symbol(&t.sym, sym, t.cdfs.intra_tx_type_set2_cdf[tx_sz_sqr][t.y_mode][:])
			}
		}
	}
	w4 := int(av1.TX_WIDTH[tx_sz]) >> 2
	h4 := int(av1.TX_HEIGHT[tx_sz]) >> 2
	for j in 0 ..< h4 {
		for i in 0 ..< w4 {
			if y4 + j < h.mi_rows && x4 + i < h.mi_cols {
				e.tile_cells[(y4 + j) * h.mi_cols + (x4 + i)].tx_type = tx_type
			}
		}
	}
}

@(private)
get_coeff_base_ctx :: proc "contextless" (
	t: ^Tile,
	tx_sz, plane, block_x, block_y, pos, c: int,
	is_eob: bool,
) -> int #no_bounds_check {
	adj_tx_sz := int(av1.ADJUSTED_TX_SIZE[tx_sz])
	bwl := uint(av1.TX_WIDTH_LOG2[adj_tx_sz])
	width := 1 << bwl
	height := int(av1.TX_HEIGHT[adj_tx_sz])
	tx_type := compute_tx_type(t, plane, tx_sz, block_x, block_y)
	if is_eob {
		if c == 0 do return av1.SIG_COEF_CONTEXTS - 4
		if c <= (height << bwl) / 8 do return av1.SIG_COEF_CONTEXTS - 3
		if c <= (height << bwl) / 4 do return av1.SIG_COEF_CONTEXTS - 2
		return av1.SIG_COEF_CONTEXTS - 1
	}
	tx_class := av1.get_tx_class(int(tx_type))
	row := pos >> bwl
	col := pos - (row << bwl)
	mag := 0
	for idx in 0 ..< av1.SIG_REF_DIFF_OFFSET_NUM {
		ref_row := row + int(av1.SIG_REF_DIFF_OFFSET[tx_class][idx][0])
		ref_col := col + int(av1.SIG_REF_DIFF_OFFSET[tx_class][idx][1])
		if ref_row >= 0 && ref_col >= 0 && ref_row < height && ref_col < width {
			mag += min(int(abs(t.quant_buf[(ref_row << bwl) + ref_col])), 3)
		}
	}
	ctx := min((mag + 1) >> 1, 4)
	if tx_class == av1.TX_CLASS_2D {
		if row == 0 && col == 0 {
			return 0
		}
		return ctx + int(av1.COEFF_BASE_CTX_OFFSET[tx_sz][min(row, 4)][min(col, 4)])
	}
	idx := tx_class == av1.TX_CLASS_VERT ? row : col
	return ctx + int(av1.COEFF_BASE_POS_CTX_OFFSET[min(idx, 2)])
}

@(private)
get_coeff_br_ctx :: proc "contextless" (
	t: ^Tile,
	tx_sz, plane, block_x, block_y, pos: int,
) -> int #no_bounds_check {
	adj_tx_sz := int(av1.ADJUSTED_TX_SIZE[tx_sz])
	bwl := uint(av1.TX_WIDTH_LOG2[adj_tx_sz])
	txw := int(av1.TX_WIDTH[adj_tx_sz])
	txh := int(av1.TX_HEIGHT[adj_tx_sz])
	row := pos >> bwl
	col := pos - (row << bwl)
	mag := 0
	tx_type := compute_tx_type(t, plane, tx_sz, block_x, block_y)
	tx_class := av1.get_tx_class(int(tx_type))
	for idx in 0 ..< 3 {
		ref_row := row + int(av1.MAG_REF_OFFSET_WITH_TX_CLASS[tx_class][idx][0])
		ref_col := col + int(av1.MAG_REF_OFFSET_WITH_TX_CLASS[tx_class][idx][1])
		if ref_row >= 0 && ref_col >= 0 && ref_row < txh && ref_col < (1 << bwl) {
			mag += min(int(t.quant_buf[ref_row * txw + ref_col]), av1.COEFF_BASE_RANGE + av1.NUM_BASE_LEVELS + 1)
		}
	}
	mag = min((mag + 1) >> 1, 6)
	if pos == 0 {
		return mag
	}
	if tx_class == av1.TX_CLASS_2D {
		return (row < 2 && col < 2) ? (mag + 7) : (mag + 14)
	}
	if tx_class == av1.TX_CLASS_HORIZ {
		return col == 0 ? (mag + 7) : (mag + 14)
	}
	return row == 0 ? (mag + 7) : (mag + 14)
}

// Section 5.11.39: Emit coefficients syntax.
@(private)
write_coeffs :: proc(t: ^Tile, plane, start_x, start_y, tx_sz: int, q_in: []i32, eob: int) #no_bounds_check {
	e := t.e
	h := &e.header
	x4 := start_x >> 2
	y4 := start_y >> 2
	w4 := int(av1.TX_WIDTH[tx_sz]) >> 2
	h4 := int(av1.TX_HEIGHT[tx_sz]) >> 2
	tx_sz_ctx := (int(av1.TX_SIZE_SQR[tx_sz]) + int(av1.TX_SIZE_SQR_UP[tx_sz]) + 1) >> 1
	ptype := plane > 0 ? 1 : 0

	sub_x := plane > 0 ? uint(e.ss_x) : 0
	sub_y := plane > 0 ? uint(e.ss_y) : 0
	max_x4 := h.mi_cols >> sub_x
	max_y4 := h.mi_rows >> sub_y
	tile_x4 := t.mi_col_start >> sub_x

	w := int(av1.TX_WIDTH[tx_sz])
	bh := int(av1.TX_HEIGHT[tx_sz])
	bsize := av1.get_plane_residual_size(t.mi_size, plane, sub_x, sub_y)
	bw := int(av1.BLOCK_WIDTH[bsize])
	bbh := int(av1.BLOCK_HEIGHT[bsize])

	skip_ctx := 0
	if plane == 0 {
		top := 0
		left := 0
		for k in 0 ..< w4 {
			if x4 + k < max_x4 {
				top = max(top, int(t.above_level_ctx[plane][x4 + k - tile_x4]))
			}
		}
		for k in 0 ..< h4 {
			if y4 + k < max_y4 {
				left = max(left, int(t.left_level_ctx[plane][(y4 + k) & 63]))
			}
		}
		top = min(top, 255)
		left = min(left, 255)
		if bw == w && bbh == bh {
			skip_ctx = 0
		} else if top == 0 && left == 0 {
			skip_ctx = 1
		} else if top == 0 || left == 0 {
			skip_ctx = 2 + (max(top, left) > 3 ? 1 : 0)
		} else if max(top, left) <= 3 {
			skip_ctx = 4
		} else if min(top, left) <= 3 {
			skip_ctx = 5
		} else {
			skip_ctx = 6
		}
	} else {
		above := u8(0)
		left := u8(0)
		for i in 0 ..< w4 {
			if x4 + i < max_x4 {
				above |= t.above_level_ctx[plane][x4 + i - tile_x4]
				above |= t.above_dc_ctx[plane][x4 + i - tile_x4]
			}
		}
		for i in 0 ..< h4 {
			if y4 + i < max_y4 {
				left |= t.left_level_ctx[plane][(y4 + i) & 63]
				left |= t.left_dc_ctx[plane][(y4 + i) & 63]
			}
		}
		skip_ctx = (above != 0 ? 1 : 0) + (left != 0 ? 1 : 0) + 7
		if bw * bbh > w * bh {
			skip_ctx += 3
		}
	}

	cul_level := 0
	dc_category := u8(0)

	all_zero := eob == 0
	put_symbol(&t.sym, int(all_zero), t.cdfs.txb_skip_cdf[tx_sz_ctx][skip_ctx][:])
	if all_zero {
		if plane == 0 {
			for j in 0 ..< h4 {
				for i in 0 ..< w4 {
					if y4 + j < h.mi_rows && x4 + i < h.mi_cols {
						e.tile_cells[(y4 + j) * h.mi_cols + (x4 + i)].tx_type = av1.DCT_DCT
					}
				}
			}
		}
	} else {
		if plane == 0 {
			write_transform_type(t, x4, y4, tx_sz, e.tile_cells[y4 * h.mi_cols + x4].tx_type)
		}
		t.plane_tx_type = compute_tx_type(t, plane, tx_sz, x4, y4)
		scan := av1.get_scan(tx_sz, int(t.plane_tx_type))
		eob_ctx := av1.get_tx_class(int(t.plane_tx_type)) == av1.TX_CLASS_2D ? 0 : 1
		eob_multisize := min(int(av1.TX_WIDTH_LOG2[tx_sz]), 5) + min(int(av1.TX_HEIGHT_LOG2[tx_sz]), 5) - 4

		eob_pt: int
		eob_shift := 0
		if eob == 1 {
			eob_pt = 1
		} else if eob == 2 {
			eob_pt = 2
		} else {
			v := eob - 1
			eob_shift = int(31 - intrinsics.count_leading_zeros(u32(v)))
			eob_pt = eob_shift + 2
		}

		switch eob_multisize {
		case 0: put_symbol(&t.sym, eob_pt - 1, t.cdfs.eob_pt_16_cdf[ptype][eob_ctx][:])
		case 1: put_symbol(&t.sym, eob_pt - 1, t.cdfs.eob_pt_32_cdf[ptype][eob_ctx][:])
		case 2: put_symbol(&t.sym, eob_pt - 1, t.cdfs.eob_pt_64_cdf[ptype][eob_ctx][:])
		case 3: put_symbol(&t.sym, eob_pt - 1, t.cdfs.eob_pt_128_cdf[ptype][eob_ctx][:])
		case 4: put_symbol(&t.sym, eob_pt - 1, t.cdfs.eob_pt_256_cdf[ptype][eob_ctx][:])
		case 5: put_symbol(&t.sym, eob_pt - 1, t.cdfs.eob_pt_512_cdf[ptype][:])
		case:   put_symbol(&t.sym, eob_pt - 1, t.cdfs.eob_pt_1024_cdf[ptype][:])
		}

		if eob_shift > 0 {
			rem := (eob - 1) - (1 << uint(eob_shift))
			eob_extra := (rem >> uint(eob_shift - 1)) & 1
			put_symbol(&t.sym, eob_extra, t.cdfs.eob_extra_cdf[tx_sz_ctx][ptype][eob_pt - 3][:])
			for i in 1 ..< eob_shift {
				bit := (rem >> uint(eob_shift - 1 - i)) & 1
				av1.write_bool(&t.sym, bit)
			}
		}

		for c := eob - 1; c >= 0; c -= 1 {
			pos := int(scan[c])
			abs_q := int(abs(q_in[pos]))
			base_level := min(abs_q, av1.NUM_BASE_LEVELS + 1)
			if c == eob - 1 {
				base_eob_ctx := get_coeff_base_ctx(t, tx_sz, plane, x4, y4, pos, c, true) - av1.SIG_COEF_CONTEXTS + av1.SIG_COEF_CONTEXTS_EOB
				put_symbol(&t.sym, base_level - 1, t.cdfs.coeff_base_eob_cdf[tx_sz_ctx][ptype][base_eob_ctx][:])
			} else {
				base_ctx := get_coeff_base_ctx(t, tx_sz, plane, x4, y4, pos, c, false)
				put_symbol(&t.sym, base_level, t.cdfs.coeff_base_cdf[tx_sz_ctx][ptype][base_ctx][:])
			}
			level := base_level
			if abs_q > av1.NUM_BASE_LEVELS {
				br_ctx := get_coeff_br_ctx(t, tx_sz, plane, x4, y4, pos)
				br_tx_ctx := min(tx_sz_ctx, av1.TX_32X32)
				rem_br := abs_q - (av1.NUM_BASE_LEVELS + 1)
				for idx := 0; idx < av1.COEFF_BASE_RANGE / (av1.BR_CDF_SIZE - 1); idx += 1 {
					coeff_br := min(rem_br, av1.BR_CDF_SIZE - 1)
					put_symbol(&t.sym, coeff_br, t.cdfs.coeff_br_cdf[br_tx_ctx][ptype][br_ctx][:])
					level += coeff_br
					rem_br -= coeff_br
					if coeff_br < av1.BR_CDF_SIZE - 1 {
						break
					}
				}
			}
			t.quant_buf[pos] = i32(level)
		}

		for c in 0 ..< eob {
			pos := int(scan[c])
			t.quant_buf[pos] = 0
			qv := q_in[pos]
			abs_q := int(abs(qv))
			sign := qv < 0 ? 1 : 0
			if abs_q != 0 {
				if c == 0 {
					dc_sign_sum := 0
					for k in 0 ..< w4 {
						if x4 + k < max_x4 {
							s := t.above_dc_ctx[plane][x4 + k - tile_x4]
							if s == 1 {
								dc_sign_sum -= 1
							} else if s == 2 {
								dc_sign_sum += 1
							}
						}
					}
					for k in 0 ..< h4 {
						if y4 + k < max_y4 {
							s := t.left_dc_ctx[plane][(y4 + k) & 63]
							if s == 1 {
								dc_sign_sum -= 1
							} else if s == 2 {
								dc_sign_sum += 1
							}
						}
					}
					dc_ctx := dc_sign_sum < 0 ? 1 : (dc_sign_sum > 0 ? 2 : 0)
					put_symbol(&t.sym, sign, t.cdfs.dc_sign_cdf[ptype][dc_ctx][:])
				} else {
					av1.write_bool(&t.sym, sign)
				}
			}
			if abs_q > av1.NUM_BASE_LEVELS + av1.COEFF_BASE_RANGE {
				rem := abs_q - (av1.NUM_BASE_LEVELS + av1.COEFF_BASE_RANGE)
				length := int(32 - intrinsics.count_leading_zeros(u32(rem)))
				for _ in 0 ..< length - 1 {
					av1.write_bool(&t.sym, 0)
				}
				av1.write_bool(&t.sym, 1)
				for i := length - 2; i >= 0; i -= 1 {
					av1.write_bool(&t.sym, (rem >> uint(i)) & 1)
				}
			}
			if pos == 0 && abs_q > 0 {
				dc_category = sign != 0 ? 1 : 2
			}
			cul_level += abs_q
		}
		cul_level = min(63, cul_level)
	}

	for i in 0 ..< w4 {
		t.above_level_ctx[plane][x4 + i - tile_x4] = u8(cul_level)
		t.above_dc_ctx[plane][x4 + i - tile_x4] = dc_category
	}
	for i in 0 ..< h4 {
		t.left_level_ctx[plane][(y4 + i) & 63] = u8(cul_level)
		t.left_dc_ctx[plane][(y4 + i) & 63] = dc_category
	}
}
