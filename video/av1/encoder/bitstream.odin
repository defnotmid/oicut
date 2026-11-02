package av1_encoder

// AV1 OBU, sequence header, uncompressed frame header, tile group, and av1C
// configuration record bitstream emission (AV1 Specification sections 5.3-5.11).

import av1 ".."

@(private)
Raw_Writer :: struct {
	data:   ^[dynamic]u8,
	bits:   int,
	failed: bool,
}

@(private)
write_bits :: proc(w: ^Raw_Writer, value, n: int) {
	for i := n - 1; i >= 0; i -= 1 {
		if (w.bits & 7) == 0 {
			if _, err := append(w.data, 0); err != nil {
				w.failed = true
				return
			}
		}
		bit := u8((value >> uint(i)) & 1)
		w.data^[len(w.data^) - 1] |= bit << uint(7 - (w.bits & 7))
		w.bits += 1
	}
}

@(private)
write_flag :: #force_inline proc(w: ^Raw_Writer, b: bool) {
	write_bits(w, int(b), 1)
}

@(private)
write_su :: proc(w: ^Raw_Writer, value: i32, n: int) {
	mask := (1 << uint(n)) - 1
	write_bits(w, int(value) & mask, n)
}

@(private)
trailing_bits :: proc(w: ^Raw_Writer) {
	write_bits(w, 1, 1)
	for (w.bits & 7) != 0 {
		write_bits(w, 0, 1)
	}
}

@(private)
byte_align :: proc(w: ^Raw_Writer) {
	for (w.bits & 7) != 0 {
		write_bits(w, 0, 1)
	}
}

@(private)
append_leb128 :: proc(dst: ^[dynamic]u8, val: int) -> bool {
	v := u64(val)
	for {
		b := u8(v & 0x7F)
		v >>= 7
		if v != 0 do b |= 0x80
		if _, err := append(dst, b); err != nil do return false
		if v == 0 do break
	}
	return true
}

@(private)
append_obu :: proc(dst: ^[dynamic]u8, obu_type: av1.Obu_Type, payload: []u8) -> bool {
	// OBU header: forbidden=0 (1), type (4), extension=0 (1), has_size=1 (1), reserved=0 (1).
	hdr := (u8(obu_type) << 3) | 0x02
	if _, err := append(dst, hdr); err != nil do return false
	if !append_leb128(dst, len(payload)) do return false
	if _, err := append(dst, ..payload); err != nil do return false
	return true
}

@(private)
tile_log2 :: proc(block, target: int) -> int {
	k := 0
	for (block << uint(k)) < target do k += 1
	return k
}

@(private)
init_sequence_header :: proc(e: ^Encoder) {
	s := &e.seq
	s^ = {
		seq_profile                    = u8(e.profile),
		seq_level_idx                  = 13, // Level 5.1
		frame_width_bits_minus_1       = 15,
		frame_height_bits_minus_1      = 15,
		max_frame_width_minus_1        = u16(e.config.width - 1),
		max_frame_height_minus_1       = u16(e.config.height - 1),
		use_128x128_superblock         = false,
		enable_filter_intra            = false,
		enable_intra_edge_filter       = true,
		enable_interintra_compound     = false,
		enable_masked_compound         = false,
		enable_warped_motion           = false,
		enable_dual_filter             = false,
		enable_order_hint              = true,
		enable_jnt_comp                = false,
		enable_ref_frame_mvs           = true,
		seq_force_screen_content_tools = 0,
		seq_force_integer_mv           = av1.SELECT_INTEGER_MV,
		order_hint_bits_minus_1        = 7,
		order_hint_bits                = 8,
		enable_superres                = false,
		enable_cdef                    = !e.config.disable_cdef,
		enable_restoration             = false,
		color = {
			bit_depth                = u8(e.bit_depth),
			mono_chrome              = e.mono_chrome,
			color_primaries          = 2,
			transfer_characteristics = 2,
			matrix_coefficients      = 2,
			color_range              = false,
			subsampling_x            = u8(e.ss_x),
			subsampling_y            = u8(e.ss_y),
			separate_uv_delta_q      = !e.mono_chrome,
		},
	}
}

@(private)
setup_tile_info :: proc(e: ^Encoder) {
	MAX_TILE_WIDTH :: 4096
	MAX_TILE_AREA :: 4096 * 2304
	sb_cols := e.sb_cols
	sb_rows := e.sb_rows
	max_tile_width_sb := MAX_TILE_WIDTH >> 6
	max_tile_area_sb := MAX_TILE_AREA >> 12
	min_log2_cols := tile_log2(max_tile_width_sb, sb_cols)
	max_log2_cols := tile_log2(1, min(sb_cols, av1.MAX_TILE_COLS))
	max_log2_rows := tile_log2(1, min(sb_rows, av1.MAX_TILE_ROWS))
	min_log2_tiles := max(min_log2_cols, tile_log2(max_tile_area_sb, sb_rows * sb_cols))

	t := &e.header.tile_info
	t.uniform_tile_spacing_flag = true
	t.tile_cols_log2 = clamp(e.config.tile_cols_log2, min_log2_cols, max_log2_cols)
	e.config.tile_cols_log2 = t.tile_cols_log2

	width_sb := (sb_cols + (1 << uint(t.tile_cols_log2)) - 1) >> uint(t.tile_cols_log2)
	i := 0
	for start := 0; start < sb_cols; start += width_sb {
		t.mi_col_starts[i] = u16(start << 4)
		t.width_in_sbs_minus_1[i] = u16(min(width_sb, sb_cols - start) - 1)
		i += 1
	}
	t.mi_col_starts[i] = u16(e.mi_cols)
	t.tile_cols = i

	min_log2_rows := max(min_log2_tiles - t.tile_cols_log2, 0)
	t.tile_rows_log2 = clamp(e.config.tile_rows_log2, min_log2_rows, max_log2_rows)
	e.config.tile_rows_log2 = t.tile_rows_log2

	height_sb := (sb_rows + (1 << uint(t.tile_rows_log2)) - 1) >> uint(t.tile_rows_log2)
	i = 0
	for start := 0; start < sb_rows; start += height_sb {
		t.mi_row_starts[i] = u16(start << 4)
		t.height_in_sbs_minus_1[i] = u16(min(height_sb, sb_rows - start) - 1)
		i += 1
	}
	t.mi_row_starts[i] = u16(e.mi_rows)
	t.tile_rows = i
	t.context_update_tile_id = 0
	t.tile_size_bytes = 4
}

// Emits the sequence header OBU into `e.seq_obu` (Section 5.5).
@(private)
build_sequence_header_obu :: proc(e: ^Encoder) -> bool {
	clear(&e.header_buf)
	w := Raw_Writer{data = &e.header_buf}
	s := &e.seq
	c := &s.color

	write_bits(&w, int(s.seq_profile), 3)
	write_flag(&w, false) // still_picture
	write_flag(&w, false) // reduced_still_picture_header
	write_flag(&w, false) // timing_info_present_flag
	write_flag(&w, false) // initial_display_delay_present_flag
	write_bits(&w, 0, 5)  // operating_points_cnt_minus_1 = 0
	write_bits(&w, 0, 12) // operating_point_idc[0] = 0
	write_bits(&w, int(s.seq_level_idx), 5)
	if s.seq_level_idx > 7 {
		write_flag(&w, false) // seq_tier[0] = 0
	}

	write_bits(&w, int(s.frame_width_bits_minus_1), 4)
	write_bits(&w, int(s.frame_height_bits_minus_1), 4)
	write_bits(&w, int(s.max_frame_width_minus_1), int(s.frame_width_bits_minus_1) + 1)
	write_bits(&w, int(s.max_frame_height_minus_1), int(s.frame_height_bits_minus_1) + 1)
	write_flag(&w, false) // frame_id_numbers_present_flag
	write_flag(&w, s.use_128x128_superblock)
	write_flag(&w, s.enable_filter_intra)
	write_flag(&w, s.enable_intra_edge_filter)
	write_flag(&w, s.enable_interintra_compound)
	write_flag(&w, s.enable_masked_compound)
	write_flag(&w, s.enable_warped_motion)
	write_flag(&w, s.enable_dual_filter)
	write_flag(&w, s.enable_order_hint)
	if s.enable_order_hint {
		write_flag(&w, s.enable_jnt_comp)
		write_flag(&w, s.enable_ref_frame_mvs)
	}
	write_flag(&w, false) // seq_choose_screen_content_tools = 0
	write_bits(&w, 0, 1)  // seq_force_screen_content_tools = 0
	if s.enable_order_hint {
		write_bits(&w, int(s.order_hint_bits_minus_1), 3)
	}
	write_flag(&w, s.enable_superres)
	write_flag(&w, s.enable_cdef)
	write_flag(&w, s.enable_restoration)

	// color_config (Section 5.5.2)
	write_flag(&w, c.bit_depth > 8)
	if s.seq_profile == 2 && c.bit_depth > 8 {
		write_flag(&w, c.bit_depth == 12)
	}
	if s.seq_profile != 1 {
		write_flag(&w, c.mono_chrome)
	}
	write_flag(&w, false) // color_description_present_flag
	write_flag(&w, c.color_range)
	if !c.mono_chrome {
		if s.seq_profile == 2 && c.bit_depth == 12 {
			write_bits(&w, int(c.subsampling_x), 1)
			if c.subsampling_x == 1 {
				write_bits(&w, int(c.subsampling_y), 1)
			}
		}
		if c.subsampling_x == 1 && c.subsampling_y == 1 {
			write_bits(&w, int(c.chroma_sample_position), 2)
		}
		write_flag(&w, c.separate_uv_delta_q)
	}
	write_flag(&w, false) // film_grain_params_present
	trailing_bits(&w)
	if w.failed do return false

	clear(&e.seq_obu)
	return append_obu(&e.seq_obu, .Sequence_Header, e.header_buf[:])
}

// Builds the 4-byte AV1CodecConfigurationRecord header followed by the
// sequence header OBU (for MP4 `av1C` and WebM `V_AV1` `CodecPrivate`).
build_av1c :: proc(e: ^Encoder, out: ^[dynamic]u8) -> bool {
	if e == nil || !e.initialized do return false
	s := &e.seq
	c := &s.color
	clear(out)
	b0 := u8(0x81) // marker = 1, version = 1
	b1 := (s.seq_profile << 5) | (s.seq_level_idx & 0x1F)
	b2 := (u8(c.bit_depth > 8) << 6) |
		(u8(c.bit_depth == 12) << 5) |
		(u8(c.mono_chrome) << 4) |
		((c.subsampling_x & 1) << 3) |
		((c.subsampling_y & 1) << 2) |
		(c.chroma_sample_position & 3)
	b3 := u8(0)
	if _, err := append(out, b0, b1, b2, b3); err != nil do return false
	if _, err := append(out, ..e.seq_obu[:]); err != nil do return false
	return true
}

@(private)
prepare_frame_header :: proc(e: ^Encoder, is_altref: bool) {
	t_info := e.header.tile_info
	h := &e.header
	h^ = {tile_info = t_info}
	s := &e.seq

	h.frame_type = e.keyframe ? .Key : .Inter
	h.frame_is_intra = e.keyframe
	h.show_frame = e.show_frame
	h.showable_frame = e.show_frame ? !e.keyframe : true
	h.error_resilient_mode = e.keyframe && e.show_frame
	h.disable_cdf_update = false
	h.allow_screen_content_tools = false
	h.force_integer_mv = h.frame_is_intra
	h.frame_size_override_flag = false
	h.order_hint = e.order_hint

	if h.frame_is_intra || h.error_resilient_mode {
		h.primary_ref_frame = av1.PRIMARY_REF_NONE
	} else {
		slot := clamp(e.primary_ref_slot, 0, 2)
		if slot == 2 && !e.has_altref do slot = 1
		e.primary_ref_slot = slot
		h.primary_ref_frame = slot == 2 ? (av1.ALTREF_FRAME - av1.LAST_FRAME) : (slot == 1 ? (av1.GOLDEN_FRAME - av1.LAST_FRAME) : 0)
	}

	h.refresh_frame_flags = e.refresh_frame_flags
	h.frame_width = e.config.width
	h.frame_height = e.config.height
	h.upscaled_width = e.config.width
	h.render_width = e.config.width
	h.render_height = e.config.height
	h.mi_cols = e.mi_cols
	h.mi_rows = e.mi_rows

	if !h.frame_is_intra {
		arf_slot := u8(e.has_altref ? 2 : 1)
		h.ref_frame_idx = {0, 0, 0, 1, arf_slot, arf_slot, arf_slot}
		for i in 0 ..< av1.REFS_PER_FRAME {
			hint := e.slot_order_hints[h.ref_frame_idx[i]]
			h.order_hints[av1.LAST_FRAME + i] = hint
			h.ref_frame_sign_bias[av1.LAST_FRAME + i] = s.enable_order_hint &&
				av1.get_relative_dist(int(hint), int(h.order_hint), s.enable_order_hint, u8(s.order_hint_bits)) > 0
		}
	}

	h.base_q_idx = u8(clamp(e.q_idx, 0, 255))
	h.delta_q_y_dc = i8(e.config.delta_q_y_dc)
	if !e.mono_chrome {
		h.delta_q_u_dc = i8(e.config.delta_q_u_dc)
		h.delta_q_u_ac = i8(e.config.delta_q_u_ac)
		h.delta_q_v_dc = i8(e.config.delta_q_v_dc)
		h.delta_q_v_ac = i8(e.config.delta_q_v_ac)
		h.diff_uv_delta = h.delta_q_u_dc != h.delta_q_v_dc || h.delta_q_u_ac != h.delta_q_v_ac
	}

	h.segmentation_enabled = e.segmentation_enabled
	if e.segmentation_enabled {
		h.feature_enabled = e.feature_enabled
		h.feature_data = e.feature_data
		if h.primary_ref_frame == av1.PRIMARY_REF_NONE {
			h.segmentation_update_map = true
			h.segmentation_update_data = true
		} else {
			h.segmentation_update_map = e.segmentation_update_map
			h.segmentation_temporal_update = e.segmentation_update_map && e.segmentation_temporal_update
			h.segmentation_update_data = e.feature_enabled != e.last_seg_enabled || e.feature_data != e.last_seg_data
		}
	}

	e.seg_id_pre_skip = false
	e.last_active_seg_id = 0
	if h.segmentation_enabled {
		for i in 0 ..< av1.MAX_SEGMENTS {
			for j in 0 ..< av1.SEG_LVL_MAX {
				if (h.feature_enabled[i] & (u8(1) << uint(j))) != 0 {
					e.last_active_seg_id = i
					if j >= av1.SEG_LVL_REF_FRAME {
						e.seg_id_pre_skip = true
					}
				}
			}
		}
	}

	h.coded_lossless = true
	for seg in 0 ..< av1.MAX_SEGMENTS {
		q := int(h.base_q_idx)
		if h.segmentation_enabled && (h.feature_enabled[seg] & (1 << av1.SEG_LVL_ALT_Q)) != 0 {
			q = clamp(q + int(h.feature_data[seg][av1.SEG_LVL_ALT_Q]), 0, 255)
		}
		seg_lossless := q == 0 &&
			h.delta_q_y_dc == 0 &&
			h.delta_q_u_ac == 0 &&
			h.delta_q_u_dc == 0 &&
			h.delta_q_v_ac == 0 &&
			h.delta_q_v_dc == 0
		e.lossless_array[seg] = seg_lossless
		if !seg_lossless do h.coded_lossless = false
	}
	h.all_lossless = h.coded_lossless
	e.lossless = h.coded_lossless

	e.allow_hp = !e.config.ultrafast && !h.frame_is_intra && !e.lossless && e.q_idx < 64
	if !h.frame_is_intra {
		h.allow_high_precision_mv = e.allow_hp
		h.is_filter_switchable = false
		h.interpolation_filter = av1.EIGHTTAP_REGULAR
		h.is_motion_mode_switchable = false
		h.use_ref_frame_mvs = !e.config.ultrafast && !e.last_keyframe && e.last_show_frame && s.enable_ref_frame_mvs
	}
	h.disable_frame_end_update_cdf = false

	// Loop filter parameters (Section 5.9.11).
	h.loop_filter_ref_deltas = {1, 0, 0, 0, -1, 0, -1, -1}
	h.loop_filter_mode_deltas = {}
	if !h.coded_lossless && !e.config.disable_loop_filter {
		lvl := u8(clamp(e.q_idx >> 2, 0, av1.MAX_LOOP_FILTER))
		uv_lvl := u8(clamp((e.q_idx * 3) >> 4, 0, av1.MAX_LOOP_FILTER))
		h.loop_filter_level = {lvl, lvl, e.mono_chrome ? 0 : uv_lvl, e.mono_chrome ? 0 : uv_lvl}
		h.loop_filter_sharpness = 0
		h.loop_filter_delta_enabled = true
		h.loop_filter_delta_update = false
	}

	// CDEF parameters (Section 5.9.19).
	if !h.coded_lossless && s.enable_cdef {
		h.cdef_damping_minus_3 = u8(clamp((e.q_idx >> 6), 0, 3))
		h.cdef_bits = 0
		pri := u8(clamp(e.q_idx >> 5, 0, 15))
		sec := u8(pri > 0 ? 1 : 0)
		h.cdef_y_pri_strength[0] = pri
		h.cdef_y_sec_strength[0] = sec
		if !e.mono_chrome {
			h.cdef_uv_pri_strength[0] = pri
			h.cdef_uv_sec_strength[0] = sec
		}
	}

	h.tx_mode = h.coded_lossless ? av1.ONLY_4X4 : av1.TX_MODE_SELECT
	h.reference_select = e.allow_comp
	h.skip_mode_present = false
	h.reduced_tx_set = true
	for ref in av1.LAST_FRAME ..= av1.ALTREF_FRAME {
		h.gm_type[ref] = av1.IDENTITY
		h.gm_params[ref] = {0, 0, 1 << av1.WARPEDMODEL_PREC_BITS, 0, 0, 1 << av1.WARPEDMODEL_PREC_BITS}
	}
}

@(private)
write_delta_q :: proc(w: ^Raw_Writer, dq: i8) {
	if dq != 0 {
		write_flag(w, true)
		write_su(w, i32(dq), 7)
	} else {
		write_flag(w, false)
	}
}

@(private)
is_skip_mode_allowed :: proc(s: ^av1.Sequence_Header, h: ^av1.Frame_Header) -> bool {
	if h.frame_is_intra || !h.reference_select || !s.enable_order_hint do return false
	forward, backward := -1, -1
	forward_hint, backward_hint: int
	for i in 0 ..< av1.REFS_PER_FRAME {
		hint := int(h.order_hints[av1.LAST_FRAME + i])
		d := av1.get_relative_dist(hint, int(h.order_hint), s.enable_order_hint, u8(s.order_hint_bits))
		if d < 0 {
			if forward < 0 || av1.get_relative_dist(hint, forward_hint, s.enable_order_hint, u8(s.order_hint_bits)) > 0 {
				forward, forward_hint = i, hint
			}
		} else if d > 0 {
			if backward < 0 || av1.get_relative_dist(hint, backward_hint, s.enable_order_hint, u8(s.order_hint_bits)) < 0 {
				backward, backward_hint = i, hint
			}
		}
	}
	if forward < 0 do return false
	if backward >= 0 do return true
	second, second_hint := -1, 0
	for i in 0 ..< av1.REFS_PER_FRAME {
		hint := int(h.order_hints[av1.LAST_FRAME + i])
		if av1.get_relative_dist(hint, forward_hint, s.enable_order_hint, u8(s.order_hint_bits)) < 0 {
			if second < 0 || av1.get_relative_dist(hint, second_hint, s.enable_order_hint, u8(s.order_hint_bits)) > 0 {
				second, second_hint = i, hint
			}
		}
	}
	return second >= 0
}

// Emits the uncompressed frame header (Section 5.9.2).
@(private)
build_frame_header :: proc(e: ^Encoder, w: ^Raw_Writer) {
	s := &e.seq
	h := &e.header

	write_flag(w, false) // show_existing_frame = 0
	write_bits(w, int(h.frame_type), 2)
	write_flag(w, h.show_frame)
	if !h.show_frame {
		write_flag(w, h.showable_frame)
	}
	if !(h.frame_type == .Switch || (h.frame_type == .Key && h.show_frame)) {
		write_flag(w, h.error_resilient_mode)
	}
	write_flag(w, h.disable_cdf_update)
	write_flag(w, h.frame_size_override_flag)
	write_bits(w, int(h.order_hint), s.order_hint_bits)
	if !(h.frame_is_intra || h.error_resilient_mode) {
		write_bits(w, int(h.primary_ref_frame), 3)
	}
	if !(h.frame_type == .Switch || (h.frame_type == .Key && h.show_frame)) {
		write_bits(w, int(h.refresh_frame_flags), 8)
	}

	if h.frame_is_intra {
		write_flag(w, false) // render_and_frame_size_different = 0
	} else {
		if s.enable_order_hint {
			write_flag(w, false) // frame_refs_short_signaling = 0
		}
		for i in 0 ..< av1.REFS_PER_FRAME {
			write_bits(w, int(h.ref_frame_idx[i]), 3)
		}
		write_flag(w, false) // render_and_frame_size_different = 0
		write_flag(w, h.allow_high_precision_mv)
		write_flag(w, h.is_filter_switchable)
		if !h.is_filter_switchable {
			write_bits(w, int(h.interpolation_filter), 2)
		}
		write_flag(w, h.is_motion_mode_switchable)
		if !h.error_resilient_mode && s.enable_ref_frame_mvs {
			write_flag(w, h.use_ref_frame_mvs)
		}
	}

	if !(s.reduced_still_picture_header || h.disable_cdf_update) {
		write_flag(w, h.disable_frame_end_update_cdf)
	}

	// tile_info (Section 5.9.15)
	MAX_TILE_WIDTH :: 4096
	MAX_TILE_AREA :: 4096 * 2304
	max_tile_width_sb := MAX_TILE_WIDTH >> 6
	max_tile_area_sb := MAX_TILE_AREA >> 12
	min_log2_cols := tile_log2(max_tile_width_sb, e.sb_cols)
	max_log2_cols := tile_log2(1, min(e.sb_cols, av1.MAX_TILE_COLS))
	max_log2_rows := tile_log2(1, min(e.sb_rows, av1.MAX_TILE_ROWS))
	min_log2_tiles := max(min_log2_cols, tile_log2(max_tile_area_sb, e.sb_rows * e.sb_cols))

	t := &h.tile_info
	write_flag(w, true) // uniform_tile_spacing_flag
	for _ in min_log2_cols ..< t.tile_cols_log2 {
		write_flag(w, true)
	}
	if t.tile_cols_log2 < max_log2_cols {
		write_flag(w, false)
	}
	min_log2_rows := max(min_log2_tiles - t.tile_cols_log2, 0)
	for _ in min_log2_rows ..< t.tile_rows_log2 {
		write_flag(w, true)
	}
	if t.tile_rows_log2 < max_log2_rows {
		write_flag(w, false)
	}
	if t.tile_cols_log2 > 0 || t.tile_rows_log2 > 0 {
		write_bits(w, int(t.context_update_tile_id), t.tile_rows_log2 + t.tile_cols_log2)
		write_bits(w, t.tile_size_bytes - 1, 2)
	}

	// quantization_params (Section 5.9.12)
	write_bits(w, int(h.base_q_idx), 8)
	write_delta_q(w, h.delta_q_y_dc)
	if !s.color.mono_chrome {
		if s.color.separate_uv_delta_q {
			write_flag(w, h.diff_uv_delta)
		}
		write_delta_q(w, h.delta_q_u_dc)
		write_delta_q(w, h.delta_q_u_ac)
		if h.diff_uv_delta {
			write_delta_q(w, h.delta_q_v_dc)
			write_delta_q(w, h.delta_q_v_ac)
		}
	}
	write_flag(w, false) // using_qmatrix = 0

	// segmentation_params (Section 5.9.14)
	write_flag(w, h.segmentation_enabled)
	if h.segmentation_enabled {
		if h.primary_ref_frame != av1.PRIMARY_REF_NONE {
			write_flag(w, h.segmentation_update_map)
			if h.segmentation_update_map {
				write_flag(w, h.segmentation_temporal_update)
			}
			write_flag(w, h.segmentation_update_data)
		}
		if h.segmentation_update_data {
			for i in 0 ..< av1.MAX_SEGMENTS {
				for j in 0 ..< av1.SEG_LVL_MAX {
					enabled := (h.feature_enabled[i] & (u8(1) << uint(j))) != 0
					write_flag(w, enabled)
					if enabled {
						bits := int(av1.SEGMENTATION_FEATURE_BITS[j])
						if av1.SEGMENTATION_FEATURE_SIGNED[j] != 0 {
							write_su(w, i32(h.feature_data[i][j]), 1 + bits)
						} else {
							write_bits(w, int(h.feature_data[i][j]), bits)
						}
					}
				}
			}
		}
	}

	// delta_q_params (Section 5.9.13)
	if h.base_q_idx > 0 {
		write_flag(w, false) // delta_q_present = 0
	}

	// loop_filter_params (Section 5.9.11)
	if !h.coded_lossless && !h.allow_intrabc {
		write_bits(w, int(h.loop_filter_level[0]), 6)
		write_bits(w, int(h.loop_filter_level[1]), 6)
		if !s.color.mono_chrome && (h.loop_filter_level[0] != 0 || h.loop_filter_level[1] != 0) {
			write_bits(w, int(h.loop_filter_level[2]), 6)
			write_bits(w, int(h.loop_filter_level[3]), 6)
		}
		write_bits(w, int(h.loop_filter_sharpness), 3)
		write_flag(w, h.loop_filter_delta_enabled)
		if h.loop_filter_delta_enabled {
			write_flag(w, false) // loop_filter_delta_update = 0
		}
	}

	// cdef_params (Section 5.9.19)
	if !h.coded_lossless && !h.allow_intrabc && s.enable_cdef {
		write_bits(w, int(h.cdef_damping_minus_3), 2)
		write_bits(w, int(h.cdef_bits), 2)
		for i in 0 ..< (1 << h.cdef_bits) {
			write_bits(w, int(h.cdef_y_pri_strength[i]), 4)
			y_sec := h.cdef_y_sec_strength[i] == 4 ? 3 : h.cdef_y_sec_strength[i]
			write_bits(w, int(y_sec), 2)
			if !s.color.mono_chrome {
				write_bits(w, int(h.cdef_uv_pri_strength[i]), 4)
				uv_sec := h.cdef_uv_sec_strength[i] == 4 ? 3 : h.cdef_uv_sec_strength[i]
				write_bits(w, int(uv_sec), 2)
			}
		}
	}

	// tx_mode (Section 5.9.21)
	if !h.coded_lossless {
		write_flag(w, h.tx_mode == av1.TX_MODE_SELECT)
	}

	// frame_reference_mode (Section 5.9.23)
	if !h.frame_is_intra {
		write_flag(w, h.reference_select)
	}

	// skip_mode_params (Section 5.9.22)
	if is_skip_mode_allowed(s, h) {
		write_flag(w, h.skip_mode_present)
	}

	write_flag(w, h.reduced_tx_set)

	// global_motion_params (Section 5.9.24)
	if !h.frame_is_intra {
		for _ in av1.LAST_FRAME ..= av1.ALTREF_FRAME {
			write_flag(w, false) // is_global = 0
		}
	}
}

@(private)
build_show_existing_obu :: proc(e: ^Encoder, slot: int) -> bool {
	clear(&e.header_buf)
	w := Raw_Writer{data = &e.header_buf}
	write_flag(&w, true) // show_existing_frame = 1
	write_bits(&w, slot, 3)
	trailing_bits(&w)
	if w.failed do return false
	clear(&e.packet)
	return append_obu(&e.packet, .Frame_Header, e.header_buf[:])
}

@(private)
assemble_frame_packet :: proc(e: ^Encoder) -> bool {
	clear(&e.header_buf)
	w := Raw_Writer{data = &e.header_buf}
	build_frame_header(e, &w)
	byte_align(&w)
	if w.failed do return false

	num_tiles := len(e.tiles)
	tile_bytes := e.header.tile_info.tile_size_bytes
	frame_payload_len := len(e.header_buf) + (num_tiles > 1 ? 1 : 0)
	for &t, i in e.tiles {
		if i + 1 < num_tiles do frame_payload_len += tile_bytes
		frame_payload_len += len(t.sym.data)
	}

	clear(&e.packet)
	total_est := (e.keyframe ? len(e.seq_obu) : 0) + 10 + frame_payload_len
	if reserve(&e.packet, total_est) != nil do return false

	if e.keyframe {
		if _, err := append(&e.packet, ..e.seq_obu[:]); err != nil do return false
	}

	hdr := (u8(av1.Obu_Type.Frame) << 3) | 0x02
	if _, err := append(&e.packet, hdr); err != nil do return false
	if !append_leb128(&e.packet, frame_payload_len) do return false
	if _, err := append(&e.packet, ..e.header_buf[:]); err != nil do return false

	if num_tiles > 1 {
		// tile_start_and_end_present_flag = 0 (1 bit) + byte_alignment (7 bits) = 0x00.
		if _, err := append(&e.packet, 0); err != nil do return false
	}

	for &t, i in e.tiles {
		if i + 1 < num_tiles {
			sz_minus_1 := len(t.sym.data) - 1
			for k in 0 ..< tile_bytes {
				if _, err := append(&e.packet, u8((sz_minus_1 >> uint(8 * k)) & 0xFF)); err != nil {
					return false
				}
			}
		}
		if _, err := append(&e.packet, ..t.sym.data[:]); err != nil do return false
	}
	return true
}
