package av1_decoder

// Native AV1 software decoder (AV1 Bitstream & Decoding Process Specification
// v1.0.0 with Errata 1, sections 5-8). Shares entropy coding, transforms,
// intra/inter prediction, motion vector referencing, loop filter, CDEF,
// superres, loop restoration, and film grain synthesis with `video/av1` and
// `video/av1/encoder`.

import common "../../../common"
import av1 "../"
import "core:mem"
import "core:sync"

Error :: av1.Error

Picture :: struct {
	planes8:                  [3][]u8,  // indexed 8-bit planes {y, u, v}
	planes16:                 [3][]u16, // indexed 10/12-bit planes {y16, u16, v16}
	strides:                  [3]int,   // indexed strides {y_stride, uv_stride, uv_stride}
	y, u, v:                  []u8,     // 8-bit planes (valid when bit_depth == 8)
	y16, u16, v16:            []u16,    // 10/12-bit planes (valid when bit_depth > 8)
	y_stride, uv_stride:      int,      // in samples
	width, height:            int,      // upscaled luma dimensions
	render_width:             int,
	render_height:            int,
	subsampling_x:            u8,
	subsampling_y:            u8,
	bit_depth:                u8,
	mono_chrome:              bool,
	color_range:              bool,    // true = full, false = studio
	color_primaries:          u8,
	transfer_characteristics: u8,
	matrix_coefficients:      u8,
}

Frame_Buffer :: struct {
	y, u, v:             [dynamic]u8,
	y16, u16, v16:       [dynamic]u16,
	y_stride, uv_stride: int,
	width, height:       int,
	subsampling_x:       u8,
	subsampling_y:       u8,
	bit_depth:           u8,
	mono_chrome:         bool,
	ref_count:           int,
}

// Per-4x4 mode cell fields used during tile decoding and post-processing.
// Stored in a frame-wide `#soa` slice (`Decoder.tile_cells`) so `skip[:cells]`
// is a contiguous `[]bool` slice passed directly to CDEF without copying.
Tile_Cell :: struct {
	skip:           bool,
	skip_mode:      u8,
	uv_mode:        u8,
	inter_tx_size:  u8,
	comp_group_idx: u8,
	compound_idx:   u8,
	interp_filters: [2]u8,
	palette_sizes:  [2]u8,
	palette_colors: [2][av1.PALETTE_COLORS]u16,
}

Ref_Slot :: struct {
	buf:               ^Frame_Buffer,
	cdfs:              av1.CDF_Context,
	saved_mvs:         #soa[dynamic]av1.Saved_MV_Cell,
	saved_segment_ids: [dynamic]u8,
	mi_rows, mi_cols:  int,
	upscaled_width:    int,
	frame_width:       int,
	frame_height:      int,
	render_width:      int,
	render_height:     int,
	subsampling_x:     u8,
	subsampling_y:     u8,
	bit_depth:         u8,
	mono_chrome:       bool,
	order_hint:        u8,
	saved_order_hints: [av1.NUM_REF_FRAMES]u8,
	frame_type:        av1.Frame_Type,
	film_grain:        av1.Film_Grain,
}

Decoder :: struct {
	pool:              ^common.Work_Pool,
	parser:            av1.Parser,
	parsed_frames:     [dynamic]av1.Frame,
	parsed_tiles:      [dynamic]av1.Tile,

	ref_slots:         [av1.NUM_REF_FRAMES]Ref_Slot,
	cur:               ^Frame_Buffer,
	shown_buf:         ^Frame_Buffer,
	cdef_buf:          Frame_Buffer,
	upscaled_curr_buf: Frame_Buffer,
	upscaled_cdef_buf: Frame_Buffer,
	grain_buf:         Frame_Buffer,
	grain_stripes:     [dynamic]i16, // apply_film_grain's scratch
	free_buffers:      [dynamic]^Frame_Buffer,
	all_buffers:       [dynamic]^Frame_Buffer,

	cdfs:              av1.CDF_Context,
	tile_cdfs:         [dynamic]av1.CDF_Context,
	tile_states:       [dynamic]Tile_State,

	// Frame-wide `#soa` mode/filter grids (sized to `mi_rows * mi_cols`).
	mi_cells:          #soa[dynamic]av1.MI_Cell,
	lf_cells:          #soa[dynamic]av1.LF_Cell,
	tile_cells:        #soa[dynamic]Tile_Cell,
	segment_ids:       [dynamic]u8,
	prev_segment_ids:  [dynamic]u8,

	// Superblock CDEF index grid (`((mi_rows + 15) >> 4) * ((mi_cols + 15) >> 4)`).
	cdef_idx:          [dynamic]i8,

	// Loop restoration unit grids per plane (`#soa` slices).
	lr_units:          [3]#soa[dynamic]av1.LR_Unit,
	lr_unit_cols:      [3]int,
	lr_unit_rows:      [3]int,

	// Temporal MFMV buffers (`(mi_rows >> 1) * (mi_cols >> 1)`).
	motion_field_mvs:  [av1.TOTAL_REFS_PER_FRAME][dynamic]av1.Mv,
	cur_saved_mvs:     #soa[dynamic]av1.Saved_MV_Cell,

	// Per-frame segmentation & quantizer matrix state.
	seg_id_pre_skip:    bool,
	last_active_seg_id: int,
	lossless_array:     [av1.MAX_SEGMENTS]bool,
	seg_qm_level:       [3][av1.MAX_SEGMENTS]u8,
}

init_decoder :: proc(d: ^Decoder, pool: ^common.Work_Pool = nil) {
	d^ = {}
	d.pool = pool
}

destroy_frame_buffer :: proc(fb: ^Frame_Buffer) {
	delete(fb.y)
	delete(fb.u)
	delete(fb.v)
	delete(fb.y16)
	delete(fb.u16)
	delete(fb.v16)
	fb^ = {}
}

destroy_decoder :: proc(d: ^Decoder) {
	delete(d.parsed_frames)
	delete(d.parsed_tiles)
	for &slot in d.ref_slots {
		delete(slot.saved_mvs)
		delete(slot.saved_segment_ids)
	}
	destroy_frame_buffer(&d.cdef_buf)
	destroy_frame_buffer(&d.upscaled_curr_buf)
	destroy_frame_buffer(&d.upscaled_cdef_buf)
	destroy_frame_buffer(&d.grain_buf)
	delete(d.grain_stripes)
	for fb in d.all_buffers {
		destroy_frame_buffer(fb)
		free(fb)
	}
	delete(d.free_buffers)
	delete(d.all_buffers)
	delete(d.tile_cdfs)
	delete(d.tile_states)
	delete(d.mi_cells)
	delete(d.lf_cells)
	delete(d.tile_cells)
	delete(d.segment_ids)
	delete(d.prev_segment_ids)
	delete(d.cdef_idx)
	for p in 0 ..< 3 do delete(d.lr_units[p])
	for r in 0 ..< av1.TOTAL_REFS_PER_FRAME do delete(d.motion_field_mvs[r])
	delete(d.cur_saved_mvs)
	d^ = {}
}

@(private)
ensure_frame_buffer :: proc(
	fb: ^Frame_Buffer,
	w, h: int,
	sub_x, sub_y, bit_depth: u8,
	mono_chrome: bool,
) {
	// Pad dimensions to multiples of 128 luma samples so superblocks and SIMD
	// stores near the right/bottom edges never overrun the buffer.
	aligned_w := (w + 127) & ~int(127)
	aligned_h := (h + 127) & ~int(127)
	uv_w := mono_chrome ? 0 : (aligned_w >> sub_x)
	uv_h := mono_chrome ? 0 : (aligned_h >> sub_y)

	fb.width = w
	fb.height = h
	fb.y_stride = aligned_w
	fb.uv_stride = uv_w
	fb.subsampling_x = sub_x
	fb.subsampling_y = sub_y
	fb.bit_depth = bit_depth
	fb.mono_chrome = mono_chrome

	y_total := aligned_w * aligned_h
	uv_total := uv_w * uv_h

	if bit_depth > 8 {
		resize(&fb.y16, y_total)
		resize(&fb.u16, uv_total)
		resize(&fb.v16, uv_total)
	} else {
		resize(&fb.y, y_total)
		resize(&fb.u, uv_total)
		resize(&fb.v, uv_total)
	}
}

@(private)
acquire_buffer :: proc(
	d: ^Decoder,
	w, h: int,
	sub_x, sub_y, bit_depth: u8,
	mono_chrome: bool,
) -> ^Frame_Buffer {
	fb: ^Frame_Buffer
	if len(d.free_buffers) > 0 {
		fb = pop(&d.free_buffers)
	} else {
		fb = new(Frame_Buffer)
		append(&d.all_buffers, fb)
	}
	fb.ref_count = 1
	ensure_frame_buffer(fb, w, h, sub_x, sub_y, bit_depth, mono_chrome)
	return fb
}

@(private)
release_buffer :: proc(d: ^Decoder, fb: ^Frame_Buffer) {
	if fb == nil do return
	fb.ref_count -= 1
	if fb.ref_count == 0 {
		append(&d.free_buffers, fb)
	}
}

@(private)
swap_frame_buffer_planes :: #force_inline proc "contextless" (a, b: ^Frame_Buffer) {
	a.y, b.y = b.y, a.y
	a.u, b.u = b.u, a.u
	a.v, b.v = b.v, a.v
	a.y16, b.y16 = b.y16, a.y16
	a.u16, b.u16 = b.u16, a.u16
	a.v16, b.v16 = b.v16, a.v16
	a.y_stride, b.y_stride = b.y_stride, a.y_stride
	a.uv_stride, b.uv_stride = b.uv_stride, a.uv_stride
	a.width, b.width = b.width, a.width
	a.height, b.height = b.height, a.height
	a.subsampling_x, b.subsampling_x = b.subsampling_x, a.subsampling_x
	a.subsampling_y, b.subsampling_y = b.subsampling_y, a.subsampling_y
	a.bit_depth, b.bit_depth = b.bit_depth, a.bit_depth
	a.mono_chrome, b.mono_chrome = b.mono_chrome, a.mono_chrome
}

@(private)
copy_frame_buffer :: proc(dst, src: ^Frame_Buffer) {
	ensure_frame_buffer(
		dst,
		src.width, src.height,
		src.subsampling_x, src.subsampling_y,
		src.bit_depth, src.mono_chrome,
	)
	num_planes := src.mono_chrome ? 1 : 3
	for plane in 0 ..< num_planes {
		sub_x := plane > 0 ? src.subsampling_x : 0
		sub_y := plane > 0 ? src.subsampling_y : 0
		pw := (src.width + int(sub_x)) >> sub_x
		ph := (src.height + int(sub_y)) >> sub_y
		s_stride := plane == 0 ? src.y_stride : src.uv_stride
		d_stride := plane == 0 ? dst.y_stride : dst.uv_stride
		if src.bit_depth > 8 {
			sp := plane == 0 ? src.y16[:] : (plane == 1 ? src.u16[:] : src.v16[:])
			dp := plane == 0 ? dst.y16[:] : (plane == 1 ? dst.u16[:] : dst.v16[:])
			for r in 0 ..< ph {
				copy(dp[r * d_stride:][:pw], sp[r * s_stride:][:pw])
			}
		} else {
			sp := plane == 0 ? src.y[:] : (plane == 1 ? src.u[:] : src.v[:])
			dp := plane == 0 ? dst.y[:] : (plane == 1 ? dst.u[:] : dst.v[:])
			for r in 0 ..< ph {
				copy(dp[r * d_stride:][:pw], sp[r * s_stride:][:pw])
			}
		}
	}
}

@(private)
make_picture :: proc(
	d: ^Decoder,
	fb: ^Frame_Buffer,
	render_w, render_h: int,
	grain: ^av1.Film_Grain,
) -> Picture {
	s := &d.parser.sequence
	c := &s.color
	out := fb

	if grain != nil && grain.apply_grain {
		copy_frame_buffer(&d.grain_buf, fb)
		out = &d.grain_buf
		fg := grain^
		strides := [3]int{out.y_stride, out.uv_stride, out.uv_stride}
		if fb.bit_depth > 8 {
			planes := [3][]u16{
				out.y16[:],
				out.mono_chrome ? ([]u16)(nil) : out.u16[:],
				out.mono_chrome ? ([]u16)(nil) : out.v16[:],
			}
			av1.apply_film_grain(
				planes, strides,
				out.width, out.height,
				&fg,
				out.mono_chrome,
				uint(out.subsampling_x), uint(out.subsampling_y),
				c.matrix_coefficients,
				&d.grain_stripes,
				int(out.bit_depth),
			)
		} else {
			planes := [3][]u8{
				out.y[:],
				out.mono_chrome ? ([]u8)(nil) : out.u[:],
				out.mono_chrome ? ([]u8)(nil) : out.v[:],
			}
			av1.apply_film_grain(
				planes, strides,
				out.width, out.height,
				&fg,
				out.mono_chrome,
				uint(out.subsampling_x), uint(out.subsampling_y),
				c.matrix_coefficients,
				&d.grain_stripes,
				int(out.bit_depth),
			)
		}
	} else {
		fb.ref_count += 1
		release_buffer(d, d.shown_buf)
		d.shown_buf = fb
	}

	pic := Picture{
		strides                  = {out.y_stride, out.uv_stride, out.uv_stride},
		y_stride                 = out.y_stride,
		uv_stride                = out.uv_stride,
		width                    = out.width,
		height                   = out.height,
		render_width             = render_w,
		render_height            = render_h,
		subsampling_x            = out.subsampling_x,
		subsampling_y            = out.subsampling_y,
		bit_depth                = out.bit_depth,
		mono_chrome              = out.mono_chrome,
		color_range              = c.color_range,
		color_primaries          = c.color_primaries,
		transfer_characteristics = c.transfer_characteristics,
		matrix_coefficients      = c.matrix_coefficients,
	}
	if out.bit_depth > 8 {
		pic.y16 = out.y16[:]
		if !out.mono_chrome {
			pic.u16 = out.u16[:]
			pic.v16 = out.v16[:]
		}
		pic.planes16 = {pic.y16, pic.u16, pic.v16}
	} else {
		pic.y = out.y[:]
		if !out.mono_chrome {
			pic.u = out.u[:]
			pic.v = out.v[:]
		}
		pic.planes8 = {pic.y, pic.u, pic.v}
	}
	return pic
}

// Decodes one packet (an AV1 temporal unit or sequence of OBUs). Returns the
// last shown frame in the packet, or `shown == false` if the packet contains
// only hidden reference frames or header OBUs.
decode_packet :: proc(d: ^Decoder, data: []u8) -> (pic: Picture, shown: bool, err: Error) {
	if len(data) == 0 do return {}, false, .Invalid_Data
	av1.parse_temporal_unit(&d.parser, data, &d.parsed_frames, &d.parsed_tiles) or_return

	for &fr in d.parsed_frames {
		h := &fr.header
		if h.show_existing_frame {
			idx := h.frame_to_show_map_idx
			slot := &d.ref_slots[idx]
			if slot.buf == nil do return {}, false, .Invalid_Data
			pic = make_picture(d, slot.buf, h.render_width, h.render_height, &h.film_grain)
			shown = true
			if h.frame_type == .Key {
				// Section 7.21 + 7.20: refresh all 8 reference slots from `idx`.
				av1.load_cdfs(&d.cdfs, &slot.cdfs)
				update_ref_slots_from_existing(d, int(idx))
			}
			continue
		}

		defer if d.cur != nil {
			release_buffer(d, d.cur)
			d.cur = nil
		}
		decode_frame(d, h, data, fr.tiles) or_return
		if h.show_frame {
			pic = make_picture(d, d.cur, h.render_width, h.render_height, &h.film_grain)
			shown = true
		}
		update_ref_slots(d, h)
	}
	return pic, shown, .None
}

@(private)
update_ref_slots_from_existing :: proc(d: ^Decoder, src_idx: int) {
	src := &d.ref_slots[src_idx]
	for i in 0 ..< av1.NUM_REF_FRAMES {
		if i == src_idx do continue
		dst := &d.ref_slots[i]
		if src.buf != nil do src.buf.ref_count += 1
		release_buffer(d, dst.buf)
		dst.buf = src.buf
		dst.cdfs = src.cdfs
		n_mvs := len(src.saved_mvs)
		resize(&dst.saved_mvs, n_mvs)
		copy(dst.saved_mvs.mv[:n_mvs], src.saved_mvs.mv[:n_mvs])
		copy(dst.saved_mvs.ref_frame[:n_mvs], src.saved_mvs.ref_frame[:n_mvs])
		resize(&dst.saved_segment_ids, len(src.saved_segment_ids))
		copy(dst.saved_segment_ids[:], src.saved_segment_ids[:])
		dst.mi_rows = src.mi_rows
		dst.mi_cols = src.mi_cols
		dst.upscaled_width = src.upscaled_width
		dst.frame_width = src.frame_width
		dst.frame_height = src.frame_height
		dst.render_width = src.render_width
		dst.render_height = src.render_height
		dst.subsampling_x = src.subsampling_x
		dst.subsampling_y = src.subsampling_y
		dst.bit_depth = src.bit_depth
		dst.mono_chrome = src.mono_chrome
		dst.order_hint = src.order_hint
		dst.saved_order_hints = src.saved_order_hints
		dst.frame_type = src.frame_type
		dst.film_grain = src.film_grain
	}
}

// Section 7.20: Reference frame update process.
@(private)
update_ref_slots :: proc(d: ^Decoder, h: ^av1.Frame_Header) {
	s := &d.parser.sequence
	c := &s.color
	n_mvs := len(d.cur_saved_mvs)
	n_seg := h.mi_rows * h.mi_cols
	for i in 0 ..< av1.NUM_REF_FRAMES {
		if (h.refresh_frame_flags >> uint(i)) & 1 == 0 do continue
		slot := &d.ref_slots[i]
		if d.cur != nil do d.cur.ref_count += 1
		release_buffer(d, slot.buf)
		slot.buf = d.cur
		slot.cdfs = d.cdfs
		if !h.frame_is_intra && s.enable_ref_frame_mvs {
			resize(&slot.saved_mvs, n_mvs)
			copy(slot.saved_mvs.mv[:n_mvs], d.cur_saved_mvs.mv[:n_mvs])
			copy(slot.saved_mvs.ref_frame[:n_mvs], d.cur_saved_mvs.ref_frame[:n_mvs])
		} else {
			clear(&slot.saved_mvs)
		}
		if h.segmentation_enabled {
			resize(&slot.saved_segment_ids, n_seg)
			copy(slot.saved_segment_ids[:], d.segment_ids[:n_seg])
		} else {
			clear(&slot.saved_segment_ids)
		}
		slot.mi_rows = h.mi_rows
		slot.mi_cols = h.mi_cols
		slot.upscaled_width = h.upscaled_width
		slot.frame_width = h.frame_width
		slot.frame_height = h.frame_height
		slot.render_width = h.render_width
		slot.render_height = h.render_height
		slot.subsampling_x = c.subsampling_x
		slot.subsampling_y = c.subsampling_y
		slot.bit_depth = c.bit_depth
		slot.mono_chrome = c.mono_chrome
		slot.order_hint = h.order_hint
		slot.saved_order_hints = h.order_hints
		slot.frame_type = h.frame_type
		slot.film_grain = h.film_grain
	}
}

@(private)
setup_frame_segmentation_and_quant :: proc(d: ^Decoder, h: ^av1.Frame_Header) {
	d.seg_id_pre_skip = false
	d.last_active_seg_id = 0
	if h.segmentation_enabled {
		for i in 0 ..< av1.MAX_SEGMENTS {
			for j in 0 ..< av1.SEG_LVL_MAX {
				if (h.feature_enabled[i] & (u8(1) << uint(j))) != 0 {
					d.last_active_seg_id = i
					if j >= av1.SEG_LVL_REF_FRAME {
						d.seg_id_pre_skip = true
					}
				}
			}
		}
	}

	for seg in 0 ..< av1.MAX_SEGMENTS {
		qindex := get_qindex(h, true, int(h.base_q_idx), seg)
		d.lossless_array[seg] = qindex == 0 &&
			h.delta_q_y_dc == 0 &&
			h.delta_q_u_ac == 0 &&
			h.delta_q_u_dc == 0 &&
			h.delta_q_v_ac == 0 &&
			h.delta_q_v_dc == 0

		if !h.using_qmatrix {
			d.seg_qm_level[0][seg] = 15
			d.seg_qm_level[1][seg] = 15
			d.seg_qm_level[2][seg] = 15
		} else {
			d.seg_qm_level[0][seg] = h.qm_y
			d.seg_qm_level[1][seg] = h.qm_u
			d.seg_qm_level[2][seg] = h.qm_v
		}
	}
}

// Section 7.12.2: get_qindex.
@(private)
get_qindex :: #force_inline proc "contextless" (
	h: ^av1.Frame_Header,
	ignore_delta_q: bool,
	current_qindex: int,
	segment_id: int,
) -> int {
	seg_qindex := int(h.base_q_idx)
	if !ignore_delta_q && h.delta_q_present {
		seg_qindex = current_qindex
	}
	if h.segmentation_enabled && (h.feature_enabled[segment_id] & (1 << av1.SEG_LVL_ALT_Q)) != 0 {
		seg_qindex += int(h.feature_data[segment_id][av1.SEG_LVL_ALT_Q])
	}
	return clamp(seg_qindex, 0, 255)
}

@(private)
decode_frame :: proc(d: ^Decoder, h: ^av1.Frame_Header, data: []u8, tiles: []av1.Tile) -> Error {
	s := &d.parser.sequence
	c := &s.color

	// Allocate/reset frame-wide SOA grids.
	cells := h.mi_rows * h.mi_cols
	if len(d.mi_cells) != cells {
		resize(&d.mi_cells, cells)
		resize(&d.lf_cells, cells)
		resize(&d.tile_cells, cells)
		resize(&d.segment_ids, cells)
		resize(&d.prev_segment_ids, cells)
	}

	if !h.frame_is_intra || h.allow_intrabc {
		mem.set(d.mi_cells.ref_frames, 0xFF, cells * size_of([2]i8))
	}

	// Previous segment IDs from primary_ref_frame when resolution matches (6.8.2).
	if h.segmentation_enabled {
		if h.primary_ref_frame == av1.PRIMARY_REF_NONE {
			for i in 0 ..< cells do d.prev_segment_ids[i] = 0
		} else {
			prev_slot := &d.ref_slots[h.ref_frame_idx[h.primary_ref_frame]]
			if prev_slot.mi_cols == h.mi_cols && prev_slot.mi_rows == h.mi_rows && len(prev_slot.saved_segment_ids) >= cells {
				copy(d.prev_segment_ids[:cells], prev_slot.saved_segment_ids[:cells])
			} else {
				for i in 0 ..< cells do d.prev_segment_ids[i] = 0
			}
		}
	}

	// Initialize or load frame CDF context (6.8.2).
	if h.primary_ref_frame == av1.PRIMARY_REF_NONE {
		av1.init_cdf_context(&d.cdfs, h.base_q_idx)
	} else {
		prev_slot := &d.ref_slots[h.ref_frame_idx[h.primary_ref_frame]]
		av1.load_cdfs(&d.cdfs, &prev_slot.cdfs)
	}

	setup_frame_segmentation_and_quant(d, h)

	// Superblock CDEF index grid (in 64x64 units = 16x16 MI units).
	cdef_rows := (h.mi_rows + 15) >> 4
	cdef_cols := (h.mi_cols + 15) >> 4
	cdef_n := cdef_rows * cdef_cols
	if len(d.cdef_idx) != cdef_n {
		resize(&d.cdef_idx, cdef_n)
	}
	mem.set(raw_data(d.cdef_idx), 0xFF, cdef_n)

	// Loop restoration unit grids per plane (5.11.58).
	num_planes := c.mono_chrome ? 1 : 3
	for plane in 0 ..< num_planes {
		sub_x := plane > 0 ? uint(c.subsampling_x) : 0
		sub_y := plane > 0 ? uint(c.subsampling_y) : 0
		unit_size := int(h.loop_restoration_size[plane])
		if h.frame_restoration_type[plane] != av1.RESTORE_NONE && unit_size > 0 {
			pw := (h.upscaled_width + int(sub_x)) >> sub_x
			ph := (h.frame_height + int(sub_y)) >> sub_y
			ucols := max(1, (pw + (unit_size >> 1)) / unit_size)
			urows := max(1, (ph + (unit_size >> 1)) / unit_size)
			d.lr_unit_cols[plane] = ucols
			d.lr_unit_rows[plane] = urows
			resize(&d.lr_units[plane], ucols * urows)
			for &u in d.lr_units[plane] do u = {restoration_type = av1.RESTORE_NONE}
		} else {
			d.lr_unit_cols[plane] = 0
			d.lr_unit_rows[plane] = 0
		}
	}

	// Motion field estimation (7.9).
	w8 := h.mi_cols >> 1
	h8 := h.mi_rows >> 1
	n8 := w8 * h8
	if !h.frame_is_intra && h.refresh_frame_flags != 0 && s.enable_ref_frame_mvs {
		resize(&d.cur_saved_mvs, n8)
	}
	if !h.frame_is_intra && h.use_ref_frame_mvs {
		mf_slices: [av1.TOTAL_REFS_PER_FRAME][]av1.Mv
		for ref in av1.LAST_FRAME ..= av1.ALTREF_FRAME {
			resize(&d.motion_field_mvs[ref], n8)
			mf_slices[ref] = d.motion_field_mvs[ref][:n8]
		}
		saved_refs: [av1.NUM_REF_FRAMES]av1.Saved_Ref_MVs
		for i in 0 ..< av1.NUM_REF_FRAMES {
			slot := &d.ref_slots[i]
			saved_refs[i] = {
				cells       = slot.saved_mvs[:],
				mi_rows     = slot.mi_rows,
				mi_cols     = slot.mi_cols,
				frame_type  = u8(slot.frame_type),
				order_hint  = slot.order_hint,
				order_hints = slot.saved_order_hints,
			}
		}
		ref_idx_i8: [av1.REFS_PER_FRAME]i8
		for i in 0 ..< av1.REFS_PER_FRAME do ref_idx_i8[i] = i8(h.ref_frame_idx[i])
		av1.motion_field_estimation(
			&mf_slices,
			&saved_refs,
			ref_idx_i8,
			h.order_hints,
			h.order_hint,
			s.enable_order_hint,
			u8(s.order_hint_bits),
			h.mi_rows,
			h.mi_cols,
		)
	}

	// Acquire output buffer for decoded coded-resolution samples (`CurrFrame`).
	d.cur = acquire_buffer(
		d,
		h.frame_width, h.frame_height,
		c.subsampling_x, c.subsampling_y,
		c.bit_depth, c.mono_chrome,
	)

	// Decode all tiles (5.11.2).
	num_tiles := h.tile_info.tile_cols * h.tile_info.tile_rows
	if len(tiles) != num_tiles do return .Invalid_Data

	if len(d.tile_states) < num_tiles {
		resize(&d.tile_states, num_tiles)
	}

	need_preserve_cdfs := h.disable_frame_end_update_cdf && !h.disable_cdf_update && h.refresh_frame_flags != 0
	if num_tiles == 1 && !need_preserve_cdfs {
		tile := tiles[0]
		end := int(tile.offset) + int(tile.size)
		if end > len(data) do return .Invalid_Data
		decode_tile(d, h, 0, data[tile.offset:end], &d.cdfs) or_return
	} else {
		// Each tile adapts its own copy of the frame's CDFs, except when updates
		// are disabled: the tiles then only read them, and share the frame's.
		shared := h.disable_cdf_update
		ctx_tile := (!h.disable_frame_end_update_cdf && d.pool == nil) ? int(h.tile_info.context_update_tile_id) : -1
		if !shared {
			if len(d.tile_cdfs) < num_tiles {
				resize(&d.tile_cdfs, num_tiles)
			}
			for t in 0 ..< num_tiles {
				if t != ctx_tile {
					d.tile_cdfs[t] = d.cdfs
				}
			}
		}

		if d.pool != nil && num_tiles > 1 {
			Batch :: struct {
				d:      ^Decoder,
				h:      ^av1.Frame_Header,
				data:   []u8,
				tiles:  []av1.Tile,
				shared: bool,
				err:    Error,
			}
			b := Batch{d = d, h = h, data = data, tiles = tiles, shared = shared}
			d.pool.run(d.pool, num_tiles, proc(ctx: rawptr, first, end: int) {
				b := (^Batch)(ctx)
				for tile_num in first ..< end {
					tile := b.tiles[tile_num]
					t_end := int(tile.offset) + int(tile.size)
					if t_end > len(b.data) {
						sync.atomic_store(&b.err, Error.Invalid_Data)
						return
					}
					cdfs := b.shared ? &b.d.cdfs : &b.d.tile_cdfs[tile_num]
					t_err := decode_tile(b.d, b.h, tile_num, b.data[tile.offset:t_end], cdfs)
					if t_err != .None {
						sync.atomic_store(&b.err, t_err)
					}
				}
			}, &b)
			if b.err != .None do return b.err
			if !h.disable_frame_end_update_cdf && int(h.tile_info.context_update_tile_id) < num_tiles {
				d.cdfs = d.tile_cdfs[h.tile_info.context_update_tile_id]
			}
		} else {
			for tile_num in 0 ..< num_tiles {
				tile := tiles[tile_num]
				end := int(tile.offset) + int(tile.size)
				if end > len(data) do return .Invalid_Data
				cdfs_ptr := shared || tile_num == ctx_tile ? &d.cdfs : &d.tile_cdfs[tile_num]
				decode_tile(d, h, tile_num, data[tile.offset:end], cdfs_ptr) or_return
			}
		}
	}

	// Post-processing wrapup pipeline (7.4).
	decode_frame_wrapup(d, h)
	return .None
}

// Section 7.4: Decode frame wrapup process (loop filter -> CDEF -> superres ->
// loop restoration -> MFMV storage -> segmentation persistence).
@(private)
decode_frame_wrapup :: proc(d: ^Decoder, h: ^av1.Frame_Header) {
	s := &d.parser.sequence
	c := &s.color
	cells := h.mi_rows * h.mi_cols
	sub_x := uint(c.subsampling_x)
	sub_y := uint(c.subsampling_y)
	num_planes := c.mono_chrome ? 1 : 3

	planes_u8 :: #force_inline proc "contextless" (fb: ^Frame_Buffer) -> [3][]u8 {
		return {
			fb.y[:],
			fb.mono_chrome ? ([]u8)(nil) : fb.u[:],
			fb.mono_chrome ? ([]u8)(nil) : fb.v[:],
		}
	}
	planes_u16 :: #force_inline proc "contextless" (fb: ^Frame_Buffer) -> [3][]u16 {
		return {
			fb.y16[:],
			fb.mono_chrome ? ([]u16)(nil) : fb.u16[:],
			fb.mono_chrome ? ([]u16)(nil) : fb.v16[:],
		}
	}
	strides_of :: #force_inline proc "contextless" (fb: ^Frame_Buffer) -> [3]int {
		return {fb.y_stride, fb.uv_stride, fb.uv_stride}
	}

	// 1. Deblocking loop filter in place on `d.cur` (section 7.14).
	if !h.all_lossless && (h.loop_filter_level[0] != 0 || h.loop_filter_level[1] != 0) {
		lf_slice := d.lf_cells[:cells]
		lf_slice.tx_size_y = d.tile_cells.inter_tx_size
		lf_slice.mi_size = d.mi_cells.mi_size
		if c.bit_depth > 8 {
			av1.loop_filter_frame(
				planes_u16(d.cur), strides_of(d.cur),
				lf_slice,
				h, num_planes, sub_x, sub_y, int(c.bit_depth),
			)
		} else {
			av1.loop_filter_frame(
				planes_u8(d.cur), strides_of(d.cur),
				lf_slice,
				h, num_planes, sub_x, sub_y, int(c.bit_depth),
			)
		}
	}

	// 2. CDEF (section 7.15), executed in-place on `d.cur`.
	has_cdef := !h.coded_lossless && s.enable_cdef && !h.allow_intrabc
	if has_cdef && h.uses_lr {
		ensure_frame_buffer(
			&d.upscaled_curr_buf,
			h.upscaled_width, h.frame_height,
			c.subsampling_x, c.subsampling_y,
			c.bit_depth, c.mono_chrome,
		)
		if h.use_superres {
			upscale_buffer(&d.upscaled_curr_buf, d.cur, h)
		} else if c.bit_depth > 8 {
			save_lr_stripe_boundaries(planes_u16(&d.upscaled_curr_buf), strides_of(&d.upscaled_curr_buf), planes_u16(d.cur), strides_of(d.cur), h, num_planes, sub_x, sub_y)
		} else {
			save_lr_stripe_boundaries(planes_u8(&d.upscaled_curr_buf), strides_of(&d.upscaled_curr_buf), planes_u8(d.cur), strides_of(d.cur), h, num_planes, sub_x, sub_y)
		}
	}
	if has_cdef {
		skips := d.tile_cells.skip[:cells]
		sb64_cols := (h.mi_cols + 15) >> 4
		if c.bit_depth > 8 {
			av1.cdef_frame(
				planes_u16(d.cur), strides_of(d.cur),
				planes_u16(d.cur), strides_of(d.cur),
				d.cdef_idx[:], sb64_cols,
				skips, h, s.enable_cdef, num_planes, sub_x, sub_y, int(c.bit_depth),
			)
		} else {
			av1.cdef_frame(
				planes_u8(d.cur), strides_of(d.cur),
				planes_u8(d.cur), strides_of(d.cur),
				d.cdef_idx[:], sb64_cols,
				skips, h, s.enable_cdef, num_planes, sub_x, sub_y, int(c.bit_depth),
			)
		}
	}

	// 3. Superres upscaling (section 7.16) & 4. Loop restoration (section 7.17).
	// `d.cur` will hold the final `LrFrame` of dimensions `(h.upscaled_width, h.frame_height)`.
	if !h.uses_lr {
		if h.use_superres {
			swap_frame_buffer_planes(&d.cdef_buf, d.cur)
			ensure_frame_buffer(
				d.cur,
				h.upscaled_width, h.frame_height,
				c.subsampling_x, c.subsampling_y,
				c.bit_depth, c.mono_chrome,
			)
			upscale_buffer(d.cur, &d.cdef_buf, h)
		}
	} else {
		// Loop restoration reads both `UpscaledCdefFrame` and `UpscaledCurrFrame`.
		up_cdef: ^Frame_Buffer
		up_curr: ^Frame_Buffer
		if h.use_superres {
			ensure_frame_buffer(
				&d.upscaled_cdef_buf,
				h.upscaled_width, h.frame_height,
				c.subsampling_x, c.subsampling_y,
				c.bit_depth, c.mono_chrome,
			)
			upscale_buffer(&d.upscaled_cdef_buf, d.cur, h)
			up_cdef = &d.upscaled_cdef_buf
			if !has_cdef {
				ensure_frame_buffer(
					&d.upscaled_curr_buf,
					h.upscaled_width, h.frame_height,
					c.subsampling_x, c.subsampling_y,
					c.bit_depth, c.mono_chrome,
				)
				upscale_buffer(&d.upscaled_curr_buf, d.cur, h)
			}
			up_curr = &d.upscaled_curr_buf
			ensure_frame_buffer(
				d.cur,
				h.upscaled_width, h.frame_height,
				c.subsampling_x, c.subsampling_y,
				c.bit_depth, c.mono_chrome,
			)
		} else if has_cdef {
			// When `!h.use_superres`, `loop_restore_frame` executes in-place on `d.cur`
			// while reading pre-CDEF stripe boundaries from `d.upscaled_curr_buf`.
			up_cdef = d.cur
			up_curr = &d.upscaled_curr_buf
		} else {
			// Neither superres nor CDEF ran: both `UpscaledCdefFrame` and `UpscaledCurrFrame`
			// are identical to `d.cur`; save only the 4-row stripe boundaries into `upscaled_curr_buf`
			// so `loop_restore_frame` can execute in-place on `d.cur`.
			ensure_frame_buffer(
				&d.upscaled_curr_buf,
				h.upscaled_width, h.frame_height,
				c.subsampling_x, c.subsampling_y,
				c.bit_depth, c.mono_chrome,
			)
			if c.bit_depth > 8 {
				save_lr_stripe_boundaries(planes_u16(&d.upscaled_curr_buf), strides_of(&d.upscaled_curr_buf), planes_u16(d.cur), strides_of(d.cur), h, num_planes, sub_x, sub_y)
			} else {
				save_lr_stripe_boundaries(planes_u8(&d.upscaled_curr_buf), strides_of(&d.upscaled_curr_buf), planes_u8(d.cur), strides_of(d.cur), h, num_planes, sub_x, sub_y)
			}
			up_cdef = d.cur
			up_curr = &d.upscaled_curr_buf
		}

		lr_slices := [3]#soa[]av1.LR_Unit{
			d.lr_units[0][:],
			d.lr_units[1][:],
			d.lr_units[2][:],
		}
		if c.bit_depth > 8 {
			av1.loop_restore_frame(
				planes_u16(d.cur), strides_of(d.cur),
				planes_u16(up_cdef), strides_of(up_cdef),
				planes_u16(up_curr), strides_of(up_curr),
				lr_slices, h, num_planes, sub_x, sub_y, int(c.bit_depth),
			)
		} else {
			av1.loop_restore_frame(
				planes_u8(d.cur), strides_of(d.cur),
				planes_u8(up_cdef), strides_of(up_cdef),
				planes_u8(up_curr), strides_of(up_curr),
				lr_slices, h, num_planes, sub_x, sub_y, int(c.bit_depth),
			)
		}
	}

	// 5. Motion field motion vector storage (section 7.19).
	if !h.frame_is_intra && h.refresh_frame_flags != 0 && s.enable_ref_frame_mvs {
		grid := av1.MI_Grid{
			cells   = d.mi_cells[:cells],
			mi_rows = h.mi_rows,
			mi_cols = h.mi_cols,
		}
		saved_dst := av1.Saved_Ref_MVs{
			cells   = d.cur_saved_mvs[:],
			mi_rows = h.mi_rows,
			mi_cols = h.mi_cols,
		}
		ref_idx_i8: [av1.REFS_PER_FRAME]i8
		for i in 0 ..< av1.REFS_PER_FRAME do ref_idx_i8[i] = i8(h.ref_frame_idx[i])
		ref_order_hint: [av1.NUM_REF_FRAMES]u8
		for i in 0 ..< av1.NUM_REF_FRAMES do ref_order_hint[i] = d.ref_slots[i].order_hint
		av1.store_motion_field_mvs(
			&saved_dst,
			&grid,
			ref_idx_i8,
			ref_order_hint,
			h.order_hint,
			s.enable_order_hint,
			u8(s.order_hint_bits),
		)
	}

	// 6. Segmentation map persistence (section 7.4).
	if h.segmentation_enabled && !h.segmentation_update_map {
		copy(d.segment_ids[:cells], d.prev_segment_ids[:cells])
	}
}

@(private)
upscale_buffer :: proc(
	dst, src: ^Frame_Buffer,
	h: ^av1.Frame_Header,
) {
	sub_x := uint(src.subsampling_x)
	sub_y := uint(src.subsampling_y)
	num_planes := src.mono_chrome ? 1 : 3
	dst_strides := [3]int{dst.y_stride, dst.uv_stride, dst.uv_stride}
	src_strides := [3]int{src.y_stride, src.uv_stride, src.uv_stride}
	if src.bit_depth > 8 {
		dst_planes := [3][]u16{
			dst.y16[:],
			src.mono_chrome ? ([]u16)(nil) : dst.u16[:],
			src.mono_chrome ? ([]u16)(nil) : dst.v16[:],
		}
		src_planes := [3][]u16{
			src.y16[:],
			src.mono_chrome ? ([]u16)(nil) : src.u16[:],
			src.mono_chrome ? ([]u16)(nil) : src.v16[:],
		}
		av1.upscale_frame(
			dst_planes, dst_strides,
			src_planes, src_strides,
			h, num_planes, sub_x, sub_y, int(src.bit_depth),
		)
	} else {
		dst_planes := [3][]u8{
			dst.y[:],
			src.mono_chrome ? ([]u8)(nil) : dst.u[:],
			src.mono_chrome ? ([]u8)(nil) : dst.v[:],
		}
		src_planes := [3][]u8{
			src.y[:],
			src.mono_chrome ? ([]u8)(nil) : src.u[:],
			src.mono_chrome ? ([]u8)(nil) : src.v[:],
		}
		av1.upscale_frame(
			dst_planes, dst_strides,
			src_planes, src_strides,
			h, num_planes, sub_x, sub_y, int(src.bit_depth),
		)
	}
}

@(private)
save_lr_stripe_boundaries :: proc "contextless" (
	dst_planes: [3][]$T,
	dst_strides: [3]int,
	src_planes: [3][]T,
	src_strides: [3]int,
	h: ^av1.Frame_Header,
	num_planes: int,
	sub_x, sub_y: uint,
) #no_bounds_check {
	for p in 0 ..< num_planes {
		if h.frame_restoration_type[p] == av1.RESTORE_NONE do continue
		sx := p == 0 ? uint(0) : sub_x
		sy := p == 0 ? uint(0) : sub_y
		plane_w := (h.upscaled_width + int(sx)) >> sx
		plane_h := (h.frame_height + int(sy)) >> sy
		plane_end_y := plane_h - 1
		if plane_end_y < 0 do continue

		dst_p := dst_planes[p]
		src_p := src_planes[p]
		dst_s := dst_strides[p]
		src_s := src_strides[p]

		// First stripe top boundary (rows 0..1) and bottom boundary.
		for r in 0 ..= min(1, plane_end_y) {
			copy(dst_p[r * dst_s:][:plane_w], src_p[r * src_s:][:plane_w])
		}
		for boundary_luma_y := 56; boundary_luma_y < h.frame_height + 64; boundary_luma_y += 64 {
			by := boundary_luma_y >> sy
			for dy in -2 ..= 1 {
				ry := clamp(by + dy, 0, plane_end_y)
				copy(dst_p[ry * dst_s:][:plane_w], src_p[ry * src_s:][:plane_w])
			}
		}
	}
}
