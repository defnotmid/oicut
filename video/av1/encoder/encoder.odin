package av1_encoder

// Native AV1 software encoder (AV1 Bitstream & Decoding Process Specification
// v1.0.0 with Errata 1). Shares entropy coding, CDF adaptation, forward/inverse
// transforms, quantization/dequantization, intra/inter prediction, motion vector
// referencing, loop filter, and CDEF with `video/av1` and `video/av1/decoder`.

import common "../../../common"
import av1 ".."
import "core:mem"
import "core:slice"

Error :: enum {
	None,
	Invalid_Argument,
	Out_Of_Memory,
	Closed,
}

Chroma_Subsampling :: enum {
	YUV420,     // ss_x = 1, ss_y = 1 (default)
	YUV422,     // ss_x = 1, ss_y = 0
	YUV444,     // ss_x = 0, ss_y = 0
	Monochrome, // mono_chrome = true, ss_x = 1, ss_y = 1
}

Encoder_Config :: struct {
	width, height:       int,
	bit_depth:           int, // 0 or 8 selects 8; 10 or 12 selects high bit depth
	subsampling:         Chroma_Subsampling,
	base_q_idx:          int, // 0..255; zero is exactly lossless
	delta_q_y_dc:        int, // -64..63
	delta_q_u_dc:        int, // -64..63
	delta_q_u_ac:        int, // -64..63
	delta_q_v_dc:        int, // -64..63
	delta_q_v_ac:        int, // -64..63
	keyframe_interval:   int, // zero selects 120
	block_log2:          int, // 2..6: 4, 8, 16, 32, or 64 pixel partitions; zero selects 6 (64)
	disable_loop_filter: bool,
	disable_cdef:        bool,
	search_range:        int, // integer pixels; zero selects 16
	tile_cols_log2:      int, // clipped to AV1 limits for the width
	tile_rows_log2:      int, // clipped to AV1 limits for the height
	ultrafast:           bool, // prefer encode speed over size
	pool:                ^common.Work_Pool,
}

@(private)
plane_t :: #force_inline proc(b: []u8, $T: typeid) -> []T #no_bounds_check {
	when T == u8 {
		return b
	} else {
		return slice.reinterpret([]T, b)
	}
}

// Per-4x4 cell state stored in an `#soa` slice (`Encoder.tile_cells`) so
// `skip[:cells]` is a contiguous `[]bool` slice passed directly to CDEF.
@(private)
Tile_Cell :: struct {
	skip:           bool,
	skip_mode:      u8,
	uv_mode:        u8,
	inter_tx_size:  u8,
	tx_type:        u8,
	comp_group_idx: u8,
	compound_idx:   u8,
	interp_filters: [2]u8,
}

Encoder :: struct {
	config:          Encoder_Config,
	allocator:       mem.Allocator,
	seq:             av1.Sequence_Header,
	header:          av1.Frame_Header,

	planes:          [3][]u8,
	reference:       [3][]u8, // slot 0 (LAST_FRAME)
	golden:          [3][]u8, // slot 1 (GOLDEN_FRAME)
	altref:          [3][]u8, // slot 2 (ALTREF_FRAME)
	last_recon:      [3][]u8,
	strides:         [3]int,
	aligned_w:       [3]int,
	aligned_h:       [3]int,

	input:           [3][]u8,
	input_strides:   [3]int,
	source:          [3][]u8,
	source_strides:  [3]int,

	// Frame-wide `#soa` mode/filter grids (sized to `mi_rows * mi_cols`).
	mi_cells:        #soa[]av1.MI_Cell,
	lf_cells:        #soa[]av1.LF_Cell,
	tile_cells:      #soa[]Tile_Cell,
	segment_ids:     []u8,
	prev_segment_ids: []u8,
	cdef_idx:        []i8,

	// Segmentation state (mirrors Frame_Header fields when enabled).
	segmentation_enabled:         bool,
	segmentation_update_map:      bool,
	segmentation_temporal_update: bool,
	feature_enabled:              [av1.MAX_SEGMENTS]u8,
	feature_data:                 [av1.MAX_SEGMENTS][av1.SEG_LVL_MAX]i16,
	last_seg_enabled:             [av1.MAX_SEGMENTS]u8,
	last_seg_data:                [av1.MAX_SEGMENTS][av1.SEG_LVL_MAX]i16,
	seg_id_pre_skip:              bool,
	last_active_seg_id:           int,
	lossless_array:               [av1.MAX_SEGMENTS]bool,

	mi_cols, mi_rows: int,
	sb_cols, sb_rows: int,

	cdfs:                   av1.CDF_Context,
	slot_cdfs:              [3]av1.CDF_Context,
	slot_order_hints:       [3]u8,
	slot_saved_order_hints: [3][av1.TOTAL_REFS_PER_FRAME]u8,
	slot_frame_type:        [3]av1.Frame_Type,
	cur_saved_mvs:          #soa[]av1.Saved_MV_Cell,
	slot_saved_mvs:         [3]#soa[]av1.Saved_MV_Cell,
	motion_field_mvs:       [av1.TOTAL_REFS_PER_FRAME][]av1.Mv,

	tiles:           []Tile,
	packet:          [dynamic]u8,
	seq_obu:         [dynamic]u8,
	header_buf:      [dynamic]u8,

	frames:          u64,
	order_hint:      u8,
	bit_depth:       int,
	profile:         int,
	ss_x, ss_y:      int,
	mono_chrome:     bool,
	q_idx:           int,
	primary_ref_slot: int,
	refresh_frame_flags: u8,
	keyframe:        bool,
	last_keyframe:   bool,
	show_frame:      bool,
	last_show_frame: bool,
	refresh_golden:  bool,
	has_altref:      bool,
	allow_comp:      bool,
	allow_hp:        bool,
	lossless:        bool,
	last_non_skip_cells: int,
	initialized:     bool,
}

init_encoder :: proc(e: ^Encoder, config: Encoder_Config) -> Error {
	if e == nil ||
	   config.width <= 0 || config.height <= 0 ||
	   config.width > 65536 || config.height > 65536 ||
	   (config.bit_depth != 0 && config.bit_depth != 8 && config.bit_depth != 10 && config.bit_depth != 12) ||
	   int(config.subsampling) < 0 || int(config.subsampling) > int(Chroma_Subsampling.Monochrome) ||
	   config.base_q_idx < 0 || config.base_q_idx > 255 ||
	   config.delta_q_y_dc < -64 || config.delta_q_y_dc > 63 ||
	   config.delta_q_u_dc < -64 || config.delta_q_u_dc > 63 ||
	   config.delta_q_u_ac < -64 || config.delta_q_u_ac > 63 ||
	   config.delta_q_v_dc < -64 || config.delta_q_v_dc > 63 ||
	   config.delta_q_v_ac < -64 || config.delta_q_v_ac > 63 ||
	   config.keyframe_interval < 0 ||
	   config.tile_cols_log2 < 0 || config.tile_rows_log2 < 0 ||
	   config.search_range < 0 || config.search_range > 64 ||
	   (config.block_log2 != 0 && (config.block_log2 < 2 || config.block_log2 > 6)) {
		return .Invalid_Argument
	}

	e^ = {config = config, allocator = context.allocator}
	e.bit_depth = config.bit_depth == 0 ? 8 : config.bit_depth
	switch config.subsampling {
	case .YUV420:     e.ss_x, e.ss_y, e.mono_chrome = 1, 1, false
	case .YUV422:     e.ss_x, e.ss_y, e.mono_chrome = 1, 0, false
	case .YUV444:     e.ss_x, e.ss_y, e.mono_chrome = 0, 0, false
	case .Monochrome: e.ss_x, e.ss_y, e.mono_chrome = 1, 1, true
	}

	// Select AV1 profile (Section 6.4.1):
	// Profile 0 (Main): 8/10-bit 4:2:0 or Monochrome.
	// Profile 1 (High): 8/10-bit 4:4:4.
	// Profile 2 (Professional): 12-bit (any) or 4:2:2 (8/10/12-bit).
	if e.bit_depth == 12 || config.subsampling == .YUV422 {
		e.profile = 2
	} else if config.subsampling == .YUV444 {
		e.profile = 1
	} else {
		e.profile = 0
	}

	if e.config.keyframe_interval == 0 do e.config.keyframe_interval = 120
	if e.config.search_range == 0 do e.config.search_range = 16
	if e.config.block_log2 == 0 do e.config.block_log2 = 6

	e.mi_cols = 2 * ((config.width + 7) >> 3)
	e.mi_rows = 2 * ((config.height + 7) >> 3)
	e.sb_cols = (e.mi_cols + 15) >> 4
	e.sb_rows = (e.mi_rows + 15) >> 4
	if i64(e.sb_cols) * i64(e.sb_rows) * 4096 > i64(max(int)) / 16 {
		e^ = {}
		return .Out_Of_Memory
	}

	init_sequence_header(e)
	setup_tile_info(e)

	ok := false
	defer if !ok do destroy_encoder(e)
	err: mem.Allocator_Error

	cells := e.mi_cols * e.mi_rows
	e.mi_cells, err = make_soa(#soa[]av1.MI_Cell, cells)
	if err != nil do return .Out_Of_Memory
	e.lf_cells, err = make_soa(#soa[]av1.LF_Cell, cells)
	if err != nil do return .Out_Of_Memory
	e.tile_cells, err = make_soa(#soa[]Tile_Cell, cells)
	if err != nil do return .Out_Of_Memory
	e.segment_ids, err = make([]u8, cells)
	if err != nil do return .Out_Of_Memory
	e.prev_segment_ids, err = make([]u8, cells)
	if err != nil do return .Out_Of_Memory
	e.cdef_idx, err = make([]i8, e.sb_cols * e.sb_rows)
	if err != nil do return .Out_Of_Memory
	n8 := (e.mi_rows >> 1) * (e.mi_cols >> 1)
	e.cur_saved_mvs, err = make_soa(#soa[]av1.Saved_MV_Cell, n8)
	if err != nil do return .Out_Of_Memory
	for slot in 0 ..< 3 {
		e.slot_saved_mvs[slot], err = make_soa(#soa[]av1.Saved_MV_Cell, n8)
		if err != nil do return .Out_Of_Memory
	}
	for ref in av1.LAST_FRAME ..= av1.ALTREF_FRAME {
		e.motion_field_mvs[ref], err = make([]av1.Mv, n8)
		if err != nil do return .Out_Of_Memory
	}

	sample_bytes := e.bit_depth > 8 ? 2 : 1
	num_planes := e.mono_chrome ? 1 : 3
	for p in 0 ..< num_planes {
		sx := p > 0 ? uint(e.ss_x) : 0
		sy := p > 0 ? uint(e.ss_y) : 0
		aw := ((config.width + 127) & ~int(127)) >> sx
		ah := ((config.height + 127) & ~int(127)) >> sy
		e.aligned_w[p], e.aligned_h[p] = aw, ah
		e.strides[p] = aw
		bytes := aw * ah * sample_bytes
		e.planes[p], err = make([]u8, bytes)
		if err != nil do return .Out_Of_Memory
		e.reference[p], err = make([]u8, bytes)
		if err != nil do return .Out_Of_Memory
		e.golden[p], err = make([]u8, bytes)
		if err != nil do return .Out_Of_Memory
		e.altref[p], err = make([]u8, bytes)
		if err != nil do return .Out_Of_Memory
		e.source_strides[p] = aw
		e.source[p], err = make([]u8, bytes)
		if err != nil do return .Out_Of_Memory
	}
	e.last_recon = e.reference

	num_tiles := e.header.tile_info.tile_cols * e.header.tile_info.tile_rows
	e.tiles, err = make([]Tile, num_tiles)
	if err != nil do return .Out_Of_Memory
	for &t, tile_num in e.tiles {
		tile_row := tile_num / e.header.tile_info.tile_cols
		tile_col := tile_num % e.header.tile_info.tile_cols
		t.e = e
		t.tile_num = tile_num
		t.mi_row_start = int(e.header.tile_info.mi_row_starts[tile_row])
		t.mi_row_end = int(e.header.tile_info.mi_row_starts[tile_row + 1])
		t.mi_col_start = int(e.header.tile_info.mi_col_starts[tile_col])
		t.mi_col_end = int(e.header.tile_info.mi_col_starts[tile_col + 1])
		tile_area := max(64, (t.mi_col_end - t.mi_col_start) * (t.mi_row_end - t.mi_row_start) * 16 * 8)
		err = reserve(&t.sym.data, tile_area)
		if err != nil do return .Out_Of_Memory
	}

	err = reserve(&e.packet, e.sb_cols * e.sb_rows * 64 * 64 * 8)
	if err != nil do return .Out_Of_Memory
	err = reserve(&e.seq_obu, 64)
	if err != nil do return .Out_Of_Memory
	err = reserve(&e.header_buf, 256)
	if err != nil do return .Out_Of_Memory

	if !build_sequence_header_obu(e) do return .Out_Of_Memory
	e.initialized = true
	ok = true
	return .None
}

destroy_encoder :: proc(e: ^Encoder) {
	if e == nil do return
	context.allocator = e.allocator
	for p in 0 ..< 3 {
		delete(e.planes[p])
		delete(e.reference[p])
		delete(e.golden[p])
		delete(e.altref[p])
		delete(e.source[p])
	}
	for &t in e.tiles {
		delete(t.sym.data)
	}
	delete(e.tiles)
	delete_soa(e.mi_cells)
	delete_soa(e.lf_cells)
	delete_soa(e.tile_cells)
	delete(e.segment_ids)
	delete(e.prev_segment_ids)
	delete(e.cdef_idx)
	delete_soa(e.cur_saved_mvs)
	for slot in 0 ..< 3 {
		delete_soa(e.slot_saved_mvs[slot])
	}
	for ref in av1.LAST_FRAME ..= av1.ALTREF_FRAME {
		delete(e.motion_field_mvs[ref])
	}
	delete(e.packet)
	delete(e.seq_obu)
	delete(e.header_buf)
	e^ = {}
}

// Packet and reconstructed planes are borrowed until the next call/destroy.
// Input is planar 8-bit, with ceil(width >> ss_x), ceil(height >> ss_y) chroma.
encode_frame :: proc(e: ^Encoder, planes: [3][]u8, strides: [3]int) -> (packet: []u8, keyframe: bool, err: Error) {
	if e != nil && e.initialized && e.bit_depth != 8 do return nil, false, .Invalid_Argument
	return encode_frame_internal(e, planes, strides, false)
}

// Input is planar 10-bit or 12-bit in u16 samples; strides count samples.
encode_frame16 :: proc(e: ^Encoder, planes: [3][]u16, strides: [3]int) -> (packet: []u8, keyframe: bool, err: Error) {
	if e == nil || !e.initialized do return nil, false, .Closed
	if e.bit_depth <= 8 do return nil, false, .Invalid_Argument
	bytes := [3][]u8{slice.to_bytes(planes[0]), slice.to_bytes(planes[1]), slice.to_bytes(planes[2])}
	return encode_frame_internal(e, bytes, strides, false)
}

// Encodes an unshown alternate reference frame (show_frame = 0, showable_frame = 1) into slot 2 (ALTREF_FRAME).
encode_altref_frame :: proc(e: ^Encoder, planes: [3][]u8, strides: [3]int) -> (packet: []u8, err: Error) {
	if e != nil && e.initialized && (e.frames == 0 || e.bit_depth != 8) do return nil, .Invalid_Argument
	pkt, _, er := encode_frame_internal(e, planes, strides, true)
	return pkt, er
}

encode_altref_frame16 :: proc(e: ^Encoder, planes: [3][]u16, strides: [3]int) -> (packet: []u8, err: Error) {
	if e == nil || !e.initialized do return nil, .Closed
	if e.frames == 0 || e.bit_depth <= 8 do return nil, .Invalid_Argument
	bytes := [3][]u8{slice.to_bytes(planes[0]), slice.to_bytes(planes[1]), slice.to_bytes(planes[2])}
	pkt, _, er := encode_frame_internal(e, bytes, strides, true)
	return pkt, er
}

// Emits an OBU_FRAME_HEADER with show_existing_frame = 1 referencing `slot` (default slot 2 = ALTREF_FRAME).
encode_existing_frame :: proc(e: ^Encoder, slot: int = 2) -> (packet: []u8, err: Error) {
	if e == nil || !e.initialized do return nil, .Closed
	if slot < 0 || slot >= 3 do return nil, .Invalid_Argument
	context.allocator = e.allocator
	if !build_show_existing_obu(e, slot) do return nil, .Out_Of_Memory
	e.last_recon = slot == 2 ? e.altref : slot == 1 ? e.golden : e.reference
	e.frames += 1
	return e.packet[:], .None
}

// Latest reconstructed (post-filter) image, same lifetime as encode_frame's packet.
reconstruction :: proc(e: ^Encoder) -> (planes: [3][]u8, strides: [3]int) {
	return e.last_recon, e.strides
}

reconstruction16 :: proc(e: ^Encoder) -> (planes: [3][]u16, strides: [3]int) {
	return {
		plane_t(e.last_recon[0], u16),
		plane_t(e.last_recon[1], u16),
		plane_t(e.last_recon[2], u16),
	}, e.strides
}

@(private)
scale_q_idx :: proc(bit_depth, base_q_idx: int, num, den: i32) -> int {
	target := (av1.ac_quant(base_q_idx, 0, bit_depth) * num + (den >> 1)) / den
	q := 1
	for q < 255 && av1.ac_quant(q, 0, bit_depth) < target do q += 1
	return q
}

@(private)
encode_frame_internal :: proc(e: ^Encoder, planes: [3][]u8, strides: [3]int, is_altref: bool) -> (packet: []u8, keyframe: bool, err: Error) {
	if e == nil || !e.initialized do return nil, false, .Closed
	sample_bytes := i128(e.bit_depth > 8 ? 2 : 1)
	num_planes := e.mono_chrome ? 1 : 3
	for p in 0 ..< num_planes {
		sx := p > 0 ? e.ss_x : 0
		sy := p > 0 ? e.ss_y : 0
		w := (e.config.width + sx) >> uint(sx)
		h := (e.config.height + sy) >> uint(sy)
		if strides[p] < w || (i128(h - 1) * i128(strides[p]) + i128(w)) * sample_bytes > i128(len(planes[p])) {
			return nil, false, .Invalid_Argument
		}
	}

	context.allocator = e.allocator
	e.input, e.input_strides = planes, strides
	gop_frame := e.frames % u64(e.config.keyframe_interval)

	if is_altref {
		e.keyframe = false
		e.show_frame = false
		e.refresh_golden = false
		e.refresh_frame_flags = 1 << 2
		e.allow_comp = !e.config.ultrafast && !e.last_keyframe && e.config.base_q_idx > 0
		e.order_hint = u8((e.frames + 2) & 0xFF)
	} else {
		e.keyframe = gop_frame == 0
		e.show_frame = true
		e.refresh_golden = !e.keyframe && (gop_frame & 7) == 0 && e.config.keyframe_interval >= 8
		e.refresh_frame_flags = e.keyframe ? 0xFF : (e.refresh_golden ? 3 : 1)
		e.allow_comp = !e.config.ultrafast && !e.keyframe && ((!e.last_keyframe && (gop_frame & 7) != 1 && e.config.keyframe_interval >= 8) || e.has_altref) && e.config.base_q_idx > 0
		e.order_hint = u8((e.frames + 1) & 0xFF)
	}

	e.q_idx = e.config.base_q_idx
	if e.config.base_q_idx > 0 && e.config.keyframe_interval > 1 {
		if e.keyframe {
			flat, total := 0, 0
			if e.bit_depth == 8 {
				for y := 1; y < e.config.height; y += 4 {
					row := planes[0][y * strides[0]:][:e.config.width]
					prev := planes[0][(y - 1) * strides[0]:][:e.config.width]
					for x := 0; x < len(row); x += 4 {
						flat += int(row[x] == prev[x])
						total += 1
					}
				}
			} else {
				p16 := plane_t(planes[0], u16)
				for y := 1; y < e.config.height; y += 4 {
					row := p16[y * strides[0]:][:e.config.width]
					prev := p16[(y - 1) * strides[0]:][:e.config.width]
					for x := 0; x < len(row); x += 4 {
						flat += int(row[x] == prev[x])
						total += 1
					}
				}
			}
			e.q_idx = flat * 3 > total * 2 ? scale_q_idx(e.bit_depth, e.config.base_q_idx, 45, 100) : scale_q_idx(e.bit_depth, e.config.base_q_idx, 75, 100)
		} else if is_altref {
			e.q_idx = scale_q_idx(e.bit_depth, e.config.base_q_idx, 65, 100)
		} else if (e.last_keyframe ? e.last_non_skip_cells * 5 : e.last_non_skip_cells * 16) < e.mi_cols * e.mi_rows && e.config.keyframe_interval >= 8 {
			gf_rates := [8]i32{65, 110, 95, 110, 80, 110, 95, 110}
			e.q_idx = scale_q_idx(e.bit_depth, e.config.base_q_idx, gf_rates[gop_frame & 7], 100)
		}
	}

	prepare_frame_header(e, is_altref)

	// Motion field estimation (Section 7.9).
	if !e.header.frame_is_intra && e.header.use_ref_frame_mvs {
		saved_refs: [av1.NUM_REF_FRAMES]av1.Saved_Ref_MVs
		for i in 0 ..< 3 {
			saved_refs[i] = {
				cells       = e.slot_saved_mvs[i],
				mi_rows     = e.mi_rows,
				mi_cols     = e.mi_cols,
				frame_type  = u8(e.slot_frame_type[i]),
				order_hint  = e.slot_order_hints[i],
				order_hints = e.slot_saved_order_hints[i],
			}
		}
		ref_idx_i8: [av1.REFS_PER_FRAME]i8
		for i in 0 ..< av1.REFS_PER_FRAME do ref_idx_i8[i] = i8(e.header.ref_frame_idx[i])
		av1.motion_field_estimation(
			&e.motion_field_mvs,
			&saved_refs,
			ref_idx_i8,
			e.header.order_hints,
			e.header.order_hint,
			e.seq.enable_order_hint,
			u8(e.seq.order_hint_bits),
			e.mi_rows,
			e.mi_cols,
		)
	}

	// Reset frame-wide SOA grids.
	cells := e.mi_rows * e.mi_cols
	for i in 0 ..< cells {
		e.mi_cells[i] = {ref_frames = {av1.NONE, av1.NONE}}
		e.lf_cells[i] = {}
		e.tile_cells[i] = {}
		if !e.segmentation_enabled || !e.segmentation_update_map {
			e.segment_ids[i] = 0
		}
	}
	for &ci in e.cdef_idx do ci = -1

	// Initialize or load frame CDF context (Section 6.8.2).
	if e.header.primary_ref_frame == av1.PRIMARY_REF_NONE {
		av1.init_cdf_context(&e.cdfs, e.header.base_q_idx)
		for &sc in e.slot_cdfs do sc = e.cdfs
		for &v in e.prev_segment_ids do v = 0
	} else {
		av1.load_cdfs(&e.cdfs, &e.slot_cdfs[e.primary_ref_slot])
	}

	for &t in e.tiles {
		t.e = e
		t.cdfs = e.cdfs
	}

	if e.config.pool != nil && e.config.pool.run != nil && len(e.tiles) > 1 {
		e.config.pool.run(e.config.pool, len(e.tiles), encode_tiles, raw_data(e.tiles))
	} else {
		encode_tiles(raw_data(e.tiles), 0, len(e.tiles))
	}

	for &t in e.tiles {
		if t.sym.failed do return nil, false, .Out_Of_Memory
	}

	if !assemble_frame_packet(e) {
		return nil, false, .Out_Of_Memory
	}

	// Frame-end CDF update from `context_update_tile_id` (Section 5.11.2).
	if !e.header.disable_frame_end_update_cdf && int(e.header.tile_info.context_update_tile_id) < len(e.tiles) {
		e.cdfs = e.tiles[e.header.tile_info.context_update_tile_id].cdfs
	}

	// Post-processing wrapup (deblocking loop filter + CDEF).
	filter_frame(e)

	non_skip := 0
	for s in e.tile_cells.skip[:cells] {
		if !s do non_skip += 1
	}
	e.last_non_skip_cells = non_skip

	// Motion field motion vector storage (Section 7.19).
	if !e.header.frame_is_intra && e.refresh_frame_flags != 0 && e.seq.enable_ref_frame_mvs {
		grid := av1.MI_Grid{
			cells   = e.mi_cells[:cells],
			mi_rows = e.mi_rows,
			mi_cols = e.mi_cols,
		}
		saved_dst := av1.Saved_Ref_MVs{
			cells   = e.cur_saved_mvs,
			mi_rows = e.mi_rows,
			mi_cols = e.mi_cols,
		}
		ref_idx_i8: [av1.REFS_PER_FRAME]i8
		for i in 0 ..< av1.REFS_PER_FRAME do ref_idx_i8[i] = i8(e.header.ref_frame_idx[i])
		ref_order_hint: [av1.NUM_REF_FRAMES]u8
		for i in 0 ..< 3 do ref_order_hint[i] = e.slot_order_hints[i]
		av1.store_motion_field_mvs(
			&saved_dst,
			&grid,
			ref_idx_i8,
			ref_order_hint,
			e.header.order_hint,
			e.seq.enable_order_hint,
			u8(e.seq.order_hint_bits),
		)
	}

	// Update reference slots and CDFs.
	n8 := (e.mi_rows >> 1) * (e.mi_cols >> 1)
	for slot in 0 ..< 3 {
		if (e.refresh_frame_flags & (1 << uint(slot))) != 0 {
			e.slot_cdfs[slot] = e.cdfs
			e.slot_order_hints[slot] = e.header.order_hint
			e.slot_saved_order_hints[slot] = e.header.order_hints
			e.slot_frame_type[slot] = e.header.frame_type
			if !e.header.frame_is_intra && e.seq.enable_ref_frame_mvs {
				copy(e.slot_saved_mvs[slot].mv[:n8], e.cur_saved_mvs.mv[:n8])
				copy(e.slot_saved_mvs[slot].ref_frame[:n8], e.cur_saved_mvs.ref_frame[:n8])
			}
		}
	}

	if e.segmentation_enabled {
		e.last_seg_enabled = e.feature_enabled
		e.last_seg_data = e.feature_data
		if e.segmentation_update_map {
			copy(e.prev_segment_ids, e.segment_ids)
		}
	}

	e.last_keyframe = e.keyframe
	e.last_show_frame = e.show_frame

	if is_altref {
		for p in 0 ..< num_planes {
			e.planes[p], e.altref[p] = e.altref[p], e.planes[p]
		}
		e.has_altref = true
		e.last_recon = e.altref
	} else {
		for p in 0 ..< num_planes {
			e.planes[p], e.reference[p] = e.reference[p], e.planes[p]
			if e.keyframe || e.refresh_golden do copy(e.golden[p], e.reference[p])
			if e.keyframe do copy(e.altref[p], e.reference[p])
		}
		if e.keyframe do e.has_altref = false
		e.last_recon = e.reference
		e.frames += 1
	}

	return e.packet[:], e.keyframe, .None
}

@(private)
encode_tiles :: proc(data: rawptr, first, end: int) {
	tiles := (cast([^]Tile)data)[first:end]
	for &t in tiles {
		context.allocator = t.e.allocator
		encode_tile(&t)
	}
}

@(private)
filter_frame :: proc(e: ^Encoder) {
	h := &e.header
	s := &e.seq
	cells := e.mi_rows * e.mi_cols
	sub_x := uint(e.ss_x)
	sub_y := uint(e.ss_y)
	num_planes := e.mono_chrome ? 1 : 3

	// 1. Deblocking loop filter in place on `e.planes` (Section 7.14).
	if !h.all_lossless && (h.loop_filter_level[0] != 0 || h.loop_filter_level[1] != 0) {
		if e.bit_depth > 8 {
			p16 := [3][]u16{
				plane_t(e.planes[0], u16),
				e.mono_chrome ? nil : plane_t(e.planes[1], u16),
				e.mono_chrome ? nil : plane_t(e.planes[2], u16),
			}
			av1.loop_filter_frame(p16, e.strides, e.lf_cells[:cells], h, num_planes, sub_x, sub_y, e.bit_depth)
		} else {
			av1.loop_filter_frame(e.planes, e.strides, e.lf_cells[:cells], h, num_planes, sub_x, sub_y, e.bit_depth)
		}
	}

	// 2. CDEF (Section 7.15).
	if !h.coded_lossless && s.enable_cdef && !h.allow_intrabc {
		skips := e.tile_cells.skip[:cells]
		if e.bit_depth > 8 {
			p16 := [3][]u16{
				plane_t(e.planes[0], u16),
				e.mono_chrome ? nil : plane_t(e.planes[1], u16),
				e.mono_chrome ? nil : plane_t(e.planes[2], u16),
			}
			av1.cdef_frame(
				p16, e.strides,
				p16, e.strides,
				e.cdef_idx, e.sb_cols,
				skips, h, s.enable_cdef, num_planes, sub_x, sub_y, e.bit_depth,
			)
		} else {
			av1.cdef_frame(
				e.planes, e.strides,
				e.planes, e.strides,
				e.cdef_idx, e.sb_cols,
				skips, h, s.enable_cdef, num_planes, sub_x, sub_y, e.bit_depth,
			)
		}
	}
}
