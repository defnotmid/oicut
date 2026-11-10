package vp9_encoder

import vp9 ".."

// The loop filter level follows the quantizer; the encoder signals no deltas
// or segments, so one level covers every block.
@(private)
encoder_filter_levels :: proc(e: ^Encoder) -> vp9.Filter_Levels {
	if e.lossless || e.config.disable_loop_filter {
		e.base_filter_level = 0
		return {}
	}
	q := int(vp9.ac_q(8, e.q_idx))
	level := (q * 20723 + 1015158 + (1 << 17)) >> 18
	if e.keyframe do level -= 4
	e.base_filter_level = clamp(level, 0, 63)
	lf := vp9.Loop_Filter{
		level         = e.base_filter_level,
		delta_enabled = !e.config.ultrafast,
		ref_deltas    = {1, 0, -1, -1},
		mode_deltas   = {0, 0},
	}
	return vp9.filter_levels(&lf, &e.segmentation)
}

@(private)
filter_frame :: proc(e: ^Encoder) {
	if e.lossless || e.config.disable_loop_filter do return
	if e.bit_depth == 8 {
		filter_frame_typed(e, u8)
	} else {
		filter_frame_typed(e, u16)
	}
}

@(private)
filter_frame_typed :: proc(e: ^Encoder, $T: typeid) {
	planes := [3][]T{plane_t(e.planes[0], T), plane_t(e.planes[1], T), plane_t(e.planes[2], T)}
	f := vp9.Loop_Filter_Frame(T){planes = planes, strides = e.strides, cells = e.filter_cells, mi_cols = e.mi_cols, mi_rows = e.mi_rows, ss_x = e.ss_x, ss_y = e.ss_y, bit_depth = e.bit_depth, limits = vp9.filter_limits(0)}
	vp9.loop_filter_frame(&f, e.sb_rows, e.sb_cols, e.config.pool)
}
