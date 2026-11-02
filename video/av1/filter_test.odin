package av1

import "core:testing"

@(test)
test_loop_filter_frame :: proc(t: ^testing.T) {
	w, h := 32, 32
	mi_cols, mi_rows := w >> 2, h >> 2
	grid := make(#soa[]LF_Cell, mi_rows * mi_cols)
	defer delete(grid)

	for i in 0 ..< len(grid) {
		grid[i].levels = {24, 24, 20, 20}
		grid[i].tx_size_y = u8(TX_4X4)
		grid[i].tx_size_uv = u8(TX_4X4)
		grid[i].mi_size = u8(BLOCK_8X8)
		grid[i].skip_inter = false
	}

	y := make([]u16, w * h)
	defer delete(y)
	u := make([]u16, (w >> 1) * (h >> 1))
	defer delete(u)
	v := make([]u16, (w >> 1) * (h >> 1))
	defer delete(v)

	// Create a step edge across x = 16 and y = 16 within the deblocking threshold
	for r in 0 ..< h {
		for c in 0 ..< w {
			val := u16(120)
			if c >= 16 {
				val += 6
			}
			if r >= 16 {
				val += 6
			}
			y[r * w + c] = val
		}
	}
	for r in 0 ..< (h >> 1) {
		for c in 0 ..< (w >> 1) {
			u[r * (w >> 1) + c] = u16(128 + (4 if c >= 8 else 0))
			v[r * (w >> 1) + c] = u16(128 + (4 if r >= 8 else 0))
		}
	}

	before_edge := y[8 * w + 15]
	hdr := Frame_Header{
		frame_width           = w,
		frame_height          = h,
		mi_cols               = mi_cols,
		mi_rows               = mi_rows,
		loop_filter_level     = {24, 24, 20, 20},
		loop_filter_sharpness = 0,
	}

	planes := [3][]u16{y, u, v}
	strides := [3]int{w, w >> 1, w >> 1}
	loop_filter_frame(
		planes,
		strides,
		grid,
		&hdr,
		3,
		1,
		1,
		8,
	)

	// Verify the step edge at x=15/16 was smoothed toward the right side
	testing.expect(t, y[8 * w + 15] > before_edge, "expected vertical loop filter to smooth step edge")
	testing.expect(t, y[8 * w + 16] < 126, "expected vertical loop filter to reduce high side of step edge")
}

@(test)
test_cdef_frame :: proc(t: ^testing.T) {
	w, h := 16, 16
	mi_cols, mi_rows := w >> 2, h >> 2
	y := make([]u16, w * h)
	defer delete(y)
	u := make([]u16, (w >> 1) * (h >> 1))
	defer delete(u)
	v := make([]u16, (w >> 1) * (h >> 1))
	defer delete(v)
	cy := make([]u16, w * h)
	defer delete(cy)
	cu := make([]u16, (w >> 1) * (h >> 1))
	defer delete(cu)
	cv := make([]u16, (w >> 1) * (h >> 1))
	defer delete(cv)

	for r in 0 ..< h {
		for c in 0 ..< w {
			// Diagonal edge with small ringing oscillation
			base := u16(100 if r + c < 14 else 180)
			ring := u16(3 if ((r + c) & 1) != 0 else 0)
			y[r * w + c] = base + ring
		}
	}
	for i in 0 ..< len(u) {
		u[i] = 128
		v[i] = 128
	}

	dir, var_out := cdef_direction(y, w, 0, 0, 8)
	testing.expect(t, dir >= 0 && dir < 8, "expected valid CDEF direction")
	testing.expect(t, var_out > 0, "expected positive CDEF directional variance on diagonal edge")

	cdef_skip := make([]bool, mi_rows * mi_cols)
	defer delete(cdef_skip)
	cdef_idx := make([]i8, 1)
	defer delete(cdef_idx)
	cdef_idx[0] = 0

	hdr := Frame_Header{
		frame_width          = w,
		frame_height         = h,
		mi_cols              = mi_cols,
		mi_rows              = mi_rows,
		cdef_damping_minus_3 = 0,
		cdef_bits            = 0,
		cdef_y_pri_strength  = {8, 0, 0, 0, 0, 0, 0, 0},
		cdef_y_sec_strength  = {2, 0, 0, 0, 0, 0, 0, 0},
		cdef_uv_pri_strength = {4, 0, 0, 0, 0, 0, 0, 0},
		cdef_uv_sec_strength = {1, 0, 0, 0, 0, 0, 0, 0},
	}

	cdef_planes := [3][]u16{cy, cu, cv}
	curr_planes := [3][]u16{y, u, v}
	strides := [3]int{w, w >> 1, w >> 1}

	cdef_frame(
		cdef_planes,
		strides,
		curr_planes,
		strides,
		cdef_idx,
		1,
		cdef_skip,
		&hdr,
		true,
		3,
		1,
		1,
		8,
	)

	for i in 0 ..< len(cy) {
		testing.expect(t, cy[i] >= 100 && cy[i] <= 183, "CDEF output must stay bounded")
	}
}

@(test)
test_superres_and_loop_restoration :: proc(t: ^testing.T) {
	// 1. Test Superres upscaling (16 -> 24 width)
	in_w, h := 16, 16
	out_w := 24
	src_y := make([]u16, in_w * h)
	defer delete(src_y)
	dst_y := make([]u16, out_w * h)
	defer delete(dst_y)

	for r in 0 ..< h {
		for c in 0 ..< in_w {
			src_y[r * in_w + c] = u16(64 + c * 8)
		}
	}

	upscale_plane(dst_y, out_w, src_y, in_w, in_w, out_w, h, in_w >> 2, 0, 0, 8)
	for r in 0 ..< h {
		testing.expect(t, dst_y[r * out_w + 0] == 64, "left boundary preserved by superres")
		testing.expect(t, dst_y[r * out_w + out_w - 1] >= 170, "right ramp preserved by superres")
	}

	// 2. Test Loop Restoration (Wiener + SGRPROJ on #soa[]LR_Unit)
	units := make(#soa[]LR_Unit, 2)
	defer delete(units)

	// Unit 0: Wiener identity filter
	units[0].restoration_type = RESTORE_WIENER
	units[0].wiener = {{0, 0, 0}, {0, 0, 0}}

	// Unit 1: SGRPROJ filter
	units[1].restoration_type = RESTORE_SGRPROJ
	units[1].sgr_set = 10
	units[1].sgr_xqd = {-32, 31}

	cdef_plane := make([]u16, out_w * h)
	defer delete(cdef_plane)
	curr_plane := make([]u16, out_w * h)
	defer delete(curr_plane)
	lr_plane := make([]u16, out_w * h)
	defer delete(lr_plane)

	for i in 0 ..< len(cdef_plane) {
		cdef_plane[i] = u16(128 + ((i * 7) & 7))
		curr_plane[i] = 130
	}

	hdr := Frame_Header{
		frame_width            = out_w,
		upscaled_width         = out_w,
		frame_height           = h,
		uses_lr                = true,
		frame_restoration_type = {RESTORE_SWITCHABLE, RESTORE_NONE, RESTORE_NONE},
		loop_restoration_size  = {64, 64, 64},
	}

	lr_planes := [3][]u16{lr_plane, nil, nil}
	cdef_planes := [3][]u16{cdef_plane, nil, nil}
	curr_planes := [3][]u16{curr_plane, nil, nil}
	strides := [3]int{out_w, 0, 0}

	loop_restore_frame(
		lr_planes,
		strides,
		cdef_planes,
		strides,
		curr_planes,
		strides,
		{units[0:1], nil, nil},
		&hdr,
		1,
		0,
		0,
		8,
	)

	// Also test SGRPROJ unit directly
	units[0] = units[1]
	loop_restore_frame(
		lr_planes,
		strides,
		cdef_planes,
		strides,
		curr_planes,
		strides,
		{units[0:1], nil, nil},
		&hdr,
		1,
		0,
		0,
		8,
	)

	for i in 0 ..< len(lr_plane) {
		testing.expect(t, lr_plane[i] <= 255, "Loop restoration must stay within 8-bit range")
	}
}

@(test)
test_film_grain_synthesis :: proc(t: ^testing.T) {
	w, h := 64, 64
	y := make([]u16, w * h)
	defer delete(y)
	u := make([]u16, (w >> 1) * (h >> 1))
	defer delete(u)
	v := make([]u16, (w >> 1) * (h >> 1))
	defer delete(v)

	for i in 0 ..< len(y) {
		y[i] = 128
	}
	for i in 0 ..< len(u) {
		u[i] = 128
		v[i] = 128
	}

	fg := Film_Grain{
		apply_grain              = true,
		grain_seed               = 7391,
		num_y_points             = 2,
		point_y_value            = {0 = 0, 1 = 255},
		point_y_scaling          = {0 = 64, 1 = 64},
		chroma_scaling_from_luma = false,
		num_cb_points            = 2,
		point_cb_value           = {0 = 0, 1 = 255},
		point_cb_scaling         = {0 = 32, 1 = 32},
		num_cr_points            = 2,
		point_cr_value           = {0 = 0, 1 = 255},
		point_cr_scaling         = {0 = 32, 1 = 32},
		grain_scaling_minus_8    = 3,
		ar_coeff_lag             = 1,
		ar_coeff_shift_minus_6   = 2,
		grain_scale_shift        = 0,
		cb_mult                  = 128,
		cb_luma_mult             = 192,
		cb_offset                = 256,
		cr_mult                  = 128,
		cr_luma_mult             = 192,
		cr_offset                = 256,
		overlap_flag             = true,
		clip_to_restricted_range = false,
	}

	planes := [3][]u16{y, u, v}
	strides := [3]int{w, w >> 1, w >> 1}
	apply_film_grain(
		planes,
		strides,
		w,
		h,
		&fg,
		false,
		1,
		1,
		0,
		8,
	)

	diff_count := 0
	for i in 0 ..< len(y) {
		if y[i] != 128 {
			diff_count += 1
		}
		testing.expect(t, y[i] <= 255, "Film grain output must stay within 8-bit range")
	}
	testing.expect(t, diff_count > (w * h) / 4, "Expected film grain to add non-zero grain samples")
}
