package av1

// Normative intra prediction (section 7.11.2), palette prediction (7.11.4),
// and Chroma-from-Luma (CfL) prediction (7.11.5), generic over 8-bit (`u8`)
// and high-bit-depth (`u16`) planes.

import "base:intrinsics"

Intra_Edges :: struct {
	have_left, have_above:               bool,
	have_above_right, have_below_left:   bool,
	max_x, max_y:                        int, // last valid coordinate in this plane
	enable_intra_edge_filter:            bool,
	filter_type:                         int, // 0 or 1 (from neighbor is_smooth, 7.11.2.8)
	angle_delta:                         int, // -3..3
	use_filter_intra:                    bool,
	filter_intra_mode:                   int, // 0..4
}

// 7.11.2.8 helper: returns true if `mode` is one of the smooth intra modes.
is_smooth_mode :: #force_inline proc "contextless" (mode: u8) -> bool {
	return mode == SMOOTH_PRED || mode == SMOOTH_V_PRED || mode == SMOOTH_H_PRED
}

// 7.11.2.4 helper: returns true if `mode` is a directional intra mode.
is_directional_mode :: #force_inline proc "contextless" (mode: int) -> bool {
	return mode >= V_PRED && mode <= D67_PRED
}

@(private)
sm_weights_for_log2 :: #force_inline proc "contextless" (log2_dim: int) -> []u8 {
	switch log2_dim {
	case 2:
		return SM_WEIGHTS_TX_4X4[:]
	case 3:
		return SM_WEIGHTS_TX_8X8[:]
	case 4:
		return SM_WEIGHTS_TX_16X16[:]
	case 5:
		return SM_WEIGHTS_TX_32X32[:]
	case:
		return SM_WEIGHTS_TX_64X64[:]
	}
}

// 7.11.2.9: Intra edge filter strength selection (0..3).
@(private)
intra_edge_filter_strength :: #force_inline proc "contextless" (w, h, filter_type, delta: int) -> int {
	d := abs(delta)
	blk_wh := w + h
	strength := 0
	if filter_type == 0 {
		if blk_wh <= 8 {
			if d >= 56 do strength = 1
		} else if blk_wh <= 12 {
			if d >= 40 do strength = 1
		} else if blk_wh <= 16 {
			if d >= 40 do strength = 1
		} else if blk_wh <= 24 {
			if d >= 8 do strength = 1
			if d >= 16 do strength = 2
			if d >= 32 do strength = 3
		} else if blk_wh <= 32 {
			strength = 1
			if d >= 4 do strength = 2
			if d >= 32 do strength = 3
		} else {
			strength = 3
		}
	} else {
		if blk_wh <= 8 {
			if d >= 40 do strength = 1
			if d >= 64 do strength = 2
		} else if blk_wh <= 16 {
			if d >= 20 do strength = 1
			if d >= 48 do strength = 2
		} else if blk_wh <= 24 {
			if d >= 4 do strength = 3
		} else {
			strength = 3
		}
	}
	return strength
}

// 7.11.2.10: Intra edge upsample selection.
@(private)
intra_edge_upsample_select :: #force_inline proc "contextless" (w, h, filter_type, delta: int) -> bool {
	d := abs(delta)
	blk_wh := w + h
	if d <= 0 || d >= 40 do return false
	return filter_type == 0 ? (blk_wh <= 16) : (blk_wh <= 8)
}

// 7.11.2.12: Intra edge 5-tap filter in place on `buf` (where `buf[2]` is index 0
// and `buf[1]` is index -1).
@(private)
intra_edge_filter :: proc "contextless" (buf: []i32, sz, strength: int) #no_bounds_check {
	if strength == 0 || sz <= 1 do return
	edge: [130]i32 = ---
	for i in 0 ..< sz do edge[i] = buf[i + 1]
	kernel := INTRA_EDGE_KERNEL[strength - 1]
	for i in 1 ..< sz {
		s := i32(0)
		#unroll for j in 0 ..< INTRA_EDGE_TAPS {
			k := clamp(i - 2 + j, 0, sz - 1)
			s += i32(kernel[j]) * edge[k]
		}
		buf[i + 1] = (s + 8) >> 4
	}
}

// 7.11.2.11: Intra edge 2x upsampling in place on `buf` (where `buf[2 + k]`
// holds logical index `k` for `k >= -2`).
@(private)
intra_edge_upsample :: #force_no_inline proc "contextless" (buf: []i32, num_px, bit_depth: int) #no_bounds_check {
	if num_px <= 0 do return
	max_val := (i32(1) << uint(bit_depth)) - 1
	dup: [132]i32 = ---
	dup[0] = buf[1] // buf[-1]
	for i := -1; i < num_px; i += 1 {
		dup[i + 2] = buf[i + 2]
	}
	dup[num_px + 2] = buf[num_px - 1 + 2]

	buf[0] = dup[0] // buf[-2] = dup[0]
	for i in 0 ..< num_px {
		s := -dup[i] + 9 * dup[i + 1] + 9 * dup[i + 2] - dup[i + 3]
		s = clamp(round2(i64(s), 4), 0, max_val)
		buf[2 * i - 1 + 2] = s
		buf[2 * i + 2] = dup[i + 2]
	}
}

@(private)
predict_filter_intra :: #force_no_inline proc "contextless" (
	plane: []$T,
	stride, x, y, w, h, filter_intra_mode: int,
	above_buf, left_buf: ^[260]i32,
	max_val: i32,
) #no_bounds_check {
	V4 :: #simd[4]i32
	VT :: #simd[4]T
	pred: [32][32]i32 = ---
	w4 := w >> 2
	h2 := h >> 1
	taps := &INTRA_FILTER_TAPS[filter_intra_mode]
	for i2 in 0 ..< h2 {
		for j4 in 0 ..< w4 {
			p: [7]i32 = ---
			for i in 0 ..< 5 {
				if i2 == 0 {
					p[i] = above_buf[(j4 << 2) + i - 1 + 2]
				} else if j4 == 0 && i == 0 {
					p[i] = left_buf[(i2 << 1) - 1 + 2]
				} else {
					p[i] = pred[(i2 << 1) - 1][(j4 << 2) + i - 1]
				}
			}
			for i in 5 ..< 7 {
				if j4 == 0 {
					p[i] = left_buf[(i2 << 1) + i - 5 + 2]
				} else {
					p[i] = pred[(i2 << 1) + i - 5][(j4 << 2) - 1]
				}
			}
			for i1 in 0 ..< 2 {
				for j1 in 0 ..< 4 {
					pr := i32(0)
					k_taps := taps[(i1 << 2) + j1]
					#unroll for i in 0 ..< 7 {
						pr += i32(k_taps[i]) * p[i]
					}
					pred[(i2 << 1) + i1][(j4 << 2) + j1] = clamp(round2_signed(i64(pr), INTRA_FILTER_SCALE_BITS), 0, max_val)
				}
			}
		}
	}
	for i in 0 ..< h {
		row := plane[(y + i) * stride + x:]
		for j := 0; j < w; j += 4 {
			pv := intrinsics.unaligned_load((^V4)(&pred[i][j]))
			intrinsics.unaligned_store((^VT)(&row[j]), VT(pv))
		}
	}
}

// 7.11.2: Predicts a single transform block at (x, y) of size `tx_size` in `plane`.
predict_intra :: proc(plane: []$T, stride, x, y, tx_size, mode: int, e: Intra_Edges, bit_depth := 8) #no_bounds_check {
	V4 :: #simd[4]i32
	VT :: #simd[4]T

	log2_w := int(TX_WIDTH_LOG2[tx_size])
	log2_h := int(TX_HEIGHT_LOG2[tx_size])
	w := 1 << uint(log2_w)
	h := 1 << uint(log2_h)
	wh := w + h
	base_mid := i32(1) << uint(bit_depth - 1)
	max_val := (i32(1) << uint(bit_depth)) - 1

	// Logical index `k` (for k in -2 .. 2*(w+h)) is stored at `buf[k + 2]`.
	above_buf: [260]i32 = ---
	left_buf: [260]i32 = ---
	above := above_buf[2:]
	left := left_buf[2:]

	// Populate AboveRow[0 .. w + h - 1].
	if !e.have_above && e.have_left {
		v := i32(plane[y * stride + x - 1])
		for i in 0 ..< wh do above[i] = v
	} else if !e.have_above && !e.have_left {
		v := base_mid - 1
		for i in 0 ..< wh do above[i] = v
	} else {
		above_limit := min(e.max_x, x + (e.have_above_right ? 2 * w : w) - 1)
		row := plane[(y - 1) * stride:]
		for i in 0 ..< wh do above[i] = i32(row[min(above_limit, x + i)])
	}

	// Populate LeftCol[0 .. w + h - 1].
	if !e.have_left && e.have_above {
		v := i32(plane[(y - 1) * stride + x])
		for i in 0 ..< wh do left[i] = v
	} else if !e.have_left && !e.have_above {
		v := base_mid + 1
		for i in 0 ..< wh do left[i] = v
	} else {
		left_limit := min(e.max_y, y + (e.have_below_left ? 2 * h : h) - 1)
		for i in 0 ..< wh do left[i] = i32(plane[min(left_limit, y + i) * stride + x - 1])
	}

	// Populate AboveRow[-1] and LeftCol[-1].
	corner := base_mid
	if e.have_above && e.have_left {
		corner = i32(plane[(y - 1) * stride + x - 1])
	} else if e.have_above {
		corner = i32(plane[(y - 1) * stride + x])
	} else if e.have_left {
		corner = i32(plane[y * stride + x - 1])
	}
	above_buf[1] = corner
	left_buf[1] = corner

	// 7.11.2.3: Recursive intra prediction (filter_intra).
	if e.use_filter_intra {
		predict_filter_intra(plane, stride, x, y, w, h, e.filter_intra_mode, &above_buf, &left_buf, max_val)
		return
	}

	// 7.11.2.4: Directional intra prediction.
	if is_directional_mode(mode) {
		p_angle := int(MODE_TO_ANGLE[mode]) + e.angle_delta * ANGLE_STEP
		upsample_above := uint(0)
		upsample_left := uint(0)

		if e.enable_intra_edge_filter {
			if p_angle != 90 && p_angle != 180 {
				if p_angle > 90 && p_angle < 180 && wh >= 24 {
					// 7.11.2.7: Filter corner process.
					s := left[0] * 5 + above_buf[1] * 6 + above[0] * 5
					c_filt := round2(i64(s), 4)
					above_buf[1] = c_filt
					left_buf[1] = c_filt
				}
				if e.have_above {
					strength := intra_edge_filter_strength(w, h, e.filter_type, p_angle - 90)
					num_px := min(w, e.max_x - x + 1) + (p_angle < 90 ? h : 0) + 1
					intra_edge_filter(above_buf[:], num_px, strength)
				}
				if e.have_left {
					strength := intra_edge_filter_strength(w, h, e.filter_type, p_angle - 180)
					num_px := min(h, e.max_y - y + 1) + (p_angle > 180 ? w : 0) + 1
					intra_edge_filter(left_buf[:], num_px, strength)
				}
			}
			if intra_edge_upsample_select(w, h, e.filter_type, p_angle - 90) {
				upsample_above = 1
				num_px := w + (p_angle < 90 ? h : 0)
				intra_edge_upsample(above_buf[:], num_px, bit_depth)
			}
			if intra_edge_upsample_select(w, h, e.filter_type, p_angle - 180) {
				upsample_left = 1
				num_px := h + (p_angle > 180 ? w : 0)
				intra_edge_upsample(left_buf[:], num_px, bit_depth)
			}
		}

		switch {
		case p_angle == 90:
			for i in 0 ..< h {
				row := plane[(y + i) * stride + x:]
				for j := 0; j < w; j += 4 {
					av := intrinsics.unaligned_load((^V4)(&above[j]))
					intrinsics.unaligned_store((^VT)(&row[j]), VT(av))
				}
			}
		case p_angle == 180:
			for i in 0 ..< h {
				row := plane[(y + i) * stride + x:]
				lv: VT = T(left[i])
				for j := 0; j < w; j += 4 {
					intrinsics.unaligned_store((^VT)(&row[j]), lv)
				}
			}
		case p_angle < 90:
			dx := int(DR_INTRA_DERIVATIVE[p_angle])
			max_base_x := (wh - 1) << upsample_above
			for i in 0 ..< h {
				row := plane[(y + i) * stride + x:]
				idx := (i + 1) * dx
				base0 := idx >> (6 - upsample_above)
				shift := i32(((idx << upsample_above) >> 1) & 0x1F)
				w0 := 32 - shift
				for j in 0 ..< w {
					base := base0 + (j << upsample_above)
					if base < max_base_x {
						v := (above_buf[base + 2] * w0 + above_buf[base + 3] * shift + 16) >> 5
						row[j] = T(v)
					} else {
						row[j] = T(above_buf[max_base_x + 2])
					}
				}
			}
		case p_angle > 90 && p_angle < 180:
			dx := int(DR_INTRA_DERIVATIVE[180 - p_angle])
			dy := int(DR_INTRA_DERIVATIVE[p_angle - 90])
			min_above_base := -(1 << upsample_above)
			for i in 0 ..< h {
				row := plane[(y + i) * stride + x:]
				for j in 0 ..< w {
					idx := (j << 6) - (i + 1) * dx
					base := idx >> (6 - upsample_above)
					if base >= min_above_base {
						shift := i32(((idx << upsample_above) >> 1) & 0x1F)
						v := (above_buf[base + 2] * (32 - shift) + above_buf[base + 3] * shift + 16) >> 5
						row[j] = T(v)
					} else {
						idx_y := (i << 6) - (j + 1) * dy
						base_y := idx_y >> (6 - upsample_left)
						shift_y := i32(((idx_y << upsample_left) >> 1) & 0x1F)
						v := (left_buf[base_y + 2] * (32 - shift_y) + left_buf[base_y + 3] * shift_y + 16) >> 5
						row[j] = T(v)
					}
				}
			}
		case: // p_angle > 180
			dy := int(DR_INTRA_DERIVATIVE[270 - p_angle])
			for i in 0 ..< h {
				row := plane[(y + i) * stride + x:]
				for j in 0 ..< w {
					idx := (j + 1) * dy
					base := (idx >> (6 - upsample_left)) + (i << upsample_left)
					shift := i32(((idx << upsample_left) >> 1) & 0x1F)
					v := (left_buf[base + 2] * (32 - shift) + left_buf[base + 3] * shift + 16) >> 5
					row[j] = T(v)
				}
			}
		}
		return
	}

	// 7.11.2.5, 7.11.2.6, 7.11.2.2: Non-directional modes.
	switch mode {
	case DC_PRED:
		v := base_mid
		sum := i32(0)
		switch {
		case e.have_left && e.have_above:
			for k in 0 ..< h do sum += left[k]
			for k in 0 ..< w do sum += above[k]
			v = (sum + i32(wh >> 1)) / i32(wh)
		case e.have_left:
			for k in 0 ..< h do sum += left[k]
			v = clamp((sum + i32(h >> 1)) >> uint(log2_h), 0, max_val)
		case e.have_above:
			for k in 0 ..< w do sum += above[k]
			v = clamp((sum + i32(w >> 1)) >> uint(log2_w), 0, max_val)
		}
		vv: VT = T(v)
		for i in 0 ..< h {
			row := plane[(y + i) * stride + x:]
			for j := 0; j < w; j += 4 {
				intrinsics.unaligned_store((^VT)(&row[j]), vv)
			}
		}

	case SMOOTH_PRED:
		sm_x := sm_weights_for_log2(log2_w)
		sm_y := sm_weights_for_log2(log2_h)
		bot_left := left[h - 1]
		top_right := above[w - 1]
		for i in 0 ..< h {
			row := plane[(y + i) * stride + x:]
			wy := i32(sm_y[i])
			wy_inv := (256 - wy) * bot_left
			li := left[i]
			for j in 0 ..< w {
				wx := i32(sm_x[j])
				sp := wy * above[j] + wy_inv + wx * li + (256 - wx) * top_right
				row[j] = T((sp + 256) >> 9)
			}
		}

	case SMOOTH_V_PRED:
		sm_y := sm_weights_for_log2(log2_h)
		bot_left := left[h - 1]
		for i in 0 ..< h {
			row := plane[(y + i) * stride + x:]
			wy := i32(sm_y[i])
			wy_inv := (256 - wy) * bot_left
			for j in 0 ..< w {
				sp := wy * above[j] + wy_inv
				row[j] = T((sp + 128) >> 8)
			}
		}

	case SMOOTH_H_PRED:
		sm_x := sm_weights_for_log2(log2_w)
		top_right := above[w - 1]
		for i in 0 ..< h {
			row := plane[(y + i) * stride + x:]
			li := left[i]
			for j in 0 ..< w {
				wx := i32(sm_x[j])
				sp := wx * li + (256 - wx) * top_right
				row[j] = T((sp + 128) >> 8)
			}
		}

	case PAETH_PRED:
		tl := above_buf[1]
		for i in 0 ..< h {
			row := plane[(y + i) * stride + x:]
			li := left[i]
			p_top := abs(li - tl)
			for j in 0 ..< w {
				aj := above[j]
				p_left := abs(aj - tl)
				p_tl := abs(aj + li - 2 * tl)
				if p_left <= p_top && p_left <= p_tl {
					row[j] = T(li)
				} else if p_top <= p_tl {
					row[j] = T(aj)
				} else {
					row[j] = T(tl)
				}
			}
		}
	}
}

// 7.11.4: Palette prediction process for one transform block at `(start_x, start_y)`
// with block-local 4x4 offset `(x4, y4)` and `color_map` stride `map_stride`.
predict_palette :: proc "contextless" (plane: []$T, stride, start_x, start_y, x4, y4, tx_size: int, palette: []u16, color_map: []u8, map_stride: int) #no_bounds_check {
	w := int(TX_WIDTH[tx_size])
	h := int(TX_HEIGHT[tx_size])
	for i in 0 ..< h {
		row := plane[(start_y + i) * stride + start_x:]
		mrow := color_map[(y4 * 4 + i) * map_stride + x4 * 4:]
		for j in 0 ..< w {
			row[j] = T(palette[mrow[j]])
		}
	}
}

// 7.11.5: Predict chroma from luma (CfL) process on a transform block at
// `(start_x, start_y)` in `chroma_plane`, after DC intra prediction has already
// populated `chroma_plane` over the `w x h` region.
predict_chroma_from_luma :: #force_no_inline proc "contextless" (
	chroma_plane: []$T,
	chroma_stride: int,
	luma_plane: []T,
	luma_stride: int,
	start_x, start_y, tx_size, alpha: int,
	sub_x, sub_y: uint,
	max_luma_w, max_luma_h: int,
	bit_depth := 8,
) #no_bounds_check {
	if alpha == 0 do return
	log2_w := uint(TX_WIDTH_LOG2[tx_size])
	log2_h := uint(TX_HEIGHT_LOG2[tx_size])
	w := 1 << log2_w
	h := 1 << log2_h
	max_val := (i32(1) << uint(bit_depth)) - 1
	shift_q3 := uint(3 - sub_x - sub_y)

	l_q3: [32 * 32]i16 = ---
	luma_avg := i32(0)
	for i in 0 ..< h {
		luma_y := min((start_y + i) << sub_y, max_luma_h - (1 << sub_y))
		for j in 0 ..< w {
			luma_x := min((start_x + j) << sub_x, max_luma_w - (1 << sub_x))
			t := i32(0)
			for dy in 0 ..= int(sub_y) {
				row := luma_plane[(luma_y + dy) * luma_stride + luma_x:]
				for dx in 0 ..= int(sub_x) {
					t += i32(row[dx])
				}
			}
			v := t << shift_q3
			l_q3[i * w + j] = i16(v)
			luma_avg += v
		}
	}
	luma_avg = round2(i64(luma_avg), log2_w + log2_h)

	for i in 0 ..< h {
		crow := chroma_plane[(start_y + i) * chroma_stride + start_x:]
		for j in 0 ..< w {
			dc := i32(crow[j])
			scaled := round2_signed(i64(alpha) * i64(i32(l_q3[i * w + j]) - luma_avg), 6)
			crow[j] = T(clamp(dc + scaled, 0, max_val))
		}
	}
}
