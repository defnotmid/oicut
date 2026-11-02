package av1

// Normative Film Grain Synthesis (section 7.18.3) shared by the AV1 decoder and encoder.
// Operates stripe-by-stripe (32 luma rows per stripe) in place on the output planes
// without heap allocations.

// Section 7.18.3.2: Random number process (16-bit LFSR).
grain_random_number :: #force_inline proc "contextless" (reg: ^u16, bits: uint) -> int {
	r := uint(reg^)
	bit := ((r >> 0) ~ (r >> 1) ~ (r >> 3) ~ (r >> 12)) & 1
	r = (r >> 1) | (bit << 15)
	result := (r >> (16 - bits)) & ((1 << bits) - 1)
	reg^ = u16(r)
	return int(result)
}

// Section 7.18.3.3: Generate grain process.
generate_grain_arrays :: proc "contextless" (
	luma_grain: ^[73][82]i32,
	cb_grain: ^[73][82]i32,
	cr_grain: ^[73][82]i32,
	fg: ^Film_Grain,
	mono_chrome: bool,
	sub_x, sub_y: uint,
	bit_depth := 8,
) #no_bounds_check {
	grain_center := i32(128) << uint(bit_depth - 8)
	grain_min := -grain_center
	grain_max := (i32(256) << uint(bit_depth - 8)) - 1 - grain_center

	shift_init := uint(12 - bit_depth + int(fg.grain_scale_shift))
	reg := fg.grain_seed
	for y in 0 ..< 73 {
		for x in 0 ..< 82 {
			g := i32(0)
			if fg.num_y_points > 0 {
				g = i32(GAUSSIAN_SEQUENCE[grain_random_number(&reg, 11)])
			}
			luma_grain[y][x] = round2(i64(g), shift_init)
		}
	}

	ar_shift := uint(fg.ar_coeff_shift_minus_6) + 6
	ar_lag := int(fg.ar_coeff_lag)
	if fg.num_y_points > 0 {
		for y in 3 ..< 73 {
			for x in 3 ..< 82 - 3 {
				s := i64(0)
				pos := 0
				for dr := -ar_lag; dr <= 0; dr += 1 {
					for dc := -ar_lag; dc <= ar_lag; dc += 1 {
						if dr == 0 && dc == 0 do break
						c := i64(int(fg.ar_coeffs_y_plus_128[pos]) - 128)
						s += i64(luma_grain[y + dr][x + dc]) * c
						pos += 1
					}
				}
				luma_grain[y][x] = clamp(luma_grain[y][x] + round2(s, ar_shift), grain_min, grain_max)
			}
		}
	}

	if mono_chrome do return

	chroma_w := sub_x != 0 ? 44 : 82
	chroma_h := sub_y != 0 ? 38 : 73

	reg = fg.grain_seed ~ 0xb524
	for y in 0 ..< chroma_h {
		for x in 0 ..< chroma_w {
			g := i32(0)
			if fg.num_cb_points > 0 || fg.chroma_scaling_from_luma {
				g = i32(GAUSSIAN_SEQUENCE[grain_random_number(&reg, 11)])
			}
			cb_grain[y][x] = round2(i64(g), shift_init)
		}
	}

	reg = fg.grain_seed ~ 0x49d8
	for y in 0 ..< chroma_h {
		for x in 0 ..< chroma_w {
			g := i32(0)
			if fg.num_cr_points > 0 || fg.chroma_scaling_from_luma {
				g = i32(GAUSSIAN_SEQUENCE[grain_random_number(&reg, 11)])
			}
			cr_grain[y][x] = round2(i64(g), shift_init)
		}
	}

	for y in 3 ..< chroma_h {
		for x in 3 ..< chroma_w - 3 {
			s0 := i64(0)
			s1 := i64(0)
			pos := 0
			for dr := -ar_lag; dr <= 0; dr += 1 {
				for dc := -ar_lag; dc <= ar_lag; dc += 1 {
					c0 := i64(int(fg.ar_coeffs_cb_plus_128[pos]) - 128)
					c1 := i64(int(fg.ar_coeffs_cr_plus_128[pos]) - 128)
					if dr == 0 && dc == 0 {
						if fg.num_y_points > 0 {
							luma := i64(0)
							luma_x := ((x - 3) << sub_x) + 3
							luma_y := ((y - 3) << sub_y) + 3
							for i in 0 ..= int(sub_y) {
								for j in 0 ..= int(sub_x) {
									luma += i64(luma_grain[luma_y + i][luma_x + j])
								}
							}
							luma = i64(round2(luma, sub_x + sub_y))
							s0 += luma * c0
							s1 += luma * c1
						}
						break
					}
					s0 += i64(cb_grain[y + dr][x + dc]) * c0
					s1 += i64(cr_grain[y + dr][x + dc]) * c1
					pos += 1
				}
			}
			cb_grain[y][x] = clamp(cb_grain[y][x] + round2(s0, ar_shift), grain_min, grain_max)
			cr_grain[y][x] = clamp(cr_grain[y][x] + round2(s1, ar_shift), grain_min, grain_max)
		}
	}
}

// Section 7.18.3.4: Scaling lookup initialization process.
init_grain_scaling_lut :: proc "contextless" (
	scaling_lut: ^[3][256]i32,
	fg: ^Film_Grain,
	num_planes: int,
) #no_bounds_check {
	get_xy :: #force_inline proc "contextless" (fg: ^Film_Grain, plane, idx: int) -> (int, int) {
		if plane == 0 || fg.chroma_scaling_from_luma {
			return int(fg.point_y_value[idx]), int(fg.point_y_scaling[idx])
		} else if plane == 1 {
			return int(fg.point_cb_value[idx]), int(fg.point_cb_scaling[idx])
		}
		return int(fg.point_cr_value[idx]), int(fg.point_cr_scaling[idx])
	}

	for plane in 0 ..< num_planes {
		num_points: int
		if plane == 0 || fg.chroma_scaling_from_luma {
			num_points = int(fg.num_y_points)
		} else if plane == 1 {
			num_points = int(fg.num_cb_points)
		} else {
			num_points = int(fg.num_cr_points)
		}

		if num_points == 0 {
			for x in 0 ..< 256 do scaling_lut[plane][x] = 0
		} else {
			x0, y0 := get_xy(fg, plane, 0)
			for x in 0 ..< x0 {
				scaling_lut[plane][x] = i32(y0)
			}
			for i in 0 ..< num_points - 1 {
				xi, yi := get_xy(fg, plane, i)
				x_next, y_next := get_xy(fg, plane, i + 1)
				delta_y := y_next - yi
				delta_x := x_next - xi
				if delta_x > 0 {
					delta := delta_y * ((65536 + (delta_x >> 1)) / delta_x)
					for x in 0 ..< delta_x {
						v := yi + ((x * delta + 32768) >> 16)
						scaling_lut[plane][xi + x] = i32(v)
					}
				}
			}
			x_last, y_last := get_xy(fg, plane, num_points - 1)
			for x in x_last ..< 256 {
				scaling_lut[plane][x] = i32(y_last)
			}
		}
	}
}

// Section 7.18.3.5: Piecewise linear lookup into ScalingLut.
@(private)
grain_scale_lut :: #force_inline proc "contextless" (
	scaling_lut: ^[3][256]i32,
	plane, index, bit_depth: int,
) -> i32 #no_bounds_check {
	shift := uint(bit_depth - 8)
	x := clamp(index >> shift, 0, 255)
	rem := index - (x << shift)
	if bit_depth == 8 || x == 255 {
		return scaling_lut[plane][x]
	}
	start := scaling_lut[plane][x]
	end := scaling_lut[plane][x + 1]
	return start + round2(i64(end - start) * i64(rem), shift)
}

// Section 7.18.3: Applies film grain synthesis in place on `planes` (`OutY`, `OutU`, `OutV`).
apply_film_grain :: proc(
	planes: [3][]$T,
	strides: [3]int,
	w, h: int,
	fg: ^Film_Grain,
	mono_chrome: bool,
	sub_x, sub_y: uint,
	matrix_coefficients: u8,
	bit_depth := 8,
) #no_bounds_check {
	if !fg.apply_grain do return
	num_planes := mono_chrome ? 1 : 3

	luma_grain: [73][82]i32 = ---
	cb_grain: [73][82]i32 = ---
	cr_grain: [73][82]i32 = ---
	generate_grain_arrays(&luma_grain, &cb_grain, &cr_grain, fg, mono_chrome, sub_x, sub_y, bit_depth)

	scaling_lut: [3][256]i32 = ---
	init_grain_scaling_lut(&scaling_lut, fg, num_planes)

	bd_shift := uint(bit_depth - 8)
	grain_center := i32(128) << bd_shift
	grain_min := -grain_center
	grain_max := (i32(256) << bd_shift) - 1 - grain_center
	max_sample := (i32(1) << uint(bit_depth)) - 1

	min_value: i32 = 0
	max_luma: i32 = max_sample
	max_chroma: i32 = max_sample
	if fg.clip_to_restricted_range {
		min_value = i32(16) << bd_shift
		max_luma = i32(235) << bd_shift
		max_chroma = (matrix_coefficients == MC_IDENTITY) ? max_luma : (i32(240) << bd_shift)
	}
	scaling_shift := uint(fg.grain_scaling_minus_8) + 8

	// Allocate two 34-row stripe buffers (current and previous stripe bottom) to support arbitrary frame widths.
	stripe_stride := w + 68
	stripe_buf := make([]i16, 3 * 34 * stripe_stride + 3 * 2 * stripe_stride)
	defer delete(stripe_buf)

	cur_stripe: [3][]i16
	prev_bottom: [3][]i16
	for p in 0 ..< 3 {
		cur_stripe[p] = stripe_buf[p * 34 * stripe_stride:][:34 * stripe_stride]
		prev_bottom[p] = stripe_buf[3 * 34 * stripe_stride + p * 2 * stripe_stride:][:2 * stripe_stride]
	}

	luma_num := 0
	half_h := (h + 1) / 2
	half_w := (w + 1) / 2
	chroma_w := (w + int(sub_x)) >> sub_x
	chroma_h := (h + int(sub_y)) >> sub_y

	for y_half := 0; y_half < half_h; y_half += 16 {
		reg := fg.grain_seed
		reg ~= u16(((luma_num * 37 + 178) & 255) << 8)
		reg ~= u16((luma_num * 173 + 105) & 255)

		for x_half := 0; x_half < half_w; x_half += 16 {
			rand := grain_random_number(&reg, 8)
			offset_x := rand >> 4
			offset_y := rand & 15

			for plane in 0 ..< num_planes {
				psx := plane > 0 ? sub_x : 0
				psy := plane > 0 ? sub_y : 0
				p_off_x := psx != 0 ? (6 + offset_x) : (9 + offset_x * 2)
				p_off_y := psy != 0 ? (6 + offset_y) : (9 + offset_y * 2)
				rows_s := 34 >> psy
				cols_s := 34 >> psx

				for i in 0 ..< rows_s {
					s_row := cur_stripe[plane][i * stripe_stride:]
					for j in 0 ..< cols_s {
						g: i32
						if plane == 0 {
							g = luma_grain[p_off_y + i][p_off_x + j]
						} else if plane == 1 {
							g = cb_grain[p_off_y + i][p_off_x + j]
						} else {
							g = cr_grain[p_off_y + i][p_off_x + j]
						}

						if psx == 0 {
							dst_x := x_half * 2 + j
							if j < 2 && fg.overlap_flag && x_half > 0 {
								old := i32(s_row[dst_x])
								g = j == 0 ? (old * 27 + g * 17) : (old * 17 + g * 27)
								g = clamp(round2(i64(g), 5), grain_min, grain_max)
							}
							s_row[dst_x] = i16(g)
						} else {
							dst_x := x_half + j
							if j == 0 && fg.overlap_flag && x_half > 0 {
								old := i32(s_row[dst_x])
								g = old * 23 + g * 22
								g = clamp(round2(i64(g), 5), grain_min, grain_max)
							}
							s_row[dst_x] = i16(g)
						}
					}
				}
			}
		}

		// Apply vertical overlap to top rows of cur_stripe (using prev_bottom from luma_num - 1),
		// then save bottom rows (32..33 or 16) into prev_bottom for luma_num + 1.
		for plane in 0 ..< num_planes {
			psx := plane > 0 ? sub_x : 0
			psy := plane > 0 ? sub_y : 0
			pw := (w + int(psx)) >> psx
			if luma_num > 0 && fg.overlap_flag {
				if psy == 0 {
					for i in 0 ..< 2 {
						s_row := cur_stripe[plane][i * stripe_stride:]
						p_row := prev_bottom[plane][i * stripe_stride:]
						w_old := i == 0 ? i32(27) : i32(17)
						w_new := i == 0 ? i32(17) : i32(27)
						for x in 0 ..< pw {
							g := i32(p_row[x]) * w_old + i32(s_row[x]) * w_new
							s_row[x] = i16(clamp(round2(i64(g), 5), grain_min, grain_max))
						}
					}
				} else {
					s_row := cur_stripe[plane][0:]
					p_row := prev_bottom[plane][0:]
					for x in 0 ..< pw {
						g := i32(p_row[x]) * 23 + i32(s_row[x]) * 22
						s_row[x] = i16(clamp(round2(i64(g), 5), grain_min, grain_max))
					}
				}
			}
			// Save rows 32..33 (or row 16 when psy == 1) for the next stripe.
			if psy == 0 {
				copy(prev_bottom[plane][0 * stripe_stride:][:pw], cur_stripe[plane][32 * stripe_stride:][:pw])
				copy(prev_bottom[plane][1 * stripe_stride:][:pw], cur_stripe[plane][33 * stripe_stride:][:pw])
			} else {
				copy(prev_bottom[plane][0 * stripe_stride:][:pw], cur_stripe[plane][16 * stripe_stride:][:pw])
			}
		}

		// Blend chroma noise for this stripe first (reads unmodified luma in this stripe).
		if !mono_chrome {
			cy_start := luma_num << (5 - sub_y)
			cy_end := min(chroma_h, (luma_num + 1) << (5 - sub_y))
			cb_luma_m := i32(fg.cb_luma_mult) - 128
			cb_m := i32(fg.cb_mult) - 128
			cb_off := (i32(fg.cb_offset) - 256) << bd_shift
			cr_luma_m := i32(fg.cr_luma_mult) - 128
			cr_m := i32(fg.cr_mult) - 128
			cr_off := (i32(fg.cr_offset) - 256) << bd_shift

			for cy in cy_start ..< cy_end {
				i := cy - cy_start
				luma_y := cy << sub_y
				y_row := planes[0][luma_y * strides[0]:]
				u_row := planes[1][cy * strides[1]:]
				v_row := planes[2][cy * strides[2]:]
				ns_u := cur_stripe[1][i * stripe_stride:]
				ns_v := cur_stripe[2][i * stripe_stride:]

				for cx in 0 ..< chroma_w {
					luma_x := cx << sub_x
					luma_next_x := min(luma_x + 1, w - 1)
					avg_luma: i32
					if sub_x != 0 {
						avg_luma = (i32(y_row[luma_x]) + i32(y_row[luma_next_x]) + 1) >> 1
					} else {
						avg_luma = i32(y_row[luma_x])
					}

					if fg.num_cb_points > 0 || fg.chroma_scaling_from_luma {
						orig := i32(u_row[cx])
						merged: i32
						if fg.chroma_scaling_from_luma {
							merged = avg_luma
						} else {
							combined := avg_luma * cb_luma_m + orig * cb_m
							merged = clamp((combined >> 6) + cb_off, 0, max_sample)
						}
						noise := round2(i64(grain_scale_lut(&scaling_lut, 1, int(merged), bit_depth)) * i64(ns_u[cx]), scaling_shift)
						u_row[cx] = T(clamp(orig + noise, min_value, max_chroma))
					}

					if fg.num_cr_points > 0 || fg.chroma_scaling_from_luma {
						orig := i32(v_row[cx])
						merged: i32
						if fg.chroma_scaling_from_luma {
							merged = avg_luma
						} else {
							combined := avg_luma * cr_luma_m + orig * cr_m
							merged = clamp((combined >> 6) + cr_off, 0, max_sample)
						}
						noise := round2(i64(grain_scale_lut(&scaling_lut, 2, int(merged), bit_depth)) * i64(ns_v[cx]), scaling_shift)
						v_row[cx] = T(clamp(orig + noise, min_value, max_chroma))
					}
				}
			}
		}

		// Blend luma noise for this stripe.
		if fg.num_y_points > 0 {
			ly_start := luma_num << 5
			ly_end := min(h, (luma_num + 1) << 5)
			for ly in ly_start ..< ly_end {
				i := ly - ly_start
				y_row := planes[0][ly * strides[0]:]
				ns_y := cur_stripe[0][i * stripe_stride:]
				for lx in 0 ..< w {
					orig := i32(y_row[lx])
					noise := round2(i64(grain_scale_lut(&scaling_lut, 0, int(orig), bit_depth)) * i64(ns_y[lx]), scaling_shift)
					y_row[lx] = T(clamp(orig + noise, min_value, max_luma))
				}
			}
		}

		luma_num += 1
	}
}
