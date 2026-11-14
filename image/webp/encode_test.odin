package webp

import common "../../common"
import "core:slice"
import "core:testing"

// Pictures of every kind the encoder treats differently: noise, gradients,
// flat runs, 1 to 256 colours and more, with and without alpha.
@(private = "file")
picture :: proc(kind, width, height: int, allocator := context.allocator) -> []u8 {
	px := make([]u8, width * height * 4, allocator)
	seed := u32(kind * 7919 + width * 31 + height)
	next :: proc(seed: ^u32) -> u32 {
		seed^ = seed^ * 1664525 + 1013904223
		return seed^ >> 8
	}
	for y in 0 ..< height do for x in 0 ..< width {
		p := px[(y * width + x) * 4:][:4]
		switch kind {
		case 0: // noise in every channel
			p[0], p[1], p[2], p[3] = u8(next(&seed)), u8(next(&seed)), u8(next(&seed)), u8(next(&seed))
		case 1: // an opaque gradient with a little noise
			p[0], p[1], p[2], p[3] = u8(x * 3 + y), u8(y * 2 + int(next(&seed) & 3)), u8(x + y), 255
		case 2: // flat bands and a transparent region whose colour varies
			p[0], p[1], p[2], p[3] = u8(y / 8 * 40), 30, u8(x / 16 * 20), x > width / 2 ? 0 : 255
			if p[3] == 0 do p[1] = u8(next(&seed))
		case 3: // a gradient in alpha
			p[0], p[1], p[2], p[3] = 200, u8(x), 10, u8(x * 255 / max(width - 1, 1))
		case: // kind - 3 colours: 1, 2, 3, ...
			c := int(next(&seed)) % (kind - 3)
			if x % 7 != 0 && x > 0 do c = int(px[(y * width + x - 1) * 4 + 3]) // runs
			p[0], p[1], p[2], p[3] = u8(c * 37), u8(c * 11), u8(255 - c), u8(c)
		}
	}
	return px
}

@(test)
lossless_encode_round_trips :: proc(t: ^testing.T) {
	pool: common.Static_Pool
	common.init_static_pool(&pool, 3)
	defer common.destroy_static_pool(&pool)
	sizes := [?][2]int{{1, 1}, {1, 9}, {9, 1}, {2, 2}, {15, 17}, {16, 16}, {33, 47}, {130, 70}, {257, 3}}
	kinds := [?]int{0, 1, 2, 3, 4, 5, 6, 8, 20, 21, 200, 259, 260, 400}
	for size in sizes do for kind in kinds {
		width, height := size[0], size[1]
		px := picture(kind, width, height)
		defer delete(px)
		for threads in 0 ..< 2 {
			data, err := encode(px, width, height, threads == 1 ? &pool.work_pool : nil)
			defer delete(data)
			if !testing.expectf(t, err == .None, "encode %dx%d kind %d: %v", width, height, kind, err) do continue
			img, e := decode(data)
			defer destroy_image(&img)
			if !testing.expectf(t, e == .None, "decode %dx%d kind %d: %v", width, height, kind, e) do continue
			testing.expectf(t, img.width == width && img.height == height && slice.equal(img.pixels, px), "%dx%d kind %d differs", width, height, kind)
			ft, _ := get_features(data)
			testing.expect(t, ft.lossless && !ft.animated && ft.width == width && ft.height == height)
		}
	}
}

@(test)
encode_refuses_bad_arguments :: proc(t: ^testing.T) {
	px: [16]u8
	_, e := encode(px[:], 2, 3)
	testing.expect_value(t, e, Error.Invalid)
	_, e = encode(px[:], 0, 1)
	testing.expect_value(t, e, Error.Invalid)
	_, e = encode(px[:], MAX_ENCODE_SIZE + 1, 1)
	testing.expect_value(t, e, Error.Invalid)
	_, e = encode_lossy(px[:], 2, 3)
	testing.expect_value(t, e, Error.Invalid)
	_, e = encode_lossy(px[:], 0, 1)
	testing.expect_value(t, e, Error.Invalid)
	_, e = encode_lossy(px[:], MAX_ENCODE_SIZE, 1) // 16384 exceeds VP8's 14-bit 16383 limit
	testing.expect_value(t, e, Error.Invalid)
	_, e = encode_lossy(px[:], 2, 2, -1)
	testing.expect_value(t, e, Error.Invalid)
	_, e = encode_lossy(px[:], 2, 2, 101)
	testing.expect_value(t, e, Error.Invalid)
	w: Animation_Writer
	testing.expect_value(t, init_animation_writer(&w, 2, 2), Error.None)
	defer destroy_animation_writer(&w)
	_, e = finish_animation(&w)
	testing.expect_value(t, e, Error.Invalid)
	testing.expect_value(t, write_frame(&w, px[:8], 10), Error.Invalid)
	testing.expect_value(t, write_lossy_frame(&w, px[:8], 10), Error.Invalid)
	testing.expect_value(t, write_lossy_frame(&w, px[:], 10, -1), Error.Invalid)
	testing.expect_value(t, write_lossy_frame(&w, px[:], 10, 101), Error.Invalid)
}

@(private = "file")
visible_rgb_mse :: proc(a, b: []u8) -> f64 {
	err, count := i64(0), 0
	for i := 0; i < len(a); i += 4 {
		if a[i + 3] == 0 do continue
		for c in 0 ..< 3 {
			d := i64(a[i + c]) - i64(b[i + c])
			err += d * d
		}
		count += 3
	}
	return count > 0 ? f64(err) / f64(count) : 0
}

@(test)
lossy_encode_round_trips :: proc(t: ^testing.T) {
	pool: common.Static_Pool
	common.init_static_pool(&pool, 3)
	defer common.destroy_static_pool(&pool)
	sizes := [?][2]int{{1, 1}, {1, 9}, {9, 1}, {2, 2}, {15, 17}, {16, 16}, {33, 47}, {130, 70}, {257, 3}}
	kinds := [?]int{0, 1, 2, 3, 4, 20, 200}
	for size in sizes do for kind in kinds {
		width, height := size[0], size[1]
		px := picture(kind, width, height)
		defer delete(px)
		want_alpha := false
		for i := 3; i < len(px); i += 4 do want_alpha ||= px[i] != 255

		single, err := encode_lossy(px, width, height, 75, nil)
		defer delete(single)
		if !testing.expectf(t, err == .None, "encode_lossy %dx%d kind %d: %v", width, height, kind, err) do continue

		threaded, err2 := encode_lossy(px, width, height, 75, &pool.work_pool)
		defer delete(threaded)
		if !testing.expectf(t, err2 == .None, "encode_lossy pool %dx%d kind %d: %v", width, height, kind, err2) do continue
		testing.expectf(t, slice.equal(single, threaded), "%dx%d kind %d pool output differs", width, height, kind)

		ft, fe := get_features(single)
		testing.expect(t, fe == .None && !ft.lossless && !ft.animated && ft.width == width && ft.height == height && ft.has_alpha == want_alpha)

		img, de := decode(single)
		defer destroy_image(&img)
		if !testing.expectf(t, de == .None, "decode %dx%d kind %d: %v", width, height, kind, de) do continue
		testing.expect(t, img.width == width && img.height == height)
		alpha_exact := true
		for i := 3; i < len(px); i += 4 do if img.pixels[i] != px[i] {alpha_exact = false; break}
		testing.expectf(t, alpha_exact, "%dx%d kind %d alpha differs", width, height, kind)
	}

	// Quality curve: higher quality lowers distortion and increases bitrate.
	grad := make([]u8, 130 * 70 * 4)
	defer delete(grad)
	for y in 0 ..< 70 do for x in 0 ..< 130 {
		p := grad[(y * 130 + x) * 4:]
		p[0], p[1], p[2], p[3] = u8((x * 255) / 129), u8((y * 255) / 69), u8(((x + y) * 255) / 198), 255
	}
	d10, _ := encode_lossy(grad, 130, 70, 10)
	d50, _ := encode_lossy(grad, 130, 70, 50)
	d95, _ := encode_lossy(grad, 130, 70, 95)
	defer delete(d10)
	defer delete(d50)
	defer delete(d95)
	i10, _ := decode(d10)
	i50, _ := decode(d50)
	i95, _ := decode(d95)
	defer destroy_image(&i10)
	defer destroy_image(&i50)
	defer destroy_image(&i95)
	mse10 := visible_rgb_mse(grad, i10.pixels)
	mse50 := visible_rgb_mse(grad, i50.pixels)
	mse95 := visible_rgb_mse(grad, i95.pixels)
	testing.expectf(t, mse95 < mse50 && mse50 < mse10 && mse95 < 4.0, "unexpected MSE curve: %.2f %.2f %.2f", mse10, mse50, mse95)
	testing.expectf(t, len(d10) < len(d95), "expected q10 (%d B) < q95 (%d B)", len(d10), len(d95))
}

@(test)
lossy_animation_round_trips :: proc(t: ^testing.T) {
	width, height :: 37, 29
	frames: [5][]u8
	frames[0] = picture(1, width, height)
	frames[1] = picture(2, width, height)
	frames[2] = slice.clone(frames[1])
	for y in 5 ..< 15 do for x in 9 ..< 23 do frames[2][(y * width + x) * 4] = 220
	frames[3] = slice.clone(frames[2]) // unchanged canvas
	frames[4] = picture(3, width, height)
	defer for f in frames do delete(f)

	w: Animation_Writer
	testing.expect_value(t, init_animation_writer(&w, width, height, 2), Error.None)
	defer destroy_animation_writer(&w)
	// Mix lossy and lossless frames in one animation.
	for f, i in frames {
		err := i == 2 ? write_frame(&w, f, 30 + i * 10) : write_lossy_frame(&w, f, 30 + i * 10, 85)
		testing.expect_value(t, err, Error.None)
	}
	data, err := finish_animation(&w)
	defer delete(data)
	testing.expect_value(t, err, Error.None)

	a: Animation
	if !testing.expect_value(t, init_animation(&a, data), Error.None) do return
	defer destroy_animation(&a)
	testing.expect(t, a.features.animated && a.features.has_alpha && a.features.frame_count == len(frames) && a.features.loop_count == 2)
	start := 0
	for f, i in frames {
		frame, e := next_frame(&a)
		if !testing.expectf(t, e == .None, "frame %d: %v", i, e) do return
		testing.expect(t, frame.start == start && frame.duration == 30 + i * 10)
		start += frame.duration
		alpha_ok := true
		for k := 3; k < len(f); k += 4 do if frame.pixels[k] != f[k] {alpha_ok = false; break}
		testing.expectf(t, alpha_ok, "frame %d alpha differs", i)
		if i == 2 {
			testing.expect(t, slice.equal(frame.pixels, f))
		} else {
			testing.expectf(t, visible_rgb_mse(f, frame.pixels) < 35.0, "frame %d RGB MSE too high", i)
		}
	}
}

// Frames that change everywhere, in one odd-placed rectangle, in a single
// pixel and not at all come back as the canvases given, with their times.
@(test)
animation_round_trips :: proc(t: ^testing.T) {
	width, height :: 37, 29
	frames: [6][]u8
	frames[0] = picture(1, width, height)
	frames[1] = picture(2, width, height)
	frames[2] = slice.clone(frames[1])
	for y in 5 ..< 12 do for x in 9 ..< 20 do frames[2][(y * width + x) * 4 + 1] ~= 0x55
	frames[3] = slice.clone(frames[2])
	frames[4] = slice.clone(frames[3])
	frames[4][(28 * width + 36) * 4 + 3] ~= 0x80
	frames[5] = picture(0, width, height)
	defer for f in frames do delete(f)
	w: Animation_Writer
	testing.expect_value(t, init_animation_writer(&w, width, height, 3), Error.None)
	defer destroy_animation_writer(&w)
	for f, i in frames do testing.expect_value(t, write_frame(&w, f, 40 + i * 10), Error.None)
	data, err := finish_animation(&w)
	defer delete(data)
	testing.expect_value(t, err, Error.None)
	a: Animation
	if !testing.expect_value(t, init_animation(&a, data), Error.None) do return
	defer destroy_animation(&a)
	testing.expect(t, a.features.animated && a.features.has_alpha && a.features.frame_count == len(frames) && a.features.loop_count == 3)
	testing.expect(t, a.features.width == width && a.features.height == height)
	start := 0
	for f, i in frames {
		frame, e := next_frame(&a)
		if !testing.expectf(t, e == .None, "frame %d: %v", i, e) do return
		testing.expectf(t, slice.equal(frame.pixels, f), "frame %d differs", i)
		testing.expect(t, frame.start == start && frame.duration == 40 + i * 10)
		start += frame.duration
	}
}

// ICC, EXIF and XMP chunks around the image come back as stored, odd sizes
// with their padding skipped; a file without them has none.
@(test)
metadata_is_found :: proc(t: ^testing.T) {
	px := [4]u8{1, 2, 3, 255}
	still, _ := encode(px[:], 1, 1)
	defer delete(still)
	m, e := get_metadata(still)
	testing.expect(t, e == .None && len(m.icc) == 0 && len(m.exif) == 0 && len(m.xmp) == 0)
	file: [dynamic]u8
	defer delete(file)
	append(&file, "RIFF\x00\x00\x00\x00WEBPVP8X\x0a\x00\x00\x00\x2c\x00\x00\x00\x00\x00\x00\x00\x00\x00")
	append(&file, "ICCP\x03\x00\x00\x00abc\x00")
	append(&file, ..still[12:])
	append(&file, "EXIF\x04\x00\x00\x00Exif")
	append(&file, "XMP \x05\x00\x00\x00<x/> \x00")
	put_u32(file[4:], u32(len(file) - 8))
	m, e = get_metadata(file[:])
	testing.expect_value(t, e, Error.None)
	testing.expect_value(t, string(m.icc), "abc")
	testing.expect_value(t, string(m.exif), "Exif")
	testing.expect_value(t, string(m.xmp), "<x/> ")
	img, de := decode(file[:])
	defer destroy_image(&img)
	testing.expect(t, de == .None && slice.equal(img.pixels, px[:]))
	_, e = get_metadata(file[:len(file) - 3])
	testing.expect_value(t, e, Error.Invalid)
}

// Every VP8 encoder SIMD kernel (predictors, forward/inverse DCT & WHT,
// Hadamard spectral weight, and branchless quantizer) matches the scalar
// decoder/reference definitions bit-for-bit.
@(test)
vp8_simd_kernels_match_scalar :: proc(t: ^testing.T) {
	ws, ws2: [WORKSPACE]u8
	state := u64(0x9e3779b97f4a7c15)
	rnd :: proc(s: ^u64) -> u8 {
		s^ = s^ * 6364136223846793005 + 1442695040888963407
		return u8(s^ >> 56)
	}
	for &b in ws do b = rnd(&state)

	// 1. 4x4 predictions
	for n in 0 ..< 16 {
		at := Y_OFF + (n & 3) * 4 + (n >> 2) * 4 * BPS
		preds := make_preds4(&ws, at)
		for m in 0 ..< 10 {
			ws2 = ws
			predict4(&ws2, at, m)
			ref := load_plane4(ws2[:], at, BPS)
			if !testing.expectf(t, preds[m] == ref, "pred4 n=%d m=%d got=%v ref=%v", n, m, preds[m], ref) do return
		}
	}

	// 2. 16x16 and UV predictions
	for mb_y in 0 ..< 2 do for mb_x in 0 ..< 2 {
		p16: [4][16]Block4
		make_preds16(&ws, mb_x, mb_y, &p16)
		for m in 0 ..< 4 {
			ws2 = ws
			predict(&ws2, Y_OFF, 16, edge_mode(m, mb_x, mb_y))
			for n in 0 ..< 16 {
				at := Y_OFF + (n & 3) * 4 + (n >> 2) * 4 * BPS
				ref := load_plane4(ws2[:], at, BPS)
				if !testing.expectf(t, p16[m][n] == ref, "pred16 m=%d n=%d", m, n) do return
			}
		}
		puv: [4][8]Block4
		make_preds_uv(&ws, mb_x, mb_y, &puv)
		for m in 0 ..< 4 {
			ws2 = ws
			em := edge_mode(m, mb_x, mb_y)
			predict(&ws2, U_OFF, 8, em)
			predict(&ws2, V_OFF, 8, em)
			for c in 0 ..< 2 do for n in 0 ..< 4 {
				base := c == 0 ? U_OFF : V_OFF
				at := base + (n & 1) * 4 + (n >> 1) * 4 * BPS
				ref := load_plane4(ws2[:], at, BPS)
				if !testing.expectf(t, puv[m][c * 4 + n] == ref, "preduv m=%d c=%d n=%d", m, c, n) do return
			}
		}
	}

	// 3. ftransform4, idct_add4, fwht4, iwht4, ttransform4
	weight_y := [16]i32{38, 32, 20, 9, 32, 28, 17, 7, 20, 17, 10, 4, 9, 7, 4, 2}
	for iter in 0 ..< 64 {
		for &b in ws do b = rnd(&state)
		for &b in ws2 do b = rnd(&state)
		s4 := load_plane4(ws[:], Y_OFF, BPS)
		p4 := load_plane4(ws2[:], Y_OFF, BPS)

		// scalar ftransform
		tmp: [16]i32
		ref_dct: [16]i16
		for i in 0 ..< 4 {
			s := ws[Y_OFF + i * BPS:]
			r := ws2[Y_OFF + i * BPS:]
			d0, d1, d2, d3 := i32(s[0]) - i32(r[0]), i32(s[1]) - i32(r[1]), i32(s[2]) - i32(r[2]), i32(s[3]) - i32(r[3])
			a0, a1, a2, a3 := d0 + d3, d1 + d2, d1 - d2, d0 - d3
			tmp[i * 4 + 0] = (a0 + a1) * 8
			tmp[i * 4 + 1] = (a2 * 2217 + a3 * 5352 + 1812) >> 9
			tmp[i * 4 + 2] = (a0 - a1) * 8
			tmp[i * 4 + 3] = (a3 * 2217 - a2 * 5352 + 937) >> 9
		}
		for i in 0 ..< 4 {
			a0, a1, a2, a3 := tmp[0 + i] + tmp[12 + i], tmp[4 + i] + tmp[8 + i], tmp[4 + i] - tmp[8 + i], tmp[0 + i] - tmp[12 + i]
			ref_dct[0 + i] = i16((a0 + a1 + 7) >> 4)
			ref_dct[4 + i] = i16(((a2 * 2217 + a3 * 5352 + 12000) >> 16) + i32(a3 != 0))
			ref_dct[8 + i] = i16((a0 - a1 + 7) >> 4)
			ref_dct[12 + i] = i16((a3 * 2217 - a2 * 5352 + 51000) >> 16)
		}
		got_dct := transmute([16]i16)ftransform4(s4, p4)
		if !testing.expectf(t, got_dct == ref_dct, "ftransform4 iter=%d got=%v ref=%v", iter, got_dct, ref_dct) do return

		// idct_add4 vs add_residual
		for kind in ([3]u8{0, 1, 2}) {
			ws_tmp := ws2
			add_residual(&ws_tmp, Y_OFF, ref_dct[:], kind)
			ref_rec := load_plane4(ws_tmp[:], Y_OFF, BPS)
			got_rec := idct_add4(p4, transmute(Coeff4)ref_dct, kind)
			if !testing.expectf(t, got_rec == ref_rec, "idct_add4 kind=%d got=%v ref=%v", kind, got_rec, ref_rec) do return
		}

		// iwht4 vs inverse_wht
		wht_out_full: [256]i16
		inverse_wht(&ref_dct, wht_out_full[:])
		var_ref: [16]i16
		for k in 0 ..< 16 do var_ref[k] = wht_out_full[k * 16]
		got_iwht := iwht4(transmute(Coeff4)ref_dct)
		if !testing.expectf(t, got_iwht == var_ref, "iwht4 got=%v ref=%v", got_iwht, var_ref) do return

		// ttransform4
		for i in 0 ..< 4 {
			r := ws[Y_OFF + i * BPS:]
			a0, a1, a2, a3 := i32(r[0]) + i32(r[2]), i32(r[1]) + i32(r[3]), i32(r[1]) - i32(r[3]), i32(r[0]) - i32(r[2])
			tmp[i * 4 + 0], tmp[i * 4 + 1], tmp[i * 4 + 2], tmp[i * 4 + 3] = a0 + a1, a3 + a2, a3 - a2, a0 - a1
		}
		sum := i32(0)
		for i in 0 ..< 4 {
			a0, a1, a2, a3 := tmp[0 + i] + tmp[8 + i], tmp[4 + i] + tmp[12 + i], tmp[4 + i] - tmp[12 + i], tmp[0 + i] - tmp[8 + i]
			sum += weight_y[0 + i] * abs(a0 + a1) + weight_y[4 + i] * abs(a3 + a2) + weight_y[8 + i] * abs(a3 - a2) + weight_y[12 + i] * abs(a0 - a1)
		}
		got_tt := ttransform4(s4)
		if !testing.expectf(t, got_tt == sum, "ttransform4 got=%d ref=%d", got_tt, sum) do return

		// fwht4
		for i in 0 ..< 4 {
			b := i * 4
			a0, a1, a2, a3 := i32(ref_dct[b + 0]) + i32(ref_dct[b + 2]), i32(ref_dct[b + 1]) + i32(ref_dct[b + 3]), i32(ref_dct[b + 1]) - i32(ref_dct[b + 3]), i32(ref_dct[b + 0]) - i32(ref_dct[b + 2])
			tmp[i * 4 + 0], tmp[i * 4 + 1], tmp[i * 4 + 2], tmp[i * 4 + 3] = a0 + a1, a3 + a2, a3 - a2, a0 - a1
		}
		ref_wht: [16]i16
		for i in 0 ..< 4 {
			a0, a1, a2, a3 := tmp[0 + i] + tmp[8 + i], tmp[4 + i] + tmp[12 + i], tmp[4 + i] - tmp[12 + i], tmp[0 + i] - tmp[8 + i]
			ref_wht[0 + i] = i16((a0 + a1) >> 1)
			ref_wht[4 + i] = i16((a3 + a2) >> 1)
			ref_wht[8 + i] = i16((a3 - a2) >> 1)
			ref_wht[12 + i] = i16((a0 - a1) >> 1)
		}
		got_wht := transmute([16]i16)fwht4(ref_dct)
		if !testing.expectf(t, got_wht == ref_wht, "fwht4 got=%v ref=%v", got_wht, ref_wht) do return

		// quantize4
		qm: Quant_Matrix
		_ = expand_matrix(&qm, 15, 23, iter % 3)
		deq4, lvls4, _, last4, kind4 := quantize4(transmute(Coeff4)ref_dct, &qm)
		q_arr := transmute([16]i16)qm.q
		iq_arr := transmute([16]u32)qm.iq
		bias_arr := transmute([16]u32)qm.bias
		sh_arr := transmute([16]u16)qm.sharpen
		ref_deq, ref_lvls: [16]i16
		ref_last := i8(-1)
		for n in 0 ..< 16 {
			j := int(ZIGZAG[n])
			v := i32(ref_dct[j])
			neg := v < 0
			coeff := u32(neg ? -v : v) + u32(sh_arr[j])
			zthr := ((1 << QFIX) - 1 - bias_arr[j]) / iq_arr[j]
			if coeff > zthr {
				lvl := int((coeff * iq_arr[j] + bias_arr[j]) >> QFIX)
				if lvl > MAX_LEVEL do lvl = MAX_LEVEL
				if lvl != 0 {
					sgn := i16(neg ? -lvl : lvl)
					ref_lvls[n] = sgn
					ref_deq[j] = sgn * q_arr[j]
					ref_last = i8(n)
				}
			}
		}
		ref_kind := ref_last > 0 ? u8(2) : (ref_deq[0] != 0 ? u8(1) : u8(0))
		if !testing.expectf(t, transmute([16]i16)deq4 == ref_deq && transmute([16]i16)lvls4 == ref_lvls && last4 == ref_last && kind4 == ref_kind, "quantize4 iter=%d", iter) do return
	}
}
