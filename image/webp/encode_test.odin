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
	w: Animation_Writer
	testing.expect_value(t, init_animation_writer(&w, 2, 2), Error.None)
	defer destroy_animation_writer(&w)
	_, e = finish_animation(&w)
	testing.expect_value(t, e, Error.Invalid)
	testing.expect_value(t, write_frame(&w, px[:8], 10), Error.Invalid)
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
