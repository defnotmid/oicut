package gif

import "core:slice"
import "core:testing"

// A frame for the test writer: indices into the 4-colour global table (red,
// green, blue, white), rows top-down.
@(private = "file")
Test_Frame :: struct {
	x, y, width, height: int,
	indices:             []u8,
	transparent:         int, // -1: none
	disposal:            int, // the graphic control extension's 0..3
	interlaced:          bool,
}

@(private = "file")
PALETTE :: [4][3]u8{{255, 0, 0}, {0, 255, 0}, {0, 0, 255}, {255, 255, 255}}

// A GIF89a with a global table, a NETSCAPE loop count, and each frame behind
// a graphic control extension (delay 10 cs).
@(private = "file")
write :: proc(width, height, loop_count: int, frames: []Test_Frame) -> [dynamic]u8 {
	b: [dynamic]u8
	put16 :: proc(b: ^[dynamic]u8, v: int) {append(b, u8(v), u8(v >> 8))}
	append(&b, "GIF89a")
	put16(&b, width)
	put16(&b, height)
	append(&b, 0x81, 0, 0) // global table of 4 entries, background 0
	for c in PALETTE do append(&b, c[0], c[1], c[2])
	append(&b, 0x21, 0xFF, 11)
	append(&b, "NETSCAPE2.0")
	append(&b, 3, 1)
	put16(&b, loop_count)
	append(&b, 0)
	for f in frames {
		append(&b, 0x21, 0xF9, 4, u8(f.disposal << 2 | (f.transparent >= 0 ? 1 : 0)), 10, 0, u8(max(f.transparent, 0)), 0)
		append(&b, 0x2C)
		put16(&b, f.x)
		put16(&b, f.y)
		put16(&b, f.width)
		put16(&b, f.height)
		append(&b, f.interlaced ? 0x40 : 0)
		// The stream carries an interlaced frame's rows pass by pass.
		order: [dynamic]u8
		defer delete(order)
		if f.interlaced {
			for pass in ([4][2]int{{0, 8}, {4, 8}, {2, 4}, {1, 2}}) do for y := pass[0]; y < f.height; y += pass[1] do append(&order, ..f.indices[y * f.width:][:f.width])
		} else {
			append(&order, ..f.indices)
		}
		append(&b, 2) // minimum code size
		codes := lzw(order[:], 2)
		defer delete(codes)
		for at := 0; at < len(codes); at += 255 {
			n := min(255, len(codes) - at)
			append(&b, u8(n))
			append(&b, ..codes[at:][:n])
		}
		append(&b, 0)
	}
	append(&b, 0x3B)
	return b
}

// GIF's LZW: codes LSB first, the width growing as the table does, a clear
// code first and when the table is full.
@(private = "file")
lzw :: proc(indices: []u8, min_size: int) -> (out: [dynamic]u8) {
	clear_code := 1 << uint(min_size)
	size, next := min_size + 1, clear_code + 2
	table: map[[2]int]int
	defer delete(table)
	bits, count: uint
	emit :: proc(out: ^[dynamic]u8, bits, count: ^uint, code, size: int) {
		bits^ |= uint(code) << count^
		count^ += uint(size)
		for count^ >= 8 {
			append(out, u8(bits^))
			bits^ >>= 8
			count^ -= 8
		}
	}
	emit(&out, &bits, &count, clear_code, size)
	if len(indices) == 0 {
		emit(&out, &bits, &count, clear_code + 1, size)
		if count > 0 do append(&out, u8(bits))
		return out
	}
	w := int(indices[0])
	for v in indices[1:] {
		if code, ok := table[{w, int(v)}]; ok {
			w = code
			continue
		}
		emit(&out, &bits, &count, w, size)
		table[{w, int(v)}] = next
		if next == 1 << uint(size) && size < 12 do size += 1
		next += 1
		if next == 4096 {
			emit(&out, &bits, &count, clear_code, size)
			clear(&table)
			size, next = min_size + 1, clear_code + 2
		}
		w = int(v)
	}
	emit(&out, &bits, &count, w, size)
	emit(&out, &bits, &count, clear_code + 1, size)
	if count > 0 do append(&out, u8(bits))
	return out
}

// Pixels from a picture: R, G, B, W for the table's colours, '.' transparent.
@(private = "file")
picture :: proc(rows: string) -> (pixels: [dynamic]u8) {
	for c in transmute([]u8)rows {
		switch c {
		case 'R':
			append(&pixels, 255, 0, 0, 255)
		case 'G':
			append(&pixels, 0, 255, 0, 255)
		case 'B':
			append(&pixels, 0, 0, 255, 255)
		case 'W':
			append(&pixels, 255, 255, 255, 255)
		case '.':
			append(&pixels, 0, 0, 0, 0)
		}
	}
	return
}

// Each disposal in turn on a 4x4 screen.
@(private = "file")
disposals :: proc() -> [dynamic]u8 {
	@(static) full := [16]u8{}
	@(static) square := [4]u8{1, 3, 3, 1}
	@(static) blue := [1]u8{2}
	@(static) green := [1]u8{1}
	return write(4, 4, 2, {
		{0, 0, 4, 4, full[:], -1, 1, false}, // red everywhere, kept
		{1, 1, 2, 2, square[:], 3, 2, false}, // a green diagonal (white transparent), then cleared
		{0, 0, 1, 1, blue[:], -1, 3, false}, // blue, then restored
		{3, 3, 1, 1, green[:], -1, 0, false},
	})
}

@(test)
animation_disposes_each_way :: proc(t: ^testing.T) {
	data := disposals()
	defer delete(data)
	ft, ferr := get_features(data[:])
	testing.expect_value(t, ferr, Error.None)
	testing.expect_value(t, ft, Features{4, 4, true, true, 4, 2})
	a: Animation
	testing.expect_value(t, init_animation(&a, data[:]), Error.None)
	defer destroy_animation(&a)
	expected := [?]string {
		"RRRR RRRR RRRR RRRR",
		"RRRR RGRR RRGR RRRR",
		"BRRR R..R R..R RRRR",
		"RRRR R..R R..R RRRG",
		"RRRR RRRR RRRR RRRR", // it starts over
	}
	for want, k in expected {
		frame, err := next_frame(&a)
		testing.expect_value(t, err, Error.None)
		pixels := picture(want)
		defer delete(pixels)
		testing.expectf(t, slice.equal(frame.pixels, pixels[:]), "frame %d: %v", k, frame.pixels)
		testing.expect_value(t, frame.start, k % 4 * 100)
		testing.expect_value(t, frame.duration, 100)
	}
}

@(test)
interlaced_rows_and_a_growing_code_table :: proc(t: ^testing.T) {
	// 61x37 of noise: some six hundred codes, so they grow to 10 bits.
	w, h :: 61, 37
	indices: [w * h]u8
	x: u32 = 1
	for &v in indices {
		x ~= x << 13; x ~= x >> 17; x ~= x << 5
		v = u8(x >> 30)
	}
	plain := write(w, h, 0, {{0, 0, w, h, indices[:], -1, 0, false}})
	defer delete(plain)
	laced := write(w, h, 0, {{0, 0, w, h, indices[:], -1, 0, true}})
	defer delete(laced)
	palette := PALETTE
	for data in ([2][]u8{plain[:], laced[:]}) {
		img, err := decode(data)
		defer destroy_image(&img)
		testing.expect_value(t, err, Error.None)
		ok := len(img.pixels) == w * h * 4
		for v, i in indices do if ok && !slice.equal(img.pixels[i * 4:][:3], palette[v][:]) do ok = false
		testing.expectf(t, ok, "interlaced %v", data[len(data) - 1] != 0x3B)
	}
}

// Every truncation fails cleanly, through decode and the animation.
@(test)
truncated_input :: proc(t: ^testing.T) {
	data := disposals()
	defer delete(data)
	for n in 0 ..< len(data) - 1 {
		cut := data[:n]
		if img, err := decode(cut); err == .None do destroy_image(&img)
		a: Animation
		if init_animation(&a, cut) != .None do continue
		for _ in 0 ..< a.frame_count do if _, err := next_frame(&a); err != .None do break
		destroy_animation(&a)
	}
}

// A frame with no pixels draws nothing and stops nothing: stb and browsers
// keep the animation.
@(test)
zero_size_frames_are_kept :: proc(t: ^testing.T) {
	green, blue: [16]u8
	for i in 0 ..< 16 do green[i], blue[i] = 1, 2
	for first_empty in ([2]bool{false, true}) {
		empty := Test_Frame{transparent = -1}
		full := Test_Frame{width = 4, height = 4, indices = blue[:], transparent = -1}
		frames := first_empty ? [3]Test_Frame{empty, full, full} : [3]Test_Frame{{width = 4, height = 4, indices = green[:], transparent = -1}, empty, full}
		data := write(4, 4, 0, frames[:])
		defer delete(data)
		img, err := decode(data[:])
		testing.expect_value(t, err, Error.None)
		if err != .None do continue
		first := picture(first_empty ? "................" : "GGGGGGGGGGGGGGGG")
		defer delete(first)
		testing.expect(t, slice.equal(img.pixels, first[:]))
		destroy_image(&img)
		a: Animation
		testing.expect_value(t, init_animation(&a, data[:]), Error.None)
		defer destroy_animation(&a)
		testing.expect_value(t, a.frame_count, 3)
		expected := first_empty ? [3]string{"................", "BBBBBBBBBBBBBBBB", "BBBBBBBBBBBBBBBB"} : [3]string{"GGGGGGGGGGGGGGGG", "GGGGGGGGGGGGGGGG", "BBBBBBBBBBBBBBBB"}
		for want, k in expected {
			frame, ferr := next_frame(&a)
			testing.expect_value(t, ferr, Error.None)
			pixels := picture(want)
			defer delete(pixels)
			testing.expectf(t, slice.equal(frame.pixels, pixels[:]), "frame %d: %v", k, frame.pixels)
		}
	}
}

// A frame whose data ends early draws what it holds and leaves the rest of
// its rectangle alone, not colour 0: as stb, in stream order when interlaced.
@(test)
short_frames_leave_the_rest :: proc(t: ^testing.T) {
	green, white: [32]u8
	for i in 0 ..< 32 do green[i], white[i] = 1, 3
	{
		frames := [2]Test_Frame {
			{width = 4, height = 4, indices = green[:16], transparent = -1},
			{width = 4, height = 4, indices = white[:6], transparent = -1},
		}
		data := write(4, 4, 0, frames[:])
		defer delete(data)
		first := picture("GGGGGGGGGGGGGGGG")
		defer delete(first)
		img, err := decode(data[:])
		testing.expect_value(t, err, Error.None)
		testing.expect(t, slice.equal(img.pixels, first[:]))
		destroy_image(&img)
		a: Animation
		testing.expect_value(t, init_animation(&a, data[:]), Error.None)
		defer destroy_animation(&a)
		next_frame(&a)
		frame, ferr := next_frame(&a)
		testing.expect_value(t, ferr, Error.None)
		want := picture("WWWW WWGG GGGG GGGG")
		defer delete(want)
		testing.expectf(t, slice.equal(frame.pixels, want[:]), "%v", frame.pixels)
	}
	{ // The first frame alone: the canvas stays transparent.
		frames := [1]Test_Frame{{width = 4, height = 4, indices = white[:6], transparent = -1}}
		data := write(4, 4, 0, frames[:])
		defer delete(data)
		img, err := decode(data[:])
		testing.expect_value(t, err, Error.None)
		want := picture("WWWW WW.. .... ....")
		defer delete(want)
		testing.expectf(t, slice.equal(img.pixels, want[:]), "%v", img.pixels)
		destroy_image(&img)
	}
	{ // Interlaced: the passes reach rows 0, 4, 2, 6, and the data ends in row 6.
		colours := [?]u8{255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255}
		f := Frame_Record {
			width       = 2,
			height      = 8,
			interlaced  = true,
			palette     = colours[:],
			transparent = -1,
		}
		canvas := picture("GGGGGGGGGGGGGGGG")
		defer delete(canvas)
		draw(canvas[:], 2, &f, white[:7])
		want := picture("WW GG WW GG WW GG WG GG")
		defer delete(want)
		testing.expectf(t, slice.equal(canvas[:], want[:]), "%v", canvas[:])
	}
}

// A first frame that ends early leaves the rest of the canvas in the background
// colour, inside its rectangle too: as stb.
@(test)
short_first_frames_show_the_background :: proc(t: ^testing.T) {
	white: [6]u8
	for i in 0 ..< 6 do white[i] = 3
	frames := [1]Test_Frame{{width = 4, height = 4, indices = white[:], transparent = -1}}
	data := write(4, 4, 0, frames[:])
	defer delete(data)
	data[11] = 1 // the logical screen's background index: green
	want := picture("WWWW WWGG GGGG GGGG")
	defer delete(want)
	img, err := decode(data[:])
	testing.expect_value(t, err, Error.None)
	testing.expect(t, slice.equal(img.pixels, want[:]))
	destroy_image(&img)
}
