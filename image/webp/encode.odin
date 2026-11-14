package webp

import "../../common"
import "base:intrinsics"
import "core:os"

// Writing WebP: lossless stills and animations from RGBA8 pixels with
// straight alpha, rows top to bottom. Every pixel decodes to the bytes
// given, also the colour of fully transparent ones.
//
//	data, err := webp.encode(pixels, width, height) // the file's bytes; delete when done
//	err = webp.write_file("frame.webp", pixels, width, height)
//
//	w: webp.Animation_Writer
//	webp.init_animation_writer(&w, width, height) or_return
//	defer webp.destroy_animation_writer(&w)
//	for frame in frames do webp.write_frame(&w, frame.pixels, frame.duration) or_return
//	data := webp.finish_animation(&w) or_return
//
// `pixels` holds at least width * height * 4 bytes and sizes are 1 to 16384;
// anything else is .Invalid. With a pool, the predictors are chosen on it.

MAX_ENCODE_SIZE :: 16384

encode :: proc(pixels: []u8, width, height: int, pool: ^common.Work_Pool = nil, allocator := context.allocator) -> (data: []u8, err: Error) {
	if !encodable(pixels, width, height) do return nil, .Invalid
	e: Lossless_Encoder
	defer destroy_lossless_encoder(&e)
	init_lossless_encoder(&e, allocator) or_return
	out := &e.w.out
	if _, oom := append(out, "RIFF\x00\x00\x00\x00WEBP"); oom != nil do return nil, .Out_Of_Memory
	write_vp8l_chunk(&e, pixels, width * 4, width, height, pool) or_return
	put_u32(out[4:], u32(len(out) - 8))
	return take_output(&e)
}

write_file :: proc(path: string, pixels: []u8, width, height: int, pool: ^common.Work_Pool = nil) -> Error {
	data := encode(pixels, width, height, pool) or_return
	defer delete(data)
	return os.write_entire_file(path, data) == nil ? .None : .Invalid
}

// An animation being written: each frame is the whole canvas, and what
// changed since the frame before is what is stored.
Animation_Writer :: struct {
	encoder:       Lossless_Encoder,
	width, height: int,
	previous:      []u8, // the canvas as the last frame left it
	frames:        int,
	has_alpha:     bool,
	pool:          ^common.Work_Pool,
}

// loop_count 0 repeats forever.
init_animation_writer :: proc(w: ^Animation_Writer, width, height: int, loop_count := 0, pool: ^common.Work_Pool = nil, allocator := context.allocator) -> Error {
	if width < 1 || height < 1 || width > MAX_ENCODE_SIZE || height > MAX_ENCODE_SIZE || loop_count < 0 || loop_count > 0xffff do return .Invalid
	w^ = {width = width, height = height, pool = pool}
	init_lossless_encoder(&w.encoder, allocator) or_return
	previous, e := make([]u8, width * height * 4, allocator)
	if e != nil do return .Out_Of_Memory
	w.previous = previous
	out := &w.encoder.w.out
	// VP8X with the animation flag (alpha is set at the end), then ANIM: a
	// transparent background and the loop count.
	header := [44]u8{}
	copy(header[:], "RIFF\x00\x00\x00\x00WEBPVP8X\x0a\x00\x00\x00\x02")
	put_u24(header[24:], u32(width - 1))
	put_u24(header[27:], u32(height - 1))
	copy(header[30:], "ANIM\x06\x00\x00\x00")
	header[42], header[43] = u8(loop_count), u8(loop_count >> 8)
	if _, e2 := append(out, ..header[:]); e2 != nil do return .Out_Of_Memory
	return .None
}

destroy_animation_writer :: proc(w: ^Animation_Writer) {
	delete(w.previous, w.encoder.w.out.allocator)
	destroy_lossless_encoder(&w.encoder)
	w^ = {}
}

// Adds the canvas's next state, shown for `duration` milliseconds. Stored is
// the rectangle that differs from the frame before, overwriting it.
write_frame :: proc(w: ^Animation_Writer, pixels: []u8, duration: int) -> Error {
	if !encodable(pixels, w.width, w.height) || duration < 0 || duration > 0xffffff do return .Invalid
	x0, y0, x1, y1 := 0, 0, w.width, w.height
	if w.frames > 0 {
		x0, y0, x1, y1 = changed_rectangle(pixels, w.previous, w.width, w.height)
		x0 &~= 1 // frame offsets are stored halved
		y0 &~= 1
	}
	out := &w.encoder.w.out
	header := [24]u8{}
	copy(header[:], "ANMF")
	put_u24(header[8:], u32(x0 / 2))
	put_u24(header[11:], u32(y0 / 2))
	put_u24(header[14:], u32(x1 - x0 - 1))
	put_u24(header[17:], u32(y1 - y0 - 1))
	put_u24(header[20:], u32(duration))
	header[23] = 0x02 // do not blend, do not dispose
	start := len(out)
	if _, e := append(out, ..header[:]); e != nil do return .Out_Of_Memory
	stride := w.width * 4
	write_vp8l_chunk(&w.encoder, pixels[y0 * stride + x0 * 4:], stride, x1 - x0, y1 - y0, w.pool) or_return
	put_u32(out[start + 4:], u32(len(out) - start - 8))
	for y in y0 ..< y1 do copy(w.previous[y * stride + x0 * 4:][:(x1 - x0) * 4], pixels[y * stride + x0 * 4:])
	w.has_alpha ||= w.encoder.has_alpha
	w.frames += 1
	return .None
}

// The file's bytes, the caller's to delete. The writer is left empty: it
// needs destroying still, and can't take more frames.
finish_animation :: proc(w: ^Animation_Writer) -> (data: []u8, err: Error) {
	if w.frames == 0 do return nil, .Invalid
	out := &w.encoder.w.out
	if w.has_alpha do out[20] |= 0x10
	put_u32(out[4:], u32(len(out) - 8))
	w.frames = 0
	return take_output(&w.encoder)
}

@(private)
encodable :: proc(pixels: []u8, width, height: int) -> bool {
	return width >= 1 && height >= 1 && width <= MAX_ENCODE_SIZE && height <= MAX_ENCODE_SIZE && len(pixels) >= width * height * 4
}

// The smallest rectangle outside which two canvases are equal; one pixel
// when they are equal everywhere.
@(private)
changed_rectangle :: proc(a, b: []u8, width, height: int) -> (x0, y0, x1, y1: int) #no_bounds_check {
	differ :: #force_inline proc(a, b: []u8, at: int) -> bool #no_bounds_check {
		return intrinsics.unaligned_load((^u32)(&a[at * 4])) != intrinsics.unaligned_load((^u32)(&b[at * 4]))
	}
	x0, y0, x1, y1 = width, height, 0, 0
	for y in 0 ..< height {
		row := y * width
		first := 0
		for first < width && !differ(a, b, row + first) do first += 1
		if first == width do continue
		last := width - 1
		for !differ(a, b, row + last) do last -= 1
		x0, x1 = min(x0, first), max(x1, last + 1)
		y0, y1 = min(y0, y), y + 1
	}
	if x1 == 0 do return 0, 0, 1, 1
	return
}

// Appends a VP8L chunk, padded to an even size.
@(private)
write_vp8l_chunk :: proc(e: ^Lossless_Encoder, rgba: []u8, stride, width, height: int, pool: ^common.Work_Pool) -> Error {
	out := &e.w.out
	if _, oom := append(out, "VP8L\x00\x00\x00\x00"); oom != nil do return .Out_Of_Memory
	start := len(out)
	encode_vp8l(e, rgba, stride, width, height, pool) or_return
	size := len(out) - start
	put_u32(out[start - 4:], u32(size))
	if size & 1 == 1 {
		if _, oom := append(out, 0); oom != nil do return .Out_Of_Memory
	}
	return .None
}

// Hands the output to the caller, sized to its bytes.
@(private)
take_output :: proc(e: ^Lossless_Encoder) -> (data: []u8, err: Error) {
	allocator := e.w.out.allocator
	if _, oom := shrink(&e.w.out); oom != nil do return nil, .Out_Of_Memory
	data = e.w.out[:]
	e.w.out = {}
	e.w.out.allocator = allocator
	return data, .None
}

@(private)
put_u24 :: proc(b: []u8, v: u32) {
	b[0], b[1], b[2] = u8(v), u8(v >> 8), u8(v >> 16)
}

@(private)
put_u32 :: proc(b: []u8, v: u32) {
	b[0], b[1], b[2], b[3] = u8(v), u8(v >> 8), u8(v >> 16), u8(v >> 24)
}
