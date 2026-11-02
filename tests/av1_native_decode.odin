package main

// Decodes a raw AV1 OBU stream with the native decoder and writes its shown
// frames as raw planar YUV (bytes at 8 bits, little-endian u16 above, chroma
// rounded up) for tests/av1_check.sh to compare with FFmpeg's libdav1d.

import av1 "../video/av1"
import decoder "../video/av1/decoder"
import "core:fmt"
import "core:mem"
import "core:os"

main :: proc() {
	if len(os.args) < 3 {
		fmt.eprintln("usage: av1_native_decode in.obu out.yuv")
		os.exit(2)
	}
	data, rerr := os.read_entire_file(os.args[1], context.allocator)
	if rerr != nil do os.exit(1)
	out, oerr := os.create(os.args[2])
	if oerr != nil do os.exit(1)
	defer os.close(out)
	d: decoder.Decoder
	decoder.init_decoder(&d)
	defer decoder.destroy_decoder(&d)
	frames := 0
	bytes: [dynamic]u8
	for at := 0; at < len(data); {
		// A temporal unit runs to the next temporal delimiter.
		end := at
		for end < len(data) {
			o, next, err := av1.next_obu(data, end)
			if err != .None {
				fmt.eprintln("obu:", err)
				os.exit(1)
			}
			if o.type == .Temporal_Delimiter && end > at do break
			end = next
		}
		pic, shown, err := decoder.decode_packet(&d, data[at:end])
		at = end
		if err != .None {
			fmt.eprintfln("frame %d: %v", frames, err)
			os.exit(1)
		}
		if !shown do continue
		clear(&bytes)
		for p in 0 ..< (pic.mono_chrome ? 1 : 3) {
			w, h := pic.width, pic.height
			if p > 0 {
				w = (w + int(pic.subsampling_x)) >> pic.subsampling_x
				h = (h + int(pic.subsampling_y)) >> pic.subsampling_y
			}
			for y in 0 ..< h {
				if pic.bit_depth == 8 {
					append(&bytes, ..pic.planes8[p][y * pic.strides[p]:][:w])
				} else {
					append(&bytes, ..mem.slice_to_bytes(pic.planes16[p][y * pic.strides[p]:][:w]))
				}
			}
		}
		os.write(out, bytes[:])
		frames += 1
	}
	fmt.eprintfln("%d frames", frames)
}
