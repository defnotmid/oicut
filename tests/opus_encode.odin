// Encodes 48 kHz 16-bit PCM with audio/opus into an opus_demo bitstream
// (.bit: per packet a big-endian length and final range, then the packet),
// which `opus_demo -d` and tests/opus_decode.odin decode and check packet
// by packet. The input is padded with silence to whole frames, past the
// encoder's delay, and its depth is 16 bits (as opus_demo tells libopus).
//
//   opus_encode in.pcm out.bit channels bit_rate [complexity: 9] [vbr|cvbr|cbr] [frame_samples: 960]
package main

import opus "../audio/opus"
import "core:fmt"
import "core:os"
import "core:strconv"
import "core:time"

num :: proc(s: string) -> int {
	v, _ := strconv.parse_int(s)
	return v
}

main :: proc() {
	if len(os.args) < 5 {fmt.eprintln("usage: opus_encode in.pcm out.bit channels bit_rate [complexity] [vbr|cvbr|cbr] [frame_samples]"); os.exit(2)}
	data, rerr := os.read_entire_file(os.args[1], context.allocator)
	if rerr != nil do os.exit(1)
	config := opus.default_encoder_config(num(os.args[3]))
	config.bit_rate = num(os.args[4])
	if len(os.args) > 5 do config.complexity = num(os.args[5])
	if len(os.args) > 6 {
		switch os.args[6] {
		case "cbr":
			config.vbr = false
		case "cvbr":
			config.constrained_vbr = true
		}
	}
	if len(os.args) > 7 do config.frame_samples = num(os.args[7])
	config.lsb_depth = 16
	e: opus.Encoder
	if err := opus.init_encoder(&e, config); err != .None {fmt.eprintln("init:", err); os.exit(1)}
	c := config.channels
	n := e.config.frame_samples
	samples := len(data) / 2
	// Enough frames to flush the delay.
	frames := (samples / c + opus.ENCODER_PRE_SKIP + n - 1) / n
	pcm := make([]f32, frames * n * c)
	for i in 0 ..< samples do pcm[i] = f32(i16(u16(data[2 * i]) | u16(data[2 * i + 1]) << 8)) / 32768
	out := make([dynamic]u8, 0, frames * (opus.MAX_PACKET_BYTES + 8))
	packet: [opus.MAX_PACKET_BYTES]u8
	encode_time: time.Duration
	total := 0
	for f in 0 ..< frames {
		start := time.now()
		size, err := opus.encode_packet(&e, pcm[f * n * c:][:n * c], packet[:])
		encode_time += time.since(start)
		if err != .None {fmt.eprintln("frame", f, err); os.exit(1)}
		rng := opus.encoder_final_range(&e)
		for v in ([2]u32{u32(size), rng}) do append(&out, u8(v >> 24), u8(v >> 16), u8(v >> 8), u8(v))
		append(&out, ..packet[:size])
		total += size
	}
	if werr := os.write_entire_file(os.args[2], out[:]); werr != nil do os.exit(1)
	seconds := f64(frames * n) / 48000
	fmt.printf("packets=%d bytes=%d kbps=%.2f encode_ms=%.3f\n", frames, total, f64(total) * 8 / seconds / 1000, time.duration_milliseconds(encode_time))
}
