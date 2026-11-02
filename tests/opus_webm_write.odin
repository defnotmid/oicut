// Writes 3 s of VP9 (a moving gradient, 30 fps) with a stereo chord as Opus
// through video.Writer, audio alongside each frame, and the chord as 16-bit
// PCM, for FFmpeg to read the file back against.
//
//   opus_webm_write out.webm source.pcm
package main

import video "../video"
import "core:fmt"
import "core:math"
import "core:os"

main :: proc() {
	W, H, FPS, SECONDS :: 320, 240, 30, 3
	if len(os.args) < 3 {fmt.eprintln("usage: opus_webm_write out.webm source.pcm"); os.exit(2)}
	config := video.default_writer_config(W, H, FPS, 1)
	config.codec = .VP9
	config.sample_rate, config.channels = 48000, 2
	w: video.Writer
	if e := video.create(&w, os.args[1], config); e != .None {fmt.eprintln("create:", e, w.error_detail); os.exit(1)}
	// Not whole Opus frames: 639 samples of padding, a DiscardPadding with its
	// top byte's high bit set.
	samples := 48000 * SECONDS + 201
	pcm := make([]f32, samples * 2)
	raw := make([]u8, samples * 4)
	for i in 0 ..< samples do for c in 0 ..< 2 {
		t := f64(i) / 48000
		v := f32(0.3 * math.sin(2 * math.PI * (330 + 110 * f64(c)) * t) + 0.1 * math.sin(2 * math.PI * 2750 * t))
		pcm[i * 2 + c] = v
		s := i16(v * 32767)
		raw[(i * 2 + c) * 2], raw[(i * 2 + c) * 2 + 1] = u8(s), u8(u16(s) >> 8)
	}
	rgba := make([]u8, W * H * 4)
	per_frame := 48000 / FPS
	for f in 0 ..< FPS * SECONDS {
		for y in 0 ..< H do for x in 0 ..< W {
			p := (y * W + x) * 4
			rgba[p], rgba[p + 1], rgba[p + 2], rgba[p + 3] = u8(x + f * 3), u8(y), 128, 255
		}
		if e := video.write_audio(&w, pcm[f * per_frame * 2:][:per_frame * 2]); e != .None {fmt.eprintln("audio:", e); os.exit(1)}
		if e := video.write_frame(&w, rgba, W * 4); e != .None {fmt.eprintln("frame:", e); os.exit(1)}
	}
	if e := video.write_audio(&w, pcm[FPS * SECONDS * per_frame * 2:]); e != .None {fmt.eprintln("audio:", e); os.exit(1)}
	if e := video.finish(&w); e != .None {fmt.eprintln("finish:", e, w.error_detail); os.exit(1)}
	video.destroy(&w)
	if err := os.write_entire_file(os.args[2], raw); err != nil do os.exit(1)
}
