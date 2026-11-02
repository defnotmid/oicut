package opus

import "core:math"
import "core:testing"

// A second of a chord, a click, noise and silence: tonal, transient and
// noisy frames, the prefilter on and off, and the silence flag.
@(private = "file")
test_signal :: proc(pcm: []f32, channels: int) {
	seed: u32 = 1
	for i in 0 ..< len(pcm) / channels {
		t := f64(i) / 48000
		for c in 0 ..< channels {
			v: f64
			switch {
			case t < 0.4:
				v = 0.3 * math.sin(2 * math.PI * 220 * f64(c + 1) * t) + 0.2 * math.sin(2 * math.PI * 1375 * t)
			case t < 0.45:
				v = i % 4800 < 48 ? 0.9 : 0
			case t < 0.8:
				seed = lcg_rand(seed)
				v = 0.2 * f64(i32(seed)) / f64(max(i32))
			}
			pcm[i * channels + c] = f32(v)
		}
	}
}

// Encodes and decodes the signal; every packet decodes to the encoder's
// final range state, and the output follows the input past the pre-skip.
@(private = "file")
round_trip :: proc(t: ^testing.T, config: Encoder_Config) {
	e: Encoder
	testing.expect_value(t, init_encoder(&e, config), Error.None)
	c := config.channels
	n := e.config.frame_samples
	frames := 48000 / n + 1
	pcm := make([]f32, frames * n * c)
	defer delete(pcm)
	test_signal(pcm[:48000 * c], c)
	out := make([]f32, frames * n * c)
	defer delete(out)
	d: Decoder
	init_decoder(&d, c)
	packet: [MAX_PACKET_BYTES]u8
	total := 0
	for f in 0 ..< frames {
		size, err := encode_packet(&e, pcm[f * n * c:][:n * c], packet[:])
		testing.expect_value(t, err, Error.None)
		if err != .None do return
		total += size
		samples, derr := decode_packet(&d, packet[:size], out[f * n * c:])
		testing.expect_value(t, derr, Error.None)
		testing.expect_value(t, samples, n)
		testing.expectf(t, d.final_range == encoder_final_range(&e), "frame %d: final range %x, encoded %x (%v)", f, d.final_range, encoder_final_range(&e), config)
	}
	// The chord, after the encoder's delay.
	signal, noise: f64
	for i in 4800 ..< 18000 do for ch in 0 ..< c {
		s := f64(pcm[i * c + ch])
		r := f64(out[(i + ENCODER_PRE_SKIP) * c + ch])
		signal += s * s
		noise += (r - s) * (r - s)
	}
	snr := 10 * math.log10(signal / noise)
	testing.expectf(t, snr > 10, "SNR %.1f dB (%v)", snr, config)
	if !config.vbr do testing.expect_value(t, total, frames * ((bitrate_to_bits(e.config.bit_rate, n) + 4) / 8))
}

@(test)
test_encoder_round_trip :: proc(t: ^testing.T) {
	for channels in 1 ..= 2 do for mode in 0 ..< 3 do for complexity in ([3]int{0, 5, 10}) do for frame in ([2]int{120, 960}) {
		config := default_encoder_config(channels)
		config.bit_rate = 48000 * channels
		config.vbr = mode != 2
		config.constrained_vbr = mode == 1
		config.complexity = complexity
		config.frame_samples = frame
		round_trip(t, config)
	}
}

@(test)
test_encoder_head :: proc(t: ^testing.T) {
	buf: [19]u8
	testing.expect_value(t, write_head(buf[:], 2), 19)
	h, err := parse_head(buf[:])
	testing.expect_value(t, err, Error.None)
	testing.expect_value(t, h.channels, 2)
	testing.expect_value(t, h.pre_skip, ENCODER_PRE_SKIP)
	testing.expect_value(t, h.mapping_family, 0)
}

@(test)
test_encoder_refuses :: proc(t: ^testing.T) {
	e: Encoder
	bad := default_encoder_config(3)
	testing.expect_value(t, init_encoder(&e, bad), Error.Unsupported)
	bad = default_encoder_config(2)
	bad.frame_samples = 1920
	testing.expect_value(t, init_encoder(&e, bad), Error.Unsupported)
	bad = default_encoder_config(1)
	bad.bit_rate = 600000
	testing.expect_value(t, init_encoder(&e, bad), Error.Unsupported)
}
