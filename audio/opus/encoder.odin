package opus

import "core:math"

// Opus encoding (RFC 6716 5): CELT-only packets of one frame, 2.5 to 20 ms,
// mono or stereo at 48 kHz. The bandwidth follows the bitrate as libopus
// picks it for music; full band from 12 kb/s.

Encoder_Config :: struct {
	channels:        int, // 1 or 2
	bit_rate:        int, // b/s, 6000 to 510000; 0 selects 64 kb/s per channel
	frame_samples:   int, // per channel: 120, 240, 480 or 960 (0 selects 960)
	complexity:      int, // 0 to 10, as libopus (-1 selects its default, 9)
	vbr:             bool,
	constrained_vbr: bool, // VBR within a one-frame reservoir
	lsb_depth:       int, // the input's significant bits, 8 to 24 (0 selects 24): 16 for 16-bit sources
}

default_encoder_config :: proc(channels: int) -> Encoder_Config {
	return {channels = channels, complexity = -1, vbr = true}
}

Encoder :: struct {
	celt:       Celt_Encoder,
	analysis:   Analysis, // from complexity 7
	config:     Encoder_Config,
	bandwidth:  Bandwidth, // the rate's; a frame may code less
	equiv_rate: int, // the rate the bandwidth decisions use
	hp_mem:     [2]f32, // DC rejection
	pcm:        [2 * (SHORT_MDCT << MAX_LM)]f32,
}

@(private, rodata)
END_BAND := [Bandwidth]int{.Narrow = 13, .Medium = 17, .Wide = 17, .Super_Wide = 19, .Full = 21}

// The samples a decoder discards at the start (OpusHead's pre-skip): the
// MDCT overlap.
ENCODER_PRE_SKIP :: OVERLAP

// The most a packet takes.
MAX_PACKET_BYTES :: 1276

init_encoder :: proc(e: ^Encoder, config: Encoder_Config) -> Error {
	c := config
	if c.channels < 1 || c.channels > 2 do return .Unsupported
	if c.frame_samples == 0 do c.frame_samples = 960
	if c.frame_samples != 120 && c.frame_samples != 240 && c.frame_samples != 480 && c.frame_samples != 960 do return .Unsupported
	if c.bit_rate == 0 do c.bit_rate = 64000 * c.channels
	if c.bit_rate < 6000 || c.bit_rate > 510000 do return .Unsupported
	if c.complexity < 0 do c.complexity = 9
	if c.complexity > 10 do return .Unsupported
	if c.lsb_depth == 0 do c.lsb_depth = 24
	if c.lsb_depth < 8 || c.lsb_depth > 24 do return .Unsupported
	e^ = {
		config = c,
	}
	// As libopus weighs a rate: less for short frames' overhead, CBR (8%)
	// and low complexity.
	equiv := c.bit_rate
	frame_rate := 48000 / c.frame_samples
	if frame_rate > 50 do equiv -= (40 * c.channels + 20) * (frame_rate - 50)
	if !c.vbr do equiv -= equiv / 12
	equiv = equiv * (90 + c.complexity) / 100
	if c.complexity < 5 do equiv = equiv * 9 / 10
	e.equiv_rate = equiv
	// libopus's music thresholds (the hysteresis does not apply: the rate
	// is fixed).
	switch {
	case equiv >= 12000:
		e.bandwidth = .Full
	case equiv >= 11000:
		e.bandwidth = .Super_Wide
	case equiv >= 9000:
		e.bandwidth = .Wide
	case:
		e.bandwidth = .Narrow
	}
	celt_encoder_init(&e.celt, c.channels, END_BAND[e.bandwidth], c.complexity, c.bit_rate, c.vbr, c.constrained_vbr, c.lsb_depth)
	analysis_reset(&e.analysis)
	return .None
}

// Forgets the signal, keeping the configuration.
reset_encoder :: proc(e: ^Encoder) {
	celt_encoder_reset(&e.celt)
	analysis_reset(&e.analysis)
	e.hp_mem = {}
}

// Encodes one frame of config.frame_samples samples per channel
// (interleaved, +-1) as one packet into out (MAX_PACKET_BYTES suffice);
// returns its size.
encode_packet :: proc(e: ^Encoder, pcm: []f32, out: []u8) -> (bytes: int, err: Error) {
	c := e.config.channels
	n := e.config.frame_samples
	if len(pcm) < n * c do return 0, .Buffer_Too_Small
	limit := min(len(out), MAX_PACKET_BYTES)
	if !e.config.vbr {
		// CBR: the rate's bytes, the TOC byte among them.
		limit = min(limit, (bitrate_to_bits(e.config.bit_rate, n) + 4) / 8)
	}
	if limit < 3 do return 0, .Buffer_Too_Small
	info: Analysis_Info
	if e.config.complexity >= 7 do info = run_analysis(&e.analysis, pcm, n, c, e.config.lsb_depth)
	bandwidth := e.bandwidth
	if info.valid {
		// A band-limited signal codes less, down to what the rate always
		// affords.
		detected := Bandwidth.Full
		switch {
		case info.bandwidth <= 12:
			detected = .Narrow
		case info.bandwidth <= 14:
			detected = .Medium
		case info.bandwidth <= 16:
			detected = .Wide
		case info.bandwidth <= 18:
			detected = .Super_Wide
		}
		least := Bandwidth.Full
		switch per_channel := e.equiv_rate / c; {
		case per_channel <= 18000:
			least = .Narrow
		case per_channel <= 24000:
			least = .Medium
		case per_channel <= 30000:
			least = .Wide
		case per_channel <= 44000:
			least = .Super_Wide
		}
		bandwidth = min(bandwidth, max(detected, least))
		if bandwidth == .Medium do bandwidth = .Wide // not in CELT
	}
	e.celt.end = END_BAND[bandwidth]
	// DC rejection at 3 Hz; a frame that is not finite is silence.
	COEF :: f32(6.3 * 3.0 / 48000)
	x := e.pcm[:n * c]
	sum: f32
	for ch in 0 ..< c {
		m := e.hp_mem[ch]
		#no_bounds_check for i in 0 ..< n {
			v := pcm[i * c + ch]
			x[i * c + ch] = v - m
			m = COEF * v + 1e-30 + (1 - COEF) * m
			sum += v * v
		}
		e.hp_mem[ch] = m
	}
	if !(sum < 1e9) || math.is_nan(sum) {
		for &v in x do v = 0
		e.hp_mem = {}
	}
	frame_code := 0
	for SHORT_MDCT << uint(frame_code) != n do frame_code += 1
	@(static, rodata)
	BANDWIDTH_CODE := [Bandwidth]int{.Narrow = 0, .Medium = 1, .Wide = 1, .Super_Wide = 2, .Full = 3}
	config := 16 + 4 * BANDWIDTH_CODE[bandwidth] + frame_code
	out[0] = u8(config << 3 | (c == 2 ? 1 : 0) << 2)
	celt_bytes := celt_encode(&e.celt, x, n, out[1:limit], info) or_return
	return 1 + celt_bytes, .None
}

// The final range coder state of the last packet, which a decoder of it
// must reach (opus_demo's bitstreams carry it).
encoder_final_range :: proc(e: ^Encoder) -> u32 {
	return e.celt.rng
}

// OpusHead (RFC 7845 5.1) for this encoder's streams of `channels` (1 or
// 2), into out (19 bytes); input_rate is informational.
write_head :: proc(out: []u8, channels: int, input_rate := 48000) -> int {
	h := out[:19]
	copy(h, "OpusHead")
	h[8] = 1
	h[9] = u8(channels)
	h[10], h[11] = u8(ENCODER_PRE_SKIP), u8(ENCODER_PRE_SKIP >> 8)
	for k in 0 ..< 4 do h[12 + k] = u8(input_rate >> uint(8 * k))
	h[16], h[17], h[18] = 0, 0, 0
	return 19
}
