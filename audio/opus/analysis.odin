package opus

import "core:math"

// The encoder's tonality analysis, after libopus's analysis.c without its
// speech/music network (a CELT-only encoder has no mode to choose): every
// 20 ms, a 480-point FFT of the last 30 ms at 24 kHz gives how tonal the
// signal is (phase predictability per bin), how active (energy against
// its recent range), how wide (the highest band above the noise floor),
// and which bands leak into their neighbours. CELT reads it for the VBR
// target, the allocation trim, the band boosts and the coded bandwidth.

@(private)
NB_TBANDS :: 18
@(private)
NB_FRAMES :: 8 // the stationarity window
@(private)
ANALYSIS_BUF :: 720 // 30 ms at 24 kHz
@(private)
DETECT_SIZE :: 100 // frames of results kept
@(private)
TONAL_SKIP_BANDS :: 9
@(private)
LEAK_BANDS :: 19

// One 20 ms frame's results.
@(private)
Analysis_Info :: struct {
	valid:           bool,
	tonality:        f32,
	tonality_slope:  f32, // tonality rising with frequency
	activity:        f32,
	max_pitch_ratio: f32, // energy below 2.7 kHz over above, at most 1
	bandwidth:       int, // the highest band with signal, of 20
	leak_boost:      [LEAK_BANDS]u8, // in 1/64 of a log2 unit
}

@(private)
Analysis :: struct {
	angle, d_angle, d2_angle: [240]f32, // per bin: phase and its two differences
	inmem:              [ANALYSIS_BUF]f32,
	mem_fill:           int,
	hp_ener_accum:      f32, // energy above 12 kHz since the last frame
	downmix_state:      [3]f32,
	prev_band_tonality: [NB_TBANDS]f32,
	prev_tonality:      f32,
	prev_bandwidth:     int,
	e:                  [NB_FRAMES][NB_TBANDS]f32,
	low_e, high_e:      [NB_TBANDS]f32, // the range of each band's log energy
	mean_e:             [NB_TBANDS + 1]f32,
	e_count, count:     int,
	write_pos:          int,
	read_pos:           int,
	read_subframe:      int,
	info:               [DETECT_SIZE]Analysis_Info,
}

@(private, rodata)
TBANDS := [NB_TBANDS + 1]int{4, 8, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64, 80, 96, 112, 136, 160, 192, 240}

// The first half of a sine window over 480 points.
@(private, rodata)
ANALYSIS_WINDOW := [240]f32 {
	0.000043, 0.000171, 0.000385, 0.000685, 0.001071, 0.001541, 0.002098, 0.002739,
	0.003466, 0.004278, 0.005174, 0.006156, 0.007222, 0.008373, 0.009607, 0.010926,
	0.012329, 0.013815, 0.015385, 0.017037, 0.018772, 0.020590, 0.022490, 0.024472,
	0.026535, 0.028679, 0.030904, 0.033210, 0.035595, 0.038060, 0.040604, 0.043227,
	0.045928, 0.048707, 0.051564, 0.054497, 0.057506, 0.060591, 0.063752, 0.066987,
	0.070297, 0.073680, 0.077136, 0.080665, 0.084265, 0.087937, 0.091679, 0.095492,
	0.099373, 0.103323, 0.107342, 0.111427, 0.115579, 0.119797, 0.124080, 0.128428,
	0.132839, 0.137313, 0.141849, 0.146447, 0.151105, 0.155823, 0.160600, 0.165435,
	0.170327, 0.175276, 0.180280, 0.185340, 0.190453, 0.195619, 0.200838, 0.206107,
	0.211427, 0.216797, 0.222215, 0.227680, 0.233193, 0.238751, 0.244353, 0.250000,
	0.255689, 0.261421, 0.267193, 0.273005, 0.278856, 0.284744, 0.290670, 0.296632,
	0.302628, 0.308658, 0.314721, 0.320816, 0.326941, 0.333097, 0.339280, 0.345492,
	0.351729, 0.357992, 0.364280, 0.370590, 0.376923, 0.383277, 0.389651, 0.396044,
	0.402455, 0.408882, 0.415325, 0.421783, 0.428254, 0.434737, 0.441231, 0.447736,
	0.454249, 0.460770, 0.467298, 0.473832, 0.480370, 0.486912, 0.493455, 0.500000,
	0.506545, 0.513088, 0.519630, 0.526168, 0.532702, 0.539230, 0.545751, 0.552264,
	0.558769, 0.565263, 0.571746, 0.578217, 0.584675, 0.591118, 0.597545, 0.603956,
	0.610349, 0.616723, 0.623077, 0.629410, 0.635720, 0.642008, 0.648271, 0.654508,
	0.660720, 0.666903, 0.673059, 0.679184, 0.685279, 0.691342, 0.697372, 0.703368,
	0.709330, 0.715256, 0.721144, 0.726995, 0.732807, 0.738579, 0.744311, 0.750000,
	0.755647, 0.761249, 0.766807, 0.772320, 0.777785, 0.783203, 0.788573, 0.793893,
	0.799162, 0.804381, 0.809547, 0.814660, 0.819720, 0.824724, 0.829673, 0.834565,
	0.839400, 0.844177, 0.848895, 0.853553, 0.858151, 0.862687, 0.867161, 0.871572,
	0.875920, 0.880203, 0.884421, 0.888573, 0.892658, 0.896677, 0.900627, 0.904508,
	0.908321, 0.912063, 0.915735, 0.919335, 0.922864, 0.926320, 0.929703, 0.933013,
	0.936248, 0.939409, 0.942494, 0.945503, 0.948436, 0.951293, 0.954072, 0.956773,
	0.959396, 0.961940, 0.964405, 0.966790, 0.969096, 0.971321, 0.973465, 0.975528,
	0.977510, 0.979410, 0.981228, 0.982963, 0.984615, 0.986185, 0.987671, 0.989074,
	0.990393, 0.991627, 0.992778, 0.993844, 0.994826, 0.995722, 0.996534, 0.997261,
	0.997902, 0.998459, 0.998929, 0.999315, 0.999615, 0.999829, 0.999957, 1.000000,
}

analysis_reset :: proc(a: ^Analysis) {
	a^ = {}
	a.mem_fill = 240
}

// Analyses one frame of interleaved +-1 input (20 ms at a time) and
// returns the results that apply to it.
@(private)
run_analysis :: proc(a: ^Analysis, pcm: []f32, frame_size, channels, lsb_depth: int) -> Analysis_Info {
	for at := 0; at < frame_size; at += 960 do tonality_analysis(a, pcm, min(960, frame_size - at), at, channels, lsb_depth)
	return analysis_get_info(a, frame_size)
}

// The frame's results, with the tonality and bandwidth of its neighbours.
@(private)
analysis_get_info :: proc(a: ^Analysis, frame_size: int) -> Analysis_Info {
	pos := a.read_pos
	a.read_subframe += frame_size / 120
	for a.read_subframe >= 8 {
		a.read_subframe -= 8
		a.read_pos += 1
	}
	if a.read_pos >= DETECT_SIZE do a.read_pos -= DETECT_SIZE
	if pos == a.write_pos do pos -= 1
	if pos < 0 do pos = DETECT_SIZE - 1
	pos0 := pos
	info := a.info[pos]
	if !info.valid do return info
	tonality_max, tonality_sum := info.tonality, info.tonality
	count := 1
	// Look ahead for a tone, past the detector's delay, and take the widest
	// bandwidth around.
	span := 6
	for _ in 0 ..< 3 {
		pos = (pos + 1) % DETECT_SIZE
		if pos == a.write_pos do break
		tonality_max = max(tonality_max, a.info[pos].tonality)
		tonality_sum += a.info[pos].tonality
		count += 1
		info.bandwidth = max(info.bandwidth, a.info[pos].bandwidth)
		span -= 1
	}
	pos = pos0
	for _ in 0 ..< span {
		pos = pos == 0 ? DETECT_SIZE - 1 : pos - 1
		if pos == a.write_pos do break
		info.bandwidth = max(info.bandwidth, a.info[pos].bandwidth)
	}
	info.tonality = max(tonality_sum / f32(count), tonality_max - 0.2)
	return info
}

// The mix of all channels at 24 kHz into y (the half-band filter's
// all-pass pair), returning the energy above 12 kHz.
@(private)
downmix_and_resample :: proc(a: ^Analysis, pcm: []f32, y: []f32, samples, offset, channels: int) -> f32 {
	if samples == 0 do return 0
	n, at := 2 * samples, 2 * offset
	scale := channels == 2 ? f32(0.5 * 32768) : 32768
	s := &a.downmix_state
	hp_ener: f32
	for k in 0 ..< n / 2 {
		even, odd: f32
		for ch in 0 ..< channels {
			even += pcm[(at + 2 * k) * channels + ch]
			odd += pcm[(at + 2 * k + 1) * channels + ch]
		}
		even, odd = even * scale, odd * scale
		// All-pass section for the even sample.
		x := 0.6074371 * (even - s[0])
		out := s[0] + x
		s[0] = even + x
		out_hp := out
		// And for the odd one, added to it.
		x = 0.15063 * (odd - s[1])
		out += s[1] + x
		s[1] = odd + x
		// The high half: the odd sample negated.
		x = 0.15063 * (-odd - s[2])
		out_hp += s[2] + x
		s[2] = -odd + x
		hp_ener += out_hp * out_hp
		y[k] = 0.5 * out
	}
	return hp_ener * (1.0 / 32768 / 32768)
}

// Rounds to the nearest integer, halves to even (lrintf), for |x| < 2^31.
@(private)
round_even :: #force_inline proc "contextless" (x: f32) -> f32 {
	t := x + 0.5
	r := f32(i32(t))
	if r > t do r -= 1 // the floor of t
	if r - x == 0.5 && i32(r) & 1 != 0 do r -= 1
	return r
}

// libopus's atan2 approximation.
@(private)
fast_atan2 :: proc "contextless" (y, x: f32) -> f32 {
	A :: 0.43157974
	B :: 0.67848403
	C :: 0.08595542
	E :: f32(math.PI / 2)
	x2, y2 := x * x, y * y
	if x2 + y2 < 1e-18 do return 0
	if x2 < y2 do return -x * y * (y2 + A * x2) / ((y2 + B * x2) * (y2 + C * x2)) + (y < 0 ? -E : E)
	return x * y * (x2 + A * y2) / ((x2 + B * y2) * (x2 + C * y2)) + (y < 0 ? -E : E) - (x * y < 0 ? -E : E)
}

@(private)
tonality_analysis :: proc(a: ^Analysis, pcm: []f32, length_48k, offset_48k, channels, lsb_depth: int) {
	N, N2 :: 480, 240
	LEAKAGE_OFFSET :: 2.5
	LEAKAGE_SLOPE :: 2
	PI4 :: f32(math.PI * math.PI * math.PI * math.PI)
	SCALE_ENER :: 1.0 / 32768 / 32768
	// Noise-floor decay for the bandwidth: -2.2 dB/s.
	alpha_e2 := a.count <= 1 ? 1 : 1 / f32(min(100, 1 + a.count))
	// At 24 kHz.
	length, offset := length_48k / 2, offset_48k / 2
	a.hp_ener_accum += downmix_and_resample(a, pcm, a.inmem[a.mem_fill:], min(length, ANALYSIS_BUF - a.mem_fill), offset, channels)
	if a.mem_fill + length < ANALYSIS_BUF {
		a.mem_fill += length
		return
	}
	hp_ener := a.hp_ener_accum
	info := &a.info[a.write_pos]
	a.write_pos = (a.write_pos + 1) % DETECT_SIZE
	silence := true
	for v in a.inmem do if abs(v) > 1 / f32(i32(1) << uint(lsb_depth)) do silence = false

	// Two real halves of the window as one complex FFT.
	re, im: [N]f32
	{
		in_re, in_im: [N]f32
		for i in 0 ..< N2 {
			w := ANALYSIS_WINDOW[i]
			in_re[i] = w * a.inmem[i]
			in_im[i] = w * a.inmem[N2 + i]
			in_re[N - i - 1] = w * a.inmem[N - i - 1]
			in_im[N - i - 1] = w * a.inmem[N + N2 - i - 1]
		}
		plan := &mdct_tables.plans[0]
		for q in 0 ..< N {
			i := int(plan.perm[q])
			re[q], im[q] = in_re[i] * (1.0 / N), in_im[i] * (1.0 / N)
		}
		fft(plan, &mdct_tables, re[:], im[:])
	}
	copy(a.inmem[:240], a.inmem[ANALYSIS_BUF - 240:])
	remaining := length - (ANALYSIS_BUF - a.mem_fill)
	a.hp_ener_accum = downmix_and_resample(a, pcm, a.inmem[240:], remaining, offset + ANALYSIS_BUF - a.mem_fill, channels)
	a.mem_fill = 240 + remaining
	if silence {
		// The last frame's analysis.
		info^ = a.info[(a.write_pos + DETECT_SIZE - 2) % DETECT_SIZE]
		return
	}
	if math.is_nan(re[0]) {
		info.valid = false
		return
	}
	bin_energy :: #force_inline proc(re, im: ^[N]f32, i: int) -> f32 {
		return re[i] * re[i] + re[N - i] * re[N - i] + im[i] * im[i] + im[N - i] * im[N - i]
	}

	// Tonality: how well each bin's phase is predicted from its last two.
	tonality, tonality2, noisiness: [N2]f32
	for i in 1 ..< N2 {
		x1r, x1i := re[i] + re[N - i], im[i] - im[N - i]
		x2r, x2i := im[i] + im[N - i], re[N - i] - re[i]
		angle := f32(0.5 / math.PI) * fast_atan2(x1i, x1r)
		d_angle := angle - a.angle[i]
		d2_angle := d_angle - a.d_angle[i]
		angle2 := f32(0.5 / math.PI) * fast_atan2(x2i, x2r)
		d_angle2 := angle2 - angle
		d2_angle2 := d_angle2 - d_angle
		mod1 := d2_angle - round_even(d2_angle)
		noisiness[i] = abs(mod1)
		mod1 *= mod1
		mod1 *= mod1
		mod2 := d2_angle2 - round_even(d2_angle2)
		noisiness[i] += abs(mod2)
		mod2 *= mod2
		mod2 *= mod2
		avg_mod := 0.25 * (a.d2_angle[i] + mod1 + 2 * mod2)
		// Two frames' delay, more reliable; and none.
		tonality[i] = 1 / (1 + 40 * 16 * PI4 * avg_mod) - 0.015
		tonality2[i] = 1 / (1 + 40 * 16 * PI4 * mod2) - 0.015
		a.angle[i], a.d_angle[i], a.d2_angle[i] = angle2, d_angle2, mod2
	}
	for i in 2 ..< N2 - 1 {
		tt := min(tonality2[i], max(tonality2[i - 1], tonality2[i + 1]))
		tonality[i] = 0.9 * max(tonality[i], tt - 0.1)
	}

	if a.count == 0 {
		for b in 0 ..< NB_TBANDS do a.low_e[b], a.high_e[b] = 1e10, -1e10
	}
	frame_tonality, max_frame_tonality, frame_noisiness, relative_e, slope: f32
	band_log2: [NB_TBANDS + 1]f32
	{
		// The first band's energy, DC doubled.
		e := 4 * re[0] * re[0] + 4 * im[0] * im[0]
		for i in 1 ..< 4 do e += bin_energy(&re, &im, i)
		band_log2[0] = 0.5 * 1.442695 * f32(math.ln(f64(SCALE_ENER * e + 1e-10)))
	}
	band_tonality: [NB_TBANDS]f32
	for b in 0 ..< NB_TBANDS {
		e, te, ne: f32
		for i in TBANDS[b] ..< TBANDS[b + 1] {
			bin_e := SCALE_ENER * bin_energy(&re, &im, i)
			e += bin_e
			te += bin_e * max(0, tonality[i])
			ne += bin_e * 2 * (0.5 - noisiness[i])
		}
		// Extreme energies would make NaNs later.
		if !(e < 1e9) || math.is_nan(e) {
			info.valid = false
			return
		}
		a.e[a.e_count][b] = e
		frame_noisiness += ne / (1e-15 + e)
		log_e := f32(math.ln(f64(e + 1e-10)))
		band_log2[b + 1] = 0.5 * 1.442695 * log_e
		if a.count == 0 do a.high_e[b], a.low_e[b] = log_e, log_e
		if a.high_e[b] > a.low_e[b] + 7.5 {
			if a.high_e[b] - log_e > log_e - a.low_e[b] {
				a.high_e[b] -= 0.01
			} else {
				a.low_e[b] += 0.01
			}
		}
		if log_e > a.high_e[b] {
			a.high_e[b] = log_e
			a.low_e[b] = max(a.high_e[b] - 15, a.low_e[b])
		} else if log_e < a.low_e[b] {
			a.low_e[b] = log_e
			a.high_e[b] = min(a.low_e[b] + 15, a.high_e[b])
		}
		relative_e += (log_e - a.low_e[b]) / (1e-5 + (a.high_e[b] - a.low_e[b]))
		// Stationarity: the band's energy steady over the window.
		l1, l2: f32
		for f in 0 ..< NB_FRAMES {
			l1 += f32(math.sqrt(f64(a.e[f][b])))
			l2 += a.e[f][b]
		}
		stationarity := min(0.99, l1 / f32(math.sqrt(1e-15 + NB_FRAMES * f64(l2))))
		stationarity *= stationarity
		stationarity *= stationarity
		band_tonality[b] = max(te / (1e-15 + e), stationarity * a.prev_band_tonality[b])
		// Over the last NB_TBANDS - TONAL_SKIP_BANDS bands.
		frame_tonality += band_tonality[b]
		if b >= NB_TBANDS - TONAL_SKIP_BANDS do frame_tonality -= band_tonality[b - NB_TBANDS + TONAL_SKIP_BANDS]
		max_frame_tonality = max(max_frame_tonality, (1 + 0.03 * f32(b - NB_TBANDS)) * frame_tonality)
		slope += band_tonality[b] * f32(b - 8)
		a.prev_band_tonality[b] = band_tonality[b]
	}

	// Leakage: how far each band sits below what its neighbours spill into
	// it, or above what it spills out.
	leakage_from, leakage_to: [NB_TBANDS + 1]f32
	leakage_from[0] = band_log2[0]
	leakage_to[0] = band_log2[0] - LEAKAGE_OFFSET
	for b in 1 ..< NB_TBANDS + 1 {
		leak_slope := LEAKAGE_SLOPE * f32(TBANDS[b] - TBANDS[b - 1]) / 4
		leakage_from[b] = min(leakage_from[b - 1] + leak_slope, band_log2[b])
		leakage_to[b] = max(leakage_to[b - 1] - leak_slope, band_log2[b] - LEAKAGE_OFFSET)
	}
	for b := NB_TBANDS - 2; b >= 0; b -= 1 {
		leak_slope := LEAKAGE_SLOPE * f32(TBANDS[b + 1] - TBANDS[b]) / 4
		leakage_from[b] = min(leakage_from[b + 1] + leak_slope, leakage_from[b])
		leakage_to[b] = max(leakage_to[b + 1] - leak_slope, leakage_to[b])
	}
	for b in 0 ..< NB_TBANDS + 1 {
		boost := max(0, leakage_to[b] - band_log2[b]) + max(0, band_log2[b] - (leakage_from[b] + LEAKAGE_OFFSET))
		info.leak_boost[b] = u8(min(255, int(math.floor(0.5 + 64 * boost))))
	}

	// Bandwidth: the last band above the PCM noise floor and not masked.
	noise_floor := f32(5.7e-4) / f32(i32(1) << uint(max(0, lsb_depth - 8)))
	noise_floor *= noise_floor
	bandwidth := 0
	bandwidth_mask, max_e, below_max_pitch, above_max_pitch: f32
	masked: [NB_TBANDS + 1]bool
	for b in 0 ..< NB_TBANDS {
		e: f32
		width := f32(TBANDS[b + 1] - TBANDS[b])
		for i in TBANDS[b] ..< TBANDS[b + 1] do e += bin_energy(&re, &im, i)
		e *= SCALE_ENER
		max_e = max(max_e, e)
		if TBANDS[b] < 64 {
			below_max_pitch += e
		} else {
			above_max_pitch += e
		}
		a.mean_e[b] = max((1 - alpha_e2) * a.mean_e[b], e)
		em := max(e, a.mean_e[b])
		// Within 90 dB of the peak and above the noise floor.
		if e * 1e9 > max_e && (em > 3 * noise_floor * width || e > noise_floor * width) do bandwidth = b + 1
		masked[b] = e < (a.prev_bandwidth >= b + 1 ? 0.01 : 0.05) * bandwidth_mask
		// A follower with a 13 dB/Bark slope for the spreading function.
		bandwidth_mask = max(0.05 * bandwidth_mask, e)
	}
	{
		// Above 12 kHz only the high band's energy is known; the half-band
		// filter leaks some low energy into it.
		e := hp_ener * (1.0 / (60 * 60))
		noise_ratio: f32 = a.prev_bandwidth == 20 ? 10 : 30
		above_max_pitch += e
		a.mean_e[NB_TBANDS] = max((1 - alpha_e2) * a.mean_e[NB_TBANDS], e)
		em := max(e, a.mean_e[NB_TBANDS])
		if em > 3 * noise_ratio * noise_floor * 160 || e > noise_ratio * noise_floor * 160 do bandwidth = 20
		masked[NB_TBANDS] = e < (a.prev_bandwidth == 20 ? 0.01 : 0.05) * bandwidth_mask
	}
	info.max_pitch_ratio = above_max_pitch > below_max_pitch ? below_max_pitch / above_max_pitch : 1
	// Resampling aliasing can put a little energy in the first band cut.
	if bandwidth == 20 && masked[NB_TBANDS] {
		bandwidth -= 2
	} else if bandwidth > 0 && bandwidth <= NB_TBANDS && masked[bandwidth - 1] {
		bandwidth -= 1
	}
	if a.count <= 2 do bandwidth = 20

	relative_e /= NB_TBANDS
	if a.count < 10 do relative_e = 0.5
	frame_noisiness /= NB_TBANDS
	info.activity = frame_noisiness + (1 - frame_noisiness) * relative_e
	frame_tonality = max(max_frame_tonality / (NB_TBANDS - TONAL_SKIP_BANDS), a.prev_tonality * 0.8)
	a.prev_tonality = frame_tonality
	info.tonality_slope = slope / (8 * 8)
	info.tonality = frame_tonality
	a.e_count = (a.e_count + 1) % NB_FRAMES
	a.count = min(a.count + 1, 10000)
	info.bandwidth = bandwidth
	a.prev_bandwidth = bandwidth
	info.valid = true
}
