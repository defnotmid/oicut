package opus

import "core:math"

// The CELT encoder (RFC 6716 5.3), after libopus's float encoder: the
// analysis chooses what the decoder reads (transients, time-frequency
// resolution, spreading, band boosts, the allocation trim, stereo coding,
// the frame size under VBR), and the shared band, allocation and energy
// coding writes it.

// What the encoder carries between frames, and its per-frame planes, so
// that encoding allocates nothing.
Celt_Encoder :: struct {
	channels:         int,
	end:              int, // coded bands: 13, 17, 19 or 21 (narrow to full band)
	complexity:       int, // 0 to 10
	bitrate:          int, // b/s
	vbr:              bool,
	constrained_vbr:  bool,
	lsb_depth:        int, // the input's significant bits, 8 to 24: its noise floor
	// Carried between frames.
	rng:              u32,
	spread:           Spread,
	delayed_intra:    f32,
	tonal_average:    int,
	hf_average:       int,
	tapset_decision:  int,
	last_coded_bands: int,
	consec_transient: int,
	intensity:        int,
	stereo_saving:    f32,
	spec_avg:         f32,
	overlap_max:      f32,
	preemph_mem:      [2]f32,
	vbr_reservoir:    int,
	vbr_drift:        int,
	vbr_offset:       int,
	vbr_count:        int,
	prefilter:        Comb, // the last frame's, negated in the filter
	// The pre-emphasised input before the prefilter: COMBFILTER_MAXPERIOD
	// samples of the past (the pitch search's), then this frame.
	history:          [2][COMBFILTER_MAXPERIOD + SHORT_MDCT << MAX_LM]f32,
	in_mem:           [2][OVERLAP]f32, // the last OVERLAP prefiltered samples
	// Band log energies as the decoder has them, the two frames before,
	// and the last coarse quantisation errors.
	old_band_e, old_log_e, old_log_e2, energy_error: [2][NB_BANDS]f32,
	// Scratch.
	input:            [2][SHORT_MDCT << MAX_LM + OVERLAP]f32, // prefiltered, the MDCTs' input
	freq:             [2][SHORT_MDCT << MAX_LM]f32,
	x:                [2][SHORT_MDCT << MAX_LM]f32, // normalised, then reconstructed
	norm:             [2][SHORT_MDCT << MAX_LM]f32,
	lowband_scratch:  [176]f32,
	collapse:         [NB_BANDS * 2]u8,
	fft:              Fft_Scratch,
}

celt_encoder_init :: proc(e: ^Celt_Encoder, channels, end, complexity, bitrate: int, vbr, constrained_vbr: bool, lsb_depth: int) {
	e^ = {
		lsb_depth       = lsb_depth,
		channels        = channels,
		end             = end,
		complexity      = complexity,
		bitrate         = bitrate,
		vbr             = vbr,
		constrained_vbr = constrained_vbr,
	}
	celt_encoder_reset(e)
}

// Forgets the signal, keeping the setup.
celt_encoder_reset :: proc(e: ^Celt_Encoder) {
	setup := e^
	e^ = {
		channels        = setup.channels,
		end             = setup.end,
		complexity      = setup.complexity,
		bitrate         = setup.bitrate,
		vbr             = setup.vbr,
		constrained_vbr = setup.constrained_vbr,
		lsb_depth       = setup.lsb_depth,
		delayed_intra   = 1,
		spread          = .Normal,
		tonal_average   = 256,
	}
	for ch in 0 ..< 2 do for i in 0 ..< NB_BANDS {
		e.old_log_e[ch][i] = -28
		e.old_log_e2[ch][i] = -28
	}
}

// Encodes frame_size samples per channel (120 << LM, interleaved, +-1) into
// at most len(out) bytes (at most 1275); returns the bytes used. `analysis`
// is the tonality analysis of the frame, if any.
celt_encode :: proc(e: ^Celt_Encoder, pcm: []f32, frame_size: int, out: []u8, analysis: Analysis_Info = {}) -> (bytes: int, err: Error) {
	lm := 0
	for lm <= MAX_LM && SHORT_MDCT << uint(lm) != frame_size do lm += 1
	if lm > MAX_LM || len(out) < 2 do return 0, .Invalid_Packet
	m := 1 << uint(lm)
	n := m * SHORT_MDCT
	c := e.channels
	start, end := 0, e.end
	eff_end := end
	tell := 1
	nb_bytes := min(len(out), 1275)
	vbr_rate, effective_bytes: int
	// CBR frames are the size of `out`; VBR picks one from the bitrate.
	if e.vbr {
		vbr_rate = bitrate_to_bits(e.bitrate, n) << BITRES
		effective_bytes = vbr_rate >> (3 + BITRES)
	} else {
		effective_bytes = nb_bytes
	}
	nb_available := nb_bytes
	// The rate the decisions are tuned for: as at 20 ms, less the
	// framing overhead of shorter frames.
	overhead := (40 * c + 20) * ((400 >> uint(lm)) - 50)
	equiv_rate := (nb_bytes * 8 * 50 << uint(3 - lm)) - overhead
	if e.vbr do equiv_rate = min(equiv_rate, e.bitrate - overhead)

	enc: Range_Encoder
	range_encoder_init(&enc, out[:nb_bytes])
	if vbr_rate > 0 && e.constrained_vbr {
		// No more than the reservoir allows, and at least two bytes.
		max_allowed := min(max(2, (2 * vbr_rate - e.vbr_reservoir) >> (BITRES + 3)), nb_available)
		if max_allowed < nb_available {
			nb_bytes, nb_available = max_allowed, max_allowed
			range_encoder_shrink(&enc, nb_bytes)
		}
	}
	total_bits := nb_bytes * 8

	samples := pcm[:c * n]
	// The peak over this frame and the overlap before it.
	sample_max := max(e.overlap_max, max_abs(samples[:c * (n - OVERLAP)]))
	e.overlap_max = max_abs(samples[c * (n - OVERLAP):])
	sample_max = max(sample_max, e.overlap_max)
	silence := sample_max <= 1 / f32(i32(1) << uint(e.lsb_depth))
	range_encode_bit_logp(&enc, silence, 15)
	if silence {
		if vbr_rate > 0 {
			nb_bytes = min(nb_bytes, 2)
			effective_bytes = nb_bytes
			total_bits = nb_bytes * 8
			nb_available = 2
			range_encoder_shrink(&enc, nb_bytes)
		}
		// The rest of the frame reads as zeros.
		tell = nb_bytes * 8
		enc.nbits_total += tell - range_tell(&enc)
	}
	// Pre-emphasis at the decoder's scale, clipped to +-2 (+6 dBFS), into
	// the history after its past.
	for ch in 0 ..< c {
		h := e.history[ch][COMBFILTER_MAXPERIOD:][:n]
		mem := e.preemph_mem[ch]
		#no_bounds_check for i in 0 ..< n {
			v := clamp(samples[i * c + ch] * 32768, -65536, 65536)
			h[i] = v - mem
			mem = PREEMPH * v
		}
		e.preemph_mem[ch] = mem
	}

	// The analyses read the frame with the overlap before it, unfiltered.
	tone_freq, toneishness := tone_detect(e, n + OVERLAP)
	transient := false
	short_blocks := 0
	tf_estimate: f32
	tf_chan := 0
	if e.complexity >= 1 do transient = transient_analysis(e, n + OVERLAP, &tf_estimate, &tf_chan, tone_freq, toneishness)
	toneishness = min(toneishness, 1 - tf_estimate)

	// The pitch prefilter, into the MDCTs' input.
	pf_enabled := nb_available > 12 * c && !silence && tell + 16 <= total_bits
	pf, qg, pf_on := run_prefilter(e, n, pf_enabled, tf_estimate, nb_available, tone_freq, toneishness, analysis)
	pitch_change := (pf.gain > 0.4 || e.prefilter.gain > 0.4) && (!analysis.valid || analysis.tonality > 0.3) && (f32(pf.period) > 1.26 * f32(e.prefilter.period) || f32(pf.period) < 0.79 * f32(e.prefilter.period))
	if pf_on {
		// Gated by nb_available above, not by the bits left.
		range_encode_bit_logp(&enc, true, 1)
		period := pf.period + 1
		octave := ilog(u32(period)) - 5
		range_encode_uint(&enc, u32(octave), 6)
		range_encode_bits(&enc, u32(period - (16 << uint(octave))), uint(4 + octave))
		range_encode_bits(&enc, u32(qg), 3)
		range_encode_icdf(&enc, pf.tapset, TAPSET_ICDF[:], 2)
	} else if tell + 16 <= total_bits {
		range_encode_bit_logp(&enc, false, 1)
	}
	transient_got_disabled := false
	if lm > 0 && range_tell(&enc) + 3 <= total_bits {
		if transient do short_blocks = m
	} else {
		transient = false
		transient_got_disabled = true
	}

	band_e: [2][NB_BANDS]f32
	band_log_e, band_log_e2: [2][NB_BANDS]f32
	second_mdct := short_blocks != 0 && e.complexity >= 8
	if second_mdct {
		// The long block's energies for the boosts, short and long scaled alike.
		compute_mdcts(e, 0, lm)
		band_energies(e, &band_e, eff_end, lm)
		amp_to_log(&band_e, &band_log_e2, eff_end, end, c)
		for ch in 0 ..< c do for i in 0 ..< end do band_log_e2[ch][i] += 0.5 * f32(lm)
	}
	compute_mdcts(e, short_blocks, lm)
	band_energies(e, &band_e, eff_end, lm)
	amp_to_log(&band_e, &band_log_e, eff_end, end, c)

	// Temporal VBR: frames louder than the running average get more.
	temporal_vbr: f32
	{
		follow: f32 = -10
		frame_avg: f32
		offset: f32 = short_blocks != 0 ? 0.5 * f32(lm) : 0
		for i in start ..< end {
			follow = max(follow - 1, band_log_e[0][i] - offset)
			if c == 2 do follow = max(follow, band_log_e[1][i] - offset)
			frame_avg += follow
		}
		frame_avg /= f32(end - start)
		temporal_vbr = min(3, max(-1.5, frame_avg - e.spec_avg))
		e.spec_avg += 0.02 * temporal_vbr
	}
	if !second_mdct do band_log_e2 = band_log_e

	// A transient the time-domain analysis missed.
	if lm > 0 && range_tell(&enc) + 3 <= total_bits && !transient && e.complexity >= 5 && patch_transient_decision(&band_log_e, &e.old_band_e, start, end, c) {
		transient = true
		short_blocks = m
		compute_mdcts(e, short_blocks, lm)
		band_energies(e, &band_e, eff_end, lm)
		amp_to_log(&band_e, &band_log_e, eff_end, end, c)
		for ch in 0 ..< c do for i in 0 ..< end do band_log_e2[ch][i] += 0.5 * f32(lm)
		tf_estimate = 0.2
	}
	if lm > 0 && range_tell(&enc) + 3 <= total_bits do range_encode_bit_logp(&enc, transient, 3)

	// Band normalisation.
	for ch in 0 ..< c do for i in 0 ..< eff_end {
		g := 1 / (1e-27 + band_e[ch][i])
		for j in m * int(EBANDS[i]) ..< m * int(EBANDS[i + 1]) do e.x[ch][j] = e.freq[ch][j] * g
	}

	offsets, importance, spread_weight: [NB_BANDS]int
	tot_boost := 0
	max_depth := dynalloc_analysis(e, &band_log_e, &band_log_e2, start, end, lm, effective_bytes, transient, &offsets, &tot_boost, &importance, &spread_weight, tone_freq, toneishness, analysis)

	tf_res: [NB_BANDS]int
	tf_select := 0
	if effective_bytes >= 15 * c && e.complexity >= 2 && toneishness < 0.98 {
		lambda := max(80, 20480 / effective_bytes + 2)
		tf_select = tf_analysis(e, eff_end, transient, &tf_res, lambda, n, lm, tf_estimate, tf_chan, &importance)
		for i in eff_end ..< end do tf_res[i] = tf_res[eff_end - 1]
	} else {
		for i in 0 ..< end do tf_res[i] = transient ? 1 : 0
	}

	// Where the energy is stable, lean toward the last error: a constant
	// offset is better than fluctuation.
	for ch in 0 ..< c do for i in start ..< end {
		if abs(band_log_e[ch][i] - e.old_band_e[ch][i]) < 2 do band_log_e[ch][i] -= 0.25 * e.energy_error[ch][i]
	}
	energy_err: [2][NB_BANDS]f32
	quant_coarse_energy(e, &enc, start, end, eff_end, &band_log_e, &energy_err, total_bits, lm, nb_available, e.complexity >= 4)

	tf_code(&enc, start, end, transient, &tf_res, lm, tf_select)

	if range_tell(&enc) + 4 <= total_bits {
		if short_blocks != 0 || e.complexity < 3 || nb_available < 10 * c {
			e.spread = e.complexity == 0 ? .None : .Normal
		} else {
			e.spread = spreading_decision(e, eff_end, m, &spread_weight, pf_on && short_blocks == 0)
		}
		range_encode_icdf(&enc, int(e.spread), SPREAD_ICDF[:], 5)
	} else {
		e.spread = .Normal
	}

	caps := band_caps(lm, c)
	// The band boosts, as the decoder reads them.
	dynalloc_logp: uint = 6
	total_bits <<= BITRES
	total_boost := 0
	tell = range_tell_frac(&enc)
	for i in start ..< end {
		width := c * int(EBANDS[i + 1] - EBANDS[i]) << uint(lm)
		quanta := min(width << BITRES, max(6 << BITRES, width))
		loop_logp := dynalloc_logp
		boost := 0
		j := 0
		for ; tell + int(loop_logp << BITRES) < total_bits - total_boost && boost < caps[i]; j += 1 {
			flag := j < offsets[i]
			range_encode_bit_logp(&enc, flag, loop_logp)
			tell = range_tell_frac(&enc)
			if !flag do break
			boost += quanta
			total_boost += quanta
			loop_logp = 1
		}
		if j > 0 do dynalloc_logp = max(2, dynalloc_logp - 1)
		offsets[i] = boost
	}

	dual_stereo := false
	if c == 2 {
		@(static, rodata)
		INTENSITY_THRESHOLDS := [21]f32{1, 2, 3, 4, 5, 6, 7, 8, 16, 24, 36, 44, 50, 56, 62, 67, 72, 79, 88, 106, 134}
		@(static, rodata)
		INTENSITY_HYSTERESIS := [21]f32{1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 4, 5, 6, 8, 8}
		// 2.5 ms frames are always mid/side.
		if lm != 0 do dual_stereo = stereo_analysis(e, lm)
		e.intensity = hysteresis_decision(f32(equiv_rate / 1000), INTENSITY_THRESHOLDS[:], INTENSITY_HYSTERESIS[:], e.intensity)
		e.intensity = min(end, max(start, e.intensity))
	}

	alloc_trim := 5
	if tell + (6 << BITRES) <= total_bits - total_boost {
		alloc_trim = alloc_trim_analysis(e, &band_log_e, end, lm, tf_estimate, equiv_rate, analysis)
		range_encode_icdf(&enc, alloc_trim, TRIM_ICDF[:], 7)
		tell = range_tell_frac(&enc)
	}

	// VBR: the frame size from the analysis, never so small that the
	// decoder's budget checks so far would read differently (2 bytes' margin).
	min_allowed := ((tell + total_boost + (1 << (BITRES + 3)) - 1) >> (BITRES + 3)) + 2
	if vbr_rate > 0 {
		lm_diff := MAX_LM - lm
		nb_bytes = min(nb_bytes, 1275 >> uint(3 - lm))
		base_target := vbr_rate - ((40 * c + 20) << BITRES)
		if e.constrained_vbr do base_target += e.vbr_offset >> uint(lm_diff)
		target := compute_vbr(e, base_target, lm, equiv_rate, tot_boost, tf_estimate, max_depth, temporal_vbr, analysis, pitch_change)
		target += tell
		nb_available = (target + (1 << (BITRES + 2))) >> (BITRES + 3)
		nb_available = min(nb_bytes, max(min_allowed, nb_available))
		delta := target - vbr_rate
		target = nb_available << (BITRES + 3)
		// Silence neither drifts the rate nor fills the reservoir.
		if silence {
			nb_available = 2
			target = 2 * 8 << BITRES
			delta = 0
		}
		alpha: f32 = 0.001
		if e.vbr_count < 970 {
			e.vbr_count += 1
			alpha = 1 / f32(e.vbr_count + 20)
		}
		if e.constrained_vbr {
			e.vbr_reservoir += target - vbr_rate
			e.vbr_drift += int(alpha * f32(delta * (1 << uint(lm_diff)) - e.vbr_offset - e.vbr_drift))
			e.vbr_offset = -e.vbr_drift
			if e.vbr_reservoir < 0 {
				// Under the floor: spend what was saved.
				if !silence do nb_available += -e.vbr_reservoir / (8 << BITRES)
				e.vbr_reservoir = 0
			}
		}
		nb_bytes = min(nb_bytes, nb_available)
		range_encoder_shrink(&enc, nb_bytes)
	}

	bits := (nb_bytes * 8 << BITRES) - range_tell_frac(&enc) - 1
	anti_collapse_rsv := transient && lm >= 2 && bits >= (lm + 2) << BITRES ? 1 << BITRES : 0
	bits -= anti_collapse_rsv
	a := Allocation {
		intensity   = e.intensity,
		dual_stereo = dual_stereo,
	}
	// Bands above the detected bandwidth may be skipped, never below a
	// rate's minimum.
	signal_bandwidth := end - 1
	if analysis.valid {
		min_bandwidth := 20
		switch {
		case equiv_rate < 32000 * c:
			min_bandwidth = 13
		case equiv_rate < 48000 * c:
			min_bandwidth = 16
		case equiv_rate < 60000 * c:
			min_bandwidth = 18
		case equiv_rate < 80000 * c:
			min_bandwidth = 19
		}
		signal_bandwidth = max(analysis.bandwidth, min_bandwidth)
	}
	compute_allocation(&a, &enc, start, end, &offsets, &caps, alloc_trim, bits, c, lm, e.last_coded_bands, signal_bandwidth)
	e.intensity = a.intensity
	if e.last_coded_bands != 0 {
		e.last_coded_bands = min(e.last_coded_bands + 1, max(e.last_coded_bands - 1, a.coded_bands))
	} else {
		e.last_coded_bands = a.coded_bands
	}
	quant_fine_energy(&enc, &e.old_band_e, &energy_err, &a.fine_quant, start, end, c)
	e.energy_error = {}

	planes := Band_Planes {
		x               = e.x[0][:],
		y               = c == 2 ? e.x[1][:] : nil,
		norm            = e.norm[0][:],
		norm2           = e.norm[1][:],
		lowband_scratch = e.lowband_scratch[:],
		collapse        = e.collapse[:],
		seed            = &e.rng,
		band_e          = &band_e,
		theta_rdo       = e.complexity >= 8,
	}
	quant_all_bands(&enc, planes, start, end, &a, short_blocks, e.spread, &tf_res, nb_bytes * (8 << BITRES) - anti_collapse_rsv, lm, c)

	if anti_collapse_rsv > 0 do range_encode_bits(&enc, e.consec_transient < 2 ? 1 : 0, 1)
	quant_energy_finalise(&enc, &e.old_band_e, &energy_err, &a.fine_quant, &a.fine_priority, nb_bytes * 8 - range_tell(&enc), start, end, c)
	for ch in 0 ..< c do for i in start ..< end do e.energy_error[ch][i] = clamp(energy_err[ch][i], -0.5, 0.5)
	if silence do for ch in 0 ..< 2 do for i in 0 ..< NB_BANDS do e.old_band_e[ch][i] = -28

	if !transient {
		e.old_log_e2 = e.old_log_e
		e.old_log_e = e.old_band_e
	} else {
		for ch in 0 ..< 2 do for i in 0 ..< NB_BANDS do e.old_log_e[ch][i] = min(e.old_log_e[ch][i], e.old_band_e[ch][i])
	}
	for ch in 0 ..< 2 do for i in end ..< NB_BANDS {
		e.old_band_e[ch][i] = 0
		e.old_log_e[ch][i], e.old_log_e2[ch][i] = -28, -28
	}
	e.consec_transient = transient || transient_got_disabled ? e.consec_transient + 1 : 0
	e.prefilter = pf
	for ch in 0 ..< c do copy(e.history[ch][:COMBFILTER_MAXPERIOD], e.history[ch][n:][:COMBFILTER_MAXPERIOD])
	e.rng = enc.rng
	range_encoder_done(&enc)
	if enc.error do return 0, .Invalid_Packet
	return nb_bytes, .None
}

@(private)
bitrate_to_bits :: #force_inline proc "contextless" (bitrate, frame_size: int) -> int {
	return bitrate * 6 / (6 * 48000 / frame_size)
}

// The first of n thresholds val is below, held at prev within its hysteresis.
@(private)
hysteresis_decision :: proc(val: f32, thresholds, hysteresis: []f32, prev: int) -> int {
	i := 0
	for i < len(thresholds) && val >= thresholds[i] do i += 1
	if i > prev && val < thresholds[prev] + hysteresis[prev] do i = prev
	if i < prev && val > thresholds[prev - 1] - hysteresis[prev - 1] do i = prev
	return i
}

// The MDCTs of every channel: one long block, or `short_blocks` short ones
// interleaved coefficient by coefficient.
@(private)
compute_mdcts :: proc(e: ^Celt_Encoder, short_blocks, lm: int) {
	b, n, shift := 1, SHORT_MDCT << uint(lm), MAX_LM - lm
	if short_blocks != 0 do b, n, shift = short_blocks, SHORT_MDCT, MAX_LM
	for ch in 0 ..< e.channels do for blk in 0 ..< b {
		mdct_forward(&mdct_tables, e.input[ch][blk * n:], e.freq[ch][blk:], b, shift, &e.fft)
	}
}

@(private)
band_energies :: proc(e: ^Celt_Encoder, band_e: ^[2][NB_BANDS]f32, end, lm: int) {
	for ch in 0 ..< e.channels do for i in 0 ..< end {
		v := e.freq[ch][int(EBANDS[i]) << uint(lm):int(EBANDS[i + 1]) << uint(lm)]
		band_e[ch][i] = f32(math.sqrt(f64(1e-27 + dot(v, v))))
	}
}

// Amplitudes to log2 energies relative to each band's mean; above eff_end,
// silent.
@(private)
amp_to_log :: proc(band_e, log_e: ^[2][NB_BANDS]f32, eff_end, end, c: int) {
	for ch in 0 ..< c {
		for i in 0 ..< eff_end do log_e[ch][i] = f32(math.log2(f64(band_e[ch][i]))) - E_MEANS[i]
		for i in eff_end ..< end do log_e[ch][i] = -14
	}
}

// The pre-emphasised input before the prefilter from OVERLAP samples before
// this frame: n = the frame size + OVERLAP.
@(private)
unfiltered :: #force_inline proc(e: ^Celt_Encoder, ch, n: int) -> []f32 {
	return e.history[ch][COMBFILTER_MAXPERIOD - OVERLAP:][:n]
}

// A nearly pure tone, by a 2-pole LPC fit: its frequency in radians per
// sample (-1 if none) and how pure (the poles' squared radius).
@(private)
tone_detect :: proc(e: ^Celt_Encoder, n: int) -> (freq: f32, toneishness: f32) {
	x: [SHORT_MDCT << MAX_LM + OVERLAP]f32
	if e.channels == 2 {
		l, r := unfiltered(e, 0, n), unfiltered(e, 1, n)
		for i in 0 ..< n do x[i] = 0.5 * l[i] + 0.5 * r[i]
	} else {
		copy(x[:n], unfiltered(e, 0, n))
	}
	lpc: [2]f32
	delay := 1
	fail := tone_lpc(x[:n], delay, &lpc)
	// Resonating too close to DC: retry downsampled.
	for delay <= 48000 / 3000 && (fail || (lpc[0] > 1 && lpc[1] < 0)) {
		delay *= 2
		fail = tone_lpc(x[:n], delay, &lpc)
	}
	// Complex roots only.
	if !fail && lpc[0] * lpc[0] + 3.999999 * lpc[1] < 0 {
		return f32(math.acos(f64(0.5 * lpc[0]))) / f32(delay), -lpc[1]
	}
	return -1, 0
}

// The 2-tap predictor at `delay` minimising forward and backward error;
// fails when the correlations are singular.
@(private)
tone_lpc :: proc(x: []f32, delay: int, lpc: ^[2]f32) -> (fail: bool) {
	n := len(x)
	r00, r01, r02: f32
	for i in 0 ..< n - 2 * delay {
		r00 += x[i] * x[i]
		r01 += x[i] * x[i + delay]
		r02 += x[i] * x[i + 2 * delay]
	}
	edges: f32
	for i in 0 ..< delay do edges += x[n + i - 2 * delay] * x[n + i - 2 * delay] - x[i] * x[i]
	r11 := r00 + edges
	edges = 0
	for i in 0 ..< delay do edges += x[n + i - delay] * x[n + i - delay] - x[i + delay] * x[i + delay]
	r22 := r11 + edges
	edges = 0
	for i in 0 ..< delay do edges += x[n + i - 2 * delay] * x[n + i - delay] - x[i] * x[i + delay]
	r12 := r01 + edges
	// Forward plus backward.
	r00, r01, r11, r02, r12 = r00 + r22, r01 + r12, 2 * r11, 2 * r02, r12 + r01
	den := r00 * r11 - r01 * r01
	if den < 0.001 * r00 * r11 do return true
	num1 := r02 * r11 - r01 * r12
	lpc[1] = num1 >= den ? 1 : (num1 <= -den ? -1 : num1 / den)
	num0 := r00 * r12 - r02 * r01
	lpc[0] = 0.5 * num0 >= den ? 1.999999 : (0.5 * num0 <= -den ? -1.999999 : num0 / den)
	return false
}

// Whether the frame needs short blocks: the energy's peak over its
// harmonic mean after forward and backward masking, per channel. Also the
// strength (for the VBR boost) and the channel it is strongest in.
@(private)
transient_analysis :: proc(e: ^Celt_Encoder, n: int, tf_estimate: ^f32, tf_chan: ^int, tone_freq, toneishness: f32) -> bool {
	// 6 * 64 / x, trained to minimise the average error.
	@(static, rodata)
	INV_TABLE := [128]u8 {
		255, 255, 156, 110, 86, 70, 59, 51, 45, 40, 37, 33, 31, 28, 26, 25,
		23, 22, 21, 20, 19, 18, 17, 16, 16, 15, 15, 14, 13, 13, 12, 12,
		12, 12, 11, 11, 11, 10, 10, 10, 9, 9, 9, 9, 9, 9, 8, 8,
		8, 8, 8, 7, 7, 7, 7, 7, 7, 6, 6, 6, 6, 6, 6, 6,
		6, 6, 6, 6, 6, 6, 6, 6, 6, 5, 5, 5, 5, 5, 5, 5,
		5, 5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
		4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3,
		3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2,
	}
	FORWARD_DECAY :: f32(0.0625) // forward masking: 6.7 dB/ms
	tmp: [SHORT_MDCT << MAX_LM + OVERLAP]f32
	n2 := n / 2
	mask_metric := 0
	for ch in 0 ..< e.channels {
		x := unfiltered(e, ch, n)
		// High-pass: (1 - 2 z^-1 + z^-2) / (1 - z^-1 + z^-2 / 2).
		mem0, mem1: f32
		for i in 0 ..< n {
			y := mem0 + x[i]
			mem00 := mem0
			mem0 = mem0 - x[i] + 0.5 * mem1
			mem1 = x[i] - mem00
			tmp[i] = y
		}
		// The filter's first samples lack history.
		for &v in tmp[:12] do v = 0
		mean: f32
		mem0 = 0
		for i in 0 ..< n2 {
			x2 := tmp[2 * i] * tmp[2 * i] + tmp[2 * i + 1] * tmp[2 * i + 1]
			mean += x2
			mem0 = x2 + (1 - FORWARD_DECAY) * mem0
			tmp[i] = FORWARD_DECAY * mem0
		}
		// Backward masking: 13.9 dB/ms.
		mem0 = 0
		max_e: f32
		for i := n2 - 1; i >= 0; i -= 1 {
			mem0 = tmp[i] + 0.875 * mem0
			tmp[i] = 0.125 * mem0
			max_e = max(max_e, 0.125 * mem0)
		}
		// The frame energy: the geometric mean of the energy and half the max.
		mean = f32(math.sqrt(f64(mean * max_e * 0.5 * f32(n2))))
		norm := f32(n2) / (1e-15 + mean)
		// The harmonic mean over every fourth sample, off the edges.
		unmask := 0
		for i := 12; i < n2 - 5; i += 4 {
			id := int(max(0, min(127, 64 * norm * (tmp[i] + 1e-15)))) // not negative: truncation is the floor
			unmask += int(INV_TABLE[id])
		}
		unmask = 64 * unmask * 4 / (6 * (n2 - 17))
		if unmask > mask_metric {
			tf_chan^ = ch
			mask_metric = unmask
		}
	}
	transient := mask_metric > 200
	// Not the partial cycle of a very low tone.
	if toneishness > 0.98 && tone_freq < 0.026 {
		transient = false
		mask_metric = 0
	}
	tf_max := max(0, f32(math.sqrt(f64(27 * mask_metric))) - 42)
	tf_estimate^ = f32(math.sqrt(f64(max(0, 0.0069 * min(163, tf_max) - 0.139))))
	return transient
}

// A sudden rise of energy against the last frame, spread -6 dB/Bark.
@(private)
patch_transient_decision :: proc(new_e, old_e: ^[2][NB_BANDS]f32, start, end, c: int) -> bool {
	spread_old: [NB_BANDS]f32
	spread_old[start] = c == 1 ? old_e[0][start] : max(old_e[0][start], old_e[1][start])
	for i in start + 1 ..< end {
		o := c == 1 ? old_e[0][i] : max(old_e[0][i], old_e[1][i])
		spread_old[i] = max(spread_old[i - 1] - 1, o)
	}
	for i := end - 2; i >= start; i -= 1 do spread_old[i] = max(spread_old[i], spread_old[i + 1] - 1)
	mean_diff: f32
	lo := max(2, start)
	for ch in 0 ..< c do for i in lo ..< end - 1 {
		mean_diff += max(0, max(0, new_e[ch][i]) - max(0, spread_old[i]))
	}
	mean_diff /= f32(c * (end - 1 - lo))
	return mean_diff > 1
}

// The L1 norm, which favours fewer, larger coefficients; biased toward
// frequency resolution.
@(private)
l1_metric :: proc(x: []f32, lm: int, bias: f32) -> f32 {
	l1: f32
	for v in x do l1 += abs(v)
	return l1 + f32(lm) * bias * l1
}

// The time-frequency resolution per band: the Haar level with the lowest
// L1 norm, then a Viterbi path over the bands that charges lambda per
// change; returns tf_select.
@(private)
tf_analysis :: proc(e: ^Celt_Encoder, length: int, transient: bool, tf_res: ^[NB_BANDS]int, lambda, n0, lm: int, tf_estimate: f32, tf_chan: int, importance: ^[NB_BANDS]int) -> int {
	bias := 0.04 * max(-0.25, 0.5 - tf_estimate)
	metric, path0, path1: [NB_BANDS]int
	tmp, tmp1: [176]f32
	x := e.x[tf_chan][:n0]
	t := transient ? 1 : 0
	for i in 0 ..< length {
		n := int(EBANDS[i + 1] - EBANDS[i]) << uint(lm)
		// Too narrow to split down to LM -1.
		narrow := EBANDS[i + 1] - EBANDS[i] == 1
		copy(tmp[:n], x[int(EBANDS[i]) << uint(lm):][:n])
		best_l1 := l1_metric(tmp[:n], transient ? lm : 0, bias)
		best_level := 0
		if transient && !narrow {
			copy(tmp1[:n], tmp[:n])
			haar1(tmp1[:], n >> uint(lm), 1 << uint(lm))
			l1 := l1_metric(tmp1[:n], lm + 1, bias)
			if l1 < best_l1 {
				best_l1 = l1
				best_level = -1
			}
		}
		levels := lm + (transient || narrow ? 0 : 1)
		for k in 0 ..< levels {
			b := transient ? lm - k - 1 : k + 1
			haar1(tmp[:], n >> uint(k), 1 << uint(k))
			l1 := l1_metric(tmp[:n], b, bias)
			if l1 < best_l1 {
				best_l1 = l1
				best_level = k + 1
			}
		}
		// In half steps, so narrow bands can sit at the midpoint.
		metric[i] = transient ? 2 * best_level : -2 * best_level
		if narrow && (metric[i] == 0 || metric[i] == -2 * lm) do metric[i] -= 1
	}
	cost :: #force_inline proc(importance: ^[NB_BANDS]int, metric: ^[NB_BANDS]int, i, lm, row: int) -> int {
		return importance[i] * abs(metric[i] - 2 * int(TF_SELECT[lm][row]))
	}
	tf_select := 0
	selcost: [2]int
	for sel in 0 ..< 2 {
		row := 4 * t + 2 * sel
		cost0 := cost(importance, &metric, 0, lm, row)
		cost1 := cost(importance, &metric, 0, lm, row + 1) + (transient ? 0 : lambda)
		for i in 1 ..< length {
			curr0 := min(cost0, cost1 + lambda)
			curr1 := min(cost0 + lambda, cost1)
			cost0 = curr0 + cost(importance, &metric, i, lm, row)
			cost1 = curr1 + cost(importance, &metric, i, lm, row + 1)
		}
		selcost[sel] = min(cost0, cost1)
	}
	// tf_select only for transients, for now (as libopus).
	if selcost[1] < selcost[0] && transient do tf_select = 1
	row := 4 * t + 2 * tf_select
	cost0 := cost(importance, &metric, 0, lm, row)
	cost1 := cost(importance, &metric, 0, lm, row + 1) + (transient ? 0 : lambda)
	for i in 1 ..< length {
		from0, from1 := cost0, cost1 + lambda
		curr0 := from0 < from1 ? from0 : from1
		path0[i] = from0 < from1 ? 0 : 1
		from0, from1 = cost0 + lambda, cost1
		curr1 := from0 < from1 ? from0 : from1
		path1[i] = from0 < from1 ? 0 : 1
		cost0 = curr0 + cost(importance, &metric, i, lm, row)
		cost1 = curr1 + cost(importance, &metric, i, lm, row + 1)
	}
	tf_res[length - 1] = cost0 < cost1 ? 0 : 1
	for i := length - 2; i >= 0; i -= 1 do tf_res[i] = tf_res[i + 1] == 1 ? path1[i + 1] : path0[i + 1]
	return tf_select
}

// The spreading from how peaky the bands are, weighted by how audible
// (spread_weight), averaged and with hysteresis; updates the tapset too.
@(private)
spreading_decision :: proc(e: ^Celt_Encoder, end, m: int, spread_weight: ^[NB_BANDS]int, update_tapset: bool) -> Spread {
	if m * int(EBANDS[end] - EBANDS[end - 1]) <= 8 do return .None
	sum, nb_bands := 0, 0
	hf_sum := 0
	for ch in 0 ..< e.channels do for i in 0 ..< end {
		n := m * int(EBANDS[i + 1] - EBANDS[i])
		if n <= 8 do continue
		x := e.x[ch][m * int(EBANDS[i]):][:n]
		// A rough CDF of |x|.
		tcount: [3]int
		for v in x {
			x2n := v * v * f32(n)
			if x2n < 0.25 do tcount[0] += 1
			if x2n < 0.0625 do tcount[1] += 1
			if x2n < 0.015625 do tcount[2] += 1
		}
		// The last four bands (8 kHz and up) for the prefilter's tapset.
		if i > NB_BANDS - 4 do hf_sum += 32 * (tcount[1] + tcount[0]) / n
		tmp := int(2 * tcount[2] >= n) + int(2 * tcount[1] >= n) + int(2 * tcount[0] >= n)
		sum += tmp * spread_weight[i]
		nb_bands += spread_weight[i]
	}
	if update_tapset {
		// Peakier high bands take a wider tapset, with hysteresis.
		if hf_sum != 0 do hf_sum /= e.channels * (4 - NB_BANDS + end)
		e.hf_average = (e.hf_average + hf_sum) >> 1
		hf_sum = e.hf_average
		if e.tapset_decision == 2 {
			hf_sum += 4
		} else if e.tapset_decision == 0 {
			hf_sum -= 4
		}
		e.tapset_decision = hf_sum > 22 ? 2 : (hf_sum > 18 ? 1 : 0)
	}
	sum = (sum << 8) / nb_bands
	sum = (sum + e.tonal_average) >> 1
	e.tonal_average = sum
	sum = (3 * sum + (((3 - int(e.spread)) << 7) + 64) + 2) >> 2
	switch {
	case sum < 80:
		return .Aggressive
	case sum < 256:
		return .Normal
	case sum < 384:
		return .Light
	}
	return .None
}

@(private)
median_of_5 :: proc(x: []f32) -> f32 {
	t0, t1 := min(x[0], x[1]), max(x[0], x[1])
	t2 := x[2]
	t3, t4 := min(x[3], x[4]), max(x[3], x[4])
	if t0 > t3 {
		t0, t3 = t3, t0
		t1, t4 = t4, t1
	}
	if t2 > t1 {
		return t1 < t3 ? min(t2, t3) : min(t4, t1)
	}
	return t2 < t3 ? min(t1, t3) : min(t2, t4)
}

@(private)
median_of_3 :: proc(x: []f32) -> f32 {
	t0, t1 := min(x[0], x[1]), max(x[0], x[1])
	if t1 < x[2] do return t1
	if t0 < x[2] do return x[2]
	return t0
}

// Band boosts where a band stands out of its neighbours' envelope (in
// quanta, into offsets), how much each band matters for the resolution
// choice, and how audible it is for the spreading choice; returns the
// signal's depth over the noise floor.
@(private)
dynalloc_analysis :: proc(e: ^Celt_Encoder, log_e, log_e2: ^[2][NB_BANDS]f32, start, end, lm, effective_bytes: int, transient: bool, offsets: ^[NB_BANDS]int, tot_boost_: ^int, importance, spread_weight: ^[NB_BANDS]int, tone_freq, toneishness: f32, analysis: Analysis_Info) -> f32 {
	c := e.channels
	offsets^ = {}
	follower: [2][NB_BANDS]f32
	noise_floor: [NB_BANDS]f32
	max_depth: f32 = -31.9
	for i in 0 ..< end {
		// The band means, the input depth, the band width and the
		// pre-emphasis (about the square of the band index).
		noise_floor[i] = 0.0625 * f32(LOG_N[i]) + 0.5 + f32(9 - e.lsb_depth) - E_MEANS[i] + 0.0062 * f32((i + 5) * (i + 5))
	}
	for ch in 0 ..< c do for i in 0 ..< end do max_depth = max(max_depth, log_e[ch][i] - noise_floor[i])
	{
		// A simple masking model, so masked bands count less for spreading.
		mask, sig: [NB_BANDS]f32
		for i in 0 ..< end do mask[i] = log_e[0][i] - noise_floor[i]
		if c == 2 do for i in 0 ..< end do mask[i] = max(mask[i], log_e[1][i] - noise_floor[i])
		sig = mask
		for i in 1 ..< end do mask[i] = max(mask[i], mask[i - 1] - 2)
		for i := end - 2; i >= 0; i -= 1 do mask[i] = max(mask[i], mask[i + 1] - 3)
		for i in 0 ..< end {
			// The mask never more than 72 dB below the peak, nor below the floor.
			smr := sig[i] - max(max(0, max_depth - 12), mask[i])
			shift := min(5, max(0, -int(math.floor(0.5 + smr))))
			spread_weight[i] = 32 >> uint(shift)
		}
	}
	tot_boost := 0
	if effective_bytes >= 30 + 5 * lm {
		last := 0
		for ch in 0 ..< c {
			e3: [NB_BANDS]f32
			copy(e3[:end], log_e2[ch][:end])
			if lm == 0 {
				// One bin per band in the first 8 at 2.5 ms: too noisy alone.
				for i in 0 ..< min(8, end) do e3[i] = max(log_e2[ch][i], e.old_band_e[ch][i])
			}
			f := &follower[ch]
			f[0] = e3[0]
			for i in 1 ..< end {
				// The last band 3 dB above the one before is the last considered.
				if e3[i] > e3[i - 1] + 0.5 do last = i
				f[i] = min(f[i - 1] + 1.5, e3[i])
			}
			for i := last - 1; i >= 0; i -= 1 do f[i] = min(f[i], min(f[i + 1] + 2, e3[i]))
			// A median filter so that dynalloc does not trigger needlessly.
			OFFSET :: 1
			for i in 2 ..< end - 2 do f[i] = max(f[i], median_of_5(e3[i - 2:]) - OFFSET)
			tmp := median_of_3(e3[:]) - OFFSET
			f[0] = max(f[0], tmp)
			f[1] = max(f[1], tmp)
			tmp = median_of_3(e3[end - 3:]) - OFFSET
			f[end - 2] = max(f[end - 2], tmp)
			f[end - 1] = max(f[end - 1], tmp)
			for i in 0 ..< end do f[i] = max(f[i], noise_floor[i])
		}
		if c == 2 {
			for i in start ..< end {
				// 24 dB of cross-talk.
				follower[1][i] = max(follower[1][i], follower[0][i] - 4)
				follower[0][i] = max(follower[0][i], follower[1][i] - 4)
				follower[0][i] = 0.5 * (max(0, log_e[0][i] - follower[0][i]) + max(0, log_e[1][i] - follower[1][i]))
			}
		} else {
			for i in start ..< end do follower[0][i] = max(0, log_e[0][i] - follower[0][i])
		}
		fo := &follower[0]
		for i in start ..< end do importance[i] = int(math.floor(0.5 + 13 * math.exp(0.6931471805599453094 * f64(min(fo[i], 4)))))
		// Half for non-transient CBR or constrained VBR frames.
		if (!e.vbr || e.constrained_vbr) && !transient do for i in start ..< end do fo[i] *= 0.5
		for i in start ..< end {
			if i < 8 do fo[i] *= 2
			if i >= 12 do fo[i] *= 0.5
		}
		// Opus under-allocates tones.
		if toneishness > 0.98 {
			bin := int(math.floor(0.5 + f64(tone_freq) * 120 / math.PI))
			for i in start ..< end {
				lo, hi := int(EBANDS[i]), int(EBANDS[i + 1])
				if bin >= lo && bin <= hi do fo[i] += 2
				if bin >= lo - 1 && bin <= hi + 1 do fo[i] += 1
				if bin >= lo - 2 && bin <= hi + 2 do fo[i] += 1
				if bin >= lo - 3 && bin <= hi + 3 do fo[i] += 0.5
			}
			if bin >= int(EBANDS[end]) {
				fo[end - 1] += 2
				fo[end - 2] += 1
			}
		}
		if analysis.valid do for i in start ..< min(LEAK_BANDS, end) do fo[i] += f32(analysis.leak_boost[i]) / 64
		for i in start ..< end {
			fo[i] = min(fo[i], 4)
			width := c * int(EBANDS[i + 1] - EBANDS[i]) << uint(lm)
			boost, boost_bits: int
			if width < 6 {
				boost = int(fo[i])
				boost_bits = boost * width << BITRES
			} else if width > 48 {
				boost = int(fo[i] * 8)
				boost_bits = (boost * width << BITRES) / 8
			} else {
				boost = int(fo[i] * f32(width) / 6)
				boost_bits = boost * 6 << BITRES
			}
			// CBR and non-transient constrained VBR: at most 2/3 of the bits.
			if (!e.vbr || (e.constrained_vbr && !transient)) && (tot_boost + boost_bits) >> BITRES >> 3 > 2 * effective_bytes / 3 {
				cap := (2 * effective_bytes / 3) << BITRES << 3
				offsets[i] = cap - tot_boost
				tot_boost = cap
				break
			}
			offsets[i] = boost
			tot_boost += boost_bits
		}
	} else {
		for i in start ..< end do importance[i] = 13
	}
	tot_boost_^ = tot_boost
	return max_depth
}

// Whether dual stereo beats mid/side: the L1 norms of L/R against M/S over
// the first 13 bands, M/S charged its angles.
@(private)
stereo_analysis :: proc(e: ^Celt_Encoder, lm: int) -> bool {
	sum_lr, sum_ms: f32 = 1e-15, 1e-15
	for j in 0 ..< int(EBANDS[13]) << uint(lm) {
		l, r := e.x[0][j], e.x[1][j]
		sum_lr += abs(l) + abs(r)
		sum_ms += abs(l + r) + abs(l - r)
	}
	sum_ms *= 0.707107
	thetas := lm <= 1 ? 5 : 13
	bins := int(EBANDS[13]) << uint(lm + 1)
	return f32(bins + thetas) * sum_ms > f32(bins) * sum_lr
}

// The allocation trim (0 to 10, 5 neutral): toward low bands for
// correlated stereo, a tilted spectrum and transients.
@(private)
alloc_trim_analysis :: proc(e: ^Celt_Encoder, log_e: ^[2][NB_BANDS]f32, end, lm: int, tf_estimate: f32, equiv_rate: int, analysis: Analysis_Info) -> int {
	c := e.channels
	trim: f32 = 5
	// Lower trims help at low rates.
	if equiv_rate < 64000 {
		trim = 4
	} else if equiv_rate < 80000 {
		trim = 4 + f32((equiv_rate - 64000) >> 10) / 16
	}
	if c == 2 {
		// The inter-channel correlation of the low bands.
		band_dot :: proc(e: ^Celt_Encoder, i, lm: int) -> f32 {
			lo, hi := int(EBANDS[i]) << uint(lm), int(EBANDS[i + 1]) << uint(lm)
			return dot(e.x[0][lo:hi], e.x[1][lo:hi])
		}
		sum: f32
		for i in 0 ..< 8 do sum += band_dot(e, i, lm)
		sum = min(1, abs(sum / 8))
		min_xc := sum
		for i in 8 ..< e.intensity do min_xc = min(min_xc, abs(band_dot(e, i, lm)))
		min_xc = min(1, abs(min_xc))
		// Mid/side savings from the average and the least correlation.
		log_xc := f32(math.log2(f64(1.001 - sum * sum)))
		log_xc2 := max(0.5 * log_xc, f32(math.log2(f64(1.001 - min_xc * min_xc))))
		trim += max(-4, 0.75 * log_xc)
		e.stereo_saving = min(e.stereo_saving + 0.25, -0.5 * log_xc2)
	}
	// The spectral tilt.
	diff: f32
	for ch in 0 ..< c do for i in 0 ..< end - 1 do diff += log_e[ch][i] * f32(2 + 2 * i - end)
	diff /= f32(c * (end - 1))
	trim -= max(-2, min(2, (diff + 1) / 6))
	trim -= 2 * tf_estimate
	if analysis.valid do trim -= clamp(2 * (analysis.tonality_slope + 0.05), -2, 2)
	return clamp(int(math.floor(0.5 + trim)), 0, 10)
}

// The VBR target in 1/8 bits: the base less the stereo savings, plus the
// boosts and the transient's, capped by the signal's depth, and moved by
// the frame's loudness against the average.
@(private)
compute_vbr :: proc(e: ^Celt_Encoder, base_target, lm, bitrate, tot_boost: int, tf_estimate, max_depth, temporal_vbr: f32, analysis: Analysis_Info, pitch_change: bool) -> int {
	c := e.channels
	coded_bands := e.last_coded_bands != 0 ? e.last_coded_bands : NB_BANDS
	coded_bins := int(EBANDS[coded_bands]) << uint(lm)
	if c == 2 do coded_bins += int(EBANDS[min(e.intensity, coded_bands)]) << uint(lm)
	target := base_target
	// Less for inactive frames.
	if analysis.valid && analysis.activity < 0.4 do target -= int(f32(coded_bins << BITRES) * (0.4 - analysis.activity))
	if c == 2 {
		stereo_bands := min(e.intensity, coded_bands)
		stereo_dof := (int(EBANDS[stereo_bands]) << uint(lm)) - stereo_bands
		// At most what a mono signal would save.
		max_frac := 0.8 * f32(stereo_dof) / f32(coded_bins)
		saving := min(e.stereo_saving, 1)
		target -= int(min(max_frac * f32(target), (saving - 0.1) * f32(stereo_dof << BITRES)))
	}
	target += tot_boost - (19 << uint(lm))
	target += int((tf_estimate - 0.044) * f32(target))
	// More for tonal frames, about the average.
	if analysis.valid {
		target += int(f32(coded_bins << BITRES) * 1.2 * (max(0, analysis.tonality - 0.15) - 0.12))
		if pitch_change do target += int(f32(coded_bins << BITRES) * 0.8)
	}
	// No more than the signal's depth over the noise floor affords.
	bins := int(EBANDS[NB_BANDS - 2]) << uint(lm)
	floor_depth := max(int(f32(c * bins << BITRES) * max_depth), target >> 2)
	target = min(target, floor_depth)
	// Constrained VBR cannot hold a higher rate for long.
	if e.constrained_vbr do target = base_target + int(0.67 * f32(target - base_target))
	if tf_estimate < 0.2 {
		amount := 0.0000031 * f32(max(0, min(32000, 96000 - bitrate)))
		target += int(temporal_vbr * amount * f32(target))
	}
	return min(2 * base_target, target)
}

// The coarse energies, intra (in frequency only) or inter, whichever
// clamps fewer residuals, or is cheaper; the band energies come in as
// log_e and the decoder's state is old_band_e.
@(private)
quant_coarse_energy :: proc(e: ^Celt_Encoder, enc: ^Range_Encoder, start, end, eff_end: int, log_e, error: ^[2][NB_BANDS]f32, budget, lm, nb_available: int, two_pass_: bool) {
	c := e.channels
	two_pass := two_pass_
	intra := !two_pass && e.delayed_intra > f32(2 * c * (end - start)) && nb_available > (end - start) * c
	new_distortion: f32
	for ch in 0 ..< c do for i in start ..< eff_end {
		d := log_e[ch][i] - e.old_band_e[ch][i]
		new_distortion += d * d
	}
	new_distortion = min(200, new_distortion)
	tell := range_tell(enc)
	if tell + 3 > budget do two_pass, intra = false, false
	max_decay: f32 = 16
	if end - start > 10 do max_decay = min(max_decay, 0.125 * f32(nb_available))
	enc_start := enc^
	old_intra := e.old_band_e
	error_intra: [2][NB_BANDS]f32
	badness_intra := 0
	if two_pass || intra do badness_intra = quant_coarse_pass(enc, start, end, log_e, &old_intra, &error_intra, budget, c, lm, true, max_decay)
	if !intra {
		tell_intra := range_tell_frac(enc)
		enc_intra := enc^
		intra_bytes: [1275]u8
		written := enc.data[enc_start.offs:enc_intra.offs]
		copy(intra_bytes[:], written)
		enc^ = enc_start
		badness_inter := quant_coarse_pass(enc, start, end, log_e, &e.old_band_e, error, budget, c, lm, false, max_decay)
		if two_pass && (badness_intra < badness_inter || (badness_intra == badness_inter && range_tell_frac(enc) > tell_intra)) {
			enc^ = enc_intra
			copy(written, intra_bytes[:len(written)])
			e.old_band_e = old_intra
			error^ = error_intra
			intra = true
		}
	} else {
		e.old_band_e = old_intra
		error^ = error_intra
	}
	e.delayed_intra = intra ? new_distortion : ENERGY_PRED[lm] * ENERGY_PRED[lm] * e.delayed_intra + new_distortion
}

// One coarse energy pass; returns how far the coded residuals are from the
// wanted ones in total.
@(private)
quant_coarse_pass :: proc(enc: ^Range_Encoder, start, end: int, log_e, old_e, error: ^[2][NB_BANDS]f32, budget, c, lm: int, intra: bool, max_decay: f32) -> int {
	if range_tell(enc) + 3 <= budget do range_encode_bit_logp(enc, intra, 3)
	model := &E_PROB_MODEL[lm][intra ? 1 : 0]
	coef := intra ? 0 : ENERGY_PRED[lm]
	beta := intra ? ENERGY_BETA_INTRA : ENERGY_BETA[lm]
	prev: [2]f32
	badness := 0
	for i in start ..< end {
		for ch in 0 ..< c {
			x := log_e[ch][i]
			old := max(-9, old_e[ch][i])
			f := x - coef * old - prev[ch]
			qi := int(math.floor(0.5 + f))
			// The energy may not fall too fast (bands of one bin).
			decay_bound := max(-28, old_e[ch][i]) - max_decay
			if qi < 0 && x < decay_bound do qi = min(0, qi + int(decay_bound - x))
			qi0 := qi
			// Short of bits: something safe.
			tell := range_tell(enc)
			bits_left := budget - tell - 3 * c * (end - i)
			if i != start && bits_left < 30 {
				if bits_left < 24 do qi = min(1, qi)
				if bits_left < 16 do qi = max(-1, qi)
			}
			switch {
			case budget - tell >= 15:
				pi := 2 * min(i, 20)
				qi = range_encode_laplace(enc, qi, u32(model[pi]) << 7, u32(model[pi + 1]) << 6)
			case budget - tell >= 2:
				qi = clamp(qi, -1, 1)
				range_encode_icdf(enc, 2 * qi ~ -int(qi < 0), SMALL_ENERGY_ICDF[:], 2)
			case budget - tell >= 1:
				qi = min(0, qi)
				range_encode_bit_logp(enc, qi != 0, 1)
			case:
				qi = -1
			}
			error[ch][i] = f - f32(qi)
			badness += abs(qi0 - qi)
			q := f32(qi)
			old_e[ch][i] = coef * old + prev[ch] + q
			prev[ch] = prev[ch] + q - beta * q
		}
	}
	return badness
}

@(private)
quant_fine_energy :: proc(enc: ^Range_Encoder, old_e, error: ^[2][NB_BANDS]f32, fine_quant: ^[NB_BANDS]int, start, end, c: int) {
	for i in start ..< end {
		extra := fine_quant[i]
		if extra <= 0 || range_tell(enc) + c * extra > len(enc.data) * 8 do continue
		levels := 1 << uint(extra)
		for ch in 0 ..< c {
			q2 := clamp(int(math.floor((error[ch][i] + 0.5) * f32(levels))), 0, levels - 1)
			range_encode_bits(enc, u32(q2), uint(extra))
			offset := (f32(q2) + 0.5) * f32(i32(1) << uint(14 - extra)) * (1.0 / 16384) - 0.5
			old_e[ch][i] += offset
			error[ch][i] -= offset
		}
	}
}

// The bits left after the shapes: one more energy bit per band and
// channel, by priority.
@(private)
quant_energy_finalise :: proc(enc: ^Range_Encoder, old_e, error: ^[2][NB_BANDS]f32, fine_quant, fine_priority: ^[NB_BANDS]int, bits_left, start, end, c: int) {
	left := bits_left
	for prio in 0 ..< 2 {
		for i := start; i < end && left >= c; i += 1 {
			if fine_quant[i] >= MAX_FINE_BITS || fine_priority[i] != prio do continue
			for ch in 0 ..< c {
				q2 := error[ch][i] < 0 ? 0 : 1
				range_encode_bits(enc, u32(q2), 1)
				offset := (f32(q2) - 0.5) * f32(i32(1) << uint(14 - fine_quant[i] - 1)) * (1.0 / 16384)
				old_e[ch][i] += offset
				error[ch][i] -= offset
				left -= 1
			}
		}
	}
}

// The pitch prefilter (the inverse of the decoder's postfilter): the period
// and gain for this frame, the input filtered by it into e.input, and its
// quantised gain; off when it does not help. A single tone takes its own
// period, anything else the pitch search's.
@(private)
run_prefilter :: proc(e: ^Celt_Encoder, n: int, enabled: bool, tf_estimate: f32, nb_available: int, tone_freq, toneishness: f32, analysis: Analysis_Info) -> (pf: Comb, qg: int, on: bool) {
	MAX :: COMBFILTER_MAXPERIOD
	MIN :: COMBFILTER_MINPERIOD
	c := e.channels
	pitch := MIN
	gain: f32
	if enabled && toneishness > 0.99 {
		// Above 24 kHz the filter aliases; too high a pitch is doubled
		// until the filter takes it, too low a one is left to the
		// filter's DC response.
		freq := tone_freq
		if freq >= 3.1416 do freq = 3.141593 - freq
		multiple := 1
		for freq >= f32(multiple) * 0.39 do multiple += 1
		if freq > 0.006148 do pitch = min(int(math.floor(0.5 + 2 * math.PI * f64(multiple) / f64(freq))), MAX - 2)
		gain = 0.75
	} else if enabled && e.complexity >= 5 {
		lp: [(MAX + SHORT_MDCT << MAX_LM) >> 1]f32
		half := (MAX + n) >> 1
		pitch_downsample({e.history[0][:MAX + n], e.history[1][:MAX + n]}, c, lp[:half])
		// Not the shortest 1.5 octaves: short-term correlation fools it.
		pitch = MAX - pitch_search(lp[MAX >> 1:half], lp[:half], n, MAX - 3 * MIN)
		gain = remove_doubling(lp[:half], MAX, MIN, n, &pitch, e.prefilter.period, e.prefilter.gain)
		pitch = min(pitch, MAX - 2)
		gain *= 0.7
	}
	if analysis.valid do gain *= analysis.max_pitch_ratio
	// The gain it takes to switch on: more without continuity and at low
	// rates, less when it was on.
	threshold: f32 = 0.2
	if abs(pitch - e.prefilter.period) * 10 > pitch {
		threshold += 0.2
		if tf_estimate > 0.98 do gain = 0 // strong transients without continuity
	}
	if nb_available < 25 do threshold += 0.1
	if nb_available < 35 do threshold += 0.1
	if e.prefilter.gain > 0.4 do threshold -= 0.1
	if e.prefilter.gain > 0.55 do threshold -= 0.1
	on = gain >= max(threshold, 0.2)
	if on {
		if abs(gain - e.prefilter.gain) < 0.1 do gain = e.prefilter.gain
		qg = clamp(int(math.floor(0.5 + gain * 32 / 3)) - 1, 0, 7)
		gain = 0.09375 * f32(qg + 1)
	} else {
		gain = 0
	}
	pf = {pitch, gain, e.tapset_decision}
	e.prefilter.period = max(e.prefilter.period, MIN)
	from := Comb{e.prefilter.period, -e.prefilter.gain, e.prefilter.tapset}
	to := Comb{pitch, -gain, pf.tapset}
	before, after: [2]f32
	for ch in 0 ..< c {
		in_ := e.input[ch][:n + OVERLAP]
		copy(in_[:OVERLAP], e.in_mem[ch][:])
		for v in e.history[ch][MAX:][:n] do before[ch] += abs(v)
		comb_filter(in_, OVERLAP, e.history[ch][:], MAX, n, from, to, false)
		for v in in_[OVERLAP:] do after[ch] += abs(v)
	}
	// Not if a channel gets much worse, and only if one gets better.
	cancel := after[0] > before[0]
	if c == 2 {
		t0 := 0.25 * gain * before[0] + 0.01 * before[1]
		t1 := 0.25 * gain * before[1] + 0.01 * before[0]
		cancel = after[0] - before[0] > t0 || after[1] - before[1] > t1 || (before[0] - after[0] < t0 && before[1] - after[1] < t1)
	}
	if cancel {
		// The unfiltered input, the last filter faded out over the overlap.
		for ch in 0 ..< c {
			in_ := e.input[ch][:n + OVERLAP]
			copy(in_[OVERLAP:], e.history[ch][MAX:][:n])
			comb_filter(in_, OVERLAP, e.history[ch][:], MAX, OVERLAP, from, {pitch, 0, pf.tapset}, false)
		}
		pf.gain, qg, on = 0, 0, false
	}
	for ch in 0 ..< c do copy(e.in_mem[ch][:], e.input[ch][n:][:OVERLAP])
	return
}

@(private)
pitch_gain :: #force_inline proc "contextless" (xy, xx, yy: f32) -> f32 {
	return xy / f32(math.sqrt(f64(1 + xx * yy)))
}

// The pitch period t0 (in samples at full rate, from the search) moved to
// a submultiple where that correlates nearly as well, at half rate over x
// (maxperiod / 2 samples of past, then n / 2); returns its gain.
@(private)
remove_doubling :: proc(x: []f32, maxperiod_, minperiod_, n_: int, t0: ^int, prev_period_: int, prev_gain: f32) -> f32 {
	@(static, rodata)
	SECOND_CHECK := [16]int{0, 0, 3, 2, 3, 2, 5, 2, 3, 2, 3, 2, 5, 2, 3, 2}
	maxperiod, minperiod, n := maxperiod_ / 2, minperiod_ / 2, n_ / 2
	prev_period := prev_period_ / 2
	b := maxperiod // x[b - t] is t samples back
	frame := x[b:][:n]
	t0^ = min(t0^ / 2, maxperiod - 1)
	T0 := t0^
	T := T0
	yy_lookup: [COMBFILTER_MAXPERIOD / 2 + 1]f32
	xx := dot(frame, frame)
	xy := dot(frame, x[b - T0:][:n])
	yy_lookup[0] = xx
	yy := xx
	for i in 1 ..= maxperiod {
		yy += x[b - i] * x[b - i] - x[b + n - i] * x[b + n - i]
		yy_lookup[i] = max(0, yy)
	}
	yy = yy_lookup[T0]
	best_xy, best_yy := xy, yy
	g0 := pitch_gain(xy, xx, yy)
	g := g0
	// Any pitch at T0 / k, checked against a second multiple of it.
	for k in 2 ..= 15 {
		T1 := (2 * T0 + k) / (2 * k)
		if T1 < minperiod do break
		T1b := (2 * SECOND_CHECK[k] * T0 + k) / (2 * k)
		if k == 2 do T1b = T1 + T0 > maxperiod ? T0 : T0 + T1
		xy = 0.5 * (dot(frame, x[b - T1:][:n]) + dot(frame, x[b - T1b:][:n]))
		yy = 0.5 * (yy_lookup[T1] + yy_lookup[T1b])
		g1 := pitch_gain(xy, xx, yy)
		cont: f32
		if abs(T1 - prev_period) <= 1 {
			cont = prev_gain
		} else if abs(T1 - prev_period) <= 2 && 5 * k * k < T0 {
			cont = 0.5 * prev_gain
		}
		// Very short periods need more: short-term correlation fools it.
		thresh := T1 < 3 * minperiod ? max(0.4, 0.85 * g0 - cont) : max(0.3, 0.7 * g0 - cont)
		if g1 > thresh do best_xy, best_yy, T, g = xy, yy, T1, g1
	}
	best_xy = max(0, best_xy)
	pg: f32 = best_yy <= best_xy ? 1 : best_xy / (best_yy + 1)
	xcorr: [3]f32
	for k in 0 ..< 3 do xcorr[k] = dot(frame, x[b - (T + k - 1):][:n])
	offset := 0
	if xcorr[2] - xcorr[0] > 0.7 * (xcorr[1] - xcorr[0]) {
		offset = 1
	} else if xcorr[0] - xcorr[2] > 0.7 * (xcorr[1] - xcorr[2]) {
		offset = -1
	}
	t0^ = max(2 * T + offset, minperiod_)
	return min(pg, g)
}
