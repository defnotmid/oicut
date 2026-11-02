package opus

import "core:math"
import "core:simd"

// CELT loss concealment (RFC 6716 4.4): the first lost frames repeat the
// last pitch period of the excitation through an LPC fit of the signal,
// decaying as the signal did; longer losses, hybrid frames and isolated
// frames fade to noise shaped by the last band energies. The next decoded
// frame overlaps the concealment through a simulated TDAC fold.

@(private)
Frame_Kind :: enum u8 {
	None,
	Normal,
	Noise,
	Periodic,
}

@(private)
PLC_LPC_ORDER :: 24
@(private)
MAX_PERIOD :: 1024
@(private)
PLC_PITCH_LAG_MAX :: 720 // 66.7 Hz
@(private)
PLC_PITCH_LAG_MIN :: 100 // 480 Hz

@(private)
celt_conceal :: proc(c: ^Celt, n, lm: int) {
	cc := c.channels
	start, end := c.start, c.end
	kind := Frame_Kind.Periodic
	if c.plc_duration >= 40 || start != 0 || c.skip_plc do kind = .Noise
	if kind == .Noise {
		for k in 0 ..< cc do copy(c.history[k][:DECODE_BUFFER - n + OVERLAP], c.history[k][n:])
		if c.fold do prefilter_and_fold(c, n)
		// The energies decay toward the background, under noise.
		decay: f32 = c.loss_duration == 0 ? 1.5 : 0.5
		for k in 0 ..< cc do for i in start ..< end do c.band_e[k][i] = max(c.background_e[k][i], c.band_e[k][i] - decay)
		eff_end := max(start, min(end, NB_BANDS))
		seed := c.rng
		for k in 0 ..< cc {
			for i in start ..< eff_end {
				band := c.x[k][int(EBANDS[i]) << uint(lm):int(EBANDS[i + 1]) << uint(lm)]
				for &v in band {
					seed = lcg_rand(seed)
					v = f32(i32(seed) >> 20)
				}
				renormalise(band, 1)
			}
		}
		c.rng = seed
		synthesis(c, start, eff_end, cc, cc, false, lm, false)
		// The postfilter continues unchanged.
		for k in 0 ..< cc {
			h := c.history[k][:]
			at := DECODE_BUFFER - n
			c.postfilter.period = max(c.postfilter.period, COMBFILTER_MINPERIOD)
			c.postfilter_old.period = max(c.postfilter_old.period, COMBFILTER_MINPERIOD)
			comb_filter(h, at, h, at, SHORT_MDCT, c.postfilter_old, c.postfilter, true)
			if lm != 0 do comb_filter(h, at + SHORT_MDCT, h, at + SHORT_MDCT, n - SHORT_MDCT, c.postfilter, c.postfilter, true)
		}
		c.postfilter_old = c.postfilter
		c.fold = false
		c.skip_plc = true // until two frames in a row decode
	} else {
		conceal_periodic(c, n)
		c.fold = true
	}
	c.loss_duration = min(10000, c.loss_duration + 1 << uint(lm))
	c.plc_duration = min(10000, c.plc_duration + 1 << uint(lm))
	c.last_frame = kind
}

// The last pitch period of the excitation, repeated and decaying, through
// the LPC synthesis filter: n samples plus the next frame's overlap.
@(private)
conceal_periodic :: proc(c: ^Celt, n: int) {
	first := c.last_frame != .Periodic
	fade: f32 = 1
	if first {
		c.last_pitch = plc_pitch_search(c)
	} else {
		fade = 0.8
	}
	pitch := c.last_pitch
	exc_length := min(2 * pitch, MAX_PERIOD)
	ext_length := n + OVERLAP
	at := DECODE_BUFFER - n
	for k in 0 ..< c.channels {
		buf := c.history[k][:]
		lpc := &c.lpc[k]
		// The last MAX_PERIOD samples, the filter's history ahead of them.
		exc_buf: [PLC_LPC_ORDER + MAX_PERIOD]f32
		copy(exc_buf[:], buf[DECODE_BUFFER - MAX_PERIOD - PLC_LPC_ORDER:DECODE_BUFFER])
		exc := exc_buf[PLC_LPC_ORDER:]
		if first {
			ac: [PLC_LPC_ORDER + 1]f32
			autocorr(exc, ac[:], WINDOW[:])
			ac[0] *= 1.0001 // a -40 dB noise floor
			// Lag windowing, for a stable recursion.
			w: f32 = 0.008
			w2 := w * w
			for i in 1 ..= PLC_LPC_ORDER do ac[i] -= ac[i] * w2 * f32(i) * f32(i)
			lpc_from_autocorr(lpc[:], ac[:])
		}
		// The excitation of the last exc_length samples (the FIR inverse).
		{
			from := MAX_PERIOD - exc_length
			tmp: [MAX_PERIOD]f32
			src := exc_buf[from:][:PLC_LPC_ORDER + exc_length]
			for i in 0 ..< exc_length {
				sum := src[PLC_LPC_ORDER + i]
				for j in 0 ..< PLC_LPC_ORDER do sum += lpc[PLC_LPC_ORDER - 1 - j] * src[i + j]
				tmp[i] = sum
			}
			copy(exc[from:][:exc_length], tmp[:exc_length])
		}
		// How fast the waveform was decaying, not to add energy.
		decay: f32
		{
			e1, e2: f32 = 1, 1
			half := exc_length >> 1
			for i in 0 ..< half {
				a := exc[MAX_PERIOD - half + i]
				b := exc[MAX_PERIOD - 2 * half + i]
				e1 += a * a
				e2 += b * b
			}
			decay = f32(math.sqrt(f64(min(e1, e2) / e2)))
		}
		copy(buf[:at], buf[n:DECODE_BUFFER])
		// Extrapolate, each period scaled down by the decay again, keeping
		// the energy of the signal whose excitation is being copied.
		offset := MAX_PERIOD - pitch
		attenuation := fade * decay
		s1: f32
		for i, j := 0, 0; i < ext_length; i, j = i + 1, j + 1 {
			if j >= pitch {
				j -= pitch
				attenuation *= decay
			}
			buf[at + i] = attenuation * exc[offset + j]
			t := buf[at - MAX_PERIOD + offset + j]
			s1 += t * t
		}
		// Back through the synthesis filter, continuing the decoded signal.
		{
			mem: [PLC_LPC_ORDER]f32
			for i in 0 ..< PLC_LPC_ORDER do mem[i] = buf[at - 1 - i]
			y := buf[at:][:ext_length]
			for i in 0 ..< ext_length {
				sum := y[i]
				for j in 0 ..< PLC_LPC_ORDER {
					past := i - 1 - (PLC_LPC_ORDER - 1 - j)
					sum -= lpc[PLC_LPC_ORDER - 1 - j] * (past >= 0 ? y[past] : mem[-past - 1])
				}
				y[i] = sum
			}
		}
		// A synthesis louder than the signal it copies (the filter can
		// ring with a changing signal) is attenuated; one exploding, or
		// not a number, is dropped.
		s2: f32
		for v in buf[at:][:ext_length] do s2 += v * v
		if !(s1 > 0.2 * s2) {
			for &v in buf[at:][:ext_length] do v = 0
		} else if s1 < s2 {
			ratio := f32(math.sqrt(f64((s1 + 1) / (s2 + 1))))
			for i in 0 ..< OVERLAP do buf[at + i] *= 1 - WINDOW[i] * (1 - ratio)
			for i in OVERLAP ..< ext_length do buf[at + i] *= ratio
		}
	}
}

// The concealment's overlap, pre-filtered (the postfilter runs again over
// it) and folded as an MDCT would, to overlap-add with the next frame.
@(private)
prefilter_and_fold :: proc(c: ^Celt, n: int) {
	at := DECODE_BUFFER - n
	// The encoder's prefilter: the postfilter with its gain negated.
	pre := c.postfilter
	pre.gain = -pre.gain
	for k in 0 ..< c.channels {
		x := c.history[k][:]
		e: [OVERLAP]f32
		comb_filter(e[:], 0, x, at, OVERLAP, pre, pre, false)
		for i in 0 ..< OVERLAP / 2 do x[at + i] = WINDOW[i] * e[OVERLAP - 1 - i] + WINDOW[OVERLAP - 1 - i] * e[i]
	}
}

// The pitch period of the history, between PLC_PITCH_LAG_MIN and _MAX.
@(private)
plc_pitch_search :: proc(c: ^Celt) -> int {
	lp: [DECODE_BUFFER / 2]f32
	pitch_downsample({c.history[0][:], c.history[1][:]}, c.channels, lp[:])
	return PLC_PITCH_LAG_MAX - pitch_search(lp[PLC_PITCH_LAG_MAX / 2:], lp[:], DECODE_BUFFER - PLC_PITCH_LAG_MAX, PLC_PITCH_LAG_MAX - PLC_PITCH_LAG_MIN)
}

// x at half rate (channels summed) into lp, whitened by a 4th-order LPC
// with a zero added.
@(private)
pitch_downsample :: proc(x_: [2][]f32, channels: int, lp: []f32) {
	for k in 0 ..< channels {
		x := x_[k]
		v := 0.25 * x[1] + 0.5 * x[0]
		lp[0] = k == 0 ? v : lp[0] + v
		for i in 1 ..< len(lp) {
			v = 0.25 * x[2 * i - 1] + 0.25 * x[2 * i + 1] + 0.5 * x[2 * i]
			lp[i] = k == 0 ? v : lp[i] + v
		}
	}
	ac: [5]f32
	autocorr(lp, ac[:], nil)
	ac[0] *= 1.0001
	for i in 1 ..= 4 do ac[i] -= ac[i] * (0.008 * f32(i)) * (0.008 * f32(i))
	lpc: [4]f32
	lpc_from_autocorr(lpc[:], ac[:])
	g: f32 = 1
	for &a in lpc {
		g *= 0.9
		a *= g
	}
	C1 :: f32(0.8)
	num := [5]f32{lpc[0] + 0.8, lpc[1] + C1 * lpc[0], lpc[2] + C1 * lpc[1], lpc[3] + C1 * lpc[2], C1 * lpc[3]}
	// The 5-tap FIR in place.
	mem: [5]f32
	for &v in lp {
		sum := v + num[0] * mem[0] + num[1] * mem[1] + num[2] * mem[2] + num[3] * mem[3] + num[4] * mem[4]
		mem = {v, mem[0], mem[1], mem[2], mem[3]}
		v = sum
	}
}

// The lag (into y) best correlating with x over len samples: a search at a
// quarter rate, refined at half rate around the two best candidates, then
// by pseudo-interpolation.
@(private)
pitch_search :: proc(x, y: []f32, length, max_pitch: int) -> int {
	// For concealment's and the prefilter's lengths and lags.
	x4: [(DECODE_BUFFER - PLC_PITCH_LAG_MAX) >> 2]f32
	y4: [DECODE_BUFFER >> 2]f32
	xcorr: [COMBFILTER_MAXPERIOD >> 1]f32
	lag := length + max_pitch
	for j in 0 ..< length >> 2 do x4[j] = x[2 * j]
	for j in 0 ..< lag >> 2 do y4[j] = y[2 * j]
	pitch_xcorr(x4[:length >> 2], y4[:], xcorr[:max_pitch >> 2])
	best := find_best_pitch(xcorr[:max_pitch >> 2], y4[:], length >> 2)
	for i in 0 ..< max_pitch >> 1 {
		xcorr[i] = 0
		if abs(i - 2 * best[0]) > 2 && abs(i - 2 * best[1]) > 2 do continue
		xcorr[i] = max(-1, dot(x[:length >> 1], y[i:][:length >> 1]))
	}
	best = find_best_pitch(xcorr[:max_pitch >> 1], y, length >> 1)
	offset := 0
	if best[0] > 0 && best[0] < (max_pitch >> 1) - 1 {
		a, b, c := xcorr[best[0] - 1], xcorr[best[0]], xcorr[best[0] + 1]
		if c - a > 0.7 * (b - a) {
			offset = 1
		} else if a - c > 0.7 * (b - c) {
			offset = -1
		}
	}
	return 2 * best[0] - offset
}

// xcorr[i] = x . y[i:][:len(x)] for every lag i: lanes take consecutive
// lags, each summing in dot's order, so the results are dot's.
@(private)
pitch_xcorr :: proc(x, y: []f32, xcorr: []f32) {
	L :: 8
	V :: #simd[L]f32
	i := 0
	for ; i + L <= len(xcorr); i += L {
		sum: V
		#no_bounds_check for v, j in x do sum += V(v) * simd.from_slice(V, y[i + j:])
		lanes := simd.to_array(sum)
		copy(xcorr[i:], lanes[:])
	}
	for ; i < len(xcorr); i += 1 do xcorr[i] = dot(x, y[i:][:len(x)])
}

// The two lags of highest normalised correlation xcorr[i]^2 / |y[i:][:len]|^2.
@(private)
find_best_pitch :: proc(xcorr: []f32, y: []f32, length: int) -> (best: [2]int) {
	syy: f32 = 1
	best_num := [2]f32{-1, -1}
	best_den: [2]f32
	best = {0, 1}
	for v in y[:length] do syy += v * v
	for xc, i in xcorr {
		if xc > 0 {
			x16 := xc * 1e-12 // keeps the square finite
			num := x16 * x16
			if num * best_den[1] > best_num[1] * syy {
				if num * best_den[0] > best_num[0] * syy {
					best_num[1], best_den[1], best[1] = best_num[0], best_den[0], best[0]
					best_num[0], best_den[0], best[0] = num, syy, i
				} else {
					best_num[1], best_den[1], best[1] = num, syy, i
				}
			}
		}
		syy += y[i + length] * y[i + length] - y[i] * y[i]
		syy = max(1, syy)
	}
	return
}

// Eight running sums, then their total: the products need not wait on
// one another.
@(private)
dot :: proc(x, y: []f32) -> f32 {
	V :: #simd[8]f32
	sums: V
	i := 0
	#no_bounds_check for ; i + 8 <= len(x); i += 8 do sums += simd.from_slice(V, x[i:]) * simd.from_slice(V, y[i:])
	sum := simd.reduce_add_ordered(sums)
	#no_bounds_check for ; i < len(x); i += 1 do sum += x[i] * y[i]
	return sum
}

// The largest magnitude of x.
@(private)
max_abs :: proc(x: []f32) -> f32 {
	V :: #simd[8]f32
	peak: V
	i := 0
	#no_bounds_check for ; i + 8 <= len(x); i += 8 do peak = simd.max(peak, simd.abs(simd.from_slice(V, x[i:])))
	m := simd.reduce_max(peak)
	#no_bounds_check for ; i < len(x); i += 1 do m = max(m, abs(x[i]))
	return m
}

// ac[k] = sum of x[i] x[i + k], x tapered by window at both ends.
@(private)
autocorr :: proc(x: []f32, ac: []f32, window: []f32) {
	xx: [MAX_PERIOD]f32
	n := len(x)
	copy(xx[:n], x)
	for w, i in window {
		xx[i] = x[i] * w
		xx[n - 1 - i] = x[n - 1 - i] * w
	}
	lag := len(ac) - 1
	fast := n - lag
	for k in 0 ..= lag {
		head := dot(xx[:fast], xx[k:][:fast])
		tail: f32
		for i in k + fast ..< n do tail += xx[i] * xx[i - k]
		ac[k] = head + tail
	}
}

// Levinson-Durbin: the prediction coefficients of an autocorrelation,
// stopping at 30 dB of prediction gain.
@(private)
lpc_from_autocorr :: proc(lpc: []f32, ac: []f32) {
	for &v in lpc do v = 0
	err := ac[0]
	if ac[0] <= 1e-10 do return
	for i in 0 ..< len(lpc) {
		rr: f32
		for j in 0 ..< i do rr += lpc[j] * ac[i - j]
		rr += ac[i + 1]
		r := -rr / err
		lpc[i] = r
		for j in 0 ..< (i + 1) >> 1 {
			t1, t2 := lpc[j], lpc[i - 1 - j]
			lpc[j] = t1 + r * t2
			lpc[i - 1 - j] = t2 + r * t1
		}
		err = err - r * r * err
		if err <= 0.001 * ac[0] do break
	}
}

// After a loss, the coded energy prediction starts from a safe level: the
// trend continued if it was falling, else the least of the last frames,
// lower still for short frames.
@(private)
safe_energy :: proc(c: ^Celt, start, end, lm: int) {
	missing := min(10, c.loss_duration >> uint(lm))
	safety: f32 = lm == 0 ? 1.5 : lm == 1 ? 0.5 : 0
	for k in 0 ..< 2 do for i in start ..< end {
		e0, e1, e2 := c.band_e[k][i], c.log_e[k][i], c.log_e2[k][i]
		if e0 < max(e1, e2) {
			slope := min(max(e1 - e0, 0.5 * (e2 - e0)), 2)
			c.band_e[k][i] = max(-20, e0 - max(0, f32(1 + missing) * slope))
		} else {
			c.band_e[k][i] = min(e0, e1, e2)
		}
		c.band_e[k][i] -= safety
	}
}
