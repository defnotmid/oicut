package opus

// The range encoder (RFC 6716 5.1), the mirror of Range_Decoder: symbols
// from the front of the frame, raw bits from its back. range_tell and
// range_tell_frac count the same bits for both.

@(private)
CODE_SHIFT :: CODE_BITS - SYM_BITS - 1
@(private)
SYM_MAX :: (1 << SYM_BITS) - 1

Range_Encoder :: struct {
	data:        []u8, // the frame: its size is the budget
	offs:        int, // bytes written at the front
	end_offs:    int, // bytes written at the back
	end_window:  u32,
	nend_bits:   int,
	nbits_total: int,
	rng, val:    u32,
	ext:         int, // 0xFF bytes waiting on a carry
	rem:         int, // the byte waiting on a carry, -1 before the first
	error:       bool,
}

range_encoder_init :: proc(e: ^Range_Encoder, data: []u8) {
	e^ = {
		data        = data,
		nbits_total = CODE_BITS + 1,
		rng         = CODE_TOP,
		rem         = -1,
	}
}

@(private)
write_byte :: #force_inline proc(e: ^Range_Encoder, v: int) {
	if e.offs + e.end_offs >= len(e.data) {
		e.error = true
		return
	}
	e.data[e.offs] = u8(v)
	e.offs += 1
}

@(private)
write_byte_at_end :: #force_inline proc(e: ^Range_Encoder, v: u32) {
	if e.offs + e.end_offs >= len(e.data) {
		e.error = true
		return
	}
	e.end_offs += 1
	e.data[len(e.data) - e.end_offs] = u8(v)
}

// A byte that cannot take a carry any more is written; 0xFF bytes wait.
@(private)
carry_out :: proc(e: ^Range_Encoder, c: int) {
	if c == SYM_MAX {
		e.ext += 1
		return
	}
	carry := c >> SYM_BITS
	if e.rem >= 0 do write_byte(e, e.rem + carry)
	for ; e.ext > 0; e.ext -= 1 do write_byte(e, (SYM_MAX + carry) & SYM_MAX)
	e.rem = c & SYM_MAX
}

@(private)
enc_normalize :: #force_inline proc(e: ^Range_Encoder) {
	for e.rng <= CODE_BOT {
		carry_out(e, int(e.val >> CODE_SHIFT))
		e.val = (e.val << SYM_BITS) & (CODE_TOP - 1)
		e.rng <<= SYM_BITS
		e.nbits_total += SYM_BITS
	}
}

// The symbol [fl, fh) out of ft.
range_encode :: proc(e: ^Range_Encoder, fl, fh, ft: u32) {
	r := e.rng / ft
	if fl > 0 {
		e.val += e.rng - r * (ft - fl)
		e.rng = r * (fh - fl)
	} else {
		e.rng -= r * (ft - fh)
	}
	enc_normalize(e)
}

// As range_encode with ft = 1 << bits.
range_encode_bin :: proc(e: ^Range_Encoder, fl, fh: u32, bits: uint) {
	r := e.rng >> bits
	if fl > 0 {
		e.val += e.rng - r * ((u32(1) << bits) - fl)
		e.rng = r * (fh - fl)
	} else {
		e.rng -= r * ((u32(1) << bits) - fh)
	}
	enc_normalize(e)
}

range_encode_bit_logp :: proc(e: ^Range_Encoder, v: bool, logp: uint) {
	s := e.rng >> logp
	r := e.rng - s
	if v do e.val += r
	e.rng = v ? s : r
	enc_normalize(e)
}

range_encode_icdf :: proc(e: ^Range_Encoder, s: int, icdf: []u8, ftb: uint) {
	r := e.rng >> ftb
	if s > 0 {
		e.val += e.rng - r * u32(icdf[s - 1])
		e.rng = r * u32(icdf[s - 1] - icdf[s])
	} else {
		e.rng -= r * u32(icdf[s])
	}
	enc_normalize(e)
}

// A uniform value fl in 0 ..< total (total > 1): the top 8 bits range
// coded, the rest raw.
range_encode_uint :: proc(e: ^Range_Encoder, fl, total: u32) {
	ft := total - 1
	ftb := ilog(ft)
	if ftb > UINT_BITS {
		ftb -= UINT_BITS
		top := (ft >> uint(ftb)) + 1
		s := fl >> uint(ftb)
		range_encode(e, s, s + 1, top)
		range_encode_bits(e, fl & ((u32(1) << uint(ftb)) - 1), uint(ftb))
		return
	}
	range_encode(e, fl, fl + 1, ft + 1)
}

// Raw bits at the end of the frame, least significant first.
range_encode_bits :: proc(e: ^Range_Encoder, v: u32, bits: uint) {
	window := e.end_window
	used := e.nend_bits
	if used + int(bits) > 32 {
		for {
			write_byte_at_end(e, window & SYM_MAX)
			window >>= SYM_BITS
			used -= SYM_BITS
			if used < SYM_BITS do break
		}
	}
	window |= v << uint(used)
	used += int(bits)
	e.end_window = window
	e.nend_bits = used
	e.nbits_total += int(bits)
}

// The Laplace-distributed coarse energy residual (range_laplace's mirror);
// a value beyond the decaying part is clamped, and the coded one returned.
range_encode_laplace :: proc(e: ^Range_Encoder, value: int, p0: u32, decay: u32) -> int {
	MIN_P :: 1
	N_MIN :: 16
	val := value
	fs := p0
	fl: u32
	if val != 0 {
		s := val < 0 ? -1 : 0
		val = (val + s) ~ s
		fl = fs
		fs = (32768 - MIN_P * (2 * N_MIN) - fs) * (16384 - decay) >> 15
		i := 1
		for ; fs > 0 && i < val; i += 1 {
			fs *= 2
			fl += fs + 2 * MIN_P
			fs = fs * decay >> 15
		}
		if fs == 0 {
			ndi_max := (32768 - int(fl) + MIN_P - 1) // log2(MIN_P) = 0
			ndi_max = (ndi_max - s) >> 1
			di := min(val - i, ndi_max - 1)
			fl += u32((2 * di + 1 + s) * MIN_P)
			fs = min(MIN_P, 32768 - fl)
			val = (i + di + s) ~ s
		} else {
			fs += MIN_P
			if s == 0 do fl += fs
			val = value
		}
	}
	range_encode_bin(e, fl, fl + fs, 15)
	return val
}

// Moves the raw bits to the end of a smaller frame.
range_encoder_shrink :: proc(e: ^Range_Encoder, size: int) {
	assert(e.offs + e.end_offs <= size)
	copy(e.data[size - e.end_offs:size], e.data[len(e.data) - e.end_offs:])
	e.data = e.data[:size]
}

// Flushes the fewest bits that decode every symbol so far, whatever
// follows, then the raw bits; the bytes between are zero.
range_encoder_done :: proc(e: ^Range_Encoder) {
	l := CODE_BITS - ilog(e.rng)
	msk := (CODE_TOP - 1) >> uint(l)
	end := (e.val + msk) & ~msk
	if (end | msk) >= e.val + e.rng {
		l += 1
		msk >>= 1
		end = (e.val + msk) & ~msk
	}
	for l > 0 {
		carry_out(e, int(end >> CODE_SHIFT))
		end = (end << SYM_BITS) & (CODE_TOP - 1)
		l -= SYM_BITS
	}
	if e.rem >= 0 || e.ext > 0 do carry_out(e, 0)
	window := e.end_window
	used := e.nend_bits
	for used >= SYM_BITS {
		write_byte_at_end(e, window & SYM_MAX)
		window >>= SYM_BITS
		used -= SYM_BITS
	}
	if e.error do return
	for &b in e.data[e.offs:len(e.data) - e.end_offs] do b = 0
	if used > 0 {
		if e.end_offs >= len(e.data) {
			e.error = true
			return
		}
		l = -l
		// Busted: the range coder's bytes win over the extra raw bits.
		if e.offs + e.end_offs >= len(e.data) && l < used {
			window &= (1 << uint(l)) - 1
			e.error = true
		}
		e.data[len(e.data) - e.end_offs - 1] |= u8(window)
	}
}
