package av1

// Multi-symbol Q15 arithmetic coder (section 8.2) and its exact encoder
// inverse, plus 1/256-bit symbol rate cost lookup for RDO.

import "base:intrinsics"
import "core:simd"

@(private, rodata)
EC_MIN_PROB_OFFSET := [24]u32{
	60, 56, 52, 48, 44, 40, 36, 32, 28, 24, 20, 16, 12, 8, 4, 0,
	0, 0, 0, 0, 0, 0, 0, 0,
}

Symbol_Decoder :: struct {
	data:               []u8,
	byte_pos:           int, // next byte index in data (or virtual padded index) to load into dif
	dif:                u64, // top 16 bits = SymbolValue; lower 48 bits = inverted lookahead bits
	rng:                u32, // SymbolRange (32768 ..= 65535)
	cnt:                i32, // valid lookahead bits minus 15 (refill when cnt < 0); i32 packs tightly with rng: u32
	disable_cdf_update: bool,
}

// 8.2.2: initializes the symbol decoder on a byte-aligned tile slice.
init_symbol :: proc "contextless" (d: ^Symbol_Decoder, data: []u8, disable_cdf_update := false) #no_bounds_check {
	d.data = data
	d.byte_pos = 7
	d.rng = 1 << 15
	d.cnt = 41
	d.disable_cdf_update = disable_cdf_update
	v := len(data) >= 8 ? ~u64(intrinsics.unaligned_load((^u64be)(raw_data(data)))) : load_tail_u64(data, 0)
	d.dif = v >> 1
}

@(private)
load_tail_u64 :: #force_no_inline proc "contextless" (data: []u8, byte_pos: int) -> u64 #no_bounds_check {
	v: u64 = 0
	for i in 0 ..< 8 {
		b: u8 = 0xFF
		if byte_pos + i < len(data) {
			b = data[byte_pos + i] ~ 0xFF
		}
		v = (v << 8) | u64(b)
	}
	return v
}

// Call-free leaf renormalization: since `init_symbol` sets `d.byte_pos = 7` and `d.byte_pos`
// only increases, `d.byte_pos < len(d.data)` guarantees `len(d.data) >= 8`. Any partial tail
// read at the end of a tile is therefore an in-bounds 8-byte load from `&d.data[len(d.data) - 8]`
// shifted left to zero-pad trailing bytes before inversion. Keeping `renorm_symbol` 100% call-free
// prevents LLVM from spilling caller-saved SIMD/mask registers across `read_symbol_*` loops.
@(private)
renorm_symbol :: #force_inline proc "contextless" (d: ^Symbol_Decoder) #no_bounds_check {
	bits := (intrinsics.count_leading_zeros(d.rng) - 16) & 15
	d.rng <<= bits
	d.dif <<= u64(bits)
	d.cnt -= i32(bits)
	if d.cnt < 0 {
		v := ~u64(0)
		if d.byte_pos + 8 <= len(d.data) {
			v = ~u64(intrinsics.unaligned_load((^u64be)(&d.data[d.byte_pos])))
		} else if d.byte_pos < len(d.data) {
			rem := len(d.data) - d.byte_pos
			v = ~(u64(intrinsics.unaligned_load((^u64be)(&d.data[len(d.data) - 8]))) << (uint(8 - rem) << 3))
		}
		shift := uint(16 + d.cnt) & 63
		d.dif |= v >> shift
		num_bytes := (48 - int(d.cnt)) >> 3
		d.byte_pos += num_bytes
		d.cnt += i32(num_bytes << 3)
	}
}

read_symbol_bool :: #force_inline proc "contextless" (d: ^Symbol_Decoder, cdf: []u16) -> int #no_bounds_check {
	c0 := cdf[0]
	cur := (((d.rng >> 8) * (u32(32768 - c0) >> EC_PROB_SHIFT)) >> (7 - EC_PROB_SHIFT)) + EC_MIN_PROB
	vw := u64(cur) << 48
	ret := u32(d.dif >= vw)
	d.dif -= u64(ret) * vw
	d.rng = cur + ret * (d.rng - 2 * cur)
	renorm_symbol(d)
	symbol := int(ret ~ 1)
	if !d.disable_cdf_update {
		count := cdf[2]
		rate := uint(4 + (count >> 4)) & 7
		cdf[0] = ret != 0 ? (c0 + ((32768 - c0) >> rate)) : (c0 - (c0 >> rate))
		cdf[2] = count + u16(count < 32)
	}
	return symbol
}

read_symbol_3 :: #force_inline proc "contextless" (d: ^Symbol_Decoder, cdf: [^]u16) -> int #no_bounds_check {
	val := u32(d.dif >> 48)
	r := d.rng >> 8
	u0 := cdf[0]
	cur0 := ((r * (u32(32768 - u0) >> EC_PROB_SHIFT)) >> (7 - EC_PROB_SHIFT)) + u32(2 * EC_MIN_PROB)
	if val >= cur0 {
		d.rng -= cur0
		d.dif -= u64(cur0) << 48
		renorm_symbol(d)
		if !d.disable_cdf_update {
			count := cdf[3]
			rate := uint(4 + (count >> 4)) & 7
			u1 := cdf[1]
			cdf[0] = u0 + ((32768 - u0) >> rate)
			cdf[1] = u1 + ((32768 - u1) >> rate)
			cdf[3] = count + u16(count < 32)
		}
		return 0
	}
	u1 := cdf[1]
	cur1 := ((r * (u32(32768 - u1) >> EC_PROB_SHIFT)) >> (7 - EC_PROB_SHIFT)) + u32(1 * EC_MIN_PROB)
	lt1 := val < cur1
	symbol := 1 + int(lt1)
	d.rng = lt1 ? cur1 : (cur0 - cur1)
	d.dif -= lt1 ? 0 : (u64(cur1) << 48)
	renorm_symbol(d)
	if !d.disable_cdf_update {
		count := cdf[3]
		rate := uint(4 + (count >> 4)) & 7
		cdf[0] = u0 - (u0 >> rate)
		cdf[1] = lt1 ? (u1 - (u1 >> rate)) : (u1 + ((32768 - u1) >> rate))
		cdf[3] = count + u16(count < 32)
	}
	return symbol
}

read_symbol_4 :: #force_inline proc "contextless" (d: ^Symbol_Decoder, cdf: [^]u16) -> int #no_bounds_check {
	val := u32(d.dif >> 48)
	r := d.rng >> 8
	u0 := cdf[0]
	cur0 := ((r * (u32(32768 - u0) >> EC_PROB_SHIFT)) >> (7 - EC_PROB_SHIFT)) + u32(3 * EC_MIN_PROB)
	if val >= cur0 {
		d.rng -= cur0
		d.dif -= u64(cur0) << 48
		renorm_symbol(d)
		if !d.disable_cdf_update {
			// Update `cdf[0]` (2B) and `cdf[1..2]` (4B) with the exact same memory access widths
			// as the `symbol > 0` branch below. Mixing an 8-byte `#simd[4]u16` store at `cdf + 0`
			// here with 2B (`cdf + 0`) + 4B (`cdf + 2`) stores/loads in `symbol > 0` would trigger
			// ~20-cycle Store-to-Load Forwarding (STLF) stalls when consecutive coefficients share `ctx`.
			count := cdf[4]
			rate := uint(5 + (count >> 4)) & 7
			u1 := cdf[1]
			u2 := cdf[2]
			cdf[0] = u0 + ((32768 - u0) >> rate)
			cdf[1] = u1 + ((32768 - u1) >> rate)
			cdf[2] = u2 + ((32768 - u2) >> rate)
			cdf[4] = count + u16(count < 32)
		}
		return 0
	}
	u1 := cdf[1]
	u2 := cdf[2]
	cur1 := ((r * (u32(32768 - u1) >> EC_PROB_SHIFT)) >> (7 - EC_PROB_SHIFT)) + u32(2 * EC_MIN_PROB)
	cur2 := ((r * (u32(32768 - u2) >> EC_PROB_SHIFT)) >> (7 - EC_PROB_SHIFT)) + u32(1 * EC_MIN_PROB)
	lt1 := val < cur1
	lt2 := val < cur2
	symbol := 1 + int(lt1) + int(lt2)
	prev := lt2 ? cur2 : (lt1 ? cur1 : cur0)
	cur := lt2 ? 0 : (lt1 ? cur2 : cur1)
	d.rng = prev - cur
	d.dif -= u64(cur) << 48
	renorm_symbol(d)
	if !d.disable_cdf_update {
		count := cdf[4]
		rate := uint(5 + (count >> 4)) & 7
		cdf[0] = u0 - (u0 >> rate)
		cdf[1] = lt1 ? (u1 - (u1 >> rate)) : (u1 + ((32768 - u1) >> rate))
		cdf[2] = lt2 ? (u2 - (u2 >> rate)) : (u2 + ((32768 - u2) >> rate))
		cdf[4] = count + u16(count < 32)
	}
	return symbol
}

// Decodes up to 4 base-range (`coeff_br`) symbols from a 4-symbol CDF (`cdf[0..4]`)
// and returns the total level (`3 + sum(coeff_br)`).
read_coeff_br :: #force_no_inline proc "contextless" (d: ^Symbol_Decoder, cdf: [^]u16) -> int #no_bounds_check {
	level := NUM_BASE_LEVELS + 1
	u0 := cdf[0]
	u1 := cdf[1]
	u2 := cdf[2]
	count := cdf[4]
	upd := !d.disable_cdf_update
	for _ in 0 ..< 4 {
		val := u32(d.dif >> 48)
		r := d.rng >> 8
		cur0 := ((r * (u32(32768 - u0) >> EC_PROB_SHIFT)) >> (7 - EC_PROB_SHIFT)) + u32(3 * EC_MIN_PROB)
		if val >= cur0 {
			d.rng -= cur0
			d.dif -= u64(cur0) << 48
			renorm_symbol(d)
			if upd {
				rate := uint(5 + (count >> 4)) & 7
				u0 += (32768 - u0) >> rate
				u1 += (32768 - u1) >> rate
				u2 += (32768 - u2) >> rate
				count += u16(count < 32)
			}
			break
		}
		cur1 := ((r * (u32(32768 - u1) >> EC_PROB_SHIFT)) >> (7 - EC_PROB_SHIFT)) + u32(2 * EC_MIN_PROB)
		cur2 := ((r * (u32(32768 - u2) >> EC_PROB_SHIFT)) >> (7 - EC_PROB_SHIFT)) + u32(1 * EC_MIN_PROB)
		lt1 := val < cur1
		lt2 := val < cur2
		coeff_br := 1 + int(lt1) + int(lt2)
		prev := lt2 ? cur2 : (lt1 ? cur1 : cur0)
		cur := lt2 ? 0 : (lt1 ? cur2 : cur1)
		d.rng = prev - cur
		d.dif -= u64(cur) << 48
		renorm_symbol(d)
		if upd {
			rate := uint(5 + (count >> 4)) & 7
			u0 -= u0 >> rate
			u1 = lt1 ? (u1 - (u1 >> rate)) : (u1 + ((32768 - u1) >> rate))
			u2 = lt2 ? (u2 - (u2 >> rate)) : (u2 + ((32768 - u2) >> rate))
			count += u16(count < 32)
		}
		level += coeff_br
		if !lt2 {
			break
		}
	}
	if upd {
		cdf[0] = u0
		cdf[1] = u1
		cdf[2] = u2
		cdf[4] = count
	}
	return level
}

@(private)
read_symbol_multi :: proc "contextless" (d: ^Symbol_Decoder, cdf: [^]u16, n: int) -> int #no_bounds_check {
	val := u32(d.dif >> 48)

	if n <= 9 {
		v_val := (#simd[8]u32)(val)
		v_r := (#simd[8]u32)(d.rng >> 8)
		zero := (#simd[8]u32)(0)
		v_32768_16 := (#simd[8]u16)(32768)

		c_v := intrinsics.unaligned_load((^#simd[8]u16)(cdf))
		f_prob := (#simd[8]u32)(simd.shr(v_32768_16 - c_v, (#simd[8]u16)(EC_PROB_SHIFT)))
		v_min := intrinsics.unaligned_load((^#simd[8]u32)(&EC_MIN_PROB_OFFSET[16 - n]))
		active := simd.lanes_gt(v_min, zero)
		v_cur_raw := simd.shr(v_r * f_prob, (#simd[8]u32)(7 - EC_PROB_SHIFT)) + v_min
		v_cur := simd.select(active, v_cur_raw, zero)
		gt_32 := simd.lanes_gt(v_cur, v_val)
		symbol := int(-simd.reduce_add_ordered(transmute(#simd[8]i32)gt_32))

		cur_arr: [10]u32 = ---
		cur_arr[0] = d.rng
		intrinsics.unaligned_store((^#simd[8]u32)(&cur_arr[1]), v_cur)
		cur_arr[9] = 0
		cur := cur_arr[symbol + 1]
		d.rng = cur_arr[symbol] - cur
		d.dif -= u64(cur) << 48
		renorm_symbol(d)

		if !d.disable_cdf_update {
			// Preserve `c_v` on inactive lanes (`>= n - 1`) and store `v_out` directly to `cdf`.
			// This avoids bouncing through a temporary stack array `out_arr` with overlapping tail
			// loads that would otherwise stall on Store-to-Load Forwarding (STLF).
			count := cdf[n]
			v_rate := (#simd[8]u16)((5 + (count >> 4)) & 7)
			dec := c_v - simd.shr(c_v, v_rate)
			inc := c_v + simd.shr(v_32768_16 - c_v, v_rate)
			v_upd := simd.select((#simd[8]u16)(gt_32), dec, inc)
			v_out := simd.select((#simd[8]u16)(active), v_upd, c_v)
			intrinsics.unaligned_store((^#simd[8]u16)(cdf), v_out)
			cdf[n] = count + u16(count < 32)
		}
		return symbol
	}

	// Single 16-lane vector path for `n in 10 ..= 16`:
	// 1. Using a single `#simd[16]u32` comparison (`gt_32`) lets LLVM lower
	//    `-simd.reduce_add_ordered(transmute(#simd[16]i32)gt_32)` to `kmovw + popcnt` (2 instructions)
	//    instead of a 14-instruction horizontal vector reduction.
	// 2. Preserving `c_v` on inactive lanes (`>= n - 1`) lets us write `v_out` directly to `cdf`
	//    in a single 32-byte store without an intermediate stack buffer or straddling STLF stall.
	v_val16 := (#simd[16]u32)(val)
	v_r16 := (#simd[16]u32)(d.rng >> 8)
	zero16 := (#simd[16]u32)(0)
	v_32768_16 := (#simd[16]u16)(32768)

	c_v := intrinsics.unaligned_load((^#simd[16]u16)(cdf))
	f_prob := (#simd[16]u32)(simd.shr(v_32768_16 - c_v, (#simd[16]u16)(EC_PROB_SHIFT)))
	v_min := intrinsics.unaligned_load((^#simd[16]u32)(&EC_MIN_PROB_OFFSET[16 - n]))
	active := simd.lanes_gt(v_min, zero16)
	v_cur_raw := simd.shr(v_r16 * f_prob, (#simd[16]u32)(7 - EC_PROB_SHIFT)) + v_min
	v_cur := simd.select(active, v_cur_raw, zero16)
	gt_32 := simd.lanes_gt(v_cur, v_val16)
	symbol := int(-simd.reduce_add_ordered(transmute(#simd[16]i32)gt_32))

	cur_arr: [18]u32 = ---
	cur_arr[0] = d.rng
	intrinsics.unaligned_store((^#simd[16]u32)(&cur_arr[1]), v_cur)
	cur_arr[17] = 0
	cur := cur_arr[symbol + 1]
	d.rng = cur_arr[symbol] - cur
	d.dif -= u64(cur) << 48
	renorm_symbol(d)

	if !d.disable_cdf_update {
		count := cdf[n]
		v_rate := (#simd[16]u16)((5 + (count >> 4)) & 7)
		dec := c_v - simd.shr(c_v, v_rate)
		inc := c_v + simd.shr(v_32768_16 - c_v, v_rate)
		v_upd := simd.select((#simd[16]u16)(gt_32), dec, inc)
		v_out := simd.select((#simd[16]u16)(active), v_upd, c_v)
		intrinsics.unaligned_store((^#simd[16]u16)(cdf), v_out)
		cdf[n] = count + u16(count < 32)
	}
	return symbol
}

// 8.2.6: decodes a symbol from cdf[0 ..= N] (where N = len(cdf) - 1 and cdf[N - 1] == 32768)
// and adapts cdf in place when disable_cdf_update is false.
read_symbol :: #force_inline proc "contextless" (d: ^Symbol_Decoder, cdf: []u16) -> int #no_bounds_check {
	n := len(cdf) - 1
	switch n {
	case 2: return read_symbol_bool(d, cdf)
	case 3: return read_symbol_3(d, raw_data(cdf))
	case 4: return read_symbol_4(d, raw_data(cdf))
	case:   return read_symbol_multi(d, raw_data(cdf), n)
	}
}

// 8.2.6: adapts cdf[0 ..= N] after coding `symbol`.
update_cdf :: #force_inline proc "contextless" (cdf: []u16, symbol: int) #no_bounds_check {
	n := len(cdf) - 1
	count := cdf[n]
	rate := uint(4 + (count >> 4) + u16(n > 3))
	for i in 0 ..< symbol {
		c := cdf[i]
		cdf[i] = c - (c >> rate)
	}
	for i in symbol ..< n - 1 {
		c := cdf[i]
		cdf[i] = c + ((32768 - c) >> rate)
	}
	cdf[n] = count + u16(count < 32)
}

// 8.2.3: decodes an equiprobable boolean without CDF adaptation.
read_bool :: #force_inline proc "contextless" (d: ^Symbol_Decoder) -> int #no_bounds_check {
	cur := ((d.rng >> 8) << 7) + EC_MIN_PROB
	vw := u64(cur) << 48
	ret := u32(d.dif >= vw)
	d.dif -= u64(ret) * vw
	d.rng = cur + ret * (d.rng - 2 * cur)
	renorm_symbol(d)
	return int(ret ~ 1)
}

// 8.2.5: decodes an n-bit unsigned integer MSB-first using read_bool.
read_literal :: proc "contextless" (d: ^Symbol_Decoder, n: int) -> u32 {
	x: u32
	for _ in 0 ..< n do x = (x << 1) | u32(read_bool(d))
	return x
}

// Non-symmetric uniform integer on [0, n) using read_literal (6.10.39).
read_ns_bool :: proc "contextless" (d: ^Symbol_Decoder, n: u32) -> u32 {
	if n <= 1 do return 0
	w := 32 - int(intrinsics.count_leading_zeros(n))
	m := (u32(1) << uint(w)) - n
	v := read_literal(d, w - 1)
	if v < m do return v
	return (v << 1) - m + u32(read_bool(d))
}

// Sub-exponential with reference coded via read_bool (5.11.56).
read_signed_subexp_with_ref_bool :: proc(d: ^Symbol_Decoder, low, high, k: int, r: i32) -> i32 {
	mx := i32(high - low)
	ref := r - i32(low)
	i, mk: i32 = 0, 0
	v: i32
	for {
		b2 := i != 0 ? i32(k) + i - 1 : i32(k)
		if b2 >= 28 do return i32(low)
		a := i32(1) << uint(b2)
		if mx <= mk + 3 * a {
			v = i32(read_ns_bool(d, u32(mx - mk))) + mk
			break
		}
		if read_bool(d) != 0 {
			i += 1
			mk += a
		} else {
			v = i32(read_literal(d, int(b2))) + mk
			break
		}
	}
	u := (ref << 1) <= mx ? inverse_recenter(ref, v) : mx - 1 - inverse_recenter(mx - 1 - ref, v)
	return u + i32(low)
}

// Exp-Golomb escape level above NUM_BASE_LEVELS + COEFF_BASE_RANGE (5.11.39).
read_golomb :: proc "contextless" (d: ^Symbol_Decoder) -> (val: i32, ok: bool) {
	length := 0
	for {
		length += 1
		if length > 20 do return 0, false
		if read_bool(d) != 0 do break
	}
	x: i32 = 1
	for _ in 0 ..< length - 1 {
		x = (x << 1) | i32(read_bool(d))
	}
	return x, true
}

// 8.2.4: verifies the trailing 1 bit and zero padding at the end of a tile.
exit_symbol :: proc "contextless" (d: ^Symbol_Decoder) -> bool {
	bit_pos := (d.byte_pos << 3) - int(d.cnt)
	max_bits := len(d.data) * 8 - bit_pos
	if max_bits < -14 do return false
	trailing_pos := bit_pos - 15
	end_pos := len(d.data) * 8
	if trailing_pos < 0 || trailing_pos >= end_pos do return false
	if (d.data[trailing_pos >> 3] >> uint(7 - (trailing_pos & 7)) & 1) != 1 do return false
	// Check zero bits strictly between trailing_pos and the next byte boundary;
	// any remaining full bytes up to end_pos must also be zero.
	for pos := trailing_pos + 1; pos < end_pos && (pos & 7) != 0; pos += 1 {
		if (d.data[pos >> 3] >> uint(7 - (pos & 7)) & 1) != 0 do return false
	}
	for b in ((trailing_pos + 8) >> 3) ..< len(d.data) {
		if d.data[b] != 0 do return false
	}
	return true
}

// ---- Symbol Writer (Encoder) ------------------------------------------------

Symbol_Writer :: struct {
	data:               [dynamic]u8,
	low:                u64,
	rng:                u32,
	cnt:                int, // un-emitted renormalization bits above the 15-bit window
	disable_cdf_update: bool,
	failed:             bool,
}

start_symbol_writer :: proc(w: ^Symbol_Writer, disable_cdf_update := false) {
	clear(&w.data)
	w.low = 0
	w.rng = 1 << 15
	w.cnt = 0
	w.disable_cdf_update = disable_cdf_update
	w.failed = false
}

@(private)
propagate_carry :: #force_inline proc(w: ^Symbol_Writer, carry: u64) #no_bounds_check {
	c := carry
	for at := len(w.data) - 1; at >= 0 && c > 0; at -= 1 {
		sum := u64(w.data[at]) + c
		w.data[at] = u8(sum & 0xFF)
		c = sum >> 8
	}
}

@(private)
flush_symbol_bytes :: #force_inline proc(w: ^Symbol_Writer) #no_bounds_check {
	if carry := w.low >> uint(15 + w.cnt); carry != 0 {
		propagate_carry(w, carry)
		w.low &= (u64(1) << uint(15 + w.cnt)) - 1
	}
	for w.cnt >= 8 {
		w.cnt -= 8
		b := u8((w.low >> uint(15 + w.cnt)) & 0xFF)
		if _, err := append(&w.data, b); err != nil do w.failed = true
		w.low &= (u64(1) << uint(15 + w.cnt)) - 1
	}
}

@(private)
write_symbol_interval :: #force_inline proc(w: ^Symbol_Writer, cur, prev: u32) #no_bounds_check {
	w.low += u64(w.rng - prev)
	w.rng = prev - cur
	bits := int(intrinsics.count_leading_zeros(w.rng) - 16)
	if bits > 0 {
		w.low <<= uint(bits)
		w.rng <<= uint(bits)
		w.cnt += bits
	}
	if w.cnt >= 8 || (w.low >> uint(15 + w.cnt)) != 0 {
		flush_symbol_bytes(w)
	}
}

// Encodes `symbol` using cdf[0 ..= N] and adapts cdf in place when disable_cdf_update is false.
write_symbol :: proc(w: ^Symbol_Writer, cdf: []u16, symbol: int) #no_bounds_check {
	n := len(cdf) - 1
	r := w.rng >> 8
	prev := w.rng
	if symbol > 0 {
		f_prev := u32(32768 - cdf[symbol - 1]) >> EC_PROB_SHIFT
		prev = ((r * f_prev) >> (7 - EC_PROB_SHIFT)) + u32(EC_MIN_PROB * (n - symbol))
	}
	cur: u32 = 0
	if symbol < n - 1 {
		f_cur := u32(32768 - cdf[symbol]) >> EC_PROB_SHIFT
		cur = ((r * f_cur) >> (7 - EC_PROB_SHIFT)) + u32(EC_MIN_PROB * (n - symbol - 1))
	}
	write_symbol_interval(w, cur, prev)
	if !w.disable_cdf_update do update_cdf(cdf, symbol)
}

// Encodes an equiprobable boolean without CDF adaptation.
write_bool :: #force_inline proc(w: ^Symbol_Writer, bit: int) #no_bounds_check {
	cur := ((w.rng >> 8) << 7) + EC_MIN_PROB
	if bit == 0 {
		write_symbol_interval(w, cur, w.rng)
	} else {
		write_symbol_interval(w, 0, cur)
	}
}

// Encodes an n-bit unsigned integer MSB-first using write_bool.
write_literal :: proc(w: ^Symbol_Writer, val: u32, n: int) {
	for i := n - 1; i >= 0; i -= 1 {
		write_bool(w, int((val >> uint(i)) & 1))
	}
}

// Non-symmetric uniform integer on [0, n) using write_literal (inverse of read_ns_bool).
write_ns_bool :: proc(w: ^Symbol_Writer, val, n: u32) {
	if n <= 1 do return
	w_bits := 32 - int(intrinsics.count_leading_zeros(n))
	m := (u32(1) << uint(w_bits)) - n
	if val < m {
		write_literal(w, val, w_bits - 1)
	} else {
		v := val + m
		write_literal(w, v >> 1, w_bits - 1)
		write_bool(w, int(v & 1))
	}
}

recenter_nonneg :: proc(r, v: i32) -> i32 {
	if v > 2 * r do return v
	if v >= r do return (v - r) << 1
	return ((r - v) << 1) - 1
}

// Inverse of read_signed_subexp_with_ref_bool.
write_signed_subexp_with_ref_bool :: proc(w: ^Symbol_Writer, val: i32, low, high, k: int, r: i32) {
	mx := i32(high - low)
	ref := r - i32(low)
	u := val - i32(low)
	v := (ref << 1) <= mx ? recenter_nonneg(ref, u) : recenter_nonneg(mx - 1 - ref, mx - 1 - u)
	i, mk: i32 = 0, 0
	for {
		b2 := i != 0 ? i32(k) + i - 1 : i32(k)
		if b2 >= 28 do return
		a := i32(1) << uint(b2)
		if mx <= mk + 3 * a {
			write_ns_bool(w, u32(v - mk), u32(mx - mk))
			return
		}
		if v >= mk + a {
			write_bool(w, 1)
			i += 1
			mk += a
		} else {
			write_bool(w, 0)
			write_literal(w, u32(v - mk), int(b2))
			return
		}
	}
}

// Inverse of read_golomb for x >= 1 (where Quant[pos] = x + COEFF_BASE_RANGE + NUM_BASE_LEVELS).
write_golomb :: proc(w: ^Symbol_Writer, x: i32) {
	length := 32 - int(intrinsics.count_leading_zeros(u32(x)))
	for _ in 0 ..< length - 1 do write_bool(w, 0)
	write_bool(w, 1)
	if length > 1 {
		write_literal(w, u32(x) & ((u32(1) << uint(length - 1)) - 1), length - 1)
	}
}

// Terminates the tile bitstream with the normative trailing 1 bit and zero padding to a byte boundary.
finish_symbol_writer :: proc(w: ^Symbol_Writer) {
	flush_symbol_bytes(w)
	w.low = ((w.low + 0x3FFF) & ~u64(0x7FFF)) | 0x4000
	if carry := w.low >> uint(15 + w.cnt); carry != 0 {
		propagate_carry(w, carry)
		w.low &= (u64(1) << uint(15 + w.cnt)) - 1
	}
	last_byte := u8((w.low >> uint(7 + w.cnt)) & 0xFF)
	if _, err := append(&w.data, last_byte); err != nil do w.failed = true
}

// ---- Symbol Rate Cost Estimation (1/256-bit units) --------------------------

// -log2(p / 512) * 256 for p in 0 ..= 512 (with p = 0 clamped to p = 1).
PROB_COST_Q9: [513]u16

@(private, init)
init_prob_costs :: proc "contextless" () {
	for p in 1 ..= 512 {
		x := u32(p) << 7 // 512 -> 65536
		res := u32(0)
		for x < 65536 {
			x <<= 1
			res += 256
		}
		for b in 0 ..< 8 {
			x = u32((u64(x) * u64(x)) >> 16)
			if x < 65536 {
				x <<= 1
				res += 128 >> uint(b)
			}
		}
		PROB_COST_Q9[p] = u16(res)
	}
	PROB_COST_Q9[0] = PROB_COST_Q9[1]
}

// Bit cost of coding `symbol` with `cdf` in 1/256-bit units.
symbol_cost :: #force_inline proc "contextless" (cdf: []u16, symbol: int) -> u32 #no_bounds_check {
	prev := symbol > 0 ? int(cdf[symbol - 1]) : 0
	prob := clamp((int(cdf[symbol]) - prev) >> EC_PROB_SHIFT, 1, 512)
	return u32(PROB_COST_Q9[prob])
}
