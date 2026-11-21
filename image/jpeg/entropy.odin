package jpeg

import "base:intrinsics"
import "core:mem"
import "core:slice"

// Huffman decoding of the scans: codes up to FAST_BITS long are found with
// one table lookup, longer ones against the largest code of each length; an
// AC table also resolves small coefficients (run, size and value) in one
// lookup.

@(private)
FAST_BITS :: 9

@(private)
Huffman :: struct {
	fast:    [1 << FAST_BITS]u8, // the symbol index of a code up to FAST_BITS long, 255: longer
	fast_ac: [1 << FAST_BITS]i16, // value << 8 | run << 4 | bits used, 0: not resolved
	values:  [256]u8,
	size:    [257]u8, // code length of each symbol index, 0 after the last
	code:    [256]u16,
	maxcode: [18]u32, // per length, the largest code + 1, shifted to 16 bits
	delta:   [17]int, // per length, symbol index - code
}

@(private)
build_huffman :: proc(h: ^Huffman, counts: [16]int) -> Error {
	k := 0
	for n, i in counts do for _ in 0 ..< n {
		h.size[k] = u8(i + 1)
		k += 1
	}
	h.size[k] = 0
	code := 0
	k = 0
	for j in 1 ..= 16 {
		h.delta[j] = k - code
		if int(h.size[k]) == j {
			for int(h.size[k]) == j {
				h.code[k] = u16(code)
				code += 1
				k += 1
			}
			if code - 1 >= 1 << uint(j) do return .Invalid
		}
		h.maxcode[j] = u32(code) << uint(16 - j)
		code <<= 1
	}
	h.maxcode[17] = max(u32)
	h.fast = 255
	for i in 0 ..< k {
		s := int(h.size[i])
		if s > FAST_BITS do continue
		c := int(h.code[i]) << uint(FAST_BITS - s)
		for j in 0 ..< 1 << uint(FAST_BITS - s) do h.fast[c + j] = u8(i)
	}
	return .None
}

@(private)
build_fast_ac :: proc(h: ^Huffman) {
	for i in 0 ..< 1 << FAST_BITS {
		h.fast_ac[i] = 0
		fast := h.fast[i]
		if fast == 255 do continue
		rs := int(h.values[fast])
		run, magnitude := rs >> 4 & 15, rs & 15
		length := int(h.size[fast])
		if magnitude == 0 || length + magnitude > FAST_BITS do continue
		// The code, then the value's bits, both in the lookup's bits.
		k := (i << uint(length)) & (1 << FAST_BITS - 1) >> uint(FAST_BITS - magnitude)
		if k < 1 << uint(magnitude - 1) do k += -1 << uint(magnitude) + 1
		if k >= -128 && k <= 127 do h.fast_ac[i] = i16(k * 256 + run * 16 + length + magnitude)
	}
}

// Annex K's tables (K.3), luminance then chrominance. Motion-JPEG frames, as
// webcams send them, carry no DHT and expect these; libjpeg-turbo gives them
// to tables 0 and 1 of a sequential file where those are undefined at its
// first scan, and so does standard_tables.
@(private, rodata)
STANDARD_COUNTS := [2][2][16]int { // [class][table], codes of each length
	{{0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0}, {0, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0}},
	{{0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 0x7d}, {0, 2, 1, 2, 4, 4, 3, 4, 7, 5, 4, 4, 0, 1, 2, 0x77}},
}

// The AC symbols; a DC table's are 0 to 11.
@(private, rodata)
STANDARD_AC_VALUES := [2][162]u8 {
	{
		0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12, 0x21, 0x31, 0x41, 0x06, 0x13, 0x51, 0x61, 0x07,
		0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xa1, 0x08, 0x23, 0x42, 0xb1, 0xc1, 0x15, 0x52, 0xd1, 0xf0,
		0x24, 0x33, 0x62, 0x72, 0x82, 0x09, 0x0a, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x25, 0x26, 0x27, 0x28,
		0x29, 0x2a, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49,
		0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69,
		0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7a, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89,
		0x8a, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5, 0xa6, 0xa7,
		0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6, 0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3, 0xc4, 0xc5,
		0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2, 0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda, 0xe1, 0xe2,
		0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9, 0xea, 0xf1, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8,
		0xf9, 0xfa,
	},
	{
		0x00, 0x01, 0x02, 0x03, 0x11, 0x04, 0x05, 0x21, 0x31, 0x06, 0x12, 0x41, 0x51, 0x07, 0x61, 0x71,
		0x13, 0x22, 0x32, 0x81, 0x08, 0x14, 0x42, 0x91, 0xa1, 0xb1, 0xc1, 0x09, 0x23, 0x33, 0x52, 0xf0,
		0x15, 0x62, 0x72, 0xd1, 0x0a, 0x16, 0x24, 0x34, 0xe1, 0x25, 0xf1, 0x17, 0x18, 0x19, 0x1a, 0x26,
		0x27, 0x28, 0x29, 0x2a, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48,
		0x49, 0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68,
		0x69, 0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7a, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87,
		0x88, 0x89, 0x8a, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5,
		0xa6, 0xa7, 0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6, 0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3,
		0xc4, 0xc5, 0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2, 0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda,
		0xe2, 0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9, 0xea, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8,
		0xf9, 0xfa,
	},
}

// Tables 0 and 1 of each class that no DHT has defined become Annex K's.
@(private)
standard_tables :: proc(d: ^Decoder) {
	for class in 0 ..< 2 do for t in 0 ..< 2 do if t not_in d.defined[class] {
		h := class == 0 ? &d.dc_tables[t] : &d.ac_tables[t]
		_ = build_huffman(h, STANDARD_COUNTS[class][t])
		if class == 0 {
			for i in 0 ..< 12 do h.values[i] = u8(i)
		} else {
			copy(h.values[:], STANDARD_AC_VALUES[t][:])
			build_fast_ac(h)
		}
		d.defined[class] += {t}
	}
}

// The entropy-coded data, MSB first through a 64-bit window. A marker (0xFF
// then anything but 0 or fill) ends it; from there on zeros are read. Kept
// in a local while blocks are decoded.
@(private)
Bits :: struct {
	data:   []u8,
	at:     int,
	window: u64, // the next bits, from the top
	count:  int,
	marker: int, // what ended the data, MARKER_NONE until then
}

@(private)
refill :: #force_inline proc(b: ^Bits) #no_bounds_check {
	// Eight bytes at once where none is 0xFF (no stuffing, no marker): as
	// many whole bytes as the window has room for.
	if b.at + 8 <= len(b.data) && b.marker == MARKER_NONE {
		v := u64(intrinsics.unaligned_load((^u64be)(&b.data[b.at])))
		ones :: 0x0101_0101_0101_0101
		x := ~v
		if (x - ones) & ~x & (ones << 7) == 0 {
			n := (64 - b.count) >> 3
			b.window |= v >> uint(b.count) & ~(max(u64) >> uint(b.count + n * 8))
			b.count += n * 8
			b.at += n
			return
		}
	}
	for b.count <= 56 {
		v: u64
		if b.marker == MARKER_NONE && b.at < len(b.data) {
			v = u64(b.data[b.at])
			b.at += 1
			if v == 0xFF {
				c := 0xFF
				for c == 0xFF && b.at < len(b.data) {
					c = int(b.data[b.at])
					b.at += 1
				}
				if c == 0xFF do c = 0 // the data ended inside the fill
				if c != 0 {
					b.marker = c
					v = 0
				}
			}
		}
		b.window |= v << uint(56 - b.count)
		b.count += 8
	}
}

@(private)
decode_huffman :: #force_inline proc(b: ^Bits, h: ^Huffman) -> int #no_bounds_check {
	if b.count < 16 do refill(b)
	k := h.fast[b.window >> (64 - FAST_BITS)]
	if k < 255 {
		s := int(h.size[k])
		b.window <<= uint(s)
		b.count -= s
		return int(h.values[k])
	}
	top := u32(b.window >> 48)
	n := FAST_BITS + 1
	for top >= h.maxcode[n] do n += 1
	if n == 17 do return -1
	symbol := int(b.window >> uint(64 - n)) + h.delta[n]
	if symbol < 0 || symbol > 255 do return -1
	b.window <<= uint(n)
	b.count -= n
	return int(h.values[symbol])
}

// n bits as a signed value: JPEG's receive and extend.
@(private)
receive_extend :: #force_inline proc(b: ^Bits, n: int) -> int {
	if b.count < n do refill(b)
	v := int(b.window >> uint(64 - n))
	b.window <<= uint(n)
	b.count -= n
	if v < 1 << uint(n - 1) do v += -1 << uint(n) + 1
	return v
}

@(private)
get_bits :: #force_inline proc(b: ^Bits, n: int) -> int {
	if b.count < n do refill(b)
	v := int(b.window >> uint(64 - n))
	b.window <<= uint(n)
	b.count -= n
	return v
}

@(private)
get_bit :: #force_inline proc(b: ^Bits) -> bool {
	if b.count < 1 do refill(b)
	v := b.window >> 63
	b.window <<= 1
	b.count -= 1
	return v != 0
}

@(private)
Scan_Kind :: enum {
	Sequential, // baseline, its blocks transformed into the bands as they come
	Sequential_Kept, // baseline in several scans: coefficients kept
	DC_First,
	DC_Refine,
	AC_First,
	AC_Refine,
}

// Starts the entropy-coded data of the scan whose header was just read.
@(private)
begin_scan :: proc(d: ^Decoder) {
	d.bits = {data = d.data, at = d.at, marker = MARKER_NONE}
	for &c in d.comps do c.dc_pred = 0
	d.eob_run = 0
	d.todo = d.restart > 0 ? d.restart : max(int)
	d.stopped = false
}

// The rows of the scan's units: MCU rows, or a lone component's block rows.
@(private)
scan_rows :: proc(d: ^Decoder) -> int {
	return d.scan_n == 1 ? d.comps[d.order[0]].blocks_y : d.mcu_y
}

// Decodes rows [first, end) of the scan's units, its kind chosen once.
@(private)
decode_rows :: proc(d: ^Decoder, first, end: int) -> (err: Error) {
	if d.stopped do return .None
	b := d.bits // the cursor in a local while blocks are decoded
	defer d.bits = b
	switch {
	case !d.features.progressive && d.coeffs == nil:
		return unit_rows(d, &b, first, end, .Sequential)
	case !d.features.progressive:
		return unit_rows(d, &b, first, end, .Sequential_Kept)
	case d.spec_start == 0 && d.spec_end != 0:
		return .Invalid // DC and AC in one progressive scan
	case d.spec_start != 0 && d.scan_n > 1:
		return .Invalid // AC scans hold one component
	case d.spec_start == 0 && d.succ_high == 0:
		return unit_rows(d, &b, first, end, .DC_First)
	case d.spec_start == 0:
		return unit_rows(d, &b, first, end, .DC_Refine)
	case d.succ_high == 0:
		return unit_rows(d, &b, first, end, .AC_First)
	}
	return unit_rows(d, &b, first, end, .AC_Refine)
}

// Blocks in stream order: a lone component's in raster order, or each MCU's
// component by component.
@(private)
unit_rows :: proc(d: ^Decoder, b: ^Bits, first, end: int, $K: Scan_Kind) -> Error {
	if d.scan_n == 1 {
		c := &d.comps[d.order[0]]
		for by in first ..< end {
			for bx in 0 ..< c.blocks_x {
				decode_one(d, b, c, bx, by, K) or_return
				if d.todo -= 1; d.todo <= 0 && !next_interval(d, b) do return .None
			}
		}
		return .None
	}
	for my in first ..< end {
		for mx in 0 ..< d.mcu_x {
			for k in 0 ..< d.scan_n {
				c := &d.comps[d.order[k]]
				for y in 0 ..< c.v do for x in 0 ..< c.h do decode_one(d, b, c, mx * c.h + x, my * c.v + y, K) or_return
			}
			if d.todo -= 1; d.todo <= 0 && !next_interval(d, b) do return .None
		}
	}
	return .None
}

// At the end of a restart interval: on a restart marker, resets the decoder
// and goes on; on anything else the scan stops there, with what it has.
@(private)
next_interval :: proc(d: ^Decoder, b: ^Bits) -> bool {
	if b.count < 24 do refill(b)
	if b.marker < 0xD0 || b.marker > 0xD7 {
		d.stopped = true
		return false
	}
	b.window, b.count, b.marker = 0, 0, MARKER_NONE
	for &c in d.comps do c.dc_pred = 0
	d.eob_run = 0
	d.todo = d.restart > 0 ? d.restart : max(int)
	return true
}

@(private)
decode_one :: #force_inline proc(d: ^Decoder, b: ^Bits, c: ^Component, bx, by: int, $K: Scan_Kind) -> Error #no_bounds_check {
	when K == .Sequential {
		block: [8]#simd[8]i16
		decode_block(b, slice.reinterpret([]i16, block[:]), &d.dc_tables[c.dc], &d.ac_tables[c.ac], &c.dc_pred) or_return
		idct_block(d.planes[plane_row(c, by * 8) + bx * 8:], c.stride, &block, &d.quant[c.tq])
		return .None
	} else {
		at := c.coeffs + by * (c.stride >> 3) + bx
		block := slice.reinterpret([]i16, d.coeffs[at][:])
		when K == .Sequential_Kept {
			return decode_block(b, block, &d.dc_tables[c.dc], &d.ac_tables[c.ac], &c.dc_pred)
		} else when K == .DC_First {
			d.coeffs[at] = {}
			return decode_dc_first(b, block, &d.dc_tables[c.dc], &c.dc_pred, d.succ_low)
		} else when K == .DC_Refine {
			if get_bit(b) do block[0] += i16(1 << uint(d.succ_low))
			return .None
		} else when K == .AC_First {
			return decode_ac_first(d, b, block, &d.ac_tables[c.ac])
		} else {
			return decode_ac_refine(d, b, block, &d.ac_tables[c.ac])
		}
	}
}

// Band m of a one-scan image: its MCU row (or a lone component's block rows)
// decoded into the bands. Once the data has stopped, the band is mid-gray.
@(private)
decode_band :: proc(d: ^Decoder, m: int) -> Error {
	if d.stopped {
		for &c in d.comps[:d.features.components] do mem.set(&d.planes[plane_row(&c, m * c.band)], 128, c.band * c.stride)
		return .None
	}
	if d.scan_n == 1 {
		c := &d.comps[d.order[0]]
		return decode_rows(d, m * c.v, min((m + 1) * c.v, c.blocks_y))
	}
	return decode_rows(d, m, m + 1)
}

// Band m of a multi-scan image, from its kept coefficients.
@(private)
transform_band :: proc(d: ^Decoder, m: int) #no_bounds_check {
	for &c in d.comps[:d.features.components] {
		quant := &d.quant[c.tq]
		for by in m * c.v ..< min((m + 1) * c.v, c.blocks_y) {
			out := plane_row(&c, by * 8)
			for bx in 0 ..< c.blocks_x {
				idct_block(d.planes[out + bx * 8:], c.stride, &d.coeffs[c.coeffs + by * (c.stride >> 3) + bx], quant)
			}
		}
	}
}

// A baseline block's coefficients, in natural order.
@(private)
decode_block :: proc(b: ^Bits, block: []i16, dc, ac: ^Huffman, pred: ^int) -> Error #no_bounds_check {
	t := decode_huffman(b, dc)
	if t < 0 || t > 15 do return .Invalid
	pred^ += t != 0 ? receive_extend(b, t) : 0
	block[0] = i16(pred^)
	k := 1
	for k < 64 {
		if b.count < 16 do refill(b)
		if r := int(ac.fast_ac[b.window >> (64 - FAST_BITS)]); r != 0 {
			k += r >> 4 & 15
			s := r & 15
			b.window <<= uint(s)
			b.count -= s
			z := DEZIGZAG[k]
			k += 1
			block[z] = i16(r >> 8)
			continue
		}
		rs := decode_huffman(b, ac)
		if rs < 0 do return .Invalid
		s, r := rs & 15, rs >> 4
		if s == 0 {
			if rs != 0xF0 do break // end of block
			k += 16
		} else {
			k += r
			z := DEZIGZAG[k]
			k += 1
			block[z] = i16(receive_extend(b, s))
		}
	}
	return .None
}

@(private)
decode_dc_first :: proc(b: ^Bits, block: []i16, dc: ^Huffman, pred: ^int, shift: int) -> Error #no_bounds_check {
	t := decode_huffman(b, dc)
	if t < 0 || t > 15 do return .Invalid
	pred^ += t != 0 ? receive_extend(b, t) : 0
	block[0] = i16(pred^ * (1 << uint(shift)))
	return .None
}

@(private)
decode_ac_first :: proc(d: ^Decoder, b: ^Bits, block: []i16, ac: ^Huffman) -> Error #no_bounds_check {
	if d.eob_run > 0 {
		d.eob_run -= 1
		return .None
	}
	shift := uint(d.succ_low)
	for k := d.spec_start; k <= d.spec_end; {
		if b.count < 16 do refill(b)
		if r := int(ac.fast_ac[b.window >> (64 - FAST_BITS)]); r != 0 {
			k += r >> 4 & 15
			s := r & 15
			b.window <<= uint(s)
			b.count -= s
			block[DEZIGZAG[k]] = i16((r >> 8) * (1 << shift))
			k += 1
			continue
		}
		rs := decode_huffman(b, ac)
		if rs < 0 do return .Invalid
		s, r := rs & 15, rs >> 4
		if s == 0 {
			if r < 15 {
				d.eob_run = 1 << uint(r)
				if r != 0 do d.eob_run += get_bits(b, r)
				d.eob_run -= 1
				break
			}
			k += 16
		} else {
			k += r
			block[DEZIGZAG[k]] = i16(receive_extend(b, s) * (1 << shift))
			k += 1
		}
	}
	return .None
}

// Refines the coefficients already set by one bit each and places the new
// ones (each +-1 at this bit).
@(private)
decode_ac_refine :: proc(d: ^Decoder, b: ^Bits, block: []i16, ac: ^Huffman) -> Error #no_bounds_check {
	bit := i16(1 << uint(d.succ_low))
	refine :: #force_inline proc(b: ^Bits, p: ^i16, bit: i16) {
		if get_bit(b) && p^ & bit == 0 {
			if p^ > 0 {p^ += bit} else {p^ -= bit}
		}
	}
	if d.eob_run > 0 {
		d.eob_run -= 1
		for k in d.spec_start ..= d.spec_end {
			p := &block[DEZIGZAG[k]]
			if p^ != 0 do refine(b, p, bit)
		}
		return .None
	}
	k := d.spec_start
	for k <= d.spec_end {
		rs := decode_huffman(b, ac)
		if rs < 0 do return .Invalid
		s, r := rs & 15, rs >> 4
		value: i16
		if s == 0 {
			if r < 15 {
				d.eob_run = 1 << uint(r) - 1
				if r != 0 do d.eob_run += get_bits(b, r)
				r = 64 // the rest of the block is refined only
			}
			// r == 15: sixteen zeros, the last of them written as 0
		} else {
			if s != 1 do return .Invalid
			value = get_bit(b) ? bit : -bit
		}
		for k <= d.spec_end {
			p := &block[DEZIGZAG[k]]
			k += 1
			if p^ != 0 {
				refine(b, p, bit)
			} else {
				if r == 0 {
					p^ = value
					break
				}
				r -= 1
			}
		}
	}
	return .None
}
