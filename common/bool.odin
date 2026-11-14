package common

import "base:intrinsics"

// Boolean (arithmetic) decoder of VP8 (RFC 6386 section 7) and VP9 (section
// 9.2), which define the same split and renormalization. `value` holds the
// decoder's 8-bit window in its top byte followed by look-ahead bits; bytes
// past the end of the data read as zero, so a truncated partition decodes as
// zeros (bool_overrun tells).
Bool_Decoder :: struct {
	data:  []u8,
	pos:   int,
	value: u64,
	bits:  int, // valid bits in value
	range: u32,
}

bool_start :: proc "contextless" (b: ^Bool_Decoder, data: []u8) {
	b^ = {data = data, range = 255}
	bool_fill(b)
}

@(private)
bool_fill_tail :: #force_no_inline proc "contextless" (b: ^Bool_Decoder) -> u64 #no_bounds_check {
	word: u64
	for i in 0 ..< 4 {
		word <<= 8
		if b.pos + i < len(b.data) do word |= u64(b.data[b.pos + i])
	}
	return word
}

// Refills load 32 stream bits once fewer than 16 remain, as the CABAC window
// does; a decision consumes at most 7, so the window never runs dry.
bool_fill :: #force_inline proc "contextless" (b: ^Bool_Decoder) #no_bounds_check {
	word := b.pos + 4 <= len(b.data) ? u64(u32(intrinsics.unaligned_load((^u32be)(&b.data[b.pos])))) : bool_fill_tail(b)
	b.value |= word << uint(32 - b.bits)
	b.pos += 4
	b.bits += 32
}

// One bool with probability prob/256 of being 0.
bool_decode :: #force_inline proc "contextless" (b: ^Bool_Decoder, prob: u8) -> int {
	if b.bits < 16 do bool_fill(b)
	split := 1 + (((b.range - 1) * u32(prob)) >> 8)
	big := u64(split) << 56
	// The outcome is data dependent, so it selects values rather than branching.
	one := b.value >= big
	b.range = one ? b.range - split : split
	b.value -= one ? big : 0
	shift := uint(intrinsics.count_leading_zeros(b.range) - 24)
	b.range <<= shift
	b.value <<= shift
	b.bits -= int(shift)
	return int(one)
}

// An n-bit unsigned literal, most significant bit first, at probability 128.
bool_literal :: #force_inline proc "contextless" (b: ^Bool_Decoder, n: int) -> int {
	x := 0
	for _ in 0 ..< n do x = x << 1 | bool_decode(b, 128)
	return x
}

// A tree value with probs[node >> 1] at each node; leaves are stored negated.
bool_tree :: #force_inline proc "contextless" (b: ^Bool_Decoder, tree: []i8, probs: []u8) -> int #no_bounds_check {
	n := 0
	for {
		n = int(tree[n + bool_decode(b, probs[n >> 1])])
		if n <= 0 do return -n
	}
}

// Whether the decoder read past its data (corrupt or truncated input).
bool_overrun :: #force_inline proc "contextless" (b: ^Bool_Decoder) -> bool {
	return b.pos - b.bits / 8 > len(b.data) + 8
}

// Boolean (arithmetic) encoder paired with Bool_Decoder: the inverse range
// subdivision of RFC 6386 section 7.3 and VP9 section 9.2, with carry
// propagation across pending 0xff bytes.
Bool_Encoder :: struct {
	data:       [dynamic]u8,
	low, range: u32,
	count:      int,
	failed:     bool, // an append ran out of memory
}

bool_encode_start :: proc(b: ^Bool_Encoder) {
	clear(&b.data)
	b.low, b.range, b.count, b.failed = 0, 255, -24, false
}

@(private)
bool_encode_byte :: #force_no_inline proc(b: ^Bool_Encoder, shift: int) #no_bounds_check {
	offset := shift - b.count
	if (b.low << uint(offset - 1)) & 0x80000000 != 0 {
		at := len(b.data) - 1
		for at >= 0 && b.data[at] == 255 {b.data[at] = 0; at -= 1}
		assert(at >= 0 || b.failed)
		if at >= 0 do b.data[at] += 1
	}
	if _, err := non_zero_append(&b.data, u8(b.low >> uint(24 - offset))); err != nil do b.failed = true
	b.low <<= uint(offset)
	rem := b.count
	b.low &= 0xffffff
	b.count -= 8
	b.low <<= uint(rem)
}

// Encodes one bit (0 or non-zero) with probability prob/256 of being 0.
bool_encode :: #force_inline proc(b: ^Bool_Encoder, bit: int, prob: u8) {
	split := 1 + (((b.range - 1) * u32(prob)) >> 8)
	b.low += bit != 0 ? split : 0
	b.range = bit != 0 ? b.range - split : split
	shift := int(intrinsics.count_leading_zeros(b.range) - 24)
	b.range <<= uint(shift)
	b.count += shift
	if b.count >= 0 {
		bool_encode_byte(b, shift)
		return
	}
	b.low <<= uint(shift)
}

// An n-bit unsigned literal, most significant bit first, at probability 128.
bool_encode_literal :: #force_inline proc(b: ^Bool_Encoder, value, bits: int) {
	for i := bits - 1; i >= 0; i -= 1 do bool_encode(b, (value >> uint(i)) & 1, 128)
}

// Flushes the encoder's pending window to `b.data`.
bool_encode_finish :: proc(b: ^Bool_Encoder) {
	if b.range == 255 do bool_encode(b, 0, 128)
	for _ in 0 ..< 32 do bool_encode(b, 0, 128)
}
