package common

import "core:testing"

// A unary run whose one bit lands just past the cache's valid bits (the
// wide refill holds it, but count doesn't cover it): it is read on the next
// refill, and the reader stays in step.
@(test)
read_unary_past_valid_bits :: proc(t: ^testing.T) {
	data: [32]u8
	// 7 bits of 1111111, then 58 zeros, then a one, then 101.
	data[0] = 0xfe
	// bits 7..64 are zero; bit 65 is one; bits 66..68 are 101
	data[8] = 0b0110_1000
	r := Bit_Reader{data = data[:], left = len(data) * 8}
	testing.expect_value(t, read_bits(&r, 7), 0x7f)
	refill_bits(&r)
	testing.expect_value(t, read_unary(&r), 58)
	testing.expect_value(t, read_bits(&r, 3), 0b101)
	testing.expect(t, r.count >= 0 && r.left == len(data) * 8 - 7 - 59 - 3)
}

@(test)
bool_coder_round_trips_and_carries :: proc(t: ^testing.T) {
	enc: Bool_Encoder
	defer delete(enc.data)

	// Empty stream flushes cleanly and decodes as zeros without overrun.
	bool_encode_start(&enc)
	bool_encode_finish(&enc)
	testing.expect(t, len(enc.data) >= 2 && !enc.failed)
	dec: Bool_Decoder
	bool_start(&dec, enc.data[:])
	testing.expect_value(t, bool_decode(&dec, 128), 0)
	testing.expect(t, !bool_overrun(&dec))

	// Extreme probabilities from bit 0 (forcing long 0xff runs and carries)
	// followed by mixed probabilities and literals.
	N :: 4096
	bits: [N]int
	probs: [N]u8
	seed := u32(0x12345678)
	next :: proc(s: ^u32) -> u32 {
		s^ = s^ * 1664525 + 1013904223
		return s^ >> 8
	}
	bool_encode_start(&enc)
	for i in 0 ..< N {
		switch {
		case i < 256:
			bits[i], probs[i] = 1, 255
		case i < 512:
			bits[i], probs[i] = i & 1, 255
		case i < 768:
			bits[i], probs[i] = 1, 1
		case:
			bits[i] = int(next(&seed) & 1)
			probs[i] = u8(max(next(&seed) & 255, 1))
		}
		bool_encode(&enc, bits[i], probs[i])
	}
	bool_encode_literal(&enc, 0x2a5, 10)
	bool_encode_finish(&enc)
	testing.expect(t, !enc.failed)

	bool_start(&dec, enc.data[:])
	for i in 0 ..< N {
		if !testing.expectf(t, bool_decode(&dec, probs[i]) == bits[i], "mismatch at %d (prob=%d)", i, probs[i]) do return
	}
	testing.expect_value(t, bool_literal(&dec, 10), 0x2a5)
	testing.expect(t, !bool_overrun(&dec))
}
