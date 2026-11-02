package av1

import "core:testing"

@(test)
test_symbol_roundtrip :: proc(t: ^testing.T) {
	enc_ctx := new(CDF_Context)
	dec_ctx := new(CDF_Context)
	defer free(enc_ctx)
	defer free(dec_ctx)

	init_cdf_context(enc_ctx, 40)
	init_cdf_context(dec_ctx, 40)

	w: Symbol_Writer
	defer delete(w.data)
	start_symbol_writer(&w, false)

	N :: 25000
	state := u64(0x9E3779B97F4A7C15)
	next_rand :: proc(s: ^u64) -> u32 {
		x := s^
		x ~= x << 13
		x ~= x >> 7
		x ~= x << 17
		s^ = x
		return u32(x >> 32)
	}

	for i in 0 ..< N {
		r := next_rand(&state)
		switch i % 8 {
		case 0:
			ctx := int(r & 3)
			sym := int((r >> 2) % 13)
			write_symbol(&w, enc_ctx.y_mode_cdf[ctx][:], sym)
		case 1:
			ctx := int(r & 1)
			sym := int((r >> 1) & 15)
			write_symbol(&w, enc_ctx.inter_tx_type_set1_cdf[ctx][:], sym)
		case 2:
			ctx := int(r % 6)
			sym := int((r >> 3) & 1)
			write_symbol(&w, enc_ctx.new_mv_cdf[ctx][:], sym)
		case 3:
			write_bool(&w, int(r & 1))
		case 4:
			nbits := int(r % 12) + 1
			val := (r >> 4) & ((u32(1) << uint(nbits)) - 1)
			write_literal(&w, val, nbits)
		case 5:
			n_max := (r % 53) + 2
			val := (r >> 8) % n_max
			write_ns_bool(&w, val, n_max)
		case 6:
			low, high := -96, 31
			ref := i32(int(r % u32(high - low + 1)) + low)
			val := i32(int((r >> 8) % u32(high - low + 1)) + low)
			write_signed_subexp_with_ref_bool(&w, val, low, high + 1, SGRPROJ_PRJ_SUBEXP_K, ref)
		case 7:
			gx := i32((r % 500) + 1)
			write_golomb(&w, gx)
		}
	}
	finish_symbol_writer(&w)
	testing.expect(t, !w.failed, "symbol writer allocation failed")
	testing.expect(t, len(w.data) > 0, "symbol writer produced empty stream")

	d: Symbol_Decoder
	init_symbol(&d, w.data[:], false)

	state = 0x9E3779B97F4A7C15
	for i in 0 ..< N {
		r := next_rand(&state)
		switch i % 8 {
		case 0:
			ctx := int(r & 3)
			expected := int((r >> 2) % 13)
			got := read_symbol(&d, dec_ctx.y_mode_cdf[ctx][:])
			if got != expected {
				testing.expectf(t, false, "step %d: y_mode got %d, expected %d", i, got, expected)
				return
			}
		case 1:
			ctx := int(r & 1)
			expected := int((r >> 1) & 15)
			got := read_symbol(&d, dec_ctx.inter_tx_type_set1_cdf[ctx][:])
			if got != expected {
				testing.expectf(t, false, "step %d: tx_type got %d, expected %d", i, got, expected)
				return
			}
		case 2:
			ctx := int(r % 6)
			expected := int((r >> 3) & 1)
			got := read_symbol(&d, dec_ctx.new_mv_cdf[ctx][:])
			if got != expected {
				testing.expectf(t, false, "step %d: new_mv got %d, expected %d", i, got, expected)
				return
			}
		case 3:
			expected := int(r & 1)
			got := read_bool(&d)
			if got != expected {
				testing.expectf(t, false, "step %d: bool got %d, expected %d", i, got, expected)
				return
			}
		case 4:
			nbits := int(r % 12) + 1
			expected := (r >> 4) & ((u32(1) << uint(nbits)) - 1)
			got := read_literal(&d, nbits)
			if got != expected {
				testing.expectf(t, false, "step %d: literal(%d) got %d, expected %d", i, nbits, got, expected)
				return
			}
		case 5:
			n_max := (r % 53) + 2
			expected := (r >> 8) % n_max
			got := read_ns_bool(&d, n_max)
			if got != expected {
				testing.expectf(t, false, "step %d: ns(%d) got %d, expected %d", i, n_max, got, expected)
				return
			}
		case 6:
			low, high := -96, 31
			ref := i32(int(r % u32(high - low + 1)) + low)
			expected := i32(int((r >> 8) % u32(high - low + 1)) + low)
			got := read_signed_subexp_with_ref_bool(&d, low, high + 1, SGRPROJ_PRJ_SUBEXP_K, ref)
			if got != expected {
				testing.expectf(t, false, "step %d: subexp got %d, expected %d", i, got, expected)
				return
			}
		case 7:
			expected := i32((r % 500) + 1)
			got, ok := read_golomb(&d)
			if !ok || got != expected {
				testing.expectf(t, false, "step %d: golomb got %d (ok=%v), expected %d", i, got, ok, expected)
				return
			}
		}
	}

	testing.expect(t, exit_symbol(&d), "exit_symbol failed after round-trip")
	testing.expect(t, enc_ctx^ == dec_ctx^, "encoder and decoder CDF contexts diverged")

	// Test load_cdfs resets all count slots while preserving probabilities.
	loaded := new(CDF_Context)
	defer free(loaded)
	load_cdfs(loaded, dec_ctx)
	for ctx in 0 ..< 4 {
		testing.expect_value(t, loaded.y_mode_cdf[ctx][INTRA_MODES], 0)
		testing.expect_value(t, loaded.y_mode_cdf[ctx][0], dec_ctx.y_mode_cdf[ctx][0])
	}
	for ctx in 0 ..< 2 {
		testing.expect_value(t, loaded.inter_tx_type_set1_cdf[ctx][16], 0)
		testing.expect_value(t, loaded.inter_tx_type_set1_cdf[ctx][0], dec_ctx.inter_tx_type_set1_cdf[ctx][0])
	}

	// Test symbol_cost on equiprobable 2-symbol and 4-symbol CDFs.
	cdf2 := [3]u16{16384, 32768, 0}
	testing.expect_value(t, symbol_cost(cdf2[:], 0), 256)
	testing.expect_value(t, symbol_cost(cdf2[:], 1), 256)
	cdf4 := [5]u16{8192, 16384, 24576, 32768, 0}
	for s in 0 ..< 4 {
		testing.expect_value(t, symbol_cost(cdf4[:], s), 512)
	}
}
