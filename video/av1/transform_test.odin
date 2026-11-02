package av1

import "core:testing"

@(test)
test_wht4x4_lossless :: proc(t: ^testing.T) {
	rng := u64(0x1234_5678_9ABC_DEF0)
	next_u32 :: proc(s: ^u64) -> u32 {
		x := s^
		x ~= x << 13
		x ~= x >> 7
		x ~= x << 17
		s^ = x
		return u32(x)
	}

	bit_depths := [3]int{8, 10, 12}
	for bd in bit_depths {
		span := i32(1) << uint(bd)
		for _ in 0 ..< 256 {
			orig: [16]i32
			work: [16]i32
			for i in 0 ..< 16 {
				orig[i] = i32(next_u32(&rng) % u32(2 * span - 1)) - (span - 1)
				work[i] = orig[i]
			}
			forward_transform_2d(work[:], TX_4X4, DCT_DCT, true, bd)
			dc_q := dc_quant(0, 0, bd)
			ac_q := ac_quant(0, 0, bd)
			for i in 0 ..< 16 {
				q, dq := quantize_coeff(work[i], i, TX_4X4, DCT_DCT, 0, dc_q, ac_q, 15, bd)
				testing.expect(t, q * 4 == work[i])
				work[i] = dq
			}
			inverse_transform_2d(work[:], TX_4X4, DCT_DCT, true, 16, bd)
			for i in 0 ..< 16 {
				if work[i] != orig[i] {
					testing.expectf(t, false, "WHT mismatch bd=%d i=%d got=%d want=%d", bd, i, work[i], orig[i])
					return
				}
			}
		}
	}
}

@(test)
test_dc_only_matches_full_itx :: proc(t: ^testing.T) {
	bit_depths := [3]int{8, 10, 12}
	test_dcs := [7]i32{-4095, -513, -17, 0, 19, 777, 4095}
	b0 := make([]i32, 64 * 64)
	b1 := make([]i32, 64 * 64)
	defer delete(b0)
	defer delete(b1)

	for bd in bit_depths {
		for tx_size in 0 ..< TX_SIZES_ALL {
			w := int(TX_WIDTH[tx_size])
			h := int(TX_HEIGHT[tx_size])
			n := w * h
			for dc in test_dcs {
				for i in 0 ..< n {
					b0[i] = 0
					b1[i] = 0
				}
				b0[0] = dc
				b1[0] = dc
				inverse_transform_2d(b0[:n], tx_size, DCT_DCT, false, 1, bd)
				inverse_transform_2d(b1[:n], tx_size, DCT_DCT, false, 2, bd)
				for i in 0 ..< n {
					if b0[i] != b1[i] {
						testing.expectf(t, false, "DC fast path mismatch tx_size=%d bd=%d dc=%d i=%d fast=%d full=%d", tx_size, bd, dc, i, b0[i], b1[i])
						return
					}
				}
			}
		}
	}
}

@(test)
test_forward_inverse_roundtrip_all_sizes_and_types :: proc(t: ^testing.T) {
	orig := make([]i32, 64 * 64)
	work := make([]i32, 64 * 64)
	defer delete(orig)
	defer delete(work)

	bit_depths := [3]int{8, 10, 12}
	for bd in bit_depths {
		scale := i32(1) << uint(bd - 8)
		for tx_size in 0 ..< TX_SIZES_ALL {
			w := int(TX_WIDTH[tx_size])
			h := int(TX_HEIGHT[tx_size])
			max_dim := max(w, h)
			for tx_type in 0 ..< TX_TYPES {
				// Per AV1 spec: max_dim == 64 only uses DCT_DCT; max_dim == 32 only uses DCT_DCT or IDTX.
				if max_dim == 64 && tx_type != DCT_DCT do continue
				if max_dim == 32 && tx_type != DCT_DCT && tx_type != IDTX do continue
				if max_dim > 16 {
					kinds := TX_1D_TYPES[tx_type]
					if (w >= 32 && kinds[0] != .DCT && kinds[0] != .IDTX) ||
					   (h >= 32 && kinds[1] != .DCT && kinds[1] != .IDTX) {
						continue
					}
				}

				// Construct a smooth low-frequency 2-D residual signal so that
				// 64-point transforms (which zero high frequencies >= 32) also reconstruct cleanly.
				for y in 0 ..< h {
					for x in 0 ..< w {
						// Smooth bilinear ramp in [-120 * scale, +120 * scale].
						vx := (2 * x - w + 1) * 60 / w
						vy := (2 * y - h + 1) * 60 / h
						v := i32(vx + vy) * scale
						orig[y * w + x] = v
						work[y * w + x] = v
					}
				}

				forward_transform_2d(work[:w * h], tx_size, tx_type, false, bd)
				inverse_transform_2d(work[:w * h], tx_size, tx_type, false, 1024, bd)

				max_err := i32(0)
				for i in 0 ..< w * h {
					err := abs(work[i] - orig[i])
					if err > max_err do max_err = err
				}
				tol := (max_dim == 64 ? i32(6) : i32(3)) * scale
				if max_err > tol {
					testing.expectf(t, false, "roundtrip error too high: tx_size=%d (%dx%d) tx_type=%d bd=%d max_err=%d tol=%d", tx_size, w, h, tx_type, bd, max_err, tol)
					return
				}
			}
		}
	}
}
