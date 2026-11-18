package vp9

import "core:testing"

@(test)
forward_inverse_pair :: proc(t: ^testing.T) {
	f: Forward_Transforms
	init_forward_transforms(&f)
	for tx in 0 ..< 4 {
		end := tx == TX_32X32 ? 1 : 4
		n := 4 << uint(tx)
		for kind in 0 ..< end {
			storage, original: [1024]i32
			block := storage[:n * n]
			for &v, i in block {v = i32((i * 137 + i / n * 23) % 511 - 255); original[i] = v}
			forward_transform(&f, block, tx, kind, false)
			inverse_transform(block, uint(tx + 2), kind, false, n)
			for v, i in block do if !testing.expect(t, abs(v - original[i]) <= 2, "lossy analysis/synthesis error exceeds two samples") do return
		}
	}
	for pattern in 0 ..< 100 {
		b, original: [16]i32
		for &v, i in b {v = i32((i * 173 + pattern * 37) % 511 - 255); original[i] = v}
		forward_transform(&f, b[:], 0, 0, true)
		inverse_transform(b[:], 2, 0, true, 4)
		testing.expect_value(t, b, original)
	}
}

// The largest residuals, flat and alternating, give the same coefficients at
// 8 bits as in the 64-bit high-depth arithmetic: none overflows. A flat 32x32
// residual of -141 once gave a positive DC and stray AC coefficients.
@(test)
forward_extremes_do_not_overflow :: proc(t: ^testing.T) {
	f: Forward_Transforms
	for tx in 0 ..< 4 {
		n := 4 << uint(tx)
		for kind in 0 ..< (tx == TX_32X32 ? 1 : 4) do for checker in ([2]bool{false, true}) do for amp in ([4]i32{255, -255, 141, -141}) {
			narrow, wide: [1024]i32
			for y in 0 ..< n do for x in 0 ..< n do narrow[y * n + x] = checker && (x + y) & 1 == 1 ? -amp : amp
			wide = narrow
			forward_transform(&f, narrow[:n * n], tx, kind, false, 8)
			forward_transform(&f, wide[:n * n], tx, kind, false, 10)
			if !testing.expectf(t, narrow == wide, "tx %d type %d amplitude %d checker %v", tx, kind, amp, checker) do return
			if kind == DCT_DCT && !checker {
				testing.expectf(t, (narrow[0] > 0) == (amp > 0), "tx %d amplitude %d: DC %d", tx, amp, narrow[0])
				for v in narrow[1:n * n] do if !testing.expectf(t, v == 0, "tx %d amplitude %d: an AC coefficient in a flat block", tx, amp) do return
			}
		}
	}
}
