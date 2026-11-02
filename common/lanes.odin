package common

import "core:simd"

// Swaps the rows and columns of an L x L tile held as L vectors of L lanes:
// lane j of the result's vector i is lane i of r[j]. Eight lanes: Odin's
// simd.interleave zips the rows into column order in one operation (register
// shuffles; a matrix transpose would go through memory). Four lanes: two
// rounds of pairwise shuffles stay within 128-bit lanes, where a single
// interleave crosses them and measured slower in H.264's 4x4 transform.
transpose :: #force_inline proc "contextless" (r: [$L]$V/#simd[L]$E) -> [L]V {
	when L == 4 {
		lo01, hi01 := simd.shuffle(r[0], r[1], 0, 4, 1, 5), simd.shuffle(r[0], r[1], 2, 6, 3, 7)
		lo23, hi23 := simd.shuffle(r[2], r[3], 0, 4, 1, 5), simd.shuffle(r[2], r[3], 2, 6, 3, 7)
		return {simd.shuffle(lo01, lo23, 0, 1, 4, 5), simd.shuffle(lo01, lo23, 2, 3, 6, 7), simd.shuffle(hi01, hi23, 0, 1, 4, 5), simd.shuffle(hi01, hi23, 2, 3, 6, 7)}
	} else when L == 8 {
		return transmute([8]V)simd.interleave(r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7])
	} else when L == 16 {
		H :: #simd[8]E
		tl, tr, bl, br: [8]H = ---, ---, ---, ---
		#unroll for i in 0 ..< 8 {
			tl[i] = simd.shuffle(r[i], r[i], 0, 1, 2, 3, 4, 5, 6, 7)
			tr[i] = simd.shuffle(r[i], r[i], 8, 9, 10, 11, 12, 13, 14, 15)
			bl[i] = simd.shuffle(r[8 + i], r[8 + i], 0, 1, 2, 3, 4, 5, 6, 7)
			br[i] = simd.shuffle(r[8 + i], r[8 + i], 8, 9, 10, 11, 12, 13, 14, 15)
		}
		tl = transpose(tl)
		tr = transpose(tr)
		bl = transpose(bl)
		br = transpose(br)
		out: [16]V = ---
		#unroll for i in 0 ..< 8 {
			out[i] = simd.shuffle(tl[i], bl[i], 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)
			out[8 + i] = simd.shuffle(tr[i], br[i], 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)
		}
		return out
	} else {
		#panic("transpose takes 4, 8 or 16 lanes")
	}
}
