package vp9

import common "../../common"
import "core:slice"
import "core:testing"

@(test)
scans_are_permutations :: proc(t: ^testing.T) {
	scans := [?][]u16 {
		DEFAULT_SCAN_4X4[:], ROW_SCAN_4X4[:], COL_SCAN_4X4[:],
		DEFAULT_SCAN_8X8[:], ROW_SCAN_8X8[:], COL_SCAN_8X8[:],
		DEFAULT_SCAN_16X16[:], ROW_SCAN_16X16[:], COL_SCAN_16X16[:],
		DEFAULT_SCAN_32X32[:],
	}
	for scan in scans {
		seen: [1024]bool
		for p in scan {
			testing.expect(t, int(p) < len(scan) && !seen[p])
			seen[p] = true
		}
	}
}

@(test)
default_probabilities_are_valid :: proc(t: ^testing.T) {
	p: Probabilities
	default_probabilities(&p)
	for tx in 0 ..< TX_SIZES do for i in 0 ..< 2 do for j in 0 ..< 2 do for band in 0 ..< COEF_BANDS {
		for ctx in 0 ..< (band == 0 ? 3 : 6) do for node in 0 ..< 3 do testing.expect(t, p.coef[tx][i][j][band][ctx][node] > 0)
	}
	for row in KF_Y_MODE_PROBS do for probs in row do for v in probs do testing.expect(t, v > 0)
}

// A DC-only block reconstructs to a constant for every DCT size.
@(test)
dc_only_dct_is_flat :: proc(t: ^testing.T) {
	block: [32 * 32]i32
	for n in uint(2) ..= 5 {
		size := 1 << n
		coefs := block[:size * size]
		for &c in coefs do c = 0
		coefs[0] = 1000
		inverse_transform(coefs, n, DCT_DCT, false, 1)
		for v in coefs do testing.expect_value(t, v, coefs[0])
		testing.expect(t, coefs[0] != 0)
	}
}

@(test)
adaptation_without_counts_keeps_probabilities :: proc(t: ^testing.T) {
	p: Probabilities
	default_probabilities(&p)
	before := p
	counts: Counts
	adapt_coef_probs(&p, &counts, false, false)
	adapt_noncoef_probs(&p, &counts, true, true, true)
	testing.expect(t, p == before)
}

// The loop filter on a pool gives the frame raster order gives: on worker
// threads, and on a pool that runs its ranges one at a time from the last.
@(test)
loop_filter_frame_matches_raster_order :: proc(t: ^testing.T) {
	reversed :: proc(_: ^common.Work_Pool, count: int, task: common.Range_Proc, data: rawptr) {
		for i := count - 1; i >= 0; i -= 1 do task(data, i, i + 1)
	}
	check :: proc(t: ^testing.T, $T: typeid, bit_depth, sb_rows, sb_cols: int, threads: ^common.Work_Pool) {
		mi_rows, mi_cols := sb_rows * 8, sb_cols * 8
		cells := make([]Filter_Cell, mi_rows * mi_cols)
		defer delete(cells)
		seed: u32 = 12345
		next :: proc(seed: ^u32) -> int {
			seed^ = seed^ * 1664525 + 1013904223
			return int(seed^ >> 16)
		}
		for &c in cells do c = {level = u8(1 + next(&seed) % 63), mi_size = BLOCK_8X8, tx_size = {TX_4X4, TX_4X4}}
		source: [3][]T
		strides := [3]int{mi_cols * 8, mi_cols * 4, mi_cols * 4}
		for &p, i in source {
			p = make([]T, strides[i] * (i == 0 ? mi_rows * 8 : mi_rows * 4))
			for &s in p do s = T((128 + next(&seed) % 9) << uint(bit_depth - 8))
		}
		defer for p in source do delete(p)
		reverse_pool := common.Work_Pool{run = reversed}
		pools := [3]^common.Work_Pool{nil, threads, &reverse_pool}
		results: [3][3][]T
		defer for r in results do for p in r do delete(p)
		for pool, k in pools {
			for p, i in source do results[k][i] = slice.clone(p)
			f := Loop_Filter_Frame(T){planes = results[k], strides = strides, cells = cells, mi_rows = mi_rows, mi_cols = mi_cols, ss_x = 1, ss_y = 1, bit_depth = bit_depth, limits = filter_limits(0)}
			loop_filter_frame(&f, sb_rows, sb_cols, pool)
		}
		for i in 0 ..< 3 {
			testing.expect(t, !slice.equal(results[0][i], source[i]), "the filter changed nothing")
			testing.expect(t, slice.equal(results[1][i], results[0][i]), "worker threads differ from raster order")
			testing.expect(t, slice.equal(results[2][i], results[0][i]), "reversed ranges differ from raster order")
		}
	}
	pool: common.Static_Pool
	common.init_static_pool(&pool, 5)
	defer common.destroy_static_pool(&pool)
	for size in ([?][2]int{{5, 7}, {6, 1}, {2, 2}, {1, 4}, {9, 3}}) {
		for _ in 0 ..< 20 {
			check(t, u8, 8, size[0], size[1], &pool.work_pool)
			check(t, u16, 10, size[0], size[1], &pool.work_pool)
		}
	}
}
