package av1

import "core:testing"

@(test)
test_intra_prediction :: proc(t: ^testing.T) {
	plane: [32 * 32]u8
	for y in 0 ..< 32 {
		for x in 0 ..< 32 {
			plane[y * 32 + x] = u8((x * 7 + y * 5 + 40) & 255)
		}
	}

	// Test all 13 nominal intra modes with angle deltas on an 8x8 block at (4, 4).
	for mode in DC_PRED ..= PAETH_PRED {
		e := Intra_Edges{
			have_left                = true,
			have_above               = true,
			have_above_right         = true,
			have_below_left          = true,
			max_x                    = 31,
			max_y                    = 31,
			enable_intra_edge_filter = true,
			filter_type              = 0,
			angle_delta              = is_directional_mode(mode) ? 2 : 0,
			use_filter_intra         = false,
			filter_intra_mode        = 0,
		}
		predict_intra(plane[:], 32, 4, 4, TX_8X8, mode, e, 8)
		for y in 4 ..< 12 {
			for x in 4 ..< 12 {
				testing.expect(t, plane[y * 32 + x] <= 255)
			}
		}
	}

	// Test all 5 filter intra modes.
	for fi_mode in 0 ..< INTRA_FILTER_MODES {
		e := Intra_Edges{
			have_left         = true,
			have_above        = true,
			max_x             = 31,
			max_y             = 31,
			use_filter_intra  = true,
			filter_intra_mode = fi_mode,
		}
		predict_intra(plane[:], 32, 4, 4, TX_8X8, DC_PRED, e, 8)
	}

	// Test CfL prediction.
	luma: [16 * 16]u8
	chroma: [16 * 16]u8
	for y in 0 ..< 16 {
		for x in 0 ..< 16 {
			luma[y * 16 + x] = u8(clamp(100 + (x - 8) * 4 + (y - 8) * 2, 0, 255))
			chroma[y * 16 + x] = 128
		}
	}
	predict_chroma_from_luma(chroma[:], 16, luma[:], 16, 0, 0, TX_8X8, 8, 1, 1, 16, 16, 8)
	testing.expectf(t, chroma[0] < 128 && chroma[7 * 16 + 7] > 128, "CfL expected gradient, got %d and %d", chroma[0], chroma[7 * 16 + 7])
}

@(test)
test_inter_prediction_and_warping :: proc(t: ^testing.T) {
	ref_buf: [64 * 64]u8
	for y in 0 ..< 64 {
		for x in 0 ..< 64 {
			ref_buf[y * 64 + x] = u8((x * 3 + y * 5) & 255)
		}
	}
	ref := Ref_Plane(u8){
		data           = ref_buf[:],
		stride         = 64,
		last_x         = 63,
		last_y         = 63,
		ref_width      = 64,
		ref_height     = 64,
		upscaled_width = 64,
	}

	// Unscaled integer MV == exact copy of displaced block.
	dst: [16 * 16]u8
	sx, sy, step_x, step_y := scale_mv_for_block(0, 0, {16 * 8, 16 * 8}, 0, 0, 64, 64, 64, 64)
	predict_inter_single(
		dst[:], 16,
		0, 0, 8, 8,
		&ref,
		sx, sy, step_x, step_y,
		EIGHTTAP_REGULAR, EIGHTTAP_REGULAR,
		8,
	)
	for y in 0 ..< 8 {
		for x in 0 ..< 8 {
			testing.expect_value(t, dst[y * 16 + x], ref_buf[(16 + y) * 64 + (16 + x)])
		}
	}

	// Compound average of identical subpel predictions == single prediction.
	sx2, sy2, stx2, sty2 := scale_mv_for_block(0, 0, {16 * 8 + 3, 16 * 8 + 5}, 0, 0, 64, 64, 64, 64)
	pred0: [8 * 8]i32
	pred1: [8 * 8]i32
	convolve_2d_pred(pred0[:], 8, &ref, sx2, sy2, stx2, sty2, 8, 8, EIGHTTAP_REGULAR, EIGHTTAP_SMOOTH, true, 8)
	pred1 = pred0
	comp_dst: [16 * 16]u8
	blend_compound_average(comp_dst[:], 16, 0, 0, 8, 8, pred0[:], pred1[:], 8, 8)
	predict_inter_single(dst[:], 16, 0, 0, 8, 8, &ref, sx2, sy2, stx2, sty2, EIGHTTAP_REGULAR, EIGHTTAP_SMOOTH, 8)
	for y in 0 ..< 8 {
		for x in 0 ..< 8 {
			testing.expect_value(t, comp_dst[y * 16 + x], dst[y * 16 + x])
		}
	}

	// Identity affine warp: params = [0, 0, 1<<16, 0, 0, 1<<16].
	gm_id := [6]i32{0, 0, 1 << WARPEDMODEL_PREC_BITS, 0, 0, 1 << WARPEDMODEL_PREC_BITS}
	valid, alpha, beta, gamma, delta := setup_shear(gm_id)
	testing.expect(t, valid)
	testing.expect_value(t, alpha, i32(0))
	testing.expect_value(t, beta, i32(0))
	testing.expect_value(t, gamma, i32(0))
	testing.expect_value(t, delta, i32(0))
	warp_plane_pred(pred0[:], 8, &ref, 16, 16, 8, 8, 0, 0, gm_id, false, 8)
	for y in 0 ..< 8 {
		for x in 0 ..< 8 {
			testing.expect_value(t, u8(clamp(pred0[y * 8 + x], 0, 255)), ref_buf[(16 + y) * 64 + (16 + x)])
		}
	}
}

@(test)
test_mvref_and_mfmv_soa :: proc(t: ^testing.T) {
	mi_rows, mi_cols :: 8, 8
	cells := make(#soa[]MI_Cell, mi_rows * mi_cols)
	defer delete(cells)
	for i in 0 ..< len(cells) {
		cells[i].ref_frames = {NONE, NONE}
	}
	grid := MI_Grid{
		cells        = cells,
		mi_rows      = mi_rows,
		mi_cols      = mi_cols,
		mi_row_start = 0,
		mi_row_end   = mi_rows,
		mi_col_start = 0,
		mi_col_end   = mi_cols,
	}

	// Populate above neighbor (row 1, col 2..3) and left neighbor (row 2..3, col 1).
	for c in 2 ..< 4 {
		idx := 1 * mi_cols + c
		grid.cells[idx] = MI_Cell{
			mvs        = {{16, -24}, {0, 0}},
			ref_frames = {LAST_FRAME, NONE},
			mi_size    = BLOCK_8X8,
			y_mode     = NEWMV,
			is_inter   = true,
		}
	}
	for r in 2 ..< 4 {
		idx := r * mi_cols + 1
		grid.cells[idx] = MI_Cell{
			mvs        = {{8, 12}, {0, 0}},
			ref_frames = {LAST_FRAME, NONE},
			mi_size    = BLOCK_8X8,
			y_mode     = NEARESTMV,
			is_inter   = true,
		}
	}

	st: MV_Stack
	gm_type: [TOTAL_REFS_PER_FRAME]u8
	gm_params: [TOTAL_REFS_PER_FRAME][6]i32
	sign_bias: [TOTAL_REFS_PER_FRAME]u8
	find_mv_stack(
		&st, &grid,
		2, 2, BLOCK_8X8,
		{LAST_FRAME, NONE},
		false,
		true, false,
		false, nil,
		gm_type, gm_params, sign_bias,
	)

	testing.expect_value(t, st.num_mv_found, 2)
	testing.expect_value(t, st.ref_stack_mv[0][0], Mv{16, -24})
	testing.expect_value(t, st.ref_stack_mv[1][0], Mv{8, 12})
	testing.expect_value(t, st.close_matches, 2)
	testing.expect_value(t, st.new_mv_context, 4)
	testing.expect_value(t, st.ref_mv_context, 5)
	testing.expect(t, has_overlappable_candidates(&grid, 2, 2, BLOCK_8X8, true, true))

	cand_list: [LEAST_SQUARES_SAMPLES_MAX][4]i32
	ns := find_warp_samples(&cand_list, &grid, 2, 2, BLOCK_8X8, true, true, LAST_FRAME, Mv{16, -24})
	testing.expect(t, ns >= 1)
}
