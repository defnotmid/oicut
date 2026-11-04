package h265_decoder

import common "../../../common"
import enc "../encoder"
import "core:testing"

@(test)
test_parse_encoder_p_slice_header :: proc(t: ^testing.T) {
	cfg := enc.Config {
		width       = 16,
		height      = 16,
		qp          = 26,
		level_idc   = 120,
		forced_mode = -1,
		deblock     = true,
		sao         = true,
	}

	e: enc.Encoder
	enc.init_encoder(&e, cfg)
	defer enc.destroy_encoder(&e)

	out := encode_test_frames(&e, 2, 16, 16)
	defer delete(out)

	nals := split_annex_b(out[:])
	defer {
		for i in 0 ..< len(nals) do destroy_nal(&nals[i])
		delete(nals)
	}

	sps_nal := find_nal(nals[:], .SPS_NUT)
	pps_nal := find_nal(nals[:], .PPS_NUT)
	testing.expect(t, sps_nal != nil && pps_nal != nil)
	if sps_nal == nil || pps_nal == nil do return

	sps, sok := parse_sps(sps_nal.rbsp)
	defer {
		if sok do destroy_sps(&sps)
	}
	pps, pok := parse_pps(pps_nal.rbsp)
	defer {
		if pok do destroy_pps(&pps)
	}
	testing.expect(t, sok && pok)

	seen := 0
	p_nal: ^Nal_Unit
	for i in 0 ..< len(nals) {
		if nal_type_is_vcl(nals[i].type) {
			seen += 1
			if seen == 2 {
				p_nal = &nals[i]
				break
			}
		}
	}
	testing.expect(t, p_nal != nil)
	if p_nal == nil do return

	header, hok := parse_slice_header(p_nal.rbsp, p_nal.type, &sps, &pps)
	defer {
		if hok do destroy_slice_header(&header)
	}
	testing.expect(t, hok)
	testing.expect_value(t, header.slice_type, Slice_Type.P)
	testing.expect_value(t, header.first_slice_segment_in_pic, true)
	testing.expect_value(t, header.pic_order_cnt_lsb, u32(1))
	testing.expect_value(t, header.st_rps_from_sps, false)
	testing.expect_value(t, len(header.st_rps.delta_poc_s0), 1)
	testing.expect_value(t, header.st_rps.delta_poc_s0[0], i32(-1))
	testing.expect_value(t, header.st_rps.used_s0[0], true)
	testing.expect_value(t, header.num_poc_total_curr, u32(1))
	testing.expect_value(t, header.sao_luma, true)
	testing.expect_value(t, header.sao_chroma, true)
	testing.expect_value(t, header.num_ref_idx_l0_active, u32(1))
	testing.expect_value(t, header.max_num_merge_cand, u32(5))
	testing.expect_value(t, header.qp_y, i32(26))
}

// ---------------------------------------------------------------------------
// pred_weight_table (7.3.6.3). A B slice with one reference per list and both
// luma and chroma weighting signalled.
// ---------------------------------------------------------------------------

@(test)
test_pred_weight_table :: proc(t: ^testing.T) {
	s := SPS {
		chroma_format_idc      = 1,
		bit_depth_chroma       = 8,
		high_precision_offsets = false,
	}

	h := Slice_Header {
		slice_type            = .B,
		num_ref_idx_l0_active = 1,
		num_ref_idx_l1_active = 1,
	}
	defer destroy_slice_header(&h)

	rbsp: [dynamic]u8
	defer delete(rbsp)

	w := common.init_bit_writer(&rbsp)
	common.write_ue(&w, 0) // luma_log2_weight_denom
	common.write_se(&w, 0) // delta_chroma_log2_weight_denom

	// List 0: luma and chroma weights present.
	common.write_bits(&w, 1, 1) // luma_weight_l0_flag
	common.write_bits(&w, 1, 1) // chroma_weight_l0_flag
	common.write_se(&w, 2) // delta_luma_weight_l0: 1 + 2 = 3
	common.write_se(&w, 1) // luma_offset_l0
	common.write_se(&w, 0) // delta_chroma_weight_l0[0]
	common.write_se(&w, 1) // delta_chroma_offset_l0[0]
	common.write_se(&w, 0) // delta_chroma_weight_l0[1]
	common.write_se(&w, -1) // delta_chroma_offset_l0[1]

	// List 1: no flags, so the defaults apply.
	common.write_bits(&w, 0, 1)
	common.write_bits(&w, 0, 1)
	common.write_rbsp_trailing_bits(&w)

	c: Get_Bits
	get_bits_init(&c, rbsp[:])

	ok := parse_pred_weight_table(&c, &h, &s)
	testing.expect(t, ok)

	testing.expect_value(t, h.weights.luma_log2_denom, u8(0))
	testing.expect_value(t, h.weights.chroma_log2_denom, u8(0))
	testing.expect_value(t, h.weights.luma_weight[0][0], i16(3))
	testing.expect_value(t, h.weights.luma_offset[0][0], i16(1))
	testing.expect_value(t, h.weights.chroma_weight[0][0], [2]i16{1, 1})
	testing.expect_value(t, h.weights.chroma_offset[0][0], [2]i16{1, -1})
	testing.expect_value(t, h.weights.luma_weight[1][0], i16(1))
	testing.expect_value(t, h.weights.chroma_weight[1][0], [2]i16{1, 1})
}

// CuQpOffsetCb and CuQpOffsetCr do not cross into the next slice: one with the
// offsets off would scale its chroma with the last slice's.
@(test)
test_slice_start_clears_cu_chroma_offsets :: proc(t: ^testing.T) {
	cfg := enc.Config {
		width       = 128,
		height      = 64,
		qp          = 26,
		level_idc   = 120,
		forced_mode = -1,
	}
	e: enc.Encoder
	enc.init_encoder(&e, cfg)
	stream := encode_test_frames(&e, 1, 128, 64)
	enc.destroy_encoder(&e)
	defer delete(stream)
	nals := split_annex_b(stream[:])
	defer {
		for &nal in nals do destroy_nal(&nal)
		delete(nals)
	}
	d: Decoder
	init_decoder(&d)
	defer destroy_decoder(&d)
	out: [dynamic]^Picture
	defer {
		for pic in out do release_picture(&d, pic)
		delete(out)
	}
	for &nal in nals do testing.expect(t, decode_nal(&d, nal, &out))
	ctu := d.ctu.(^Ctu_Decoder(u8))
	ctu.cu_cb_offset, ctu.cu_cr_offset = 6, -6
	next := Slice_Header {
		slice_type            = .I,
		slice_segment_address = 1,
		data_offset           = 1,
	}
	decode_slice_data(ctu, {rbsp = nil}, &next)
	testing.expect_value(t, ctu.cu_cb_offset, i32(0))
	testing.expect_value(t, ctu.cu_cr_offset, i32(0))
}
