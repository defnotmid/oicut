package av1

// Frame and tile CDF context state (sections 6.8.2, 8.2.2, 8.2.4).

CDF_Context :: struct {
	// Intra mode & angle CDFs.
	intra_frame_y_mode_cdf:      [INTRA_MODE_CONTEXTS][INTRA_MODE_CONTEXTS][INTRA_MODES + 1]u16,
	y_mode_cdf:                  [BLOCK_SIZE_GROUPS][INTRA_MODES + 1]u16,
	uv_mode_cfl_not_allowed_cdf: [INTRA_MODES][UV_INTRA_MODES_CFL_NOT_ALLOWED + 1]u16,
	uv_mode_cfl_allowed_cdf:     [INTRA_MODES][UV_INTRA_MODES_CFL_ALLOWED + 1]u16,
	angle_delta_cdf:             [DIRECTIONAL_MODES][(2 * MAX_ANGLE_DELTA + 1) + 1]u16,
	intrabc_cdf:                 [3]u16,
	filter_intra_mode_cdf:       [INTRA_FILTER_MODES + 1]u16,
	filter_intra_cdf:            [BLOCK_SIZES][3]u16,
	cfl_sign_cdf:                [CFL_JOINT_SIGNS + 1]u16,
	cfl_alpha_cdf:               [CFL_ALPHA_CONTEXTS][CFL_ALPHABET_SIZE + 1]u16,

	// Partition & segmentation CDFs.
	partition_w8_cdf:            [PARTITION_CONTEXTS][5]u16,
	partition_w16_cdf:           [PARTITION_CONTEXTS][11]u16,
	partition_w32_cdf:           [PARTITION_CONTEXTS][11]u16,
	partition_w64_cdf:           [PARTITION_CONTEXTS][11]u16,
	partition_w128_cdf:          [PARTITION_CONTEXTS][9]u16,
	segment_id_cdf:              [SEGMENT_ID_CONTEXTS][MAX_SEGMENTS + 1]u16,
	segment_id_predicted_cdf:    [SEGMENT_ID_PREDICTED_CONTEXTS][3]u16,

	// Transform size & type CDFs.
	tx_8x8_cdf:                  [TX_SIZE_CONTEXTS][3]u16,
	tx_16x16_cdf:                [TX_SIZE_CONTEXTS][4]u16,
	tx_32x32_cdf:                [TX_SIZE_CONTEXTS][4]u16,
	tx_64x64_cdf:                [TX_SIZE_CONTEXTS][4]u16,
	txfm_split_cdf:              [TXFM_PARTITION_CONTEXTS][3]u16,
	intra_tx_type_set1_cdf:      [2][INTRA_MODES][8]u16,
	intra_tx_type_set2_cdf:      [3][INTRA_MODES][6]u16,
	inter_tx_type_set1_cdf:      [2][17]u16,
	inter_tx_type_set2_cdf:      [13]u16,
	inter_tx_type_set3_cdf:      [4][3]u16,

	// Inter mode, reference & compound CDFs.
	interp_filter_cdf:           [INTERP_FILTER_CONTEXTS][INTERP_FILTERS + 1]u16,
	motion_mode_cdf:             [BLOCK_SIZES][MOTION_MODES + 1]u16,
	new_mv_cdf:                  [NEW_MV_CONTEXTS][3]u16,
	zero_mv_cdf:                 [ZERO_MV_CONTEXTS][3]u16,
	ref_mv_cdf:                  [REF_MV_CONTEXTS][3]u16,
	compound_mode_cdf:           [COMPOUND_MODE_CONTEXTS][COMPOUND_MODES + 1]u16,
	drl_mode_cdf:                [DRL_MODE_CONTEXTS][3]u16,
	is_inter_cdf:                [IS_INTER_CONTEXTS][3]u16,
	comp_mode_cdf:               [COMP_INTER_CONTEXTS][3]u16,
	skip_mode_cdf:               [SKIP_MODE_CONTEXTS][3]u16,
	skip_cdf:                    [SKIP_CONTEXTS][3]u16,
	comp_ref_cdf:                [REF_CONTEXTS][FWD_REFS - 1][3]u16,
	comp_bwd_ref_cdf:            [REF_CONTEXTS][BWD_REFS - 1][3]u16,
	single_ref_cdf:              [REF_CONTEXTS][SINGLE_REFS - 1][3]u16,
	comp_ref_type_cdf:           [COMP_REF_TYPE_CONTEXTS][3]u16,
	uni_comp_ref_cdf:            [REF_CONTEXTS][UNIDIR_COMP_REFS - 1][3]u16,
	comp_group_idx_cdf:          [COMP_GROUP_IDX_CONTEXTS][3]u16,
	compound_idx_cdf:            [COMPOUND_IDX_CONTEXTS][3]u16,
	compound_type_cdf:           [BLOCK_SIZES][COMPOUND_TYPES + 1]u16,
	inter_intra_cdf:             [BLOCK_SIZE_GROUPS - 1][3]u16,
	inter_intra_mode_cdf:        [BLOCK_SIZE_GROUPS - 1][INTERINTRA_MODES + 1]u16,
	wedge_index_cdf:             [BLOCK_SIZES][17]u16,
	wedge_inter_intra_cdf:       [BLOCK_SIZES][3]u16,
	use_obmc_cdf:                [BLOCK_SIZES][3]u16,

	// Motion vector component CDFs (indexed by 0 = inter, 1 = intrabc).
	mv_joint_cdf:                [MV_CONTEXTS][MV_JOINTS + 1]u16,
	mv_class_cdf:                [MV_CONTEXTS][2][MV_CLASSES + 1]u16,
	mv_class0_bit_cdf:           [MV_CONTEXTS][2][3]u16,
	mv_fr_cdf:                   [MV_CONTEXTS][2][MV_JOINTS + 1]u16,
	mv_class0_fr_cdf:            [MV_CONTEXTS][2][CLASS0_SIZE][MV_JOINTS + 1]u16,
	mv_class0_hp_cdf:            [MV_CONTEXTS][2][3]u16,
	mv_sign_cdf:                 [MV_CONTEXTS][2][3]u16,
	mv_bit_cdf:                  [MV_CONTEXTS][2][MV_OFFSET_BITS][3]u16,
	mv_hp_cdf:                   [MV_CONTEXTS][2][3]u16,

	// Palette CDFs.
	palette_y_mode_cdf:          [PALETTE_BLOCK_SIZE_CONTEXTS][PALETTE_Y_MODE_CONTEXTS][3]u16,
	palette_uv_mode_cdf:         [PALETTE_UV_MODE_CONTEXTS][3]u16,
	palette_y_size_cdf:          [PALETTE_BLOCK_SIZE_CONTEXTS][PALETTE_SIZES + 1]u16,
	palette_uv_size_cdf:         [PALETTE_BLOCK_SIZE_CONTEXTS][PALETTE_SIZES + 1]u16,
	palette_size_2_y_color_cdf:  [PALETTE_COLOR_CONTEXTS][3]u16,
	palette_size_2_uv_color_cdf: [PALETTE_COLOR_CONTEXTS][3]u16,
	palette_size_3_y_color_cdf:  [PALETTE_COLOR_CONTEXTS][4]u16,
	palette_size_3_uv_color_cdf: [PALETTE_COLOR_CONTEXTS][4]u16,
	palette_size_4_y_color_cdf:  [PALETTE_COLOR_CONTEXTS][5]u16,
	palette_size_4_uv_color_cdf: [PALETTE_COLOR_CONTEXTS][5]u16,
	palette_size_5_y_color_cdf:  [PALETTE_COLOR_CONTEXTS][6]u16,
	palette_size_5_uv_color_cdf: [PALETTE_COLOR_CONTEXTS][6]u16,
	palette_size_6_y_color_cdf:  [PALETTE_COLOR_CONTEXTS][7]u16,
	palette_size_6_uv_color_cdf: [PALETTE_COLOR_CONTEXTS][7]u16,
	palette_size_7_y_color_cdf:  [PALETTE_COLOR_CONTEXTS][8]u16,
	palette_size_7_uv_color_cdf: [PALETTE_COLOR_CONTEXTS][8]u16,
	palette_size_8_y_color_cdf:  [PALETTE_COLOR_CONTEXTS][9]u16,
	palette_size_8_uv_color_cdf: [PALETTE_COLOR_CONTEXTS][9]u16,

	// Delta-Q, delta-LF & loop restoration CDFs.
	delta_q_cdf:                 [DELTA_Q_SMALL + 2]u16,
	delta_lf_cdf:                [DELTA_LF_SMALL + 2]u16,
	delta_lf_multi_cdf:          [FRAME_LF_COUNT][DELTA_LF_SMALL + 2]u16,
	use_wiener_cdf:              [3]u16,
	use_sgrproj_cdf:             [3]u16,
	restoration_type_cdf:        [RESTORE_SWITCHABLE + 1]u16,

	// Transform coefficient CDFs (quantizer-context dependent).
	txb_skip_cdf:                [TX_SIZES][TXB_SKIP_CONTEXTS][3]u16,
	eob_pt_16_cdf:               [PLANE_TYPES][2][6]u16,
	eob_pt_32_cdf:               [PLANE_TYPES][2][7]u16,
	eob_pt_64_cdf:               [PLANE_TYPES][2][8]u16,
	eob_pt_128_cdf:              [PLANE_TYPES][2][9]u16,
	eob_pt_256_cdf:              [PLANE_TYPES][2][10]u16,
	eob_pt_512_cdf:              [PLANE_TYPES][11]u16,
	eob_pt_1024_cdf:             [PLANE_TYPES][12]u16,
	eob_extra_cdf:               [TX_SIZES][PLANE_TYPES][EOB_COEF_CONTEXTS][3]u16,
	dc_sign_cdf:                 [PLANE_TYPES][DC_SIGN_CONTEXTS][3]u16,
	coeff_base_eob_cdf:          [TX_SIZES][PLANE_TYPES][SIG_COEF_CONTEXTS_EOB][4]u16,
	coeff_base_cdf:              [TX_SIZES][PLANE_TYPES][SIG_COEF_CONTEXTS][5]u16,
	coeff_br_cdf:                [TX_SIZES][PLANE_TYPES][LEVEL_CONTEXTS][BR_CDF_SIZE + 1]u16,
	// Trailing 16-byte SIMD padding so 8-lane and 16-lane `u16` loads/stores in
	// `read_symbol_multi` never read or write past the end of `CDF_Context`.
	_:                           [8]u16,
}

// 6.8.2: initializes all non-coefficient CDF tables to their spec defaults.
init_non_coeff_cdfs :: proc(c: ^CDF_Context) {
	c.intra_frame_y_mode_cdf = DEFAULT_INTRA_FRAME_Y_MODE_CDF
	c.y_mode_cdf = DEFAULT_Y_MODE_CDF
	c.uv_mode_cfl_not_allowed_cdf = DEFAULT_UV_MODE_CFL_NOT_ALLOWED_CDF
	c.uv_mode_cfl_allowed_cdf = DEFAULT_UV_MODE_CFL_ALLOWED_CDF
	c.angle_delta_cdf = DEFAULT_ANGLE_DELTA_CDF
	c.intrabc_cdf = DEFAULT_INTRABC_CDF
	c.filter_intra_mode_cdf = DEFAULT_FILTER_INTRA_MODE_CDF
	c.filter_intra_cdf = DEFAULT_FILTER_INTRA_CDF
	c.cfl_sign_cdf = DEFAULT_CFL_SIGN_CDF
	c.cfl_alpha_cdf = DEFAULT_CFL_ALPHA_CDF

	c.partition_w8_cdf = DEFAULT_PARTITION_W8_CDF
	c.partition_w16_cdf = DEFAULT_PARTITION_W16_CDF
	c.partition_w32_cdf = DEFAULT_PARTITION_W32_CDF
	c.partition_w64_cdf = DEFAULT_PARTITION_W64_CDF
	c.partition_w128_cdf = DEFAULT_PARTITION_W128_CDF
	c.segment_id_cdf = DEFAULT_SEGMENT_ID_CDF
	c.segment_id_predicted_cdf = DEFAULT_SEGMENT_ID_PREDICTED_CDF

	c.tx_8x8_cdf = DEFAULT_TX_8X8_CDF
	c.tx_16x16_cdf = DEFAULT_TX_16X16_CDF
	c.tx_32x32_cdf = DEFAULT_TX_32X32_CDF
	c.tx_64x64_cdf = DEFAULT_TX_64X64_CDF
	c.txfm_split_cdf = DEFAULT_TXFM_SPLIT_CDF
	c.intra_tx_type_set1_cdf = DEFAULT_INTRA_TX_TYPE_SET1_CDF
	c.intra_tx_type_set2_cdf = DEFAULT_INTRA_TX_TYPE_SET2_CDF
	c.inter_tx_type_set1_cdf = DEFAULT_INTER_TX_TYPE_SET1_CDF
	c.inter_tx_type_set2_cdf = DEFAULT_INTER_TX_TYPE_SET2_CDF
	c.inter_tx_type_set3_cdf = DEFAULT_INTER_TX_TYPE_SET3_CDF

	c.interp_filter_cdf = DEFAULT_INTERP_FILTER_CDF
	c.motion_mode_cdf = DEFAULT_MOTION_MODE_CDF
	c.new_mv_cdf = DEFAULT_NEW_MV_CDF
	c.zero_mv_cdf = DEFAULT_ZERO_MV_CDF
	c.ref_mv_cdf = DEFAULT_REF_MV_CDF
	c.compound_mode_cdf = DEFAULT_COMPOUND_MODE_CDF
	c.drl_mode_cdf = DEFAULT_DRL_MODE_CDF
	c.is_inter_cdf = DEFAULT_IS_INTER_CDF
	c.comp_mode_cdf = DEFAULT_COMP_MODE_CDF
	c.skip_mode_cdf = DEFAULT_SKIP_MODE_CDF
	c.skip_cdf = DEFAULT_SKIP_CDF
	c.comp_ref_cdf = DEFAULT_COMP_REF_CDF
	c.comp_bwd_ref_cdf = DEFAULT_COMP_BWD_REF_CDF
	c.single_ref_cdf = DEFAULT_SINGLE_REF_CDF
	c.comp_ref_type_cdf = DEFAULT_COMP_REF_TYPE_CDF
	c.uni_comp_ref_cdf = DEFAULT_UNI_COMP_REF_CDF
	c.comp_group_idx_cdf = DEFAULT_COMP_GROUP_IDX_CDF
	c.compound_idx_cdf = DEFAULT_COMPOUND_IDX_CDF
	c.compound_type_cdf = DEFAULT_COMPOUND_TYPE_CDF
	c.inter_intra_cdf = DEFAULT_INTER_INTRA_CDF
	c.inter_intra_mode_cdf = DEFAULT_INTER_INTRA_MODE_CDF
	c.wedge_index_cdf = DEFAULT_WEDGE_INDEX_CDF
	c.wedge_inter_intra_cdf = DEFAULT_WEDGE_INTER_INTRA_CDF
	c.use_obmc_cdf = DEFAULT_USE_OBMC_CDF

	for i in 0 ..< MV_CONTEXTS {
		c.mv_joint_cdf[i] = DEFAULT_MV_JOINT_CDF
		c.mv_class_cdf[i] = DEFAULT_MV_CLASS_CDF
		c.mv_fr_cdf[i] = DEFAULT_MV_FR_CDF
		c.mv_class0_fr_cdf[i] = DEFAULT_MV_CLASS0_FR_CDF
		for comp in 0 ..< 2 {
			c.mv_class0_bit_cdf[i][comp] = DEFAULT_MV_CLASS0_BIT_CDF
			c.mv_class0_hp_cdf[i][comp] = DEFAULT_MV_CLASS0_HP_CDF
			c.mv_sign_cdf[i][comp] = DEFAULT_MV_SIGN_CDF
			c.mv_bit_cdf[i][comp] = DEFAULT_MV_BIT_CDF
			c.mv_hp_cdf[i][comp] = DEFAULT_MV_HP_CDF
		}
	}

	c.palette_y_mode_cdf = DEFAULT_PALETTE_Y_MODE_CDF
	c.palette_uv_mode_cdf = DEFAULT_PALETTE_UV_MODE_CDF
	c.palette_y_size_cdf = DEFAULT_PALETTE_Y_SIZE_CDF
	c.palette_uv_size_cdf = DEFAULT_PALETTE_UV_SIZE_CDF
	c.palette_size_2_y_color_cdf = DEFAULT_PALETTE_SIZE_2_Y_COLOR_CDF
	c.palette_size_2_uv_color_cdf = DEFAULT_PALETTE_SIZE_2_UV_COLOR_CDF
	c.palette_size_3_y_color_cdf = DEFAULT_PALETTE_SIZE_3_Y_COLOR_CDF
	c.palette_size_3_uv_color_cdf = DEFAULT_PALETTE_SIZE_3_UV_COLOR_CDF
	c.palette_size_4_y_color_cdf = DEFAULT_PALETTE_SIZE_4_Y_COLOR_CDF
	c.palette_size_4_uv_color_cdf = DEFAULT_PALETTE_SIZE_4_UV_COLOR_CDF
	c.palette_size_5_y_color_cdf = DEFAULT_PALETTE_SIZE_5_Y_COLOR_CDF
	c.palette_size_5_uv_color_cdf = DEFAULT_PALETTE_SIZE_5_UV_COLOR_CDF
	c.palette_size_6_y_color_cdf = DEFAULT_PALETTE_SIZE_6_Y_COLOR_CDF
	c.palette_size_6_uv_color_cdf = DEFAULT_PALETTE_SIZE_6_UV_COLOR_CDF
	c.palette_size_7_y_color_cdf = DEFAULT_PALETTE_SIZE_7_Y_COLOR_CDF
	c.palette_size_7_uv_color_cdf = DEFAULT_PALETTE_SIZE_7_UV_COLOR_CDF
	c.palette_size_8_y_color_cdf = DEFAULT_PALETTE_SIZE_8_Y_COLOR_CDF
	c.palette_size_8_uv_color_cdf = DEFAULT_PALETTE_SIZE_8_UV_COLOR_CDF

	c.delta_q_cdf = DEFAULT_DELTA_Q_CDF
	c.delta_lf_cdf = DEFAULT_DELTA_LF_CDF
	for i in 0 ..< FRAME_LF_COUNT do c.delta_lf_multi_cdf[i] = DEFAULT_DELTA_LF_CDF
	c.use_wiener_cdf = DEFAULT_USE_WIENER_CDF
	c.use_sgrproj_cdf = DEFAULT_USE_SGRPROJ_CDF
	c.restoration_type_cdf = DEFAULT_RESTORATION_TYPE_CDF
}

// 6.8.2: initializes the transform coefficient CDF tables for `base_q_idx`.
init_coeff_cdfs :: proc(c: ^CDF_Context, base_q_idx: u8) {
	idx := 3
	switch {
	case base_q_idx <= 20:  idx = 0
	case base_q_idx <= 60:  idx = 1
	case base_q_idx <= 120: idx = 2
	}
	c.txb_skip_cdf = DEFAULT_TXB_SKIP_CDF[idx]
	c.eob_pt_16_cdf = DEFAULT_EOB_PT_16_CDF[idx]
	c.eob_pt_32_cdf = DEFAULT_EOB_PT_32_CDF[idx]
	c.eob_pt_64_cdf = DEFAULT_EOB_PT_64_CDF[idx]
	c.eob_pt_128_cdf = DEFAULT_EOB_PT_128_CDF[idx]
	c.eob_pt_256_cdf = DEFAULT_EOB_PT_256_CDF[idx]
	c.eob_pt_512_cdf = DEFAULT_EOB_PT_512_CDF[idx]
	c.eob_pt_1024_cdf = DEFAULT_EOB_PT_1024_CDF[idx]
	c.eob_extra_cdf = DEFAULT_EOB_EXTRA_CDF[idx]
	c.dc_sign_cdf = DEFAULT_DC_SIGN_CDF[idx]
	c.coeff_base_eob_cdf = DEFAULT_COEFF_BASE_EOB_CDF[idx]
	c.coeff_base_cdf = DEFAULT_COEFF_BASE_CDF[idx]
	c.coeff_br_cdf = DEFAULT_COEFF_BR_CDF[idx]
}

// Initializes all CDF tables for a frame with PRIMARY_REF_NONE.
init_cdf_context :: proc(c: ^CDF_Context, base_q_idx: u8) {
	init_non_coeff_cdfs(c)
	init_coeff_cdfs(c, base_q_idx)
}

// 6.8.2 load_cdfs: copies `src` into `dst`, resets intra_frame_y_mode_cdf to
// its default, and zeroes the symbol count slot (the entry after 32768) of
// every CDF row.
load_cdfs :: proc(dst: ^CDF_Context, src: ^CDF_Context) #no_bounds_check {
	dst^ = src^
	dst.intra_frame_y_mode_cdf = DEFAULT_INTRA_FRAME_Y_MODE_CDF
	words := (^[size_of(CDF_Context) / size_of(u16)]u16)(dst)
	for i in 1 ..< len(words) {
		if words[i - 1] == 32768 do words[i] = 0
	}
}
