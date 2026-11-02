// AV1 bitstream syntax for hardware decoding (AV1 Bitstream & Decoding
// Process Specification, sections 5 and 6): OBUs, the sequence header and
// the uncompressed frame header with the reference state it depends on
// (7.20/7.21), and tile group layout. Pixels are decoded elsewhere (Vulkan
// Video); this package says what each frame is and where its tiles are.
// Depends only on Odin core.
package av1

Error :: enum {
	None,
	Invalid_Data, // malformed or out of spec
	Unsupported,  // valid, outside the implemented scope (tile lists, large scale tile)
}

NUM_REF_FRAMES :: 8
REFS_PER_FRAME :: 7
PRIMARY_REF_NONE :: 7
MAX_TILE_COLS :: 64
MAX_TILE_ROWS :: 64
MAX_SEGMENTS :: 8
SEG_LVL_MAX :: 8

// Reference frame names (ref frame index 0 is INTRA_FRAME).
NONE :: -1
NONE_FRAME :: -1
INTRA_FRAME :: 0
LAST_FRAME :: 1
LAST2_FRAME :: 2
LAST3_FRAME :: 3
GOLDEN_FRAME :: 4
BWDREF_FRAME :: 5
ALTREF2_FRAME :: 6
ALTREF_FRAME :: 7
TOTAL_REFS_PER_FRAME :: 8

// Frame types (section 6.8.2).
KEY_FRAME :: 0
INTER_FRAME :: 1
INTRA_ONLY_FRAME :: 2
SWITCH_FRAME :: 3

// Dimensions & grid units (section 3).
MI_SIZE :: 4
MI_SIZE_LOG2 :: 2
MAX_SB_SIZE :: 128
BLOCK_SIZE_GROUPS :: 4
BLOCK_SIZES :: 22
BLOCK_INVALID :: 22
PARTITION_CONTEXTS :: 4
PLANE_TYPES :: 2
TX_SIZES :: 5
TX_SIZES_ALL :: 19
TX_SIZE_CONTEXTS :: 3
MAX_VARTX_DEPTH :: 2
TXFM_PARTITION_CONTEXTS :: 21
TX_MODES :: 3
ONLY_4X4 :: 0
TX_MODE_LARGEST :: 1
TX_MODE_SELECT :: 2

// Block sizes (6.10.4).
BLOCK_4X4 :: 0
BLOCK_4X8 :: 1
BLOCK_8X4 :: 2
BLOCK_8X8 :: 3
BLOCK_8X16 :: 4
BLOCK_16X8 :: 5
BLOCK_16X16 :: 6
BLOCK_16X32 :: 7
BLOCK_32X16 :: 8
BLOCK_32X32 :: 9
BLOCK_32X64 :: 10
BLOCK_64X32 :: 11
BLOCK_64X64 :: 12
BLOCK_64X128 :: 13
BLOCK_128X64 :: 14
BLOCK_128X128 :: 15
BLOCK_4X16 :: 16
BLOCK_16X4 :: 17
BLOCK_8X32 :: 18
BLOCK_32X8 :: 19
BLOCK_16X64 :: 20
BLOCK_64X16 :: 21

@(rodata)
BLOCK_WIDTH := [BLOCK_SIZES]u8{
	4, 4, 8, 8, 8, 16, 16, 16, 32, 32, 32, 64, 64, 64, 128, 128,
	4, 16, 8, 32, 16, 64,
}

@(rodata)
BLOCK_HEIGHT := [BLOCK_SIZES]u8{
	4, 8, 4, 8, 16, 8, 16, 32, 16, 32, 64, 32, 64, 128, 64, 128,
	16, 4, 32, 8, 64, 16,
}

// Section 5.9.3: get_relative_dist.
get_relative_dist :: #force_inline proc "contextless" (a, b: int, enable_order_hint: bool, order_hint_bits: u8) -> int {
	if !enable_order_hint || order_hint_bits == 0 do return 0
	diff := a - b
	m := 1 << uint(order_hint_bits - 1)
	return (diff & (m - 1)) - (diff & m)
}

// Partition types (6.10.4).
PARTITION_NONE :: 0
PARTITION_HORZ :: 1
PARTITION_VERT :: 2
PARTITION_SPLIT :: 3
PARTITION_HORZ_A :: 4
PARTITION_HORZ_B :: 5
PARTITION_VERT_A :: 6
PARTITION_VERT_B :: 7
PARTITION_HORZ_4 :: 8
PARTITION_VERT_4 :: 9
PARTITION_TYPES :: 10

// Transform sizes (6.10.16).
TX_4X4 :: 0
TX_8X8 :: 1
TX_16X16 :: 2
TX_32X32 :: 3
TX_64X64 :: 4
TX_4X8 :: 5
TX_8X4 :: 6
TX_8X16 :: 7
TX_16X8 :: 8
TX_16X32 :: 9
TX_32X16 :: 10
TX_32X64 :: 11
TX_64X32 :: 12
TX_4X16 :: 13
TX_16X4 :: 14
TX_8X32 :: 15
TX_32X8 :: 16
TX_16X64 :: 17
TX_64X16 :: 18

// Transform types & sets (section 3, 6.10.19).
DCT_DCT :: 0
ADST_DCT :: 1
DCT_ADST :: 2
ADST_ADST :: 3
FLIPADST_DCT :: 4
DCT_FLIPADST :: 5
FLIPADST_FLIPADST :: 6
ADST_FLIPADST :: 7
FLIPADST_ADST :: 8
IDTX :: 9
V_DCT :: 10
H_DCT :: 11
V_ADST :: 12
H_ADST :: 13
V_FLIPADST :: 14
H_FLIPADST :: 15
TX_TYPES :: 16

TX_SET_DCTONLY :: 0
TX_SET_INTRA_1 :: 1
TX_SET_INTRA_2 :: 2
TX_SET_INTER_1 :: 1
TX_SET_INTER_2 :: 2
TX_SET_INTER_3 :: 3
TX_SET_TYPES_INTRA :: 3
TX_SET_TYPES_INTER :: 4

TX_CLASS_2D :: 0
TX_CLASS_HORIZ :: 1
TX_CLASS_VERT :: 2

// Prediction modes (6.10.6, 6.10.22, 6.10.23).
DC_PRED :: 0
V_PRED :: 1
H_PRED :: 2
D45_PRED :: 3
D135_PRED :: 4
D113_PRED :: 5
D157_PRED :: 6
D203_PRED :: 7
D67_PRED :: 8
SMOOTH_PRED :: 9
SMOOTH_V_PRED :: 10
SMOOTH_H_PRED :: 11
PAETH_PRED :: 12
UV_CFL_PRED :: 13
NEARESTMV :: 14
NEARMV :: 15
GLOBALMV :: 16
NEWMV :: 17
NEAREST_NEARESTMV :: 18
NEAR_NEARMV :: 19
NEAREST_NEWMV :: 20
NEW_NEARESTMV :: 21
NEAR_NEWMV :: 22
NEW_NEARMV :: 23
GLOBAL_GLOBALMV :: 24
NEW_NEWMV :: 25

INTRA_MODES :: 13
UV_INTRA_MODES_CFL_NOT_ALLOWED :: 13
UV_INTRA_MODES_CFL_ALLOWED :: 14
INTRA_MODE_CONTEXTS :: 5
DIRECTIONAL_MODES :: 8
MAX_ANGLE_DELTA :: 3
ANGLE_STEP :: 3
INTRA_EDGE_KERNELS :: 3
INTRA_EDGE_TAPS :: 5
INTRA_FILTER_MODES :: 5
INTRA_FILTER_SCALE_BITS :: 4
FILTER_DC_PRED :: 0
FILTER_V_PRED :: 1
FILTER_H_PRED :: 2
FILTER_D157_PRED :: 3
FILTER_PAETH_PRED :: 4

CFL_JOINT_SIGNS :: 8
CFL_ALPHABET_SIZE :: 16
CFL_ALPHA_CONTEXTS :: 6

// Palette & screen content (section 3).
PALETTE_COLOR_CONTEXTS :: 5
PALETTE_MAX_COLOR_CONTEXT_HASH :: 8
PALETTE_BLOCK_SIZE_CONTEXTS :: 7
PALETTE_Y_MODE_CONTEXTS :: 3
PALETTE_UV_MODE_CONTEXTS :: 2
PALETTE_SIZES :: 7
PALETTE_COLORS :: 8
PALETTE_NUM_NEIGHBORS :: 3
INTRABC_DELAY_PIXELS :: 256
INTRABC_DELAY_SB64 :: 4

// Inter prediction & motion vectors (section 3, 6.10).
IS_INTER_CONTEXTS :: 4
SKIP_MODE_CONTEXTS :: 3
SKIP_CONTEXTS :: 3
REF_CONTEXTS :: 3
FWD_REFS :: 4
BWD_REFS :: 3
SINGLE_REFS :: 7
UNIDIR_COMP_REFS :: 4
SINGLE_REFERENCE :: 0
COMPOUND_REFERENCE :: 1
UNIDIR_COMP_REFERENCE :: 0
BIDIR_COMP_REFERENCE :: 1
COMP_INTER_CONTEXTS :: 5
COMP_REF_TYPE_CONTEXTS :: 5
COMP_GROUP_IDX_CONTEXTS :: 6
COMPOUND_IDX_CONTEXTS :: 6
COMPOUND_MODES :: 8
COMPOUND_MODE_CONTEXTS :: 8
COMP_NEWMV_CTXS :: 5
NEW_MV_CONTEXTS :: 6
ZERO_MV_CONTEXTS :: 2
REF_MV_CONTEXTS :: 6
DRL_MODE_CONTEXTS :: 3

MV_CONTEXTS :: 2
MV_INTRABC_CONTEXT :: 1
MV_JOINTS :: 4
MV_JOINT_ZERO :: 0
MV_JOINT_HNZVZ :: 1
MV_JOINT_HZVNZ :: 2
MV_JOINT_HNZVNZ :: 3
MV_CLASSES :: 11
CLASS0_SIZE :: 2
MV_OFFSET_BITS :: 10
MV_BORDER :: 128
MAX_REF_MV_STACK_SIZE :: 8
MFMV_STACK_SIZE :: 3
REF_CAT_LEVEL :: 640
REFMVS_LIMIT :: (1 << 12) - 1

INTERP_FILTERS :: 3
INTERP_FILTER_CONTEXTS :: 16
EIGHTTAP_REGULAR :: 0
EIGHTTAP_SMOOTH :: 1
EIGHTTAP_SHARP :: 2
BILINEAR :: 3
SWITCHABLE :: 4
SUBPEL_BITS :: 4
SUBPEL_MASK :: 15
SCALE_SUBPEL_BITS :: 10
REF_SCALE_SHIFT :: 14

MOTION_MODES :: 3
SIMPLE :: 0
OBMC :: 1
LOCALWARP :: 2

INTERINTRA_MODES :: 4
II_DC_PRED :: 0
II_V_PRED :: 1
II_H_PRED :: 2
II_SMOOTH_PRED :: 3
COMPOUND_TYPES :: 2
COMPOUND_WEDGE :: 0
COMPOUND_DIFFWTD :: 1
COMPOUND_AVERAGE :: 2
COMPOUND_INTRA :: 3
COMPOUND_DISTANCE :: 4
WEDGE_TYPES :: 16
MASK_MASTER_SIZE :: 64
WEDGE_HORIZONTAL :: 0
WEDGE_VERTICAL :: 1
WEDGE_OBLIQUE27 :: 2
WEDGE_OBLIQUE63 :: 3
WEDGE_OBLIQUE117 :: 4
WEDGE_OBLIQUE153 :: 5
MAX_FRAME_DISTANCE :: 31
MAX_OFFSET_WIDTH :: 8
MAX_OFFSET_HEIGHT :: 0

// Warped motion (section 3).
IDENTITY :: 0
TRANSLATION :: 1
ROTZOOM :: 2
AFFINE :: 3
GM_ABS_TRANS_BITS :: 12
GM_ABS_TRANS_ONLY_BITS :: 9
GM_ABS_ALPHA_BITS :: 12
GM_ALPHA_PREC_BITS :: 15
GM_TRANS_PREC_BITS :: 6
GM_TRANS_ONLY_PREC_BITS :: 3
DIV_LUT_PREC_BITS :: 14
DIV_LUT_BITS :: 8
DIV_LUT_NUM :: 257
LEAST_SQUARES_SAMPLES_MAX :: 8
LS_MV_MAX :: 256
WARPEDMODEL_TRANS_CLAMP :: 1 << 23
WARPEDMODEL_NONDIAGAFFINE_CLAMP :: 1 << 13
WARPEDPIXEL_PREC_SHIFTS :: 1 << 6
WARPEDDIFF_PREC_BITS :: 10
WARP_PARAM_REDUCE_BITS :: 6

// Segmentation, delta-Q, loop filter, CDEF, superres, restoration (section 3).
SEGMENT_ID_CONTEXTS :: 3
SEGMENT_ID_PREDICTED_CONTEXTS :: 3
SEG_LVL_ALT_Q :: 0
SEG_LVL_ALT_LF_Y_V :: 1
SEG_LVL_REF_FRAME :: 5
SEG_LVL_SKIP :: 6
SEG_LVL_GLOBALMV :: 7
DELTA_Q_SMALL :: 3
DELTA_LF_SMALL :: 3
FRAME_LF_COUNT :: 4
MAX_LOOP_FILTER :: 63
QM_TOTAL_SIZE :: 3344

SUPERRES_NUM :: 8
SUPERRES_DENOM_MIN :: 9
SUPERRES_DENOM_BITS :: 3
SUPERRES_FILTER_BITS :: 6
SUPERRES_FILTER_SHIFTS :: 1 << SUPERRES_FILTER_BITS
SUPERRES_FILTER_TAPS :: 8
SUPERRES_FILTER_OFFSET :: 3
SUPERRES_SCALE_BITS :: 14
SUPERRES_SCALE_MASK :: (1 << 14) - 1
SUPERRES_EXTRA_BITS :: 8

RESTORE_NONE :: 0
RESTORE_WIENER :: 1
RESTORE_SGRPROJ :: 2
RESTORE_SWITCHABLE :: 3
RESTORATION_TILESIZE_MAX :: 256
FILTER_BITS :: 7
WIENER_COEFFS :: 3
SGRPROJ_PARAMS_BITS :: 4
SGRPROJ_PRJ_SUBEXP_K :: 4
SGRPROJ_PRJ_BITS :: 7
SGRPROJ_RST_BITS :: 4
SGRPROJ_MTABLE_BITS :: 20
SGRPROJ_RECIP_BITS :: 12
SGRPROJ_SGR_BITS :: 8

// Entropy coding & coefficients (section 3).
EC_PROB_SHIFT :: 6
EC_MIN_PROB :: 4
COEFF_CDF_Q_CTXS :: 4
NUM_BASE_LEVELS :: 2
COEFF_BASE_RANGE :: 12
BR_CDF_SIZE :: 4
SIG_COEF_CONTEXTS_EOB :: 4
SIG_COEF_CONTEXTS_2D :: 26
SIG_COEF_CONTEXTS :: 42
SIG_REF_DIFF_OFFSET_NUM :: 5
TXB_SKIP_CONTEXTS :: 13
EOB_COEF_CONTEXTS :: 9
DC_SIGN_CONTEXTS :: 3
LEVEL_CONTEXTS :: 21

Obu_Type :: enum u8 {
	Reserved_0,
	Sequence_Header,
	Temporal_Delimiter,
	Frame_Header,
	Tile_Group,
	Metadata,
	Frame,
	Redundant_Frame_Header,
	Tile_List,
	Padding = 15,
}

Obu :: struct {
	type:                    Obu_Type,
	temporal_id, spatial_id: u8,
	offset:                  int, // of the OBU header in the data
	payload:                 []u8,
	payload_offset:          int,
}

// The OBU starting at data[at:] (low overhead format: every OBU sized).
next_obu :: proc(data: []u8, at: int) -> (o: Obu, next: int, err: Error) {
	if at >= len(data) do return o, at, .Invalid_Data
	h := data[at]
	if h & 0x80 != 0 do return o, at, .Invalid_Data // forbidden bit
	o.type = Obu_Type(h >> 3 & 15)
	o.offset = at
	p := at + 1
	if h & 4 != 0 {
		if p >= len(data) do return o, at, .Invalid_Data
		o.temporal_id, o.spatial_id = data[p] >> 5, data[p] >> 3 & 3
		p += 1
	}
	size := len(data) - p
	if h & 2 != 0 {
		v, n, ok := leb128(data[p:])
		if !ok || v > u64(len(data) - p - n) do return o, at, .Invalid_Data
		p += n
		size = int(v)
	}
	o.payload_offset = p
	o.payload = data[p:][:size]
	return o, p + size, .None
}

@(private)
leb128 :: proc(b: []u8) -> (value: u64, n: int, ok: bool) {
	for i in 0 ..< 8 {
		if i >= len(b) do return
		value |= u64(b[i] & 0x7F) << uint(7 * i)
		if b[i] & 0x80 == 0 do return value, i + 1, true
	}
	return
}

// ---- Bit reader (4.10, 8.1) -------------------------------------------------

@(private)
Bits :: struct {
	data: []u8,
	pos:  int, // bits read; reads past the end give zeros and set overrun
	overrun: bool,
}

@(private)
f :: proc(b: ^Bits, n: int) -> u32 {
	v: u32
	for _ in 0 ..< n {
		bit: u32
		if b.pos >> 3 < len(b.data) {
			bit = u32(b.data[b.pos >> 3] >> uint(7 - b.pos & 7) & 1)
		} else {
			b.overrun = true
		}
		v = v << 1 | bit
		b.pos += 1
	}
	return v
}

@(private)
flag :: proc(b: ^Bits) -> bool {return f(b, 1) == 1}

@(private)
su :: proc(b: ^Bits, n: int) -> i32 {
	v := i32(f(b, n))
	sign := i32(1) << uint(n - 1)
	return v & sign != 0 ? v - 2 * sign : v
}

@(private)
ns :: proc(b: ^Bits, n: u32) -> u32 {
	w := 0
	for x := n; x != 0; x >>= 1 do w += 1 // FloorLog2(n) + 1
	m := (u32(1) << uint(w)) - n
	v := f(b, w - 1)
	if v < m do return v
	return (v << 1) - m + f(b, 1)
}

@(private)
uvlc :: proc(b: ^Bits) -> u32 {
	zeros := 0
	for !flag(b) {
		zeros += 1
		if b.overrun do return 0
	}
	if zeros >= 32 do return max(u32)
	return f(b, zeros) + (u32(1) << uint(zeros)) - 1
}

// ---- Sequence header (5.5) --------------------------------------------------

Color_Config :: struct {
	bit_depth:                      u8,
	mono_chrome:                    bool,
	color_description_present_flag: bool,
	color_primaries:                u8,
	transfer_characteristics:       u8,
	matrix_coefficients:            u8,
	color_range:                    bool,
	subsampling_x, subsampling_y:   u8,
	chroma_sample_position:         u8,
	separate_uv_delta_q:            bool,
}

Sequence_Header :: struct {
	seq_profile:                        u8,
	still_picture:                      bool,
	reduced_still_picture_header:       bool,
	timing_info_present_flag:           bool,
	num_units_in_display_tick:          u32,
	time_scale:                         u32,
	equal_picture_interval:             bool,
	num_ticks_per_picture_minus_1:      u32,
	decoder_model_info_present_flag:    bool,
	buffer_removal_time_length_minus_1: u8,
	frame_presentation_time_length_minus_1: u8,
	initial_display_delay_present_flag: bool,
	operating_points_cnt_minus_1:       u8,
	operating_point_idc:                [32]u16,
	decoder_model_present_for_this_op:  [32]bool,
	seq_level_idx:                      u8, // of operating point 0
	frame_width_bits_minus_1:           u8,
	frame_height_bits_minus_1:          u8,
	max_frame_width_minus_1:            u16,
	max_frame_height_minus_1:           u16,
	frame_id_numbers_present_flag:      bool,
	delta_frame_id_length_minus_2:      u8,
	additional_frame_id_length_minus_1: u8,
	use_128x128_superblock:             bool,
	enable_filter_intra:                bool,
	enable_intra_edge_filter:           bool,
	enable_interintra_compound:         bool,
	enable_masked_compound:             bool,
	enable_warped_motion:               bool,
	enable_dual_filter:                 bool,
	enable_order_hint:                  bool,
	enable_jnt_comp:                    bool,
	enable_ref_frame_mvs:               bool,
	seq_force_screen_content_tools:     u8, // 2: SELECT_SCREEN_CONTENT_TOOLS
	seq_force_integer_mv:               u8, // 2: SELECT_INTEGER_MV
	order_hint_bits_minus_1:            u8,
	order_hint_bits:                    int,
	enable_superres:                    bool,
	enable_cdef:                        bool,
	enable_restoration:                 bool,
	color:                              Color_Config,
	film_grain_params_present:          bool,
}

SELECT_SCREEN_CONTENT_TOOLS :: 2
SELECT_INTEGER_MV :: 2
MC_IDENTITY :: 0

parse_sequence_header :: proc(payload: []u8) -> (s: Sequence_Header, err: Error) {
	b := Bits{data = payload}
	s.seq_profile = u8(f(&b, 3))
	if s.seq_profile > 2 do return s, .Unsupported
	s.still_picture = flag(&b)
	s.reduced_still_picture_header = flag(&b)
	if s.reduced_still_picture_header {
		s.seq_level_idx = u8(f(&b, 5))
		return s, parse_sequence_rest(&b, &s)
	}
	buffer_delay_length := 0
	s.timing_info_present_flag = flag(&b)
	if s.timing_info_present_flag {
		s.num_units_in_display_tick = f(&b, 32)
		s.time_scale = f(&b, 32)
		s.equal_picture_interval = flag(&b)
		if s.equal_picture_interval do s.num_ticks_per_picture_minus_1 = uvlc(&b)
		s.decoder_model_info_present_flag = flag(&b)
		if s.decoder_model_info_present_flag {
			buffer_delay_length = int(f(&b, 5)) + 1
			f(&b, 32) // num_units_in_decoding_tick
			s.buffer_removal_time_length_minus_1 = u8(f(&b, 5))
			s.frame_presentation_time_length_minus_1 = u8(f(&b, 5))
		}
	}
	return s, parse_operating_points(&b, &s, buffer_delay_length)
}

@(private)
parse_operating_points :: proc(b: ^Bits, s: ^Sequence_Header, buffer_delay_length: int) -> Error {
	s.initial_display_delay_present_flag = flag(b)
	s.operating_points_cnt_minus_1 = u8(f(b, 5))
	for i in 0 ..= int(s.operating_points_cnt_minus_1) {
		s.operating_point_idc[i] = u16(f(b, 12))
		level := u8(f(b, 5))
		if i == 0 do s.seq_level_idx = level
		if level > 7 do f(b, 1) // seq_tier
		if s.decoder_model_info_present_flag {
			s.decoder_model_present_for_this_op[i] = flag(b)
			if s.decoder_model_present_for_this_op[i] {
				f(b, buffer_delay_length) // decoder_buffer_delay
				f(b, buffer_delay_length) // encoder_buffer_delay
				f(b, 1) // low_delay_mode_flag
			}
		}
		if s.initial_display_delay_present_flag {
			if flag(b) do f(b, 4) // initial_display_delay_minus_1
		}
	}
	return parse_sequence_rest(b, s)
}

@(private)
parse_sequence_rest :: proc(b: ^Bits, s: ^Sequence_Header) -> Error {
	s.frame_width_bits_minus_1 = u8(f(b, 4))
	s.frame_height_bits_minus_1 = u8(f(b, 4))
	s.max_frame_width_minus_1 = u16(f(b, int(s.frame_width_bits_minus_1) + 1))
	s.max_frame_height_minus_1 = u16(f(b, int(s.frame_height_bits_minus_1) + 1))
	if !s.reduced_still_picture_header do s.frame_id_numbers_present_flag = flag(b)
	if s.frame_id_numbers_present_flag {
		s.delta_frame_id_length_minus_2 = u8(f(b, 4))
		s.additional_frame_id_length_minus_1 = u8(f(b, 3))
	}
	s.use_128x128_superblock = flag(b)
	s.enable_filter_intra = flag(b)
	s.enable_intra_edge_filter = flag(b)
	s.seq_force_screen_content_tools = SELECT_SCREEN_CONTENT_TOOLS
	s.seq_force_integer_mv = SELECT_INTEGER_MV
	if !s.reduced_still_picture_header {
		s.enable_interintra_compound = flag(b)
		s.enable_masked_compound = flag(b)
		s.enable_warped_motion = flag(b)
		s.enable_dual_filter = flag(b)
		s.enable_order_hint = flag(b)
		if s.enable_order_hint {
			s.enable_jnt_comp = flag(b)
			s.enable_ref_frame_mvs = flag(b)
		}
		if !flag(b) do s.seq_force_screen_content_tools = u8(f(b, 1)) // seq_choose_screen_content_tools
		if s.seq_force_screen_content_tools > 0 {
			if !flag(b) do s.seq_force_integer_mv = u8(f(b, 1)) // seq_choose_integer_mv
		}
		if s.enable_order_hint {
			s.order_hint_bits_minus_1 = u8(f(b, 3))
			s.order_hint_bits = int(s.order_hint_bits_minus_1) + 1
		}
	}
	s.enable_superres = flag(b)
	s.enable_cdef = flag(b)
	s.enable_restoration = flag(b)
	// color_config (5.5.2)
	c := &s.color
	high := flag(b)
	if s.seq_profile == 2 && high {
		c.bit_depth = flag(b) ? 12 : 10
	} else {
		c.bit_depth = high ? 10 : 8
	}
	if s.seq_profile != 1 do c.mono_chrome = flag(b)
	c.color_description_present_flag = flag(b)
	c.color_primaries, c.transfer_characteristics, c.matrix_coefficients = 2, 2, 2
	if c.color_description_present_flag {
		c.color_primaries = u8(f(b, 8))
		c.transfer_characteristics = u8(f(b, 8))
		c.matrix_coefficients = u8(f(b, 8))
	}
	switch {
	case c.mono_chrome:
		c.color_range = flag(b)
		c.subsampling_x, c.subsampling_y = 1, 1
	case c.color_primaries == 1 && c.transfer_characteristics == 13 && c.matrix_coefficients == 0:
		c.color_range = true
	case:
		c.color_range = flag(b)
		switch s.seq_profile {
		case 0:
			c.subsampling_x, c.subsampling_y = 1, 1
		case 1:
		case:
			if c.bit_depth == 12 {
				c.subsampling_x = u8(f(b, 1))
				if c.subsampling_x == 1 do c.subsampling_y = u8(f(b, 1))
			} else {
				c.subsampling_x = 1
			}
		}
		if c.subsampling_x == 1 && c.subsampling_y == 1 do c.chroma_sample_position = u8(f(b, 2))
	}
	if !c.mono_chrome do c.separate_uv_delta_q = flag(b)
	s.film_grain_params_present = flag(b)
	return b.overrun ? .Invalid_Data : .None
}
