package webp

import common "../../common"
import "base:intrinsics"
import "core:math"
import "core:simd"

// VP8 lossy key-frame encoding (RFC 6386) and ALPH chunk compression
// (RFC 9649).
//
// Macroblocks are loaded once into flat 16-byte 4x4 vectors (`Block4`),
// predicted in vector registers across all modes at once, transformed and
// quantized branchlessly in SIMD, and reconstructed in registers so only the
// winning macroblock touches frame memory. Coefficient decisions are written
// to one pre-reserved token stream, probabilities are updated where they save
// bits, and both partitions go through `common.Bool_Encoder`.

@(private)
MAX_LOSSY_SIZE :: 16383 // VP8 stores width and height in 14 bits without -1

@(private)
QFIX :: 17
@(private)
MAX_LEVEL :: 2047
@(private)
MAX_VARIABLE_LEVEL :: 67
@(private)
FLATNESS_LIMIT_I16 :: 0
@(private)
FLATNESS_LIMIT_I4 :: 3
@(private)
FLATNESS_LIMIT_UV :: 2
@(private)
FLATNESS_PENALTY :: 140
@(private)
RD_DISTO_MULT :: 256
@(private)
FIXED_PROB_FLAG :: u16(0x4000)
@(private)
MAX_MB_TOKENS :: 25 * 16 * 24 // worst-case bits per macroblock (25 blocks, Cat6)

// A 4x4 pixel patch (4 rows of 4 bytes) and its 4x4 transform block.
@(private)
Block4 :: #simd[16]u8
@(private)
Coeff4 :: #simd[16]i16

@(private, rodata)
ENTROPY_COST := [256]u16{
	1792, 1792, 1792, 1536, 1536, 1408, 1366, 1280, 1280, 1216,
	1178, 1152, 1110, 1076, 1061, 1024, 1024,  992,  968,  951,
	 939,  911,  896,  878,  871,  854,  838,  820,  811,  794,
	 786,  768,  768,  752,  740,  732,  720,  709,  704,  690,
	 683,  672,  666,  655,  647,  640,  631,  622,  615,  607,
	 598,  592,  586,  576,  572,  564,  559,  555,  547,  541,
	 534,  528,  522,  512,  512,  504,  500,  494,  488,  483,
	 477,  473,  467,  461,  458,  452,  448,  443,  438,  434,
	 427,  424,  419,  415,  410,  406,  403,  399,  394,  390,
	 384,  384,  377,  374,  370,  366,  362,  359,  355,  351,
	 347,  342,  342,  336,  333,  330,  326,  323,  320,  316,
	 312,  308,  305,  302,  299,  296,  293,  288,  287,  283,
	 280,  277,  274,  272,  268,  266,  262,  256,  256,  256,
	 251,  248,  245,  242,  240,  237,  234,  232,  228,  226,
	 223,  221,  218,  216,  214,  211,  208,  205,  203,  201,
	 198,  196,  192,  191,  188,  187,  183,  181,  179,  176,
	 175,  171,  171,  168,  165,  163,  160,  159,  156,  154,
	 152,  150,  148,  146,  144,  142,  139,  138,  135,  133,
	 131,  128,  128,  125,  123,  121,  119,  117,  115,  113,
	 111,  110,  107,  105,  103,  102,  100,   98,   96,   94,
	  92,   91,   89,   86,   86,   83,   82,   80,   77,   76,
	  74,   73,   71,   69,   67,   66,   64,   63,   61,   59,
	  57,   55,   54,   52,   51,   49,   47,   46,   44,   43,
	  41,   40,   38,   36,   35,   33,   32,   30,   29,   27,
	  25,   24,   22,   21,   19,   18,   16,   15,   13,   12,
	  10,    9,    7,    6,    4,    3,
}

@(private, rodata)
LEVEL_CODES := [MAX_VARIABLE_LEVEL][2]u16{
	{0x001, 0x000}, {0x007, 0x001}, {0x00f, 0x005},
	{0x00f, 0x00d}, {0x033, 0x003}, {0x033, 0x003}, {0x033, 0x023},
	{0x033, 0x023}, {0x033, 0x023}, {0x033, 0x023}, {0x0d3, 0x013},
	{0x0d3, 0x013}, {0x0d3, 0x013}, {0x0d3, 0x013}, {0x0d3, 0x013},
	{0x0d3, 0x013}, {0x0d3, 0x013}, {0x0d3, 0x013}, {0x0d3, 0x093},
	{0x0d3, 0x093}, {0x0d3, 0x093}, {0x0d3, 0x093}, {0x0d3, 0x093},
	{0x0d3, 0x093}, {0x0d3, 0x093}, {0x0d3, 0x093}, {0x0d3, 0x093},
	{0x0d3, 0x093}, {0x0d3, 0x093}, {0x0d3, 0x093}, {0x0d3, 0x093},
	{0x0d3, 0x093}, {0x0d3, 0x093}, {0x0d3, 0x093}, {0x153, 0x053},
	{0x153, 0x053}, {0x153, 0x053}, {0x153, 0x053}, {0x153, 0x053},
	{0x153, 0x053}, {0x153, 0x053}, {0x153, 0x053}, {0x153, 0x053},
	{0x153, 0x053}, {0x153, 0x053}, {0x153, 0x053}, {0x153, 0x053},
	{0x153, 0x053}, {0x153, 0x053}, {0x153, 0x053}, {0x153, 0x053},
	{0x153, 0x053}, {0x153, 0x053}, {0x153, 0x053}, {0x153, 0x053},
	{0x153, 0x053}, {0x153, 0x053}, {0x153, 0x053}, {0x153, 0x053},
	{0x153, 0x053}, {0x153, 0x053}, {0x153, 0x053}, {0x153, 0x053},
	{0x153, 0x053}, {0x153, 0x053}, {0x153, 0x053}, {0x153, 0x153},
}

@(private, rodata)
I16_MODE_COST := [4]i64{663, 919, 872, 919}
@(private, rodata)
UV_MODE_COST := [4]i64{302, 984, 439, 642}
@(private, rodata)
FREQ_SHARPENING := [16]u16{0, 30, 60, 90, 30, 60, 90, 90, 60, 90, 90, 90, 90, 90, 90, 90}
@(private, rodata)
BIAS_MATRICES := [3][2]u32{{96, 110}, {96, 108}, {110, 115}}
@(private, rodata)
CAT3_PROBS := [3]u8{173, 148, 140}
@(private, rodata)
CAT4_PROBS := [4]u8{176, 155, 140, 135}
@(private, rodata)
CAT5_PROBS := [5]u8{180, 157, 141, 134, 130}
@(private, rodata)
CAT6_PROBS := [11]u8{254, 254, 243, 230, 196, 177, 153, 140, 133, 130, 129}

@(private)
WEIGHT_Y0 :: #simd[4]i32{38, 32, 20, 9}
@(private)
WEIGHT_Y1 :: #simd[4]i32{32, 28, 17, 7}
@(private)
WEIGHT_Y2 :: #simd[4]i32{20, 17, 10, 4}
@(private)
WEIGHT_Y3 :: #simd[4]i32{9, 7, 4, 2}

@(private)
LEVEL_FIXED_COST: [MAX_LEVEL + 1]u16
@(private)
BMODE_COST: [10][10][10]u16

@(private)
bit_cost :: #force_inline proc "contextless" (bit: int, prob: u8) -> int #no_bounds_check {
	return bit != 0 ? int(ENTROPY_COST[255 - prob]) : int(ENTROPY_COST[prob])
}

@(private, init)
init_vp8_cost_tables :: proc "contextless" () {
	LEVEL_FIXED_COST[0] = 0
	for v in 1 ..= 4 do LEVEL_FIXED_COST[v] = 256
	LEVEL_FIXED_COST[5] = u16(256 + bit_cost(0, 159))
	LEVEL_FIXED_COST[6] = u16(256 + bit_cost(1, 159))
	for v in 7 ..= 10 {
		LEVEL_FIXED_COST[v] = u16(256 + bit_cost(int(v >= 9), 165) + bit_cost((v - 7) & 1, 145))
	}
	for v in 11 ..= MAX_LEVEL {
		r := v - 3
		c := 256
		switch {
		case r < 16:
			r -= 8
			for i in 0 ..< 3 do c += bit_cost((r >> uint(2 - i)) & 1, CAT3_PROBS[i])
		case r < 32:
			r -= 16
			for i in 0 ..< 4 do c += bit_cost((r >> uint(3 - i)) & 1, CAT4_PROBS[i])
		case r < 64:
			r -= 32
			for i in 0 ..< 5 do c += bit_cost((r >> uint(4 - i)) & 1, CAT5_PROBS[i])
		case:
			r -= 64
			for i in 0 ..< 11 do c += bit_cost((r >> uint(10 - i)) & 1, CAT6_PROBS[i])
		}
		LEVEL_FIXED_COST[v] = u16(c)
	}

	for top in 0 ..< 10 do for left in 0 ..< 10 {
		p := &BMODE_PROBS[top][left]
		c := &BMODE_COST[top][left]
		c[B_DC] = u16(bit_cost(0, p[0]))
		b0 := bit_cost(1, p[0])
		c[B_TM] = u16(b0 + bit_cost(0, p[1]))
		b1 := b0 + bit_cost(1, p[1])
		c[B_VE] = u16(b1 + bit_cost(0, p[2]))
		b2 := b1 + bit_cost(1, p[2])
		l := b2 + bit_cost(0, p[3])
		c[B_HE] = u16(l + bit_cost(0, p[4]))
		l4 := l + bit_cost(1, p[4])
		c[B_RD] = u16(l4 + bit_cost(0, p[5]))
		c[B_VR] = u16(l4 + bit_cost(1, p[5]))
		r := b2 + bit_cost(1, p[3])
		c[B_LD] = u16(r + bit_cost(0, p[6]))
		r6 := r + bit_cost(1, p[6])
		c[B_VL] = u16(r6 + bit_cost(0, p[7]))
		r7 := r6 + bit_cost(1, p[7])
		c[B_HD] = u16(r7 + bit_cost(0, p[8]))
		c[B_HU] = u16(r7 + bit_cost(1, p[8]))
	}
}

@(private)
Quant_Matrix :: struct {
	q:       Coeff4,
	iq:      #simd[16]u32,
	bias:    #simd[16]u32,
	sharpen: #simd[16]u16,
	dc_q:    i32,
	dc_iq:   u32,
	dc_bias: u32,
	dc_zthr: u32,
}

@(private)
Segment_Quant :: struct {
	y1, y1_ac, y2, uv: Quant_Matrix,
	alpha, beta:       int,
	quant:             int,
	fstrength:         int,
	lambda_i16:        i64,
	lambda_i4:         i64,
	lambda_uv:         i64,
	lambda_mode:       i64,
	tlambda:           i64,
	min_disto:         int,
	max_edge:          int,
}

@(private)
Enc_Mb :: struct {
	tok_start: u32,
	segment:   u8,
	alpha:     u8,
	is_4x4:    bool,
	skip:      bool,
	ymode:     u8,
	uvmode:    u8,
	modes:     [16]u8,
}

@(private)
Analysis_Row :: struct {
	alphas:   [256]i32,
	uv_alpha: i64,
}

// Heap-allocated probability/cost tables so Lossy_Encoder stays small on a
// 1 MB stack. `level_cost` includes `LEVEL_FIXED_COST` for v <= 67 so RD rate
// evaluation needs a single table lookup per coefficient.
@(private)
Enc_Tables :: struct {
	probs:      [4][8][3][11]u8,
	stats:      [4][8][3][11][2]u32,
	skip_stats: [4][8][3][11][2]u32,
	level_cost: [4][8][3][MAX_VARIABLE_LEVEL + 1]u16,
}

@(private)
Lossy_Encoder :: struct {
	allocator:       Allocator,
	out:             [dynamic]u8,
	planes:          [dynamic]u8, // one arena for src YUV, rec YUV, alpha, and row contexts
	mbs:             [dynamic]Enc_Mb, // length mb_w * mb_h + 1 (last is tok_start sentinel)
	top_derr:        [dynamic][2][2]i8,
	row_stats:       [dynamic]Analysis_Row,
	tokens:          [dynamic]u16,
	part0:           common.Bool_Encoder,
	part1:           common.Bool_Encoder,
	d:               Vp8,
	src_y:           []u8,
	src_u:           []u8,
	src_v:           []u8,
	alpha_plane:     []u8,
	alpha_filtered:  []u8,
	row_has_alpha:   []u8,
	tables:          ^Enc_Tables,
	dqm:             [4]Segment_Quant,
	num_segments:    int,
	base_quant:      int,
	dq_uv_dc:        int,
	dq_uv_ac:        int,
	has_alpha:       bool,
	use_derr:        bool,
	max_i4_bits:     i64,
	alpha_enc:       Lossless_Encoder,
	alpha_enc_ready: bool,
}

@(private)
init_lossy_encoder :: proc(e: ^Lossy_Encoder, allocator: Allocator) -> Error {
	e.allocator = allocator
	e.out.allocator = allocator
	e.planes.allocator = allocator
	e.mbs.allocator = allocator
	e.top_derr.allocator = allocator
	e.row_stats.allocator = allocator
	e.tokens.allocator = allocator
	e.part0.data.allocator = allocator
	e.part1.data.allocator = allocator
	tables, err := new(Enc_Tables, allocator)
	if err != nil do return .Out_Of_Memory
	e.tables = tables
	return .None
}

@(private)
destroy_lossy_encoder :: proc(e: ^Lossy_Encoder) {
	delete(e.out)
	delete(e.planes)
	delete(e.mbs)
	delete(e.top_derr)
	delete(e.row_stats)
	delete(e.tokens)
	delete(e.part0.data)
	delete(e.part1.data)
	free(e.tables, e.allocator)
	if e.alpha_enc_ready do destroy_lossless_encoder(&e.alpha_enc)
	e^ = {}
}

@(private)
take_lossy_output :: proc(e: ^Lossy_Encoder) -> (data: []u8, err: Error) {
	allocator := e.out.allocator
	if _, oom := shrink(&e.out); oom != nil do return nil, .Out_Of_Memory
	data = e.out[:]
	e.out = {}
	e.out.allocator = allocator
	return data, .None
}

// Appends a complete RIFF WEBP still (VP8 alone when opaque, VP8X + ALPH + VP8
// when any pixel is not opaque) to `e.out`.
@(private)
write_lossy_still :: proc(e: ^Lossy_Encoder, rgba: []u8, stride, width, height, quality: int, pool: ^common.Work_Pool) -> Error {
	prepare_lossy_frame(e, rgba, stride, width, height, quality, pool) or_return
	out := &e.out
	clear(out)
	if e.has_alpha {
		hdr := [30]u8{}
		copy(hdr[:], "RIFF\x00\x00\x00\x00WEBPVP8X\x0a\x00\x00\x00\x10")
		put_u24(hdr[24:], u32(width - 1))
		put_u24(hdr[27:], u32(height - 1))
		if _, oom := non_zero_append(out, ..hdr[:]); oom != nil do return .Out_Of_Memory
		write_alph_chunk(e, out, width, height) or_return
	} else {
		if _, oom := non_zero_append(out, "RIFF\x00\x00\x00\x00WEBP"); oom != nil do return .Out_Of_Memory
	}
	write_vp8_chunk(e, out, width, height) or_return
	put_u32(out[4:], u32(len(out) - 8))
	return .None
}

// Appends an ANMF frame's lossy payload ([ALPH] + VP8) to `dst`.
@(private)
write_lossy_anmf :: proc(e: ^Lossy_Encoder, dst: ^[dynamic]u8, rgba: []u8, stride, width, height, quality: int, pool: ^common.Work_Pool) -> Error {
	prepare_lossy_frame(e, rgba, stride, width, height, quality, pool) or_return
	if e.has_alpha do write_alph_chunk(e, dst, width, height) or_return
	return write_vp8_chunk(e, dst, width, height)
}

@(private)
prepare_lossy_frame :: proc(e: ^Lossy_Encoder, rgba: []u8, stride, width, height, quality: int, pool: ^common.Work_Pool) -> Error {
	mb_w, mb_h := (width + 15) >> 4, (height + 15) >> 4
	ys, uvs := mb_w * 16, mb_w * 8
	y_len := ys * mb_h * 16
	uv_len := uvs * mb_h * 8
	pix_len := width * height
	pair_rows := (height + 1) >> 1
	top_len := mb_w * 6

	plane_bytes := 2 * (y_len + 2 * uv_len) + 2 * pix_len + pair_rows + top_len
	if non_zero_resize(&e.planes, plane_bytes) != nil ||
	   non_zero_resize(&e.mbs, mb_w * mb_h + 1) != nil ||
	   non_zero_resize(&e.top_derr, mb_w) != nil ||
	   non_zero_resize(&e.row_stats, mb_h) != nil {
		return .Out_Of_Memory
	}

	at := 0
	carve :: proc(buf: []u8, at: ^int, n: int) -> []u8 {
		s := buf[at^ : at^ + n]
		at^ += n
		return s
	}
	e.src_y = carve(e.planes[:], &at, y_len)
	e.src_u = carve(e.planes[:], &at, uv_len)
	e.src_v = carve(e.planes[:], &at, uv_len)
	rec_y := carve(e.planes[:], &at, y_len)
	rec_u := carve(e.planes[:], &at, uv_len)
	rec_v := carve(e.planes[:], &at, uv_len)
	e.alpha_plane = carve(e.planes[:], &at, pix_len)
	e.alpha_filtered = carve(e.planes[:], &at, pix_len)
	e.row_has_alpha = carve(e.planes[:], &at, pair_rows)
	intra_top := carve(e.planes[:], &at, mb_w * 4)
	nz_top := carve(e.planes[:], &at, mb_w)
	nz_dc_top := carve(e.planes[:], &at, mb_w)

	e.d = {
		frame = {
			width     = width,
			height    = height,
			y         = rec_y,
			u         = rec_u,
			v         = rec_v,
			y_stride  = ys,
			uv_stride = uvs,
		},
		mb_w      = mb_w,
		mb_h      = mb_h,
		intra_top = intra_top,
		nz_top    = nz_top,
		nz_dc_top = nz_dc_top,
	}

	rgba_to_yuva(e, rgba, stride, width, height, pool)
	if e.has_alpha do cleanup_transparent_yuv(e, width, height)
	pad_yuv_borders(e, width, height)

	analyze_segments(e, pool)
	setup_segment_params(e, quality)
	e.use_derr = quality <= 98
	e.max_i4_bits = min(i64(256 * 16 * 16), (i64(256) * 500 * 8 * 1024) / i64(mb_w * mb_h))

	for {
		encode_vp8_pass(e) or_return
		if len(e.part0.data) < 1 << 19 || e.max_i4_bits == 0 do break
		e.max_i4_bits >>= 1
	}
	return len(e.part0.data) < 1 << 19 ? .None : .Invalid
}

@(private)
ashr4 :: #force_inline proc "contextless" (v: #simd[4]i32, $n: u32) -> #simd[4]i32 {
	return simd.shr_masked(v, (#simd[4]u32)(n))
}

@(private)
rgba_to_yuva :: proc(e: ^Lossy_Encoder, rgba: []u8, stride, width, height: int, pool: ^common.Work_Pool) {
	Job :: struct {
		e:            ^Lossy_Encoder,
		rgba:         []u8,
		stride, w, h: int,
	}
	pair_rows := (height + 1) >> 1
	job := Job{e, rgba, stride, width, height}
	rows :: proc(data: rawptr, first, end: int) #no_bounds_check {
		j := (^Job)(data)
		e := j.e
		w, h, stride := j.w, j.h, j.stride
		ys, uvs := e.d.y_stride, e.d.uv_stride
		cw := (w + 1) >> 1
		for py in first ..< end {
			y0 := py * 2
			opaque := u8(0xff)
			for dy in 0 ..< 2 {
				y := y0 + dy
				if y >= h do break
				srow := j.rgba[y * stride:]
				yrow := e.src_y[y * ys:]
				arow := e.alpha_plane[y * w:]
				x := 0
				op_vec := (#simd[8]u32)(0xff)
				for ; x + 8 <= w; x += 8 {
					raw := intrinsics.unaligned_load((^#simd[32]u8)(&srow[x * 4]))
					r := cast(#simd[8]u32)simd.swizzle(raw, 0, 4, 8, 12, 16, 20, 24, 28)
					g := cast(#simd[8]u32)simd.swizzle(raw, 1, 5, 9, 13, 17, 21, 25, 29)
					b := cast(#simd[8]u32)simd.swizzle(raw, 2, 6, 10, 14, 18, 22, 26, 30)
					a := cast(#simd[8]u32)simd.swizzle(raw, 3, 7, 11, 15, 19, 23, 27, 31)
					yv := cast(#simd[8]u8)simd.shr(16839 * r + 33059 * g + 6420 * b + (1 << 15) + (16 << 16), 16)
					(^[8]u8)(&yrow[x])^ = transmute([8]u8)yv
					(^[8]u8)(&arow[x])^ = transmute([8]u8)cast(#simd[8]u8)a
					op_vec &= a
				}
				if x > 0 && simd.reduce_and(op_vec) != 0xff do opaque = 0
				for ; x < w; x += 1 {
					r, g, b, a := int(srow[x * 4]), int(srow[x * 4 + 1]), int(srow[x * 4 + 2]), srow[x * 4 + 3]
					yrow[x] = u8((16839 * r + 33059 * g + 6420 * b + (1 << 15) + (16 << 16)) >> 16)
					arow[x] = a
					opaque &= a
				}
			}
			e.row_has_alpha[py] = u8(opaque != 0xff)

			y1 := min(y0 + 1, h - 1)
			r0 := j.rgba[y0 * stride:]
			r1 := j.rgba[y1 * stride:]
			urow := e.src_u[py * uvs:]
			vrow := e.src_v[py * uvs:]
			cx := 0
			if opaque == 0xff {
				for ; cx * 2 + 8 <= w; cx += 4 {
					t0 := intrinsics.unaligned_load((^#simd[32]u8)(&r0[cx * 8]))
					t1 := intrinsics.unaligned_load((^#simd[32]u8)(&r1[cx * 8]))
					r4 := cast(#simd[4]i32)simd.swizzle(t0, 0, 8, 16, 24) + cast(#simd[4]i32)simd.swizzle(t0, 4, 12, 20, 28) +
					      cast(#simd[4]i32)simd.swizzle(t1, 0, 8, 16, 24) + cast(#simd[4]i32)simd.swizzle(t1, 4, 12, 20, 28)
					g4 := cast(#simd[4]i32)simd.swizzle(t0, 1, 9, 17, 25) + cast(#simd[4]i32)simd.swizzle(t0, 5, 13, 21, 29) +
					      cast(#simd[4]i32)simd.swizzle(t1, 1, 9, 17, 25) + cast(#simd[4]i32)simd.swizzle(t1, 5, 13, 21, 29)
					b4 := cast(#simd[4]i32)simd.swizzle(t0, 2, 10, 18, 26) + cast(#simd[4]i32)simd.swizzle(t0, 6, 14, 22, 30) +
					      cast(#simd[4]i32)simd.swizzle(t1, 2, 10, 18, 26) + cast(#simd[4]i32)simd.swizzle(t1, 6, 14, 22, 30)
					u := cast(#simd[4]u8)simd.clamp(ashr4(-9719 * r4 - 19081 * g4 + 28800 * b4 + (1 << 17) + (128 << 18), 18), 0, 255)
					v := cast(#simd[4]u8)simd.clamp(ashr4(28800 * r4 - 24116 * g4 - 4684 * b4 + (1 << 17) + (128 << 18), 18), 0, 255)
					(^[4]u8)(&urow[cx])^ = transmute([4]u8)u
					(^[4]u8)(&vrow[cx])^ = transmute([4]u8)v
				}
			}
			for ; cx < cw; cx += 1 {
				x0 := cx * 2
				x1 := min(x0 + 1, w - 1)
				p00, p10 := r0[x0 * 4:], r0[x1 * 4:]
				p01, p11 := r1[x0 * 4:], r1[x1 * 4:]
				a00, a10, a01, a11 := p00[3], p10[3], p01[3], p11[3]
				r4, g4, b4: int
				if (a00 & a10 & a01 & a11) == 0xff || (a00 | a10 | a01 | a11) == 0 {
					r4 = int(p00[0]) + int(p10[0]) + int(p01[0]) + int(p11[0])
					g4 = int(p00[1]) + int(p10[1]) + int(p01[1]) + int(p11[1])
					b4 = int(p00[2]) + int(p10[2]) + int(p01[2]) + int(p11[2])
				} else {
					sr, sg, sb, n := 0, 0, 0, 0
					if a00 != 0 {sr += int(p00[0]); sg += int(p00[1]); sb += int(p00[2]); n += 1}
					if a10 != 0 {sr += int(p10[0]); sg += int(p10[1]); sb += int(p10[2]); n += 1}
					if a01 != 0 {sr += int(p01[0]); sg += int(p01[1]); sb += int(p01[2]); n += 1}
					if a11 != 0 {sr += int(p11[0]); sg += int(p11[1]); sb += int(p11[2]); n += 1}
					r4 = (sr * 4 + n / 2) / n
					g4 = (sg * 4 + n / 2) / n
					b4 = (sb * 4 + n / 2) / n
				}
				urow[cx] = u8(clamp((-9719 * r4 - 19081 * g4 + 28800 * b4 + (1 << 17) + (128 << 18)) >> 18, 0, 255))
				vrow[cx] = u8(clamp((28800 * r4 - 24116 * g4 - 4684 * b4 + (1 << 17) + (128 << 18)) >> 18, 0, 255))
			}
		}
	}
	if pool != nil && pool.run != nil && pair_rows > 1 {
		pool.run(pool, pair_rows, rows, &job)
	} else {
		rows(&job, 0, pair_rows)
	}

	has_alpha := false
	for f in e.row_has_alpha do has_alpha ||= f != 0
	e.has_alpha = has_alpha
}

@(private)
pad_yuv_borders :: proc(e: ^Lossy_Encoder, width, height: int) #no_bounds_check {
	ys, uvs := e.d.y_stride, e.d.uv_stride
	pair_rows := (height + 1) >> 1
	cw := (width + 1) >> 1
	for y in 0 ..< height {
		yrow := e.src_y[y * ys:]
		last_y := yrow[width - 1]
		for x in width ..< ys do yrow[x] = last_y
	}
	for py in 0 ..< pair_rows {
		urow := e.src_u[py * uvs:]
		vrow := e.src_v[py * uvs:]
		last_u, last_v := urow[cw - 1], vrow[cw - 1]
		for cx in cw ..< uvs {
			urow[cx] = last_u
			vrow[cx] = last_v
		}
	}
	for y in height ..< e.d.mb_h * 16 {
		copy(e.src_y[y * ys:][:ys], e.src_y[(height - 1) * ys:][:ys])
	}
	for py in pair_rows ..< e.d.mb_h * 8 {
		copy(e.src_u[py * uvs:][:uvs], e.src_u[(pair_rows - 1) * uvs:][:uvs])
		copy(e.src_v[py * uvs:][:uvs], e.src_v[(pair_rows - 1) * uvs:][:uvs])
	}
}

// Smooths luma of transparent pixels in mixed 8x8 blocks to the block's
// visible mean and flattens runs of completely transparent 8x8 blocks,
// matching libwebp's WebPCleanupTransparentArea.
@(private)
cleanup_transparent_yuv :: proc(e: ^Lossy_Encoder, width, height: int) #no_bounds_check {
	smoothen_block :: proc(a: []u8, a_stride: int, yp: []u8, y_stride, bw, bh: int) -> bool #no_bounds_check {
		sum, count := 0, 0
		for dy in 0 ..< bh {
			ar := a[dy * a_stride:]
			yr := yp[dy * y_stride:]
			for dx in 0 ..< bw do if ar[dx] != 0 {
				count += 1
				sum += int(yr[dx])
			}
		}
		if count > 0 && count < bw * bh {
			avg := u8(sum / count)
			for dy in 0 ..< bh {
				ar := a[dy * a_stride:]
				yr := yp[dy * y_stride:]
				for dx in 0 ..< bw do if ar[dx] == 0 {
					yr[dx] = avg
				}
			}
		}
		return count == 0
	}

	ys, uvs := e.d.y_stride, e.d.uv_stride
	y := 0
	for y + 8 <= height {
		need_reset := true
		yv, uv, vv: u8
		x := 0
		for x + 8 <= width {
			if smoothen_block(e.alpha_plane[y * width + x:], width, e.src_y[y * ys + x:], ys, 8, 8) {
				if need_reset {
					yv = e.src_y[y * ys + x]
					uv = e.src_u[(y >> 1) * uvs + (x >> 1)]
					vv = e.src_v[(y >> 1) * uvs + (x >> 1)]
					need_reset = false
				}
				for dy in 0 ..< 8 do for dx in 0 ..< 8 do e.src_y[(y + dy) * ys + x + dx] = yv
				for dy in 0 ..< 4 do for dx in 0 ..< 4 {
					e.src_u[((y >> 1) + dy) * uvs + (x >> 1) + dx] = uv
					e.src_v[((y >> 1) + dy) * uvs + (x >> 1) + dx] = vv
				}
			} else {
				need_reset = true
			}
			x += 8
		}
		if x < width {
			_ = smoothen_block(e.alpha_plane[y * width + x:], width, e.src_y[y * ys + x:], ys, width - x, 8)
		}
		y += 8
	}
	if y < height {
		sub_h := height - y
		x := 0
		for x + 8 <= width {
			_ = smoothen_block(e.alpha_plane[y * width + x:], width, e.src_y[y * ys + x:], ys, 8, sub_h)
			x += 8
		}
		if x < width {
			_ = smoothen_block(e.alpha_plane[y * width + x:], width, e.src_y[y * ys + x:], ys, width - x, sub_h)
		}
	}
}

@(private)
load_plane4 :: #force_inline proc "contextless" (plane: []u8, at, stride: int) -> Block4 #no_bounds_check {
	r0 := intrinsics.unaligned_load((^u32)(&plane[at + 0 * stride]))
	r1 := intrinsics.unaligned_load((^u32)(&plane[at + 1 * stride]))
	r2 := intrinsics.unaligned_load((^u32)(&plane[at + 2 * stride]))
	r3 := intrinsics.unaligned_load((^u32)(&plane[at + 3 * stride]))
	return transmute(Block4)[4]u32{r0, r1, r2, r3}
}

@(private)
store_ws4 :: #force_inline proc "contextless" (ws: ^[WORKSPACE]u8, at: int, b: Block4) #no_bounds_check {
	rows := transmute([4]u32)b
	intrinsics.unaligned_store((^u32)(&ws[at + 0 * BPS]), rows[0])
	intrinsics.unaligned_store((^u32)(&ws[at + 1 * BPS]), rows[1])
	intrinsics.unaligned_store((^u32)(&ws[at + 2 * BPS]), rows[2])
	intrinsics.unaligned_store((^u32)(&ws[at + 3 * BPS]), rows[3])
}

@(private)
load_mb_source :: #force_inline proc "contextless" (e: ^Lossy_Encoder, mb_x, mb_y: int, y4: ^[16]Block4, uv4: ^[8]Block4) #no_bounds_check {
	ys, uvs := e.d.y_stride, e.d.uv_stride
	y_base := mb_y * 16 * ys + mb_x * 16
	for n in 0 ..< 16 {
		y4[n] = load_plane4(e.src_y, y_base + (n >> 2) * 4 * ys + (n & 3) * 4, ys)
	}
	uv_base := mb_y * 8 * uvs + mb_x * 8
	for n in 0 ..< 4 {
		off := uv_base + (n >> 1) * 4 * uvs + (n & 1) * 4
		uv4[n] = load_plane4(e.src_u, off, uvs)
		uv4[4 + n] = load_plane4(e.src_v, off, uvs)
	}
}

@(private)
store_mb_recon :: #force_inline proc "contextless" (d: ^Vp8, mb_x, mb_y: int, y4: ^[16]Block4, uv4: ^[8]Block4) #no_bounds_check {
	ys, uvs := d.y_stride, d.uv_stride
	y_base := mb_y * 16 * ys + mb_x * 16
	for by in 0 ..< 4 {
		b0 := transmute([4]u32)y4[by * 4 + 0]
		b1 := transmute([4]u32)y4[by * 4 + 1]
		b2 := transmute([4]u32)y4[by * 4 + 2]
		b3 := transmute([4]u32)y4[by * 4 + 3]
		for r in 0 ..< 4 {
			intrinsics.unaligned_store((^[4]u32)(&d.y[y_base + (by * 4 + r) * ys]), [4]u32{b0[r], b1[r], b2[r], b3[r]})
		}
	}
	uv_base := mb_y * 8 * uvs + mb_x * 8
	for by in 0 ..< 2 {
		u0 := transmute([4]u32)uv4[by * 2 + 0]
		u1 := transmute([4]u32)uv4[by * 2 + 1]
		v0 := transmute([4]u32)uv4[4 + by * 2 + 0]
		v1 := transmute([4]u32)uv4[4 + by * 2 + 1]
		for r in 0 ..< 4 {
			row := uv_base + (by * 4 + r) * uvs
			intrinsics.unaligned_store((^[2]u32)(&d.u[row]), [2]u32{u0[r], u1[r]})
			intrinsics.unaligned_store((^[2]u32)(&d.v[row]), [2]u32{v0[r], v1[r]})
		}
	}
}

@(private)
transpose4x4 :: #force_inline proc "contextless" (c0, c1, c2, c3: #simd[4]i32) -> (r0, r1, r2, r3: #simd[4]i32) {
	t0 := simd.shuffle(c0, c1, 0, 4, 1, 5)
	t1 := simd.shuffle(c2, c3, 0, 4, 1, 5)
	t2 := simd.shuffle(c0, c1, 2, 6, 3, 7)
	t3 := simd.shuffle(c2, c3, 2, 6, 3, 7)
	r0 = simd.shuffle(t0, t1, 0, 1, 4, 5)
	r1 = simd.shuffle(t0, t1, 2, 3, 6, 7)
	r2 = simd.shuffle(t2, t3, 0, 1, 4, 5)
	r3 = simd.shuffle(t2, t3, 2, 3, 6, 7)
	return
}

@(private)
unpack_rows4x4 :: #force_inline proc "contextless" (v: Coeff4) -> (r0, r1, r2, r3: #simd[4]i32) {
	r0 = cast(#simd[4]i32)simd.swizzle(v, 0, 1, 2, 3)
	r1 = cast(#simd[4]i32)simd.swizzle(v, 4, 5, 6, 7)
	r2 = cast(#simd[4]i32)simd.swizzle(v, 8, 9, 10, 11)
	r3 = cast(#simd[4]i32)simd.swizzle(v, 12, 13, 14, 15)
	return
}

@(private)
unpack_cols4x4 :: #force_inline proc "contextless" (v: Coeff4) -> (c0, c1, c2, c3: #simd[4]i32) {
	c0 = cast(#simd[4]i32)simd.swizzle(v, 0, 4, 8, 12)
	c1 = cast(#simd[4]i32)simd.swizzle(v, 1, 5, 9, 13)
	c2 = cast(#simd[4]i32)simd.swizzle(v, 2, 6, 10, 14)
	c3 = cast(#simd[4]i32)simd.swizzle(v, 3, 7, 11, 15)
	return
}

@(private)
pack_i16x16 :: #force_inline proc "contextless" (r0, r1, r2, r3: #simd[4]i32) -> Coeff4 {
	h0 := simd.shuffle(cast(#simd[4]i16)r0, cast(#simd[4]i16)r1, 0, 1, 2, 3, 4, 5, 6, 7)
	h1 := simd.shuffle(cast(#simd[4]i16)r2, cast(#simd[4]i16)r3, 0, 1, 2, 3, 4, 5, 6, 7)
	return simd.shuffle(h0, h1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)
}

// Forward 4x4 DCT of (src - pred), bit-identical to libwebp's FTransform_C.
@(private)
ftransform4 :: #force_inline proc "contextless" (src, pred: Block4) -> Coeff4 {
	c0, c1, c2, c3 := unpack_cols4x4(cast(Coeff4)src - cast(Coeff4)pred)

	a0, a1, a2, a3 := c0 + c3, c1 + c2, c1 - c2, c0 - c3
	t0 := (a0 + a1) * 8
	t1 := ashr4(a2 * 2217 + a3 * 5352 + 1812, 9)
	t2 := (a0 - a1) * 8
	t3 := ashr4(a3 * 2217 - a2 * 5352 + 937, 9)

	r0, r1, r2, r3 := transpose4x4(t0, t1, t2, t3)
	b0, b1, b2, b3 := r0 + r3, r1 + r2, r1 - r2, r0 - r3
	nz := simd.select(simd.lanes_ne(b3, 0), (#simd[4]i32)(1), (#simd[4]i32)(0))
	o0 := ashr4(b0 + b1 + 7, 4)
	o1 := ashr4(b2 * 2217 + b3 * 5352 + 12000, 16) + nz
	o2 := ashr4(b0 - b1 + 7, 4)
	o3 := ashr4(b3 * 2217 - b2 * 5352 + 51000, 16)
	return pack_i16x16(o0, o1, o2, o3)
}

// Inverse 4x4 DCT added to `pred`, bit-identical to vp8_recon.odin's add_residual.
@(private)
idct_add4 :: #force_inline proc "contextless" (pred: Block4, c: Coeff4, kind: u8) -> Block4 {
	if kind == 0 do return pred
	if kind == 1 {
		dc := i16((i32(simd.extract(c, 0)) + 4) >> 3)
		return cast(Block4)simd.clamp(cast(Coeff4)pred + Coeff4(dc), 0, 255)
	}
	mul1 :: #force_inline proc "contextless" (a: #simd[4]i32) -> #simd[4]i32 {return ashr4(a * 20091, 16) + a}
	mul2 :: #force_inline proc "contextless" (a: #simd[4]i32) -> #simd[4]i32 {return ashr4(a * 35468, 16)}

	r0, r1, r2, r3 := unpack_rows4x4(c)

	a, b := r0 + r2, r0 - r2
	cc, dd := mul2(r1) - mul1(r3), mul1(r1) + mul2(r3)
	u0, u1, u2, u3 := transpose4x4(a + dd, b + cc, b - cc, a - dd)

	dc := u0 + 4
	a2, b2 := dc + u2, dc - u2
	cc2, dd2 := mul2(u1) - mul1(u3), mul1(u1) + mul2(u3)
	w0, w1, w2, w3 := transpose4x4(ashr4(a2 + dd2, 3), ashr4(b2 + cc2, 3), ashr4(b2 - cc2, 3), ashr4(a2 - dd2, 3))
	return cast(Block4)simd.clamp(cast(Coeff4)pred + pack_i16x16(w0, w1, w2, w3), 0, 255)
}

// Forward and inverse 4x4 Walsh-Hadamard transforms of the 16 luma block DCs.
@(private)
fwht4 :: #force_inline proc "contextless" (dcs: [16]i16) -> Coeff4 {
	c0, c1, c2, c3 := unpack_cols4x4(transmute(Coeff4)dcs)
	a0, a1, a2, a3 := c0 + c2, c1 + c3, c1 - c3, c0 - c2
	r0, r1, r2, r3 := transpose4x4(a0 + a1, a3 + a2, a3 - a2, a0 - a1)
	b0, b1, b2, b3 := r0 + r2, r1 + r3, r1 - r3, r0 - r2
	return pack_i16x16(ashr4(b0 + b1, 1), ashr4(b3 + b2, 1), ashr4(b3 - b2, 1), ashr4(b0 - b1, 1))
}

@(private)
iwht4 :: #force_inline proc "contextless" (input: Coeff4) -> [16]i16 {
	r0, r1, r2, r3 := unpack_rows4x4(input)
	a0, a1, a2, a3 := r0 + r3, r1 + r2, r1 - r2, r0 - r3
	c0, c1, c2, c3 := transpose4x4(a0 + a1, a3 + a2, a0 - a1, a3 - a2)
	dc := c0 + 3
	b0, b1, b2, b3 := dc + c3, c1 + c2, c1 - c2, dc - c3
	o0, o1, o2, o3 := transpose4x4(ashr4(b0 + b1, 3), ashr4(b3 + b2, 3), ashr4(b0 - b1, 3), ashr4(b3 - b2, 3))
	return transmute([16]i16)pack_i16x16(o0, o1, o2, o3)
}

// Branchless SIMD quantization of a 4x4 block. Because zthresh was defined as
// floor(((1<<QFIX)-1-bias)/iq), `(coeff * iq + bias) >> QFIX` is already 0 iff
// coeff <= zthresh. Returns dequantized raster coeffs, zigzag levels, the
// 16-bit zigzag non-zero mask, `last` (-1 when empty), and `kind` (0/1/2).
@(private)
quantize4 :: #force_inline proc "contextless" (dct: Coeff4, m: ^Quant_Matrix) -> (deq: Coeff4, levels: Coeff4, nz_mask: u16, last: i8, kind: u8) {
	abs_c := cast(#simd[16]u16)simd.abs(dct) + m.sharpen
	raw := simd.shr(cast(#simd[16]u32)abs_c * m.iq + m.bias, QFIX)
	lvl := cast(Coeff4)simd.min(raw, MAX_LEVEL)
	if simd.reduce_or(lvl) == 0 do return {}, {}, 0, -1, 0
	signed := simd.select(simd.lanes_lt(dct, 0), -lvl, lvl)
	deq = signed * m.q
	levels = simd.swizzle(signed, 0, 1, 4, 8, 5, 2, 3, 6, 9, 12, 13, 10, 7, 11, 14, 15)
	nz_mask = transmute(u16)simd.extract_msbs(simd.lanes_ne(levels, 0))
	last = i8(15 - intrinsics.count_leading_zeros(nz_mask))
	kind = nz_mask > 1 ? 2 : 1
	return
}

@(private)
quantize_uv_dc :: #force_inline proc "contextless" (dc_in: i16, m: ^Quant_Matrix) -> (deq_dc: i16, level: i16, err: i8) {
	val := i32(dc_in)
	neg := val < 0
	av := u32(neg ? -val : val)
	if av > m.dc_zthr {
		lvl := i32((av * m.dc_iq + m.dc_bias) >> QFIX)
		if lvl > MAX_LEVEL do lvl = MAX_LEVEL
		qv := lvl * m.dc_q
		rem := i32(av) - qv
		return i16(neg ? -qv : qv), i16(neg ? -lvl : lvl), i8((neg ? -rem : rem) >> 1)
	}
	return 0, 0, i8(val >> 1)
}

// 4x4 SSE without i32 multiplies: |a - b|^2 <= 255^2 = 65025 <= max(u16).
@(private)
sse4 :: #force_inline proc "contextless" (a, b: Block4) -> int {
	d := cast(#simd[16]u16)(simd.max(a, b) - simd.min(a, b))
	return int(simd.reduce_add_bisect(cast(#simd[16]u32)(d * d)))
}

@(private)
ttransform4 :: #force_inline proc "contextless" (b: Block4) -> i32 {
	c0, c1, c2, c3 := unpack_cols4x4(cast(Coeff4)b)
	a0, a1, a2, a3 := c0 + c2, c1 + c3, c1 - c3, c0 - c2
	r0, r1, r2, r3 := transpose4x4(a0 + a1, a3 + a2, a3 - a2, a0 - a1)
	b0, b1, b2, b3 := r0 + r2, r1 + r3, r1 - r3, r0 - r2
	acc := WEIGHT_Y0 * simd.abs(b0 + b1) + WEIGHT_Y1 * simd.abs(b3 + b2) +
	       WEIGHT_Y2 * simd.abs(b3 - b2) + WEIGHT_Y3 * simd.abs(b0 - b1)
	return simd.reduce_add_bisect(acc)
}

// Generates all 10 Intra-4x4 predictions at once from the 13 border samples
// around `ws[at]` using two 16-lane averages over [L, L, K, J, I, X, A..H, H].
@(private)
make_preds4 :: #force_inline proc "contextless" (ws: ^[WORKSPACE]u8, at: int) -> [10]Block4 #no_bounds_check {
	I := ws[at - 1]
	J := ws[at + BPS - 1]
	K := ws[at + 2 * BPS - 1]
	L := ws[at + 3 * BPS - 1]
	X := ws[at - BPS - 1]
	top8 := (^[8]u8)(&ws[at - BPS])^

	strip := #simd[16]u8{L, L, K, J, I, X, top8[0], top8[1], top8[2], top8[3], top8[4], top8[5], top8[6], top8[7], top8[7], 0}
	s0 := cast(#simd[16]u16)strip
	s1 := simd.swizzle(s0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 15)
	s2 := simd.swizzle(s0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 15, 15)
	m2 := transmute([16]u8)cast(#simd[16]u8)simd.shr(s0 + s1 + 1, 1)
	m3 := transmute([16]u8)cast(#simd[16]u8)simd.shr(s0 + 2 * s1 + s2 + 2, 2)

	u32_at :: #force_inline proc "contextless" (a: ^[16]u8, i: int) -> u32 #no_bounds_check {
		return intrinsics.unaligned_load((^u32)(&a[i]))
	}
	rows4 :: #force_inline proc "contextless" (r0, r1, r2, r3: u32) -> Block4 {
		return transmute(Block4)[4]u32{r0, r1, r2, r3}
	}
	pack4 :: #force_inline proc "contextless" (b0, b1, b2, b3: u8) -> u32 {
		return transmute(u32)[4]u8{b0, b1, b2, b3}
	}

	dc := u8((int(top8[0]) + int(top8[1]) + int(top8[2]) + int(top8[3]) + int(I) + int(J) + int(K) + int(L) + 4) >> 3)
	top_tm := cast(Coeff4)simd.swizzle(strip, 6, 7, 8, 9, 6, 7, 8, 9, 6, 7, 8, 9, 6, 7, 8, 9)
	left_tm := cast(Coeff4)simd.swizzle(strip, 4, 4, 4, 4, 3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1)
	ve_row := u32_at(&m3, 5)

	preds: [10]Block4
	preds[B_DC] = Block4(dc)
	preds[B_TM] = cast(Block4)simd.clamp(top_tm + left_tm - Coeff4(i16(X)), 0, 255)
	preds[B_VE] = rows4(ve_row, ve_row, ve_row, ve_row)
	preds[B_HE] = rows4(u32(m3[3]) * 0x01010101, u32(m3[2]) * 0x01010101, u32(m3[1]) * 0x01010101, u32(m3[0]) * 0x01010101)
	preds[B_RD] = rows4(u32_at(&m3, 4), u32_at(&m3, 3), u32_at(&m3, 2), u32_at(&m3, 1))
	preds[B_VR] = rows4(u32_at(&m2, 5), u32_at(&m3, 4), pack4(m3[3], m2[5], m2[6], m2[7]), pack4(m3[2], m3[4], m3[5], m3[6]))
	preds[B_LD] = rows4(u32_at(&m3, 6), u32_at(&m3, 7), u32_at(&m3, 8), u32_at(&m3, 9))
	preds[B_VL] = rows4(u32_at(&m2, 6), u32_at(&m3, 6), pack4(m2[7], m2[8], m2[9], m3[10]), pack4(m3[7], m3[8], m3[9], m3[11]))
	preds[B_HD] = rows4(
		pack4(m2[4], m3[4], m3[5], m3[6]),
		pack4(m2[3], m3[3], m2[4], m3[4]),
		pack4(m2[2], m3[2], m2[3], m3[3]),
		pack4(m2[1], m3[1], m2[2], m3[2]),
	)
	preds[B_HU] = rows4(
		pack4(m2[3], m3[2], m2[2], m3[1]),
		pack4(m2[2], m3[1], m2[1], m3[0]),
		pack4(m2[1], m3[0], L, L),
		u32(L) * 0x01010101,
	)
	return preds
}

// Builds all 4 Intra-16x16 luma predictions directly as `[4][16]Block4` from
// `ws`'s filled borders without writing into `ws`.
@(private)
make_preds16 :: proc(ws: ^[WORKSPACE]u8, mb_x, mb_y: int, out: ^[4][16]Block4) #no_bounds_check {
	top_u8 := intrinsics.unaligned_load((^Block4)(&ws[Y_OFF - BPS]))
	top_i16 := cast(Coeff4)top_u8
	top_words := transmute([4]u32)top_u8
	corner := i16(ws[Y_OFF - BPS - 1])

	left_sum := 0
	for j in 0 ..< 16 do left_sum += int(ws[Y_OFF + j * BPS - 1])
	top_sum := int(simd.reduce_add_bisect(cast(#simd[16]u16)top_u8))

	dc: u8
	switch edge_mode(DC_PRED, mb_x, mb_y) {
	case DC_PRED:        dc = u8((top_sum + left_sum + 16) >> 5)
	case DC_NO_TOP:      dc = u8((left_sum + 8) >> 4)
	case DC_NO_LEFT:     dc = u8((top_sum + 8) >> 4)
	case DC_NO_TOP_LEFT: dc = 128
	}
	dc_blk := Block4(dc)
	for n in 0 ..< 16 do out[DC_PRED][n] = dc_blk

	for bx in 0 ..< 4 {
		w := top_words[bx]
		vb := transmute(Block4)[4]u32{w, w, w, w}
		for by in 0 ..< 4 do out[V_PRED][by * 4 + bx] = vb
	}

	for by in 0 ..< 4 {
		y0 := by * 4
		l0 := ws[Y_OFF + (y0 + 0) * BPS - 1]
		l1 := ws[Y_OFF + (y0 + 1) * BPS - 1]
		l2 := ws[Y_OFF + (y0 + 2) * BPS - 1]
		l3 := ws[Y_OFF + (y0 + 3) * BPS - 1]
		hb := transmute(Block4)[4]u32{
			u32(l0) * 0x01010101,
			u32(l1) * 0x01010101,
			u32(l2) * 0x01010101,
			u32(l3) * 0x01010101,
		}
		tm0 := transmute([4]u32)cast(Block4)simd.clamp(top_i16 + Coeff4(i16(l0) - corner), 0, 255)
		tm1 := transmute([4]u32)cast(Block4)simd.clamp(top_i16 + Coeff4(i16(l1) - corner), 0, 255)
		tm2 := transmute([4]u32)cast(Block4)simd.clamp(top_i16 + Coeff4(i16(l2) - corner), 0, 255)
		tm3 := transmute([4]u32)cast(Block4)simd.clamp(top_i16 + Coeff4(i16(l3) - corner), 0, 255)
		for bx in 0 ..< 4 {
			out[H_PRED][by * 4 + bx] = hb
			out[TM_PRED][by * 4 + bx] = transmute(Block4)[4]u32{tm0[bx], tm1[bx], tm2[bx], tm3[bx]}
		}
	}
}

// Builds all 4 8x8 chroma predictions for U (blocks 0..3) and V (blocks 4..7).
@(private)
make_preds_uv :: proc(ws: ^[WORKSPACE]u8, mb_x, mb_y: int, out: ^[4][8]Block4) #no_bounds_check {
	em := edge_mode(DC_PRED, mb_x, mb_y)
	for c in 0 ..< 2 {
		off := c == 0 ? U_OFF : V_OFF
		b_off := c * 4
		top8 := intrinsics.unaligned_load((^#simd[8]u8)(&ws[off - BPS]))
		top_i16 := cast(#simd[8]i16)top8
		top_words := transmute([2]u32)top8
		corner := i16(ws[off - BPS - 1])

		left_sum := 0
		for j in 0 ..< 8 do left_sum += int(ws[off + j * BPS - 1])
		top_sum := int(simd.reduce_add_bisect(cast(#simd[8]u16)top8))

		dc: u8
		switch em {
		case DC_PRED:        dc = u8((top_sum + left_sum + 8) >> 4)
		case DC_NO_TOP:      dc = u8((left_sum + 4) >> 3)
		case DC_NO_LEFT:     dc = u8((top_sum + 4) >> 3)
		case DC_NO_TOP_LEFT: dc = 128
		}
		dc_blk := Block4(dc)
		for n in 0 ..< 4 do out[DC_PRED][b_off + n] = dc_blk

		for bx in 0 ..< 2 {
			w := top_words[bx]
			vb := transmute(Block4)[4]u32{w, w, w, w}
			out[V_PRED][b_off + bx] = vb
			out[V_PRED][b_off + 2 + bx] = vb
		}

		for by in 0 ..< 2 {
			y0 := by * 4
			l0 := ws[off + (y0 + 0) * BPS - 1]
			l1 := ws[off + (y0 + 1) * BPS - 1]
			l2 := ws[off + (y0 + 2) * BPS - 1]
			l3 := ws[off + (y0 + 3) * BPS - 1]
			hb := transmute(Block4)[4]u32{
				u32(l0) * 0x01010101,
				u32(l1) * 0x01010101,
				u32(l2) * 0x01010101,
				u32(l3) * 0x01010101,
			}
			tm0 := transmute([2]u32)cast(#simd[8]u8)simd.clamp(top_i16 + (#simd[8]i16)(i16(l0) - corner), 0, 255)
			tm1 := transmute([2]u32)cast(#simd[8]u8)simd.clamp(top_i16 + (#simd[8]i16)(i16(l1) - corner), 0, 255)
			tm2 := transmute([2]u32)cast(#simd[8]u8)simd.clamp(top_i16 + (#simd[8]i16)(i16(l2) - corner), 0, 255)
			tm3 := transmute([2]u32)cast(#simd[8]u8)simd.clamp(top_i16 + (#simd[8]i16)(i16(l3) - corner), 0, 255)
			for bx in 0 ..< 2 {
				out[H_PRED][b_off + by * 2 + bx] = hb
				out[TM_PRED][b_off + by * 2 + bx] = transmute(Block4)[4]u32{tm0[bx], tm1[bx], tm2[bx], tm3[bx]}
			}
		}
	}
}

@(private)
dct_alpha_blocks :: proc "contextless" (src, pred: []Block4) -> int #no_bounds_check {
	dist: [32]u16
	for n in 0 ..< len(src) {
		bins := transmute([16]u16)simd.min(simd.shr(cast(#simd[16]u16)simd.abs(ftransform4(src[n], pred[n])), 3), 31)
		for b in bins do dist[b] += 1
	}
	max_val, last_nz := 0, 1
	for v, k in dist do if v > 0 {
		if int(v) > max_val do max_val = int(v)
		last_nz = k
	}
	return max_val > 1 ? 510 * last_nz / max_val : 0
}

@(private)
analyze_segments :: proc(e: ^Lossy_Encoder, pool: ^common.Work_Pool) {
	rows :: proc(data: rawptr, first, end: int) #no_bounds_check {
		e := (^Lossy_Encoder)(data)
		src_d := Vp8{
			frame = {
				y         = e.src_y,
				u         = e.src_u,
				v         = e.src_v,
				y_stride  = e.d.y_stride,
				uv_stride = e.d.uv_stride,
			},
			mb_w = e.d.mb_w,
			mb_h = e.d.mb_h,
		}
		ws: [WORKSPACE]u8
		src_y4: [16]Block4
		src_uv4: [8]Block4
		preds_y: [4][16]Block4
		preds_uv: [4][8]Block4
		for mb_y in first ..< end {
			rstat := &e.row_stats[mb_y]
			rstat^ = {}
			for mb_x in 0 ..< e.d.mb_w {
				load_mb_source(e, mb_x, mb_y, &src_y4, &src_uv4)
				fill_borders(&src_d, &ws, mb_x, mb_y)
				make_preds16(&ws, mb_x, mb_y, &preds_y)
				make_preds_uv(&ws, mb_x, mb_y, &preds_uv)

				best_y := max(dct_alpha_blocks(src_y4[:], preds_y[DC_PRED][:]), dct_alpha_blocks(src_y4[:], preds_y[TM_PRED][:]))
				best_uv := max(dct_alpha_blocks(src_uv4[:], preds_uv[DC_PRED][:]), dct_alpha_blocks(src_uv4[:], preds_uv[TM_PRED][:]))
				mixed := clamp(255 - ((3 * best_y + best_uv + 2) >> 2), 0, 255)
				e.mbs[mb_y * e.d.mb_w + mb_x] = {alpha = u8(mixed)}
				rstat.alphas[mixed] += 1
				rstat.uv_alpha += i64(best_uv)
			}
		}
	}
	if pool != nil && pool.run != nil && e.d.mb_h > 1 {
		pool.run(pool, e.d.mb_h, rows, e)
	} else {
		rows(e, 0, e.d.mb_h)
	}

	alphas: [256]int
	uv_sum := i64(0)
	for &r in e.row_stats {
		for c, i in r.alphas do alphas[i] += int(c)
		uv_sum += r.uv_alpha
	}
	total_mbs := e.d.mb_w * e.d.mb_h
	uv_alpha := int(uv_sum / i64(total_mbs))
	e.dq_uv_ac = clamp(((uv_alpha - 64) * 10 / 70) * 50 / 100, -4, 6)
	e.dq_uv_dc = -2

	e.num_segments = 4
	min_a, max_a := 0, 255
	for min_a < 255 && alphas[min_a] == 0 do min_a += 1
	for max_a > min_a && alphas[max_a] == 0 do max_a -= 1
	range_a := max_a - min_a

	centers: [4]int
	for k in 0 ..< 4 do centers[k] = min_a + ((2 * k + 1) * range_a) / 8

	seg_map: [256]u8
	weighted_avg := min_a
	for _ in 0 ..< 6 {
		accum, dist_accum: [4]int
		n := 0
		for a in min_a ..= max_a do if alphas[a] > 0 {
			for n + 1 < 4 && abs(a - centers[n + 1]) < abs(a - centers[n]) do n += 1
			seg_map[a] = u8(n)
			dist_accum[n] += a * alphas[a]
			accum[n] += alphas[a]
		}
		displaced, wsum, wtot := 0, 0, 0
		for k in 0 ..< 4 do if accum[k] > 0 {
			nc := (dist_accum[k] + accum[k] / 2) / accum[k]
			displaced += abs(centers[k] - nc)
			centers[k] = nc
			wsum += nc * accum[k]
			wtot += accum[k]
		}
		if wtot > 0 do weighted_avg = (wsum + wtot / 2) / wtot
		if displaced < 5 do break
	}

	mb_slice := e.mbs[:total_mbs]
	for &mb in mb_slice do mb.segment = seg_map[mb.alpha]

	cmin, cmax := centers[0], centers[0]
	for c in centers {
		cmin = min(cmin, c)
		cmax = max(cmax, c)
	}
	span := max(cmax - cmin, 1)
	for s in 0 ..< 4 {
		e.dqm[s] = {
			alpha = clamp(255 * (centers[s] - weighted_avg) / span, -127, 127),
			beta  = clamp(255 * (centers[s] - cmin) / span, 0, 255),
		}
	}
}

@(private)
expand_matrix :: proc(m: ^Quant_Matrix, q_dc, q_ac: u32, kind: int, zero_dc := false) -> i64 {
	q_pair := [2]u32{q_dc, q_ac}
	iq_pair, bias_pair, z_pair: [2]u32
	for i in 0 ..< 2 {
		iq_pair[i] = (1 << QFIX) / q_pair[i]
		bias_pair[i] = BIAS_MATRICES[kind][i] << (QFIX - 8)
		z_pair[i] = ((1 << QFIX) - 1 - bias_pair[i]) / iq_pair[i]
	}
	q_arr: [16]i16
	iq_arr, bias_arr: [16]u32
	sh_arr: [16]u16
	sum := u32(0)
	for i in 0 ..< 16 {
		idx := i > 0 ? 1 : 0
		q_arr[i] = i16(q_pair[idx])
		iq_arr[i] = iq_pair[idx]
		bias_arr[i] = bias_pair[idx]
		sh_arr[i] = kind == 0 ? u16((u32(FREQ_SHARPENING[i]) * q_pair[idx]) >> 11) : 0
		sum += q_pair[idx]
	}
	if zero_dc {
		q_arr[0], iq_arr[0], bias_arr[0], sh_arr[0] = 0, 0, 0, 0
	}
	m^ = {
		q       = transmute(Coeff4)q_arr,
		iq      = transmute(#simd[16]u32)iq_arr,
		bias    = transmute(#simd[16]u32)bias_arr,
		sharpen = transmute(#simd[16]u16)sh_arr,
		dc_q    = i32(q_pair[0]),
		dc_iq   = iq_pair[0],
		dc_bias = bias_pair[0],
		dc_zthr = z_pair[0],
	}
	return i64((sum + 8) >> 4)
}

@(private)
setup_segment_params :: proc(e: ^Lossy_Encoder, quality: int) {
	total_mbs := e.d.mb_w * e.d.mb_h
	mb_slice := e.mbs[:total_mbs]
	qn := f64(quality) / 100.0
	lin := qn < 0.75 ? qn * (2.0 / 3.0) : 2.0 * qn - 1.0
	c_base := lin > 0.0 ? math.pow(lin, 1.0 / 3.0) : 0.0
	amp := 0.9 * 50.0 / 100.0 / 128.0

	for s in 0 ..< e.num_segments {
		m := &e.dqm[s]
		c := c_base > 0.0 ? math.pow(c_base, 1.0 - amp * f64(m.alpha)) : 0.0
		m.quant = clamp(int(127.0 * (1.0 - c)), 0, 127)
		base_f := clamp(int(AC_STEP[m.quant]) >> 2, 0, 63)
		f := base_f * 300 / (256 + m.beta)
		m.fstrength = f < 2 ? 0 : min(f, 63)
	}
	e.base_quant = e.dqm[0].quant

	remap := [4]u8{0, 1, 2, 3}
	final_segs := 1
	for s1 in 1 ..< e.num_segments {
		found := -1
		for s2 in 0 ..< final_segs {
			if e.dqm[s1].quant == e.dqm[s2].quant && e.dqm[s1].fstrength == e.dqm[s2].fstrength {
				found = s2
				break
			}
		}
		if found >= 0 {
			remap[s1] = u8(found)
		} else {
			remap[s1] = u8(final_segs)
			if final_segs != s1 do e.dqm[final_segs] = e.dqm[s1]
			final_segs += 1
		}
	}
	if final_segs < e.num_segments {
		for &mb in mb_slice do mb.segment = remap[mb.segment]
		e.num_segments = final_segs
		for s in final_segs ..< 4 do e.dqm[s] = e.dqm[final_segs - 1]
	}

	counts: [4]int
	for &mb in mb_slice do counts[mb.segment] += 1
	seg_prob :: proc(a, b: int) -> u8 {
		tot := a + b
		return tot == 0 ? 255 : u8((255 * a + tot / 2) / tot)
	}
	e.d.segments = {enabled = e.num_segments > 1, absolute = true, tree_probs = {255, 255, 255}}
	if e.num_segments > 1 {
		tp := [3]u8{
			seg_prob(counts[0] + counts[1], counts[2] + counts[3]),
			seg_prob(counts[0], counts[1]),
			seg_prob(counts[2], counts[3]),
		}
		e.d.segments.tree_probs = tp
		e.d.segments.update_map = tp[0] != 255 || tp[1] != 255 || tp[2] != 255
		if !e.d.segments.update_map {
			for &mb in mb_slice do mb.segment = 0
		}
	}

	for s in 0 ..< e.num_segments {
		m := &e.dqm[s]
		q := clamp(m.quant, 0, 127)
		y1_dc := u32(DC_STEP[q])
		y1_ac := u32(AC_STEP[q])
		y2_dc := u32(DC_STEP[q]) * 2
		y2_ac := u32(max((i32(AC_STEP[q]) * 101581) >> 16, 8))
		uv_dc := u32(DC_STEP[clamp(q + e.dq_uv_dc, 0, 117)])
		uv_ac := u32(AC_STEP[clamp(q + e.dq_uv_ac, 0, 127)])

		q_i4 := expand_matrix(&m.y1, y1_dc, y1_ac, 0)
		_ = expand_matrix(&m.y1_ac, y1_dc, y1_ac, 0, true)
		q_i16 := expand_matrix(&m.y2, y2_dc, y2_ac, 1)
		q_uv := expand_matrix(&m.uv, uv_dc, uv_ac, 2)

		m.lambda_i4 = max((3 * q_i4 * q_i4) >> 7, 1)
		m.lambda_i16 = max(3 * q_i16 * q_i16, 1)
		m.lambda_uv = max((3 * q_uv * q_uv) >> 6, 1)
		m.lambda_mode = max((q_i4 * q_i4) >> 7, 1)
		m.tlambda = max((50 * q_i4) >> 5, 1)
		m.min_disto = 20 * int(y1_dc)
		m.max_edge = 0
	}
}

@(private)
update_level_costs :: proc(t: ^Enc_Tables) #no_bounds_check {
	for ctype in 0 ..< 4 do for band in 0 ..< 8 do for ctx in 0 ..< 3 {
		p := &t.probs[ctype][band][ctx]
		dst := &t.level_cost[ctype][band][ctx]
		cost0 := ctx > 0 ? bit_cost(1, p[0]) : 0
		base := bit_cost(1, p[1]) + cost0
		dst[0] = u16(bit_cost(0, p[1]) + cost0)
		for v in 1 ..= MAX_VARIABLE_LEVEL {
			pat := int(LEVEL_CODES[v - 1][0])
			bits := int(LEVEL_CODES[v - 1][1])
			c := base + int(LEVEL_FIXED_COST[v])
			for i := 2; pat != 0; i += 1 {
				if pat & 1 != 0 do c += bit_cost(bits & 1, p[i])
				bits >>= 1
				pat >>= 1
			}
			dst[v] = u16(c)
		}
	}
}

@(private)
residual_cost :: #force_inline proc "contextless" (t: ^Enc_Tables, ctype, first, ctx0: int, levels: ^Coeff4, last: i8) -> i64 #no_bounds_check {
	probs := &t.probs[ctype]
	p0 := probs[first][ctx0][0]
	if int(last) < first do return i64(bit_cost(0, p0))
	lv := (^[16]i16)(levels)
	cost := ctx0 == 0 ? bit_cost(1, p0) : 0
	tab := &t.level_cost[ctype][BANDS[first]][ctx0]
	last_i := int(last)
	for n in first ..< last_i {
		v := int(abs(lv[n]))
		if v <= MAX_VARIABLE_LEVEL {
			cost += int(tab[v])
		} else {
			cost += int(tab[MAX_VARIABLE_LEVEL]) - int(LEVEL_FIXED_COST[MAX_VARIABLE_LEVEL]) + int(LEVEL_FIXED_COST[v])
		}
		tab = &t.level_cost[ctype][BANDS[n + 1]][min(v, 2)]
	}
	v := int(abs(lv[last_i]))
	if v <= MAX_VARIABLE_LEVEL {
		cost += int(tab[v])
	} else {
		cost += int(tab[MAX_VARIABLE_LEVEL]) - int(LEVEL_FIXED_COST[MAX_VARIABLE_LEVEL]) + int(LEVEL_FIXED_COST[v])
	}
	if last_i < 15 {
		cost += bit_cost(0, probs[BANDS[last_i + 1]][v == 1 ? 1 : 2][0])
	}
	return i64(cost)
}

@(private)
rd_score :: #force_inline proc "contextless" (lambda, rate, disto: i64) -> i64 {
	return rate * lambda + RD_DISTO_MULT * disto
}

@(private)
Mb_Choice :: struct {
	is_4x4:    bool,
	ymode:     u8,
	uvmode:    u8,
	y_dc_last: i8,
	modes:     [16]u8,
	y_last:    [16]i8,
	uv_last:   [8]i8,
	derr:      [2][3]i8,
	y_dc:      Coeff4,
	y_ac:      [16]Coeff4,
	uv:        [8]Coeff4,
	rec_y:     [16]Block4,
	rec_uv:    [8]Block4,
}

@(private)
decide_macroblock :: proc(
	e: ^Lossy_Encoder,
	src_y4: ^[16]Block4,
	src_uv4: ^[8]Block4,
	ws: ^[WORKSPACE]u8,
	mb_x, mb_y, seg: int,
	intra_left: ^[4]u8,
	nz_left, nz_dc_left: u8,
	left_derr: ^[2][2]i8,
	choice: ^Mb_Choice,
) #no_bounds_check {
	dqm := &e.dqm[seg]
	tabs := e.tables

	// Precompute source 4x4 Hadamard weights once for all I16 + I4 candidates.
	src_tw: [16]i32
	first_row := simd.extract(transmute(#simd[4]u32)src_y4[0], 0)
	flat_val := Block4(u8(first_row))
	flat_src := true
	for n in 0 ..< 16 {
		src_tw[n] = ttransform4(src_y4[n])
		if flat_src && transmute(u16)simd.extract_msbs(simd.lanes_ne(src_y4[n], flat_val)) != 0 {
			flat_src = false
		}
	}

	preds16: [4][16]Block4
	make_preds16(ws, mb_x, mb_y, &preds16)

	best_i16_score := max(i64)
	i16_D, i16_SD, i16_H, i16_R: i64
	dc_ctx := int(e.d.nz_dc_top[mb_x] + nz_dc_left)

	for mode in 0 ..< 4 {
		dequant: [16]Coeff4
		dcs_in: [16]i16
		for n in 0 ..< 16 {
			dequant[n] = ftransform4(src_y4[n], preds16[mode][n])
			dcs_in[n] = simd.extract(dequant[n], 0)
		}
		dc_wht, dc_levels, _, dc_last, _ := quantize4(fwht4(dcs_in), &dqm.y2)
		dc_recon := iwht4(dc_wht)

		ac_levels: [16]Coeff4
		ac_last: [16]i8
		rec16: [16]Block4
		ac_masks := u16(0)
		D, SD_raw := 0, 0
		for n in 0 ..< 16 {
			deq_ac, lvls, mask, last, kind := quantize4(dequant[n], &dqm.y1_ac)
			ac_levels[n] = lvls
			ac_last[n] = last
			ac_masks |= mask
			dc_n := dc_recon[n]
			if kind == 0 && dc_n != 0 do kind = 1
			deq_ac = simd.replace(deq_ac, 0, dc_n)
			rec := idct_add4(preds16[mode][n], deq_ac, kind)
			rec16[n] = rec
			D += sse4(src_y4[n], rec)
			SD_raw += int(abs(ttransform4(rec) - src_tw[n]) >> 5)
		}

		D64 := i64(D)
		SD := (dqm.tlambda * i64(SD_raw) + 128) >> 8
		H := I16_MODE_COST[mode]
		R := residual_cost(tabs, 1, 0, dc_ctx, &dc_levels, dc_last)
		top := e.d.nz_top[mb_x]
		left := nz_left
		for y in 0 ..< 4 {
			l := (left >> uint(y)) & 1
			for x in 0 ..< 4 {
				t := (top >> uint(x)) & 1
				b := y * 4 + x
				R += residual_cost(tabs, 0, 1, int(l + t), &ac_levels[b], ac_last[b])
				l = u8(ac_last[b] >= 1)
				top = top &~ (1 << uint(x)) | l << uint(x)
			}
			left = left &~ (1 << uint(y)) | l << uint(y)
		}
		if flat_src {
			flat_src = ac_masks == 0
			if flat_src {
				D64 *= 2
				SD *= 2
			}
		}
		score := rd_score(dqm.lambda_i16, R + H, D64 + SD)
		if score < best_i16_score {
			best_i16_score = score
			i16_D, i16_SD, i16_H, i16_R = D64, SD, H, R
			choice.is_4x4 = false
			choice.ymode = u8(mode)
			choice.y_dc = dc_levels
			choice.y_dc_last = dc_last
			choice.y_ac = ac_levels
			choice.y_last = ac_last
			choice.rec_y = rec16
		}
	}

	i16_mode_score := rd_score(dqm.lambda_mode, i16_R + i16_H, i16_D + i16_SD)

	// Intra-4x4 subblock search with early bail-out against Intra-16x16.
	// `ws` borders are still untouched, so `ws` serves directly as the running
	// reconstruction surface without copying the workspace.
	if e.max_i4_bits > 0 {
		top_modes := (^[4]u8)(&e.d.intra_top[mb_x * 4])^
		left_modes := intra_left^
		top_nz := e.d.nz_top[mb_x]
		left_nz := nz_left
		modes_i4: [16]u8
		i4_levels: [16]Coeff4
		i4_last: [16]i8
		i4_rec: [16]Block4
		i4_bits := i64(0)
		acc_score := rd_score(dqm.lambda_mode, 211, 0) // bit_cost(0, 145) == 211
		keep_i4 := true

		for n in 0 ..< 16 {
			bx, by := n & 3, n >> 2
			at := Y_OFF + bx * 4 + by * 4 * BPS
			mcosts := &BMODE_COST[top_modes[bx]][left_modes[by]]
			nz_ctx := int(((top_nz >> uint(bx)) & 1) + ((left_nz >> uint(by)) & 1))
			preds := make_preds4(ws, at)
			src4 := src_y4[n]
			tw_src := src_tw[n]

			best_sub_score := max(i64)
			best_sub_mode := u8(0)
			best_D, best_SD, best_H, best_R: i64
			best_blk_levels: Coeff4
			best_blk_last := i8(-1)
			best_rec: Block4

			for mode in 0 ..< 10 {
				pred := preds[mode]
				deq, lvls, nz_mask, last, kind := quantize4(ftransform4(src4, pred), &dqm.y1)
				rec := idct_add4(pred, deq, kind)

				D := i64(sse4(src4, rec))
				SD := (dqm.tlambda * i64(abs(ttransform4(rec) - tw_src) >> 5) + 128) >> 8
				H := i64(mcosts[mode])
				R := mode > 0 && intrinsics.count_ones(nz_mask &~ 1) <= FLATNESS_LIMIT_I4 ? i64(FLATNESS_PENALTY) : 0
				if best_sub_score < max(i64) && rd_score(dqm.lambda_i4, R + H, D + SD) >= best_sub_score {
					continue
				}
				R += residual_cost(tabs, 3, 0, nz_ctx, &lvls, last)
				sc := rd_score(dqm.lambda_i4, R + H, D + SD)
				if sc < best_sub_score {
					best_sub_score = sc
					best_sub_mode = u8(mode)
					best_D, best_SD, best_H, best_R = D, SD, H, R
					best_blk_levels = lvls
					best_blk_last = last
					best_rec = rec
				}
			}

			acc_score += rd_score(dqm.lambda_mode, best_R + best_H, best_D + best_SD)
			i4_bits += best_H
			if acc_score >= i16_mode_score || i4_bits > e.max_i4_bits {
				keep_i4 = false
				break
			}
			store_ws4(ws, at, best_rec)
			i4_rec[n] = best_rec
			modes_i4[n] = best_sub_mode
			top_modes[bx] = best_sub_mode
			left_modes[by] = best_sub_mode
			i4_levels[n] = best_blk_levels
			i4_last[n] = best_blk_last
			nz_bit := u8(best_blk_last >= 0)
			top_nz = top_nz &~ (1 << uint(bx)) | nz_bit << uint(bx)
			left_nz = left_nz &~ (1 << uint(by)) | nz_bit << uint(by)
		}

		if keep_i4 {
			choice.is_4x4 = true
			choice.modes = modes_i4
			choice.y_ac = i4_levels
			choice.y_last = i4_last
			choice.rec_y = i4_rec
		}
	}

	if !choice.is_4x4 {
		ac_any := false
		for l in choice.y_last do ac_any ||= l >= 1
		if choice.y_dc_last >= 0 && !ac_any && int(i16_D) > dqm.min_disto {
			dc_arr := (^[16]i16)(&choice.y_dc)
			m_edge := max(int(abs(dc_arr[1])), max(int(abs(dc_arr[2])), int(abs(dc_arr[4]))))
			if m_edge > dqm.max_edge do dqm.max_edge = m_edge
		}
		(^[4]u8)(&e.d.intra_top[mb_x * 4])^ = choice.ymode
		intra_left^ = choice.ymode
	} else {
		for y in 0 ..< 4 {
			intra_left[y] = choice.modes[y * 4 + 3]
			e.d.intra_top[mb_x * 4 + y] = choice.modes[12 + y]
		}
	}

	// 8x8 chroma mode selection with optional DC quantization error diffusion.
	preds_uv: [4][8]Block4
	make_preds_uv(ws, mb_x, mb_y, &preds_uv)
	best_uv_score := max(i64)

	for mode in 0 ..< 4 {
		uv_lvls: [8]Coeff4
		uv_last: [8]i8
		rec_uv: [8]Block4
		derr: [2][3]i8
		ac_nz := 0
		D := 0

		for c in 0 ..< 2 {
			b := c * 4
			dct: [4]Coeff4
			for n in 0 ..< 4 do dct[n] = ftransform4(src_uv4[b + n], preds_uv[mode][b + n])

			if e.use_derr {
				top_d := &e.top_derr[mb_x][c]
				left_d := &left_derr[c]
				dc0_in := simd.extract(dct[0], 0) + i16((7 * int(top_d[0]) + 8 * int(left_d[0])) >> 3)
				q0, dc0, e0 := quantize_uv_dc(dc0_in, &dqm.uv)
				dc1_in := simd.extract(dct[1], 0) + i16((7 * int(top_d[1]) + 8 * int(e0)) >> 3)
				q1, dc1, e1 := quantize_uv_dc(dc1_in, &dqm.uv)
				dc2_in := simd.extract(dct[2], 0) + i16((7 * int(e0) + 8 * int(left_d[1])) >> 3)
				q2, dc2, e2 := quantize_uv_dc(dc2_in, &dqm.uv)
				dc3_in := simd.extract(dct[3], 0) + i16((7 * int(e1) + 8 * int(e2)) >> 3)
				q3, dc3, e3 := quantize_uv_dc(dc3_in, &dqm.uv)
				derr[c] = {e1, e2, e3}

				qs := [4]i16{q0, q1, q2, q3}
				dcs := [4]i16{dc0, dc1, dc2, dc3}
				for n in 0 ..< 4 {
					deq, lvls, mask, last, kind := quantize4(simd.replace(dct[n], 0, 0), &dqm.uv)
					lvls = simd.replace(lvls, 0, dcs[n])
					deq = simd.replace(deq, 0, qs[n])
					if last < 0 && dcs[n] != 0 do last = 0
					if kind == 0 && qs[n] != 0 do kind = 1
					ac_nz += int(intrinsics.count_ones(mask &~ 1))
					uv_lvls[b + n] = lvls
					uv_last[b + n] = last
					rec := idct_add4(preds_uv[mode][b + n], deq, kind)
					rec_uv[b + n] = rec
					D += sse4(src_uv4[b + n], rec)
				}
			} else {
				for n in 0 ..< 4 {
					deq, lvls, mask, last, kind := quantize4(dct[n], &dqm.uv)
					ac_nz += int(intrinsics.count_ones(mask &~ 1))
					uv_lvls[b + n] = lvls
					uv_last[b + n] = last
					rec := idct_add4(preds_uv[mode][b + n], deq, kind)
					rec_uv[b + n] = rec
					D += sse4(src_uv4[b + n], rec)
				}
			}
		}

		H := UV_MODE_COST[mode]
		R := i64(0)
		top := e.d.nz_top[mb_x]
		left := nz_left
		for c in 0 ..< 2 {
			shift := uint(4 + 2 * c)
			for y in 0 ..< 2 {
				l := (left >> (shift + uint(y))) & 1
				for x in 0 ..< 2 {
					t := (top >> (shift + uint(x))) & 1
					idx := c * 4 + y * 2 + x
					R += residual_cost(tabs, 2, 0, int(l + t), &uv_lvls[idx], uv_last[idx])
					l = u8(uv_last[idx] >= 0)
					top = top &~ (1 << (shift + uint(x))) | l << (shift + uint(x))
				}
				left = left &~ (1 << (shift + uint(y))) | l << (shift + uint(y))
			}
		}
		if mode > 0 && ac_nz <= FLATNESS_LIMIT_UV {
			R += FLATNESS_PENALTY * 8
		}
		sc := rd_score(dqm.lambda_uv, R + H, i64(D))
		if sc < best_uv_score {
			best_uv_score = sc
			choice.uvmode = u8(mode)
			choice.uv = uv_lvls
			choice.uv_last = uv_last
			choice.derr = derr
			choice.rec_uv = rec_uv
		}
	}

	if e.use_derr {
		for c in 0 ..< 2 {
			l1 := i8((3 * int(choice.derr[c][2])) >> 2)
			left_derr[c] = {choice.derr[c][0], l1}
			e.top_derr[mb_x][c] = {choice.derr[c][1], choice.derr[c][2] - l1}
		}
	}
}

@(private)
put_dyn_tok :: #force_inline proc "contextless" (dst: [^]u16, pos: ^int, stats: ^[4][8][3][11][2]u32, ctype, band, ctx, node, bit: int) #no_bounds_check {
	id := u16(node + 11 * (ctx + 3 * (band + 8 * ctype)))
	dst[pos^] = u16(bit << 15) | id
	pos^ += 1
	s := &stats[ctype][band][ctx][node]
	s[0] += u32(bit)
	s[1] += 1
}

@(private)
put_const_tok :: #force_inline proc "contextless" (dst: [^]u16, pos: ^int, bit: int, prob: u8) #no_bounds_check {
	dst[pos^] = u16(bit << 15) | FIXED_PROB_FLAG | u16(prob)
	pos^ += 1
}

@(private)
record_block_tokens :: proc "contextless" (
	dst: [^]u16,
	pos: ^int,
	stats: ^[4][8][3][11][2]u32,
	ctype, first, ctx0: int,
	levels: ^Coeff4,
	last: i8,
) -> u8 #no_bounds_check {
	n := first
	band := int(BANDS[n])
	ctx := ctx0
	last_i := int(last)
	if last_i < first {
		put_dyn_tok(dst, pos, stats, ctype, band, ctx, 0, 0)
		return 0
	}
	lv := (^[16]i16)(levels)
	put_dyn_tok(dst, pos, stats, ctype, band, ctx, 0, 1)
	for n < 16 {
		c := int(lv[n])
		n += 1
		neg := int(c < 0)
		v := neg != 0 ? -c : c
		if v == 0 {
			put_dyn_tok(dst, pos, stats, ctype, band, ctx, 1, 0)
			band = int(BANDS[n])
			ctx = 0
			continue
		}
		put_dyn_tok(dst, pos, stats, ctype, band, ctx, 1, 1)
		if v == 1 {
			put_dyn_tok(dst, pos, stats, ctype, band, ctx, 2, 0)
			band = int(BANDS[n])
			ctx = 1
		} else {
			put_dyn_tok(dst, pos, stats, ctype, band, ctx, 2, 1)
			if v <= 4 {
				put_dyn_tok(dst, pos, stats, ctype, band, ctx, 3, 0)
				if v == 2 {
					put_dyn_tok(dst, pos, stats, ctype, band, ctx, 4, 0)
				} else {
					put_dyn_tok(dst, pos, stats, ctype, band, ctx, 4, 1)
					put_dyn_tok(dst, pos, stats, ctype, band, ctx, 5, int(v == 4))
				}
			} else if v <= 10 {
				put_dyn_tok(dst, pos, stats, ctype, band, ctx, 3, 1)
				put_dyn_tok(dst, pos, stats, ctype, band, ctx, 6, 0)
				if v <= 6 {
					put_dyn_tok(dst, pos, stats, ctype, band, ctx, 7, 0)
					put_const_tok(dst, pos, int(v == 6), 159)
				} else {
					put_dyn_tok(dst, pos, stats, ctype, band, ctx, 7, 1)
					put_const_tok(dst, pos, int(v >= 9), 165)
					put_const_tok(dst, pos, (v - 7) & 1, 145)
				}
			} else {
				put_dyn_tok(dst, pos, stats, ctype, band, ctx, 3, 1)
				put_dyn_tok(dst, pos, stats, ctype, band, ctx, 6, 1)
				r := v - 3
				switch {
				case r < 16:
					put_dyn_tok(dst, pos, stats, ctype, band, ctx, 8, 0)
					put_dyn_tok(dst, pos, stats, ctype, band, ctx, 9, 0)
					r -= 8
					for i in 0 ..< 3 do put_const_tok(dst, pos, (r >> uint(2 - i)) & 1, CAT3_PROBS[i])
				case r < 32:
					put_dyn_tok(dst, pos, stats, ctype, band, ctx, 8, 0)
					put_dyn_tok(dst, pos, stats, ctype, band, ctx, 9, 1)
					r -= 16
					for i in 0 ..< 4 do put_const_tok(dst, pos, (r >> uint(3 - i)) & 1, CAT4_PROBS[i])
				case r < 64:
					put_dyn_tok(dst, pos, stats, ctype, band, ctx, 8, 1)
					put_dyn_tok(dst, pos, stats, ctype, band, ctx, 10, 0)
					r -= 32
					for i in 0 ..< 5 do put_const_tok(dst, pos, (r >> uint(4 - i)) & 1, CAT5_PROBS[i])
				case:
					put_dyn_tok(dst, pos, stats, ctype, band, ctx, 8, 1)
					put_dyn_tok(dst, pos, stats, ctype, band, ctx, 10, 1)
					r -= 64
					for i in 0 ..< 11 do put_const_tok(dst, pos, (r >> uint(10 - i)) & 1, CAT6_PROBS[i])
				}
			}
			band = int(BANDS[n])
			ctx = 2
		}
		put_const_tok(dst, pos, neg, 128)
		if n == 16 || n > last_i {
			if n < 16 do put_dyn_tok(dst, pos, stats, ctype, band, ctx, 0, 0)
			return 1
		}
		put_dyn_tok(dst, pos, stats, ctype, band, ctx, 0, 1)
	}
	return 1
}

@(private)
finalize_token_probs :: proc(t: ^Enc_Tables) #no_bounds_check {
	for ti in 0 ..< 4 do for bi in 0 ..< 8 do for ci in 0 ..< 3 do for pi in 0 ..< 11 {
		ones := int(t.stats[ti][bi][ci][pi][0])
		tot := int(t.stats[ti][bi][ci][pi][1])
		old_p := COEFF_PROBS[ti][bi][ci][pi]
		upd_p := COEFF_UPDATE_PROBS[ti][bi][ci][pi]
		new_p := tot > 0 && ones > 0 ? u8(255 - (ones * 255) / tot) : 255
		old_cost := ones * bit_cost(1, old_p) + (tot - ones) * bit_cost(0, old_p) + bit_cost(0, upd_p)
		new_cost := ones * bit_cost(1, new_p) + (tot - ones) * bit_cost(0, new_p) + bit_cost(1, upd_p) + 8 * 256
		t.probs[ti][bi][ci][pi] = new_cost < old_cost ? new_p : old_p
	}
}

@(private)
encode_vp8_pass :: proc(e: ^Lossy_Encoder) -> Error {
	tabs := e.tables
	tabs.probs = COEFF_PROBS
	tabs.stats = {}
	tabs.skip_stats = {}
	update_level_costs(tabs)
	clear(&e.tokens)

	total_mbs := e.d.mb_w * e.d.mb_h
	if cap(e.tokens) < total_mbs * 32 {
		if non_zero_reserve(&e.tokens, max(total_mbs * 32, MAX_MB_TOKENS)) != nil do return .Out_Of_Memory
	}

	for &v in e.d.intra_top do v = B_DC
	for &v in e.d.nz_top do v = 0
	for &v in e.d.nz_dc_top do v = 0
	for &v in e.top_derr do v = {}
	for s in 0 ..< e.num_segments do e.dqm[s].max_edge = 0

	refresh_period := max(total_mbs >> 3, 96)
	refresh_cnt := refresh_period
	skip_count := 0
	tok_pos := 0

	ws: [WORKSPACE]u8
	src_y4: [16]Block4
	src_uv4: [8]Block4
	choice: Mb_Choice

	#no_bounds_check for mb_y in 0 ..< e.d.mb_h {
		intra_left := [4]u8{B_DC, B_DC, B_DC, B_DC}
		nz_left, nz_dc_left: u8
		left_derr: [2][2]i8
		for mb_x in 0 ..< e.d.mb_w {
			refresh_cnt -= 1
			if refresh_cnt < 0 {
				finalize_token_probs(tabs)
				update_level_costs(tabs)
				refresh_cnt = refresh_period
			}
			if cap(e.tokens) - tok_pos < MAX_MB_TOKENS {
				if non_zero_reserve(&e.tokens, max(cap(e.tokens) * 2, tok_pos + MAX_MB_TOKENS)) != nil do return .Out_Of_Memory
			}
			at := mb_y * e.d.mb_w + mb_x
			seg := int(e.mbs[at].segment)
			load_mb_source(e, mb_x, mb_y, &src_y4, &src_uv4)
			fill_borders(&e.d, &ws, mb_x, mb_y)
			decide_macroblock(e, &src_y4, &src_uv4, &ws, mb_x, mb_y, seg, &intra_left, nz_left, nz_dc_left, &left_derr, &choice)
			store_mb_recon(&e.d, mb_x, mb_y, &choice.rec_y, &choice.rec_uv)

			first_ac := choice.is_4x4 ? i8(0) : i8(1)
			all_zero := choice.is_4x4 || choice.y_dc_last < 0
			if all_zero {
				for l in choice.y_last do if l >= first_ac {all_zero = false; break}
			}
			if all_zero {
				for l in choice.uv_last do if l >= 0 {all_zero = false; break}
			}
			if all_zero do skip_count += 1

			stat_dst := all_zero ? &tabs.skip_stats : &tabs.stats
			tok_start := u32(tok_pos)
			tok_ptr := raw_data(e.tokens)
			if !choice.is_4x4 {
				dc_ctx := int(e.d.nz_dc_top[mb_x] + nz_dc_left)
				flag := record_block_tokens(tok_ptr, &tok_pos, stat_dst, 1, 0, dc_ctx, &choice.y_dc, choice.y_dc_last)
				e.d.nz_dc_top[mb_x], nz_dc_left = flag, flag
			}
			ac_type := choice.is_4x4 ? 3 : 0
			first_i := int(first_ac)
			top := e.d.nz_top[mb_x]
			left := nz_left
			for y in 0 ..< 4 {
				l := (left >> uint(y)) & 1
				for x in 0 ..< 4 {
					t := (top >> uint(x)) & 1
					b := y * 4 + x
					l = record_block_tokens(tok_ptr, &tok_pos, stat_dst, ac_type, first_i, int(l + t), &choice.y_ac[b], choice.y_last[b])
					top = top &~ (1 << uint(x)) | l << uint(x)
				}
				left = left &~ (1 << uint(y)) | l << uint(y)
			}
			for c in 0 ..< 2 {
				shift := uint(4 + 2 * c)
				for y in 0 ..< 2 {
					l := (left >> (shift + uint(y))) & 1
					for x in 0 ..< 2 {
						t := (top >> (shift + uint(x))) & 1
						b := c * 4 + y * 2 + x
						l = record_block_tokens(tok_ptr, &tok_pos, stat_dst, 2, 0, int(l + t), &choice.uv[b], choice.uv_last[b])
						top = top &~ (1 << (shift + uint(x))) | l << (shift + uint(x))
					}
					left = left &~ (1 << (shift + uint(y))) | l << (shift + uint(y))
				}
			}
			e.d.nz_top[mb_x], nz_left = top, left

			e.mbs[at] = {
				tok_start = tok_start,
				segment   = u8(seg),
				alpha     = e.mbs[at].alpha,
				is_4x4    = choice.is_4x4,
				skip      = all_zero,
				ymode     = choice.ymode,
				uvmode    = choice.uvmode,
				modes     = choice.modes,
			}
		}
	}
	e.mbs[total_mbs].tok_start = u32(tok_pos)
	if non_zero_resize(&e.tokens, tok_pos) != nil do return .Out_Of_Memory

	max_level := 0
	for s in 0 ..< e.num_segments {
		m := &e.dqm[s]
		delta := clamp((m.max_edge * int(simd.extract(m.y2.q, 1))) >> 3, 0, 63)
		if delta > m.fstrength do m.fstrength = delta
		if m.fstrength > max_level do max_level = m.fstrength
	}
	e.d.filter = {level = max_level}

	skip_prob := u8(((total_mbs - skip_count) * 255) / total_mbs)
	use_skip := skip_count > 0 && skip_prob < 250
	e.d.use_skip = use_skip
	e.d.skip_prob = skip_prob
	if !use_skip && skip_count > 0 {
		for ti in 0 ..< 4 do for bi in 0 ..< 8 do for ci in 0 ..< 3 do for pi in 0 ..< 11 {
			tabs.stats[ti][bi][ci][pi][0] += tabs.skip_stats[ti][bi][ci][pi][0]
			tabs.stats[ti][bi][ci][pi][1] += tabs.skip_stats[ti][bi][ci][pi][1]
		}
	}
	finalize_token_probs(tabs)

	emit_partition0(e)
	emit_partition1(e)
	return e.part0.failed || e.part1.failed ? .Out_Of_Memory : .None
}

@(private)
put_signed_opt :: proc(bw: ^common.Bool_Encoder, v, bits: int) {
	if v == 0 {
		common.bool_encode(bw, 0, 128)
		return
	}
	common.bool_encode(bw, 1, 128)
	common.bool_encode_literal(bw, abs(v), bits)
	common.bool_encode(bw, int(v < 0), 128)
}

@(private)
put_i4_mode :: proc(bw: ^common.Bool_Encoder, mode: u8, p: ^[9]u8) #no_bounds_check {
	if mode == B_DC {
		common.bool_encode(bw, 0, p[0])
		return
	}
	common.bool_encode(bw, 1, p[0])
	if mode == B_TM {
		common.bool_encode(bw, 0, p[1])
		return
	}
	common.bool_encode(bw, 1, p[1])
	if mode == B_VE {
		common.bool_encode(bw, 0, p[2])
		return
	}
	common.bool_encode(bw, 1, p[2])
	if mode < B_LD {
		common.bool_encode(bw, 0, p[3])
		if mode == B_HE {
			common.bool_encode(bw, 0, p[4])
		} else {
			common.bool_encode(bw, 1, p[4])
			common.bool_encode(bw, int(mode != B_RD), p[5])
		}
	} else {
		common.bool_encode(bw, 1, p[3])
		if mode == B_LD {
			common.bool_encode(bw, 0, p[6])
		} else {
			common.bool_encode(bw, 1, p[6])
			if mode == B_VL {
				common.bool_encode(bw, 0, p[7])
			} else {
				common.bool_encode(bw, 1, p[7])
				common.bool_encode(bw, int(mode != B_HD), p[8])
			}
		}
	}
}

@(private)
emit_partition0 :: proc(e: ^Lossy_Encoder) #no_bounds_check {
	bw_val := e.part0
	bw := &bw_val
	defer e.part0 = bw_val
	common.bool_encode_start(bw)
	_ = non_zero_reserve(&bw.data, e.d.mb_w * e.d.mb_h + 256)
	common.bool_encode(bw, 0, 128) // colour space
	common.bool_encode(bw, 0, 128) // clamping

	common.bool_encode(bw, int(e.d.segments.enabled), 128)
	if e.d.segments.enabled {
		common.bool_encode(bw, int(e.d.segments.update_map), 128)
		common.bool_encode(bw, 1, 128) // update data
		common.bool_encode(bw, 1, 128) // absolute values
		for s in 0 ..< 4 do put_signed_opt(bw, e.dqm[s].quant, 7)
		for s in 0 ..< 4 do put_signed_opt(bw, e.dqm[s].fstrength, 6)
		if e.d.segments.update_map {
			for p in e.d.segments.tree_probs {
				common.bool_encode(bw, int(p != 255), 128)
				if p != 255 do common.bool_encode_literal(bw, int(p), 8)
			}
		}
	}

	// Normal loop filter, sharpness 0, no mode/ref deltas, 1 token partition.
	common.bool_encode(bw, 0, 128)
	common.bool_encode_literal(bw, e.d.filter.level, 6)
	common.bool_encode_literal(bw, 0, 3)
	common.bool_encode(bw, 0, 128)
	common.bool_encode_literal(bw, 0, 2)

	common.bool_encode_literal(bw, e.base_quant, 7)
	put_signed_opt(bw, 0, 4)
	put_signed_opt(bw, 0, 4)
	put_signed_opt(bw, 0, 4)
	put_signed_opt(bw, e.dq_uv_dc, 4)
	put_signed_opt(bw, e.dq_uv_ac, 4)

	common.bool_encode(bw, 0, 128) // refresh entropy probs

	tabs := e.tables
	for ti in 0 ..< 4 do for bi in 0 ..< 8 do for ci in 0 ..< 3 do for pi in 0 ..< 11 {
		p := tabs.probs[ti][bi][ci][pi]
		upd := p != COEFF_PROBS[ti][bi][ci][pi]
		common.bool_encode(bw, int(upd), COEFF_UPDATE_PROBS[ti][bi][ci][pi])
		if upd do common.bool_encode_literal(bw, int(p), 8)
	}

	common.bool_encode(bw, int(e.d.use_skip), 128)
	if e.d.use_skip do common.bool_encode_literal(bw, int(e.d.skip_prob), 8)

	for &v in e.d.intra_top do v = B_DC
	tp := &e.d.segments.tree_probs
	for mb_y in 0 ..< e.d.mb_h {
		left := [4]u8{B_DC, B_DC, B_DC, B_DC}
		for mb_x in 0 ..< e.d.mb_w {
			mb := &e.mbs[mb_y * e.d.mb_w + mb_x]
			if e.d.segments.update_map {
				s := int(mb.segment)
				if s < 2 {
					common.bool_encode(bw, 0, tp[0])
					common.bool_encode(bw, s & 1, tp[1])
				} else {
					common.bool_encode(bw, 1, tp[0])
					common.bool_encode(bw, s & 1, tp[2])
				}
			}
			if e.d.use_skip do common.bool_encode(bw, int(mb.skip), e.d.skip_prob)

			top := (^[4]u8)(&e.d.intra_top[mb_x * 4])
			common.bool_encode(bw, int(!mb.is_4x4), 145)
			if !mb.is_4x4 {
				ym := int(mb.ymode)
				if ym == TM_PRED || ym == H_PRED {
					common.bool_encode(bw, 1, 156)
					common.bool_encode(bw, int(ym == TM_PRED), 128)
				} else {
					common.bool_encode(bw, 0, 156)
					common.bool_encode(bw, int(ym == V_PRED), 163)
				}
				top^ = mb.ymode
				left = mb.ymode
			} else {
				for y in 0 ..< 4 {
					lm := left[y]
					for x in 0 ..< 4 {
						m := mb.modes[y * 4 + x]
						put_i4_mode(bw, m, &BMODE_PROBS[top[x]][lm])
						top[x] = m
						lm = m
					}
					left[y] = lm
				}
			}

			uvm := int(mb.uvmode)
			if uvm == DC_PRED {
				common.bool_encode(bw, 0, 142)
			} else {
				common.bool_encode(bw, 1, 142)
				if uvm == V_PRED {
					common.bool_encode(bw, 0, 114)
				} else {
					common.bool_encode(bw, 1, 114)
					common.bool_encode(bw, int(uvm == TM_PRED), 183)
				}
			}
		}
	}
	common.bool_encode_finish(bw)
}

@(private)
emit_partition1 :: proc(e: ^Lossy_Encoder) #no_bounds_check {
	bw_val := e.part1
	bw := &bw_val
	defer e.part1 = bw_val
	common.bool_encode_start(bw)
	tokens := e.tokens[:]
	_ = non_zero_reserve(&bw.data, len(tokens) / 8 + 64)
	flat_probs := (^[4 * 8 * 3 * 11]u8)(&e.tables.probs)^
	use_skip := e.d.use_skip
	total_mbs := e.d.mb_w * e.d.mb_h
	if !use_skip {
		for tok in tokens {
			prob := (tok & FIXED_PROB_FLAG) != 0 ? u8(tok) : flat_probs[tok & 0x3fff]
			common.bool_encode(bw, int(tok >> 15), prob)
		}
	} else {
		for i in 0 ..< total_mbs {
			if e.mbs[i].skip do continue
			for tok in tokens[e.mbs[i].tok_start:e.mbs[i + 1].tok_start] {
				prob := (tok & FIXED_PROB_FLAG) != 0 ? u8(tok) : flat_probs[tok & 0x3fff]
				common.bool_encode(bw, int(tok >> 15), prob)
			}
		}
	}
	common.bool_encode_finish(bw)
}

@(private)
write_vp8_chunk :: proc(e: ^Lossy_Encoder, dst: ^[dynamic]u8, width, height: int) -> Error {
	p0 := e.part0.data[:]
	p1 := e.part1.data[:]
	vp8_size := 10 + len(p0) + len(p1)
	tag := u32(len(p0)) << 5 | 0x10 // keyframe, profile 0, shown
	hdr := [18]u8{
		'V', 'P', '8', ' ',
		u8(vp8_size), u8(vp8_size >> 8), u8(vp8_size >> 16), u8(vp8_size >> 24),
		u8(tag), u8(tag >> 8), u8(tag >> 16),
		0x9d, 0x01, 0x2a,
		u8(width), u8(width >> 8),
		u8(height), u8(height >> 8),
	}
	if _, oom := non_zero_append(dst, ..hdr[:]); oom != nil do return .Out_Of_Memory
	if _, oom := non_zero_append(dst, ..p0); oom != nil do return .Out_Of_Memory
	if _, oom := non_zero_append(dst, ..p1); oom != nil do return .Out_Of_Memory
	if vp8_size & 1 == 1 {
		if _, oom := non_zero_append(dst, 0); oom != nil do return .Out_Of_Memory
	}
	return .None
}

// Inverse of webp.odin's unfilter_alpha.
@(private)
filter_alpha :: proc(dst, src: []u8, width, height: int, filter: u8) #no_bounds_check {
	if filter == 0 {
		copy(dst, src)
		return
	}
	dst[0] = src[0]
	for x in 1 ..< width do dst[x] = src[x] - src[x - 1]
	for y in 1 ..< height {
		srow := src[y * width:]
		sprev := src[(y - 1) * width:]
		drow := dst[y * width:]
		switch filter {
		case 1:
			drow[0] = srow[0] - sprev[0]
			for x in 1 ..< width do drow[x] = srow[x] - srow[x - 1]
		case 2:
			for x in 0 ..< width do drow[x] = srow[x] - sprev[x]
		case 3:
			drow[0] = srow[0] - sprev[0]
			for x in 1 ..< width {
				pred := u8(clamp(int(srow[x - 1]) + int(sprev[x]) - int(sprev[x - 1]), 0, 255))
				drow[x] = srow[x] - pred
			}
		}
	}
}

@(private)
filter_entropy_score :: proc(buf: []u8) -> f64 #no_bounds_check {
	counts: [256]int
	for b in buf do counts[b] += 1
	inv := 1.0 / f64(len(buf))
	bits := 0.0
	for c in counts do if c > 0 {
		bits -= f64(c) * math.log2(f64(c) * inv)
	}
	return bits
}

// Appends an ALPH chunk (lossless VP8L with the smaller of unfiltered or the
// lowest-entropy spatial filter, falling back to raw bytes when smaller).
@(private)
write_alph_chunk :: proc(e: ^Lossy_Encoder, dst: ^[dynamic]u8, width, height: int) -> Error {
	total := width * height
	if !e.alpha_enc_ready {
		init_lossless_encoder(&e.alpha_enc, e.allocator) or_return
		e.alpha_enc_ready = true
	}

	best_spatial := u8(1)
	best_entropy := max(f64)
	for f in u8(1) ..= 3 {
		filter_alpha(e.alpha_filtered, e.alpha_plane, width, height, f)
		ent := filter_entropy_score(e.alpha_filtered)
		if ent < best_entropy {
			best_entropy = ent
			best_spatial = f
		}
	}

	ae := &e.alpha_enc
	ae.failed = false
	ae.w.failed = false
	clear(&ae.w.out)
	encode_vp8l_alpha(ae, e.alpha_plane, width, height, true) or_return
	len0 := len(ae.w.out)

	mark := len0
	if total >= 16 {
		filter_alpha(e.alpha_filtered, e.alpha_plane, width, height, best_spatial)
		encode_vp8l_alpha(ae, e.alpha_filtered, width, height, false) or_return
	}
	len_sp := len(ae.w.out) - mark

	best_vp8l_len := len0
	best_vp8l_off := 0
	best_filter := u8(0)
	if total >= 16 && len_sp < len0 {
		best_vp8l_len = len_sp
		best_vp8l_off = mark
		best_filter = best_spatial
	}

	if best_vp8l_len < total {
		chunk_size := 1 + best_vp8l_len
		hdr := [9]u8{
			'A', 'L', 'P', 'H',
			u8(chunk_size), u8(chunk_size >> 8), u8(chunk_size >> 16), u8(chunk_size >> 24),
			1 | (best_filter << 2),
		}
		if _, oom := non_zero_append(dst, ..hdr[:]); oom != nil do return .Out_Of_Memory
		if _, oom := non_zero_append(dst, ..ae.w.out[best_vp8l_off : best_vp8l_off + best_vp8l_len]); oom != nil do return .Out_Of_Memory
		if chunk_size & 1 == 1 {
			if _, oom := non_zero_append(dst, 0); oom != nil do return .Out_Of_Memory
		}
		return .None
	}

	chunk_size := 1 + total
	hdr := [9]u8{
		'A', 'L', 'P', 'H',
		u8(chunk_size), u8(chunk_size >> 8), u8(chunk_size >> 16), u8(chunk_size >> 24),
		0,
	}
	if _, oom := non_zero_append(dst, ..hdr[:]); oom != nil do return .Out_Of_Memory
	if _, oom := non_zero_append(dst, ..e.alpha_plane); oom != nil do return .Out_Of_Memory
	if chunk_size & 1 == 1 {
		if _, oom := non_zero_append(dst, 0); oom != nil do return .Out_Of_Memory
	}
	return .None
}
