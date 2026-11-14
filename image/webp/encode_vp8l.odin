package webp

import common "../../common"
import "base:intrinsics"
import "base:runtime"
import "core:math"
import "core:simd"
import "core:slice"

// WebP lossless encoding (VP8L, RFC 9649). An image is either colour-indexed
// (up to 256 colours, packed 1, 2, 4 or 8 to a pixel) or has green subtracted
// from red and blue and each 16x16 block predicted by the mode with the
// lowest entropy. The result is coded as literals, colour-cache hits and
// copies from three places: the pixel before, the pixel above, and the last
// position that began with the same three pixels. One set of five prefix
// codes covers the image, with the colour cache size that codes it smallest.

@(private)
PREDICTOR_BITS :: 4
@(private)
MAX_COPY :: 4096
@(private)
MIN_COPY :: 3
@(private)
MAX_DISTANCE :: 1 << 20 - 120 // the largest distance code is 2^20
@(private)
MAX_CACHE_BITS :: 11
@(private)
GREEN_SYMBOLS :: 256 + 24 + 1 << MAX_CACHE_BITS
@(private)
CACHE_HASH :: 0x1e35a7bd

// LSB-first bits appended to `out`. A dry writer only counts.
@(private)
Bit_Writer :: struct {
	out:    [dynamic]u8,
	value:  u64,
	count:  uint,
	total:  int, // bits put
	dry:    bool,
	failed: bool, // an append ran out of memory
}

@(private)
put_bits :: #force_inline proc(w: ^Bit_Writer, value: u32, n: uint) {
	w.total += int(n)
	if w.dry do return
	w.value |= u64(value) << w.count
	w.count += n
	if w.count >= 32 {
		word := transmute([4]u8)u32le(w.value)
		if _, e := non_zero_append(&w.out, ..word[:]); e != nil do w.failed = true
		w.value >>= 32
		w.count -= 32
	}
}

@(private)
flush_bits :: proc(w: ^Bit_Writer) {
	for w.count > 0 {
		if _, e := non_zero_append(&w.out, u8(w.value)); e != nil do w.failed = true
		w.value >>= 8
		w.count -= min(w.count, 8)
	}
	w.value = 0
}

// The counts of one entropy-coded image's symbols.
@(private)
Histogram :: struct {
	green:            [GREEN_SYMBOLS]u32, // green or index, copy lengths, cache keys
	red, blue, alpha: [256]u32,
	dist:             [40]u32,
}

// The five prefix codes of an image, each alphabet at its offset.
@(private)
CODE_OFFSET := [6]int{0, GREEN_SYMBOLS, GREEN_SYMBOLS + 256, GREEN_SYMBOLS + 512, GREEN_SYMBOLS + 768, GREEN_SYMBOLS + 768 + 40}

@(private)
Prefix_Codes :: struct {
	lengths: [GREEN_SYMBOLS + 768 + 40]u8,
	codes:   [GREEN_SYMBOLS + 768 + 40]u16,
}

// One file's encoder: the output and the scratch every image of it reuses.
@(private)
Lossless_Encoder :: struct {
	w:           Bit_Writer,
	argb:        [dynamic]u32, // the frame's pixels, then green-subtracted
	coded:       [dynamic]u32, // residuals or packed indices
	modes:       [dynamic]u32, // the predictor's block image
	lengths:     [dynamic]u16, // tokens: 0 for a literal or a copy's length
	distances:   [dynamic]u32, // a copy's distance code
	last:        [dynamic]i32, // the last position of each three-pixel hash
	histograms:  []Histogram, // one per colour cache size, 0 to MAX_CACHE_BITS
	codes:       []Prefix_Codes, // the chosen codes, and the trial
	plane:       [128]u8, // distance code - 1 of (dy << 4) | (8 - dx), or 0xff
	log2:        [1 << (2 * PREDICTOR_BITS) + 1]u32, // log2(n) in 1/4096ths
	failed:      bool,
	has_alpha:   bool, // the last image encoded has pixels that are not opaque
}

@(private)
init_lossless_encoder :: proc(e: ^Lossless_Encoder, allocator: Allocator) -> Error {
	e.w.out.allocator = allocator
	e.argb.allocator = allocator
	e.coded.allocator = allocator
	e.modes.allocator = allocator
	e.lengths.allocator = allocator
	e.distances.allocator = allocator
	e.last.allocator = allocator
	err: runtime.Allocator_Error
	e.histograms, err = make([]Histogram, MAX_CACHE_BITS + 1, allocator)
	if err != nil do return .Out_Of_Memory
	e.codes, err = make([]Prefix_Codes, 2, allocator)
	if err != nil do return .Out_Of_Memory
	for &p in e.plane do p = 0xff
	for m, code in DISTANCE_MAP do e.plane[m] = u8(code)
	for &l, n in e.log2 do if n > 0 do l = u32(math.log2(f64(n)) * 4096 + 0.5)
	return .None
}

@(private)
destroy_lossless_encoder :: proc(e: ^Lossless_Encoder) {
	allocator := e.w.out.allocator
	delete(e.w.out)
	delete(e.argb)
	delete(e.coded)
	delete(e.modes)
	delete(e.lengths)
	delete(e.distances)
	delete(e.last)
	delete(e.histograms, allocator)
	delete(e.codes, allocator)
	e^ = {}
}

@(private)
sized :: proc(e: ^Lossless_Encoder, a: ^[dynamic]$T, n: int) -> []T {
	if err := non_zero_resize(a, n); err != nil {
		e.failed = true
		return nil
	}
	return a[:]
}

@(private)
sub_pixels :: #force_inline proc(a, b: u32) -> u32 {
	ag := 0x00ff00ff + (a & 0xff00ff00) - (b & 0xff00ff00)
	rb := 0xff00ff00 + (a & 0x00ff00ff) - (b & 0x00ff00ff)
	return ag & 0xff00ff00 | rb & 0x00ff00ff
}

// The prefix symbol and extra bits of a copy length or distance code.
@(private)
prefix_code :: #force_inline proc(value: int) -> (symbol: int, extra: uint, extra_value: u32) {
	d := value - 1
	if d < 4 do return d, 0, 0
	highest := uint(63 - intrinsics.count_leading_zeros(u64(d)))
	extra = highest - 1
	return int(2 * highest) + (d >> extra) & 1, extra, u32(d & (1 << extra - 1))
}

// The distance code of a distance: 1 to 120 for the nearest neighbours.
@(private)
distance_code :: #force_inline proc(e: ^Lossless_Encoder, width, dist: int) -> int {
	dy := dist / width
	dx := dist - dy * width
	code := u8(0xff)
	if dx <= 8 && dy < 8 {
		code = e.plane[dy << 4 | (8 - dx)]
	} else if dx > width - 8 && dy < 7 {
		code = e.plane[(dy + 1) << 4 | (8 + width - dx)]
	}
	return code != 0xff ? int(code) + 1 : dist + 120
}

// Appends the VP8L bitstream of an image to e.w.out. `rgba` holds rows of
// `width` RGBA8 pixels `stride` bytes apart.
@(private)
encode_vp8l :: proc(e: ^Lossless_Encoder, rgba: []u8, stride, width, height: int, pool: ^common.Work_Pool) -> Error {
	total := width * height
	argb := sized(e, &e.argb, total)
	coded := sized(e, &e.coded, total)
	if e.failed do return .Out_Of_Memory
	opaque: u32 = 0xff000000
	#no_bounds_check for y in 0 ..< height {
		row := rgba[y * stride:]
		for x in 0 ..< width {
			p := u32(row[x * 4 + 3]) << 24 | u32(row[x * 4]) << 16 | u32(row[x * 4 + 1]) << 8 | u32(row[x * 4 + 2])
			argb[y * width + x] = p
			opaque &= p
		}
	}
	w := &e.w
	w.value, w.count = 0, 0
	put_bits(w, 0x2f, 8)
	put_bits(w, u32(width - 1), 14)
	put_bits(w, u32(height - 1), 14)
	e.has_alpha = opaque >> 24 != 0xff
	put_bits(w, e.has_alpha ? 1 : 0, 1)
	put_bits(w, 0, 3)

	palette: [256]u32
	if colours := find_palette(argb, &palette); colours > 0 {
		bits: uint = colours > 16 ? 0 : colours > 4 ? 1 : colours > 2 ? 2 : 3
		put_bits(w, 1, 1)
		put_bits(w, u32(Transform_Kind.Colour_Indexing), 2)
		put_bits(w, u32(colours - 1), 8)
		deltas: [256]u32
		deltas[0] = palette[0]
		for i in 1 ..< colours do deltas[i] = sub_pixels(palette[i], palette[i - 1])
		encode_image(e, deltas[:colours], colours, false)
		packed_width := subsample(width, bits)
		pack_indices(coded[:packed_width * height], argb, width, height, palette[:colours], bits)
		put_bits(w, 0, 1)
		encode_image(e, coded[:packed_width * height], packed_width, true)
	} else {
		put_bits(w, 1, 1)
		put_bits(w, u32(Transform_Kind.Subtract_Green), 2)
		for &p in argb {
			g := (p >> 8) & 0xff
			p = p & 0xff00ff00 | ((p & 0x00ff00ff) + 0x01000100 - (g << 16 | g)) & 0x00ff00ff
		}
		blocks_x, blocks_y := subsample(width, PREDICTOR_BITS), subsample(height, PREDICTOR_BITS)
		modes := sized(e, &e.modes, blocks_x * blocks_y)
		if e.failed do return .Out_Of_Memory
		job := Predict_Job{argb, coded, modes, &e.log2, width, height, blocks_x}
		if pool != nil && pool.run != nil && blocks_y > 1 {
			pool.run(pool, blocks_y, predict_block_rows, &job)
		} else {
			predict_block_rows(&job, 0, blocks_y)
		}
		put_bits(w, 1, 1)
		put_bits(w, u32(Transform_Kind.Predictor), 2)
		put_bits(w, PREDICTOR_BITS - 2, 3)
		encode_image(e, modes, blocks_x, false)
		put_bits(w, 0, 1)
		encode_image(e, coded, width, true)
	}
	flush_bits(w)
	return e.failed || w.failed ? .Out_Of_Memory : .None
}

// The image's colours in ascending order when there are at most 256, else 0.
@(private)
find_palette :: proc(argb: []u32, palette: ^[256]u32) -> int {
	slots: [1024]u32
	used: [1024]bool
	count := 0
	last := ~argb[0]
	for p in argb {
		if p == last do continue
		last = p
		at := (p * 0x9e3779b1) >> 22
		for used[at] && slots[at] != p do at = (at + 1) & 1023
		if used[at] do continue
		if count == 256 do return 0
		used[at], slots[at] = true, p
		palette[count] = p
		count += 1
	}
	slice.sort(palette[:count])
	return count
}

// Palette indices in the green channel, 8 >> bits bits each, the first of a
// packed pixel lowest.
@(private)
pack_indices :: proc(packed, argb: []u32, width, height: int, palette: []u32, bits: uint) #no_bounds_check {
	slots: [1024]u32
	index: [1024]u16
	used: [1024]bool
	for p, i in palette {
		at := (p * 0x9e3779b1) >> 22
		for used[at] do at = (at + 1) & 1023
		used[at], slots[at], index[at] = true, p, u16(i)
	}
	packed_width := subsample(width, bits)
	index_bits := uint(8) >> bits
	per := 1 << bits
	last, last_index := palette[0], u32(0)
	for y in 0 ..< height {
		row := argb[y * width:][:width]
		out := packed[y * packed_width:][:packed_width]
		for &o in out do o = 0xff000000
		for p, x in row {
			if p != last {
				at := (p * 0x9e3779b1) >> 22
				for slots[at] != p do at = (at + 1) & 1023
				last, last_index = p, u32(index[at])
			}
			out[x >> bits] |= last_index << (8 + uint(x & (per - 1)) * index_bits)
		}
	}
}

@(private)
Predict_Job :: struct {
	src:      []u32, // green-subtracted pixels
	residual: []u32,
	modes:    []u32, // per block: 0xff000000 | mode << 8
	log2:     ^[1 << (2 * PREDICTOR_BITS) + 1]u32,
	width:    int,
	height:   int,
	blocks_x: int,
}

// Chooses each block's predictor and writes its rows' residuals. The first
// row predicts from the left and the first column from above, whatever the
// mode, as the decoder does.
@(private)
predict_block_rows :: proc(data: rawptr, first, end: int) #no_bounds_check {
	j := (^Predict_Job)(data)
	width := j.width
	SIZE :: 1 << PREDICTOR_BITS
	trial: [SIZE * SIZE]u32 = ---
	counts: [4][256]u16
	for by in first ..< end {
		y0, y1 := by * SIZE, min((by + 1) * SIZE, j.height)
		for bx in 0 ..< j.blocks_x {
			x0, x1 := max(bx * SIZE, 1), min((bx + 1) * SIZE, width)
			best, best_score := 0, u64(0)
			if x0 < x1 do for mode in 0 ..< 14 {
				n := 0
				for y in max(y0, 1) ..< y1 {
					predict_run(mode, j.src, trial[n:], y * width + x0, y * width + x1, width)
					n += x1 - x0
				}
				// The sum of n log2 n over the residuals' counts: the higher,
				// the lower their entropy.
				for r in trial[:n] {
					counts[0][r >> 24] += 1
					counts[1][(r >> 16) & 0xff] += 1
					counts[2][(r >> 8) & 0xff] += 1
					counts[3][r & 0xff] += 1
				}
				score: u64
				for r in trial[:n] {
					score += u64(j.log2[counts[0][r >> 24]]) + u64(j.log2[counts[1][(r >> 16) & 0xff]])
					score += u64(j.log2[counts[2][(r >> 8) & 0xff]]) + u64(j.log2[counts[3][r & 0xff]])
				}
				for r in trial[:n] {
					counts[0][r >> 24] = 0
					counts[1][(r >> 16) & 0xff] = 0
					counts[2][(r >> 8) & 0xff] = 0
					counts[3][r & 0xff] = 0
				}
				if score > best_score do best, best_score = mode, score
			}
			j.modes[by * j.blocks_x + bx] = 0xff000000 | u32(best) << 8
		}
		for y in y0 ..< y1 {
			row := y * width
			if y == 0 {
				j.residual[0] = sub_pixels(j.src[0], 0xff000000)
				predict_run(1, j.src, j.residual[1:], 1, width, width)
				continue
			}
			j.residual[row] = sub_pixels(j.src[row], j.src[row - width])
			for bx in 0 ..< j.blocks_x {
				x0, x1 := max(bx * SIZE, 1), min((bx + 1) * SIZE, width)
				if x0 < x1 do predict_run(int(j.modes[by * j.blocks_x + bx] >> 8) & 0xf, j.src, j.residual[row + x0:], row + x0, row + x1, width)
			}
		}
	}
}

// The residuals of pixels first ..< end, all of one row, under one predictor
// (RFC 9649 4.1): the inverse of inverse_predictor's cases.
@(private)
predict_run :: proc(mode: int, px: []u32, out: []u32, first, end, width: int) #no_bounds_check {
	switch mode {
	case 0:
		for i in first ..< end do out[i - first] = sub_pixels(px[i], 0xff000000)
	case 1:
		for i in first ..< end do out[i - first] = sub_pixels(px[i], px[i - 1])
	case 2:
		for i in first ..< end do out[i - first] = sub_pixels(px[i], px[i - width])
	case 3:
		for i in first ..< end do out[i - first] = sub_pixels(px[i], px[i - width + 1])
	case 4:
		for i in first ..< end do out[i - first] = sub_pixels(px[i], px[i - width - 1])
	case 5:
		for i in first ..< end do out[i - first] = sub_pixels(px[i], average2(average2(px[i - 1], px[i - width + 1]), px[i - width]))
	case 6:
		for i in first ..< end do out[i - first] = sub_pixels(px[i], average2(px[i - 1], px[i - width - 1]))
	case 7:
		for i in first ..< end do out[i - first] = sub_pixels(px[i], average2(px[i - 1], px[i - width]))
	case 8:
		for i in first ..< end do out[i - first] = sub_pixels(px[i], average2(px[i - width - 1], px[i - width]))
	case 9:
		for i in first ..< end do out[i - first] = sub_pixels(px[i], average2(px[i - width], px[i - width + 1]))
	case 10:
		for i in first ..< end {
			out[i - first] = sub_pixels(px[i], average2(average2(px[i - 1], px[i - width - 1]), average2(px[i - width], px[i - width + 1])))
		}
	case 11:
		for i in first ..< end do out[i - first] = sub_pixels(px[i], select(px[i - width], px[i - 1], px[i - width - 1]))
	case 12:
		for i in first ..< end {
			out[i - first] = sub_pixels(px[i], pixel(simd.clamp(lanes(px[i - 1]) + lanes(px[i - width]) - lanes(px[i - width - 1]), 0, 255)))
		}
	case 13:
		for i in first ..< end do out[i - first] = sub_pixels(px[i], clamped_add_subtract_half(average2(px[i - 1], px[i - width]), px[i - width - 1]))
	}
}

// Writes one entropy-coded image: its colour cache size, its five prefix
// codes and its pixels. `main` images say they use one group of codes.
@(private)
encode_image :: proc(e: ^Lossless_Encoder, px: []u32, width: int, main: bool) {
	tokens := tokenize(e, px, width)
	if e.failed do return
	count_symbols(e, px, tokens)
	// The cache size whose codes and symbols take the fewest bits. Copies'
	// extra bits are the same for all.
	w := &e.w
	best, best_bits := 0, max(int)
	chosen, trial := &e.codes[0], &e.codes[1]
	for cache_bits in 0 ..= MAX_CACHE_BITS {
		dry := Bit_Writer{dry = true}
		bits := build_codes(&dry, &e.histograms[cache_bits], trial, cache_bits)
		if bits + dry.total < best_bits {
			best, best_bits = cache_bits, bits + dry.total
			chosen, trial = trial, chosen
		}
	}
	put_bits(w, best > 0 ? 1 : 0, 1)
	if best > 0 do put_bits(w, u32(best), 4)
	if main do put_bits(w, 0, 1)
	build_codes(w, &e.histograms[best], chosen, best)

	lengths, codes := &chosen.lengths, &chosen.codes
	cache: [1 << MAX_CACHE_BITS]u32
	shift := uint(32 - best)
	pos := 0
	#no_bounds_check for t in 0 ..< tokens {
		length := int(e.lengths[t])
		if length == 0 {
			p := px[pos]
			pos += 1
			if best > 0 {
				key := (CACHE_HASH * p) >> shift
				if cache[key] == p {
					put_bits(w, u32(codes[256 + 24 + key]), uint(lengths[256 + 24 + key]))
					continue
				}
				cache[key] = p
			}
			g, r, b, a := int(p >> 8) & 0xff, CODE_OFFSET[RED] + int(p >> 16) & 0xff, CODE_OFFSET[BLUE] + int(p & 0xff), CODE_OFFSET[ALPHA] + int(p >> 24)
			put_bits(w, u32(codes[g]) | u32(codes[r]) << lengths[g], uint(lengths[g]) + uint(lengths[r]))
			put_bits(w, u32(codes[b]) | u32(codes[a]) << lengths[b], uint(lengths[b]) + uint(lengths[a]))
			continue
		}
		symbol, extra, extra_value := prefix_code(length)
		symbol += 256
		put_bits(w, u32(codes[symbol]) | extra_value << lengths[symbol], uint(lengths[symbol]) + extra)
		symbol, extra, extra_value = prefix_code(int(e.distances[t]))
		symbol += CODE_OFFSET[DIST]
		put_bits(w, u32(codes[symbol]), uint(lengths[symbol]))
		put_bits(w, extra_value, extra)
		if best > 0 do for p in px[pos:][:length] do cache[(CACHE_HASH * p) >> shift] = p
		pos += length
	}
}

// Splits the pixels into literals and copies: e.lengths and e.distances.
// A copy comes from the pixel before, the pixel above or the last position
// that began with the same three pixels, whichever runs longest.
@(private)
tokenize :: proc(e: ^Lossless_Encoder, px: []u32, width: int) -> (tokens: int) #no_bounds_check {
	total := len(px)
	lengths := sized(e, &e.lengths, total)
	distances := sized(e, &e.distances, total)
	hash_bits := clamp(uint(64 - intrinsics.count_leading_zeros(u64(total))), 8, 18)
	last := sized(e, &e.last, 1 << hash_bits)
	if e.failed do return 0
	for &l in last do l = -1
	run :: #force_inline proc(px: []u32, at, dist, limit: int) -> int #no_bounds_check {
		n := 0
		for n < limit && px[at + n] == px[at + n - dist] do n += 1
		return n
	}
	hash :: #force_inline proc(px: []u32, at: int, bits: uint) -> u32 #no_bounds_check {
		return (px[at] * 0x9e3779b1 + px[at + 1] * 0x85ebca6b + px[at + 2] * 0xc2b2ae35) >> (32 - bits)
	}
	for at := 0; at < total; {
		limit := min(MAX_COPY, total - at)
		length, dist := 0, 0
		if at >= 1 do length, dist = run(px, at, 1, limit), 1
		if at >= width {
			if n := run(px, at, width, limit); n > length do length, dist = n, width
		}
		if at + 2 < total {
			if from := int(last[hash(px, at, hash_bits)]); from >= 0 && at - from <= MAX_DISTANCE {
				if n := run(px, at, at - from, limit); n > length do length, dist = n, at - from
			}
		}
		if length < MIN_COPY {
			lengths[tokens] = 0
			length = 1
		} else {
			lengths[tokens] = u16(length)
			distances[tokens] = u32(distance_code(e, width, dist))
		}
		tokens += 1
		for i in at ..< min(at + length, total - 2) do last[hash(px, i, hash_bits)] = i32(i)
		at += length
	}
	return
}

// Fills e.histograms: the tokens' symbols under every colour cache size.
@(private)
count_symbols :: proc(e: ^Lossless_Encoder, px: []u32, tokens: int) #no_bounds_check {
	histograms := e.histograms
	for &h in histograms do h = {}
	caches: [2 << MAX_CACHE_BITS]u32 // the cache of `bits` bits starts at 1 << bits
	pos := 0
	for t in 0 ..< tokens {
		length := int(e.lengths[t])
		if length == 0 {
			p := px[pos]
			pos += 1
			g, r, b, a := (p >> 8) & 0xff, (p >> 16) & 0xff, p & 0xff, p >> 24
			h := &histograms[0]
			h.green[g] += 1
			h.red[r] += 1
			h.blue[b] += 1
			h.alpha[a] += 1
			for bits in 1 ..= uint(MAX_CACHE_BITS) {
				h = &histograms[bits]
				key := (CACHE_HASH * p) >> (32 - bits)
				slot := &caches[1 << bits + key]
				if slot^ == p {
					h.green[256 + 24 + key] += 1
				} else {
					slot^ = p
					h.green[g] += 1
					h.red[r] += 1
					h.blue[b] += 1
					h.alpha[a] += 1
				}
			}
			continue
		}
		length_symbol, _, _ := prefix_code(length)
		dist_symbol, _, _ := prefix_code(int(e.distances[t]))
		for &h in histograms {
			h.green[256 + length_symbol] += 1
			h.dist[dist_symbol] += 1
		}
		for p in px[pos:][:length] {
			for bits in 1 ..= uint(MAX_CACHE_BITS) do caches[1 << bits + (CACHE_HASH * p) >> (32 - bits)] = p
		}
		pos += length
	}
}

// Builds the five codes of a histogram and writes their lengths. Returns the
// bits the histogram's symbols take under them.
@(private)
build_codes :: proc(w: ^Bit_Writer, h: ^Histogram, c: ^Prefix_Codes, cache_bits: int) -> (bits: int) {
	counts := [5][]u32{h.green[:256 + 24 + (cache_bits > 0 ? 1 << uint(cache_bits) : 0)], h.red[:], h.blue[:], h.alpha[:], h.dist[:]}
	for count, kind in counts {
		lengths := c.lengths[CODE_OFFSET[kind]:][:len(count)]
		used := huffman_lengths(count, lengths, 15)
		write_code_lengths(w, lengths, used)
		// A code of one symbol takes no bits.
		if used == 1 do for &l in lengths do l = 0
		huffman_codes(lengths, c.codes[CODE_OFFSET[kind]:][:len(count)])
		for n, s in count do bits += int(n) * int(lengths[s])
	}
	return
}

// A prefix code's lengths (RFC 9649 3.7.2.1): one or two symbols below 256
// directly, else the lengths under a code of their own, with runs.
@(private)
write_code_lengths :: proc(w: ^Bit_Writer, lengths: []u8, used: int) {
	if used <= 2 {
		symbols: [2]int
		n := 0
		for l, s in lengths do if l != 0 {
			symbols[n] = s
			n += 1
		}
		if symbols[0] < 256 && symbols[1] < 256 {
			put_bits(w, 1, 1)
			put_bits(w, u32(max(used, 1) - 1), 1)
			if symbols[0] <= 1 {
				put_bits(w, 0, 1)
				put_bits(w, u32(symbols[0]), 1)
			} else {
				put_bits(w, 1, 1)
				put_bits(w, u32(symbols[0]), 8)
			}
			if used == 2 do put_bits(w, u32(symbols[1]), 8)
			return
		}
	}
	put_bits(w, 0, 1)
	// Runs: 16 repeats the last nonzero length 3 to 6 times (it starts as 8),
	// 17 is 3 to 10 zeros and 18 is 11 to 138.
	Run :: struct {
		code, extra: u8,
	}
	runs: [GREEN_SYMBOLS]Run = ---
	n := 0
	count: [19]u32
	emit :: #force_inline proc(runs: []Run, n: ^int, count: ^[19]u32, code, extra: int) {
		runs[n^] = {u8(code), u8(extra)}
		n^ += 1
		count[code] += 1
	}
	prev: u8 = 8
	for at := 0; at < len(lengths); {
		value := lengths[at]
		length := 1
		for at + length < len(lengths) && lengths[at + length] == value do length += 1
		at += length
		if value == 0 {
			for ; length >= 11; length -= min(length, 138) do emit(runs[:], &n, &count, 18, min(length, 138) - 11)
			for ; length >= 3; length -= min(length, 10) do emit(runs[:], &n, &count, 17, min(length, 10) - 3)
		} else {
			if value != prev {
				emit(runs[:], &n, &count, int(value), 0)
				length -= 1
				prev = value
			}
			for ; length >= 3; length -= min(length, 6) do emit(runs[:], &n, &count, 16, min(length, 6) - 3)
		}
		for _ in 0 ..< length do emit(runs[:], &n, &count, int(value), 0)
	}
	code_lengths: [19]u8
	codes: [19]u16
	// A lone code is sent with length 1 and written in no bits.
	lone := huffman_lengths(count[:], code_lengths[:], 7) == 1
	write_code_length_code(w, code_lengths)
	if lone do code_lengths = {}
	huffman_codes(code_lengths[:], codes[:])
	put_bits(w, 0, 1) // every length follows
	for r in runs[:n] {
		put_bits(w, u32(codes[r.code]), uint(code_lengths[r.code]))
		switch r.code {
		case 16:
			put_bits(w, u32(r.extra), 2)
		case 17:
			put_bits(w, u32(r.extra), 3)
		case 18:
			put_bits(w, u32(r.extra), 7)
		}
	}
}

@(private)
write_code_length_code :: proc(w: ^Bit_Writer, code_lengths: [19]u8) {
	@(static, rodata)
	ORDER := [19]u8{17, 18, 0, 1, 2, 3, 4, 5, 16, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}
	sent := 19
	for sent > 4 && code_lengths[ORDER[sent - 1]] == 0 do sent -= 1
	put_bits(w, u32(sent - 4), 4)
	for i in 0 ..< sent do put_bits(w, u32(code_lengths[ORDER[i]]), 3)
}

// Code lengths of at most `limit` bits for the symbols with a count:
// minimum-redundancy lengths (Moffat and Katajainen's in-place method), then
// the longest codes folded under the limit. A lone symbol gets length 1.
// Returns the number of symbols with a count.
@(private)
huffman_lengths :: proc(counts: []u32, lengths: []u8, limit: int) -> (n: int) {
	Symbol :: struct {
		key:    u32,
		symbol: u16,
	}
	symbols: [GREEN_SYMBOLS]Symbol = ---
	for c, s in counts {
		lengths[s] = 0
		if c > 0 {
			symbols[n] = {c, u16(s)}
			n += 1
		}
	}
	a := symbols[:n]
	if n == 0 do return
	if n == 1 {
		lengths[a[0].symbol] = 1
		return
	}
	slice.sort_by(a, proc(x, y: Symbol) -> bool {return x.key < y.key || (x.key == y.key && x.symbol < y.symbol)})
	// a[i].key becomes the length of the i-th rarest symbol.
	a[0].key += a[1].key
	root, leaf := 0, 2
	for next in 1 ..< n - 1 {
		if leaf >= n || a[root].key < a[leaf].key {
			a[next].key = a[root].key
			a[root].key = u32(next)
			root += 1
		} else {
			a[next].key = a[leaf].key
			leaf += 1
		}
		if leaf >= n || (root < next && a[root].key < a[leaf].key) {
			a[next].key += a[root].key
			a[root].key = u32(next)
			root += 1
		} else {
			a[next].key += a[leaf].key
			leaf += 1
		}
	}
	a[n - 2].key = 0
	for next := n - 3; next >= 0; next -= 1 do a[next].key = a[a[next].key].key + 1
	available, used, depth := 1, 0, u32(0)
	internal, next := n - 2, n - 1
	for available > 0 {
		for internal >= 0 && a[internal].key == depth {
			used += 1
			internal -= 1
		}
		for available > used {
			a[next].key = depth
			next -= 1
			available -= 1
		}
		available = 2 * used
		depth += 1
		used = 0
	}
	count: [64]int
	for s in a do count[min(int(s.key), 63)] += 1
	for l in limit + 1 ..< 64 do count[limit] += count[l]
	// The Kraft sum in units of 2^-limit: lengthen codes until it fits.
	total := 0
	for l in 1 ..= limit do total += count[l] << uint(limit - l)
	for total != 1 << uint(limit) {
		count[limit] -= 1
		for l := limit - 1; l > 0; l -= 1 {
			if count[l] != 0 {
				count[l] -= 1
				count[l + 1] += 2
				break
			}
		}
		total -= 1
	}
	// The shortest codes go to the most frequent symbols.
	at := n
	for l in 1 ..= limit {
		for _ in 0 ..< count[l] {
			at -= 1
			lengths[a[at].symbol] = u8(l)
		}
	}
	return
}

// Canonical codes for `lengths`, bit-reversed: the stream is read from the
// least significant bit.
@(private)
huffman_codes :: proc(lengths: []u8, codes: []u16) {
	count: [16]u16
	for l in lengths do count[l] += 1
	count[0] = 0
	next: [16]u16
	code: u16
	for l in 1 ..< 16 {
		code = (code + count[l - 1]) << 1
		next[l] = code
	}
	for l, s in lengths {
		if l == 0 {
			codes[s] = 0 // written in no bits, also beside another code
			continue
		}
		c := next[l]
		next[l] += 1
		r: u16
		for _ in 0 ..< l {
			r = r << 1 | c & 1
			c >>= 1
		}
		codes[s] = r
	}
}
