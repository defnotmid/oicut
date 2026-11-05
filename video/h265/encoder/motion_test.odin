package h265_encoder

import "core:testing"

// A walk's only limit is the largest vector the encoder writes, so motion far
// from the prediction is found.
@(test)
motion_far_from_the_prediction_is_found :: proc(t: ^testing.T) {
	SIZE :: 128
	ref := alloc_frame(SIZE, SIZE)
	defer free_frame(&ref)
	source := alloc_frame(SIZE, SIZE)
	defer free_frame(&source)
	// A bowl centred on the block: the slope turns all the way round it, so no
	// line of vectors matches as well, and the cost falls to the true vector.
	bowl :: proc(x, y: int) -> Sample {return Sample((x - 64) * (x - 64) + (y - 64) * (y - 64))}
	for y in 0 ..< SIZE {
		for x in 0 ..< SIZE {
			set_at(ref.y, x, y, bowl(x, y))
			set_at(source.y, x, y, bowl(x + 25, y - 18))
		}
	}
	enc := Encoder {
		source       = source,
		motion_range = 4,
		bit_depth    = 8,
	}
	best, cost := motion_estimate(&enc, &ref, 48, 48, 32, 32, 0, 0, Wp_Plane{weight = 1})
	testing.expect_value(t, best, [2]i16{25 * 4, -18 * 4})
	testing.expect_value(t, cost, u32(0))
}

// Motion past that limit stops at it: the cost is the returned vector's, not
// that of an exact match the vector cannot reach.
@(test)
motion_past_the_limit_stops_at_it :: proc(t: ^testing.T) {
	W, H :: 2048, 128
	ref := alloc_frame(W, H)
	defer free_frame(&ref)
	source := alloc_frame(W, H)
	defer free_frame(&source)
	for y in 0 ..< H {
		for x in 0 ..< W {
			set_at(ref.y, x, y, Sample(x * 16 + y))
			set_at(source.y, x, y, Sample((x + 700) * 16 + y))
		}
	}
	enc := Encoder {
		source       = source,
		motion_range = 4,
		bit_depth    = 12,
	}
	best, cost := motion_estimate(&enc, &ref, 64, 48, 32, 32, 0, 0, Wp_Plane{weight = 1})
	testing.expect(t, best[0] >= MAX_MV_SAMPLES * 4 && best[0] <= 2047)
	testing.expect(t, cost > 0)
}
