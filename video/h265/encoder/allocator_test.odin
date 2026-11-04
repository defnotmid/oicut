package h265_encoder

import "core:mem"
import "core:testing"

// destroy_encoder frees through the allocator init_encoder was given, whatever
// the caller's context holds by then. An encoder never initialised is a no-op.
@(test)
destroy_frees_through_the_init_allocator :: proc(t: ^testing.T) {
	never: Encoder
	destroy_encoder(&never)
	configs := [3]Config {
		{width = 128, height = 128, qp = 26, level_idc = 120, forced_mode = -1},
		{width = 128, height = 128, qp = 26, level_idc = 120, forced_mode = -1, tiles_cols = 2, tiles_rows = 2},
		{width = 128, height = 128, qp = 26, level_idc = 120, forced_mode = -1, hierarchical = true, gop_size = 4},
	}
	for config in configs {
		tracker: mem.Tracking_Allocator
		mem.tracking_allocator_init(&tracker, context.allocator)
		defer mem.tracking_allocator_destroy(&tracker)
		tracker.bad_free_callback = mem.tracking_allocator_bad_free_callback_add_to_array
		e: Encoder
		init_encoder(&e, config, mem.tracking_allocator(&tracker))
		testing.expect(t, len(tracker.allocation_map) > 0)
		destroy_encoder(&e)
		testing.expect_value(t, len(tracker.allocation_map), 0)
		testing.expect_value(t, len(tracker.bad_free_array), 0)
	}
}
