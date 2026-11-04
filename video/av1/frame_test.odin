package av1

import "core:testing"

// A redundant frame header only repeats one already parsed: alone it starts no
// frame, even before any sequence header.
@(test)
redundant_frame_headers_are_skipped :: proc(t: ^testing.T) {
	p: Parser
	frames: [dynamic]Frame
	tiles: [dynamic]Tile
	defer {delete(frames); delete(tiles)}
	unit := [?]u8{7 << 3 | 2, 1, 0x80} // type 7, has_size, one byte
	testing.expect_value(t, parse_temporal_unit(&p, unit[:], &frames, &tiles), Error.None)
	testing.expect_value(t, len(frames), 0)
}
