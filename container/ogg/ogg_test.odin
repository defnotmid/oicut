package ogg

import "core:io"
import "core:testing"

@(private = "file")
bytes_stream :: proc(b: ^[]u8) -> io.Stream {
	proc_ :: proc(data: rawptr, mode: io.Stream_Mode, p: []byte, offset: i64, whence: io.Seek_From) -> (n: i64, err: io.Error) {
		b := (^[]u8)(data)^
		#partial switch mode {
		case .Read_At:
			if offset >= i64(len(b)) do return 0, .EOF
			return i64(copy(p, b[offset:])), .None
		case .Size:
			return i64(len(b)), .None
		case .Query:
			return io.query_utility({.Read_At, .Size, .Query})
		}
		return 0, .Unsupported
	}
	return {proc_, b}
}

// One page: its flags and serial, its lacing values, and as many body bytes
// as they say, all `fill`.
@(private = "file")
page :: proc(out: ^[dynamic]u8, flags: u8, serial: u8, lacing: []u8, fill: u8) {
	append(out, "OggS")
	append(out, 0, flags)
	for _ in 0 ..< 8 do append(out, 0) // granule
	append(out, serial, 0, 0, 0)
	for _ in 0 ..< 8 do append(out, 0) // sequence, checksum
	append(out, u8(len(lacing)))
	append(out, ..lacing)
	total := 0
	for v in lacing do total += int(v)
	for _ in 0 ..< total do append(out, fill)
}

// A page continuing a packet whose start was lost: the fragment is dropped
// wherever it falls, and the page's next packet is kept.
@(test)
an_orphan_continuation_is_dropped :: proc(t: ^testing.T) {
	doc: [dynamic]u8
	defer delete(doc)
	page(&doc, 2, 1, {3}, 'a') // a complete packet
	page(&doc, 1, 1, {5}, 'b') // continues a packet nothing started
	page(&doc, 0, 1, {4}, 'c')
	data := doc[:]
	r: Reader
	testing.expect_value(t, open_stream(&r, bytes_stream(&data)), Error.None)
	defer close(&r)
	testing.expect_value(t, len(r.packets), 2)
	first, _ := read_packet(&r, 0)
	testing.expect_value(t, string(first), "aaa")
	second, _ := read_packet(&r, 1)
	testing.expect_value(t, string(second), "cccc")
}
