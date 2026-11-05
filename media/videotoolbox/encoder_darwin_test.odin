#+build darwin
package media_videotoolbox

import "core:testing"

@(private = "file")
Picture :: struct {
	y, uv: []u8,
}

// A picture of its own: its number moves the pattern.
@(private = "file")
draw :: proc(p: Picture, width, height, number: int) {
	for row in 0 ..< height {
		for col in 0 ..< width {
			p.y[row * width + col] = u8((col + row + 3 * number) % 200 + 16)
		}
	}
	for &v in p.uv {
		v = 128
	}
}

@(test)
test_pictures_in_flight :: proc(t: ^testing.T) {
	WIDTH, HEIGHT :: 320, 240
	FRAMES :: 20
	INTERVAL :: 8
	e: Encoder
	// A machine with no encoder to open (a CI virtual machine may be one).
	if encoder_create(&e, {codec = .H264, width = WIDTH, height = HEIGHT, qp = 30, keyframe_interval = INTERVAL}) != .None {
		return
	}
	defer encoder_destroy(&e)
	p := Picture{make([]u8, WIDTH * HEIGHT), make([]u8, WIDTH * HEIGHT / 2)}
	defer delete(p.y)
	defer delete(p.uv)
	published := 0
	check :: proc(t: ^testing.T, packet: []u8, keyframe: bool, number: int) {
		testing.expect(t, len(packet) > 4)
		// Forced ones come where they were asked for, so packets are in input
		// order; an encoder may add its own.
		testing.expectf(t, keyframe || number % INTERVAL != 0, "packet %d: no keyframe", number)
	}
	for number in 0 ..< FRAMES {
		draw(p, WIDTH, HEIGHT, number)
		packet, keyframe, err := encoder_encode(&e, p.y, WIDTH, {planes = {p.uv, nil}, strides = {WIDTH, 0}, interleaved = true})
		testing.expect_value(t, err, Error.None)
		if number < ENCODE_DEPTH - 1 {
			testing.expect(t, packet == nil)
			continue
		}
		testing.expect(t, packet != nil)
		check(t, packet, keyframe, published)
		published += 1
	}
	for {
		packet, keyframe, more, err := encoder_drain(&e)
		testing.expect_value(t, err, Error.None)
		if !more do break
		check(t, packet, keyframe, published)
		published += 1
	}
	testing.expect_value(t, published, FRAMES)
}

@(test)
test_destroy_with_pictures_in_flight :: proc(t: ^testing.T) {
	WIDTH, HEIGHT :: 320, 240
	e: Encoder
	if encoder_create(&e, {codec = .H264, width = WIDTH, height = HEIGHT, qp = 30, keyframe_interval = 8}) != .None {
		return
	}
	p := Picture{make([]u8, WIDTH * HEIGHT), make([]u8, WIDTH * HEIGHT / 2)}
	defer delete(p.y)
	defer delete(p.uv)
	// However many a depth keeps; at depth 1 none are left in flight.
	for number := 0; number < ENCODE_DEPTH - 1; number += 1 {
		draw(p, WIDTH, HEIGHT, number)
		_, _, err := encoder_encode(&e, p.y, WIDTH, {planes = {p.uv, nil}, strides = {WIDTH, 0}, interleaved = true})
		testing.expect_value(t, err, Error.None)
	}
	encoder_destroy(&e)
}
