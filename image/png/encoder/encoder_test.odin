package png_encoder

import "core:bytes"
import "core:image"
import core_png "core:image/png"
import "core:slice"
import "core:testing"

@(test)
encode_round_trips_through_core_image :: proc(t: ^testing.T) {
	for size in ([][2]int{{1, 1}, {3, 2}, {67, 41}, {640, 9}}) {
		w, h := size[0], size[1]
		pixels := make([]u8, w * h * 4)
		defer delete(pixels)
		s := u32(w * 31 + h)
		for &v, i in pixels {s = s * 1664525 + 1013904223; v = i % 7 < 3 ? u8(s >> 24) : u8(i / 9)}
		data, err := encode(pixels, w, h)
		defer delete(data)
		testing.expect_value(t, err, Error.None)
		img, derr := core_png.load_from_bytes(data, {.alpha_add_if_missing})
		testing.expect_value(t, derr, nil)
		if derr != nil do continue
		defer image.destroy(img)
		testing.expect_value(t, img.width, w)
		testing.expect_value(t, img.height, h)
		testing.expect_value(t, img.channels, 4)
		testing.expect(t, bytes.equal(img.pixels[:], pixels), "decoded pixels differ")
	}
	_, err := encode(nil, 1, 1)
	testing.expect_value(t, err, Error.Invalid_Argument)
}

// Dimensions whose byte count wraps a 64-bit int are refused, not taken as a
// fit for an empty buffer.
@(test)
huge_dimensions_are_refused :: proc(t: ^testing.T) {
	testing.expect(t, !valid(nil, 1 << 31, 1 << 31))
	testing.expect(t, !valid(nil, 1 << 31 - 1, 1 << 31 - 1))
	pixels: [16]u8
	testing.expect(t, valid(pixels[:], 2, 2))
	testing.expect(t, !valid(pixels[:], 2, 3))
}

// Premultiplied pixels write the same file as the same pixels un-premultiplied
// first by the formula of unpremultiply_row, with every alpha from 0 to 255,
// and the caller's pixels are left as they were.
@(test)
premultiplied_matches_straight :: proc(t: ^testing.T) {
	for size in ([][2]int{{1, 1}, {3, 2}, {67, 41}, {256, 3}}) {
		w, h := size[0], size[1]
		pre := make([]u8, w * h * 4)
		defer delete(pre)
		s := u32(w * 17 + h)
		for i := 0; i < len(pre); i += 4 {
			s = s * 1664525 + 1013904223
			a := (i / 4) % 256
			pre[i + 3] = u8(a)
			for k in 0 ..< 3 {
				s = s * 1664525 + 1013904223
				pre[i + k] = u8(u32(s >> 24) * u32(a) / 255 + (k == 2 && a < 255 ? 1 : 0)) // the +1 reaches past 255 after dividing
			}
		}
		straight := make([]u8, len(pre))
		defer delete(straight)
		for i := 0; i < len(pre); i += 4 {
			a := u32(pre[i + 3])
			for k in 0 ..< 3 do straight[i + k] = a == 0 || a == 255 ? pre[i + k] : u8(min((u32(pre[i + k]) * 255 + a / 2) / a, 255))
			straight[i + 3] = u8(a)
		}
		original := slice.clone(pre)
		defer delete(original)
		want, e1 := encode(straight, w, h)
		defer delete(want)
		got, e2 := encode(pre, w, h, premultiplied = true)
		defer delete(got)
		testing.expect_value(t, e1, Error.None)
		testing.expect_value(t, e2, Error.None)
		testing.expectf(t, bytes.equal(want, got), "%dx%d: premultiplied file differs", w, h)
		testing.expect(t, bytes.equal(pre, original), "caller's pixels changed")
	}
}
