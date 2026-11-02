package av1_decoder

import av1 "../"
import "core:encoding/endian"
import "core:fmt"
import "core:os"
import "core:testing"

@(test)
malformed_packets_fail_cleanly :: proc(t: ^testing.T) {
	d: Decoder
	init_decoder(&d)
	defer destroy_decoder(&d)

	_, _, err := decode_packet(&d, nil)
	testing.expect_value(t, err, Error.Invalid_Data)

	// Invalid OBU forbidden bit (bit 7 = 1).
	_, _, err = decode_packet(&d, {0x80, 0x00})
	testing.expect_value(t, err, Error.Invalid_Data)

	// Frame OBU before sequence header.
	_, _, err = decode_packet(&d, {0x32, 0x02, 0x00, 0x00})
	testing.expect(t, err != .None)

	// Truncated sequence header OBU.
	seq := [?]u8{0x0A, 0x0A, 0x00, 0x00, 0x00}
	_, _, err = decode_packet(&d, seq[:])
	testing.expect(t, err != .None)
}

@(test)
decode_libaom_ivf_bit_exact :: proc(t: ^testing.T) {
	ivf_data, ivf_err := os.read_entire_file(".work/tmp/test_av1.ivf", context.allocator)
	if ivf_err != nil {
		return
	}
	defer delete(ivf_data)

	ref_yuv, ref_err := os.read_entire_file(".work/tmp/test_av1_ref.yuv", context.allocator)
	if ref_err != nil {
		return
	}
	defer delete(ref_yuv)

	d: Decoder
	init_decoder(&d)
	defer destroy_decoder(&d)

	// Skip 32-byte IVF header.
	at := 32
	frame_idx := 0
	for at + 12 <= len(ivf_data) {
		pkt_size := int(endian.unchecked_get_u32le(ivf_data[at:]))
		at += 12
		if at + pkt_size > len(ivf_data) do break
		pkt := ivf_data[at:][:pkt_size]
		at += pkt_size

		pic, shown, err := decode_packet(&d, pkt)
		if err != .None {
			testing.expect(t, false, fmt.tprintf("10b frame_idx=%d err=%v", frame_idx, err))
			return
		}
		if !shown do continue

		w := pic.width
		h := pic.height
		uv_w := w >> pic.subsampling_x
		uv_h := h >> pic.subsampling_y
		frame_bytes := w * h + 2 * uv_w * uv_h
		ref_base := frame_idx * frame_bytes
		testing.expect(t, ref_base + frame_bytes <= len(ref_yuv))

		// Compare Y, U, V pixel-by-pixel against libdav1d.
		for y in 0 ..< h {
			for x in 0 ..< w {
				got := pic.y[y * pic.y_stride + x]
				exp := ref_yuv[ref_base + y * w + x]
				if got != exp {
					testing.expect(t, false, fmt.tprintf("frame=%d Y(%d,%d) got=%d exp=%d", frame_idx, x, y, got, exp))
					return
				}
			}
		}
		u_base := ref_base + w * h
		v_base := u_base + uv_w * uv_h
		for y in 0 ..< uv_h {
			for x in 0 ..< uv_w {
				got_u := pic.u[y * pic.uv_stride + x]
				exp_u := ref_yuv[u_base + y * uv_w + x]
				if got_u != exp_u {
					testing.expect(t, false, fmt.tprintf("frame=%d U(%d,%d) got=%d exp=%d", frame_idx, x, y, got_u, exp_u))
					return
				}
				got_v := pic.v[y * pic.uv_stride + x]
				exp_v := ref_yuv[v_base + y * uv_w + x]
				if got_v != exp_v {
					testing.expect(t, false, fmt.tprintf("frame=%d V(%d,%d) got=%d exp=%d", frame_idx, x, y, got_v, exp_v))
					return
				}
			}
		}
		frame_idx += 1
	}
	testing.expect(t, frame_idx > 0)
}

@(test)
decode_libaom_ivf_10bit_multitile :: proc(t: ^testing.T) {
	ivf_data, ivf_err := os.read_entire_file(".work/tmp/test_av1_10b.ivf", context.allocator)
	if ivf_err != nil {
		return
	}
	defer delete(ivf_data)

	ref_yuv, ref_err := os.read_entire_file(".work/tmp/test_av1_10b_ref.yuv", context.allocator)
	if ref_err != nil {
		return
	}
	defer delete(ref_yuv)

	d: Decoder
	init_decoder(&d)
	defer destroy_decoder(&d)

	at := 32
	frame_idx := 0
	for at + 12 <= len(ivf_data) {
		pkt_size := int(endian.unchecked_get_u32le(ivf_data[at:]))
		at += 12
		if at + pkt_size > len(ivf_data) do break
		pkt := ivf_data[at:][:pkt_size]
		at += pkt_size

		pic, shown, err := decode_packet(&d, pkt)
		if err != .None {
			testing.expect(t, false, fmt.tprintf("10b frame_idx=%d err=%v", frame_idx, err))
			return
		}
		if !shown do continue
		testing.expect_value(t, pic.bit_depth, u8(10))

		w := pic.width
		h := pic.height
		uv_w := w >> pic.subsampling_x
		uv_h := h >> pic.subsampling_y
		frame_samples := w * h + 2 * uv_w * uv_h
		ref_base := frame_idx * frame_samples * 2
		testing.expect(t, ref_base + frame_samples * 2 <= len(ref_yuv))

		for y in 0 ..< h {
			for x in 0 ..< w {
				got := pic.y16[y * pic.y_stride + x]
				exp := endian.unchecked_get_u16le(ref_yuv[ref_base + (y * w + x) * 2:])
				if got != exp {
					testing.expect(t, false, fmt.tprintf("10b frame=%d Y(%d,%d) got=%d exp=%d", frame_idx, x, y, got, exp))
					return
				}
			}
		}
		u_base := ref_base + w * h * 2
		v_base := u_base + uv_w * uv_h * 2
		for y in 0 ..< uv_h {
			for x in 0 ..< uv_w {
				got_u := pic.u16[y * pic.uv_stride + x]
				exp_u := endian.unchecked_get_u16le(ref_yuv[u_base + (y * uv_w + x) * 2:])
				if got_u != exp_u {
					testing.expect(t, false, fmt.tprintf("10b frame=%d U(%d,%d) got=%d exp=%d", frame_idx, x, y, got_u, exp_u))
					return
				}
				got_v := pic.v16[y * pic.uv_stride + x]
				exp_v := endian.unchecked_get_u16le(ref_yuv[v_base + (y * uv_w + x) * 2:])
				if got_v != exp_v {
					testing.expect(t, false, fmt.tprintf("10b frame=%d V(%d,%d) got=%d exp=%d", frame_idx, x, y, got_v, exp_v))
					return
				}
			}
		}
		frame_idx += 1
	}
	testing.expect(t, frame_idx > 0)
}

@(test)
decode_av1fx_obus :: proc(t: ^testing.T) {
	files := [?][2]string{
		{".work/av1fx/svt8.obu", ".work/av1gate/svt8.ref.yuv"},
		{".work/av1fx/svt10.obu", ".work/av1gate/svt10.ref.yuv"},
		{".work/av1fx/aom_tiles.obu", ".work/av1gate/aom_tiles.ref.yuv"},
		{".work/av1fx/aom_resilient.obu", ".work/av1gate/aom_resilient.ref.yuv"},
		{".work/av1fx/aom_scaled_down.obu", ".work/av1gate/aom_scaled_down.ref.yuv"},
		{".work/av1fx/aom_scaled_up.obu", ".work/av1gate/aom_scaled_up.ref.yuv"},
		{".work/av1fx/aom_segments.obu", ".work/av1gate/aom_segments.ref.yuv"},
		{".work/av1fx/aom_tilegroups.obu", ".work/av1gate/aom_tilegroups.ref.yuv"},
		{".work/av1fx/rav1e.obu", ".work/av1gate/rav1e.ref.yuv"},
		{".work/av1fx/svt_grain.obu", ".work/av1gate/svt_grain.ref.yuv"},
		{".work/av1fx/svt_superres10.obu", ".work/av1gate/svt_superres10.ref.yuv"},
	}
	for pair in files {
		path := pair[0]
		ref_path := pair[1]
		data, rerr := os.read_entire_file(path, context.allocator)
		if rerr != nil do continue
		defer delete(data)
		ref_yuv, yrr := os.read_entire_file(ref_path, context.allocator)
		if yrr != nil do continue
		defer delete(ref_yuv)

		d: Decoder
		init_decoder(&d)
		defer destroy_decoder(&d)

		unit := 0
		shown_idx := 0
		ref_offset := 0
		for at := 0; at < len(data); {
			end := at
			for end < len(data) {
				o, next, oerr := av1.next_obu(data, end)
				if oerr != .None do break
				if o.type == .Temporal_Delimiter && end > at do break
				end = next
			}
			pic, shown, derr := decode_packet(&d, data[at:end])
			if derr != .None {
				perr := av1.parse_temporal_unit(&d.parser, data[at:end], &d.parsed_frames, &d.parsed_tiles)
				testing.expect(t, false, fmt.tprintf("%s unit=%d shown=%d err=%v perr=%v frames=%d tiles=%d", path, unit, shown_idx, derr, perr, len(d.parsed_frames), len(d.parsed_tiles)))
				break
			}
			if shown {
				w, h := pic.width, pic.height
				uv_w := (w + int(pic.subsampling_x)) >> pic.subsampling_x
				uv_h := (h + int(pic.subsampling_y)) >> pic.subsampling_y
				bps := pic.bit_depth > 8 ? 2 : 1
				frame_bytes := (w * h + 2 * uv_w * uv_h) * bps
				diff := false
				if pic.bit_depth == 8 {
					for y in 0 ..< h {
						for x in 0 ..< w {
							got := pic.y[y * pic.y_stride + x]
							exp := ref_yuv[ref_offset + y * w + x]
							if got != exp && !diff {
								testing.expect(t, false, fmt.tprintf("%s shown=%d Y(%d,%d) got=%d exp=%d", path, shown_idx, x, y, got, exp))
								diff = true
							}
						}
					}
					u_off := ref_offset + w * h
					v_off := u_off + uv_w * uv_h
					for y in 0 ..< uv_h {
						for x in 0 ..< uv_w {
							got_u := pic.u[y * pic.uv_stride + x]
							exp_u := ref_yuv[u_off + y * uv_w + x]
							if got_u != exp_u && !diff {
								testing.expect(t, false, fmt.tprintf("%s shown=%d U(%d,%d) got=%d exp=%d", path, shown_idx, x, y, got_u, exp_u))
								diff = true
							}
							got_v := pic.v[y * pic.uv_stride + x]
							exp_v := ref_yuv[v_off + y * uv_w + x]
							if got_v != exp_v && !diff {
								testing.expect(t, false, fmt.tprintf("%s shown=%d V(%d,%d) got=%d exp=%d", path, shown_idx, x, y, got_v, exp_v))
								diff = true
							}
						}
					}
				} else {
					for y in 0 ..< h {
						for x in 0 ..< w {
							got := pic.y16[y * pic.y_stride + x]
							idx := ref_offset + (y * w + x) * 2
							exp := u16(ref_yuv[idx]) | (u16(ref_yuv[idx + 1]) << 8)
							if got != exp && !diff {
								testing.expect(t, false, fmt.tprintf("%s shown=%d Y16(%d,%d) got=%d exp=%d", path, shown_idx, x, y, got, exp))
								diff = true
							}
						}
					}
					u_off := ref_offset + w * h * 2
					v_off := u_off + uv_w * uv_h * 2
					for y in 0 ..< uv_h {
						for x in 0 ..< uv_w {
							got_u := pic.u16[y * pic.uv_stride + x]
							idx_u := u_off + (y * uv_w + x) * 2
							exp_u := u16(ref_yuv[idx_u]) | (u16(ref_yuv[idx_u + 1]) << 8)
							if got_u != exp_u && !diff {
								testing.expect(t, false, fmt.tprintf("%s shown=%d U16(%d,%d) got=%d exp=%d", path, shown_idx, x, y, got_u, exp_u))
								diff = true
							}
							got_v := pic.v16[y * pic.uv_stride + x]
							idx_v := v_off + (y * uv_w + x) * 2
							exp_v := u16(ref_yuv[idx_v]) | (u16(ref_yuv[idx_v + 1]) << 8)
							if got_v != exp_v && !diff {
								testing.expect(t, false, fmt.tprintf("%s shown=%d V16(%d,%d) got=%d exp=%d", path, shown_idx, x, y, got_v, exp_v))
								diff = true
							}
						}
					}
				}
				ref_offset += frame_bytes
				if diff do break
				shown_idx += 1
			}
			unit += 1
			at = end
		}
	}
}

