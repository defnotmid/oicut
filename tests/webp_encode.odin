// Re-encodes a WebP file losslessly (default) or lossily (-q quality) with
// webp's encoder, for checking the result with libwebp's tools
// (tests/webp_encode_check.sh).
//
//   webp_encode [-q quality] in.webp out.webp [threads] [repeat]
//   webp_encode -cmp ref.pam ours.pam [other.pam]
//   webp_encode -psnr se channels
//
// A still is decoded and encoded `repeat` times (the fastest is reported);
// an animation's canvases go through an Animation_Writer with their durations.
package main

import "../common"
import webp "../image/webp"
import "core:fmt"
import "core:math"
import "core:os"
import "core:strconv"
import "core:strings"
import "core:time"

fail :: proc(args: ..any) {
	fmt.eprintln(..args)
	os.exit(1)
}

pam_raster :: proc(path: string) -> (hdr: string, body: []u8) {
	data, err := os.read_entire_file(path, context.allocator)
	if err != nil do fail("cannot read", path)
	at := strings.index(string(data), "ENDHDR\n")
	if at < 0 do fail("bad PAM header:", path)
	return string(data[:at]), data[at + 7:]
}

psnr_db :: proc(se, n: u64) -> f64 {
	if n == 0 || se == 0 do return 99.0
	return 10.0 * math.log10(255.0 * 255.0 * f64(n) / f64(se))
}

compare_pam :: proc(args: []string) {
	if len(args) < 2 do fail("usage: webp_encode -cmp ref.pam ours.pam [other.pam]")
	href, ref := pam_raster(args[0])
	hours, ours := pam_raster(args[1])
	if href != hours || len(ref) != len(ours) do fail("PAM header/size mismatch:", args[1])
	has_other := len(args) > 2
	other: []u8
	if has_other {
		hother: string
		hother, other = pam_raster(args[2])
		if href != hother || len(ref) != len(other) do fail("PAM header/size mismatch:", args[2])
	}
	pixels := len(ref) / 4
	vis, se_ours, se_other: u64
	for i in 0 ..< pixels {
		a := ref[i * 4 + 3]
		if ours[i * 4 + 3] != a do fail("alpha mismatch at pixel", i, "expected", a, "got", ours[i * 4 + 3])
		if a == 0 do continue
		vis += 3
		for c in 0 ..< 3 {
			d := int(ref[i * 4 + c]) - int(ours[i * 4 + c])
			se_ours += u64(d * d)
			if has_other {
				d2 := int(ref[i * 4 + c]) - int(other[i * 4 + c])
				se_other += u64(d2 * d2)
			}
		}
	}
	fmt.printfln("%d %d %d %.2f %.2f", vis, se_ours, se_other, psnr_db(se_ours, vis), psnr_db(se_other, vis))
}

main :: proc() {
	args := os.args[1:]
	if len(args) >= 1 && args[0] == "-cmp" {
		compare_pam(args[1:])
		return
	}
	if len(args) == 3 && args[0] == "-psnr" {
		se, _ := strconv.parse_u64(args[1])
		n, _ := strconv.parse_u64(args[2])
		fmt.printf("%.2f\n", psnr_db(se, n))
		return
	}
	lossy := false
	quality := 75
	if len(args) >= 2 && args[0] == "-q" {
		q, ok := strconv.parse_int(args[1])
		if !ok do fail("invalid quality:", args[1])
		lossy, quality = true, q
		args = args[2:]
	}
	if len(args) < 2 do fail("usage: webp_encode [-q quality] in.webp out.webp [threads] [repeat]")
	data, rerr := os.read_entire_file(args[0], context.allocator)
	if rerr != nil do fail("cannot read", args[0])
	threads, repeat := 1, 1
	if len(args) > 2 do threads, _ = strconv.parse_int(args[2])
	if len(args) > 3 do repeat, _ = strconv.parse_int(args[3])
	pool: common.Static_Pool
	work_pool: ^common.Work_Pool
	if threads > 1 {
		common.init_static_pool(&pool, threads - 1)
		work_pool = &pool.work_pool
	}
	ft, ferr := webp.get_features(data)
	if ferr != .None do fail("features:", ferr)
	out: []u8
	best := max(f64)
	if ft.animated {
		a: webp.Animation
		if e := webp.init_animation(&a, data); e != .None do fail("init:", e)
		w: webp.Animation_Writer
		if e := webp.init_animation_writer(&w, ft.width, ft.height, ft.loop_count, work_pool); e != .None do fail("writer:", e)
		start := time.tick_now()
		for i in 0 ..< ft.frame_count {
			frame, e := webp.next_frame(&a)
			if e != .None do fail("frame", i, e)
			e2 := lossy ? webp.write_lossy_frame(&w, frame.pixels, frame.duration, quality) : webp.write_frame(&w, frame.pixels, frame.duration)
			if e2 != .None do fail("write_frame", i, e2)
		}
		e: webp.Error
		out, e = webp.finish_animation(&w)
		if e != .None do fail("finish:", e)
		best = time.duration_milliseconds(time.tick_since(start))
	} else {
		img, e := webp.decode(data)
		if e != .None do fail("decode:", e)
		for _ in 0 ..< max(repeat, 1) {
			delete(out)
			start := time.tick_now()
			if lossy {
				out, e = webp.encode_lossy(img.pixels, img.width, img.height, quality, work_pool)
			} else {
				out, e = webp.encode(img.pixels, img.width, img.height, work_pool)
			}
			best = min(best, time.duration_milliseconds(time.tick_since(start)))
			if e != .None do fail("encode:", e)
		}
	}
	if os.write_entire_file(args[1], out) != nil do fail("cannot write", args[1])
	fmt.printfln("%dx%d frames=%d bytes=%d ms=%.3f", ft.width, ft.height, max(ft.frame_count, 1), len(out), best)
}
