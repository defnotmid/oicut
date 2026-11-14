// Re-encodes a WebP file losslessly with webp's encoder, for checking the
// result with libwebp's tools (tests/webp_encode_check.sh).
//
//   webp_encode in.webp out.webp [threads] [repeat]
//
// A still is decoded and encoded `repeat` times (the fastest is reported);
// an animation's canvases go through an Animation_Writer with their durations.
package main

import "../common"
import webp "../image/webp"
import "core:fmt"
import "core:os"
import "core:strconv"
import "core:time"

fail :: proc(args: ..any) {
	fmt.eprintln(..args)
	os.exit(1)
}

main :: proc() {
	if len(os.args) < 3 do fail("usage: webp_encode in.webp out.webp [threads] [repeat]")
	data, rerr := os.read_entire_file(os.args[1], context.allocator)
	if rerr != nil do fail("cannot read", os.args[1])
	threads, repeat := 1, 1
	if len(os.args) > 3 do threads, _ = strconv.parse_int(os.args[3])
	if len(os.args) > 4 do repeat, _ = strconv.parse_int(os.args[4])
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
			if e2 := webp.write_frame(&w, frame.pixels, frame.duration); e2 != .None do fail("write_frame", i, e2)
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
			out, e = webp.encode(img.pixels, img.width, img.height, work_pool)
			best = min(best, time.duration_milliseconds(time.tick_since(start)))
			if e != .None do fail("encode:", e)
		}
	}
	if os.write_entire_file(os.args[2], out) != nil do fail("cannot write", os.args[2])
	fmt.printfln("%dx%d frames=%d bytes=%d ms=%.3f", ft.width, ft.height, max(ft.frame_count, 1), len(out), best)
}
