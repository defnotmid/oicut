# WebP

WebP (RFC 9649) from memory to RGBA8 with straight alpha, and back. Decoding:
- lossy images (VP8 key frames, RFC 6386);
- lossless images (VP8L);
- alpha (ALPH, raw or lossless-compressed, with its three filters);
- the extended format (VP8X);
- animation (ANIM/ANMF), composited frame by frame;
- ICC, EXIF and XMP chunks, returned as stored.

Encoding is lossless: stills and animations whose pixels decode to the bytes
given. There is no lossy encoder yet.

```odin
import webp "../webp"

img, err := webp.decode(data)            // a still image, or an animation's first frame
defer webp.destroy_image(&img)           // img.pixels: width * height * 4, RGBA
ft, _ := webp.get_features(data)         // size, alpha, animated, frame and loop counts

a: webp.Animation
webp.init_animation(&a, data) or_return
defer webp.destroy_animation(&a)
for _ in 0 ..< a.features.frame_count {
	frame := webp.next_frame(&a) or_return // frame.pixels: the whole canvas; frame.start, frame.duration (ms)
}

// Lossy images as decoded, for conversion on the GPU (like video.Reader's
// planar output): 4:2:0 planes sized to whole macroblocks, and alpha.
yuv, alpha, e := webp.decode_yuva(data)

meta, _ := webp.get_metadata(data)       // meta.icc, meta.exif, meta.xmp: slices of data

file, err := webp.encode(pixels, width, height)  // lossless; delete when done
err = webp.write_file("frame.webp", pixels, width, height)

w: webp.Animation_Writer
webp.init_animation_writer(&w, width, height) or_return
defer webp.destroy_animation_writer(&w)
for f in frames do webp.write_frame(&w, f.pixels, f.duration) or_return  // the whole canvas each time
file := webp.finish_animation(&w) or_return
```

`decode`, `init_animation`, `encode` and `init_animation_writer` take an
optional `^common.Work_Pool`. The conversion of lossy images to RGBA runs on
it, row by row, and so does the encoder's choice of predictors.

## Scope

- **Every coding tool of both bitstreams:**
  - lossy: segments, the simple and normal loop filters and their sharpness, multiple token partitions, and every intra mode;
  - lossless: all four transforms (predictor, cross-colour, subtract-green, colour indexing with pixel packing), colour caches up to 11 bits, and meta prefix codes per block.
- **Colour:** RGB is BT.601 studio range with libwebp's fixed-point constants, and chroma is upsampled with libwebp's "fancy" 9-3-3-1 filter. Output is identical to `dwebp`'s.
- **Animation:**
  - frames overwrite or alpha-blend (libwebp's integer non-premultiplied blend) and dispose to transparent;
  - the canvas starts transparent, and `ANIM`'s background colour is reported (`Features.background`) but not painted, as libwebp does.
- **Not read:** raw VP8/VP8L bitstreams without the RIFF container. The ICC profile is returned, not applied (sRGB is assumed).
- **Damaged input:**
  - a file shorter than its RIFF size is refused (`.Invalid`), by `get_features` too, as libwebp refuses to decode it;
  - inside a whole file, a lossy chunk with no room for its first partition, partition sizes or last partition is refused (`.Invalid`), and a partition that runs out after that decodes as zeros, as a damaged frame would;
  - inside a whole file, a lossless stream that runs out is refused (`.Invalid`).

## Encoding

`encode` writes a lossless file (a `VP8L` chunk); an `Animation_Writer` takes
the whole canvas for each frame and stores the rectangle that changed since
the frame before, overwriting it. Pixels come back exactly, also the colour
of fully transparent ones.

- **Up to 256 colours:** colour indexing, the indices packed 1, 2, 4 or 8 to a pixel.
- **Otherwise:** green is subtracted from red and blue, and each 16×16 block takes the predictor whose residuals have the lowest entropy (all 14 are tried).
- **Coding:** literals, colour-cache hits and copies from the pixel before, the pixel above, or the last place the same three pixels began. One set of prefix codes covers the image, with the colour cache size (none, or 1 to 11 bits) that codes it smallest.
- **Not used:** the cross-colour transform, prefix codes per block, and longer match searches. libwebp's default effort uses them and is about 5% smaller on its test corpus, at several times the time.

## Layout

- **VP8:** macroblocks are predicted and reconstructed in a small bordered workspace. The loop filter then runs over the whole frame: each edge is 16 lanes, each with the 8 pixels across it, computed branch-free so the compiler vectorizes it. Vertical edges use one transposed copy of the macroblock.
- **VP8L:**
  - prefix codes decode through 8-bit root tables stored flat, one offset per code;
  - groups whose red, blue and alpha codes have a single symbol skip those reads;
  - groups whose codes are all short decode a whole pixel with one 6-bit lookup;
  - predictors run per block run, the per-channel ones on `#simd[4]i16` lanes.
- **Colour conversion:** 16 pixels at a time in 16-bit lanes.
- **Tables:** `tables.odin` is generated from libwebp by [`tools/tables`](../../tools/tables/README.md).

## Verification

`bash tests/webp_check.sh`, with fixtures from `bash tests/webp_fixtures.sh`. The fixture script:
- fetches libwebp's test corpus at a pinned commit;
- builds libwebp's tools from `third_party/libwebp`;
- writes `dwebp`'s decodes of every file;
- encodes animations from FFmpeg frames.

The gate checks:
- **All 131 corpus files** (88 lossy, among them the 61 VP8 conformance streams; 43 lossless): RGBA identical to `dwebp -pam`, and for lossy files the Y, U, V and alpha planes identical to `dwebp -pgm`. This covers odd sizes (1×1, 13×1), every alpha filter and compression method, and an 11158×2156 lossy image.
- **The 11158×2156 lossy image on 8 threads:** identical.
- **Animations** (lossless, lossy, mixed, opaque, and one that blends semi-transparent frames over a frame disposed to transparent): every canvas identical to `anim_dump`'s. `anim_dump` writes fully transparent pixels as transparent black, so ours are compared the same way.
- **Fuzzing:** 3,000 rounds of mutated and truncated files through every entry point, under AddressSanitizer with leak checking.

`bash tests/webp_encode_check.sh` covers the encoder, with the same fixtures:
- **All 131 corpus files** re-encoded: `dwebp` decodes each to the same RGBA as it does the original. The 11158×2156 image encodes to the same bytes on 8 threads.
- **Five animations** re-encoded through `Animation_Writer`: every canvas identical in `anim_dump`.
- **Size:** over the corpus 48.2 MB, against 45.7 MB from `cwebp -lossless` and 47.8 MB from `cwebp -lossless -z 0`.

`odin test image/webp` round-trips pictures of every kind the encoder treats
differently (1 to 256 colours and more, noise, gradients, alpha) at sizes from
1×1, and an animation whose frames change everywhere, in one rectangle, in one
pixel and not at all.

Speed against libwebp 1.6.0 (`WebPDecodeRGBA`, `-O3`, its SSE2/SSE4.1/AVX2 code)
on one core, per image and summed over the corpus, is in
[BENCHMARKS.md](../../BENCHMARKS.md) (`bash tests/compare.sh run decode/webp`).

Lossy decoding is bound by the boolean entropy decoder, as libwebp's scalar
code is. Most of the remaining gap is libwebp's SIMD inverse transforms and
predictors.
