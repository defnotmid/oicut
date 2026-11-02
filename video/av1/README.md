# AV1

Native decoding (`video/av1/decoder`) and encoding (`video/av1/encoder`) after
the AV1 Bitstream & Decoding Process Specification v1.0.0 with Errata 1, and
the syntax (`video/av1`) that Vulkan Video decodes from
([`media/vulkan/av1_frontend.odin`](../../media/vulkan/av1_frontend.odin),
`VK_KHR_video_decode_av1`).

## Packages

- `video/av1`: OBU framing, the sequence and frame headers (reference state,
  segmentation, loop filter, CDEF, loop restoration, global motion, film
  grain) and tile groups, with field names after the specification and
  Vulkan's `StdVideoAV1*` structures; the multi-symbol entropy coder and its
  CDF adaptation; every transform type and size, forward and inverse, and the
  lossless Walsh-Hadamard; intra prediction (directional, smooth, Paeth,
  filter intra, CfL, palette, intra block copy); inter prediction (8-tap and
  scaled convolution, warped and global motion, compound and OBMC); the
  motion vector stack and projected motion field; deblocking, CDEF, loop
  restoration (Wiener and self-guided) and film grain synthesis.
  `tables.odin` is generated from the specification's text by
  [`tools/tables`](../../tools/tables/README.md) (`av1`).
- `video/av1/decoder`: profiles 0, 1 and 2 (Main, High, Professional), 8, 10
  and 12 bits, monochrome, 4:2:0, 4:2:2 and 4:4:4; tiles decode in parallel
  on a `common.Work_Pool`. Output is identical to dav1d's on every fixture of
  `tests/av1_fixtures.sh` (libaom, SVT-AV1 and rav1e streams with tiles, tile
  groups, superres, scaled references, segmentation, error resilience and
  film grain) and on `tests/compare.sh`'s 720p 10-bit clip.
- `video/av1/encoder`: the same profiles and formats; intra and inter mode
  decision (single and compound references, golden and altref frames with
  `show_existing_frame`), segmentation and per-superblock `delta_q`,
  deblocking and CDEF; tiles encode in parallel. `ultrafast` trades size for
  speed.

`media/native` decodes and encodes AV1 (one temporal unit per packet,
`Framing.Frame`); `video.Reader` reads it from MP4 (`av01`) and
WebM/Matroska (`V_AV1`), through Vulkan Video where the device decodes it and
natively otherwise, and `video.Writer` writes it to MP4.

## Performance

One core, `-o:speed -microarch:native`: the 162-frame 720p 10-bit SVT-AV1 clip
decodes in 1.34 times dav1d's time (with its assembly). Encoding 1080p through
`video.Writer` at QP 22 to 37 runs at 33 to 61 fps on one core, at the
compression of oicut's own H.265 encoder (-0.5% BD-rate), which runs at 4 to
9 fps. Figures are in [BENCHMARKS.md](../../BENCHMARKS.md).

## Verification

- `odin test video/av1`, `video/av1/decoder` and `video/av1/encoder` (the
  decoder's comparisons with dav1d read fixtures from `.work/` and are skipped
  without them).
- `bash tests/av1_fixtures.sh && bash tests/av1_check.sh`: frame headers
  against FFmpeg's `trace_headers`, then Vulkan Video and native decoding of
  every fixture against libdav1d, bit-exact.
