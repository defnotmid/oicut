#!/usr/bin/env bash
# Opus encoder gate:
# - every packet of 648 configurations (vectors 01, 07 and 11; mono and
#   stereo; 16 to 256 kb/s; VBR, constrained VBR and CBR; complexity 0, 5
#   and 10; 2.5, 10 and 20 ms frames) decodes to the encoder's final range
#   in libopus's opus_demo and in audio/opus;
# - rate and SNR against libopus's CELT-only encoder (restricted-celt) at
#   complexity 9 on a minute of music, VBR and CBR, 32 to 192 kb/s: the rate
#   within 2% and the SNR within 0.3 dB;
# - a -microarch:native build writes the same bytes;
# - a VP9 and Opus WebM from video.Writer reads back in FFmpeg with the
#   source's sample count.
# Fixtures: tests/opus_fixtures.sh (the vectors and libopus); the music is
# compare.sh's source (WORK, default .work/compare), else vector 01.
#   bash tests/opus_encode_check.sh [fixture-dir]
set -uo pipefail
fx=${1:-.work/opusfx}
work=.work/opusenc-gate
mkdir -p "$work"
export TMPDIR=${TMPDIR:-$PWD/.work/tmp}
mkdir -p "$TMPDIR"
odin build tests/opus_encode.odin -file -out:"$work/opus_encode" -o:speed -vet -strict-style || exit 1
odin build tests/opus_encode.odin -file -out:"$work/opus_encode_native" -o:speed -microarch:native || exit 1
odin build tests/opus_decode.odin -file -out:"$work/opus_decode" -o:speed || exit 1
odin build tests/audio_compare.odin -file -out:"$work/audio_compare" -o:speed || exit 1
odin build tests/opus_webm_write.odin -file -out:"$work/opus_webm_write" -o:speed -vet -strict-style || exit 1
demo=$fx/opus-build/opus_demo
v=$fx/vectors/opus_newvectors
ff=(ffmpeg -hide_banner -loglevel error -y)
pass=0 fail=0
result() { # ok? message
	if [ "$1" = 0 ]; then echo "ok   $2"; pass=$((pass + 1)); else echo "FAIL $2"; fail=$((fail + 1)); fi
}

# Final ranges in both decoders.
configs=0 bad=0
for vec in 01 07 11; do
	for ch in 1 2; do
		in=$v/testvector$vec.dec
		if [ "$ch" = 1 ]; then
			"${ff[@]}" -f s16le -ar 48000 -ac 2 -i "$in" -ac 1 -f s16le "$work/mono.pcm"
			in=$work/mono.pcm
		fi
		for rate in 16000 48000 96000 256000; do
			for mode in vbr cvbr cbr; do
				for cx in 0 5 10; do
					for frame in 120 480 960; do
						configs=$((configs + 1))
						if ! "$work/opus_encode" "$in" "$work/s.bit" "$ch" "$rate" "$cx" "$mode" "$frame" >/dev/null; then
							bad=$((bad + 1)); echo "  encoding failed: vector $vec, $ch ch, $rate, $mode, c$cx, $frame"; continue
						fi
						libopus=$("$demo" -d 48000 "$ch" "$work/s.bit" "$work/s.pcm" 2>&1 | grep -ci "mismatch\|error")
						ours=$("$work/opus_decode" "$work/s.bit" "$work/s.ours.pcm" "$ch" 2>&1 | grep -o "[0-9]* wrong")
						if [ "$libopus" != 0 ] || [ "$ours" != "0 wrong" ]; then
							bad=$((bad + 1)); echo "  range mismatch: vector $vec, $ch ch, $rate, $mode, c$cx, $frame"
						fi
					done
				done
			done
		done
	done
done
result $bad "final ranges: $configs configurations in opus_demo and audio/opus, $bad wrong"

# Against libopus. Both decode with opus_demo, 120 samples late.
music=${WORK:-.work/compare}/inputs/source.s16le
if [ -f "$music" ]; then head -c $((192000 * 60)) "$music" > "$work/music.pcm"; else cp "$v/testvector01.dec" "$work/music.pcm"; fi
seconds=$(( $(stat -c %s "$work/music.pcm") / 192000 ))
snr() { "$work/audio_compare" quality "$work/music.pcm" "$1" 48000 2 120 | sed -n 's/^section=overall.*snr_db=\([^ ]*\).*/\1/p'; }
kbps() { echo "($(stat -c %s "$1") - 8 * $2) * 8 / $seconds / 1000" | bc -l; }
for mode in vbr cbr; do
	flag=; [ "$mode" = cbr ] && flag=-cbr
	for rate in 32000 64000 96000 128000 192000; do
		"$work/opus_encode" "$work/music.pcm" "$work/ours.bit" 2 $rate 9 $mode >/dev/null
		"$demo" -e restricted-celt 48000 2 $rate -complexity 9 $flag "$work/music.pcm" "$work/lib.bit" >/dev/null 2>&1
		"$demo" -d 48000 2 "$work/ours.bit" "$work/ours.pcm" >/dev/null 2>&1
		"$demo" -d 48000 2 "$work/lib.bit" "$work/lib.pcm" >/dev/null 2>&1
		ours_kbps=$(kbps "$work/ours.bit" $(( $(stat -c %s "$work/ours.pcm") / 3840 )))
		lib_kbps=$(kbps "$work/lib.bit" $(( $(stat -c %s "$work/lib.pcm") / 3840 )))
		ours_db=$(snr "$work/ours.pcm") lib_db=$(snr "$work/lib.pcm")
		ok=$(awk -v a="$ours_kbps" -v b="$lib_kbps" -v x="$ours_db" -v y="$lib_db" 'BEGIN {exit !(a <= b * 1.02 && a >= b * 0.98 && x >= y - 0.3)}'; echo $?)
		result "$ok" "$mode $((rate / 1000)) kb/s: $(printf '%.2f' "$ours_kbps") kb/s at $ours_db dB, libopus $(printf '%.2f' "$lib_kbps") kb/s at $lib_db dB"
	done
done

# The native build's bytes.
"$work/opus_encode_native" "$work/music.pcm" "$work/native.bit" 2 128000 9 vbr >/dev/null
"$work/opus_encode" "$work/music.pcm" "$work/default.bit" 2 128000 9 vbr >/dev/null
cmp -s "$work/native.bit" "$work/default.bit"
result $? "-microarch:native writes the same bitstream"

# A WebM from video.Writer in FFmpeg.
"$work/opus_webm_write" "$work/out.webm" "$work/webm_source.pcm"
"${ff[@]}" -i "$work/out.webm" -map 0:v -f null - && "${ff[@]}" -i "$work/out.webm" -map 0:a -f s16le -ac 2 "$work/webm_ff.pcm"
same=$([ "$(stat -c %s "$work/webm_ff.pcm")" = "$(stat -c %s "$work/webm_source.pcm")" ] && echo 0 || echo 1)
db=$("$work/audio_compare" quality "$work/webm_source.pcm" "$work/webm_ff.pcm" 48000 2 0 | sed -n 's/^section=overall.*snr_db=\([^ ]*\).*/\1/p')
result "$same" "video.Writer VP9+Opus WebM: FFmpeg decodes both tracks, $(( $(stat -c %s "$work/webm_ff.pcm") / 4 )) samples (source $(( $(stat -c %s "$work/webm_source.pcm") / 4 ))), $db dB"

echo "$pass passed, $fail failed"
[ "$fail" = 0 ]
