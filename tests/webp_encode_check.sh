#!/usr/bin/env bash
# WebP encoder gate: every file of libwebp's test corpus re-encoded
# losslessly by webp and decoded by dwebp to the same RGBA as dwebp gives
# for the original; the largest again on a Work_Pool; and the animations
# re-encoded, every canvas identical in anim_dump. Prints the total size
# against cwebp -lossless at its default and fastest settings.
# Fixtures: tests/webp_fixtures.sh.
#   bash tests/webp_encode_check.sh [fixture-dir]
set -uo pipefail
fx=${1:-.work/webpfx}
work=.work/webpencgate
b=$fx/build
rm -rf "${work:?}" && mkdir -p "$work/out" "$work/cwebp"
odin build tests/webp_encode.odin -file -out:"$work/webp_encode" -o:speed -vet -strict-style || exit 1
pass=0 fail=0 ours=0 default=0 fastest=0 count=0
for f in "$fx"/testdata/*.webp; do
	n=$(basename "$f" .webp)
	if log=$("$work/webp_encode" "$f" "$work/out/$n.webp" 2>&1) && "$b/dwebp" -quiet "$work/out/$n.webp" -pam -o "$work/out/$n.pam" && cmp -s "$work/out/$n.pam" "$fx/ref/$n.pam"; then
		pass=$((pass + 1))
	else
		echo "FAIL $n: $log"
		fail=$((fail + 1))
		continue
	fi
	"$b/cwebp" -quiet -lossless -exact "$fx/ref/$n.pam" -o "$work/cwebp/$n.webp"
	"$b/cwebp" -quiet -lossless -exact -z 0 "$fx/ref/$n.pam" -o "$work/cwebp/$n.z0.webp"
	ours=$((ours + $(stat -c %s "$work/out/$n.webp")))
	default=$((default + $(stat -c %s "$work/cwebp/$n.webp")))
	fastest=$((fastest + $(stat -c %s "$work/cwebp/$n.z0.webp")))
	count=$((count + 1))
	rm -f "$work/out/$n.pam"
done
echo "ok   corpus: $pass of $((pass + fail)) files decode in dwebp to the original's pixels"
echo "     bytes over $count files: ours $ours, cwebp -lossless $default, cwebp -lossless -z 0 $fastest"
if log=$("$work/webp_encode" "$fx/testdata/bryce.webp" "$work/out/bryce8.webp" 8 2>&1) && cmp -s "$work/out/bryce8.webp" "$work/out/bryce.webp"; then
	pass=$((pass + 1))
	echo "ok   bryce on 8 threads: the same bytes ($log)"
else
	echo "FAIL bryce on 8 threads: $log"
	fail=$((fail + 1))
fi
for n in lossless lossy mixed opaque blend; do
	rm -rf "$work/anim_$n" && mkdir -p "$work/anim_$n"
	ok=1
	log=$("$work/webp_encode" "$fx/anim/$n.webp" "$work/out/anim_$n.webp" 2>&1) && "$b/anim_dump" -pam -folder "$work/anim_$n" "$work/out/anim_$n.webp" >/dev/null && ok=0
	frames=0
	for ref in "$fx/anim/ref_$n"/dump_*.pam; do
		cmp -s "$ref" "$work/anim_$n/$(basename "$ref")" || ok=1
		frames=$((frames + 1))
	done
	[ "$(ls "$work/anim_$n" | wc -l)" = "$frames" ] || ok=1
	if [ $ok = 0 ]; then
		pass=$((pass + 1))
		echo "ok   animation $n: $frames canvases identical in anim_dump ($log; the original is $(stat -c %s "$fx/anim/$n.webp") bytes)"
	else
		echo "FAIL animation $n: $log"
		fail=$((fail + 1))
	fi
	rm -rf "$work/anim_$n"
done
echo "$pass passed, $fail failed"
[ "$fail" -eq 0 ]
