#!/usr/bin/env bash
# WebP encoder gate: every file of libwebp's test corpus re-encoded
# losslessly and lossily (-q 75) by webp, checked with dwebp and webp's own
# decoder; the largest again on a Work_Pool; and the animations re-encoded
# both ways, checked in anim_dump and webp_anim. Prints total sizes and
# visible-pixel RGB PSNR against cwebp.
# Fixtures: tests/webp_fixtures.sh.
#   bash tests/webp_encode_check.sh [fixture-dir]
set -uo pipefail
fx=${1:-.work/webpfx}
work=.work/webpencgate
b=$fx/build
rm -rf "${work:?}" && mkdir -p "$work/out" "$work/cwebp" "$work/lossy" "$work/cwebp_lossy"
odin build tests/webp_encode.odin -file -out:"$work/webp_encode" -o:speed -vet -strict-style || exit 1
odin build tests/webp_decode.odin -file -out:"$work/webp_decode" -o:speed -vet -strict-style || exit 1
odin build tests/webp_anim.odin -file -out:"$work/webp_anim" -o:speed -vet -strict-style || exit 1
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
echo "ok   corpus (lossless): $pass of $((pass + fail)) files decode in dwebp to the original's pixels"
echo "     bytes over $count files: ours $ours, cwebp -lossless $default, cwebp -lossless -z 0 $fastest"
if log=$("$work/webp_encode" "$fx/testdata/bryce.webp" "$work/out/bryce8.webp" 8 2>&1) && cmp -s "$work/out/bryce8.webp" "$work/out/bryce.webp"; then
	pass=$((pass + 1))
	echo "ok   bryce lossless on 8 threads: the same bytes ($log)"
else
	echo "FAIL bryce lossless on 8 threads: $log"
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
		echo "ok   animation $n (lossless): $frames canvases identical in anim_dump ($log; the original is $(stat -c %s "$fx/anim/$n.webp") bytes)"
	else
		echo "FAIL animation $n (lossless): $log"
		fail=$((fail + 1))
	fi
	rm -rf "$work/anim_$n"
done

lossy_pass=0 lossy_fail=0 lossy_ours=0 lossy_cwebp=0 lossy_count=0
tot_vis=0 tot_se_ours=0 tot_se_cwebp=0
for f in "$fx"/testdata/*.webp; do
	n=$(basename "$f" .webp)
	if log=$("$work/webp_encode" -q 75 "$f" "$work/lossy/$n.webp" 2>&1) &&
		"$b/dwebp" -quiet "$work/lossy/$n.webp" -pam -o "$work/lossy/$n.pam" &&
		"$b/dwebp" -quiet "$work/lossy/$n.webp" -pgm -o "$work/lossy/$n.pgm" &&
		dlog=$("$work/webp_decode" "$work/lossy/$n.webp" "$work/lossy/$n.pam" "$work/lossy/$n.pgm" 2>&1) &&
		[[ "$dlog" == *"RGBA identical to dwebp; YUVA identical to dwebp" ]] &&
		"$b/cwebp" -quiet -q 75 "$fx/ref/$n.pam" -o "$work/cwebp_lossy/$n.webp" &&
		"$b/dwebp" -quiet "$work/cwebp_lossy/$n.webp" -pam -o "$work/cwebp_lossy/$n.pam" &&
		clog=$("$work/webp_encode" -cmp "$fx/ref/$n.pam" "$work/lossy/$n.pam" "$work/cwebp_lossy/$n.pam" 2>&1); then
		read -r vis se_o se_c _ _ <<< "$clog"
		tot_vis=$((tot_vis + vis))
		tot_se_ours=$((tot_se_ours + se_o))
		tot_se_cwebp=$((tot_se_cwebp + se_c))
		lossy_ours=$((lossy_ours + $(stat -c %s "$work/lossy/$n.webp")))
		lossy_cwebp=$((lossy_cwebp + $(stat -c %s "$work/cwebp_lossy/$n.webp")))
		lossy_count=$((lossy_count + 1))
		lossy_pass=$((lossy_pass + 1))
		pass=$((pass + 1))
	else
		echo "FAIL lossy $n: ${log:-} | ${dlog:-} | ${clog:-}"
		lossy_fail=$((lossy_fail + 1))
		fail=$((fail + 1))
	fi
	rm -f "$work/lossy/$n.pam" "$work/lossy/$n.pgm" "$work/cwebp_lossy/$n.pam"
done
psnr_ours=$("$work/webp_encode" -psnr "$tot_se_ours" "$tot_vis")
psnr_cwebp=$("$work/webp_encode" -psnr "$tot_se_cwebp" "$tot_vis")
echo "ok   corpus (lossy -q 75): $lossy_pass of $((lossy_pass + lossy_fail)) files match dwebp RGBA+YUVA and preserve exact alpha"
echo "     bytes over $lossy_count files: ours $lossy_ours (${psnr_ours} dB), cwebp -q 75 $lossy_cwebp (${psnr_cwebp} dB)"
if log=$("$work/webp_encode" -q 75 "$fx/testdata/bryce.webp" "$work/lossy/bryce8.webp" 8 2>&1) && cmp -s "$work/lossy/bryce8.webp" "$work/lossy/bryce.webp"; then
	pass=$((pass + 1))
	echo "ok   bryce lossy on 8 threads: the same bytes ($log)"
else
	echo "FAIL bryce lossy on 8 threads: $log"
	fail=$((fail + 1))
fi
for n in lossless lossy mixed opaque blend; do
	rm -rf "$work/lossy_anim_$n" && mkdir -p "$work/lossy_anim_$n"
	ok=1
	if log=$("$work/webp_encode" -q 75 "$fx/anim/$n.webp" "$work/lossy/anim_$n.webp" 2>&1) &&
		"$b/anim_dump" -pam -folder "$work/lossy_anim_$n" "$work/lossy/anim_$n.webp" >/dev/null &&
		alog=$("$work/webp_anim" "$work/lossy/anim_$n.webp" "$work/lossy_anim_$n" 2>&1) &&
		[[ "$alog" == *"all identical to anim_dump"* ]]; then
		ok=0
	fi
	frames=0 anim_vis=0 anim_se=0
	for ref in "$fx/anim/ref_$n"/dump_*.pam; do
		if [ $ok = 0 ] && clog=$("$work/webp_encode" -cmp "$ref" "$work/lossy_anim_$n/$(basename "$ref")" 2>&1); then
			read -r vis se_o _ <<< "$clog"
			anim_vis=$((anim_vis + vis))
			anim_se=$((anim_se + se_o))
		else
			ok=1
		fi
		frames=$((frames + 1))
	done
	[ "$(ls "$work/lossy_anim_$n" | wc -l)" = "$frames" ] || ok=1
	anim_psnr=$("$work/webp_encode" -psnr "$anim_se" "$anim_vis")
	if [ $ok = 0 ]; then
		pass=$((pass + 1))
		echo "ok   animation $n (lossy -q 75): $frames canvases match anim_dump, exact alpha, ${anim_psnr} dB ($log; original $(stat -c %s "$fx/anim/$n.webp") bytes)"
	else
		echo "FAIL animation $n (lossy -q 75): ${log:-} | ${alog:-} | ${clog:-}"
		fail=$((fail + 1))
	fi
	rm -rf "$work/lossy_anim_$n"
done
echo "$pass passed, $fail failed"
[ "$fail" -eq 0 ]
