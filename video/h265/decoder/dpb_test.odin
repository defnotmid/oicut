package h265_decoder

import "core:testing"

// 8.3.2: a picture stays a reference only if the set names it, and is long-term
// only if a long-term list does.
@(test)
dpb_mark_follows_the_reference_picture_set :: proc(t: ^testing.T) {
	d: Decoder
	defer delete(d.dpb)
	pictures: [7]Picture
	for &p, i in pictures {
		p.poc = i
		append(&d.dpb, Dpb_Picture{pic = &p, ref = true, long_term = true})
	}
	rps: Ref_Pic_Set
	defer destroy_ref_pic_set(&rps)
	append(&rps.st_curr_before, 0)
	append(&rps.st_curr_after, 1)
	append(&rps.st_foll, 2)
	append(&rps.lt_curr, 3)
	append(&rps.lt_foll, 4)
	dpb_mark(&d, &rps)
	for e, i in d.dpb {
		testing.expect_value(t, e.ref, i <= 4)
		testing.expect_value(t, e.long_term, i == 3 || i == 4)
	}
}
