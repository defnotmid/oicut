package vp9_encoder

import vp9 ".."
import "../../../common"

// VP9's boolean coder: common.Bool_Encoder framed with section 9.2's leading
// zero marker bit and superframe-index padding at the end.
@(private)
Bool_Writer :: common.Bool_Encoder

@(private)
start_bool :: proc(b: ^Bool_Writer) {
	common.bool_encode_start(b)
	common.bool_encode(b, 0, 128)
}

@(private)
put_bool :: #force_inline proc(b: ^Bool_Writer, bit: int, prob: u8) {
	common.bool_encode(b, bit, prob)
}

@(private)
put_literal :: proc(b: ^Bool_Writer, value, bits: int) {
	common.bool_encode_literal(b, value, bits)
}

@(private)
finish_bool :: proc(b: ^Bool_Writer) {
	common.bool_encode_finish(b)
	// A frame whose last byte looks like a superframe index needs padding.
	if len(b.data) > 0 && b.data[len(b.data) - 1] & 0xE0 == 0xC0 {
		if _, err := non_zero_append(&b.data, 0); err != nil do b.failed = true
	}
}

// One symbol's path from the root of a VP9 binary tree: the bits and the
// probability indices (node >> 1) visited along the way. Built once at init
// so coding and rate estimation walk flat steps without searching the tree.
@(private)
Tree_Code :: struct {
	bits:  u16,
	count: u8,
	nodes: [9]u8,
}

@(private)
PARTITION_CODES: [vp9.PARTITION_TYPES]Tree_Code
@(private)
INTRA_MODE_CODES: [vp9.INTRA_MODES]Tree_Code
@(private)
INTER_MODE_CODES: [vp9.INTER_MODES]Tree_Code
@(private)
INTERP_FILTER_CODES: [vp9.SWITCHABLE_FILTERS]Tree_Code
@(private)
TX_SIZE_CODES: [vp9.TX_SIZES][vp9.TX_SIZES]Tree_Code
@(private)
MV_JOINT_CODES: [vp9.MV_JOINTS]Tree_Code
@(private)
MV_CLASS_CODES: [vp9.MV_CLASSES]Tree_Code
@(private)
MV_FR_CODES: [vp9.MV_FR_SIZE]Tree_Code
@(private)
TOKEN_PARETO_CODES: [11]Tree_Code
@(private)
SEG_CODES: [vp9.MAX_SEGMENTS]Tree_Code

@(private, init)
init_tree_codes :: proc "contextless" () {
	build_tree_codes(PARTITION_CODES[:], vp9.PARTITION_TREE[:], 0, {}, 0)
	build_tree_codes(INTRA_MODE_CODES[:], vp9.INTRA_MODE_TREE[:], 0, {}, 0)
	build_tree_codes(INTER_MODE_CODES[:], vp9.INTER_MODE_TREE[:], 0, {}, 0)
	build_tree_codes(INTERP_FILTER_CODES[:], vp9.INTERP_FILTER_TREE[:], 0, {}, 0)
	build_tree_codes(TX_SIZE_CODES[vp9.TX_8X8][:], vp9.TX_SIZE_8_TREE[:], 0, {}, 0)
	build_tree_codes(TX_SIZE_CODES[vp9.TX_16X16][:], vp9.TX_SIZE_16_TREE[:], 0, {}, 0)
	build_tree_codes(TX_SIZE_CODES[vp9.TX_32X32][:], vp9.TX_SIZE_32_TREE[:], 0, {}, 0)
	build_tree_codes(MV_JOINT_CODES[:], vp9.MV_JOINT_TREE[:], 0, {}, 0)
	build_tree_codes(MV_CLASS_CODES[:], vp9.MV_CLASS_TREE[:], 0, {}, 0)
	build_tree_codes(MV_FR_CODES[:], vp9.MV_FR_TREE[:], 0, {}, 0)
	build_tree_codes(TOKEN_PARETO_CODES[:], vp9.TOKEN_TREE[:], 4, {}, 2)
	build_tree_codes(SEG_CODES[:], vp9.SEGMENT_TREE[:], 0, {}, 0)
}

@(private)
build_tree_codes :: proc "contextless" (out: []Tree_Code, tree: []i8, node: int, code: Tree_Code, base_node: int) {
	for bit in 0 ..< 2 {
		next := code
		next.bits |= u16(bit) << uint(next.count)
		next.nodes[next.count] = u8((node >> 1) - base_node)
		next.count += 1
		child := int(tree[node + bit])
		if child <= 0 {
			out[-child] = next
		} else {
			build_tree_codes(out, tree, child, next, base_node)
		}
	}
}
