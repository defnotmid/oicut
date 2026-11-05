package media_videotoolbox

import "core:sync"
import "core:time"

// Pictures in flight, as the Vulkan encoder keeps them: each encode call
// publishes the picture submitted ENCODE_DEPTH-1 calls earlier and drain
// publishes the rest, so one picture's encoding overlaps the next one's
// submission.
ENCODE_DEPTH :: #config(OICUT_ENCODE_DEPTH, 3)

// How long a picture may take before it is forced out (`take_oldest`): only
// for an encoder that broke its word on how many it may hold back.
ENCODE_PATIENCE :: time.Second

// A VideoToolbox compression session. Packets come out in input order. The
// output callback only keeps VideoToolbox's sample, in its picture's slot;
// the packet is built on the caller's thread, once the slot is marked done.
// The encoder's address is the callback context: do not move it while open.
Encoder :: struct {
	config:   Encoder_Config,
	vt:       VTCompressionSessionRef,
	frames:   i64, // pictures submitted
	taken:    i64, // pictures published; those between are in flight
	slots:    [ENCODE_DEPTH]Encoder_Slot,
	finished: sync.Futex, // bumped as each picture comes back: what `take_oldest` waits on
	sample:   CMSampleBufferRef, // the picture being published, retained
	packet:   [dynamic]u8, // Annex B: parameter sets before keyframes, then the picture
	scratch:  [dynamic]u8, // the sample's length-prefixed NAL units
	status:   i32, // the last OSStatus VideoToolbox returned, for diagnostics
	qp_held:  bool, // the frame QP limits took; otherwise quality approximates qp
}

// What the output callback left for a picture; `done` publishes the rest.
@(private)
Encoder_Slot :: struct {
	sample: CMSampleBufferRef, // retained; nil if it failed or was dropped
	status: OSStatus,
	done:   bool,
}

@(private)
on_encoded :: proc "c" (ref_con, frame_ref_con: rawptr, status: OSStatus, info_flags: u32, sample: CMSampleBufferRef) {
	e := (^Encoder)(ref_con)
	slot := &e.slots[uintptr(frame_ref_con) % ENCODE_DEPTH]
	slot.status = status
	if status == 0 && sample != nil do slot.sample = CMSampleBufferRef(CFRetain(CFTypeRef(sample)))
	sync.atomic_store(&slot.done, true)
	sync.atomic_add(&e.finished, 1)
	sync.futex_signal(&e.finished)
}

// A picture's presentation time: its number, which is also how it is found.
@(private)
picture_time :: proc "contextless" (picture: i64) -> CMTime {
	return {value = picture, timescale = 1, flags = 1} // kCMTimeFlags_Valid
}

encoder_create :: proc(e: ^Encoder, config: Encoder_Config) -> Error {
	e^ = {config = config}
	specification: CFDictionaryRef
	if config.require_hardware {
		keys := [1]rawptr{rawptr(kVTVideoEncoderSpecification_RequireHardwareAcceleratedVideoEncoder)}
		values := [1]rawptr{rawptr(kCFBooleanTrue)}
		specification = CFDictionaryCreate(nil, &keys[0], &values[0], 1, &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks)
	}
	defer if specification != nil do CFRelease(CFTypeRef(specification))
	format := i32(PIXEL_NV12_VIDEO)
	number := CFNumberCreate(nil, CF_NUMBER_SINT32, &format)
	defer CFRelease(CFTypeRef(number))
	attribute_keys := [1]rawptr{rawptr(kCVPixelBufferPixelFormatTypeKey)}
	attribute_values := [1]rawptr{rawptr(number)}
	attributes := CFDictionaryCreate(nil, &attribute_keys[0], &attribute_values[0], 1, &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks)
	defer CFRelease(CFTypeRef(attributes))
	codec: u32 = config.codec == .H264 ? CODEC_H264 : CODEC_HEVC
	e.status = VTCompressionSessionCreate(nil, i32(config.width), i32(config.height), codec, specification, attributes, nil, on_encoded, e, &e.vt)
	if e.status != 0 {
		e.vt = nil
		return .Unavailable
	}
	profile := config.codec == .H264 ? kVTProfileLevel_H264_High_AutoLevel : kVTProfileLevel_HEVC_Main_AutoLevel
	set_property(e, kVTCompressionPropertyKey_ProfileLevel, rawptr(profile)) or_return
	set_property(e, kVTCompressionPropertyKey_AllowFrameReordering, rawptr(kCFBooleanFalse)) or_return
	set_number(e, kVTCompressionPropertyKey_MaxKeyFrameInterval, config.keyframe_interval) or_return
	// A picture must come back before the depth asks for it (`take_oldest`);
	// an encoder without the property keeps its own count.
	_ = set_number(e, kVTCompressionPropertyKey_MaxFrameDelayCount, ENCODE_DEPTH - 1)
	set_property(e, kVTCompressionPropertyKey_ColorPrimaries, rawptr(kCVImageBufferColorPrimaries_ITU_R_709_2)) or_return
	set_property(e, kVTCompressionPropertyKey_TransferFunction, rawptr(kCVImageBufferTransferFunction_ITU_R_709_2)) or_return
	set_property(e, kVTCompressionPropertyKey_YCbCrMatrix, rawptr(kCVImageBufferYCbCrMatrix_ITU_R_709_2)) or_return
	// A constant QP where the encoder honours equal frame QP limits;
	// otherwise its quality setting, from the same 1..51 scale.
	e.qp_held = set_number(e, kVTCompressionPropertyKey_MinAllowedFrameQP, config.qp) == .None
	e.qp_held &&= set_number(e, kVTCompressionPropertyKey_MaxAllowedFrameQP, config.qp) == .None
	if !e.qp_held {
		quality := f32(51 - config.qp) / 50
		value := CFNumberCreate(nil, CF_NUMBER_FLOAT32, &quality)
		defer CFRelease(CFTypeRef(value))
		set_property(e, kVTCompressionPropertyKey_Quality, rawptr(value)) or_return
	}
	e.status = VTCompressionSessionPrepareToEncodeFrames(e.vt)
	return e.status == 0 ? .None : .Encoder_Error
}

@(private)
set_property :: proc(e: ^Encoder, key: CFStringRef, value: rawptr) -> Error {
	e.status = VTSessionSetProperty(rawptr(e.vt), key, value)
	return e.status == 0 ? .None : .Unsupported
}

@(private)
set_number :: proc(e: ^Encoder, key: CFStringRef, value: int) -> Error {
	v := i32(value)
	number := CFNumberCreate(nil, CF_NUMBER_SINT32, &v)
	defer CFRelease(CFTypeRef(number))
	return set_property(e, key, rawptr(number))
}

// Submits one picture and, once ENCODE_DEPTH are in flight, returns the
// oldest's Annex B packet, borrowed until the next call; nil before that.
// The picture is copied before this returns.
encoder_encode :: proc(e: ^Encoder, y: []u8, y_stride: int, chroma: Encoder_Chroma) -> (packet: []u8, keyframe: bool, err: Error) {
	if e.vt == nil do return nil, false, .Invalid_Argument
	image: CVPixelBufferRef
	if CVPixelBufferPoolCreatePixelBuffer(nil, VTCompressionSessionGetPixelBufferPool(e.vt), &image) != 0 do return nil, false, .Encoder_Error
	defer CVBufferRelease(image)
	fill_nv12(e, image, y, y_stride, chroma) or_return

	force := e.frames % i64(max(e.config.keyframe_interval, 1)) == 0
	keys := [1]rawptr{rawptr(kVTEncodeFrameOptionKey_ForceKeyFrame)}
	values := [1]rawptr{rawptr(force ? kCFBooleanTrue : kCFBooleanFalse)}
	options := CFDictionaryCreate(nil, &keys[0], &values[0], 1, &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks)
	defer CFRelease(CFTypeRef(options))
	info: u32
	if status := VTCompressionSessionEncodeFrame(e.vt, image, picture_time(e.frames), kCMTimeInvalid, options, rawptr(uintptr(e.frames)), &info); status != 0 {
		e.status = status
		return nil, false, .Encoder_Error
	}
	e.frames += 1
	if e.frames - e.taken < ENCODE_DEPTH do return nil, false, .None
	return take_oldest(e)
}

// The packets still in flight, oldest first, one a call, each borrowed until
// the next; `more` is false when none are left.
encoder_drain :: proc(e: ^Encoder) -> (packet: []u8, keyframe, more: bool, err: Error) {
	if e.vt == nil || e.taken >= e.frames do return nil, false, false, .None
	packet, keyframe, err = take_oldest(e)
	return packet, keyframe, err == .None, err
}

// Waits for the callback: VTCompressionSessionCompleteFrames, even only up to
// this picture's time, waits for the pictures behind it too, and with it each
// call would see the encoder through one picture at a time again (hello_world
// on the M4: 4.88 s, the same as completing every frame, against 4.06 s).
// MaxFrameDelayCount holds VideoToolbox to the depth, so the picture is coming;
// completing is what's left if it doesn't.
@(private)
take_oldest :: proc(e: ^Encoder) -> (packet: []u8, keyframe: bool, err: Error) {
	slot := &e.slots[e.taken % ENCODE_DEPTH]
	for !sync.atomic_load(&slot.done) {
		seen := sync.atomic_load(&e.finished)
		if sync.atomic_load(&slot.done) do break
		if !sync.futex_wait_with_timeout(&e.finished, u32(seen), ENCODE_PATIENCE) {
			VTCompressionSessionCompleteFrames(e.vt, picture_time(e.taken))
			break
		}
	}
	e.taken += 1
	e.status = slot.status
	e.sample = slot.sample
	slot^ = {}
	if e.sample == nil do return nil, false, .Encoder_Error
	defer {CFRelease(CFTypeRef(e.sample)); e.sample = nil}
	return build_packet(e)
}

// Copies the picture into VideoToolbox's NV12 buffer row by row.
@(private)
fill_nv12 :: proc(e: ^Encoder, image: CVPixelBufferRef, y: []u8, y_stride: int, chroma: Encoder_Chroma) -> Error {
	if CVPixelBufferLockBaseAddress(image, 0) != 0 do return .Encoder_Error
	defer CVPixelBufferUnlockBaseAddress(image, 0)
	w, h := e.config.width, e.config.height
	luma := ([^]u8)(CVPixelBufferGetBaseAddressOfPlane(image, 0))
	luma_stride := int(CVPixelBufferGetBytesPerRowOfPlane(image, 0))
	for row in 0 ..< h do copy(luma[row * luma_stride:][:w], y[row * y_stride:][:w])
	pairs := ([^]u8)(CVPixelBufferGetBaseAddressOfPlane(image, 1))
	pair_stride := int(CVPixelBufferGetBytesPerRowOfPlane(image, 1))
	for row in 0 ..< h / 2 {
		out := pairs[row * pair_stride:][:w]
		if chroma.interleaved {
			copy(out, chroma.planes[0][row * chroma.strides[0]:][:w])
		} else {
			u, v := chroma.planes[0][row * chroma.strides[0]:][:w / 2], chroma.planes[1][row * chroma.strides[1]:][:w / 2]
			for x in 0 ..< w / 2 do out[2 * x], out[2 * x + 1] = u[x], v[x]
		}
	}
	return .None
}

// The sample as Annex B: a keyframe carries the parameter sets of the
// session's format description first.
@(private)
build_packet :: proc(e: ^Encoder) -> (packet: []u8, keyframe: bool, err: Error) {
	clear(&e.packet)
	keyframe = true
	if attachments := CMSampleBufferGetSampleAttachmentsArray(e.sample, false); attachments != nil && CFArrayGetCount(attachments) > 0 {
		first := CFDictionaryRef(CFArrayGetValueAtIndex(attachments, 0))
		keyframe = CFDictionaryGetValue(first, rawptr(kCMSampleAttachmentKey_NotSync)) == nil
	}
	if keyframe {
		format := CMSampleBufferGetFormatDescription(e.sample)
		count: uint = 1
		for i: uint = 0; i < count; i += 1 {
			set: [^]u8
			size: uint
			header: i32
			get := e.config.codec == .H264 ? CMVideoFormatDescriptionGetH264ParameterSetAtIndex : CMVideoFormatDescriptionGetHEVCParameterSetAtIndex
			if e.status = get(format, i, &set, &size, &count, &header); e.status != 0 do return nil, false, .Encoder_Error
			append(&e.packet, 0, 0, 0, 1)
			append(&e.packet, ..set[:size])
		}
	}
	block := CMSampleBufferGetDataBuffer(e.sample)
	resize(&e.scratch, int(CMBlockBufferGetDataLength(block)))
	if e.status = CMBlockBufferCopyDataBytes(block, 0, uint(len(e.scratch)), raw_data(e.scratch)); e.status != 0 do return nil, false, .Encoder_Error
	// 4-byte big-endian lengths, as VideoToolbox writes them.
	for at := 0; at + 4 <= len(e.scratch); {
		n := int(e.scratch[at]) << 24 | int(e.scratch[at + 1]) << 16 | int(e.scratch[at + 2]) << 8 | int(e.scratch[at + 3])
		at += 4
		if n > len(e.scratch) - at do return nil, false, .Encoder_Error
		append(&e.packet, 0, 0, 0, 1)
		append(&e.packet, ..e.scratch[at:][:n])
		at += n
	}
	return e.packet[:], keyframe, .None
}

encoder_destroy :: proc(e: ^Encoder) {
	if e.vt != nil {
		VTCompressionSessionCompleteFrames(e.vt, kCMTimeInvalid)
		VTCompressionSessionInvalidate(e.vt)
		CFRelease(CFTypeRef(e.vt))
	}
	for slot in e.slots {
		if slot.sample != nil do CFRelease(CFTypeRef(slot.sample))
	}
	delete(e.packet)
	delete(e.scratch)
	e^ = {}
}
