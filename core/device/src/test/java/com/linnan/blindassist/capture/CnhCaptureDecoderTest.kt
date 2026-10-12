package com.linnan.blindassist.capture

import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Test

class CnhCaptureDecoderTest {
    @Test fun joinsFragmentedPacketAndUsesReceiptOfFinalNewline() {
        val packet = bytes(frame())
        val decoder = CnhCaptureDecoder()
        assertTrue(decoder.feed(packet.copyOfRange(0, 100), 10).isEmpty())
        val record = decoder.feed(packet.copyOfRange(100, packet.size), 20).single()
        assertArrayEquals(packet, record.rawLine)
        assertEquals(20L, record.hostReceivedMonotonicNs)
        assertEquals(1L, record.frame!!.sequence)
        assertEquals(64, record.frame!!.validCellCount)
        assertTrue(record.issues.isEmpty())
    }

    @Test fun preservesSignedHistogramScalersAndUnsignedDeviceFieldsWithoutNormalization() {
        val obj = frame(0xffffffffL, 0xffffffffL)
        obj.getJSONArray("hist_raw").getJSONArray(0).put(0, Int.MIN_VALUE)
        obj.getJSONArray("hist_raw").getJSONArray(0).put(1, Int.MAX_VALUE)
        obj.getJSONArray("hist_scaler").getJSONArray(0).put(0, -128)
        obj.getJSONArray("hist_scaler").getJSONArray(0).put(1, 127)
        obj.getJSONArray("distance_mm").put(0, -1)
        obj.getJSONArray("target_status").put(1, 4)
        val record = CnhCaptureDecoder().feed(bytes(obj), 30).single()
        val result = record.frame!!
        assertEquals(0xffffffffL, result.sequence)
        assertEquals(0xffffffffL, result.deviceMs)
        assertEquals(listOf(Int.MIN_VALUE, Int.MAX_VALUE), result.histRaw[0].take(2))
        assertEquals(listOf(-128, 127), result.histScaler[0].take(2))
        assertEquals(62, result.validCellCount)
        assertArrayEquals(bytes(obj), record.rawLine)
    }

    @Test fun keepsBootConfigAndCorruptPacketsAlongsideLaterValidFrames() {
        val boot = "{\"type\":\"boot\",\"boot_id\":\"same-boot\"}\r\n".toByteArray()
        val corrupt = "not-json\n".toByteArray()
        val decoder = CnhCaptureDecoder()
        val records = decoder.feed(boot + corrupt + bytes(frame()), 50)
        assertEquals(3, records.size)
        assertEquals("boot", records[0].kind)
        assertNull(records[0].frame)
        assertArrayEquals(boot, records[0].rawLine)
        assertNull(records[1].frame)
        assertTrue(records[1].issues.single().startsWith("invalid_packet:"))
        assertNotNull(records[2].frame)
        assertArrayEquals(boot + corrupt + bytes(frame()), records.flatMap { it.rawLine.asIterable() }.toByteArray())
    }

    @Test fun reportsGapDuplicateAndUnannouncedResetButDoesNotDiscardRawFrames() {
        val decoder = CnhCaptureDecoder()
        decoder.feed(bytes(frame(10, 1000)), 1)
        val gap = decoder.feed(bytes(frame(13, 1600)), 2).single()
        assertEquals(listOf("sequence_gap:2"), gap.issues)
        val duplicate = decoder.feed(bytes(frame(13, 1600)), 3).single()
        assertTrue(duplicate.issues.contains("duplicate_sequence"))
        assertTrue(duplicate.issues.contains("device_clock_discontinuity"))
        val reset = decoder.feed(bytes(frame(1, 200)), 4).single()
        assertTrue(reset.issues.contains("sequence_reset_or_reorder"))
        assertTrue(reset.issues.contains("device_clock_discontinuity"))
        assertNotNull(reset.frame)
    }

    @Test fun acceptsUint32WrapAndCachedBootDoesNotResetContinuity() {
        val decoder = CnhCaptureDecoder()
        decoder.feed(bytes(frame(0xffffffffL, 0xffffff80L)), 1)
        decoder.feed("{\"type\":\"boot\",\"boot_id\":\"same\"}\n".toByteArray(), 2)
        val wrapped = decoder.feed(bytes(frame(0, 72)), 3).single()
        assertTrue(wrapped.issues.isEmpty())
        val afterCachedBoot = decoder.feed(bytes(frame(2, 472)), 4).single()
        assertEquals(listOf("sequence_gap:1"), afterCachedBoot.issues)
    }

    @Test fun rejectsWrongLayoutHeaderAndCoercedIntegerFields() {
        listOf<(JSONObject) -> Unit>(
            { it.put("rows", 4) },
            { it.put("bins", 24) },
            { it.put("seq", "1") },
            { it.put("seq", 1.5) },
            { it.put("ms", 0x100000000L) },
            { it.put("distance_mm", JSONArray(listOf(1000))) },
            { it.getJSONArray("hist_scaler").getJSONArray(0).put(0, 128) },
            { it.getJSONArray("cnh_header").put(4, (24 shl 16) + 16) }
        ).forEach { mutate ->
            val obj = frame().also(mutate)
            val raw = bytes(obj)
            val record = CnhCaptureDecoder().feed(raw, 0).single()
            assertNull(record.frame)
            assertArrayEquals(raw, record.rawLine)
            assertTrue(record.issues.single().startsWith("invalid_packet:"))
        }
    }

    @Test fun rejectsMalformedUtf8AndPreservesUnterminatedTail() {
        val decoder = CnhCaptureDecoder()
        val invalid = byteArrayOf(0xc3.toByte(), 0x28, 0x0a)
        val bad = decoder.feed(invalid, 100).single()
        assertNull(bad.frame)
        assertTrue(bad.issues.isNotEmpty())
        val tail = bytes(frame()).dropLast(1).toByteArray()
        assertTrue(decoder.feed(tail, 200).isEmpty())
        val partial = decoder.finish()!!
        assertArrayEquals(tail, partial.rawLine)
        assertEquals(listOf("truncated_line"), partial.issues)
        assertEquals(200L, partial.hostReceivedMonotonicNs)
        assertNull(partial.frame)
        assertNull(decoder.finish())
    }

    @Test fun overlongPacketIsRetainedInPartsAndDecoderRecoversAtNextNewline() {
        val decoder = CnhCaptureDecoder(16)
        val overlong = ("x".repeat(35) + "\n").toByteArray()
        val event = "{\"type\":\"ok\"}\n".toByteArray()
        val records = decoder.feed(overlong + event, 300)
        assertArrayEquals(overlong + event, records.flatMap { it.rawLine.asIterable() }.toByteArray())
        assertTrue(records.dropLast(1).all { it.frame == null && it.issues.single().startsWith("line_too_long") })
        assertEquals("ok", records.last().kind)
        assertTrue(records.last().issues.isEmpty())
    }

    @Test fun rangeOnlyFirmwareNeverCountsAsFullCnhFrame() {
        val record = CnhCaptureDecoder().feed("{\"type\":\"frame\",\"seq\":1}\n".toByteArray(), 5).single()
        assertNull(record.frame)
        assertEquals(listOf("range_only_frame_without_cnh"), record.issues)
    }

    @Test fun boundedAssemblyPreservesBytesAcrossEveryChunkBoundaryAndRecovers() {
        val event = "{\"type\":\"ok\"}\r\n".toByteArray()
        val oversized = ("x".repeat(64) + "\n").toByteArray()
        val tail = byteArrayOf(0xff.toByte(), 0x00, 0x7f)
        val raw = oversized + event + "\n".toByteArray() + tail
        for (chunkSize in 1..37) {
            val decoder = CnhCaptureDecoder(32)
            val records = raw.asList().chunked(chunkSize).flatMapIndexed { index, chunk ->
                decoder.feed(chunk.toByteArray(), index.toLong() + 1)
            }.toMutableList()
            records += decoder.finish()!!
            assertArrayEquals("chunk size $chunkSize", raw,
                records.flatMap { it.rawLine.asIterable() }.toByteArray())
            assertEquals(listOf("ok"), records.mapNotNull { it.kind })
            assertTrue(records.take(3).all { it.issues.single().startsWith("line_too_long") })
            assertEquals(listOf("truncated_line"), records.last().issues)
            assertEquals((raw.size + chunkSize - 1L) / chunkSize, records.last().hostReceivedMonotonicNs)
            assertNull(decoder.finish())
        }
    }

    @Test fun newlineAtSizeLimitRemainsACompleteRecord() {
        val packet = "{\"type\":\"ok\"}\n".toByteArray()
        val decoder = CnhCaptureDecoder(packet.size)
        assertTrue(decoder.feed(packet.copyOf(packet.size - 1), 10).isEmpty())
        val record = decoder.feed(byteArrayOf('\n'.code.toByte()), 20).single()
        assertArrayEquals(packet, record.rawLine)
        assertEquals("ok", record.kind)
        assertEquals(20L, record.hostReceivedMonotonicNs)
        assertTrue(record.issues.isEmpty())
    }

    private fun frame(seq: Long = 1, ms: Long = 200): JSONObject = JSONObject().apply {
        put("type", "cnh_frame")
        put("seq", seq)
        put("ms", ms)
        put("read_us", 90000)
        put("stream", 1)
        put("rows", 8)
        put("cols", 8)
        put("bins", 16)
        put("distance_mm", JSONArray(List(64) { 1000 }))
        put("target_status", JSONArray(List(64) { 5 }))
        put("nb_target", JSONArray(List(64) { 1 }))
        put("hist_raw", JSONArray(List(64) { List(16) { bin -> bin - 8 } }))
        put("hist_scaler", JSONArray(List(64) { List(16) { 0 } }))
        put("ambient_raw", JSONArray(List(64) { -5 }))
        put("ambient_scaler", JSONArray(List(64) { 0 }))
        put("cnh_header", JSONArray(listOf(0, 0, 0, 0, (16 shl 16) + 64)))
    }

    private fun bytes(obj: JSONObject) = (obj.toString() + "\n").toByteArray(Charsets.UTF_8)
}
