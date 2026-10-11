package com.linnan.blindassist.capture

import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Test

class ImuCaptureDecoderTest {
    @Test fun joinsFragmentedPacketAndKeepsNativeAxesIndependentClockAndEstimatedPose() {
        val raw = bytes(frame().put("device_us", 5_000_000_000L))
        val decoder = ImuCaptureDecoder()
        assertTrue(decoder.feed(raw.copyOfRange(0, 90), 100).isEmpty())
        val record = decoder.feed(raw.copyOfRange(90, raw.size), 200).single()
        assertArrayEquals(raw, record.rawLine)
        assertEquals(200L, record.hostReceivedMonotonicNs)
        val sample = record.frame!!
        assertEquals(5_000_000_000L, sample.deviceUs)
        assertEquals(listOf(-32768, 0, 32767), sample.accelRaw)
        assertEquals(listOf(-1.5, 0.0, 0.8), sample.accelG)
        assertEquals(listOf(12.5, -25.0, 0.0), sample.gyroDps)
        assertEquals(-10.25, sample.pitchDeg, 0.0)
        assertEquals(20.5, sample.rollDeg, 0.0)
        assertEquals(listOf(0.0, 0.0, 1.0), sample.gravityUnit)
        assertTrue(record.issues.isEmpty())
    }

    @Test fun retainsBootConfigAndReadFailureRecordsWithoutCountingThemAsSamples() {
        val boot = "{\"type\":\"imu_boot\",\"ready\":true,\"imu_version\":\"atom-bmi270-v1\"}\r\n".toByteArray()
        val config = "{\"type\":\"imu_config\",\"sample_hz\":100}\n".toByteArray()
        val status = "{\"type\":\"imu_status\",\"ready\":false,\"read_errors\":1}\n".toByteArray()
        val failed = bytes(frame().put("valid", false))
        val raw = boot + config + status + failed + bytes(frame())
        val records = ImuCaptureDecoder().feed(raw, 300)
        assertEquals(5, records.size)
        assertEquals(listOf("imu_boot", "imu_config", "imu_status", "imu_frame", "imu_frame"), records.map { it.kind })
        assertTrue(records.take(4).all { it.frame == null })
        assertEquals(listOf("invalid_imu_read"), records[3].issues)
        assertNotNull(records[4].frame)
        assertArrayEquals(raw, records.flatMap { it.rawLine.asIterable() }.toByteArray())
    }

    @Test fun reportsSampleGapsDuplicatesAndClockDiscontinuityWithoutDiscardingMeasurements() {
        val decoder = ImuCaptureDecoder()
        decoder.feed(bytes(frame(10, 1000)), 1)
        val gap = decoder.feed(bytes(frame(13, 1300).put("sample_gap", true)), 2).single()
        assertEquals(listOf("sequence_gap:2", "device_sample_gap"), gap.issues)
        val duplicate = decoder.feed(bytes(frame(13, 1300)), 3).single()
        assertEquals(listOf("duplicate_sequence", "device_clock_discontinuity"), duplicate.issues)
        assertNotNull(duplicate.frame)
        val reset = decoder.feed(bytes(frame(1, 200)), 4).single()
        assertTrue(reset.issues.contains("sequence_reset_or_reorder"))
        assertTrue(reset.issues.contains("device_clock_discontinuity"))
    }

    @Test fun sequenceWrapAndCachedBootDoNotResetContinuityButPhysicalBootChangeDoes() {
        val decoder = ImuCaptureDecoder()
        decoder.feed(bytes(frame(0xffffffffL, 1000)), 1)
        decoder.feed("{\"type\":\"imu_boot\",\"boot_id\":\"boot-a\"}\n".toByteArray(), 2)
        assertTrue(decoder.feed(bytes(frame(0, 2000)), 3).single().issues.isEmpty())
        val next = decoder.feed(bytes(frame(2, 3000)), 4).single()
        assertEquals(listOf("sequence_gap:1"), next.issues)
        val reboot = decoder.feed(bytes(frame(1, 10).put("boot_id", "boot-b")), 5).single()
        assertEquals(listOf("boot_id_changed"), reboot.issues)
    }

    @Test fun invalidSchemaAxesCoercedIntegersAndMissingPoseAreRetainedAsRawOnly() {
        listOf<(JSONObject) -> Unit>(
            { it.put("schema_version", 2) },
            { it.put("seq", "1") },
            { it.put("device_us", -1) },
            { it.put("accel_raw", JSONArray(listOf(1, 2))) },
            { it.getJSONArray("gyro_raw").put(0, 32768) },
            { it.getJSONArray("accel_g").put(0, "0.2") },
            { it.put("pitch_deg", JSONObject.NULL) },
            { it.put("valid", "true") },
            { it.put("gravity_unit", JSONArray(listOf(0.0, 1.0))) },
        ).forEach { mutation ->
            val raw = bytes(frame().also(mutation))
            val record = ImuCaptureDecoder().feed(raw, 1).single()
            assertNull(record.frame)
            assertTrue(record.issues.single().startsWith("invalid_packet:"))
            assertArrayEquals(raw, record.rawLine)
        }
    }

    @Test fun preservesMalformedUtf8AndTruncatedPacketWithoutPromotingTailToFrame() {
        val decoder = ImuCaptureDecoder()
        val invalid = byteArrayOf(0xff.toByte(), 0x0a)
        val record = decoder.feed(invalid, 100).single()
        assertArrayEquals(invalid, record.rawLine)
        assertNull(record.frame)
        assertTrue(record.issues.isNotEmpty())
        val tail = bytes(frame()).dropLast(1).toByteArray()
        assertTrue(decoder.feed(tail, 200).isEmpty())
        val partial = decoder.finish()!!
        assertArrayEquals(tail, partial.rawLine)
        assertNull(partial.frame)
        assertEquals(200L, partial.hostReceivedMonotonicNs)
        assertEquals(listOf("truncated_line"), partial.issues)
        assertNull(decoder.finish())
    }

    @Test fun earlierSchemaOneFrameWithoutAdditiveGravityFieldStillDecodes() {
        val obj = frame().apply { remove("gravity_unit") }
        val record = ImuCaptureDecoder().feed(bytes(obj), 1).single()
        assertNotNull(record.frame)
        assertNull(record.frame!!.gravityUnit)
        assertTrue(record.issues.isEmpty())
    }

    @Test fun overlongInputKeepsEveryByteAndRecoversOnNextPacket() {
        val decoder = ImuCaptureDecoder(16)
        val oversized = ("x".repeat(35) + "\n").toByteArray()
        val event = "{\"type\":\"ok\"}\n".toByteArray()
        val records = decoder.feed(oversized + event, 1)
        assertArrayEquals(oversized + event, records.flatMap { it.rawLine.asIterable() }.toByteArray())
        assertEquals("ok", records.last().kind)
        assertTrue(records.last().issues.isEmpty())
        assertTrue(records.dropLast(1).all { it.frame == null })
    }

    companion object {
        fun frame(seq: Long = 1, deviceUs: Long = 200): JSONObject = JSONObject().apply {
            put("type", "imu_frame")
            put("schema_version", 1)
            put("seq", seq)
            put("device_us", deviceUs)
            put("device_id", "atom-a")
            put("boot_id", "boot-a")
            put("accel_raw", JSONArray(listOf(-32768, 0, 32767)))
            put("gyro_raw", JSONArray(listOf(100, -200, 0)))
            put("accel_g", JSONArray(listOf(-1.5, 0.0, 0.8)))
            put("gyro_dps", JSONArray(listOf(12.5, -25.0, 0.0)))
            put("pitch_deg", -10.25)
            put("roll_deg", 20.5)
            put("gravity_unit", JSONArray(listOf(0.0, 0.0, 1.0)))
            put("valid", true)
            put("accel_correction", true)
            put("dt_us", 10_000)
            put("read_us", 200)
            put("sample_gap", false)
        }

        fun bytes(obj: JSONObject): ByteArray = (obj.toString() + "\n").toByteArray(Charsets.UTF_8)
    }
}
