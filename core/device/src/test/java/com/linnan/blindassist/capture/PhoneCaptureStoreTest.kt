package com.linnan.blindassist.capture

import org.json.JSONObject
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.io.ByteArrayOutputStream
import java.io.File
import java.security.MessageDigest
import java.util.Random
import java.util.zip.ZipEntry
import java.util.zip.ZipInputStream

class PhoneCaptureStoreTest {
    @get:Rule val temporary = TemporaryFolder()

    @Test
    fun atomJpegHeadersAndUnmappedTimeRemainExactThroughStorageAndZip() {
        val root = temporary.newFolder("atom-captures")
        val store = PhoneCaptureStore.create(root, sourceMetadata = mapOf("rgb_source" to "atom_mjpeg_camera"))
        val jpeg = byteArrayOf(0xff.toByte(), 0xd8.toByte(), 12, 0, 0xff.toByte(), 0xd9.toByte())
        val headers = mapOf("x-sequence-id" to "atom-boot", "x-frame-sequence" to "17",
            "x-capture-timestamp-us" to "123456", "x-jpeg-ready-timestamp-us" to "123999",
            "content-length" to jpeg.size.toString())
        val rgbPath = store.recordRgb(jpeg, 123456000, 900000000, 640, 480, 90,
            "ATOM_DEVICE_MONOTONIC_UNMAPPED", 17, "atom-mjpeg:http://192.168.5.9:81/stream:boot:atom-boot",
            sourceHeaders = headers, displayMirrorHorizontal = true)
        store.recordRgb(jpeg, null, 900001000, 640, 480, 90,
            "ATOM_DEVICE_MONOTONIC_UNMAPPED", sourceId = "atom-mjpeg:missing-device-header")
        val saved = store.finish()
        val indexes = File(saved.directory, "rgb.jsonl").readLines().map(::JSONObject)
        val row = indexes.first()
        assertEquals("atom-boot", row.getString("boot_id"))
        assertEquals("123456", row.getJSONObject("source_headers").getString("x-capture-timestamp-us"))
        assertEquals(123456L, row.getLong("device_capture_timestamp_us"))
        assertEquals(123999L, row.getLong("device_jpeg_ready_timestamp_us"))
        assertEquals(123456000L, row.getLong("camera_timestamp_ns"))
        assertEquals(900000000L, row.getLong("received_elapsed_ns"))
        assertEquals(jpeg.size, row.getInt("bytes"))
        assertEquals(MessageDigest.getInstance("SHA-256").digest(jpeg).joinToString("") {
            "%02x".format(it.toInt() and 255)
        }, row.getString("sha256"))
        assertTrue(row.getBoolean("display_mirror_horizontal"))
        assertEquals(90, row.getInt("rotation_degrees"))
        assertTrue(indexes.last().isNull("camera_timestamp_ns"))
        assertTrue(indexes.last().isNull("device_capture_timestamp_us"))
        val output = ByteArrayOutputStream()
        PhoneCaptureStore.exportZip(saved, output)
        val files = mutableMapOf<String, ByteArray>()
        ZipInputStream(output.toByteArray().inputStream()).use { zip ->
            while (true) {
                val entry = zip.nextEntry ?: break
                files[entry.name] = zip.readBytes()
            }
        }
        assertArrayEquals(jpeg, files["${saved.id}/$rgbPath"])
        assertArrayEquals(File(saved.directory, "rgb.jsonl").readBytes(), files["${saved.id}/rgb.jsonl"])
    }

    @Test
    fun writesSeparateRawStreamsAndClockMetadataWithoutChangingPayloads() {
        val root = temporary.newFolder("captures")
        val store = PhoneCaptureStore.create(root, "走廊\n未标注", mapOf("tof" to "usb"), startedElapsedNs = 1_000)
        val jpeg = byteArrayOf(0xff.toByte(), 0xd8.toByte(), 1, 2, 0xff.toByte(), 0xd9.toByte())
        val rgbPath = store.recordRgb(jpeg, 111, 222, 640, 480, 90,
            sourceFrameId = 12, sourceId = "rear", intrinsics = mapOf("fx" to 500.0))
        val raw = "{\"seq\":5,\"cn_h\":[1, 2]}\r\n"
        store.recordTof(raw, 333, 5, 444)
        store.recordEvent("usb_disconnected", "线缆断开", 555)
        val saved = store.finish()

        assertArrayEquals(jpeg, File(saved.directory, rgbPath).readBytes())
        assertEquals(raw, File(saved.directory, "tof.raw.jsonl").readText())
        val rgb = JSONObject(File(saved.directory, "rgb.jsonl").readText())
        assertEquals(111L, rgb.getLong("camera_timestamp_ns"))
        assertEquals(222L, rgb.getLong("received_elapsed_ns"))
        assertEquals(90, rgb.getInt("rotation_degrees"))
        assertEquals("CAMERA_HARDWARE_UNMAPPED", rgb.getString("camera_clock_domain"))
        assertEquals(500.0, rgb.getJSONObject("camera_intrinsics").getDouble("fx"), 0.0)
        val tof = JSONObject(File(saved.directory, "tof.index.jsonl").readText())
        assertEquals(333L, tof.getLong("received_elapsed_ns"))
        assertEquals(444L, tof.getLong("device_timestamp_ms"))
        assertEquals(raw.toByteArray().size, tof.getInt("raw_byte_length"))
        assertEquals(1L, saved.rgbCount)
        assertEquals(1L, saved.tofCount)
        assertEquals("COMPLETED", saved.status)
        val manifest = JSONObject(File(saved.directory, "manifest.json").readText())
        assertTrue(manifest.isNull("labels"))
        assertTrue(manifest.getString("synchronization").contains("no hardware clock mapping"))
        assertEquals("走廊\n未标注", manifest.getString("scene_note"))
        assertTrue(JSONObject(File(saved.directory, "events.jsonl").readText()).getString("kind") == "usb_disconnected")
    }

    @Test
    fun interruptedSessionRemainsDiscoverableAndCountsOnlyCompleteIndexRows() {
        val root = temporary.newFolder("captures")
        val store = PhoneCaptureStore.create(root)
        store.recordRgb(byteArrayOf(1), 1, 2, 1, 1, 0)
        store.recordTof("{\"seq\":1}", 3)
        File(store.directory, "rgb.jsonl").appendText("{\"index\":2")

        val recovered = PhoneCaptureStore.listSessions(root).single()
        assertEquals(store.id, recovered.id)
        assertTrue(recovered.isInterrupted)
        assertEquals("RECORDING", recovered.status)
        assertEquals(1L, recovered.rgbCount)
        assertEquals(1L, recovered.tofCount)
        assertEquals("{\"seq\":1}\n", File(store.directory, "tof.raw.jsonl").readText())
        store.close()
    }

    @Test
    fun writeFailureHaltsRecordingAndPreservesUnindexedPayloadForExport() {
        val root = temporary.newFolder("captures")
        val store = PhoneCaptureStore.create(root)
        val index = File(store.directory, "tof.index.jsonl")
        assertTrue(index.delete())
        assertTrue(index.mkdir())

        var failed = false
        try {
            store.recordTof("{\"seq\":1}", 3)
        } catch (_: Exception) {
            failed = true
        }
        assertTrue(failed)
        assertEquals(0L, store.tofCount)
        assertEquals("{\"seq\":1}\n", File(store.directory, "tof.raw.jsonl").readText())
        assertEquals("FAILED", PhoneCaptureStore.listSessions(root).single().status)
        try {
            store.recordTof("{\"seq\":2}", 4)
            throw AssertionError("Failed storage must not accept more records")
        } catch (_: IllegalStateException) {
            // Explicit failure, instead of silently losing or acknowledging data.
        }
    }

    @Test
    fun rawByteRecordsRemainExactWhileBootAndMalformedPacketsDoNotCountAsFrames() {
        val root = temporary.newFolder("captures")
        val store = PhoneCaptureStore.create(root)
        val malformed = byteArrayOf(0xff.toByte(), 0x00, 0x7b)
        val boot = "{\"type\":\"boot\"}\r\n".toByteArray()
        val frame = "{\"seq\":3}\n".toByteArray()
        store.recordTof(malformed, 1, validFrame = false)
        store.recordTof(boot, 2, validFrame = false)
        store.recordTof(frame, 3, 3, 100, validFrame = true)
        val saved = store.finish()

        assertArrayEquals(malformed + boot + frame, File(saved.directory, "tof.raw.jsonl").readBytes())
        assertEquals(1L, saved.tofCount)
        assertEquals(3L, saved.tofRecordCount)
        val indexes = File(saved.directory, "tof.index.jsonl").readLines().map(::JSONObject)
        assertEquals(0L, indexes[0].getLong("raw_byte_offset"))
        assertEquals(malformed.size.toLong(), indexes[1].getLong("raw_byte_offset"))
        assertFalse(indexes[0].getBoolean("valid_frame"))
        assertTrue(indexes[2].getBoolean("valid_frame"))
    }

    @Test
    fun zipIncludesOriginalFilesAndLeavesCallerOutputOpen() {
        val root = temporary.newFolder("captures")
        val store = PhoneCaptureStore.create(root)
        val jpeg = byteArrayOf(1, 2, 3)
        val relativePath = store.recordRgb(jpeg, 1, 2, 3, 4, 0)
        store.recordTof("{\"seq\":9}\n", 3)
        val saved = store.finish()
        val output = object : ByteArrayOutputStream() {
            var closed = false
            override fun close() { closed = true; super.close() }
        }
        PhoneCaptureStore.exportZip(saved, output)
        assertFalse(output.closed)
        val files = mutableMapOf<String, ByteArray>()
        ZipInputStream(output.toByteArray().inputStream()).use { zip ->
            while (true) {
                val entry = zip.nextEntry ?: break
                files[entry.name] = zip.readBytes()
            }
        }
        assertArrayEquals(jpeg, files["${saved.id}/$relativePath"])
        assertEquals("{\"seq\":9}\n", files.getValue("${saved.id}/tof.raw.jsonl").toString(Charsets.UTF_8))
        assertTrue(files.containsKey("${saved.id}/manifest.json"))
        assertTrue(files.containsKey("${saved.id}/rgb.jsonl"))
        assertTrue(files.containsKey("${saved.id}/tof.index.jsonl"))
    }

    @Test
    fun exportUsesProviderBulkWritesAndReportsExactSourceBytesAfterFlush() {
        val store = PhoneCaptureStore.create(temporary.newFolder("bulk-export"))
        val jpeg = ByteArray(1024 * 1024).also { Random(42L).nextBytes(it) }
        val rgbPath = store.recordRgb(jpeg, 1, 2, 640, 480, 0)
        // Text still benefits from compression while the JPEG stays streaming.
        val raw = ("{\"padding\":\"" + "0".repeat(16_384) + "\"}\n").toByteArray()
        store.recordTof(raw, 3)
        val saved = store.finish()
        val expectedBytes = saved.directory.walkTopDown().filter { it.isFile }.sumOf { it.length() }
        val output = object : ByteArrayOutputStream() {
            var singleWrites = 0
            var bulkWrites = 0
            var flushed = false
            var closed = false
            override fun write(value: Int) { singleWrites++; super.write(value) }
            override fun write(bytes: ByteArray, offset: Int, length: Int) {
                bulkWrites++
                flushed = false
                super.write(bytes, offset, length)
            }
            override fun flush() { flushed = true }
            override fun close() { closed = true }
        }
        val progress = mutableListOf<Pair<Long, Long>>()
        var lastProgressFollowedFlush = false
        PhoneCaptureStore.exportZip(saved, output) { written, total ->
            progress += written to total
            lastProgressFollowedFlush = output.flushed
        }
        assertEquals(0, output.singleWrites)
        assertTrue("Provider writes must be blocks, not per ZIP header or byte", output.bulkWrites in 1..8)
        assertFalse(output.closed)
        assertTrue(lastProgressFollowedFlush)
        assertEquals(0L to expectedBytes, progress.first())
        assertEquals(expectedBytes to expectedBytes, progress.last())
        assertTrue(progress.zipWithNext().all { (before, after) -> after.first >= before.first })
        assertTrue(progress.all { it.second == expectedBytes })

        ZipInputStream(output.toByteArray().inputStream()).use { zip ->
            var sawJpeg = false
            var sawRaw = false
            while (true) {
                val entry = zip.nextEntry ?: break
                val restored = zip.readBytes()
                if (entry.name == "${saved.id}/$rgbPath") {
                    sawJpeg = true
                    assertEquals(ZipEntry.DEFLATED, entry.method)
                    assertArrayEquals(jpeg, restored)
                    assertTrue(entry.compressedSize >= jpeg.size.toLong())
                    assertTrue(entry.compressedSize < jpeg.size + 1024L)
                } else if (entry.name == "${saved.id}/tof.raw.jsonl") {
                    sawRaw = true
                    assertArrayEquals(raw, restored)
                    assertTrue(entry.compressedSize < raw.size / 4L)
                }
            }
            assertTrue(sawJpeg)
            assertTrue(sawRaw)
        }
    }

    @Test
    fun imuRawAndSixAxisIndexRoundTripThroughExportWithExplicitUncalibratedTofReference() {
        val root = temporary.newFolder("captures")
        val store = PhoneCaptureStore.create(root)
        val boot = "{\"type\":\"imu_boot\",\"ready\":true}\r\n".toByteArray()
        val invalid = byteArrayOf(0xff.toByte(), 0x0a)
        val raw = ImuCaptureDecoderTest.bytes(ImuCaptureDecoderTest.frame(8, 500_000))
        val frame = ImuCaptureDecoder().feed(raw, 100_000_000).single().frame!!
        store.recordImu(boot, 10, validFrame = false)
        store.recordImu(invalid, 20, validFrame = false)
        val recordIndex = store.recordImu(raw, 100_000_000, 8, 500_000, true, frame)
        store.recordTof("{\"seq\":9}\n", 110_000_000, imuAlignment = mapOf(
            "status" to "MATCHED_RECEIPT_ONLY",
            "age_ns" to 10_000_000L,
            "imu_minus_tof_ns" to -10_000_000L,
            "raw_reference" to mapOf("index_file" to "imu.index.jsonl", "record_index" to recordIndex,
                "raw_file" to "imu.raw.jsonl"),
        ))
        val saved = store.finish()
        assertEquals(1L, saved.imuCount)
        assertEquals(3L, saved.imuRecordCount)
        val output = ByteArrayOutputStream()
        PhoneCaptureStore.exportZip(saved, output)
        val files = mutableMapOf<String, ByteArray>()
        ZipInputStream(output.toByteArray().inputStream()).use { zip ->
            while (true) {
                val entry = zip.nextEntry ?: break
                files[entry.name] = zip.readBytes()
            }
        }
        assertArrayEquals(boot + invalid + raw, files.getValue("${saved.id}/imu.raw.jsonl"))
        val indexes = files.getValue("${saved.id}/imu.index.jsonl").toString(Charsets.UTF_8)
            .trimEnd().lines().map(::JSONObject)
        val sample = indexes.last()
        assertEquals(100_000_000L, sample.getLong("received_elapsed_ns"))
        assertEquals(500_000L, sample.getLong("device_timestamp_us"))
        assertEquals(listOf(-32768, 0, 32767), (0..2).map { sample.getJSONArray("accel_raw").getInt(it) })
        assertEquals(-25.0, sample.getJSONArray("gyro_dps").getDouble(1), 0.0)
        assertEquals(-10.25, sample.getDouble("pitch_deg"), 0.0)
        assertEquals(1.0, sample.getJSONArray("gravity_unit").getDouble(2), 0.0)
        assertEquals("BMI270_NATIVE_NOT_TOF_OPTICAL", sample.getString("pose_frame"))
        assertEquals((boot.size + invalid.size).toLong(), sample.getLong("raw_byte_offset"))
        assertFalse(indexes.first().getBoolean("valid_frame"))
        val tof = JSONObject(files.getValue("${saved.id}/tof.index.jsonl").toString(Charsets.UTF_8))
        assertEquals(recordIndex, tof.getJSONObject("imu_alignment").getJSONObject("raw_reference").getLong("record_index"))
        assertEquals(-10_000_000L, tof.getJSONObject("imu_alignment").getLong("imu_minus_tof_ns"))
        val manifest = JSONObject(files.getValue("${saved.id}/manifest.json").toString(Charsets.UTF_8))
        assertEquals("blindassist-phone-capture-v1", manifest.getString("schema"))
        assertEquals(1L, manifest.getLong("imu_count"))
        assertEquals(3L, manifest.getLong("imu_record_count"))
    }

    @Test
    fun earlierSessionsWithoutImuFilesRemainDiscoverableWithZeroCounts() {
        val root = temporary.newFolder("captures")
        val store = PhoneCaptureStore.create(root)
        val saved = store.finish()
        assertTrue(File(saved.directory, "imu.raw.jsonl").delete())
        assertTrue(File(saved.directory, "imu.index.jsonl").delete())
        val manifestFile = File(saved.directory, "manifest.json")
        val manifest = JSONObject(manifestFile.readText()).apply {
            remove("imu_count")
            remove("imu_record_count")
        }
        manifestFile.writeText(manifest.toString())
        val previous = PhoneCaptureStore.listSessions(root).single()
        assertEquals(saved.id, previous.id)
        assertEquals("COMPLETED", previous.status)
        assertEquals(0L, previous.imuCount)
        assertEquals(0L, previous.imuRecordCount)
    }

    @Test
    fun interruptedImuIndexIgnoresPartialRowWhileKeepingPacketBytesAndUnmatchedTofExplicit() {
        val root = temporary.newFolder("captures")
        val store = PhoneCaptureStore.create(root)
        val raw = ImuCaptureDecoderTest.bytes(ImuCaptureDecoderTest.frame())
        store.recordImu(raw, 100)
        File(store.directory, "imu.index.jsonl").appendText("{\"index\":2")
        store.recordTof("{\"seq\":1}\n", 200)
        val interrupted = PhoneCaptureStore.listSessions(root).single()
        assertEquals(1L, interrupted.imuCount)
        assertEquals(1L, interrupted.imuRecordCount)
        assertArrayEquals(raw, File(store.directory, "imu.raw.jsonl").readBytes())
        val tof = JSONObject(File(store.directory, "tof.index.jsonl").readText())
        assertEquals("UNAVAILABLE", tof.getJSONObject("imu_alignment").getString("status"))
        assertTrue(tof.getJSONObject("imu_alignment").isNull("raw_reference"))
        assertTrue(tof.getJSONObject("imu_alignment").isNull("imu_minus_tof_ns"))
        store.close()
    }

    @Test
    fun acknowledgedImuRowsRemainReadableWhileOpenAndAllHandlesCloseAtFinish() {
        val root = temporary.newFolder("persistent-streams")
        val store = PhoneCaptureStore.create(root)
        val records = (1L..12L).map { sequence ->
            ImuCaptureDecoderTest.bytes(ImuCaptureDecoderTest.frame(sequence, sequence * 10_000L))
        }
        records.forEachIndexed { index, raw ->
            assertEquals(index + 1L, store.recordImu(raw, index + 100L))
            val rows = File(store.directory, "imu.index.jsonl").readLines().map(::JSONObject)
            assertEquals(index + 1, rows.size)
            assertEquals(index + 1L, rows.last().getLong("index"))
            assertEquals(records.take(index).sumOf { it.size }.toLong(), rows.last().getLong("raw_byte_offset"))
        }
        assertArrayEquals(records.reduce { accumulated, raw -> accumulated + raw },
            File(store.directory, "imu.raw.jsonl").readBytes())
        val saved = store.finish()
        assertEquals(12L, saved.imuCount)
        assertEquals(12L, PhoneCaptureStore.listSessions(root).single().imuCount)
        val rawFile = File(store.directory, "imu.raw.jsonl")
        assertTrue(rawFile.renameTo(File(store.directory, "imu.raw.renamed")))
        store.finish() // Finalization is idempotent and does not reopen the append descriptors.
        assertTrue(File(store.directory, "imu.raw.renamed").renameTo(rawFile))
    }

    @Test
    fun damagedFinalManifestCountersFallBackToCompleteIndexRows() {
        val root = temporary.newFolder("damaged-final-counters")
        val store = PhoneCaptureStore.create(root)
        store.recordTof("{\"seq\":1}\n", 100)
        store.recordImu(ImuCaptureDecoderTest.bytes(ImuCaptureDecoderTest.frame()), 200)
        val saved = store.finish()
        val manifestFile = File(store.directory, "manifest.json")
        manifestFile.writeText(JSONObject(manifestFile.readText()).apply {
            put("tof_record_count", -1)
            put("imu_record_count", "damaged")
        }.toString())
        val recovered = PhoneCaptureStore.listSessions(root).single()
        assertEquals(saved.id, recovered.id)
        assertEquals(1L, recovered.tofCount)
        assertEquals(1L, recovered.tofRecordCount)
        assertEquals(1L, recovered.imuCount)
        assertEquals(1L, recovered.imuRecordCount)
    }
}
