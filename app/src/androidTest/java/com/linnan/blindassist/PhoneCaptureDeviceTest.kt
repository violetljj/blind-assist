package com.linnan.blindassist

import android.graphics.BitmapFactory
import android.os.SystemClock
import androidx.compose.ui.semantics.SemanticsProperties
import androidx.compose.ui.state.ToggleableState
import androidx.compose.ui.test.assertIsDisplayed
import androidx.compose.ui.test.assertIsEnabled
import androidx.compose.ui.test.assertIsNotEnabled
import androidx.compose.ui.test.junit4.createAndroidComposeRule
import androidx.compose.ui.test.onAllNodesWithTag
import androidx.compose.ui.test.onNodeWithTag
import androidx.compose.ui.test.performClick
import androidx.compose.ui.test.performScrollTo
import androidx.compose.ui.test.performTextReplacement
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.linnan.blindassist.capture.PhoneCaptureStore
import com.linnan.blindassist.capture.PhoneCaptureSummary
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.io.FilterOutputStream
import java.io.OutputStream
import java.security.MessageDigest
import java.util.zip.ZipEntry
import java.util.zip.ZipFile
import java.util.zip.ZipOutputStream

/**
 * Real device acquisition. H3 and optional Atom IMU each require an explicit runner opt-in.
 * phoneCaptureDurationSeconds defaults to 5; phoneCaptureSessions defaults to 1.
 * phoneCaptureExportBenchmark opts into the saved-session export test without live sensors.
 */
@RunWith(AndroidJUnit4::class)
class PhoneCaptureDeviceTest {
    @get:Rule
    val compose = createAndroidComposeRule<PhoneCaptureActivity>()

    @Test
    fun capturePageShowsThreePreviewPanels() {
        compose.onNodeWithTag("capture_preview").performScrollTo().assertIsDisplayed()
        compose.onNodeWithTag("capture_tof_preview").performScrollTo().assertIsDisplayed()
        compose.onNodeWithTag("capture_imu_preview").performScrollTo().assertIsDisplayed()
        compose.onNodeWithTag("capture_camera_permission").assertDoesNotExist()
        compose.onNodeWithTag("capture_connect").performScrollTo().assertIsEnabled()
        compose.onNodeWithTag("capture_start").performScrollTo().assertIsNotEnabled()
        compose.onNodeWithTag("capture_stop").assertDoesNotExist()
    }

    @Test
    fun captureTimingCountsSamplesSharingOneReceipt() {
        val frames = listOf(10_000_000L, 10_000_000L, 30_000_000L).mapIndexed { index, receipt ->
            JSONObject().put("sequence", index + 1L).put("device_us", 100_000L + index * 10_000L)
                .put("received_elapsed_ns", receipt)
        }
        val result = streamTiming(frames, "device_us", 1_000L, "sequence")
            .getJSONArray("by_boot").getJSONObject(0)
        assertEquals(100.0, result.getDouble("observed_source_hz"), 1e-9)
        assertEquals(100.0, result.getDouble("observed_receipt_hz"), 1e-9)
        assertEquals(1, result.getInt("receipt_shared_timestamp_count"))
        assertEquals(0, result.getInt("receipt_clock_decrease_count"))
    }

    @Test
    fun savedSessionExportBenchmarkWithoutSensors() {
        assumeTrue("Opt in to benchmark preserved real sessions without live sensors",
            InstrumentationRegistry.getArguments().getString("phoneCaptureExportBenchmark") == "true")
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val session = PhoneCaptureStore.listSessions(context)
            .filter { it.status == "COMPLETED" && it.rgbCount > 0 && it.tofCount > 0 }
            .maxByOrNull { it.bytes }
        assumeTrue("A completed RGB + ToF session must already exist on this phone", session != null)
        try {
            val saved = requireNotNull(session)
            val manifest = JSONObject(File(saved.directory, "manifest.json").readText())
            val evidence = File(context.cacheDir, "phone-capture-export-test").apply { mkdirs() }
            val receipt = benchmarkExport(saved, evidence)
                .put("schema", "blindassist-phone-capture-export-test-v1")
                .put("session_id", saved.id)
                .put("session_directory", saved.directory.absolutePath)
                .put("rgb_count", manifest.getLong("rgb_count"))
                .put("tof_count", manifest.getLong("tof_count"))
                .put("imu_count", manifest.optLong("imu_count", 0))
                .put("session_status", manifest.getString("status"))
                .put("rgb_source", manifest.getJSONObject("sources").optString("rgb_source"))
                .put("boundary", "Preserved completed capture on this phone; no new sensor acquisition; app-private output, no SAF provider measurement")
            File(evidence, "${saved.id}-receipt.json").writeText(receipt.toString(2))
        } finally {
            compose.activityRule.scenario.close()
        }
    }

    @Test
    fun liveH3CapturePersistsRawCnhAndExportsZip() {
        val args = InstrumentationRegistry.getArguments()
        assumeTrue("Opt in only with the actual H3 on the phone LAN and no other TCP consumer",
            args.getString("phoneCaptureLive") == "true")
        val host = args.getString("phoneCaptureTofHost") ?: "192.168.5.10"
        val imuEnabled = args.getString("phoneCaptureImu") == "true"
        val atomHost = args.getString("phoneCaptureAtomHost") ?: args.getString("phoneCaptureImuHost") ?: "192.168.5.9"
        val discovery = args.getString("phoneCaptureDiscovery") == "true"
        val measurePerformance = args.getString("phoneCapturePerformance") == "true"
        val recordingSeconds = args.getString("phoneCaptureDurationSeconds")?.toLong() ?: 5L
        val recordingSessions = args.getString("phoneCaptureSessions")?.toInt() ?: 1
        require(recordingSeconds in 5L..300L) { "phoneCaptureDurationSeconds must be 5..300" }
        require(recordingSessions in 1..3) { "phoneCaptureSessions must be 1..3" }
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val existing = PhoneCaptureStore.listSessions(context).map { it.id }.toMutableSet()
        try {
            // The product defaults IMU on; the existing two-stream test remains explicitly opt-out.
            compose.onNodeWithTag("capture_settings_toggle").performScrollTo().performClick()
            val imuToggle = compose.onNodeWithTag("capture_imu_enabled").performScrollTo()
            val checked = imuToggle.fetchSemanticsNode().config[SemanticsProperties.ToggleableState] ==
                ToggleableState.On
            if (checked != imuEnabled) imuToggle.performClick()
            compose.onNodeWithTag("capture_host").performScrollTo().performTextReplacement(host)
            compose.onNodeWithTag("capture_atom_host").performScrollTo().performTextReplacement(atomHost)
            compose.onNodeWithTag("capture_note").performScrollTo().performTextReplacement(
                "Device integration test: real Atom RGB + real H3" +
                    (if (imuEnabled) " + real Atom IMU" else "") + ", unlabeled")
            compose.onNodeWithTag("capture_settings_toggle").performScrollTo().performClick()
            val firstConnectMs = SystemClock.elapsedRealtime()
            compose.onNodeWithTag(if (discovery) "capture_discover" else "capture_connect")
                .performScrollTo().performClick()
            waitForStreams(imuEnabled)
            val firstReadyMs = SystemClock.elapsedRealtime() - firstConnectMs
            assertLivePreviews(imuEnabled)
            // Recreating closes all previous sockets, then real previews must reconnect and update.
            compose.activityRule.scenario.recreate()
            compose.onNodeWithTag("capture_camera_permission").assertDoesNotExist()
            val reconnectMs = SystemClock.elapsedRealtime()
            compose.onNodeWithTag(if (discovery) "capture_discover" else "capture_connect").performClick()
            waitForStreams(imuEnabled)
            val recreatedReadyMs = SystemClock.elapsedRealtime() - reconnectMs
            assertLivePreviews(imuEnabled)
            val idlePerformance = if (measurePerformance) {
                compose.onNodeWithTag("capture_preview").performScrollTo()
                compose.activityRule.scenario.onActivity { it.resetPreviewPerformance() }
                val previewStartMs = SystemClock.elapsedRealtime()
                // waitUntil advances the Compose test frame clock while external I/O uses real time.
                // A raw sleep pauses test-managed frames and cannot measure actual preview drawing.
                compose.waitUntil(13_000) { SystemClock.elapsedRealtime() - previewStartMs >= 12_000 }
                previewPerformance().also {
                    val performanceEvidence = File(context.cacheDir, "phone-capture-performance-test").apply { mkdirs() }
                    File(performanceEvidence, "preview-${SystemClock.elapsedRealtime()}.json").writeText(it.toString(2))
                    assertTrue("Real Atom image must be drawn: $it", it.getLong("drawn_unique_frames") > 0)
                }
            } else JSONObject.NULL
            compose.onNodeWithTag("capture_settings_toggle").performScrollTo().performClick()
            val connectedHost = editableText("capture_host")
            val connectedAtomHost = editableText("capture_atom_host")
            compose.onNodeWithTag("capture_settings_toggle").performScrollTo().performClick()
            repeat(recordingSessions) { sessionNumber ->
                compose.onNodeWithTag("capture_start").performScrollTo().assertIsEnabled().performClick()
                // The start control can scroll the image offscreen; measure visible drawing.
                compose.onNodeWithTag("capture_preview").performScrollTo()
                if (measurePerformance) compose.activityRule.scenario.onActivity { it.resetPreviewPerformance() }
                val startMs = SystemClock.elapsedRealtime()
                compose.waitUntil(recordingSeconds * 1_000 + 7_000) {
                    val counts = counts()
                    SystemClock.elapsedRealtime() - startMs >= recordingSeconds * 1_000 && counts.first >= 5 && counts.second >= 3 &&
                        (!imuEnabled || counts.third >= 10)
                }
                val recordingPerformance = if (measurePerformance) previewPerformance().also {
                    assertTrue("Real Atom image must be drawn during recording: $it", it.getLong("drawn_unique_frames") > 0)
                } else JSONObject.NULL
                val stopStartMs = SystemClock.elapsedRealtime()
                compose.onNodeWithTag("capture_stop").performScrollTo().performClick()
                compose.waitUntil(10_000) {
                    compose.onAllNodesWithTag("capture_stop").fetchSemanticsNodes().isEmpty()
                }
                val stopCompletedMs = SystemClock.elapsedRealtime()
                val session = PhoneCaptureStore.listSessions(context).single { it.id !in existing }
                existing += session.id
                val directory = session.directory
                val manifest = JSONObject(File(directory, "manifest.json").readText())
                assertEquals("blindassist-phone-capture-v1", manifest.getString("schema"))
                assertEquals("COMPLETED", manifest.getString("status"))
                assertEquals("user_stop", manifest.getString("finish_reason"))
                assertEquals("android_elapsed_realtime_nanos", manifest.getString("receipt_clock"))
                assertTrue(manifest.getString("synchronization").contains("no hardware clock mapping"))
                assertTrue(manifest.isNull("labels"))
                assertTrue(manifest.getJSONObject("sources").getString("geometry").contains("NOT_CALIBRATED"))
                assertEquals("atom_mjpeg_camera", manifest.getJSONObject("sources").getString("rgb_source"))

                val rgb = rows(File(directory, "rgb.jsonl"))
                val tofIndex = rows(File(directory, "tof.index.jsonl"))
                val raw = File(directory, "tof.raw.jsonl").readBytes()
                assertTrue("Too few captured real RGB frames: ${rgb.size}", rgb.size >= 5)
                val validTof = tofIndex.filter { it.getBoolean("valid_frame") }
                assertTrue("Too few captured real H3 frames: ${validTof.size}", validTof.size >= 3)
                assertEquals(rgb.size.toLong(), manifest.getLong("rgb_count"))
                assertEquals(validTof.size.toLong(), manifest.getLong("tof_count"))
                assertEquals(tofIndex.size.toLong(), manifest.getLong("tof_record_count"))
                assertEquals(imuEnabled.toString(), manifest.getJSONObject("sources").getString("imu_enabled"))
                val imuReceipt = if (imuEnabled) verifyImuCapture(directory, manifest, validTof) else {
                    assertEquals(0L, manifest.getLong("imu_count"))
                    assertEquals(0L, manifest.getLong("imu_record_count"))
                    assertEquals(0L, File(directory, "imu.raw.jsonl").length())
                    assertEquals(0L, File(directory, "imu.index.jsonl").length())
                    validTof.forEach { assertEquals("DISABLED", it.getJSONObject("imu_alignment").getString("status")) }
                    JSONObject().put("enabled", false)
                }

                rgb.forEach { row ->
                    val headers = row.getJSONObject("source_headers")
                    assertFalse(headers.getString("x-sequence-id").isBlank())
                    assertEquals(headers.getString("x-sequence-id"), row.getString("boot_id"))
                    assertEquals(headers.getString("x-frame-sequence").toLong(), row.getLong("source_frame_id"))
                    val captureUs = headers.getString("x-capture-timestamp-us").toLong()
                    assertTrue(captureUs > 0)
                    assertEquals(captureUs, row.getLong("device_capture_timestamp_us"))
                    assertEquals(captureUs * 1000L, row.getLong("camera_timestamp_ns"))
                    assertTrue(row.getLong("received_elapsed_ns") > 0)
                    assertTrue(row.getString("camera_clock_domain").contains("ATOM"))
                    assertTrue(row.getString("source_id").startsWith("atom-mjpeg:http://"))
                    assertFalse(row.getString("boot_id").isBlank())
                    if (imuEnabled) assertTrue(imuReceipt.getJSONArray("boot_ids").let { boots ->
                        (0 until boots.length()).any { boots.getString(it) == row.getString("boot_id") }
                    })
                    val jpeg = File(directory, row.getString("file"))
                    assertEquals(row.getLong("bytes"), jpeg.length())
                    assertEquals(headers.getString("content-length").toLong(), jpeg.length())
                    assertEquals(row.getString("sha256"), sha256(jpeg.readBytes()))
                    assertEquals(90, row.getInt("rotation_degrees"))
                    assertTrue(row.getBoolean("display_mirror_horizontal"))
                    val dimensions = BitmapFactory.Options().apply { inJustDecodeBounds = true }
                    BitmapFactory.decodeFile(jpeg.path, dimensions)
                    assertEquals(row.getInt("width"), dimensions.outWidth)
                    assertEquals(row.getInt("height"), dimensions.outHeight)
                }
                val decodedTof = tofIndex.mapNotNull { index ->
                    assertTrue(index.getLong("received_elapsed_ns") > 0)
                    val offset = index.getInt("raw_byte_offset")
                    val length = index.getInt("raw_byte_length")
                    assertTrue(offset >= 0 && length > 0 && offset + length <= raw.size)
                    val text = raw.copyOfRange(offset, offset + length).toString(Charsets.UTF_8)
                    if (index.getBoolean("valid_frame")) {
                        val record = JSONObject(text)
                        assertEquals("cnh_frame", record.getString("type"))
                        assertEquals(8, record.getInt("rows"))
                        assertEquals(8, record.getInt("cols"))
                        assertEquals(16, record.getInt("bins"))
                        assertEquals(index.getLong("sequence"), record.getLong("seq"))
                        assertEquals(index.getLong("device_timestamp_ms"), record.getLong("ms"))
                        listOf("distance_mm", "target_status", "nb_target", "ambient_raw", "ambient_scaler")
                            .forEach { assertEquals(64, record.getJSONArray(it).length()) }
                        listOf("hist_raw", "hist_scaler").forEach { key ->
                            val matrix = record.getJSONArray(key)
                            assertEquals(64, matrix.length())
                            repeat(64) { assertEquals(16, matrix.getJSONArray(it).length()) }
                        }
                        assertTrue(record.has("read_us"))
                        assertTrue(record.has("cnh_header"))
                        record
                    } else {
                        // Interrupted tails and malformed bytes remain indexed/exported evidence.
                        // Only valid_frame records are required to contain a complete sensor JSON.
                        runCatching { JSONObject(text) }.getOrNull()
                    }
                }
                val boot = decodedTof.firstOrNull { it.optString("type") == "boot" }
                assertNotNull("Physical boot identity must be retained with the real capture", boot)
                assertEquals("xiao-cnh-h3-wifi-v1", requireNotNull(boot).getString("firmware"))
                assertFalse(boot.getString("boot_id").isBlank())

                val evidence = File(context.cacheDir, "phone-capture-device-test").apply { mkdirs() }
                val exportReceipt = benchmarkExport(session, evidence)
                File(evidence, "${session.id}-receipt.json").writeText(JSONObject()
                    .put("schema", "blindassist-phone-capture-device-test-v1")
                    .put("session_id", session.id)
                    .put("session_directory", directory.absolutePath)
                    .put("session_number", sessionNumber + 1)
                    .put("requested_session_count", recordingSessions)
                    .put("requested_recording_seconds", recordingSeconds)
                    .put("recording_observed_ms", stopStartMs - startMs)
                    .put("stop_completion_ms", stopCompletedMs - stopStartMs)
                    .put("export_zip", exportReceipt.getJSONObject("optimized").getString("zip_path"))
                    .put("export", exportReceipt)
                    .put("stream_timing", JSONObject()
                        .put("rgb", streamTiming(rgb, "device_capture_timestamp_us", 1_000L, "source_frame_id", "boot_id"))
                        .put("tof", streamTiming(validTof, "device_timestamp_ms", 1_000_000L, "sequence"))
                        .put("imu", if (imuEnabled) streamTiming(rows(File(directory, "imu.index.jsonl"))
                            .filter { it.getBoolean("valid_frame") }, "device_timestamp_us", 1_000L, "sequence", "boot_id")
                            else JSONObject.NULL))
                    .put("rgb_count", rgb.size)
                    .put("tof_count", validTof.size)
                    .put("tof_record_count", tofIndex.size)
                    .put("tof_host", connectedHost)
                    .put("atom_host", connectedAtomHost)
                    .put("imu_host", if (imuEnabled) connectedAtomHost else JSONObject.NULL)
                    .put("rgb_source", "atom_mjpeg_camera")
                    .put("rgb_boot_ids", JSONArray(rgb.map { it.getString("boot_id") }.distinct()))
                    .put("activity_recreation_preview_passed", true)
                    .put("first_connection_ready_ms", firstReadyMs)
                    .put("recreated_connection_ready_ms", recreatedReadyMs)
                    .put("preview_idle_performance", idlePerformance)
                    .put("preview_recording_performance", recordingPerformance)
                    .put("udp_discovery", discovery)
                    .put("imu", imuReceipt)
                    .put("boot_id", boot.getString("boot_id"))
                    .put("boundary", "Actual sensor acquisition and byte-preserving export; no labels, calibration or exposure synchronization")
                    .toString(2))
            }
        } finally {
            // Closing runs the product's foreground-stop path and preserves partial
            // recordings. Cleanup must also work after the UI hierarchy is gone.
            compose.activityRule.scenario.close()
        }
    }

    private fun waitForStreams(imuEnabled: Boolean) {
        try {
            compose.waitUntil(15_000) {
                tagText("capture_readiness").contains("RGB 已就绪") &&
                    tagText("capture_readiness").contains("ToF 已就绪") &&
                    compose.onAllNodesWithTag("capture_atom_image").fetchSemanticsNodes().isNotEmpty() &&
                    (!imuEnabled || tagText("capture_imu_readiness").contains("IMU 已就绪"))
            }
        } catch (failure: Throwable) {
            throw AssertionError("Live inputs did not become ready: RGB=${tagText("capture_camera_status")}; " +
                "ToF=${tagText("capture_status")}" + (if (imuEnabled) "; IMU=${tagText("capture_imu_status")}" else ""), failure)
        }
    }

    private fun assertLivePreviews(imuEnabled: Boolean) {
        // Bring the preview block into view: all three panels fit together.
        compose.onNodeWithTag("capture_preview").performScrollTo()
        compose.onNodeWithTag("capture_atom_image").assertIsDisplayed()
        compose.onNodeWithTag("capture_tof_grid").assertIsDisplayed()
        compose.onNodeWithTag("capture_imu_preview").assertIsDisplayed()
        assertFooterFits("capture_tof_no_reading_legend", "capture_tof_preview")
        assertFooterFits("capture_imu_native_axes", "capture_imu_preview")
        assertTrue(tagText("capture_rgb_freshness").contains("ms前"))
        assertTrue(tagText("capture_tof_freshness").contains("ms前"))
        if (imuEnabled) {
            assertTrue(tagText("capture_imu_freshness").contains("ms前"))
            assertFalse(tagText("capture_imu_angles").contains("—"))
            assertFalse(tagText("capture_imu_accel").contains("—"))
            assertFalse(tagText("capture_imu_gyro").contains("—"))
        }
    }

    private fun assertFooterFits(footer: String, panel: String) {
        compose.onNodeWithTag(footer).assertIsDisplayed()
        val footerNode = compose.onNodeWithTag(footer).fetchSemanticsNode()
        val panelNode = compose.onNodeWithTag(panel).fetchSemanticsNode()
        // boundsInRoot is already clipped; compare the full measured text size instead.
        val footerBottom = footerNode.positionInRoot.y + footerNode.size.height
        val panelBottom = panelNode.positionInRoot.y + panelNode.size.height
        assertTrue("$footer is clipped by $panel", footerBottom <= panelBottom + 1f)
    }

    private fun previewPerformance(): JSONObject {
        var receipt = JSONObject()
        compose.activityRule.scenario.onActivity { receipt = it.previewPerformanceReceipt() }
        return receipt
    }

    private fun benchmarkExport(session: PhoneCaptureSummary, evidence: File): JSONObject {
        val files = session.directory.walkTopDown().filter { it.isFile }.toList()
        val sourceBytes = files.sumOf { it.length() }
        val runId = SystemClock.elapsedRealtime()
        fun measure(name: String, export: (OutputStream) -> Unit): JSONObject {
            val zip = File(evidence, "${session.id}-$runId-$name.zip")
            val startedNs = SystemClock.elapsedRealtimeNanos()
            zip.outputStream().use(export)
            val elapsedNs = SystemClock.elapsedRealtimeNanos() - startedNs
            return JSONObject().put("zip_path", zip.absolutePath)
                .put("elapsed_ms", elapsedNs / 1e6).put("zip_bytes", zip.length())
                .put("source_mib_per_second", if (elapsedNs > 0)
                    sourceBytes * 1e9 / elapsedNs / 1_048_576.0 else JSONObject.NULL)
                .put("output_mib_per_second", if (elapsedNs > 0)
                    zip.length() * 1e9 / elapsedNs / 1_048_576.0 else JSONObject.NULL)
        }
        // The old byte-at-a-time exporter is a slow control, not part of an
        // integrity regression. Run it only for an explicitly requested benchmark.
        val legacy = if (InstrumentationRegistry.getArguments().getString("phoneCaptureLegacyExport") == "true")
            measure("legacy") { legacyExportZip(session, it) } else null
        val optimized = measure("optimized") { PhoneCaptureStore.exportZip(session, it) }
        ZipFile(File(optimized.getString("zip_path"))).use { archive ->
            assertEquals(files.size, archive.size())
            files.forEach { file ->
                val name = "${session.id}/${file.relativeTo(session.directory).invariantSeparatorsPath}"
                val entry = archive.getEntry(name)
                assertNotNull("Missing ZIP entry $name", entry)
                archive.getInputStream(requireNotNull(entry)).use {
                    assertArrayEquals("ZIP bytes differ for $name", file.readBytes(), it.readBytes())
                }
            }
        }
        optimized.put("byte_integrity_verified", true)
        return JSONObject().put("source_bytes", sourceBytes).put("file_count", files.size)
            .put("legacy", legacy ?: JSONObject.NULL).put("optimized", optimized)
            .put("elapsed_speedup", if (legacy != null && optimized.getDouble("elapsed_ms") > 0)
                legacy.getDouble("elapsed_ms") / optimized.getDouble("elapsed_ms") else JSONObject.NULL)
            .put("measurement_order", if (legacy != null) "legacy_once_then_optimized_once_on_same_phone_same_source_files" else "optimized_once")
            .put("boundary", "App-private cache file output including close; excludes ZIP verification and SAF provider; filesystem cache/order may affect timings")
    }

    /** Exact previous exporter retained only as an on-device performance control. */
    private fun legacyExportZip(session: PhoneCaptureSummary, output: OutputStream) {
        val directory = session.directory.canonicalFile
        require(directory.isDirectory)
        val prefix = directory.path + File.separator
        ZipOutputStream(object : FilterOutputStream(output) {
            override fun close() { flush() }
        }).use { zip ->
            directory.walkTopDown().filter { it.isFile }
                .sortedBy { it.relativeTo(directory).path }.forEach { file ->
                    require(file.canonicalPath.startsWith(prefix))
                    val name = "${session.id}/${file.relativeTo(directory).invariantSeparatorsPath}"
                    zip.putNextEntry(ZipEntry(name).apply { time = file.lastModified() })
                    file.inputStream().use { it.copyTo(zip) }
                    zip.closeEntry()
                }
        }
    }

    private fun streamTiming(
        frames: List<JSONObject>,
        sourceTimestampKey: String,
        sourceUnitNs: Long,
        sequenceKey: String,
        bootKey: String? = null,
    ): JSONObject {
        val groups = frames.groupBy { row -> bootKey?.let { row.optString(it, "unknown") } ?: "unavailable" }
        val perBoot = JSONArray()
        groups.forEach { (bootId, samples) ->
            val sourceIntervals = mutableListOf<Long>()
            val receiptIntervals = mutableListOf<Long>()
            var sequenceGaps = 0
            var missingSequences = 0L
            var sequenceResets = 0
            var sourceClockResets = 0
            var receiptClockResets = 0
            var receiptSharedTimestamps = 0
            samples.zipWithNext().forEach { (previous, current) ->
                val sequenceDelta = current.getLong(sequenceKey) - previous.getLong(sequenceKey)
                if (sequenceDelta > 1) {
                    sequenceGaps++
                    missingSequences += sequenceDelta - 1
                } else if (sequenceDelta <= 0) sequenceResets++
                val sourceDelta = (current.getLong(sourceTimestampKey) - previous.getLong(sourceTimestampKey)) * sourceUnitNs
                if (sourceDelta > 0) sourceIntervals += sourceDelta else sourceClockResets++
                val receiptDelta = current.getLong("received_elapsed_ns") - previous.getLong("received_elapsed_ns")
                when {
                    receiptDelta > 0 -> receiptIntervals += receiptDelta
                    receiptDelta == 0L -> receiptSharedTimestamps++
                    else -> receiptClockResets++
                }
            }
            fun spanRate(key: String, unitNs: Long): Any {
                if (samples.size < 2) return JSONObject.NULL
                val span = (samples.last().getLong(key) - samples.first().getLong(key)) * unitNs
                return if (span > 0) (samples.size - 1) * 1e9 / span else JSONObject.NULL
            }
            perBoot.put(JSONObject().put("boot_id", if (bootKey == null) JSONObject.NULL else bootId)
                .put("captured_frames", samples.size)
                .put("observed_source_hz", if (sourceClockResets == 0) spanRate(sourceTimestampKey, sourceUnitNs) else JSONObject.NULL)
                .put("observed_receipt_hz", if (receiptClockResets == 0) spanRate("received_elapsed_ns", 1L) else JSONObject.NULL)
                .put("source_positive_interval_count", sourceIntervals.size)
                .put("receipt_positive_interval_count", receiptIntervals.size)
                .put("source_interval_ms", distribution(sourceIntervals.map { it / 1e6 }))
                .put("receipt_interval_ms", distribution(receiptIntervals.map { it / 1e6 }))
                .put("sequence_gap_events", sequenceGaps).put("missing_sequence_count", missingSequences)
                .put("duplicate_or_reset_sequence_count", sequenceResets)
                .put("source_clock_nonincreasing_count", sourceClockResets)
                .put("receipt_shared_timestamp_count", receiptSharedTimestamps)
                .put("receipt_clock_decrease_count", receiptClockResets))
        }
        return JSONObject().put("captured_frames", frames.size).put("by_boot", perBoot)
            .put("source_timestamp_field", sourceTimestampKey)
            .put("boundary", "Captured sample rate over first-to-last span, including shared chunk receipt times; positive interval distribution reported separately; clocks remain unmapped")
    }

    private fun sha256(bytes: ByteArray): String = MessageDigest.getInstance("SHA-256")
        .digest(bytes).joinToString("") { "%02x".format(it) }

    private fun tagText(tag: String): String = compose.onNodeWithTag(tag).fetchSemanticsNode()
        .config[SemanticsProperties.Text].joinToString(" ") { it.text }

    private fun editableText(tag: String): String = compose.onNodeWithTag(tag).fetchSemanticsNode()
        .config[SemanticsProperties.EditableText].text

    private fun counts(): Triple<Long, Long, Long> {
        val values = Regex("[0-9]+").findAll(tagText("capture_counts")).map { it.value.toLong() }.toList()
        return Triple(values.getOrNull(0) ?: 0, values.getOrNull(1) ?: 0, values.getOrNull(2) ?: 0)
    }

    private fun verifyImuCapture(directory: File, manifest: JSONObject, validTof: List<JSONObject>): JSONObject {
        val index = rows(File(directory, "imu.index.jsonl"))
        val raw = File(directory, "imu.raw.jsonl").readBytes()
        val decoded = mutableMapOf<Long, JSONObject>()
        var expectedOffset = 0L
        index.forEachIndexed { position, row ->
            assertEquals(position + 1L, row.getLong("index"))
            assertTrue(row.getLong("received_elapsed_ns") > 0)
            val offset = row.getLong("raw_byte_offset")
            val length = row.getLong("raw_byte_length")
            assertEquals("Every original IMU byte must be indexed once", expectedOffset, offset)
            assertTrue(length > 0 && offset >= 0 && offset + length <= raw.size.toLong())
            expectedOffset += length
            val text = raw.copyOfRange(offset.toInt(), (offset + length).toInt()).toString(Charsets.UTF_8)
            val record = if (row.getBoolean("valid_frame")) JSONObject(text)
                else runCatching { JSONObject(text) }.getOrNull()
            if (record != null) decoded[row.getLong("index")] = record
            if (row.getBoolean("valid_frame")) {
                requireNotNull(record)
                assertEquals("imu_frame", record.getString("type"))
                assertEquals(1, record.getInt("schema_version"))
                assertTrue(record.get("valid") is Boolean && record.getBoolean("valid"))
                assertEquals(record.getLong("seq"), row.getLong("sequence"))
                assertEquals(record.getLong("device_us"), row.getLong("device_timestamp_us"))
                listOf("device_id", "boot_id").forEach {
                    assertFalse(record.getString(it).isBlank())
                    assertEquals(record.getString(it), row.getString(it))
                }
                listOf("accel_raw", "gyro_raw", "accel_g", "gyro_dps").forEach { key ->
                    val source = record.getJSONArray(key)
                    val saved = row.getJSONArray(key)
                    assertEquals(3, source.length())
                    assertEquals(3, saved.length())
                    repeat(3) { axis ->
                        assertTrue(source.getDouble(axis).isFinite())
                        assertEquals(source.getDouble(axis), saved.getDouble(axis), 0.0)
                        if (key.endsWith("_raw")) {
                            assertTrue(source.get(axis) is Int || source.get(axis) is Long)
                            assertTrue(source.getLong(axis) in -32768L..32767L)
                        }
                    }
                }
                if (record.has("gravity_unit") && !record.isNull("gravity_unit")) {
                    val source = record.getJSONArray("gravity_unit")
                    val saved = row.getJSONArray("gravity_unit")
                    assertEquals(3, source.length())
                    assertEquals(3, saved.length())
                    repeat(3) { axis ->
                        assertTrue(source.getDouble(axis).isFinite())
                        assertEquals(source.getDouble(axis), saved.getDouble(axis), 0.0)
                    }
                } else assertTrue(row.isNull("gravity_unit"))
                listOf("pitch_deg", "roll_deg").forEach {
                    assertTrue(record.getDouble(it).isFinite())
                    assertEquals(record.getDouble(it), row.getDouble(it), 0.0)
                }
                listOf("dt_us", "read_us").forEach {
                    assertTrue(record.getLong(it) in 0L..0xffffffffL)
                    assertEquals(record.getLong(it), row.getLong(it))
                }
                listOf("accel_correction", "sample_gap").forEach {
                    assertTrue(record.get(it) is Boolean)
                    assertEquals(record.getBoolean(it), row.getBoolean(it))
                }
                assertEquals("atom_device_microseconds_unmapped", row.getString("device_clock"))
                assertEquals("BMI270_NATIVE_NOT_TOF_OPTICAL", row.getString("pose_frame"))
                assertEquals("REQUIRED_FOR_TOF_OPTICAL_POSE", row.getString("mount_calibration"))
            }
        }
        assertEquals(raw.size.toLong(), expectedOffset)
        val valid = index.filter { it.getBoolean("valid_frame") }
        assertTrue("Too few actual IMU samples: ${valid.size}", valid.size >= 10)
        assertEquals(valid.size.toLong(), manifest.getLong("imu_count"))
        assertEquals(index.size.toLong(), manifest.getLong("imu_record_count"))
        assertEquals("atom_device_microseconds_unmapped", manifest.getString("imu_device_clock"))
        assertEquals("BMI270_NATIVE_NOT_TOF_OPTICAL", manifest.getString("imu_pose_frame"))
        val boots = decoded.values.filter { it.optString("type") == "imu_boot" }
        assertTrue("Actual Atom boot metadata must be preserved", boots.isNotEmpty())
        boots.forEach {
            assertEquals(1, it.getInt("schema_version"))
            assertEquals("BMI270", it.getString("sensor"))
            assertEquals("atom-bmi270-v1", it.getString("imu_version"))
            assertEquals(100, it.getInt("sample_hz")) // Firmware declaration; measured rate is reported below.
            assertFalse(it.getString("boot_id").isBlank())
        }
        valid.forEach { row ->
            assertTrue("Sample identity must match retained boot metadata", boots.any {
                it.getString("boot_id") == row.getString("boot_id") &&
                    it.getString("device_id") == row.getString("device_id")
            })
        }

        val byIndex = index.associateBy { it.getLong("index") }
        val residuals = mutableListOf<Long>()
        validTof.forEach { tof ->
            val alignment = tof.getJSONObject("imu_alignment")
            assertEquals("BMI270_NATIVE_NOT_TOF_OPTICAL", alignment.getString("pose_frame"))
            assertEquals("UNMAPPED", alignment.getString("device_clock_mapping"))
            if (alignment.getString("status") == "ALIGNED_RECEIPT_CAUSAL") {
                val reference = alignment.getJSONObject("raw_reference")
                assertEquals("imu.index.jsonl", reference.getString("index_file"))
                assertEquals("imu.raw.jsonl", reference.getString("raw_file"))
                val recordIndex = alignment.getLong("imu_record_index")
                assertEquals(recordIndex, reference.getLong("record_index"))
                val row = requireNotNull(byIndex[recordIndex]) { "Missing referenced IMU record $recordIndex" }
                assertTrue(row.getBoolean("valid_frame"))
                val source = requireNotNull(decoded[recordIndex])
                val residual = alignment.getLong("imu_minus_tof_ns")
                assertTrue("Alignment must be causal and at most 100ms old: $residual", residual in -100_000_000L..0L)
                assertEquals(row.getLong("received_elapsed_ns") - tof.getLong("received_elapsed_ns"), residual)
                assertEquals(source.getString("boot_id"), alignment.getString("imu_boot_id"))
                assertEquals(source.getLong("seq"), alignment.getLong("imu_sequence"))
                assertEquals(source.getLong("device_us"), alignment.getLong("imu_device_us"))
                assertEquals(source.getDouble("pitch_deg"), alignment.getDouble("imu_pitch_deg"), 0.0)
                assertEquals(source.getDouble("roll_deg"), alignment.getDouble("imu_roll_deg"), 0.0)
                residuals += residual
            } else {
                assertEquals("UNAVAILABLE", alignment.getString("status"))
                assertTrue(alignment.isNull("raw_reference"))
                assertTrue(alignment.isNull("imu_minus_tof_ns"))
            }
        }
        assertTrue("At least one real ToF frame must reference a real causal IMU sample", residuals.isNotEmpty())
        val rates = JSONArray()
        valid.groupBy { it.getString("boot_id") }.forEach { (bootId, samples) ->
            val deviceSpanUs = samples.maxOf { it.getLong("device_timestamp_us") } -
                samples.minOf { it.getLong("device_timestamp_us") }
            val receiptSpanNs = samples.maxOf { it.getLong("received_elapsed_ns") } -
                samples.minOf { it.getLong("received_elapsed_ns") }
            rates.put(JSONObject().put("boot_id", bootId).put("samples", samples.size)
                .put("device_span_us", deviceSpanUs).put("receipt_span_ns", receiptSpanNs)
                .put("observed_device_hz", if (deviceSpanUs > 0) (samples.size - 1) * 1e6 / deviceSpanUs else JSONObject.NULL)
                .put("observed_receipt_hz", if (receiptSpanNs > 0) (samples.size - 1) * 1e9 / receiptSpanNs else JSONObject.NULL))
        }
        return JSONObject().put("enabled", true).put("imu_count", valid.size)
            .put("imu_record_count", index.size).put("raw_bytes", raw.size)
            .put("raw_sha256", MessageDigest.getInstance("SHA-256").digest(raw).joinToString("") { "%02x".format(it) })
            .put("boot_ids", JSONArray(valid.map { it.getString("boot_id") }.distinct()))
            .put("device_ids", JSONArray(valid.map { it.getString("device_id") }.distinct()))
            .put("sample_gap_count", valid.count { it.getBoolean("sample_gap") })
            .put("nominal_sample_hz", 100).put("rates_by_boot", rates)
            .put("aligned_tof_count", residuals.size).put("valid_tof_count", validTof.size)
            .put("alignment_coverage", residuals.size.toDouble() / validTof.size)
            .put("imu_minus_tof_ms", distribution(residuals.map { it / 1e6 }))
            .put("imu_receipt_age_ms", distribution(residuals.map { -it / 1e6 }))
            .put("boundary", "Measured sensor rates; closest causal Android receipt within 100ms; native BMI270 pose, no exposure or device clock synchronization")
    }

    private fun distribution(values: List<Double>): JSONObject {
        if (values.isEmpty()) return JSONObject().put("count", 0)
        val sorted = values.sorted()
        val middle = sorted.size / 2
        val median = if (sorted.size % 2 == 0) (sorted[middle - 1] + sorted[middle]) / 2 else sorted[middle]
        val p95 = sorted[(kotlin.math.ceil(sorted.size * 0.95).toInt() - 1).coerceAtLeast(0)]
        return JSONObject().put("min", sorted.first()).put("median", median).put("p95", p95)
            .put("max", sorted.last()).put("count", sorted.size)
    }

    private fun rows(file: File): List<JSONObject> = file.readLines(Charsets.UTF_8)
        .filter { it.isNotBlank() }.map(::JSONObject)
}
