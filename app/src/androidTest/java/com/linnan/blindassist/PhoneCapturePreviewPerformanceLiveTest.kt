package com.linnan.blindassist

import android.os.SystemClock
import androidx.compose.ui.test.assertIsEnabled
import androidx.compose.ui.test.junit4.createAndroidComposeRule
import androidx.compose.ui.test.onNodeWithTag
import androidx.compose.ui.test.performClick
import androidx.compose.ui.test.performScrollTo
import androidx.compose.ui.test.performTextReplacement
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.linnan.blindassist.capture.PhoneCaptureController
import com.linnan.blindassist.capture.PhoneCaptureStore
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertFalse
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.security.MessageDigest

/** Short opt-in engineering baseline: visible preview before/during recording, no recreation/export. */
@RunWith(AndroidJUnit4::class)
class PhoneCapturePreviewPerformanceLiveTest {
    @get:Rule val compose = createAndroidComposeRule<PhoneCaptureActivity>()

    @Test fun visiblePreviewBeforeAndDuringRecording() {
        val args = InstrumentationRegistry.getArguments()
        assumeTrue(args.getString("phoneCapturePreviewPerformanceLive") == "true")
        val durationMs = (args.getString("phoneCapturePreviewSeconds")?.toLong() ?: 12L) * 1000
        require(durationMs in 5_000..30_000)
        val recordingMs = (args.getString("phoneCaptureRecordingSeconds")?.toLong()?.times(1000)) ?: durationMs
        require(recordingMs in 5_000..30_000)
        val runId = args.getString("phoneCapturePreviewRunId") ?: System.currentTimeMillis().toString()
        require(runId.matches(Regex("[A-Za-z0-9_-]+")))
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val controller = PhoneCaptureActivity::class.java.getDeclaredField("capture").let {
            it.isAccessible = true; it.get(compose.activity) as PhoneCaptureController
        }
        val receipt = JSONObject().put("schema", "blindassist-preview-idle-recording-v1")
            .put("run_id", runId).put("duration_per_window_ms", durationMs)
            .put("recording_window_ms", recordingMs)
            .put("boundary", "Performance engineering only; sampled latest frames can miss received frames; not exposure latency or frozen evaluation")
        val evidence = File(context.cacheDir, "phone-capture-performance-test").apply { mkdirs() }
        val output = File(evidence, "preview-comparison-$runId.json")
        check(!output.exists()) { "Use a fresh run id to preserve prior evidence" }
        try {
            compose.onNodeWithTag("capture_settings_toggle").performScrollTo().performClick()
            compose.onNodeWithTag("capture_note").performScrollTo().performTextReplacement(
                "Performance engineering $runId: visible preview idle/recording, unlabeled, not frozen evaluation")
            compose.onNodeWithTag("capture_settings_toggle").performScrollTo().performClick()
            compose.onNodeWithTag("capture_connect").performScrollTo().assertIsEnabled().performClick()
            compose.waitUntil(20_000) { controller.snapshot().let {
                it.cameraReady && it.tofReady && (!it.imuEnabled || it.imuReady)
            } }
            receipt.put("imu_enabled", controller.snapshot().imuEnabled)
            receipt.put("idle", sampleVisibleWindow(controller, durationMs))
            compose.onNodeWithTag("capture_start").performScrollTo().assertIsEnabled().performClick()
            compose.waitUntil(5_000) { controller.snapshot().recording }
            val sessionId = requireNotNull(controller.snapshot().sessionId)
            receipt.put("session_id", sessionId)
            receipt.put("recording", sampleVisibleWindow(controller, recordingMs))
            val stopNs = SystemClock.elapsedRealtimeNanos()
            compose.onNodeWithTag("capture_stop").performScrollTo().performClick()
            compose.waitUntil(12_000) { !controller.snapshot().recording }
            receipt.put("stop_completed_ms", (SystemClock.elapsedRealtimeNanos() - stopNs) / 1e6)
            val session = PhoneCaptureStore.listSessions(context).single { it.id == sessionId }
            receipt.put("session_directory", session.directory.absolutePath)
            val manifest = JSONObject(File(session.directory, "manifest.json").readText())
            receipt.put("manifest", manifest)
            receipt.put("integrity", verifyFinalizedRecording(session.directory, manifest))
            assertTrue("RGB recorded", session.rgbCount >= 5)
            assertTrue("ToF recorded", session.tofCount >= 3)
        } finally {
            compose.activityRule.scenario.close()
            val deadline = SystemClock.elapsedRealtime() + 12_000
            while (SystemClock.elapsedRealtime() < deadline && controller.snapshot().let {
                    it.cameraConnected || it.connected || it.imuConnected || it.recording
                }) SystemClock.sleep(50)
            val final = controller.snapshot()
            val closed = !final.cameraConnected && !final.connected && !final.imuConnected && !final.recording
            receipt.put("clients_closed", closed)
            output.writeText(receipt.toString(2))
            assertTrue("Task sensor clients released", closed)
        }
    }

    private fun sampleVisibleWindow(controller: PhoneCaptureController, durationMs: Long): JSONObject {
        compose.onNodeWithTag("capture_preview").performScrollTo()
        compose.activityRule.scenario.onActivity { it.resetPreviewPerformance() }
        // Keep this test runnable against a preserved older app APK for before/after evidence.
        controller.javaClass.methods.firstOrNull { it.name == "resetCapturePerformance" }?.invoke(controller)
        val frames = JSONArray()
        var lastReceiptNs = -1L
        val startMs = SystemClock.elapsedRealtime()
        compose.waitUntil(durationMs + 5_000) {
            controller.peekCameraFrame()?.let { frame ->
                if (frame.receivedElapsedNs != lastReceiptNs) {
                    lastReceiptNs = frame.receivedElapsedNs
                    frames.put(JSONObject().put("receipt_ns", frame.receivedElapsedNs)
                        .put("sequence", frame.sequence).put("boot_id", frame.bootId)
                        .put("device_capture_us", frame.deviceCaptureTimestampUs).put("bytes", frame.jpeg.size))
                }
            }
            SystemClock.elapsedRealtime() - startMs >= durationMs
        }
        val result = compose.activity.previewPerformanceReceipt()
            .put("sampled_latest_frames", frames)
        val captureMetrics = controller.javaClass.methods.firstOrNull { it.name == "capturePerformanceReceipt" }
        result.put("capture_performance", captureMetrics?.invoke(controller)
            ?: JSONObject().put("supported", false).put("reason", "not available in baseline app"))
        val state = controller.snapshot()
        result.put("recording", state.recording).put("rgb_count", state.rgbCount)
            .put("tof_count", state.tofCount).put("imu_count", state.imuCount)
            .put("camera_age_ms", state.cameraAgeMs)
        assertTrue("Actual visible frames required: $result", result.getLong("drawn_unique_frames") > 0)
        assertFalse("Sampled Atom frames required", frames.length() == 0)
        return result
    }

    private fun verifyFinalizedRecording(directory: File, manifest: JSONObject): JSONObject {
        assertEquals("COMPLETED", manifest.getString("status"))
        assertEquals("user_stop", manifest.getString("finish_reason"))
        fun rows(name: String) = File(directory, name).readLines().filter(String::isNotBlank).map(::JSONObject)
        val rgb = rows("rgb.jsonl")
        assertEquals(manifest.getLong("rgb_count"), rgb.size.toLong())
        rgb.forEach { row ->
            val bytes = File(directory, row.getString("file")).readBytes()
            assertEquals(row.getLong("bytes"), bytes.size.toLong())
            val digest = MessageDigest.getInstance("SHA-256").digest(bytes).joinToString("") { "%02x".format(it) }
            assertEquals(row.getString("sha256"), digest)
            val headers = row.getJSONObject("source_headers")
            assertEquals(row.getString("boot_id"), headers.getString("x-sequence-id"))
            assertEquals(row.getLong("source_frame_id"), headers.getLong("x-frame-sequence"))
        }
        fun verifyRaw(stream: String): JSONObject {
            val index = rows("$stream.index.jsonl")
            val raw = File(directory, "$stream.raw.jsonl").readBytes()
            var offset = 0L
            index.forEach { row ->
                assertEquals("$stream raw offsets must be contiguous", offset, row.getLong("raw_byte_offset"))
                val length = row.getLong("raw_byte_length")
                assertTrue(length > 0 && offset + length <= raw.size)
                if (row.getBoolean("valid_frame")) {
                    val decoded = JSONObject(String(raw, offset.toInt(), length.toInt(), Charsets.UTF_8))
                    assertEquals(row.getLong("sequence"), decoded.getLong("seq"))
                }
                offset += length
            }
            assertEquals("$stream raw tail must be indexed after stop drain", raw.size.toLong(), offset)
            assertEquals(manifest.getLong("${stream}_count"), index.count { it.getBoolean("valid_frame") }.toLong())
            assertEquals(manifest.getLong("${stream}_record_count"), index.size.toLong())
            return JSONObject().put("index_rows", index.size).put("raw_bytes", raw.size)
                .put("complete_contiguous_raw_coverage", true)
        }
        return JSONObject().put("jpeg_hashes_verified", rgb.size)
            .put("rgb_boot_ids", JSONArray(rgb.map { it.getString("boot_id") }.distinct()))
            .put("rgb_sequence_gap_frames", rgb.zipWithNext().filter { (a,b) ->
                a.getString("boot_id") == b.getString("boot_id")
            }.sumOf { (a,b) -> (b.getLong("source_frame_id") - a.getLong("source_frame_id") - 1).coerceAtLeast(0) })
            .put("tof", verifyRaw("tof")).put("imu", verifyRaw("imu"))
    }
}
