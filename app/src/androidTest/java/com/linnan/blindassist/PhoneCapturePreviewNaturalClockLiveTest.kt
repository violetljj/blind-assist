package com.linnan.blindassist

import android.os.SystemClock
import androidx.test.core.app.ActivityScenario
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.linnan.blindassist.capture.PhoneCaptureController
import com.linnan.blindassist.capture.PhoneCaptureStore
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.security.MessageDigest
import java.util.concurrent.atomic.AtomicInteger

/** Opt-in live preview measurement without a Compose rule or its substituted frame clock. */
@RunWith(AndroidJUnit4::class)
class PhoneCapturePreviewNaturalClockLiveTest {
    @Test fun visiblePreviewWithNaturalFrameClock() {
        val args = InstrumentationRegistry.getArguments()
        assumeTrue(args.getString("phoneCapturePreviewNaturalClockLive") == "true")
        val idleMs = (args.getString("phoneCapturePreviewSeconds")?.toLong() ?: 5L) * 1000
        val recordingMs = (args.getString("phoneCaptureRecordingSeconds")?.toLong() ?: 30L) * 1000
        require(idleMs in 5_000..30_000 && recordingMs in 5_000..30_000)
        val runId = args.getString("phoneCapturePreviewRunId") ?: System.currentTimeMillis().toString()
        require(runId.matches(Regex("[A-Za-z0-9_-]+")))
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val evidence = File(context.cacheDir, "phone-capture-performance-test").apply { mkdirs() }
        val output = File(evidence, "preview-comparison-$runId.json")
        check(!output.exists()) { "Use a fresh run id to preserve prior evidence" }
        val receipt = JSONObject().put("schema", "blindassist-preview-idle-recording-v1")
            .put("run_id", runId).put("duration_per_window_ms", idleMs)
            .put("recording_window_ms", recordingMs).put("clock_mode", "natural_activity_scenario")
            .put("boundary", "Performance engineering only; no Compose test rule or substituted frame clock; Canvas submission is not physical display or exposure timing; latest-frame sampling may miss received frames")
        val scenario = ActivityScenario.launch(PhoneCaptureActivity::class.java)
        var controller: PhoneCaptureController? = null
        try {
            lateinit var controls: AtomicInteger
            scenario.onActivity { activity ->
                controller = PhoneCaptureActivity::class.java.getDeclaredField("capture").let {
                    it.isAccessible = true; it.get(activity) as PhoneCaptureController
                }
                controls = PhoneCaptureActivity::class.java.getDeclaredField("pendingControls").let {
                    it.isAccessible = true; it.get(activity) as AtomicInteger
                }
                // Use the same connection path and retained addresses as the page's connect button.
                PhoneCaptureActivity::class.java.getDeclaredMethod("connectDevices").apply {
                    isAccessible = true
                }.invoke(activity)
            }
            val active = requireNotNull(controller)
            awaitCondition(20_000, "All enabled sensor streams ready") {
                controls.get() == 0 && active.snapshot().let {
                    it.cameraReady && it.tofReady && (!it.imuEnabled || it.imuReady)
                }
            }
            receipt.put("imu_enabled", active.snapshot().imuEnabled)
            receipt.put("idle", sampleVisibleWindow(scenario, active, idleMs))
            active.startRecording("Natural-clock preview engineering $runId; unlabeled, not frozen evaluation")
            awaitCondition(5_000, "Recording started") { active.snapshot().recording }
            val sessionId = requireNotNull(active.snapshot().sessionId)
            receipt.put("session_id", sessionId)
            receipt.put("recording", sampleVisibleWindow(scenario, active, recordingMs))
            val stopNs = SystemClock.elapsedRealtimeNanos()
            active.stopRecording()
            receipt.put("stop_completed_ms", (SystemClock.elapsedRealtimeNanos() - stopNs) / 1e6)
            val session = PhoneCaptureStore.listSessions(context).single { it.id == sessionId }
            receipt.put("session_directory", session.directory.absolutePath)
            val manifest = JSONObject(File(session.directory, "manifest.json").readText())
            receipt.put("manifest", manifest)
            receipt.put("integrity", verifyFinalizedRecording(session.directory, manifest))
            assertTrue("RGB recorded", session.rgbCount >= 5)
            assertTrue("ToF recorded", session.tofCount >= 3)
        } finally {
            try {
                scenario.close()
                val active = controller
                if (active != null) awaitCondition(12_000, "Task sensor clients released") {
                    active.snapshot().let { !it.cameraConnected && !it.connected && !it.imuConnected && !it.recording }
                }
            } finally {
                val closed = controller?.snapshot()?.let {
                    !it.cameraConnected && !it.connected && !it.imuConnected && !it.recording
                } ?: false
                receipt.put("clients_closed", closed)
                output.writeText(receipt.toString(2))
            }
        }
    }

    private fun sampleVisibleWindow(
        scenario: ActivityScenario<PhoneCaptureActivity>, controller: PhoneCaptureController, durationMs: Long,
    ): JSONObject {
        scenario.onActivity { it.resetPreviewPerformance() }
        controller.resetCapturePerformance()
        val frames = JSONArray()
        var lastReceiptNs = -1L
        val startMs = SystemClock.elapsedRealtime()
        while (SystemClock.elapsedRealtime() - startMs < durationMs) {
            controller.peekCameraFrame()?.let { frame ->
                if (frame.receivedElapsedNs != lastReceiptNs) {
                    lastReceiptNs = frame.receivedElapsedNs
                    frames.put(JSONObject().put("receipt_ns", frame.receivedElapsedNs)
                        .put("sequence", frame.sequence).put("boot_id", frame.bootId)
                        .put("device_capture_us", frame.deviceCaptureTimestampUs).put("bytes", frame.jpeg.size))
                }
            }
            // Only the instrumentation thread sleeps; the activity uses its ordinary frame clock.
            SystemClock.sleep(10)
        }
        var measured: JSONObject? = null
        scenario.onActivity { measured = it.previewPerformanceReceipt() }
        val result = requireNotNull(measured).put("sampled_latest_frames", frames)
            .put("capture_performance", controller.capturePerformanceReceipt())
        val state = controller.snapshot()
        result.put("recording", state.recording).put("rgb_count", state.rgbCount)
            .put("tof_count", state.tofCount).put("imu_count", state.imuCount)
            .put("camera_age_ms", state.cameraAgeMs)
        assertTrue("Default page layout must actually draw the Atom preview: $result",
            result.getLong("drawn_unique_frames") > 0 && result.getJSONArray("canvas_timeline").length() > 0)
        assertTrue("Sampled Atom frames required", frames.length() > 0)
        return result
    }

    private fun awaitCondition(timeoutMs: Long, description: String, condition: () -> Boolean) {
        val deadline = SystemClock.elapsedRealtime() + timeoutMs
        while (!condition() && SystemClock.elapsedRealtime() < deadline) SystemClock.sleep(20)
        assertTrue(description, condition())
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
            .put("rgb_sequence_gap_frames", rgb.zipWithNext().filter { (a, b) ->
                a.getString("boot_id") == b.getString("boot_id")
            }.sumOf { (a, b) -> (b.getLong("source_frame_id") - a.getLong("source_frame_id") - 1).coerceAtLeast(0) })
            .put("tof", verifyRaw("tof")).put("imu", verifyRaw("imu"))
    }
}
