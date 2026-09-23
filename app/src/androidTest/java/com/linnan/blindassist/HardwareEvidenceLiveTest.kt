package com.linnan.blindassist

import android.os.SystemClock
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.linnan.blindassist.device.glasses.HardwareDemoSnapshot
import com.linnan.blindassist.device.glasses.HardwareEvidenceRecorder
import com.linnan.blindassist.device.glasses.HardwareEvidenceStore
import com.linnan.blindassist.device.glasses.HardwareWifiDemoClient
import com.linnan.blindassist.device.glasses.HardwareWifiDiscovery
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.util.UUID
import kotlin.math.ceil

/** Explicit real transport smoke; ages/offer timings do not measure optical or audible latency. */
@RunWith(AndroidJUnit4::class)
class HardwareEvidenceLiveTest {
    @Test fun boundedRecordingAndIndependentAvailability() {
        assumeTrue(InstrumentationRegistry.getArguments().getString("hardwareWifiLive") == "true")
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val fixture = File(context.cacheDir, "hardware-evidence-live-${UUID.randomUUID()}")
        val phases = JSONArray()
        val degraded = JSONArray()
        val receipt = JSONObject().put("schema", "hardware-evidence-live-v1")
            .put("boundary", "Phone client ages and recorder offer overhead only; no optical latency, speech, perception, calibration, or user outcome evidence")
            .put("phase_window_ms", 8000).put("poll_target_hz", 20)
            .put("phases", phases).put("degraded_checks", degraded).put("passed", false)
        try {
            val devices = HardwareWifiDiscovery.discover(4000)
            val camera = requireNotNull(devices.firstOrNull { it.role == "camera" }) { "Camera not discovered" }
            val tof = requireNotNull(devices.firstOrNull { it.role == "tof" }) { "ToF not discovered" }
            receipt.put("camera_endpoint", camera.endpoint).put("tof_endpoint", tof.endpoint)
            HardwareWifiDemoClient(camera.endpoint, tof.endpoint).use { client ->
                client.start()
                warmup(client) { it.liveUsable }
                for (phase in listOf("off", "scalar", "thumbnails")) {
                    val recorder = if (phase == "off") null else HardwareEvidenceRecorder("live-smoke").apply {
                        captureImages = phase == "thumbnails"
                    }
                    val report = JSONObject().put("phase", phase)
                    phases.put(report)
                    try {
                        val offers = mutableListOf<Long>()
                        val cameraAges = mutableListOf<Long>()
                        val tofAges = mutableListOf<Long>()
                        var samples = 0
                        var full = 0
                        val start = SystemClock.elapsedRealtime()
                        while (SystemClock.elapsedRealtime() - start < 8000) {
                            val tick = SystemClock.elapsedRealtime()
                            val snapshot = client.poll()
                            samples++
                            if (snapshot.liveUsable) full++
                            snapshot.cameraAgeMs?.let(cameraAges::add)
                            snapshot.tofAgeMs?.let(tofAges::add)
                            if (recorder != null) {
                                val before = System.nanoTime()
                                recorder.offer(snapshot, tick)
                                offers += System.nanoTime() - before
                            }
                            val remaining = 50 - (SystemClock.elapsedRealtime() - tick)
                            if (remaining > 0) Thread.sleep(remaining)
                        }
                        report.put("elapsed_ms", SystemClock.elapsedRealtime() - start)
                            .put("samples", samples).put("full_live_samples", full)
                            .put("camera_age_p95_ms", p95(cameraAges) ?: JSONObject.NULL)
                            .put("tof_age_p95_ms", p95(tofAges) ?: JSONObject.NULL)
                            .put("offer_p95_ns", p95(offers) ?: JSONObject.NULL)
                            .put("offer_count", offers.size)
                        if (recorder != null) {
                            val clip = recorder.freeze()
                            val store = HardwareEvidenceStore(fixture)
                            val saved = store.save(clip)
                            val loaded = store.load(saved.id)
                            report.put("saved_frames", loaded.frames.size).put("saved_bytes", saved.bytes)
                                .put("saved_images", loaded.frames.count { it.thumbnail != null })
                            assertEquals(clip.frames.size, loaded.frames.size)
                            loaded.frames.forEachIndexed { index, frame ->
                                assertEquals(clip.frames[index].decision, frame.decision)
                                val replay = loaded.snapshotAt(frame.atMs - loaded.frames.first().atMs)
                                try {
                                    assertEquals("replay", replay.mode)
                                    assertFalse(replay.liveUsable)
                                    assertEquals(frame.decision, replay.decision)
                                } finally { replay.image?.recycle() }
                            }
                            report.put("exact_saved_decisions", true)
                            if (phase == "thumbnails") assertTrue("No thumbnails saved", loaded.frames.any { it.thumbnail != null })
                            assertTrue("Recorder offer p95 >= 10 ms", requireNotNull(p95(offers)) < 10_000_000L)
                        } else {
                            report.put("saved_frames", 0).put("saved_bytes", 0).put("saved_images", 0)
                        }
                        assertTrue("No phase samples", samples > 0)
                        assertTrue("$phase full-live fraction below 90%: $full/$samples", full >= samples * .9)
                        report.put("passed", true)
                    } finally { recorder?.close() }
                }
            }
            checkDegraded(tof.endpoint, tof.endpoint, "camera_role_rejected", degraded) {
                it.tofUsable && !it.cameraUsable && !it.liveUsable && it.image == null
            }
            checkDegraded(camera.endpoint, camera.endpoint, "tof_unavailable", degraded) {
                it.cameraUsable && !it.tofUsable && !it.liveUsable && !it.decision.alert
            }
            receipt.put("passed", true)
        } catch (failure: Throwable) {
            receipt.put("failure", "${failure.javaClass.simpleName}: ${failure.message}")
            throw failure
        } finally {
            // Only this test's UUID-named private cache fixtures are removed; receipt survives.
            val cleaned = !fixture.exists() || fixture.deleteRecursively()
            receipt.put("fixture_cleanup_ok", cleaned)
            File(context.filesDir, "hardware-evidence-live.json").writeText(receipt.toString(2))
            check(cleaned) { "Task-owned evidence fixture cleanup failed: $fixture" }
        }
    }

    private fun warmup(client: HardwareWifiDemoClient, ready: (HardwareDemoSnapshot) -> Boolean) {
        val start = SystemClock.elapsedRealtime()
        while (SystemClock.elapsedRealtime() - start < 8000) {
            if (ready(client.poll())) return
            Thread.sleep(50)
        }
        fail("Expected transport state not reached within 8 seconds")
    }

    private fun checkDegraded(camera: String, tof: String, name: String, reports: JSONArray,
        expected: (HardwareDemoSnapshot) -> Boolean) {
        val report = JSONObject().put("case", name).put("passed", false)
        reports.put(report)
        HardwareWifiDemoClient(camera, tof).use { client ->
            client.start()
            warmup(client, expected)
            val start = SystemClock.elapsedRealtime()
            var samples = 0
            var matching = 0
            var consecutive = 0
            var longest = 0
            while (SystemClock.elapsedRealtime() - start < 2000) {
                samples++
                if (expected(client.poll())) { matching++; consecutive++; longest = maxOf(longest, consecutive) }
                else consecutive = 0
                Thread.sleep(50)
            }
            report.put("samples", samples).put("matching_samples", matching)
                .put("longest_consecutive", longest).put("elapsed_ms", SystemClock.elapsedRealtime() - start)
            assertTrue("$name did not remain degraded", matching >= samples * .9 && longest >= 10)
            report.put("passed", true)
        }
    }

    private fun p95(values: List<Long>): Long? = if (values.isEmpty()) null else
        values.sorted()[(ceil(values.size * .95).toInt() - 1).coerceAtLeast(0)]
}
