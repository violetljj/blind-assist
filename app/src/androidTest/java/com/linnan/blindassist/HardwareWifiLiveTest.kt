package com.linnan.blindassist

import android.content.Intent
import android.os.SystemClock
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.linnan.blindassist.device.glasses.HardwareWifiDemoClient
import com.linnan.blindassist.device.glasses.HardwareWifiDiscovery
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File

/** Explicit opt-in hardware receipt, not a deterministic CI test or display latency claim. */
@RunWith(AndroidJUnit4::class)
class HardwareWifiLiveTest {
    @Test fun phoneHotspotDirectTransport() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val args = InstrumentationRegistry.getArguments()
        assumeTrue(args.getString("hardwareWifiLive") == "true")
        val devices = HardwareWifiDiscovery.discover(4000)
        val camera = requireNotNull(devices.firstOrNull { it.role == "camera" }) { "Camera not discovered" }
        val tof = requireNotNull(devices.firstOrNull { it.role == "tof" }) { "ToF not discovered" }
        val context = instrumentation.targetContext
        val seconds = args.getString("sampleSeconds")?.toIntOrNull()?.coerceIn(10, 90) ?: 45
        val client = HardwareWifiDemoClient(camera.endpoint, tof.endpoint)
        val rows = JSONArray()
        val cameraSequences = mutableSetOf<Long>()
        val tofSequences = mutableSetOf<Long>()
        var live = 0
        var warm = false
        val start = SystemClock.elapsedRealtime()
        try {
            client.start()
            while (SystemClock.elapsedRealtime() - start < (seconds + 8) * 1000L) {
                val elapsed = SystemClock.elapsedRealtime() - start
                val snapshot = client.poll()
                if (elapsed >= 8000) {
                    warm = true
                    if (snapshot.liveUsable) {
                        live++
                        snapshot.cameraSequence?.let(cameraSequences::add)
                        snapshot.tofSequence?.let(tofSequences::add)
                        assertEquals(64, snapshot.cells.size)
                        assertNotNull(snapshot.image)
                    }
                    val diag = client.diagnostics
                    rows.put(JSONObject().put("elapsed_ms", elapsed).put("live", snapshot.liveUsable)
                        .put("camera_seq", diag.cameraSequence).put("tof_seq", diag.tofSequence)
                        .put("camera_age_upper_ms", diag.cameraAgeUpperMs)
                        .put("tof_age_upper_ms", diag.tofAgeUpperMs)
                        .put("camera_clock_error_ms", diag.cameraClockErrorMs)
                        .put("camera_acquisition_ms", diag.cameraAcquisitionMs)
                        .put("camera_transfer_upper_ms", diag.cameraTransferUpperMs)
                        .put("camera_decode_ms", diag.cameraDecodeMs)
                        .put("camera_copy_ms", diag.cameraCopyMs)
                        .put("tof_arrival_age_upper_ms", diag.tofArrivalAgeUpperMs)
                        .put("valid_tof_zones", snapshot.decision.validZones)
                        .put("status", snapshot.status))
                }
                Thread.sleep(33)
            }
        } finally {
            client.close()
            val stopped = client.poll()
            assertFalse(stopped.liveUsable)
            assertNull(stopped.image)
            assertFalse(stopped.decision.alert)
            val receipt = JSONObject().put("camera_endpoint", camera.endpoint).put("tof_endpoint", tof.endpoint)
                .put("sample_seconds", seconds).put("live_samples", live).put("total_samples", rows.length())
                .put("unique_camera_frames", cameraSequences.size).put("unique_tof_frames", tofSequences.size)
                .put("boundary", "client snapshot ages, not screen or speech latency; independent sensor samples")
                .put("samples", rows)
            File(context.filesDir, "hardware-wifi-live.json").writeText(receipt.toString())
        }
        assertTrue(warm)
        assertTrue("Live fraction below 90%: $live/${rows.length()}", live >= rows.length() * 0.9)
        assertTrue("Camera stalled", cameraSequences.size > seconds * 5)
        assertTrue("ToF stalled", tofSequences.size > seconds * 3)
    }
}
