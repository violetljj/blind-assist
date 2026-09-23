package com.linnan.blindassist

import android.graphics.Bitmap
import android.graphics.Color
import android.os.SystemClock
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.linnan.blindassist.device.glasses.*
import com.linnan.blindassist.risk.*
import org.junit.Assert.*
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.json.JSONObject
import org.json.JSONArray
import java.io.File

@RunWith(AndroidJUnit4::class)
class HardwareLocalRuntimeTest {
    private fun model() = InstrumentationRegistry.getInstrumentation().targetContext.assets
        .open("hardware_local/local-v1.bin").use { HardwareLocalModel.read(it) }
    private fun sample(bitmap: Bitmap) = HardwareDemoSnapshot("live", "test", 1, 1, bitmap,
        List(64) { DemoTofCell(it, 1200.0, 5, 1, "KNOWN") }, 8, 8,
        TofCorridorDemo.evaluate(8, 8, emptyList(), false), true, "test", true, true, 100, 100)

    @Test fun cropUsesCentreWithoutStretching() {
        val bitmap = Bitmap.createBitmap(640, 480, Bitmap.Config.ARGB_8888)
        try {
            bitmap.eraseColor(Color.RED)
            bitmap.setPixels(IntArray(640 * 360) { Color.BLUE }, 0, 640, 0, 60, 640, 360)
            assertTrue(HardwareLocalRuntime.modelPixels(bitmap).all { it == Color.BLUE })
        } finally { bitmap.recycle() }
    }

    @Test fun unavailableCameraDisablesLocalAndReplayNeverReevaluates() {
        val bitmap = Bitmap.createBitmap(640, 480, Bitmap.Config.ARGB_8888)
        try {
            val runtime = HardwareLocalRuntime(model())
            val input = sample(bitmap)
            val replay = input.copy(mode = "replay")
            assertSame(replay, runtime.evaluate(replay))
            val onlyTof = runtime.evaluate(input.copy(cameraUsable = false, image = null))
            val a = HardwareCalibratedA.evaluate(HardwareLocalGeometry.tokens(input.cells))
            assertEquals(a.alert, onlyTof.decision.alert)
            assertTrue(onlyTof.status.contains("仅 A"))
            val invalid = runtime.evaluate(input.copy(cells = input.cells.map { it.copy(status = 255) }))
            assertFalse(invalid.decision.alert)
            assertEquals(0, invalid.decision.validZones)
        } finally { bitmap.recycle() }
    }

    @Test fun expiryDuringInferenceRejectsOldAlert() {
        val bitmap = Bitmap.createBitmap(640, 480, Bitmap.Config.ARGB_8888)
        try {
            var now = 0L
            val runtime = HardwareLocalRuntime(model()) { now.also { now += 700 } }
            val result = runtime.evaluate(sample(bitmap))
            assertFalse(result.tofUsable)
            assertFalse(result.decision.alert)
            assertEquals("UNKNOWN", result.decision.state)
        } finally { bitmap.recycle() }
    }

    @Test fun mainThreadDelayCannotKeepLocalOnlyAlertAfterCameraExpiry() {
        val bitmap = Bitmap.createBitmap(640, 480, Bitmap.Config.ARGB_8888)
        try {
            val input = sample(bitmap).copy(cameraAgeMs = 745, tofAgeMs = 100)
            val local = input.copy(decision = DemoTofDecision(true, "LOCAL_EXPERIMENT_UNKNOWN", null, emptyList(), 64))
            val result = HardwareLocalRuntime.afterProcessing(input, local, 10)
            assertTrue(result.tofUsable)
            assertFalse(result.cameraUsable)
            assertFalse(result.decision.alert)
            val a = local.copy(decision = local.decision.copy(state = "A_SUPPORTED"))
            assertTrue(HardwareLocalRuntime.afterProcessing(input, a, 10).decision.alert)
            // A newer preview must not refresh the older RGB used by a cached LOCAL verdict.
            val freshPreview = input.copy(cameraAgeMs = 20)
            assertFalse(HardwareLocalRuntime.afterProcessing(freshPreview, local, 10).decision.alert)
        } finally { bitmap.recycle() }
    }

    /** Explicit real-input mechanics/timing smoke; has no obstacle labels or accuracy claim. */
    @Test fun liveFrozenModelOnPhone() {
        assumeTrue(InstrumentationRegistry.getArguments().getString("hardwareLocalLive") == "true")
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val devices = HardwareWifiDiscovery.discover(4000)
        val camera = requireNotNull(devices.firstOrNull { it.role == "camera" })
        val tof = requireNotNull(devices.firstOrNull { it.role == "tof" })
        val client = HardwareWifiDemoClient(camera.endpoint, tof.endpoint)
        val runtime = HardwareLocalRuntime(model())
        val rows = JSONArray()
        var complete = 0
        var lastTof: Long? = null
        try {
            client.start()
            val start = SystemClock.elapsedRealtime()
            while (SystemClock.elapsedRealtime() - start < 18_000) {
                val raw = client.poll()
                if (raw.liveUsable && raw.tofSequence != lastTof && SystemClock.elapsedRealtime() - start >= 8000) {
                    lastTof = raw.tofSequence
                    val at = SystemClock.elapsedRealtime()
                    val inferred = runtime.evaluate(raw)
                    val cost = SystemClock.elapsedRealtime() - at
                    val result = HardwareLocalRuntime.afterProcessing(raw, inferred, cost)
                    if (result.status.contains(" / LOCAL=")) complete++
                    rows.put(JSONObject().put("ms", cost).put("status", result.status)
                        .put("alert", result.decision.alert).put("state", result.decision.state)
                        .put("tof_usable", result.tofUsable).put("camera_usable", result.cameraUsable))
                }
                Thread.sleep(20)
            }
        } finally { client.close() }
        File(context.filesDir, "hardware-local-live.json").writeText(JSONObject()
            .put("scope", "real input mechanics only; no accuracy ground truth")
            .put("completed", complete).put("samples", rows).toString(2))
        assertTrue("No sustained LOCAL inference; see hardware-local-live.json", complete >= 10)
    }
}
