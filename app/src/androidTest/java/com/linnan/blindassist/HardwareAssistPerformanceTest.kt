package com.linnan.blindassist

import android.content.SharedPreferences
import android.os.ParcelFileDescriptor
import android.os.SystemClock
import android.accessibilityservice.AccessibilityServiceInfo
import android.app.UiAutomation
import android.graphics.Rect
import android.view.InputDevice
import android.view.MotionEvent
import android.view.accessibility.AccessibilityNodeInfo
import androidx.test.core.app.ActivityScenario
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import androidx.test.runner.lifecycle.ActivityLifecycleMonitorRegistry
import androidx.test.runner.lifecycle.Stage
import com.linnan.blindassist.device.glasses.HardwareDemoSnapshot
import com.linnan.blindassist.device.glasses.HardwareWifiDemoClient
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File

/** Paired on-device UI diagnostics, not a sensor FPS, exposure latency or accuracy benchmark. */
@RunWith(AndroidJUnit4::class)
class HardwareAssistPerformanceTest {
    @Test fun foregroundDrawingAndSnapshotFreshness() {
        val arguments = InstrumentationRegistry.getArguments()
        assumeTrue(arguments.getString("hardwareAssistPerformance") == "true")
        val durationMs = (arguments.getString("hardwareAssistPerformanceSeconds")?.toLongOrNull() ?: 15L)
            .coerceIn(5L, 60L) * 1_000L
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val automation = instrumentation.uiAutomation
        val originalAccessibilityFlags = automation.serviceInfo.flags
        val context = instrumentation.targetContext
        val user = context.getSharedPreferences("blindassist_user_preferences", 0)
        val hardware = context.getSharedPreferences("hardware_demo", 0)
        val userBefore = user.all
        val hardwareBefore = hardware.all
        val userKeys = listOf("onboarding_completed", "app_language", "speech_enabled", "vibration_enabled")
        val hardwareKeys = listOf("a_local", "wireless", "speech", "vibration", "alert_profile",
            "speech_style", "vibration_strength", "language", "care")
        val rows = JSONArray()
        val installedPackage = context.packageManager.getPackageInfo(context.packageName, 0)
        val receipt = JSONObject().put("scope", "foreground UI draw statistics and sampled published snapshot progression")
            .put("clock", "real Android frame clock; no Compose test rule")
            .put("installed_version_name", installedPackage.versionName)
            .put("installed_version_code", installedPackage.longVersionCode)
            .put("rate_boundary", "Observed snapshot sequence changes only; sampling can miss source frames. Not camera FPS or exposure latency.")
            .put("nominal_sampling_interval_ms", 20).put("requested_duration_ms", durationMs)
            .put("samples", rows)
        var activity: HardwareDemoActivity? = null
        var client: HardwareWifiDemoClient? = null
        var scenario: ActivityScenario<MainActivity>? = null
        var compositionTrace: HardwareComposeTrace? = null
        var released = false
        val enteredAt = SystemClock.elapsedRealtime()
        fun readSnapshot(): HardwareDemoSnapshot? {
            var result: HardwareDemoSnapshot? = null
            instrumentation.runOnMainSync {
                activity?.let {
                    result = HardwareDemoActivity::class.java.getDeclaredMethod("getSnapshot")
                        .apply { isAccessible = true }.invoke(it) as? HardwareDemoSnapshot
                }
            }
            return result
        }
        fun shell(command: String): String = ParcelFileDescriptor.AutoCloseInputStream(
            instrumentation.uiAutomation.executeShellCommand(command)).bufferedReader().use { it.readText() }
        try {
            user.edit().putBoolean("onboarding_completed", true).putString("app_language", "zh")
                .putBoolean("speech_enabled", false).putBoolean("vibration_enabled", false).commit()
            hardware.edit().putBoolean("wireless", true).commit()
            scenario = ActivityScenario.launch(MainActivity::class.java)
            automation.serviceInfo = automation.serviceInfo.apply {
                flags = flags or AccessibilityServiceInfo.FLAG_RETRIEVE_INTERACTIVE_WINDOWS
            }
            assertTrue("Actual home assistance button must be clickable", clickHomeAssist(automation))
            val readyDeadline = SystemClock.elapsedRealtime() + 30_000L
            var ready = false
            while (!ready && SystemClock.elapsedRealtime() < readyDeadline) {
                instrumentation.runOnMainSync {
                    activity = ActivityLifecycleMonitorRegistry.getInstance().getActivitiesInStage(Stage.RESUMED)
                        .filterIsInstance<HardwareDemoActivity>().firstOrNull()
                    activity?.let {
                        client = HardwareDemoActivity::class.java.getDeclaredField("wifiClient")
                            .apply { isAccessible = true }.get(it) as? HardwareWifiDemoClient
                    }
                }
                ready = readSnapshot()?.let { it.cameraUsable && it.tofUsable && it.tofTimingBasis == "host_receipt_only" } == true
                if (!ready) Thread.sleep(50)
            }
            receipt.put("ready_after_ms", SystemClock.elapsedRealtime() - enteredAt).put("ready", ready)
            assertTrue("Both real H3 and camera streams must become usable", ready)
            // Allow initial entry/layout to settle. No Compose idle waits or forced redraws during measurement.
            Thread.sleep(2_000)
            File(context.cacheDir, "hardware-assist-performance-gfx-reset.txt")
                .writeText(shell("dumpsys gfxinfo ${context.packageName} reset"))
            val traceRequested = arguments.getString("hardwareAssistComposeTrace") == "true"
            receipt.put("compose_trace_enabled", traceRequested)
            if (traceRequested) instrumentation.runOnMainSync {
                compositionTrace = HardwareComposeTrace.install()
            }
            val start = SystemClock.elapsedRealtime()
            val cpuStart = android.os.Process.getElapsedCpuTime()
            val camera = Progression()
            val tof = Progression()
            var cameraFresh = 0
            var tofFresh = 0
            var bothFresh = 0
            var foregroundSamples = 0
            var maxSampleGap = 0L
            var previousAt = start
            while (SystemClock.elapsedRealtime() - start < durationMs) {
                val sample = readSnapshot()
                val now = SystemClock.elapsedRealtime()
                val elapsed = now - start
                maxSampleGap = maxOf(maxSampleGap, now - previousAt)
                previousAt = now
                var foreground = false
                instrumentation.runOnMainSync {
                    foreground = ActivityLifecycleMonitorRegistry.getInstance().getActivitiesInStage(Stage.RESUMED).contains(activity)
                }
                if (foreground) foregroundSamples++
                if (sample?.cameraUsable == true) cameraFresh++
                if (sample?.tofUsable == true) tofFresh++
                if (sample?.cameraUsable == true && sample.tofUsable) bothFresh++
                camera.observe(sample?.cameraSequence, elapsed)
                tof.observe(sample?.tofSequence, elapsed)
                rows.put(JSONObject().put("elapsed_ms", elapsed).put("foreground", foreground)
                    .put("camera_sequence", sample?.cameraSequence ?: JSONObject.NULL)
                    .put("tof_sequence", sample?.tofSequence ?: JSONObject.NULL)
                    .put("camera_usable", sample?.cameraUsable == true).put("tof_usable", sample?.tofUsable == true)
                    .put("camera_age_ms", sample?.cameraAgeMs ?: JSONObject.NULL)
                    .put("tof_exposure_age_ms", sample?.tofAgeMs ?: JSONObject.NULL)
                    .put("tof_receipt_age_ms", sample?.tofReceiptAgeMs ?: JSONObject.NULL)
                    .put("tof_timing_basis", sample?.tofTimingBasis ?: JSONObject.NULL))
                Thread.sleep(20)
            }
            val elapsed = SystemClock.elapsedRealtime() - start
            val cpuElapsed = android.os.Process.getElapsedCpuTime() - cpuStart
            val gfx = shell("dumpsys gfxinfo ${context.packageName}")
            File(context.cacheDir, "hardware-assist-performance-gfx.txt").writeText(gfx)
            val renderedFrames = Regex("Total frames rendered:\\s*(\\d+)").find(gfx)
                ?.groupValues?.get(1)?.toIntOrNull() ?: -1
            receipt.put("measured_duration_ms", elapsed).put("sample_count", rows.length())
                .put("process_cpu_ms", cpuElapsed)
                .put("rendered_frames", renderedFrames)
                .put("max_sample_gap_ms", maxSampleGap).put("camera_progression", camera.summary(elapsed))
                .put("tof_progression", tof.summary(elapsed)).put("camera_usable_samples", cameraFresh)
                .put("tof_usable_samples", tofFresh).put("both_usable_samples", bothFresh)
                .put("foreground_samples", foregroundSamples)
            assertTrue("Real UI must render measured frames; a zero-frame result is invalid", renderedFrames > 0)
            assertTrue("Assistance must remain foreground throughout the sample", foregroundSamples == rows.length())
            assertTrue("Camera snapshot must advance during the sample", camera.changes > 0)
            assertTrue("ToF snapshot must advance during the sample", tof.changes > 0)
        } finally {
            try {
                instrumentation.runOnMainSync { compositionTrace?.close() }
                instrumentation.runOnMainSync { activity?.takeUnless { it.isFinishing }?.finish() }
                val deadline = SystemClock.elapsedRealtime() + 5_000L
                while (!released && SystemClock.elapsedRealtime() < deadline) {
                    instrumentation.runOnMainSync {
                        released = activity?.let {
                            HardwareDemoActivity::class.java.getDeclaredField("wifiClient")
                                .apply { isAccessible = true }.get(it) == null
                        } ?: true
                    }
                    if (!released) Thread.sleep(50)
                }
                released = released && client?.poll()?.let { !it.cameraUsable && !it.tofUsable } != false
                scenario?.close()
            } finally {
                automation.serviceInfo = automation.serviceInfo.apply { flags = originalAccessibilityFlags }
                restore(user, userBefore, userKeys)
                restore(hardware, hardwareBefore, hardwareKeys)
                receipt.put("stop_released_client", released)
                File(context.cacheDir, "hardware-assist-performance.json").writeText(receipt.toString(2))
            }
            assertTrue("Finishing assistance must release its client", released)
        }
    }

    private fun clickHomeAssist(automation: UiAutomation): Boolean {
        val deadline = SystemClock.elapsedRealtime() + 10_000L
        while (SystemClock.elapsedRealtime() < deadline) {
            val queue = java.util.ArrayDeque<AccessibilityNodeInfo>()
            automation.rootInActiveWindow?.let(queue::add)
            automation.windows.mapNotNull { it.root }.forEach(queue::add)
            var visited = 0
            while (queue.isNotEmpty() && visited++ < 1_000) {
                val node = queue.removeFirst()
                val text = node.text?.toString()
                val description = node.contentDescription?.toString()
                if (text in setOf("开始辅助", "Start assist") ||
                    description?.startsWith("开始辅助。") == true || description?.startsWith("Start assist.") == true) {
                    var target = node
                    var ancestors = 0
                    while (!target.isClickable && ancestors++ < 12) target = target.parent ?: break
                    if (target.performAction(AccessibilityNodeInfo.ACTION_CLICK)) return true
                    val bounds = Rect().also(target::getBoundsInScreen)
                    if (target.isVisibleToUser && !bounds.isEmpty) {
                        val time = SystemClock.uptimeMillis()
                        val down = MotionEvent.obtain(time, time, MotionEvent.ACTION_DOWN,
                            bounds.exactCenterX(), bounds.exactCenterY(), 0)
                        val up = MotionEvent.obtain(time, time + 40, MotionEvent.ACTION_UP,
                            bounds.exactCenterX(), bounds.exactCenterY(), 0)
                        try {
                            down.source = InputDevice.SOURCE_TOUCHSCREEN
                            up.source = InputDevice.SOURCE_TOUCHSCREEN
                            val acceptedDown = automation.injectInputEvent(down, true)
                            val acceptedUp = automation.injectInputEvent(up, true)
                            if (acceptedDown && acceptedUp) return true
                        } finally { down.recycle(); up.recycle() }
                    }
                }
                repeat(node.childCount) { node.getChild(it)?.let(queue::add) }
            }
            Thread.sleep(100)
        }
        return false
    }

    private class Progression {
        private var previous: Long? = null
        private var changedAt = 0L
        private var longestPlateau = 0L
        private val sequences = mutableSetOf<Long>()
        var changes = 0
            private set

        fun observe(sequence: Long?, elapsedMs: Long) {
            longestPlateau = maxOf(longestPlateau, elapsedMs - changedAt)
            if (sequence != null) sequences += sequence
            if (sequence != previous) {
                if (previous != null && sequence != null) changes++
                changedAt = elapsedMs
                previous = sequence
            }
        }

        fun summary(durationMs: Long) = JSONObject().put("unique_sequences", sequences.size)
            .put("observed_sequence_changes", changes)
            .put("observed_changes_per_second", changes * 1_000.0 / durationMs)
            .put("longest_observed_plateau_ms", maxOf(longestPlateau, durationMs - changedAt))
    }

    private fun restore(preferences: SharedPreferences, before: Map<String, *>, keys: List<String>) {
        preferences.edit().apply {
            keys.forEach { key ->
                when (val value = before[key]) {
                    is Boolean -> putBoolean(key, value)
                    is String -> putString(key, value)
                    null -> remove(key)
                    else -> error("Unexpected preference type: $key")
                }
            }
        }.commit()
    }
}
