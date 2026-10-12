package com.linnan.blindassist

import android.os.SystemClock
import android.os.ParcelFileDescriptor
import android.content.SharedPreferences
import android.view.accessibility.AccessibilityNodeInfo
import android.accessibilityservice.AccessibilityServiceInfo
import android.graphics.Rect
import android.view.InputDevice
import android.view.MotionEvent
import androidx.compose.ui.test.junit4.createEmptyComposeRule
import androidx.compose.ui.test.onAllNodesWithText
import androidx.compose.ui.test.onNodeWithTag
import androidx.compose.ui.test.onNodeWithText
import androidx.compose.ui.test.performClick
import androidx.compose.ui.test.performScrollTo
import androidx.test.core.app.ActivityScenario
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import androidx.test.runner.lifecycle.ActivityLifecycleMonitorRegistry
import androidx.test.runner.lifecycle.Stage
import com.linnan.blindassist.device.glasses.HardwareDemoSnapshot
import com.linnan.blindassist.device.glasses.HardwareWifiDemoClient
import com.linnan.blindassist.feedback.HardwareFeedbackSettings
import com.linnan.blindassist.feedback.SpeechStyle
import com.linnan.blindassist.feedback.VibrationStrength
import com.linnan.blindassist.alert.AlertProfile
import com.linnan.blindassist.localization.AppLanguage
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Assume.assumeTrue
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File

/** Opt-in real transport/entry/cleanup mechanics, without obstacle labels or accuracy claims. */
@RunWith(AndroidJUnit4::class)
class HardwareAssistLiveTest {
    @get:Rule val compose = createEmptyComposeRule()

    @Test fun homeHandsSelectedFeedbackSettingsToAssistInBothLanguages() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val context = instrumentation.targetContext
        val user = context.getSharedPreferences("blindassist_user_preferences", 0)
        val hardware = context.getSharedPreferences("hardware_demo", 0)
        val userKeys = listOf("onboarding_completed", "app_language", "alert_profile", "speech_style",
            "vibration_strength", "care_mode_enabled", "speech_enabled", "vibration_enabled")
        val hardwareKeys = listOf("a_local", "wireless", "speech", "vibration", "alert_profile",
            "speech_style", "vibration_strength", "language", "care")
        val userBefore = user.all
        val hardwareBefore = hardware.all
        try {
            for (language in listOf(AppLanguage.ZH, AppLanguage.EN)) {
                user.edit().putBoolean("onboarding_completed", true)
                    .putString("app_language", language.storageValue)
                    .putString("alert_profile", AlertProfile.QUIET.storageValue)
                    .putString("speech_style", SpeechStyle.DETAILED.storageValue)
                    .putString("vibration_strength", VibrationStrength.SOFT.storageValue)
                    .putBoolean("care_mode_enabled", true).putBoolean("speech_enabled", false)
                    .putBoolean("vibration_enabled", false).commit()
                var assist: HardwareDemoActivity? = null
                try {
                    ActivityScenario.launch(MainActivity::class.java).use {
                        compose.onNodeWithTag("home_primary_assist").performScrollTo().performClick()
                        compose.waitUntil(5_000) {
                            instrumentation.runOnMainSync {
                                assist = ActivityLifecycleMonitorRegistry.getInstance().getActivitiesInStage(Stage.RESUMED)
                                    .filterIsInstance<HardwareDemoActivity>().firstOrNull()
                            }
                            assist != null
                        }
                        var actual: HardwareFeedbackSettings? = null
                        instrumentation.runOnMainSync { actual = field(requireNotNull(assist), "feedbackSettings") as HardwareFeedbackSettings }
                        assertEquals(HardwareFeedbackSettings(AlertProfile.QUIET, SpeechStyle.DETAILED,
                            VibrationStrength.SOFT, language, true), actual)
                        assertEquals(language.storageValue, hardware.getString("language", null))
                        assertFalse(hardware.getBoolean("speech", true))
                        assertFalse(hardware.getBoolean("vibration", true))
                        compose.onNodeWithTag("hardware_demo_feedback_settings").performScrollTo()
                        compose.onNodeWithTag("hardware_demo_more").performScrollTo().performClick()
                    }
                } finally { instrumentation.runOnMainSync { assist?.takeUnless { it.isFinishing }?.finish() } }
            }
        } finally {
            restore(user, userBefore, userKeys)
            restore(hardware, hardwareBefore, hardwareKeys)
        }
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

    @Test fun homeStartsFreshH3AssistAndEndingReleasesClient() {
        assumeTrue(InstrumentationRegistry.getArguments().getString("hardwareAssistLive") == "true")
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val rows = JSONArray()
        val endDiagnostics = JSONArray()
        val automation = instrumentation.uiAutomation
        val originalFlags = automation.serviceInfo.flags
        var activity: HardwareDemoActivity? = null
        var client: HardwareWifiDemoClient? = null
        var stopped = false
        val started = SystemClock.elapsedRealtime()
        try {
            ActivityScenario.launch(MainActivity::class.java).use {
                compose.waitForIdle()
                if (compose.onAllNodesWithText("跳过引导").fetchSemanticsNodes().isNotEmpty()) {
                    compose.onNodeWithText("跳过引导").performClick()
                }
                compose.onNodeWithTag("home_primary_assist").performScrollTo().performClick()
                compose.waitUntil(10_000) {
                    instrumentation.runOnMainSync {
                        activity = ActivityLifecycleMonitorRegistry.getInstance()
                            .getActivitiesInStage(Stage.RESUMED).filterIsInstance<HardwareDemoActivity>().firstOrNull()
                    }
                    activity != null
                }
                val samples = mutableListOf<HardwareDemoSnapshot>()
                val deadline = SystemClock.elapsedRealtime() + 30_000
                while (SystemClock.elapsedRealtime() < deadline && samples.size < 12) {
                    var sample: HardwareDemoSnapshot? = null
                    instrumentation.runOnMainSync {
                        sample = snapshot(requireNotNull(activity))
                        client = field(requireNotNull(activity), "wifiClient") as? HardwareWifiDemoClient
                    }
                    sample?.takeIf { it.cameraUsable && it.tofUsable && it.tofSequence != samples.lastOrNull()?.tofSequence }?.let {
                        samples += it
                        rows.put(JSONObject().put("elapsed_ms", SystemClock.elapsedRealtime() - started)
                            .put("camera_sequence", it.cameraSequence).put("tof_sequence", it.tofSequence)
                            .put("tof_timing_basis", it.tofTimingBasis)
                            .put("tof_exposure_age_ms", it.tofAgeMs ?: JSONObject.NULL)
                            .put("tof_receipt_age_ms", it.tofReceiptAgeMs ?: JSONObject.NULL)
                            .put("status", it.status))
                    }
                    Thread.sleep(80)
                }
                assertTrue("Expected 12 fresh H3 samples; see hardware-assist-live.json", samples.size >= 12)
                assertTrue("Camera sequence must advance", samples.map { it.cameraSequence }.distinct().size > 1)
                samples.forEach {
                    assertEquals(64, it.cells.size)
                    assertEquals("host_receipt_only", it.tofTimingBasis)
                    assertNull("H3 receipt is not an exposure clock bound", it.tofAgeMs)
                    assertTrue(requireNotNull(it.tofReceiptAgeMs) in 0..750)
                    assertFalse("Unaligned H3 must not run LOCAL", it.status.contains(" / LOCAL="))
                }
                assertNotNull(client)
                // Live frames continually invalidate Compose. Exercise the actual accessible button
                // through Android UI automation instead of waiting for the stream to become idle.
                automation.serviceInfo = automation.serviceInfo.apply {
                    flags = flags or AccessibilityServiceInfo.FLAG_RETRIEVE_INTERACTIVE_WINDOWS
                }
                val endDeadline = SystemClock.elapsedRealtime() + 5_000
                var clicked = false
                while (!clicked && SystemClock.elapsedRealtime() < endDeadline) {
                    val queue = java.util.ArrayDeque<AccessibilityNodeInfo>()
                    automation.rootInActiveWindow?.let(queue::add)
                    automation.windows.mapNotNull { it.root }.forEach(queue::add)
                    val attempt = JSONObject().put("root_count", queue.size)
                    val nodes = JSONArray()
                    var visited = 0
                    while (queue.isNotEmpty() && visited++ < 1_000 && !clicked) {
                        val current = queue.removeFirst()
                        val bounds = Rect().also(current::getBoundsInScreen)
                        if (endDiagnostics.length() == 0 && nodes.length() < 120) {
                            nodes.put(JSONObject().put("text", current.text?.toString())
                                .put("description", current.contentDescription?.toString())
                                .put("class", current.className?.toString())
                                .put("clickable", current.isClickable).put("bounds", bounds.flattenToString()))
                        }
                        val labels = setOf("结束辅助", "End assistance")
                        if (current.text?.toString() in labels || current.contentDescription?.toString() in labels) {
                            var target = current
                            var ancestors = 0
                            while (!target.isClickable && ancestors++ < 12) {
                                target = target.parent ?: break
                            }
                            val targetBounds = Rect().also(target::getBoundsInScreen)
                            val action = target.performAction(AccessibilityNodeInfo.ACTION_CLICK)
                            attempt.put("matched_text", current.text?.toString())
                                .put("target_bounds", targetBounds.flattenToString())
                                .put("action_click", action)
                            clicked = action
                            if (!clicked && target.isVisibleToUser && !targetBounds.isEmpty) {
                                val time = SystemClock.uptimeMillis()
                                val down = MotionEvent.obtain(time, time, MotionEvent.ACTION_DOWN,
                                    targetBounds.exactCenterX(), targetBounds.exactCenterY(), 0)
                                val up = MotionEvent.obtain(time, time + 40, MotionEvent.ACTION_UP,
                                    targetBounds.exactCenterX(), targetBounds.exactCenterY(), 0)
                                try {
                                    down.source = InputDevice.SOURCE_TOUCHSCREEN
                                    up.source = InputDevice.SOURCE_TOUCHSCREEN
                                    val downAccepted = automation.injectInputEvent(down, true)
                                    val upAccepted = automation.injectInputEvent(up, true)
                                    clicked = downAccepted && upAccepted
                                    attempt.put("injected_down", downAccepted).put("injected_up", upAccepted)
                                } finally { down.recycle(); up.recycle() }
                            }
                        }
                        repeat(current.childCount) { index -> current.getChild(index)?.let(queue::add) }
                    }
                    attempt.put("visited", visited).put("nodes", nodes)
                    endDiagnostics.put(attempt)
                    if (!clicked) Thread.sleep(100)
                }
                assertTrue("Accessible End assistance button was not clickable", clicked)
                val releaseDeadline = SystemClock.elapsedRealtime() + 5_000
                var released = false
                while (!released && SystemClock.elapsedRealtime() < releaseDeadline) {
                    instrumentation.runOnMainSync { released = field(requireNotNull(activity), "wifiClient") == null }
                    if (!released) Thread.sleep(50)
                }
                assertTrue("End assistance must release its client", released)
                val closed = requireNotNull(client).poll()
                assertFalse(closed.cameraUsable)
                assertFalse(closed.tofUsable)
                stopped = true
            }
        } finally {
            automation.serviceInfo = automation.serviceInfo.apply { flags = originalFlags }
            instrumentation.runOnMainSync { activity?.takeUnless { it.isFinishing }?.finish() }
            File(instrumentation.targetContext.cacheDir, "hardware-assist-live.json").writeText(JSONObject()
                .put("scope", "real home entry, independent fresh streams and foreground cleanup; no accuracy or synchronization claim")
                .put("samples", rows).put("end_ui_diagnostics", endDiagnostics)
                .put("stop_released_client", stopped).toString(2))
        }
    }

    @Test fun foregroundH3AssistRecoversAfterWifiLossWithoutManualReconnect() {
        assumeTrue(InstrumentationRegistry.getArguments().getString("hardwareAssistReconnect") == "true")
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val context = instrumentation.targetContext
        fun shell(command: String): String = ParcelFileDescriptor.AutoCloseInputStream(
            instrumentation.uiAutomation.executeShellCommand(command)).bufferedReader().use { it.readText() }
        fun wifiEnabled() = shell("cmd wifi status").lineSequence()
            .any { it.trim().equals("Wifi is enabled", ignoreCase = true) ||
                it.trim().equals("Wi-Fi is enabled", ignoreCase = true) }
        fun advanced(sequence: Long?, previous: Long?) = sequence != null && previous != null &&
            ((sequence - previous) and 0xffffffffL) in 1L..0x7fffffffL
        // Only exercise an already-enabled client connection; never turn on Wi-Fi for a hotspot-only run.
        assumeTrue("Reconnect test requires Wi-Fi already enabled", wifiEnabled())
        val user = context.getSharedPreferences("blindassist_user_preferences", 0)
        val hardware = context.getSharedPreferences("hardware_demo", 0)
        val userKeys = listOf("onboarding_completed", "app_language", "speech_enabled", "vibration_enabled")
        val hardwareKeys = listOf("a_local", "wireless", "speech", "vibration", "alert_profile",
            "speech_style", "vibration_strength", "language", "care", "camera", "tof")
        val userBefore = user.all
        val hardwareBefore = hardware.all
        val rows = JSONArray()
        val started = SystemClock.elapsedRealtime()
        var activity: HardwareDemoActivity? = null
        var client: HardwareWifiDemoClient? = null
        var wifiNeedsRestore = false
        var recovered = false
        var released = false
        var restoredWifi = false
        fun observe(phase: String): HardwareDemoSnapshot? {
            var sample: HardwareDemoSnapshot? = null
            instrumentation.runOnMainSync { activity?.let { sample = snapshot(it) } }
            sample?.let {
                rows.put(JSONObject().put("phase", phase).put("elapsed_ms", SystemClock.elapsedRealtime() - started)
                    .put("camera_usable", it.cameraUsable).put("tof_usable", it.tofUsable)
                    .put("camera_sequence", it.cameraSequence ?: JSONObject.NULL)
                    .put("tof_sequence", it.tofSequence ?: JSONObject.NULL)
                    .put("tof_timing_basis", it.tofTimingBasis)
                    .put("tof_receipt_age_ms", it.tofReceiptAgeMs ?: JSONObject.NULL)
                    .put("alert", it.decision.alert).put("status", it.status))
            }
            return sample
        }
        fun awaitSample(phase: String, timeoutMs: Long, predicate: (HardwareDemoSnapshot) -> Boolean): HardwareDemoSnapshot {
            val deadline = SystemClock.elapsedRealtime() + timeoutMs
            while (SystemClock.elapsedRealtime() < deadline) {
                observe(phase)?.takeIf(predicate)?.let { return it }
                Thread.sleep(200)
            }
            throw AssertionError("Timed out in $phase; see hardware-assist-reconnect.json")
        }
        try {
            user.edit().putBoolean("onboarding_completed", true).putString("app_language", AppLanguage.ZH.storageValue)
                .putBoolean("speech_enabled", false).putBoolean("vibration_enabled", false).commit()
            hardware.edit().putBoolean("wireless", true).commit()
            ActivityScenario.launch(MainActivity::class.java).use {
                compose.onNodeWithTag("home_primary_assist").performScrollTo().performClick()
                val activityDeadline = SystemClock.elapsedRealtime() + 10_000
                while (activity == null && SystemClock.elapsedRealtime() < activityDeadline) {
                    instrumentation.runOnMainSync {
                        activity = ActivityLifecycleMonitorRegistry.getInstance().getActivitiesInStage(Stage.RESUMED)
                            .filterIsInstance<HardwareDemoActivity>().firstOrNull()
                    }
                    if (activity == null) Thread.sleep(100)
                }
                assertNotNull("Home must open assistance", activity)
                val initial = awaitSample("initial_ready", 30_000) {
                    it.cameraUsable && it.tofUsable && it.tofTimingBasis == "host_receipt_only"
                }
                var generation: Any? = null
                instrumentation.runOnMainSync {
                    client = field(requireNotNull(activity), "wifiClient") as? HardwareWifiDemoClient
                    generation = field(requireNotNull(activity), "connectionGeneration")
                }
                assertNotNull(client)
                wifiNeedsRestore = true
                shell("svc wifi disable")
                val disabledAt = SystemClock.elapsedRealtime()
                val lost = awaitSample("wifi_disabled", 15_000) {
                    !it.cameraUsable && !it.tofUsable && SystemClock.elapsedRealtime() - disabledAt >= 4_000
                }
                assertFalse("Unavailable data must not retain an alert", lost.decision.alert)
                assertFalse("Wi-Fi disable must take effect", wifiEnabled())
                shell("svc wifi enable")
                val after = awaitSample("automatic_recovery", 45_000) {
                    it.cameraUsable && it.tofUsable && it.tofTimingBasis == "host_receipt_only" &&
                        it.runId == initial.runId && advanced(it.cameraSequence, initial.cameraSequence) &&
                        advanced(it.tofSequence, initial.tofSequence)
                }
                assertNull(after.tofAgeMs)
                assertEquals(64, after.cells.size)
                assertTrue(requireNotNull(after.tofReceiptAgeMs) in 0..750)
                instrumentation.runOnMainSync {
                    assertSame("Recovery must reuse the existing client", client, field(requireNotNull(activity), "wifiClient"))
                    assertEquals("Recovery must not restart the activity connection", generation,
                        field(requireNotNull(activity), "connectionGeneration"))
                    assertTrue("Assistance must stay foreground", ActivityLifecycleMonitorRegistry.getInstance()
                        .getActivitiesInStage(Stage.RESUMED).contains(activity))
                }
                recovered = true
            }
        } finally {
            try {
                if (wifiNeedsRestore) shell("svc wifi enable")
                val restoreDeadline = SystemClock.elapsedRealtime() + 10_000
                while (!wifiEnabled() && SystemClock.elapsedRealtime() < restoreDeadline) Thread.sleep(200)
                restoredWifi = wifiEnabled()
            } finally {
                instrumentation.runOnMainSync { activity?.takeUnless { it.isFinishing }?.finish() }
                val releaseDeadline = SystemClock.elapsedRealtime() + 5_000
                while (!released && SystemClock.elapsedRealtime() < releaseDeadline) {
                    instrumentation.runOnMainSync { released = activity?.let { field(it, "wifiClient") == null } ?: true }
                    if (!released) Thread.sleep(50)
                }
                released = released && client?.poll()?.let { !it.cameraUsable && !it.tofUsable } != false
                restore(user, userBefore, userKeys)
                restore(hardware, hardwareBefore, hardwareKeys)
                File(context.cacheDir, "hardware-assist-reconnect.json").writeText(JSONObject()
                    .put("scope", "opt-in real phone Wi-Fi loss and automatic foreground H3/camera recovery; no accuracy claim")
                    .put("samples", rows).put("automatic_recovery", recovered)
                    .put("wifi_restored_enabled", restoredWifi).put("stop_released_client", released).toString(2))
            }
            assertTrue("Original enabled Wi-Fi state must be restored", restoredWifi)
            assertTrue("Finishing assistance must release its client", released)
        }
    }

    private fun field(activity: HardwareDemoActivity, name: String): Any? =
        HardwareDemoActivity::class.java.getDeclaredField(name).apply { isAccessible = true }.get(activity)

    private fun snapshot(activity: HardwareDemoActivity): HardwareDemoSnapshot? =
        HardwareDemoActivity::class.java.getDeclaredMethod("getSnapshot").apply { isAccessible = true }
            .invoke(activity) as? HardwareDemoSnapshot
}
