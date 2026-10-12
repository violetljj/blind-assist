package com.linnan.blindassist

import android.os.ParcelFileDescriptor
import android.os.SystemClock
import androidx.compose.ui.semantics.SemanticsProperties
import androidx.compose.ui.state.ToggleableState
import androidx.compose.ui.test.assertIsEnabled
import androidx.compose.ui.test.junit4.createAndroidComposeRule
import androidx.compose.ui.test.onNodeWithTag
import androidx.compose.ui.test.performClick
import androidx.compose.ui.test.performScrollTo
import androidx.compose.ui.test.performTextReplacement
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.linnan.blindassist.capture.CaptureConnectionReason
import com.linnan.blindassist.capture.PhoneCaptureController
import com.linnan.blindassist.capture.PhoneCaptureState
import com.linnan.blindassist.capture.PhoneCaptureStore
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Assume.assumeTrue
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File

/** Opt-in real Wi-Fi interruption, with no firmware reset and no second TCP consumer. */
@RunWith(AndroidJUnit4::class)
class PhoneCaptureWifiDiagnosticsLiveTest {
    @get:Rule val compose = createAndroidComposeRule<PhoneCaptureActivity>()

    @Test fun wifiLossIsDiagnosedPersistedAndRecoversWithoutResumingRecording() {
        val args = InstrumentationRegistry.getArguments()
        assumeTrue(args.getString("phoneCaptureWifiDiagnosticsLive") == "true")
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val context = instrumentation.targetContext
        fun shell(command: String) = ParcelFileDescriptor.AutoCloseInputStream(
            instrumentation.uiAutomation.executeShellCommand(command)).bufferedReader().use { it.readText() }
        fun wifiEnabled() = shell("cmd wifi status").lineSequence().any {
            it.trim().equals("Wifi is enabled", true) || it.trim().equals("Wi-Fi is enabled", true)
        }
        assumeTrue("Requires an already enabled Wi-Fi client connection", wifiEnabled())
        val prefs = context.getSharedPreferences("phone_capture", 0)
        val keys = listOf("host", "atom_host", "imu_host", "imu_enabled")
        val previous = prefs.all
        val existing = PhoneCaptureStore.listSessions(context).map { it.id }.toSet()
        var wifiChanged = false
        var controller: PhoneCaptureController? = null
        val receipt = JSONObject()
        try {
            compose.onNodeWithTag("capture_settings_toggle").performScrollTo().performClick()
            val imu = compose.onNodeWithTag("capture_imu_enabled").performScrollTo()
            if (imu.fetchSemanticsNode().config[SemanticsProperties.ToggleableState] != ToggleableState.On) imu.performClick()
            compose.onNodeWithTag("capture_host").performScrollTo().performTextReplacement(
                args.getString("phoneCaptureTofHost") ?: "192.168.5.10")
            compose.onNodeWithTag("capture_atom_host").performScrollTo().performTextReplacement(
                args.getString("phoneCaptureAtomHost") ?: "192.168.5.9")
            compose.onNodeWithTag("capture_note").performScrollTo().performTextReplacement(
                "Intentional Wi-Fi loss diagnostics test; unlabeled")
            compose.onNodeWithTag("capture_settings_toggle").performScrollTo().performClick()
            compose.onNodeWithTag("capture_connect").performScrollTo().performClick()
            controller = captureController()
            val activeController = requireNotNull(controller)
            awaitState(activeController, 30_000) { it.cameraReady && it.tofReady && it.imuReady }
            compose.onNodeWithTag("capture_start").performScrollTo().assertIsEnabled().performClick()
            val recording = awaitState(activeController, 10_000) {
                it.recording && it.rgbCount >= 5 && it.tofCount >= 3 && it.imuCount >= 10
            }
            val id = requireNotNull(recording.sessionId)
            val startedNs = SystemClock.elapsedRealtimeNanos()
            wifiChanged = true
            shell("svc wifi disable")
            val faultReasons = setOf(CaptureConnectionReason.NO_VALID_FRAMES,
                CaptureConnectionReason.TRANSPORT_CLOSED, CaptureConnectionReason.SOCKET_TIMEOUT,
                CaptureConnectionReason.NETWORK_UNREACHABLE, CaptureConnectionReason.SOCKET_ERROR,
                CaptureConnectionReason.CONNECTION_FAILED, CaptureConnectionReason.SOURCE_ERROR)
            val interrupted = awaitState(activeController, 25_000) { state ->
                !state.recording && state.connectionDiagnostics.history.any {
                    it.elapsedRealtimeNs >= startedNs && it.reason in faultReasons
                }
            }
            assertFalse("Wi-Fi disable must take effect", wifiEnabled())
            val firstFault = interrupted.connectionDiagnostics.history.first {
                it.elapsedRealtimeNs >= startedNs && it.reason in faultReasons
            }
            val session = PhoneCaptureStore.listSessions(context).single { it.id == id }
            val manifestBefore = File(session.directory, "manifest.json").readText()
            assertEquals("INTERRUPTED", JSONObject(manifestBefore).getString("status"))
            val filesBefore = session.directory.walkTopDown().filter { it.isFile }
                .associate { it.relativeTo(session.directory).path to it.length() }
            shell("svc wifi enable")
            val recovered = awaitState(activeController, 50_000) { state ->
                state.cameraReady && state.tofReady && state.imuReady &&
                    state.connectionDiagnostics.history.any {
                        it.stream == firstFault.stream && it.elapsedRealtimeNs > firstFault.elapsedRealtimeNs &&
                            it.reason == CaptureConnectionReason.RECOVERED
                    }
            }
            assertFalse("Reconnect must not resume recording", recovered.recording)
            val recovery = recovered.connectionDiagnostics.history.last {
                it.stream == firstFault.stream && it.reason == CaptureConnectionReason.RECOVERED
            }
            assertTrue(requireNotNull(recovery.recoveryElapsedMs) >= 0L)
            assertTrue("Recovery should require an actual reconnect attempt", recovery.attempts >= 1)
            assertNull(recovered.connectionDiagnostics.persistenceError)
            assertEquals(1, PhoneCaptureStore.listSessions(context).count { it.id !in existing })
            assertEquals(manifestBefore, File(session.directory, "manifest.json").readText())
            assertEquals(filesBefore, session.directory.walkTopDown().filter { it.isFile }
                .associate { it.relativeTo(session.directory).path to it.length() })
            val persisted = JSONArray(File(context.filesDir, "capture-connection-history.json").readText())
            assertTrue((0 until persisted.length()).any { index ->
                persisted.getJSONObject(index).let {
                    it.optString("reason") == "RECOVERED" &&
                        it.optLong("observed_elapsed_realtime_ns") == recovery.elapsedRealtimeNs
                }
            })
            compose.activityRule.scenario.recreate()
            val reopenedController = captureController()
            controller = reopenedController
            awaitState(activeController, 10_000) { !it.cameraConnected && !it.connected && !it.imuConnected }
            val reopened = reopenedController.snapshot()
            assertFalse(reopened.recording)
            assertTrue("Fault history must survive reopening", reopened.connectionDiagnostics.history.contains(firstFault))
            assertTrue("Recovery history must survive reopening", reopened.connectionDiagnostics.history.contains(recovery))
            receipt.put("session_id", id).put("fault_reason", firstFault.reason.name)
                .put("fault_stream", firstFault.stream).put("recovery_attempts", recovery.attempts)
                .put("recovery_elapsed_ms", recovery.recoveryElapsedMs)
                .put("recording_not_resumed", true).put("finalized_segment_unchanged", true)
                .put("history_survived_reopen", true)
        } finally {
            if (wifiChanged) shell("svc wifi enable")
            val restoreDeadline = SystemClock.elapsedRealtime() + 10_000
            while (!wifiEnabled() && SystemClock.elapsedRealtime() < restoreDeadline) Thread.sleep(200)
            val wifiRestored = wifiEnabled()
            var clientsClosed = false
            try {
                compose.activityRule.scenario.close()
                controller?.let { awaitState(it, 10_000) { state ->
                    !state.cameraConnected && !state.connected && !state.imuConnected && !state.recording
                } }
                clientsClosed = true
            } finally {
                prefs.edit().apply {
                    keys.forEach { key ->
                        when (val value = previous[key]) {
                            is String -> putString(key, value)
                            is Boolean -> putBoolean(key, value)
                            else -> remove(key)
                        }
                    }
                }.commit()
                receipt.put("wifi_restored_enabled", wifiRestored).put("clients_closed", clientsClosed)
                File(context.cacheDir, "capture-wifi-diagnostics-live-receipt.json").writeText(receipt.toString(2))
            }
            assertTrue("Wi-Fi must be restored", wifiRestored)
        }
    }

    private fun captureController(): PhoneCaptureController {
        var result: PhoneCaptureController? = null
        compose.activityRule.scenario.onActivity { activity ->
            result = PhoneCaptureActivity::class.java.getDeclaredField("capture").apply { isAccessible = true }
                .get(activity) as PhoneCaptureController
        }
        return requireNotNull(result)
    }

    private fun awaitState(controller: PhoneCaptureController, timeoutMs: Long,
        predicate: (PhoneCaptureState) -> Boolean): PhoneCaptureState {
        val deadline = SystemClock.elapsedRealtime() + timeoutMs
        var state = controller.snapshot()
        while (!predicate(state) && SystemClock.elapsedRealtime() < deadline) {
            Thread.sleep(150)
            state = controller.snapshot()
        }
        assertTrue("Capture state did not reach expected condition: ${state.message}; ${state.connectionDiagnostics.latest}", predicate(state))
        return state
    }
}
