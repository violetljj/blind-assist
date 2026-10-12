package com.linnan.blindassist

import android.os.SystemClock
import androidx.compose.ui.semantics.SemanticsProperties
import androidx.compose.ui.state.ToggleableState
import androidx.compose.ui.test.assertIsEnabled
import androidx.compose.ui.test.junit4.createAndroidComposeRule
import androidx.compose.ui.test.onAllNodesWithTag
import androidx.compose.ui.test.onNodeWithTag
import androidx.compose.ui.test.performClick
import androidx.compose.ui.test.performScrollTo
import androidx.compose.ui.test.performTextReplacement
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.linnan.blindassist.capture.PhoneCaptureStore
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File

/** Opt-in hardware interruption: host resets ToF only after the fresh ready marker. */
@RunWith(AndroidJUnit4::class)
class PhoneCaptureReconnectLiveTest {
    @get:Rule
    val compose = createAndroidComposeRule<PhoneCaptureActivity>()

    @Test
    fun tofResetPreservesInterruptedSegmentAndAutomaticallyReconnects() {
        val args = InstrumentationRegistry.getArguments()
        assumeTrue("Requires real Atom/H3 and a host-controlled ToF reset",
            args.getString("phoneCaptureReconnectLive") == "true")
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val marker = File(context.cacheDir, "reconnect-live-ready.json")
        val runId = args.getString("phoneCaptureReconnectRunId") ?: SystemClock.elapsedRealtime().toString()
        val existing = PhoneCaptureStore.listSessions(context).map { it.id }.toSet()
        // Remove only this test's previous handshake; run_id also prevents a stale host reset.
        check(!marker.exists() || marker.delete())
        try {
            compose.onNodeWithTag("capture_settings_toggle").performScrollTo().performClick()
            val imu = compose.onNodeWithTag("capture_imu_enabled").performScrollTo()
            if (imu.fetchSemanticsNode().config[SemanticsProperties.ToggleableState] != ToggleableState.On) {
                imu.performClick()
            }
            compose.onNodeWithTag("capture_host").performScrollTo().performTextReplacement(
                args.getString("phoneCaptureTofHost") ?: "192.168.5.10")
            compose.onNodeWithTag("capture_atom_host").performScrollTo().performTextReplacement(
                args.getString("phoneCaptureAtomHost") ?: "192.168.5.9")
            compose.onNodeWithTag("capture_note").performScrollTo().performTextReplacement(
                "Live reconnect test $runId: intentional ToF reset, unlabeled")
            compose.onNodeWithTag("capture_settings_toggle").performScrollTo().performClick()
            compose.onNodeWithTag("capture_connect").performScrollTo().performClick()
            compose.waitUntil(20_000) { streamsReady() }
            startAndWaitForSamples()
            val first = PhoneCaptureStore.listSessions(context).single { it.id !in existing }
            val firstBoot = bootId(first.directory)
            assertFalse(firstBoot.isBlank())
            val resetRequestedNs = SystemClock.elapsedRealtimeNanos()
            marker.writeText(JSONObject().put("run_id", runId).put("session_id", first.id)
                .put("tof_boot_id", firstBoot).put("ready_elapsed_ns", resetRequestedNs).toString(2))

            // No connect/discover action after this marker: exercise Activity's retry wiring.
            compose.waitUntil(45_000) {
                compose.onAllNodesWithTag("capture_stop").fetchSemanticsNodes().isEmpty() &&
                    manifest(first.directory).optString("status") == "INTERRUPTED" && streamsReady()
            }
            val recoveredNs = SystemClock.elapsedRealtimeNanos()
            val interruptedManifest = File(first.directory, "manifest.json").readText()
            val firstFileSizes = first.directory.walkTopDown().filter { it.isFile }
                .associate { it.relativeTo(first.directory).path to it.length() }
            assertEquals("INTERRUPTED", JSONObject(interruptedManifest).getString("status"))
            assertFalse(JSONObject(interruptedManifest).getString("finish_reason") == "user_stop")
            // Recovery restores preview only; a separate explicit start creates the next segment.
            assertEquals(1, PhoneCaptureStore.listSessions(context).count { it.id !in existing })
            startAndWaitForSamples()
            val secondStartMs = SystemClock.elapsedRealtime()
            compose.waitUntil(7_000) { SystemClock.elapsedRealtime() - secondStartMs >= 5_000 }
            compose.onNodeWithTag("capture_stop").performScrollTo().performClick()
            compose.waitUntil(10_000) {
                compose.onAllNodesWithTag("capture_stop").fetchSemanticsNodes().isEmpty()
            }
            val second = PhoneCaptureStore.listSessions(context).single { it.id !in existing && it.id != first.id }
            val secondManifest = manifest(second.directory)
            assertEquals("COMPLETED", secondManifest.getString("status"))
            assertEquals("user_stop", secondManifest.getString("finish_reason"))
            listOf("rgb_count", "tof_count", "imu_count").forEach { assertTrue(secondManifest.getLong(it) > 0) }
            val secondBoot = bootId(second.directory)
            assertFalse("Reset must change physical ToF boot identity", firstBoot == secondBoot)
            assertEquals("Interrupted segment must stay finalized", interruptedManifest,
                File(first.directory, "manifest.json").readText())
            assertEquals("Recovery must not append to the interrupted segment", firstFileSizes,
                first.directory.walkTopDown().filter { it.isFile }
                    .associate { it.relativeTo(first.directory).path to it.length() })
            val firstLastNs = rows(File(first.directory, "tof.index.jsonl")).maxOf { it.getLong("received_elapsed_ns") }
            val secondFirstNs = rows(File(second.directory, "tof.index.jsonl"))
                .filter { it.getBoolean("valid_frame") }.minOf { it.getLong("received_elapsed_ns") }
            assertTrue("New segment must begin after interruption and recovery", secondManifest.getLong("started_elapsed_ns") >= recoveredNs)
            assertTrue("No pre-start frame may leak into the new segment", secondFirstNs >= secondManifest.getLong("started_elapsed_ns"))
            assertTrue("Preserve the real gap between segments", secondFirstNs > firstLastNs)
            File(context.cacheDir, "reconnect-live-$runId-receipt.json").writeText(JSONObject()
                .put("run_id", runId).put("interrupted_session", first.id).put("completed_session", second.id)
                .put("first_tof_boot_id", firstBoot).put("second_tof_boot_id", secondBoot)
                .put("reset_marker_to_ready_ms", (recoveredNs - resetRequestedNs) / 1e6)
                .put("tof_segment_gap_ms", (secondFirstNs - firstLastNs) / 1e6)
                .put("automatic_reconnect_without_recording_resume", true)
                .put("interrupted_segment_unchanged", true).toString(2))
        } finally {
            compose.activityRule.scenario.close()
            check(!marker.exists() || marker.delete())
        }
    }

    private fun startAndWaitForSamples() {
        compose.onNodeWithTag("capture_start").performScrollTo().assertIsEnabled().performClick()
        compose.waitUntil(8_000) {
            val values = Regex("[0-9]+").findAll(text("capture_counts")).map { it.value.toLong() }.toList()
            values.size == 3 && values[0] >= 5 && values[1] >= 3 && values[2] >= 10
        }
    }

    private fun streamsReady(): Boolean = text("capture_readiness").let {
        it.contains("RGB 已就绪") && it.contains("ToF 已就绪")
    } && text("capture_imu_readiness").contains("IMU 已就绪")

    private fun text(tag: String): String = compose.onNodeWithTag(tag).fetchSemanticsNode()
        .config[SemanticsProperties.Text].joinToString(" ") { it.text }

    private fun manifest(directory: File): JSONObject = JSONObject(File(directory, "manifest.json").readText())

    private fun rows(file: File): List<JSONObject> = file.readLines().filter { it.isNotBlank() }.map(::JSONObject)

    private fun bootId(directory: File): String = File(directory, "tof.raw.jsonl").useLines { lines ->
        lines.mapNotNull { runCatching { JSONObject(it) }.getOrNull() }
            .first { it.optString("type") == "boot" && it.has("boot_id") }.getString("boot_id")
    }
}
