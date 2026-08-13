package com.linnan.blindassist.ustrfbenchmark

import android.Manifest
import android.content.Intent
import android.os.Bundle
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import androidx.test.rule.GrantPermissionRule
import com.google.ar.core.ArCoreApk
import com.google.ar.core.Config
import com.google.ar.core.Session
import com.google.ar.core.TrackingState
import com.linnan.blindassist.ustrf.TaroPoseDiverseSelection
import com.linnan.blindassist.ustrf.UstrfFrameStamp
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith

/**
 * Project-owned, device-only TARO canary. It tests anchor-relative camera-history selection inside
 * the isolated benchmark package. Passing cannot authorize risk fusion, guidance, or default-App
 * integration because no pixel/depth payload or camera-to-body transform enters this test.
 */
@RunWith(AndroidJUnit4::class)
class TaroArCoreAnchorPoseDiverseCanaryTest {
    @get:Rule
    val cameraPermission: GrantPermissionRule = GrantPermissionRule.grant(Manifest.permission.CAMERA)

    @Test
    fun trackedAnchorRelativePose_drivesIsolatedPoseDiverseSelection() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val frameAttempts = InstrumentationRegistry.getArguments()
            .getString("taroFrameAttempts")
            ?.toIntOrNull()
            ?.coerceIn(MINIMUM_FRAME_ATTEMPTS, MAXIMUM_FRAME_ATTEMPTS)
            ?: DEFAULT_FRAME_ATTEMPTS
        val activity = instrumentation.startActivitySync(
            Intent(instrumentation.targetContext, UstrfArCoreBenchmarkActivity::class.java)
                .addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
        ) as UstrfArCoreBenchmarkActivity
        assertTrue("ARCore benchmark GL host did not initialize", activity.awaitGlReady())
        assertTrue("ARCore benchmark texture was not allocated", activity.cameraTextureName() != 0)

        val availability = awaitArCoreAvailability(activity)
        assertEquals(
            "ARCore must be supported and installed for this isolated canary",
            ArCoreApk.Availability.SUPPORTED_INSTALLED,
            availability
        )

        var trackingFrameCount = 0
        var advancingTimestampCount = 0
        var availableAdmissionCount = 0
        var evaluatedSelectionCount = 0
        var availableSelectionCount = 0
        var maximumSelectedTranslationM = 0f
        var maximumSelectedYawDeltaRad = 0f
        var minimumSelectedGapNs = Long.MAX_VALUE
        var maximumSelectedGapNs = 0L
        var longestContinuousTrackingRun = 0
        var lastTimestampNs = -1L
        val admissionFailureCounts = linkedMapOf<String, Int>()
        val selectionFailureCounts = linkedMapOf<String, Int>()
        var session: Session? = null
        try {
            activity.runOnGlThreadAndWait(
                timeoutSeconds = maxOf(20L, frameAttempts * FRAME_SETTLE_MS / 1_000L + 25L)
            ) {
                session = Session(activity).also { created ->
                    created.configure(Config(created).apply { updateMode = Config.UpdateMode.LATEST_CAMERA_IMAGE })
                    created.setCameraTextureName(activity.cameraTextureName())
                    created.resume()
                }
                val adapter = TaroArCoreAnchorPoseAdmissionAdapter(
                    session = requireNotNull(session),
                    sessionToken = SESSION_TOKEN
                )
                val harness = TaroPoseDiverseCanaryHarness()
                try {
                    repeat(frameAttempts) { attempt ->
                        val frame = requireNotNull(session).update()
                        val timestampNs = frame.timestamp
                        if (timestampNs > lastTimestampNs) advancingTimestampCount++
                        lastTimestampNs = maxOf(lastTimestampNs, timestampNs)
                        if (frame.camera.trackingState == TrackingState.TRACKING) trackingFrameCount++
                        val sourceFrame = UstrfFrameStamp(attempt.toLong(), timestampNs, ARCORE_CAMERA_FRAME)
                        val admission = adapter.observe(frame, sourceFrame)
                        when (admission) {
                            is TaroArCoreAnchorPoseAdmission.Available -> {
                                availableAdmissionCount++
                                longestContinuousTrackingRun = maxOf(
                                    longestContinuousTrackingRun,
                                    admission.continuousTrackingFrames
                                )
                            }
                            is TaroArCoreAnchorPoseAdmission.Unavailable -> {
                                val key = admission.failure.name
                                admissionFailureCounts[key] = (admissionFailureCounts[key] ?: 0) + 1
                                longestContinuousTrackingRun = maxOf(
                                    longestContinuousTrackingRun,
                                    admission.continuousTrackingFrames
                                )
                            }
                        }
                        when (val step = harness.observe(sourceFrame, admission)) {
                            is TaroPoseDiverseCanaryStep.Evaluated -> {
                                evaluatedSelectionCount++
                                when (val selection = step.selection) {
                                    is TaroPoseDiverseSelection.Available -> {
                                        availableSelectionCount++
                                        maximumSelectedTranslationM = maxOf(
                                            maximumSelectedTranslationM,
                                            selection.translationM
                                        )
                                        maximumSelectedYawDeltaRad = maxOf(
                                            maximumSelectedYawDeltaRad,
                                            selection.yawDeltaRad
                                        )
                                        minimumSelectedGapNs = minOf(minimumSelectedGapNs, selection.gapNs)
                                        maximumSelectedGapNs = maxOf(maximumSelectedGapNs, selection.gapNs)
                                    }
                                    is TaroPoseDiverseSelection.Unavailable -> {
                                        val key = selection.failure.name
                                        selectionFailureCounts[key] = (selectionFailureCounts[key] ?: 0) + 1
                                    }
                                }
                            }
                            is TaroPoseDiverseCanaryStep.AnchorAdmissionRejected -> Unit
                            is TaroPoseDiverseCanaryStep.AdmissionRejected -> error(
                                "Anchor canary unexpectedly entered the VIO admission branch"
                            )
                        }
                        Thread.sleep(FRAME_SETTLE_MS)
                    }
                } finally {
                    adapter.close()
                    session?.pause()
                    session?.close()
                }
            }
        } finally {
            activity.runOnUiThread { activity.finish() }
        }

        val selectionWindowGatePass = availableSelectionCount > 0 &&
            minimumSelectedGapNs >= MINIMUM_SELECTION_GAP_NS &&
            maximumSelectedGapNs <= MAXIMUM_SELECTION_GAP_NS
        val observedMotionGatePass = maximumSelectedTranslationM >= MINIMUM_OBSERVED_TRANSLATION_M ||
            maximumSelectedYawDeltaRad >= MINIMUM_OBSERVED_YAW_DELTA_RAD
        val report = JSONObject()
            .put("schema", "blindassist_taro_arcore_anchor_pose_diverse_canary_v1")
            .put("package", instrumentation.targetContext.packageName)
            .put("availability", availability.name)
            .put("frame_attempts", frameAttempts)
            .put("advancing_timestamp_count", advancingTimestampCount)
            .put("tracking_frame_count", trackingFrameCount)
            .put("longest_continuous_tracking_run_frames", longestContinuousTrackingRun)
            .put("available_anchor_pose_admission_count", availableAdmissionCount)
            .put("evaluated_selector_count", evaluatedSelectionCount)
            .put("available_selector_count", availableSelectionCount)
            .put("maximum_selected_translation_m", maximumSelectedTranslationM)
            .put("maximum_selected_yaw_delta_rad", maximumSelectedYawDeltaRad)
            .put("minimum_selected_gap_ns", minimumSelectedGapNs.takeIf { it != Long.MAX_VALUE } ?: JSONObject.NULL)
            .put("maximum_selected_gap_ns", maximumSelectedGapNs.takeIf { availableSelectionCount > 0 } ?: JSONObject.NULL)
            .put("selection_window_gate_pass", selectionWindowGatePass)
            .put("observed_motion_gate_pass", observedMotionGatePass)
            .put("admission_failure_counts", JSONObject(admissionFailureCounts as Map<*, *>))
            .put("selection_failure_counts", JSONObject(selectionFailureCounts as Map<*, *>))
            .put("pose_coordinate_contract", "camera pose relative to one local ARCore anchor in one session")
            .put("authorization", JSONObject()
                .put("benchmark_only", true)
                .put("pose_diverse_selector_exercised", availableSelectionCount > 0)
                .put("pixels_or_depth_consumed", false)
                .put("camera_to_body_extrinsics_required_for_this_camera_only_selector", false)
                .put("camera_to_body_transform_authorized", false)
                .put("risk_field_fusion_authorized", false)
                .put("guidance_authorized", false)
                .put("default_app_changed", false)
                .put("production_authorized", false))
        Log.i(TAG, "TARO_ARCORE_ANCHOR_POSE_DIVERSE_CANARY_JSON $report")
        instrumentation.sendStatus(2, Bundle().apply { putString(REPORT_KEY, report.toString()) })

        assertTrue("ARCore did not expose an advancing frame timestamp", advancingTimestampCount > 0)
        assertTrue("No anchor-relative pose passed continuous-tracking admission", availableAdmissionCount > 0)
        assertTrue("TARO selector never returned an eligible historical frame", availableSelectionCount > 0)
        assertTrue("Selected history fell outside the frozen 150ms to 1s window", selectionWindowGatePass)
        assertTrue(
            "No material camera motion was observed; move the device by at least 2cm or 2 degrees",
            observedMotionGatePass
        )
    }

    private fun awaitArCoreAvailability(activity: UstrfArCoreBenchmarkActivity): ArCoreApk.Availability {
        var availability = ArCoreApk.Availability.UNKNOWN_CHECKING
        repeat(AVAILABILITY_ATTEMPTS) {
            availability = ArCoreApk.getInstance().checkAvailability(activity)
            if (!availability.isTransient) return availability
            Thread.sleep(AVAILABILITY_RETRY_MS)
        }
        return availability
    }

    private companion object {
        const val AVAILABILITY_ATTEMPTS = 10
        const val AVAILABILITY_RETRY_MS = 200L
        const val MINIMUM_FRAME_ATTEMPTS = 60
        const val DEFAULT_FRAME_ATTEMPTS = 120
        const val MAXIMUM_FRAME_ATTEMPTS = 1_800
        const val FRAME_SETTLE_MS = 33L
        const val MINIMUM_SELECTION_GAP_NS = 150_000_000L
        const val MAXIMUM_SELECTION_GAP_NS = 1_000_000_000L
        const val MINIMUM_OBSERVED_TRANSLATION_M = .02f
        const val MINIMUM_OBSERVED_YAW_DELTA_RAD = .0349066f
        const val ARCORE_CAMERA_FRAME = "arcore-camera-v1"
        const val SESSION_TOKEN = "isolated-device-canary-r0"
        const val REPORT_KEY = "taro_arcore_anchor_pose_diverse_canary"
        const val TAG = "UstrfShadowBenchmark"
    }
}
