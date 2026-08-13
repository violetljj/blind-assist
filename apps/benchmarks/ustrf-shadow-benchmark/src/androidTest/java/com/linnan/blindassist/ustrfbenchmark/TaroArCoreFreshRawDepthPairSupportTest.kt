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
import com.linnan.blindassist.ustrf.TaroPoseDiverseSelection
import com.linnan.blindassist.ustrf.UstrfFrameStamp
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith

/**
 * Device-only pair-support canary for TARO's next payload boundary.
 *
 * Only raw depth whose depth and confidence timestamps exactly equal the ARCore source frame is
 * admitted. The test buffers pose metadata only for those fresh frames and asks the frozen TARO
 * selector whether repeated 150 ms..1 s pairs exist. Raw depth pixels are never decoded, copied,
 * fused, or exposed to the default app.
 */
@RunWith(AndroidJUnit4::class)
class TaroArCoreFreshRawDepthPairSupportTest {
    @get:Rule
    val cameraPermission: GrantPermissionRule = GrantPermissionRule.grant(Manifest.permission.CAMERA)

    @Test
    fun sourceAlignedRawDepthFrames_formRepeatedPoseDiversePairs() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val frameAttempts = InstrumentationRegistry.getArguments()
            .getString("taroFreshDepthFrameAttempts")
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

        var rawDepthOnlySupported = false
        var rawDepthCandidateCount = 0
        var freshSourceAlignedDepthCount = 0
        var reprojectedOrStaleDepthCount = 0
        var freshPoseAdmissionCount = 0
        var availablePairSelectionCount = 0
        var maximumSelectedTranslationM = 0f
        var maximumSelectedYawDeltaRad = 0f
        var minimumSelectedGapNs = Long.MAX_VALUE
        var maximumSelectedGapNs = 0L
        val poseAdmissionFailureCounts = linkedMapOf<String, Int>()
        val selectionFailureCounts = linkedMapOf<String, Int>()
        var session: Session? = null
        try {
            activity.runOnGlThreadAndWait(
                timeoutSeconds = maxOf(20L, frameAttempts * FRAME_SETTLE_MS / 1_000L + 25L)
            ) {
                session = Session(activity).also { created ->
                    rawDepthOnlySupported = created.isDepthModeSupported(Config.DepthMode.RAW_DEPTH_ONLY)
                    val config = Config(created).apply {
                        updateMode = Config.UpdateMode.LATEST_CAMERA_IMAGE
                        if (rawDepthOnlySupported) depthMode = Config.DepthMode.RAW_DEPTH_ONLY
                    }
                    created.configure(config)
                    created.setCameraTextureName(activity.cameraTextureName())
                    created.resume()
                }
                val poseAdapter = TaroArCoreAnchorPoseAdmissionAdapter(
                    session = requireNotNull(session),
                    sessionToken = SESSION_TOKEN
                )
                val rawDepthAdapter = UstrfArCoreRawDepthReceiptAdapter()
                val harness = TaroPoseDiverseCanaryHarness()
                try {
                    repeat(frameAttempts) { attempt ->
                        val frame = requireNotNull(session).update()
                        val sourceFrame = UstrfFrameStamp(
                            frameId = attempt.toLong(),
                            capturedAtNs = frame.timestamp,
                            coordinateFrame = ARCORE_CAMERA_FRAME
                        )
                        val poseAdmission = poseAdapter.observe(frame, sourceFrame)
                        if (poseAdmission is TaroArCoreAnchorPoseAdmission.Unavailable) {
                            val key = poseAdmission.failure.name
                            poseAdmissionFailureCounts[key] = (poseAdmissionFailureCounts[key] ?: 0) + 1
                        }
                        val rawDepth = rawDepthAdapter.observeOrNull(frame, sourceFrame)
                        if (rawDepth != null) {
                            rawDepthCandidateCount++
                            if (rawDepth.isFreshForSourceFrame) {
                                freshSourceAlignedDepthCount++
                                if (poseAdmission is TaroArCoreAnchorPoseAdmission.Available) {
                                    freshPoseAdmissionCount++
                                    when (val step = harness.observe(sourceFrame, poseAdmission)) {
                                        is TaroPoseDiverseCanaryStep.Evaluated -> when (val selection = step.selection) {
                                            is TaroPoseDiverseSelection.Available -> {
                                                availablePairSelectionCount++
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
                                        is TaroPoseDiverseCanaryStep.AnchorAdmissionRejected -> error(
                                            "Available anchor admission was rejected by the harness"
                                        )
                                        is TaroPoseDiverseCanaryStep.AdmissionRejected -> error(
                                            "Fresh-depth canary unexpectedly entered the VIO admission branch"
                                        )
                                    }
                                }
                            } else {
                                reprojectedOrStaleDepthCount++
                            }
                        }
                        Thread.sleep(FRAME_SETTLE_MS)
                    }
                } finally {
                    poseAdapter.close()
                    session?.pause()
                    session?.close()
                }
            }
        } finally {
            activity.runOnUiThread { activity.finish() }
        }

        val repeatedPairGatePass = freshPoseAdmissionCount >= MINIMUM_FRESH_ADMITTED_FRAMES &&
            availablePairSelectionCount >= MINIMUM_AVAILABLE_PAIR_SELECTIONS
        val selectionWindowGatePass = availablePairSelectionCount > 0 &&
            minimumSelectedGapNs >= MINIMUM_SELECTION_GAP_NS &&
            maximumSelectedGapNs <= MAXIMUM_SELECTION_GAP_NS
        val observedMotionGatePass = maximumSelectedTranslationM >= MINIMUM_OBSERVED_TRANSLATION_M ||
            maximumSelectedYawDeltaRad >= MINIMUM_OBSERVED_YAW_DELTA_RAD
        val report = JSONObject()
            .put("schema", "blindassist_taro_arcore_fresh_raw_depth_pair_support_v1")
            .put("package", instrumentation.targetContext.packageName)
            .put("availability", availability.name)
            .put("depth_mode", "RAW_DEPTH_ONLY")
            .put("raw_depth_only_supported", rawDepthOnlySupported)
            .put("frame_attempts", frameAttempts)
            .put("raw_depth_candidate_count", rawDepthCandidateCount)
            .put("fresh_source_aligned_depth_count", freshSourceAlignedDepthCount)
            .put("reprojected_or_stale_depth_count", reprojectedOrStaleDepthCount)
            .put("fresh_anchor_pose_admission_count", freshPoseAdmissionCount)
            .put("available_pair_selection_count", availablePairSelectionCount)
            .put("maximum_selected_translation_m", maximumSelectedTranslationM)
            .put("maximum_selected_yaw_delta_rad", maximumSelectedYawDeltaRad)
            .put("minimum_selected_gap_ns", minimumSelectedGapNs.takeIf { it != Long.MAX_VALUE } ?: JSONObject.NULL)
            .put("maximum_selected_gap_ns", maximumSelectedGapNs.takeIf { availablePairSelectionCount > 0 } ?: JSONObject.NULL)
            .put("repeated_pair_gate_pass", repeatedPairGatePass)
            .put("selection_window_gate_pass", selectionWindowGatePass)
            .put("observed_motion_gate_pass", observedMotionGatePass)
            .put("pose_admission_failure_counts", JSONObject(poseAdmissionFailureCounts as Map<*, *>))
            .put("selection_failure_counts", JSONObject(selectionFailureCounts as Map<*, *>))
            .put("authorization", JSONObject()
                .put("benchmark_only", true)
                .put("fresh_depth_metadata_bound", freshPoseAdmissionCount > 0)
                .put("raw_depth_pixels_consumed", false)
                .put("depth_registration_validated", false)
                .put("risk_field_fusion_authorized", false)
                .put("guidance_authorized", false)
                .put("default_app_changed", false)
                .put("production_authorized", false))
        Log.i(TAG, "TARO_ARCORE_FRESH_RAW_DEPTH_PAIR_SUPPORT_JSON $report")
        instrumentation.sendStatus(2, Bundle().apply { putString(REPORT_KEY, report.toString()) })

        assertTrue("RAW_DEPTH_ONLY is not supported on this device", rawDepthOnlySupported)
        assertTrue(
            "Fewer than $MINIMUM_FRESH_ADMITTED_FRAMES fresh anchor-bound depth frames or fewer than " +
                "$MINIMUM_AVAILABLE_PAIR_SELECTIONS repeated pairs were observed",
            repeatedPairGatePass
        )
        assertTrue("Fresh-depth pair selection fell outside the frozen 150ms to 1s window", selectionWindowGatePass)
        assertTrue(
            "Fresh-depth pairs did not observe at least 2cm translation or 2 degrees yaw",
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
        const val MINIMUM_FRAME_ATTEMPTS = 300
        const val DEFAULT_FRAME_ATTEMPTS = 600
        const val MAXIMUM_FRAME_ATTEMPTS = 1_800
        const val FRAME_SETTLE_MS = 33L
        const val MINIMUM_FRESH_ADMITTED_FRAMES = 4
        const val MINIMUM_AVAILABLE_PAIR_SELECTIONS = 3
        const val MINIMUM_SELECTION_GAP_NS = 150_000_000L
        const val MAXIMUM_SELECTION_GAP_NS = 1_000_000_000L
        const val MINIMUM_OBSERVED_TRANSLATION_M = .02f
        const val MINIMUM_OBSERVED_YAW_DELTA_RAD = .0349066f
        const val ARCORE_CAMERA_FRAME = "arcore-camera-v1"
        const val SESSION_TOKEN = "fresh-raw-depth-pair-canary-r0"
        const val REPORT_KEY = "taro_arcore_fresh_raw_depth_pair_support"
        const val TAG = "UstrfShadowBenchmark"
    }
}
