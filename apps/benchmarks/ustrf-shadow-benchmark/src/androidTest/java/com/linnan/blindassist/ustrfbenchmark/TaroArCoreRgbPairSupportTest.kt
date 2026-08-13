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
import com.google.ar.core.exceptions.NotYetAvailableException
import com.linnan.blindassist.ustrf.TaroPoseDiverseSelection
import com.linnan.blindassist.ustrf.UstrfFrameStamp
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith
import java.security.MessageDigest

/**
 * Device-only RGB pair-support canary for TARO.
 *
 * A camera image is admitted only through acquireCameraImage() on the same current ARCore Frame
 * whose advancing timestamp and anchor-relative pose passed admission. ARCore documents that call
 * as returning the image corresponding to the current Frame and fails an expired Frame rather than
 * accepting a nearest image. Image/Frame/Android timestamps remain explicit diagnostics because
 * their equality changes with tracking state on the reference device. Only a bounded luminance-plane digest
 * is read; no image is retained, decoded, inferred, fused, or exposed to the default app.
 */
@RunWith(AndroidJUnit4::class)
class TaroArCoreRgbPairSupportTest {
    @get:Rule
    val cameraPermission: GrantPermissionRule = GrantPermissionRule.grant(Manifest.permission.CAMERA)

    @Test
    fun sourceBoundRgbFrames_formRepeatedPoseDiversePairs() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val frameAttempts = InstrumentationRegistry.getArguments()
            .getString("taroRgbFrameAttempts")
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

        var cameraImageAvailableCount = 0
        var cameraImageNotYetAvailableCount = 0
        var cameraImageFrameTimestampMismatchCount = 0
        var cameraImageAndroidTimestampMatchCount = 0
        var cameraImageRecentFrameTimestampMatchCount = 0
        var cameraImagePoseAdmissionUnavailableCount = 0
        var cameraImageNonmonotonicBindingCount = 0
        var sourceBoundRgbPoseAdmissionCount = 0
        var availablePairSelectionCount = 0
        var maximumSelectedTranslationM = 0f
        var maximumSelectedYawDeltaRad = 0f
        var minimumSelectedGapNs = Long.MAX_VALUE
        var maximumSelectedGapNs = 0L
        var imageWidthPx: Int? = null
        var imageHeightPx: Int? = null
        var imageFormat: Int? = null
        val distinctLuminanceDigests = linkedSetOf<String>()
        val timestampSamples = JSONArray()
        val poseAdmissionFailureCounts = linkedMapOf<String, Int>()
        val selectionFailureCounts = linkedMapOf<String, Int>()
        val recentFramesByTimestamp = linkedMapOf<Long, BoundFrame>()
        var lastBoundImageTimestampNs = -1L
        var session: Session? = null
        try {
            activity.runOnGlThreadAndWait(
                timeoutSeconds = maxOf(20L, frameAttempts * FRAME_SETTLE_MS / 1_000L + 25L)
            ) {
                session = Session(activity).also { created ->
                    created.configure(Config(created).apply {
                        // CPU payload binding requires a new current camera frame. LATEST may
                        // return the most recent Frame again after its matching CPU image moved on.
                        updateMode = Config.UpdateMode.BLOCKING
                    })
                    created.setCameraTextureName(activity.cameraTextureName())
                    created.resume()
                }
                val poseAdapter = TaroArCoreAnchorPoseAdmissionAdapter(
                    session = requireNotNull(session),
                    sessionToken = SESSION_TOKEN
                )
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
                        val oldestBindingTimestampNs = (frame.timestamp - MAXIMUM_IMAGE_BINDING_LAG_NS).coerceAtLeast(0L)
                        recentFramesByTimestamp.entries.removeAll { it.key < oldestBindingTimestampNs }
                        recentFramesByTimestamp[frame.timestamp] = BoundFrame(sourceFrame, poseAdmission)
                        val image = try {
                            frame.acquireCameraImage()
                        } catch (_: NotYetAvailableException) {
                            cameraImageNotYetAvailableCount++
                            null
                        }
                        image?.use { cameraImage ->
                            cameraImageAvailableCount++
                            imageWidthPx = cameraImage.width
                            imageHeightPx = cameraImage.height
                            imageFormat = cameraImage.format
                            if (timestampSamples.length() < MAXIMUM_TIMESTAMP_SAMPLES) {
                                timestampSamples.put(JSONObject()
                                    .put("image_timestamp_ns", cameraImage.timestamp)
                                    .put("frame_timestamp_ns", frame.timestamp)
                                    .put("android_camera_timestamp_ns", frame.androidCameraTimestamp)
                                    .put("tracking_state", frame.camera.trackingState.name)
                                    .put("pose_admission", poseAdmission.javaClass.simpleName))
                            }
                            if (cameraImage.timestamp == frame.androidCameraTimestamp) {
                                cameraImageAndroidTimestampMatchCount++
                            }
                            if (cameraImage.timestamp != frame.timestamp) {
                                cameraImageFrameTimestampMismatchCount++
                            }
                            val exactTimestampBoundFrame = recentFramesByTimestamp[cameraImage.timestamp]
                            if (exactTimestampBoundFrame != null) {
                                cameraImageRecentFrameTimestampMatchCount++
                                if (exactTimestampBoundFrame.poseAdmission !is TaroArCoreAnchorPoseAdmission.Available) {
                                    cameraImagePoseAdmissionUnavailableCount++
                                }
                            }
                            if (poseAdmission !is TaroArCoreAnchorPoseAdmission.Available) {
                                return@use
                            }
                            if (frame.timestamp <= lastBoundImageTimestampNs) {
                                cameraImageNonmonotonicBindingCount++
                                return@use
                            }
                            lastBoundImageTimestampNs = frame.timestamp
                            run {
                                distinctLuminanceDigests += boundedLuminanceDigest(cameraImage.planes.first().buffer)
                                sourceBoundRgbPoseAdmissionCount++
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
                                        "RGB canary unexpectedly entered the VIO admission branch"
                                    )
                                }
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

        val repeatedPairGatePass = sourceBoundRgbPoseAdmissionCount >= MINIMUM_SOURCE_BOUND_RGB_FRAMES &&
            availablePairSelectionCount >= MINIMUM_AVAILABLE_PAIR_SELECTIONS &&
            distinctLuminanceDigests.size >= MINIMUM_DISTINCT_LUMINANCE_DIGESTS
        val selectionWindowGatePass = availablePairSelectionCount > 0 &&
            minimumSelectedGapNs >= MINIMUM_SELECTION_GAP_NS &&
            maximumSelectedGapNs <= MAXIMUM_SELECTION_GAP_NS
        val observedMotionGatePass = maximumSelectedTranslationM >= MINIMUM_OBSERVED_TRANSLATION_M ||
            maximumSelectedYawDeltaRad >= MINIMUM_OBSERVED_YAW_DELTA_RAD
        val report = JSONObject()
            .put("schema", "blindassist_taro_arcore_rgb_pair_support_v1")
            .put("package", instrumentation.targetContext.packageName)
            .put("availability", availability.name)
            .put("frame_attempts", frameAttempts)
            .put("camera_image_available_count", cameraImageAvailableCount)
            .put("camera_image_not_yet_available_count", cameraImageNotYetAvailableCount)
            .put("camera_image_frame_timestamp_mismatch_count", cameraImageFrameTimestampMismatchCount)
            .put("camera_image_android_timestamp_match_count", cameraImageAndroidTimestampMatchCount)
            .put("camera_image_recent_frame_timestamp_match_count", cameraImageRecentFrameTimestampMatchCount)
            .put("camera_image_pose_admission_unavailable_count", cameraImagePoseAdmissionUnavailableCount)
            .put("camera_image_nonmonotonic_binding_count", cameraImageNonmonotonicBindingCount)
            .put("source_bound_rgb_anchor_pose_admission_count", sourceBoundRgbPoseAdmissionCount)
            .put("distinct_bounded_luminance_digest_count", distinctLuminanceDigests.size)
            .put("available_pair_selection_count", availablePairSelectionCount)
            .put("image_width_px", imageWidthPx ?: JSONObject.NULL)
            .put("image_height_px", imageHeightPx ?: JSONObject.NULL)
            .put("image_format", imageFormat ?: JSONObject.NULL)
            .put("timestamp_samples", timestampSamples)
            .put("maximum_selected_translation_m", maximumSelectedTranslationM)
            .put("maximum_selected_yaw_delta_rad", maximumSelectedYawDeltaRad)
            .put("minimum_selected_gap_ns", minimumSelectedGapNs.takeIf { it != Long.MAX_VALUE } ?: JSONObject.NULL)
            .put("maximum_selected_gap_ns", maximumSelectedGapNs.takeIf { availablePairSelectionCount > 0 } ?: JSONObject.NULL)
            .put("repeated_pair_gate_pass", repeatedPairGatePass)
            .put("selection_window_gate_pass", selectionWindowGatePass)
            .put("observed_motion_gate_pass", observedMotionGatePass)
            .put("pose_admission_failure_counts", JSONObject(poseAdmissionFailureCounts as Map<*, *>))
            .put("selection_failure_counts", JSONObject(selectionFailureCounts as Map<*, *>))
            .put("clock_binding", JSONObject()
                .put("pose_clock", "Frame.getTimestamp")
                .put("image_provenance", "Frame.acquireCameraImage on the same current non-expired ARCore Frame")
                .put("image_timestamp_relation", "diagnostic only; exact-current/recent matches reported separately")
                .put("maximum_binding_lag_ns", MAXIMUM_IMAGE_BINDING_LAG_NS)
                .put("android_camera_timestamp_role", "Camera2 metadata correlation diagnostic only")
                .put("binding", "same current ARCore Frame API contract; no nearest-frame fallback"))
            .put("authorization", JSONObject()
                .put("benchmark_only", true)
                .put("rgb_metadata_and_bounded_digest_bound", sourceBoundRgbPoseAdmissionCount > 0)
                .put("rgb_frame_retained", false)
                .put("pixels_decoded_or_inferred", false)
                .put("risk_field_fusion_authorized", false)
                .put("guidance_authorized", false)
                .put("default_app_changed", false)
                .put("production_authorized", false))
        Log.i(TAG, "TARO_ARCORE_RGB_PAIR_SUPPORT_JSON $report")
        instrumentation.sendStatus(2, Bundle().apply { putString(REPORT_KEY, report.toString()) })

        assertTrue("RGB source binding did not produce repeated distinct pair support", repeatedPairGatePass)
        assertTrue("RGB pair selection fell outside the frozen 150ms to 1s window", selectionWindowGatePass)
        assertTrue(
            "RGB pairs did not observe at least 2cm translation or 2 degrees yaw",
            observedMotionGatePass
        )
    }

    private fun boundedLuminanceDigest(buffer: java.nio.ByteBuffer): String {
        val input = buffer.duplicate()
        val digest = MessageDigest.getInstance("SHA-256")
        val bytes = ByteArray(minOf(DIGEST_CHUNK_BYTES, maxOf(1, input.remaining())))
        var remainingBudget = DIGEST_MAX_BYTES
        while (input.hasRemaining() && remainingBudget > 0) {
            val count = minOf(input.remaining(), bytes.size, remainingBudget)
            input.get(bytes, 0, count)
            digest.update(bytes, 0, count)
            remainingBudget -= count
        }
        return digest.digest().joinToString("") { "%02x".format(it) }
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

    private data class BoundFrame(
        val sourceFrame: UstrfFrameStamp,
        val poseAdmission: TaroArCoreAnchorPoseAdmission
    )

    private companion object {
        const val AVAILABILITY_ATTEMPTS = 10
        const val AVAILABILITY_RETRY_MS = 200L
        const val MINIMUM_FRAME_ATTEMPTS = 300
        const val DEFAULT_FRAME_ATTEMPTS = 600
        const val MAXIMUM_FRAME_ATTEMPTS = 1_800
        const val FRAME_SETTLE_MS = 33L
        const val MINIMUM_SOURCE_BOUND_RGB_FRAMES = 30
        const val MINIMUM_AVAILABLE_PAIR_SELECTIONS = 20
        const val MINIMUM_DISTINCT_LUMINANCE_DIGESTS = 10
        const val MINIMUM_SELECTION_GAP_NS = 150_000_000L
        const val MAXIMUM_SELECTION_GAP_NS = 1_000_000_000L
        const val MAXIMUM_IMAGE_BINDING_LAG_NS = 1_000_000_000L
        const val MINIMUM_OBSERVED_TRANSLATION_M = .02f
        const val MINIMUM_OBSERVED_YAW_DELTA_RAD = .0349066f
        const val DIGEST_CHUNK_BYTES = 16 * 1024
        const val DIGEST_MAX_BYTES = 64 * 1024
        const val MAXIMUM_TIMESTAMP_SAMPLES = 8
        const val ARCORE_CAMERA_FRAME = "arcore-camera-v1"
        const val SESSION_TOKEN = "rgb-pair-canary-r0"
        const val REPORT_KEY = "taro_arcore_rgb_pair_support"
        const val TAG = "UstrfShadowBenchmark"
    }
}
