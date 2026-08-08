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
import com.google.ar.core.Frame
import com.google.ar.core.Session
import com.google.ar.core.TrackingState
import com.linnan.blindassist.ustrf.UstrfFrameStamp
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith

/**
 * Isolated, benchmark-only ARCore audit. A passing test means only that this exact device exposed
 * tracked pose/depth candidates in this run; it does not authorize USTRF safety actions.
 */
@RunWith(AndroidJUnit4::class)
class UstrfArCoreCapabilityAuditTest {
    @get:Rule
    val cameraPermission: GrantPermissionRule = GrantPermissionRule.grant(Manifest.permission.CAMERA)

    @Test
    fun installedArCore_auditsPoseAndDepthCapabilityWithoutAuthorizingIt() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val frameAttempts = InstrumentationRegistry.getArguments()
            .getString("ustrfArcoreFrameAttempts")
            ?.toIntOrNull()
            ?.coerceIn(1, MAX_FRAME_ATTEMPTS)
            ?: DEFAULT_FRAME_ATTEMPTS
        val activity = instrumentation.startActivitySync(
            Intent(instrumentation.targetContext, UstrfArCoreBenchmarkActivity::class.java)
                .addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
        ) as UstrfArCoreBenchmarkActivity
        assertTrue("ARCore benchmark GL host did not initialize", activity.awaitGlReady())
        assertTrue("ARCore benchmark texture was not allocated", activity.cameraTextureName() != 0)
        var availability = ArCoreApk.Availability.UNKNOWN_CHECKING
        repeat(AVAILABILITY_ATTEMPTS) {
            availability = ArCoreApk.getInstance().checkAvailability(activity)
            if (!availability.isTransient) return@repeat
            Thread.sleep(AVAILABILITY_RETRY_MS)
        }
        assertEquals("ARCore must be supported and installed for this isolated audit", ArCoreApk.Availability.SUPPORTED_INSTALLED, availability)

        var session: Session? = null
        var trackingFrames = 0
        var monotonicTimestampFrames = 0
        var rawDepthFrames = 0
        var freshRawDepthFrames = 0
        var reprojectedOrStaleRawDepthFrames = 0
        var rawPoseReceiptCount = 0
        var rawIntrinsicsObservationCount = 0
        val intrinsicsSignatures = linkedSetOf<String>()
        var firstIntrinsics: UstrfArCoreRawCameraIntrinsicsObservation? = null
        var lastIntrinsics: UstrfArCoreRawCameraIntrinsicsObservation? = null
        val trackingStateCounts = linkedMapOf<String, Int>()
        var currentTrackingRun = 0
        var longestTrackingRun = 0
        var trackingStateTransitions = 0
        var previousTrackingState: String? = null
        var lastTimestampNs = -1L
        var firstTrackingPose: FloatArray? = null
        var lastTrackingPose: FloatArray? = null
        var depthSupported = false
        val rawPoseAdapter = UstrfArCoreRawPoseReceiptAdapter()
        val rawDepthAdapter = UstrfArCoreRawDepthReceiptAdapter()
        val rawIntrinsicsAdapter = UstrfArCoreRawCameraIntrinsicsAdapter()
        // The frozen ARCore 1.33 SDK uses its default camera-texture binding path. The audit does
        // not consume the texture; it only measures whether the session can expose pose/depth.
        val textureMode = "BIND_TO_TEXTURE_EXTERNAL_OES (ARCore 1.33 default)"
        try {
            // Keep every ARCore interaction, including close, in one GL task. Re-queueing close
            // after the final update proved lifecycle-racy on the target device.
            activity.runOnGlThreadAndWait(
                timeoutSeconds = maxOf(15L, frameAttempts * FRAME_SETTLE_MS / 1_000L + 20L)
            ) {
                session = Session(activity).also { created ->
                    val config = Config(created)
                    depthSupported = created.isDepthModeSupported(Config.DepthMode.AUTOMATIC)
                    if (depthSupported) config.depthMode = Config.DepthMode.AUTOMATIC
                    config.updateMode = Config.UpdateMode.LATEST_CAMERA_IMAGE
                    created.configure(config)
                    created.setCameraTextureName(activity.cameraTextureName())
                    created.resume()
                }
                try {
                    repeat(frameAttempts) {
                        val current = requireNotNull(session).update()
                        val timestampNs = current.timestamp
                        val sourceFrame = UstrfFrameStamp(rawPoseReceiptCount.toLong(), timestampNs, ARCORE_CAMERA_FRAME)
                        val rawPose = rawPoseAdapter.observe(current, sourceFrame)
                        val rawIntrinsics = rawIntrinsicsAdapter.observe(current, sourceFrame)
                        rawPoseReceiptCount++
                        rawIntrinsicsObservationCount++
                        intrinsicsSignatures += rawIntrinsics.signature()
                        if (firstIntrinsics == null) firstIntrinsics = rawIntrinsics
                        lastIntrinsics = rawIntrinsics
                        trackingStateCounts[rawPose.trackingState] = (trackingStateCounts[rawPose.trackingState] ?: 0) + 1
                        if (previousTrackingState != null && previousTrackingState != rawPose.trackingState) trackingStateTransitions++
                        previousTrackingState = rawPose.trackingState
                        if (rawPoseAdapter.isTracking(rawPose)) {
                            currentTrackingRun++
                            longestTrackingRun = maxOf(longestTrackingRun, currentTrackingRun)
                        } else {
                            currentTrackingRun = 0
                        }
                        if (timestampNs > lastTimestampNs) monotonicTimestampFrames++
                        lastTimestampNs = maxOf(lastTimestampNs, timestampNs)
                        if (current.camera.trackingState == TrackingState.TRACKING) {
                            trackingFrames++
                            val pose = current.camera.pose
                            val translation = floatArrayOf(pose.tx(), pose.ty(), pose.tz())
                            if (firstTrackingPose == null) firstTrackingPose = translation
                            lastTrackingPose = translation
                            if (depthSupported) {
                                val rawDepth = rawDepthAdapter.observeOrNull(current, sourceFrame)
                                if (rawDepth != null) {
                                    rawDepthFrames++
                                    if (rawDepth.isFreshForSourceFrame) {
                                        freshRawDepthFrames++
                                    } else {
                                        reprojectedOrStaleRawDepthFrames++
                                    }
                                }
                            }
                        }
                        Thread.sleep(FRAME_SETTLE_MS)
                    }
                } finally {
                    session?.pause()
                    session?.close()
                }
            }
        } finally {
            activity.runOnUiThread { activity.finish() }
        }

        val report = JSONObject()
            .put("schema", "blindassist_ustrf_arcore_capability_audit_v1")
            .put("package", instrumentation.targetContext.packageName)
            .put("availability", availability.name)
            .put("depth_mode_automatic_supported", depthSupported)
            .put("texture_update_mode", textureMode)
            .put("frame_attempts", frameAttempts)
            .put("tracking_frame_count", trackingFrames)
            .put("strictly_monotonic_timestamp_count", monotonicTimestampFrames)
            .put("raw_depth_frame_count", rawDepthFrames)
            .put("fresh_raw_depth_frame_count", freshRawDepthFrames)
            .put("reprojected_or_stale_raw_depth_frame_count", reprojectedOrStaleRawDepthFrames)
            .put("raw_pose_receipt_count", rawPoseReceiptCount)
            .put("raw_image_intrinsics_observation_count", rawIntrinsicsObservationCount)
            .put("raw_image_intrinsics_distinct_signature_count", intrinsicsSignatures.size)
            .put("first_raw_image_intrinsics", firstIntrinsics?.let(::intrinsicsJson) ?: JSONObject.NULL)
            .put("last_raw_image_intrinsics", lastIntrinsics?.let(::intrinsicsJson) ?: JSONObject.NULL)
            .put("tracking_state_counts", JSONObject(trackingStateCounts as Map<*, *>))
            .put("longest_tracking_run_frames", longestTrackingRun)
            .put("tracking_state_transition_count", trackingStateTransitions)
            .put("raw_pose_world_frame_stability", "EPHEMERAL_PER_FRAME")
            .put("first_tracking_translation_m", firstTrackingPose?.toList() ?: JSONObject.NULL)
            .put("last_tracking_translation_m", lastTrackingPose?.toList() ?: JSONObject.NULL)
            .put("authorization", JSONObject()
                .put("benchmark_only", true)
                .put("vio_candidate_observed", trackingFrames > 0)
                .put("vio_gate_open", false)
                .put("camera_intrinsics_observed", rawIntrinsicsObservationCount > 0)
                .put("camera_intrinsics_independently_verified", false)
                .put("metric_geometry_validated", false)
                .put("camera_to_body_extrinsics_validated", false)
                .put("production_authorized", false))
        Log.i(TAG, "USTRF_ARCORE_CAPABILITY_AUDIT_JSON $report")
        instrumentation.sendStatus(2, Bundle().apply { putString(REPORT_KEY, report.toString()) })

        assertTrue("ARCore did not expose any advancing frame timestamp", monotonicTimestampFrames > 0)
        if (depthSupported) assertNotNull("depth-supported audit did not retain report state", report)
    }

    private companion object {
        const val AVAILABILITY_ATTEMPTS = 10
        const val AVAILABILITY_RETRY_MS = 200L
        const val DEFAULT_FRAME_ATTEMPTS = 60
        const val MAX_FRAME_ATTEMPTS = 1_800
        const val FRAME_SETTLE_MS = 33L
        const val REPORT_KEY = "ustrf_arcore_capability_audit"
        const val TAG = "UstrfShadowBenchmark"
        const val ARCORE_CAMERA_FRAME = "arcore-camera-v1"

        fun intrinsicsJson(observation: UstrfArCoreRawCameraIntrinsicsObservation) = JSONObject()
            .put("image_width_px", observation.imageWidthPx)
            .put("image_height_px", observation.imageHeightPx)
            .put("focal_x_px", observation.focalXpx)
            .put("focal_y_px", observation.focalYpx)
            .put("principal_x_px", observation.principalXpx)
            .put("principal_y_px", observation.principalYpx)
    }
}
