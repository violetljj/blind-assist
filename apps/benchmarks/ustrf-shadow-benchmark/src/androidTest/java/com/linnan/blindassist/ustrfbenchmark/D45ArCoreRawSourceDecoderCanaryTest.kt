package com.linnan.blindassist.ustrfbenchmark

import android.Manifest
import android.content.Context
import android.content.Intent
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraManager
import android.os.Build
import android.os.Bundle
import android.util.AtomicFile
import android.util.Log
import android.view.Surface
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import androidx.test.rule.GrantPermissionRule
import com.google.ar.core.ArCoreApk
import com.google.ar.core.CameraConfig
import com.google.ar.core.CameraConfigFilter
import com.google.ar.core.Config
import com.google.ar.core.Session
import com.google.ar.core.TrackingState
import org.json.JSONObject
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.util.EnumSet

/**
 * D45 benchmark-only source canary.
 *
 * This test answers only whether one device/run can expose, timestamp-bind, and decode ARCore raw
 * depth plus confidence, and whether ARCore's CPU-image -> depth mapping is affine-consistent for
 * the detector's explicit rotation. It stores no raster and performs no person measurement, event
 * evaluation, navigation, or production authorization.
 */
@RunWith(AndroidJUnit4::class)
class D45ArCoreRawSourceDecoderCanaryTest {
    @get:Rule
    val cameraPermission: GrantPermissionRule = GrantPermissionRule.grant(Manifest.permission.CAMERA)

    @Test
    fun observesRawDepthSourceWithoutConvertingAbsenceIntoAlgorithmFailure() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val arguments = InstrumentationRegistry.getArguments()
        val frameAttempts = arguments.getString(ARG_FRAME_ATTEMPTS)
            ?.toIntOrNull()
            ?.coerceIn(1, MAX_FRAME_ATTEMPTS)
            ?: DEFAULT_FRAME_ATTEMPTS
        val runId = arguments.getString(ARG_RUN_ID)
            ?.takeIf(RUN_ID::matches)
            ?: "autonomous-${System.currentTimeMillis()}"
        val motionProtocol = arguments.getString(ARG_MOTION_PROTOCOL)?.let { requested ->
            require(requested in ALLOWED_MOTION_PROTOCOLS) {
                "unsupported D45 raw-source motion protocol: $requested"
            }
            requested
        } ?: MOTION_PROTOCOL_AUTONOMOUS
        val outputRoot = File(
            instrumentation.targetContext.getExternalFilesDir(null)
                ?: instrumentation.targetContext.filesDir,
            "$OUTPUT_PARENT/$runId"
        )
        check(outputRoot.mkdirs() || outputRoot.isDirectory) {
            "cannot create D45 source canary output directory"
        }
        val outputFile = File(outputRoot, OUTPUT_FILE)
        val startedAtMs = System.currentTimeMillis()
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

        var rawDepthSupported = false
        var automaticDepthSupported = false
        var cameraConfigCount = 0
        var hardwareDepthCameraConfigCount = 0
        var attemptedUpdates = 0
        var trackingFrames = 0
        var strictlyAdvancingTimestamps = 0
        var lastTimestampNs = -1L
        val distinctTimestamps = linkedSetOf<Long>()
        var availableObservations = 0
        var totalDecodedPixels = 0L
        var totalNonzeroDepthPixels = 0L
        var totalInRangeDepthPixels = 0L
        var totalNonzeroConfidencePixels = 0L
        var totalThresholdConfidencePixels = 0L
        var totalValidPixels = 0L
        var maximumDecodedDepthMm = 0
        var maximumDecodedConfidence = 0f
        val coverageValues = mutableListOf<Double>()
        val latencyValuesMs = mutableListOf<Double>()
        val dimensions = linkedMapOf<String, Int>()
        val trackingStates = linkedMapOf<String, Int>()
        val failures = linkedMapOf<String, Int>()
        var captureError: String? = null
        var session: Session? = null
        val adapter = D45ArCoreRawDepthObservationAdapter()
        val registrationAdapter = D45ArCoreMetricDepthRegistrationAdapter()
        val registrationFailures = linkedMapOf<String, Int>()
        val registrationTransformIds = linkedMapOf<String, Int>()
        val registrationResidualsPx = mutableListOf<Double>()
        var registrationObservations = 0
        var cameraId: String? = null
        var sensorOrientationDegrees: Int? = null
        var lensFacing: Int? = null
        var detectorRotationDegrees: Int? = null
        val displayRotation = activity.display?.rotation ?: Surface.ROTATION_0
        val displayWidth = activity.window.decorView.width.coerceAtLeast(1)
        val displayHeight = activity.window.decorView.height.coerceAtLeast(1)

        try {
            if (availability == ArCoreApk.Availability.SUPPORTED_INSTALLED) {
                activity.runOnGlThreadAndWait(
                    timeoutSeconds = maxOf(
                        MIN_GL_TIMEOUT_SECONDS,
                        frameAttempts * FRAME_SETTLE_MS / 1_000L + GL_TIMEOUT_MARGIN_SECONDS
                    )
                ) {
                    session = Session(activity).also { created ->
                        automaticDepthSupported =
                            created.isDepthModeSupported(Config.DepthMode.AUTOMATIC)
                        rawDepthSupported =
                            created.isDepthModeSupported(Config.DepthMode.RAW_DEPTH_ONLY)
                        val cameraConfigs = created.getSupportedCameraConfigs(
                            CameraConfigFilter(created).setDepthSensorUsage(
                                EnumSet.allOf(CameraConfig.DepthSensorUsage::class.java)
                            )
                        )
                        cameraConfigCount = cameraConfigs.size
                        hardwareDepthCameraConfigCount = cameraConfigs.count {
                            it.depthSensorUsage == CameraConfig.DepthSensorUsage.REQUIRE_AND_USE
                        }
                        if (rawDepthSupported) {
                            val config = Config(created).apply {
                                depthMode = Config.DepthMode.RAW_DEPTH_ONLY
                                updateMode = Config.UpdateMode.LATEST_CAMERA_IMAGE
                            }
                            created.configure(config)
                            created.setDisplayGeometry(
                                displayRotation,
                                displayWidth,
                                displayHeight
                            )
                            cameraId = created.cameraConfig.cameraId
                            val characteristics = (
                                activity.getSystemService(Context.CAMERA_SERVICE) as CameraManager
                                ).getCameraCharacteristics(requireNotNull(cameraId))
                            sensorOrientationDegrees =
                                characteristics.get(CameraCharacteristics.SENSOR_ORIENTATION)
                            lensFacing = characteristics.get(CameraCharacteristics.LENS_FACING)
                            detectorRotationDegrees = detectorRotationDegrees(
                                sensorOrientationDegrees = sensorOrientationDegrees,
                                targetRotation = DETECTOR_ANALYSIS_TARGET_ROTATION,
                                lensFacing = lensFacing
                            )
                            created.setCameraTextureName(activity.cameraTextureName())
                            created.resume()
                        }
                    }
                    try {
                        if (rawDepthSupported) {
                            repeat(frameAttempts) { frameId ->
                                val frame = requireNotNull(session).update()
                                attemptedUpdates++
                                val trackingState = frame.camera.trackingState.name
                                trackingStates.increment(trackingState)
                                if (frame.timestamp > lastTimestampNs) {
                                    strictlyAdvancingTimestamps++
                                }
                                lastTimestampNs = maxOf(lastTimestampNs, frame.timestamp)
                                distinctTimestamps += frame.timestamp
                                if (frame.camera.trackingState == TrackingState.TRACKING) {
                                    trackingFrames++
                                    when (val result = adapter.observe(frame, frameId.toLong())) {
                                        is D45ArCoreRawDepthObservationResult.Available -> {
                                            availableObservations++
                                            val observation = result.observation
                                            val raster = observation.raster
                                            val nonzeroDepthPixels =
                                                raster.depthMillimeters.count { it > 0 }
                                            val inRangeDepthPixels =
                                                raster.depthMillimeters.count {
                                                    it in MIN_VALID_DEPTH_MM..MAX_VALID_DEPTH_MM
                                                }
                                            val nonzeroConfidencePixels =
                                                raster.confidence.count { it > 0f }
                                            val thresholdConfidencePixels =
                                                raster.confidence.count {
                                                    it >= MIN_VALID_CONFIDENCE
                                                }
                                            val validPixels = raster.depthMillimeters.indices.count { index ->
                                                raster.depthMillimeters[index] in
                                                    MIN_VALID_DEPTH_MM..MAX_VALID_DEPTH_MM &&
                                                    raster.confidence[index] >= MIN_VALID_CONFIDENCE
                                            }
                                            totalDecodedPixels += raster.depthMillimeters.size
                                            totalNonzeroDepthPixels += nonzeroDepthPixels
                                            totalInRangeDepthPixels += inRangeDepthPixels
                                            totalNonzeroConfidencePixels += nonzeroConfidencePixels
                                            totalThresholdConfidencePixels +=
                                                thresholdConfidencePixels
                                            totalValidPixels += validPixels
                                            maximumDecodedDepthMm = maxOf(
                                                maximumDecodedDepthMm,
                                                raster.depthMillimeters.maxOrNull() ?: 0
                                            )
                                            maximumDecodedConfidence = maxOf(
                                                maximumDecodedConfidence,
                                                raster.confidence.maxOrNull() ?: 0f
                                            )
                                            coverageValues +=
                                                validPixels.toDouble() / raster.depthMillimeters.size
                                            latencyValuesMs +=
                                                (observation.producedAtNs -
                                                    observation.sourceFrame.receivedAtNs) /
                                                    NANOS_PER_MILLISECOND
                                            dimensions.increment(
                                                "${raster.widthPx}x${raster.heightPx}"
                                            )
                                            val rotation = detectorRotationDegrees
                                            if (rotation == null) {
                                                registrationFailures.increment(
                                                    "DETECTOR_ROTATION_UNAVAILABLE"
                                                )
                                            } else {
                                                when (
                                                    val registration = registrationAdapter.observe(
                                                        frame = frame,
                                                        rawObservation = observation,
                                                        detectorRotationDegrees = rotation
                                                    )
                                                ) {
                                                    is D45ArCoreRegistrationObservationResult
                                                        .Available -> {
                                                        registrationObservations++
                                                        registrationTransformIds.increment(
                                                            registration.observation.transform
                                                                .transformId
                                                        )
                                                        registrationResidualsPx +=
                                                            registration.observation.transform
                                                                .maximumFitResidualPx.toDouble()
                                                    }

                                                    is D45ArCoreRegistrationObservationResult
                                                        .Unavailable -> {
                                                        val key = buildString {
                                                            append(registration.failure.name)
                                                            registration.registrationFailure?.let {
                                                                append(':')
                                                                append(it.name)
                                                            }
                                                        }
                                                        registrationFailures.increment(key)
                                                    }
                                                }
                                            }
                                        }

                                        is D45ArCoreRawDepthObservationResult.Unavailable ->
                                            failures.increment(result.failure.name)
                                    }
                                }
                                Thread.sleep(FRAME_SETTLE_MS)
                            }
                        }
                    } finally {
                        session?.pause()
                        session?.close()
                    }
                }
            }
        } catch (error: Throwable) {
            captureError = "${error.javaClass.name}:${error.message.orEmpty()}"
        } finally {
            activity.runOnUiThread { activity.finish() }
        }

        val terminal = when {
            availability != ArCoreApk.Availability.SUPPORTED_INSTALLED ->
                "NOT_EVALUABLE_ARCORE_UNAVAILABLE"
            captureError != null ->
                "NOT_EVALUABLE_CAPTURE_ERROR"
            !rawDepthSupported ->
                "NOT_EVALUABLE_RAW_DEPTH_UNSUPPORTED"
            trackingFrames == 0 ->
                "NOT_EVALUABLE_NO_TRACKING_FRAMES"
            availableObservations == 0 ->
                "NOT_EVALUABLE_NO_RAW_DEPTH_OBSERVATIONS"
            else ->
                "RAW_SOURCE_DECODER_OBSERVED"
        }
        val registrationTerminal = when {
            availableObservations == 0 ->
                "NOT_EVALUATED_NO_RAW_DEPTH_OBSERVATIONS"
            detectorRotationDegrees == null ->
                "NOT_EVALUABLE_DETECTOR_ROTATION_UNAVAILABLE"
            registrationObservations == 0 ->
                "NOT_EVALUABLE_NO_AFFINE_REGISTRATION_OBSERVATIONS"
            else ->
                "AFFINE_REGISTRATION_OBSERVED_DEVICE_ONLY"
        }
        val report = JSONObject()
            .put("schema", REPORT_SCHEMA)
            .put("run_id", runId)
            .put("capture_started_at_epoch_ms", startedAtMs)
            .put("capture_finished_at_epoch_ms", System.currentTimeMillis())
            .put("terminal", terminal)
            .put("capture_error", captureError ?: JSONObject.NULL)
            .put("package", instrumentation.targetContext.packageName)
            .put("capture_test", this::class.java.name)
            .put("device", JSONObject()
                .put("manufacturer", Build.MANUFACTURER)
                .put("brand", Build.BRAND)
                .put("model", Build.MODEL)
                .put("device", Build.DEVICE)
                .put("build_fingerprint", Build.FINGERPRINT)
                .put("android_sdk_int", Build.VERSION.SDK_INT)
                .put("android_release", Build.VERSION.RELEASE))
            .put("arcore", JSONObject()
                .put("availability", availability.name)
                .put("sdk_dependency_version", ARCORE_SDK_VERSION)
                .put("automatic_depth_supported", automaticDepthSupported)
                .put("raw_depth_only_supported", rawDepthSupported)
                .put("camera_config_count", cameraConfigCount)
                .put(
                    "hardware_depth_camera_config_count",
                    hardwareDepthCameraConfigCount
                )
                .put("camera_id", cameraId ?: JSONObject.NULL)
                .put(
                    "sensor_orientation_degrees",
                    sensorOrientationDegrees ?: JSONObject.NULL
                )
                .put("lens_facing", lensFacing ?: JSONObject.NULL)
                .put("display_rotation", displayRotation)
                .put("display_width_px", displayWidth)
                .put("display_height_px", displayHeight)
                .put(
                    "detector_analysis_target_rotation",
                    DETECTOR_ANALYSIS_TARGET_ROTATION
                )
                .put(
                    "detector_rotation_degrees",
                    detectorRotationDegrees ?: JSONObject.NULL
                ))
            .put("capture", JSONObject()
                .put("requested_frame_attempts", frameAttempts)
                .put("attempted_session_updates", attemptedUpdates)
                .put("tracking_frame_count", trackingFrames)
                .put("strictly_advancing_timestamp_count", strictlyAdvancingTimestamps)
                .put("distinct_frame_timestamp_count", distinctTimestamps.size)
                .put("available_raw_depth_observation_count", availableObservations)
                .put("tracking_state_counts", JSONObject(trackingStates as Map<*, *>))
                .put("unavailable_failure_counts", JSONObject(failures as Map<*, *>))
                .put("decoded_dimension_counts", JSONObject(dimensions as Map<*, *>))
                .put("motion_protocol", motionProtocol))
            .put("valid_pixel_contract", JSONObject()
                .put("minimum_depth_mm", MIN_VALID_DEPTH_MM)
                .put("maximum_depth_mm", MAX_VALID_DEPTH_MM)
                .put("minimum_confidence", MIN_VALID_CONFIDENCE.toDouble())
                .put("decoded_pixel_count", totalDecodedPixels)
                .put("nonzero_depth_pixel_count", totalNonzeroDepthPixels)
                .put("in_range_depth_pixel_count", totalInRangeDepthPixels)
                .put("nonzero_confidence_pixel_count", totalNonzeroConfidencePixels)
                .put(
                    "threshold_confidence_pixel_count",
                    totalThresholdConfidencePixels
                )
                .put("valid_pixel_count", totalValidPixels)
                .put("maximum_decoded_depth_mm", maximumDecodedDepthMm)
                .put(
                    "maximum_decoded_confidence",
                    maximumDecodedConfidence.toDouble()
                )
                .put(
                    "aggregate_valid_pixel_coverage",
                    ratioOrNull(totalValidPixels, totalDecodedPixels)
                )
                .put("per_frame_coverage_p50", percentileOrNull(coverageValues, 0.50))
                .put("per_frame_coverage_p95", percentileOrNull(coverageValues, 0.95)))
            .put("source_latency_ms", JSONObject()
                .put("sample_count", latencyValuesMs.size)
                .put("p50", percentileOrNull(latencyValuesMs, 0.50))
                .put("p95", percentileOrNull(latencyValuesMs, 0.95)))
            .put("registration_candidate", JSONObject()
                .put("terminal", registrationTerminal)
                .put("coordinate_contract", REGISTRATION_COORDINATE_CONTRACT)
                .put("observation_count", registrationObservations)
                .put(
                    "distinct_transform_id_count",
                    registrationTransformIds.size
                )
                .put(
                    "transform_id_counts",
                    JSONObject(registrationTransformIds as Map<*, *>)
                )
                .put(
                    "failure_counts",
                    JSONObject(registrationFailures as Map<*, *>)
                )
                .put(
                    "maximum_fit_residual_px_p50",
                    percentileOrNull(registrationResidualsPx, 0.50)
                )
                .put(
                    "maximum_fit_residual_px_p95",
                    percentileOrNull(registrationResidualsPx, 0.95)
                )
                .put("external_alignment_verified", false)
                .put("person_registration_verified", false))
            .put("evidence_boundary", JSONObject()
                .put("benchmark_only", true)
                .put("app_runtime_involved", false)
                .put("raster_pixels_persisted", false)
                .put(
                    "affine_registration_candidate_observed",
                    registrationObservations > 0
                )
                .put("external_registration_verified", false)
                .put("person_measurement_performed", false)
                .put("event_outcome_evaluated", false)
                .put("navigation_output_issued", false)
                .put("production_authorized", false)
                .put("zero_observations_are_algorithm_negative", false))

        writeAtomicUtf8(outputFile, report.toString(2))
        Log.i(TAG, "HFTF_D45_RAW_SOURCE_DECODER_CANARY_JSON $report")
        instrumentation.sendStatus(2, Bundle().apply {
            putString(REPORT_KEY, report.toString())
            putString(REPORT_PATH_KEY, outputFile.absolutePath)
        })

        assertTrue("D45 aggregate receipt was not written", outputFile.isFile)
        assertTrue(
            "D45 receipt unexpectedly exceeded its small-control-plane ceiling",
            outputFile.length() in 1..MAX_RECEIPT_BYTES
        )
    }

    private fun MutableMap<String, Int>.increment(key: String) {
        this[key] = (this[key] ?: 0) + 1
    }

    private fun ratioOrNull(numerator: Long, denominator: Long): Any =
        if (denominator == 0L) JSONObject.NULL else numerator.toDouble() / denominator

    private fun percentileOrNull(values: List<Double>, percentile: Double): Any {
        if (values.isEmpty()) return JSONObject.NULL
        val sorted = values.sorted()
        val index = ((sorted.size - 1) * percentile).toInt().coerceIn(0, sorted.lastIndex)
        return sorted[index]
    }

    private fun detectorRotationDegrees(
        sensorOrientationDegrees: Int?,
        targetRotation: Int,
        lensFacing: Int?
    ): Int? {
        val sensor = sensorOrientationDegrees ?: return null
        val targetDegrees = when (targetRotation) {
            Surface.ROTATION_0 -> 0
            Surface.ROTATION_90 -> 90
            Surface.ROTATION_180 -> 180
            Surface.ROTATION_270 -> 270
            else -> return null
        }
        return if (lensFacing == CameraCharacteristics.LENS_FACING_FRONT) {
            (sensor + targetDegrees) % 360
        } else {
            (sensor - targetDegrees + 360) % 360
        }.takeIf { it in setOf(0, 90, 180, 270) }
    }

    private fun writeAtomicUtf8(file: File, content: String) {
        val atomicFile = AtomicFile(file)
        val stream = atomicFile.startWrite()
        try {
            stream.write(content.toByteArray(Charsets.UTF_8))
            stream.flush()
            atomicFile.finishWrite(stream)
        } catch (error: Throwable) {
            atomicFile.failWrite(stream)
            throw error
        }
    }

    private companion object {
        const val ARG_FRAME_ATTEMPTS = "hftfD45RawSourceFrameAttempts"
        const val ARG_RUN_ID = "hftfD45RawSourceRunId"
        const val ARG_MOTION_PROTOCOL = "hftfD45RawSourceMotionProtocol"
        const val DEFAULT_FRAME_ATTEMPTS = 300
        const val MAX_FRAME_ATTEMPTS = 1_800
        const val AVAILABILITY_ATTEMPTS = 10
        const val AVAILABILITY_RETRY_MS = 200L
        const val FRAME_SETTLE_MS = 33L
        const val MIN_GL_TIMEOUT_SECONDS = 20L
        const val GL_TIMEOUT_MARGIN_SECONDS = 20L
        const val MIN_VALID_DEPTH_MM = 200
        const val MAX_VALID_DEPTH_MM = 20_000
        const val MIN_VALID_CONFIDENCE = 0.50f
        const val NANOS_PER_MILLISECOND = 1_000_000.0
        const val MAX_RECEIPT_BYTES = 256L * 1_024L
        const val ARCORE_SDK_VERSION = "1.33.0"
        const val MOTION_PROTOCOL_AUTONOMOUS =
            "AUTONOMOUS_STATIONARY_OR_UNCONTROLLED_NO_USER_INSTRUCTION"
        const val MOTION_PROTOCOL_OPERATOR_CONTROLLED =
            "OPERATOR_CONTROLLED_TRANSLATION_TEXTURED_SCENE"
        val ALLOWED_MOTION_PROTOCOLS = setOf(
            MOTION_PROTOCOL_AUTONOMOUS,
            MOTION_PROTOCOL_OPERATOR_CONTROLLED
        )
        const val REGISTRATION_COORDINATE_CONTRACT =
            "DETECTOR_DISPLAY_TO_CPU_IMAGE_TO_TEXTURE_NORMALIZED_RAW_DEPTH"
        const val DETECTOR_ANALYSIS_TARGET_ROTATION = Surface.ROTATION_0
        const val OUTPUT_PARENT = "hftf-d45/raw-source-registration-r0"
        const val OUTPUT_FILE = "summary.json"
        const val REPORT_SCHEMA =
            "blindassist_hftf_d45_raw_source_registration_canary_v1"
        const val REPORT_KEY = "hftf_d45_raw_source_decoder_canary"
        const val REPORT_PATH_KEY = "hftf_d45_raw_source_decoder_canary_path"
        const val TAG = "UstrfShadowBenchmark"
        val RUN_ID = Regex("[A-Za-z0-9][A-Za-z0-9._-]{0,79}")
    }
}
