package com.linnan.blindassist.ustrfbenchmark

import android.Manifest
import android.content.Intent
import android.os.Build
import android.os.Bundle
import android.util.AtomicFile
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import androidx.test.rule.GrantPermissionRule
import com.google.ar.core.ArCoreApk
import com.google.ar.core.Config
import com.google.ar.core.Session
import com.google.ar.core.TrackingState
import org.json.JSONObject
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File

/**
 * D45 benchmark-only source canary.
 *
 * This test answers only whether one device/run can expose, timestamp-bind, and decode ARCore raw
 * depth plus confidence. It stores no raster and performs no registration, person measurement,
 * event evaluation, navigation, or production authorization.
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
        var attemptedUpdates = 0
        var trackingFrames = 0
        var strictlyAdvancingTimestamps = 0
        var lastTimestampNs = -1L
        val distinctTimestamps = linkedSetOf<Long>()
        var availableObservations = 0
        var totalDecodedPixels = 0L
        var totalValidPixels = 0L
        val coverageValues = mutableListOf<Double>()
        val latencyValuesMs = mutableListOf<Double>()
        val dimensions = linkedMapOf<String, Int>()
        val trackingStates = linkedMapOf<String, Int>()
        val failures = linkedMapOf<String, Int>()
        var captureError: String? = null
        var session: Session? = null
        val adapter = D45ArCoreRawDepthObservationAdapter()

        try {
            if (availability == ArCoreApk.Availability.SUPPORTED_INSTALLED) {
                activity.runOnGlThreadAndWait(
                    timeoutSeconds = maxOf(
                        MIN_GL_TIMEOUT_SECONDS,
                        frameAttempts * FRAME_SETTLE_MS / 1_000L + GL_TIMEOUT_MARGIN_SECONDS
                    )
                ) {
                    session = Session(activity).also { created ->
                        rawDepthSupported =
                            created.isDepthModeSupported(Config.DepthMode.RAW_DEPTH_ONLY)
                        if (rawDepthSupported) {
                            val config = Config(created).apply {
                                depthMode = Config.DepthMode.RAW_DEPTH_ONLY
                                updateMode = Config.UpdateMode.LATEST_CAMERA_IMAGE
                            }
                            created.configure(config)
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
                                            val validPixels = raster.depthMillimeters.indices.count { index ->
                                                raster.depthMillimeters[index] in
                                                    MIN_VALID_DEPTH_MM..MAX_VALID_DEPTH_MM &&
                                                    raster.confidence[index] >= MIN_VALID_CONFIDENCE
                                            }
                                            totalDecodedPixels += raster.depthMillimeters.size
                                            totalValidPixels += validPixels
                                            coverageValues +=
                                                validPixels.toDouble() / raster.depthMillimeters.size
                                            latencyValuesMs +=
                                                (observation.producedAtNs -
                                                    observation.sourceFrame.receivedAtNs) /
                                                    NANOS_PER_MILLISECOND
                                            dimensions.increment(
                                                "${raster.widthPx}x${raster.heightPx}"
                                            )
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
                .put("raw_depth_only_supported", rawDepthSupported))
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
                .put("motion_protocol", MOTION_PROTOCOL))
            .put("valid_pixel_contract", JSONObject()
                .put("minimum_depth_mm", MIN_VALID_DEPTH_MM)
                .put("maximum_depth_mm", MAX_VALID_DEPTH_MM)
                .put("minimum_confidence", MIN_VALID_CONFIDENCE.toDouble())
                .put("decoded_pixel_count", totalDecodedPixels)
                .put("valid_pixel_count", totalValidPixels)
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
            .put("evidence_boundary", JSONObject()
                .put("benchmark_only", true)
                .put("app_runtime_involved", false)
                .put("raster_pixels_persisted", false)
                .put("registration_verified", false)
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
        const val MOTION_PROTOCOL =
            "AUTONOMOUS_STATIONARY_OR_UNCONTROLLED_NO_USER_INSTRUCTION"
        const val OUTPUT_PARENT = "hftf-d45/raw-source-decoder-r0"
        const val OUTPUT_FILE = "summary.json"
        const val REPORT_SCHEMA = "blindassist_hftf_d45_raw_source_decoder_canary_v1"
        const val REPORT_KEY = "hftf_d45_raw_source_decoder_canary"
        const val REPORT_PATH_KEY = "hftf_d45_raw_source_decoder_canary_path"
        const val TAG = "UstrfShadowBenchmark"
        val RUN_ID = Regex("[A-Za-z0-9][A-Za-z0-9._-]{0,79}")
    }
}
