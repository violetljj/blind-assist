package com.linnan.blindassist.ustrfbenchmark

import android.Manifest
import android.content.Context
import android.content.Intent
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraManager
import android.os.Build
import android.os.Bundle
import android.os.SystemClock
import android.util.AtomicFile
import android.util.Log
import android.view.Surface
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import androidx.test.rule.GrantPermissionRule
import com.google.ar.core.ArCoreApk
import com.google.ar.core.Config
import com.google.ar.core.Session
import com.google.ar.core.TrackingState
import com.linnan.blindassist.hftf.metricdepth.D45MetricDepthCanaryStatistics
import com.linnan.blindassist.hftf.metricdepth.D45MetricDepthFrameRegistrar
import com.linnan.blindassist.hftf.metricdepth.D45MetricDepthErrorSummary
import com.linnan.blindassist.hftf.metricdepth.MetricDepthHistoryResult
import com.linnan.blindassist.hftf.metricdepth.MetricDepthHistorySolver
import com.linnan.blindassist.hftf.metricdepth.MetricDepthSampleResult
import com.linnan.blindassist.hftf.metricdepth.MetricDepthTargetMeasurement
import com.linnan.blindassist.hftf.metricdepth.MetricDepthTargetSampler
import com.linnan.blindassist.model.DetectionSource
import com.linnan.blindassist.vision.DetectorExecutionBackend
import com.linnan.blindassist.vision.TfliteYoloDetector
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.io.InputStream
import java.security.MessageDigest

/**
 * One-distance D45 physical measurement runner.
 *
 * The controlled scene must contain exactly one detected person at the declared torso-plane to
 * camera-optical-centre distance. Each 1/2/3/5 m run writes one aggregate receipt and no images.
 */
@RunWith(AndroidJUnit4::class)
class D45ArCorePersonMeasurementCanaryTest {
    @get:Rule
    val cameraPermission: GrantPermissionRule = GrantPermissionRule.grant(Manifest.permission.CAMERA)

    @Test
    fun measuresSinglePersonAtDeclaredReferenceDistanceWithoutRuntimeAuthority() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val arguments = InstrumentationRegistry.getArguments()
        val referenceDistanceMeters = arguments.getString(ARG_REFERENCE_DISTANCE_METERS)
            ?.toFloatOrNull()
        assumeTrue(
            "D45 person measurement requires hftfD45ReferenceDistanceMeters=1|2|3|5",
            referenceDistanceMeters != null &&
                referenceDistanceMeters in
                D45MetricDepthCanaryStatistics.ALLOWED_REFERENCE_DISTANCES_METERS
        )
        val referenceDistance = requireNotNull(referenceDistanceMeters)
        val frameAttempts = arguments.getString(ARG_FRAME_ATTEMPTS)
            ?.toIntOrNull()
            ?.coerceIn(1, MAX_FRAME_ATTEMPTS)
            ?: DEFAULT_FRAME_ATTEMPTS
        val runId = arguments.getString(ARG_RUN_ID)
            ?.takeIf(RUN_ID::matches)
            ?: "manual-${referenceDistance.toInt()}m-${System.currentTimeMillis()}"
        val outputRoot = File(
            instrumentation.targetContext.getExternalFilesDir(null)
                ?: instrumentation.targetContext.filesDir,
            "$OUTPUT_PARENT/${referenceDistance.toInt()}m/$runId"
        )
        check(outputRoot.mkdirs() || outputRoot.isDirectory) {
            "cannot create D45 person measurement output directory"
        }
        val outputFile = File(outputRoot, OUTPUT_FILE)
        val startedAtMs = System.currentTimeMillis()
        val detector = TfliteYoloDetector(
            context = instrumentation.targetContext,
            executionBackend = DetectorExecutionBackend.CPU_XNNPACK
        )
        val detectorReadyAtStart = detector.isReady
        val detectorStatusAtStart = detector.statusMessage
        val modelSha256 = assetSha256(
            instrumentation.targetContext,
            TfliteYoloDetector.MODEL_ASSET
        )
        val targetApk = File(instrumentation.targetContext.applicationInfo.sourceDir)
        val instrumentationApk = File(instrumentation.context.applicationInfo.sourceDir)
        val targetApkSha256 = fileSha256(targetApk)
        val instrumentationApkSha256 = fileSha256(instrumentationApk)
        val activity = instrumentation.startActivitySync(
            Intent(instrumentation.targetContext, UstrfArCoreBenchmarkActivity::class.java)
                .putExtra(
                    UstrfArCoreBenchmarkActivity.EXTRA_STATUS_TEXT,
                    "D45 metric-depth measurement active\\n" +
                        "Keep exactly one person torso at ${referenceDistance.toInt()} m.\\n" +
                        "Gently translate the phone to maintain ARCore depth; no navigation " +
                        "output will be issued."
                )
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
        var rawObservations = 0
        var registrationObservations = 0
        var detectorFrames = 0
        var exactSinglePersonFrames = 0
        var zeroPersonFrames = 0
        var multiplePersonFrames = 0
        var eligibleHistoryWindows = 0
        var availableHistoryForecasts = 0
        val acceptedMeasurements = mutableListOf<MetricDepthTargetMeasurement>()
        val sourceToMeasurementLatencyMs = mutableListOf<Float>()
        val yuvLatencyMs = mutableListOf<Float>()
        val detectorLatencyMs = mutableListOf<Float>()
        val failures = linkedMapOf<String, Int>()
        val sampleFailures = linkedMapOf<String, Int>()
        val historyFailures = linkedMapOf<String, Int>()
        val transformIds = linkedMapOf<String, Int>()
        var captureError: String? = null
        var session: Session? = null
        var cameraId: String? = null
        var detectorRotationDegrees: Int? = null
        val displayRotation = activity.display?.rotation ?: Surface.ROTATION_0
        val displayWidth = activity.window.decorView.width.coerceAtLeast(1)
        val displayHeight = activity.window.decorView.height.coerceAtLeast(1)
        val rawAdapter = D45ArCoreRawDepthObservationAdapter()
        val registrationAdapter = D45ArCoreMetricDepthRegistrationAdapter()
        val detectorAdapter = D45ArCorePersonDetectorAdapter(detector)
        val sampler = MetricDepthTargetSampler()
        val historySolver = MetricDepthHistorySolver()

        try {
            if (
                availability == ArCoreApk.Availability.SUPPORTED_INSTALLED &&
                detectorReadyAtStart
            ) {
                activity.runOnGlThreadAndWait(
                    timeoutSeconds = maxOf(
                        MIN_GL_TIMEOUT_SECONDS,
                        frameAttempts * MAX_EXPECTED_FRAME_MS / 1_000L +
                            GL_TIMEOUT_MARGIN_SECONDS
                    )
                ) {
                    session = Session(activity).also { created ->
                        rawDepthSupported =
                            created.isDepthModeSupported(Config.DepthMode.RAW_DEPTH_ONLY)
                        if (rawDepthSupported) {
                            created.configure(
                                Config(created).apply {
                                    depthMode = Config.DepthMode.RAW_DEPTH_ONLY
                                    updateMode = Config.UpdateMode.LATEST_CAMERA_IMAGE
                                }
                            )
                            created.setDisplayGeometry(
                                displayRotation,
                                displayWidth,
                                displayHeight
                            )
                            cameraId = created.cameraConfig.cameraId
                            val characteristics = (
                                activity.getSystemService(Context.CAMERA_SERVICE) as CameraManager
                                ).getCameraCharacteristics(requireNotNull(cameraId))
                            detectorRotationDegrees = detectorRotationDegrees(
                                sensorOrientationDegrees =
                                    characteristics.get(
                                        CameraCharacteristics.SENSOR_ORIENTATION
                                    ),
                                targetRotation = DETECTOR_ANALYSIS_TARGET_ROTATION,
                                lensFacing =
                                    characteristics.get(CameraCharacteristics.LENS_FACING)
                            )
                            created.setCameraTextureName(activity.cameraTextureName())
                            created.resume()
                        }
                    }
                    try {
                        val rotation = detectorRotationDegrees
                        if (rawDepthSupported && rotation != null) {
                            repeat(frameAttempts) { frameId ->
                                val frame = requireNotNull(session).update()
                                attemptedUpdates++
                                if (frame.camera.trackingState != TrackingState.TRACKING) {
                                    failures.increment("CAMERA_NOT_TRACKING")
                                    Thread.sleep(FRAME_SETTLE_MS)
                                    return@repeat
                                }
                                trackingFrames++
                                val raw = when (
                                    val result = rawAdapter.observe(frame, frameId.toLong())
                                ) {
                                    is D45ArCoreRawDepthObservationResult.Available -> {
                                        rawObservations++
                                        result.observation
                                    }
                                    is D45ArCoreRawDepthObservationResult.Unavailable -> {
                                        failures.increment("RAW:${result.failure.name}")
                                        Thread.sleep(FRAME_SETTLE_MS)
                                        return@repeat
                                    }
                                }
                                val registration = when (
                                    val result = registrationAdapter.observe(
                                        frame,
                                        raw,
                                        rotation
                                    )
                                ) {
                                    is D45ArCoreRegistrationObservationResult.Available -> {
                                        registrationObservations++
                                        transformIds.increment(
                                            result.observation.transform.transformId
                                        )
                                        result.observation
                                    }
                                    is D45ArCoreRegistrationObservationResult.Unavailable -> {
                                        failures.increment("REGISTRATION:${result.failure.name}")
                                        Thread.sleep(FRAME_SETTLE_MS)
                                        return@repeat
                                    }
                                }
                                val detectorResult = when (
                                    val result = detectorAdapter.observe(frame, raw, rotation)
                                ) {
                                    is D45ArCorePersonDetectorResult.Available -> {
                                        detectorFrames++
                                        yuvLatencyMs += result.yuvToRgbaLatencyNs /
                                            NANOS_PER_MILLISECOND_FLOAT
                                        detectorLatencyMs += result.detectorLatencyNs /
                                            NANOS_PER_MILLISECOND_FLOAT
                                        result
                                    }
                                    is D45ArCorePersonDetectorResult.Unavailable -> {
                                        failures.increment("DETECTOR:${result.failure.name}")
                                        Thread.sleep(FRAME_SETTLE_MS)
                                        return@repeat
                                    }
                                }
                                val persons = detectorResult.detectorFrame.detections.filter {
                                    it.label == PERSON_LABEL &&
                                        it.source == DetectionSource.OBJECT_DETECTOR
                                }
                                if (persons.isEmpty()) {
                                    zeroPersonFrames++
                                    Thread.sleep(FRAME_SETTLE_MS)
                                    return@repeat
                                }
                                if (persons.size != 1) {
                                    multiplePersonFrames++
                                    Thread.sleep(FRAME_SETTLE_MS)
                                    return@repeat
                                }
                                exactSinglePersonFrames++
                                val registeredFrame = try {
                                    D45MetricDepthFrameRegistrar.register(
                                        frame = raw,
                                        registrationObservation = registration,
                                        validUntilNs =
                                            raw.sourceFrame.capturedAtNs +
                                                MAXIMUM_RECEIPT_AGE_NS
                                    )
                                } catch (error: IllegalArgumentException) {
                                    failures.increment("FRAME_REGISTRAR_REJECTED")
                                    Thread.sleep(FRAME_SETTLE_MS)
                                    return@repeat
                                }
                                when (
                                    val sample = sampler.sample(
                                        frame = registeredFrame,
                                        detection = persons.single(),
                                        targetKey = TARGET_KEY,
                                        observedAtNs = detectorResult.producedAtNs
                                    )
                                ) {
                                    is MetricDepthSampleResult.Available -> {
                                        val measurement = sample.measurement.copy(
                                            producedAtNs = SystemClock.elapsedRealtimeNanos()
                                        )
                                        acceptedMeasurements += measurement
                                        sourceToMeasurementLatencyMs +=
                                            (measurement.producedAtNs -
                                                measurement.capturedAtNs) /
                                                NANOS_PER_MILLISECOND_FLOAT
                                        if (acceptedMeasurements.size >= HISTORY_SIZE) {
                                            eligibleHistoryWindows++
                                            when (
                                                val history = historySolver.forecast(
                                                    acceptedMeasurements.takeLast(HISTORY_SIZE)
                                                )
                                            ) {
                                                is MetricDepthHistoryResult.Available ->
                                                    availableHistoryForecasts++
                                                is MetricDepthHistoryResult.Unavailable ->
                                                    historyFailures.increment(
                                                        history.failure.name
                                                    )
                                            }
                                        }
                                    }
                                    is MetricDepthSampleResult.Unavailable ->
                                        sampleFailures.increment(sample.failure.name)
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
            detector.close()
            activity.runOnUiThread { activity.finish() }
        }

        val errorSummary = D45MetricDepthCanaryStatistics.summarize(
            referenceDistance,
            acceptedMeasurements
        )
        val terminal = when {
            availability != ArCoreApk.Availability.SUPPORTED_INSTALLED ->
                "NOT_EVALUABLE_ARCORE_UNAVAILABLE"
            !detectorReadyAtStart ->
                "NOT_EVALUABLE_DETECTOR_UNAVAILABLE"
            captureError != null ->
                "NOT_EVALUABLE_CAPTURE_ERROR"
            !rawDepthSupported ->
                "NOT_EVALUABLE_RAW_DEPTH_UNSUPPORTED"
            detectorRotationDegrees == null ->
                "NOT_EVALUABLE_DETECTOR_ROTATION_UNAVAILABLE"
            rawObservations == 0 ->
                "NOT_EVALUABLE_NO_RAW_DEPTH_OBSERVATIONS"
            registrationObservations == 0 ->
                "NOT_EVALUABLE_NO_REGISTRATION_OBSERVATIONS"
            detectorFrames == 0 ->
                "NOT_EVALUABLE_NO_DETECTOR_FRAMES"
            exactSinglePersonFrames == 0 ->
                "NOT_EVALUABLE_NO_UNAMBIGUOUS_PERSON_FRAMES"
            acceptedMeasurements.isEmpty() ->
                "DISTANCE_MEASUREMENT_OBSERVED_NO_ACCEPTED_SAMPLES"
            acceptedMeasurements.size < MINIMUM_ACCEPTED_PER_DISTANCE ->
                "DISTANCE_MEASUREMENT_OBSERVED_BELOW_ACCEPTANCE_COUNT"
            else ->
                "DISTANCE_MEASUREMENT_OBSERVED"
        }
        val report = JSONObject()
            .put("schema", REPORT_SCHEMA)
            .put("run_id", runId)
            .put("terminal", terminal)
            .put("capture_started_at_epoch_ms", startedAtMs)
            .put("capture_finished_at_epoch_ms", System.currentTimeMillis())
            .put("capture_error", captureError ?: JSONObject.NULL)
            .put("reference_distance", JSONObject()
                .put("meters", referenceDistance.toDouble())
                .put("definition", REFERENCE_DISTANCE_DEFINITION)
                .put("source", "OPERATOR_DECLARED_INSTRUMENTATION_ARGUMENT"))
            .put("device", JSONObject()
                .put("manufacturer", Build.MANUFACTURER)
                .put("brand", Build.BRAND)
                .put("model", Build.MODEL)
                .put("device", Build.DEVICE)
                .put("build_fingerprint", Build.FINGERPRINT)
                .put("android_sdk_int", Build.VERSION.SDK_INT)
                .put("android_release", Build.VERSION.RELEASE))
            .put("build", JSONObject()
                .put("target_apk_bytes", targetApk.length())
                .put("target_apk_sha256", targetApkSha256)
                .put("instrumentation_apk_bytes", instrumentationApk.length())
                .put("instrumentation_apk_sha256", instrumentationApkSha256))
            .put("source", JSONObject()
                .put("arcore_availability", availability.name)
                .put("arcore_sdk_dependency_version", ARCORE_SDK_VERSION)
                .put("raw_depth_only_supported", rawDepthSupported)
                .put("camera_id", cameraId ?: JSONObject.NULL)
                .put(
                    "detector_rotation_degrees",
                    detectorRotationDegrees ?: JSONObject.NULL
                ))
            .put("detector", JSONObject()
                .put("model_asset", TfliteYoloDetector.MODEL_ASSET)
                .put("model_sha256", modelSha256)
                .put("backend", DetectorExecutionBackend.CPU_XNNPACK.wireName)
                .put("ready", detectorReadyAtStart)
                .put("status", detectorStatusAtStart)
                .put("selection_contract", "EXACTLY_ONE_PERSON_DETECTION"))
            .put("coverage", JSONObject()
                .put("requested_frame_attempts", frameAttempts)
                .put("attempted_session_updates", attemptedUpdates)
                .put("tracking_frame_count", trackingFrames)
                .put("raw_observation_count", rawObservations)
                .put("registration_observation_count", registrationObservations)
                .put("detector_frame_count", detectorFrames)
                .put("exact_single_person_frame_count", exactSinglePersonFrames)
                .put("zero_person_frame_count", zeroPersonFrames)
                .put("multiple_person_frame_count", multiplePersonFrames)
                .put("accepted_measurement_count", acceptedMeasurements.size)
                .put(
                    "accepted_person_coverage",
                    ratioOrNull(acceptedMeasurements.size, exactSinglePersonFrames)
                )
                .put("failure_counts", JSONObject(failures as Map<*, *>))
                .put("sample_failure_counts", JSONObject(sampleFailures as Map<*, *>)))
            .put("metric_error", errorSummary.toJson())
            .put("bounded_measurement_values", JSONObject()
                .put(
                    "optical_axis_depth_m",
                    JSONArray(
                        acceptedMeasurements.map {
                            it.opticalAxisDepthMeters.toDouble()
                        }
                    )
                )
                .put(
                    "source_to_measurement_latency_ms",
                    JSONArray(sourceToMeasurementLatencyMs.map(Float::toDouble))
                )
                .put("maximum_value_count", MAX_FRAME_ATTEMPTS))
            .put("latency_ms", JSONObject()
                .put("source_to_measurement_p50", percentileOrNull(
                    sourceToMeasurementLatencyMs,
                    0.50f
                ))
                .put("source_to_measurement_p95", percentileOrNull(
                    sourceToMeasurementLatencyMs,
                    0.95f
                ))
                .put("yuv_to_rgba_p50", percentileOrNull(yuvLatencyMs, 0.50f))
                .put("yuv_to_rgba_p95", percentileOrNull(yuvLatencyMs, 0.95f))
                .put("detector_p50", percentileOrNull(detectorLatencyMs, 0.50f))
                .put("detector_p95", percentileOrNull(detectorLatencyMs, 0.95f)))
            .put("history", JSONObject()
                .put("eligible_window_count", eligibleHistoryWindows)
                .put("available_forecast_count", availableHistoryForecasts)
                .put(
                    "availability",
                    ratioOrNull(availableHistoryForecasts, eligibleHistoryWindows)
                )
                .put("failure_counts", JSONObject(historyFailures as Map<*, *>)))
            .put("registration", JSONObject()
                .put("distinct_transform_id_count", transformIds.size)
                .put("transform_id_counts", JSONObject(transformIds as Map<*, *>))
                .put("external_alignment_verified_by_metric_error", false))
            .put("evidence_boundary", JSONObject()
                .put("benchmark_only", true)
                .put("app_runtime_involved", false)
                .put("risk_feedback_invocation_count", 0)
                .put("raster_pixels_persisted", false)
                .put("camera_images_persisted", false)
                .put("person_boxes_persisted", false)
                .put("operator_distance_independently_verified", false)
                .put("event_outcome_evaluated", false)
                .put("navigation_output_issued", false)
                .put("production_authorized", false))

        writeAtomicUtf8(outputFile, report.toString(2))
        Log.i(TAG, "HFTF_D45_PERSON_MEASUREMENT_CANARY_JSON $report")
        instrumentation.sendStatus(2, Bundle().apply {
            putString(REPORT_KEY, report.toString())
            putString(REPORT_PATH_KEY, outputFile.absolutePath)
        })

        assertTrue("D45 person aggregate receipt was not written", outputFile.isFile)
        assertTrue(
            "D45 person receipt exceeded its small-control-plane ceiling",
            outputFile.length() in 1..MAX_RECEIPT_BYTES
        )
    }

    private fun MutableMap<String, Int>.increment(key: String) {
        this[key] = (this[key] ?: 0) + 1
    }

    private fun D45MetricDepthErrorSummary.toJson() = JSONObject()
        .put("accepted_observation_count", acceptedObservationCount)
        .put(
            "absolute_error_median_m",
            absoluteErrorMedianMeters?.toDouble() ?: JSONObject.NULL
        )
        .put(
            "absolute_error_p90_m",
            absoluteErrorP90Meters?.toDouble() ?: JSONObject.NULL
        )
        .put(
            "relative_error_median",
            relativeErrorMedian?.toDouble() ?: JSONObject.NULL
        )

    private fun percentileOrNull(values: List<Float>, percentile: Float): Any =
        D45MetricDepthCanaryStatistics.percentileOrNull(values.sorted(), percentile)
            ?.toDouble()
            ?: JSONObject.NULL

    private fun ratioOrNull(numerator: Int, denominator: Int): Any =
        if (denominator == 0) JSONObject.NULL else numerator.toDouble() / denominator

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

    private fun assetSha256(context: Context, assetName: String): String {
        return context.assets.open(assetName).use(::sha256)
    }

    private fun fileSha256(file: File): String {
        check(file.isFile && file.length() > 0L) {
            "D45 build binding APK is unavailable"
        }
        return file.inputStream().buffered().use(::sha256)
    }

    private fun sha256(input: InputStream): String {
        val digest = MessageDigest.getInstance("SHA-256")
        input.use {
            val buffer = ByteArray(HASH_BUFFER_BYTES)
            while (true) {
                val count = it.read(buffer)
                if (count < 0) break
                digest.update(buffer, 0, count)
            }
        }
        return digest.digest().joinToString("") { "%02x".format(it) }
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
        const val ARG_REFERENCE_DISTANCE_METERS = "hftfD45ReferenceDistanceMeters"
        const val ARG_FRAME_ATTEMPTS = "hftfD45PersonFrameAttempts"
        const val ARG_RUN_ID = "hftfD45PersonRunId"
        const val DEFAULT_FRAME_ATTEMPTS = 900
        const val MAX_FRAME_ATTEMPTS = 1_800
        const val MINIMUM_ACCEPTED_PER_DISTANCE = 20
        const val HISTORY_SIZE = 7
        const val MAXIMUM_RECEIPT_AGE_NS = 150_000_000L
        const val AVAILABILITY_ATTEMPTS = 10
        const val AVAILABILITY_RETRY_MS = 200L
        const val FRAME_SETTLE_MS = 20L
        const val MAX_EXPECTED_FRAME_MS = 150L
        const val MIN_GL_TIMEOUT_SECONDS = 30L
        const val GL_TIMEOUT_MARGIN_SECONDS = 60L
        const val NANOS_PER_MILLISECOND_FLOAT = 1_000_000f
        const val MAX_RECEIPT_BYTES = 256L * 1_024L
        const val HASH_BUFFER_BYTES = 64 * 1_024
        const val DETECTOR_ANALYSIS_TARGET_ROTATION = Surface.ROTATION_0
        const val ARCORE_SDK_VERSION = "1.33.0"
        const val PERSON_LABEL = "person"
        const val TARGET_KEY = "manual-single-person"
        const val REFERENCE_DISTANCE_DEFINITION =
            "PERSON_TORSO_PLANE_TO_CAMERA_OPTICAL_CENTER"
        const val OUTPUT_PARENT = "hftf-d45/person-measurement-r0"
        const val OUTPUT_FILE = "summary.json"
        const val REPORT_SCHEMA = "blindassist_hftf_d45_person_measurement_canary_v1"
        const val REPORT_KEY = "hftf_d45_person_measurement_canary"
        const val REPORT_PATH_KEY = "hftf_d45_person_measurement_canary_path"
        const val TAG = "UstrfShadowBenchmark"
        val RUN_ID = Regex("[A-Za-z0-9][A-Za-z0-9._-]{0,79}")
    }
}
