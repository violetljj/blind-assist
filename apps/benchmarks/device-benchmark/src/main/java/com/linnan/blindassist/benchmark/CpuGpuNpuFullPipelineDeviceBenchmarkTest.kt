package com.linnan.blindassist.benchmark

import android.content.Intent
import android.content.IntentFilter
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Build
import android.os.PowerManager
import android.os.SystemClock
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.linnan.blindassist.alert.AlertProfile
import com.linnan.blindassist.alert.AssistScenario
import com.linnan.blindassist.feedback.FeedbackDecision
import com.linnan.blindassist.feedback.FeedbackGateway
import com.linnan.blindassist.feedback.FeedbackPlanner
import com.linnan.blindassist.feedback.FeedbackReason
import com.linnan.blindassist.model.Detection
import com.linnan.blindassist.model.FrameSize
import com.linnan.blindassist.risk.DenseSemanticMask
import com.linnan.blindassist.risk.RiskAnalyzer
import com.linnan.blindassist.risk.RiskResult
import com.linnan.blindassist.risk.TraversabilitySegmentationAnalyzer
import com.linnan.blindassist.session.AssistDecisionKernel
import com.linnan.blindassist.session.AssistEngine
import com.linnan.blindassist.session.DetectorMetrics
import com.linnan.blindassist.vision.DetectorBackendPolicy
import com.linnan.blindassist.vision.DetectorExecutionBackend
import com.linnan.blindassist.vision.RuntimeObjectDetectorFactory
import com.linnan.blindassist.vision.TfliteYoloDetector
import com.qualcomm.qti.QnnDelegate
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import java.io.File
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

/**
 * Benchmark-only CPU/GPU/QNN HTP comparison through the production
 * TfliteYoloDetector decode/NMS and AssistDecisionKernel risk path.
 */
@RunWith(AndroidJUnit4::class)
class CpuGpuNpuFullPipelineDeviceBenchmarkTest {
    private val instrumentation = InstrumentationRegistry.getInstrumentation()
    private val testContext = instrumentation.context
    private val targetContext = instrumentation.targetContext
    private val arguments = InstrumentationRegistry.getArguments()

    @Test
    fun compareFullPipelineOnHundredEvalSetImages() {
        val imageLimit = arguments.getString(ARG_IMAGE_LIMIT)
            ?.toIntOrNull()
            ?.coerceIn(1, MAX_IMAGE_LIMIT)
            ?: DEFAULT_IMAGE_LIMIT
        val imagePaths = loadImagePaths().take(imageLimit)
        assertEquals("The requested EvalSet image count must be available", imageLimit, imagePaths.size)

        val temperatureBefore = batteryTemperatureC()
        val backendResults = VALIDATION_BACKENDS.associateWith { backend ->
            runBackend(backend, imagePaths)
        }
        val temperatureAfter = batteryTemperatureC()
        val cpu = requireNotNull(backendResults[BackendId.CPU])
        val comparisons = JSONArray()
        listOf(BackendId.GPU, BackendId.NPU, BackendId.PRODUCTION_ROUTE).forEach { backend ->
            comparisons.put(compareBackendResults(cpu, requireNotNull(backendResults[backend])))
        }

        val report = JSONObject()
            .put("schema", FULL_REPORT_SCHEMA)
            .put("disposition", "FORMAL_PRODUCTION_ROUTE_VALIDATION")
            .put("backend_policy", backendPolicyJson())
            .put("device", deviceJson())
            .put("protocol", JSONObject()
                .put("image_count", imagePaths.size)
                .put("dataset", "blindassist-evalset-20260527-impl")
                .put("decode", "production_TfliteYoloDetector")
                .put("nms", "production_YoloOutputDecoder")
                .put("risk", "production_AssistDecisionKernel")
                .put("independent_images_reset_between_frames", true)
                .put("warmup_images_per_backend", WARMUP_IMAGES))
            .put("temperature_c", JSONObject()
                .put("before", temperatureBefore)
                .put("after", temperatureAfter))
            .put("backends", JSONArray(VALIDATION_BACKENDS.map { requireNotNull(backendResults[it]).toJson() }))
            .put("comparisons_against_cpu", comparisons)

        writeArtifact(FULL_REPORT_FILE, report)
        assertTrue("One or more backends produced frame failures", backendResults.values.all { it.failures.isEmpty() })
    }

    @Test
    fun soakSelectedBackendForTenMinutes() {
        val backend = BackendId.valueOf(
            (arguments.getString(ARG_BACKEND) ?: BackendId.NPU.name).uppercase()
        )
        val durationSeconds = arguments.getString(ARG_DURATION_SECONDS)
            ?.toIntOrNull()
            ?.coerceIn(MIN_SOAK_SECONDS, MAX_SOAK_SECONDS)
            ?: DEFAULT_SOAK_SECONDS
        val targetFps = arguments.getString(ARG_TARGET_FPS)
            ?.toIntOrNull()
            ?.coerceIn(1, MAX_TARGET_FPS)
            ?: DEFAULT_TARGET_FPS
        val imagePaths = loadImagePaths().take(DEFAULT_IMAGE_LIMIT)
        assertEquals(DEFAULT_IMAGE_LIMIT, imagePaths.size)

        val runtime = openBackend(backend)
        val detector = runtime.detector
        val riskAnalyzer = RiskAnalyzer()
        val totalMs = mutableListOf<Double>()
        val preprocessMs = mutableListOf<Double>()
        val inferenceMs = mutableListOf<Double>()
        val postprocessMs = mutableListOf<Double>()
        val thermalSamples = JSONArray()
        val failures = JSONArray()
        val startBattery = batterySnapshot()
        val startElapsed = SystemClock.elapsedRealtime()
        val deadline = startElapsed + durationSeconds * 1_000L
        val framePeriodMs = 1_000L / targetFps
        var nextSampleMs = startElapsed
        var frameCount = 0
        var abortedForSafety = false

        try {
            repeat(WARMUP_IMAGES) { index ->
                decodeBitmap(imagePaths[index % imagePaths.size]).use { detector.detect(it) }
            }
            while (SystemClock.elapsedRealtime() < deadline) {
                val frameStart = SystemClock.elapsedRealtime()
                val path = imagePaths[frameCount % imagePaths.size]
                try {
                    decodeBitmap(path).use { bitmap ->
                        val result = detector.detect(bitmap)
                        riskAnalyzer.analyze(result.detections, result.frameSize)
                    }
                    preprocessMs += detector.lastPreprocessMs.toDouble()
                    inferenceMs += detector.lastInferenceMs.toDouble()
                    postprocessMs += detector.lastPostprocessMs.toDouble()
                    totalMs += detector.lastTotalDetectMs.toDouble()
                    frameCount += 1
                } catch (error: Throwable) {
                    failures.put("${error.javaClass.simpleName}: ${error.message}")
                    break
                }

                val now = SystemClock.elapsedRealtime()
                if (now >= nextSampleMs) {
                    val snapshot = thermalSnapshot(now - startElapsed)
                    thermalSamples.put(snapshot)
                    if (snapshot.getDouble("battery_temperature_c") >= MAX_SAFE_BATTERY_TEMPERATURE_C ||
                        snapshot.getInt("thermal_status") >= PowerManager.THERMAL_STATUS_SEVERE
                    ) {
                        abortedForSafety = true
                        break
                    }
                    nextSampleMs = now + THERMAL_SAMPLE_INTERVAL_MS
                }
                val remaining = framePeriodMs - (SystemClock.elapsedRealtime() - frameStart)
                if (remaining > 0L) SystemClock.sleep(remaining)
            }
        } finally {
            runtime.close()
        }

        val elapsedMs = SystemClock.elapsedRealtime() - startElapsed
        thermalSamples.put(thermalSnapshot(elapsedMs))
        val report = JSONObject()
            .put("schema", SOAK_REPORT_SCHEMA)
            .put(
                "disposition",
                if (backend == BackendId.PRODUCTION_ROUTE) {
                    "FORMAL_PRODUCTION_ROUTE_STABILITY_VALIDATION"
                } else {
                    "BENCHMARK_ONLY_NO_PRODUCTION_BACKEND_CHANGE"
                }
            )
            .put("backend_policy", backendPolicyJson())
            .put("device", deviceJson())
            .put("backend", backend.name)
            .put("backend_init_ms", round3(runtime.initMs))
            .put("protocol", JSONObject()
                .put("requested_duration_seconds", durationSeconds)
                .put("actual_duration_seconds", round3(elapsedMs / 1_000.0))
                .put("target_fps", targetFps)
                .put("equal_workload_pacing", true)
                .put("includes_asset_decode", true)
                .put("includes_preprocess_decode_nms_risk", true)
                .put("image_count_in_cycle", imagePaths.size))
            .put("frame_count", frameCount)
            .put("effective_fps", round3(frameCount * 1_000.0 / max(1L, elapsedMs)))
            .put("aborted_for_safety", abortedForSafety)
            .put("safety_limits", JSONObject()
                .put("battery_temperature_c", MAX_SAFE_BATTERY_TEMPERATURE_C)
                .put("thermal_status", PowerManager.THERMAL_STATUS_SEVERE))
            .put("latency_ms", JSONObject()
                .put("preprocess", stats(preprocessMs))
                .put("inference", stats(inferenceMs))
                .put("postprocess", stats(postprocessMs))
                .put("detector_total", stats(totalMs)))
            .put("battery_start", startBattery)
            .put("battery_end", batterySnapshot())
            .put("thermal_samples", thermalSamples)
            .put("failures", failures)
        writeArtifact("soak-${backend.name.lowercase()}.json", report)
        assertTrue("Soak test produced failures: $failures", failures.length() == 0)
        assertTrue("Soak test stopped at a safety threshold", !abortedForSafety)
        assertTrue("Soak test ended too early", elapsedMs >= durationSeconds * 1_000L - 1_000L)
    }

    @Test
    fun compareEventLifecycleOnSanpoNinetyFrameSequences() {
        val frames = loadEventFrames()
        assertEquals("SANPO event-labeled frame count", EVENT_FRAME_COUNT, frames.size)
        assertEquals("SANPO event-labeled sequence count", EVENT_SEQUENCE_COUNT, frames.map { it.sequenceId }.toSet().size)
        assertEquals("SANPO event-labeled event count", EVENT_SEQUENCE_COUNT, frames.map { it.riskEventId }.toSet().size)
        assertEquals("SANPO source masks must cover every frame", EVENT_FRAME_COUNT, frames.count { assetExists(it.maskPath) })

        val temperatureBefore = batteryTemperatureC()
        val results = VALIDATION_BACKENDS.associateWith { backend -> runEventBackend(backend, frames) }
        val cpu = requireNotNull(results[BackendId.CPU])
        val comparisons = JSONArray(
            listOf(BackendId.GPU, BackendId.NPU, BackendId.PRODUCTION_ROUTE).map { backend ->
                compareEventBackends(cpu, requireNotNull(results[backend]))
            }
        )
        val report = JSONObject()
            .put("schema", EVENT_REPORT_SCHEMA)
            .put("disposition", "FORMAL_PRODUCTION_ROUTE_VALIDATION")
            .put("backend_policy", backendPolicyJson())
            .put("device", deviceJson())
            .put("protocol", JSONObject()
                .put("dataset", "blindassist-sanpo-v2-event-labeled-20260711")
                .put("frame_count", frames.size)
                .put("sequence_count", frames.map { it.sequenceId }.toSet().size)
                .put("expected_event_count", frames.map { it.riskEventId }.toSet().size)
                .put("frame_step_ms", FRAME_STEP_MS)
                .put("continuous_within_sequence", true)
                .put("reset_only_at_sequence_boundary", true)
                .put("yolo_decode_nms", "production_TfliteYoloDetector")
                .put("traversability_oracle", "production_TraversabilitySegmentationAnalyzer_from_source_mask")
                .put("risk_event_path", "production_AssistDecisionKernel_and_RiskEventTracker"))
            .put("temperature_c", JSONObject()
                .put("before", temperatureBefore)
                .put("after", batteryTemperatureC()))
            .put("backends", JSONArray(VALIDATION_BACKENDS.map { requireNotNull(results[it]).toJson() }))
            .put("comparisons_against_cpu", comparisons)
        writeArtifact(EVENT_REPORT_FILE, report)
        assertTrue("One or more event backends failed", results.values.all { it.failures.isEmpty() })
    }

    private fun runEventBackend(backend: BackendId, frames: List<EventFrame>): EventBackendResult {
        val runtime = openBackend(backend)
        val detector = runtime.detector
        val traversability = TraversabilitySegmentationAnalyzer()
        val kernel = AssistDecisionKernel(assistEngine = AssistEngine(riskAnalyzer = RiskAnalyzer()))
        val gateway = PlannerGateway()
        val rows = mutableListOf<EventFrameResult>()
        val failures = mutableListOf<String>()
        var activeSequence: String? = null
        try {
            frames.forEach { frame ->
                try {
                    if (activeSequence != frame.sequenceId) {
                        kernel.reset()
                        gateway.resetSession()
                        activeSequence = frame.sequenceId
                    }
                    decodeBitmap(frame.imagePath).use { bitmap ->
                        val detected = detector.detect(bitmap)
                        val mask = decodeSemanticMask(frame.maskPath)
                        val segmentationDetections = traversability
                            .analyze(mask, detected.frameSize)
                            .riskDetections
                        val decision = kernel.processFrame(
                            detections = detected.detections + segmentationDetections,
                            frameSize = detected.frameSize,
                            profile = AlertProfile.STANDARD,
                            scenario = frame.scenario,
                            metrics = DetectorMetrics(
                                totalMs = detector.lastTotalDetectMs,
                                preprocessMs = detector.lastPreprocessMs,
                                inferenceMs = detector.lastInferenceMs,
                                postprocessMs = detector.lastPostprocessMs,
                                fps = 10f,
                                modelStatus = backend.name
                            ),
                            feedbackGateway = gateway,
                            nowMs = frame.frameIndex * FRAME_STEP_MS
                        )
                        rows += EventFrameResult(
                            expected = frame,
                            yoloDetectionCount = detected.detections.size,
                            segmentationDetectionCount = segmentationDetections.size,
                            rawRisk = decision.evaluation.rawRisk,
                            stableRisk = decision.evaluation.stableRisk,
                            actualAlert = decision.feedbackDecision.triggered,
                            feedbackReason = decision.feedbackDecision.reason.name,
                            runtimeEventId = decision.evaluation.riskEvent.eventId,
                            runtimeEventState = decision.evaluation.riskEvent.state?.name,
                            runtimeEventActive = decision.evaluation.riskEvent.active,
                            runtimeEventClearReason = decision.evaluation.riskEvent.clearReason?.name,
                            inferenceMs = detector.lastInferenceMs.toDouble(),
                            totalMs = detector.lastTotalDetectMs.toDouble()
                        )
                    }
                } catch (error: Throwable) {
                    failures += "${frame.id}: ${error.javaClass.simpleName}: ${error.message}"
                }
            }
        } finally {
            runtime.close()
        }
        val alertSummary = EventAlertMetrics.summarize(rows.map { row ->
            EventAlertSample(
                riskEventId = row.expected.riskEventId,
                sequenceId = row.expected.sequenceId,
                fallbackFrameId = row.expected.id,
                expectedShouldAlert = row.expected.expectedShouldAlert,
                expectedCritical = row.expected.expectedDistanceBand == "CRITICAL",
                actualAlert = row.actualAlert
            )
        })
        val lifecycle = EventLifecycleMetrics.summarize(rows.map { row ->
            EventLifecycleSample(
                riskEventId = row.expected.riskEventId,
                sequenceId = row.expected.sequenceId,
                fallbackFrameId = row.expected.id,
                frameIndex = row.expected.frameIndex,
                expectedEventPhase = row.expected.expectedEventPhase,
                expectedShouldAlert = row.expected.expectedShouldAlert,
                actualAlert = row.actualAlert,
                suppressedDuplicateAttempt = row.feedbackReason == FeedbackReason.EVENT_ALREADY_ALERTED.name,
                runtimeEventId = row.runtimeEventId,
                runtimeEventActive = row.runtimeEventActive
            )
        })
        return EventBackendResult(
            backend = backend,
            initMs = runtime.initMs,
            frames = rows,
            alertSummary = alertSummary,
            lifecycle = lifecycle,
            failures = failures
        )
    }

    private fun compareEventBackends(
        cpu: EventBackendResult,
        candidate: EventBackendResult
    ): JSONObject {
        assertEquals(cpu.frames.size, candidate.frames.size)
        var rawRiskEqual = 0
        var stableRiskEqual = 0
        var alertEqual = 0
        var eventStateEqual = 0
        var runtimeEventIdentityEqual = 0
        val differences = JSONArray()
        cpu.frames.zip(candidate.frames).forEach { (baseline, other) ->
            val rawEqual = riskKey(baseline.rawRisk) == riskKey(other.rawRisk)
            val stableEqual = riskKey(baseline.stableRisk) == riskKey(other.stableRisk)
            val alertSame = baseline.actualAlert == other.actualAlert &&
                baseline.feedbackReason == other.feedbackReason
            val stateSame = baseline.runtimeEventState == other.runtimeEventState &&
                baseline.runtimeEventActive == other.runtimeEventActive &&
                baseline.runtimeEventClearReason == other.runtimeEventClearReason
            val idSame = baseline.runtimeEventId == other.runtimeEventId
            if (rawEqual) rawRiskEqual += 1
            if (stableEqual) stableRiskEqual += 1
            if (alertSame) alertEqual += 1
            if (stateSame) eventStateEqual += 1
            if (idSame) runtimeEventIdentityEqual += 1
            if (!rawEqual || !stableEqual || !alertSame || !stateSame || !idSame) {
                differences.put(JSONObject()
                    .put("frame_id", baseline.expected.id)
                    .put("sequence_id", baseline.expected.sequenceId)
                    .put("frame_index", baseline.expected.frameIndex)
                    .put("expected_event_phase", baseline.expected.expectedEventPhase)
                    .put("cpu_raw_risk", riskKey(baseline.rawRisk))
                    .put("candidate_raw_risk", riskKey(other.rawRisk))
                    .put("cpu_stable_risk", riskKey(baseline.stableRisk))
                    .put("candidate_stable_risk", riskKey(other.stableRisk))
                    .put("cpu_feedback", "${baseline.actualAlert}|${baseline.feedbackReason}")
                    .put("candidate_feedback", "${other.actualAlert}|${other.feedbackReason}")
                    .put("cpu_event", "${baseline.runtimeEventId}|${baseline.runtimeEventState}|${baseline.runtimeEventActive}|${baseline.runtimeEventClearReason}")
                    .put("candidate_event", "${other.runtimeEventId}|${other.runtimeEventState}|${other.runtimeEventActive}|${other.runtimeEventClearReason}"))
            }
        }
        return JSONObject()
            .put("candidate", candidate.backend.name)
            .put("frame_count", cpu.frames.size)
            .put("raw_risk_equal_frames", rawRiskEqual)
            .put("stable_risk_equal_frames", stableRiskEqual)
            .put("feedback_equal_frames", alertEqual)
            .put("event_state_equal_frames", eventStateEqual)
            .put("runtime_event_identity_equal_frames", runtimeEventIdentityEqual)
            .put("different_frames", differences.length())
            .put("differences", differences)
    }

    private fun loadEventFrames(): List<EventFrame> {
        return testContext.assets.open(EVENT_MANIFEST).bufferedReader(Charsets.UTF_8).useLines { lines ->
            lines.filter { it.isNotBlank() }
                .map { JSONObject(it) }
                .map { row ->
                    val relativeImage = row.getString("image_path")
                    EventFrame(
                        id = row.getString("id"),
                        imagePath = "$EVENT_PREFIX/$relativeImage",
                        maskPath = "$EVENT_PREFIX/source_masks/test/${relativeImage.substringAfterLast('/')}",
                        sequenceId = row.getString("sequence_id"),
                        frameIndex = row.getInt("frame_index"),
                        riskEventId = row.getString("risk_event_id"),
                        expectedEventPhase = row.getString("expected_event_phase"),
                        expectedShouldAlert = row.getBoolean("expected_should_alert"),
                        expectedRiskLevel = row.getString("expected_risk_level"),
                        expectedDistanceBand = row.getString("expected_distance_band"),
                        scenario = AssistScenario.valueOf(row.getString("assist_scenario")),
                        sceneBucket = row.getString("scene_bucket")
                    )
                }
                .sortedWith(compareBy<EventFrame> { it.sequenceId }.thenBy { it.frameIndex })
                .toList()
        }
    }

    private fun decodeSemanticMask(assetName: String): DenseSemanticMask {
        val original = decodeBitmap(assetName)
        val bitmap = if (original.width == TRAVERSABILITY_MASK_SIZE &&
            original.height == TRAVERSABILITY_MASK_SIZE
        ) {
            original
        } else {
            Bitmap.createScaledBitmap(
                original,
                TRAVERSABILITY_MASK_SIZE,
                TRAVERSABILITY_MASK_SIZE,
                false
            )
        }
        try {
            val pixels = IntArray(bitmap.width * bitmap.height)
            bitmap.getPixels(pixels, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)
            return DenseSemanticMask(
                width = bitmap.width,
                height = bitmap.height,
                classIds = IntArray(pixels.size) { index -> (pixels[index] shr 16) and 0xff }
            )
        } finally {
            bitmap.recycle()
            if (bitmap !== original) original.recycle()
        }
    }

    private fun assetExists(path: String): Boolean = try {
        testContext.assets.open(path).close()
        true
    } catch (_: Throwable) {
        false
    }

    private fun runBackend(backend: BackendId, imagePaths: List<String>): BackendResult {
        val runtime = openBackend(backend)
        val detector = runtime.detector
        val rows = mutableListOf<FrameResult>()
        val failures = mutableListOf<String>()
        try {
            repeat(WARMUP_IMAGES) { index ->
                decodeBitmap(imagePaths[index % imagePaths.size]).use { detector.detect(it) }
            }
            imagePaths.forEachIndexed { index, path ->
                try {
                    decodeBitmap(path).use { bitmap ->
                        val detected = detector.detect(bitmap)
                        val kernel = AssistDecisionKernel(
                            assistEngine = AssistEngine(riskAnalyzer = RiskAnalyzer())
                        )
                        val gateway = PlannerGateway()
                        kernel.reset()
                        gateway.resetSession()
                        val decision = kernel.processFrame(
                            detections = detected.detections,
                            frameSize = detected.frameSize,
                            profile = AlertProfile.STANDARD,
                            scenario = AssistScenario.GENERAL,
                            metrics = DetectorMetrics(
                                totalMs = detector.lastTotalDetectMs,
                                preprocessMs = detector.lastPreprocessMs,
                                inferenceMs = detector.lastInferenceMs,
                                postprocessMs = detector.lastPostprocessMs,
                                fps = 0f,
                                modelStatus = backend.name
                            ),
                            feedbackGateway = gateway,
                            nowMs = index * FRAME_STEP_MS
                        )
                        rows += FrameResult(
                            path = path,
                            detections = detected.detections,
                            rawRisk = decision.evaluation.rawRisk,
                            stableRisk = decision.evaluation.stableRisk,
                            feedbackReason = decision.feedbackDecision.reason.name,
                            feedbackTriggered = decision.feedbackDecision.triggered,
                            eventIdPresent = decision.evaluation.riskEvent.eventId != null,
                            eventState = decision.evaluation.riskEvent.state?.name,
                            preprocessMs = detector.lastPreprocessMs.toDouble(),
                            inferenceMs = detector.lastInferenceMs.toDouble(),
                            postprocessMs = detector.lastPostprocessMs.toDouble(),
                            totalMs = detector.lastTotalDetectMs.toDouble()
                        )
                    }
                } catch (error: Throwable) {
                    failures += "$path: ${error.javaClass.simpleName}: ${error.message}"
                }
            }
        } finally {
            runtime.close()
        }
        return BackendResult(backend, runtime.initMs, rows, failures)
    }

    private fun openBackend(backend: BackendId): BackendRuntime {
        var delegateCloser: (() -> Unit)? = null
        val start = System.nanoTime()
        val detector = when (backend) {
            BackendId.CPU -> TfliteYoloDetector(
                context = testContext,
                executionBackend = DetectorExecutionBackend.CPU_XNNPACK,
                externalInterpreterOptionsFactory = {
                    Interpreter.Options().setNumThreads(CPU_THREADS)
                }
            )
            BackendId.GPU -> {
                val compatibility = CompatibilityList()
                assertTrue("LiteRT GPU delegate is not supported on this device", compatibility.isDelegateSupportedOnThisDevice)
                val delegate = GpuDelegate(compatibility.bestOptionsForThisDevice)
                delegateCloser = { delegate.close() }
                TfliteYoloDetector(
                    context = testContext,
                    executionBackend = DetectorExecutionBackend.GPU_DELEGATE,
                    externalInterpreterOptionsFactory = {
                        Interpreter.Options()
                            .setNumThreads(CPU_THREADS)
                            .addDelegate(delegate)
                    }
                )
            }
            BackendId.NPU -> {
                System.loadLibrary("cdsprpc")
                assertTrue(
                    "QNN HTP FP16 capability is unavailable",
                    QnnDelegate.checkCapability(QnnDelegate.Capability.HTP_RUNTIME_FP16)
                )
                val options = QnnDelegate.Options().apply {
                    setBackendType(QnnDelegate.Options.BackendType.HTP_BACKEND)
                    setSkelLibraryDir(targetContext.applicationInfo.nativeLibraryDir)
                    setHtpPrecision(QnnDelegate.Options.HtpPrecision.HTP_PRECISION_FP16)
                    setHtpPerformanceMode(
                        QnnDelegate.Options.HtpPerformanceMode.HTP_PERFORMANCE_SUSTAINED_HIGH_PERFORMANCE
                    )
                    setLogLevel(QnnDelegate.Options.LogLevel.LOG_LEVEL_INFO)
                    setCacheDir(targetContext.codeCacheDir.absolutePath)
                    setModelToken(QNN_MODEL_TOKEN)
                }
                val delegate = QnnDelegate(options)
                assertTrue("QNN HTP delegate is unavailable", delegate.isAvailable)
                delegateCloser = { delegate.close() }
                TfliteYoloDetector(
                    context = testContext,
                    executionBackend = DetectorExecutionBackend.QUALCOMM_QNN_HTP,
                    externalInterpreterOptionsFactory = {
                        Interpreter.Options()
                            .setNumThreads(CPU_THREADS)
                            .addDelegate(delegate)
                    }
                )
            }
            BackendId.PRODUCTION_ROUTE -> {
                val routed = RuntimeObjectDetectorFactory.create(targetContext)
                require(routed is TfliteYoloDetector) {
                    "Production route returned unsupported detector type ${routed.javaClass.name}"
                }
                routed
            }
        }
        assertTrue("$backend detector initialization failed: ${detector.statusMessage}", detector.isReady)
        return BackendRuntime(
            detector = detector,
            initMs = elapsedMs(start),
            delegateCloser = delegateCloser
        )
    }

    private fun compareBackendResults(cpu: BackendResult, candidate: BackendResult): JSONObject {
        assertEquals(cpu.frames.size, candidate.frames.size)
        var detectionCountEqual = 0
        var detectionSetEquivalent = 0
        var rawRiskEqual = 0
        var stableRiskEqual = 0
        var feedbackEqual = 0
        var eventEqual = 0
        val attributionCounts = linkedMapOf<String, Int>()
        val differences = JSONArray()
        cpu.frames.zip(candidate.frames).forEach { (baseline, other) ->
            val detectionComparison = compareDetections(baseline.detections, other.detections)
            if (baseline.detections.size == other.detections.size) detectionCountEqual += 1
            if (detectionComparison.equivalent) detectionSetEquivalent += 1
            val rawEqual = riskKey(baseline.rawRisk) == riskKey(other.rawRisk)
            val stableEqual = riskKey(baseline.stableRisk) == riskKey(other.stableRisk)
            val feedbackSame = baseline.feedbackTriggered == other.feedbackTriggered &&
                baseline.feedbackReason == other.feedbackReason
            val eventSame = baseline.eventIdPresent == other.eventIdPresent &&
                baseline.eventState == other.eventState
            if (rawEqual) rawRiskEqual += 1
            if (stableEqual) stableRiskEqual += 1
            if (feedbackSame) feedbackEqual += 1
            if (eventSame) eventEqual += 1
            if (!detectionComparison.equivalent || !rawEqual || !stableEqual || !feedbackSame || !eventSame) {
                detectionComparison.attributions.forEach { attribution ->
                    attributionCounts[attribution.category] =
                        (attributionCounts[attribution.category] ?: 0) + 1
                }
                differences.put(JSONObject()
                    .put("image_path", baseline.path)
                    .put("cpu_detection_count", baseline.detections.size)
                    .put("candidate_detection_count", other.detections.size)
                    .put("matched_detections", detectionComparison.matched)
                    .put("detection_attribution", detectionComparison.toJson())
                    .put("cpu_raw_risk", riskKey(baseline.rawRisk))
                    .put("candidate_raw_risk", riskKey(other.rawRisk))
                    .put("cpu_stable_risk", riskKey(baseline.stableRisk))
                    .put("candidate_stable_risk", riskKey(other.stableRisk))
                    .put("cpu_feedback", "${baseline.feedbackTriggered}|${baseline.feedbackReason}")
                    .put("candidate_feedback", "${other.feedbackTriggered}|${other.feedbackReason}")
                    .put("cpu_event", "${baseline.eventIdPresent}|${baseline.eventState}")
                    .put("candidate_event", "${other.eventIdPresent}|${other.eventState}"))
            }
        }
        return JSONObject()
            .put("candidate", candidate.backend.name)
            .put("frame_count", cpu.frames.size)
            .put("detection_count_equal_frames", detectionCountEqual)
            .put("detection_set_equivalent_frames", detectionSetEquivalent)
            .put("raw_risk_equal_frames", rawRiskEqual)
            .put("stable_risk_equal_frames", stableRiskEqual)
            .put("feedback_equal_frames", feedbackEqual)
            .put("risk_event_equal_frames", eventEqual)
            .put("different_frames", differences.length())
            .put("detection_attribution_counts", JSONObject(attributionCounts as Map<*, *>))
            .put("differences", differences)
    }

    private fun compareDetections(cpu: List<Detection>, candidate: List<Detection>): DetectionComparison {
        val used = BooleanArray(candidate.size)
        var matched = 0
        val attributions = mutableListOf<DetectionAttribution>()
        cpu.forEach { expected ->
            var bestIndex = -1
            var bestIou = 0f
            candidate.forEachIndexed { index, actual ->
                if (!used[index] && expected.classId == actual.classId) {
                    val overlap = iou(expected, actual)
                    if (overlap > bestIou) {
                        bestIou = overlap
                        bestIndex = index
                    }
                }
            }
            if (bestIndex < 0) {
                attributions += DetectionAttribution(
                    category = "CANDIDATE_MISSING_CLASS_DETECTION",
                    cpu = expected,
                    candidate = null,
                    iou = null,
                    confidenceDelta = null
                )
            } else if (bestIou < DETECTION_EQUIVALENCE_IOU) {
                attributions += DetectionAttribution(
                    category = "BOX_GEOMETRY_DELTA",
                    cpu = expected,
                    candidate = candidate[bestIndex],
                    iou = bestIou,
                    confidenceDelta = kotlin.math.abs(expected.confidence - candidate[bestIndex].confidence)
                )
            } else if (
                kotlin.math.abs(expected.confidence - candidate[bestIndex].confidence) >
                DETECTION_EQUIVALENCE_CONFIDENCE_DELTA
            ) {
                attributions += DetectionAttribution(
                    category = "CONFIDENCE_DELTA",
                    cpu = expected,
                    candidate = candidate[bestIndex],
                    iou = bestIou,
                    confidenceDelta = kotlin.math.abs(expected.confidence - candidate[bestIndex].confidence)
                )
            } else {
                used[bestIndex] = true
                matched += 1
            }
        }
        candidate.forEachIndexed { index, detection ->
            if (!used[index] && attributions.none { it.candidate === detection }) {
                attributions += DetectionAttribution(
                    category = "CANDIDATE_EXTRA_DETECTION",
                    cpu = null,
                    candidate = detection,
                    iou = null,
                    confidenceDelta = null
                )
            }
        }
        return DetectionComparison(
            matched = matched,
            equivalent = matched == cpu.size && matched == candidate.size,
            attributions = attributions
        )
    }

    private fun iou(a: Detection, b: Detection): Float {
        val left = max(a.boundingBox.left, b.boundingBox.left)
        val top = max(a.boundingBox.top, b.boundingBox.top)
        val right = min(a.boundingBox.right, b.boundingBox.right)
        val bottom = min(a.boundingBox.bottom, b.boundingBox.bottom)
        val intersection = max(0f, right - left) * max(0f, bottom - top)
        val union = a.boundingBox.width * a.boundingBox.height +
            b.boundingBox.width * b.boundingBox.height - intersection
        return if (union <= 0f) 0f else intersection / union
    }

    private fun riskKey(risk: RiskResult): String =
        "${risk.level}|${risk.direction}|${risk.proximity}|${risk.sourceDetection?.label ?: "none"}"

    private fun loadImagePaths(): List<String> {
        return testContext.assets.open(BLINDASSIST_MANIFEST).bufferedReader(Charsets.UTF_8).useLines { lines ->
            lines.filter { it.isNotBlank() }
                .map { JSONObject(it).getString("image_path") }
                .map { "$BLINDASSIST_PREFIX/$it" }
                .toList()
        }
    }

    private fun decodeBitmap(path: String): Bitmap {
        return testContext.assets.open(path).use { stream ->
            requireNotNull(BitmapFactory.decodeStream(stream)) { "Failed to decode $path" }
        }
    }

    private fun batteryTemperatureC(): Double {
        val battery = targetContext.registerReceiver(null, IntentFilter(Intent.ACTION_BATTERY_CHANGED))
        val tenths = battery?.getIntExtra(android.os.BatteryManager.EXTRA_TEMPERATURE, -1) ?: -1
        return if (tenths >= 0) tenths / 10.0 else -1.0
    }

    private fun batterySnapshot(): JSONObject {
        val battery = targetContext.registerReceiver(null, IntentFilter(Intent.ACTION_BATTERY_CHANGED))
        val level = battery?.getIntExtra(android.os.BatteryManager.EXTRA_LEVEL, -1) ?: -1
        val scale = battery?.getIntExtra(android.os.BatteryManager.EXTRA_SCALE, -1) ?: -1
        val status = battery?.getIntExtra(android.os.BatteryManager.EXTRA_STATUS, -1) ?: -1
        return JSONObject()
            .put("level_percent", if (level >= 0 && scale > 0) round3(level * 100.0 / scale) else -1)
            .put("temperature_c", batteryTemperatureC())
            .put("status", status)
            .put("plugged", battery?.getIntExtra(android.os.BatteryManager.EXTRA_PLUGGED, -1) ?: -1)
    }

    private fun thermalSnapshot(elapsedMs: Long): JSONObject {
        val power = targetContext.getSystemService(PowerManager::class.java)
        return JSONObject()
            .put("elapsed_seconds", round3(elapsedMs / 1_000.0))
            .put("battery_temperature_c", batteryTemperatureC())
            .put("thermal_status", power.currentThermalStatus)
    }

    private fun deviceJson(): JSONObject = JSONObject()
        .put("manufacturer", Build.MANUFACTURER)
        .put("model", Build.MODEL)
        .put("soc_model", Build.SOC_MODEL)
        .put("android_release", Build.VERSION.RELEASE)
        .put("sdk_int", Build.VERSION.SDK_INT)

    private fun backendPolicyJson(): JSONObject = JSONObject()
        .put("policy_id", DetectorBackendPolicy.POLICY_ID)
        .put("production_default", DetectorBackendPolicy.productionDefault.wireName)
        .put(
            "next_default_candidate",
            DetectorBackendPolicy.nextDefaultCandidate?.wireName ?: JSONObject.NULL
        )
        .put("candidate_promotion_ready", DetectorBackendPolicy.decision.candidatePromotionReady)
        .put("candidate_gates", JSONArray(
            DetectorBackendPolicy.decision.candidateGates.map { gate ->
                JSONObject()
                    .put("gate", gate.gate.name)
                    .put("gate_class", gate.gateClass.name)
                    .put("state", gate.state.name)
                    .put("reason", gate.reason)
            }
        ))

    private fun writeArtifact(fileName: String, payload: JSONObject) {
        val dir = File(targetContext.filesDir, ARTIFACT_DIR).apply { mkdirs() }
        File(dir, fileName).writeText(payload.toString(2), Charsets.UTF_8)
    }

    private fun stats(values: List<Double>): JSONObject {
        if (values.isEmpty()) return JSONObject().put("count", 0)
        val sorted = values.sorted()
        return JSONObject()
            .put("count", sorted.size)
            .put("min", round3(sorted.first()))
            .put("p50", round3(percentile(sorted, 0.50)))
            .put("p95", round3(percentile(sorted, 0.95)))
            .put("max", round3(sorted.last()))
            .put("mean", round3(sorted.average()))
    }

    private fun percentile(sorted: List<Double>, quantile: Double): Double {
        val index = ((sorted.size - 1) * quantile).roundToInt().coerceIn(sorted.indices)
        return sorted[index]
    }

    private fun elapsedMs(startNs: Long): Double = (System.nanoTime() - startNs) / 1_000_000.0
    private fun round3(value: Double): Double = (value * 1_000.0).roundToInt() / 1_000.0

    private fun Bitmap.use(block: (Bitmap) -> Unit) {
        try {
            block(this)
        } finally {
            recycle()
        }
    }

    private class PlannerGateway : FeedbackGateway {
        override fun resetSession() = Unit

        override fun notify(
            risk: RiskResult,
            profile: AlertProfile,
            scenario: AssistScenario
        ): FeedbackDecision {
            val plan = FeedbackPlanner.planFor(risk, profile = profile, scenario = scenario)
            return if (plan == null) {
                FeedbackDecision(null, false, FeedbackReason.NO_FEEDBACK_RISK)
            } else {
                FeedbackDecision(plan, true, FeedbackReason.TRIGGERED)
            }
        }
    }

    private enum class BackendId { CPU, GPU, NPU, PRODUCTION_ROUTE }

    private data class BackendRuntime(
        val detector: TfliteYoloDetector,
        val initMs: Double,
        val delegateCloser: (() -> Unit)?
    ) {
        fun close() {
            detector.close()
            delegateCloser?.invoke()
        }
    }

    private data class BackendResult(
        val backend: BackendId,
        val initMs: Double,
        val frames: List<FrameResult>,
        val failures: List<String>
    ) {
        fun toJson(): JSONObject = JSONObject()
            .put("backend", backend.name)
            .put("init_ms", round3Static(initMs))
            .put("frame_count", frames.size)
            .put("frames_with_detections", frames.count { it.detections.isNotEmpty() })
            .put("total_detections", frames.sumOf { it.detections.size })
            .put("feedback_triggered_frames", frames.count { it.feedbackTriggered })
            .put("risk_event_present_frames", frames.count { it.eventIdPresent })
            .put("latency_ms", JSONObject()
                .put("preprocess", statsStatic(frames.map { it.preprocessMs }))
                .put("inference", statsStatic(frames.map { it.inferenceMs }))
                .put("postprocess", statsStatic(frames.map { it.postprocessMs }))
                .put("detector_total", statsStatic(frames.map { it.totalMs })))
            .put("failures", JSONArray(failures))
    }

    private data class FrameResult(
        val path: String,
        val detections: List<Detection>,
        val rawRisk: RiskResult,
        val stableRisk: RiskResult,
        val feedbackReason: String,
        val feedbackTriggered: Boolean,
        val eventIdPresent: Boolean,
        val eventState: String?,
        val preprocessMs: Double,
        val inferenceMs: Double,
        val postprocessMs: Double,
        val totalMs: Double
    )

    private data class DetectionComparison(
        val matched: Int,
        val equivalent: Boolean,
        val attributions: List<DetectionAttribution>
    ) {
        fun toJson(): JSONArray = JSONArray(attributions.map { it.toJson() })
    }

    private data class DetectionAttribution(
        val category: String,
        val cpu: Detection?,
        val candidate: Detection?,
        val iou: Float?,
        val confidenceDelta: Float?
    ) {
        fun toJson(): JSONObject = JSONObject()
            .put("category", category)
            .put("cpu", cpu?.let(::detectionJsonStatic) ?: JSONObject.NULL)
            .put("candidate", candidate?.let(::detectionJsonStatic) ?: JSONObject.NULL)
            .put("iou", iou?.toDouble()?.let(::round3Static) ?: JSONObject.NULL)
            .put("confidence_delta", confidenceDelta?.toDouble()?.let(::round3Static) ?: JSONObject.NULL)
    }

    private data class EventFrame(
        val id: String,
        val imagePath: String,
        val maskPath: String,
        val sequenceId: String,
        val frameIndex: Int,
        val riskEventId: String,
        val expectedEventPhase: String,
        val expectedShouldAlert: Boolean,
        val expectedRiskLevel: String,
        val expectedDistanceBand: String,
        val scenario: AssistScenario,
        val sceneBucket: String
    )

    private data class EventFrameResult(
        val expected: EventFrame,
        val yoloDetectionCount: Int,
        val segmentationDetectionCount: Int,
        val rawRisk: RiskResult,
        val stableRisk: RiskResult,
        val actualAlert: Boolean,
        val feedbackReason: String,
        val runtimeEventId: String?,
        val runtimeEventState: String?,
        val runtimeEventActive: Boolean,
        val runtimeEventClearReason: String?,
        val inferenceMs: Double,
        val totalMs: Double
    ) {
        fun toJson(): JSONObject = JSONObject()
            .put("frame_id", expected.id)
            .put("sequence_id", expected.sequenceId)
            .put("frame_index", expected.frameIndex)
            .put("risk_event_id", expected.riskEventId)
            .put("expected_event_phase", expected.expectedEventPhase)
            .put("expected_should_alert", expected.expectedShouldAlert)
            .put("expected_risk_level", expected.expectedRiskLevel)
            .put("expected_distance_band", expected.expectedDistanceBand)
            .put("scene_bucket", expected.sceneBucket)
            .put("yolo_detection_count", yoloDetectionCount)
            .put("segmentation_detection_count", segmentationDetectionCount)
            .put("raw_risk", riskKeyStatic(rawRisk))
            .put("stable_risk", riskKeyStatic(stableRisk))
            .put("actual_alert", actualAlert)
            .put("feedback_reason", feedbackReason)
            .put("runtime_event_id", runtimeEventId ?: JSONObject.NULL)
            .put("runtime_event_state", runtimeEventState ?: JSONObject.NULL)
            .put("runtime_event_active", runtimeEventActive)
            .put("runtime_event_clear_reason", runtimeEventClearReason ?: JSONObject.NULL)
            .put("inference_ms", inferenceMs)
            .put("total_ms", totalMs)
    }

    private data class EventBackendResult(
        val backend: BackendId,
        val initMs: Double,
        val frames: List<EventFrameResult>,
        val alertSummary: EventAlertSummary,
        val lifecycle: EventLifecycleSummary,
        val failures: List<String>
    ) {
        fun toJson(): JSONObject {
            val passedEventRows = frames
                .filter { it.expected.expectedEventPhase == "PASSED" }
                .groupBy { it.expected.riskEventId }
            val passedFalseAlerts = frames.count {
                it.expected.expectedEventPhase == "PASSED" && it.actualAlert
            }
            val passedRuntimeActiveFrames = passedEventRows.values.sumOf { event ->
                event.count { it.runtimeEventActive }
            }
            val passedRuntimeAlertedFrames = passedEventRows.values.sumOf { event ->
                event.count { it.runtimeEventState == "ALERTED" }
            }
            val runtimeExitedPassedEvents = passedEventRows.values.count { event ->
                val last = event.maxByOrNull { it.expected.frameIndex }
                last?.runtimeEventActive == false
            }
            val runtimeLifecycleClearanceRate = if (passedEventRows.isEmpty()) {
                0.0
            } else {
                runtimeExitedPassedEvents.toDouble() / passedEventRows.size.toDouble()
            }
            val eventGatePass =
                alertSummary.recall == 1.0 &&
                    alertSummary.criticalEventMissCount == 0 &&
                    lifecycle.deliveredRepeatedAlertCount == 0 &&
                    lifecycle.eventRegenerationCount == 0 &&
                    lifecycle.postEventClearanceRate == 1.0 &&
                    runtimeLifecycleClearanceRate == 1.0 &&
                    passedFalseAlerts == 0
            return JSONObject()
                .put("backend", backend.name)
                .put("init_ms", round3Static(initMs))
                .put("frame_count", frames.size)
                .put("event_gate_pass", eventGatePass)
                .put("event_alert", JSONObject()
                    .put("event_count", alertSummary.eventCount)
                    .put("hit_count", alertSummary.hitCount)
                    .put("recall", alertSummary.recall)
                    .put("critical_event_count", alertSummary.criticalEventCount)
                    .put("critical_event_miss_count", alertSummary.criticalEventMissCount))
                .put("lifecycle", JSONObject()
                    .put("observed_event_count", lifecycle.observedEventCount)
                    .put("delivered_alert_count", lifecycle.deliveredAlertCount)
                    .put("delivered_repeated_alert_count", lifecycle.deliveredRepeatedAlertCount)
                    .put("delivered_repeated_alert_rate", lifecycle.deliveredRepeatedAlertRate)
                    .put("suppressed_duplicate_attempt_count", lifecycle.suppressedDuplicateAttemptCount)
                    .put("passed_event_count", lifecycle.passedEventCount)
                    .put("cleared_passed_event_count", lifecycle.clearedPassedEventCount)
                    .put("post_event_delivery_suppression_rate", lifecycle.postEventClearanceRate)
                    .put("runtime_exited_passed_event_count", runtimeExitedPassedEvents)
                    .put("runtime_lifecycle_clearance_rate", runtimeLifecycleClearanceRate)
                    .put("passed_runtime_active_frame_count", passedRuntimeActiveFrames)
                    .put("passed_runtime_alerted_frame_count", passedRuntimeAlertedFrames)
                    .put("event_regeneration_count", lifecycle.eventRegenerationCount)
                    .put("mean_post_event_alert_latency_frames", lifecycle.meanPostEventAlertLatencyFrames)
                    .put("false_alert_count", lifecycle.falseAlertCount)
                    .put("false_alerts_per_minute", lifecycle.falseAlertsPerMinute ?: JSONObject.NULL)
                    .put("passed_window_false_alert_count", frames.count {
                        it.expected.expectedEventPhase == "PASSED" && it.actualAlert
                    }))
                .put("runtime_event_state_counts", JSONObject(
                    frames.groupingBy { it.runtimeEventState ?: "NONE" }.eachCount()
                ))
                .put("latency_ms", JSONObject()
                    .put("inference", statsStatic(frames.map { it.inferenceMs }))
                    .put("detector_total", statsStatic(frames.map { it.totalMs })))
                .put("first_alert_frame_by_sequence", JSONObject().apply {
                    frames.groupBy { it.expected.sequenceId }.forEach { (sequence, rows) ->
                        put(sequence, rows.firstOrNull { it.actualAlert }?.expected?.frameIndex ?: JSONObject.NULL)
                    }
                })
                .put("frames", JSONArray(frames.map { it.toJson() }))
                .put("failures", JSONArray(failures))
        }
    }

    companion object {
        private val VALIDATION_BACKENDS = listOf(
            BackendId.CPU,
            BackendId.GPU,
            BackendId.NPU,
            BackendId.PRODUCTION_ROUTE
        )
        private const val FULL_REPORT_SCHEMA = "blindassist_cpu_gpu_npu_full_pipeline_v2"
        private const val SOAK_REPORT_SCHEMA = "blindassist_cpu_gpu_npu_soak_v1"
        private const val EVENT_REPORT_SCHEMA = "blindassist_cpu_gpu_npu_sanpo_event_lifecycle_v1"
        private const val ARTIFACT_DIR = "cpu_gpu_npu_full_pipeline"
        private const val FULL_REPORT_FILE = "full-pipeline.json"
        private const val EVENT_REPORT_FILE = "sanpo-event-lifecycle.json"
        private const val BLINDASSIST_PREFIX = "blindassist_evalset"
        private const val BLINDASSIST_MANIFEST = "$BLINDASSIST_PREFIX/manifest.jsonl"
        private const val EVENT_PREFIX = "sanpo_event_lifecycle"
        private const val EVENT_MANIFEST = "$EVENT_PREFIX/manifest.jsonl"
        private const val QNN_MODEL_TOKEN = "blindassist_yolo11n_fp16_320_cpu_gpu_npu_qnn_2_47_v1"
        private const val ARG_IMAGE_LIMIT = "imageLimit"
        private const val ARG_BACKEND = "backend"
        private const val ARG_DURATION_SECONDS = "durationSeconds"
        private const val ARG_TARGET_FPS = "targetFps"
        private const val DEFAULT_IMAGE_LIMIT = 100
        private const val MAX_IMAGE_LIMIT = 150
        private const val WARMUP_IMAGES = 10
        private const val CPU_THREADS = 4
        private const val FRAME_STEP_MS = 100L
        private const val DEFAULT_SOAK_SECONDS = 600
        private const val MIN_SOAK_SECONDS = 60
        private const val MAX_SOAK_SECONDS = 1_800
        private const val DEFAULT_TARGET_FPS = 10
        private const val MAX_TARGET_FPS = 30
        private const val THERMAL_SAMPLE_INTERVAL_MS = 30_000L
        private const val MAX_SAFE_BATTERY_TEMPERATURE_C = 45.0
        private const val DETECTION_EQUIVALENCE_IOU = 0.95f
        private const val DETECTION_EQUIVALENCE_CONFIDENCE_DELTA = 0.03f
        private const val EVENT_FRAME_COUNT = 90
        private const val EVENT_SEQUENCE_COUNT = 3
        private const val TRAVERSABILITY_MASK_SIZE = 256

        private fun round3Static(value: Double): Double =
            (value * 1_000.0).roundToInt() / 1_000.0

        private fun statsStatic(values: List<Double>): JSONObject {
            if (values.isEmpty()) return JSONObject().put("count", 0)
            val sorted = values.sorted()
            fun percentile(q: Double): Double {
                val index = ((sorted.size - 1) * q).roundToInt().coerceIn(sorted.indices)
                return sorted[index]
            }
            return JSONObject()
                .put("count", sorted.size)
                .put("min", round3Static(sorted.first()))
                .put("p50", round3Static(percentile(0.50)))
                .put("p95", round3Static(percentile(0.95)))
                .put("max", round3Static(sorted.last()))
                .put("mean", round3Static(sorted.average()))
        }

        private fun riskKeyStatic(risk: RiskResult): String =
            "${risk.level}|${risk.direction}|${risk.proximity}|${risk.sourceDetection?.label ?: "none"}"

        private fun detectionJsonStatic(detection: Detection): JSONObject = JSONObject()
            .put("class_id", detection.classId)
            .put("label", detection.label)
            .put("confidence", round3Static(detection.confidence.toDouble()))
            .put("box", JSONObject()
                .put("left", round3Static(detection.boundingBox.left.toDouble()))
                .put("top", round3Static(detection.boundingBox.top.toDouble()))
                .put("right", round3Static(detection.boundingBox.right.toDouble()))
                .put("bottom", round3Static(detection.boundingBox.bottom.toDouble())))
    }
}
