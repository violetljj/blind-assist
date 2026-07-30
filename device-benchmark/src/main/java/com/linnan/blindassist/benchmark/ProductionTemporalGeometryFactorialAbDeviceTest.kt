package com.linnan.blindassist.benchmark

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Build
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.linnan.blindassist.alert.AlertProfile
import com.linnan.blindassist.alert.AssistScenario
import com.linnan.blindassist.feedback.FeedbackClock
import com.linnan.blindassist.feedback.FeedbackController
import com.linnan.blindassist.feedback.FeedbackPlan
import com.linnan.blindassist.feedback.FeedbackRuntimeSettings
import com.linnan.blindassist.feedback.HapticOutput
import com.linnan.blindassist.feedback.SpeechOutput
import com.linnan.blindassist.localization.AppLanguage
import com.linnan.blindassist.model.BoundingBox
import com.linnan.blindassist.model.Detection
import com.linnan.blindassist.model.DetectionSource
import com.linnan.blindassist.model.FrameSize
import com.linnan.blindassist.risk.ObjectDetectorTemporalGeometryMode
import com.linnan.blindassist.risk.RiskAnalyzer
import com.linnan.blindassist.risk.RiskResult
import com.linnan.blindassist.risk.TemporalRiskTracker
import com.linnan.blindassist.risk.TemporalRiskTrackerConfig
import com.linnan.blindassist.session.AssistDecisionKernel
import com.linnan.blindassist.session.AssistEngine
import com.linnan.blindassist.session.DetectorMetrics
import com.linnan.blindassist.session.DualLoopRuntimeMode
import com.linnan.blindassist.vision.DetectorExecutionBackend
import com.linnan.blindassist.vision.FrameClockDomain
import com.linnan.blindassist.vision.FrameStamp
import com.linnan.blindassist.vision.ProductionDetectorRoutePolicy
import com.linnan.blindassist.vision.ProductionDeviceProfile
import com.linnan.blindassist.vision.TfliteYoloDetector
import com.qualcomm.qti.QnnDelegate
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.tensorflow.lite.Interpreter
import java.io.BufferedWriter
import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import java.nio.file.StandardOpenOption
import java.security.MessageDigest

/**
 * Truth-blind producer for DUAL_LOOP_PRODUCTION_TEMPORAL_GEOMETRY_FACTORIAL_AB_R0.
 *
 * This class has no truth-ledger input. Formal output is generated only by the explicitly named
 * test method after a separate prestart receipt and activation have been completed.
 */
@RunWith(AndroidJUnit4::class)
class ProductionTemporalGeometryFactorialAbDeviceTest {
    private val instrumentation = InstrumentationRegistry.getInstrumentation()
    private val testContext = instrumentation.context
    private val targetContext = instrumentation.targetContext

    /**
     * Reuses the already-burned RGB universe to expose the full production detector output for a
     * Development-only multi-track falsification screen. This writes to a separate namespace and
     * neither reads truth nor modifies/replaces the completed formal A/B output.
     */
    @Test
    fun runDevelopmentAllDetectionDump() {
        val inventory = verifyInputInventory(inputRoot())
        val outputRoot = File(baseRoot(), DEVELOPMENT_DUMP_DIRECTORY)
        val temporaryRoot = File(baseRoot(), "$DEVELOPMENT_DUMP_DIRECTORY.tmp")
        assertTrue("development output already exists", !outputRoot.exists())
        assertTrue("stale development temporary output exists", !temporaryRoot.exists())

        val detector = strictQnnDetector()
        assertTrue("development temporary output could not be created", temporaryRoot.mkdirs())
        var frameCount = 0
        val startedAtNs = System.nanoTime()
        try {
            File(temporaryRoot, TRACE_FILE).bufferedWriter(Charsets.UTF_8).use { writer ->
                SESSIONS.forEach { session ->
                    readFrameLedger(File(inputRoot(), "$session/frames.jsonl")).forEach { frame ->
                        val imageFile = File(inputRoot(), "$session/${frame.rgbPath}")
                        check(sha256File(imageFile) == frame.rgbSha256) {
                            "development RGB identity drift: $session/${frame.frameId}"
                        }
                        val bitmap = BitmapFactory.decodeFile(imageFile.absolutePath)
                            ?: error("decode failed: $session/${frame.frameId}")
                        try {
                            check(bitmap.width == EXPECTED_WIDTH && bitmap.height == EXPECTED_HEIGHT)
                            val detectorResult = detector.detect(bitmap)
                            check(
                                detectorResult.frameSize ==
                                    FrameSize(EXPECTED_WIDTH, EXPECTED_HEIGHT)
                            )
                            val detections = detectorResult.detections.toList()
                            val row = JSONObject()
                                .put(
                                    "schema_version",
                                    "blindassist.dual_loop_all_detection_dump.v1"
                                )
                                .put("authority", "DEVELOPMENT_ONLY_BURNED_RGB")
                                .put("session_id", session)
                                .put("frame_id", frame.frameId)
                                .put(
                                    "source_capture_timestamp_ns",
                                    frame.sourceCaptureTimestampNs
                                )
                                .put(
                                    "detector_output_sha256",
                                    canonicalDetectionsSha256(detections)
                                )
                                .put("detection_count", detections.size)
                                .put(
                                    "detections",
                                    JSONArray().apply {
                                        detections.forEach { put(detectionJson(it)) }
                                    }
                                )
                            writer.write(row.toString())
                            writer.newLine()
                            frameCount += 1
                            if (frameCount % PROGRESS_INTERVAL == 0) {
                                val elapsedSeconds =
                                    (System.nanoTime() - startedAtNs).toDouble() /
                                        NANOS_PER_SECOND
                                Log.i(
                                    TAG,
                                    "stage=DEVELOPMENT_ALL_DETECTIONS state=RUNNING " +
                                        "completed=$frameCount total=$EXPECTED_FRAME_COUNT " +
                                        "throughput_fps=${frameCount / elapsedSeconds}"
                                )
                            }
                        } finally {
                            bitmap.recycle()
                        }
                    }
                }
            }
            check(frameCount == EXPECTED_FRAME_COUNT)
            val trace = File(temporaryRoot, TRACE_FILE)
            val receipt = JSONObject()
                .put(
                    "schema_version",
                    "blindassist.dual_loop_all_detection_dump_receipt.v1"
                )
                .put("status", "COMPLETE")
                .put("authority", "DEVELOPMENT_ONLY_BURNED_RGB")
                .put("truth_read", false)
                .put("formal_output_modified", false)
                .put("frame_count", frameCount)
                .put("trace_sha256", sha256File(trace))
                .put(
                    "input_inventory_sha256",
                    inventory.getString("canonical_rgb_inventory_sha256")
                )
                .put("backend", DetectorExecutionBackend.QUALCOMM_QNN_HTP.wireName)
                .put("qnn_runtime_version", JSONArray(QnnDelegate.getVersion().toList()))
                .put("elapsed_ms", (System.nanoTime() - startedAtNs) / NANOS_PER_MILLISECOND)
                .put("device", deviceJson())
            atomicWrite(File(temporaryRoot, PRODUCER_RECEIPT), receipt.toString(2))
            check(temporaryRoot.renameTo(outputRoot)) {
                "development output publication failed"
            }
            Log.i(
                TAG,
                "stage=DEVELOPMENT_ALL_DETECTIONS state=COMPLETE " +
                    "completed=$frameCount total=$EXPECTED_FRAME_COUNT"
            )
        } finally {
            detector.close()
        }
    }

    /**
     * Replays the immutable Development detector dump through the real decision kernel.
     *
     * Baseline and candidate receive byte-identical detections. The candidate may suppress only
     * an otherwise-eligible feedback opportunity when the frozen scene-scale source emits an
     * admitted contradiction. Risk values are required to remain identical on every frame.
     */
    @Test
    fun runDevelopmentSceneScaleActiveReplay() {
        val dumpRoot = File(baseRoot(), DEVELOPMENT_DUMP_DIRECTORY)
        val dumpTrace = File(dumpRoot, TRACE_FILE)
        val dumpReceiptFile = File(dumpRoot, PRODUCER_RECEIPT)
        assertTrue("development detector dump is missing", dumpTrace.isFile)
        assertTrue("development detector dump receipt is missing", dumpReceiptFile.isFile)
        val dumpReceipt = JSONObject(dumpReceiptFile.readText(Charsets.UTF_8))
        assertEquals("COMPLETE", dumpReceipt.getString("status"))
        assertEquals(false, dumpReceipt.getBoolean("truth_read"))
        assertEquals(false, dumpReceipt.getBoolean("formal_output_modified"))
        assertEquals(EXPECTED_FRAME_COUNT, dumpReceipt.getInt("frame_count"))
        assertEquals(EXPECTED_INVENTORY_SHA256, dumpReceipt.getString("input_inventory_sha256"))
        assertEquals(dumpReceipt.getString("trace_sha256"), sha256File(dumpTrace))

        val outputRoot = File(baseRoot(), DEVELOPMENT_ACTIVE_REPLAY_DIRECTORY)
        val temporaryRoot = File(baseRoot(), "$DEVELOPMENT_ACTIVE_REPLAY_DIRECTORY.tmp")
        assertTrue("development active replay output already exists", !outputRoot.exists())
        assertTrue("stale development active replay temporary output exists", !temporaryRoot.exists())
        assertTrue("development active replay temporary output could not be created", temporaryRoot.mkdirs())

        val baseline = branch(ObjectDetectorTemporalGeometryMode.APPLY)
        val candidate = branch(ObjectDetectorTemporalGeometryMode.APPLY)
        val metrics = DetectorMetrics(
            totalMs = 0L,
            preprocessMs = 0L,
            inferenceMs = 0L,
            postprocessMs = 0L,
            fps = 0f,
            modelStatus = "DEVELOPMENT_IMMUTABLE_DETECTION_REPLAY"
        )
        var currentSession: String? = null
        var sessionOriginNs = 0L
        var frameCount = 0
        var baselineTriggerCount = 0
        var candidateTriggerCount = 0
        var contradictionCount = 0
        var vetoedOpportunityCount = 0
        val startedAtNs = System.nanoTime()
        try {
            File(temporaryRoot, TRACE_FILE).bufferedWriter(Charsets.UTF_8).use { writer ->
                dumpTrace.forEachLine(Charsets.UTF_8) { line ->
                    if (line.isBlank()) return@forEachLine
                    val input = JSONObject(line)
                    val session = input.getString("session_id")
                    val capturedAtNs = input.getLong("source_capture_timestamp_ns")
                    if (session != currentSession) {
                        currentSession = session
                        sessionOriginNs = capturedAtNs
                        baseline.startSession(0L)
                        candidate.startSession(0L)
                    }
                    val nowMs = (capturedAtNs - sessionOriginNs) / NANOS_PER_MILLISECOND
                    baseline.clock.nowMs = nowMs
                    candidate.clock.nowMs = nowMs
                    val detectionsJson = input.getJSONArray("detections")
                    val detections = buildList {
                        repeat(detectionsJson.length()) { index ->
                            val detection = detectionsJson.getJSONObject(index)
                            add(
                                Detection(
                                    classId = detection.getInt("class_id"),
                                    label = detection.getString("label"),
                                    confidence = detection.getDouble("confidence").toFloat(),
                                    boundingBox = BoundingBox(
                                        detection.getDouble("left").toFloat(),
                                        detection.getDouble("top").toFloat(),
                                        detection.getDouble("right").toFloat(),
                                        detection.getDouble("bottom").toFloat()
                                    ),
                                    frameSize = FrameSize(
                                        detection.getInt("frame_width"),
                                        detection.getInt("frame_height")
                                    ),
                                    source = DetectionSource.valueOf(detection.getString("source")),
                                    temporalPromotionEligible =
                                        detection.getBoolean("temporal_promotion_eligible")
                                )
                            )
                        }
                    }
                    val detectionHash = canonicalDetectionsSha256(detections)
                    check(detectionHash == input.getString("detector_output_sha256")) {
                        "development detector dump drift: $session/${input.getString("frame_id")}"
                    }
                    val frameId = input.getString("frame_id")
                    val stamp = FrameStamp(
                        frameId = frameId.toLong(),
                        capturedAtNs = capturedAtNs,
                        receivedAtNs = capturedAtNs,
                        sourceId = session,
                        coordinateFrame = "CROWDBOT_RGB_STORED_PIXELS",
                        clockDomain = FrameClockDomain.REPLAY_TIMELINE
                    )
                    val baselineResult = baseline.kernel.processFrame(
                        detections = detections,
                        frameSize = FrameSize(EXPECTED_WIDTH, EXPECTED_HEIGHT),
                        profile = AlertProfile.STANDARD,
                        scenario = AssistScenario.GENERAL,
                        metrics = metrics,
                        feedbackGateway = baseline.feedback,
                        nowMs = nowMs,
                        sourceFrame = stamp,
                        decisionAtNs = capturedAtNs,
                        dualLoopMode = DualLoopRuntimeMode.OFF
                    )
                    val candidateResult = candidate.kernel.processFrame(
                        detections = detections,
                        frameSize = FrameSize(EXPECTED_WIDTH, EXPECTED_HEIGHT),
                        profile = AlertProfile.STANDARD,
                        scenario = AssistScenario.GENERAL,
                        metrics = metrics,
                        feedbackGateway = candidate.feedback,
                        nowMs = nowMs,
                        sourceFrame = stamp,
                        decisionAtNs = capturedAtNs,
                        dualLoopMode = DualLoopRuntimeMode.ACTIVE_CONTRADICT_ONLY
                    )
                    val baselineRawHash = canonicalRiskSha256(baselineResult.evaluation.rawRisk)
                    val candidateRawHash = canonicalRiskSha256(candidateResult.evaluation.rawRisk)
                    val baselineStableHash =
                        canonicalRiskSha256(baselineResult.evaluation.stableRisk)
                    val candidateStableHash =
                        canonicalRiskSha256(candidateResult.evaluation.stableRisk)
                    check(baselineRawHash == candidateRawHash) {
                        "active replay raw-risk mutation: $session/$frameId"
                    }
                    check(baselineStableHash == candidateStableHash) {
                        "active replay stable-risk mutation: $session/$frameId"
                    }
                    val observation = candidateResult.evaluation.dualLoopShadow
                    val contradicted =
                        observation.correctionDecision?.name == "CONTRADICT_APPROACH"
                    if (baselineResult.feedbackDecision.triggered) baselineTriggerCount += 1
                    if (candidateResult.feedbackDecision.triggered) candidateTriggerCount += 1
                    if (contradicted) contradictionCount += 1
                    if (candidateResult.feedbackDecision.reason.name == "DUAL_LOOP_CONTRADICTED") {
                        vetoedOpportunityCount += 1
                    }
                    writer.write(
                        JSONObject()
                            .put(
                                "schema_version",
                                "blindassist.dual_loop_scene_scale_active_replay.v1"
                            )
                            .put("protocol_id", DEVELOPMENT_ACTIVE_PROTOCOL_ID)
                            .put("authority", "DEVELOPMENT_ONLY_BURNED_RGB")
                            .put("session_id", session)
                            .put("frame_id", frameId)
                            .put("source_capture_timestamp_ns", capturedAtNs)
                            .put("detector_output_sha256", detectionHash)
                            .put("raw_risk_sha256", baselineRawHash)
                            .put("stable_risk_sha256", baselineStableHash)
                            .put(
                                "baseline_feedback_triggered",
                                baselineResult.feedbackDecision.triggered
                            )
                            .put(
                                "baseline_feedback_reason",
                                baselineResult.feedbackDecision.reason.name
                            )
                            .put(
                                "candidate_feedback_triggered",
                                candidateResult.feedbackDecision.triggered
                            )
                            .put(
                                "candidate_feedback_reason",
                                candidateResult.feedbackDecision.reason.name
                            )
                            .put(
                                "dual_loop",
                                JSONObject()
                                    .put("mode", observation.mode.name)
                                    .put("disposition", observation.disposition.name)
                                    .put(
                                        "correction_decision",
                                        observation.correctionDecision?.name ?: JSONObject.NULL
                                    )
                                    .put(
                                        "signed_approach_rate_per_s",
                                        observation.signedApproachRatePerS
                                            ?.toDouble()
                                            ?: JSONObject.NULL
                                    )
                                    .put(
                                        "feedback_mutation_allowed",
                                        observation.feedbackMutationAllowed
                                    )
                                    .put(
                                        "event_mutation_allowed",
                                        observation.eventMutationAllowed
                                    )
                            )
                            .toString()
                    )
                    writer.newLine()
                    frameCount += 1
                }
            }
            check(frameCount == EXPECTED_FRAME_COUNT)
            check(baselineTriggerCount == EXPECTED_BASELINE_TRIGGER_COUNT) {
                "baseline replay drift: $baselineTriggerCount"
            }
            check(candidateTriggerCount == EXPECTED_ACTIVE_TRIGGER_COUNT) {
                "candidate implementation drift: $candidateTriggerCount"
            }
            val trace = File(temporaryRoot, TRACE_FILE)
            val receipt = JSONObject()
                .put(
                    "schema_version",
                    "blindassist.dual_loop_scene_scale_active_replay_receipt.v1"
                )
                .put("protocol_id", DEVELOPMENT_ACTIVE_PROTOCOL_ID)
                .put("status", "COMPLETE")
                .put("authority", "DEVELOPMENT_ONLY_BURNED_RGB")
                .put("truth_read", false)
                .put("formal_output_modified", false)
                .put("frame_count", frameCount)
                .put("baseline_feedback_trigger_count", baselineTriggerCount)
                .put("candidate_feedback_trigger_count", candidateTriggerCount)
                .put("scene_contradict_frame_count", contradictionCount)
                .put("vetoed_feedback_opportunity_count", vetoedOpportunityCount)
                .put("risk_mutation_count", 0)
                .put("trace_sha256", sha256File(trace))
                .put("input_dump_sha256", sha256File(dumpTrace))
                .put("elapsed_ms", (System.nanoTime() - startedAtNs) / NANOS_PER_MILLISECOND)
                .put("device", deviceJson())
            atomicWrite(File(temporaryRoot, PRODUCER_RECEIPT), receipt.toString(2))
            check(temporaryRoot.renameTo(outputRoot)) {
                "development active replay publication failed"
            }
            Log.i(
                TAG,
                "stage=DEVELOPMENT_SCENE_SCALE_ACTIVE state=COMPLETE " +
                    "completed=$frameCount baseline_triggers=$baselineTriggerCount " +
                    "candidate_triggers=$candidateTriggerCount vetoed=$vetoedOpportunityCount"
            )
        } finally {
            baseline.close()
            candidate.close()
        }
    }

    /**
     * Locked cross-source Development run on the full 10 Hz Matoaka replay.
     *
     * The device input contains RGB plus timestamps only. Labels and event windows remain host-side
     * and are joined only after this producer has atomically published its truth-blind trace.
     */
    @Test
    fun runDevelopmentSceneScaleMatoakaCrossSource() {
        val runRoot = File(baseRoot(), DEVELOPMENT_MATOAKA_DIRECTORY)
        val inputRoot = File(runRoot, "input")
        val inputReceiptFile = File(inputRoot, "input_receipt.json")
        val manifestFile = File(inputRoot, "manifest.jsonl")
        assertTrue("Matoaka input receipt is missing", inputReceiptFile.isFile)
        assertTrue("Matoaka input manifest is missing", manifestFile.isFile)
        val inputReceipt = JSONObject(inputReceiptFile.readText(Charsets.UTF_8))
        assertEquals(DEVELOPMENT_MATOAKA_PROTOCOL_ID, inputReceipt.getString("protocol_id"))
        assertEquals("COMPLETE", inputReceipt.getString("status"))
        assertEquals(false, inputReceipt.getBoolean("truth_read"))
        assertEquals(MATOAKA_VIDEO_SHA256, inputReceipt.getString("video_sha256"))
        assertEquals(MATOAKA_FRAME_COUNT, inputReceipt.getInt("frame_count"))
        assertEquals(MATOAKA_MANIFEST_SHA256, inputReceipt.getString("manifest_sha256"))
        assertEquals(MATOAKA_MANIFEST_SHA256, sha256File(manifestFile))

        val outputRoot = File(runRoot, "output")
        val temporaryRoot = File(runRoot, "output.tmp")
        assertTrue("Matoaka output already exists", !outputRoot.exists())
        assertTrue("stale Matoaka temporary output exists", !temporaryRoot.exists())
        assertTrue("Matoaka temporary output could not be created", temporaryRoot.mkdirs())

        val detector = strictQnnDetector()
        val baseline = branch(ObjectDetectorTemporalGeometryMode.APPLY)
        val candidate = branch(ObjectDetectorTemporalGeometryMode.APPLY)
        baseline.startSession(0L)
        candidate.startSession(0L)
        var frameCount = 0
        var baselineTriggerCount = 0
        var candidateTriggerCount = 0
        var riskMutationCount = 0
        val startedAtNs = System.nanoTime()
        try {
            File(temporaryRoot, TRACE_FILE).bufferedWriter(Charsets.UTF_8).use { writer ->
                manifestFile.forEachLine(Charsets.UTF_8) { line ->
                    if (line.isBlank()) return@forEachLine
                    val frame = JSONObject(line)
                    val frameId = frame.getLong("frame_id")
                    check(frameId == frameCount.toLong()) {
                        "Matoaka frame sequence drift at $frameId"
                    }
                    val capturedAtNs = frame.getLong("source_capture_timestamp_ns")
                    check(capturedAtNs == frameId * MATOAKA_FRAME_STEP_NS) {
                        "Matoaka timestamp schedule drift at $frameId"
                    }
                    val image = File(inputRoot, frame.getString("image_path"))
                    check(sha256File(image) == frame.getString("image_sha256")) {
                        "Matoaka image identity drift at $frameId"
                    }
                    val bitmap = BitmapFactory.decodeFile(image.absolutePath)
                        ?: error("Matoaka image decode failed at $frameId")
                    try {
                        check(bitmap.width == EXPECTED_WIDTH && bitmap.height == EXPECTED_HEIGHT)
                        val detectorResult = detector.detect(bitmap)
                        val detections = detectorResult.detections.toList()
                        val detectionHash = canonicalDetectionsSha256(detections)
                        val nowMs = capturedAtNs / NANOS_PER_MILLISECOND
                        baseline.clock.nowMs = nowMs
                        candidate.clock.nowMs = nowMs
                        val stamp = FrameStamp(
                            frameId = frameId,
                            capturedAtNs = capturedAtNs,
                            receivedAtNs = capturedAtNs,
                            sourceId = MATOAKA_SOURCE_ID,
                            coordinateFrame = "MATOAKA_LETTERBOX_640X480",
                            clockDomain = FrameClockDomain.REPLAY_TIMELINE
                        )
                        val baselineResult = baseline.kernel.processFrame(
                            detections = detections,
                            frameSize = detectorResult.frameSize,
                            profile = AlertProfile.STANDARD,
                            scenario = AssistScenario.GENERAL,
                            metrics = detectorResult.metrics,
                            feedbackGateway = baseline.feedback,
                            nowMs = nowMs,
                            sourceFrame = stamp,
                            decisionAtNs = capturedAtNs,
                            dualLoopMode = DualLoopRuntimeMode.OFF
                        )
                        val candidateResult = candidate.kernel.processFrame(
                            detections = detections,
                            frameSize = detectorResult.frameSize,
                            profile = AlertProfile.STANDARD,
                            scenario = AssistScenario.GENERAL,
                            metrics = detectorResult.metrics,
                            feedbackGateway = candidate.feedback,
                            nowMs = nowMs,
                            sourceFrame = stamp,
                            decisionAtNs = capturedAtNs,
                            dualLoopMode = DualLoopRuntimeMode.ACTIVE_CONTRADICT_ONLY
                        )
                        val baselineRawHash =
                            canonicalRiskSha256(baselineResult.evaluation.rawRisk)
                        val candidateRawHash =
                            canonicalRiskSha256(candidateResult.evaluation.rawRisk)
                        val baselineStableHash =
                            canonicalRiskSha256(baselineResult.evaluation.stableRisk)
                        val candidateStableHash =
                            canonicalRiskSha256(candidateResult.evaluation.stableRisk)
                        if (baselineRawHash != candidateRawHash ||
                            baselineStableHash != candidateStableHash
                        ) {
                            riskMutationCount += 1
                        }
                        check(riskMutationCount == 0) {
                            "Matoaka active branch mutated risk at $frameId"
                        }
                        if (baselineResult.feedbackDecision.triggered) {
                            baselineTriggerCount += 1
                        }
                        if (candidateResult.feedbackDecision.triggered) {
                            candidateTriggerCount += 1
                        }
                        val observation = candidateResult.evaluation.dualLoopShadow
                        writer.write(
                            JSONObject()
                                .put(
                                    "schema_version",
                                    "blindassist.dual_loop_scene_scale_cross_source_trace.v1"
                                )
                                .put("protocol_id", DEVELOPMENT_MATOAKA_PROTOCOL_ID)
                                .put("authority", "LOCKED_CROSS_SOURCE_DEVELOPMENT")
                                .put("source_id", MATOAKA_SOURCE_ID)
                                .put("frame_id", frameId)
                                .put("source_capture_timestamp_ns", capturedAtNs)
                                .put("image_sha256", frame.getString("image_sha256"))
                                .put("detector_output_sha256", detectionHash)
                                .put("detection_count", detections.size)
                                .put("raw_risk", riskJson(baselineResult.evaluation.rawRisk))
                                .put(
                                    "stable_risk",
                                    riskJson(baselineResult.evaluation.stableRisk)
                                )
                                .put(
                                    "baseline_feedback_triggered",
                                    baselineResult.feedbackDecision.triggered
                                )
                                .put(
                                    "baseline_feedback_reason",
                                    baselineResult.feedbackDecision.reason.name
                                )
                                .put(
                                    "candidate_feedback_triggered",
                                    candidateResult.feedbackDecision.triggered
                                )
                                .put(
                                    "candidate_feedback_reason",
                                    candidateResult.feedbackDecision.reason.name
                                )
                                .put(
                                    "dual_loop",
                                    JSONObject()
                                        .put("mode", observation.mode.name)
                                        .put("disposition", observation.disposition.name)
                                        .put(
                                            "correction_decision",
                                            observation.correctionDecision?.name
                                                ?: JSONObject.NULL
                                        )
                                        .put(
                                            "signed_approach_rate_per_s",
                                            observation.signedApproachRatePerS
                                                ?.toDouble()
                                                ?: JSONObject.NULL
                                        )
                                        .put(
                                            "feedback_mutation_allowed",
                                            observation.feedbackMutationAllowed
                                        )
                                        .put(
                                            "event_mutation_allowed",
                                            observation.eventMutationAllowed
                                        )
                                )
                                .toString()
                        )
                        writer.newLine()
                        frameCount += 1
                        if (frameCount % MATOAKA_PROGRESS_INTERVAL == 0) {
                            val elapsedSeconds =
                                (System.nanoTime() - startedAtNs).toDouble() /
                                    NANOS_PER_SECOND
                            Log.i(
                                TAG,
                                "stage=DEVELOPMENT_MATOAKA_CROSS_SOURCE state=RUNNING " +
                                    "completed=$frameCount total=$MATOAKA_FRAME_COUNT " +
                                    "throughput_fps=${frameCount / elapsedSeconds}"
                            )
                        }
                    } finally {
                        bitmap.recycle()
                    }
                }
            }
            check(frameCount == MATOAKA_FRAME_COUNT)
            val trace = File(temporaryRoot, TRACE_FILE)
            val receipt = JSONObject()
                .put(
                    "schema_version",
                    "blindassist.dual_loop_scene_scale_cross_source_receipt.v1"
                )
                .put("protocol_id", DEVELOPMENT_MATOAKA_PROTOCOL_ID)
                .put("status", "COMPLETE")
                .put("authority", "LOCKED_CROSS_SOURCE_DEVELOPMENT")
                .put("truth_read", false)
                .put("frame_count", frameCount)
                .put("baseline_feedback_trigger_count", baselineTriggerCount)
                .put("candidate_feedback_trigger_count", candidateTriggerCount)
                .put("risk_mutation_count", riskMutationCount)
                .put("trace_sha256", sha256File(trace))
                .put("input_manifest_sha256", MATOAKA_MANIFEST_SHA256)
                .put("backend", DetectorExecutionBackend.QUALCOMM_QNN_HTP.wireName)
                .put("model_sha256", sha256Asset(TfliteYoloDetector.MODEL_ASSET))
                .put("elapsed_ms", (System.nanoTime() - startedAtNs) / NANOS_PER_MILLISECOND)
                .put("device", deviceJson())
            atomicWrite(File(temporaryRoot, PRODUCER_RECEIPT), receipt.toString(2))
            check(temporaryRoot.renameTo(outputRoot)) {
                "Matoaka cross-source output publication failed"
            }
            Log.i(
                TAG,
                "stage=DEVELOPMENT_MATOAKA_CROSS_SOURCE state=COMPLETE " +
                    "completed=$frameCount baseline_triggers=$baselineTriggerCount " +
                    "candidate_triggers=$candidateTriggerCount"
            )
        } finally {
            baseline.close()
            candidate.close()
            detector.close()
        }
    }

    @Test
    fun verifyFormalInputAndQnnPrestart() {
        val inputRoot = inputRoot()
        val inventory = verifyInputInventory(inputRoot)
        verifySyntheticWholeChainOrderInvariance()
        val qnn = strictQnnDetector()
        try {
            val synthetic = Bitmap.createBitmap(EXPECTED_WIDTH, EXPECTED_HEIGHT, Bitmap.Config.ARGB_8888)
            try {
                val probe = qnn.detect(synthetic)
                assertEquals(FrameSize(EXPECTED_WIDTH, EXPECTED_HEIGHT), probe.frameSize)
                assertTrue(probe.detections.all { it.confidence.isFinite() })
            } finally {
                synthetic.recycle()
            }
        } finally {
            qnn.close()
        }

        val receipt = JSONObject()
            .put("schema_version", "blindassist.production_temporal_ab_prestart.v1")
            .put("protocol_id", PROTOCOL_ID)
            .put("status", "VALID")
            .put("decision_rgb_decoded", false)
            .put("synthetic_qnn_probe_completed", true)
            .put("synthetic_branch_order_invariance_completed", true)
            .put("backend", DetectorExecutionBackend.QUALCOMM_QNN_HTP.wireName)
            .put("qnn_maven_version", QNN_MAVEN_VERSION)
            .put("qnn_runtime_version", JSONArray(QnnDelegate.getVersion().toList()))
            .put("device", deviceJson())
            .put(
                "installed_app_apk_sha256",
                sha256File(File(targetContext.applicationInfo.sourceDir))
            )
            .put(
                "installed_test_apk_sha256",
                sha256File(File(testContext.applicationInfo.sourceDir))
            )
            .put("input", inventory)
            .put("candidate_output_written", false)
            .put("errors", JSONArray())
        atomicWrite(File(prestartRoot(), PRESTART_RECEIPT), receipt.toString(2))
        Log.i(TAG, "prestart=VALID frames=${inventory.getInt("frame_count")}")
    }

    @Test
    fun runFormalTruthBlindProducer() {
        try {
            runFormalTruthBlindProducerInternal()
        } catch (error: Throwable) {
            if (!File(baseRoot(), FORMAL_START_MARKER).exists()) {
                val failure = JSONObject()
                    .put("protocol_id", PROTOCOL_ID)
                    .put("implementation_id", IMPLEMENTATION_ID)
                    .put("status", "NOT_EVALUABLE_PRESTART")
                    .put("formal_start_consumed", false)
                    .put("error_class", error.javaClass.name)
                    .put("error_message", error.message)
                atomicWrite(
                    File(prestartRoot(), FORMAL_ENTRY_FAILURE_RECEIPT),
                    failure.toString(2)
                )
            }
            throw error
        }
    }

    private fun runFormalTruthBlindProducerInternal() {
        val prestartFile = File(prestartRoot(), PRESTART_RECEIPT)
        assertTrue("prestart receipt is missing", prestartFile.isFile)
        val prestart = JSONObject(prestartFile.readText(Charsets.UTF_8))
        assertEquals("VALID", prestart.getString("status"))
        assertEquals(PROTOCOL_ID, prestart.getString("protocol_id"))
        assertEquals(
            sha256File(File(targetContext.applicationInfo.sourceDir)),
            prestart.getString("installed_app_apk_sha256")
        )
        assertEquals(
            sha256File(File(testContext.applicationInfo.sourceDir)),
            prestart.getString("installed_test_apk_sha256")
        )
        assertTrue(prestart.getBoolean("synthetic_qnn_probe_completed"))
        assertTrue(prestart.getBoolean("synthetic_branch_order_invariance_completed"))
        assertEquals(EXPECTED_INVENTORY_SHA256, prestart.getJSONObject("input").getString("canonical_rgb_inventory_sha256"))
        assertEquals(EXPECTED_FRAME_COUNT, prestart.getJSONObject("input").getInt("frame_count"))
        val authorizationRoot = File(baseRoot(), AUTHORIZATION_DIRECTORY)
        val activationFile = File(authorizationRoot, ACTIVATION_RECEIPT)
        val implementationLockFile = File(authorizationRoot, IMPLEMENTATION_LOCK)
        assertTrue("device activation receipt is missing", activationFile.isFile)
        assertTrue("device implementation lock is missing", implementationLockFile.isFile)
        val activation = JSONObject(activationFile.readText(Charsets.UTF_8))
        val implementationLock =
            JSONObject(implementationLockFile.readText(Charsets.UTF_8))
        assertEquals(PROTOCOL_ID, activation.getString("protocol_id"))
        assertEquals(IMPLEMENTATION_ID, activation.getString("implementation_id"))
        assertEquals("ACTIVATED", activation.getString("status"))
        assertTrue(activation.getBoolean("formal_execution_authorized"))
        assertEquals(
            sha256File(implementationLockFile),
            activation.getString("implementation_lock_sha256")
        )
        assertEquals(PROTOCOL_ID, implementationLock.getString("protocol_id"))
        assertEquals(IMPLEMENTATION_ID, implementationLock.getString("implementation_id"))
        assertEquals("LOCKED", implementationLock.getString("status"))
        assertEquals(
            sha256File(prestartFile),
            implementationLock.getJSONObject("device_prestart").getString("sha256")
        )
        val currentAppApkSha256 =
            sha256File(File(targetContext.applicationInfo.sourceDir))
        val currentTestApkSha256 =
            sha256File(File(testContext.applicationInfo.sourceDir))
        assertEquals(
            currentAppApkSha256,
            implementationLock.getJSONObject("app_apk").getString("sha256")
        )
        assertEquals(
            currentTestApkSha256,
            implementationLock.getJSONObject("test_apk").getString("sha256")
        )
        assertEquals(
            currentAppApkSha256,
            activation.getString("installed_app_apk_sha256")
        )
        assertEquals(
            currentTestApkSha256,
            activation.getString("installed_test_apk_sha256")
        )

        val detector = strictQnnDetector()
        assertSameDevice(
            deviceJson(),
            implementationLock.getJSONObject("device_prestart").getJSONObject("device")
        )
        assertEquals(
            "formal QNN runtime drift from prestart",
            JSONArray(QnnDelegate.getVersion().toList()).toString(),
            implementationLock.getJSONObject("device_prestart")
                .getJSONArray("qnn_runtime_version")
                .toString()
        )
        val outputRoot = formalOutputRoot()
        val formalStart = File(baseRoot(), FORMAL_START_MARKER)
        assertTrue("formal-start marker already exists", !formalStart.exists())
        assertTrue("formal output already exists", !outputRoot.exists())
        val marker = JSONObject()
            .put("protocol_id", PROTOCOL_ID)
            .put("implementation_id", IMPLEMENTATION_ID)
            .put("state", "FORMAL_STARTED")
            .put("first_frame_inference_pending", true)

        var detectorInvocationCount = 0
        var traceRowCount = 0
        var completedFrames = 0
        var formalStartConsumed = false
        val traceTemporary = File(baseRoot(), "$TRACE_FILE.tmp")
        assertTrue("stale temporary trace exists", !traceTemporary.exists())
        val startedAtNs = System.nanoTime()
        try {
            traceTemporary.bufferedWriter(Charsets.UTF_8).use { writer ->
                SESSIONS.forEach { session ->
                    val rows = readFrameLedger(File(inputRoot(), "$session/frames.jsonl"))
                    val sessionOriginNs = rows.first().sourceCaptureTimestampNs
                    val a = branch(ObjectDetectorTemporalGeometryMode.NEUTRALIZE_OUTPUT)
                    val b = branch(ObjectDetectorTemporalGeometryMode.APPLY)
                    a.startSession(0L)
                    b.startSession(0L)

                    rows.forEach { frame ->
                        val sourceNowNs = frame.sourceCaptureTimestampNs - sessionOriginNs
                        val sourceNowMs = sourceNowNs / NANOS_PER_MILLISECOND
                        a.clock.nowMs = sourceNowMs
                        b.clock.nowMs = sourceNowMs
                        val imageFile = File(inputRoot(), "$session/${frame.rgbPath}")
                        check(sha256File(imageFile) == frame.rgbSha256) {
                            "formal RGB identity drift: $session/${frame.frameId}"
                        }
                        val bitmap = BitmapFactory.decodeFile(imageFile.absolutePath)
                            ?: error("decode failed: $session/${frame.frameId}")
                        try {
                            check(bitmap.width == EXPECTED_WIDTH && bitmap.height == EXPECTED_HEIGHT) {
                                "decoded size mismatch: $session/${frame.frameId}"
                            }
                            if (!formalStartConsumed) {
                                atomicCreateMarker(formalStart, marker.toString(2))
                                formalStartConsumed = true
                            }
                            val detectorResult = detector.detect(bitmap)
                            detectorInvocationCount += 1
                            check(detectorResult.frameSize == FrameSize(EXPECTED_WIDTH, EXPECTED_HEIGHT))
                            val immutableDetections = detectorResult.detections.toList()
                            val detectionHashBefore = canonicalDetectionsSha256(immutableDetections)
                            val stamp = FrameStamp(
                                frameId = frame.frameId.toLong(),
                                capturedAtNs = frame.sourceCaptureTimestampNs,
                                receivedAtNs = frame.sourceCaptureTimestampNs,
                                sourceId = session,
                                coordinateFrame = "CROWDBOT_RGB_STORED_PIXELS",
                                clockDomain = FrameClockDomain.REPLAY_TIMELINE
                            )
                            val aResult = a.kernel.processFrame(
                                detections = immutableDetections,
                                frameSize = detectorResult.frameSize,
                                profile = AlertProfile.STANDARD,
                                scenario = AssistScenario.GENERAL,
                                metrics = detectorResult.metrics,
                                feedbackGateway = a.feedback,
                                nowMs = sourceNowMs,
                                sourceFrame = stamp,
                                decisionAtNs = frame.sourceCaptureTimestampNs
                            )
                            check(canonicalDetectionsSha256(immutableDetections) == detectionHashBefore)
                            val bResult = b.kernel.processFrame(
                                detections = immutableDetections,
                                frameSize = detectorResult.frameSize,
                                profile = AlertProfile.STANDARD,
                                scenario = AssistScenario.GENERAL,
                                metrics = detectorResult.metrics,
                                feedbackGateway = b.feedback,
                                nowMs = sourceNowMs,
                                sourceFrame = stamp,
                                decisionAtNs = frame.sourceCaptureTimestampNs
                            )
                            check(canonicalDetectionsSha256(immutableDetections) == detectionHashBefore)
                            val aPreTemporalHash =
                                canonicalRiskSha256(aResult.evaluation.preTemporalRisk)
                            val bPreTemporalHash =
                                canonicalRiskSha256(bResult.evaluation.preTemporalRisk)
                            check(aPreTemporalHash == bPreTemporalHash) {
                                "branch analyzer drift: $session/${frame.frameId}"
                            }
                            writeTraceRow(
                                writer,
                                session,
                                frame,
                                BRANCH_A,
                                detectionHashBefore,
                                aPreTemporalHash,
                                immutableDetections.size,
                                aResult
                            )
                            writeTraceRow(
                                writer,
                                session,
                                frame,
                                BRANCH_B,
                                detectionHashBefore,
                                bPreTemporalHash,
                                immutableDetections.size,
                                bResult
                            )
                            traceRowCount += 2
                            completedFrames += 1
                            if (completedFrames % PROGRESS_INTERVAL == 0) {
                                val elapsedSeconds =
                                    (System.nanoTime() - startedAtNs).toDouble() / NANOS_PER_SECOND
                                val throughput = completedFrames / elapsedSeconds
                                val etaSeconds =
                                    (EXPECTED_FRAME_COUNT - completedFrames) / throughput
                                Log.i(
                                    TAG,
                                    "stage=FORMAL_PRODUCER state=RUNNING session=$session " +
                                        "completed=$completedFrames total=$EXPECTED_FRAME_COUNT " +
                                        "throughput_fps=$throughput eta_seconds=$etaSeconds"
                                )
                            }
                        } finally {
                            bitmap.recycle()
                        }
                    }
                    a.close()
                    b.close()
                }
            }
            check(detectorInvocationCount == EXPECTED_FRAME_COUNT)
            check(traceRowCount == EXPECTED_TRACE_ROWS)
            assertTrue("formal output directory could not be created", outputRoot.mkdirs())
            val trace = File(outputRoot, TRACE_FILE)
            check(traceTemporary.renameTo(trace)) { "atomic trace publication failed" }
            val receipt = JSONObject()
                .put("schema_version", "blindassist.production_temporal_ab_producer_receipt.v1")
                .put("protocol_id", PROTOCOL_ID)
                .put("implementation_id", IMPLEMENTATION_ID)
                .put("status", "COMPLETE")
                .put("truth_joined", false)
                .put("backend", DetectorExecutionBackend.QUALCOMM_QNN_HTP.wireName)
                .put("qnn_maven_version", QNN_MAVEN_VERSION)
                .put("qnn_runtime_version", JSONArray(QnnDelegate.getVersion().toList()))
                .put("model_sha256", sha256Asset(TfliteYoloDetector.MODEL_ASSET))
                .put("labels_sha256", sha256Asset(TfliteYoloDetector.LABELS_ASSET))
                .put("input_inventory_sha256", EXPECTED_INVENTORY_SHA256)
                .put("session_count", SESSIONS.size)
                .put("frame_count", completedFrames)
                .put("detector_invocation_count", detectorInvocationCount)
                .put("trace_row_count", traceRowCount)
                .put("trace_sha256", sha256File(trace))
                .put("implementation_lock_sha256", sha256File(implementationLockFile))
                .put("activation_sha256", sha256File(activationFile))
                .put("elapsed_ms", (System.nanoTime() - startedAtNs) / NANOS_PER_MILLISECOND)
                .put("device", deviceJson())
                .put("failure_count", 0)
                .put("formal_start_consumed", formalStart.exists())
            atomicWrite(File(outputRoot, PRODUCER_RECEIPT), receipt.toString(2))
            val elapsedSeconds =
                (System.nanoTime() - startedAtNs).toDouble() / NANOS_PER_SECOND
            Log.i(
                TAG,
                "stage=FORMAL_PRODUCER state=COMPLETE session=ALL completed=$completedFrames " +
                    "total=$EXPECTED_FRAME_COUNT throughput_fps=${completedFrames / elapsedSeconds} " +
                    "eta_seconds=0"
            )
        } catch (error: Throwable) {
            val markerConsumed = formalStart.exists()
            val failure = JSONObject()
                .put("protocol_id", PROTOCOL_ID)
                .put("implementation_id", IMPLEMENTATION_ID)
                .put(
                    "status",
                    if (markerConsumed) "INVALID_EXECUTION" else "NOT_EVALUABLE_PRESTART"
                )
                .put("formal_start_consumed", markerConsumed)
                .put("completed_frames", completedFrames)
                .put("detector_invocation_count", detectorInvocationCount)
                .put("trace_row_count", traceRowCount)
                .put("error_class", error.javaClass.name)
                .put("error_message", error.message)
            val failureRoot = if (markerConsumed) {
                outputRoot.also { it.mkdirs() }
            } else {
                prestartRoot()
            }
            atomicWrite(File(failureRoot, FAILURE_RECEIPT), failure.toString(2))
            Log.e(
                TAG,
                "stage=FORMAL_PRODUCER state=${failure.getString("status")} session=UNKNOWN " +
                    "completed=$completedFrames total=$EXPECTED_FRAME_COUNT " +
                    "throughput_fps=NOT_AVAILABLE eta_seconds=NOT_AVAILABLE",
                error
            )
            throw error
        } finally {
            detector.close()
            if (traceTemporary.exists()) {
                traceTemporary.delete()
            }
        }
    }

    private fun branch(mode: ObjectDetectorTemporalGeometryMode): Branch {
        val clock = MutableFeedbackClock()
        val feedback = FeedbackController(
            speechOutput = AcceptingSpeechOutput(),
            hapticOutput = HapticOutput { _ -> true },
            clock = clock,
            initialSettings = FeedbackRuntimeSettings()
        )
        val tracker = TemporalRiskTracker(
            TemporalRiskTrackerConfig(objectDetectorTemporalGeometryMode = mode)
        )
        return Branch(
            kernel = AssistDecisionKernel(
                assistEngine = AssistEngine(
                    riskAnalyzer = RiskAnalyzer(),
                    temporalRiskTracker = tracker
                )
            ),
            feedback = feedback,
            clock = clock
        )
    }

    private fun verifySyntheticWholeChainOrderInvariance() {
        fun replay(aFirst: Boolean): Pair<List<String>, List<String>> {
            val a = branch(ObjectDetectorTemporalGeometryMode.NEUTRALIZE_OUTPUT)
            val b = branch(ObjectDetectorTemporalGeometryMode.APPLY)
            val aResults = mutableListOf<String>()
            val bResults = mutableListOf<String>()
            val metrics = DetectorMetrics(
                totalMs = 4L,
                preprocessMs = 1L,
                inferenceMs = 2L,
                postprocessMs = 1L,
                fps = 10f,
                modelStatus = "SYNTHETIC_PRESTART"
            )
            val boxes = listOf(
                BoundingBox(450f, 120f, 520f, 280f),
                BoundingBox(445f, 130f, 525f, 310f),
                BoundingBox(440f, 140f, 530f, 340f),
                BoundingBox(435f, 150f, 535f, 370f)
            )
            fun process(branch: Branch, detection: Detection, nowMs: Long): String {
                branch.clock.nowMs = nowMs
                val result = branch.kernel.processFrame(
                    detections = listOf(detection),
                    frameSize = detection.frameSize,
                    profile = AlertProfile.STANDARD,
                    scenario = AssistScenario.GENERAL,
                    metrics = metrics,
                    feedbackGateway = branch.feedback,
                    nowMs = nowMs,
                    decisionAtNs = nowMs * NANOS_PER_MILLISECOND
                )
                return listOf(
                    canonicalRiskSha256(result.evaluation.rawRisk),
                    canonicalRiskSha256(result.evaluation.stableRisk),
                    result.feedbackDecision.triggered.toString(),
                    result.feedbackDecision.reason.name,
                    result.feedbackDecision.speechTriggered.toString(),
                    result.feedbackDecision.vibrationTriggered.toString(),
                    result.evaluation.riskEvent.eventId ?: "NONE",
                    result.evaluation.riskEvent.state?.name ?: "NONE",
                    result.evaluation.riskEvent.active.toString(),
                    result.evaluation.riskEvent.suppressesFeedback.toString()
                ).joinToString("\t")
            }
            try {
                a.startSession(0L)
                b.startSession(0L)
                boxes.forEachIndexed { index, box ->
                    val nowMs = 100L + index * 100L
                    val detection = Detection(
                        classId = 0,
                        label = "person",
                        confidence = 0.9f,
                        boundingBox = box,
                        frameSize = FrameSize(EXPECTED_WIDTH, EXPECTED_HEIGHT)
                    )
                    if (aFirst) {
                        aResults += process(a, detection, nowMs)
                        bResults += process(b, detection, nowMs)
                    } else {
                        bResults += process(b, detection, nowMs)
                        aResults += process(a, detection, nowMs)
                    }
                }
            } finally {
                a.close()
                b.close()
            }
            return aResults to bResults
        }

        val aThenB = replay(aFirst = true)
        val bThenA = replay(aFirst = false)
        assertEquals(aThenB, bThenA)
        assertTrue("synthetic mutation did not distinguish A from B", aThenB.first != aThenB.second)
    }

    private fun strictQnnDetector(): TfliteYoloDetector {
        val profile = ProductionDeviceProfile(
            socModel = Build.SOC_MODEL.orEmpty(),
            supportedAbis = Build.SUPPORTED_ABIS.toList()
        )
        assertTrue("device is outside the production QNN route scope", ProductionDetectorRoutePolicy.isQnnProbeEligible(profile))
        System.loadLibrary("cdsprpc")
        assertTrue(
            "QNN HTP FP16 capability is unavailable",
            QnnDelegate.checkCapability(QnnDelegate.Capability.HTP_RUNTIME_FP16)
        )
        var delegate: QnnDelegate? = null
        val detector = TfliteYoloDetector(
            context = testContext,
            executionBackend = DetectorExecutionBackend.QUALCOMM_QNN_HTP,
            externalInterpreterOptionsFactory = {
                val options = QnnDelegate.Options().apply {
                    setBackendType(QnnDelegate.Options.BackendType.HTP_BACKEND)
                    // The QNN delegate is loaded into the target app process. Its DSP skeleton
                    // must therefore be resolved from the production app's native namespace,
                    // not from the separate instrumentation APK namespace.
                    setSkelLibraryDir(targetContext.applicationInfo.nativeLibraryDir)
                    setHtpPrecision(QnnDelegate.Options.HtpPrecision.HTP_PRECISION_FP16)
                    setHtpPerformanceMode(
                        QnnDelegate.Options.HtpPerformanceMode.HTP_PERFORMANCE_SUSTAINED_HIGH_PERFORMANCE
                    )
                    setLogLevel(QnnDelegate.Options.LogLevel.LOG_LEVEL_INFO)
                    setCacheDir(targetContext.codeCacheDir.absolutePath)
                    setModelToken(QNN_MODEL_TOKEN)
                }
                QnnDelegate(options).also {
                    check(it.isAvailable) { "QNN delegate is unavailable" }
                    delegate = it
                }.let {
                    Interpreter.Options()
                        .setNumThreads(CPU_THREADS)
                        .addDelegate(it)
                }
            },
            externalBackendCloser = {
                delegate?.close()
                delegate = null
            }
        )
        assertTrue("strict QNN detector initialization failed: ${detector.statusMessage}", detector.isReady)
        assertEquals(DetectorExecutionBackend.QUALCOMM_QNN_HTP, detector.executionBackend)
        return detector
    }

    private fun verifyInputInventory(root: File): JSONObject {
        assertTrue("input root is missing: ${root.absolutePath}", root.isDirectory)
        val global = MessageDigest.getInstance("SHA-256")
        var frameCount = 0
        var totalBytes = 0L
        val sessionsJson = JSONArray()
        SESSIONS.forEach { session ->
            val ledger = File(root, "$session/frames.jsonl")
            assertEquals(EXPECTED_LEDGER_SHA256.getValue(session), sha256File(ledger))
            val rows = readFrameLedger(ledger)
            assertEquals(EXPECTED_SESSION_FRAME_COUNTS.getValue(session), rows.size)
            var previousTimestamp: Long? = null
            val frameIds = mutableSetOf<String>()
            val timestamps = mutableSetOf<Long>()
            var sessionBytes = 0L
            val sessionDigest = MessageDigest.getInstance("SHA-256")
            rows.forEach { frame ->
                check(frameIds.add(frame.frameId)) { "duplicate frame id: $session/${frame.frameId}" }
                check(timestamps.add(frame.sourceCaptureTimestampNs)) { "duplicate timestamp: $session/${frame.frameId}" }
                previousTimestamp?.let { check(frame.sourceCaptureTimestampNs > it) }
                previousTimestamp = frame.sourceCaptureTimestampNs
                val image = File(root, "$session/${frame.rgbPath}")
                val actualHash = sha256File(image)
                check(actualHash == frame.rgbSha256) { "RGB hash mismatch: $session/${frame.frameId}" }
                val (width, height) = pngDimensions(image)
                check(width == EXPECTED_WIDTH && height == EXPECTED_HEIGHT)
                val canonical = (
                    "$session\t${frame.frameId}\t${frame.sourceCaptureTimestampNs}\t${frame.rgbPath}\t" +
                        "$actualHash\t${image.length()}\t$width\t$height\n"
                    ).toByteArray(Charsets.UTF_8)
                sessionDigest.update(canonical)
                global.update(canonical)
                sessionBytes += image.length()
                frameCount += 1
            }
            totalBytes += sessionBytes
            sessionsJson.put(
                JSONObject()
                    .put("session_id", session)
                    .put("frame_count", rows.size)
                    .put("rgb_total_bytes", sessionBytes)
                    .put("canonical_rgb_inventory_sha256", hex(sessionDigest.digest()))
            )
        }
        val globalHash = hex(global.digest())
        assertEquals(EXPECTED_FRAME_COUNT, frameCount)
        assertEquals(EXPECTED_RGB_BYTES, totalBytes)
        assertEquals(EXPECTED_INVENTORY_SHA256, globalHash)
        return JSONObject()
            .put("frame_count", frameCount)
            .put("rgb_total_bytes", totalBytes)
            .put("canonical_rgb_inventory_sha256", globalHash)
            .put("dimensions", "${EXPECTED_WIDTH}x$EXPECTED_HEIGHT")
            .put("sessions", sessionsJson)
    }

    private fun readFrameLedger(path: File): List<FrameLedgerRow> =
        path.readLines(Charsets.UTF_8)
            .filter { it.isNotBlank() }
            .map { line ->
                val row = JSONObject(line)
                FrameLedgerRow(
                    frameId = row.getString("frame_id"),
                    sourceCaptureTimestampNs = row.getLong("source_capture_timestamp_ns"),
                    rgbPath = row.getString("rgb_path"),
                    rgbSha256 = row.getString("rgb_sha256")
                )
            }

    private fun writeTraceRow(
        writer: BufferedWriter,
        sessionId: String,
        frame: FrameLedgerRow,
        branchId: String,
        detectionHash: String,
        preTemporalHash: String,
        detectionCount: Int,
        result: com.linnan.blindassist.session.AssistFrameResult
    ) {
        val row = JSONObject()
            .put("protocol_id", PROTOCOL_ID)
            .put("implementation_id", IMPLEMENTATION_ID)
            .put("session_id", sessionId)
            .put("frame_id", frame.frameId)
            .put("source_capture_timestamp_ns", frame.sourceCaptureTimestampNs)
            .put("branch_id", branchId)
            .put("pre_temporal_raw_risk_sha256", preTemporalHash)
            .put("detector_output_sha256", detectionHash)
            .put("detection_count", detectionCount)
            .put("raw_risk", riskJson(result.evaluation.rawRisk))
            .put("stable_risk", riskJson(result.evaluation.stableRisk))
            .put("feedback_triggered", result.feedbackDecision.triggered)
            .put("feedback_reason", result.feedbackDecision.reason.name)
            .put("speech_triggered", result.feedbackDecision.speechTriggered)
            .put("vibration_triggered", result.feedbackDecision.vibrationTriggered)
            .put(
                "risk_event",
                JSONObject()
                    .put("event_id", result.evaluation.riskEvent.eventId ?: JSONObject.NULL)
                    .put("state", result.evaluation.riskEvent.state?.name ?: JSONObject.NULL)
                    .put("active", result.evaluation.riskEvent.active)
                    .put("suppresses_feedback", result.evaluation.riskEvent.suppressesFeedback)
                    .put("clear_reason", result.evaluation.riskEvent.clearReason?.name ?: JSONObject.NULL)
            )
            .put(
                "timing",
                JSONObject()
                    .put("detector_total_ms", result.evaluation.metrics.totalMs)
                    .put("preprocess_ms", result.evaluation.metrics.preprocessMs)
                    .put("inference_ms", result.evaluation.metrics.inferenceMs)
                    .put("postprocess_ms", result.evaluation.metrics.postprocessMs)
            )
            .put("failure", JSONObject.NULL)
        writer.write(row.toString())
        writer.newLine()
    }

    private fun riskJson(risk: RiskResult): JSONObject = JSONObject()
        .put("level", risk.level.name)
        .put("direction", risk.direction.name)
        .put("proximity", risk.proximity.name)
        .put("message", risk.message)
        .put("urgency_score", risk.urgencyScore.toDouble())
        .put("risk_score", risk.riskScore.toDouble())
        .put("approach_trend", risk.approachTrend.name)
        .put("evidence_state", risk.evidenceState.name)
        .put("fusion_summary", risk.scoreBreakdown.fusionSummary)
        .put("approach_score", risk.scoreBreakdown.approachTrend.toDouble())
        .put("total_score", risk.scoreBreakdown.total.toDouble())
        .put("source_detection", risk.sourceDetection?.let(::detectionJson) ?: JSONObject.NULL)

    private fun detectionJson(detection: Detection): JSONObject = JSONObject()
        .put("class_id", detection.classId)
        .put("label", detection.label)
        .put("confidence", detection.confidence.toDouble())
        .put("left", detection.boundingBox.left.toDouble())
        .put("top", detection.boundingBox.top.toDouble())
        .put("right", detection.boundingBox.right.toDouble())
        .put("bottom", detection.boundingBox.bottom.toDouble())
        .put("frame_width", detection.frameSize.width)
        .put("frame_height", detection.frameSize.height)
        .put("source", detection.source.name)
        .put("temporal_promotion_eligible", detection.temporalPromotionEligible)

    private fun canonicalDetectionsSha256(detections: List<Detection>): String {
        val text = buildString {
            detections.forEachIndexed { index, detection ->
                append(index).append('\t')
                append(detection.classId).append('\t')
                append(detection.label.length).append(':').append(detection.label).append('\t')
                append(detection.confidence.toRawBits()).append('\t')
                append(detection.boundingBox.left.toRawBits()).append('\t')
                append(detection.boundingBox.top.toRawBits()).append('\t')
                append(detection.boundingBox.right.toRawBits()).append('\t')
                append(detection.boundingBox.bottom.toRawBits()).append('\t')
                append(detection.frameSize.width).append('\t')
                append(detection.frameSize.height).append('\t')
                append(detection.source.name).append('\t')
                append(detection.temporalPromotionEligible).append('\t')
                val distance = detection.distanceEvidence
                if (distance == null) {
                    append("DISTANCE_NONE")
                } else {
                    append(distance.band.name).append('\t')
                    append(distance.confidence.toRawBits()).append('\t')
                    append(distance.source.name).append('\t')
                    append(distance.relativeDepthScore.toRawBits())
                }
                append('\n')
            }
        }
        return sha256Bytes(text.toByteArray(Charsets.UTF_8))
    }

    private fun canonicalRiskSha256(risk: RiskResult): String {
        val sourceHash = risk.sourceDetection?.let { canonicalDetectionsSha256(listOf(it)) } ?: "NONE"
        val text = listOf(
            risk.level.name,
            risk.direction.name,
            risk.proximity.name,
            risk.message,
            risk.urgencyScore.toRawBits().toString(),
            risk.riskScore.toRawBits().toString(),
            risk.approachTrend.name,
            risk.evidenceState.name,
            risk.scoreBreakdown.fusionSummary,
            risk.scoreBreakdown.confidence.toRawBits().toString(),
            risk.scoreBreakdown.classWeight.toRawBits().toString(),
            risk.scoreBreakdown.directionWeight.toRawBits().toString(),
            risk.scoreBreakdown.proximityWeight.toRawBits().toString(),
            risk.scoreBreakdown.bottomPosition.toRawBits().toString(),
            risk.scoreBreakdown.area.toRawBits().toString(),
            risk.scoreBreakdown.centerLane.toRawBits().toString(),
            risk.scoreBreakdown.distanceEvidence.toRawBits().toString(),
            risk.scoreBreakdown.approachTrend.toRawBits().toString(),
            risk.scoreBreakdown.total.toRawBits().toString(),
            risk.distanceEvidence?.let {
                listOf(
                    it.band.name,
                    it.confidence.toRawBits().toString(),
                    it.source.name,
                    it.relativeDepthScore.toRawBits().toString()
                ).joinToString(":")
            } ?: "DISTANCE_NONE",
            sourceHash
        ).joinToString("\t")
        return sha256Bytes(text.toByteArray(Charsets.UTF_8))
    }

    private fun pngDimensions(file: File): Pair<Int, Int> {
        val header = ByteArray(24)
        file.inputStream().use { stream ->
            check(stream.read(header) == header.size) { "short PNG header: ${file.absolutePath}" }
        }
        check(header.copyOfRange(0, 8).contentEquals(PNG_SIGNATURE))
        check(String(header, 12, 4, Charsets.US_ASCII) == "IHDR")
        val buffer = ByteBuffer.wrap(header, 16, 8).order(ByteOrder.BIG_ENDIAN)
        return buffer.int to buffer.int
    }

    private fun sha256Asset(name: String): String =
        testContext.assets.open(name).use { stream ->
            val digest = MessageDigest.getInstance("SHA-256")
            val buffer = ByteArray(DEFAULT_BUFFER_SIZE)
            while (true) {
                val count = stream.read(buffer)
                if (count < 0) break
                digest.update(buffer, 0, count)
            }
            hex(digest.digest())
        }

    private fun sha256File(file: File): String =
        file.inputStream().use { stream ->
            val digest = MessageDigest.getInstance("SHA-256")
            val buffer = ByteArray(DEFAULT_BUFFER_SIZE)
            while (true) {
                val count = stream.read(buffer)
                if (count < 0) break
                digest.update(buffer, 0, count)
            }
            hex(digest.digest())
        }

    private fun sha256Bytes(bytes: ByteArray): String =
        hex(MessageDigest.getInstance("SHA-256").digest(bytes))

    private fun hex(bytes: ByteArray): String = bytes.joinToString("") { "%02x".format(it) }

    private fun atomicWrite(target: File, content: String) {
        val parent = requireNotNull(target.parentFile)
        assertTrue("output parent could not be created", ensureDirectory(parent))
        val temporary = File(parent, "${target.name}.tmp")
        FileOutputStream(temporary).use { stream ->
            stream.write(content.toByteArray(Charsets.UTF_8))
            stream.write('\n'.code)
            stream.fd.sync()
        }
        check(temporary.renameTo(target)) { "atomic write failed: ${target.absolutePath}" }
    }

    private fun atomicCreateMarker(target: File, content: String) {
        val parent = requireNotNull(target.parentFile)
        assertTrue("marker parent could not be created", ensureDirectory(parent))
        val bytes = ByteBuffer.wrap((content + "\n").toByteArray(Charsets.UTF_8))
        FileChannel.open(
            target.toPath(),
            StandardOpenOption.CREATE_NEW,
            StandardOpenOption.WRITE
        ).use { channel ->
            while (bytes.hasRemaining()) {
                channel.write(bytes)
            }
            channel.force(true)
        }
    }

    private fun ensureDirectory(directory: File): Boolean =
        directory.isDirectory || directory.mkdirs()

    private fun deviceJson(): JSONObject = JSONObject()
        .put("manufacturer", Build.MANUFACTURER)
        .put("model", Build.MODEL)
        .put("soc_model", Build.SOC_MODEL)
        .put("supported_abis", JSONArray(Build.SUPPORTED_ABIS.toList()))
        .put("android_release", Build.VERSION.RELEASE)
        .put("sdk_int", Build.VERSION.SDK_INT)

    private fun assertSameDevice(actual: JSONObject, expected: JSONObject) {
        for (field in listOf("manufacturer", "model", "soc_model", "android_release")) {
            assertEquals("formal device field drift: $field", expected.getString(field), actual.getString(field))
        }
        assertEquals(
            "formal device SDK drift",
            expected.getInt("sdk_int"),
            actual.getInt("sdk_int")
        )
        assertEquals(
            "formal device ABI drift",
            expected.getJSONArray("supported_abis").toString(),
            actual.getJSONArray("supported_abis").toString()
        )
    }

    private fun baseRoot(): File =
        File(requireNotNull(targetContext.getExternalFilesDir(null)), BASE_DIRECTORY)

    private fun inputRoot(): File = File(baseRoot(), "input")

    private fun prestartRoot(): File = File(baseRoot(), "prestart")

    private fun formalOutputRoot(): File = File(baseRoot(), "output/device-producer")

    private data class FrameLedgerRow(
        val frameId: String,
        val sourceCaptureTimestampNs: Long,
        val rgbPath: String,
        val rgbSha256: String
    )

    private data class Branch(
        val kernel: AssistDecisionKernel,
        val feedback: FeedbackController,
        val clock: MutableFeedbackClock
    ) {
        fun startSession(nowMs: Long) {
            feedback.resetSession()
            kernel.startSession(nowMs)
        }

        fun close() {
            feedback.shutdown()
        }
    }

    private class MutableFeedbackClock(var nowMs: Long = 0L) : FeedbackClock {
        override fun nowMs(): Long = nowMs
    }

    private class AcceptingSpeechOutput : SpeechOutput {
        override val ready: Boolean = true
        override fun setLanguage(language: AppLanguage) = Unit
        override fun speak(message: String, utteranceId: String): Boolean = true
        override fun shutdown() = Unit
    }

    companion object {
        private const val TAG = "ProductionTemporalAb"
        private const val PROTOCOL_ID = "DUAL_LOOP_PRODUCTION_TEMPORAL_GEOMETRY_FACTORIAL_AB_R0"
        private const val IMPLEMENTATION_ID = "PRODUCTION_TEMPORAL_GEOMETRY_FACTORIAL_AB_IMPL_R0"
        private const val BASE_DIRECTORY = "dual_loop_production_temporal_ab_r0"
        private const val DEVELOPMENT_DUMP_DIRECTORY = "development-multitrack-dump-r0"
        private const val DEVELOPMENT_ACTIVE_REPLAY_DIRECTORY =
            "development-scene-scale-active-r1"
        private const val DEVELOPMENT_ACTIVE_PROTOCOL_ID =
            "DUAL_LOOP_SCENE_SCALE_VETO_R1"
        private const val DEVELOPMENT_MATOAKA_DIRECTORY =
            "development-matoaka-cross-source-r1"
        private const val DEVELOPMENT_MATOAKA_PROTOCOL_ID =
            "DUAL_LOOP_SCENE_SCALE_VETO_R1_CROSS_SOURCE_MATOAKA"
        private const val MATOAKA_SOURCE_ID =
            "wikimedia_commons_matoaka_west_virginia_walk_2019"
        private const val MATOAKA_VIDEO_SHA256 =
            "017f860e002c75d093206772800bd68cb1c19f226b74e4d5a933798916347821"
        private const val MATOAKA_MANIFEST_SHA256 =
            "97cb1c1ee68ab0b9a401f085c6819e696cbd951e01cb2c576c2c4615247be5f2"
        private const val PRESTART_RECEIPT = "prestart_receipt.json"
        private const val AUTHORIZATION_DIRECTORY = "authorization"
        private const val ACTIVATION_RECEIPT = "activation.json"
        private const val IMPLEMENTATION_LOCK = "implementation_lock.json"
        private const val FORMAL_ENTRY_FAILURE_RECEIPT = "formal_entry_failure_receipt.json"
        private const val FORMAL_START_MARKER = "formal_start.json"
        private const val TRACE_FILE = "trace.jsonl"
        private const val PRODUCER_RECEIPT = "producer_receipt.json"
        private const val FAILURE_RECEIPT = "failure_receipt.json"
        private const val BRANCH_A = "PRODUCTION_SEMANTIC_WITH_OBJECT_DETECTOR_TEMPORAL_GEOMETRY_NEUTRALIZED"
        private const val BRANCH_B = "CURRENT_FULL_PRODUCTION_TEMPORAL_GEOMETRY"
        private const val QNN_MAVEN_VERSION = "2.47.0"
        private const val QNN_MODEL_TOKEN = "blindassist_yolo11n_fp16_320_qnn_2_47_production_v1"
        private const val CPU_THREADS = 4
        private const val EXPECTED_FRAME_COUNT = 4422
        private const val EXPECTED_BASELINE_TRIGGER_COUNT = 373
        private const val EXPECTED_ACTIVE_TRIGGER_COUNT = 357
        private const val MATOAKA_FRAME_COUNT = 10724
        private const val MATOAKA_FRAME_STEP_NS = 100_000_000L
        private const val MATOAKA_PROGRESS_INTERVAL = 250
        private const val EXPECTED_TRACE_ROWS = 8844
        private const val EXPECTED_WIDTH = 640
        private const val EXPECTED_HEIGHT = 480
        private const val EXPECTED_RGB_BYTES = 2612679375L
        private const val EXPECTED_INVENTORY_SHA256 =
            "45621b226b4f6286962ec39c548234f92c3a34331cc4a1b2c413ef0bd3f7dd3b"
        private const val NANOS_PER_MILLISECOND = 1_000_000L
        private const val NANOS_PER_SECOND = 1_000_000_000.0
        private const val PROGRESS_INTERVAL = 50
        private val PNG_SIGNATURE = byteArrayOf(
            0x89.toByte(), 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a
        )
        private val SESSIONS = listOf(
            "defaced_2021-03-27-11-51-18_filtered_lidar_odom",
            "defaced_2021-03-27-11-55-00_filtered_lidar_odom"
        )
        private val EXPECTED_SESSION_FRAME_COUNTS = mapOf(
            SESSIONS[0] to 2239,
            SESSIONS[1] to 2183
        )
        private val EXPECTED_LEDGER_SHA256 = mapOf(
            SESSIONS[0] to "5b99e32b4d2ccbb75089cf1b1796c2e1d0a29c3ca0164feb3844e051051cc748",
            SESSIONS[1] to "88115772b70e0498875925aa909302e5662e0e0edd614f64b68ce1c433a39073"
        )
    }
}
