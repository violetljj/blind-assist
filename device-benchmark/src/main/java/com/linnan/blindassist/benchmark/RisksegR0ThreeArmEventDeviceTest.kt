package com.linnan.blindassist.benchmark

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.util.Log
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
import com.linnan.blindassist.risk.TraversabilityClass
import com.linnan.blindassist.risk.TraversabilitySegmentationAnalyzer
import com.linnan.blindassist.risk.TraversabilityTaxonomy
import com.linnan.blindassist.session.AssistDecisionKernel
import com.linnan.blindassist.session.AssistEngine
import com.linnan.blindassist.session.DetectorMetrics
import com.linnan.blindassist.vision.TfliteYoloDetector
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import java.io.File
import java.io.FileInputStream
import java.io.InputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.security.MessageDigest
import kotlin.math.roundToInt

/**
 * Benchmark-only execution of the frozen RISKSEG-R0 three-arm event contract.
 *
 * The host supplies a hash-closed compact view under target external files. This
 * class writes raw per-frame traces plus parent-event summaries and never changes
 * the production App detector path.
 */
@RunWith(AndroidJUnit4::class)
class RisksegR0ThreeArmEventDeviceTest {
    private val instrumentation = InstrumentationRegistry.getInstrumentation()
    private val testContext = instrumentation.context
    private val targetContext = instrumentation.targetContext
    private val arguments = InstrumentationRegistry.getArguments()

    @Test
    fun evaluateFrozenThreeArmParentEvents() {
        val seed = requireArgument(ARG_SEED).toInt()
        val expectedModelSha256 = requireSha256Argument(ARG_MODEL_SHA256)
        val expectedManifestSha256 = requireSha256Argument(ARG_MANIFEST_SHA256)
        val viewRoot = externalChild(requireArgument(ARG_VIEW_RELATIVE_ROOT))
        val outputRoot = externalChild(requireArgument(ARG_OUTPUT_RELATIVE_ROOT))
        assertFalse("refusing to overwrite event output", outputRoot.exists())
        assertTrue("event view root missing: $viewRoot", viewRoot.isDirectory)
        val manifestFile = File(viewRoot, MANIFEST_NAME)
        assertEquals(expectedManifestSha256, sha256File(manifestFile))
        val manifest = JSONObject(manifestFile.readText(Charsets.UTF_8))
        assertEquals(MANIFEST_SCHEMA, manifest.getString("schema_version"))
        assertEquals(PROTOCOL_ID, manifest.getString("protocol_id"))
        assertEquals(30, manifest.getInt("event_count"))
        assertEquals(30, manifest.getInt("source_session_count"))
        assertEquals(1_920, manifest.getInt("frame_count"))
        assertEquals(
            listOf(
                "walkable",
                "blocking_obstacle",
                "boundary_level_change",
                "unknown_nonwalkable"
            ),
            manifest.getJSONArray("class_order").strings()
        )
        assertEquals(FROZEN_YOLO_SHA256, sha256Asset(TfliteYoloDetector.MODEL_ASSET))
        assertEquals(expectedModelSha256, sha256Asset(RISKSEG_MODEL_ASSET))

        outputRoot.mkdirs()
        val traceFile = File(outputRoot, TRACE_NAME)
        val eventSummaries = JSONArray()
        val armAggregate = linkedMapOf(
            ARM_YOLO to ArmAggregate(),
            ARM_LEARNED to ArmAggregate(),
            ARM_ORACLE to ArmAggregate()
        )
        val yolo = TfliteYoloDetector(
            context = testContext,
            modelAssetName = TfliteYoloDetector.MODEL_ASSET,
            labelsAssetName = TfliteYoloDetector.LABELS_ASSET
        )
        assertTrue("frozen YOLO failed to initialize: ${yolo.statusMessage}", yolo.isReady)
        val learnedSegmenter = RisksegSegmenter(testContext, RISKSEG_MODEL_ASSET)
        val learnedAnalyzer = TraversabilitySegmentationAnalyzer(taxonomy = RisksegTaxonomy)
        val oracleAnalyzer = TraversabilitySegmentationAnalyzer(taxonomy = RisksegTaxonomy)
        val events = manifest.getJSONArray("events")
        val frameTimings = mutableListOf<Double>()

        try {
            traceFile.bufferedWriter(Charsets.UTF_8).use { trace ->
                for (eventIndex in 0 until events.length()) {
                    val event = events.getJSONObject(eventIndex)
                    val eventId = event.getString("parent_event_id")
                    val bucket = event.getString("bucket")
                    val positive = event.getBoolean("positive")
                    val scenario = AssistScenario.valueOf(event.getString("assist_scenario"))
                    val branches = linkedMapOf(
                        ARM_YOLO to Branch(),
                        ARM_LEARNED to Branch(),
                        ARM_ORACLE to Branch()
                    )
                    branches.values.forEach(Branch::reset)
                    val samples = linkedMapOf(
                        ARM_YOLO to mutableListOf<FrameObservation>(),
                        ARM_LEARNED to mutableListOf<FrameObservation>(),
                        ARM_ORACLE to mutableListOf<FrameObservation>()
                    )
                    val frames = event.getJSONArray("frames")
                    for (framePosition in 0 until frames.length()) {
                        val frameStart = System.nanoTime()
                        val frame = frames.getJSONObject(framePosition)
                        assertEquals(framePosition, frame.getInt("frame_index"))
                        val imageFile = checkedViewFile(
                            viewRoot,
                            frame.getString("image_path"),
                            frame.getString("image_sha256")
                        )
                        val oracleMaskFile = checkedViewFile(
                            viewRoot,
                            frame.getString("oracle_mask_path"),
                            frame.getString("oracle_mask_sha256")
                        )
                        val bitmap = requireNotNull(BitmapFactory.decodeFile(imageFile.absolutePath)) {
                            "cannot decode $imageFile"
                        }
                        try {
                            assertEquals(INPUT_WIDTH, bitmap.width)
                            assertEquals(INPUT_HEIGHT, bitmap.height)
                            val frameSize = FrameSize(bitmap.width, bitmap.height)
                            val yoloStart = System.nanoTime()
                            val yoloDetections = yolo.detect(bitmap).detections
                            val yoloMs = elapsedMs(yoloStart)
                            val learnedStart = System.nanoTime()
                            val learnedMask = learnedSegmenter.segment(bitmap)
                            val learnedDetections =
                                learnedAnalyzer.analyze(learnedMask, frameSize).riskDetections
                            val learnedMs = elapsedMs(learnedStart)
                            val oracleStart = System.nanoTime()
                            val oracleMask = decodeOracleMask(oracleMaskFile)
                            val oracleDetections =
                                oracleAnalyzer.analyze(oracleMask, frameSize).riskDetections
                            val oracleMs = elapsedMs(oracleStart)
                            val nowMs = frame.getLong("timestamp_ms")
                            val inputs = mapOf(
                                ARM_YOLO to Pair(yoloDetections, yoloMs),
                                ARM_LEARNED to Pair(learnedDetections, learnedMs),
                                ARM_ORACLE to Pair(oracleDetections, oracleMs)
                            )
                            for ((arm, input) in inputs) {
                                val observation = branches.getValue(arm).process(
                                    detections = input.first,
                                    frameSize = frameSize,
                                    scenario = scenario,
                                    nowMs = nowMs,
                                    perceptionMs = input.second
                                )
                                samples.getValue(arm) += observation
                                trace.append(
                                    JSONObject()
                                        .put("schema_version", TRACE_SCHEMA)
                                        .put("protocol_id", PROTOCOL_ID)
                                        .put("seed", seed)
                                        .put("model_sha256", expectedModelSha256)
                                        .put("parent_event_id", eventId)
                                        .put("event_candidate_id", event.getString("event_candidate_id"))
                                        .put("source_session_id", event.getString("source_session_id"))
                                        .put("bucket", bucket)
                                        .put("positive", positive)
                                        .put("arm", arm)
                                        .put("frame_index", framePosition)
                                        .put("source_frame_index", frame.getInt("source_frame_index"))
                                        .put("timestamp_ms", nowMs)
                                        .put("detection_count", input.first.size)
                                        .put("perception_ms", input.second)
                                        .put("actual_alert", observation.actualAlert)
                                        .put("feedback_reason", observation.feedbackReason)
                                        .put("raw_risk_level", observation.rawRiskLevel)
                                        .put("stable_risk_level", observation.stableRiskLevel)
                                        .put("risk_direction", observation.riskDirection)
                                        .put("risk_event_id", observation.riskEventId ?: JSONObject.NULL)
                                        .put("risk_event_state", observation.riskEventState ?: JSONObject.NULL)
                                        .put("risk_event_active", observation.riskEventActive)
                                        .put("risk_event_clear_reason", observation.riskEventClearReason ?: JSONObject.NULL)
                                        .toString()
                                )
                                trace.newLine()
                            }
                        } finally {
                            bitmap.recycle()
                        }
                        frameTimings += elapsedMs(frameStart)
                    }
                    for ((arm, observations) in samples) {
                        val summary = summarizeEvent(event, arm, observations)
                        eventSummaries.put(summary)
                        armAggregate.getValue(arm).add(summary)
                    }
                    trace.flush()
                    Log.i(TAG, "RISKSEG_EVENT_COMPLETE index=$eventIndex id=$eventId")
                }
            }
        } finally {
            learnedSegmenter.close()
            yolo.close()
        }

        val report = JSONObject()
            .put("schema_version", REPORT_SCHEMA)
            .put("protocol_id", PROTOCOL_ID)
            .put("seed", seed)
            .put("model_asset", RISKSEG_MODEL_ASSET)
            .put("model_sha256", expectedModelSha256)
            .put("frozen_yolo_sha256", FROZEN_YOLO_SHA256)
            .put("manifest_sha256", expectedManifestSha256)
            .put("decision_kernel_contract_id", AssistDecisionKernel.CONTRACT_ID)
            .put("event_count", events.length())
            .put("frame_count", manifest.getInt("frame_count"))
            .put("trace_file", TRACE_NAME)
            .put("trace_sha256", sha256File(traceFile))
            .put("event_summaries", eventSummaries)
            .put(
                "arm_aggregates",
                JSONObject().also { value ->
                    armAggregate.forEach { (arm, aggregate) -> value.put(arm, aggregate.toJson()) }
                }
            )
            .put("combined_frame_pipeline_p50_ms", percentile(frameTimings, 0.50))
            .put("combined_frame_pipeline_p95_ms", percentile(frameTimings, 0.95))
        val reportFile = File(outputRoot, REPORT_NAME)
        reportFile.writeText(report.toString(2) + "\n", Charsets.UTF_8)
        val receipt = JSONObject()
            .put("schema_version", RECEIPT_SCHEMA)
            .put("protocol_id", PROTOCOL_ID)
            .put("status", "RISKSEG_R0_THREE_ARM_DEVICE_EXECUTION_COMPLETE")
            .put("seed", seed)
            .put("model_sha256", expectedModelSha256)
            .put("manifest_sha256", expectedManifestSha256)
            .put("report_sha256", sha256File(reportFile))
            .put("trace_sha256", sha256File(traceFile))
            .put("failure_count", 0)
        File(outputRoot, RECEIPT_NAME).writeText(receipt.toString(2) + "\n", Charsets.UTF_8)
        Log.i(TAG, "RISKSEG_THREE_ARM_REPORT ${reportFile.absolutePath}")
    }

    private fun summarizeEvent(
        event: JSONObject,
        arm: String,
        observations: List<FrameObservation>
    ): JSONObject {
        val positive = event.getBoolean("positive")
        val alertable = event.optJSONArray("alertable_interval_frames")
        val passed = event.optJSONArray("passed_interval_frames")
        val hitFrames = if (positive) {
            requireNotNull(alertable).indicesInclusive().filter { observations[it].actualAlert }
        } else {
            emptyList()
        }
        val passedAlertFrames = if (positive) {
            requireNotNull(passed).indicesInclusive().filter { observations[it].actualAlert }
        } else {
            emptyList()
        }
        val falseAlert = !positive && observations.any { it.actualAlert }
        val passedCleared = positive && passedAlertFrames.isEmpty()
        val runtimeExitedByPassedEnd = if (positive) {
            !observations[requireNotNull(passed).getInt(1)].riskEventActive
        } else {
            false
        }
        return JSONObject()
            .put("parent_event_id", event.getString("parent_event_id"))
            .put("source_session_id", event.getString("source_session_id"))
            .put("bucket", event.getString("bucket"))
            .put("positive", positive)
            .put("arm", arm)
            .put("event_hit", positive && hitFrames.isNotEmpty())
            .put("first_alertable_alert_frame", hitFrames.firstOrNull() ?: JSONObject.NULL)
            .put("critical_miss", positive && hitFrames.isEmpty())
            .put("false_alert_event", falseAlert)
            .put("passed_cleared", passedCleared)
            .put("passed_alert_frame_count", passedAlertFrames.size)
            .put("runtime_exited_by_passed_end", runtimeExitedByPassedEnd)
            .put("delivered_alert_count", observations.count { it.actualAlert })
    }

    private fun decodeOracleMask(file: File): DenseSemanticMask {
        val bitmap = requireNotNull(BitmapFactory.decodeFile(file.absolutePath)) {
            "cannot decode oracle mask $file"
        }
        try {
            assertEquals(ORACLE_MASK_SIZE, bitmap.width)
            assertEquals(ORACLE_MASK_SIZE, bitmap.height)
            val pixels = IntArray(bitmap.width * bitmap.height)
            bitmap.getPixels(pixels, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)
            val classes = IntArray(pixels.size) { index -> (pixels[index] shr 16) and 0xFF }
            assertTrue("oracle class outside 0..3", classes.all { it in 0..3 })
            return DenseSemanticMask(bitmap.width, bitmap.height, classes)
        } finally {
            bitmap.recycle()
        }
    }

    private fun checkedViewFile(root: File, relative: String, expectedSha256: String): File {
        val file = File(root, relative).canonicalFile
        assertTrue("view path escaped root: $relative", file.path.startsWith(root.canonicalPath + File.separator))
        assertTrue("view file missing: $file", file.isFile)
        assertEquals(expectedSha256, sha256File(file))
        return file
    }

    private fun externalChild(relative: String): File {
        val base = requireNotNull(targetContext.getExternalFilesDir(null)).canonicalFile
        val child = File(base, relative).canonicalFile
        require(child.path.startsWith(base.path + File.separator)) { "external path escaped root" }
        return child
    }

    private fun requireArgument(name: String): String =
        requireNotNull(arguments.getString(name)?.takeIf(String::isNotBlank)) {
            "missing instrumentation argument: $name"
        }

    private fun requireSha256Argument(name: String): String =
        requireArgument(name).lowercase().also { value ->
            require(value.matches(Regex("[0-9a-f]{64}"))) { "$name is not SHA-256" }
        }

    private fun sha256Asset(name: String): String =
        testContext.assets.open(name).use(::sha256Stream)

    private fun sha256File(file: File): String =
        file.inputStream().use(::sha256Stream)

    private fun sha256Stream(stream: InputStream): String {
        val digest = MessageDigest.getInstance("SHA-256")
        val buffer = ByteArray(1024 * 1024)
        while (true) {
            val count = stream.read(buffer)
            if (count < 0) break
            digest.update(buffer, 0, count)
        }
        return digest.digest().joinToString("") { "%02x".format(it) }
    }

    private fun JSONArray.strings(): List<String> =
        List(length()) { index -> getString(index) }

    private fun JSONArray.indicesInclusive(): IntRange =
        getInt(0)..getInt(1)

    private fun percentile(values: List<Double>, quantile: Double): Double {
        if (values.isEmpty()) return 0.0
        val sorted = values.sorted()
        val index = ((sorted.size - 1) * quantile).roundToInt().coerceIn(sorted.indices)
        return sorted[index]
    }

    private fun elapsedMs(startNs: Long): Double =
        (System.nanoTime() - startNs) / 1_000_000.0

    private class Branch {
        private val kernel = AssistDecisionKernel(assistEngine = AssistEngine(riskAnalyzer = RiskAnalyzer()))
        private val gateway = PlannerGateway()

        fun reset() {
            kernel.reset()
            gateway.resetSession()
        }

        fun process(
            detections: List<Detection>,
            frameSize: FrameSize,
            scenario: AssistScenario,
            nowMs: Long,
            perceptionMs: Double
        ): FrameObservation {
            val result = kernel.processFrame(
                detections = detections,
                frameSize = frameSize,
                profile = AlertProfile.STANDARD,
                scenario = scenario,
                metrics = DetectorMetrics(
                    totalMs = perceptionMs.toLong(),
                    preprocessMs = 0L,
                    inferenceMs = perceptionMs.toLong(),
                    postprocessMs = 0L,
                    fps = 10f,
                    modelStatus = "ready"
                ),
                feedbackGateway = gateway,
                nowMs = nowMs
            )
            val event = result.evaluation.riskEvent
            return FrameObservation(
                actualAlert = result.feedbackDecision.triggered,
                feedbackReason = result.feedbackDecision.reason.name,
                rawRiskLevel = result.evaluation.rawRisk.level.name,
                stableRiskLevel = result.evaluation.stableRisk.level.name,
                riskDirection = result.evaluation.stableRisk.direction.name,
                riskEventId = event.eventId,
                riskEventState = event.state?.name,
                riskEventActive = event.active,
                riskEventClearReason = event.clearReason?.name
            )
        }
    }

    private class PlannerGateway : FeedbackGateway {
        override fun resetSession() = Unit

        override fun notify(
            risk: com.linnan.blindassist.risk.RiskResult,
            profile: AlertProfile,
            scenario: AssistScenario
        ): FeedbackDecision {
            val plan = FeedbackPlanner.planFor(
                risk = risk,
                profile = profile,
                scenario = scenario
            )
            return if (plan == null) {
                FeedbackDecision(null, false, FeedbackReason.NO_FEEDBACK_RISK)
            } else {
                FeedbackDecision(plan, true, FeedbackReason.TRIGGERED)
            }
        }
    }

    private class RisksegSegmenter(
        context: Context,
        assetName: String
    ) : AutoCloseable {
        private val interpreter: Interpreter
        private val inputScale: Float
        private val inputZeroPoint: Int
        private val input = ByteBuffer
            .allocateDirect(INPUT_WIDTH * INPUT_HEIGHT * 3)
            .order(ByteOrder.nativeOrder())
        private val output = ByteBuffer
            .allocateDirect(INPUT_WIDTH * INPUT_HEIGHT * CLASS_COUNT)
            .order(ByteOrder.nativeOrder())
        private val pixels = IntArray(INPUT_WIDTH * INPUT_HEIGHT)

        init {
            interpreter = Interpreter(
                loadMappedAsset(context, assetName),
                Interpreter.Options().setNumThreads(4)
            )
            interpreter.allocateTensors()
            val inputTensor = interpreter.getInputTensor(0)
            val outputTensor = interpreter.getOutputTensor(0)
            require(inputTensor.shape().contentEquals(intArrayOf(1, INPUT_HEIGHT, INPUT_WIDTH, 3)))
            require(outputTensor.shape().contentEquals(intArrayOf(1, INPUT_HEIGHT, INPUT_WIDTH, CLASS_COUNT)))
            require(inputTensor.dataType() == DataType.INT8)
            require(outputTensor.dataType() == DataType.INT8)
            inputScale = inputTensor.quantizationParams().scale
            inputZeroPoint = inputTensor.quantizationParams().zeroPoint
            require(inputScale > 0f && outputTensor.quantizationParams().scale > 0f)
        }

        fun segment(bitmap: Bitmap): DenseSemanticMask {
            require(bitmap.width == INPUT_WIDTH && bitmap.height == INPUT_HEIGHT)
            bitmap.getPixels(pixels, 0, INPUT_WIDTH, 0, 0, INPUT_WIDTH, INPUT_HEIGHT)
            input.rewind()
            pixels.forEach { pixel ->
                quantize((pixel shr 16) and 0xFF, 0)
                quantize((pixel shr 8) and 0xFF, 1)
                quantize(pixel and 0xFF, 2)
            }
            output.rewind()
            interpreter.run(input, output)
            output.rewind()
            val classes = IntArray(INPUT_WIDTH * INPUT_HEIGHT)
            for (index in classes.indices) {
                var bestClass = 0
                var bestScore = Int.MIN_VALUE
                repeat(CLASS_COUNT) { classId ->
                    val score = output.get().toInt()
                    if (score > bestScore) {
                        bestClass = classId
                        bestScore = score
                    }
                }
                classes[index] = bestClass
            }
            return DenseSemanticMask(INPUT_WIDTH, INPUT_HEIGHT, classes)
        }

        private fun quantize(channel: Int, channelIndex: Int) {
            val normalized = (channel / 255f - IMAGENET_MEAN[channelIndex]) /
                IMAGENET_STD[channelIndex]
            val quantized = (normalized / inputScale + inputZeroPoint)
                .roundToInt()
                .coerceIn(-128, 127)
            input.put(quantized.toByte())
        }

        override fun close() = interpreter.close()

        private fun loadMappedAsset(context: Context, name: String): MappedByteBuffer {
            val descriptor = context.assets.openFd(name)
            FileInputStream(descriptor.fileDescriptor).use { stream ->
                return stream.channel.map(
                    FileChannel.MapMode.READ_ONLY,
                    descriptor.startOffset,
                    descriptor.declaredLength
                )
            }
        }
    }

    private object RisksegTaxonomy : TraversabilityTaxonomy {
        override fun traversabilityFor(classId: Int): TraversabilityClass = when (classId) {
            WALKABLE -> TraversabilityClass.SAFE_TO_WALK
            BLOCKING_OBSTACLE -> TraversabilityClass.OBSTACLE
            BOUNDARY_LEVEL_CHANGE, UNKNOWN_NONWALKABLE -> TraversabilityClass.NOT_SAFE_TO_WALK
            else -> error("RISKSEG-R0 class outside 0..3: $classId")
        }

        override fun isNavigationHazard(classId: Int): Boolean =
            classId == BLOCKING_OBSTACLE || classId == BOUNDARY_LEVEL_CHANGE

        override fun riskLabelFor(classId: Int): String? = when (classId) {
            BLOCKING_OBSTACLE -> "generic obstacle"
            BOUNDARY_LEVEL_CHANGE -> "curb"
            else -> null
        }

        override fun permitsBoundaryDetection(classId: Int): Boolean =
            classId == BOUNDARY_LEVEL_CHANGE

        override fun isBoundaryEvidence(classId: Int): Boolean =
            classId == BOUNDARY_LEVEL_CHANGE
    }

    private data class FrameObservation(
        val actualAlert: Boolean,
        val feedbackReason: String,
        val rawRiskLevel: String,
        val stableRiskLevel: String,
        val riskDirection: String,
        val riskEventId: String?,
        val riskEventState: String?,
        val riskEventActive: Boolean,
        val riskEventClearReason: String?
    )

    private class ArmAggregate {
        var positiveEvents = 0
        var hitEvents = 0
        var criticalMisses = 0
        var negativeEvents = 0
        var falseAlertEvents = 0
        var passedEvents = 0
        var clearedEvents = 0
        var runtimeExitedEvents = 0

        fun add(summary: JSONObject) {
            if (summary.getBoolean("positive")) {
                positiveEvents += 1
                if (summary.getBoolean("event_hit")) hitEvents += 1
                if (summary.getBoolean("critical_miss")) criticalMisses += 1
                passedEvents += 1
                if (summary.getBoolean("passed_cleared")) clearedEvents += 1
                if (summary.getBoolean("runtime_exited_by_passed_end")) runtimeExitedEvents += 1
            } else {
                negativeEvents += 1
                if (summary.getBoolean("false_alert_event")) falseAlertEvents += 1
            }
        }

        fun toJson(): JSONObject = JSONObject()
            .put("positive_event_count", positiveEvents)
            .put("hit_event_count", hitEvents)
            .put("event_recall", if (positiveEvents == 0) 0.0 else hitEvents.toDouble() / positiveEvents)
            .put("critical_miss_count", criticalMisses)
            .put("negative_event_count", negativeEvents)
            .put("false_alert_event_count", falseAlertEvents)
            .put("passed_event_count", passedEvents)
            .put("cleared_event_count", clearedEvents)
            .put("clearance_rate", if (passedEvents == 0) 0.0 else clearedEvents.toDouble() / passedEvents)
            .put("runtime_exited_by_passed_end_count", runtimeExitedEvents)
    }

    companion object {
        private const val TAG = "RisksegR0ThreeArm"
        private const val PROTOCOL_ID = "RISKSEG_R0_EVENT_EVAL_V1"
        private const val MANIFEST_SCHEMA = "blindassist.riskseg_r0.device_event_view.v1"
        private const val TRACE_SCHEMA = "blindassist.riskseg_r0.three_arm_frame_trace.v1"
        private const val REPORT_SCHEMA = "blindassist.riskseg_r0.three_arm_device_report.v1"
        private const val RECEIPT_SCHEMA = "blindassist.riskseg_r0.three_arm_device_receipt.v1"
        private const val MANIFEST_NAME = "manifest.json"
        private const val TRACE_NAME = "frame_traces.jsonl"
        private const val REPORT_NAME = "event_report.json"
        private const val RECEIPT_NAME = "receipt.json"
        private const val ARG_SEED = "risksegSeed"
        private const val ARG_MODEL_SHA256 = "risksegExpectedModelSha256"
        private const val ARG_MANIFEST_SHA256 = "risksegExpectedManifestSha256"
        private const val ARG_VIEW_RELATIVE_ROOT = "risksegEvalViewRelativeRoot"
        private const val ARG_OUTPUT_RELATIVE_ROOT = "risksegEvalOutputRelativeRoot"
        private const val ARM_YOLO = "A_CURRENT_YOLO_ONLY"
        private const val ARM_LEARNED = "B_LEARNED_SEGMENTATION_ONLY"
        private const val ARM_ORACLE = "C_TRUTH_MASK_ORACLE_REFERENCE"
        private const val RISKSEG_MODEL_ASSET =
            "riskseg_pidnet/pidnet_s_512x288_4class_preflight_full_integer_quant.tflite"
        private const val FROZEN_YOLO_SHA256 =
            "00edb41a528b0a7e709c4af8ce3e685491492c4539274804e5cfc17a1a867cd2"
        private const val INPUT_WIDTH = 512
        private const val INPUT_HEIGHT = 288
        private const val ORACLE_MASK_SIZE = 256
        private const val CLASS_COUNT = 4
        private const val WALKABLE = 0
        private const val BLOCKING_OBSTACLE = 1
        private const val BOUNDARY_LEVEL_CHANGE = 2
        private const val UNKNOWN_NONWALKABLE = 3
        private val IMAGENET_MEAN = floatArrayOf(0.485f, 0.456f, 0.406f)
        private val IMAGENET_STD = floatArrayOf(0.229f, 0.224f, 0.225f)
    }
}
