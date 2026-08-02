package com.linnan.blindassist.session

import com.linnan.blindassist.alert.AlertProfile
import com.linnan.blindassist.alert.AssistScenario
import com.linnan.blindassist.feedback.FeedbackDecision
import com.linnan.blindassist.feedback.FeedbackGateway
import com.linnan.blindassist.feedback.FeedbackPlanner
import com.linnan.blindassist.feedback.FeedbackReason
import com.linnan.blindassist.model.BoundingBox
import com.linnan.blindassist.model.Detection
import com.linnan.blindassist.model.FrameSize
import com.linnan.blindassist.risk.RiskResult
import com.linnan.blindassist.vision.FrameClockDomain
import com.linnan.blindassist.vision.FrameStamp
import java.io.File
import java.nio.file.Files
import java.nio.file.StandardCopyOption
import kotlin.math.exp

/**
 * Frozen D40 source-only continuous-session replay.
 *
 * It turns a causal selected-target log-height slope into a one-second
 * bottom-center anchored box forecast, then sends that future observation
 * through an independent production kernel. Outcome truth is unavailable.
 */
object HftfD40ThorMagniContinuousTrackProjectedRiskReplayMain {
    private const val EXPECTED_SAMPLES = 530
    private const val EXPECTED_SESSIONS = 19
    private const val EXPECTED_FRAMES_PER_SAMPLE = 7
    private const val EXPECTED_UNIQUE_FRAMES = 3_710
    private const val EXPECTED_DETECTIONS = 14_364
    private const val MAXIMUM_CONTINUOUS_GAP_NS = 500_000_000L
    private const val FORECAST_HORIZON_S = 1.0

    @JvmStatic
    fun main(args: Array<String>) {
        require(args.size == 2) { "usage: <detections.tsv> <kernel_replay.tsv>" }
        val samples = readSamples(File(args[0]))
        require(samples.size == EXPECTED_SAMPLES) {
            "D40 expected $EXPECTED_SAMPLES samples, got ${samples.size}"
        }
        require(samples.map { it.sourceSessionId }.toSet().size == EXPECTED_SESSIONS) {
            "D40 source-session census drift"
        }
        val uniqueFrames = uniqueFrames(samples)
        require(uniqueFrames.size == EXPECTED_UNIQUE_FRAMES) {
            "D40 unique-frame census drift: ${uniqueFrames.size}"
        }
        require(uniqueFrames.sumOf { it.detections.size } == EXPECTED_DETECTIONS) {
            "D40 detection census drift"
        }
        val frameResults = replayContinuous(uniqueFrames)
        require(frameResults.size == uniqueFrames.size)
        val sampleResults = samples.map { sample ->
            summarizeSample(sample, frameResults)
        }
        writeResults(File(args[1]), sampleResults)
        println(
            "D40_KERNEL_REPLAY samples=${sampleResults.size} " +
                "unique_frames=${uniqueFrames.size} " +
                "detections=${uniqueFrames.sumOf { it.detections.size }} " +
                "forecast_frames=${frameResults.values.count { it.forecastAvailable }} " +
                "forecast_windows=${sampleResults.count { it.forecastFrames > 0 }} " +
                "baseline_windows=${sampleResults.count { it.baselineAnyTriggered }} " +
                "candidate_windows=${sampleResults.count { it.candidateAnyTriggered }}"
        )
    }

    private fun replayContinuous(
        frames: List<Frame>
    ): Map<FrameKey, FrameReplayResult> {
        val output = linkedMapOf<FrameKey, FrameReplayResult>()
        frames.groupBy { it.sourceSessionId }.toSortedMap().forEach {
                (sourceSessionId, sessionFrames) ->
            var baseline = AssistDecisionKernel()
            var candidate = AssistDecisionKernel()
            var producer = CausalTrackTristateGeometryProducer()
            var baselineGateway = PlannerGateway()
            var candidateGateway = PlannerGateway()
            var previousCapturedAtNs: Long? = null
            var segment = -1
            sessionFrames.sortedWith(
                compareBy<Frame>({ it.capturedAtNs }, { it.sourceSceneFrame })
            ).forEach { frame ->
                val gap = previousCapturedAtNs?.let {
                    frame.capturedAtNs - it
                }
                if (gap == null || gap !in 1..MAXIMUM_CONTINUOUS_GAP_NS) {
                    segment += 1
                    baseline = AssistDecisionKernel()
                    candidate = AssistDecisionKernel()
                    producer = CausalTrackTristateGeometryProducer()
                    baselineGateway = PlannerGateway()
                    candidateGateway = PlannerGateway()
                    val startMs = frame.capturedAtNs / 1_000_000L
                    baseline.startSession(startMs)
                    candidate.startSession(startMs)
                }
                previousCapturedAtNs = frame.capturedAtNs
                val stamp = FrameStamp(
                    frameId = frame.sourceSceneFrame.toLong(),
                    capturedAtNs = frame.capturedAtNs,
                    receivedAtNs = frame.capturedAtNs + 1_000_000L,
                    sourceId = "d40:$sourceSessionId:$segment",
                    coordinateFrame = "thor-magni:pupil-rgb",
                    clockDomain = FrameClockDomain.REPLAY_TIMELINE
                )
                val decisionAtNs = frame.capturedAtNs + 2_000_000L
                val nowMs = frame.capturedAtNs / 1_000_000L
                val baselineResult = baseline.processFrame(
                    detections = frame.detections,
                    frameSize = frame.frameSize,
                    profile = AlertProfile.STANDARD,
                    scenario = AssistScenario.GENERAL,
                    metrics = metrics(),
                    feedbackGateway = baselineGateway,
                    nowMs = nowMs,
                    sourceFrame = stamp,
                    decisionAtNs = decisionAtNs,
                    dualLoopMode = DualLoopRuntimeMode.OFF
                )
                val selected = baselineResult.evaluation.rawRisk.sourceDetection
                val evidence = producer.produce(
                    sourceFrame = stamp,
                    selectedTarget = selected,
                    decisionAtNs = decisionAtNs
                )
                val forecast = forecastDetections(
                    detections = frame.detections,
                    selected = selected,
                    slopePerS = evidence?.signedApproachRatePerS,
                    frameSize = frame.frameSize
                )
                val candidateResult = candidate.processFrame(
                    detections = forecast?.detections ?: frame.detections,
                    frameSize = frame.frameSize,
                    profile = AlertProfile.STANDARD,
                    scenario = AssistScenario.GENERAL,
                    metrics = metrics(),
                    feedbackGateway = candidateGateway,
                    nowMs = nowMs,
                    sourceFrame = stamp,
                    decisionAtNs = decisionAtNs,
                    dualLoopMode = DualLoopRuntimeMode.OFF
                )
                val key = FrameKey(
                    sourceSessionId,
                    frame.sourceSceneFrame
                )
                check(key !in output)
                output[key] = FrameReplayResult(
                    baselineTriggered =
                        baselineResult.feedbackDecision.triggered,
                    candidateTriggered =
                        candidateResult.feedbackDecision.triggered,
                    forecastAvailable = forecast != null,
                    slopePerS = forecast?.slopePerS,
                    scale = forecast?.scale,
                    segment = segment
                )
            }
        }
        return output
    }

    private fun forecastDetections(
        detections: List<Detection>,
        selected: Detection?,
        slopePerS: Float?,
        frameSize: FrameSize
    ): Forecast? {
        if (selected == null || slopePerS == null || !slopePerS.isFinite()) {
            return null
        }
        val selectedIndices = detections.withIndex()
            .filter { it.value == selected }
            .map { it.index }
        if (selectedIndices.size != 1) return null
        val scale = exp(slopePerS.toDouble() * FORECAST_HORIZON_S)
        if (!scale.isFinite() || scale <= 0.0) return null
        val box = selected.boundingBox
        val projectedWidth = box.width.toDouble() * scale
        val projectedHeight = box.height.toDouble() * scale
        if (
            !projectedWidth.isFinite() ||
            !projectedHeight.isFinite() ||
            projectedWidth <= 0.0 ||
            projectedHeight <= 0.0
        ) {
            return null
        }
        val projected = BoundingBox(
            left = (box.centerX - projectedWidth / 2.0).toFloat(),
            top = (box.bottom - projectedHeight).toFloat(),
            right = (box.centerX + projectedWidth / 2.0).toFloat(),
            bottom = box.bottom
        ).clamped(frameSize)
        if (
            !projected.left.isFinite() ||
            !projected.top.isFinite() ||
            !projected.right.isFinite() ||
            !projected.bottom.isFinite() ||
            projected.width <= 0f ||
            projected.height <= 0f
        ) {
            return null
        }
        val mutable = detections.toMutableList()
        mutable[selectedIndices.single()] = selected.copy(
            boundingBox = projected
        )
        return Forecast(
            detections = mutable.toList(),
            slopePerS = slopePerS,
            scale = scale
        )
    }

    private fun summarizeSample(
        sample: Sample,
        results: Map<FrameKey, FrameReplayResult>
    ): SampleReplayResult {
        require(sample.frames.size == EXPECTED_FRAMES_PER_SAMPLE)
        require(
            sample.frames.map { it.ordinal } ==
                (0 until EXPECTED_FRAMES_PER_SAMPLE).toList()
        )
        val frameResults = sample.frames.map { frame ->
            requireNotNull(
                results[
                    FrameKey(
                        sample.sourceSessionId,
                        frame.sourceSceneFrame
                    )
                ]
            ) {
                "D40 sample frame missing from continuous replay"
            }
        }
        return SampleReplayResult(
            sampleId = sample.sampleId,
            sourceSessionId = sample.sourceSessionId,
            fold = sample.fold,
            anchorSceneFrame = sample.frames.last().sourceSceneFrame,
            baselineTriggeredFrames = sample.frames.zip(frameResults)
                .filter { it.second.baselineTriggered }
                .map { it.first.ordinal },
            candidateTriggeredFrames = sample.frames.zip(frameResults)
                .filter { it.second.candidateTriggered }
                .map { it.first.ordinal },
            forecastFrames = frameResults.count { it.forecastAvailable },
            positiveSlopeFrames = frameResults.count {
                it.slopePerS?.let { slope -> slope > 0f } == true
            },
            negativeSlopeFrames = frameResults.count {
                it.slopePerS?.let { slope -> slope < 0f } == true
            },
            maximumAbsoluteSlope = frameResults
                .mapNotNull { it.slopePerS }
                .maxOfOrNull { kotlin.math.abs(it) },
            segmentCount = frameResults.map { it.segment }.toSet().size
        )
    }

    private fun uniqueFrames(samples: List<Sample>): List<Frame> {
        val unique = linkedMapOf<FrameKey, Frame>()
        samples.forEach { sample ->
            sample.frames.forEach { frame ->
                val key = FrameKey(
                    sample.sourceSessionId,
                    frame.sourceSceneFrame
                )
                val existing = unique[key]
                if (existing == null) {
                    unique[key] = frame
                } else {
                    require(
                        existing.sourceSessionId == frame.sourceSessionId &&
                            existing.sourceSceneFrame == frame.sourceSceneFrame &&
                            existing.capturedAtNs == frame.capturedAtNs &&
                            existing.frameSize == frame.frameSize &&
                            existing.detections == frame.detections
                    ) {
                        "D40 overlapping sample windows disagree on frame $key"
                    }
                }
            }
        }
        return unique.values.toList()
    }

    private fun readSamples(input: File): List<Sample> {
        val lines = input.readLines(Charsets.UTF_8)
        require(lines.isNotEmpty()) { "D40 detector input is empty" }
        val header = lines.first().split('\t')
        val index = header.withIndex().associate { it.value to it.index }
        fun value(parts: List<String>, name: String): String =
            parts[requireNotNull(index[name]) { "D40 missing column $name" }]

        val samples = linkedMapOf<String, MutableSample>()
        lines.drop(1).filter { it.isNotBlank() }.forEach { line ->
            val parts = line.split('\t')
            val sampleId = value(parts, "sample_id")
            val sample = samples.getOrPut(sampleId) {
                MutableSample(
                    sampleId = sampleId,
                    sourceSessionId = value(parts, "source_session_id"),
                    fold = value(parts, "fold").toInt()
                )
            }
            require(sample.sourceSessionId == value(parts, "source_session_id"))
            require(sample.fold == value(parts, "fold").toInt())
            val ordinal = value(parts, "frame_ordinal").toInt()
            val frame = sample.frames.getOrPut(ordinal) {
                MutableFrame(
                    ordinal = ordinal,
                    sourceSceneFrame = value(parts, "source_scene_frame").toInt(),
                    capturedAtNs = value(parts, "captured_at_ns").toLong(),
                    frameSize = FrameSize(
                        value(parts, "frame_width").toInt(),
                        value(parts, "frame_height").toInt()
                    )
                )
            }
            val detectionIndex = value(parts, "detection_index").toInt()
            if (detectionIndex >= 0) {
                require(detectionIndex == frame.detections.size)
                frame.detections += Detection(
                    classId = 0,
                    label = "person",
                    confidence = value(parts, "confidence").toFloat(),
                    boundingBox = BoundingBox(
                        value(parts, "left").toFloat(),
                        value(parts, "top").toFloat(),
                        value(parts, "right").toFloat(),
                        value(parts, "bottom").toFloat()
                    ),
                    frameSize = frame.frameSize
                )
            }
        }
        return samples.values.map { sample ->
            Sample(
                sampleId = sample.sampleId,
                sourceSessionId = sample.sourceSessionId,
                fold = sample.fold,
                frames = sample.frames.values.sortedBy { it.ordinal }.map {
                    Frame(
                        sourceSessionId = sample.sourceSessionId,
                        ordinal = it.ordinal,
                        sourceSceneFrame = it.sourceSceneFrame,
                        capturedAtNs = it.capturedAtNs,
                        frameSize = it.frameSize,
                        detections = it.detections.toList()
                    )
                }
            )
        }
    }

    private fun writeResults(
        output: File,
        results: List<SampleReplayResult>
    ) {
        output.parentFile.mkdirs()
        val header = listOf(
            "sample_id",
            "source_session_id",
            "fold",
            "anchor_scene_frame",
            "baseline_any_triggered",
            "candidate_any_triggered",
            "candidate_only_triggered_window",
            "baseline_only_triggered_window",
            "baseline_triggered_frames",
            "candidate_triggered_frames",
            "forecast_frames",
            "positive_slope_frames",
            "negative_slope_frames",
            "maximum_absolute_slope",
            "segment_count"
        )
        val text = buildString {
            appendLine(header.joinToString("\t"))
            results.forEach { row ->
                appendLine(
                    listOf(
                        row.sampleId,
                        row.sourceSessionId,
                        row.fold,
                        row.anchorSceneFrame,
                        row.baselineAnyTriggered,
                        row.candidateAnyTriggered,
                        row.candidateAnyTriggered &&
                            !row.baselineAnyTriggered,
                        row.baselineAnyTriggered &&
                            !row.candidateAnyTriggered,
                        row.baselineTriggeredFrames.joinToString(","),
                        row.candidateTriggeredFrames.joinToString(","),
                        row.forecastFrames,
                        row.positiveSlopeFrames,
                        row.negativeSlopeFrames,
                        row.maximumAbsoluteSlope ?: "",
                        row.segmentCount
                    ).joinToString("\t")
                )
            }
        }
        val partial = File(output.parentFile, output.name + ".partial")
        partial.writeText(text, Charsets.UTF_8)
        Files.move(
            partial.toPath(),
            output.toPath(),
            StandardCopyOption.REPLACE_EXISTING,
            StandardCopyOption.ATOMIC_MOVE
        )
    }

    private fun metrics() = DetectorMetrics(
        totalMs = 0L,
        preprocessMs = 0L,
        inferenceMs = 0L,
        postprocessMs = 0L,
        fps = 15f,
        modelStatus = "replay"
    )

    private class PlannerGateway : FeedbackGateway {
        override fun resetSession() = Unit

        override fun notify(
            risk: RiskResult,
            profile: AlertProfile,
            scenario: AssistScenario
        ): FeedbackDecision {
            val plan = FeedbackPlanner.planFor(
                risk,
                profile = profile,
                scenario = scenario
            )
            return if (plan == null) {
                FeedbackDecision(
                    plan = null,
                    triggered = false,
                    reason = FeedbackReason.NO_FEEDBACK_RISK
                )
            } else {
                FeedbackDecision(
                    plan = plan,
                    triggered = true,
                    reason = FeedbackReason.TRIGGERED
                )
            }
        }
    }

    private data class FrameKey(
        val sourceSessionId: String,
        val sourceSceneFrame: Int
    )

    private data class MutableSample(
        val sampleId: String,
        val sourceSessionId: String,
        val fold: Int,
        val frames: MutableMap<Int, MutableFrame> = linkedMapOf()
    )

    private data class MutableFrame(
        val ordinal: Int,
        val sourceSceneFrame: Int,
        val capturedAtNs: Long,
        val frameSize: FrameSize,
        val detections: MutableList<Detection> = mutableListOf()
    )

    private data class Sample(
        val sampleId: String,
        val sourceSessionId: String,
        val fold: Int,
        val frames: List<Frame>
    )

    private data class Frame(
        val sourceSessionId: String,
        val ordinal: Int,
        val sourceSceneFrame: Int,
        val capturedAtNs: Long,
        val frameSize: FrameSize,
        val detections: List<Detection>
    )

    private data class Forecast(
        val detections: List<Detection>,
        val slopePerS: Float,
        val scale: Double
    )

    private data class FrameReplayResult(
        val baselineTriggered: Boolean,
        val candidateTriggered: Boolean,
        val forecastAvailable: Boolean,
        val slopePerS: Float?,
        val scale: Double?,
        val segment: Int
    )

    private data class SampleReplayResult(
        val sampleId: String,
        val sourceSessionId: String,
        val fold: Int,
        val anchorSceneFrame: Int,
        val baselineTriggeredFrames: List<Int>,
        val candidateTriggeredFrames: List<Int>,
        val forecastFrames: Int,
        val positiveSlopeFrames: Int,
        val negativeSlopeFrames: Int,
        val maximumAbsoluteSlope: Float?,
        val segmentCount: Int
    ) {
        val baselineAnyTriggered: Boolean
            get() = baselineTriggeredFrames.isNotEmpty()
        val candidateAnyTriggered: Boolean
            get() = candidateTriggeredFrames.isNotEmpty()
    }
}
