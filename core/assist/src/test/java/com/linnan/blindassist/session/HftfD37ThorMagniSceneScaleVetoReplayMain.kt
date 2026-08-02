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

/**
 * Source-only production-kernel replay for the frozen D37 scene-scale veto.
 *
 * Event truth is deliberately unavailable here and is joined only by the
 * Python evaluator after the atomic replay artifact exists.
 */
object HftfD37ThorMagniSceneScaleVetoReplayMain {
    private const val EXPECTED_SAMPLES = 530
    private const val EXPECTED_SESSIONS = 19
    private const val EXPECTED_FRAMES = 7

    @JvmStatic
    fun main(args: Array<String>) {
        require(args.size == 2) { "usage: <detections.tsv> <kernel_replay.tsv>" }
        val samples = readSamples(File(args[0]))
        require(samples.size == EXPECTED_SAMPLES) {
            "D37 expected $EXPECTED_SAMPLES samples, got ${samples.size}"
        }
        require(samples.map { it.sourceSessionId }.toSet().size == EXPECTED_SESSIONS) {
            "D37 source-session census drift"
        }
        val results = samples.map(::replay)
        require(results.sumOf { it.rawRiskMismatches } == 0) {
            "D37 baseline/candidate raw-risk drift"
        }
        require(results.sumOf { it.stableRiskMismatches } == 0) {
            "D37 baseline/candidate stable-risk drift"
        }
        require(results.sumOf { it.nonSceneSourceObservations } == 0) {
            "D37 observed non-scene dual-loop source"
        }
        writeResults(File(args[1]), results)
        println(
            "D37_KERNEL_REPLAY samples=${results.size} " +
                "baseline_windows=${results.count { it.baselineAnyTriggered }} " +
                "candidate_windows=${results.count { it.candidateAnyTriggered }} " +
                "admitted_contradict_frames=" +
                results.sumOf { it.admittedContradictFrames } +
                " suppressions=${results.sumOf { it.suppressedFrames }}"
        )
    }

    private fun replay(sample: Sample): ReplayResult {
        require(sample.frames.size == EXPECTED_FRAMES)
        require(sample.frames.map { it.ordinal } == (0 until EXPECTED_FRAMES).toList())
        val baseline = AssistDecisionKernel()
        val candidate = AssistDecisionKernel()
        val baselineGateway = PlannerGateway()
        val candidateGateway = PlannerGateway()
        val startMs = sample.frames.first().capturedAtNs / 1_000_000L
        baseline.startSession(startMs)
        candidate.startSession(startMs)
        val baselineTriggered = mutableListOf<Int>()
        val candidateTriggered = mutableListOf<Int>()
        var rawRiskMismatches = 0
        var stableRiskMismatches = 0
        var candidateOnlyTriggeredFrames = 0
        var suppressedFrames = 0
        var admittedContradictFrames = 0
        var admittedConfirmFrames = 0
        var sceneContradictObservations = 0
        var sceneAbstainObservations = 0
        var evidenceAbsentFrames = 0
        var nonSceneSourceObservations = 0
        sample.frames.forEach { frame ->
            val stamp = FrameStamp(
                frameId = frame.sourceSceneFrame.toLong(),
                capturedAtNs = frame.capturedAtNs,
                receivedAtNs = frame.capturedAtNs + 1_000_000L,
                sourceId = "d37:${sample.sampleId}",
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
            val candidateResult = candidate.processFrame(
                detections = frame.detections,
                frameSize = frame.frameSize,
                profile = AlertProfile.STANDARD,
                scenario = AssistScenario.GENERAL,
                metrics = metrics(),
                feedbackGateway = candidateGateway,
                nowMs = nowMs,
                sourceFrame = stamp,
                decisionAtNs = decisionAtNs,
                dualLoopMode = DualLoopRuntimeMode.ACTIVE_CONTRADICT_ONLY
            )
            if (baselineResult.evaluation.rawRisk != candidateResult.evaluation.rawRisk) {
                rawRiskMismatches += 1
            }
            if (baselineResult.evaluation.stableRisk != candidateResult.evaluation.stableRisk) {
                stableRiskMismatches += 1
            }
            if (baselineResult.feedbackDecision.triggered) {
                baselineTriggered += frame.ordinal
            }
            if (candidateResult.feedbackDecision.triggered) {
                candidateTriggered += frame.ordinal
            }
            if (
                candidateResult.feedbackDecision.triggered &&
                !baselineResult.feedbackDecision.triggered
            ) {
                candidateOnlyTriggeredFrames += 1
            }
            if (
                candidateResult.feedbackDecision.reason ==
                FeedbackReason.DUAL_LOOP_CONTRADICTED
            ) {
                suppressedFrames += 1
            }
            val shadow = candidateResult.evaluation.dualLoopShadow
            if (
                shadow.sourceId != null &&
                shadow.sourceId != CausalSceneScaleTristateGeometryProducer.SOURCE_ID
            ) {
                nonSceneSourceObservations += 1
            }
            when {
                shadow.disposition == DualLoopShadowDisposition.EVIDENCE_ABSENT ->
                    evidenceAbsentFrames += 1
                shadow.sourceId ==
                    CausalSceneScaleTristateGeometryProducer.SOURCE_ID &&
                    shadow.correctionDecision ==
                    DualLoopCorrectionDecision.CONTRADICT_APPROACH ->
                    sceneContradictObservations += 1
                shadow.sourceId ==
                    CausalSceneScaleTristateGeometryProducer.SOURCE_ID ->
                    sceneAbstainObservations += 1
            }
            if (shadow.admitted) {
                when (shadow.correctionDecision) {
                    DualLoopCorrectionDecision.CONTRADICT_APPROACH ->
                        admittedContradictFrames += 1
                    DualLoopCorrectionDecision.CONFIRM_APPROACH ->
                        admittedConfirmFrames += 1
                    else -> Unit
                }
            }
        }
        return ReplayResult(
            sampleId = sample.sampleId,
            sourceSessionId = sample.sourceSessionId,
            fold = sample.fold,
            anchorSceneFrame = sample.frames.last().sourceSceneFrame,
            baselineTriggeredFrames = baselineTriggered,
            candidateTriggeredFrames = candidateTriggered,
            rawRiskMismatches = rawRiskMismatches,
            stableRiskMismatches = stableRiskMismatches,
            candidateOnlyTriggeredFrames = candidateOnlyTriggeredFrames,
            suppressedFrames = suppressedFrames,
            admittedContradictFrames = admittedContradictFrames,
            admittedConfirmFrames = admittedConfirmFrames,
            sceneContradictObservations = sceneContradictObservations,
            sceneAbstainObservations = sceneAbstainObservations,
            evidenceAbsentFrames = evidenceAbsentFrames,
            nonSceneSourceObservations = nonSceneSourceObservations
        )
    }

    private fun readSamples(input: File): List<Sample> {
        val lines = input.readLines(Charsets.UTF_8)
        require(lines.isNotEmpty()) { "D37 detector input is empty" }
        val header = lines.first().split('\t')
        val index = header.withIndex().associate { it.value to it.index }
        fun value(parts: List<String>, name: String): String =
            parts[requireNotNull(index[name]) { "D37 missing column $name" }]

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

    private fun writeResults(output: File, results: List<ReplayResult>) {
        output.parentFile.mkdirs()
        val header = listOf(
            "sample_id",
            "source_session_id",
            "fold",
            "anchor_scene_frame",
            "baseline_any_triggered",
            "candidate_any_triggered",
            "candidate_only_triggered_window",
            "baseline_triggered_frames",
            "candidate_triggered_frames",
            "candidate_only_triggered_frames",
            "suppressed_frames",
            "admitted_contradict_frames",
            "admitted_confirm_frames",
            "scene_contradict_observations",
            "scene_abstain_observations",
            "evidence_absent_frames",
            "raw_risk_mismatches",
            "stable_risk_mismatches",
            "non_scene_source_observations"
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
                        row.candidateAnyTriggered && !row.baselineAnyTriggered,
                        row.baselineTriggeredFrames.joinToString(","),
                        row.candidateTriggeredFrames.joinToString(","),
                        row.candidateOnlyTriggeredFrames,
                        row.suppressedFrames,
                        row.admittedContradictFrames,
                        row.admittedConfirmFrames,
                        row.sceneContradictObservations,
                        row.sceneAbstainObservations,
                        row.evidenceAbsentFrames,
                        row.rawRiskMismatches,
                        row.stableRiskMismatches,
                        row.nonSceneSourceObservations
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
        val ordinal: Int,
        val sourceSceneFrame: Int,
        val capturedAtNs: Long,
        val frameSize: FrameSize,
        val detections: List<Detection>
    )

    private data class ReplayResult(
        val sampleId: String,
        val sourceSessionId: String,
        val fold: Int,
        val anchorSceneFrame: Int,
        val baselineTriggeredFrames: List<Int>,
        val candidateTriggeredFrames: List<Int>,
        val rawRiskMismatches: Int,
        val stableRiskMismatches: Int,
        val candidateOnlyTriggeredFrames: Int,
        val suppressedFrames: Int,
        val admittedContradictFrames: Int,
        val admittedConfirmFrames: Int,
        val sceneContradictObservations: Int,
        val sceneAbstainObservations: Int,
        val evidenceAbsentFrames: Int,
        val nonSceneSourceObservations: Int
    ) {
        val baselineAnyTriggered: Boolean get() = baselineTriggeredFrames.isNotEmpty()
        val candidateAnyTriggered: Boolean get() = candidateTriggeredFrames.isNotEmpty()
    }
}
