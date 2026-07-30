package com.linnan.blindassist.session

import com.linnan.blindassist.model.BoundingBox
import com.linnan.blindassist.model.Detection
import com.linnan.blindassist.model.FrameSize
import com.linnan.blindassist.vision.FrameStamp
import java.util.ArrayDeque
import kotlin.math.ln
import kotlin.math.sqrt

/**
 * Minimal causal second-loop source.
 *
 * It estimates only whether the currently selected target's box height grows
 * or shrinks consistently. It does not attribute motion, estimate metric TTC,
 * consume depth/pose/IMU, or mutate the production decision.
 */
class CausalTrackTristateGeometryProducer(
    private val historyFrames: Int = DEFAULT_HISTORY_FRAMES,
    private val slopeThresholdPerS: Float = DEFAULT_SLOPE_THRESHOLD_PER_S,
    private val maximumAdjacentGapNs: Long = DEFAULT_MAXIMUM_ADJACENT_GAP_NS
) {
    private data class Sample(
        val frame: FrameStamp,
        val detection: Detection,
        val logHeight: Double
    )

    private val history = ArrayDeque<Sample>()
    private var epochCounter = 0L

    init {
        require(historyFrames >= 2)
        require(slopeThresholdPerS.isFinite() && slopeThresholdPerS > 0f)
        require(maximumAdjacentGapNs > 0L)
    }

    fun reset() {
        history.clear()
        epochCounter = 0L
    }

    fun produce(
        sourceFrame: FrameStamp?,
        selectedTarget: Detection?,
        decisionAtNs: Long
    ): DualLoopGeometryEvidence? {
        if (sourceFrame == null || selectedTarget == null) {
            resetTrack()
            return null
        }
        val height = selectedTarget.boundingBox.height
        if (!height.isFinite() || height <= 0f) {
            resetTrack()
            return null
        }

        val previous = history.lastOrNull()
        if (previous != null && !continues(previous, sourceFrame, selectedTarget)) {
            resetTrack()
        }
        history.addLast(Sample(sourceFrame, selectedTarget, ln(height.toDouble())))
        while (history.size > historyFrames) history.removeFirst()

        val priorFrame = history.elementAtOrNull(history.size - 2)?.frame ?: sourceFrame
        val (decision, slope, reason) = decide()
        return DualLoopGeometryEvidence(
            sourceContractId = DualLoopShadowAdmitter.CONTRACT_ID,
            sourceId = SOURCE_ID,
            previousFrame = priorFrame,
            currentFrame = sourceFrame,
            availableAtNs = decisionAtNs,
            validUntilNs = saturatingAdd(decisionAtNs, EVIDENCE_TTL_NS),
            availabilityClockDomain = sourceFrame.clockDomain,
            trackEpoch = "${sourceFrame.sourceId}:$epochCounter",
            targetClassId = selectedTarget.classId,
            targetLabel = selectedTarget.label,
            targetBoundingBox = selectedTarget.boundingBox,
            targetFrameSize = selectedTarget.frameSize,
            targetSource = selectedTarget.source,
            correctionDecision = decision,
            signedApproachRatePerS = slope,
            quality = if (decision == DualLoopCorrectionDecision.ABSTAIN) null else 1f,
            sourceAbstentionReason =
                if (decision == DualLoopCorrectionDecision.ABSTAIN) reason else null
        )
    }

    private fun decide(): Triple<DualLoopCorrectionDecision, Float?, String> {
        if (history.size != historyFrames) {
            return Triple(
                DualLoopCorrectionDecision.ABSTAIN,
                null,
                "INSUFFICIENT_CONTIGUOUS_HISTORY"
            )
        }
        val samples = history.toList()
        val timeOriginNs = samples.first().frame.capturedAtNs
        val times = samples.map {
            (it.frame.capturedAtNs - timeOriginNs).toDouble() / NANOS_PER_SECOND
        }
        val values = samples.map { it.logHeight }
        val meanTime = times.average()
        val meanValue = values.average()
        val denominator = times.sumOf { (it - meanTime) * (it - meanTime) }
        if (denominator <= 0.0) {
            return Triple(DualLoopCorrectionDecision.ABSTAIN, null, "DEGENERATE_TIMESTAMP")
        }
        val slope = (
            times.zip(values).sumOf { (time, value) ->
                (time - meanTime) * (value - meanValue)
            } / denominator
            ).toFloat()
        val deltas = values.zipWithNext { left, right -> right - left }
        return when {
            slope >= slopeThresholdPerS && deltas.all { it > 0.0 } ->
                Triple(
                    DualLoopCorrectionDecision.CONFIRM_APPROACH,
                    slope,
                    "SEVEN_FRAME_UNANIMOUS_GROWTH"
                )
            slope <= -slopeThresholdPerS && deltas.all { it < 0.0 } ->
                Triple(
                    DualLoopCorrectionDecision.CONTRADICT_APPROACH,
                    slope,
                    "SEVEN_FRAME_UNANIMOUS_SHRINKAGE"
                )
            else ->
                Triple(DualLoopCorrectionDecision.ABSTAIN, slope, "TREND_NOT_SELECTIVE")
        }
    }

    private fun continues(
        previous: Sample,
        currentFrame: FrameStamp,
        current: Detection
    ): Boolean {
        val previousFrame = previous.frame
        val gap = currentFrame.capturedAtNs - previousFrame.capturedAtNs
        return currentFrame.sourceId == previousFrame.sourceId &&
            currentFrame.coordinateFrame == previousFrame.coordinateFrame &&
            currentFrame.clockDomain == previousFrame.clockDomain &&
            currentFrame.frameId > previousFrame.frameId &&
            gap in 1..maximumAdjacentGapNs &&
            current.classId == previous.detection.classId &&
            current.label == previous.detection.label &&
            current.source == previous.detection.source &&
            current.frameSize == previous.detection.frameSize &&
            sameTarget(previous.detection.boundingBox, current.boundingBox, current.frameSize)
    }

    private fun sameTarget(previous: BoundingBox, current: BoundingBox, frameSize: FrameSize): Boolean {
        val intersectionLeft = maxOf(previous.left, current.left)
        val intersectionTop = maxOf(previous.top, current.top)
        val intersectionRight = minOf(previous.right, current.right)
        val intersectionBottom = minOf(previous.bottom, current.bottom)
        val intersection = maxOf(0f, intersectionRight - intersectionLeft) *
            maxOf(0f, intersectionBottom - intersectionTop)
        val union = previous.width * previous.height +
            current.width * current.height - intersection
        val iou = if (union > 0f) intersection / union else 0f
        val dx = (previous.centerX - current.centerX) / frameSize.width.coerceAtLeast(1)
        val dy = (previous.centerY - current.centerY) / frameSize.height.coerceAtLeast(1)
        val normalizedCenterDistance = sqrt(dx * dx + dy * dy)
        return iou >= MINIMUM_IOU || normalizedCenterDistance <= MAXIMUM_CENTER_DISTANCE
    }

    private fun resetTrack() {
        history.clear()
        epochCounter += 1L
    }

    private fun saturatingAdd(value: Long, increment: Long): Long =
        if (value > Long.MAX_VALUE - increment) Long.MAX_VALUE else value + increment

    companion object {
        const val SOURCE_ID = "CAUSAL_TRACK_TRISTATE_R0"
        const val DEFAULT_HISTORY_FRAMES = 7
        const val DEFAULT_SLOPE_THRESHOLD_PER_S = 0.2f
        const val DEFAULT_MAXIMUM_ADJACENT_GAP_NS = 500_000_000L
        private const val EVIDENCE_TTL_NS = 250_000_000L
        private const val NANOS_PER_SECOND = 1_000_000_000.0
        private const val MINIMUM_IOU = 0.25f
        private const val MAXIMUM_CENTER_DISTANCE = 0.12f
    }
}
