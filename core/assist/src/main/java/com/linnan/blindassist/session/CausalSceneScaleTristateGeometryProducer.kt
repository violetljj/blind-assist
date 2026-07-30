package com.linnan.blindassist.session

import com.linnan.blindassist.model.Detection
import com.linnan.blindassist.vision.FrameStamp
import kotlin.math.hypot
import kotlin.math.ln

/**
 * Minimal scene-level veto source.
 *
 * It asks only whether at least two uniquely associated detector boxes shrink together. It does
 * not select an obstacle, attribute motion, estimate TTC, or emit a positive alert signal.
 */
class CausalSceneScaleTristateGeometryProducer(
    private val rateThresholdPerS: Float = DEFAULT_RATE_THRESHOLD_PER_S,
    private val maximumGapNs: Long = DEFAULT_MAXIMUM_GAP_NS
) {
    private data class Track(
        val epoch: Long,
        var frame: FrameStamp,
        var detection: Detection
    )

    private data class PairCandidate(
        val negativeScore: Float,
        val trackEpoch: Long,
        val detectionIndex: Int,
        val trackIndex: Int
    )

    private val tracks = mutableListOf<Track>()
    private var nextEpoch = 0L
    private var lastFrame: FrameStamp? = null

    init {
        require(rateThresholdPerS.isFinite() && rateThresholdPerS < 0f)
        require(maximumGapNs > 0L)
    }

    fun reset() {
        tracks.clear()
        nextEpoch = 0L
        lastFrame = null
    }

    fun produce(
        sourceFrame: FrameStamp?,
        detections: List<Detection>,
        selectedTarget: Detection?,
        decisionAtNs: Long
    ): DualLoopGeometryEvidence? {
        if (sourceFrame == null) {
            reset()
            return null
        }
        if (lastFrame?.let { !continuesFrame(it, sourceFrame) } == true) {
            tracks.clear()
            nextEpoch += 1L
        }
        val previousFrame = lastFrame
        val rates = associateAndUpdate(sourceFrame, detections)
        lastFrame = sourceFrame
        if (selectedTarget == null) return null

        val medianRate = rates.takeIf { it.size >= MINIMUM_MATCHES }
            ?.sorted()
            ?.let { sorted ->
                val middle = sorted.size / 2
                if (sorted.size % 2 == 1) sorted[middle]
                else (sorted[middle - 1] + sorted[middle]) / 2f
            }
        val contradicts = medianRate != null && medianRate <= rateThresholdPerS
        val reason = when {
            rates.size < MINIMUM_MATCHES -> "INSUFFICIENT_SCENE_MATCHES"
            !contradicts -> "SCENE_NOT_COLLECTIVELY_RECEDING"
            else -> null
        }
        return DualLoopGeometryEvidence(
            sourceContractId = DualLoopShadowAdmitter.CONTRACT_ID,
            sourceId = SOURCE_ID,
            previousFrame = previousFrame ?: sourceFrame,
            currentFrame = sourceFrame,
            availableAtNs = decisionAtNs,
            validUntilNs = saturatingAdd(decisionAtNs, EVIDENCE_TTL_NS),
            availabilityClockDomain = sourceFrame.clockDomain,
            trackEpoch = "${sourceFrame.sourceId}:scene:$nextEpoch",
            targetClassId = selectedTarget.classId,
            targetLabel = selectedTarget.label,
            targetBoundingBox = selectedTarget.boundingBox,
            targetFrameSize = selectedTarget.frameSize,
            targetSource = selectedTarget.source,
            correctionDecision = if (contradicts) {
                DualLoopCorrectionDecision.CONTRADICT_APPROACH
            } else {
                DualLoopCorrectionDecision.ABSTAIN
            },
            signedApproachRatePerS = medianRate,
            quality = if (contradicts) {
                (rates.size.toFloat() / QUALITY_FULL_MATCH_COUNT).coerceAtMost(1f)
            } else {
                null
            },
            sourceAbstentionReason = reason
        )
    }

    private fun associateAndUpdate(
        frame: FrameStamp,
        detections: List<Detection>
    ): List<Float> {
        tracks.removeAll { track ->
            val gap = frame.capturedAtNs - track.frame.capturedAtNs
            gap !in 1..maximumGapNs ||
                track.frame.sourceId != frame.sourceId ||
                track.frame.coordinateFrame != frame.coordinateFrame ||
                track.frame.clockDomain != frame.clockDomain
        }
        val pairs = buildList {
            tracks.forEachIndexed { trackIndex, track ->
                detections.forEachIndexed { detectionIndex, detection ->
                    associationScore(track.detection, detection)?.let { score ->
                        add(
                            PairCandidate(
                                negativeScore = -score,
                                trackEpoch = track.epoch,
                                detectionIndex = detectionIndex,
                                trackIndex = trackIndex
                            )
                        )
                    }
                }
            }
        }.sortedWith(
            compareBy<PairCandidate>(
                { it.negativeScore },
                { it.trackEpoch },
                { it.detectionIndex },
                { it.trackIndex }
            )
        )
        val usedTracks = mutableSetOf<Int>()
        val usedDetections = mutableSetOf<Int>()
        val matches = mutableMapOf<Int, Track>()
        val rates = mutableListOf<Float>()
        pairs.forEach { pair ->
            if (pair.trackIndex in usedTracks || pair.detectionIndex in usedDetections) {
                return@forEach
            }
            usedTracks += pair.trackIndex
            usedDetections += pair.detectionIndex
            val track = tracks[pair.trackIndex]
            val detection = detections[pair.detectionIndex]
            val previousHeight = track.detection.boundingBox.height
            val currentHeight = detection.boundingBox.height
            val gapS = (frame.capturedAtNs - track.frame.capturedAtNs) / NANOS_PER_SECOND
            if (previousHeight.isFinite() && previousHeight > 0f &&
                currentHeight.isFinite() && currentHeight > 0f &&
                gapS > 0.0
            ) {
                rates += (ln(currentHeight / previousHeight) / gapS).toFloat()
            }
            matches[pair.detectionIndex] = track
        }
        detections.forEachIndexed { index, detection ->
            val track = matches[index] ?: Track(nextEpoch++, frame, detection).also(tracks::add)
            track.frame = frame
            track.detection = detection
        }
        return rates
    }

    private fun associationScore(previous: Detection, current: Detection): Float? {
        if (previous.classId != current.classId ||
            previous.label != current.label ||
            previous.source != current.source ||
            previous.frameSize != current.frameSize
        ) {
            return null
        }
        val first = previous.boundingBox
        val second = current.boundingBox
        val intersection = maxOf(0f, minOf(first.right, second.right) - maxOf(first.left, second.left)) *
            maxOf(0f, minOf(first.bottom, second.bottom) - maxOf(first.top, second.top))
        val union = first.width * first.height + second.width * second.height - intersection
        val iou = if (union > 0f) intersection / union else 0f
        val dx = (second.centerX - first.centerX) / previous.frameSize.width.coerceAtLeast(1)
        val dy = (second.centerY - first.centerY) / previous.frameSize.height.coerceAtLeast(1)
        val distance = hypot(dx, dy)
        return if (iou >= MINIMUM_IOU || distance <= MAXIMUM_CENTER_DISTANCE) {
            2f * iou + maxOf(0f, 1f - distance)
        } else {
            null
        }
    }

    private fun continuesFrame(previous: FrameStamp, current: FrameStamp): Boolean {
        val gap = current.capturedAtNs - previous.capturedAtNs
        return previous.sourceId == current.sourceId &&
            previous.coordinateFrame == current.coordinateFrame &&
            previous.clockDomain == current.clockDomain &&
            current.frameId > previous.frameId &&
            gap in 1..maximumGapNs
    }

    private fun saturatingAdd(value: Long, increment: Long): Long =
        if (value > Long.MAX_VALUE - increment) Long.MAX_VALUE else value + increment

    companion object {
        const val SOURCE_ID = "CAUSAL_SCENE_SCALE_VETO_R1"
        const val DEFAULT_RATE_THRESHOLD_PER_S = -0.05f
        const val DEFAULT_MAXIMUM_GAP_NS = 500_000_000L
        private const val MINIMUM_MATCHES = 2
        private const val QUALITY_FULL_MATCH_COUNT = 4f
        private const val MINIMUM_IOU = 0.25f
        private const val MAXIMUM_CENTER_DISTANCE = 0.12f
        private const val EVIDENCE_TTL_NS = 250_000_000L
        private const val NANOS_PER_SECOND = 1_000_000_000.0
    }
}
