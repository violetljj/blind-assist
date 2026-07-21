package com.linnan.blindassist.benchmark

import com.linnan.blindassist.model.BoundingBox
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min

data class ContinuousTargetDetection(val detectionId: String, val label: String, val box: BoundingBox)
data class ContinuousTargetFrame(val timestampMs: Long, val detections: List<ContinuousTargetDetection>)
data class ContinuousTargetAnchor(val timestampMs: Long, val visibility: String, val box: BoundingBox?)
data class ContinuousAssociationFrame(
    val timestampMs: Long,
    val matchedDetectionId: String?,
    val ambiguous: Boolean,
    val eligibleDetectionCount: Int,
)
data class ContinuousAssociationResult(
    val frames: List<ContinuousAssociationFrame>,
    val primaryAnchorMatched: Boolean,
    val visibleAnchorCount: Int,
    val visibleAnchorReacquiredCount: Int,
    val identitySwitchCount: Int,
)

/** Seed-anchored benchmark association. It is diagnostic trace, never continuous event truth. */
object UstrfContinuousTargetAssociation {
    const val CONTRACT_ID = "frozen_anchor_bidirectional_diagnostic_v1"

    fun associate(
        frames: List<ContinuousTargetFrame>,
        anchors: List<ContinuousTargetAnchor>,
        primaryAnchorTimestampMs: Long,
        detectorLabels: List<String>,
        allowlist: List<String>,
        minimumIou: Double,
        maximumCenterDistanceRatio: Double,
        minimumAreaRatio: Double,
        maximumAreaRatio: Double,
        ambiguityMargin: Double,
        clearAfterMisses: Int,
        frameWidth: Int,
        frameHeight: Int,
    ): ContinuousAssociationResult {
        require(frames.zipWithNext().all { it.first.timestampMs < it.second.timestampMs })
        require(clearAfterMisses > 0 && frameWidth > 0 && frameHeight > 0)
        val supported = allowlist.toSet().intersect(detectorLabels.toSet())
        val visibleAnchors = anchors.filter { it.visibility == "visible" && it.box != null }
        val primary = visibleAnchors.singleOrNull { it.timestampMs == primaryAnchorTimestampMs }
            ?: error("primary visible anchor missing")
        val primaryFrameIndex = frames.indices.minByOrNull { kotlin.math.abs(frames[it].timestampMs - primaryAnchorTimestampMs) }
            ?: error("continuous frames missing")
        val primaryFrame = frames[primaryFrameIndex]
        val anchorCandidates = primaryFrame.detections.filter { it.label in supported }
        val anchorMatch = UstrfTargetInstanceMatcher.match(
            checkNotNull(primary.box), detectorLabels, allowlist,
            anchorCandidates.map { TargetMatchCandidate(it.detectionId, it.label, it.box) },
        )
        val assignments = arrayOfNulls<ContinuousTargetDetection>(frames.size)
        val ambiguous = BooleanArray(frames.size)
        if (anchorMatch.status == "matched") {
            assignments[primaryFrameIndex] = anchorCandidates.single { it.detectionId == anchorMatch.matchedDetectionId }
            extend(frames, assignments, ambiguous, primaryFrameIndex, 1, supported, minimumIou,
                maximumCenterDistanceRatio, minimumAreaRatio, maximumAreaRatio, ambiguityMargin,
                clearAfterMisses, frameWidth, frameHeight)
            extend(frames, assignments, ambiguous, primaryFrameIndex, -1, supported, minimumIou,
                maximumCenterDistanceRatio, minimumAreaRatio, maximumAreaRatio, ambiguityMargin,
                clearAfterMisses, frameWidth, frameHeight)
        }
        var reacquired = 0
        var switches = 0
        visibleAnchors.forEach { anchor ->
            val frameIndex = frames.indices.minByOrNull { kotlin.math.abs(frames[it].timestampMs - anchor.timestampMs) } ?: return@forEach
            val assigned = assignments[frameIndex]
            if (assigned != null && UstrfTargetInstanceMatcher.iou(checkNotNull(anchor.box), assigned.box) >= UstrfTargetInstanceMatcher.MIN_IOU) {
                reacquired++
            } else if (assigned != null) {
                val otherMatches = frames[frameIndex].detections.any {
                    it.label in supported && UstrfTargetInstanceMatcher.iou(checkNotNull(anchor.box), it.box) >= UstrfTargetInstanceMatcher.MIN_IOU
                }
                if (otherMatches) switches++
            }
        }
        return ContinuousAssociationResult(
            frames.mapIndexed { index, frame -> ContinuousAssociationFrame(
                frame.timestampMs, assignments[index]?.detectionId, ambiguous[index],
                frame.detections.count { it.label in supported },
            ) },
            anchorMatch.status == "matched", visibleAnchors.size, reacquired, switches,
        )
    }

    private fun extend(
        frames: List<ContinuousTargetFrame>,
        assignments: Array<ContinuousTargetDetection?>,
        ambiguousFrames: BooleanArray,
        startIndex: Int,
        direction: Int,
        supportedLabels: Set<String>,
        minimumIou: Double,
        maximumCenterDistanceRatio: Double,
        minimumAreaRatio: Double,
        maximumAreaRatio: Double,
        ambiguityMargin: Double,
        clearAfterMisses: Int,
        frameWidth: Int,
        frameHeight: Int,
    ) {
        var previous = checkNotNull(assignments[startIndex])
        var misses = 0
        var index = startIndex + direction
        while (index in frames.indices) {
            val scored = frames[index].detections.asSequence()
                .filter { it.label in supportedLabels }
                .mapNotNull { candidate ->
                    val areaRatio = candidate.box.area() / max(previous.box.area(), 1f)
                    val iou = UstrfTargetInstanceMatcher.iou(previous.box, candidate.box)
                    val centerDistance = centerDistanceRatio(previous.box, candidate.box, frameWidth, frameHeight)
                    if (areaRatio !in minimumAreaRatio..maximumAreaRatio ||
                        (iou < minimumIou && centerDistance > maximumCenterDistanceRatio)) null
                    else candidate to (iou + (1.0 - min(centerDistance / maximumCenterDistanceRatio, 1.0)))
                }.sortedByDescending { it.second }.toList()
            val best = scored.firstOrNull()
            val isAmbiguous = best != null && scored.getOrNull(1)?.let { best.second - it.second <= ambiguityMargin } == true
            if (best == null || isAmbiguous) {
                ambiguousFrames[index] = isAmbiguous
                misses++
                if (misses >= clearAfterMisses) break
            } else {
                assignments[index] = best.first
                previous = best.first
                misses = 0
            }
            index += direction
        }
    }

    private fun BoundingBox.area() = max(0f, width) * max(0f, height)
    private fun centerDistanceRatio(a: BoundingBox, b: BoundingBox, width: Int, height: Int): Double {
        val ax = (a.left + a.right) / 2.0
        val ay = (a.top + a.bottom) / 2.0
        val bx = (b.left + b.right) / 2.0
        val by = (b.top + b.bottom) / 2.0
        return hypot(ax - bx, ay - by) / hypot(width.toDouble(), height.toDouble())
    }
}
