package com.linnan.blindassist.benchmark

import com.linnan.blindassist.model.BoundingBox
import kotlin.math.max
import kotlin.math.min

data class TargetMatchCandidate(
    val detectionId: String,
    val label: String,
    val boundingBox: BoundingBox
)

data class TargetMatchResult(
    val status: String,
    val matchedDetectionId: String?,
    val matchedIou: Double?,
    val eligibleDetectionIds: List<String>
)

/** Benchmark-only target matcher. It never turns an arbitrary route-inside detection into an event. */
object UstrfTargetInstanceMatcher {
    const val CONTRACT_ID = "target_label_allowlist_max_iou_030_v1"
    const val MIN_IOU = 0.30

    fun match(
        targetBox: BoundingBox,
        detectorLabels: List<String>,
        targetLabelAllowlist: List<String>,
        candidates: List<TargetMatchCandidate>
    ): TargetMatchResult {
        require(targetLabelAllowlist.distinct().size == targetLabelAllowlist.size) { "target allowlist repeats a label" }
        val supported = targetLabelAllowlist.toSet().intersect(detectorLabels.toSet())
        if (supported.isEmpty()) return TargetMatchResult("unsupported_taxonomy", null, null, emptyList())
        val eligible = candidates.filter { it.label in supported }
        if (eligible.isEmpty()) return TargetMatchResult("unmatched", null, null, emptyList())
        val scored = eligible.map { it to iou(targetBox, it.boundingBox) }
        val best = scored.maxOf { it.second }
        if (best < MIN_IOU) return TargetMatchResult("unmatched", null, best, eligible.map { it.detectionId })
        val winners = scored.filter { kotlin.math.abs(it.second - best) <= 1e-9 }
        if (winners.size != 1) return TargetMatchResult("ambiguous", null, best, eligible.map { it.detectionId })
        return TargetMatchResult("matched", winners.single().first.detectionId, best, eligible.map { it.detectionId })
    }

    fun iou(a: BoundingBox, b: BoundingBox): Double {
        val intersection = max(0f, min(a.right, b.right) - max(a.left, b.left)) *
            max(0f, min(a.bottom, b.bottom) - max(a.top, b.top))
        val union = a.width * a.height + b.width * b.height - intersection
        return if (union <= 0f) 0.0 else (intersection / union).toDouble()
    }
}
