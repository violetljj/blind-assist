package com.linnan.blindassist.risk

import com.linnan.blindassist.model.BoundingBox
import com.linnan.blindassist.model.DetectionSource
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min

data class TemporalRiskTrackerConfig(
    val maxFrames: Int = TemporalRiskTracker.DEFAULT_MAX_FRAMES,
    val maxWindowMs: Long = TemporalRiskTracker.DEFAULT_MAX_WINDOW_MS,
    val minApproachFrames: Int = TemporalRiskTracker.DEFAULT_MIN_APPROACH_FRAMES,
    val minIouForSameTarget: Float = TemporalRiskTracker.DEFAULT_MIN_IOU_FOR_SAME_TARGET,
    val maxCenterDeltaForSameTarget: Float = TemporalRiskTracker.DEFAULT_MAX_CENTER_DELTA_FOR_SAME_TARGET,
    val minBottomDeltaForApproach: Float = TemporalRiskTracker.DEFAULT_MIN_BOTTOM_DELTA_FOR_APPROACH,
    val minAreaGrowthForApproach: Float = TemporalRiskTracker.DEFAULT_MIN_AREA_GROWTH_FOR_APPROACH,
    val minDepthScoreDeltaForApproach: Float = TemporalRiskTracker.DEFAULT_MIN_DEPTH_SCORE_DELTA_FOR_APPROACH,
    val approachScoreBoost: Float = TemporalRiskTracker.DEFAULT_APPROACH_SCORE_BOOST,
    val minStableSegmentationFrames: Int = TemporalRiskTracker.DEFAULT_MIN_STABLE_SEGMENTATION_FRAMES,
    val minStableSegmentationBottomRatio: Float = TemporalRiskTracker.DEFAULT_MIN_STABLE_SEGMENTATION_BOTTOM_RATIO
) {
    init {
        require(maxFrames >= minApproachFrames) { "maxFrames must be >= minApproachFrames" }
        require(maxWindowMs > 0L) { "maxWindowMs must be positive" }
        require(minApproachFrames >= 2) { "minApproachFrames must be at least 2" }
        require(minStableSegmentationFrames >= 2) { "minStableSegmentationFrames must be at least 2" }
        require(minStableSegmentationBottomRatio in 0f..1f) {
            "minStableSegmentationBottomRatio must be in [0, 1]"
        }
    }
}

class TemporalRiskTracker(
    private val config: TemporalRiskTrackerConfig = TemporalRiskTrackerConfig()
) {
    private val observations = ArrayDeque<TargetObservation>()
    private val fusionPolicy = ConservativeRiskFusionPolicy(
        ConservativeRiskFusionConfig(motionMaxPromotionSteps = 1)
    )

    fun update(raw: RiskResult, nowMs: Long): RiskResult {
        val detection = raw.sourceDetection ?: run {
            reset()
            return raw.copy(approachTrend = ApproachTrend.UNKNOWN)
        }
        val current = TargetObservation.from(raw, detection.boundingBox, nowMs)
        trim(nowMs)

        val latest = observations.lastOrNull()
        if (latest != null && !latest.matches(current, config)) {
            observations.clear()
        }

        observations.addLast(current)
        while (observations.size > config.maxFrames) {
            observations.removeFirst()
        }

        val trend = trendFor(observations.toList())
        return applyTrend(raw, trend, observations.size)
    }

    fun reset() {
        observations.clear()
    }

    private fun trim(nowMs: Long) {
        while (observations.isNotEmpty() && nowMs - observations.first().observedAtMs > config.maxWindowMs) {
            observations.removeFirst()
        }
    }

    private fun trendFor(track: List<TargetObservation>): ApproachTrend {
        if (track.size < config.minApproachFrames) {
            return ApproachTrend.UNKNOWN
        }
        val first = track.first()
        val last = track.last()
        val bottomDelta = last.bottomRatio - first.bottomRatio
        val areaGrowth = if (first.areaRatio <= 0f) 0f else (last.areaRatio / first.areaRatio) - 1f
        val depthBandDelta = (last.distanceEvidence?.band?.ordinal ?: -1) -
            (first.distanceEvidence?.band?.ordinal ?: -1)
        val depthScoreDelta = (last.distanceEvidence?.relativeDepthScore ?: 0f) -
            (first.distanceEvidence?.relativeDepthScore ?: 0f)

        return when {
            bottomDelta >= config.minBottomDeltaForApproach ||
                areaGrowth >= config.minAreaGrowthForApproach ||
                depthBandDelta > 0 ||
                depthScoreDelta >= config.minDepthScoreDeltaForApproach -> ApproachTrend.APPROACHING
            bottomDelta <= -config.minBottomDeltaForApproach ||
                areaGrowth <= -config.minAreaGrowthForApproach ||
                depthBandDelta < 0 ||
                depthScoreDelta <= -config.minDepthScoreDeltaForApproach -> ApproachTrend.RECEDING
            else -> ApproachTrend.STABLE
        }
    }

    private fun applyTrend(raw: RiskResult, trend: ApproachTrend, stableFrameCount: Int): RiskResult {
        if (trend != ApproachTrend.APPROACHING) {
            val stableSegmentation = isStableSegmentationEligible(raw) &&
                trend != ApproachTrend.RECEDING && stableFrameCount >= config.minStableSegmentationFrames
            val stabilityFusion = fusionPolicy.fuseStableSegmentation(raw, stableSegmentation)
            return raw.copy(
                level = stabilityFusion.level,
                message = if (stabilityFusion.level >= RiskLevel.MEDIUM) "前方障碍稳定存在，减速观察" else raw.message,
                urgencyScore = stabilityFusion.score,
                riskScore = stabilityFusion.score,
                scoreBreakdown = stabilityFusion.scoreBreakdown,
                approachTrend = trend
            )
        }
        val motionFusion = fusionPolicy.fuseMotion(
            raw = raw,
            trend = trend,
            scoreBoost = config.approachScoreBoost
        )
        return raw.copy(
            level = motionFusion.level,
            message = trendMessageFor(raw, motionFusion.level),
            urgencyScore = motionFusion.score,
            riskScore = motionFusion.score,
            scoreBreakdown = motionFusion.scoreBreakdown,
            approachTrend = trend
        )
    }

    private fun trendMessageFor(raw: RiskResult, boostedLevel: RiskLevel): String {
        if (boostedLevel == raw.level && raw.level != RiskLevel.NONE) {
            return raw.message
        }
        return when (raw.direction) {
            RiskDirection.CENTER -> {
                if (boostedLevel >= RiskLevel.MEDIUM) "前方目标正在靠近，减速观察" else "前方目标正在靠近，继续观察"
            }
            RiskDirection.LEFT -> "左前方目标正在靠近，注意观察"
            RiskDirection.RIGHT -> "右前方目标正在靠近，注意观察"
            RiskDirection.NONE -> raw.message
        }
    }

    private fun isStableSegmentationEligible(raw: RiskResult): Boolean {
        val detection = raw.sourceDetection ?: return false
        if (detection.source != DetectionSource.SEGMENTATION) return false
        if (detection.label == "stairs") return true
        if (detection.label !in STABLE_SEGMENTATION_OBSTACLE_LABELS) return false
        return detection.boundingBox.bottom / detection.frameSize.height >= config.minStableSegmentationBottomRatio
    }

    private data class TargetObservation(
        val label: String,
        val direction: RiskDirection,
        val box: BoundingBox,
        val bottomRatio: Float,
        val areaRatio: Float,
        val centerXRatio: Float,
        val source: DetectionSource,
        val distanceEvidence: DistanceEvidence?,
        val observedAtMs: Long
    ) {
        fun matches(other: TargetObservation, config: TemporalRiskTrackerConfig): Boolean {
            if (label != other.label) {
                return false
            }
            if (source == DetectionSource.SEGMENTATION && other.source == DetectionSource.SEGMENTATION) {
                // Segmentation component bounds can deform substantially while the same broad
                // stairway or fixed obstacle remains in the center walking path.
                return abs(centerXRatio - other.centerXRatio) <= SEGMENTATION_MAX_CENTER_DELTA
            }
            if (direction != other.direction) return false
            return iou(box, other.box) >= config.minIouForSameTarget ||
                abs(centerXRatio - other.centerXRatio) <= config.maxCenterDeltaForSameTarget
        }

        companion object {
            fun from(raw: RiskResult, box: BoundingBox, observedAtMs: Long): TargetObservation {
                val frameSize = requireNotNull(raw.sourceDetection).frameSize
                return TargetObservation(
                    label = raw.sourceDetection.label,
                    direction = raw.direction,
                    box = box,
                    bottomRatio = box.bottom / frameSize.height.toFloat(),
                    areaRatio = raw.sourceDetection.areaRatio,
                    centerXRatio = box.centerX / frameSize.width.toFloat(),
                    source = raw.sourceDetection.source,
                    distanceEvidence = raw.distanceEvidence,
                    observedAtMs = observedAtMs
                )
            }

            private fun iou(first: BoundingBox, second: BoundingBox): Float {
                val left = max(first.left, second.left)
                val top = max(first.top, second.top)
                val right = min(first.right, second.right)
                val bottom = min(first.bottom, second.bottom)
                val intersection = max(0f, right - left) * max(0f, bottom - top)
                val union = first.width * first.height + second.width * second.height - intersection
                return if (union <= 0f) 0f else intersection / union
            }
        }
    }

    companion object {
        const val DEFAULT_MAX_FRAMES = 5
        const val DEFAULT_MAX_WINDOW_MS = 900L
        const val DEFAULT_MIN_APPROACH_FRAMES = 3
        const val DEFAULT_MIN_IOU_FOR_SAME_TARGET = 0.25f
        const val DEFAULT_MAX_CENTER_DELTA_FOR_SAME_TARGET = 0.12f
        const val DEFAULT_MIN_BOTTOM_DELTA_FOR_APPROACH = 0.05f
        const val DEFAULT_MIN_AREA_GROWTH_FOR_APPROACH = 0.20f
        const val DEFAULT_MIN_DEPTH_SCORE_DELTA_FOR_APPROACH = 0.12f
        const val DEFAULT_APPROACH_SCORE_BOOST = 1.1f
        const val DEFAULT_MIN_STABLE_SEGMENTATION_FRAMES = 2
        const val DEFAULT_MIN_STABLE_SEGMENTATION_BOTTOM_RATIO = 0.65f
        const val SEGMENTATION_MAX_CENTER_DELTA = 0.25f
        private val STABLE_SEGMENTATION_OBSTACLE_LABELS = setOf("generic obstacle", "pole", "inaccessible surface")
    }
}
