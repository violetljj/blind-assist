package com.linnan.blindassist.vision

import android.graphics.Bitmap
import com.linnan.blindassist.model.BoundingBox
import com.linnan.blindassist.model.FrameSize
import com.linnan.blindassist.risk.DistanceEvidence
import com.linnan.blindassist.risk.DistanceEvidenceSource
import com.linnan.blindassist.risk.ProximityBand
import kotlin.math.ceil
import kotlin.math.floor
import kotlin.math.max
import kotlin.math.roundToInt

interface DepthEstimator : AutoCloseable {
    val isReady: Boolean
    val statusMessage: String

    fun estimate(bitmap: Bitmap): DepthFrameResult

    override fun close()
}

data class DepthFrameResult(
    val depthMap: RelativeDepthMap,
    val metrics: DepthEstimatorMetrics
)

data class DepthEstimatorMetrics(
    val totalMs: Long,
    val preprocessMs: Long,
    val inferenceMs: Long,
    val postprocessMs: Long,
    val modelStatus: String
)

data class DepthEvidenceSamplingConfig(
    val samplePercentile: Float = 0.50f,
    val innerCropRatio: Float = 1.0f,
    val lowerHalfOnly: Boolean = true,
    val minSamples: Int = 4,
    val minLocalRange: Float = 0f,
    val minConfidence: Float = 0.55f,
    val criticalThreshold: Float = 0.78f,
    val nearThreshold: Float = 0.58f,
    val midThreshold: Float = 0.35f
) {
    init {
        require(samplePercentile in 0f..1f) { "samplePercentile must be in [0, 1]" }
        require(innerCropRatio > 0f && innerCropRatio <= 1f) { "innerCropRatio must be in (0, 1]" }
        require(minSamples >= 1) { "minSamples must be positive" }
        require(minLocalRange >= 0f) { "minLocalRange must be non-negative" }
        require(minConfidence in 0f..1f) { "minConfidence must be in [0, 1]" }
        require(criticalThreshold in 0f..1f) { "criticalThreshold must be in [0, 1]" }
        require(nearThreshold in 0f..1f) { "nearThreshold must be in [0, 1]" }
        require(midThreshold in 0f..1f) { "midThreshold must be in [0, 1]" }
        require(criticalThreshold >= nearThreshold && nearThreshold >= midThreshold) {
            "depth thresholds must be ordered critical >= near >= mid"
        }
    }
}

data class RelativeDepthMap(
    val width: Int,
    val height: Int,
    val closeness: FloatArray
) {
    init {
        require(width > 0) { "width must be positive" }
        require(height > 0) { "height must be positive" }
        require(closeness.size == width * height) { "closeness size must equal width * height" }
    }

    fun sampleEvidence(
        box: BoundingBox,
        frameSize: FrameSize,
        config: DepthEvidenceSamplingConfig = DepthEvidenceSamplingConfig()
    ): DistanceEvidence? {
        val fullLeft = floor(box.left / frameSize.width.toFloat() * width).toInt().coerceIn(0, width - 1)
        val fullRight = ceil(box.right / frameSize.width.toFloat() * width).toInt().coerceIn(fullLeft + 1, width)
        val fullTop = floor(box.top / frameSize.height.toFloat() * height).toInt().coerceIn(0, height - 1)
        val fullBottom = ceil(box.bottom / frameSize.height.toFloat() * height).toInt().coerceIn(fullTop + 1, height)

        val left = cropStart(fullLeft, fullRight, config.innerCropRatio)
        val right = cropEnd(fullLeft, fullRight, config.innerCropRatio).coerceAtLeast(left + 1)
        val cropTop = cropStart(fullTop, fullBottom, config.innerCropRatio)
        val bottom = cropEnd(fullTop, fullBottom, config.innerCropRatio).coerceAtLeast(cropTop + 1)
        val top = if (config.lowerHalfOnly) {
            cropTop + max(0, (bottom - cropTop) / 2)
        } else {
            cropTop
        }

        val samples = ArrayList<Float>((right - left) * max(1, bottom - top))
        for (y in top until bottom) {
            val row = y * width
            for (x in left until right) {
                val value = closeness[row + x]
                if (value.isFinite()) {
                    samples += value.coerceIn(0f, 1f)
                }
            }
        }
        if (samples.size < config.minSamples) return null

        samples.sort()
        val score = samples[percentileIndex(samples.lastIndex, config.samplePercentile)]
        val low = samples[(samples.lastIndex * 0.25f).toInt()]
        val high = samples[(samples.lastIndex * 0.75f).toInt()]
        val localRange = high - low
        if (localRange < config.minLocalRange) return null
        val stability = (1f - (high - low)).coerceIn(0f, 1f)
        val confidence = (0.45f + stability * 0.45f).coerceIn(0f, 0.95f)
        if (confidence < config.minConfidence) return null

        return DistanceEvidence(
            band = bandFor(score, config),
            confidence = confidence,
            source = DistanceEvidenceSource.MONOCULAR_DEPTH,
            relativeDepthScore = score.coerceIn(0f, 1f)
        )
    }

    private fun bandFor(score: Float, config: DepthEvidenceSamplingConfig): ProximityBand {
        return when {
            score >= config.criticalThreshold -> ProximityBand.CRITICAL
            score >= config.nearThreshold -> ProximityBand.NEAR
            score >= config.midThreshold -> ProximityBand.MID
            else -> ProximityBand.FAR
        }
    }

    private fun cropStart(start: Int, end: Int, ratio: Float): Int {
        val inset = ((end - start) * (1f - ratio) / 2f).toInt()
        return (start + inset).coerceIn(start, end - 1)
    }

    private fun cropEnd(start: Int, end: Int, ratio: Float): Int {
        val inset = ((end - start) * (1f - ratio) / 2f).toInt()
        return (end - inset).coerceIn(start + 1, end)
    }

    private fun percentileIndex(lastIndex: Int, percentile: Float): Int {
        return (lastIndex * percentile).roundToInt().coerceIn(0, lastIndex)
    }
}
