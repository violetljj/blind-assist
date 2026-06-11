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
import kotlin.math.min

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

    fun sampleEvidence(box: BoundingBox, frameSize: FrameSize): DistanceEvidence? {
        val left = floor(box.left / frameSize.width.toFloat() * width).toInt().coerceIn(0, width - 1)
        val right = ceil(box.right / frameSize.width.toFloat() * width).toInt().coerceIn(left + 1, width)
        val top = floor(box.top / frameSize.height.toFloat() * height).toInt().coerceIn(0, height - 1)
        val bottom = ceil(box.bottom / frameSize.height.toFloat() * height).toInt().coerceIn(top + 1, height)
        val lowerHalfTop = top + max(0, (bottom - top) / 2)

        val samples = ArrayList<Float>((right - left) * max(1, bottom - lowerHalfTop))
        for (y in lowerHalfTop until bottom) {
            val row = y * width
            for (x in left until right) {
                val value = closeness[row + x]
                if (value.isFinite()) {
                    samples += value.coerceIn(0f, 1f)
                }
            }
        }
        if (samples.size < MIN_SAMPLES) return null

        samples.sort()
        val median = samples[samples.size / 2]
        val low = samples[(samples.lastIndex * 0.25f).toInt()]
        val high = samples[(samples.lastIndex * 0.75f).toInt()]
        val stability = (1f - (high - low)).coerceIn(0f, 1f)
        val confidence = (0.45f + stability * 0.45f).coerceIn(0f, 0.95f)
        if (confidence < MIN_CONFIDENCE) return null

        return DistanceEvidence(
            band = bandFor(median),
            confidence = confidence,
            source = DistanceEvidenceSource.MONOCULAR_DEPTH,
            relativeDepthScore = median.coerceIn(0f, 1f)
        )
    }

    private fun bandFor(score: Float): ProximityBand {
        return when {
            score >= 0.78f -> ProximityBand.CRITICAL
            score >= 0.58f -> ProximityBand.NEAR
            score >= 0.35f -> ProximityBand.MID
            else -> ProximityBand.FAR
        }
    }

    companion object {
        private const val MIN_SAMPLES = 4
        private const val MIN_CONFIDENCE = 0.55f
    }
}
