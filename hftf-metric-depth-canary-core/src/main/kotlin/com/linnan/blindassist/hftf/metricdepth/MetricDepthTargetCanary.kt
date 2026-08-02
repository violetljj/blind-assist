package com.linnan.blindassist.hftf.metricdepth

import com.linnan.blindassist.model.Detection
import com.linnan.blindassist.model.DetectionSource
import com.linnan.blindassist.model.FrameSize
import com.linnan.blindassist.vision.FrameClockDomain
import com.linnan.blindassist.vision.FrameStamp
import kotlin.math.ceil
import kotlin.math.floor

enum class MetricDepthSource {
    ARCORE_AUTOMATIC,
    ARCORE_RAW_REGISTERED,
    CAMERA2_DEPTH16,
    CALIBRATED_MONOCULAR
}

data class MetricDepthCameraIntrinsics(
    val imageWidthPx: Int,
    val imageHeightPx: Int,
    val focalXpx: Float,
    val focalYpx: Float,
    val principalXpx: Float,
    val principalYpx: Float
) {
    init {
        require(imageWidthPx > 0 && imageHeightPx > 0)
        require(focalXpx.isFinite() && focalXpx > 0f)
        require(focalYpx.isFinite() && focalYpx > 0f)
        require(principalXpx.isFinite() && principalYpx.isFinite())
        require(principalXpx in 0f..imageWidthPx.toFloat())
        require(principalYpx in 0f..imageHeightPx.toFloat())
    }
}

/**
 * A metric depth raster already registered into the detector's display-oriented image grid.
 *
 * Source-specific rotation and registration are deliberately outside this contract. A source
 * adapter must provide a stable [registrationTransformId] before a frame can be sampled.
 */
data class RegisteredMetricDepthFrame(
    val sourceFrame: FrameStamp,
    val detectorDisplaySize: FrameSize,
    val intrinsics: MetricDepthCameraIntrinsics,
    val depthMillimeters: IntArray,
    val confidence: FloatArray,
    val source: MetricDepthSource,
    val registrationTransformId: String,
    val producedAtNs: Long,
    val validUntilNs: Long
) {
    init {
        require(detectorDisplaySize.width > 0 && detectorDisplaySize.height > 0)
        require(intrinsics.imageWidthPx * intrinsics.imageHeightPx == depthMillimeters.size)
        require(confidence.size == depthMillimeters.size)
        require(depthMillimeters.all { it >= 0 })
        require(confidence.all { it.isFinite() && it in 0f..1f })
        require(registrationTransformId.isNotBlank())
        require(sourceFrame.clockDomain == FrameClockDomain.ANDROID_ELAPSED_REALTIME) {
            "live metric depth requires an elapsed-realtime source frame"
        }
        require(producedAtNs >= sourceFrame.receivedAtNs)
        require(validUntilNs >= producedAtNs)
    }
}

data class MetricDepthTargetSamplerConfig(
    val innerCropRatio: Float = 0.60f,
    val minimumDepthMeters: Float = 0.20f,
    val maximumDepthMeters: Float = 20.0f,
    val minimumSampleConfidence: Float = 0.50f,
    val minimumValidSamples: Int = 12,
    val minimumCoverage: Float = 0.25f,
    val minimumMeanConfidence: Float = 0.50f,
    val maximumRelativeIqr: Float = 0.50f,
    val maximumReceiptAgeNs: Long = 150_000_000L
) {
    init {
        require(innerCropRatio > 0f && innerCropRatio <= 1f)
        require(minimumDepthMeters > 0f && maximumDepthMeters > minimumDepthMeters)
        require(minimumSampleConfidence in 0f..1f)
        require(minimumValidSamples >= 1)
        require(minimumCoverage in 0f..1f)
        require(minimumMeanConfidence in 0f..1f)
        require(maximumRelativeIqr >= 0f)
        require(maximumReceiptAgeNs >= 0L)
    }
}

data class MetricVector3Meters(
    val x: Float,
    val y: Float,
    val z: Float
) {
    init {
        require(x.isFinite() && y.isFinite() && z.isFinite())
    }
}

data class MetricDepthSampleDiagnostics(
    val candidateSampleCount: Int,
    val validSampleCount: Int,
    val coverage: Float,
    val meanConfidence: Float?,
    val relativeIqr: Float?,
    val receiptAgeNs: Long
)

data class MetricDepthTargetMeasurement(
    val targetKey: String,
    val frameId: Long,
    val capturedAtNs: Long,
    val source: MetricDepthSource,
    val registrationTransformId: String,
    val opticalAxisDepthMeters: Float,
    /** Camera axes: x right, y down, z forward. */
    val positionCameraMeters: MetricVector3Meters,
    val qualityScore: Float,
    val diagnostics: MetricDepthSampleDiagnostics,
    val producedAtNs: Long
) {
    init {
        require(targetKey.isNotBlank())
        require(frameId >= 0L && capturedAtNs >= 0L && producedAtNs >= capturedAtNs)
        require(opticalAxisDepthMeters.isFinite() && opticalAxisDepthMeters > 0f)
        require(qualityScore in 0f..1f)
    }
}

enum class MetricDepthSampleFailure {
    NOT_PERSON,
    DETECTION_FRAME_MISMATCH,
    RECEIPT_FROM_FUTURE,
    RECEIPT_STALE,
    NO_REGISTERED_PIXELS,
    INSUFFICIENT_VALID_SAMPLES,
    COVERAGE_BELOW_FLOOR,
    CONFIDENCE_BELOW_FLOOR,
    DISPERSION_ABOVE_CEILING
}

sealed interface MetricDepthSampleResult {
    data class Available(val measurement: MetricDepthTargetMeasurement) : MetricDepthSampleResult
    data class Unavailable(
        val failure: MetricDepthSampleFailure,
        val diagnostics: MetricDepthSampleDiagnostics
    ) : MetricDepthSampleResult
}

class MetricDepthTargetSampler(
    private val config: MetricDepthTargetSamplerConfig = MetricDepthTargetSamplerConfig()
) {
    fun sample(
        frame: RegisteredMetricDepthFrame,
        detection: Detection,
        targetKey: String,
        observedAtNs: Long
    ): MetricDepthSampleResult {
        require(targetKey.isNotBlank())
        require(observedAtNs >= 0L)
        val ageNs = observedAtNs - frame.sourceFrame.capturedAtNs
        val emptyDiagnostics = MetricDepthSampleDiagnostics(0, 0, 0f, null, null, ageNs)
        if (
            detection.label != PERSON_LABEL ||
            detection.source != DetectionSource.OBJECT_DETECTOR
        ) {
            return MetricDepthSampleResult.Unavailable(MetricDepthSampleFailure.NOT_PERSON, emptyDiagnostics)
        }
        if (detection.frameSize != frame.detectorDisplaySize) {
            return MetricDepthSampleResult.Unavailable(
                MetricDepthSampleFailure.DETECTION_FRAME_MISMATCH,
                emptyDiagnostics
            )
        }
        if (ageNs < 0L) {
            return MetricDepthSampleResult.Unavailable(
                MetricDepthSampleFailure.RECEIPT_FROM_FUTURE,
                emptyDiagnostics
            )
        }
        if (ageNs > config.maximumReceiptAgeNs || observedAtNs > frame.validUntilNs) {
            return MetricDepthSampleResult.Unavailable(
                MetricDepthSampleFailure.RECEIPT_STALE,
                emptyDiagnostics
            )
        }

        val region = registeredInnerRegion(frame, detection)
        if (region.right <= region.left || region.bottom <= region.top) {
            return MetricDepthSampleResult.Unavailable(
                MetricDepthSampleFailure.NO_REGISTERED_PIXELS,
                emptyDiagnostics
            )
        }
        val candidateCount = (region.right - region.left) * (region.bottom - region.top)
        val depths = ArrayList<Float>(candidateCount)
        var confidenceSum = 0f
        for (y in region.top until region.bottom) {
            val row = y * frame.intrinsics.imageWidthPx
            for (x in region.left until region.right) {
                val index = row + x
                val depthMeters = frame.depthMillimeters[index] / MILLIMETERS_PER_METER
                val sampleConfidence = frame.confidence[index]
                if (
                    depthMeters in config.minimumDepthMeters..config.maximumDepthMeters &&
                    sampleConfidence >= config.minimumSampleConfidence
                ) {
                    depths += depthMeters
                    confidenceSum += sampleConfidence
                }
            }
        }
        val coverage = if (candidateCount == 0) 0f else depths.size.toFloat() / candidateCount
        val meanConfidence = if (depths.isEmpty()) null else confidenceSum / depths.size
        if (depths.size < config.minimumValidSamples) {
            return unavailable(
                MetricDepthSampleFailure.INSUFFICIENT_VALID_SAMPLES,
                candidateCount,
                depths.size,
                coverage,
                meanConfidence,
                null,
                ageNs
            )
        }

        depths.sort()
        val medianDepth = median(depths)
        val lowerQuartile = percentile(depths, 0.25f)
        val upperQuartile = percentile(depths, 0.75f)
        val relativeIqr = (upperQuartile - lowerQuartile) / medianDepth
        val diagnostics = MetricDepthSampleDiagnostics(
            candidateSampleCount = candidateCount,
            validSampleCount = depths.size,
            coverage = coverage,
            meanConfidence = meanConfidence,
            relativeIqr = relativeIqr,
            receiptAgeNs = ageNs
        )
        if (coverage < config.minimumCoverage) {
            return MetricDepthSampleResult.Unavailable(
                MetricDepthSampleFailure.COVERAGE_BELOW_FLOOR,
                diagnostics
            )
        }
        if (requireNotNull(meanConfidence) < config.minimumMeanConfidence) {
            return MetricDepthSampleResult.Unavailable(
                MetricDepthSampleFailure.CONFIDENCE_BELOW_FLOOR,
                diagnostics
            )
        }
        if (relativeIqr > config.maximumRelativeIqr) {
            return MetricDepthSampleResult.Unavailable(
                MetricDepthSampleFailure.DISPERSION_ABOVE_CEILING,
                diagnostics
            )
        }

        val depthU = detection.centerX / frame.detectorDisplaySize.width *
            frame.intrinsics.imageWidthPx
        val depthV = detection.centerY / frame.detectorDisplaySize.height *
            frame.intrinsics.imageHeightPx
        val position = MetricVector3Meters(
            x = (depthU - frame.intrinsics.principalXpx) * medianDepth / frame.intrinsics.focalXpx,
            y = (depthV - frame.intrinsics.principalYpx) * medianDepth / frame.intrinsics.focalYpx,
            z = medianDepth
        )
        val stability = (1f / (1f + relativeIqr)).coerceIn(0f, 1f)
        val quality = minOf(coverage, meanConfidence, stability).coerceIn(0f, 1f)
        return MetricDepthSampleResult.Available(
            MetricDepthTargetMeasurement(
                targetKey = targetKey,
                frameId = frame.sourceFrame.frameId,
                capturedAtNs = frame.sourceFrame.capturedAtNs,
                source = frame.source,
                registrationTransformId = frame.registrationTransformId,
                opticalAxisDepthMeters = medianDepth,
                positionCameraMeters = position,
                qualityScore = quality,
                diagnostics = diagnostics,
                producedAtNs = frame.producedAtNs
            )
        )
    }

    private fun registeredInnerRegion(
        frame: RegisteredMetricDepthFrame,
        detection: Detection
    ): PixelRegion {
        val box = detection.boundingBox.clamped(frame.detectorDisplaySize)
        val halfWidth = box.width * config.innerCropRatio / 2f
        val halfHeight = box.height * config.innerCropRatio / 2f
        val leftDisplay = box.centerX - halfWidth
        val rightDisplay = box.centerX + halfWidth
        val topDisplay = box.centerY - halfHeight
        val bottomDisplay = box.centerY + halfHeight
        val depthWidth = frame.intrinsics.imageWidthPx
        val depthHeight = frame.intrinsics.imageHeightPx
        return PixelRegion(
            left = floor(leftDisplay / frame.detectorDisplaySize.width * depthWidth)
                .toInt().coerceIn(0, depthWidth),
            right = ceil(rightDisplay / frame.detectorDisplaySize.width * depthWidth)
                .toInt().coerceIn(0, depthWidth),
            top = floor(topDisplay / frame.detectorDisplaySize.height * depthHeight)
                .toInt().coerceIn(0, depthHeight),
            bottom = ceil(bottomDisplay / frame.detectorDisplaySize.height * depthHeight)
                .toInt().coerceIn(0, depthHeight)
        )
    }

    private fun unavailable(
        failure: MetricDepthSampleFailure,
        candidateCount: Int,
        validCount: Int,
        coverage: Float,
        meanConfidence: Float?,
        relativeIqr: Float?,
        ageNs: Long
    ) = MetricDepthSampleResult.Unavailable(
        failure,
        MetricDepthSampleDiagnostics(
            candidateSampleCount = candidateCount,
            validSampleCount = validCount,
            coverage = coverage,
            meanConfidence = meanConfidence,
            relativeIqr = relativeIqr,
            receiptAgeNs = ageNs
        )
    )

    private fun median(sorted: List<Float>): Float {
        val middle = sorted.size / 2
        return if (sorted.size % 2 == 0) {
            (sorted[middle - 1] + sorted[middle]) / 2f
        } else {
            sorted[middle]
        }
    }

    private fun percentile(sorted: List<Float>, percentile: Float): Float {
        val index = (percentile * (sorted.size - 1)).toInt().coerceIn(0, sorted.lastIndex)
        return sorted[index]
    }

    private data class PixelRegion(
        val left: Int,
        val right: Int,
        val top: Int,
        val bottom: Int
    )

    private companion object {
        const val PERSON_LABEL = "person"
        const val MILLIMETERS_PER_METER = 1_000f
    }
}

data class MetricDepthHistorySolverConfig(
    val historySize: Int = 7,
    val minimumHistorySpanNs: Long = 200_000_000L,
    val maximumAdjacentGapNs: Long = 200_000_000L,
    val forecastHorizonNs: Long = 1_000_000_000L
) {
    init {
        require(historySize >= 2)
        require(minimumHistorySpanNs > 0L)
        require(maximumAdjacentGapNs > 0L)
        require(forecastHorizonNs > 0L)
    }
}

enum class MetricDepthHistoryFailure {
    INSUFFICIENT_HISTORY,
    NON_MONOTONIC_HISTORY,
    MIXED_TARGET,
    MIXED_SOURCE_OR_REGISTRATION,
    HISTORY_SPAN_BELOW_FLOOR,
    ADJACENT_GAP_ABOVE_CEILING
}

data class MetricDepthTargetForecast(
    val targetKey: String,
    val source: MetricDepthSource,
    val registrationTransformId: String,
    val currentFrameId: Long,
    val currentPositionCameraMeters: MetricVector3Meters,
    val predictedPositionCameraMeters: MetricVector3Meters,
    val velocityCameraMetersPerSecond: MetricVector3Meters,
    val predictedAtNs: Long,
    val historySize: Int,
    val historySpanNs: Long,
    val minimumInputQuality: Float
)

sealed interface MetricDepthHistoryResult {
    data class Available(val forecast: MetricDepthTargetForecast) : MetricDepthHistoryResult
    data class Unavailable(val failure: MetricDepthHistoryFailure) : MetricDepthHistoryResult
}

class MetricDepthHistorySolver(
    private val config: MetricDepthHistorySolverConfig = MetricDepthHistorySolverConfig()
) {
    fun forecast(history: List<MetricDepthTargetMeasurement>): MetricDepthHistoryResult {
        if (history.size < config.historySize) {
            return MetricDepthHistoryResult.Unavailable(
                MetricDepthHistoryFailure.INSUFFICIENT_HISTORY
            )
        }
        val window = history.takeLast(config.historySize)
        if (window.zipWithNext().any { (left, right) -> right.capturedAtNs <= left.capturedAtNs }) {
            return MetricDepthHistoryResult.Unavailable(
                MetricDepthHistoryFailure.NON_MONOTONIC_HISTORY
            )
        }
        if (window.any { it.targetKey != window.first().targetKey }) {
            return MetricDepthHistoryResult.Unavailable(MetricDepthHistoryFailure.MIXED_TARGET)
        }
        if (
            window.any {
                it.source != window.first().source ||
                    it.registrationTransformId != window.first().registrationTransformId
            }
        ) {
            return MetricDepthHistoryResult.Unavailable(
                MetricDepthHistoryFailure.MIXED_SOURCE_OR_REGISTRATION
            )
        }
        val spanNs = window.last().capturedAtNs - window.first().capturedAtNs
        if (spanNs < config.minimumHistorySpanNs) {
            return MetricDepthHistoryResult.Unavailable(
                MetricDepthHistoryFailure.HISTORY_SPAN_BELOW_FLOOR
            )
        }
        if (
            window.zipWithNext().any { (left, right) ->
                right.capturedAtNs - left.capturedAtNs > config.maximumAdjacentGapNs
            }
        ) {
            return MetricDepthHistoryResult.Unavailable(
                MetricDepthHistoryFailure.ADJACENT_GAP_ABOVE_CEILING
            )
        }

        val firstTimeNs = window.first().capturedAtNs
        val timesSeconds = window.map {
            (it.capturedAtNs - firstTimeNs).toDouble() / NANOS_PER_SECOND
        }
        val targetTimeSeconds = (
            window.last().capturedAtNs - firstTimeNs + config.forecastHorizonNs
            ).toDouble() / NANOS_PER_SECOND
        val xLine = fitLine(timesSeconds, window.map { it.positionCameraMeters.x.toDouble() })
        val yLine = fitLine(timesSeconds, window.map { it.positionCameraMeters.y.toDouble() })
        val zLine = fitLine(timesSeconds, window.map { it.positionCameraMeters.z.toDouble() })
        val latest = window.last()
        return MetricDepthHistoryResult.Available(
            MetricDepthTargetForecast(
                targetKey = latest.targetKey,
                source = latest.source,
                registrationTransformId = latest.registrationTransformId,
                currentFrameId = latest.frameId,
                currentPositionCameraMeters = latest.positionCameraMeters,
                predictedPositionCameraMeters = MetricVector3Meters(
                    x = xLine.valueAt(targetTimeSeconds).toFloat(),
                    y = yLine.valueAt(targetTimeSeconds).toFloat(),
                    z = zLine.valueAt(targetTimeSeconds).toFloat()
                ),
                velocityCameraMetersPerSecond = MetricVector3Meters(
                    x = xLine.slope.toFloat(),
                    y = yLine.slope.toFloat(),
                    z = zLine.slope.toFloat()
                ),
                predictedAtNs = latest.capturedAtNs + config.forecastHorizonNs,
                historySize = window.size,
                historySpanNs = spanNs,
                minimumInputQuality = window.minOf { it.qualityScore }
            )
        )
    }

    private fun fitLine(times: List<Double>, values: List<Double>): Line {
        val meanTime = times.average()
        val meanValue = values.average()
        var covariance = 0.0
        var variance = 0.0
        for (index in times.indices) {
            val centeredTime = times[index] - meanTime
            covariance += centeredTime * (values[index] - meanValue)
            variance += centeredTime * centeredTime
        }
        require(variance > 0.0)
        val slope = covariance / variance
        return Line(intercept = meanValue - slope * meanTime, slope = slope)
    }

    private data class Line(val intercept: Double, val slope: Double) {
        fun valueAt(timeSeconds: Double): Double = intercept + slope * timeSeconds
    }

    private companion object {
        const val NANOS_PER_SECOND = 1_000_000_000.0
    }
}
