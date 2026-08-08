package com.linnan.blindassist.hftf.metricdepth

import kotlin.math.ceil
import kotlin.math.floor

data class D45MetricDepthErrorSummary(
    val referenceDistanceMeters: Float,
    val acceptedObservationCount: Int,
    val absoluteErrorMedianMeters: Float?,
    val absoluteErrorP90Meters: Float?,
    val relativeErrorMedian: Float?
)

object D45MetricDepthCanaryStatistics {
    val ALLOWED_REFERENCE_DISTANCES_METERS = setOf(1f, 2f, 3f, 5f)

    fun summarize(
        referenceDistanceMeters: Float,
        measurements: List<MetricDepthTargetMeasurement>
    ): D45MetricDepthErrorSummary {
        require(referenceDistanceMeters in ALLOWED_REFERENCE_DISTANCES_METERS) {
            "reference distance must be one of 1/2/3/5 metres"
        }
        val absoluteErrors = measurements
            .map { kotlin.math.abs(it.opticalAxisDepthMeters - referenceDistanceMeters) }
            .sorted()
        val relativeErrors = absoluteErrors
            .map { it / referenceDistanceMeters }
            .sorted()
        return D45MetricDepthErrorSummary(
            referenceDistanceMeters = referenceDistanceMeters,
            acceptedObservationCount = measurements.size,
            absoluteErrorMedianMeters = percentileOrNull(absoluteErrors, 0.50f),
            absoluteErrorP90Meters = percentileOrNull(absoluteErrors, 0.90f),
            relativeErrorMedian = percentileOrNull(relativeErrors, 0.50f)
        )
    }

    fun percentileOrNull(sortedValues: List<Float>, percentile: Float): Float? {
        require(percentile in 0f..1f)
        if (sortedValues.isEmpty()) return null
        require(sortedValues.zipWithNext().all { (left, right) -> left <= right })
        val position = percentile * (sortedValues.size - 1)
        val lower = floor(position).toInt()
        val upper = ceil(position).toInt()
        if (lower == upper) return sortedValues[lower]
        val weight = position - lower
        return sortedValues[lower] * (1f - weight) + sortedValues[upper] * weight
    }
}
