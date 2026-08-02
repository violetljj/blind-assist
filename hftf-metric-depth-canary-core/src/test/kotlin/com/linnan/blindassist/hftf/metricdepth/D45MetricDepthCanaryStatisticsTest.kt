package com.linnan.blindassist.hftf.metricdepth

import org.junit.Assert.assertEquals
import org.junit.Test

class D45MetricDepthCanaryStatisticsTest {
    @Test
    fun exactReferenceMeasurementsHaveZeroError() {
        val summary = D45MetricDepthCanaryStatistics.summarize(
            referenceDistanceMeters = 2f,
            measurements = listOf(1.8f, 2f, 2.2f).mapIndexed(::measurement)
        )

        assertEquals(3, summary.acceptedObservationCount)
        assertEquals(0.2f, summary.absoluteErrorMedianMeters!!, 0.0001f)
        assertEquals(0.2f, summary.absoluteErrorP90Meters!!, 0.0001f)
        assertEquals(0.1f, summary.relativeErrorMedian!!, 0.0001f)
    }

    @Test
    fun emptyMeasurementRunRemainsExplicitlyUnmeasured() {
        val summary = D45MetricDepthCanaryStatistics.summarize(5f, emptyList())

        assertEquals(0, summary.acceptedObservationCount)
        assertEquals(null, summary.absoluteErrorMedianMeters)
        assertEquals(null, summary.absoluteErrorP90Meters)
        assertEquals(null, summary.relativeErrorMedian)
    }

    @Test(expected = IllegalArgumentException::class)
    fun uncontractedReferenceDistanceIsRejectedBeforeCapture() {
        D45MetricDepthCanaryStatistics.summarize(4f, emptyList())
    }

    private fun measurement(index: Int, depthMeters: Float) = MetricDepthTargetMeasurement(
        targetKey = "manual-single-person",
        frameId = index.toLong(),
        capturedAtNs = 1_000L + index * 50L,
        source = MetricDepthSource.ARCORE_RAW_REGISTERED,
        registrationTransformId = "registration",
        opticalAxisDepthMeters = depthMeters,
        positionCameraMeters = MetricVector3Meters(0f, 0f, depthMeters),
        qualityScore = 0.8f,
        diagnostics = MetricDepthSampleDiagnostics(
            candidateSampleCount = 20,
            validSampleCount = 18,
            coverage = 0.9f,
            meanConfidence = 0.9f,
            relativeIqr = 0.1f,
            receiptAgeNs = 20L
        ),
        producedAtNs = 1_020L + index * 50L
    )
}
