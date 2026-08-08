package com.linnan.blindassist.hftf.metricdepth

import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Test

class MetricTraversabilityFieldTest {
    @Test
    fun richFieldKeepsContinuousDirectionsAndAlertIsSeparateShadowProjection() {
        val field = validField()

        assertEquals(5, field.clearanceProfile.size)
        assertEquals(listOf(1f, 1.5f, 2f), field.sweepEnvelopes.map { it.horizonMeters })
        val alert = TraversabilityAlertMapper().map(field)
        assertEquals(ShadowAlertStatus.CENTER_RISK, alert.status)
        assertEquals("SHADOW_DEMO_ONLY", alert.authority)
        assertEquals(20, alert.observedOpeningAngleDegrees)
    }

    @Test
    fun unknownCenterSupportNeverBecomesClearOrDirectionGuidance() {
        val field = validField().copy(
            sweepEnvelopes = validField().sweepEnvelopes.map { envelope ->
                if (envelope.horizonMeters != 1.5f) envelope else envelope.copy(
                    directions = envelope.directions.map {
                        if (it.angleDegrees == 0) it.copy(state = SweepObservationState.UNKNOWN_SUPPORT) else it
                    }
                )
            }
        )

        val alert = TraversabilityAlertMapper().map(field)

        assertEquals(ShadowAlertStatus.SILENT_UNKNOWN, alert.status)
        assertEquals(listOf("UNKNOWN_CENTER_SUPPORT"), alert.unknownReasons)
        assertEquals(null, alert.observedOpeningAngleDegrees)
    }

    @Test
    fun unavailableMetricDepthCannotRetainMeterValues() {
        assertThrows(IllegalArgumentException::class.java) {
            MetricDepthSummary(
                available = false,
                sourceModel = "candidate",
                scaleStatus = "UNKNOWN_STALE_METRIC_SCALE_ANCHOR",
                scale = null,
                anchorAgeNs = null,
                anchorSource = null,
                finiteFraction = 0.8f,
                p05Meters = 0.5f,
                p50Meters = null,
                p95Meters = null
            )
        }
    }

    private fun validField(): MetricTraversabilityField {
        val angles = listOf(-20, -10, 0, 10, 20)
        val provenance = TraversabilityProvenance("metric-source", "depth-ransac", "tof")
        val profile = angles.map { angle ->
            DirectionalClearanceSample(
                angleDegrees = angle,
                nearestIntrusionMeters = if (angle == 0) 1.0f else 2.5f,
                riskScore = if (angle == 0) 0.75f else 0.375f,
                knownScore = 0.9f,
                intrusionPoints = 30,
                supportPoints = 60,
                observedForwardMeters = 3f,
                provenance = provenance
            )
        }
        val envelopes = listOf(1f, 1.5f, 2f).map { horizon ->
            BodySweepEnvelope(
                horizonMeters = horizon,
                bodyHalfWidthMeters = 0.32f,
                lateralMarginMeters = 0.1f,
                directions = profile.map {
                    DirectionalSweepState(
                        it.angleDegrees,
                        if (it.nearestIntrusionMeters!! <= horizon) {
                            SweepObservationState.OCCUPIED_OBSERVED
                        } else {
                            SweepObservationState.CLEAR_OBSERVED
                        }
                    )
                }
            )
        }
        return MetricTraversabilityField(
            frameId = 7,
            capturedAtNs = 1_000,
            sourceId = "candidate-sidecar",
            status = TraversabilityFieldStatus.VALID,
            calibratedDepth = MetricDepthSummary(
                available = true,
                sourceModel = "metric-source",
                scaleStatus = "VALID",
                scale = 1f,
                anchorAgeNs = 10,
                anchorSource = "tof",
                finiteFraction = 0.9f,
                p05Meters = 0.5f,
                p50Meters = 2f,
                p95Meters = 4f
            ),
            groundPlane = TraversabilityGroundPlane(
                "depth-ransac", listOf(0f, -1f, 0f), 1.2f, 0.01f
            ),
            clearanceProfile = profile,
            sweepEnvelopes = envelopes,
            intrusionRegions = listOf(TraversabilityIntrusionRegion(1, -10, 10, 1f)),
            bestObservedClearanceDirection = ObservedClearanceCandidate(20, 2.5f),
            temporalTrend = TraversabilityTemporalTrend(
                "UNKNOWN_NO_COMPARABLE_PREVIOUS_FIELD", null, 0,
                "observed deltas only; not motion prediction"
            ),
            quality = TraversabilityQuality(
                imageQualityPass = true,
                laplacianVariance320x240 = 40f,
                underexposedFraction = 0.01f,
                overexposedFraction = 0.01f,
                depthFiniteFraction = 0.9f,
                depthSupportPass = true,
                groundSupportPass = true,
                directionSupportFraction = 1f,
                overallConfidence = 0.8f
            ),
            unknownReasons = emptyList()
        )
    }
}
