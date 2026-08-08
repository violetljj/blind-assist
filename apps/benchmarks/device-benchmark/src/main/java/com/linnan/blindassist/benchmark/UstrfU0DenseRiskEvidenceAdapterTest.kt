package com.linnan.blindassist.benchmark

import androidx.test.ext.junit.runners.AndroidJUnit4
import com.linnan.blindassist.risk.ApproachTrend
import com.linnan.blindassist.risk.ProximityBand
import com.linnan.blindassist.risk.RiskLevel
import com.linnan.blindassist.session.AssistDecisionKernel
import com.linnan.blindassist.ustrf.UstrfFrameStamp
import com.linnan.blindassist.ustrf.UstrfRouteConditionedRiskFailure
import com.linnan.blindassist.ustrf.UstrfRouteConditionedRiskResolution
import com.linnan.blindassist.ustrf.UstrfRouteIntrusionEvidence
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class UstrfU0DenseRiskEvidenceAdapterTest {
    @Test
    fun localPeakIsPreservedWithoutManufacturingDetectionLifecycleOrFeedback() {
        val result = available(
            UstrfU0DenseRiskEvidenceAdapter.normalize(
                resolution(average = 0.40f, peak = 1.0f),
                episodeId = "episode-1",
                decisionFrameId = "frame-1",
                decisionAtMs = 500
            )
        )

        assertEquals(0.75f, result.normalizedRiskScore, 0f)
        assertEquals(RiskLevel.HIGH, result.kernelEvidence.rawRisk.level)
        assertEquals(ProximityBand.CRITICAL, result.kernelEvidence.rawRisk.proximity)
        assertNull(result.kernelEvidence.rawRisk.sourceDetection)
        assertEquals(ApproachTrend.UNKNOWN, result.kernelEvidence.rawRisk.approachTrend)
        assertEquals("episode-1:route-1", result.kernelEvidence.eventKey)
        assertEquals(AssistDecisionKernel.RISK_EVIDENCE_INPUT_CONTRACT_ID, result.kernelEvidence.sourceContractId)
        assertEquals(4, result.kernelEvidence.evidenceCount)
    }

    @Test
    fun continuousScoreUsesFrozenNoneLowMediumHighBands() {
        val values = listOf(0.34f, 0.35f, 0.55f, 0.75f)
        val levels = values.map { score ->
            available(
                UstrfU0DenseRiskEvidenceAdapter.normalize(
                    resolution(average = score, peak = score), "episode-1", "frame-1", 500
                )
            ).kernelEvidence.rawRisk.level
        }
        assertEquals(listOf(RiskLevel.NONE, RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH), levels)
    }

    @Test
    fun unknownFieldUpstreamFailureAndTimeMismatchFailClosed() {
        val unknown = UstrfU0DenseRiskEvidenceAdapter.normalize(
            resolution(average = 0.9f, peak = 1.0f, unknown = 0.51f), "episode-1", "frame-1", 500
        )
        val upstream = UstrfU0DenseRiskEvidenceAdapter.normalize(
            UstrfRouteConditionedRiskResolution.Unavailable(UstrfRouteConditionedRiskFailure.ROUTE_STALE),
            "episode-1", "frame-1", 500
        )
        val time = UstrfU0DenseRiskEvidenceAdapter.normalize(
            resolution(average = 0.9f, peak = 1.0f), "episode-1", "frame-1", 501
        )

        assertEquals(
            UstrfU0DenseRiskEvidenceFailure.ROUTE_UNKNOWN_FRACTION_TOO_HIGH,
            unavailable(unknown).failure
        )
        assertEquals(
            UstrfU0DenseRiskEvidenceFailure.ROUTE_CONDITIONED_EVIDENCE_UNAVAILABLE,
            unavailable(upstream).failure
        )
        assertEquals("ROUTE_STALE", unavailable(upstream).sourceFailure)
        assertEquals(UstrfU0DenseRiskEvidenceFailure.INVALID_TIME_BINDING, unavailable(time).failure)
    }

    private fun resolution(
        average: Float,
        peak: Float,
        unknown: Float = 0f
    ) = UstrfRouteConditionedRiskResolution.Available(
        UstrfRouteIntrusionEvidence(
            sourceFrame = UstrfFrameStamp(1, 500_000_000, "u0-camera-grid"),
            routeIntentId = "route-1",
            routeIntrusionScore = average,
            maximumRouteCellRisk = peak,
            routeUnknownFraction = unknown,
            contributingCellCount = 4,
            validUntilNs = 900_000_000,
            riskSources = setOf("dense-teacher-v1")
        )
    )

    private fun available(value: UstrfU0DenseRiskEvidenceResolution): UstrfU0DenseRiskEvidenceResolution.Available {
        assertTrue(value is UstrfU0DenseRiskEvidenceResolution.Available)
        return value as UstrfU0DenseRiskEvidenceResolution.Available
    }

    private fun unavailable(value: UstrfU0DenseRiskEvidenceResolution): UstrfU0DenseRiskEvidenceResolution.Unavailable {
        assertTrue(value is UstrfU0DenseRiskEvidenceResolution.Unavailable)
        return value as UstrfU0DenseRiskEvidenceResolution.Unavailable
    }
}
