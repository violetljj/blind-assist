package com.linnan.blindassist.ustrf

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class UstrfStructuredSafetyOutputTest {
    private val mapper = UstrfStructuredSafetyOutputMapper(cellMeters = .5f, lookaheadMeters = 2f)

    @Test
    fun nominalSafeCorridorIsReportedAsContinueOrLateralAdjustmentWithoutBypassingShadowMode() {
        val centre = mapper.map(decision(UstrfSafetyAction.SLOW_DOWN, 0), UstrfCorridorCandidate(0, true, .1f, false))
        val left = mapper.map(decision(UstrfSafetyAction.SLOW_DOWN, 1), UstrfCorridorCandidate(1, true, .1f, false))

        assertEquals(UstrfStructuredAction.CONTINUE, centre.action)
        assertEquals(1f, centre.speedScale)
        assertEquals(UstrfStructuredAction.ADJUST_LEFT, left.action)
        assertEquals(.70f, left.speedScale)
        assertTrue(left.headingDeltaRadians > 0f)
        assertTrue(centre.shadowOnly && left.shadowOnly)
    }

    @Test
    fun anyNonNominalSupervisorReasonKeepsSlowDownInsteadOfPublishingContinueOrAdjust() {
        val output = mapper.map(
            decision(UstrfSafetyAction.SLOW_DOWN, 1, setOf(UstrfSafetyReason.SHADOW_ONLY, UstrfSafetyReason.THERMAL_DEGRADED)),
            UstrfCorridorCandidate(1, true, .1f, false)
        )

        assertEquals(UstrfStructuredAction.SLOW_DOWN, output.action)
        assertEquals(.50f, output.speedScale)
        assertEquals(0f, output.headingDeltaRadians)
    }

    @Test
    fun stopAndScanPreserveSupervisorOutcomeAndNeverCarryHeadingOrSpeed() {
        val stop = mapper.map(decision(UstrfSafetyAction.STOP_AND_REASSESS, null), null)
        val scan = mapper.map(decision(UstrfSafetyAction.SCAN, null, setOf(UstrfSafetyReason.SHADOW_ONLY, UstrfSafetyReason.ROUTE_INVALID)), null)

        assertEquals(UstrfStructuredAction.STOP, stop.action)
        assertEquals(UstrfStructuredAction.SCAN, scan.action)
        assertEquals(0f, stop.speedScale)
        assertEquals(0f, scan.speedScale)
    }

    @Test
    fun sessionRecordCarriesTheDerivedStructuredOutputAndDigestBindsIt() {
        val frame = UstrfFrameStamp(1L, 1_000L, "synthetic-local")
        val packet = UstrfPerceptionPacket(
            frame,
            1_000L,
            2_000L,
            (1..4).map { forward -> UstrfRiskObservation(UstrfGridCoordinate(0, forward), 0f, 1f, 0f, 0f, null, 0f, "fixture", 2_000L) }
        )
        val record = UstrfSafetySession().evaluate(
            UstrfSessionInput(
                frame,
                UstrfHealth(UstrfPoseState.TRACKING, UstrfEvidenceState.VALID, UstrfEvidenceState.VALID, UstrfEvidenceState.VALID),
                UstrfPerceptionAssembly.Available(packet),
                UstrfRouteIntent("synthetic-local", 0, 1f, 2_000L)
            )
        )

        assertEquals(UstrfStructuredAction.CONTINUE, record.structuredOutput.action)
        assertTrue(record.structuredOutput.shadowOnly)
        assertTrue(UstrfSessionTraceDigest.canonicalText(listOf(record)).contains("CONTINUE"))
    }

    private fun decision(
        action: UstrfSafetyAction,
        offset: Int?,
        reasons: Set<UstrfSafetyReason> = setOf(UstrfSafetyReason.SHADOW_ONLY)
    ) = UstrfSafetyDecision(action, .1f, .5f, 10L, reasons, offset)
}
