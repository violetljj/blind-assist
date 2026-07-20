package com.linnan.blindassist.ustrf

import org.junit.Assert.assertEquals
import org.junit.Test

class UstrfDocumentFiveMeterProfileTest {
    @Test
    fun completeFiveMeterGridCanProduceShadowContinueAndAHighRiskAtFiveMetersStops() {
        val frame = UstrfFrameStamp(1L, 1_000L, "offline-five-meter")
        val safe = UstrfDocumentFiveMeterProfile.safetySession().evaluate(input(frame, null))
        val blocked = UstrfDocumentFiveMeterProfile.safetySession().evaluate(input(frame, UstrfGridCoordinate(0, 10)))

        assertEquals(UstrfStructuredAction.CONTINUE, safe.structuredOutput.action)
        assertEquals(1f, safe.structuredOutput.speedScale)
        assertEquals(UstrfSafetyAction.STOP_AND_REASSESS, blocked.decision.action)
        assertEquals(UstrfStructuredAction.STOP, blocked.structuredOutput.action)
    }

    @Test
    fun profileSharesOneHalfMeterGridAcrossGeometryMotionAndRiskField() {
        val frame = UstrfFrameStamp(1L, 1_000L, "offline-five-meter")
        val geometry = UstrfGeometryPacket(
            frame, 1_000L, 2_000L, UstrfDepthScale.METRIC,
            listOf(UstrfMetricGeometryEvidence(4.9f, 0f, UstrfHeightBand.TORSO, UstrfGeometryKind.OCCUPIED, 1f, "fixture", 2_000L))
        )
        val assembled = UstrfDocumentFiveMeterProfile.perceptionAssembler().assemble(frame, geometry, emptyList(), 1_000L) as UstrfPerceptionAssembly.Available
        val next = UstrfFrameStamp(2L, 2_000L, "offline-five-meter")
        val motion = UstrfDocumentFiveMeterProfile.egoMotionPromoter().promote(
            UstrfDynamicTrackPair("track", frame, next, UstrfVector2(4.5f, 0f), UstrfVector2(4.5f, 0f), 1f, "fixture", 3_000L),
            UstrfVerifiedPoseDelta(frame, next, 0f, 0f, 0f, true),
            next.capturedAtNs
        ) as UstrfEgoCompensatedMotionResolution.Available

        assertEquals(UstrfGridCoordinate(0, 9), assembled.packet.observations.single().coordinate)
        assertEquals(UstrfGridCoordinate(0, 9), motion.evidence.coordinate)
    }

    private fun input(frame: UstrfFrameStamp, blocked: UstrfGridCoordinate?) = UstrfSessionInput(
        frame = frame,
        health = UstrfHealth(UstrfPoseState.TRACKING, UstrfEvidenceState.VALID, UstrfEvidenceState.VALID, UstrfEvidenceState.VALID),
        perception = UstrfPerceptionAssembly.Available(
            UstrfPerceptionPacket(
                frame, frame.capturedAtNs, frame.capturedAtNs + 1_000_000L,
                (-2..2).flatMap { lateral -> (1..10).map { forward ->
                    val coordinate = UstrfGridCoordinate(lateral, forward)
                    UstrfRiskObservation(coordinate, if (coordinate == blocked) .95f else 0f, 1f, 0f, 0f, null, 0f, "fixture", frame.capturedAtNs + 1_000_000L)
                } }
            )
        ),
        route = UstrfRouteIntent("offline-five-meter", 0, 1f, frame.capturedAtNs + 1_000_000L)
    )
}
