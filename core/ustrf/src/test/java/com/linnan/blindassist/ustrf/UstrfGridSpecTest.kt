package com.linnan.blindassist.ustrf

import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Test

class UstrfGridSpecTest {
    @Test
    fun fiveCandidateEnvelopeRequiresThreeCellsOfLateralCoverage() {
        assertThrows(IllegalArgumentException::class.java) {
            UstrfGridSpec(.5f, 2, 10, 1, listOf(-2, -1, 0, 1, 2))
        }

        assertEquals(3, UstrfGridSpec.DOCUMENT_FIVE_METER.halfWidthCells)
    }

    @Test
    fun fieldBuilderRejectsPacketsFromAnotherGridContract() {
        val frame = UstrfFrameStamp(1L, 1_000L, "grid")
        val packet = UstrfPerceptionPacket(
            frame,
            1_000L,
            2_000L,
            emptyList(),
            UstrfGridSpec.DOCUMENT_FIVE_METER
        )

        assertThrows(IllegalArgumentException::class.java) { UstrfRiskFieldBuilder().update(packet) }
    }

    @Test
    fun structuredCorridorWidthComesFromTheSharedBodyProfile() {
        val spec = UstrfGridSpec.DOCUMENT_FIVE_METER.copy(bodyWidthMeters = .72f)
        val decision = UstrfSafetyDecision(
            UstrfSafetyAction.SLOW_DOWN,
            0f,
            1f,
            2_000L,
            setOf(UstrfSafetyReason.SHADOW_ONLY),
            0
        )

        val output = UstrfStructuredSafetyOutputMapper(spec).map(
            decision,
            UstrfCorridorCandidate(0, true, 0f, false)
        )

        assertEquals(.72f, output.corridorWidthMeters!!, .0001f)
    }

    @Test
    fun safetySessionReportsGridMismatchAsASeparateFailClosedReason() {
        val frame = UstrfFrameStamp(1L, 1_000L, "grid")
        val packet = UstrfPerceptionPacket(
            frame, 1_000L, 2_000L, emptyList(), UstrfGridSpec.DOCUMENT_FIVE_METER
        )

        val record = UstrfSafetySession().evaluate(
            UstrfSessionInput(
                frame,
                UstrfHealth(
                    UstrfPoseState.TRACKING,
                    UstrfEvidenceState.VALID,
                    UstrfEvidenceState.VALID,
                    UstrfEvidenceState.VALID
                ),
                UstrfPerceptionAssembly.Available(packet),
                UstrfRouteIntent("grid", 0, 1f, 2_000L)
            )
        )

        assertTrue(UstrfPerceptionAssemblyFailure.GRID_SPEC_MISMATCH in record.assemblyFailures)
        assertTrue(UstrfSafetyReason.GRID_SPEC_MISMATCH in record.decision.reasons)
        assertEquals(UstrfSafetyAction.STOP_AND_REASSESS, record.decision.action)
    }
}
