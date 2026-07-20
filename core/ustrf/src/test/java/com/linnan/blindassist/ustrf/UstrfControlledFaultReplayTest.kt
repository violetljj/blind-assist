package com.linnan.blindassist.ustrf

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class UstrfControlledFaultReplayTest {
    @Test
    fun injectedReceiptFaultsAreDeterministicAndNeverPublishAPartialField() {
        val faults = UstrfFastLoopFault.entries
        fun run() = UstrfControlledFaultReplay().run(
            listOf(UstrfFaultScenario(input(1L), emptySet())) + faults.mapIndexed { index, fault ->
                UstrfFaultScenario(input((index + 2).toLong()), setOf(fault))
            }
        )

        val first = run()
        val second = run()
        assertEquals(UstrfSessionTraceDigest.canonicalText(first.map { it.result }), UstrfSessionTraceDigest.canonicalText(second.map { it.result }))
        assertTrue(first.first().result.field != null)
        first.drop(1).forEach { record ->
            assertNull("${record.injectedFaults} must not publish a partial field", record.result.field)
            assertEquals(UstrfSafetyAction.STOP_AND_REASSESS, record.result.decision.action)
            assertTrue(UstrfSafetyReason.PERCEPTION_ASSEMBLY_UNAVAILABLE in record.result.decision.reasons)
        }
    }

    private fun input(frameId: Long): UstrfSessionInput {
        val capturedAtNs = frameId * 1_000L
        val frame = UstrfFrameStamp(frameId, capturedAtNs, "camera-v1")
        return UstrfSessionInput(
            frame = frame,
            health = UstrfHealth(UstrfPoseState.TRACKING, UstrfEvidenceState.VALID, UstrfEvidenceState.VALID, UstrfEvidenceState.VALID),
            perception = UstrfPerceptionAssembly.Available(
                UstrfPerceptionPacket(frame, capturedAtNs, capturedAtNs + 500L, clearObservations(capturedAtNs + 500L))
            ),
            route = UstrfRouteIntent("camera-v1", 0, 1f, capturedAtNs + 500L)
        )
    }

    private fun clearObservations(validUntilNs: Long): List<UstrfRiskObservation> =
        (-1..1).flatMap { lateral -> (1..4).map { forward ->
            UstrfRiskObservation(UstrfGridCoordinate(lateral, forward), 0f, 1f, 0f, 0f, null, 0f, "fixture", validUntilNs)
        } }
}
