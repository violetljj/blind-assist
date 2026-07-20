package com.linnan.blindassist.ustrf

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class UstrfSafetySessionAdmissionTest {
    @Test
    fun validPerceptionCannotUpdateFieldWhenAnyFastLoopHealthReceiptIsUnavailable() {
        val expected = listOf(
            UstrfHealth(UstrfPoseState.LOST, UstrfEvidenceState.VALID, UstrfEvidenceState.VALID, UstrfEvidenceState.VALID) to UstrfPerceptionAssemblyFailure.POSE_UNAVAILABLE,
            UstrfHealth(UstrfPoseState.TRACKING, UstrfEvidenceState.MISSING, UstrfEvidenceState.VALID, UstrfEvidenceState.VALID) to UstrfPerceptionAssemblyFailure.CAPTURE_UNAVAILABLE,
            UstrfHealth(UstrfPoseState.TRACKING, UstrfEvidenceState.VALID, UstrfEvidenceState.DEGRADED, UstrfEvidenceState.VALID) to UstrfPerceptionAssemblyFailure.GEOMETRY_UNAVAILABLE,
            UstrfHealth(UstrfPoseState.TRACKING, UstrfEvidenceState.VALID, UstrfEvidenceState.VALID, UstrfEvidenceState.MISSING) to UstrfPerceptionAssemblyFailure.MOTION_EVIDENCE_UNAVAILABLE
        )
        expected.forEachIndexed { index, (health, failure) ->
            val frame = UstrfFrameStamp(index.toLong(), 1_000L + index, "camera-v1")
            val packet = UstrfPerceptionPacket(frame, frame.capturedAtNs, 2_000L, clearObservations(2_000L))
            val record = UstrfSafetySession().evaluate(
                UstrfSessionInput(
                    frame = frame,
                    health = health,
                    perception = UstrfPerceptionAssembly.Available(packet),
                    route = UstrfRouteIntent("camera-v1", 0, 1f, 2_000L)
                )
            )
            assertNull("$failure must prevent a partial risk-field write", record.field)
            assertTrue(record.assemblyFailures.contains(failure))
            assertEquals(UstrfSafetyAction.STOP_AND_REASSESS, record.decision.action)
            assertTrue(UstrfSafetyReason.PERCEPTION_ASSEMBLY_UNAVAILABLE in record.decision.reasons)
        }
    }

    @Test
    fun rejectedFrameResetsPriorFieldBeforeAHealthyRecoveryFrame() {
        val session = UstrfSafetySession()
        val first = input(1L, UstrfHealth(UstrfPoseState.TRACKING, UstrfEvidenceState.VALID, UstrfEvidenceState.VALID, UstrfEvidenceState.VALID), occupied = true)
        assertEquals(1f, session.evaluate(first).field!!.cellAt(UstrfGridCoordinate(0, 1)).occupancy)

        val rejected = input(2L, UstrfHealth(UstrfPoseState.LOST, UstrfEvidenceState.VALID, UstrfEvidenceState.VALID, UstrfEvidenceState.VALID))
        assertNull(session.evaluate(rejected).field)

        val recovered = session.evaluate(input(3L, UstrfHealth(UstrfPoseState.TRACKING, UstrfEvidenceState.VALID, UstrfEvidenceState.VALID, UstrfEvidenceState.VALID)))
        assertEquals(0f, recovered.field!!.cellAt(UstrfGridCoordinate(0, 1)).occupancy)
    }

    private fun input(frameId: Long, health: UstrfHealth, occupied: Boolean = false): UstrfSessionInput {
        val capturedAtNs = frameId * 1_000L
        val frame = UstrfFrameStamp(frameId, capturedAtNs, "camera-v1")
        val observations = clearObservations(capturedAtNs + 500L).map { observation ->
            if (occupied && observation.coordinate == UstrfGridCoordinate(0, 1)) observation.copy(occupancy = 1f) else observation
        }
        return UstrfSessionInput(
            frame = frame,
            health = health,
            perception = UstrfPerceptionAssembly.Available(UstrfPerceptionPacket(frame, capturedAtNs, capturedAtNs + 500L, observations)),
            route = UstrfRouteIntent("camera-v1", 0, 1f, capturedAtNs + 500L)
        )
    }

    private fun clearObservations(validUntilNs: Long): List<UstrfRiskObservation> =
        (-1..1).flatMap { lateral -> (1..4).map { forward ->
            UstrfRiskObservation(
                coordinate = UstrfGridCoordinate(lateral, forward),
                occupancy = 0f,
                traversability = 1f,
                dropRisk = 0f,
                headRisk = 0f,
                dynamicTtcMs = null,
                uncertainty = 0f,
                source = "fixture",
                validUntilNs = validUntilNs
            )
        } }
}
