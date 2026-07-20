package com.linnan.blindassist.ustrf

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class UstrfSafetySessionPoseWarpTest {
    @Test
    fun sessionAppliesOnlyFrameBoundVerifiedOfflinePoseDeltaToRetainedStaticRisk() {
        val session = UstrfSafetySession()
        val first = frame(1L, 1_000L)
        val second = frame(2L, 101_000L)
        session.evaluate(input(first, listOf(observation(UstrfGridCoordinate(0, 3)))))

        val record = session.evaluate(
            input(
                second,
                emptyList(),
                UstrfVerifiedPoseDelta(first, second, forwardMeters = 1f, lateralMeters = 0f, yawRadians = 0f, verifiedForOfflineReplay = true)
            )
        )

        assertNotNull(record.field)
        assertTrue(record.field!!.cellAt(UstrfGridCoordinate(0, 2)).occupancy > .8f)
        assertEquals(UstrfStructuredAction.STOP, record.structuredOutput.action)
    }

    @Test
    fun invalidPoseDeltaResetsTheSessionFieldAndProducesFailClosedReceipt() {
        val session = UstrfSafetySession()
        val first = frame(1L, 1_000L)
        val second = frame(2L, 101_000L)
        session.evaluate(input(first, listOf(observation(UstrfGridCoordinate(0, 3)))))

        val record = session.evaluate(
            input(
                second,
                emptyList(),
                UstrfVerifiedPoseDelta(first, second, 1f, 0f, 0f, verifiedForOfflineReplay = false)
            )
        )

        assertNull(record.field)
        assertTrue(UstrfPerceptionAssemblyFailure.POSE_DELTA_INVALID in record.assemblyFailures)
        assertEquals(UstrfSafetyAction.STOP_AND_REASSESS, record.decision.action)
    }

    private fun frame(id: Long, timestampNs: Long) = UstrfFrameStamp(id, timestampNs, "offline-local")

    private fun input(frame: UstrfFrameStamp, observations: List<UstrfRiskObservation>, poseDelta: UstrfVerifiedPoseDelta? = null) =
        UstrfSessionInput(
            frame = frame,
            health = UstrfHealth(UstrfPoseState.TRACKING, UstrfEvidenceState.VALID, UstrfEvidenceState.VALID, UstrfEvidenceState.VALID),
            perception = UstrfPerceptionAssembly.Available(UstrfPerceptionPacket(frame, frame.capturedAtNs, frame.capturedAtNs + 1_000_000L, observations)),
            route = UstrfRouteIntent("offline-local", 0, 1f, frame.capturedAtNs + 1_000_000L),
            poseDelta = poseDelta
        )

    private fun observation(coordinate: UstrfGridCoordinate) = UstrfRiskObservation(
        coordinate, .9f, 1f, 0f, 0f, null, 0f, "offline-fixture", 2_000_000L
    )
}
