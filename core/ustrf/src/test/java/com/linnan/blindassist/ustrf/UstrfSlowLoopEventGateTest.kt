package com.linnan.blindassist.ustrf

import org.junit.Assert.assertEquals
import org.junit.Test

class UstrfSlowLoopEventGateTest {
    private val gate = UstrfSlowLoopEventGate()

    @Test
    fun periodicKeyframesAreBoundToFramesAndThrottled() {
        val first = gate.request(request(1L, 1_000L, UstrfSlowLoopTrigger.PERIODIC_KEYFRAME)) as UstrfSlowLoopEventGateResult.Accepted
        assertEquals("slow-camera-v1-1", first.event.eventId)
        assertEquals(2_000_001_000L, first.event.validUntilNs)

        assertEquals(
            UstrfSlowLoopEventGateResult.Suppressed(UstrfSlowLoopEventSuppression.PERIODIC_KEYFRAME_INTERVAL_NOT_REACHED),
            gate.request(request(2L, 1_500L, UstrfSlowLoopTrigger.PERIODIC_KEYFRAME))
        )
        assertEquals(
            UstrfSlowLoopEventGateResult.Accepted(UstrfSlowLoopEvent("slow-camera-v1-3", UstrfSlowLoopTrigger.PERIODIC_KEYFRAME, frame(3L, 2_000_001_000L), 2_000_001_000L, 4_000_001_000L)),
            gate.request(request(3L, 2_000_001_000L, UstrfSlowLoopTrigger.PERIODIC_KEYFRAME))
        )
    }

    @Test
    fun entropyRequiresThresholdAndOtherEventsRespectMinimumInterval() {
        assertEquals(
            UstrfSlowLoopEventGateResult.Suppressed(UstrfSlowLoopEventSuppression.HIGH_ENTROPY_BELOW_THRESHOLD),
            gate.request(request(1L, 1_000L, UstrfSlowLoopTrigger.HIGH_ENTROPY, .59f))
        )
        assertEquals(
            UstrfSlowLoopEventGateResult.Accepted(UstrfSlowLoopEvent("slow-camera-v1-2", UstrfSlowLoopTrigger.HIGH_ENTROPY, frame(2L, 2_000L), 2_000L, 2_000_002_000L)),
            gate.request(request(2L, 2_000L, UstrfSlowLoopTrigger.HIGH_ENTROPY, .60f))
        )
        assertEquals(
            UstrfSlowLoopEventGateResult.Suppressed(UstrfSlowLoopEventSuppression.MINIMUM_INTER_EVENT_INTERVAL),
            gate.request(request(3L, 2_100L, UstrfSlowLoopTrigger.USER_QUERY))
        )
    }

    @Test
    fun frameRollbackAndCoordinateChangesAreSuppressedWithoutResettingScheduler() {
        gate.request(request(3L, 3_000L, UstrfSlowLoopTrigger.USER_QUERY))
        assertEquals(
            UstrfSlowLoopEventGateResult.Suppressed(UstrfSlowLoopEventSuppression.SOURCE_FRAME_ROLLBACK),
            gate.request(request(3L, 4_000L, UstrfSlowLoopTrigger.GOAL_CHANGED))
        )
        assertEquals(
            UstrfSlowLoopEventGateResult.Suppressed(UstrfSlowLoopEventSuppression.COORDINATE_FRAME_CHANGED),
            gate.request(UstrfSlowLoopTriggerRequest(UstrfSlowLoopTrigger.GOAL_CHANGED, UstrfFrameStamp(4L, 4_000L, "other-camera"), 4_000L))
        )
        assertEquals(
            UstrfSlowLoopEventGateResult.Accepted(UstrfSlowLoopEvent("slow-camera-v1-5", UstrfSlowLoopTrigger.GOAL_CHANGED, frame(5L, 600_000_000L), 600_000_000L, 2_600_000_000L)),
            gate.request(request(5L, 600_000_000L, UstrfSlowLoopTrigger.GOAL_CHANGED))
        )
    }

    private fun request(
        id: Long,
        timeNs: Long,
        trigger: UstrfSlowLoopTrigger,
        uncertainty: Float? = null
    ) = UstrfSlowLoopTriggerRequest(trigger, frame(id, timeNs), timeNs, uncertainty)

    private fun frame(id: Long, timeNs: Long) = UstrfFrameStamp(id, timeNs, "camera-v1")
}
