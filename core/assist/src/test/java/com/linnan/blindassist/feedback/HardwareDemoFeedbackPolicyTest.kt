package com.linnan.blindassist.feedback

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class HardwareDemoFeedbackPolicyTest {
    private val policy = HardwareDemoFeedbackPolicy()
    private fun update(t: Long, alert: Boolean = true, mm: Int? = 1500,
                       usable: Boolean = true, live: Boolean = true) =
        policy.update(t, live, usable, alert, mm)

    @Test fun firstAlertIsImmediateAndSteadyObstacleRemindsAtTwelveSeconds() {
        assertEquals(HardwareDemoFeedbackEvent.OBSTACLE, update(0))
        assertNull(update(3000))
        assertNull(update(11999))
        assertEquals(HardwareDemoFeedbackEvent.REMINDER, update(12000))
        assertNull(update(12001))
    }

    @Test fun approachingUsesLastAnnouncedDistanceAndOneSecondPacing() {
        update(0)
        assertNull(update(500, mm = 1100))
        assertEquals(HardwareDemoFeedbackEvent.CLOSER, update(1000, mm = 1200))
        assertNull(update(2000, mm = 1000))
        assertEquals(HardwareDemoFeedbackEvent.CLOSER, update(2100, mm = 900))
        assertNull(update(3100, mm = null))
        assertNull(update(3200, mm = -1))
    }

    @Test fun shortNonAlertGapDoesNotRepeatButFullReleaseStartsNewEvent() {
        update(0)
        assertNull(update(100, alert = false))
        assertNull(update(1099))
        assertNull(update(1200, alert = false))
        assertEquals(HardwareDemoFeedbackEvent.OBSTACLE, update(2200))
    }

    @Test fun unknownNeverSpeaksClearAndColdStartDoesNotReportLost() {
        assertNull(update(0, usable = false))
        assertNull(update(1000, usable = false))
        assertNull(update(2000, alert = false, mm = null))
        assertNull(update(3000, alert = false, mm = null))
    }

    @Test fun lossIsSingleAndRecoveryDoesNotHideAnObstacle() {
        update(0)
        assertEquals(HardwareDemoFeedbackEvent.CONNECTION_LOST, update(100, usable = false))
        assertNull(update(2000, usable = false))
        assertEquals(HardwareDemoFeedbackEvent.RECOVERED_OBSTACLE, update(3000))
        assertNull(update(3100))
        assertEquals(HardwareDemoFeedbackEvent.CONNECTION_LOST, update(4000, usable = false))
        assertEquals(HardwareDemoFeedbackEvent.RECOVERED, update(5000, alert = false))
    }

    @Test fun replayIsSilentAndCannotCarryLiveConnectionOrObstacleState() {
        update(0)
        assertNull(update(1000, live = false, usable = false))
        assertNull(update(2000, live = false))
        assertNull(update(3000, usable = false))
        assertEquals(HardwareDemoFeedbackEvent.OBSTACLE, update(4000))
    }

    @Test fun lifecycleResetAndClockRollbackStartFresh() {
        update(5000)
        policy.reset()
        assertNull(update(6000, usable = false))
        assertEquals(HardwareDemoFeedbackEvent.OBSTACLE, update(7000))
        assertEquals(HardwareDemoFeedbackEvent.OBSTACLE, update(0))
    }
}
