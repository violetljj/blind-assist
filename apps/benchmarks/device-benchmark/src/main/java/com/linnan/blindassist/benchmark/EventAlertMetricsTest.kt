package com.linnan.blindassist.benchmark

import androidx.test.ext.junit.runners.AndroidJUnit4
import org.junit.Assert.assertEquals
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class EventAlertMetricsTest {
    @Test
    fun oneAlertWithinMultiFrameEventCountsAsOneHit() {
        val result = EventAlertMetrics.summarize(
            listOf(
                sample(eventId = "stairs", actualAlert = false, critical = true),
                sample(eventId = "stairs", actualAlert = true, critical = true),
                sample(eventId = "stairs", actualAlert = false, critical = false)
            )
        )

        assertEquals(1, result.eventCount)
        assertEquals(1, result.hitCount)
        assertEquals(1, result.criticalEventCount)
        assertEquals(1.0, result.recall, 0.0)
        assertEquals(0, result.criticalEventMissCount)
    }

    @Test
    fun alertOutsideExpectedWindowDoesNotHitEvent() {
        val result = EventAlertMetrics.summarize(
            listOf(
                sample(eventId = "obstacle", expectedShouldAlert = false, actualAlert = true),
                sample(eventId = "obstacle", actualAlert = false, critical = true)
            )
        )

        assertEquals(1, result.eventCount)
        assertEquals(0, result.hitCount)
        assertEquals(1, result.criticalEventCount)
        assertEquals(1, result.criticalEventMissCount)
    }

    @Test
    fun sequenceIdIsFallbackAndEventsRemainIndependent() {
        val result = EventAlertMetrics.summarize(
            listOf(
                sample(sequenceId = "stairs-sequence", actualAlert = true),
                sample(sequenceId = "stairs-sequence", actualAlert = false),
                sample(sequenceId = "obstacle-sequence", actualAlert = false, critical = true)
            )
        )

        assertEquals(2, result.eventCount)
        assertEquals(1, result.hitCount)
        assertEquals(1, result.criticalEventCount)
        assertEquals(0.5, result.recall, 0.0)
        assertEquals(1, result.criticalEventMissCount)
    }

    private fun sample(
        eventId: String? = null,
        sequenceId: String? = null,
        expectedShouldAlert: Boolean = true,
        critical: Boolean = false,
        actualAlert: Boolean = false
    ) = EventAlertSample(
        riskEventId = eventId,
        sequenceId = sequenceId,
        fallbackFrameId = "fixture-${eventId ?: sequenceId ?: "frame"}",
        expectedShouldAlert = expectedShouldAlert,
        expectedCritical = critical,
        actualAlert = actualAlert
    )
}
