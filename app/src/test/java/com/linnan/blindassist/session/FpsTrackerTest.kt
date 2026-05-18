package com.linnan.blindassist.session

import org.junit.Assert.assertEquals
import org.junit.Test

class FpsTrackerTest {
    private var nowMs = 0L

    @Test
    fun returnsZeroUntilFirstWindowCompletes() {
        val tracker = FpsTracker(clock = { nowMs })

        assertEquals(0f, tracker.onFrame(), 0.01f)
        nowMs = 500L
        assertEquals(0f, tracker.onFrame(), 0.01f)
    }

    @Test
    fun calculatesFpsWhenWindowCompletes() {
        val tracker = FpsTracker(clock = { nowMs })

        tracker.onFrame()
        nowMs = 500L
        tracker.onFrame()
        nowMs = 1000L

        assertEquals(3f, tracker.onFrame(), 0.01f)
    }

    @Test
    fun resetStartsANewWindow() {
        val tracker = FpsTracker(clock = { nowMs })
        nowMs = 1000L
        tracker.onFrame()
        tracker.reset()
        nowMs = 1500L

        assertEquals(0f, tracker.onFrame(), 0.01f)
    }
}
