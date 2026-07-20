package com.linnan.blindassist.benchmark

import androidx.test.ext.junit.runners.AndroidJUnit4
import org.junit.Assert.assertEquals
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class EventLifecycleMetricsTest {
    @Test
    fun separatesDeliveredRepeatFromSuppressedAttemptAndDetectsRegeneration() {
        val result = EventLifecycleMetrics.summarize(
            listOf(
                sample(frame = 0, actualAlert = true, runtimeEventId = "seg-1"),
                sample(frame = 1, suppressed = true, runtimeEventId = "seg-1"),
                sample(frame = 2, phase = "PASSED", actualAlert = true, runtimeEventId = "seg-2"),
                sample(frame = 3, phase = "PASSED", runtimeEventId = "seg-2")
            )
        )

        assertEquals(1, result.observedEventCount)
        assertEquals(2, result.deliveredAlertCount)
        assertEquals(1, result.deliveredRepeatedAlertCount)
        assertEquals(1, result.suppressedDuplicateAttemptCount)
        assertEquals(1, result.passedEventCount)
        assertEquals(0, result.clearedPassedEventCount)
        assertEquals(1, result.eventRegenerationCount)
        assertEquals(1.0, result.meanPostEventAlertLatencyFrames, 0.0)
        assertEquals(0.5, result.deliveredRepeatedAlertRate, 0.0)
        assertEquals(0.0, result.postEventClearanceRate, 0.0)
        assertEquals(1, result.falseAlertCount)
        assertEquals(400L, result.sequenceDurationMs)
        assertEquals(150.0, requireNotNull(result.falseAlertsPerMinute), 0.0)
    }

    @Test
    fun cleanPassedWindowHasFullClearanceWithoutInventingARepeat() {
        val result = EventLifecycleMetrics.summarize(
            listOf(
                sample(frame = 0, actualAlert = true, runtimeEventId = "seg-1"),
                sample(frame = 1, phase = "PASSED", runtimeEventId = "seg-1"),
                sample(frame = 2, phase = "PASSED", suppressed = true, runtimeEventId = "seg-1")
            )
        )

        assertEquals(1, result.deliveredAlertCount)
        assertEquals(0, result.deliveredRepeatedAlertCount)
        assertEquals(1, result.suppressedDuplicateAttemptCount)
        assertEquals(1, result.clearedPassedEventCount)
        assertEquals(0, result.eventRegenerationCount)
        assertEquals(0.0, result.meanPostEventAlertLatencyFrames, 0.0)
        assertEquals(1.0, result.postEventClearanceRate, 0.0)
        assertEquals(0, result.falseAlertCount)
        assertEquals(300L, result.sequenceDurationMs)
        assertEquals(0.0, requireNotNull(result.falseAlertsPerMinute), 0.0)
    }

    @Test
    fun countsEachAdditionalRuntimeIdAsARegeneration() {
        val result = EventLifecycleMetrics.summarize(
            listOf(
                sample(frame = 0, actualAlert = true, runtimeEventId = "seg-1"),
                sample(frame = 1, phase = "PASSED", actualAlert = true, runtimeEventId = "seg-2"),
                sample(frame = 2, phase = "PASSED", actualAlert = true, runtimeEventId = "seg-3")
            )
        )

        assertEquals(2, result.eventRegenerationCount)
        assertEquals(2, result.deliveredRepeatedAlertCount)
        assertEquals(2, result.falseAlertCount)
    }

    private fun sample(
        frame: Int,
        phase: String? = null,
        actualAlert: Boolean = false,
        suppressed: Boolean = false,
        runtimeEventId: String? = null
    ) = EventLifecycleSample(
        riskEventId = "stairs-1",
        sequenceId = "stairs-sequence",
        fallbackFrameId = "stairs-$frame",
        frameIndex = frame,
        expectedEventPhase = phase,
        expectedShouldAlert = phase != "PASSED",
        actualAlert = actualAlert,
        suppressedDuplicateAttempt = suppressed,
        runtimeEventId = runtimeEventId
    )
}
