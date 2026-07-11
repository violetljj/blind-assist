package com.linnan.blindassist.risk

import com.linnan.blindassist.feedback.FeedbackDecision
import com.linnan.blindassist.feedback.FeedbackReason
import com.linnan.blindassist.model.BoundingBox
import com.linnan.blindassist.model.Detection
import com.linnan.blindassist.model.DetectionSource
import com.linnan.blindassist.model.FrameSize
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class RiskEventTrackerTest {
    private val frame = FrameSize(1000, 1000)

    @Test
    fun firstApproachAlertsOnlyOnceUntilPassed() {
        val tracker = RiskEventTracker()
        val first = tracker.update(risk(ApproachTrend.APPROACHING))
        tracker.recordFeedback(first, FeedbackDecision(null, true, FeedbackReason.TRIGGERED))

        val repeated = tracker.update(risk(ApproachTrend.APPROACHING))

        assertEquals(RiskEventState.ALERTED, repeated.state)
        assertTrue(repeated.suppressesFeedback)
    }

    @Test
    fun threeRecedingFramesClearAndAllowNewAlert() {
        val tracker = RiskEventTracker()
        val first = tracker.update(risk(ApproachTrend.APPROACHING))
        tracker.recordFeedback(first, FeedbackDecision(null, true, FeedbackReason.TRIGGERED))
        repeat(3) { tracker.update(risk(ApproachTrend.RECEDING)) }

        val next = tracker.update(risk(ApproachTrend.APPROACHING))

        assertEquals(RiskEventState.APPROACHING, next.state)
        assertFalse(next.suppressesFeedback)
        assertTrue(next.eventId != first.eventId)
    }

    @Test
    fun oneMissingFrameDoesNotClearAlertedEvent() {
        val tracker = RiskEventTracker()
        val first = tracker.update(risk(ApproachTrend.APPROACHING))
        tracker.recordFeedback(first, FeedbackDecision(null, true, FeedbackReason.TRIGGERED))

        val held = tracker.update(RiskResult(RiskLevel.NONE, RiskDirection.NONE, "none"))

        assertEquals(RiskEventState.PASSED_OR_RECEDING, held.state)
        assertTrue(held.suppressesFeedback)
    }

    @Test
    fun unavailableFeedbackDoesNotConsumeEventAlert() {
        val tracker = RiskEventTracker()
        val first = tracker.update(risk(ApproachTrend.APPROACHING))
        tracker.recordFeedback(first, FeedbackDecision(null, false, FeedbackReason.FEEDBACK_UNAVAILABLE))

        val retry = tracker.update(risk(ApproachTrend.APPROACHING))

        assertEquals(RiskEventState.APPROACHING, retry.state)
        assertFalse(retry.suppressesFeedback)
    }

    @Test
    fun resetDropsPriorEvent() {
        val tracker = RiskEventTracker()
        val first = tracker.update(risk(ApproachTrend.APPROACHING))
        tracker.recordFeedback(first, FeedbackDecision(null, true, FeedbackReason.TRIGGERED))
        tracker.reset()

        val afterReset = tracker.update(risk(ApproachTrend.APPROACHING))

        assertFalse(afterReset.suppressesFeedback)
        assertTrue(afterReset.eventId != first.eventId)
    }

    private fun risk(trend: ApproachTrend): RiskResult = RiskResult(
        level = RiskLevel.MEDIUM,
        direction = RiskDirection.CENTER,
        message = "risk",
        sourceDetection = Detection(
            classId = 10_020,
            label = "stairs",
            confidence = 1f,
            boundingBox = BoundingBox(400f, 400f, 600f, 800f),
            frameSize = frame,
            source = DetectionSource.SEGMENTATION
        ),
        proximity = ProximityBand.MID,
        approachTrend = trend
    )
}
