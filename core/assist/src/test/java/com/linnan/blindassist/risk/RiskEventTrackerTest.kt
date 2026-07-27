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
    fun postPassReappearanceIsSuppressedBeforeSameAnchorCanAlertAgain() {
        val tracker = RiskEventTracker()
        val first = tracker.update(risk(ApproachTrend.APPROACHING), nowMs = 0L)
        tracker.recordFeedback(first, FeedbackDecision(null, true, FeedbackReason.TRIGGERED))
        repeat(3) { index -> tracker.update(risk(ApproachTrend.RECEDING), nowMs = (index + 1) * 100L) }

        val rebound = tracker.update(risk(ApproachTrend.APPROACHING), nowMs = 350L)

        assertEquals(first.eventId, rebound.eventId)
        assertEquals(RiskEventState.PASSED_OR_RECEDING, rebound.state)
        assertFalse(rebound.active)
        assertTrue(rebound.suppressesFeedback)

        val next = tracker.update(risk(ApproachTrend.APPROACHING), nowMs = 1_301L)

        assertEquals(RiskEventState.APPROACHING, next.state)
        assertFalse(next.suppressesFeedback)
        assertTrue(next.eventId != first.eventId)
    }

    @Test
    fun unalertedEventRestoresIdentityAfterShortMissingGap() {
        val tracker = RiskEventTracker()
        val first = tracker.update(risk(ApproachTrend.APPROACHING), nowMs = 0L)
        repeat(3) { index ->
            tracker.update(
                RiskResult(RiskLevel.NONE, RiskDirection.NONE, "none"),
                nowMs = (index + 1) * 100L
            )
        }

        val restored = tracker.update(risk(ApproachTrend.APPROACHING), nowMs = 350L)

        assertEquals(first.eventId, restored.eventId)
        assertTrue(restored.active)
        assertFalse(restored.suppressesFeedback)
    }

    @Test
    fun unalertedTombstoneExpiresBeforeNewIdentityIsAllocated() {
        val tracker = RiskEventTracker()
        val first = tracker.update(risk(ApproachTrend.APPROACHING), nowMs = 0L)
        repeat(3) { index ->
            tracker.update(
                RiskResult(RiskLevel.NONE, RiskDirection.NONE, "none"),
                nowMs = (index + 1) * 100L
            )
        }

        val next = tracker.update(risk(ApproachTrend.APPROACHING), nowMs = 1_301L)

        assertTrue(next.active)
        assertTrue(next.eventId != first.eventId)
    }

    @Test
    fun postPassReappearanceWithCenterShiftInsideCorridorIsStillSuppressed() {
        val tracker = RiskEventTracker()
        val first = tracker.update(risk(ApproachTrend.APPROACHING), nowMs = 0L)
        tracker.recordFeedback(first, FeedbackDecision(null, true, FeedbackReason.TRIGGERED))
        repeat(3) { index -> tracker.update(risk(ApproachTrend.RECEDING), nowMs = (index + 1) * 100L) }

        val rebound = tracker.update(risk(ApproachTrend.APPROACHING, centerX = 850f), nowMs = 350L)

        assertEquals(first.eventId, rebound.eventId)
        assertEquals(RiskEventState.PASSED_OR_RECEDING, rebound.state)
        assertTrue(rebound.suppressesFeedback)
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

    @Test
    fun objectDetectorPersonTrackingIsExplicitlyOptIn() {
        val defaultTracker = RiskEventTracker()
        assertEquals(null, defaultTracker.update(personRisk(ApproachTrend.APPROACHING)).eventId)

        val candidateTracker = RiskEventTracker(
            RiskEventTrackerConfig(trackCenterObjectDetectorPerson = true)
        )
        val first = candidateTracker.update(personRisk(ApproachTrend.APPROACHING))
        candidateTracker.recordFeedback(first, FeedbackDecision(null, true, FeedbackReason.TRIGGERED))
        val repeated = candidateTracker.update(personRisk(ApproachTrend.APPROACHING))

        assertEquals(RiskEventState.ALERTED, repeated.state)
        assertTrue(repeated.suppressesFeedback)
    }

    private fun risk(trend: ApproachTrend, centerX: Float = 500f): RiskResult = RiskResult(
        level = RiskLevel.MEDIUM,
        direction = RiskDirection.CENTER,
        message = "risk",
        sourceDetection = Detection(
            classId = 10_020,
            label = "stairs",
            confidence = 1f,
            boundingBox = BoundingBox(centerX - 100f, 400f, centerX + 100f, 800f),
            frameSize = frame,
            source = DetectionSource.SEGMENTATION
        ),
        proximity = ProximityBand.MID,
        approachTrend = trend
    )

    private fun personRisk(trend: ApproachTrend): RiskResult = risk(trend).copy(
        sourceDetection = risk(trend).sourceDetection?.copy(
            classId = 0,
            label = "person",
            source = DetectionSource.OBJECT_DETECTOR
        )
    )
}
