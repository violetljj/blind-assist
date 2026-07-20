package com.linnan.blindassist.session

import com.linnan.blindassist.alert.AlertProfile
import com.linnan.blindassist.alert.AssistScenario
import com.linnan.blindassist.feedback.FeedbackGateway
import com.linnan.blindassist.feedback.FeedbackDecision
import com.linnan.blindassist.feedback.FeedbackReason
import com.linnan.blindassist.model.Detection
import com.linnan.blindassist.model.DetectionSource
import com.linnan.blindassist.risk.RiskEventTracker
import com.linnan.blindassist.risk.RiskDirection
import com.linnan.blindassist.risk.RiskResult
import com.linnan.blindassist.vision.DetectorFrameResult
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min

class AssistSessionCoordinator(
    private val assistEngine: AssistEngine = AssistEngine(),
    private val feedbackGateway: FeedbackGateway,
    private val fpsTracker: FpsTracker = FpsTracker(),
    private val riskEventTracker: RiskEventTracker = RiskEventTracker()
) {
    private val lowConfidenceSidePersonConfirmation = LowConfidenceSidePersonConfirmation()

    fun startSession(nowMs: Long = System.currentTimeMillis()) {
        feedbackGateway.resetSession()
        fpsTracker.reset()
        assistEngine.startSession(nowMs)
        riskEventTracker.reset()
        lowConfidenceSidePersonConfirmation.reset()
    }

    fun reset() {
        fpsTracker.reset()
        assistEngine.reset()
        riskEventTracker.reset()
        lowConfidenceSidePersonConfirmation.reset()
    }

    fun sessionSummary(): SessionSummary {
        return assistEngine.sessionSummary()
    }

    fun processFrame(
        detectorFrame: DetectorFrameResult,
        profile: AlertProfile,
        scenario: AssistScenario,
        nowMs: Long = System.currentTimeMillis()
    ): AssistFrameResult {
        val fps = fpsTracker.onFrame()
        val evaluation = assistEngine.evaluate(
            detections = detectorFrame.detections,
            frameSize = detectorFrame.frameSize,
            profile = profile,
            scenario = scenario,
            metrics = detectorFrame.metrics.copy(fps = fps),
            nowMs = nowMs
        )
        val event = riskEventTracker.update(evaluation.stableRisk, nowMs)
        val eventEvaluation = evaluation.copy(riskEvent = event)
        val feedbackDecision = if (event.suppressesFeedback) {
            FeedbackDecision(null, triggered = false, reason = FeedbackReason.EVENT_ALREADY_ALERTED)
        } else if (!lowConfidenceSidePersonConfirmation.isConfirmed(eventEvaluation.stableRisk)) {
            FeedbackDecision(null, triggered = false, reason = FeedbackReason.UNSTABLE_RISK)
        } else {
            feedbackGateway.notify(eventEvaluation.stableRisk, profile, scenario)
        }
        val completedEvent = riskEventTracker.recordFeedback(event, feedbackDecision)
        return assistEngine.completeFeedback(eventEvaluation.copy(riskEvent = completedEvent), feedbackDecision)
    }
}

/**
 * Keeps a low-confidence, side-lane YOLO person out of feedback until the same target
 * appears in two consecutive frames. Center persons and confidence >= 0.50 retain the
 * existing response path.
 */
/**
 * Applies the same consecutive-frame gate for production feedback and benchmark simulation.
 */
class LowConfidenceSidePersonConfirmation {
    private var previous: Detection? = null

    fun isConfirmed(risk: RiskResult): Boolean {
        val current = risk.sourceDetection ?: return resetAndAllow()
        if (!requiresConfirmation(risk, current)) {
            return resetAndAllow()
        }
        val matchesPrevious = previous?.let { matches(it, current) } == true
        previous = current
        return matchesPrevious
    }

    fun reset() {
        previous = null
    }

    private fun resetAndAllow(): Boolean {
        previous = null
        return true
    }

    private fun requiresConfirmation(risk: RiskResult, detection: Detection?): Boolean {
        return detection?.source == DetectionSource.OBJECT_DETECTOR &&
            detection.label == PERSON_LABEL &&
            detection.confidence < MIN_CONFIDENCE_WITHOUT_CONFIRMATION &&
            risk.direction != RiskDirection.CENTER
    }

    private fun matches(previous: Detection, current: Detection): Boolean {
        if (previous.label != current.label || previous.source != current.source) return false
        val previousCenter = previous.boundingBox.centerX / previous.frameSize.width.toFloat()
        val currentCenter = current.boundingBox.centerX / current.frameSize.width.toFloat()
        return iou(previous, current) >= MIN_IOU_FOR_SAME_PERSON ||
            abs(previousCenter - currentCenter) <= MAX_CENTER_DELTA_FOR_SAME_PERSON
    }

    private fun iou(first: Detection, second: Detection): Float {
        val firstBox = first.boundingBox
        val secondBox = second.boundingBox
        val left = max(firstBox.left, secondBox.left)
        val top = max(firstBox.top, secondBox.top)
        val right = min(firstBox.right, secondBox.right)
        val bottom = min(firstBox.bottom, secondBox.bottom)
        val intersection = max(0f, right - left) * max(0f, bottom - top)
        val union = firstBox.width * firstBox.height + secondBox.width * secondBox.height - intersection
        return if (union <= 0f) 0f else intersection / union
    }

    private companion object {
        const val PERSON_LABEL = "person"
        const val MIN_CONFIDENCE_WITHOUT_CONFIRMATION = 0.50f
        const val MIN_IOU_FOR_SAME_PERSON = 0.25f
        const val MAX_CENTER_DELTA_FOR_SAME_PERSON = 0.12f
    }
}
