package com.linnan.blindassist.session

import com.linnan.blindassist.alert.AlertProfile
import com.linnan.blindassist.alert.AssistScenario
import com.linnan.blindassist.feedback.FeedbackGateway
import com.linnan.blindassist.vision.DetectorFrameResult

class AssistSessionCoordinator(
    private val assistEngine: AssistEngine = AssistEngine(),
    private val feedbackGateway: FeedbackGateway,
    private val fpsTracker: FpsTracker = FpsTracker()
) {
    fun startSession(nowMs: Long = System.currentTimeMillis()) {
        feedbackGateway.resetSession()
        fpsTracker.reset()
        assistEngine.startSession(nowMs)
    }

    fun reset() {
        fpsTracker.reset()
        assistEngine.reset()
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
        val feedbackDecision = feedbackGateway.notify(evaluation.stableRisk, profile, scenario)
        return assistEngine.completeFeedback(evaluation, feedbackDecision)
    }
}
