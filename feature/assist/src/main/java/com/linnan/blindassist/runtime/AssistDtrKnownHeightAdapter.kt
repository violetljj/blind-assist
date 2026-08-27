package com.linnan.blindassist.runtime

import com.linnan.blindassist.alert.AlertProfile
import com.linnan.blindassist.alert.AssistScenario
import com.linnan.blindassist.risk.DtrKnownHeightRiskProducer
import com.linnan.blindassist.risk.DtrPrediction
import com.linnan.blindassist.risk.DtrSignal
import com.linnan.blindassist.risk.ProximityBand
import com.linnan.blindassist.risk.RiskDirection
import com.linnan.blindassist.risk.RiskEvidenceState
import com.linnan.blindassist.risk.RiskLevel
import com.linnan.blindassist.risk.RiskResult
import com.linnan.blindassist.risk.RiskScoreBreakdown
import com.linnan.blindassist.session.AssistDecisionKernel
import com.linnan.blindassist.session.AssistDtrEvidenceFrame
import com.linnan.blindassist.session.AssistFrameResult
import com.linnan.blindassist.session.AssistSessionCoordinator
import com.linnan.blindassist.vision.DetectorFrameResult

/** Isolated phone-camera adapter for the fixed known-height DTR route. */
internal class AssistDtrKnownHeightAdapter(
    private val coordinator: AssistSessionCoordinator,
    private val producer: DtrKnownHeightRiskProducer = DtrKnownHeightRiskProducer()
) {
    fun process(
        frame: DetectorFrameResult,
        profile: AlertProfile,
        scenario: AssistScenario,
        nowMs: Long,
        decisionAtNs: Long
    ): AssistFrameResult {
        val prediction = producer.process(
            detections = frame.detections,
            frameSize = frame.frameSize,
            cameraIntrinsics = frame.cameraIntrinsics,
            sourceFrame = frame.sourceFrame
        )
        return coordinator.processDtrEvidence(
            evidence = prediction.toEvidence(nowMs, frame.sourceFrame?.frameId),
            frameSize = frame.frameSize,
            profile = profile,
            scenario = scenario,
            metrics = frame.metrics,
            nowMs = nowMs,
            sourceFrame = frame.sourceFrame,
            decisionAtNs = decisionAtNs
        )
    }

    fun reset() = producer.reset()

    private fun DtrPrediction.toEvidence(nowMs: Long, frameId: Long?): AssistDtrEvidenceFrame {
        val actionable = eventActive
        val score = if (actionable) 1f else 0f
        val risk = RiskResult(
            level = if (actionable) RiskLevel.HIGH else RiskLevel.NONE,
            direction = if (actionable) RiskDirection.CENTER else RiskDirection.NONE,
            message = message(),
            proximity = if (actionable) ProximityBand.CRITICAL else ProximityBand.FAR,
            urgencyScore = score,
            riskScore = score,
            scoreBreakdown = RiskScoreBreakdown(
                total = score,
                fusionSummary = SOURCE_ID
            ),
            evidenceState = if (actionable) {
                RiskEvidenceState.SUPPORTED_TARGET_EVIDENCE
            } else {
                RiskEvidenceState.NO_SUPPORTED_TARGET_EVIDENCE
            }
        )
        return AssistDtrEvidenceFrame(
            sourceContractId = AssistDecisionKernel.DTR_EVIDENCE_INPUT_CONTRACT_ID,
            frameId = frameId?.let { "camera-frame-$it" } ?: "unstamped-frame-$nowMs",
            eventKey = eventKey,
            signal = signal,
            observedAtMs = nowMs,
            validUntilMs = saturatingAdd(nowMs, EVIDENCE_TTL_MS),
            risk = risk,
            evidenceCount = maxOf(1, metricTrackCount, personDetectionCount)
        )
    }

    private fun DtrPrediction.message(): String = when {
        signal == DtrSignal.ONSET -> "短时路线将与行人轨迹相交，减速并停步确认"
        signal == DtrSignal.HOLD && rawAlert == true -> "行人仍在短时路线风险事件中"
        signal == DtrSignal.HOLD -> "等待连续安全证据后清除路线风险"
        signal == DtrSignal.UNKNOWN && eventActive -> "路线风险仍保持，当前帧几何不可用"
        signal == DtrSignal.UNKNOWN -> "正在建立行人短时运动轨迹"
        else -> "短时路线暂未发现行人交叉风险"
    }

    private fun saturatingAdd(value: Long, increment: Long): Long =
        if (value > Long.MAX_VALUE - increment) Long.MAX_VALUE else value + increment

    private companion object {
        const val SOURCE_ID = "DTR_KNOWN_HEIGHT_ROUTE_INTERSECTION_V1"
        const val EVIDENCE_TTL_MS = 250L
    }
}
