package com.linnan.blindassist.goal

import com.linnan.blindassist.vision.FrameClockDomain
import com.linnan.blindassist.vision.FrameStamp

enum class GoalObservationActionOutcomeState {
    IMPROVED,
    NO_GAIN,
    CONTRADICTED,
    UNKNOWN
}

/**
 * A causal receipt for one observation-seeking instruction.
 *
 * It deliberately records the pre-action semantic state. The following frame
 * can therefore say whether the issued action improved evidence; a generic
 * image-quality proxy cannot claim that authority.
 */
data class GoalObservationActionReceipt(
    val receiptId: String,
    val goalId: String,
    val sessionId: String,
    val parentBindingId: String,
    val action: GoalCopilotAction,
    val issuedFrame: FrameStamp,
    val issuedAtNs: Long,
    val validUntilNs: Long,
    val clockDomain: FrameClockDomain,
    val priorSemanticState: GoalSemanticState,
    val priorDeficit: GoalObservationDeficit,
    val priorBearing: CameraRelativeBearing
)

/** A caller acknowledgement that the previously issued instruction happened. */
data class GoalObservationActionExecution(
    val receiptId: String,
    val goalId: String,
    val sessionId: String,
    val parentBindingId: String,
    val action: GoalCopilotAction,
    val executedAtNs: Long,
    val clockDomain: FrameClockDomain
)

data class GoalObservationActionOutcome(
    val receipt: GoalObservationActionReceipt,
    val state: GoalObservationActionOutcomeState,
    val observedFrameId: Long,
    val reason: String
)
