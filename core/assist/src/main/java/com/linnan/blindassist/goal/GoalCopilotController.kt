package com.linnan.blindassist.goal

import com.linnan.blindassist.vision.FrameClockDomain
import com.linnan.blindassist.vision.FrameStamp

enum class GoalSemanticState {
    TARGET,
    UNCERTAIN,
    UNKNOWN,
    AUTHORIZED_ABSENT
}

/** Which source is allowed to make the accompanying semantic statement. */
enum class GoalSemanticAuthority {
    FRESH_SEMANTIC,
    CONTINUITY_ONLY,
    VERIFIED_ABSENCE,
    NONE
}

enum class CameraRelativeBearing {
    LEFT,
    FORWARD,
    RIGHT,
    UNKNOWN
}

enum class GoalObservationDeficit {
    NONE,
    NO_LOCALIZABLE_EVIDENCE,
    TARGET_NOT_PROPOSED,
    DECISIVE_IDENTITY_UNREADABLE,
    ASSOCIATION_AMBIGUOUS,
    REACQUIRE_CONFIRMATION_PENDING,
    METRIC_POSITION_UNKNOWN,
    FUNCTIONAL_TARGET_UNKNOWN,
    ENDPOINT_NOT_READY
}

data class GoalObservationEvidence(
    val sourceContractId: String,
    val sourceId: String,
    val goalId: String,
    val sessionId: String,
    val parentBindingId: String,
    val frame: FrameStamp,
    val availableAtNs: Long,
    val validUntilNs: Long,
    val availabilityClockDomain: FrameClockDomain,
    val semanticState: GoalSemanticState,
    val semanticAuthority: GoalSemanticAuthority,
    val bearing: CameraRelativeBearing,
    val deficit: GoalObservationDeficit
)

enum class GoalObservationDisposition {
    EVIDENCE_ABSENT,
    SOURCE_NOT_ADMITTED,
    IDENTITY_MISMATCH,
    CURRENT_FRAME_MISMATCH,
    CLOCK_DOMAIN_MISMATCH,
    EVIDENCE_NOT_AVAILABLE,
    EVIDENCE_STALE,
    AUTHORITY_STATE_MISMATCH,
    ADMITTED
}

enum class GoalContinuityState {
    UNBOUND,
    BOUND,
    COASTING,
    PROVISIONAL_REACQUIRE,
    LOST
}

enum class GoalCopilotAction {
    SWEEP_SEARCH,
    SCAN_LAST_LEFT,
    SCAN_LAST_FORWARD,
    SCAN_LAST_RIGHT,
    PAN_LEFT_TO_IDENTITY,
    PAN_RIGHT_TO_IDENTITY,
    APPROACH_FOR_IDENTITY,
    SIDESTEP_FOR_DISAMBIGUATION,
    HOLD_STEADY_CONFIRM,
    HOLD_STEADY_LOCALIZE,
    GUIDE_LEFT,
    GUIDE_FORWARD,
    GUIDE_RIGHT
}

data class GoalCopilotStep(
    val observationDisposition: GoalObservationDisposition,
    val semanticState: GoalSemanticState,
    val continuityState: GoalContinuityState,
    val action: GoalCopilotAction,
    val freshSemanticIdentity: Boolean,
    val frameId: Long,
    val handoffState: GoalHandoffState,
    val readinessDecision: GoalHandoffReadinessDecision? = null,
    val priorActionOutcome: GoalObservationActionOutcome? = null,
    val issuedActionReceipt: GoalObservationActionReceipt? = null
)

/**
 * Runtime landing of the L10 SC1W/SC2 authority split.
 *
 * Fresh semantics alone may acquire, reacquire, or navigate. Appearance/track
 * continuity can request a better view for a bounded interval but can never
 * silently become identity. Missing proposals are UNKNOWN + SEARCH, not proof
 * of absence. A lost or stale endpoint also revokes HANDOFF_READY.
 */
class GoalCopilotController(
    private val goalId: String,
    private val sessionId: String,
    private val parentBindingId: String,
    admittedSources: Set<GoalObservationSourceIdentity> = setOf(
        GoalObservationSourceIdentity(CONTRACT_ID, SEMANTIC_CARRIER_SOURCE_ID)
    ),
    private val coastLimitFrames: Int = DEFAULT_COAST_LIMIT_FRAMES,
    initialHandoffState: GoalHandoffState = GoalHandoffState.Inactive
) {
    private val admittedSources = admittedSources.toSet()
    private var continuityState = GoalContinuityState.UNBOUND
    private var lastBearing = CameraRelativeBearing.UNKNOWN
    private var coastAge = 0
    private var pendingReacquireFrameId: Long? = null
    private var pendingReacquireHits = 0
    private var everBound = false
    private var handoffState: GoalHandoffState = initialHandoffState
    private var pendingActionReceipt: GoalObservationActionReceipt? = null

    init {
        require(goalId.isNotBlank() && sessionId.isNotBlank() && parentBindingId.isNotBlank())
        require(coastLimitFrames >= 0)
        require(
            initialHandoffState == GoalHandoffState.Inactive ||
                initialHandoffState.goalAndSessionOrNull() == (goalId to sessionId)
        ) { "initial handoff state does not belong to this goal session" }
    }

    fun step(
        evidence: GoalObservationEvidence?,
        currentFrame: FrameStamp,
        decisionAtNs: Long,
        decisionClockDomain: FrameClockDomain,
        sessionTimestampMs: Long,
        endpointEvidence: GoalEndpointEvidence? = null,
        priorActionExecution: GoalObservationActionExecution? = null
    ): GoalCopilotStep {
        val disposition = admit(
            evidence = evidence,
            currentFrame = currentFrame,
            decisionAtNs = decisionAtNs,
            decisionClockDomain = decisionClockDomain
        )
        val priorActionOutcome = consumeActionOutcome(
            evidence = evidence,
            disposition = disposition,
            currentFrame = currentFrame,
            decisionAtNs = decisionAtNs,
            decisionClockDomain = decisionClockDomain,
            execution = priorActionExecution
        )
        if (evidence == null || disposition != GoalObservationDisposition.ADMITTED) {
            return lose(
                disposition = disposition,
                semanticState = GoalSemanticState.UNKNOWN,
                frame = currentFrame,
                decisionAtNs = decisionAtNs,
                decisionClockDomain = decisionClockDomain,
                priorActionOutcome = priorActionOutcome
            )
        }

        return when (evidence.semanticState) {
            GoalSemanticState.TARGET -> onFreshTarget(
                evidence = evidence,
                currentFrame = currentFrame,
                decisionAtNs = decisionAtNs,
                decisionClockDomain = decisionClockDomain,
                sessionTimestampMs = sessionTimestampMs,
                endpointEvidence = endpointEvidence,
                priorActionOutcome = priorActionOutcome
            )
            GoalSemanticState.UNCERTAIN -> onUncertain(
                evidence,
                currentFrame,
                decisionAtNs,
                decisionClockDomain,
                priorActionOutcome
            )
            GoalSemanticState.UNKNOWN,
            GoalSemanticState.AUTHORIZED_ABSENT -> lose(
                disposition = disposition,
                semanticState = evidence.semanticState,
                frame = currentFrame,
                decisionAtNs = decisionAtNs,
                decisionClockDomain = decisionClockDomain,
                priorActionOutcome = priorActionOutcome,
                deficit = evidence.deficit,
                bearing = evidence.bearing
            )
        }
    }

    fun confirmUser(
        timestamp: Long,
        modality: ConfirmationModality
    ): GoalHandoffTransition {
        val transition = GoalHandoffReducer.reduce(
            handoffState,
            GoalHandoffEvent.UserConfirmed(timestamp, modality)
        )
        if (transition is GoalHandoffTransition.Accepted) handoffState = transition.state
        return transition
    }

    fun currentHandoffState(): GoalHandoffState = handoffState

    private fun onFreshTarget(
        evidence: GoalObservationEvidence,
        currentFrame: FrameStamp,
        decisionAtNs: Long,
        decisionClockDomain: FrameClockDomain,
        sessionTimestampMs: Long,
        endpointEvidence: GoalEndpointEvidence?,
        priorActionOutcome: GoalObservationActionOutcome?
    ): GoalCopilotStep {
        if (evidence.bearing != CameraRelativeBearing.UNKNOWN) lastBearing = evidence.bearing
        if (continuityState == GoalContinuityState.LOST ||
            continuityState == GoalContinuityState.PROVISIONAL_REACQUIRE
        ) {
            val priorFrame = pendingReacquireFrameId
            if (priorFrame != null && currentFrame.frameId > priorFrame) {
                pendingReacquireHits += 1
            } else {
                pendingReacquireHits = 1
            }
            pendingReacquireFrameId = currentFrame.frameId
            if (pendingReacquireHits < REQUIRED_REACQUIRE_HITS) {
                continuityState = GoalContinuityState.PROVISIONAL_REACQUIRE
                revokeReadiness("REACQUIRE_CONFIRMATION_PENDING")
                return result(
                    disposition = GoalObservationDisposition.ADMITTED,
                    semanticState = GoalSemanticState.UNCERTAIN,
                    action = GoalCopilotAction.HOLD_STEADY_CONFIRM,
                    freshSemantic = true,
                    frame = currentFrame,
                    decisionAtNs = decisionAtNs,
                    decisionClockDomain = decisionClockDomain,
                    priorActionOutcome = priorActionOutcome,
                    deficit = GoalObservationDeficit.REACQUIRE_CONFIRMATION_PENDING,
                    bearing = evidence.bearing
                )
            }
        }

        everBound = true
        continuityState = GoalContinuityState.BOUND
        coastAge = 0
        pendingReacquireFrameId = null
        pendingReacquireHits = 0
        advanceFoundApproach()

        val readiness = endpointEvidence?.let {
            GoalHandoffReadinessGuard.evaluate(
                evidence = it,
                expectedGoalId = goalId,
                expectedSessionId = sessionId,
                expectedParentBindingId = parentBindingId,
                currentFrame = currentFrame,
                decisionAtNs = decisionAtNs,
                decisionClockDomain = decisionClockDomain
            )
        }
        when {
            handoffState is GoalHandoffState.Approach && readiness is GoalHandoffReadinessDecision.Ready -> {
                applyHandoff(
                    GoalHandoffEvent.HandoffReady(
                        timestamp = sessionTimestampMs,
                        reason = CURRENT_FRAME_ENDPOINT_READY,
                        readiness = readiness
                    )
                )
            }
            handoffState is GoalHandoffState.HandoffReady && readiness !is GoalHandoffReadinessDecision.Ready -> {
                revokeReadiness("CURRENT_ENDPOINT_EVIDENCE_NOT_READY")
            }
        }
        return result(
            disposition = GoalObservationDisposition.ADMITTED,
            semanticState = GoalSemanticState.TARGET,
            action = guidanceAction(evidence.bearing),
            freshSemantic = true,
            frame = currentFrame,
            readiness = readiness,
            decisionAtNs = decisionAtNs,
            decisionClockDomain = decisionClockDomain,
            priorActionOutcome = priorActionOutcome,
            deficit = evidence.deficit,
            bearing = evidence.bearing
        )
    }

    private fun onUncertain(
        evidence: GoalObservationEvidence,
        currentFrame: FrameStamp,
        decisionAtNs: Long,
        decisionClockDomain: FrameClockDomain,
        priorActionOutcome: GoalObservationActionOutcome?
    ): GoalCopilotStep {
        revokeReadiness("FRESH_TARGET_IDENTITY_NOT_AVAILABLE")
        val mayCoast = continuityState in setOf(
            GoalContinuityState.BOUND,
            GoalContinuityState.COASTING
        ) && coastAge < coastLimitFrames
        if (mayCoast) {
            continuityState = GoalContinuityState.COASTING
            coastAge += 1
            if (evidence.bearing != CameraRelativeBearing.UNKNOWN) lastBearing = evidence.bearing
            return result(
                disposition = GoalObservationDisposition.ADMITTED,
                semanticState = GoalSemanticState.UNCERTAIN,
                action = observationAction(evidence),
                freshSemantic = false,
                frame = currentFrame,
                decisionAtNs = decisionAtNs,
                decisionClockDomain = decisionClockDomain,
                priorActionOutcome = priorActionOutcome,
                deficit = evidence.deficit,
                bearing = evidence.bearing
            )
        }
        return lose(
            disposition = GoalObservationDisposition.ADMITTED,
            semanticState = GoalSemanticState.UNCERTAIN,
            frame = currentFrame,
            decisionAtNs = decisionAtNs,
            decisionClockDomain = decisionClockDomain,
            priorActionOutcome = priorActionOutcome,
            deficit = evidence.deficit,
            bearing = evidence.bearing
        )
    }

    private fun lose(
        disposition: GoalObservationDisposition,
        semanticState: GoalSemanticState,
        frame: FrameStamp,
        decisionAtNs: Long,
        decisionClockDomain: FrameClockDomain,
        priorActionOutcome: GoalObservationActionOutcome?,
        deficit: GoalObservationDeficit = GoalObservationDeficit.NO_LOCALIZABLE_EVIDENCE,
        bearing: CameraRelativeBearing = CameraRelativeBearing.UNKNOWN
    ): GoalCopilotStep {
        revokeReadiness("CURRENT_TARGET_EVIDENCE_LOST")
        continuityState = if (everBound) GoalContinuityState.LOST else GoalContinuityState.UNBOUND
        coastAge = 0
        pendingReacquireFrameId = null
        pendingReacquireHits = 0
        return result(
            disposition = disposition,
            semanticState = semanticState,
            action = searchAction(),
            freshSemantic = false,
            frame = frame,
            decisionAtNs = decisionAtNs,
            decisionClockDomain = decisionClockDomain,
            priorActionOutcome = priorActionOutcome,
            deficit = deficit,
            bearing = bearing
        )
    }

    private fun consumeActionOutcome(
        evidence: GoalObservationEvidence?,
        disposition: GoalObservationDisposition,
        currentFrame: FrameStamp,
        decisionAtNs: Long,
        decisionClockDomain: FrameClockDomain,
        execution: GoalObservationActionExecution?
    ): GoalObservationActionOutcome? {
        val receipt = pendingActionReceipt ?: return null
        pendingActionReceipt = null
        val unknownReason = when {
            receipt.goalId != goalId || receipt.sessionId != sessionId ||
                receipt.parentBindingId != parentBindingId -> "ACTION_RECEIPT_IDENTITY_MISMATCH"
            receipt.clockDomain != decisionClockDomain ||
                currentFrame.clockDomain != decisionClockDomain -> "ACTION_OUTCOME_CLOCK_MISMATCH"
            currentFrame.frameId <= receipt.issuedFrame.frameId -> "ACTION_OUTCOME_NOT_AFTER_ACTION"
            decisionAtNs > receipt.validUntilNs -> "ACTION_OUTCOME_RECEIPT_EXPIRED"
            execution == null -> "ACTION_EXECUTION_NOT_CONFIRMED"
            execution.receiptId != receipt.receiptId || execution.action != receipt.action ||
                execution.goalId != receipt.goalId || execution.sessionId != receipt.sessionId ||
                execution.parentBindingId != receipt.parentBindingId -> "ACTION_EXECUTION_MISMATCH"
            execution.clockDomain != receipt.clockDomain -> "ACTION_EXECUTION_CLOCK_MISMATCH"
            execution.executedAtNs < receipt.issuedAtNs ||
                execution.executedAtNs > decisionAtNs ||
                currentFrame.capturedAtNs < execution.executedAtNs -> "ACTION_EXECUTION_TIME_INVALID"
            evidence == null -> "ACTION_OUTCOME_EVIDENCE_ABSENT"
            disposition != GoalObservationDisposition.ADMITTED -> "ACTION_OUTCOME_EVIDENCE_NOT_ADMITTED"
            else -> null
        }
        if (unknownReason != null) {
            return GoalObservationActionOutcome(
                receipt = receipt,
                state = GoalObservationActionOutcomeState.UNKNOWN,
                observedFrameId = currentFrame.frameId,
                reason = unknownReason
            )
        }
        check(evidence != null)
        if (evidence.semanticState == GoalSemanticState.AUTHORIZED_ABSENT) {
            return GoalObservationActionOutcome(
                receipt = receipt,
                state = GoalObservationActionOutcomeState.CONTRADICTED,
                observedFrameId = currentFrame.frameId,
                reason = "VERIFIED_ABSENCE_AFTER_ACTION"
            )
        }
        val priorRank = semanticRank(receipt.priorSemanticState)
        val currentRank = semanticRank(evidence.semanticState)
        val bearingResolved = receipt.priorBearing == CameraRelativeBearing.UNKNOWN &&
            evidence.bearing != CameraRelativeBearing.UNKNOWN
        val deficitResolved = receipt.priorDeficit != GoalObservationDeficit.NONE &&
            evidence.deficit == GoalObservationDeficit.NONE
        return if (currentRank > priorRank || bearingResolved || deficitResolved) {
            GoalObservationActionOutcome(
                receipt = receipt,
                state = GoalObservationActionOutcomeState.IMPROVED,
                observedFrameId = currentFrame.frameId,
                reason = when {
                    currentRank > priorRank -> "SEMANTIC_EVIDENCE_RANK_INCREASED"
                    bearingResolved -> "CAMERA_RELATIVE_BEARING_RESOLVED"
                    else -> "OBSERVATION_DEFICIT_RESOLVED"
                }
            )
        } else {
            GoalObservationActionOutcome(
                receipt = receipt,
                state = GoalObservationActionOutcomeState.NO_GAIN,
                observedFrameId = currentFrame.frameId,
                reason = "COMPARABLE_EVIDENCE_DID_NOT_IMPROVE"
            )
        }
    }

    private fun semanticRank(state: GoalSemanticState): Int = when (state) {
        GoalSemanticState.TARGET -> 2
        GoalSemanticState.UNCERTAIN -> 1
        GoalSemanticState.UNKNOWN,
        GoalSemanticState.AUTHORIZED_ABSENT -> 0
    }

    private fun admit(
        evidence: GoalObservationEvidence?,
        currentFrame: FrameStamp,
        decisionAtNs: Long,
        decisionClockDomain: FrameClockDomain
    ): GoalObservationDisposition {
        if (evidence == null) return GoalObservationDisposition.EVIDENCE_ABSENT
        if (GoalObservationSourceIdentity(evidence.sourceContractId, evidence.sourceId) !in admittedSources) {
            return GoalObservationDisposition.SOURCE_NOT_ADMITTED
        }
        if (evidence.goalId != goalId || evidence.sessionId != sessionId ||
            evidence.parentBindingId != parentBindingId
        ) return GoalObservationDisposition.IDENTITY_MISMATCH
        if (evidence.frame != currentFrame) return GoalObservationDisposition.CURRENT_FRAME_MISMATCH
        if (evidence.availabilityClockDomain != decisionClockDomain ||
            evidence.frame.clockDomain != decisionClockDomain ||
            decisionClockDomain == FrameClockDomain.CAMERA_HARDWARE_UNMAPPED
        ) return GoalObservationDisposition.CLOCK_DOMAIN_MISMATCH
        if (evidence.availableAtNs < evidence.frame.capturedAtNs || decisionAtNs < evidence.availableAtNs) {
            return GoalObservationDisposition.EVIDENCE_NOT_AVAILABLE
        }
        if (evidence.validUntilNs < evidence.availableAtNs || decisionAtNs > evidence.validUntilNs) {
            return GoalObservationDisposition.EVIDENCE_STALE
        }
        if (!validAuthorityState(evidence.semanticState, evidence.semanticAuthority)) {
            return GoalObservationDisposition.AUTHORITY_STATE_MISMATCH
        }
        return GoalObservationDisposition.ADMITTED
    }

    private fun validAuthorityState(
        state: GoalSemanticState,
        authority: GoalSemanticAuthority
    ): Boolean = when (state) {
        GoalSemanticState.TARGET -> authority == GoalSemanticAuthority.FRESH_SEMANTIC
        GoalSemanticState.UNCERTAIN -> authority in setOf(
            GoalSemanticAuthority.FRESH_SEMANTIC,
            GoalSemanticAuthority.CONTINUITY_ONLY
        )
        GoalSemanticState.UNKNOWN -> authority == GoalSemanticAuthority.NONE
        GoalSemanticState.AUTHORIZED_ABSENT -> authority == GoalSemanticAuthority.VERIFIED_ABSENCE
    }

    private fun advanceFoundApproach() {
        when (handoffState) {
            GoalHandoffState.Inactive -> applyHandoff(GoalHandoffEvent.Found(goalId, sessionId))
            is GoalHandoffState.Found -> applyHandoff(GoalHandoffEvent.Approach)
            else -> Unit
        }
    }

    private fun revokeReadiness(reason: String) {
        if (handoffState is GoalHandoffState.HandoffReady) {
            applyHandoff(GoalHandoffEvent.ReadinessRevoked(reason))
        }
    }

    private fun applyHandoff(event: GoalHandoffEvent) {
        val transition = GoalHandoffReducer.reduce(handoffState, event)
        check(transition is GoalHandoffTransition.Accepted) {
            "internal handoff transition rejected: ${(transition as GoalHandoffTransition.Rejected).reason}"
        }
        handoffState = transition.state
    }

    private fun guidanceAction(bearing: CameraRelativeBearing): GoalCopilotAction = when (bearing) {
        CameraRelativeBearing.LEFT -> GoalCopilotAction.GUIDE_LEFT
        CameraRelativeBearing.RIGHT -> GoalCopilotAction.GUIDE_RIGHT
        CameraRelativeBearing.FORWARD -> GoalCopilotAction.GUIDE_FORWARD
        CameraRelativeBearing.UNKNOWN -> GoalCopilotAction.HOLD_STEADY_LOCALIZE
    }

    private fun observationAction(evidence: GoalObservationEvidence): GoalCopilotAction {
        if (evidence.deficit == GoalObservationDeficit.ASSOCIATION_AMBIGUOUS) {
            return GoalCopilotAction.SIDESTEP_FOR_DISAMBIGUATION
        }
        if (evidence.deficit == GoalObservationDeficit.REACQUIRE_CONFIRMATION_PENDING) {
            return GoalCopilotAction.HOLD_STEADY_CONFIRM
        }
        return when (evidence.bearing) {
            CameraRelativeBearing.LEFT -> GoalCopilotAction.PAN_LEFT_TO_IDENTITY
            CameraRelativeBearing.RIGHT -> GoalCopilotAction.PAN_RIGHT_TO_IDENTITY
            CameraRelativeBearing.FORWARD -> GoalCopilotAction.APPROACH_FOR_IDENTITY
            CameraRelativeBearing.UNKNOWN -> GoalCopilotAction.HOLD_STEADY_LOCALIZE
        }
    }

    private fun searchAction(): GoalCopilotAction = when (lastBearing) {
        CameraRelativeBearing.LEFT -> GoalCopilotAction.SCAN_LAST_LEFT
        CameraRelativeBearing.RIGHT -> GoalCopilotAction.SCAN_LAST_RIGHT
        CameraRelativeBearing.FORWARD -> GoalCopilotAction.SCAN_LAST_FORWARD
        CameraRelativeBearing.UNKNOWN -> GoalCopilotAction.SWEEP_SEARCH
    }

    private fun result(
        disposition: GoalObservationDisposition,
        semanticState: GoalSemanticState,
        action: GoalCopilotAction,
        freshSemantic: Boolean,
        frame: FrameStamp,
        readiness: GoalHandoffReadinessDecision? = null,
        decisionAtNs: Long,
        decisionClockDomain: FrameClockDomain,
        priorActionOutcome: GoalObservationActionOutcome?,
        deficit: GoalObservationDeficit,
        bearing: CameraRelativeBearing
    ): GoalCopilotStep {
        val repairedAction = repairAction(action, priorActionOutcome)
        val receipt = if (repairedAction.isObservationSeeking()) {
            GoalObservationActionReceipt(
                receiptId = "$sessionId:${frame.frameId}:$decisionAtNs:${repairedAction.name}",
                goalId = goalId,
                sessionId = sessionId,
                parentBindingId = parentBindingId,
                action = repairedAction,
                issuedFrame = frame,
                issuedAtNs = decisionAtNs,
                validUntilNs = decisionAtNs + ACTION_OUTCOME_TTL_NS,
                clockDomain = decisionClockDomain,
                priorSemanticState = semanticState,
                priorDeficit = deficit,
                priorBearing = bearing
            )
        } else {
            null
        }
        pendingActionReceipt = receipt
        return GoalCopilotStep(
            observationDisposition = disposition,
            semanticState = semanticState,
            continuityState = continuityState,
            action = repairedAction,
            freshSemanticIdentity = freshSemantic,
            frameId = frame.frameId,
            handoffState = handoffState,
            readinessDecision = readiness,
            priorActionOutcome = priorActionOutcome,
            issuedActionReceipt = receipt
        )
    }

    private fun repairAction(
        proposed: GoalCopilotAction,
        outcome: GoalObservationActionOutcome?
    ): GoalCopilotAction {
        if (outcome == null || outcome.state !in setOf(
                GoalObservationActionOutcomeState.NO_GAIN,
                GoalObservationActionOutcomeState.CONTRADICTED
            )
        ) return proposed
        if (outcome.state == GoalObservationActionOutcomeState.CONTRADICTED) {
            return if (outcome.receipt.action == GoalCopilotAction.SWEEP_SEARCH) {
                GoalCopilotAction.HOLD_STEADY_LOCALIZE
            } else {
                GoalCopilotAction.SWEEP_SEARCH
            }
        }
        if (proposed != outcome.receipt.action) return proposed
        return when (proposed) {
            GoalCopilotAction.SCAN_LAST_LEFT,
            GoalCopilotAction.SCAN_LAST_FORWARD,
            GoalCopilotAction.SCAN_LAST_RIGHT -> GoalCopilotAction.SWEEP_SEARCH
            GoalCopilotAction.PAN_LEFT_TO_IDENTITY,
            GoalCopilotAction.PAN_RIGHT_TO_IDENTITY,
            GoalCopilotAction.APPROACH_FOR_IDENTITY,
            GoalCopilotAction.HOLD_STEADY_CONFIRM -> GoalCopilotAction.SIDESTEP_FOR_DISAMBIGUATION
            GoalCopilotAction.SIDESTEP_FOR_DISAMBIGUATION -> GoalCopilotAction.APPROACH_FOR_IDENTITY
            GoalCopilotAction.HOLD_STEADY_LOCALIZE,
            GoalCopilotAction.SWEEP_SEARCH -> GoalCopilotAction.HOLD_STEADY_LOCALIZE
            GoalCopilotAction.GUIDE_LEFT,
            GoalCopilotAction.GUIDE_FORWARD,
            GoalCopilotAction.GUIDE_RIGHT -> proposed
        }
    }

    private fun GoalCopilotAction.isObservationSeeking(): Boolean = this !in setOf(
        GoalCopilotAction.GUIDE_LEFT,
        GoalCopilotAction.GUIDE_FORWARD,
        GoalCopilotAction.GUIDE_RIGHT
    )

    private fun GoalHandoffState.goalAndSessionOrNull(): Pair<String, String>? = when (this) {
        GoalHandoffState.Inactive -> null
        is GoalHandoffState.Found -> goalId to sessionId
        is GoalHandoffState.Approach -> goalId to sessionId
        is GoalHandoffState.HandoffReady -> goalId to sessionId
        is GoalHandoffState.CompletedByUser -> receipt.goalId to receipt.sessionId
    }

    companion object {
        const val CONTRACT_ID = "blindassist_goal_observation_input_v1"
        const val SEMANTIC_CARRIER_SOURCE_ID = "semantic_carrier_identity_continuity_v1"
        const val DEFAULT_COAST_LIMIT_FRAMES = 2
        const val REQUIRED_REACQUIRE_HITS = 2
        const val CURRENT_FRAME_ENDPOINT_READY = "CURRENT_FRAME_ENDPOINT_READY"
        const val ACTION_OUTCOME_TTL_NS = 2_000_000_000L
    }
}

data class GoalObservationSourceIdentity(
    val sourceContractId: String,
    val sourceId: String
)
