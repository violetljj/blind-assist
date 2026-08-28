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
    val readinessDecision: GoalHandoffReadinessDecision? = null
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
        endpointEvidence: GoalEndpointEvidence? = null
    ): GoalCopilotStep {
        val disposition = admit(
            evidence = evidence,
            currentFrame = currentFrame,
            decisionAtNs = decisionAtNs,
            decisionClockDomain = decisionClockDomain
        )
        if (evidence == null || disposition != GoalObservationDisposition.ADMITTED) {
            return lose(
                disposition = disposition,
                semanticState = GoalSemanticState.UNKNOWN,
                frame = currentFrame
            )
        }

        return when (evidence.semanticState) {
            GoalSemanticState.TARGET -> onFreshTarget(
                evidence = evidence,
                currentFrame = currentFrame,
                decisionAtNs = decisionAtNs,
                decisionClockDomain = decisionClockDomain,
                sessionTimestampMs = sessionTimestampMs,
                endpointEvidence = endpointEvidence
            )
            GoalSemanticState.UNCERTAIN -> onUncertain(evidence, currentFrame)
            GoalSemanticState.UNKNOWN,
            GoalSemanticState.AUTHORIZED_ABSENT -> lose(
                disposition = disposition,
                semanticState = evidence.semanticState,
                frame = currentFrame
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
        endpointEvidence: GoalEndpointEvidence?
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
                    frame = currentFrame
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
            readiness = readiness
        )
    }

    private fun onUncertain(
        evidence: GoalObservationEvidence,
        currentFrame: FrameStamp
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
                frame = currentFrame
            )
        }
        return lose(
            disposition = GoalObservationDisposition.ADMITTED,
            semanticState = GoalSemanticState.UNCERTAIN,
            frame = currentFrame
        )
    }

    private fun lose(
        disposition: GoalObservationDisposition,
        semanticState: GoalSemanticState,
        frame: FrameStamp
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
            frame = frame
        )
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
        readiness: GoalHandoffReadinessDecision? = null
    ) = GoalCopilotStep(
        observationDisposition = disposition,
        semanticState = semanticState,
        continuityState = continuityState,
        action = action,
        freshSemanticIdentity = freshSemantic,
        frameId = frame.frameId,
        handoffState = handoffState,
        readinessDecision = readiness
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
    }
}

data class GoalObservationSourceIdentity(
    val sourceContractId: String,
    val sourceId: String
)
