package com.linnan.blindassist.goal

import com.linnan.blindassist.vision.FrameClockDomain
import com.linnan.blindassist.vision.FrameStamp
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class GoalCopilotControllerTest {
    @Test
    fun freshIdentityGuidesApproachesAndCompletesOnlyAfterEndpointAndUser() {
        val controller = controller()
        val found = step(controller, frame(1L), CameraRelativeBearing.LEFT)
        assertEquals(GoalContinuityState.BOUND, found.continuityState)
        assertEquals(GoalCopilotAction.GUIDE_LEFT, found.action)
        assertTrue(found.handoffState is GoalHandoffState.Found)

        val approach = step(controller, frame(2L), CameraRelativeBearing.FORWARD)
        assertTrue(approach.handoffState is GoalHandoffState.Approach)

        val readyFrame = frame(3L)
        val ready = step(
            controller,
            readyFrame,
            CameraRelativeBearing.FORWARD,
            endpointEvidence = endpointEvidence(readyFrame)
        )
        assertTrue(ready.handoffState is GoalHandoffState.HandoffReady)
        assertTrue(ready.readinessDecision is GoalHandoffReadinessDecision.Ready)

        val completed = controller.confirmUser(1_030L, ConfirmationModality.BUTTON)
        assertTrue(completed is GoalHandoffTransition.Accepted)
        assertTrue(completed.state is GoalHandoffState.CompletedByUser)
    }

    @Test
    fun targetLossRevokesReadyAndRequiresTwoFreshHitsToReacquire() {
        val controller = controller()
        step(controller, frame(1L), CameraRelativeBearing.FORWARD)
        step(controller, frame(2L), CameraRelativeBearing.FORWARD)
        val readyFrame = frame(3L)
        step(
            controller,
            readyFrame,
            CameraRelativeBearing.FORWARD,
            endpointEvidence = endpointEvidence(readyFrame)
        )

        val lostFrame = frame(4L)
        val lost = controller.step(
            evidence = null,
            currentFrame = lostFrame,
            decisionAtNs = decisionAt(lostFrame),
            decisionClockDomain = CLOCK,
            sessionTimestampMs = 1_040L
        )
        assertEquals(GoalContinuityState.LOST, lost.continuityState)
        assertEquals(GoalCopilotAction.SCAN_LAST_FORWARD, lost.action)
        assertTrue(lost.handoffState is GoalHandoffState.Approach)
        assertTrue(controller.confirmUser(1_041L, ConfirmationModality.VOICE) is GoalHandoffTransition.Rejected)

        val provisional = step(controller, frame(5L), CameraRelativeBearing.RIGHT)
        assertEquals(GoalContinuityState.PROVISIONAL_REACQUIRE, provisional.continuityState)
        assertEquals(GoalCopilotAction.HOLD_STEADY_CONFIRM, provisional.action)

        val reacquired = step(controller, frame(6L), CameraRelativeBearing.RIGHT)
        assertEquals(GoalContinuityState.BOUND, reacquired.continuityState)
        assertEquals(GoalCopilotAction.GUIDE_RIGHT, reacquired.action)
        assertTrue(reacquired.handoffState is GoalHandoffState.Approach)
    }

    @Test
    fun continuityCanOnlyRequestTwoBetterViewsThenFallsBackToSearch() {
        val controller = controller()
        step(controller, frame(1L), CameraRelativeBearing.FORWARD)

        val first = uncertainStep(controller, frame(2L), CameraRelativeBearing.LEFT)
        val second = uncertainStep(controller, frame(3L), CameraRelativeBearing.LEFT)
        val expired = uncertainStep(controller, frame(4L), CameraRelativeBearing.LEFT)

        assertEquals(GoalContinuityState.COASTING, first.continuityState)
        assertEquals(GoalCopilotAction.PAN_LEFT_TO_IDENTITY, first.action)
        assertEquals(GoalContinuityState.COASTING, second.continuityState)
        assertEquals(GoalContinuityState.LOST, expired.continuityState)
        assertEquals(GoalCopilotAction.SCAN_LAST_LEFT, expired.action)
        assertTrue(listOf(first, second, expired).none(GoalCopilotStep::freshSemanticIdentity))
    }

    @Test
    fun comparableNoGainRepairsActionAndFreshTargetClosesReceipt() {
        val controller = controller()
        step(controller, frame(1L), CameraRelativeBearing.LEFT)

        val issued = uncertainStep(controller, frame(2L), CameraRelativeBearing.LEFT)
        assertEquals(GoalCopilotAction.PAN_LEFT_TO_IDENTITY, issued.action)
        assertEquals(GoalCopilotAction.PAN_LEFT_TO_IDENTITY, issued.issuedActionReceipt?.action)

        val repaired = uncertainStep(
            controller,
            frame(3L),
            CameraRelativeBearing.LEFT,
            execution = execution(issued.issuedActionReceipt!!)
        )
        assertEquals(GoalObservationActionOutcomeState.NO_GAIN, repaired.priorActionOutcome?.state)
        assertEquals(GoalCopilotAction.SIDESTEP_FOR_DISAMBIGUATION, repaired.action)
        assertEquals(GoalCopilotAction.SIDESTEP_FOR_DISAMBIGUATION, repaired.issuedActionReceipt?.action)
        assertEquals(
            GoalObservationDeficit.DECISIVE_IDENTITY_UNREADABLE,
            repaired.actionUtilitySelection?.context?.deficit
        )
        assertEquals(1L, controller.actionUtilitySnapshot().single().noGain)

        val improved = step(
            controller,
            frame(4L),
            CameraRelativeBearing.RIGHT,
            priorActionExecution = execution(repaired.issuedActionReceipt!!)
        )
        assertEquals(GoalObservationActionOutcomeState.IMPROVED, improved.priorActionOutcome?.state)
        assertEquals(GoalCopilotAction.GUIDE_RIGHT, improved.action)
        assertEquals(null, improved.issuedActionReceipt)
    }

    @Test
    fun absentFollowupIsUnknownOutcomeNotActionFailure() {
        val controller = controller()
        step(controller, frame(1L), CameraRelativeBearing.LEFT)
        val issued = uncertainStep(controller, frame(2L), CameraRelativeBearing.LEFT)

        val missingFrame = frame(3L)
        val missing = controller.step(
            evidence = null,
            currentFrame = missingFrame,
            decisionAtNs = decisionAt(missingFrame),
            decisionClockDomain = CLOCK,
            sessionTimestampMs = 1_030L,
            priorActionExecution = execution(issued.issuedActionReceipt!!)
        )

        assertEquals(GoalObservationActionOutcomeState.UNKNOWN, missing.priorActionOutcome?.state)
        assertEquals("ACTION_OUTCOME_EVIDENCE_ABSENT", missing.priorActionOutcome?.reason)
        assertEquals(GoalCopilotAction.SCAN_LAST_LEFT, missing.action)
    }

    @Test
    fun issuedButUnconfirmedActionCannotClaimOutcomeOrTriggerRepair() {
        val controller = controller()
        step(controller, frame(1L), CameraRelativeBearing.LEFT)
        val issued = uncertainStep(controller, frame(2L), CameraRelativeBearing.LEFT)
        assertEquals(GoalCopilotAction.PAN_LEFT_TO_IDENTITY, issued.action)

        val unconfirmed = uncertainStep(controller, frame(3L), CameraRelativeBearing.LEFT)

        assertEquals(GoalObservationActionOutcomeState.UNKNOWN, unconfirmed.priorActionOutcome?.state)
        assertEquals("ACTION_EXECUTION_NOT_CONFIRMED", unconfirmed.priorActionOutcome?.reason)
        assertEquals(GoalCopilotAction.PAN_LEFT_TO_IDENTITY, unconfirmed.action)
    }

    private fun controller() = GoalCopilotController(
        goalId = GOAL,
        sessionId = SESSION,
        parentBindingId = BINDING
    )

    private fun step(
        controller: GoalCopilotController,
        frame: FrameStamp,
        bearing: CameraRelativeBearing,
        endpointEvidence: GoalEndpointEvidence? = null,
        priorActionExecution: GoalObservationActionExecution? = null
    ): GoalCopilotStep = controller.step(
        evidence = observation(
            frame = frame,
            state = GoalSemanticState.TARGET,
            authority = GoalSemanticAuthority.FRESH_SEMANTIC,
            bearing = bearing,
            deficit = GoalObservationDeficit.NONE
        ),
        currentFrame = frame,
        decisionAtNs = decisionAt(frame),
        decisionClockDomain = CLOCK,
        sessionTimestampMs = 1_000L + frame.frameId * 10L,
        endpointEvidence = endpointEvidence,
        priorActionExecution = priorActionExecution
    )

    private fun uncertainStep(
        controller: GoalCopilotController,
        frame: FrameStamp,
        bearing: CameraRelativeBearing,
        execution: GoalObservationActionExecution? = null
    ): GoalCopilotStep = controller.step(
        evidence = observation(
            frame = frame,
            state = GoalSemanticState.UNCERTAIN,
            authority = GoalSemanticAuthority.CONTINUITY_ONLY,
            bearing = bearing,
            deficit = GoalObservationDeficit.DECISIVE_IDENTITY_UNREADABLE
        ),
        currentFrame = frame,
        decisionAtNs = decisionAt(frame),
        decisionClockDomain = CLOCK,
        sessionTimestampMs = 1_000L + frame.frameId * 10L,
        priorActionExecution = execution
    )

    private fun execution(receipt: GoalObservationActionReceipt): GoalObservationActionExecution {
        return GoalObservationActionExecution(
            receiptId = receipt.receiptId,
            goalId = GOAL,
            sessionId = SESSION,
            parentBindingId = BINDING,
            action = receipt.action,
            executedAtNs = receipt.issuedAtNs + 1L,
            clockDomain = CLOCK
        )
    }

    private fun observation(
        frame: FrameStamp,
        state: GoalSemanticState,
        authority: GoalSemanticAuthority,
        bearing: CameraRelativeBearing,
        deficit: GoalObservationDeficit
    ) = GoalObservationEvidence(
        sourceContractId = GoalCopilotController.CONTRACT_ID,
        sourceId = GoalCopilotController.SEMANTIC_CARRIER_SOURCE_ID,
        goalId = GOAL,
        sessionId = SESSION,
        parentBindingId = BINDING,
        frame = frame,
        availableAtNs = frame.receivedAtNs,
        validUntilNs = frame.receivedAtNs + 10L,
        availabilityClockDomain = CLOCK,
        semanticState = state,
        semanticAuthority = authority,
        bearing = bearing,
        deficit = deficit
    )

    private fun endpointEvidence(frame: FrameStamp): GoalEndpointEvidence {
        val estimate = CausalActionGeometryEstimate(
            state = ActionGeometryBeliefState.LOCKED,
            motionType = ActionMotionType.TRANSLATION,
            axis = ActionVector3(1.0, 0.0, 0.0),
            pairCount = 6
        )
        return GoalEndpointEvidence(
            sourceContractId = GoalHandoffReadinessGuard.CONTRACT_ID,
            goalId = GOAL,
            sessionId = SESSION,
            parentBindingId = BINDING,
            currentFrame = frame,
            availableAtNs = frame.receivedAtNs,
            validUntilNs = frame.receivedAtNs + 10L,
            availabilityClockDomain = CLOCK,
            position = GoalEndpointCondition.READY,
            visibility = GoalEndpointCondition.READY,
            grounding = GoalEndpointCondition.READY,
            orientation = GoalEndpointCondition.READY,
            reachability = GoalEndpointCondition.READY,
            actionGeometry = CausalActionGeometryObservation(
                disposition = CausalActionGeometryDisposition.ADMITTED,
                state = ActionGeometryBeliefState.LOCKED,
                goalId = GOAL,
                sessionId = SESSION,
                parentBindingId = BINDING,
                currentFrame = frame,
                sourceId = CausalActionGeometryAdmitter.PAIRED_RGBD_SOURCE_ID,
                estimate = estimate
            )
        )
    }

    private fun frame(id: Long): FrameStamp {
        val capturedAt = id * 100L
        return FrameStamp(
            frameId = id,
            capturedAtNs = capturedAt,
            receivedAtNs = capturedAt + 1L,
            sourceId = "camera",
            coordinateFrame = "camera",
            clockDomain = CLOCK
        )
    }

    private fun decisionAt(frame: FrameStamp): Long = frame.receivedAtNs + 1L

    companion object {
        private const val GOAL = "room-307"
        private const val SESSION = "session-1"
        private const val BINDING = "door-307"
        private val CLOCK = FrameClockDomain.ANDROID_ELAPSED_REALTIME
    }
}
