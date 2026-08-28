package com.linnan.blindassist.goal

import com.linnan.blindassist.vision.FrameClockDomain
import com.linnan.blindassist.vision.FrameStamp
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class GoalActionUtilityPolicyTest {
    @Test
    fun unknownAndDuplicateReceiptsCannotBiasUtility() {
        val policy = GoalActionUtilityPolicy()
        val unknown = outcome(
            id = "unknown",
            action = GoalCopilotAction.PAN_LEFT_TO_IDENTITY,
            state = GoalObservationActionOutcomeState.UNKNOWN
        )
        assertFalse(policy.observe(unknown))
        assertTrue(policy.snapshot().isEmpty())

        val improved = outcome(
            id = "improved",
            action = GoalCopilotAction.PAN_LEFT_TO_IDENTITY,
            state = GoalObservationActionOutcomeState.IMPROVED
        )
        assertTrue(policy.observe(improved))
        assertFalse(policy.observe(improved))
        val stats = policy.snapshot().single()
        assertEquals(1L, stats.improved)
        assertEquals(1L, stats.evaluated)
    }

    @Test
    fun learnedRepairUsesObservedUtilityInsideSafeContextBoundary() {
        val policy = GoalActionUtilityPolicy()
        repeat(8) { index ->
            policy.observe(
                outcome(
                    id = "hold-$index",
                    action = GoalCopilotAction.HOLD_STEADY_CONFIRM,
                    state = if (index < 6) {
                        GoalObservationActionOutcomeState.IMPROVED
                    } else {
                        GoalObservationActionOutcomeState.NO_GAIN
                    }
                )
            )
            policy.observe(
                outcome(
                    id = "side-$index",
                    action = GoalCopilotAction.SIDESTEP_FOR_DISAMBIGUATION,
                    state = if (index < 2) {
                        GoalObservationActionOutcomeState.IMPROVED
                    } else {
                        GoalObservationActionOutcomeState.NO_GAIN
                    }
                )
            )
            policy.observe(
                outcome(
                    id = "pan-$index",
                    action = GoalCopilotAction.PAN_LEFT_TO_IDENTITY,
                    state = if (index == 0) {
                        GoalObservationActionOutcomeState.IMPROVED
                    } else {
                        GoalObservationActionOutcomeState.NO_GAIN
                    }
                )
            )
            policy.observe(
                outcome(
                    id = "sweep-$index",
                    action = GoalCopilotAction.SWEEP_SEARCH,
                    state = if (index == 0) {
                        GoalObservationActionOutcomeState.IMPROVED
                    } else {
                        GoalObservationActionOutcomeState.NO_GAIN
                    }
                )
            )
        }
        val failed = outcome(
            id = "current-failure",
            action = GoalCopilotAction.SIDESTEP_FOR_DISAMBIGUATION,
            state = GoalObservationActionOutcomeState.NO_GAIN
        )
        policy.observe(failed)

        val selected = policy.selectRepair(failed)!!

        assertEquals(GoalCopilotAction.HOLD_STEADY_CONFIRM, selected.action)
        assertEquals(GoalObservationDeficit.ASSOCIATION_AMBIGUOUS, selected.context.deficit)
        assertTrue(selected.action != GoalCopilotAction.APPROACH_FOR_IDENTITY)
    }

    private fun outcome(
        id: String,
        action: GoalCopilotAction,
        state: GoalObservationActionOutcomeState
    ): GoalObservationActionOutcome = GoalObservationActionOutcome(
        receipt = GoalObservationActionReceipt(
            receiptId = id,
            goalId = "goal",
            sessionId = "session",
            parentBindingId = "binding",
            action = action,
            issuedFrame = FrameStamp(
                frameId = 1,
                capturedAtNs = 100,
                receivedAtNs = 101,
                sourceId = "camera",
                coordinateFrame = "camera",
                clockDomain = CLOCK
            ),
            issuedAtNs = 102,
            validUntilNs = 1_000,
            clockDomain = CLOCK,
            priorSemanticState = GoalSemanticState.UNCERTAIN,
            priorDeficit = GoalObservationDeficit.ASSOCIATION_AMBIGUOUS,
            priorBearing = CameraRelativeBearing.LEFT
        ),
        state = state,
        observedFrameId = 2,
        reason = state.name
    )

    companion object {
        private val CLOCK = FrameClockDomain.ANDROID_ELAPSED_REALTIME
    }
}
