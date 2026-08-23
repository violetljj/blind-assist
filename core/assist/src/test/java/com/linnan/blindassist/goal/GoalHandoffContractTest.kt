package com.linnan.blindassist.goal

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertSame
import org.junit.Assert.assertTrue
import org.junit.Test

class GoalHandoffContractTest {
    @Test
    fun legalSequenceCreatesExplicitUserReceiptOnlyAfterHandoff() {
        val found = accepted(GoalHandoffState.Inactive, GoalHandoffEvent.Found("goal-1", "session-1"))
        val approach = accepted(found.state, GoalHandoffEvent.Approach)
        val handoff = accepted(
            approach.state,
            GoalHandoffEvent.HandoffReady(1_000L, "CURRENT_FRAME_HANDOFF_READY")
        )

        assertNull(handoff.completionReceipt)
        assertTrue(handoff.state is GoalHandoffState.HandoffReady)

        val completed = accepted(
            handoff.state,
            GoalHandoffEvent.UserConfirmed(1_250L, ConfirmationModality.BUTTON)
        )
        val receipt = requireNotNull(completed.completionReceipt)
        assertTrue(completed.state is GoalHandoffState.CompletedByUser)
        assertEquals("goal-1", receipt.goalId)
        assertEquals("session-1", receipt.sessionId)
        assertEquals(1_000L, receipt.handoffTimestamp)
        assertEquals("CURRENT_FRAME_HANDOFF_READY", receipt.handoffReason)
        assertTrue(receipt.userConfirmation)
        assertEquals(ConfirmationModality.BUTTON, receipt.confirmationModality)
        assertEquals(1_250L, receipt.confirmationTimestamp)
        assertEquals(1, receipt.attemptCount)
    }

    @Test
    fun handoffBeforeApproachAndConfirmationBeforeHandoffFailClosed() {
        val inactive = GoalHandoffState.Inactive
        val prematureConfirmation = GoalHandoffReducer.reduce(
            inactive,
            GoalHandoffEvent.UserConfirmed(1L, ConfirmationModality.VOICE)
        )
        assertTrue(prematureConfirmation is GoalHandoffTransition.Rejected)
        assertSame(inactive, prematureConfirmation.state)

        val found = accepted(inactive, GoalHandoffEvent.Found("g", "s"))
        val prematureHandoff = GoalHandoffReducer.reduce(
            found.state,
            GoalHandoffEvent.HandoffReady(2L, "too-early")
        )
        assertTrue(prematureHandoff is GoalHandoffTransition.Rejected)
        assertSame(found.state, prematureHandoff.state)
    }

    @Test
    fun duplicateConfirmationIsRejectedAndCannotCreateSecondAttempt() {
        val completed = complete()

        val duplicate = GoalHandoffReducer.reduce(
            completed.state,
            GoalHandoffEvent.UserConfirmed(1_300L, ConfirmationModality.VOICE)
        )

        assertTrue(duplicate is GoalHandoffTransition.Rejected)
        assertSame(completed.state, duplicate.state)
    }

    @Test
    fun confirmationTimestampBeforeHandoffFailsClosed() {
        val handoff = handoffReady(handoffTimestamp = 2_000L)

        val result = GoalHandoffReducer.reduce(
            handoff.state,
            GoalHandoffEvent.UserConfirmed(1_999L, ConfirmationModality.BUTTON)
        )

        assertTrue(result is GoalHandoffTransition.Rejected)
        assertSame(handoff.state, result.state)
    }

    @Test
    fun jsonV1ContainsAuthorityFieldsAndEscapesUserControlledValues() {
        val receipt = GoalCompletionReceipt(
            goalId = "goal-\"1",
            sessionId = "session\\1",
            handoffTimestamp = 10L,
            handoffReason = "near\nentrance",
            confirmationModality = ConfirmationModality.VOICE,
            confirmationTimestamp = 12L
        )

        val encoded = GoalCompletionReceiptJsonV1.encode(receipt)

        assertEquals(
            "{\"schema\":\"blindassist.goal_completion_receipt\",\"version\":1," +
                "\"goal_id\":\"goal-\\\"1\",\"session_id\":\"session\\\\1\"," +
                "\"handoff_timestamp\":10,\"handoff_reason\":\"near\\nentrance\"," +
                "\"user_confirmation\":true,\"confirmation_modality\":\"VOICE\"," +
                "\"confirmation_timestamp\":12,\"attempt_count\":1}",
            encoded
        )
    }

    @Test
    fun voicePhraseNormalizationAcceptsOnlyTheExactExplicitPhrase() {
        listOf(
            "找到了",
            "  找到了  ",
            "\u3000找到了。\u3000",
            "找到了！",
            "找到了!!"
        ).forEach { phrase ->
            assertTrue(phrase, VoiceConfirmationPhrase.isExplicitConfirmation(phrase))
        }

        listOf(
            "我找到了",
            "找到了吗",
            "没找到",
            "找到",
            "找 到 了",
            "found it",
            "找到了？",
            ""
        ).forEach { phrase ->
            assertFalse(phrase, VoiceConfirmationPhrase.isExplicitConfirmation(phrase))
        }
    }

    private fun complete(): GoalHandoffTransition.Accepted {
        val handoff = handoffReady()
        return accepted(
            handoff.state,
            GoalHandoffEvent.UserConfirmed(1_250L, ConfirmationModality.BUTTON)
        )
    }

    private fun handoffReady(handoffTimestamp: Long = 1_000L): GoalHandoffTransition.Accepted {
        val found = accepted(GoalHandoffState.Inactive, GoalHandoffEvent.Found("g", "s"))
        val approach = accepted(found.state, GoalHandoffEvent.Approach)
        return accepted(
            approach.state,
            GoalHandoffEvent.HandoffReady(handoffTimestamp, "CURRENT_FRAME_HANDOFF_READY")
        )
    }

    private fun accepted(
        state: GoalHandoffState,
        event: GoalHandoffEvent
    ): GoalHandoffTransition.Accepted {
        return GoalHandoffReducer.reduce(state, event) as GoalHandoffTransition.Accepted
    }
}
