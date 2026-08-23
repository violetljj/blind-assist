package com.linnan.blindassist.goal

import org.junit.Assert.assertEquals
import org.junit.Assert.assertSame
import org.junit.Assert.assertTrue
import org.junit.Test

class GoalHandoffStateOwnerTest {
    @Test
    fun buttonConfirmationPersistsOneReceiptBeforePublishingCompletion() {
        val writes = mutableListOf<String>()
        val owner = GoalHandoffStateOwner(CompletionReceiptSink(writes::add))
        advanceToHandoff(owner)

        val result = owner.dispatch(
            GoalHandoffEvent.UserConfirmed(1_100L, ConfirmationModality.BUTTON)
        )
        val duplicate = owner.dispatch(
            GoalHandoffEvent.UserConfirmed(1_200L, ConfirmationModality.VOICE)
        )

        assertTrue(result is GoalHandoffTransition.Accepted)
        assertTrue(owner.state.value is GoalHandoffState.CompletedByUser)
        assertEquals(1, writes.size)
        assertTrue(writes.single().contains("\"confirmation_modality\":\"BUTTON\""))
        assertTrue(duplicate is GoalHandoffTransition.Rejected)
    }

    @Test
    fun exactVoicePhraseUsesVoiceModalityWhileOtherPhrasesAreIgnored() {
        val writes = mutableListOf<String>()
        val owner = GoalHandoffStateOwner(CompletionReceiptSink(writes::add))
        advanceToHandoff(owner)

        val ignored = owner.onVoicePhrase("我找到了", 1_050L)
        assertTrue(ignored is VoicePhraseResult.Ignored)
        assertTrue(owner.state.value is GoalHandoffState.HandoffReady)
        assertTrue(writes.isEmpty())

        val accepted = owner.onVoicePhrase("\u3000找到了。\u3000", 1_100L)
        assertTrue(accepted is VoicePhraseResult.Dispatched)
        assertTrue(owner.state.value is GoalHandoffState.CompletedByUser)
        assertTrue(writes.single().contains("\"confirmation_modality\":\"VOICE\""))
    }

    @Test
    fun receiptWriteFailureKeepsHandoffReady() {
        val owner = GoalHandoffStateOwner(
            CompletionReceiptSink { throw IllegalStateException("disk unavailable") }
        )
        advanceToHandoff(owner)
        val before = owner.state.value

        val failure = runCatching {
            owner.dispatch(GoalHandoffEvent.UserConfirmed(1_100L, ConfirmationModality.BUTTON))
        }.exceptionOrNull()

        assertTrue(failure is IllegalStateException)
        assertSame(before, owner.state.value)
    }

    private fun advanceToHandoff(owner: GoalHandoffStateOwner) {
        owner.dispatch(GoalHandoffEvent.Found("goal-1", "session-1"))
        owner.dispatch(GoalHandoffEvent.Approach)
        owner.dispatch(GoalHandoffEvent.HandoffReady(1_000L, "CURRENT_FRAME_HANDOFF_READY"))
    }
}
