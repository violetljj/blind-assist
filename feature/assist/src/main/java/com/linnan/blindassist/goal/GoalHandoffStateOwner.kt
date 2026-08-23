package com.linnan.blindassist.goal

import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow

/**
 * Injectable product owner for a single goal/session handoff sequence.
 *
 * The default app does not construct this owner. A future Goal Copilot integration may feed
 * [GoalHandoffEvent] values into it and expose [state] to the existing Compose app contract.
 */
class GoalHandoffStateOwner(
    private val receiptSink: CompletionReceiptSink,
    private val receiptEncoder: CompletionReceiptEncoder = GoalCompletionReceiptJsonV1
) {
    private val mutableState = MutableStateFlow<GoalHandoffState>(GoalHandoffState.Inactive)
    val state: StateFlow<GoalHandoffState> = mutableState.asStateFlow()

    /**
     * Persists a completion receipt before publishing COMPLETED_BY_USER. If persistence throws,
     * the observable state remains HANDOFF_READY so completion fails closed.
     */
    @Synchronized
    fun dispatch(event: GoalHandoffEvent): GoalHandoffTransition {
        val transition = GoalHandoffReducer.reduce(mutableState.value, event)
        if (transition is GoalHandoffTransition.Accepted) {
            transition.completionReceipt?.let { receipt ->
                receiptSink.write(receiptEncoder.encode(receipt))
            }
            mutableState.value = transition.state
        }
        return transition
    }

    /**
     * Adapter seam for text produced by an external ASR owner. No microphone or ASR runtime is
     * provided here. Only the normalized exact phrase "找到了" becomes an explicit confirmation.
     */
    fun onVoicePhrase(
        phrase: String,
        confirmationTimestamp: Long
    ): VoicePhraseResult {
        if (!VoiceConfirmationPhrase.isExplicitConfirmation(phrase)) {
            return VoicePhraseResult.Ignored(normalizedPhrase = VoiceConfirmationPhrase.normalize(phrase))
        }
        return VoicePhraseResult.Dispatched(
            dispatch(
                GoalHandoffEvent.UserConfirmed(
                    timestamp = confirmationTimestamp,
                    modality = ConfirmationModality.VOICE
                )
            )
        )
    }
}

sealed interface VoicePhraseResult {
    data class Ignored(val normalizedPhrase: String) : VoicePhraseResult
    data class Dispatched(val transition: GoalHandoffTransition) : VoicePhraseResult
}
