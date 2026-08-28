package com.linnan.blindassist.goal

/**
 * Product-layer handoff states. Perception may advance only through [GoalHandoffEvent];
 * it never has authority to create [GoalHandoffState.CompletedByUser].
 */
sealed interface GoalHandoffState {
    data object Inactive : GoalHandoffState

    data class Found(
        val goalId: String,
        val sessionId: String
    ) : GoalHandoffState {
        init {
            require(goalId.isNotBlank()) { "goalId is required" }
            require(sessionId.isNotBlank()) { "sessionId is required" }
        }
    }

    data class Approach(
        val goalId: String,
        val sessionId: String
    ) : GoalHandoffState {
        init {
            require(goalId.isNotBlank()) { "goalId is required" }
            require(sessionId.isNotBlank()) { "sessionId is required" }
        }
    }

    data class HandoffReady(
        val goalId: String,
        val sessionId: String,
        /** Epoch milliseconds from the caller-owned session clock. */
        val handoffTimestamp: Long,
        val handoffReason: String
    ) : GoalHandoffState {
        init {
            require(goalId.isNotBlank()) { "goalId is required" }
            require(sessionId.isNotBlank()) { "sessionId is required" }
            require(handoffTimestamp >= 0L) { "handoffTimestamp must be non-negative" }
            require(handoffReason.isNotBlank()) { "handoffReason is required" }
        }
    }

    data class CompletedByUser(
        val receipt: GoalCompletionReceipt
    ) : GoalHandoffState
}

sealed interface GoalHandoffEvent {
    data class Found(
        val goalId: String,
        val sessionId: String
    ) : GoalHandoffEvent

    data object Approach : GoalHandoffEvent

    data class HandoffReady(
        val timestamp: Long,
        val reason: String,
        /** A current-frame endpoint join; perception alone cannot fabricate completion. */
        val readiness: GoalHandoffReadinessDecision
    ) : GoalHandoffEvent

    /** Revokes a previously ready endpoint when its current evidence is lost or expires. */
    data class ReadinessRevoked(
        val reason: String
    ) : GoalHandoffEvent

    data class UserConfirmed(
        val timestamp: Long,
        val modality: ConfirmationModality
    ) : GoalHandoffEvent
}

enum class ConfirmationModality {
    BUTTON,
    VOICE
}

data class GoalCompletionReceipt(
    val schema: String = SCHEMA,
    val version: Int = VERSION,
    val goalId: String,
    val sessionId: String,
    /** Epoch milliseconds copied from the HANDOFF_READY event. */
    val handoffTimestamp: Long,
    val handoffReason: String,
    val userConfirmation: Boolean = true,
    val confirmationModality: ConfirmationModality,
    /** Epoch milliseconds from the explicit confirmation event, never from visual arrival. */
    val confirmationTimestamp: Long,
    /**
     * Ordinal of the accepted explicit confirmation in this state machine.
     * Completion is terminal, so the only legal value is 1; duplicate confirmations are rejected.
     */
    val attemptCount: Int = 1
) {
    init {
        require(schema == SCHEMA) { "unsupported completion receipt schema" }
        require(version == VERSION) { "unsupported completion receipt version" }
        require(goalId.isNotBlank()) { "goalId is required" }
        require(sessionId.isNotBlank()) { "sessionId is required" }
        require(handoffTimestamp >= 0L) { "handoffTimestamp must be non-negative" }
        require(handoffReason.isNotBlank()) { "handoffReason is required" }
        require(userConfirmation) { "completion requires explicit user confirmation" }
        require(confirmationTimestamp >= handoffTimestamp) {
            "confirmation cannot precede handoff"
        }
        require(attemptCount == 1) {
            "completion is terminal; accepted confirmation attemptCount must be 1"
        }
    }

    companion object {
        const val SCHEMA = "blindassist.goal_completion_receipt"
        const val VERSION = 1
    }
}

sealed interface GoalHandoffTransition {
    val state: GoalHandoffState

    data class Accepted(
        override val state: GoalHandoffState,
        val completionReceipt: GoalCompletionReceipt? = null
    ) : GoalHandoffTransition

    data class Rejected(
        override val state: GoalHandoffState,
        val reason: String
    ) : GoalHandoffTransition
}

object GoalHandoffReducer {
    fun reduce(
        current: GoalHandoffState,
        event: GoalHandoffEvent
    ): GoalHandoffTransition {
        return when {
            current is GoalHandoffState.Inactive && event is GoalHandoffEvent.Found -> {
                if (event.goalId.isBlank() || event.sessionId.isBlank()) {
                    rejected(current, "FOUND requires non-blank goalId and sessionId")
                } else {
                    accepted(GoalHandoffState.Found(event.goalId, event.sessionId))
                }
            }

            current is GoalHandoffState.Found && event is GoalHandoffEvent.Approach -> {
                accepted(GoalHandoffState.Approach(current.goalId, current.sessionId))
            }

            current is GoalHandoffState.Approach && event is GoalHandoffEvent.HandoffReady -> {
                val readiness = event.readiness
                if (event.timestamp < 0L || event.reason.isBlank()) {
                    rejected(current, "HANDOFF_READY requires a timestamp and reason")
                } else if (readiness !is GoalHandoffReadinessDecision.Ready) {
                    val detail = (readiness as GoalHandoffReadinessDecision.Blocked).reason
                    rejected(current, "HANDOFF_READY blocked by $detail")
                } else if (readiness.receipt.goalId != current.goalId ||
                    readiness.receipt.sessionId != current.sessionId
                ) {
                    rejected(current, "HANDOFF_READY readiness identity does not match active goal")
                } else {
                    accepted(
                        GoalHandoffState.HandoffReady(
                            goalId = current.goalId,
                            sessionId = current.sessionId,
                            handoffTimestamp = event.timestamp,
                            handoffReason = event.reason
                        )
                    )
                }
            }

            current is GoalHandoffState.HandoffReady && event is GoalHandoffEvent.UserConfirmed -> {
                if (event.timestamp < current.handoffTimestamp) {
                    rejected(current, "user confirmation cannot precede HANDOFF_READY")
                } else {
                    val receipt = GoalCompletionReceipt(
                        goalId = current.goalId,
                        sessionId = current.sessionId,
                        handoffTimestamp = current.handoffTimestamp,
                        handoffReason = current.handoffReason,
                        confirmationModality = event.modality,
                        confirmationTimestamp = event.timestamp
                    )
                    GoalHandoffTransition.Accepted(
                        state = GoalHandoffState.CompletedByUser(receipt),
                        completionReceipt = receipt
                    )
                }
            }

            current is GoalHandoffState.HandoffReady && event is GoalHandoffEvent.ReadinessRevoked -> {
                if (event.reason.isBlank()) {
                    rejected(current, "HANDOFF_READY revocation requires a reason")
                } else {
                    accepted(GoalHandoffState.Approach(current.goalId, current.sessionId))
                }
            }

            else -> rejected(
                current,
                "illegal transition ${current.javaClass.simpleName} -> ${event.javaClass.simpleName}"
            )
        }
    }

    private fun accepted(state: GoalHandoffState): GoalHandoffTransition.Accepted {
        return GoalHandoffTransition.Accepted(state)
    }

    private fun rejected(
        state: GoalHandoffState,
        reason: String
    ): GoalHandoffTransition.Rejected {
        return GoalHandoffTransition.Rejected(state, reason)
    }
}

fun interface CompletionReceiptSink {
    /** Writes one already serialized, versioned receipt. Implementations must be durable or throw. */
    fun write(serializedReceipt: String)
}

fun interface CompletionReceiptEncoder {
    fun encode(receipt: GoalCompletionReceipt): String
}

object GoalCompletionReceiptJsonV1 : CompletionReceiptEncoder {
    override fun encode(receipt: GoalCompletionReceipt): String {
        return buildString {
            append('{')
            appendJsonString("schema", receipt.schema)
            append(',').appendJsonNumber("version", receipt.version.toLong())
            append(',').appendJsonString("goal_id", receipt.goalId)
            append(',').appendJsonString("session_id", receipt.sessionId)
            append(',').appendJsonNumber("handoff_timestamp", receipt.handoffTimestamp)
            append(',').appendJsonString("handoff_reason", receipt.handoffReason)
            append(',').appendJsonBoolean("user_confirmation", receipt.userConfirmation)
            append(',').appendJsonString(
                "confirmation_modality",
                receipt.confirmationModality.name
            )
            append(',').appendJsonNumber(
                "confirmation_timestamp",
                receipt.confirmationTimestamp
            )
            append(',').appendJsonNumber("attempt_count", receipt.attemptCount.toLong())
            append('}')
        }
    }

    private fun StringBuilder.appendJsonString(name: String, value: String): StringBuilder {
        return append('"').append(name).append("\":\"").append(value.jsonEscaped()).append('"')
    }

    private fun StringBuilder.appendJsonNumber(name: String, value: Long): StringBuilder {
        return append('"').append(name).append("\":").append(value)
    }

    private fun StringBuilder.appendJsonBoolean(name: String, value: Boolean): StringBuilder {
        return append('"').append(name).append("\":").append(value)
    }

    private fun String.jsonEscaped(): String = buildString {
        this@jsonEscaped.forEach { char ->
            when (char) {
                '"' -> append("\\\"")
                '\\' -> append("\\\\")
                '\b' -> append("\\b")
                '\u000C' -> append("\\f")
                '\n' -> append("\\n")
                '\r' -> append("\\r")
                '\t' -> append("\\t")
                else -> if (char.code < 0x20) {
                    append("\\u")
                    append(char.code.toString(16).padStart(4, '0'))
                } else {
                    append(char)
                }
            }
        }
    }
}
