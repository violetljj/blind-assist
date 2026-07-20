package com.linnan.blindassist.ustrf

/** Event-driven only: slow-loop work is not a per-frame safety authority. */
enum class UstrfSlowLoopTrigger { USER_QUERY, GOAL_CHANGED, HIGH_ENTROPY, SAFETY_REASSESS, PERIODIC_KEYFRAME }

data class UstrfSlowLoopEvent(
    val eventId: String,
    val trigger: UstrfSlowLoopTrigger,
    val queryFrame: UstrfFrameStamp,
    val issuedAtNs: Long,
    val validUntilNs: Long
) {
    init {
        require(eventId.isNotBlank())
        require(issuedAtNs >= queryFrame.capturedAtNs && validUntilNs >= issuedAtNs)
    }
}

/** A model/OCR observation; it has no route, action, speed, or direction field. */
data class UstrfSemanticReceipt(
    val sourceFrame: UstrfFrameStamp,
    val producedAtNs: Long,
    val validUntilNs: Long,
    val confidence: Float,
    val label: String,
    val source: String
) {
    init {
        require(producedAtNs >= sourceFrame.capturedAtNs && validUntilNs >= producedAtNs)
        require(confidence in 0f..1f && label.isNotBlank() && source.isNotBlank())
    }
}

/** Candidate fact for persistent scene memory. It must carry its spatial-stability boundary. */
data class UstrfSceneMemoryCandidate(
    val sourceFrame: UstrfFrameStamp,
    val sceneKey: String,
    val label: String,
    val confidence: Float,
    val worldFrameStability: UstrfWorldFrameStability,
    val validUntilNs: Long
) {
    init {
        require(sceneKey.isNotBlank() && label.isNotBlank() && confidence in 0f..1f)
        require(validUntilNs >= sourceFrame.capturedAtNs)
    }
}

/** User/task level proposal. It may change future slow-loop retrieval, never a safety action. */
data class UstrfTaskGoalProposal(
    val eventId: String,
    val goalLabel: String,
    val confidence: Float,
    val issuedAtNs: Long,
    val validUntilNs: Long,
    val source: String
) {
    init {
        require(eventId.isNotBlank() && goalLabel.isNotBlank() && source.isNotBlank())
        require(confidence in 0f..1f && issuedAtNs >= 0L && validUntilNs >= issuedAtNs)
    }
}

data class UstrfPersistentSceneFact(
    val sceneKey: String,
    val label: String,
    val confidence: Float,
    val sourceFrame: UstrfFrameStamp,
    val validUntilNs: Long
)

enum class UstrfSlowLoopFailure {
    EVENT_ISSUED_IN_FUTURE,
    EVENT_STALE,
    SEMANTIC_SOURCE_FRAME_MISMATCH,
    SEMANTIC_PRODUCED_IN_FUTURE,
    SEMANTIC_STALE,
    SEMANTIC_LOW_CONFIDENCE,
    SCENE_SOURCE_FRAME_MISMATCH,
    SCENE_STALE,
    SCENE_LOW_CONFIDENCE,
    TASK_EVENT_MISMATCH,
    TASK_ISSUED_IN_FUTURE,
    TASK_STALE,
    TASK_LOW_CONFIDENCE
}

sealed interface UstrfSlowLoopResolution {
    data class Available(
        val semanticHint: UstrfSemanticHint,
        val persistentSceneFact: UstrfPersistentSceneFact?,
        /** A non-persistent receipt is still useful to explain the current frame. */
        val sceneMemoryDeferredForEphemeralWorldFrame: Boolean,
        val taskGoal: UstrfTaskGoalProposal?
    ) : UstrfSlowLoopResolution

    data class Unavailable(val failure: UstrfSlowLoopFailure) : UstrfSlowLoopResolution
}

/**
 * Resolves an event-driven slow-loop result. Its output can enter [UstrfTick.semanticHints] for
 * traceability, but no output type is capable of authorizing [UstrfSafetyAction] or a corridor.
 */
class UstrfSlowLoopReceiptResolver(
    private val minimumSemanticConfidence: Float = .70f,
    private val minimumSceneConfidence: Float = .80f,
    private val minimumTaskConfidence: Float = .70f
) {
    init {
        require(minimumSemanticConfidence in 0f..1f)
        require(minimumSceneConfidence in 0f..1f)
        require(minimumTaskConfidence in 0f..1f)
    }

    fun resolve(
        event: UstrfSlowLoopEvent,
        semantic: UstrfSemanticReceipt,
        sceneCandidate: UstrfSceneMemoryCandidate?,
        taskGoal: UstrfTaskGoalProposal?,
        decisionAtNs: Long
    ): UstrfSlowLoopResolution {
        if (event.issuedAtNs > decisionAtNs) return unavailable(UstrfSlowLoopFailure.EVENT_ISSUED_IN_FUTURE)
        if (event.validUntilNs < decisionAtNs) return unavailable(UstrfSlowLoopFailure.EVENT_STALE)
        if (semantic.sourceFrame != event.queryFrame) return unavailable(UstrfSlowLoopFailure.SEMANTIC_SOURCE_FRAME_MISMATCH)
        if (semantic.producedAtNs > decisionAtNs) return unavailable(UstrfSlowLoopFailure.SEMANTIC_PRODUCED_IN_FUTURE)
        if (semantic.validUntilNs < decisionAtNs) return unavailable(UstrfSlowLoopFailure.SEMANTIC_STALE)
        if (semantic.confidence < minimumSemanticConfidence) return unavailable(UstrfSlowLoopFailure.SEMANTIC_LOW_CONFIDENCE)

        val sceneFact = when {
            sceneCandidate == null -> null
            sceneCandidate.sourceFrame != event.queryFrame -> return unavailable(UstrfSlowLoopFailure.SCENE_SOURCE_FRAME_MISMATCH)
            sceneCandidate.validUntilNs < decisionAtNs -> return unavailable(UstrfSlowLoopFailure.SCENE_STALE)
            sceneCandidate.confidence < minimumSceneConfidence -> return unavailable(UstrfSlowLoopFailure.SCENE_LOW_CONFIDENCE)
            sceneCandidate.worldFrameStability != UstrfWorldFrameStability.INTER_FRAME_STABLE -> null
            else -> UstrfPersistentSceneFact(
                sceneKey = sceneCandidate.sceneKey,
                label = sceneCandidate.label,
                confidence = sceneCandidate.confidence,
                sourceFrame = sceneCandidate.sourceFrame,
                validUntilNs = minOf(sceneCandidate.validUntilNs, semantic.validUntilNs)
            )
        }
        if (taskGoal != null) {
            if (taskGoal.eventId != event.eventId) return unavailable(UstrfSlowLoopFailure.TASK_EVENT_MISMATCH)
            if (taskGoal.issuedAtNs > decisionAtNs) return unavailable(UstrfSlowLoopFailure.TASK_ISSUED_IN_FUTURE)
            if (taskGoal.validUntilNs < decisionAtNs) return unavailable(UstrfSlowLoopFailure.TASK_STALE)
            if (taskGoal.confidence < minimumTaskConfidence) return unavailable(UstrfSlowLoopFailure.TASK_LOW_CONFIDENCE)
        }
        return UstrfSlowLoopResolution.Available(
            semanticHint = UstrfSemanticHint(
                sourceFrame = semantic.sourceFrame,
                producedAtNs = semantic.producedAtNs,
                validUntilNs = semantic.validUntilNs,
                confidence = semantic.confidence,
                label = semantic.label
            ),
            persistentSceneFact = sceneFact,
            sceneMemoryDeferredForEphemeralWorldFrame = sceneCandidate?.worldFrameStability == UstrfWorldFrameStability.EPHEMERAL_PER_FRAME,
            taskGoal = taskGoal
        )
    }

    private fun unavailable(failure: UstrfSlowLoopFailure) = UstrfSlowLoopResolution.Unavailable(failure)
}
