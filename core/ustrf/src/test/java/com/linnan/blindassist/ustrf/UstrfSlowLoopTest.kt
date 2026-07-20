package com.linnan.blindassist.ustrf

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class UstrfSlowLoopTest {
    private val frame = UstrfFrameStamp(21L, 1_000L, "camera-v1")
    private val resolver = UstrfSlowLoopReceiptResolver()

    @Test
    fun validSemanticCanBeTracedButEphemeralWorldFrameCannotPersistSceneMemory() {
        val resolved = resolver.resolve(event(), semantic(), scene(UstrfWorldFrameStability.EPHEMERAL_PER_FRAME), task(), 1_100L)
            as UstrfSlowLoopResolution.Available
        assertEquals("EXIT", resolved.semanticHint.label)
        assertNull(resolved.persistentSceneFact)
        assertTrue(resolved.sceneMemoryDeferredForEphemeralWorldFrame)
        assertEquals("find elevator", resolved.taskGoal?.goalLabel)
    }

    @Test
    fun stableSceneMemoryRequiresInterFrameWorldStability() {
        val resolved = resolver.resolve(event(), semantic(), scene(UstrfWorldFrameStability.INTER_FRAME_STABLE), null, 1_100L)
            as UstrfSlowLoopResolution.Available
        assertEquals("door-17", resolved.persistentSceneFact?.sceneKey)
        assertEquals("EXIT", resolved.persistentSceneFact?.label)
        assertTrue(!resolved.sceneMemoryDeferredForEphemeralWorldFrame)
    }

    @Test
    fun staleWrongFrameAndLowConfidenceSemanticReceiptsFailClosed() {
        assertFailure(semantic(sourceFrame = frame.copy(frameId = 22L)), UstrfSlowLoopFailure.SEMANTIC_SOURCE_FRAME_MISMATCH)
        assertFailure(semantic(validUntilNs = 1_050L), UstrfSlowLoopFailure.SEMANTIC_STALE)
        assertFailure(semantic(confidence = .69f), UstrfSlowLoopFailure.SEMANTIC_LOW_CONFIDENCE)
    }

    @Test
    fun taskGoalCannotDetachFromEventOrTime() {
        val mismatch = resolver.resolve(event(), semantic(), null, task(eventId = "other"), 1_100L)
        assertEquals(UstrfSlowLoopResolution.Unavailable(UstrfSlowLoopFailure.TASK_EVENT_MISMATCH), mismatch)
        val stale = resolver.resolve(event(), semantic(), null, task(validUntilNs = 1_050L), 1_100L)
        assertEquals(UstrfSlowLoopResolution.Unavailable(UstrfSlowLoopFailure.TASK_STALE), stale)
    }

    @Test
    fun staleSceneCandidateCannotBecomePersistentMemory() {
        val stale = resolver.resolve(event(), semantic(), scene(UstrfWorldFrameStability.INTER_FRAME_STABLE, validUntilNs = 1_050L), null, 1_100L)
        assertEquals(UstrfSlowLoopResolution.Unavailable(UstrfSlowLoopFailure.SCENE_STALE), stale)
    }

    private fun assertFailure(semantic: UstrfSemanticReceipt, expected: UstrfSlowLoopFailure) = assertEquals(
        UstrfSlowLoopResolution.Unavailable(expected),
        resolver.resolve(event(), semantic, null, null, 1_100L)
    )

    private fun event() = UstrfSlowLoopEvent("event-1", UstrfSlowLoopTrigger.USER_QUERY, frame, 1_010L, 1_500L)

    private fun semantic(
        sourceFrame: UstrfFrameStamp = frame,
        validUntilNs: Long = 1_400L,
        confidence: Float = .9f
    ) = UstrfSemanticReceipt(sourceFrame, 1_020L, validUntilNs, confidence, "EXIT", "local-ocr")

    private fun scene(
        stability: UstrfWorldFrameStability,
        validUntilNs: Long = 1_400L
    ) = UstrfSceneMemoryCandidate(
        sourceFrame = frame,
        sceneKey = "door-17",
        label = "EXIT",
        confidence = .9f,
        worldFrameStability = stability,
        validUntilNs = validUntilNs
    )

    private fun task(
        eventId: String = "event-1",
        validUntilNs: Long = 1_400L
    ) = UstrfTaskGoalProposal(eventId, "find elevator", .9f, 1_030L, validUntilNs, "user-query")
}
