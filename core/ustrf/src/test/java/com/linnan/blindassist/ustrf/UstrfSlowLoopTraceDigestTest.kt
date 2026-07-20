package com.linnan.blindassist.ustrf

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Test

class UstrfSlowLoopTraceDigestTest {
    private val frame = UstrfFrameStamp(31L, 1_000L, "camera-v1")

    @Test
    fun canonicalTraceIsDeterministicAndExcludesSemanticAndGoalText() {
        val first = record(label = "EXIT A", goal = "find elevator")
        val text = UstrfSlowLoopTraceDigest.canonicalText(listOf(first))
        assertFalse(text.contains("EXIT A"))
        assertFalse(text.contains("find elevator"))

        val changedText = record(label = "private OCR text", goal = "private user goal")
        assertEquals(
            UstrfSlowLoopTraceDigest.sha256(listOf(first)),
            UstrfSlowLoopTraceDigest.sha256(listOf(changedText))
        )
    }

    @Test
    fun changedAdmissionStateChangesTraceDigest() {
        val available = record()
        val unavailable = UstrfSlowLoopTraceRecord(
            queryFrame = frame,
            trigger = UstrfSlowLoopTrigger.USER_QUERY,
            resolution = UstrfSlowLoopResolution.Unavailable(UstrfSlowLoopFailure.SEMANTIC_STALE)
        )
        assertFalse(UstrfSlowLoopTraceDigest.sha256(listOf(available)) == UstrfSlowLoopTraceDigest.sha256(listOf(unavailable)))
    }

    private fun record(
        label: String = "EXIT",
        goal: String = "find elevator"
    ): UstrfSlowLoopTraceRecord = UstrfSlowLoopTraceRecord(
        queryFrame = frame,
        trigger = UstrfSlowLoopTrigger.USER_QUERY,
        resolution = UstrfSlowLoopResolution.Available(
            semanticHint = UstrfSemanticHint(frame, 1_010L, 1_400L, .9f, label),
            persistentSceneFact = null,
            sceneMemoryDeferredForEphemeralWorldFrame = true,
            taskGoal = UstrfTaskGoalProposal("event", goal, .9f, 1_010L, 1_400L, "fixture")
        )
    )
}
