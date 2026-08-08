package com.linnan.blindassist.benchmark

import androidx.test.ext.junit.runners.AndroidJUnit4
import com.linnan.blindassist.model.BoundingBox
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class UstrfContinuousTargetAssociationTest {
    @Test
    fun followsFrozenTargetAndDoesNotSwapToCooccurrence() {
        val frames = listOf(
            frame(0, targetX = 100f, cooccurrenceX = 300f),
            frame(500, targetX = 110f, cooccurrenceX = 290f),
            frame(1000, targetX = 120f, cooccurrenceX = 280f),
        )
        val result = associate(frames, listOf(anchor(500, 110f)), 500)
        assertTrue(result.primaryAnchorMatched)
        assertEquals(listOf("target", "target", "target"), result.frames.map { it.matchedDetectionId })
        assertEquals(0, result.identitySwitchCount)
    }

    @Test
    fun clearsInsteadOfReplacingExitedTargetWithSimilarObject() {
        val frames = listOf(
            frame(0, targetX = 100f, cooccurrenceX = 300f),
            frame(500, targetX = 110f, cooccurrenceX = 290f),
            frame(1000, targetX = null, cooccurrenceX = 280f),
            frame(1500, targetX = null, cooccurrenceX = 270f),
        )
        val result = associate(frames, listOf(anchor(500, 110f)), 500)
        assertEquals(listOf("target", "target", null, null), result.frames.map { it.matchedDetectionId })
        assertFalse(result.frames.last().ambiguous)
    }

    private fun associate(frames: List<ContinuousTargetFrame>, anchors: List<ContinuousTargetAnchor>, primary: Long) =
        UstrfContinuousTargetAssociation.associate(frames, anchors, primary, listOf("traffic cone"), listOf("traffic cone"),
            0.10, 0.12, 0.33, 3.0, 0.05, 2, 640, 480)

    private fun frame(timestamp: Long, targetX: Float?, cooccurrenceX: Float) = ContinuousTargetFrame(timestamp, buildList {
        if (targetX != null) add(ContinuousTargetDetection("target", "traffic cone", box(targetX)))
        add(ContinuousTargetDetection("other", "traffic cone", box(cooccurrenceX)))
    })
    private fun anchor(timestamp: Long, x: Float) = ContinuousTargetAnchor(timestamp, "visible", box(x))
    private fun box(x: Float) = BoundingBox(x, 100f, x + 50f, 200f)
}
