package com.linnan.blindassist.benchmark

import androidx.test.ext.junit.runners.AndroidJUnit4
import com.linnan.blindassist.model.BoundingBox
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class UstrfTargetInstanceMatcherTest {
    private val target = BoundingBox(100f, 100f, 200f, 300f)

    @Test
    fun unsupportedTaxonomyCannotFallBackToArbitraryPerson() {
        val result = UstrfTargetInstanceMatcher.match(
            target, listOf("person", "car"), listOf("traffic cone"),
            listOf(TargetMatchCandidate("d0", "person", target))
        )
        assertEquals("unsupported_taxonomy", result.status)
        assertNull(result.matchedDetectionId)
    }

    @Test
    fun choosesUniqueEligibleMaximumIou() {
        val result = UstrfTargetInstanceMatcher.match(
            target, listOf("traffic cone", "person"), listOf("traffic cone"),
            listOf(
                TargetMatchCandidate("cooccurrence", "person", target),
                TargetMatchCandidate("weak", "traffic cone", BoundingBox(150f, 150f, 250f, 350f)),
                TargetMatchCandidate("target", "traffic cone", BoundingBox(105f, 105f, 195f, 295f))
            )
        )
        assertEquals("matched", result.status)
        assertEquals("target", result.matchedDetectionId)
    }

    @Test
    fun belowFrozenIouRemainsUnmatched() {
        val result = UstrfTargetInstanceMatcher.match(
            target, listOf("traffic cone"), listOf("traffic cone"),
            listOf(TargetMatchCandidate("far", "traffic cone", BoundingBox(300f, 100f, 400f, 300f)))
        )
        assertEquals("unmatched", result.status)
        assertNull(result.matchedDetectionId)
    }
}
