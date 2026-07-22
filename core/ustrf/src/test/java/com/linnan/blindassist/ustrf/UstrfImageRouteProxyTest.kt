package com.linnan.blindassist.ustrf

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class UstrfImageRouteProxyTest {
    private val subject = UstrfImageRouteProxy()

    @Test
    fun nearCentralFootprintCreatesHighRouteRisk() {
        val result = subject.evaluate(
            listOf(UstrfImageObstacle(.42f, .35f, .62f, .98f, .95f))
        )

        assertEquals(1, result.routeIntrusionCount)
        assertTrue(result.routeRisk >= .75f)
        assertTrue(result.candidates.first { it.centerX == .50f }.hardBlocked)
    }

    @Test
    fun sideObstacleDoesNotBecomeCentralRouteRisk() {
        val result = subject.evaluate(
            listOf(UstrfImageObstacle(.02f, .35f, .18f, .98f, .95f))
        )

        assertEquals(0, result.routeIntrusionCount)
        assertEquals(0f, result.routeRisk, .0001f)
        assertTrue(result.candidates.first().risk > 0f)
    }

    @Test
    fun emptyDetectorFrameMeansNoProxyRiskNotProvenSafety() {
        val result = subject.evaluate(emptyList())

        assertEquals(0f, result.routeRisk, .0001f)
        assertEquals(1, result.evidenceCount)
        assertEquals(5, result.candidates.size)
    }

    @Test(expected = IllegalArgumentException::class)
    fun invalidNormalizedObstacleIsRejected() {
        UstrfImageObstacle(-.1f, .2f, .3f, .8f, .9f)
    }
}
