package com.linnan.blindassist.benchmark

import androidx.test.ext.junit.runners.AndroidJUnit4
import org.junit.Assert.assertEquals
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class ExplicitRouteIntentFusionTest {
    @Test
    fun twoActiveSamplesOpenAndTwoClearSamplesClose() {
        val transitions = ExplicitRouteIntentFusion.decode(
            listOf(
                sample(0, 1.0),
                sample(1_000, 1.0),
                sample(2_000, 0.0),
                sample(3_000, 0.0)
            )
        )

        assertEquals(
            listOf(
                ExplicitRouteRiskTransition(ExplicitRouteRiskState.INTERVENTION_NEEDED, 1_000),
                ExplicitRouteRiskTransition(ExplicitRouteRiskState.ROUTE_CLEAR, 3_000)
            ),
            transitions
        )
    }

    @Test
    fun missingRouteDoesNotOpenOrFabricateClear() {
        val transitions = ExplicitRouteIntentFusion.decode(
            listOf(
                sample(0, 1.0),
                sample(1_000, 1.0),
                ExplicitRouteRiskSample(2_000, routeValid = false, intersectionFraction = null),
                ExplicitRouteRiskSample(3_000, routeValid = false, intersectionFraction = null)
            )
        )

        assertEquals(
            listOf(ExplicitRouteRiskTransition(ExplicitRouteRiskState.INTERVENTION_NEEDED, 1_000)),
            transitions
        )
    }

    @Test
    fun timestampGapResetsConsecutiveRun() {
        val transitions = ExplicitRouteIntentFusion.decode(
            listOf(sample(0, 1.0), sample(2_000, 1.0))
        )

        assertEquals(emptyList<ExplicitRouteRiskTransition>(), transitions)
    }

    private fun sample(timestampMs: Long, score: Double) = ExplicitRouteRiskSample(
        timestampMs = timestampMs,
        routeValid = true,
        intersectionFraction = score
    )
}
