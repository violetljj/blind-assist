package com.linnan.blindassist.benchmark

import androidx.test.ext.junit.runners.AndroidJUnit4
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class ExplicitRouteGeometryFusionTest {
    @Test
    fun oneOfThreeWaypointHitsProducesFrozenThresholdScore() {
        val result = ExplicitRouteGeometryFusion.score(
            frame(
                waypoints = FixedExplicitRouteChoiceProvider.waypoints(ExplicitRouteChoice.STRAIGHT),
                obstacles = listOf(box(0.49, 0.78, 0.51, 0.82))
            ),
            obstacleExpansionObjectHeights = 0.0
        )

        assertTrue(result.routeValid)
        assertEquals(1.0 / 3.0, result.intersectionFraction!!, 1e-12)
    }

    @Test
    fun objectHeightExpansionMatchesPythonGeometry() {
        val result = ExplicitRouteGeometryFusion.score(
            frame(
                waypoints = listOf(
                    ExplicitRouteWaypoint(1_000, 0.50, 0.90),
                    ExplicitRouteWaypoint(2_000, 0.50, 0.86),
                    ExplicitRouteWaypoint(3_000, 0.50, 0.80)
                ),
                obstacles = listOf(box(0.55, 0.80, 0.60, 0.90))
            )
        )

        assertEquals(1.0, result.intersectionFraction!!, 1e-12)
    }

    @Test
    fun nonSquareFrameUsesPixelHeightForHorizontalExpansion() {
        val result = ExplicitRouteGeometryFusion.score(
            frame(
                frameWidthPx = 1_000,
                frameHeightPx = 2_000,
                waypoints = listOf(
                    ExplicitRouteWaypoint(1_000, 0.50, 0.40),
                    ExplicitRouteWaypoint(2_000, 0.50, 0.40),
                    ExplicitRouteWaypoint(3_000, 0.50, 0.40)
                ),
                obstacles = listOf(box(0.55, 0.40, 0.60, 0.45))
            )
        )

        assertEquals(1.0, result.intersectionFraction!!, 1e-12)
    }

    @Test
    fun staleLowConfidenceOrIncompleteRouteFailsClosed() {
        val stale = ExplicitRouteGeometryFusion.score(
            frame(
                validUntilTimestampMs = -1,
                waypoints = FixedExplicitRouteChoiceProvider.waypoints(ExplicitRouteChoice.STRAIGHT)
            )
        )
        val lowConfidence = ExplicitRouteGeometryFusion.score(
            frame(
                confidence = 0.49,
                waypoints = FixedExplicitRouteChoiceProvider.waypoints(ExplicitRouteChoice.STRAIGHT)
            )
        )
        val incomplete = ExplicitRouteGeometryFusion.score(
            frame(waypoints = listOf(ExplicitRouteWaypoint(1_000, 0.5, 0.9)))
        )

        listOf(stale, lowConfidence, incomplete).forEach { result ->
            assertFalse(result.routeValid)
            assertNull(result.intersectionFraction)
        }
    }

    @Test
    fun emptyDetectorSetIsValidClearEvidence() {
        val result = ExplicitRouteGeometryFusion.score(
            frame(waypoints = FixedExplicitRouteChoiceProvider.waypoints(ExplicitRouteChoice.STRAIGHT))
        )

        assertTrue(result.routeValid)
        assertEquals(0.0, result.intersectionFraction!!, 0.0)
    }

    @Test
    fun fixedChoicesAreDirectionSelectiveAndUnknownIsClosed() {
        val leftObstacle = listOf(box(0.34, 0.78, 0.38, 0.82))
        val left = ExplicitRouteGeometryFusion.score(
            frame(waypoints = FixedExplicitRouteChoiceProvider.waypoints(ExplicitRouteChoice.LEFT), obstacles = leftObstacle),
            obstacleExpansionObjectHeights = 0.0
        )
        val right = ExplicitRouteGeometryFusion.score(
            frame(waypoints = FixedExplicitRouteChoiceProvider.waypoints(ExplicitRouteChoice.RIGHT), obstacles = leftObstacle),
            obstacleExpansionObjectHeights = 0.0
        )
        val unknown = ExplicitRouteGeometryFusion.score(
            frame(routeValid = false, waypoints = FixedExplicitRouteChoiceProvider.waypoints(ExplicitRouteChoice.UNKNOWN))
        )

        assertEquals(1.0 / 3.0, left.intersectionFraction!!, 1e-12)
        assertEquals(0.0, right.intersectionFraction!!, 0.0)
        assertFalse(unknown.routeValid)
    }

    @Test
    fun geometryScoresFeedExistingOpenAndClearLifecycle() {
        val route = FixedExplicitRouteChoiceProvider.waypoints(ExplicitRouteChoice.STRAIGHT)
        val blocking = listOf(box(0.49, 0.78, 0.51, 0.82))
        val samples = listOf(
            ExplicitRouteGeometryFusion.score(frame(timestampMs = 0, waypoints = route, obstacles = blocking), 0.5, 0.0),
            ExplicitRouteGeometryFusion.score(frame(timestampMs = 1_000, validUntilTimestampMs = 1_500, waypoints = route, obstacles = blocking), 0.5, 0.0),
            ExplicitRouteGeometryFusion.score(frame(timestampMs = 2_000, validUntilTimestampMs = 2_500, waypoints = route), 0.5, 0.0),
            ExplicitRouteGeometryFusion.score(frame(timestampMs = 3_000, validUntilTimestampMs = 3_500, waypoints = route), 0.5, 0.0)
        )

        assertEquals(
            listOf(
                ExplicitRouteRiskTransition(ExplicitRouteRiskState.INTERVENTION_NEEDED, 1_000),
                ExplicitRouteRiskTransition(ExplicitRouteRiskState.ROUTE_CLEAR, 3_000)
            ),
            ExplicitRouteIntentFusion.decode(samples)
        )
    }

    private fun frame(
        timestampMs: Long = 0,
        validUntilTimestampMs: Long = 500,
        confidence: Double = 1.0,
        routeValid: Boolean = true,
        frameWidthPx: Int = 1_000,
        frameHeightPx: Int = 1_000,
        waypoints: List<ExplicitRouteWaypoint> = emptyList(),
        obstacles: List<ExplicitRouteObstacleBox> = emptyList()
    ) = ExplicitRouteFrameInput(
        timestampMs = timestampMs,
        validUntilTimestampMs = validUntilTimestampMs,
        confidence = confidence,
        routeValid = routeValid,
        frameWidthPx = frameWidthPx,
        frameHeightPx = frameHeightPx,
        waypoints = waypoints,
        obstacles = obstacles
    )

    private fun box(left: Double, top: Double, right: Double, bottom: Double) =
        ExplicitRouteObstacleBox(left, top, right, bottom)
}
