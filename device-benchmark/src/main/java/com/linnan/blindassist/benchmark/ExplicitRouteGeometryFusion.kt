package com.linnan.blindassist.benchmark

import kotlin.math.max

internal data class ExplicitRouteWaypoint(
    val horizonMs: Long,
    val xNorm: Double,
    val yNorm: Double
)

internal data class ExplicitRouteObstacleBox(
    val leftNorm: Double,
    val topNorm: Double,
    val rightNorm: Double,
    val bottomNorm: Double
)

internal data class ExplicitRouteFrameInput(
    val timestampMs: Long,
    val validUntilTimestampMs: Long,
    val confidence: Double,
    val routeValid: Boolean,
    val frameWidthPx: Int,
    val frameHeightPx: Int,
    val waypoints: List<ExplicitRouteWaypoint>,
    val obstacles: List<ExplicitRouteObstacleBox>
)

internal enum class ExplicitRouteChoice {
    LEFT,
    STRAIGHT,
    RIGHT,
    UNKNOWN
}

/**
 * Benchmark-only fixed camera-space route templates.
 *
 * These mirror r7.99 and are not a production route provider. UNKNOWN deliberately emits no
 * waypoints so downstream geometry remains fail closed.
 */
internal object FixedExplicitRouteChoiceProvider {
    private val horizons = listOf(1_000L, 2_000L, 3_000L)
    private val y = listOf(0.92, 0.86, 0.80)

    fun waypoints(choice: ExplicitRouteChoice): List<ExplicitRouteWaypoint> {
        val x = when (choice) {
            ExplicitRouteChoice.LEFT -> listOf(0.47, 0.42, 0.36)
            ExplicitRouteChoice.STRAIGHT -> listOf(0.50, 0.50, 0.50)
            ExplicitRouteChoice.RIGHT -> listOf(0.53, 0.58, 0.64)
            ExplicitRouteChoice.UNKNOWN -> return emptyList()
        }
        return horizons.indices.map { index ->
            ExplicitRouteWaypoint(horizons[index], x[index], y[index])
        }
    }
}

/**
 * Converts a validated explicit route and normalized detector boxes into the frozen lifecycle
 * input. No model score or future-video inference is used here.
 */
internal object ExplicitRouteGeometryFusion {
    private val requiredHorizonsMs = listOf(1_000L, 2_000L, 3_000L)

    fun score(
        input: ExplicitRouteFrameInput,
        minimumConfidence: Double = 0.5,
        obstacleExpansionObjectHeights: Double = 0.5
    ): ExplicitRouteRiskSample {
        require(minimumConfidence in 0.0..1.0) { "minimum confidence must be in [0, 1]" }
        require(obstacleExpansionObjectHeights >= 0.0) { "obstacle expansion must be non-negative" }
        require(input.frameWidthPx > 0 && input.frameHeightPx > 0) { "frame dimensions must be positive" }
        validateObstacles(input.obstacles)

        val routeUsable = input.routeValid &&
            input.timestampMs <= input.validUntilTimestampMs &&
            input.validUntilTimestampMs - input.timestampMs <= 1_000L &&
            input.confidence.isFinite() &&
            input.confidence >= minimumConfidence
        if (!routeUsable) {
            return ExplicitRouteRiskSample(input.timestampMs, routeValid = false, intersectionFraction = null)
        }

        val byHorizon = input.waypoints.groupBy { it.horizonMs }
        val exactWaypoints = requiredHorizonsMs.map { horizon ->
            val rows = byHorizon[horizon]
            if (rows?.size != 1) {
                return ExplicitRouteRiskSample(input.timestampMs, routeValid = false, intersectionFraction = null)
            }
            rows.single()
        }
        if (input.waypoints.size != requiredHorizonsMs.size || exactWaypoints.any { !validPoint(it) }) {
            return ExplicitRouteRiskSample(input.timestampMs, routeValid = false, intersectionFraction = null)
        }

        val hitCount = exactWaypoints.count { waypoint ->
            input.obstacles.any { obstacle ->
                pointHitsExpandedObstacle(
                    waypoint,
                    obstacle,
                    obstacleExpansionObjectHeights,
                    input.frameWidthPx,
                    input.frameHeightPx
                )
            }
        }
        return ExplicitRouteRiskSample(
            timestampMs = input.timestampMs,
            routeValid = true,
            intersectionFraction = hitCount.toDouble() / exactWaypoints.size
        )
    }

    private fun validPoint(point: ExplicitRouteWaypoint): Boolean =
        point.xNorm.isFinite() && point.yNorm.isFinite() &&
            point.xNorm in 0.0..1.0 && point.yNorm in 0.0..1.0

    private fun validateObstacles(obstacles: List<ExplicitRouteObstacleBox>) {
        obstacles.forEach { obstacle ->
            require(
                obstacle.leftNorm.isFinite() && obstacle.topNorm.isFinite() &&
                    obstacle.rightNorm.isFinite() && obstacle.bottomNorm.isFinite() &&
                    obstacle.leftNorm in 0.0..1.0 && obstacle.rightNorm in 0.0..1.0 &&
                    obstacle.topNorm in 0.0..1.0 && obstacle.bottomNorm in 0.0..1.0 &&
                    obstacle.leftNorm < obstacle.rightNorm && obstacle.topNorm < obstacle.bottomNorm
            ) { "obstacle boxes must be finite, normalized, and non-empty" }
        }
    }

    private fun pointHitsExpandedObstacle(
        point: ExplicitRouteWaypoint,
        obstacle: ExplicitRouteObstacleBox,
        expansionHeights: Double,
        frameWidthPx: Int,
        frameHeightPx: Int
    ): Boolean {
        val objectHeightPx = max(1.0, (obstacle.bottomNorm - obstacle.topNorm) * frameHeightPx)
        val marginPx = expansionHeights * objectHeightPx
        val marginXNorm = marginPx / frameWidthPx
        val marginYNorm = marginPx / frameHeightPx
        return point.xNorm >= obstacle.leftNorm - marginXNorm &&
            point.xNorm <= obstacle.rightNorm + marginXNorm &&
            point.yNorm >= obstacle.topNorm - marginYNorm &&
            point.yNorm <= obstacle.bottomNorm + marginYNorm
    }
}
