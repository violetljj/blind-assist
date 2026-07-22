package com.linnan.blindassist.ustrf

import kotlin.math.sqrt

/**
 * Explicitly provisional image-plane input for the separately packaged USTRF experiment app.
 *
 * This is not metric geometry: it has no depth, body-frame pose, ground plane, or physical TTC.
 * It only answers whether a detector footprint intersects an assumed image-plane route corridor.
 */
data class UstrfImageObstacle(
    val left: Float,
    val top: Float,
    val right: Float,
    val bottom: Float,
    val confidence: Float
) {
    init {
        require(left in 0f..1f && top in 0f..1f && right in 0f..1f && bottom in 0f..1f)
        require(right > left && bottom > top)
        require(confidence in 0f..1f)
    }

    val width: Float get() = right - left
    val height: Float get() = bottom - top
    val area: Float get() = width * height
}

data class UstrfImageCorridorCandidate(
    val centerX: Float,
    val risk: Float,
    val intrusionCount: Int,
    val hardBlocked: Boolean
)

data class UstrfImageRouteProxyResult(
    val routeRisk: Float,
    val routeIntrusionCount: Int,
    val evidenceCount: Int,
    val candidates: List<UstrfImageCorridorCandidate>,
    /** Diagnostic only. It is not a user-direction instruction. */
    val lowestProxyRiskCenterX: Float
)

data class UstrfImageRouteProxyConfig(
    val routeCenterX: Float = .50f,
    val corridorHalfWidth: Float = .10f,
    val footprintHeightRatio: Float = .25f,
    val hardRiskThreshold: Float = .60f,
    val candidateCenters: List<Float> = listOf(.18f, .34f, .50f, .66f, .82f)
) {
    init {
        require(routeCenterX in 0f..1f)
        require(corridorHalfWidth > 0f && corridorHalfWidth <= .5f)
        require(footprintHeightRatio > 0f && footprintHeightRatio <= 1f)
        require(hardRiskThreshold in 0f..1f)
        require(candidateCenters.size == 5 && candidateCenters.all { it in 0f..1f })
    }
}

class UstrfImageRouteProxy(
    private val config: UstrfImageRouteProxyConfig = UstrfImageRouteProxyConfig()
) {
    fun evaluate(obstacles: List<UstrfImageObstacle>): UstrfImageRouteProxyResult {
        val candidates = config.candidateCenters.map { center -> evaluateCorridor(center, obstacles) }
        val route = evaluateCorridor(config.routeCenterX, obstacles)
        val lowest = candidates.minWithOrNull(
            compareBy<UstrfImageCorridorCandidate> { it.risk }
                .thenBy { kotlin.math.abs(it.centerX - config.routeCenterX) }
                .thenBy { it.centerX }
        ) ?: error("candidate corridor inventory must not be empty")
        return UstrfImageRouteProxyResult(
            routeRisk = route.risk,
            routeIntrusionCount = route.intrusionCount,
            evidenceCount = maxOf(1, obstacles.size),
            candidates = candidates,
            lowestProxyRiskCenterX = lowest.centerX
        )
    }

    private fun evaluateCorridor(
        centerX: Float,
        obstacles: List<UstrfImageObstacle>
    ): UstrfImageCorridorCandidate {
        val corridorLeft = (centerX - config.corridorHalfWidth).coerceAtLeast(0f)
        val corridorRight = (centerX + config.corridorHalfWidth).coerceAtMost(1f)
        val risks = obstacles.mapNotNull { obstacle ->
            val overlap = overlapWidth(obstacle.left, obstacle.right, corridorLeft, corridorRight)
            if (overlap <= 0f) return@mapNotNull null
            val footprintTop = obstacle.bottom - obstacle.height * config.footprintHeightRatio
            val footprintArea = obstacle.width * (obstacle.bottom - footprintTop)
            val overlapStrength = (overlap / minOf(obstacle.width, corridorRight - corridorLeft)).coerceIn(0f, 1f)
            val nearness = ((obstacle.bottom - .35f) / .65f).coerceIn(0f, 1f)
            val areaSignal = sqrt((footprintArea / .08f).coerceIn(0f, 1f))
            obstacle.confidence * (.45f * nearness + .30f * overlapStrength + .25f * areaSignal)
        }
        val risk = risks.maxOrNull()?.coerceIn(0f, 1f) ?: 0f
        return UstrfImageCorridorCandidate(
            centerX = centerX,
            risk = risk,
            intrusionCount = risks.size,
            hardBlocked = risk >= config.hardRiskThreshold
        )
    }

    private fun overlapWidth(leftA: Float, rightA: Float, leftB: Float, rightB: Float): Float =
        (minOf(rightA, rightB) - maxOf(leftA, leftB)).coerceAtLeast(0f)
}
