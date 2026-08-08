package com.linnan.blindassist.benchmark

internal data class ExplicitRouteRiskSample(
    val timestampMs: Long,
    val routeValid: Boolean,
    val intersectionFraction: Double?
)

internal enum class ExplicitRouteRiskState {
    INTERVENTION_NEEDED,
    ROUTE_CLEAR
}

internal data class ExplicitRouteRiskTransition(
    val state: ExplicitRouteRiskState,
    val timestampMs: Long
)

/**
 * Benchmark-only mirror of the frozen explicit route-intent lifecycle contract.
 *
 * Missing route intent never opens an intervention and never fabricates a clear transition.
 * This object is intentionally isolated from the application runtime until independent route
 * providers and the offline/INT8/device event gates pass.
 */
internal object ExplicitRouteIntentFusion {
    fun decode(
        samples: List<ExplicitRouteRiskSample>,
        threshold: Double = 1.0 / 3.0,
        openConsecutive: Int = 2,
        clearConsecutive: Int = 2,
        expectedStepMs: Long = 1_000L
    ): List<ExplicitRouteRiskTransition> {
        require(openConsecutive > 0 && clearConsecutive > 0) {
            "consecutive counts must be positive"
        }
        val transitions = mutableListOf<ExplicitRouteRiskTransition>()
        var interventionOpen = false
        var activeRun = 0
        var clearRun = 0
        var previousTimestamp: Long? = null

        samples.sortedBy { it.timestampMs }.forEach { sample ->
            val contiguous = previousTimestamp?.let {
                sample.timestampMs - it == expectedStepMs
            } == true
            if (!contiguous) {
                activeRun = 0
                clearRun = 0
            }
            previousTimestamp = sample.timestampMs

            val score = sample.intersectionFraction
            if (!sample.routeValid || score == null) {
                activeRun = 0
                clearRun = 0
                return@forEach
            }
            val active = score >= threshold
            if (!interventionOpen) {
                activeRun = if (active) activeRun + 1 else 0
                clearRun = 0
                if (activeRun >= openConsecutive) {
                    interventionOpen = true
                    transitions += ExplicitRouteRiskTransition(
                        ExplicitRouteRiskState.INTERVENTION_NEEDED,
                        sample.timestampMs
                    )
                    activeRun = 0
                }
            } else {
                clearRun = if (!active) clearRun + 1 else 0
                activeRun = 0
                if (clearRun >= clearConsecutive) {
                    interventionOpen = false
                    transitions += ExplicitRouteRiskTransition(
                        ExplicitRouteRiskState.ROUTE_CLEAR,
                        sample.timestampMs
                    )
                    clearRun = 0
                }
            }
        }
        return transitions
    }
}
