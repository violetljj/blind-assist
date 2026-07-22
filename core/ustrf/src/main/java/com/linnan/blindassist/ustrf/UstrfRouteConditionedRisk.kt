package com.linnan.blindassist.ustrf

/** External route authority. The risk model is deliberately absent from this enum. */
enum class UstrfRouteFieldProviderType {
    NAVIGATION,
    EXPLICIT_USER_CHOICE,
    OFFLINE_AI_TEACHER
}

/**
 * A frame-bound continuous route field in the same user-local grid as [UstrfRiskField].
 *
 * Values are membership weights, not obstacle scores. A receipt must come from navigation,
 * explicit user choice, or an independently reviewed offline AI teacher; it cannot be inferred by the model whose
 * route-relative risk is being evaluated.
 */
data class UstrfRouteFieldReceipt(
    val sourceFrame: UstrfFrameStamp,
    val routeIntentId: String,
    val providerId: String,
    val providerType: UstrfRouteFieldProviderType,
    val coordinateFrame: String,
    val issuedAtNs: Long,
    val validUntilNs: Long,
    val confidence: Float,
    val routeValid: Boolean,
    val inferredByRiskModel: Boolean,
    val derivedFromFutureFrames: Boolean,
    val weights: Map<UstrfGridCoordinate, Float>,
    val gridSpec: UstrfGridSpec = UstrfGridSpec.LEGACY_KERNEL
) {
    init {
        require(routeIntentId.isNotBlank() && providerId.isNotBlank() && coordinateFrame.isNotBlank())
        require(issuedAtNs >= 0L && validUntilNs >= issuedAtNs)
        require(confidence in 0f..1f)
        require(weights.isNotEmpty())
        require(weights.values.all { it.isFinite() && it in 0f..1f })
        require(weights.values.any { it > 0f })
        require(weights.keys.all(gridSpec::contains)) { "route weights exceed the declared grid spec" }
    }
}

enum class UstrfRouteConditionedRiskFailure {
    SOURCE_FRAME_MISMATCH,
    COORDINATE_FRAME_MISMATCH,
    RISK_FIELD_FROM_FUTURE,
    RISK_FIELD_STALE,
    ROUTE_ISSUED_IN_FUTURE,
    ROUTE_STALE,
    ROUTE_LOW_CONFIDENCE,
    ROUTE_INVALID,
    ROUTE_INFERRED_BY_RISK_MODEL,
    FUTURE_DERIVED_ROUTE_FORBIDDEN,
    PROVIDER_NOT_ALLOWED_AT_RUNTIME,
    GRID_SPEC_MISMATCH
}

data class UstrfRouteIntrusionEvidence(
    val sourceFrame: UstrfFrameStamp,
    val routeIntentId: String,
    val routeIntrusionScore: Float,
    val maximumRouteCellRisk: Float,
    val routeUnknownFraction: Float,
    val contributingCellCount: Int,
    val validUntilNs: Long,
    val riskSources: Set<String>
)

sealed interface UstrfRouteConditionedRiskResolution {
    data class Available(val evidence: UstrfRouteIntrusionEvidence) : UstrfRouteConditionedRiskResolution

    data class Unavailable(
        val failure: UstrfRouteConditionedRiskFailure,
        val abstainReason: String = ROUTE_UNKNOWN_OR_INVALID
    ) : UstrfRouteConditionedRiskResolution {
        companion object {
            const val ROUTE_UNKNOWN_OR_INVALID = "route_unknown_or_invalid"
            const val RISK_FIELD_UNKNOWN_OR_INVALID = "risk_field_unknown_or_invalid"
        }
    }
}

/**
 * Deterministic interaction seam for an object-agnostic risk field and an externally supplied
 * route. It produces evidence only; it does not open an alert lifecycle or authorize movement.
 */
class UstrfRouteConditionedRiskInteractor(
    private val minimumRouteConfidence: Float = .70f,
    private val dynamicRiskHorizonMs: Long = 3_000L,
    private val maximumRiskFieldAgeNs: Long = 500_000_000L
) {
    init {
        require(minimumRouteConfidence in 0f..1f)
        require(dynamicRiskHorizonMs > 0L)
        require(maximumRiskFieldAgeNs > 0L)
    }

    fun interact(
        field: UstrfRiskField,
        route: UstrfRouteFieldReceipt,
        decisionAtNs: Long,
        runtime: Boolean
    ): UstrfRouteConditionedRiskResolution {
        val failure = when {
            route.sourceFrame != field.frame -> UstrfRouteConditionedRiskFailure.SOURCE_FRAME_MISMATCH
            route.gridSpec != field.gridSpec -> UstrfRouteConditionedRiskFailure.GRID_SPEC_MISMATCH
            route.coordinateFrame != field.frame.coordinateFrame -> UstrfRouteConditionedRiskFailure.COORDINATE_FRAME_MISMATCH
            decisionAtNs < field.frame.capturedAtNs -> UstrfRouteConditionedRiskFailure.RISK_FIELD_FROM_FUTURE
            decisionAtNs - field.frame.capturedAtNs > maximumRiskFieldAgeNs -> UstrfRouteConditionedRiskFailure.RISK_FIELD_STALE
            route.issuedAtNs > decisionAtNs -> UstrfRouteConditionedRiskFailure.ROUTE_ISSUED_IN_FUTURE
            route.validUntilNs < decisionAtNs -> UstrfRouteConditionedRiskFailure.ROUTE_STALE
            route.confidence < minimumRouteConfidence -> UstrfRouteConditionedRiskFailure.ROUTE_LOW_CONFIDENCE
            !route.routeValid -> UstrfRouteConditionedRiskFailure.ROUTE_INVALID
            route.inferredByRiskModel -> UstrfRouteConditionedRiskFailure.ROUTE_INFERRED_BY_RISK_MODEL
            route.derivedFromFutureFrames -> UstrfRouteConditionedRiskFailure.FUTURE_DERIVED_ROUTE_FORBIDDEN
            runtime && route.providerType == UstrfRouteFieldProviderType.OFFLINE_AI_TEACHER ->
                UstrfRouteConditionedRiskFailure.PROVIDER_NOT_ALLOWED_AT_RUNTIME
            else -> null
        }
        if (failure != null) {
            val reason = when (failure) {
                UstrfRouteConditionedRiskFailure.RISK_FIELD_FROM_FUTURE,
                UstrfRouteConditionedRiskFailure.RISK_FIELD_STALE ->
                    UstrfRouteConditionedRiskResolution.Unavailable.RISK_FIELD_UNKNOWN_OR_INVALID
                else -> UstrfRouteConditionedRiskResolution.Unavailable.ROUTE_UNKNOWN_OR_INVALID
            }
            return UstrfRouteConditionedRiskResolution.Unavailable(failure, reason)
        }

        val activeWeights = route.weights.filterValues { it > 0f }
        val totalWeight = activeWeights.values.sum()
        var weightedRisk = 0f
        var weightedUnknown = 0f
        var maximumRisk = 0f
        val sources = mutableSetOf<String>()
        activeWeights.forEach { (coordinate, weight) ->
            val cell = field.cellAt(coordinate)
            val dynamicRisk = cell.dynamicTtcMs?.let { ttc ->
                if (ttc <= 0L) 1f else (1f - ttc.toFloat() / dynamicRiskHorizonMs.toFloat()).coerceIn(0f, 1f)
            } ?: 0f
            val cellRisk = maxOf(cell.occupancy, cell.dropRisk, cell.headRisk, dynamicRisk)
            weightedRisk += weight * cellRisk
            weightedUnknown += weight * if (cell.isUnknown) 1f else 0f
            maximumRisk = maxOf(maximumRisk, cellRisk)
            sources += cell.sources
        }
        return UstrfRouteConditionedRiskResolution.Available(
            UstrfRouteIntrusionEvidence(
                sourceFrame = field.frame,
                routeIntentId = route.routeIntentId,
                routeIntrusionScore = (weightedRisk / totalWeight).coerceIn(0f, 1f),
                maximumRouteCellRisk = maximumRisk,
                routeUnknownFraction = (weightedUnknown / totalWeight).coerceIn(0f, 1f),
                contributingCellCount = activeWeights.size,
                validUntilNs = minOf(route.validUntilNs, riskFieldValidUntil(field.frame.capturedAtNs)),
                riskSources = sources
            )
        )
    }

    private fun riskFieldValidUntil(capturedAtNs: Long): Long =
        if (capturedAtNs > Long.MAX_VALUE - maximumRiskFieldAgeNs) Long.MAX_VALUE
        else capturedAtNs + maximumRiskFieldAgeNs
}
