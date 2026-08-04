package com.linnan.blindassist.hftf.metricdepth

import kotlin.math.abs

enum class ShadowAlertStatus {
    SILENT_UNKNOWN,
    SILENT_NO_NEAR_INTRUSION_OBSERVED,
    LEFT_RISK,
    CENTER_RISK,
    RIGHT_RISK,
    BOTH_SIDES_RISK
}

data class ShadowAlertDecision(
    val status: ShadowAlertStatus,
    val observedOpeningAngleDegrees: Int?,
    val unknownReasons: List<String>,
    val authority: String = "SHADOW_DEMO_ONLY",
    val claimCeiling: String = "not connected to user alerts or safe-route guidance"
)

data class TraversabilityAlertMapperConfig(
    val alertHorizonMeters: Float = 1.5f,
    val centerHalfAngleDegrees: Int = 10
) {
    init {
        require(alertHorizonMeters.isFinite() && alertHorizonMeters > 0f)
        require(centerHalfAngleDegrees in 0..45)
    }
}

/** Lossy and non-actuating shadow/demo projection. */
class TraversabilityAlertMapper(
    private val config: TraversabilityAlertMapperConfig = TraversabilityAlertMapperConfig()
) {
    fun map(field: MetricTraversabilityField): ShadowAlertDecision {
        if (field.status != TraversabilityFieldStatus.VALID) {
            return ShadowAlertDecision(
                ShadowAlertStatus.SILENT_UNKNOWN,
                observedOpeningAngleDegrees = null,
                unknownReasons = field.unknownReasons
            )
        }
        val envelope = field.sweepEnvelopes.singleOrNull {
            abs(it.horizonMeters - config.alertHorizonMeters) < 0.0001f
        } ?: return ShadowAlertDecision(
            ShadowAlertStatus.SILENT_UNKNOWN,
            observedOpeningAngleDegrees = null,
            unknownReasons = listOf("UNKNOWN_ALERT_HORIZON")
        )
        val center = envelope.directions.filter { abs(it.angleDegrees) <= config.centerHalfAngleDegrees }
        if (center.isEmpty() || center.any { it.state == SweepObservationState.UNKNOWN_SUPPORT }) {
            return ShadowAlertDecision(
                ShadowAlertStatus.SILENT_UNKNOWN,
                observedOpeningAngleDegrees = null,
                unknownReasons = listOf("UNKNOWN_CENTER_SUPPORT")
            )
        }
        val centerRisk = center.any { it.state == SweepObservationState.OCCUPIED_OBSERVED }
        val leftRisk = envelope.directions.any {
            it.angleDegrees < -config.centerHalfAngleDegrees &&
                it.state == SweepObservationState.OCCUPIED_OBSERVED
        }
        val rightRisk = envelope.directions.any {
            it.angleDegrees > config.centerHalfAngleDegrees &&
                it.state == SweepObservationState.OCCUPIED_OBSERVED
        }
        val status = when {
            centerRisk -> ShadowAlertStatus.CENTER_RISK
            leftRisk && rightRisk -> ShadowAlertStatus.BOTH_SIDES_RISK
            leftRisk -> ShadowAlertStatus.LEFT_RISK
            rightRisk -> ShadowAlertStatus.RIGHT_RISK
            else -> ShadowAlertStatus.SILENT_NO_NEAR_INTRUSION_OBSERVED
        }
        return ShadowAlertDecision(
            status = status,
            observedOpeningAngleDegrees = field.bestObservedClearanceDirection?.angleDegrees,
            unknownReasons = emptyList()
        )
    }
}
