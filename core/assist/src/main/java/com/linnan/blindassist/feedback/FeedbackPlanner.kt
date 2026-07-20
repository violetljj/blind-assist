package com.linnan.blindassist.feedback

import com.linnan.blindassist.alert.AlertPolicy
import com.linnan.blindassist.alert.AlertProfile
import com.linnan.blindassist.alert.AssistScenario
import com.linnan.blindassist.model.DetectionSource
import com.linnan.blindassist.risk.ApproachTrend
import com.linnan.blindassist.risk.ProximityBand
import com.linnan.blindassist.risk.RiskFusionReason
import com.linnan.blindassist.risk.RiskLevel
import com.linnan.blindassist.risk.RiskResult

/** Optional experimental feedback branches; the shipped default is conservative. */
data class FeedbackPlannerConfig(
    val enableApproachingCenterPersonMidAlert: Boolean = false
)

object FeedbackPlanner {
    const val DEFAULT_AMPLITUDE = -1
    const val STANDARD_NEAR_ALERT_COOLDOWN_MS = 1500L
    const val STANDARD_CRITICAL_ALERT_COOLDOWN_MS = 850L
    const val STANDARD_NEAR_VIBRATION_MS = 160L
    const val STANDARD_CRITICAL_VIBRATION_MS = 420L

    fun planFor(
        risk: RiskResult,
        profile: AlertProfile = AlertProfile.STANDARD,
        vibrationStrength: VibrationStrength = VibrationStrength.STANDARD,
        scenario: AssistScenario = AssistScenario.GENERAL,
        config: FeedbackPlannerConfig = FeedbackPlannerConfig()
    ): FeedbackPlan? {
        val policy = AlertPolicy.forProfile(profile, scenario)
        val basePlan = when {
            risk.proximity == ProximityBand.CRITICAL && risk.level == RiskLevel.HIGH -> {
                FeedbackPlan(policy.criticalCooldownMs, policy.criticalVibrationMs, DEFAULT_AMPLITUDE)
            }
            risk.proximity == ProximityBand.NEAR &&
                (risk.level == RiskLevel.HIGH || risk.level == RiskLevel.MEDIUM) -> {
                FeedbackPlan(policy.nearCooldownMs, policy.nearVibrationMs, DEFAULT_AMPLITUDE)
            }
            risk.sourceDetection?.source == DetectionSource.SEGMENTATION &&
                risk.level == RiskLevel.MEDIUM &&
                risk.proximity == ProximityBand.MID &&
                (RiskFusionReason.STABILITY_PROMOTED.name in risk.scoreBreakdown.fusionSummary ||
                    RiskFusionReason.MOTION_PROMOTED.name in risk.scoreBreakdown.fusionSummary) -> {
                // A center-path segmentation region reaches this branch only after the
                // temporal tracker has confirmed it in multiple frames.
                FeedbackPlan(policy.nearCooldownMs, policy.nearVibrationMs, DEFAULT_AMPLITUDE)
            }
            config.enableApproachingCenterPersonMidAlert &&
                risk.sourceDetection?.source == DetectionSource.OBJECT_DETECTOR &&
                risk.sourceDetection.label == "person" &&
                risk.direction == com.linnan.blindassist.risk.RiskDirection.CENTER &&
                risk.proximity == ProximityBand.MID &&
                risk.level >= RiskLevel.MEDIUM &&
                risk.approachTrend == ApproachTrend.APPROACHING -> {
                FeedbackPlan(policy.nearCooldownMs, policy.nearVibrationMs, DEFAULT_AMPLITUDE)
            }
            else -> null
        }
        return basePlan?.copy(
            vibrationMs = vibrationStrength.scaleDuration(basePlan.vibrationMs),
            amplitude = if (vibrationStrength == VibrationStrength.STANDARD) {
                basePlan.amplitude
            } else {
                vibrationStrength.amplitude
            }
        )
    }
}
