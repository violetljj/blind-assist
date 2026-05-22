package com.linnan.blindassist.feedback

import com.linnan.blindassist.alert.AlertPolicy
import com.linnan.blindassist.alert.AlertProfile
import com.linnan.blindassist.alert.AssistScenario
import com.linnan.blindassist.risk.ProximityBand
import com.linnan.blindassist.risk.RiskLevel
import com.linnan.blindassist.risk.RiskResult

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
        scenario: AssistScenario = AssistScenario.GENERAL
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
