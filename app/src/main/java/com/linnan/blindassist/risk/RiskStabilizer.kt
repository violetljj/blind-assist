package com.linnan.blindassist.risk

import com.linnan.blindassist.alert.AlertPolicy
import com.linnan.blindassist.alert.AlertProfile

class RiskStabilizer {
    private var pendingKey: RiskKey? = null
    private var pendingFrames: Int = 0
    private var confirmed: RiskResult? = null
    private var confirmedAtMs: Long = 0L

    fun update(
        raw: RiskResult,
        profile: AlertProfile = AlertProfile.STANDARD,
        nowMs: Long = System.currentTimeMillis()
    ): RiskResult {
        val policy = AlertPolicy.forProfile(profile)
        return when (raw.level) {
            RiskLevel.HIGH -> confirm(raw, policy, nowMs)
            RiskLevel.MEDIUM -> updateMedium(raw, policy, nowMs)
            RiskLevel.LOW,
            RiskLevel.NONE -> fallback(raw, policy, nowMs)
        }
    }

    fun reset() {
        pendingKey = null
        pendingFrames = 0
        confirmed = null
        confirmedAtMs = 0L
    }

    private fun updateMedium(raw: RiskResult, policy: AlertPolicy, nowMs: Long): RiskResult {
        val key = RiskKey.from(raw)
        val previousKey = pendingKey
        pendingFrames = when {
            previousKey == key -> pendingFrames + 1
            previousKey != null && previousKey.direction == key.direction &&
                key.proximity.ordinal > previousKey.proximity.ordinal -> policy.mediumConfirmFrames
            else -> 1
        }
        pendingKey = key

        if (pendingFrames >= policy.mediumConfirmFrames) {
            return confirm(raw, policy, nowMs)
        }
        return heldOr(raw.copy(level = RiskLevel.NONE, direction = RiskDirection.NONE, message = "未发现风险"), policy, nowMs)
    }

    private fun confirm(raw: RiskResult, policy: AlertPolicy, nowMs: Long): RiskResult {
        pendingKey = RiskKey.from(raw)
        pendingFrames = if (raw.level == RiskLevel.HIGH) policy.mediumConfirmFrames else pendingFrames
        confirmed = raw
        confirmedAtMs = nowMs
        return raw
    }

    private fun fallback(raw: RiskResult, policy: AlertPolicy, nowMs: Long): RiskResult {
        pendingKey = null
        pendingFrames = 0
        return heldOr(raw, policy, nowMs)
    }

    private fun heldOr(raw: RiskResult, policy: AlertPolicy, nowMs: Long): RiskResult {
        val current = confirmed
        if (current != null && nowMs - confirmedAtMs <= policy.holdAlertMs) {
            return current
        }
        confirmed = null
        return raw
    }

    private data class RiskKey(
        val direction: RiskDirection,
        val message: String,
        val proximity: ProximityBand
    ) {
        companion object {
            fun from(risk: RiskResult): RiskKey = RiskKey(risk.direction, risk.message, risk.proximity)
        }
    }

    companion object {
        const val STANDARD_MEDIUM_CONFIRM_FRAMES = 2
        const val STANDARD_HOLD_ALERT_MS = 600L
    }
}
