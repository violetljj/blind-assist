package com.linnan.blindassist.risk

class RiskStabilizer(
    private val mediumConfirmFrames: Int = MEDIUM_CONFIRM_FRAMES,
    private val holdAlertMs: Long = HOLD_ALERT_MS
) {
    private var pendingKey: RiskKey? = null
    private var pendingFrames: Int = 0
    private var confirmed: RiskResult? = null
    private var confirmedAtMs: Long = 0L

    fun update(raw: RiskResult, nowMs: Long = System.currentTimeMillis()): RiskResult {
        return when (raw.level) {
            RiskLevel.HIGH -> confirm(raw, nowMs)
            RiskLevel.MEDIUM -> updateMedium(raw, nowMs)
            RiskLevel.LOW,
            RiskLevel.NONE -> fallback(raw, nowMs)
        }
    }

    fun reset() {
        pendingKey = null
        pendingFrames = 0
        confirmed = null
        confirmedAtMs = 0L
    }

    private fun updateMedium(raw: RiskResult, nowMs: Long): RiskResult {
        val key = RiskKey.from(raw)
        val previousKey = pendingKey
        pendingFrames = when {
            previousKey == key -> pendingFrames + 1
            previousKey != null && previousKey.direction == key.direction &&
                key.proximity.ordinal > previousKey.proximity.ordinal -> mediumConfirmFrames
            else -> 1
        }
        pendingKey = key

        if (pendingFrames >= mediumConfirmFrames) {
            return confirm(raw, nowMs)
        }
        return heldOr(raw.copy(level = RiskLevel.NONE, direction = RiskDirection.NONE, message = "未发现风险"), nowMs)
    }

    private fun confirm(raw: RiskResult, nowMs: Long): RiskResult {
        pendingKey = RiskKey.from(raw)
        pendingFrames = if (raw.level == RiskLevel.HIGH) mediumConfirmFrames else pendingFrames
        confirmed = raw
        confirmedAtMs = nowMs
        return raw
    }

    private fun fallback(raw: RiskResult, nowMs: Long): RiskResult {
        pendingKey = null
        pendingFrames = 0
        return heldOr(raw, nowMs)
    }

    private fun heldOr(raw: RiskResult, nowMs: Long): RiskResult {
        val current = confirmed
        if (current != null && nowMs - confirmedAtMs <= holdAlertMs) {
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
        const val MEDIUM_CONFIRM_FRAMES = 2
        const val HOLD_ALERT_MS = 600L
    }
}
