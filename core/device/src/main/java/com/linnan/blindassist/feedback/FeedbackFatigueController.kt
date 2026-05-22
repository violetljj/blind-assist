package com.linnan.blindassist.feedback

import com.linnan.blindassist.risk.ProximityBand
import com.linnan.blindassist.risk.RiskLevel
import com.linnan.blindassist.risk.RiskResult

internal class FeedbackFatigueController {
    private var lastNonCriticalAtMs: Long = 0L
    private var consecutiveNonCriticalTriggers: Int = 0

    fun effectiveCooldownMs(risk: RiskResult, baseCooldownMs: Long, nowMs: Long): Long {
        if (risk.proximity == ProximityBand.CRITICAL && risk.level == RiskLevel.HIGH) {
            return baseCooldownMs
        }
        val isFreshSeries = nowMs - lastNonCriticalAtMs <= FATIGUE_WINDOW_MS
        val fatigueLevel = if (isFreshSeries) consecutiveNonCriticalTriggers else 0
        val multiplier = when {
            fatigueLevel >= 4 -> 2.0f
            fatigueLevel >= 2 -> 1.5f
            else -> 1.0f
        }
        return (baseCooldownMs * multiplier).toLong()
    }

    fun recordTriggered(risk: RiskResult, nowMs: Long) {
        if (risk.proximity == ProximityBand.CRITICAL && risk.level == RiskLevel.HIGH) {
            consecutiveNonCriticalTriggers = 0
            lastNonCriticalAtMs = 0L
            return
        }
        consecutiveNonCriticalTriggers = if (nowMs - lastNonCriticalAtMs <= FATIGUE_WINDOW_MS) {
            consecutiveNonCriticalTriggers + 1
        } else {
            1
        }
        lastNonCriticalAtMs = nowMs
    }

    fun reset() {
        lastNonCriticalAtMs = 0L
        consecutiveNonCriticalTriggers = 0
    }

    companion object {
        const val FATIGUE_WINDOW_MS = 12_000L
    }
}
