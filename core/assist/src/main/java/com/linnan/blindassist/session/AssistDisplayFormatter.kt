package com.linnan.blindassist.session

import com.linnan.blindassist.localization.AppLanguage
import com.linnan.blindassist.localization.LocalizedText
import com.linnan.blindassist.risk.ProximityBand
import com.linnan.blindassist.risk.RiskDirection
import com.linnan.blindassist.risk.RiskLevel
import com.linnan.blindassist.risk.RiskResult

object AssistDisplayFormatter {
    fun targetLine(
        rawRisk: RiskResult,
        stableRisk: RiskResult,
        detectionCount: Int,
        language: AppLanguage = AppLanguage.ZH
    ): String {
        val sourceName = objectName(stableRisk.sourceDetection?.label, language)
        if (isHoldingPreviousRisk(rawRisk, stableRisk, detectionCount) && sourceName != null) {
            return if (language == AppLanguage.EN) {
                "Reminder held: previous $sourceName, current frame 0 objects"
            } else {
                "提醒保持：上一帧 $sourceName · 当前帧 0 个"
            }
        }
        if (detectionCount <= 0) {
            return if (language == AppLanguage.EN) "No main object locked" else "当前未锁定主要目标"
        }
        return if (sourceName != null && stableRisk.level != RiskLevel.NONE) {
            if (language == AppLanguage.EN) "$detectionCount objects, main object $sourceName" else "当前目标 $detectionCount 个 · 主要目标 $sourceName"
        } else {
            if (language == AppLanguage.EN) "$detectionCount objects, no main object needs a reminder" else "当前目标 $detectionCount 个 · 暂无需要提醒的主要目标"
        }
    }

    fun careTargetLine(
        rawRisk: RiskResult,
        stableRisk: RiskResult,
        detectionCount: Int,
        language: AppLanguage = AppLanguage.ZH
    ): String {
        val sourceName = objectName(stableRisk.sourceDetection?.label, language)
        if (isHoldingPreviousRisk(rawRisk, stableRisk, detectionCount)) {
            return if (language == AppLanguage.EN) "Reminder is briefly held; no object relocked in this frame" else "提醒短暂保持，当前帧未重新锁定目标"
        }
        if (stableRisk.level == RiskLevel.NONE) {
            return if (language == AppLanguage.EN) "No obstacle needs an immediate reminder" else "没有发现需要立即提醒的障碍"
        }
        return sourceName?.let {
            if (language == AppLanguage.EN) "Main object: $it" else "主要目标：$it"
        } ?: if (language == AppLanguage.EN) "Keep following direction reminders" else "请继续听从方向提醒"
    }

    fun detailFor(risk: RiskResult, language: AppLanguage = AppLanguage.ZH): String {
        val proximityText = proximityText(risk.proximity, language)
        val directionText = directionText(risk.direction, language)
        return if (language == AppLanguage.EN) {
            when {
                risk.proximity == ProximityBand.CRITICAL -> "$proximityText, $directionText is very close. Slow down and confirm the environment."
                risk.level == RiskLevel.HIGH -> "$proximityText. Slow down and check $directionText first."
                risk.level == RiskLevel.MEDIUM -> "$proximityText. Watch $directionText and pass carefully."
                risk.level == RiskLevel.LOW -> "$proximityText. Object found, no strong reminder yet."
                else -> "Still observing. No obstacle needs an immediate reminder."
            }
        } else {
            when {
                risk.proximity == ProximityBand.CRITICAL -> "$proximityText · $directionText 很近，请立刻放慢并确认环境"
                risk.level == RiskLevel.HIGH -> "$proximityText · 请减速，先确认 $directionText 方向"
                risk.level == RiskLevel.MEDIUM -> "$proximityText · 留意 $directionText 方向，谨慎通过"
                risk.level == RiskLevel.LOW -> "$proximityText · 发现目标，暂不触发强提醒"
                else -> "持续观察中，未发现需要立即提醒的障碍"
            }
        }
    }

    fun careDetailFor(risk: RiskResult, language: AppLanguage = AppLanguage.ZH): String {
        val directionText = directionText(risk.direction, language)
        return if (language == AppLanguage.EN) {
            when {
                risk.proximity == ProximityBand.CRITICAL -> "Very close, slow down first"
                risk.level == RiskLevel.HIGH -> "Risk at $directionText, slow down first"
                risk.level == RiskLevel.MEDIUM -> "Watch $directionText, pass carefully"
                risk.level == RiskLevel.LOW -> "Keep observing, no strong reminder yet"
                else -> "Keep observing ahead"
            }
        } else {
            when {
                risk.proximity == ProximityBand.CRITICAL -> "很近，先放慢"
                risk.level == RiskLevel.HIGH -> "$directionText 有风险，先减速"
                risk.level == RiskLevel.MEDIUM -> "留意 $directionText，谨慎通过"
                risk.level == RiskLevel.LOW -> "保持观察，暂不强提醒"
                else -> "继续观察前方"
            }
        }
    }

    fun urgencyLine(rawRisk: RiskResult, stableRisk: RiskResult, language: AppLanguage = AppLanguage.ZH): String {
        return if (language == AppLanguage.EN) {
            "Urgency: raw ${formatScore(rawRisk.urgencyScore)} / stable ${formatScore(stableRisk.urgencyScore)}"
        } else {
            "紧急度：原始 ${formatScore(rawRisk.urgencyScore)} / 稳定 ${formatScore(stableRisk.urgencyScore)}"
        }
    }

    fun accessibilityTargetSummary(
        rawRisk: RiskResult,
        stableRisk: RiskResult,
        detectionCount: Int,
        language: AppLanguage = AppLanguage.ZH
    ): String {
        return targetLine(rawRisk, stableRisk, detectionCount, language)
    }

    private fun isHoldingPreviousRisk(
        rawRisk: RiskResult,
        stableRisk: RiskResult,
        detectionCount: Int
    ): Boolean {
        return detectionCount <= 0 &&
            rawRisk.level == RiskLevel.NONE &&
            stableRisk.level != RiskLevel.NONE &&
            stableRisk.sourceDetection != null
    }

    private fun objectName(label: String?, language: AppLanguage): String? {
        return LocalizedText.objectName(label, language)
    }

    private fun directionText(direction: RiskDirection, language: AppLanguage): String {
        return LocalizedText.direction(direction, language, short = true)
    }

    private fun proximityText(proximity: ProximityBand, language: AppLanguage): String {
        return LocalizedText.proximity(proximity, language)
    }

    private fun formatScore(score: Float): String {
        return "%.2f".format(score)
    }
}
