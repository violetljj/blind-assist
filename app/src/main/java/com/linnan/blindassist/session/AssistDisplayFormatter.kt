package com.linnan.blindassist.session

import com.linnan.blindassist.risk.ProximityBand
import com.linnan.blindassist.risk.RiskDirection
import com.linnan.blindassist.risk.RiskLevel
import com.linnan.blindassist.risk.RiskResult

object AssistDisplayFormatter {
    fun targetLine(
        rawRisk: RiskResult,
        stableRisk: RiskResult,
        detectionCount: Int
    ): String {
        val sourceName = objectName(stableRisk.sourceDetection?.label)
        if (isHoldingPreviousRisk(rawRisk, stableRisk, detectionCount) && sourceName != null) {
            return "提醒保持：上一帧 $sourceName · 当前帧 0 个"
        }
        if (detectionCount <= 0) {
            return "当前未锁定主要目标"
        }
        return if (sourceName != null && stableRisk.level != RiskLevel.NONE) {
            "当前目标 $detectionCount 个 · 主要目标 $sourceName"
        } else {
            "当前目标 $detectionCount 个 · 暂无需要提醒的主要目标"
        }
    }

    fun careTargetLine(
        rawRisk: RiskResult,
        stableRisk: RiskResult,
        detectionCount: Int
    ): String {
        val sourceName = objectName(stableRisk.sourceDetection?.label)
        if (isHoldingPreviousRisk(rawRisk, stableRisk, detectionCount)) {
            return "提醒短暂保持，当前帧未重新锁定目标"
        }
        if (stableRisk.level == RiskLevel.NONE) {
            return "没有发现需要立即提醒的障碍"
        }
        return sourceName?.let { "主要目标：$it" } ?: "请继续听从方向提醒"
    }

    fun detailFor(risk: RiskResult): String {
        val proximityText = proximityText(risk.proximity)
        val directionText = directionText(risk.direction)
        return when {
            risk.proximity == ProximityBand.CRITICAL -> "$proximityText · $directionText 很近，请立刻放慢并确认环境"
            risk.level == RiskLevel.HIGH -> "$proximityText · 请减速，先确认 $directionText 方向"
            risk.level == RiskLevel.MEDIUM -> "$proximityText · 留意 $directionText 方向，谨慎通过"
            risk.level == RiskLevel.LOW -> "$proximityText · 发现目标，暂不触发强提醒"
            else -> "持续观察中，未发现需要立即提醒的障碍"
        }
    }

    fun careDetailFor(risk: RiskResult): String {
        val directionText = directionText(risk.direction)
        return when {
            risk.proximity == ProximityBand.CRITICAL -> "很近，先放慢"
            risk.level == RiskLevel.HIGH -> "$directionText 有风险，先减速"
            risk.level == RiskLevel.MEDIUM -> "留意 $directionText，谨慎通过"
            risk.level == RiskLevel.LOW -> "保持观察，暂不强提醒"
            else -> "继续观察前方"
        }
    }

    fun urgencyLine(rawRisk: RiskResult, stableRisk: RiskResult): String {
        return "紧急度：原始 ${formatScore(rawRisk.urgencyScore)} / 稳定 ${formatScore(stableRisk.urgencyScore)}"
    }

    fun accessibilityTargetSummary(
        rawRisk: RiskResult,
        stableRisk: RiskResult,
        detectionCount: Int
    ): String {
        return targetLine(rawRisk, stableRisk, detectionCount)
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

    private fun objectName(label: String?): String? {
        return when (label) {
            null -> null
            "person" -> "人"
            "car", "bus", "truck", "motorcycle", "bicycle" -> "车辆"
            "bench", "chair", "potted plant" -> "障碍"
            "traffic light" -> "交通灯"
            "stop sign" -> "停止标志"
            else -> label
        }
    }

    private fun directionText(direction: RiskDirection): String {
        return when (direction) {
            RiskDirection.LEFT -> "左前"
            RiskDirection.CENTER -> "正前"
            RiskDirection.RIGHT -> "右前"
            RiskDirection.NONE -> "前方"
        }
    }

    private fun proximityText(proximity: ProximityBand): String {
        return when (proximity) {
            ProximityBand.CRITICAL -> "迫近"
            ProximityBand.NEAR -> "近处"
            ProximityBand.MID -> "中距"
            ProximityBand.FAR -> "远处"
        }
    }

    private fun formatScore(score: Float): String {
        return "%.2f".format(score)
    }
}
