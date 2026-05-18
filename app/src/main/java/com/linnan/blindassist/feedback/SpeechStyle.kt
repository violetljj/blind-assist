package com.linnan.blindassist.feedback

import com.linnan.blindassist.risk.ProximityBand
import com.linnan.blindassist.risk.RiskDirection
import com.linnan.blindassist.risk.RiskLevel
import com.linnan.blindassist.risk.RiskResult

enum class SpeechStyle(
    val storageValue: String,
    val displayName: String,
    val description: String
) {
    BRIEF("brief", "简短", "只播报方向和紧急程度"),
    STANDARD("standard", "标准", "使用当前短句提醒"),
    DETAILED("detailed", "详细", "补充目标类别和避让建议");

    fun messageFor(risk: RiskResult): String {
        if (risk.level == RiskLevel.NONE) return "未发现风险"
        return when (this) {
            BRIEF -> briefMessage(risk)
            STANDARD -> risk.message
            DETAILED -> detailedMessage(risk)
        }
    }

    private fun briefMessage(risk: RiskResult): String {
        return when {
            risk.proximity == ProximityBand.CRITICAL -> "${directionText(risk.direction)}很近"
            risk.proximity == ProximityBand.NEAR -> "${directionText(risk.direction)}近处"
            risk.level == RiskLevel.MEDIUM -> "${directionText(risk.direction)}留意"
            else -> "注意前方"
        }
    }

    private fun detailedMessage(risk: RiskResult): String {
        val objectName = objectName(risk.sourceDetection?.label)
        return when {
            risk.proximity == ProximityBand.CRITICAL -> {
                "${directionText(risk.direction)}有${objectName}迫近，请立刻放慢"
            }
            risk.proximity == ProximityBand.NEAR -> {
                "${directionText(risk.direction)}近处有${objectName}，请注意避让"
            }
            else -> {
                "${directionText(risk.direction)}有${objectName}，请保持观察"
            }
        }
    }

    companion object {
        fun fromStorageValue(value: String?): SpeechStyle {
            return values().firstOrNull { it.storageValue == value } ?: STANDARD
        }

        private fun directionText(direction: RiskDirection): String {
            return when (direction) {
                RiskDirection.LEFT -> "左前方"
                RiskDirection.CENTER -> "正前方"
                RiskDirection.RIGHT -> "右前方"
                RiskDirection.NONE -> "前方"
            }
        }

        private fun objectName(label: String?): String {
            return when (label) {
                "person" -> "行人"
                "car", "bus", "truck", "motorcycle", "bicycle" -> "车辆"
                "bench", "chair", "potted plant" -> "障碍物"
                "traffic light", "stop sign" -> "交通标志"
                else -> "障碍物"
            }
        }
    }
}
