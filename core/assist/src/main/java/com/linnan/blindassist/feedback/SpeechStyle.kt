package com.linnan.blindassist.feedback

import com.linnan.blindassist.risk.ProximityBand
import com.linnan.blindassist.risk.RiskDirection
import com.linnan.blindassist.risk.RiskLevel
import com.linnan.blindassist.risk.RiskResult
import com.linnan.blindassist.localization.AppLanguage
import com.linnan.blindassist.localization.LocalizedText

enum class SpeechStyle(
    val storageValue: String,
    val displayName: String,
    val description: String
) {
    BRIEF("brief", "简短", "只播报方向和紧急程度"),
    STANDARD("standard", "标准", "使用当前短句提醒"),
    DETAILED("detailed", "详细", "补充目标类别和避让建议");

    fun messageFor(risk: RiskResult, language: AppLanguage = AppLanguage.ZH): String {
        if (risk.level == RiskLevel.NONE) return if (language == AppLanguage.EN) "No risk detected" else "未发现风险"
        return when (this) {
            BRIEF -> briefMessage(risk, language)
            STANDARD -> if (language == AppLanguage.EN) standardMessage(risk, language) else risk.message
            DETAILED -> detailedMessage(risk, language)
        }
    }

    fun displayName(language: AppLanguage): String {
        return LocalizedText.speechStyleName(this, language)
    }

    fun description(language: AppLanguage): String {
        return LocalizedText.speechStyleDescription(this, language)
    }

    private fun briefMessage(risk: RiskResult, language: AppLanguage): String {
        return if (language == AppLanguage.EN) {
            when {
                risk.proximity == ProximityBand.CRITICAL -> "${directionText(risk.direction, language)} very close"
                risk.proximity == ProximityBand.NEAR -> "${directionText(risk.direction, language)} near"
                risk.level == RiskLevel.MEDIUM -> "watch ${directionText(risk.direction, language)}"
                else -> "watch ahead"
            }
        } else {
            when {
                risk.proximity == ProximityBand.CRITICAL -> "${directionText(risk.direction, language)}很近"
                risk.proximity == ProximityBand.NEAR -> "${directionText(risk.direction, language)}近处"
                risk.level == RiskLevel.MEDIUM -> "${directionText(risk.direction, language)}留意"
                else -> "注意前方"
            }
        }
    }

    private fun detailedMessage(risk: RiskResult, language: AppLanguage): String {
        val objectName = objectName(risk.sourceDetection?.label, language)
        return if (language == AppLanguage.EN) {
            when {
                risk.proximity == ProximityBand.CRITICAL -> {
                    "${directionText(risk.direction, language)} $objectName approaching, slow down now"
                }
                risk.proximity == ProximityBand.NEAR -> {
                    "$objectName near ${directionText(risk.direction, language)}, avoid carefully"
                }
                else -> {
                    "$objectName ${directionText(risk.direction, language)}, keep observing"
                }
            }
        } else {
            when {
                risk.proximity == ProximityBand.CRITICAL -> {
                    "${directionText(risk.direction, language)}有${objectName}迫近，请立刻放慢"
                }
                risk.proximity == ProximityBand.NEAR -> {
                    "${directionText(risk.direction, language)}近处有${objectName}，请注意避让"
                }
                else -> {
                    "${directionText(risk.direction, language)}有${objectName}，请保持观察"
                }
            }
        }
    }

    private fun standardMessage(risk: RiskResult, language: AppLanguage): String {
        val direction = directionText(risk.direction, language)
        return when {
            risk.proximity == ProximityBand.CRITICAL -> "$direction very close, slow down"
            risk.proximity == ProximityBand.NEAR -> "$direction near risk, avoid carefully"
            risk.level == RiskLevel.MEDIUM -> "watch $direction"
            else -> "keep observing ahead"
        }
    }

    companion object {
        fun fromStorageValue(value: String?): SpeechStyle {
            return values().firstOrNull { it.storageValue == value } ?: STANDARD
        }

        private fun directionText(direction: RiskDirection, language: AppLanguage): String {
            return LocalizedText.direction(direction, language)
        }

        private fun objectName(label: String?, language: AppLanguage): String {
            return LocalizedText.objectName(label, language) ?: if (language == AppLanguage.EN) "obstacle" else "障碍物"
        }
    }
}
