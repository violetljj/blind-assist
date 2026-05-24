package com.linnan.blindassist.localization

import com.linnan.blindassist.alert.AlertProfile
import com.linnan.blindassist.alert.AssistScenario
import com.linnan.blindassist.feedback.FeedbackReason
import com.linnan.blindassist.feedback.SpeechStyle
import com.linnan.blindassist.feedback.VibrationStrength
import com.linnan.blindassist.preferences.DailyUsageMode
import com.linnan.blindassist.risk.ProximityBand
import com.linnan.blindassist.risk.RiskDirection
import com.linnan.blindassist.risk.RiskLevel

object LocalizedText {
    fun enabled(enabled: Boolean, language: AppLanguage): String {
        return if (language == AppLanguage.EN) {
            if (enabled) "on" else "off"
        } else {
            if (enabled) "已开启" else "已关闭"
        }
    }

    fun alertProfileName(profile: AlertProfile, language: AppLanguage): String {
        return when (profile) {
            AlertProfile.QUIET -> if (language == AppLanguage.EN) "Quiet" else "安静"
            AlertProfile.STANDARD -> if (language == AppLanguage.EN) "Standard" else "标准"
            AlertProfile.SENSITIVE -> if (language == AppLanguage.EN) "Sensitive" else "敏感"
        }
    }

    fun assistScenarioName(scenario: AssistScenario, language: AppLanguage): String {
        return when (scenario) {
            AssistScenario.GENERAL -> if (language == AppLanguage.EN) "General" else "通用"
            AssistScenario.INDOOR -> if (language == AppLanguage.EN) "Indoor slow walk" else "室内慢行"
            AssistScenario.CORRIDOR -> if (language == AppLanguage.EN) "Corridor" else "走廊通行"
            AssistScenario.CROWDED -> if (language == AppLanguage.EN) "Crowded area" else "密集区域"
            AssistScenario.OUTDOOR_SLOW -> if (language == AppLanguage.EN) "Outdoor slow walk" else "户外慢行"
        }
    }

    fun assistScenarioDescription(scenario: AssistScenario, language: AppLanguage): String {
        return when (scenario) {
            AssistScenario.GENERAL -> if (language == AppLanguage.EN) "Keep the standard strategy" else "保持当前标准策略"
            AssistScenario.INDOOR -> if (language == AppLanguage.EN) "Reduce indoor near-risk false alarms" else "减少室内近距误触发"
            AssistScenario.CORRIDOR -> if (language == AppLanguage.EN) "Notice sustained front risks earlier" else "更早关注正前方持续风险"
            AssistScenario.CROWDED -> if (language == AppLanguage.EN) "Reduce reminder fatigue in dense spaces" else "降低密集环境提醒疲劳"
            AssistScenario.OUTDOOR_SLOW -> if (language == AppLanguage.EN) "Make outdoor slow-walk reminders clearer" else "让户外慢行提醒更清晰"
        }
    }

    fun speechStyleName(style: SpeechStyle, language: AppLanguage): String {
        return when (style) {
            SpeechStyle.BRIEF -> if (language == AppLanguage.EN) "Brief" else "简短"
            SpeechStyle.STANDARD -> if (language == AppLanguage.EN) "Standard" else "标准"
            SpeechStyle.DETAILED -> if (language == AppLanguage.EN) "Detailed" else "详细"
        }
    }

    fun speechStyleDescription(style: SpeechStyle, language: AppLanguage): String {
        return when (style) {
            SpeechStyle.BRIEF -> if (language == AppLanguage.EN) "Only speaks direction and urgency" else "只播报方向和紧急程度"
            SpeechStyle.STANDARD -> if (language == AppLanguage.EN) "Use the current short reminder" else "使用当前短句提醒"
            SpeechStyle.DETAILED -> if (language == AppLanguage.EN) "Adds object type and avoidance guidance" else "补充目标类别和避让建议"
        }
    }

    fun vibrationStrengthName(strength: VibrationStrength, language: AppLanguage): String {
        return when (strength) {
            VibrationStrength.SOFT -> if (language == AppLanguage.EN) "Soft" else "轻柔"
            VibrationStrength.STANDARD -> if (language == AppLanguage.EN) "Standard" else "标准"
            VibrationStrength.STRONG -> if (language == AppLanguage.EN) "Strong" else "强"
        }
    }

    fun vibrationStrengthDescription(strength: VibrationStrength, language: AppLanguage): String {
        return when (strength) {
            VibrationStrength.SOFT -> if (language == AppLanguage.EN) "Lower vibration duration and intensity" else "降低震动时长和强度"
            VibrationStrength.STANDARD -> if (language == AppLanguage.EN) "Use the default tactile feedback" else "使用默认触觉反馈"
            VibrationStrength.STRONG -> if (language == AppLanguage.EN) "Strengthen near and critical reminders" else "增强近处和迫近提醒"
        }
    }

    fun dailyUsageModeName(mode: DailyUsageMode, language: AppLanguage): String {
        return when (mode) {
            DailyUsageMode.GENERAL_DAILY -> if (language == AppLanguage.EN) "Daily" else "通用日常"
            DailyUsageMode.INDOOR_SLOW -> if (language == AppLanguage.EN) "Indoor slow walk" else "室内慢行"
            DailyUsageMode.CORRIDOR -> if (language == AppLanguage.EN) "Corridor" else "走廊通行"
            DailyUsageMode.CROWDED -> if (language == AppLanguage.EN) "Crowded area" else "密集区域"
            DailyUsageMode.OUTDOOR_SLOW -> if (language == AppLanguage.EN) "Outdoor slow walk" else "户外慢行"
            DailyUsageMode.CUSTOM -> if (language == AppLanguage.EN) "Custom" else "自定义"
        }
    }

    fun dailyUsageModeDescription(mode: DailyUsageMode, language: AppLanguage): String {
        return when (mode) {
            DailyUsageMode.GENERAL_DAILY -> if (language == AppLanguage.EN) "Balanced reminders for everyday walking" else "日常行走的平衡提醒"
            DailyUsageMode.INDOOR_SLOW -> if (language == AppLanguage.EN) "Quieter prompts for slow indoor movement" else "适合室内慢行的低打扰提醒"
            DailyUsageMode.CORRIDOR -> if (language == AppLanguage.EN) "Earlier attention to sustained front risks" else "更早关注正前方持续风险"
            DailyUsageMode.CROWDED -> if (language == AppLanguage.EN) "Lower reminder fatigue in dense spaces" else "降低密集环境中的提醒疲劳"
            DailyUsageMode.OUTDOOR_SLOW -> if (language == AppLanguage.EN) "Clearer tactile prompts for outdoor slow walking" else "户外慢行时强化清晰触觉提醒"
            DailyUsageMode.CUSTOM -> if (language == AppLanguage.EN) "Manual preference combination" else "手动调整后的偏好组合"
        }
    }

    fun dailyUsageModeAccessibility(mode: DailyUsageMode, language: AppLanguage): String {
        val config = mode.config
        if (config == null) {
            return if (language == AppLanguage.EN) {
                "Custom daily mode, current preferences were manually adjusted"
            } else {
                "自定义日常模式，当前偏好已手动调整"
            }
        }
        return if (language == AppLanguage.EN) {
            "Choose ${dailyUsageModeName(mode, language)} daily mode, ${dailyUsageModeDescription(mode, language)}. " +
                "Profile ${alertProfileName(config.profile, language)}, scenario ${assistScenarioName(config.scenario, language)}, " +
                "speech ${speechStyleName(config.speechStyle, language)}, vibration ${vibrationStrengthName(config.vibrationStrength, language)}, " +
                "Care Mode ${enabled(config.careModeEnabled, language)}."
        } else {
            "选择${dailyUsageModeName(mode, language)}日常模式，${dailyUsageModeDescription(mode, language)}。" +
                "提醒档位${alertProfileName(config.profile, language)}，场景${assistScenarioName(config.scenario, language)}，" +
                "语音${speechStyleName(config.speechStyle, language)}，震动${vibrationStrengthName(config.vibrationStrength, language)}，" +
                "关怀模式${enabled(config.careModeEnabled, language)}。"
        }
    }

    fun feedbackReason(reason: FeedbackReason, language: AppLanguage): String {
        return when (reason) {
            FeedbackReason.TRIGGERED -> if (language == AppLanguage.EN) "Feedback triggered" else "已触发反馈"
            FeedbackReason.HELD_ALERT -> if (language == AppLanguage.EN) "Reminder held" else "提醒保持"
            FeedbackReason.DISTANCE_TOO_FAR -> if (language == AppLanguage.EN) "Farther away" else "距离较远"
            FeedbackReason.UNSTABLE_RISK -> if (language == AppLanguage.EN) "Risk not stable" else "风险未稳定"
            FeedbackReason.COOLDOWN -> if (language == AppLanguage.EN) "Cooling down" else "冷却中"
            FeedbackReason.SPEECH_DISABLED -> if (language == AppLanguage.EN) "Speech off" else "语音关闭"
            FeedbackReason.VIBRATION_DISABLED -> if (language == AppLanguage.EN) "Vibration off" else "震动关闭"
            FeedbackReason.FEEDBACK_UNAVAILABLE -> if (language == AppLanguage.EN) "Feedback unavailable" else "反馈不可用"
            FeedbackReason.NO_FEEDBACK_RISK -> if (language == AppLanguage.EN) "No feedback risk" else "无可反馈风险"
        }
    }

    fun level(level: RiskLevel, language: AppLanguage): String {
        return when (level) {
            RiskLevel.HIGH -> if (language == AppLanguage.EN) "High risk" else "高风险"
            RiskLevel.MEDIUM -> if (language == AppLanguage.EN) "Medium risk" else "中风险"
            RiskLevel.LOW -> if (language == AppLanguage.EN) "Low risk" else "低风险"
            RiskLevel.NONE -> if (language == AppLanguage.EN) "Safe" else "安全"
        }
    }

    fun direction(direction: RiskDirection, language: AppLanguage, short: Boolean = false): String {
        return when (direction) {
            RiskDirection.LEFT -> if (language == AppLanguage.EN) "front left" else "左前"
            RiskDirection.CENTER -> if (language == AppLanguage.EN) "front center" else if (short) "正前" else "正前方"
            RiskDirection.RIGHT -> if (language == AppLanguage.EN) "front right" else "右前"
            RiskDirection.NONE -> if (language == AppLanguage.EN) "ahead" else if (short) "前方" else "前方"
        }
    }

    fun proximity(proximity: ProximityBand, language: AppLanguage): String {
        return when (proximity) {
            ProximityBand.CRITICAL -> if (language == AppLanguage.EN) "critical" else "迫近"
            ProximityBand.NEAR -> if (language == AppLanguage.EN) "near" else "近处"
            ProximityBand.MID -> if (language == AppLanguage.EN) "mid range" else "中距"
            ProximityBand.FAR -> if (language == AppLanguage.EN) "far" else "远处"
        }
    }

    fun objectName(label: String?, language: AppLanguage): String? {
        return when (label) {
            null -> null
            "person" -> if (language == AppLanguage.EN) "person" else "人"
            "car", "bus", "truck", "motorcycle", "bicycle" -> if (language == AppLanguage.EN) "vehicle" else "车辆"
            "bench", "chair", "potted plant" -> if (language == AppLanguage.EN) "obstacle" else "障碍"
            "traffic light" -> if (language == AppLanguage.EN) "traffic light" else "交通灯"
            "stop sign" -> if (language == AppLanguage.EN) "stop sign" else "停止标志"
            else -> if (language == AppLanguage.EN) label else label
        }
    }

    fun durationText(hasStarted: Boolean, durationMs: Long, language: AppLanguage): String {
        if (!hasStarted) return if (language == AppLanguage.EN) "not started" else "尚未开始"
        val totalSeconds = durationMs / 1000L
        val minutes = totalSeconds / 60L
        val seconds = totalSeconds % 60L
        return if (language == AppLanguage.EN) {
            if (minutes > 0L) "${minutes}m ${seconds}s" else "${seconds}s"
        } else {
            if (minutes > 0L) "${minutes}分${seconds}秒" else "${seconds}秒"
        }
    }
}
