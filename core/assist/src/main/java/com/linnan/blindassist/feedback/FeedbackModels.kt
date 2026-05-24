package com.linnan.blindassist.feedback

import com.linnan.blindassist.localization.AppLanguage
import com.linnan.blindassist.localization.LocalizedText

data class FeedbackPlan(
    val cooldownMs: Long,
    val vibrationMs: Long,
    val amplitude: Int
)

data class FeedbackDecision(
    val plan: FeedbackPlan?,
    val triggered: Boolean,
    val reason: FeedbackReason,
    val speechTriggered: Boolean = false,
    val vibrationTriggered: Boolean = false
) {
    fun withDisplayReason(displayReason: FeedbackReason): FeedbackDecision {
        return if (triggered) this else copy(reason = displayReason)
    }
}

enum class FeedbackReason(val displayText: String) {
    TRIGGERED("已触发反馈"),
    HELD_ALERT("提醒保持"),
    DISTANCE_TOO_FAR("距离较远"),
    UNSTABLE_RISK("风险未稳定"),
    COOLDOWN("冷却中"),
    SPEECH_DISABLED("语音关闭"),
    VIBRATION_DISABLED("震动关闭"),
    FEEDBACK_UNAVAILABLE("反馈不可用"),
    NO_FEEDBACK_RISK("无可反馈风险");

    fun displayText(language: AppLanguage): String {
        return LocalizedText.feedbackReason(this, language)
    }
}
