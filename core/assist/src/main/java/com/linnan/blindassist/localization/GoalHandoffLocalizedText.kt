package com.linnan.blindassist.localization

import com.linnan.blindassist.goal.GoalHandoffState

/** Product copy kept outside the composable so language selection and copy tests stay deterministic. */
object GoalHandoffLocalizedText {
    fun message(state: GoalHandoffState, language: AppLanguage): String? {
        val english = language == AppLanguage.EN
        return when (state) {
            GoalHandoffState.Inactive -> null
            is GoalHandoffState.Found -> if (english) {
                "Target found, ahead to your right."
            } else {
                "找到目标，在你的右前方。"
            }
            is GoalHandoffState.Approach -> if (english) {
                "Bear slightly right and continue forward."
            } else {
                "稍向右，继续向前。"
            }
            is GoalHandoffState.HandoffReady -> if (english) {
                "You are in front of the target. Confirm the entrance with your hand or cane."
            } else {
                "已经到目标前，请用手或盲杖确认入口。"
            }
            is GoalHandoffState.CompletedByUser -> if (english) {
                "You confirmed that you found it."
            } else {
                "你已确认找到了。"
            }
        }
    }

    fun stateDescription(state: GoalHandoffState, language: AppLanguage): String? {
        val english = language == AppLanguage.EN
        return when (state) {
            GoalHandoffState.Inactive -> null
            is GoalHandoffState.Found -> if (english) "Target found" else "已找到目标"
            is GoalHandoffState.Approach -> if (english) "Approaching target" else "正在接近目标"
            is GoalHandoffState.HandoffReady -> if (english) {
                "Handoff ready, waiting for explicit user confirmation"
            } else {
                "已到交接点，等待用户明确确认"
            }
            is GoalHandoffState.CompletedByUser -> if (english) {
                "Completed by explicit user confirmation"
            } else {
                "已由用户明确确认完成"
            }
        }
    }

    fun confirmationButton(language: AppLanguage): String {
        return if (language == AppLanguage.EN) "Found it" else "找到了"
    }

    fun confirmationButtonDescription(language: AppLanguage): String {
        return if (language == AppLanguage.EN) {
            "Confirm that you found the target"
        } else {
            "确认已找到目标"
        }
    }
}
