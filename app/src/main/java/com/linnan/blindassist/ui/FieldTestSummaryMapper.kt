package com.linnan.blindassist.ui

import com.linnan.blindassist.alert.AlertProfile
import com.linnan.blindassist.alert.AssistScenario
import com.linnan.blindassist.session.SessionSummary
import com.linnan.blindassist.ui.compose.FieldTestSummaryUiState

object FieldTestSummaryMapper {
    fun fromSummary(
        summary: SessionSummary,
        active: Boolean,
        profile: AlertProfile,
        scenario: AssistScenario
    ): FieldTestSummaryUiState {
        val status = when {
            active && summary.frameCount > 0 -> "本次相机会话进行中"
            active -> "本次相机会话已开始，等待检测帧"
            summary.hasStarted || summary.frameCount > 0 -> "上一场相机会话摘要"
            else -> "等待相机会话"
        }
        return FieldTestSummaryUiState(
            title = "现场测试摘要",
            detailText = summary.fieldTestText(profile.displayName, scenario.displayName),
            statusText = status,
            accessibilityText = "现场测试摘要，$status，运行时长${summary.durationText()}，" +
                "最近${summary.frameCount}帧风险${summary.riskyFrameCount}次，" +
                "语音提醒${summary.speechTriggerCount}次，震动提醒${summary.vibrationTriggerCount}次，" +
                "平均FPS ${"%.1f".format(summary.averageFps)}，平均推理${summary.averageInferenceMs}毫秒，" +
                "当前档位${profile.displayName}，当前场景${scenario.displayName}，最近解释${summary.latestExplanation}。"
        )
    }
}
