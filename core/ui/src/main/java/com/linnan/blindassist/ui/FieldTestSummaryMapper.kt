package com.linnan.blindassist.ui

import com.linnan.blindassist.alert.AlertProfile
import com.linnan.blindassist.alert.AssistScenario
import com.linnan.blindassist.localization.AppLanguage
import com.linnan.blindassist.session.SessionSummary
import com.linnan.blindassist.ui.compose.FieldTestSummaryUiState
import java.util.Locale

object FieldTestSummaryMapper {
    fun fromSummary(
        summary: SessionSummary,
        active: Boolean,
        profile: AlertProfile,
        scenario: AssistScenario,
        language: AppLanguage = AppLanguage.ZH
    ): FieldTestSummaryUiState {
        val english = language == AppLanguage.EN
        val status = when {
            active && summary.frameCount > 0 -> if (english) "Current camera session running" else "本次相机会话进行中"
            active -> if (english) "Camera session started, waiting for detection frames" else "本次相机会话已开始，等待检测帧"
            summary.hasStarted || summary.frameCount > 0 -> if (english) "Previous camera session summary" else "上一场相机会话摘要"
            else -> if (english) "Waiting for camera session" else "等待相机会话"
        }
        val profileName = profile.displayName(language)
        val scenarioName = scenario.displayName(language)
        return FieldTestSummaryUiState(
            title = if (english) "Field test summary" else "现场测试摘要",
            detailText = summary.fieldTestText(profileName, scenarioName, language),
            statusText = status,
            accessibilityText = if (english) {
                "Field test summary, $status, runtime ${summary.durationText(language)}, " +
                    "last ${summary.frameCount} frames include ${summary.riskyFrameCount} risky frames, " +
                    "speech reminders ${summary.speechTriggerCount}, vibration reminders ${summary.vibrationTriggerCount}, " +
                    "average FPS ${String.format(Locale.US, "%.1f", summary.averageFps)}, average inference ${summary.averageInferenceMs} milliseconds, " +
                    "current profile $profileName, current scenario $scenarioName."
            } else {
                "现场测试摘要，$status，运行时长${summary.durationText()}，" +
                    "最近${summary.frameCount}帧风险${summary.riskyFrameCount}次，" +
                    "语音提醒${summary.speechTriggerCount}次，震动提醒${summary.vibrationTriggerCount}次，" +
                    "平均FPS ${"%.1f".format(summary.averageFps)}，平均推理${summary.averageInferenceMs}毫秒，" +
                    "当前档位$profileName，当前场景$scenarioName，最近解释${summary.latestExplanation}。"
            }
        )
    }
}
