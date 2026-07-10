package com.linnan.blindassist.ui.compose

import com.linnan.blindassist.alert.AlertProfile
import com.linnan.blindassist.alert.AssistScenario
import com.linnan.blindassist.feedback.SpeechStyle
import com.linnan.blindassist.feedback.VibrationStrength
import com.linnan.blindassist.localization.AppLanguage
import com.linnan.blindassist.model.AssistInputSource
import com.linnan.blindassist.model.ReplayScenario
import com.linnan.blindassist.preferences.DailyUsageMode

enum class GlassesConnectionState {
    DISCONNECTED,
    CONNECTING,
    CONNECTED,
    CONNECTION_LOST
}

data class GlassesSimulatorUiState(
    val connectionState: GlassesConnectionState = GlassesConnectionState.DISCONNECTED,
    val batteryPercent: Int? = null,
    val selectedInput: AssistInputSource = AssistInputSource.PHONE_CAMERA,
    val selectedReplayScenario: ReplayScenario = ReplayScenario.HIGH_CENTER,
    val debugReplayAvailable: Boolean = false
)

data class AssistControlsUiState(
    val detectionEnabled: Boolean,
    val speechEnabled: Boolean,
    val vibrationEnabled: Boolean,
    val careModeEnabled: Boolean,
    val debugVisible: Boolean,
    val alertProfile: AlertProfile,
    val assistScenario: AssistScenario,
    val speechStyle: SpeechStyle,
    val vibrationStrength: VibrationStrength,
    val appLanguage: AppLanguage,
    val dailyUsageMode: DailyUsageMode
)

data class CameraGuidanceUiState(
    val title: String,
    val detail: String,
    val targetLine: String,
    val careTitle: String,
    val careDetail: String,
    val careTargetLine: String,
    val debugText: String,
    val scenarioName: String,
    val explanationHeadline: String,
    val explanationDetail: String,
    val careExplanation: String,
    val titleColor: Int,
    val statusBadge: String,
    val badgeColor: Int,
    val badgeTextColor: Int,
    val careAccessibilitySummary: String,
    val accessibilitySummary: String,
    val accessibilityKey: String
) {
    companion object {
        fun initial(
            modelStatus: String,
            scenarioName: String = AssistScenario.GENERAL.displayName,
            language: AppLanguage = AppLanguage.ZH
        ): CameraGuidanceUiState {
            val english = language == AppLanguage.EN
            return CameraGuidanceUiState(
                title = if (english) "Initializing" else "初始化中",
                detail = if (english) "Preparing the local detection model" else "正在准备本地检测模型",
                targetLine = if (english) "Model status: $modelStatus" else "模型状态：$modelStatus",
                careTitle = if (english) "Preparing" else "正在准备",
                careDetail = if (english) "Detection model is starting, please wait" else "识别模型正在启动，请稍等",
                careTargetLine = if (english) "The phone camera will observe ahead after entry" else "进入手机摄像头后会开始观察前方",
                debugText = if (english) "Model status: $modelStatus" else "模型状态：$modelStatus",
                scenarioName = scenarioName,
                explanationHeadline = if (english) "No risk explanation yet" else "暂无风险解释",
                explanationDetail = if (english) "After a camera session starts, the latest reminder reason will appear here." else "进入相机会话后会显示最近一次提醒或暂不提醒的原因。",
                careExplanation = if (english) "The reminder reason will appear after the camera starts" else "进入相机会话后会说明提醒原因",
                titleColor = -1,
                statusBadge = if (english) "Preparing" else "准备中",
                badgeColor = rgb(206, 221, 235),
                badgeTextColor = rgb(10, 22, 32),
                careAccessibilitySummary = if (english) "Preparing, the detection model is starting" else "正在准备，识别模型正在启动",
                accessibilitySummary = if (english) "Initializing, preparing the local detection model" else "初始化中，正在准备本地检测模型",
                accessibilityKey = "initial"
            )
        }

        private fun rgb(red: Int, green: Int, blue: Int): Int {
            return (0xFF shl 24) or (red shl 16) or (green shl 8) or blue
        }
    }
}

data class FieldTestSummaryUiState(
    val title: String,
    val detailText: String,
    val statusText: String,
    val accessibilityText: String
) {
    companion object {
        fun empty(profileName: String, scenarioName: String, language: AppLanguage = AppLanguage.ZH): FieldTestSummaryUiState {
            val english = language == AppLanguage.EN
            return FieldTestSummaryUiState(
                title = if (english) "Field test summary" else "现场测试摘要",
                detailText = if (english) {
                    "Runtime: not started\nLast 0 frames: 0 risky, 0 critical, high/medium/low/none 0/0/0/0\nReminders: speech 0, vibration 0\nAverage performance: FPS 0.0, inference 0ms\nCurrent profile: $profileName\nCurrent scenario: $scenarioName\nLatest explanation: none yet"
                } else {
                    "运行时长：尚未开始\n最近0帧：风险0次，迫近0次，高/中/低/无 0/0/0/0\n提醒触发：语音0次，震动0次\n平均性能：FPS 0.0，推理 0ms\n当前档位：$profileName\n当前场景：$scenarioName\n最近解释：暂无风险解释"
                },
                statusText = if (english) "Waiting for camera session" else "等待相机会话",
                accessibilityText = if (english) {
                    "Field test summary, not started. After a camera session runs, runtime, risk counts, reminder counts, average performance, current profile, current scenario, and the latest explanation will appear."
                } else {
                    "现场测试摘要，尚未开始，相机会话运行后会显示运行时长、风险次数、提醒次数、平均性能、当前档位、当前场景和最近解释。"
                }
            )
        }
    }
}
