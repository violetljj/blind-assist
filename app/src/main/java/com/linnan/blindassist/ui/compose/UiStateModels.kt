package com.linnan.blindassist.ui.compose

import com.linnan.blindassist.alert.AlertProfile
import com.linnan.blindassist.alert.AssistScenario
import com.linnan.blindassist.feedback.SpeechStyle
import com.linnan.blindassist.feedback.VibrationStrength

data class AssistControlsUiState(
    val detectionEnabled: Boolean,
    val speechEnabled: Boolean,
    val vibrationEnabled: Boolean,
    val careModeEnabled: Boolean,
    val debugVisible: Boolean,
    val alertProfile: AlertProfile,
    val assistScenario: AssistScenario,
    val speechStyle: SpeechStyle,
    val vibrationStrength: VibrationStrength
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
        fun initial(modelStatus: String, scenarioName: String = AssistScenario.GENERAL.displayName): CameraGuidanceUiState {
            return CameraGuidanceUiState(
                title = "初始化中",
                detail = "正在准备本地检测模型",
                targetLine = "模型状态：$modelStatus",
                careTitle = "正在准备",
                careDetail = "识别模型正在启动，请稍等",
                careTargetLine = "进入手机摄像头后会开始观察前方",
                debugText = "模型状态：$modelStatus",
                scenarioName = scenarioName,
                explanationHeadline = "暂无风险解释",
                explanationDetail = "进入相机会话后会显示最近一次提醒或暂不提醒的原因。",
                careExplanation = "进入相机会话后会说明提醒原因",
                titleColor = -1,
                statusBadge = "准备中",
                badgeColor = rgb(206, 221, 235),
                badgeTextColor = rgb(10, 22, 32),
                careAccessibilitySummary = "正在准备，识别模型正在启动",
                accessibilitySummary = "初始化中，正在准备本地检测模型",
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
        fun empty(profileName: String, scenarioName: String): FieldTestSummaryUiState {
            return FieldTestSummaryUiState(
                title = "现场测试摘要",
                detailText = "运行时长：尚未开始\n最近0帧：风险0次，迫近0次，高/中/低/无 0/0/0/0\n提醒触发：语音0次，震动0次\n平均性能：FPS 0.0，推理 0ms\n当前档位：$profileName\n当前场景：$scenarioName\n最近解释：暂无风险解释",
                statusText = "等待相机会话",
                accessibilityText = "现场测试摘要，尚未开始，相机会话运行后会显示运行时长、风险次数、提醒次数、平均性能、当前档位、当前场景和最近解释。"
            )
        }
    }
}
