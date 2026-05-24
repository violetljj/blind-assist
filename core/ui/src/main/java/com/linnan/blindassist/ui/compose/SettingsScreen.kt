package com.linnan.blindassist.ui.compose

import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.height
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.rounded.BugReport
import androidx.compose.material.icons.rounded.Favorite
import androidx.compose.material.icons.rounded.Info
import androidx.compose.material.icons.rounded.Shield
import androidx.compose.material.icons.rounded.Vibration
import androidx.compose.material.icons.rounded.VolumeUp
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.semantics.heading
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.linnan.blindassist.alert.AlertProfile
import com.linnan.blindassist.alert.AssistScenario
import com.linnan.blindassist.feedback.SpeechStyle
import com.linnan.blindassist.feedback.VibrationStrength
import com.linnan.blindassist.localization.AppLanguage


@Composable
fun SettingsScreen(
    controls: AssistControlsUiState,
    fieldTestSummary: FieldTestSummaryUiState,
    onSpeechChange: (Boolean) -> Unit,
    onVibrationChange: (Boolean) -> Unit,
    onCareModeChange: (Boolean) -> Unit,
    onDebugVisibleChange: (Boolean) -> Unit,
    onProfileChange: (AlertProfile) -> Unit,
    onScenarioChange: (AssistScenario) -> Unit,
    onSpeechStyleChange: (SpeechStyle) -> Unit,
    onVibrationStrengthChange: (VibrationStrength) -> Unit,
    onLanguageChange: (AppLanguage) -> Unit,
    onShowOnboarding: () -> Unit,
    modifier: Modifier = Modifier
) {
    val language = controls.appLanguage
    ScreenColumn(modifier = modifier) {
        Text(
            text = if (language == AppLanguage.EN) "Settings" else "设置",
            style = MaterialTheme.typography.headlineMedium,
            color = BaText,
            fontWeight = FontWeight.Bold,
            modifier = Modifier.semantics { heading() }
        )
        Text(
            text = if (language == AppLanguage.EN) {
                "These preferences affect reminders on the camera page. Detection can still be controlled after entering the camera."
            } else {
                "这些偏好会影响摄像头页的提醒方式。检测开关进入相机后仍可随时控制。"
            },
            style = MaterialTheme.typography.bodyMedium,
            color = BaTextMuted
        )
        Spacer(Modifier.height(18.dp))

        LanguageSelector(
            selected = controls.appLanguage,
            onLanguageChange = onLanguageChange
        )
        Spacer(Modifier.height(16.dp))

        SettingSwitchRow(
            icon = Icons.Rounded.VolumeUp,
            title = if (language == AppLanguage.EN) "Speech reminders" else "语音提醒",
            body = if (language == AppLanguage.EN) "Speak short risk prompts" else "播报短句式风险提示",
            checked = controls.speechEnabled,
            language = language,
            onCheckedChange = onSpeechChange
        )
        SettingSwitchRow(
            icon = Icons.Rounded.Vibration,
            title = if (language == AppLanguage.EN) "Vibration reminders" else "震动提醒",
            body = if (language == AppLanguage.EN) "Give tactile feedback for near and critical risks" else "在近处和迫近风险时给出触觉反馈",
            checked = controls.vibrationEnabled,
            language = language,
            onCheckedChange = onVibrationChange
        )
        SettingSwitchRow(
            icon = Icons.Rounded.Favorite,
            title = if (language == AppLanguage.EN) "Care Mode" else "关怀模式",
            body = if (language == AppLanguage.EN) "Enlarge main guidance and reduce debug noise" else "放大主要指导语并减少调试干扰",
            checked = controls.careModeEnabled,
            language = language,
            onCheckedChange = onCareModeChange
        )
        SettingSwitchRow(
            icon = Icons.Rounded.BugReport,
            title = if (language == AppLanguage.EN) "Debug details" else "调试信息",
            body = if (language == AppLanguage.EN) "Show FPS, timing, and risk summary on the camera page" else "在相机页显示 FPS、耗时和风险判定摘要",
            checked = controls.debugVisible,
            language = language,
            onCheckedChange = onDebugVisibleChange
        )
        Spacer(Modifier.height(16.dp))
        ProfileSelector(
            selected = controls.alertProfile,
            language = language,
            onProfileChange = onProfileChange
        )
        Spacer(Modifier.height(16.dp))
        ScenarioSelector(
            selected = controls.assistScenario,
            language = language,
            onScenarioChange = onScenarioChange
        )
        Spacer(Modifier.height(16.dp))
        SpeechStyleSelector(
            selected = controls.speechStyle,
            language = language,
            onSpeechStyleChange = onSpeechStyleChange
        )
        Spacer(Modifier.height(16.dp))
        VibrationStrengthSelector(
            selected = controls.vibrationStrength,
            language = language,
            onVibrationStrengthChange = onVibrationStrengthChange
        )
        Spacer(Modifier.height(16.dp))
        FieldTestSummaryCard(fieldTestSummary)
        Spacer(Modifier.height(16.dp))
        SettingsActionRow(
            icon = Icons.Rounded.Info,
            title = if (language == AppLanguage.EN) "Replay onboarding" else "查看新手引导",
            body = if (language == AppLanguage.EN) "Review camera, local reminders, and safety boundaries" else "重新查看摄像头、本地提醒和安全边界说明",
            onClick = onShowOnboarding
        )
        Spacer(Modifier.height(16.dp))
        InfoStrip(
            icon = Icons.Rounded.Shield,
            title = if (language == AppLanguage.EN) "Usage boundary" else "使用边界",
            body = if (language == AppLanguage.EN) {
                "Reminders are generated by a local model and rules, and can be affected by lighting, occlusion, and device performance. Keep human judgment while walking."
            } else {
                "提醒由本地模型与规则层生成，受光照、遮挡和设备性能影响。行走时仍需保留人工判断。"
            }
        )
    }
}
