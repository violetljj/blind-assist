package com.linnan.blindassist.ui.compose

import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.height
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.rounded.BugReport
import androidx.compose.material.icons.rounded.Favorite
import androidx.compose.material.icons.rounded.Info
import androidx.compose.material.icons.rounded.Shield
import androidx.compose.material.icons.rounded.Vibration
import androidx.compose.material.icons.automirrored.rounded.VolumeUp
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.testTag
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
        ScreenIntro(
            eyebrow = if (language == AppLanguage.EN) "PERSONALIZE" else "个性化辅助",
            title = if (language == AppLanguage.EN) "Settings" else "设置",
            body = if (language == AppLanguage.EN) {
                "These preferences affect reminders on the camera page. Detection can still be controlled after entering the camera."
            } else {
                "这些偏好会影响摄像头页的提醒方式。检测开关进入相机后仍可随时控制。"
            }
        )
        Spacer(Modifier.height(26.dp))

        SettingsSectionHeader(if (language == AppLanguage.EN) "Interface and assistance" else "界面与辅助")
        LanguageSelector(
            selected = controls.appLanguage,
            onLanguageChange = onLanguageChange
        )
        Spacer(Modifier.height(20.dp))
        SettingSwitchRow(
            icon = Icons.Rounded.Favorite,
            title = if (language == AppLanguage.EN) "Care Mode" else "关怀模式",
            body = if (language == AppLanguage.EN) "Use a larger, quieter camera panel with less debug noise" else "使用更大、更安静的相机面板并减少调试干扰",
            checked = controls.careModeEnabled,
            language = language,
            modifier = Modifier.testTag("settings_care_mode_toggle"),
            onCheckedChange = onCareModeChange
        )
        Spacer(Modifier.height(24.dp))

        SettingsSectionHeader(if (language == AppLanguage.EN) "Reminder feedback" else "提醒方式")
        SettingSwitchRow(
            icon = Icons.AutoMirrored.Rounded.VolumeUp,
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
        Spacer(Modifier.height(12.dp))
        SpeechStyleSelector(
            selected = controls.speechStyle,
            language = language,
            onSpeechStyleChange = onSpeechStyleChange
        )
        Spacer(Modifier.height(20.dp))
        VibrationStrengthSelector(
            selected = controls.vibrationStrength,
            language = language,
            onVibrationStrengthChange = onVibrationStrengthChange
        )
        Spacer(Modifier.height(24.dp))

        SettingsSectionHeader(if (language == AppLanguage.EN) "Walking scenario" else "行走场景")
        ProfileSelector(
            selected = controls.alertProfile,
            language = language,
            onProfileChange = onProfileChange
        )
        Spacer(Modifier.height(20.dp))
        ScenarioSelector(
            selected = controls.assistScenario,
            language = language,
            onScenarioChange = onScenarioChange
        )
        Spacer(Modifier.height(24.dp))

        SettingsSectionHeader(if (language == AppLanguage.EN) "Debug and records" else "调试与记录")
        SettingSwitchRow(
            icon = Icons.Rounded.BugReport,
            title = if (language == AppLanguage.EN) "Debug details" else "调试信息",
            body = if (language == AppLanguage.EN) "Show FPS, timing, and risk summary on the camera page" else "在相机页显示 FPS、耗时和风险判定摘要",
            checked = controls.debugVisible,
            language = language,
            modifier = Modifier.testTag("settings_debug_toggle"),
            onCheckedChange = onDebugVisibleChange
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

@Composable
private fun SettingsSectionHeader(text: String) {
    Text(
        text = text.uppercase(),
        style = MaterialTheme.typography.labelMedium,
        color = BaMint,
        fontWeight = FontWeight.Bold,
        modifier = Modifier.semantics { heading() }
    )
    Spacer(Modifier.height(10.dp))
}
