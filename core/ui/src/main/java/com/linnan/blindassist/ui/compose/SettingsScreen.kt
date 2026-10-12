package com.linnan.blindassist.ui.compose

import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.height
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.outlined.BugReport
import androidx.compose.material.icons.outlined.FavoriteBorder
import androidx.compose.material.icons.outlined.Info
import androidx.compose.material.icons.outlined.Shield
import androidx.compose.material.icons.outlined.Vibration
import androidx.compose.material.icons.automirrored.outlined.VolumeUp
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
                "Language, feedback, reminder cadence and Care Mode apply next time you start assistance. Legacy-only settings are marked below."
            } else {
                "语言、提醒方式、提醒频率和关怀模式在下次开始辅助时生效。仅适用于旧版相机的选项会单独标明。"
            }
        )
        Spacer(Modifier.height(26.dp))

        SettingsSectionHeader(if (language == AppLanguage.EN) "Interface and assistance" else "界面与辅助")
        LanguageSelector(
            selected = controls.appLanguage,
            onLanguageChange = onLanguageChange
        )
        Spacer(Modifier.height(8.dp))
        SettingSwitchRow(
            icon = Icons.Outlined.FavoriteBorder,
            title = if (language == AppLanguage.EN) "Care Mode" else "关怀模式",
            body = if (language == AppLanguage.EN) "Use larger guidance text during assistance" else "辅助时使用更大的状态与提醒文字",
            checked = controls.careModeEnabled,
            language = language,
            modifier = Modifier.testTag("settings_care_mode_toggle"),
            onCheckedChange = onCareModeChange
        )
        Spacer(Modifier.height(24.dp))

        SettingsSectionHeader(if (language == AppLanguage.EN) "Reminder feedback" else "提醒方式")
        SettingsGroup {
            SettingSwitchRow(
                icon = Icons.AutoMirrored.Outlined.VolumeUp,
                title = if (language == AppLanguage.EN) "Speech reminders" else "语音提醒",
                body = if (language == AppLanguage.EN) "Speak short risk prompts" else "播报短句式风险提示",
                checked = controls.speechEnabled,
                language = language,
                modifier = Modifier.testTag("settings_speech_toggle"),
                onCheckedChange = onSpeechChange
            )
            SettingsDivider()
            SettingSwitchRow(
                icon = Icons.Outlined.Vibration,
                title = if (language == AppLanguage.EN) "Vibration reminders" else "震动提醒",
                body = if (language == AppLanguage.EN) "Give tactile feedback for near and critical risks" else "在近处和迫近风险时给出触觉反馈",
                checked = controls.vibrationEnabled,
                language = language,
                modifier = Modifier.testTag("settings_vibration_toggle"),
                onCheckedChange = onVibrationChange
            )
        }
        Spacer(Modifier.height(18.dp))
        SpeechStyleSelector(
            selected = controls.speechStyle,
            language = language,
            onSpeechStyleChange = onSpeechStyleChange
        )
        Spacer(Modifier.height(18.dp))
        VibrationStrengthSelector(
            selected = controls.vibrationStrength,
            language = language,
            onVibrationStrengthChange = onVibrationStrengthChange
        )
        Spacer(Modifier.height(24.dp))

        SettingsSectionHeader(if (language == AppLanguage.EN) "Reminder cadence" else "提醒频率")
        ProfileSelector(
            selected = controls.alertProfile,
            language = language,
            onProfileChange = onProfileChange
        )
        Spacer(Modifier.height(20.dp))
        SettingsSectionHeader(if (language == AppLanguage.EN) "Legacy camera settings" else "旧版相机设置")
        Text(
            if (language == AppLanguage.EN) "These scene and debug options do not change Camera + ToF assistance."
            else "以下场景与调试选项不改变当前相机 + ToF 辅助。",
            style = MaterialTheme.typography.bodySmall
        )
        Spacer(Modifier.height(12.dp))
        ScenarioSelector(
            selected = controls.assistScenario,
            language = language,
            onScenarioChange = onScenarioChange
        )
        Spacer(Modifier.height(24.dp))

        SettingsSectionHeader(if (language == AppLanguage.EN) "Debug and records" else "调试与记录")
        SettingsGroup {
            SettingSwitchRow(
                icon = Icons.Outlined.BugReport,
                title = if (language == AppLanguage.EN) "Debug details" else "调试信息",
                body = if (language == AppLanguage.EN) "Legacy camera only: show FPS, timing and risk summary" else "仅旧版相机：显示 FPS、耗时和风险判定摘要",
                checked = controls.debugVisible,
                language = language,
                modifier = Modifier.testTag("settings_debug_toggle"),
                onCheckedChange = onDebugVisibleChange
            )
            SettingsDivider()
            FieldTestSummaryCard(fieldTestSummary)
            SettingsDivider()
            SettingsActionRow(
                icon = Icons.Outlined.Info,
                title = if (language == AppLanguage.EN) "Replay onboarding" else "查看新手引导",
                body = if (language == AppLanguage.EN) "Review camera, local reminders, and safety boundaries" else "重新查看摄像头、本地提醒和安全边界说明",
                onClick = onShowOnboarding
            )
        }
        Spacer(Modifier.height(22.dp))
        InfoStrip(
            icon = Icons.Outlined.Shield,
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
        text = text,
        style = MaterialTheme.typography.titleMedium,
        color = BaHomeInk,
        fontWeight = FontWeight.SemiBold,
        modifier = Modifier.semantics { heading() }
    )
    Spacer(Modifier.height(12.dp))
}
