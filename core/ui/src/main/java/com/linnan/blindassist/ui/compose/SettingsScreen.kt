package com.linnan.blindassist.ui.compose

import android.view.ViewGroup
import android.widget.FrameLayout
import androidx.activity.compose.BackHandler
import androidx.camera.view.PreviewView
import androidx.compose.animation.AnimatedContent
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.core.FastOutSlowInEasing
import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.slideInVertically
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ColumnScope
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.rounded.ArrowBack
import androidx.compose.material.icons.rounded.Bluetooth
import androidx.compose.material.icons.rounded.BugReport
import androidx.compose.material.icons.rounded.CameraAlt
import androidx.compose.material.icons.rounded.ChevronRight
import androidx.compose.material.icons.rounded.Favorite
import androidx.compose.material.icons.rounded.Home
import androidx.compose.material.icons.rounded.Info
import androidx.compose.material.icons.rounded.Person
import androidx.compose.material.icons.rounded.PhoneAndroid
import androidx.compose.material.icons.rounded.Settings
import androidx.compose.material.icons.rounded.Shield
import androidx.compose.material.icons.rounded.Tune
import androidx.compose.material.icons.rounded.Vibration
import androidx.compose.material.icons.rounded.Visibility
import androidx.compose.material.icons.rounded.VolumeUp
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FilterChip
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.NavigationBar
import androidx.compose.material3.NavigationBarItem
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Surface
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.heading
import androidx.compose.ui.semantics.role
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.semantics.stateDescription
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import com.linnan.blindassist.alert.AlertProfile
import com.linnan.blindassist.alert.AssistScenario
import com.linnan.blindassist.feedback.SpeechStyle
import com.linnan.blindassist.feedback.VibrationStrength
import com.linnan.blindassist.localization.AppLanguage
import com.linnan.blindassist.localization.LocalizedText
import com.linnan.blindassist.preferences.DailyUsageMode
import com.linnan.blindassist.ui.DetectionOverlayView
import kotlinx.coroutines.delay


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


@Composable
private fun SettingSwitchRow(
    icon: ImageVector,
    title: String,
    body: String,
    checked: Boolean,
    language: AppLanguage = AppLanguage.ZH,
    onCheckedChange: (Boolean) -> Unit
) {
    val stateText = LocalizedText.enabled(checked, language)
    val actionText = if (language == AppLanguage.EN) {
        if (checked) "Turn off $title" else "Turn on $title"
    } else {
        if (checked) "关闭$title" else "开启$title"
    }
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .heightIn(min = 72.dp)
            .clickable(
                role = Role.Switch,
                onClickLabel = actionText,
                onClick = { onCheckedChange(!checked) }
            )
            .semantics(mergeDescendants = true) {
                stateDescription = stateText
                contentDescription = if (language == AppLanguage.EN) {
                    "$title, $body, currently $stateText"
                } else {
                    "$title，$body，当前$stateText"
                }
            }
            .padding(vertical = 6.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        Icon(icon, contentDescription = null, tint = if (checked) BaMint else BaTextMuted)
        Spacer(Modifier.width(14.dp))
        Column(Modifier.weight(1f)) {
            Text(title, color = BaText, fontWeight = FontWeight.Bold)
            Text(body, color = BaTextMuted, style = MaterialTheme.typography.bodySmall)
        }
        Switch(
            checked = checked,
            onCheckedChange = onCheckedChange,
            modifier = Modifier.semantics {
                role = Role.Switch
                stateDescription = stateText
            }
        )
    }
}

@Composable
private fun SettingsActionRow(
    icon: ImageVector,
    title: String,
    body: String,
    onClick: () -> Unit
) {
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .heightIn(min = 76.dp)
            .clickable(role = Role.Button, onClick = onClick)
            .semantics(mergeDescendants = true) {},
        shape = RoundedCornerShape(16.dp),
        colors = CardDefaults.cardColors(containerColor = BaPanel)
    ) {
        Row(
            modifier = Modifier.padding(16.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Icon(icon, contentDescription = null, tint = BaMint)
            Spacer(Modifier.width(14.dp))
            Column(Modifier.weight(1f)) {
                Text(title, color = BaText, fontWeight = FontWeight.Bold)
                Text(body, color = BaTextMuted, style = MaterialTheme.typography.bodySmall)
            }
            Icon(Icons.Rounded.ChevronRight, contentDescription = null, tint = BaTextMuted)
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun ProfileSelector(
    selected: AlertProfile,
    language: AppLanguage,
    onProfileChange: (AlertProfile) -> Unit
) {
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .semantics {
                contentDescription = if (language == AppLanguage.EN) {
                    "Reminder profile, current ${selected.displayName(language)}. Quiet reduces interruption, Sensitive confirms medium risk earlier."
                } else {
                    "提醒档位，当前${selected.displayName(language)}。安静减少打扰，敏感更早确认中风险。"
                }
            },
        shape = RoundedCornerShape(16.dp),
        colors = CardDefaults.cardColors(containerColor = BaPanel)
    ) {
        Column(Modifier.padding(16.dp)) {
            Text(if (language == AppLanguage.EN) "Reminder profile" else "提醒档位", color = BaText, fontWeight = FontWeight.Bold, modifier = Modifier.semantics { heading() })
            Text(if (language == AppLanguage.EN) "Quiet reduces interruption, Sensitive confirms medium risk earlier." else "安静减少打扰，敏感更早确认中风险。", color = BaTextMuted, style = MaterialTheme.typography.bodySmall)
            Spacer(Modifier.height(10.dp))
            Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                AlertProfile.values().forEach { profile ->
                    FilterChip(
                        selected = selected == profile,
                        onClick = { onProfileChange(profile) },
                        label = { Text(profile.displayName(language)) },
                        modifier = Modifier.fillMaxWidth().heightIn(min = 48.dp).semantics {
                            role = Role.Button
                            stateDescription = if (selected == profile) {
                                if (language == AppLanguage.EN) "Current profile" else "当前档位"
                            } else {
                                if (language == AppLanguage.EN) "Not selected" else "未选择"
                            }
                            contentDescription = if (language == AppLanguage.EN) {
                                "Choose ${profile.displayName(language)} reminder profile"
                            } else {
                                "选择${profile.displayName(language)}提醒档位"
                            }
                        }
                    )
                }
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun LanguageSelector(
    selected: AppLanguage,
    onLanguageChange: (AppLanguage) -> Unit
) {
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .testTag("language_selector")
            .semantics {
                contentDescription = if (selected == AppLanguage.EN) {
                    "Interface language, current English"
                } else {
                    "界面语言，当前中文"
                }
            },
        shape = RoundedCornerShape(16.dp),
        colors = CardDefaults.cardColors(containerColor = BaPanel)
    ) {
        Column(Modifier.padding(16.dp)) {
            Text(if (selected == AppLanguage.EN) "Interface language" else "界面语言", color = BaText, fontWeight = FontWeight.Bold, modifier = Modifier.semantics { heading() })
            Text(if (selected == AppLanguage.EN) "Choose Chinese or English for core reminders and settings." else "选择核心提醒和设置界面的中文或英文。", color = BaTextMuted, style = MaterialTheme.typography.bodySmall)
            Spacer(Modifier.height(10.dp))
            Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                AppLanguage.values().forEach { language ->
                    val isEnglishUi = selected == AppLanguage.EN
                    val name = language.displayName(selected)
                    FilterChip(
                        selected = selected == language,
                        onClick = { onLanguageChange(language) },
                        label = { Text(name) },
                        modifier = Modifier
                            .fillMaxWidth()
                            .heightIn(min = 48.dp)
                            .semantics {
                                role = Role.Button
                                stateDescription = if (selected == language) {
                                    if (isEnglishUi) "Current language" else "当前语言"
                                } else {
                                    if (isEnglishUi) "Not selected" else "未选择"
                                }
                                contentDescription = if (isEnglishUi) {
                                    "Choose $name interface language"
                                } else {
                                    "选择$name 界面语言"
                                }
                            }
                    )
                }
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun ScenarioSelector(
    selected: AssistScenario,
    language: AppLanguage,
    onScenarioChange: (AssistScenario) -> Unit
) {
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .testTag("scenario_selector")
            .semantics {
                contentDescription = if (language == AppLanguage.EN) {
                    "Usage scenario, current ${selected.displayName(language)}. ${selected.description(language)}"
                } else {
                    "使用场景，当前${selected.displayName(language)}。${selected.description(language)}"
                }
            },
        shape = RoundedCornerShape(16.dp),
        colors = CardDefaults.cardColors(containerColor = BaPanel)
    ) {
        Column(Modifier.padding(16.dp)) {
            Text(if (language == AppLanguage.EN) "Usage scenario" else "使用场景", color = BaText, fontWeight = FontWeight.Bold, modifier = Modifier.semantics { heading() })
            Text(if (language == AppLanguage.EN) "Manually choose the walking environment to tune confirmation, cooldown, and vibration." else "手动选择行走环境，调整提醒确认、冷却和震动计划。", color = BaTextMuted, style = MaterialTheme.typography.bodySmall)
            Spacer(Modifier.height(10.dp))
            AssistScenario.values().forEach { scenario ->
                FilterChip(
                    selected = selected == scenario,
                    onClick = { onScenarioChange(scenario) },
                    label = {
                        Text(
                            "${scenario.displayName(language)} · ${scenario.description(language)}",
                            maxLines = 2,
                            overflow = TextOverflow.Ellipsis
                        )
                    },
                    modifier = Modifier
                        .fillMaxWidth()
                        .heightIn(min = 48.dp)
                        .semantics {
                            role = Role.Button
                            stateDescription = if (selected == scenario) {
                                if (language == AppLanguage.EN) "Current scenario" else "当前场景"
                            } else {
                                if (language == AppLanguage.EN) "Not selected" else "未选择"
                            }
                            contentDescription = if (language == AppLanguage.EN) {
                                "Choose ${scenario.displayName(language)} usage scenario, ${scenario.description(language)}"
                            } else {
                                "选择${scenario.displayName(language)}使用场景，${scenario.description(language)}"
                            }
                        }
                )
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun SpeechStyleSelector(
    selected: SpeechStyle,
    language: AppLanguage,
    onSpeechStyleChange: (SpeechStyle) -> Unit
) {
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .semantics {
                contentDescription = if (language == AppLanguage.EN) {
                    "Speech style, current ${selected.displayName(language)}. ${selected.description(language)}"
                } else {
                    "语音风格，当前${selected.displayName(language)}。${selected.description(language)}"
                }
            },
        shape = RoundedCornerShape(16.dp),
        colors = CardDefaults.cardColors(containerColor = BaPanel)
    ) {
        Column(Modifier.padding(16.dp)) {
            Text(if (language == AppLanguage.EN) "Speech style" else "语音风格", color = BaText, fontWeight = FontWeight.Bold, modifier = Modifier.semantics { heading() })
            Text(if (language == AppLanguage.EN) "Brief reduces interruption, Detailed adds object type." else "简短减少打扰，详细会补充目标类别。", color = BaTextMuted, style = MaterialTheme.typography.bodySmall)
            Spacer(Modifier.height(10.dp))
            Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                SpeechStyle.values().forEach { style ->
                    FilterChip(
                        selected = selected == style,
                        onClick = { onSpeechStyleChange(style) },
                        label = { Text(style.displayName(language), maxLines = 1, overflow = TextOverflow.Ellipsis) },
                        modifier = Modifier.fillMaxWidth().heightIn(min = 48.dp).semantics {
                            role = Role.Button
                            stateDescription = if (selected == style) {
                                if (language == AppLanguage.EN) "Current style" else "当前风格"
                            } else {
                                if (language == AppLanguage.EN) "Not selected" else "未选择"
                            }
                            contentDescription = if (language == AppLanguage.EN) {
                                "Choose ${style.displayName(language)} speech style, ${style.description(language)}"
                            } else {
                                "选择${style.displayName(language)}语音风格，${style.description(language)}"
                            }
                        }
                    )
                }
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun VibrationStrengthSelector(
    selected: VibrationStrength,
    language: AppLanguage,
    onVibrationStrengthChange: (VibrationStrength) -> Unit
) {
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .semantics {
                contentDescription = if (language == AppLanguage.EN) {
                    "Vibration strength, current ${selected.displayName(language)}. ${selected.description(language)}"
                } else {
                    "震动强度，当前${selected.displayName(language)}。${selected.description(language)}"
                }
            },
        shape = RoundedCornerShape(16.dp),
        colors = CardDefaults.cardColors(containerColor = BaPanel)
    ) {
        Column(Modifier.padding(16.dp)) {
            Text(if (language == AppLanguage.EN) "Vibration strength" else "震动强度", color = BaText, fontWeight = FontWeight.Bold, modifier = Modifier.semantics { heading() })
            Text(if (language == AppLanguage.EN) "Choose soft, standard, or stronger feedback for tactile sensitivity." else "按触觉敏感度选择轻柔、标准或更强提醒。", color = BaTextMuted, style = MaterialTheme.typography.bodySmall)
            Spacer(Modifier.height(10.dp))
            Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                VibrationStrength.values().forEach { strength ->
                    FilterChip(
                        selected = selected == strength,
                        onClick = { onVibrationStrengthChange(strength) },
                        label = { Text(strength.displayName(language), maxLines = 1, overflow = TextOverflow.Ellipsis) },
                        modifier = Modifier.fillMaxWidth().heightIn(min = 48.dp).semantics {
                            role = Role.Button
                            stateDescription = if (selected == strength) {
                                if (language == AppLanguage.EN) "Current strength" else "当前强度"
                            } else {
                                if (language == AppLanguage.EN) "Not selected" else "未选择"
                            }
                            contentDescription = if (language == AppLanguage.EN) {
                                "Choose ${strength.displayName(language)} vibration strength, ${strength.description(language)}"
                            } else {
                                "选择${strength.displayName(language)}震动强度，${strength.description(language)}"
                            }
                        }
                    )
                }
            }
        }
    }
}

