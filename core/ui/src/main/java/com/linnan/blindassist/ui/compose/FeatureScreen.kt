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
fun FeatureScreen(
    controls: AssistControlsUiState,
    modelStatus: String,
    appVersion: String,
    onOpenCamera: () -> Unit,
    onGlassesPlaceholder: () -> Unit,
    onDailyUsageModeChange: (DailyUsageMode) -> Unit,
    modifier: Modifier = Modifier
) {
    val language = controls.appLanguage
    var modesExpanded by rememberSaveable { mutableStateOf(false) }
    ScreenColumn(modifier = modifier) {
        Text(
            text = if (language == AppLanguage.EN) "Start assist" else "开始辅助",
            style = MaterialTheme.typography.headlineMedium,
            color = BaText,
            fontWeight = FontWeight.Bold,
            modifier = Modifier.semantics { heading() }
        )
        Text(
            text = if (language == AppLanguage.EN) {
                "Current mode is ready. Start the phone camera when you are ready to observe ahead."
            } else {
                "当前模式已准备好。需要观察前方时，直接打开手机摄像头。"
            },
            style = MaterialTheme.typography.bodyMedium,
            color = BaTextMuted
        )
        Spacer(Modifier.height(16.dp))
        CurrentModeSummary(
            controls = controls,
            language = language,
        )
        Spacer(Modifier.height(16.dp))
        ActionFeatureCard(
            title = if (language == AppLanguage.EN) "Use phone camera" else "使用手机摄像头",
            subtitle = if (language == AppLanguage.EN) {
                "Start live recognition with ${controls.dailyUsageMode.displayName(language)} mode"
            } else {
                "按${controls.dailyUsageMode.displayName(language)}模式打开实时识别"
            },
            badge = if (language == AppLanguage.EN) "Ready" else "可用",
            icon = Icons.Rounded.CameraAlt,
            accent = BaMint,
            onClick = onOpenCamera,
            emphasis = true,
            accessibilityText = if (language == AppLanguage.EN) {
                "Use phone camera, start live recognition with ${controls.dailyUsageMode.displayName(language)} mode"
            } else {
                "使用手机摄像头，按${controls.dailyUsageMode.displayName(language)}模式打开实时识别"
            }
        )
        Spacer(Modifier.height(16.dp))

        CollapsibleDailyUsageModeSelector(
            selected = controls.dailyUsageMode,
            language = language,
            expanded = modesExpanded,
            onExpandedChange = { modesExpanded = it },
            onModeChange = onDailyUsageModeChange
        )
        Spacer(Modifier.height(12.dp))
        ActionFeatureCard(
            title = if (language == AppLanguage.EN) "Connect glasses device" else "连接眼镜设备",
            subtitle = if (language == AppLanguage.EN) {
                "Reserved for future external vision devices; no Bluetooth permission is requested"
            } else {
                "预留给蓝牙眼镜和外接视觉设备，不会申请蓝牙权限"
            },
            badge = if (language == AppLanguage.EN) "Future" else "占位",
            icon = Icons.Rounded.Bluetooth,
            accent = BaSky,
            onClick = onGlassesPlaceholder,
            accessibilityText = if (language == AppLanguage.EN) {
                "Connect glasses device, reserved future extension, no Bluetooth permission is requested"
            } else {
                "连接眼镜设备，未来扩展占位，不会申请蓝牙权限"
            }
        )
        Spacer(Modifier.height(18.dp))

        InfoStrip(
            icon = Icons.Rounded.Shield,
            title = if (language == AppLanguage.EN) "Usage boundary" else "安全边界",
            body = if (language == AppLanguage.EN) {
                "BlindAssist is a local assistive prototype. Reminders are only references and cannot replace human judgment or professional safety devices."
            } else {
                "BlindAssist 是本地助盲避障原型，提醒结果只作为辅助参考，不能替代人工判断或专业安全设备。"
            }
        )
        Spacer(Modifier.height(12.dp))
        StatusGrid(
            leftTitle = if (language == AppLanguage.EN) "Model" else "模型",
            leftBody = modelStatus,
            rightTitle = if (language == AppLanguage.EN) "Version" else "版本",
            rightBody = "v$appVersion"
        )
    }
}


@Composable
private fun CurrentModeSummary(
    controls: AssistControlsUiState,
    language: AppLanguage,
    modifier: Modifier = Modifier
) {
    val mode = controls.dailyUsageMode.displayName(language)
    val scenario = controls.assistScenario.displayName(language)
    val profile = controls.alertProfile.displayName(language)
    val care = LocalizedText.enabled(controls.careModeEnabled, language)
    Card(
        modifier = modifier
            .fillMaxWidth()
            .semantics(mergeDescendants = true) {
                contentDescription = if (language == AppLanguage.EN) {
                    "Current assist setup, mode $mode, scenario $scenario, reminder profile $profile, Care Mode $care"
                } else {
                    "当前辅助设置，模式$mode，场景$scenario，提醒档位$profile，关怀模式$care"
                }
            },
        shape = RoundedCornerShape(18.dp),
        colors = CardDefaults.cardColors(containerColor = BaPanelSoft)
    ) {
        Column(Modifier.padding(16.dp)) {
            Text(
                text = if (language == AppLanguage.EN) "Current walking task" else "当前行走任务",
                color = BaTextMuted,
                style = MaterialTheme.typography.labelLarge
            )
            Spacer(Modifier.height(6.dp))
            Text(
                text = mode,
                color = BaText,
                style = MaterialTheme.typography.headlineSmall,
                fontWeight = FontWeight.Bold,
                maxLines = 1,
                overflow = TextOverflow.Ellipsis
            )
            Spacer(Modifier.height(6.dp))
            Text(
                text = if (language == AppLanguage.EN) {
                    "$scenario · $profile reminders · Care Mode $care"
                } else {
                    "$scenario · $profile 提醒 · 关怀模式$care"
                },
                color = BaMint,
                style = MaterialTheme.typography.bodyMedium,
                fontWeight = FontWeight.SemiBold
            )
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun CollapsibleDailyUsageModeSelector(
    selected: DailyUsageMode,
    language: AppLanguage,
    expanded: Boolean,
    onExpandedChange: (Boolean) -> Unit,
    onModeChange: (DailyUsageMode) -> Unit,
    modifier: Modifier = Modifier
) {
    Card(
        modifier = modifier
            .fillMaxWidth()
            .testTag("daily_usage_mode_selector")
            .semantics {
                contentDescription = if (language == AppLanguage.EN) {
                    "Daily mode selector, current ${selected.displayName(language)}"
                } else {
                    "日常模式选择，当前${selected.displayName(language)}"
                }
            },
        shape = RoundedCornerShape(16.dp),
        colors = CardDefaults.cardColors(containerColor = BaPanel)
    ) {
        Column(Modifier.padding(16.dp)) {
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .clickable(role = Role.Button) { onExpandedChange(!expanded) }
                    .semantics {
                        contentDescription = if (language == AppLanguage.EN) {
                            if (expanded) "Collapse daily mode choices" else "Change daily mode"
                        } else {
                            if (expanded) "收起日常模式选择" else "更换日常模式"
                        }
                        stateDescription = if (expanded) {
                            if (language == AppLanguage.EN) "Expanded" else "已展开"
                        } else {
                            if (language == AppLanguage.EN) "Collapsed" else "已收起"
                        }
                    },
                verticalAlignment = Alignment.CenterVertically
            ) {
                Column(Modifier.weight(1f)) {
                    Text(
                        text = if (language == AppLanguage.EN) "Change daily mode" else "更换日常模式",
                        color = BaText,
                        fontWeight = FontWeight.Bold,
                        modifier = Modifier.semantics { heading() }
                    )
                    Text(
                        text = if (language == AppLanguage.EN) {
                            "Current: ${selected.displayName(language)}"
                        } else {
                            "当前：${selected.displayName(language)}"
                        },
                        color = BaTextMuted,
                        style = MaterialTheme.typography.bodySmall
                    )
                }
                Icon(Icons.Rounded.ChevronRight, contentDescription = null, tint = BaTextMuted)
            }
            AnimatedVisibility(
                visible = expanded,
                enter = fadeIn() + slideInVertically { it / 3 },
                exit = fadeOut()
            ) {
                Column {
                    Spacer(Modifier.height(10.dp))
                    Text(
                        text = if (language == AppLanguage.EN) {
                            "Pick the walking task once; BlindAssist applies a matching reminder bundle."
                        } else {
                            "先选今天的行走任务，系统会套用对应提醒组合。"
                        },
                        color = BaTextMuted,
                        style = MaterialTheme.typography.bodySmall
                    )
                    Spacer(Modifier.height(10.dp))
                    Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                        DailyUsageMode.selectableModes.forEach { mode ->
                            FilterChip(
                                selected = selected == mode,
                                onClick = { onModeChange(mode) },
                                label = {
                                    Column {
                                        Text(mode.displayName(language), fontWeight = FontWeight.SemiBold)
                                        Text(
                                            mode.description(language),
                                            style = MaterialTheme.typography.bodySmall,
                                            color = BaTextMuted,
                                            maxLines = 2,
                                            overflow = TextOverflow.Ellipsis
                                        )
                                    }
                                },
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .heightIn(min = 56.dp)
                                    .semantics {
                                        role = Role.Button
                                        stateDescription = if (selected == mode) {
                                            if (language == AppLanguage.EN) "Current daily mode" else "当前日常模式"
                                        } else {
                                            if (language == AppLanguage.EN) "Not selected" else "未选择"
                                        }
                                        contentDescription = mode.accessibilitySummary(language)
                                    }
                            )
                        }
                    }
                    if (selected == DailyUsageMode.CUSTOM) {
                        Spacer(Modifier.height(10.dp))
                        Text(
                            text = if (language == AppLanguage.EN) {
                                "Current preferences are custom because one or more settings were adjusted manually."
                            } else {
                                "当前为自定义组合，因为部分设置已被手动调整。"
                            },
                            color = BaAmber,
                            style = MaterialTheme.typography.bodySmall,
                            modifier = Modifier.testTag("daily_usage_custom_notice")
                        )
                    }
                }
            }
        }
    }
}

@Composable
private fun ActionFeatureCard(
    title: String,
    subtitle: String,
    badge: String,
    icon: ImageVector,
    accent: Color,
    onClick: () -> Unit,
    emphasis: Boolean = false,
    accessibilityText: String = title
) {
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .heightIn(min = if (emphasis) 136.dp else 116.dp)
            .clickable(role = Role.Button, onClick = onClick)
            .semantics(mergeDescendants = true) {
                contentDescription = accessibilityText
            },
        shape = RoundedCornerShape(if (emphasis) 22.dp else 18.dp),
        colors = CardDefaults.cardColors(containerColor = if (emphasis) BaPanelSoft else BaPanel)
    ) {
        Row(
            verticalAlignment = Alignment.CenterVertically,
            modifier = Modifier.padding(if (emphasis) 20.dp else 18.dp)
        ) {
            Box(
                modifier = Modifier
                    .size(if (emphasis) 64.dp else 54.dp)
                    .clip(RoundedCornerShape(if (emphasis) 20.dp else 16.dp))
                    .background(accent.copy(alpha = if (emphasis) 0.22f else 0.16f)),
                contentAlignment = Alignment.Center
            ) {
                Icon(icon, contentDescription = null, tint = accent, modifier = Modifier.size(if (emphasis) 32.dp else 24.dp))
            }
            Spacer(Modifier.width(14.dp))
            Column(Modifier.weight(1f)) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text(
                        text = title,
                        color = BaText,
                        style = if (emphasis) MaterialTheme.typography.titleLarge else MaterialTheme.typography.titleMedium,
                        fontWeight = FontWeight.Bold,
                        maxLines = 1,
                        overflow = TextOverflow.Ellipsis,
                        modifier = Modifier.weight(1f)
                    )
                    Text(
                        text = badge,
                        color = BaInk,
                        style = MaterialTheme.typography.labelMedium,
                        fontWeight = FontWeight.Bold,
                        modifier = Modifier
                            .clip(RoundedCornerShape(50))
                            .background(accent)
                            .padding(horizontal = 10.dp, vertical = 5.dp)
                    )
                }
                Spacer(Modifier.height(6.dp))
                Text(text = subtitle, color = BaTextMuted, style = MaterialTheme.typography.bodyMedium)
            }
            Icon(Icons.Rounded.ChevronRight, contentDescription = null, tint = BaTextMuted)
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun DailyUsageModeSelector(
    selected: DailyUsageMode,
    language: AppLanguage,
    onModeChange: (DailyUsageMode) -> Unit,
    modifier: Modifier = Modifier
) {
    Card(
        modifier = modifier
            .fillMaxWidth()
            .testTag("daily_usage_mode_selector")
            .semantics {
                contentDescription = if (language == AppLanguage.EN) {
                    "Daily usage guide, current ${selected.displayName(language)}. Choose a mode to apply scenario, reminder profile, speech style, vibration strength, and Care Mode."
                } else {
                    "日常使用向导，当前${selected.displayName(language)}。选择模式会应用场景、提醒档位、语音风格、震动强度和关怀模式。"
                }
            },
        shape = RoundedCornerShape(16.dp),
        colors = CardDefaults.cardColors(containerColor = BaPanel)
    ) {
        Column(Modifier.padding(16.dp)) {
            Text(
                text = if (language == AppLanguage.EN) "Daily usage guide" else "日常使用向导",
                color = BaText,
                fontWeight = FontWeight.Bold,
                modifier = Modifier.semantics { heading() }
            )
            Text(
                text = if (language == AppLanguage.EN) {
                    "Pick the walking task once; BlindAssist applies a matching reminder bundle."
                } else {
                    "先选今天的行走任务，系统会套用对应提醒组合。"
                },
                color = BaTextMuted,
                style = MaterialTheme.typography.bodySmall
            )
            Spacer(Modifier.height(10.dp))
            Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                DailyUsageMode.selectableModes.forEach { mode ->
                    FilterChip(
                        selected = selected == mode,
                        onClick = { onModeChange(mode) },
                        label = {
                            Column {
                                Text(mode.displayName(language), fontWeight = FontWeight.SemiBold)
                                Text(
                                    mode.description(language),
                                    style = MaterialTheme.typography.bodySmall,
                                    color = BaTextMuted,
                                    maxLines = 2,
                                    overflow = TextOverflow.Ellipsis
                                )
                            }
                        },
                        modifier = Modifier
                            .fillMaxWidth()
                            .heightIn(min = 56.dp)
                            .semantics {
                                role = Role.Button
                                stateDescription = if (selected == mode) {
                                    if (language == AppLanguage.EN) "Current daily mode" else "当前日常模式"
                                } else {
                                    if (language == AppLanguage.EN) "Not selected" else "未选择"
                                }
                                contentDescription = mode.accessibilitySummary(language)
                            }
                    )
                }
            }
            if (selected == DailyUsageMode.CUSTOM) {
                Spacer(Modifier.height(10.dp))
                Text(
                    text = if (language == AppLanguage.EN) "Current preferences are custom because one or more settings were adjusted manually." else "当前为自定义组合，因为部分设置已被手动调整。",
                    color = BaAmber,
                    style = MaterialTheme.typography.bodySmall,
                    modifier = Modifier.testTag("daily_usage_custom_notice")
                )
            }
        }
    }
}

@Composable
internal fun InfoStrip(
    icon: ImageVector,
    title: String,
    body: String
) {
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .semantics(mergeDescendants = true) {},
        shape = RoundedCornerShape(16.dp),
        colors = CardDefaults.cardColors(containerColor = BaPanelSoft)
    ) {
        Row(
            modifier = Modifier.padding(16.dp),
            verticalAlignment = Alignment.Top
        ) {
            Icon(icon, contentDescription = null, tint = BaAmber, modifier = Modifier.size(22.dp))
            Spacer(Modifier.width(12.dp))
            Column {
                Text(text = title, color = BaText, fontWeight = FontWeight.Bold)
                Spacer(Modifier.height(4.dp))
                Text(text = body, color = BaTextMuted, style = MaterialTheme.typography.bodyMedium)
            }
        }
    }
}

