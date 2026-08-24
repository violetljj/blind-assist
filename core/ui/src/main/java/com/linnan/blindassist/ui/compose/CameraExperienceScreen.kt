package com.linnan.blindassist.ui.compose

import android.graphics.BitmapFactory
import android.view.ViewGroup
import android.widget.FrameLayout
import android.widget.ImageView
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
import androidx.compose.foundation.layout.BoxWithConstraints
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
import androidx.compose.foundation.layout.navigationBarsPadding
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
import androidx.compose.material.icons.automirrored.rounded.ArrowBack
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
import androidx.compose.material.icons.automirrored.rounded.VolumeUp
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
import com.linnan.blindassist.goal.GoalHandoffState
import com.linnan.blindassist.localization.AppLanguage
import com.linnan.blindassist.localization.LocalizedText
import com.linnan.blindassist.model.AssistInputSource
import com.linnan.blindassist.model.ReplayScenario
import com.linnan.blindassist.preferences.DailyUsageMode
import com.linnan.blindassist.ui.DetectionOverlayView
import kotlinx.coroutines.delay


@Composable
fun CameraExperienceScreen(
    controls: AssistControlsUiState,
    guidance: CameraGuidanceUiState,
    fieldTestSummary: FieldTestSummaryUiState,
    goalHandoffState: GoalHandoffState? = null,
    inputSource: AssistInputSource,
    replayScenario: ReplayScenario?,
    onBack: () -> Unit,
    onDetectionChange: (Boolean) -> Unit,
    onSpeechChange: (Boolean) -> Unit,
    onVibrationChange: (Boolean) -> Unit,
    onCareModeChange: (Boolean) -> Unit,
    onDebugVisibleChange: (Boolean) -> Unit,
    onProfileChange: (AlertProfile) -> Unit,
    onScenarioChange: (AssistScenario) -> Unit,
    onQuietShortcut: () -> Unit,
    onSensitiveShortcut: () -> Unit,
    onGoalCompletionConfirmed: () -> Unit = {},
    onCameraViewsReady: (PreviewView?, ImageView?, DetectionOverlayView) -> Unit,
    modifier: Modifier = Modifier
) {
    Box(modifier = modifier.fillMaxSize().background(Color.Black)) {
        CameraPreviewHost(
            inputSource = inputSource,
            replayScenario = replayScenario,
            onCameraViewsReady = onCameraViewsReady,
            modifier = Modifier.fillMaxSize()
        )
        CameraTopBar(
            statusBadge = guidance.statusBadge,
            language = controls.appLanguage,
            onBack = onBack,
            modifier = Modifier
                .align(Alignment.TopCenter)
                .statusBarsPadding()
                .padding(12.dp)
        )
        if (inputSource == AssistInputSource.OFFLINE_REPLAY) {
            Text(
                text = if (controls.appLanguage == AppLanguage.EN) {
                    "Offline test asset · debug replay"
                } else {
                    "离线测试素材 · 调试回放"
                },
                color = BaAmber,
                style = MaterialTheme.typography.labelLarge,
                modifier = Modifier
                    .align(Alignment.TopCenter)
                    .statusBarsPadding()
                    .padding(top = 72.dp)
            )
        }
        if (inputSource == AssistInputSource.GLASSES_HARDWARE) {
            Text(
                text = if (controls.appLanguage == AppLanguage.EN) {
                    "AtomS3R external camera · live MJPEG"
                } else {
                    "AtomS3R 外界摄像头 · 实时 MJPEG"
                },
                color = BaMint,
                style = MaterialTheme.typography.labelLarge,
                modifier = Modifier
                    .align(Alignment.TopCenter)
                    .statusBarsPadding()
                    .padding(top = 72.dp)
            )
        }
        CameraControlPanel(
            controls = controls,
            guidance = guidance,
            fieldTestSummary = fieldTestSummary,
            goalHandoffState = goalHandoffState,
            onDetectionChange = onDetectionChange,
            onSpeechChange = onSpeechChange,
            onVibrationChange = onVibrationChange,
            onCareModeChange = onCareModeChange,
            onDebugVisibleChange = onDebugVisibleChange,
            onProfileChange = onProfileChange,
            onScenarioChange = onScenarioChange,
            onQuietShortcut = onQuietShortcut,
            onSensitiveShortcut = onSensitiveShortcut,
            onGoalCompletionConfirmed = onGoalCompletionConfirmed,
            modifier = Modifier
                .align(Alignment.BottomCenter)
                .padding(12.dp)
        )
    }
}

@Composable
fun CameraPreviewHost(
    inputSource: AssistInputSource,
    replayScenario: ReplayScenario?,
    onCameraViewsReady: (PreviewView?, ImageView?, DetectionOverlayView) -> Unit,
    modifier: Modifier = Modifier
) {
    val context = LocalContext.current
    AndroidView(
        modifier = modifier,
        factory = {
            val frame = FrameLayout(context)
            val preview = if (inputSource == AssistInputSource.PHONE_CAMERA) {
                PreviewView(context).apply {
                    scaleType = PreviewView.ScaleType.FILL_CENTER
                    layoutParams = FrameLayout.LayoutParams(
                        ViewGroup.LayoutParams.MATCH_PARENT,
                        ViewGroup.LayoutParams.MATCH_PARENT
                    )
                }
            } else {
                null
            }
            val replayImage = if (inputSource == AssistInputSource.OFFLINE_REPLAY) {
                val scenario = requireNotNull(replayScenario) {
                    "ReplayScenario is required for offline replay preview"
                }
                ImageView(context).apply {
                    scaleType = ImageView.ScaleType.CENTER_CROP
                    layoutParams = FrameLayout.LayoutParams(
                        ViewGroup.LayoutParams.MATCH_PARENT,
                        ViewGroup.LayoutParams.MATCH_PARENT
                    )
                    val bitmap = try {
                        context.assets.open(scenario.assetPath).use(BitmapFactory::decodeStream)
                    } catch (_: Exception) {
                        null
                    }
                    if (bitmap != null) {
                        setImageBitmap(bitmap)
                    } else {
                        setBackgroundColor(android.graphics.Color.DKGRAY)
                        contentDescription = "Offline replay preview unavailable"
                    }
                }
            } else {
                null
            }
            val externalImage = if (inputSource == AssistInputSource.GLASSES_HARDWARE) {
                ImageView(context).apply {
                    scaleType = ImageView.ScaleType.CENTER_CROP
                    setBackgroundColor(android.graphics.Color.BLACK)
                    contentDescription = "AtomS3R live MJPEG preview"
                    layoutParams = FrameLayout.LayoutParams(
                        ViewGroup.LayoutParams.MATCH_PARENT,
                        ViewGroup.LayoutParams.MATCH_PARENT
                    )
                }
            } else null
            val overlay = DetectionOverlayView(context).apply {
                layoutParams = FrameLayout.LayoutParams(
                    ViewGroup.LayoutParams.MATCH_PARENT,
                    ViewGroup.LayoutParams.MATCH_PARENT
                )
            }
            preview?.let(frame::addView)
            replayImage?.let(frame::addView)
            externalImage?.let(frame::addView)
            frame.addView(overlay)
            onCameraViewsReady(preview, externalImage, overlay)
            frame
        }
    )
}

@Composable
fun CameraControlPanel(
    controls: AssistControlsUiState,
    guidance: CameraGuidanceUiState,
    fieldTestSummary: FieldTestSummaryUiState,
    goalHandoffState: GoalHandoffState? = null,
    onDetectionChange: (Boolean) -> Unit,
    onSpeechChange: (Boolean) -> Unit,
    onVibrationChange: (Boolean) -> Unit,
    onCareModeChange: (Boolean) -> Unit,
    onDebugVisibleChange: (Boolean) -> Unit,
    onProfileChange: (AlertProfile) -> Unit,
    onScenarioChange: (AssistScenario) -> Unit,
    onQuietShortcut: () -> Unit,
    onSensitiveShortcut: () -> Unit,
    onGoalCompletionConfirmed: () -> Unit = {},
    modifier: Modifier = Modifier
) {
    val language = controls.appLanguage
    val title = if (controls.careModeEnabled) guidance.careTitle else guidance.title
    val detail = if (controls.careModeEnabled) guidance.careDetail else guidance.detail
    val target = if (controls.careModeEnabled) guidance.careTargetLine else guidance.targetLine
    val titleStyle = if (controls.careModeEnabled) {
        MaterialTheme.typography.headlineLarge
    } else {
        MaterialTheme.typography.headlineMedium
    }
    var debugExpanded by rememberSaveable { mutableStateOf(false) }

    BoxWithConstraints(
        modifier = modifier
            .fillMaxWidth()
            .navigationBarsPadding()
    ) {
        val panelMaxHeight = maxHeight * 0.75f
        Card(
            modifier = Modifier
                .fillMaxWidth()
                .heightIn(max = panelMaxHeight),
            shape = RoundedCornerShape(if (controls.careModeEnabled) 26.dp else 22.dp),
            colors = CardDefaults.cardColors(
                containerColor = if (controls.careModeEnabled) {
                    BaHomeActionEnd.copy(alpha = 0.97f)
                } else {
                    BaHomeActionStart.copy(alpha = 0.95f)
                }
            ),
            elevation = CardDefaults.cardElevation(defaultElevation = 8.dp)
        ) {
            Column(
                Modifier
                    .verticalScroll(rememberScrollState())
                    .padding(if (controls.careModeEnabled) 20.dp else 16.dp)
            ) {
            goalHandoffState?.let { handoffState ->
                GoalHandoffCard(
                    state = handoffState,
                    language = language,
                    onUserConfirmed = onGoalCompletionConfirmed
                )
                if (handoffState !is GoalHandoffState.Inactive) {
                    Spacer(Modifier.height(16.dp))
                }
            }
            Text(
                text = title,
                style = titleStyle,
                color = Color(guidance.titleColor),
                fontWeight = FontWeight.Bold,
                maxLines = 2,
                overflow = TextOverflow.Ellipsis,
                modifier = Modifier.semantics {
                    heading()
                    stateDescription = if (controls.careModeEnabled) {
                        guidance.careAccessibilitySummary
                    } else {
                        guidance.accessibilitySummary
                    }
                }
            )
            Spacer(Modifier.height(if (controls.careModeEnabled) 8.dp else 6.dp))
            Text(
                text = detail,
                color = BaText,
                style = if (controls.careModeEnabled) MaterialTheme.typography.titleMedium else MaterialTheme.typography.bodyMedium,
                fontWeight = if (controls.careModeEnabled) FontWeight.SemiBold else FontWeight.Normal
            )
            Text(
                text = target,
                color = if (controls.careModeEnabled) BaText else BaTextMuted,
                style = if (controls.careModeEnabled) MaterialTheme.typography.bodyMedium else MaterialTheme.typography.bodySmall
            )
            Spacer(Modifier.height(8.dp))
            CameraModeStatusRow(controls = controls, language = language)
            Spacer(Modifier.height(8.dp))
            Text(
                text = if (controls.careModeEnabled) guidance.careExplanation else guidance.explanationHeadline,
                color = BaText,
                style = if (controls.careModeEnabled) MaterialTheme.typography.bodyMedium else MaterialTheme.typography.bodySmall,
                fontWeight = FontWeight.SemiBold,
                modifier = Modifier.testTag("risk_explanation_headline")
            )
            if (!controls.careModeEnabled) {
                Text(
                    text = guidance.explanationDetail,
                    color = BaTextMuted,
                    style = MaterialTheme.typography.bodySmall
                )
            }
            Spacer(Modifier.height(if (controls.careModeEnabled) 16.dp else 12.dp))
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                CompactToggle(if (language == AppLanguage.EN) "Detection" else "检测", controls.detectionEnabled, Icons.Rounded.Visibility, onDetectionChange, Modifier.weight(1f), language)
                CompactToggle(if (language == AppLanguage.EN) "Speech" else "语音", controls.speechEnabled, Icons.AutoMirrored.Rounded.VolumeUp, onSpeechChange, Modifier.weight(1f), language)
                CompactToggle(if (language == AppLanguage.EN) "Vibration" else "震动", controls.vibrationEnabled, Icons.Rounded.Vibration, onVibrationChange, Modifier.weight(1f), language)
            }
            Spacer(Modifier.height(8.dp))
            AnimatedVisibility(
                visible = false,
                enter = fadeIn() + slideInVertically { it / 3 },
                exit = fadeOut()
            ) {
                Column {
                    Text(
                        text = if (language == AppLanguage.EN) "More adjustments" else "更多调整",
                        color = BaTextMuted,
                        style = MaterialTheme.typography.labelLarge,
                        fontWeight = FontWeight.SemiBold
                    )
                    Spacer(Modifier.height(6.dp))
                    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        CompactAction(
                            text = if (language == AppLanguage.EN) "Quiet" else "调安静",
                            icon = Icons.Rounded.Tune,
                            onClick = onQuietShortcut,
                            modifier = Modifier
                                .weight(1f)
                                .testTag("camera_quiet_shortcut"),
                            accessibilityText = if (language == AppLanguage.EN) {
                                "Apply quiet reminder shortcut, keep current scenario, use Quiet profile, Brief speech, and Soft vibration"
                            } else {
                                "应用安静提醒快捷设置，保留当前场景，使用安静档位、简短语音和轻柔震动"
                            }
                        )
                        CompactAction(
                            text = if (language == AppLanguage.EN) "Sensitive" else "调敏感",
                            icon = Icons.Rounded.Tune,
                            onClick = onSensitiveShortcut,
                            modifier = Modifier
                                .weight(1f)
                                .testTag("camera_sensitive_shortcut"),
                            accessibilityText = if (language == AppLanguage.EN) {
                                "Apply sensitive reminder shortcut, keep current scenario, use Sensitive profile, Standard speech, and Strong vibration"
                            } else {
                                "应用敏感提醒快捷设置，保留当前场景，使用敏感档位、标准语音和强震动"
                            }
                        )
                    }
                    Spacer(Modifier.height(8.dp))
                    CompactAction(
                        text = if (language == AppLanguage.EN) "Scenario ${controls.assistScenario.displayName(language)}" else "场景 ${controls.assistScenario.displayName(language)}",
                        icon = Icons.Rounded.Shield,
                        onClick = { onScenarioChange(controls.assistScenario.next()) },
                        modifier = Modifier
                            .fillMaxWidth()
                            .testTag("camera_scenario_toggle"),
                        accessibilityText = if (language == AppLanguage.EN) {
                            "Usage scenario, current ${controls.assistScenario.displayName(language)}, tap to switch to ${controls.assistScenario.next().displayName(language)}"
                        } else {
                            "使用场景，当前${controls.assistScenario.displayName(language)}，点击切换到${controls.assistScenario.next().displayName(language)}"
                        }
                    )
                    Spacer(Modifier.height(8.dp))
                    CompactAction(
                        text = if (controls.debugVisible) {
                            if (language == AppLanguage.EN) "Hide debug details" else "收起调试信息"
                        } else {
                            if (language == AppLanguage.EN) "Show debug details" else "展开调试信息"
                        },
                        icon = Icons.Rounded.BugReport,
                        onClick = { onDebugVisibleChange(!controls.debugVisible) },
                        modifier = Modifier
                            .fillMaxWidth()
                            .testTag("camera_debug_toggle"),
                        selected = controls.debugVisible,
                        accessibilityText = if (controls.debugVisible) {
                            if (language == AppLanguage.EN) "Hide camera debug details" else "收起相机调试信息"
                        } else {
                            if (language == AppLanguage.EN) "Show camera debug details" else "展开相机调试信息"
                        },
                        stateDescriptionText = if (language == AppLanguage.EN) {
                            if (controls.debugVisible) "Expanded" else "Collapsed"
                        } else {
                            if (controls.debugVisible) "已展开" else "已收起"
                        }
                    )
                    AnimatedVisibility(visible = controls.debugVisible) {
                        Column(Modifier.padding(top = 4.dp)) {
                            Text(
                                text = guidance.debugText,
                                color = BaTextMuted,
                                style = MaterialTheme.typography.bodySmall
                            )
                            Spacer(Modifier.height(10.dp))
                            FieldTestSummaryBlock(fieldTestSummary)
                        }
                    }
                }
            }
            if (controls.debugVisible) {
                Spacer(Modifier.height(8.dp))
                CompactAction(
                    text = if (debugExpanded) {
                        if (language == AppLanguage.EN) "Hide debug details" else "收起调试信息"
                    } else {
                        if (language == AppLanguage.EN) "Show debug details" else "展开调试信息"
                    },
                    icon = Icons.Rounded.BugReport,
                    onClick = { debugExpanded = !debugExpanded },
                    modifier = Modifier
                        .fillMaxWidth()
                        .testTag("camera_debug_toggle"),
                    selected = debugExpanded,
                    accessibilityText = if (debugExpanded) {
                        if (language == AppLanguage.EN) "Hide camera debug details" else "收起相机调试信息"
                    } else {
                        if (language == AppLanguage.EN) "Show camera debug details" else "展开相机调试信息"
                    },
                    stateDescriptionText = if (language == AppLanguage.EN) {
                        if (debugExpanded) "Expanded" else "Collapsed"
                    } else {
                        if (debugExpanded) "已展开" else "已收起"
                    }
                )
                AnimatedVisibility(visible = debugExpanded) {
                    Column(Modifier.padding(top = 4.dp)) {
                        Text(
                            text = guidance.debugText,
                            color = BaTextMuted,
                            style = MaterialTheme.typography.bodySmall
                        )
                        Spacer(Modifier.height(10.dp))
                        FieldTestSummaryBlock(
                            summary = fieldTestSummary,
                            contentColor = BaText,
                            mutedColor = BaTextMuted,
                            accentColor = BaMint
                        )
                    }
                }
            }
            Spacer(Modifier.height(8.dp))
            CompactToggle(if (language == AppLanguage.EN) "Care" else "关怀", controls.careModeEnabled, Icons.Rounded.Favorite, onCareModeChange, Modifier.fillMaxWidth(), language)
            }
        }
    }
}

@Composable
private fun CameraModeStatusRow(
    controls: AssistControlsUiState,
    language: AppLanguage
) {
    Row(horizontalArrangement = Arrangement.spacedBy(8.dp), modifier = Modifier.fillMaxWidth()) {
        Text(
            text = if (language == AppLanguage.EN) "Scenario: ${controls.assistScenario.displayName(language)}" else "场景：${controls.assistScenario.displayName(language)}",
            color = BaMint,
            style = MaterialTheme.typography.labelLarge,
            fontWeight = FontWeight.SemiBold,
            maxLines = 1,
            overflow = TextOverflow.Ellipsis,
            modifier = Modifier
                .weight(1f)
                .testTag("camera_scenario_label")
        )
        Text(
            text = if (language == AppLanguage.EN) "Mode: ${controls.dailyUsageMode.displayName(language)}" else "模式：${controls.dailyUsageMode.displayName(language)}",
            color = BaSky,
            style = MaterialTheme.typography.labelLarge,
            fontWeight = FontWeight.SemiBold,
            maxLines = 1,
            overflow = TextOverflow.Ellipsis,
            modifier = Modifier
                .weight(1f)
                .testTag("camera_daily_mode_label")
        )
    }
}

@Composable
private fun CameraTopBar(
    statusBadge: String,
    language: AppLanguage,
    onBack: () -> Unit,
    modifier: Modifier = Modifier
) {
    Row(
        modifier = modifier
            .fillMaxWidth()
            .heightIn(min = 56.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        IconButton(
            onClick = onBack,
            modifier = Modifier
                .size(48.dp)
                .clip(CircleShape)
                .background(BaHomeActionEnd.copy(alpha = 0.88f))
        ) {
            Icon(
                Icons.AutoMirrored.Rounded.ArrowBack,
                contentDescription = if (language == AppLanguage.EN) "Back to features" else "返回功能页",
                tint = BaHomeOnAction
            )
        }
        Spacer(Modifier.weight(1f))
        Text(
            text = statusBadge,
            color = BaHomeInk,
            style = MaterialTheme.typography.labelLarge,
            fontWeight = FontWeight.Bold,
            modifier = Modifier
                .clip(RoundedCornerShape(50))
                .background(BaHomeSurface.copy(alpha = 0.94f))
                .padding(horizontal = 14.dp, vertical = 8.dp)
        )
    }
}
