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
fun BlindAssistTheme(content: @Composable () -> Unit) {
    val colors = androidx.compose.material3.darkColorScheme(
        primary = BaMint,
        onPrimary = BaInk,
        secondary = BaSky,
        tertiary = BaAmber,
        background = BaNight,
        surface = BaPanel,
        surfaceVariant = BaPanelSoft,
        onBackground = BaText,
        onSurface = BaText,
        onSurfaceVariant = BaTextMuted,
        error = BaDanger
    )
    MaterialTheme(
        colorScheme = colors,
        typography = MaterialTheme.typography,
        content = content
    )
}

@Composable
fun BlindAssistApp(
    controls: AssistControlsUiState,
    cameraGuidance: CameraGuidanceUiState,
    fieldTestSummary: FieldTestSummaryUiState,
    modelStatus: String,
    appVersion: String,
    cameraActive: Boolean,
    showOnboarding: Boolean,
    onOpenCamera: () -> Unit,
    onCloseCamera: () -> Unit,
    onCompleteOnboarding: () -> Unit,
    onShowOnboarding: () -> Unit,
    onGlassesPlaceholder: () -> Unit,
    onDetectionChange: (Boolean) -> Unit,
    onSpeechChange: (Boolean) -> Unit,
    onVibrationChange: (Boolean) -> Unit,
    onCareModeChange: (Boolean) -> Unit,
    onDebugVisibleChange: (Boolean) -> Unit,
    onProfileChange: (AlertProfile) -> Unit,
    onScenarioChange: (AssistScenario) -> Unit,
    onSpeechStyleChange: (SpeechStyle) -> Unit,
    onVibrationStrengthChange: (VibrationStrength) -> Unit,
    onDailyUsageModeChange: (DailyUsageMode) -> Unit,
    onQuietShortcut: () -> Unit,
    onSensitiveShortcut: () -> Unit,
    onLanguageChange: (AppLanguage) -> Unit,
    onCameraViewsReady: (PreviewView, DetectionOverlayView) -> Unit,
    modifier: Modifier = Modifier
) {
    var splashVisible by rememberSaveable { mutableStateOf(true) }

    BackHandler(enabled = cameraActive) {
        onCloseCamera()
    }

    Surface(
        modifier = modifier.fillMaxSize(),
        color = BaNight
    ) {
        AnimatedContent(
            targetState = cameraActive,
            label = "camera-shell"
        ) { isCameraActive ->
            when {
                isCameraActive -> CameraExperienceScreen(
                    controls = controls,
                    guidance = cameraGuidance,
                    fieldTestSummary = fieldTestSummary,
                    onBack = onCloseCamera,
                    onDetectionChange = onDetectionChange,
                    onSpeechChange = onSpeechChange,
                    onVibrationChange = onVibrationChange,
                    onCareModeChange = onCareModeChange,
                    onDebugVisibleChange = onDebugVisibleChange,
                    onProfileChange = onProfileChange,
                    onScenarioChange = onScenarioChange,
                    onQuietShortcut = onQuietShortcut,
                    onSensitiveShortcut = onSensitiveShortcut,
                    onCameraViewsReady = onCameraViewsReady
                )
                splashVisible -> BrandSplashScreen(onFinished = { splashVisible = false })
                showOnboarding -> OnboardingScreen(onFinished = onCompleteOnboarding)
                else -> MainShell(
                    controls = controls,
                    fieldTestSummary = fieldTestSummary,
                    modelStatus = modelStatus,
                    appVersion = appVersion,
                    onOpenCamera = onOpenCamera,
                    onGlassesPlaceholder = onGlassesPlaceholder,
                    onSpeechChange = onSpeechChange,
                    onVibrationChange = onVibrationChange,
                    onCareModeChange = onCareModeChange,
                    onDebugVisibleChange = onDebugVisibleChange,
                    onProfileChange = onProfileChange,
                    onScenarioChange = onScenarioChange,
                    onSpeechStyleChange = onSpeechStyleChange,
                    onVibrationStrengthChange = onVibrationStrengthChange,
                    onDailyUsageModeChange = onDailyUsageModeChange,
                    onLanguageChange = onLanguageChange,
                    onShowOnboarding = onShowOnboarding
                )
            }
        }
    }
}

@Composable
fun BrandSplashScreen(
    onFinished: () -> Unit,
    modifier: Modifier = Modifier
) {
    val transition = rememberInfiniteTransition(label = "splash")
    val scan by transition.animateFloat(
        initialValue = 0.05f,
        targetValue = 0.95f,
        animationSpec = infiniteRepeatable(
            animation = tween(1400, easing = FastOutSlowInEasing),
            repeatMode = RepeatMode.Reverse
        ),
        label = "scan"
    )
    val pulse by transition.animateFloat(
        initialValue = 0.78f,
        targetValue = 1f,
        animationSpec = infiniteRepeatable(
            animation = tween(900, easing = FastOutSlowInEasing),
            repeatMode = RepeatMode.Reverse
        ),
        label = "pulse"
    )

    LaunchedEffect(Unit) {
        delay(1150)
        onFinished()
    }

    Box(
        modifier = modifier
            .fillMaxSize()
            .clickable(
                role = Role.Button,
                onClickLabel = "跳过启动页",
                onClick = onFinished
            )
            .background(
                Brush.verticalGradient(
                    listOf(BaNight, Color(0xFF092128), Color(0xFF061115))
                )
            ),
        contentAlignment = Alignment.Center
    ) {
        Canvas(modifier = Modifier.fillMaxSize()) {
            val y = size.height * scan
            drawLine(
                color = BaMint.copy(alpha = 0.36f),
                start = Offset(size.width * 0.12f, y),
                end = Offset(size.width * 0.88f, y),
                strokeWidth = 3.dp.toPx(),
                cap = StrokeCap.Round
            )
            drawCircle(
                color = BaSky.copy(alpha = 0.09f * pulse),
                radius = size.minDimension * 0.34f * pulse,
                center = center
            )
        }
        Column(
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.Center,
            modifier = Modifier.padding(32.dp)
        ) {
            Box(
                modifier = Modifier
                    .size(116.dp)
                    .clip(CircleShape)
                    .background(BaPanelSoft),
                contentAlignment = Alignment.Center
            ) {
                Icon(
                    imageVector = Icons.Rounded.Visibility,
                    contentDescription = null,
                    tint = BaMint,
                    modifier = Modifier.size(58.dp)
                )
            }
            Spacer(Modifier.height(24.dp))
            Text(
                text = "BlindAssist",
                style = MaterialTheme.typography.headlineLarge,
                color = BaText,
                fontWeight = FontWeight.Bold,
                modifier = Modifier.semantics { heading() }
            )
            Spacer(Modifier.height(8.dp))
            Text(
                text = "本地视觉辅助引擎启动中",
                style = MaterialTheme.typography.bodyLarge,
                color = BaTextMuted
            )
        }
    }
}

@Composable
fun OnboardingScreen(
    onFinished: () -> Unit,
    modifier: Modifier = Modifier
) {
    var pageIndex by rememberSaveable { mutableStateOf(0) }
    val pages = onboardingPages()
    val page = pages[pageIndex]
    val isLastPage = pageIndex == pages.lastIndex

    ScreenColumn(modifier = modifier) {
        Spacer(Modifier.height(8.dp))
        Text(
            text = "开始使用 BlindAssist",
            style = MaterialTheme.typography.headlineMedium,
            color = BaText,
            fontWeight = FontWeight.Bold,
            modifier = Modifier.semantics { heading() }
        )
        Text(
            text = "先了解三件事，再进入本地视觉辅助体验。",
            style = MaterialTheme.typography.bodyMedium,
            color = BaTextMuted
        )
        Spacer(Modifier.height(28.dp))

        Card(
            modifier = Modifier
                .fillMaxWidth()
                .heightIn(min = 340.dp),
            shape = RoundedCornerShape(24.dp),
            colors = CardDefaults.cardColors(containerColor = BaPanel)
        ) {
            Column(
                modifier = Modifier.padding(22.dp),
                horizontalAlignment = Alignment.Start
            ) {
                Box(
                    modifier = Modifier
                        .size(68.dp)
                        .clip(RoundedCornerShape(20.dp))
                        .background(page.accent.copy(alpha = 0.16f)),
                    contentAlignment = Alignment.Center
                ) {
                    Icon(
                        imageVector = page.icon,
                        contentDescription = null,
                        tint = page.accent,
                        modifier = Modifier.size(34.dp)
                    )
                }
                Spacer(Modifier.height(22.dp))
                Text(
                    text = page.title,
                    style = MaterialTheme.typography.headlineSmall,
                    color = BaText,
                    fontWeight = FontWeight.Bold,
                    modifier = Modifier.semantics { heading() }
                )
                Spacer(Modifier.height(10.dp))
                Text(
                    text = page.body,
                    style = MaterialTheme.typography.bodyLarge,
                    color = BaText,
                    lineHeight = MaterialTheme.typography.bodyLarge.lineHeight
                )
                Spacer(Modifier.height(12.dp))
                Text(
                    text = page.detail,
                    style = MaterialTheme.typography.bodyMedium,
                    color = BaTextMuted
                )
            }
        }

        Spacer(Modifier.height(20.dp))
        Row(
            horizontalArrangement = Arrangement.spacedBy(8.dp),
            verticalAlignment = Alignment.CenterVertically,
            modifier = Modifier.fillMaxWidth()
        ) {
            pages.forEachIndexed { index, _ ->
                Box(
                    modifier = Modifier
                        .height(8.dp)
                        .weight(1f)
                        .clip(RoundedCornerShape(50))
                        .background(if (index == pageIndex) BaMint else BaPanelSoft)
                )
            }
        }
        Spacer(Modifier.height(20.dp))
        Button(
            onClick = {
                if (isLastPage) {
                    onFinished()
                } else {
                    pageIndex += 1
                }
            },
            modifier = Modifier
                .fillMaxWidth()
                .heightIn(min = 52.dp),
            shape = RoundedCornerShape(16.dp),
            colors = ButtonDefaults.buttonColors(containerColor = BaMint, contentColor = BaInk)
        ) {
            Text(if (isLastPage) "开始使用" else "下一步")
        }
        if (!isLastPage) {
            TextButton(
                onClick = onFinished,
                modifier = Modifier
                    .fillMaxWidth()
                    .heightIn(min = 48.dp)
            ) {
                Text("跳过引导")
            }
        }
    }
}

@Composable
private fun MainShell(
    controls: AssistControlsUiState,
    fieldTestSummary: FieldTestSummaryUiState,
    modelStatus: String,
    appVersion: String,
    onOpenCamera: () -> Unit,
    onGlassesPlaceholder: () -> Unit,
    onSpeechChange: (Boolean) -> Unit,
    onVibrationChange: (Boolean) -> Unit,
    onCareModeChange: (Boolean) -> Unit,
    onDebugVisibleChange: (Boolean) -> Unit,
    onProfileChange: (AlertProfile) -> Unit,
    onScenarioChange: (AssistScenario) -> Unit,
    onSpeechStyleChange: (SpeechStyle) -> Unit,
    onVibrationStrengthChange: (VibrationStrength) -> Unit,
    onDailyUsageModeChange: (DailyUsageMode) -> Unit,
    onLanguageChange: (AppLanguage) -> Unit,
    onShowOnboarding: () -> Unit,
    modifier: Modifier = Modifier
) {
    var selectedTab by rememberSaveable { mutableStateOf(BottomTab.Features.name) }
    val tabs = BottomTab.entries

    Scaffold(
        modifier = modifier.fillMaxSize(),
        containerColor = BaNight,
        bottomBar = {
            NavigationBar(containerColor = BaPanel, tonalElevation = 0.dp) {
                tabs.forEach { tab ->
                    NavigationBarItem(
                        selected = selectedTab == tab.name,
                        onClick = { selectedTab = tab.name },
                        icon = {
                            Icon(
                                imageVector = tab.icon,
                                contentDescription = tab.label
                            )
                        },
                        label = { Text(tab.label, maxLines = 1) }
                    )
                }
            }
        }
    ) { innerPadding ->
        Box(
            modifier = Modifier
                .fillMaxSize()
                .padding(innerPadding)
        ) {
            when (BottomTab.valueOf(selectedTab)) {
                BottomTab.Features -> FeatureScreen(
                    controls = controls,
                    modelStatus = modelStatus,
                    appVersion = appVersion,
                    onOpenCamera = onOpenCamera,
                    onGlassesPlaceholder = onGlassesPlaceholder,
                    onDailyUsageModeChange = onDailyUsageModeChange
                )
                BottomTab.Profile -> ProfileScreen(
                    controls = controls,
                    appVersion = appVersion
                )
                BottomTab.Settings -> SettingsScreen(
                    controls = controls,
                    fieldTestSummary = fieldTestSummary,
                    onSpeechChange = onSpeechChange,
                    onVibrationChange = onVibrationChange,
                    onCareModeChange = onCareModeChange,
                    onDebugVisibleChange = onDebugVisibleChange,
                    onProfileChange = onProfileChange,
                    onScenarioChange = onScenarioChange,
                    onSpeechStyleChange = onSpeechStyleChange,
                    onVibrationStrengthChange = onVibrationStrengthChange,
                    onLanguageChange = onLanguageChange,
                    onShowOnboarding = onShowOnboarding
                )
            }
        }
    }
}

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
    ScreenColumn(modifier = modifier) {
        Text(
            text = if (language == AppLanguage.EN) "Features" else "功能",
            style = MaterialTheme.typography.headlineMedium,
            color = BaText,
            fontWeight = FontWeight.Bold,
            modifier = Modifier.semantics { heading() }
        )
        Text(
            text = if (language == AppLanguage.EN) {
                "Choose a daily assist mode, then start the phone camera. The glasses path remains a future extension."
            } else {
                "先选择一种日常辅助模式，再打开手机摄像头。眼镜连接作为后续扩展入口保留。"
            },
            style = MaterialTheme.typography.bodyMedium,
            color = BaTextMuted
        )
        Spacer(Modifier.height(18.dp))

        DailyUsageModeSelector(
            selected = controls.dailyUsageMode,
            language = language,
            onModeChange = onDailyUsageModeChange
        )
        Spacer(Modifier.height(18.dp))

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
            onClick = onOpenCamera
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
            onClick = onGlassesPlaceholder
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
fun ProfileScreen(
    controls: AssistControlsUiState,
    appVersion: String,
    modifier: Modifier = Modifier
) {
    val language = controls.appLanguage
    ScreenColumn(modifier = modifier) {
        Row(
            verticalAlignment = Alignment.CenterVertically,
            modifier = Modifier.fillMaxWidth()
        ) {
            Box(
                modifier = Modifier
                    .size(64.dp)
                    .clip(CircleShape)
                    .background(BaMint.copy(alpha = 0.18f)),
                contentAlignment = Alignment.Center
            ) {
                Icon(Icons.Rounded.Person, contentDescription = null, tint = BaMint)
            }
            Spacer(Modifier.width(14.dp))
            Column {
                Text(
                    text = "BlindAssist 用户",
                    style = MaterialTheme.typography.titleLarge,
                    color = BaText,
                    fontWeight = FontWeight.Bold,
                    modifier = Modifier.semantics { heading() }
                )
                Text(
                    text = "本地原型模式 · 未接入账号系统",
                    style = MaterialTheme.typography.bodyMedium,
                    color = BaTextMuted
                )
            }
        }
        Spacer(Modifier.height(20.dp))
        StatusGrid(
            leftTitle = "设备",
            leftBody = "手机摄像头可用",
            rightTitle = "眼镜",
            rightBody = "等待未来扩展"
        )
        Spacer(Modifier.height(12.dp))
        StatusGrid(
            leftTitle = if (language == AppLanguage.EN) "Reminder profile" else "提醒档位",
            leftBody = controls.alertProfile.displayName(language),
            rightTitle = if (language == AppLanguage.EN) "Usage scenario" else "使用场景",
            rightBody = controls.assistScenario.displayName(language)
        )
        Spacer(Modifier.height(12.dp))
        StatusGrid(
            leftTitle = "当前版本",
            leftBody = "v$appVersion",
            rightTitle = "提醒解释",
            rightBody = "已开启"
        )
        Spacer(Modifier.height(12.dp))
        InfoStrip(
            icon = Icons.Rounded.Favorite,
            title = if (language == AppLanguage.EN) "Assist preferences" else "辅助偏好",
            body = if (language == AppLanguage.EN) {
                "Speech ${enabledText(controls.speechEnabled, language)} (${controls.speechStyle.displayName(language)}), vibration ${enabledText(controls.vibrationEnabled, language)} (${controls.vibrationStrength.displayName(language)}), Care Mode ${enabledText(controls.careModeEnabled, language)}."
            } else {
                "语音${enabledText(controls.speechEnabled, language)}（${controls.speechStyle.displayName(language)}），震动${enabledText(controls.vibrationEnabled, language)}（${controls.vibrationStrength.displayName(language)}），关怀模式${enabledText(controls.careModeEnabled, language)}。"
            }
        )
    }
}

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
fun CameraExperienceScreen(
    controls: AssistControlsUiState,
    guidance: CameraGuidanceUiState,
    fieldTestSummary: FieldTestSummaryUiState,
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
    onCameraViewsReady: (PreviewView, DetectionOverlayView) -> Unit,
    modifier: Modifier = Modifier
) {
    Box(modifier = modifier.fillMaxSize().background(Color.Black)) {
        CameraPreviewHost(
            onCameraViewsReady = onCameraViewsReady,
            modifier = Modifier.fillMaxSize()
        )
        CameraTopBar(
            statusBadge = guidance.statusBadge,
            onBack = onBack,
            modifier = Modifier
                .align(Alignment.TopCenter)
                .statusBarsPadding()
                .padding(12.dp)
        )
        CameraControlPanel(
            controls = controls,
            guidance = guidance,
            fieldTestSummary = fieldTestSummary,
            onDetectionChange = onDetectionChange,
            onSpeechChange = onSpeechChange,
            onVibrationChange = onVibrationChange,
            onCareModeChange = onCareModeChange,
            onDebugVisibleChange = onDebugVisibleChange,
            onProfileChange = onProfileChange,
            onScenarioChange = onScenarioChange,
            onQuietShortcut = onQuietShortcut,
            onSensitiveShortcut = onSensitiveShortcut,
            modifier = Modifier
                .align(Alignment.BottomCenter)
                .padding(12.dp)
        )
    }
}

@Composable
fun CameraPreviewHost(
    onCameraViewsReady: (PreviewView, DetectionOverlayView) -> Unit,
    modifier: Modifier = Modifier
) {
    val context = LocalContext.current
    AndroidView(
        modifier = modifier,
        factory = {
            val frame = FrameLayout(context)
            val preview = PreviewView(context).apply {
                scaleType = PreviewView.ScaleType.FILL_CENTER
                layoutParams = FrameLayout.LayoutParams(
                    ViewGroup.LayoutParams.MATCH_PARENT,
                    ViewGroup.LayoutParams.MATCH_PARENT
                )
            }
            val overlay = DetectionOverlayView(context).apply {
                layoutParams = FrameLayout.LayoutParams(
                    ViewGroup.LayoutParams.MATCH_PARENT,
                    ViewGroup.LayoutParams.MATCH_PARENT
                )
            }
            frame.addView(preview)
            frame.addView(overlay)
            onCameraViewsReady(preview, overlay)
            frame
        }
    )
}

@Composable
fun CameraControlPanel(
    controls: AssistControlsUiState,
    guidance: CameraGuidanceUiState,
    fieldTestSummary: FieldTestSummaryUiState,
    onDetectionChange: (Boolean) -> Unit,
    onSpeechChange: (Boolean) -> Unit,
    onVibrationChange: (Boolean) -> Unit,
    onCareModeChange: (Boolean) -> Unit,
    onDebugVisibleChange: (Boolean) -> Unit,
    onProfileChange: (AlertProfile) -> Unit,
    onScenarioChange: (AssistScenario) -> Unit,
    onQuietShortcut: () -> Unit,
    onSensitiveShortcut: () -> Unit,
    modifier: Modifier = Modifier
) {
    val language = controls.appLanguage
    val title = if (controls.careModeEnabled) guidance.careTitle else guidance.title
    val detail = if (controls.careModeEnabled) guidance.careDetail else guidance.detail
    val target = if (controls.careModeEnabled) guidance.careTargetLine else guidance.targetLine

    Card(
        modifier = modifier.fillMaxWidth(),
        shape = RoundedCornerShape(22.dp),
        colors = CardDefaults.cardColors(containerColor = BaPanel.copy(alpha = 0.94f)),
        elevation = CardDefaults.cardElevation(defaultElevation = 8.dp)
    ) {
        Column(Modifier.padding(16.dp)) {
            Text(
                text = title,
                style = if (controls.careModeEnabled) MaterialTheme.typography.headlineMedium else MaterialTheme.typography.headlineSmall,
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
            Spacer(Modifier.height(6.dp))
            Text(text = detail, color = BaText, style = MaterialTheme.typography.bodyMedium)
            Text(text = target, color = BaTextMuted, style = MaterialTheme.typography.bodySmall)
            Spacer(Modifier.height(8.dp))
            Text(
                text = if (language == AppLanguage.EN) "Scenario: ${controls.assistScenario.displayName(language)}" else "场景：${controls.assistScenario.displayName(language)}",
                color = BaMint,
                style = MaterialTheme.typography.labelLarge,
                fontWeight = FontWeight.SemiBold,
                modifier = Modifier.testTag("camera_scenario_label")
            )
            Text(
                text = if (language == AppLanguage.EN) "Daily mode: ${controls.dailyUsageMode.displayName(language)}" else "日常模式：${controls.dailyUsageMode.displayName(language)}",
                color = BaSky,
                style = MaterialTheme.typography.labelLarge,
                fontWeight = FontWeight.SemiBold,
                modifier = Modifier.testTag("camera_daily_mode_label")
            )
            Text(
                text = if (controls.careModeEnabled) guidance.careExplanation else guidance.explanationHeadline,
                color = BaText,
                style = MaterialTheme.typography.bodySmall,
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
            Spacer(Modifier.height(12.dp))
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                CompactToggle(if (language == AppLanguage.EN) "Detection" else "检测", controls.detectionEnabled, Icons.Rounded.Visibility, onDetectionChange, Modifier.weight(1f), language)
                CompactToggle(if (language == AppLanguage.EN) "Speech" else "语音", controls.speechEnabled, Icons.Rounded.VolumeUp, onSpeechChange, Modifier.weight(1f), language)
                CompactToggle(if (language == AppLanguage.EN) "Vibration" else "震动", controls.vibrationEnabled, Icons.Rounded.Vibration, onVibrationChange, Modifier.weight(1f), language)
            }
            Spacer(Modifier.height(8.dp))
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                CompactAction(
                    text = if (language == AppLanguage.EN) "Quiet" else "调安静",
                    icon = Icons.Rounded.Tune,
                    onClick = onQuietShortcut,
                    modifier = Modifier.weight(1f),
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
                    modifier = Modifier.weight(1f),
                    accessibilityText = if (language == AppLanguage.EN) {
                        "Apply sensitive reminder shortcut, keep current scenario, use Sensitive profile, Standard speech, and Strong vibration"
                    } else {
                        "应用敏感提醒快捷设置，保留当前场景，使用敏感档位、标准语音和强震动"
                    }
                )
            }
            Spacer(Modifier.height(8.dp))
            CompactToggle(if (language == AppLanguage.EN) "Care" else "关怀", controls.careModeEnabled, Icons.Rounded.Favorite, onCareModeChange, Modifier.fillMaxWidth(), language)
            Spacer(Modifier.height(8.dp))
            CompactAction(
                text = if (language == AppLanguage.EN) "Scenario ${controls.assistScenario.displayName(language)}" else "场景 ${controls.assistScenario.displayName(language)}",
                icon = Icons.Rounded.Shield,
                onClick = { onScenarioChange(controls.assistScenario.next()) },
                modifier = Modifier.fillMaxWidth(),
                accessibilityText = if (language == AppLanguage.EN) {
                    "Usage scenario, current ${controls.assistScenario.displayName(language)}, tap to switch to ${controls.assistScenario.next().displayName(language)}"
                } else {
                    "使用场景，当前${controls.assistScenario.displayName(language)}，点击切换到${controls.assistScenario.next().displayName(language)}"
                }
            )
            AnimatedVisibility(
                visible = !controls.careModeEnabled,
                enter = fadeIn() + slideInVertically { it / 3 },
                exit = fadeOut()
            ) {
                Column {
                    TextButton(
                        onClick = { onDebugVisibleChange(!controls.debugVisible) },
                        modifier = Modifier.heightIn(min = 48.dp)
                    ) {
                        Icon(Icons.Rounded.BugReport, contentDescription = null, modifier = Modifier.size(18.dp))
                        Spacer(Modifier.width(6.dp))
                        Text(if (controls.debugVisible) {
                            if (language == AppLanguage.EN) "Hide debug details" else "收起调试信息"
                        } else {
                            if (language == AppLanguage.EN) "Show debug details" else "展开调试信息"
                        })
                    }
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
        }
    }
}

@Composable
private fun CameraTopBar(
    statusBadge: String,
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
                .background(BaPanel.copy(alpha = 0.78f))
        ) {
            Icon(Icons.Rounded.ArrowBack, contentDescription = "返回功能页", tint = BaText)
        }
        Spacer(Modifier.weight(1f))
        Text(
            text = statusBadge,
            color = BaInk,
            style = MaterialTheme.typography.labelLarge,
            fontWeight = FontWeight.Bold,
            modifier = Modifier
                .clip(RoundedCornerShape(50))
                .background(BaMint.copy(alpha = 0.92f))
                .padding(horizontal = 14.dp, vertical = 8.dp)
        )
    }
}

@Composable
private fun ScreenColumn(
    modifier: Modifier = Modifier,
    content: @Composable ColumnScope.() -> Unit
) {
    Column(
        modifier = modifier
            .fillMaxSize()
            .background(BaNight)
            .verticalScroll(rememberScrollState())
            .padding(horizontal = 20.dp, vertical = 22.dp),
        content = content
    )
}

@Composable
private fun ActionFeatureCard(
    title: String,
    subtitle: String,
    badge: String,
    icon: ImageVector,
    accent: Color,
    onClick: () -> Unit
) {
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .heightIn(min = 124.dp)
            .clickable(role = Role.Button, onClick = onClick)
            .semantics(mergeDescendants = true) {},
        shape = RoundedCornerShape(18.dp),
        colors = CardDefaults.cardColors(containerColor = BaPanel)
    ) {
        Row(
            verticalAlignment = Alignment.CenterVertically,
            modifier = Modifier.padding(18.dp)
        ) {
            Box(
                modifier = Modifier
                    .size(54.dp)
                    .clip(RoundedCornerShape(16.dp))
                    .background(accent.copy(alpha = 0.16f)),
                contentAlignment = Alignment.Center
            ) {
                Icon(icon, contentDescription = null, tint = accent)
            }
            Spacer(Modifier.width(14.dp))
            Column(Modifier.weight(1f)) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text(
                        text = title,
                        color = BaText,
                        style = MaterialTheme.typography.titleMedium,
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
private fun InfoStrip(
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

@Composable
private fun FieldTestSummaryCard(summary: FieldTestSummaryUiState) {
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .semantics(mergeDescendants = true) {
                contentDescription = summary.accessibilityText
            },
        shape = RoundedCornerShape(16.dp),
        colors = CardDefaults.cardColors(containerColor = BaPanel)
    ) {
        FieldTestSummaryBlock(
            summary = summary,
            modifier = Modifier.padding(16.dp)
        )
    }
}

@Composable
private fun FieldTestSummaryBlock(
    summary: FieldTestSummaryUiState,
    modifier: Modifier = Modifier
) {
    Column(
        modifier = modifier.semantics(mergeDescendants = true) {
            contentDescription = summary.accessibilityText
        }
    ) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Icon(Icons.Rounded.BugReport, contentDescription = null, tint = BaMint, modifier = Modifier.size(22.dp))
            Spacer(Modifier.width(10.dp))
            Column(Modifier.weight(1f)) {
                Text(
                    text = summary.title,
                    color = BaText,
                    fontWeight = FontWeight.Bold,
                    modifier = Modifier.semantics { heading() }
                )
                Text(summary.statusText, color = BaTextMuted, style = MaterialTheme.typography.bodySmall)
            }
        }
        Spacer(Modifier.height(10.dp))
        Text(
            text = summary.detailText,
            color = BaTextMuted,
            style = MaterialTheme.typography.bodySmall
        )
    }
}

@Composable
private fun StatusGrid(
    leftTitle: String,
    leftBody: String,
    rightTitle: String,
    rightBody: String
) {
    Row(horizontalArrangement = Arrangement.spacedBy(12.dp), modifier = Modifier.fillMaxWidth()) {
        StatusCell(leftTitle, leftBody, Modifier.weight(1f))
        StatusCell(rightTitle, rightBody, Modifier.weight(1f))
    }
}

@Composable
private fun StatusCell(title: String, body: String, modifier: Modifier = Modifier) {
    Card(
        modifier = modifier.heightIn(min = 92.dp),
        shape = RoundedCornerShape(16.dp),
        colors = CardDefaults.cardColors(containerColor = BaPanel)
    ) {
        Column(Modifier.padding(14.dp)) {
            Text(text = title, color = BaTextMuted, style = MaterialTheme.typography.labelLarge)
            Spacer(Modifier.height(6.dp))
            Text(
                text = body,
                color = BaText,
                style = MaterialTheme.typography.bodyMedium,
                fontWeight = FontWeight.SemiBold,
                maxLines = 3,
                overflow = TextOverflow.Ellipsis
            )
        }
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

@Composable
private fun CompactToggle(
    text: String,
    checked: Boolean,
    icon: ImageVector,
    onCheckedChange: (Boolean) -> Unit,
    modifier: Modifier = Modifier,
    language: AppLanguage = AppLanguage.ZH
) {
    val stateShort = if (language == AppLanguage.EN) {
        if (checked) "on" else "off"
    } else {
        if (checked) "开" else "关"
    }
    val stateLong = LocalizedText.enabled(checked, language)
    CompactAction(
        text = "$text $stateShort",
        icon = icon,
        selected = checked,
        onClick = { onCheckedChange(!checked) },
        modifier = modifier,
        accessibilityText = if (language == AppLanguage.EN) {
            "$text, currently $stateLong, tap to ${if (checked) "turn off" else "turn on"}"
        } else {
            "$text，当前$stateLong，点击${if (checked) "关闭" else "开启"}"
        }
    )
}

@Composable
private fun CompactAction(
    text: String,
    icon: ImageVector,
    onClick: () -> Unit,
    modifier: Modifier = Modifier,
    selected: Boolean = true,
    accessibilityText: String = text
) {
    val background = if (selected) BaMint else BaPanelSoft
    val foreground = if (selected) BaInk else BaText
    Button(
        onClick = onClick,
        modifier = modifier
            .heightIn(min = 48.dp)
            .widthIn(min = 64.dp)
            .semantics {
                contentDescription = accessibilityText
                stateDescription = if (selected) "已启用" else "未启用"
            },
        shape = RoundedCornerShape(14.dp),
        colors = ButtonDefaults.buttonColors(
            containerColor = background,
            contentColor = foreground
        ),
        contentPadding = PaddingValues(horizontal = 8.dp, vertical = 8.dp)
    ) {
        Icon(icon, contentDescription = null, modifier = Modifier.size(17.dp))
        Spacer(Modifier.width(5.dp))
        Text(text = text, maxLines = 1, overflow = TextOverflow.Ellipsis)
    }
}

private enum class BottomTab(val label: String, val icon: ImageVector) {
    Features("功能", Icons.Rounded.Home),
    Profile("个人主页", Icons.Rounded.Person),
    Settings("设置", Icons.Rounded.Settings)
}

private data class OnboardingPage(
    val title: String,
    val body: String,
    val detail: String,
    val icon: ImageVector,
    val accent: Color
)

private fun onboardingPages(): List<OnboardingPage> = listOf(
    OnboardingPage(
        title = "使用手机摄像头进行本地识别",
        body = "打开手机摄像头后，App 会在本机运行 YOLO11n 模型，识别画面中的常见目标和相对方向。",
        detail = "画面不上传、不联网、不保存视频，当前版本优先支持手机摄像头。",
        icon = Icons.Rounded.CameraAlt,
        accent = BaMint
    ),
    OnboardingPage(
        title = "通过语音和震动给出辅助提醒",
        body = "当规则层判断近处或迫近风险时，系统会用短句语音和震动帮助你注意前方变化。",
        detail = "你可以在设置页调整语音、震动、关怀模式和提醒档位。",
        icon = Icons.Rounded.Vibration,
        accent = BaSky
    ),
    OnboardingPage(
        title = "不能替代盲杖、导盲犬或人工判断",
        body = "BlindAssist 是助盲避障原型，提醒可能受光照、遮挡、设备性能和模型识别结果影响。",
        detail = "行走时请继续保留人工判断和专业辅助方式，把 App 提醒作为额外参考。",
        icon = Icons.Rounded.Shield,
        accent = BaAmber
    )
)

private fun enabledText(enabled: Boolean, language: AppLanguage = AppLanguage.ZH): String {
    return LocalizedText.enabled(enabled, language)
}

private val BaNight = Color(0xFF061115)
private val BaPanel = Color(0xFF111D23)
private val BaPanelSoft = Color(0xFF1C2D34)
internal val BaMint = Color(0xFF71F6C5)
internal val BaSky = Color(0xFF8AC7FF)
internal val BaAmber = Color(0xFFFFD66B)
private val BaDanger = Color(0xFFFF6B7E)
private val BaText = Color(0xFFEAF3F4)
private val BaTextMuted = Color(0xFFAFC0C6)
private val BaInk = Color(0xFF061115)

@Preview(showBackground = true, backgroundColor = 0xFF061115)
@Composable
private fun SplashPreview() {
    BlindAssistTheme {
        BrandSplashScreen(onFinished = {})
    }
}

@Preview(showBackground = true, backgroundColor = 0xFF061115)
@Composable
private fun OnboardingPreview() {
    BlindAssistTheme {
        OnboardingScreen(onFinished = {})
    }
}

@Preview(showBackground = true, backgroundColor = 0xFF061115)
@Composable
private fun FeaturePreview() {
    BlindAssistTheme {
        FeatureScreen(
            controls = previewControls(),
            modelStatus = "YOLO11n ready",
            appVersion = "5.8.0",
            onOpenCamera = {},
            onGlassesPlaceholder = {},
            onDailyUsageModeChange = {}
        )
    }
}

@Preview(showBackground = true, backgroundColor = 0xFF061115)
@Composable
private fun SettingsPreview() {
    BlindAssistTheme {
        SettingsScreen(
            controls = previewControls(),
            fieldTestSummary = previewFieldTestSummary(),
            onSpeechChange = {},
            onVibrationChange = {},
            onCareModeChange = {},
            onDebugVisibleChange = {},
            onProfileChange = {},
            onScenarioChange = {},
            onSpeechStyleChange = {},
            onVibrationStrengthChange = {},
            onLanguageChange = {},
            onShowOnboarding = {}
        )
    }
}

@Preview(showBackground = true, backgroundColor = 0xFF061115)
@Composable
private fun CameraPanelPreview() {
    BlindAssistTheme {
        Box(Modifier.fillMaxSize().background(Color.Black), contentAlignment = Alignment.BottomCenter) {
            CameraControlPanel(
                controls = previewControls(),
                guidance = previewGuidance(),
                fieldTestSummary = previewFieldTestSummary(),
                onDetectionChange = {},
                onSpeechChange = {},
                onVibrationChange = {},
                onCareModeChange = {},
                onDebugVisibleChange = {},
                onProfileChange = {},
                onScenarioChange = {},
                onQuietShortcut = {},
                onSensitiveShortcut = {},
                modifier = Modifier.padding(12.dp)
            )
        }
    }
}

private fun previewControls(): AssistControlsUiState {
    return AssistControlsUiState(
        detectionEnabled = true,
        speechEnabled = true,
        vibrationEnabled = true,
        careModeEnabled = false,
        debugVisible = true,
        alertProfile = AlertProfile.STANDARD,
        assistScenario = AssistScenario.CORRIDOR,
        speechStyle = SpeechStyle.STANDARD,
        vibrationStrength = VibrationStrength.STANDARD,
        appLanguage = AppLanguage.ZH,
        dailyUsageMode = DailyUsageMode.fromPreferences(
            scenario = AssistScenario.CORRIDOR,
            profile = AlertProfile.STANDARD,
            speechStyle = SpeechStyle.STANDARD,
            vibrationStrength = VibrationStrength.STANDARD,
            careModeEnabled = false
        )
    )
}

private fun previewFieldTestSummary(): FieldTestSummaryUiState {
    return FieldTestSummaryUiState(
        title = "现场测试摘要",
        statusText = "本次相机会话进行中",
        detailText = "运行时长：1分12秒\n最近30帧：风险8次，迫近2次，高/中/低/无 2/4/2/22\n提醒触发：语音3次，震动4次\n平均性能：FPS 18.4，推理 28ms\n当前档位：标准\n当前场景：走廊通行\n最近解释：暂不重复：提醒冷却中",
        accessibilityText = "现场测试摘要，本次相机会话进行中，运行时长1分12秒，最近30帧风险8次，语音提醒3次，震动提醒4次，平均FPS 18.4，平均推理28毫秒，当前档位标准，当前场景走廊通行。"
    )
}

private fun previewGuidance(): CameraGuidanceUiState {
    return CameraGuidanceUiState(
        title = "安全观察中",
        detail = "等待实时画面和稳定风险结果",
        targetLine = "模型状态：YOLO11n ready",
        careTitle = "前方平稳",
        careDetail = "请自然前进，系统会在前方有风险时提醒",
        careTargetLine = "建议同时保留语音和震动提醒",
        debugText = "FPS：18.0\n模型：YOLO11n ready\n最近风险判定：安全",
        scenarioName = "走廊通行",
        explanationHeadline = "继续观察：暂无可反馈风险",
        explanationDetail = "检测到目标，但未达到近处或迫近提醒条件。",
        careExplanation = "继续观察，暂无可反馈风险",
        titleColor = android.graphics.Color.rgb(99, 230, 166),
        statusBadge = "平稳",
        badgeColor = android.graphics.Color.rgb(160, 255, 215),
        badgeTextColor = android.graphics.Color.rgb(6, 24, 18),
        careAccessibilitySummary = "前方平稳",
        accessibilitySummary = "安全观察中",
        accessibilityKey = "preview"
    )
}
