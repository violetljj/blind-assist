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
import androidx.compose.material3.AlertDialog
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
import com.linnan.blindassist.ui.DetectionOverlayView
import kotlinx.coroutines.delay

data class AssistControlsUiState(
    val detectionEnabled: Boolean,
    val speechEnabled: Boolean,
    val vibrationEnabled: Boolean,
    val careModeEnabled: Boolean,
    val debugVisible: Boolean,
    val alertProfile: AlertProfile
)

data class CameraGuidanceUiState(
    val title: String,
    val detail: String,
    val targetLine: String,
    val careTitle: String,
    val careDetail: String,
    val careTargetLine: String,
    val debugText: String,
    val titleColor: Int,
    val statusBadge: String,
    val badgeColor: Int,
    val badgeTextColor: Int,
    val careAccessibilitySummary: String,
    val accessibilitySummary: String,
    val accessibilityKey: String
) {
    companion object {
        fun initial(modelStatus: String): CameraGuidanceUiState {
            return CameraGuidanceUiState(
                title = "初始化中",
                detail = "正在准备本地检测模型",
                targetLine = "模型状态：$modelStatus",
                careTitle = "正在准备",
                careDetail = "识别模型正在启动，请稍等",
                careTargetLine = "进入手机摄像头后会开始观察前方",
                debugText = "模型状态：$modelStatus",
                titleColor = 0xFFFFFFFF.toInt(),
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
        fun empty(profileName: String): FieldTestSummaryUiState {
            return FieldTestSummaryUiState(
                title = "现场测试摘要",
                detailText = "运行时长：尚未开始\n最近0帧：风险0次，迫近0次，高/中/低/无 0/0/0/0\n提醒触发：语音0次，震动0次\n平均性能：FPS 0.0，推理 0ms\n当前档位：$profileName",
                statusText = "等待相机会话",
                accessibilityText = "现场测试摘要，尚未开始，相机会话运行后会显示运行时长、风险次数、提醒次数、平均性能和当前档位。"
            )
        }
    }
}

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
                    modelStatus = modelStatus,
                    appVersion = appVersion,
                    onOpenCamera = onOpenCamera,
                    onGlassesPlaceholder = onGlassesPlaceholder
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
                    onShowOnboarding = onShowOnboarding
                )
            }
        }
    }
}

@Composable
fun FeatureScreen(
    modelStatus: String,
    appVersion: String,
    onOpenCamera: () -> Unit,
    onGlassesPlaceholder: () -> Unit,
    modifier: Modifier = Modifier
) {
    ScreenColumn(modifier = modifier) {
        Text(
            text = "功能",
            style = MaterialTheme.typography.headlineMedium,
            color = BaText,
            fontWeight = FontWeight.Bold,
            modifier = Modifier.semantics { heading() }
        )
        Text(
            text = "选择一种辅助方式开始。当前版本优先提供手机摄像头本地识别，眼镜连接作为后续扩展入口保留。",
            style = MaterialTheme.typography.bodyMedium,
            color = BaTextMuted
        )
        Spacer(Modifier.height(18.dp))

        ActionFeatureCard(
            title = "使用手机摄像头",
            subtitle = "打开实时画面、目标框、语音与震动提醒",
            badge = "可用",
            icon = Icons.Rounded.CameraAlt,
            accent = BaMint,
            onClick = onOpenCamera
        )
        Spacer(Modifier.height(12.dp))
        ActionFeatureCard(
            title = "连接眼镜设备",
            subtitle = "预留给蓝牙眼镜和外接视觉设备，不会申请蓝牙权限",
            badge = "占位",
            icon = Icons.Rounded.Bluetooth,
            accent = BaSky,
            onClick = onGlassesPlaceholder
        )
        Spacer(Modifier.height(18.dp))

        InfoStrip(
            icon = Icons.Rounded.Shield,
            title = "安全边界",
            body = "BlindAssist 是本地助盲避障原型，提醒结果只作为辅助参考，不能替代人工判断或专业安全设备。"
        )
        Spacer(Modifier.height(12.dp))
        StatusGrid(
            leftTitle = "模型",
            leftBody = modelStatus,
            rightTitle = "版本",
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
            leftTitle = "提醒档位",
            leftBody = controls.alertProfile.displayName,
            rightTitle = "当前版本",
            rightBody = "v$appVersion"
        )
        Spacer(Modifier.height(12.dp))
        InfoStrip(
            icon = Icons.Rounded.Favorite,
            title = "辅助偏好",
            body = "语音${enabledText(controls.speechEnabled)}，震动${enabledText(controls.vibrationEnabled)}，关怀模式${enabledText(controls.careModeEnabled)}。"
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
    onShowOnboarding: () -> Unit,
    modifier: Modifier = Modifier
) {
    ScreenColumn(modifier = modifier) {
        Text(
            text = "设置",
            style = MaterialTheme.typography.headlineMedium,
            color = BaText,
            fontWeight = FontWeight.Bold,
            modifier = Modifier.semantics { heading() }
        )
        Text(
            text = "这些偏好会影响摄像头页的提醒方式。检测开关进入相机后仍可随时控制。",
            style = MaterialTheme.typography.bodyMedium,
            color = BaTextMuted
        )
        Spacer(Modifier.height(18.dp))

        SettingSwitchRow(
            icon = Icons.Rounded.VolumeUp,
            title = "语音提醒",
            body = "播报短句式风险提示",
            checked = controls.speechEnabled,
            onCheckedChange = onSpeechChange
        )
        SettingSwitchRow(
            icon = Icons.Rounded.Vibration,
            title = "震动提醒",
            body = "在近处和迫近风险时给出触觉反馈",
            checked = controls.vibrationEnabled,
            onCheckedChange = onVibrationChange
        )
        SettingSwitchRow(
            icon = Icons.Rounded.Favorite,
            title = "关怀模式",
            body = "放大主要指导语并减少调试干扰",
            checked = controls.careModeEnabled,
            onCheckedChange = onCareModeChange
        )
        SettingSwitchRow(
            icon = Icons.Rounded.BugReport,
            title = "调试信息",
            body = "在相机页显示 FPS、耗时和风险判定摘要",
            checked = controls.debugVisible,
            onCheckedChange = onDebugVisibleChange
        )
        Spacer(Modifier.height(16.dp))
        ProfileSelector(
            selected = controls.alertProfile,
            onProfileChange = onProfileChange
        )
        Spacer(Modifier.height(16.dp))
        FieldTestSummaryCard(fieldTestSummary)
        Spacer(Modifier.height(16.dp))
        SettingsActionRow(
            icon = Icons.Rounded.Info,
            title = "查看新手引导",
            body = "重新查看摄像头、本地提醒和安全边界说明",
            onClick = onShowOnboarding
        )
        Spacer(Modifier.height(16.dp))
        InfoStrip(
            icon = Icons.Rounded.Shield,
            title = "使用边界",
            body = "提醒由本地模型与规则层生成，受光照、遮挡和设备性能影响。行走时仍需保留人工判断。"
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
    modifier: Modifier = Modifier
) {
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
            Spacer(Modifier.height(12.dp))
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                CompactToggle("检测", controls.detectionEnabled, Icons.Rounded.Visibility, onDetectionChange, Modifier.weight(1f))
                CompactToggle("语音", controls.speechEnabled, Icons.Rounded.VolumeUp, onSpeechChange, Modifier.weight(1f))
                CompactToggle("震动", controls.vibrationEnabled, Icons.Rounded.Vibration, onVibrationChange, Modifier.weight(1f))
            }
            Spacer(Modifier.height(8.dp))
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                CompactAction(
                    text = "模式 ${controls.alertProfile.displayName}",
                    icon = Icons.Rounded.Tune,
                    onClick = { onProfileChange(controls.alertProfile.next()) },
                    modifier = Modifier.weight(1f),
                    accessibilityText = "提醒档位，当前${controls.alertProfile.displayName}，点击切换到${controls.alertProfile.next().displayName}"
                )
                CompactToggle("关怀", controls.careModeEnabled, Icons.Rounded.Favorite, onCareModeChange, Modifier.weight(1f))
            }
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
                        Text(if (controls.debugVisible) "收起调试信息" else "展开调试信息")
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
    onCheckedChange: (Boolean) -> Unit
) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .heightIn(min = 72.dp)
            .clickable(
                role = Role.Switch,
                onClickLabel = if (checked) "关闭$title" else "开启$title",
                onClick = { onCheckedChange(!checked) }
            )
            .semantics(mergeDescendants = true) {
                stateDescription = if (checked) "已开启" else "已关闭"
                contentDescription = "$title，$body，当前${if (checked) "已开启" else "已关闭"}"
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
                stateDescription = if (checked) "已开启" else "已关闭"
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
    onProfileChange: (AlertProfile) -> Unit
) {
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .semantics(mergeDescendants = true) {
                contentDescription = "提醒档位，当前${selected.displayName}。安静减少打扰，敏感更早确认中风险。"
            },
        shape = RoundedCornerShape(16.dp),
        colors = CardDefaults.cardColors(containerColor = BaPanel)
    ) {
        Column(Modifier.padding(16.dp)) {
            Text("提醒档位", color = BaText, fontWeight = FontWeight.Bold, modifier = Modifier.semantics { heading() })
            Text("安静减少打扰，敏感更早确认中风险。", color = BaTextMuted, style = MaterialTheme.typography.bodySmall)
            Spacer(Modifier.height(10.dp))
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                AlertProfile.values().forEach { profile ->
                    FilterChip(
                        selected = selected == profile,
                        onClick = { onProfileChange(profile) },
                        label = { Text(profile.displayName) },
                        modifier = Modifier.heightIn(min = 48.dp).semantics {
                            role = Role.Button
                            stateDescription = if (selected == profile) "当前档位" else "未选择"
                            contentDescription = "选择${profile.displayName}提醒档位"
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
    modifier: Modifier = Modifier
) {
    CompactAction(
        text = "$text ${if (checked) "开" else "关"}",
        icon = icon,
        selected = checked,
        onClick = { onCheckedChange(!checked) },
        modifier = modifier,
        accessibilityText = "$text，当前${if (checked) "已开启" else "已关闭"}，点击${if (checked) "关闭" else "开启"}"
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

@Composable
fun GlassesPlaceholderDialog(
    onDismiss: () -> Unit
) {
    AlertDialog(
        onDismissRequest = onDismiss,
        confirmButton = {
            TextButton(onClick = onDismiss) {
                Text("知道了")
            }
        },
        icon = { Icon(Icons.Rounded.Bluetooth, contentDescription = null, tint = BaSky) },
        title = { Text("眼镜设备连接") },
        text = {
            Text("该入口为未来蓝牙眼镜或外接视觉设备预留。当前版本不会扫描蓝牙、不会联网，也不会申请额外权限。")
        }
    )
}

@Composable
fun CameraPermissionExplanationDialog(
    onContinue: () -> Unit,
    onDismiss: () -> Unit
) {
    AlertDialog(
        onDismissRequest = onDismiss,
        confirmButton = {
            Button(onClick = onContinue, modifier = Modifier.heightIn(min = 48.dp)) {
                Text("继续并授权")
            }
        },
        dismissButton = {
            TextButton(onClick = onDismiss, modifier = Modifier.heightIn(min = 48.dp)) {
                Text("暂不打开")
            }
        },
        icon = { Icon(Icons.Rounded.CameraAlt, contentDescription = null, tint = BaMint) },
        title = { Text("需要相机权限") },
        text = {
            Text("相机仅用于手机端实时识别。BlindAssist 不上传画面、不联网、不保存视频；语音和震动提醒只作为辅助参考，不能替代盲杖、导盲犬或人工判断。")
        }
    )
}

@Composable
fun CameraPermissionDeniedDialog(
    onDismiss: () -> Unit
) {
    AlertDialog(
        onDismissRequest = onDismiss,
        confirmButton = {
            TextButton(onClick = onDismiss, modifier = Modifier.heightIn(min = 48.dp)) {
                Text("知道了")
            }
        },
        icon = { Icon(Icons.Rounded.Shield, contentDescription = null, tint = BaAmber) },
        title = { Text("相机权限未开启") },
        text = {
            Text("未获得相机权限时，手机摄像头辅助无法启动。你仍可留在主界面查看设置，稍后再次点击“使用手机摄像头”重新授权。")
        }
    )
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

private fun enabledText(enabled: Boolean): String = if (enabled) "已开启" else "已关闭"

private val BaNight = Color(0xFF061115)
private val BaPanel = Color(0xFF111D23)
private val BaPanelSoft = Color(0xFF1C2D34)
private val BaMint = Color(0xFF71F6C5)
private val BaSky = Color(0xFF8AC7FF)
private val BaAmber = Color(0xFFFFD66B)
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
            modelStatus = "YOLO11n ready",
            appVersion = "3.4.0",
            onOpenCamera = {},
            onGlassesPlaceholder = {}
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
        alertProfile = AlertProfile.STANDARD
    )
}

private fun previewFieldTestSummary(): FieldTestSummaryUiState {
    return FieldTestSummaryUiState(
        title = "现场测试摘要",
        statusText = "本次相机会话进行中",
        detailText = "运行时长：1分12秒\n最近30帧：风险8次，迫近2次，高/中/低/无 2/4/2/22\n提醒触发：语音3次，震动4次\n平均性能：FPS 18.4，推理 28ms\n当前档位：标准",
        accessibilityText = "现场测试摘要，本次相机会话进行中，运行时长1分12秒，最近30帧风险8次，语音提醒3次，震动提醒4次，平均FPS 18.4，平均推理28毫秒，当前档位标准。"
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
        titleColor = android.graphics.Color.rgb(99, 230, 166),
        statusBadge = "平稳",
        badgeColor = android.graphics.Color.rgb(160, 255, 215),
        badgeTextColor = android.graphics.Color.rgb(6, 24, 18),
        careAccessibilitySummary = "前方平稳",
        accessibilitySummary = "安全观察中",
        accessibilityKey = "preview"
    )
}
