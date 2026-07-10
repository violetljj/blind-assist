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
            appVersion = "8.2.0",
            onOpenCamera = {},
            onShowGlassesCenter = {},
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
        title = "持续检测中",
        detail = "等待实时画面和稳定风险结果",
        targetLine = "模型状态：YOLO11n ready",
        careTitle = "持续检测中",
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
        careAccessibilitySummary = "持续检测中，请继续确认周围环境",
        accessibilitySummary = "持续检测中，请继续确认周围环境",
        accessibilityKey = "preview"
    )
}
