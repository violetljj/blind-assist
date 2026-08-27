package com.linnan.blindassist.ui.compose

import android.app.Activity
import android.os.Build
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
import androidx.compose.foundation.BorderStroke
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
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.navigationBarsPadding
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
import androidx.compose.material.icons.outlined.Home
import androidx.compose.material.icons.outlined.Settings
import androidx.compose.material.icons.rounded.Info
import androidx.compose.material.icons.rounded.Person
import androidx.compose.material.icons.rounded.PhoneAndroid
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
import androidx.compose.material3.NavigationBarItemDefaults
import androidx.compose.material3.NavigationBarItem
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Surface
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.SideEffect
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
import androidx.compose.ui.platform.LocalView
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
import androidx.core.view.WindowCompat
import com.linnan.blindassist.alert.AlertProfile
import com.linnan.blindassist.alert.AssistScenario
import com.linnan.blindassist.feedback.SpeechStyle
import com.linnan.blindassist.feedback.VibrationStrength
import com.linnan.blindassist.localization.AppLanguage
import com.linnan.blindassist.localization.LocalizedText
import com.linnan.blindassist.model.AssistInputSource
import com.linnan.blindassist.model.ReplayScenario
import com.linnan.blindassist.preferences.DailyUsageMode
import com.linnan.blindassist.ui.DetectionOverlayView


@Composable
fun BlindAssistApp(
    state: BlindAssistAppState,
    actions: BlindAssistAppActions,
    modifier: Modifier = Modifier
) {
    BackHandler(enabled = state.cameraActive) {
        actions.runtime.onCloseCamera()
    }

    if (state.cameraActive || state.showOnboarding || state.showGlassesCenter) {
        SystemBarAppearance(
            darkStatusIcons = !state.cameraActive,
            darkNavigationIcons = !state.cameraActive
        )
    }

    Surface(
        modifier = modifier.fillMaxSize(),
        color = BaHomeBackground
    ) {
        Box(modifier = Modifier.fillMaxSize()) {
            AnimatedContent(
                targetState = state.cameraActive,
                label = "camera-shell"
            ) { isCameraActive ->
                when {
                    isCameraActive -> CameraExperienceScreen(
                        controls = state.controls,
                        guidance = state.cameraGuidance,
                        fieldTestSummary = state.fieldTestSummary,
                        goalHandoffState = state.goalHandoff,
                        inputSource = state.activeInputSource,
                        replayScenario = state.activeReplayScenario,
                        onBack = actions.runtime.onCloseCamera,
                        onDetectionChange = actions.runtime.onDetectionChange,
                        onSpeechChange = actions.runtime.onSpeechChange,
                        onVibrationChange = actions.runtime.onVibrationChange,
                        onCareModeChange = actions.runtime.onCareModeChange,
                        onDebugVisibleChange = actions.runtime.onDebugVisibleChange,
                        onProfileChange = actions.runtime.onProfileChange,
                        onScenarioChange = actions.runtime.onScenarioChange,
                        onQuietShortcut = actions.runtime.onQuietShortcut,
                        onSensitiveShortcut = actions.runtime.onSensitiveShortcut,
                        onGoalCompletionConfirmed = actions.goalHandoff.onConfirmByButton,
                        onCameraViewsReady = actions.runtime.onCameraViewsReady
                    )
                    state.showOnboarding -> OnboardingScreen(onFinished = actions.navigation.onCompleteOnboarding)
                    state.showGlassesCenter -> GlassesHardwareScreen(
                        state = state.glassesSimulator,
                        language = state.controls.appLanguage,
                        onBack = actions.navigation.onDismissGlassesCenter,
                        onConnect = actions.glasses.onConnect,
                        onDisconnect = actions.glasses.onDisconnect,
                        onStartLiveAssist = actions.glasses.onStartLiveAssist,
                        onReplayScenarioSelected = actions.glasses.onReplayScenarioSelected,
                        onStartReplay = actions.runtime.onStartOfflineReplay
                    )
                    else -> MainShell(
                        controls = state.controls,
                        fieldTestSummary = state.fieldTestSummary,
                        modelStatus = state.modelStatus,
                        appVersion = state.appVersion,
                        glassesState = state.glassesSimulator,
                        onOpenCamera = actions.runtime.onOpenCamera,
                        onShowGlassesCenter = actions.navigation.onShowGlassesCenter,
                        onSpeechChange = actions.runtime.onSpeechChange,
                        onVibrationChange = actions.runtime.onVibrationChange,
                        onCareModeChange = actions.runtime.onCareModeChange,
                        onDebugVisibleChange = actions.runtime.onDebugVisibleChange,
                        onProfileChange = actions.runtime.onProfileChange,
                        onScenarioChange = actions.runtime.onScenarioChange,
                        onSpeechStyleChange = actions.runtime.onSpeechStyleChange,
                        onVibrationStrengthChange = actions.runtime.onVibrationStrengthChange,
                        onDailyUsageModeChange = actions.runtime.onDailyUsageModeChange,
                        onQuietShortcut = actions.runtime.onQuietShortcut,
                        onSensitiveShortcut = actions.runtime.onSensitiveShortcut,
                        onLanguageChange = actions.runtime.onLanguageChange,
                        onShowOnboarding = actions.navigation.onShowOnboarding
                    )
                }
            }
            state.editionLabel?.let { label ->
                ExperimentalEditionBanner(
                    label = label,
                    modifier = Modifier
                        .align(Alignment.TopCenter)
                        .statusBarsPadding()
                        .padding(top = 8.dp, start = 12.dp, end = 12.dp)
                )
            }
        }
    }
}

@Composable
fun ExperimentalEditionBanner(
    label: String,
    modifier: Modifier = Modifier
) {
    Surface(
        modifier = modifier.testTag("experimental_edition_banner"),
        color = MaterialTheme.colorScheme.errorContainer,
        contentColor = MaterialTheme.colorScheme.onErrorContainer,
        shape = RoundedCornerShape(12.dp),
        tonalElevation = 4.dp
    ) {
        Text(
            text = label,
            modifier = Modifier.padding(horizontal = 12.dp, vertical = 8.dp),
            style = MaterialTheme.typography.labelLarge
        )
    }
}

@Composable
private fun MainShell(
    controls: AssistControlsUiState,
    fieldTestSummary: FieldTestSummaryUiState,
    modelStatus: String,
    appVersion: String,
    glassesState: GlassesSimulatorUiState,
    onOpenCamera: () -> Unit,
    onShowGlassesCenter: () -> Unit,
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
    onShowOnboarding: () -> Unit,
    modifier: Modifier = Modifier
) {
    var selectedTab by rememberSaveable { mutableStateOf(BottomTab.Features.name) }
    val tabs = BottomTab.entries
    val language = controls.appLanguage
    val currentTab = tabs.firstOrNull { it.name == selectedTab } ?: BottomTab.Features

    SystemBarAppearance(
        darkStatusIcons = true,
        darkNavigationIcons = true
    )

    LaunchedEffect(selectedTab) {
        if (selectedTab != currentTab.name) selectedTab = currentTab.name
    }

    Scaffold(
        modifier = modifier.fillMaxSize(),
        containerColor = BaHomeBackground,
        bottomBar = {
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .navigationBarsPadding()
                    .padding(horizontal = 18.dp, vertical = 8.dp)
            ) {
                Surface(
                    modifier = Modifier.fillMaxWidth(),
                    color = BaHomeSurface.copy(alpha = 0.98f),
                    contentColor = BaHomeInk,
                    shape = BaShapeCard,
                    shadowElevation = 6.dp,
                    tonalElevation = 0.dp,
                    border = BorderStroke(1.dp, BaHomeHairline.copy(alpha = 0.8f))
                ) {
                    NavigationBar(
                        modifier = Modifier.height(72.dp),
                        containerColor = Color.Transparent,
                        contentColor = BaHomeInk,
                        tonalElevation = 0.dp,
                        windowInsets = WindowInsets(0, 0, 0, 0)
                    ) {
                        tabs.forEach { tab ->
                            NavigationBarItem(
                                selected = currentTab == tab,
                                onClick = { selectedTab = tab.name },
                                icon = {
                                    Icon(
                                        imageVector = tab.icon,
                                        contentDescription = null,
                                        modifier = Modifier.size(24.dp)
                                    )
                                },
                                label = {
                                    Text(
                                        tab.label(language),
                                        maxLines = 1,
                                        style = MaterialTheme.typography.labelLarge,
                                        fontWeight = if (currentTab == tab) FontWeight.Bold else FontWeight.Medium
                                    )
                                },
                                colors = NavigationBarItemDefaults.colors(
                                    selectedIconColor = BaHomeGreen,
                                    selectedTextColor = BaHomeGreen,
                                    indicatorColor = BaHomeNavIndicator,
                                    unselectedIconColor = BaHomeNavInactive,
                                    unselectedTextColor = BaHomeNavInactive
                                )
                            )
                        }
                    }
                }
            }
        }
    ) { innerPadding ->
        Box(
            modifier = Modifier
                .fillMaxSize()
                .padding(innerPadding)
        ) {
            when (currentTab) {
                BottomTab.Features -> FeatureScreen(
                    controls = controls,
                    modelStatus = modelStatus,
                    appVersion = appVersion,
                    onOpenCamera = onOpenCamera,
                    onShowGlassesCenter = onShowGlassesCenter,
                    onDailyUsageModeChange = onDailyUsageModeChange,
                    glassesConnectionState = glassesState.connectionState,
                    onQuietShortcut = onQuietShortcut,
                    onSensitiveShortcut = onSensitiveShortcut
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
private fun SystemBarAppearance(
    darkStatusIcons: Boolean,
    darkNavigationIcons: Boolean
) {
    val view = LocalView.current
    SideEffect {
        val window = (view.context as? Activity)?.window ?: return@SideEffect
        WindowCompat.getInsetsController(window, view).apply {
            isAppearanceLightStatusBars = darkStatusIcons
            isAppearanceLightNavigationBars = darkNavigationIcons
        }
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            window.isNavigationBarContrastEnforced = false
        }
    }
}


private enum class BottomTab(
    private val zhLabel: String,
    private val enLabel: String,
    val icon: ImageVector
) {
    Features("辅助", "Assist", Icons.Outlined.Home),
    Settings("设置", "Settings", Icons.Outlined.Settings);

    fun label(language: AppLanguage): String {
        return if (language == AppLanguage.EN) enLabel else zhLabel
    }
}
