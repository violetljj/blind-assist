package com.linnan.blindassist.ui.compose

import androidx.camera.view.PreviewView
import android.widget.ImageView
import com.linnan.blindassist.alert.AlertProfile
import com.linnan.blindassist.alert.AssistScenario
import com.linnan.blindassist.feedback.SpeechStyle
import com.linnan.blindassist.feedback.VibrationStrength
import com.linnan.blindassist.goal.GoalHandoffState
import com.linnan.blindassist.localization.AppLanguage
import com.linnan.blindassist.model.AssistInputSource
import com.linnan.blindassist.model.ReplayScenario
import com.linnan.blindassist.preferences.DailyUsageMode
import com.linnan.blindassist.ui.DetectionOverlayView

/** Stable UI interface for the full BlindAssist screen tree. */
data class BlindAssistAppState(
    val controls: AssistControlsUiState,
    val cameraGuidance: CameraGuidanceUiState,
    val fieldTestSummary: FieldTestSummaryUiState,
    val modelStatus: String,
    val appVersion: String,
    val cameraActive: Boolean,
    val activeInputSource: AssistInputSource,
    val activeReplayScenario: ReplayScenario?,
    val showOnboarding: Boolean,
    val showGlassesCenter: Boolean,
    val glassesSimulator: GlassesSimulatorUiState,
    /** Null keeps Goal Copilot handoff UI inactive in the default app. */
    val goalHandoff: GoalHandoffState? = null,
    val editionLabel: String? = null
)

data class BlindAssistAppActions(
    val runtime: AssistRuntimeUiActions,
    val navigation: AssistNavigationActions,
    val glasses: GlassesSimulatorActions,
    val goalHandoff: GoalHandoffActions = GoalHandoffActions()
)

data class GoalHandoffActions(
    val onConfirmByButton: () -> Unit = {}
)

data class AssistRuntimeUiActions(
    val onOpenCamera: () -> Unit,
    val onCloseCamera: () -> Unit,
    val onStartOfflineReplay: (ReplayScenario) -> Unit,
    val onDetectionChange: (Boolean) -> Unit,
    val onSpeechChange: (Boolean) -> Unit,
    val onVibrationChange: (Boolean) -> Unit,
    val onCareModeChange: (Boolean) -> Unit,
    val onDebugVisibleChange: (Boolean) -> Unit,
    val onProfileChange: (AlertProfile) -> Unit,
    val onScenarioChange: (AssistScenario) -> Unit,
    val onSpeechStyleChange: (SpeechStyle) -> Unit,
    val onVibrationStrengthChange: (VibrationStrength) -> Unit,
    val onDailyUsageModeChange: (DailyUsageMode) -> Unit,
    val onQuietShortcut: () -> Unit,
    val onSensitiveShortcut: () -> Unit,
    val onLanguageChange: (AppLanguage) -> Unit,
    val onCameraViewsReady: (PreviewView?, ImageView?, DetectionOverlayView) -> Unit
)

data class AssistNavigationActions(
    val onCompleteOnboarding: () -> Unit,
    val onShowOnboarding: () -> Unit,
    val onShowGlassesCenter: () -> Unit,
    val onDismissGlassesCenter: () -> Unit
)

data class GlassesSimulatorActions(
    val onConnect: () -> Unit,
    val onDisconnect: () -> Unit,
    val onStartLiveAssist: (String) -> Unit,
    val onReplayScenarioSelected: (ReplayScenario) -> Unit
)
