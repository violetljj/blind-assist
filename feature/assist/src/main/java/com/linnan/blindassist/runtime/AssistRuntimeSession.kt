package com.linnan.blindassist.runtime

import androidx.camera.view.PreviewView
import com.linnan.blindassist.alert.AlertProfile
import com.linnan.blindassist.alert.AssistScenario
import com.linnan.blindassist.feedback.SpeechStyle
import com.linnan.blindassist.feedback.VibrationStrength
import com.linnan.blindassist.localization.AppLanguage
import com.linnan.blindassist.model.ReplayScenario
import com.linnan.blindassist.preferences.DailyUsageMode
import com.linnan.blindassist.ui.DetectionOverlayView

/** The small external seam for the lifecycle of one assist session. */
interface AssistSession {
    fun initialize()
    fun dispatch(intent: AssistRuntimeIntent)
    fun shutdown()
}

sealed interface AssistRuntimeIntent {
    data object OpenPhoneCamera : AssistRuntimeIntent
    data class OpenOfflineReplay(val scenario: ReplayScenario) : AssistRuntimeIntent
    data object CloseCamera : AssistRuntimeIntent
    data class PermissionExplanationAccepted(val launchPermissionRequest: () -> Unit) : AssistRuntimeIntent
    data object DismissPermissionFlow : AssistRuntimeIntent
    data class CameraPermissionResult(val granted: Boolean) : AssistRuntimeIntent
    data class CameraViewsReady(val preview: PreviewView?, val overlay: DetectionOverlayView) : AssistRuntimeIntent
    data class DetectionEnabled(val enabled: Boolean) : AssistRuntimeIntent
    data class SpeechEnabled(val enabled: Boolean) : AssistRuntimeIntent
    data class VibrationEnabled(val enabled: Boolean) : AssistRuntimeIntent
    data class SpeechStyleChanged(val style: SpeechStyle) : AssistRuntimeIntent
    data class VibrationStrengthChanged(val strength: VibrationStrength) : AssistRuntimeIntent
    data class AppLanguageChanged(val language: AppLanguage) : AssistRuntimeIntent
    data class CareModeEnabled(val enabled: Boolean) : AssistRuntimeIntent
    data class AlertProfileChanged(val profile: AlertProfile) : AssistRuntimeIntent
    data class AssistScenarioChanged(val scenario: AssistScenario) : AssistRuntimeIntent
    data class DailyUsageModeChanged(val mode: DailyUsageMode) : AssistRuntimeIntent
    data object QuietShortcutSelected : AssistRuntimeIntent
    data object SensitiveShortcutSelected : AssistRuntimeIntent
}
