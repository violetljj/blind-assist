package com.linnan.blindassist.runtime

import com.linnan.blindassist.feedback.FeedbackController
import com.linnan.blindassist.feedback.FeedbackRuntimeSettings
import com.linnan.blindassist.ui.DetectionOverlayView
import javax.inject.Inject

class RuntimeConfigApplier @Inject constructor(
    private val feedbackController: FeedbackController
) {
    fun apply(config: AssistRuntimeConfig, overlayView: DetectionOverlayView? = null) {
        feedbackController.applySettings(
            FeedbackRuntimeSettings(
                speechEnabled = config.speechEnabled,
                vibrationEnabled = config.vibrationEnabled,
                speechStyle = config.speechStyle,
                vibrationStrength = config.vibrationStrength,
                appLanguage = config.appLanguage
            )
        )
        overlayView?.setCareMode(config.careModeEnabled)
    }
}
