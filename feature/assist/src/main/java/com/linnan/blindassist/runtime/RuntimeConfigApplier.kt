package com.linnan.blindassist.runtime

import com.linnan.blindassist.feedback.FeedbackController
import com.linnan.blindassist.ui.DetectionOverlayView
import javax.inject.Inject

class RuntimeConfigApplier @Inject constructor(
    private val feedbackController: FeedbackController
) {
    fun apply(config: AssistRuntimeConfig, overlayView: DetectionOverlayView? = null) {
        feedbackController.speechEnabled = config.speechEnabled
        feedbackController.vibrationEnabled = config.vibrationEnabled
        feedbackController.speechStyle = config.speechStyle
        feedbackController.vibrationStrength = config.vibrationStrength
        feedbackController.appLanguage = config.appLanguage
        overlayView?.setCareMode(config.careModeEnabled)
    }
}
