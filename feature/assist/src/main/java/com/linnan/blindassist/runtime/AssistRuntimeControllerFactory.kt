package com.linnan.blindassist.runtime

import androidx.activity.ComponentActivity
import com.linnan.blindassist.camera.FrameSourceFactory
import com.linnan.blindassist.feedback.FeedbackController
import com.linnan.blindassist.session.AssistSessionCoordinator
import com.linnan.blindassist.ui.BlindAssistViewModel
import com.linnan.blindassist.vision.ObjectDetector
import javax.inject.Inject

class AssistRuntimeControllerFactory @Inject constructor(
    private val detector: ObjectDetector,
    private val feedbackController: FeedbackController,
    private val coordinator: AssistSessionCoordinator,
    private val frameSourceFactory: FrameSourceFactory,
    private val configApplier: RuntimeConfigApplier
) {
    fun create(activity: ComponentActivity, appViewModel: BlindAssistViewModel): AssistRuntimeController {
        return AssistRuntimeController(
            activity = activity,
            appViewModel = appViewModel,
            detector = detector,
            feedbackController = feedbackController,
            coordinator = coordinator,
            frameSourceFactory = frameSourceFactory,
            configApplier = configApplier
        )
    }
}
