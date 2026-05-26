package com.linnan.blindassist.runtime

import com.linnan.blindassist.session.AssistSessionCoordinator
import com.linnan.blindassist.ui.BlindAssistViewModel

internal class AssistRuntimeEffectExecutor(
    private val appViewModel: BlindAssistViewModel,
    private val coordinator: AssistSessionCoordinator,
    private val renderer: AssistRuntimeRenderer,
    private val cameraLifecycleAdapter: AssistCameraLifecycleAdapter,
    private val frameProcessor: AssistFrameProcessor,
    private val configSnapshot: AssistRuntimeConfigSnapshot,
    private val syncConfigFromViewModel: () -> Unit,
    private val startCameraIfReady: () -> Unit
) {
    fun execute(
        transition: AssistRuntimeTransition,
        launchPermissionRequest: (() -> Unit)? = null
    ) {
        transition.effects.forEach { effect ->
            when (effect) {
                AssistRuntimeEffect.ShowPermissionExplanation -> appViewModel.onShowCameraPermissionDialog()
                AssistRuntimeEffect.DismissPermissionExplanation -> appViewModel.onDismissCameraPermissionDialog()
                AssistRuntimeEffect.LaunchPermissionRequest -> launchPermissionRequest?.invoke()
                AssistRuntimeEffect.StartSession -> {
                    coordinator.startSession()
                    frameProcessor.reset()
                    renderer.updateFieldTestSummary(active = true, configSnapshot.get())
                }
                AssistRuntimeEffect.ActivateCamera -> renderer.activateCamera(configSnapshot.get())
                AssistRuntimeEffect.StartCameraIfReady -> startCameraIfReady()
                AssistRuntimeEffect.StopCamera -> {
                    cameraLifecycleAdapter.stop()
                    frameProcessor.reset()
                }
                AssistRuntimeEffect.ClearOverlay -> renderer.clearOverlay()
                AssistRuntimeEffect.CloseCamera -> {
                    renderer.closeCamera(configSnapshot.get())
                    cameraLifecycleAdapter.clearViews()
                }
                AssistRuntimeEffect.ResetSession -> {
                    coordinator.reset()
                    frameProcessor.reset()
                }
                AssistRuntimeEffect.ShowPermissionDenied -> renderer.showPermissionDenied()
                AssistRuntimeEffect.ApplyConfig -> syncConfigFromViewModel()
                is AssistRuntimeEffect.Render -> renderer.renderTarget(effect.target, effect.message)
            }
        }
    }
}
