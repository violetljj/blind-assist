package com.linnan.blindassist.runtime

import com.linnan.blindassist.model.AssistInputSource

class AssistRuntimeStateMachine(
    initialState: AssistRuntimeState = AssistRuntimeState.Idle,
    private var cameraViewsReady: Boolean = false
) {
    var currentState: AssistRuntimeState = initialState
        private set

    fun onEvent(event: AssistRuntimeEvent): AssistRuntimeTransition {
        val effects = mutableListOf<AssistRuntimeEffect>()
        when (event) {
            is AssistRuntimeEvent.OpenCamera -> {
                if (event.inputSource != AssistInputSource.PHONE_CAMERA || event.hasCameraPermission) {
                    currentState = AssistRuntimeState.Starting
                    effects += startCameraSessionEffects(event.modelReady)
                } else {
                    currentState = AssistRuntimeState.PermissionExplaining
                    effects += AssistRuntimeEffect.ShowPermissionExplanation
                }
            }
            AssistRuntimeEvent.PermissionExplanationAccepted -> {
                currentState = AssistRuntimeState.PermissionRequesting
                effects += AssistRuntimeEffect.DismissPermissionExplanation
                effects += AssistRuntimeEffect.LaunchPermissionRequest
            }
            AssistRuntimeEvent.PermissionFlowDismissed -> {
                if (
                    currentState == AssistRuntimeState.PermissionExplaining ||
                    currentState == AssistRuntimeState.PermissionRequesting ||
                    currentState == AssistRuntimeState.PermissionDenied
                ) {
                    currentState = AssistRuntimeState.Idle
                    effects += AssistRuntimeEffect.DismissPermissionExplanation
                    effects += AssistRuntimeEffect.DismissPermissionDenied
                }
            }
            is AssistRuntimeEvent.PermissionResult -> {
                if (event.granted) {
                    currentState = AssistRuntimeState.Starting
                    effects += AssistRuntimeEffect.DismissPermissionExplanation
                    effects += startCameraSessionEffects(event.modelReady)
                } else {
                    currentState = AssistRuntimeState.PermissionDenied
                    effects += AssistRuntimeEffect.DismissPermissionExplanation
                    effects += AssistRuntimeEffect.Render(AssistRuntimeRenderTarget.PermissionDenied)
                    effects += AssistRuntimeEffect.ShowPermissionDenied
                }
            }
            AssistRuntimeEvent.CameraViewsReady -> {
                cameraViewsReady = true
                effects += AssistRuntimeEffect.ApplyConfig
                if (currentState == AssistRuntimeState.Starting) {
                    effects += AssistRuntimeEffect.StartCameraIfReady
                }
            }
            is AssistRuntimeEvent.CameraStarted -> {
                currentState = if (event.modelReady) AssistRuntimeState.Running else AssistRuntimeState.Error("Model unavailable")
                effects += if (event.modelReady) {
                    AssistRuntimeEffect.Render(AssistRuntimeRenderTarget.Waiting)
                } else {
                    AssistRuntimeEffect.Render(AssistRuntimeRenderTarget.ModelUnavailable)
                }
            }
            is AssistRuntimeEvent.CameraSourceFailed -> {
                currentState = AssistRuntimeState.Error(event.message)
                effects += AssistRuntimeEffect.StopCamera
                effects += AssistRuntimeEffect.ClearOverlay
                effects += AssistRuntimeEffect.StopSession
                effects += AssistRuntimeEffect.Render(AssistRuntimeRenderTarget.CameraError, event.message)
            }
            AssistRuntimeEvent.CloseCamera -> {
                cameraViewsReady = false
                currentState = AssistRuntimeState.Idle
                effects += AssistRuntimeEffect.StopCamera
                effects += AssistRuntimeEffect.ClearOverlay
                effects += AssistRuntimeEffect.CloseCamera
                effects += AssistRuntimeEffect.StopSession
            }
            is AssistRuntimeEvent.DetectionChanged -> {
                if (event.enabled) {
                    if (currentState == AssistRuntimeState.DetectionPaused) {
                        currentState = if (cameraViewsReady) AssistRuntimeState.Running else AssistRuntimeState.Starting
                        effects += AssistRuntimeEffect.StartSession
                        effects += AssistRuntimeEffect.Render(AssistRuntimeRenderTarget.Waiting)
                        if (cameraViewsReady) {
                            effects += AssistRuntimeEffect.StartCameraIfReady
                        }
                    }
                } else if (currentState == AssistRuntimeState.Running || currentState == AssistRuntimeState.Starting) {
                    currentState = AssistRuntimeState.DetectionPaused
                    effects += AssistRuntimeEffect.ClearOverlay
                    effects += AssistRuntimeEffect.StopSession
                    effects += AssistRuntimeEffect.Render(AssistRuntimeRenderTarget.Paused)
                }
            }
        }
        return AssistRuntimeTransition(currentState, effects)
    }

    private fun startCameraSessionEffects(modelReady: Boolean): List<AssistRuntimeEffect> {
        return buildList {
            add(AssistRuntimeEffect.StartSession)
            add(AssistRuntimeEffect.ActivateCamera)
            add(AssistRuntimeEffect.Render(AssistRuntimeRenderTarget.Starting))
            if (!modelReady) {
                add(AssistRuntimeEffect.Render(AssistRuntimeRenderTarget.ModelUnavailable))
            }
            if (cameraViewsReady) {
                add(AssistRuntimeEffect.StartCameraIfReady)
            }
        }
    }
}

sealed interface AssistRuntimeState {
    data object Idle : AssistRuntimeState
    data object PermissionExplaining : AssistRuntimeState
    data object PermissionRequesting : AssistRuntimeState
    data object Starting : AssistRuntimeState
    data object Running : AssistRuntimeState
    data object DetectionPaused : AssistRuntimeState
    data object PermissionDenied : AssistRuntimeState
    data class Error(val message: String) : AssistRuntimeState
}

sealed interface AssistRuntimeEvent {
    data class OpenCamera(
        val hasCameraPermission: Boolean,
        val modelReady: Boolean,
        val inputSource: AssistInputSource = AssistInputSource.PHONE_CAMERA
    ) : AssistRuntimeEvent
    data object PermissionExplanationAccepted : AssistRuntimeEvent
    data object PermissionFlowDismissed : AssistRuntimeEvent
    data class PermissionResult(val granted: Boolean, val modelReady: Boolean) : AssistRuntimeEvent
    data object CameraViewsReady : AssistRuntimeEvent
    data class CameraStarted(val modelReady: Boolean) : AssistRuntimeEvent
    data class CameraSourceFailed(val message: String) : AssistRuntimeEvent
    data object CloseCamera : AssistRuntimeEvent
    data class DetectionChanged(val enabled: Boolean) : AssistRuntimeEvent
}

data class AssistRuntimeTransition(
    val state: AssistRuntimeState,
    val effects: List<AssistRuntimeEffect>
)

sealed interface AssistRuntimeEffect {
    data object ShowPermissionExplanation : AssistRuntimeEffect
    data object DismissPermissionExplanation : AssistRuntimeEffect
    data object LaunchPermissionRequest : AssistRuntimeEffect
    data object ActivateCamera : AssistRuntimeEffect
    data object CloseCamera : AssistRuntimeEffect
    data object StartCameraIfReady : AssistRuntimeEffect
    data object StopCamera : AssistRuntimeEffect
    data object StartSession : AssistRuntimeEffect
    data object StopSession : AssistRuntimeEffect
    data object ClearOverlay : AssistRuntimeEffect
    data object ShowPermissionDenied : AssistRuntimeEffect
    data object DismissPermissionDenied : AssistRuntimeEffect
    data object ApplyConfig : AssistRuntimeEffect
    data class Render(val target: AssistRuntimeRenderTarget, val message: String? = null) : AssistRuntimeEffect
}

enum class AssistRuntimeRenderTarget {
    Starting,
    Waiting,
    Paused,
    PermissionDenied,
    ModelUnavailable,
    CameraError
}
