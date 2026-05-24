package com.linnan.blindassist.runtime

import com.linnan.blindassist.ui.CameraGuidanceMapper
import com.linnan.blindassist.ui.compose.CameraGuidanceUiState
import com.linnan.blindassist.vision.ObjectDetector

internal class AssistRuntimeGuidanceFactory(
    private val detector: ObjectDetector,
    private val configProvider: () -> AssistRuntimeConfig
) {
    fun initial(): CameraGuidanceUiState {
        val config = configProvider()
        return CameraGuidanceMapper.initial(detector.statusMessage, config.assistScenario, config.appLanguage)
    }

    fun starting(): CameraGuidanceUiState {
        val config = configProvider()
        return CameraGuidanceMapper.starting(detector.statusMessage, config.assistScenario, config.appLanguage)
    }

    fun waiting(): CameraGuidanceUiState {
        val config = configProvider()
        return CameraGuidanceMapper.waiting(detector.statusMessage, config.assistScenario, config.appLanguage)
    }

    fun modelUnavailable(): CameraGuidanceUiState {
        val config = configProvider()
        return CameraGuidanceMapper.modelUnavailable(detector.statusMessage, config.assistScenario, config.appLanguage)
    }

    fun paused(): CameraGuidanceUiState {
        val config = configProvider()
        return CameraGuidanceMapper.paused(config.appLanguage)
            .copy(scenarioName = config.assistScenario.displayName(config.appLanguage))
    }

    fun permissionDenied(): CameraGuidanceUiState {
        return CameraGuidanceMapper.permissionDenied(configProvider().appLanguage)
    }

    fun cameraError(message: String?): CameraGuidanceUiState {
        return CameraGuidanceMapper.cameraError(message ?: "未知错误", configProvider().appLanguage)
    }

    fun forTarget(target: AssistRuntimeRenderTarget, message: String?): CameraGuidanceUiState {
        return when (target) {
            AssistRuntimeRenderTarget.Starting -> starting()
            AssistRuntimeRenderTarget.Waiting -> if (detector.isReady) waiting() else modelUnavailable()
            AssistRuntimeRenderTarget.Paused -> paused()
            AssistRuntimeRenderTarget.PermissionDenied -> permissionDenied()
            AssistRuntimeRenderTarget.ModelUnavailable -> modelUnavailable()
            AssistRuntimeRenderTarget.CameraError -> cameraError(message)
        }
    }

    fun forState(state: AssistRuntimeState): CameraGuidanceUiState {
        return when (state) {
            AssistRuntimeState.Idle -> initial()
            AssistRuntimeState.Starting -> starting()
            AssistRuntimeState.Running -> waiting()
            AssistRuntimeState.DetectionPaused -> paused()
            AssistRuntimeState.PermissionDenied -> permissionDenied()
            is AssistRuntimeState.Error -> if (detector.isReady) cameraError(state.message) else modelUnavailable()
            AssistRuntimeState.PermissionExplaining,
            AssistRuntimeState.PermissionRequesting -> permissionDenied()
        }
    }
}
