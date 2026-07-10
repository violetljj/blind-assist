package com.linnan.blindassist.runtime

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class AssistRuntimeStateMachineTest {
    @Test
    fun openCameraWithoutPermissionShowsExplanation() {
        val transition = AssistRuntimeStateMachine().onEvent(
            AssistRuntimeEvent.OpenCamera(hasCameraPermission = false, modelReady = true)
        )

        assertEquals(AssistRuntimeState.PermissionExplaining, transition.state)
        assertTrue(transition.effects.contains(AssistRuntimeEffect.ShowPermissionExplanation))
    }

    @Test
    fun permissionAcceptedLaunchesSystemRequest() {
        val machine = AssistRuntimeStateMachine()
        machine.onEvent(AssistRuntimeEvent.OpenCamera(hasCameraPermission = false, modelReady = true))

        val transition = machine.onEvent(AssistRuntimeEvent.PermissionExplanationAccepted)

        assertEquals(AssistRuntimeState.PermissionRequesting, transition.state)
        assertTrue(transition.effects.contains(AssistRuntimeEffect.DismissPermissionExplanation))
        assertTrue(transition.effects.contains(AssistRuntimeEffect.LaunchPermissionRequest))
    }

    @Test
    fun grantedPermissionWaitsForLateCameraViewsBeforeStartingSource() {
        val machine = AssistRuntimeStateMachine()

        val granted = machine.onEvent(AssistRuntimeEvent.PermissionResult(granted = true, modelReady = true))

        assertEquals(AssistRuntimeState.Starting, granted.state)
        assertTrue(granted.effects.contains(AssistRuntimeEffect.StartSession))
        assertTrue(granted.effects.contains(AssistRuntimeEffect.ActivateCamera))
        assertTrue(!granted.effects.contains(AssistRuntimeEffect.StartCameraIfReady))

        val viewsReady = machine.onEvent(AssistRuntimeEvent.CameraViewsReady)

        assertTrue(viewsReady.effects.contains(AssistRuntimeEffect.ApplyConfig))
        assertTrue(viewsReady.effects.contains(AssistRuntimeEffect.StartCameraIfReady))
    }

    @Test
    fun deniedPermissionShowsDeniedStateAndDialog() {
        val transition = AssistRuntimeStateMachine().onEvent(
            AssistRuntimeEvent.PermissionResult(granted = false, modelReady = true)
        )

        assertEquals(AssistRuntimeState.PermissionDenied, transition.state)
        assertTrue(transition.effects.contains(AssistRuntimeEffect.ShowPermissionDenied))
        assertTrue(transition.hasRenderTarget(AssistRuntimeRenderTarget.PermissionDenied))
    }

    @Test
    fun detectionCanPauseAndResumeRunningSession() {
        val machine = AssistRuntimeStateMachine(initialState = AssistRuntimeState.Running, cameraViewsReady = true)

        val paused = machine.onEvent(AssistRuntimeEvent.DetectionChanged(false))

        assertEquals(AssistRuntimeState.DetectionPaused, paused.state)
        assertTrue(paused.effects.contains(AssistRuntimeEffect.ClearOverlay))
        assertTrue(paused.effects.contains(AssistRuntimeEffect.StopSession))
        assertTrue(paused.hasRenderTarget(AssistRuntimeRenderTarget.Paused))

        val resumed = machine.onEvent(AssistRuntimeEvent.DetectionChanged(true))

        assertEquals(AssistRuntimeState.Running, resumed.state)
        assertTrue(resumed.effects.contains(AssistRuntimeEffect.StartSession))
        assertTrue(resumed.effects.contains(AssistRuntimeEffect.StartCameraIfReady))
        assertTrue(resumed.hasRenderTarget(AssistRuntimeRenderTarget.Waiting))
    }

    @Test
    fun cameraSourceFailureMovesToErrorState() {
        val transition = AssistRuntimeStateMachine(initialState = AssistRuntimeState.Starting)
            .onEvent(AssistRuntimeEvent.CameraSourceFailed("camera busy"))

        assertEquals(AssistRuntimeState.Error("camera busy"), transition.state)
        assertTrue(transition.hasRenderTarget(AssistRuntimeRenderTarget.CameraError))
        assertTrue(transition.effects.contains(AssistRuntimeEffect.StopCamera))
        assertTrue(transition.effects.contains(AssistRuntimeEffect.ClearOverlay))
    }

    @Test
    fun cameraSourceFailureStopsSourceAndClearsSession() {
        val transition = AssistRuntimeStateMachine(initialState = AssistRuntimeState.Running, cameraViewsReady = true)
            .onEvent(AssistRuntimeEvent.CameraSourceFailed("analyzer failed"))

        assertEquals(AssistRuntimeState.Error("analyzer failed"), transition.state)
        assertTrue(transition.effects.contains(AssistRuntimeEffect.StopCamera))
        assertTrue(transition.effects.contains(AssistRuntimeEffect.ClearOverlay))
        assertTrue(transition.effects.contains(AssistRuntimeEffect.StopSession))
        assertTrue(transition.hasRenderTarget(AssistRuntimeRenderTarget.CameraError))
    }

    @Test
    fun modelUnavailableIsRenderedWhenOpeningWithMissingModel() {
        val transition = AssistRuntimeStateMachine(initialState = AssistRuntimeState.Idle, cameraViewsReady = true)
            .onEvent(AssistRuntimeEvent.OpenCamera(hasCameraPermission = true, modelReady = false))

        assertEquals(AssistRuntimeState.Starting, transition.state)
        assertTrue(transition.hasRenderTarget(AssistRuntimeRenderTarget.Starting))
        assertTrue(transition.hasRenderTarget(AssistRuntimeRenderTarget.ModelUnavailable))
        assertTrue(transition.effects.contains(AssistRuntimeEffect.StartCameraIfReady))
    }

    @Test
    fun closeCameraStopsSourceClosesUiAndResetsSession() {
        val transition = AssistRuntimeStateMachine(initialState = AssistRuntimeState.Running, cameraViewsReady = true)
            .onEvent(AssistRuntimeEvent.CloseCamera)

        assertEquals(AssistRuntimeState.Idle, transition.state)
        assertTrue(transition.effects.contains(AssistRuntimeEffect.StopCamera))
        assertTrue(transition.effects.contains(AssistRuntimeEffect.ClearOverlay))
        assertTrue(transition.effects.contains(AssistRuntimeEffect.CloseCamera))
        assertTrue(transition.effects.contains(AssistRuntimeEffect.StopSession))
    }

    @Test
    fun reopeningAfterCloseWaitsForFreshCameraViews() {
        val machine = AssistRuntimeStateMachine(initialState = AssistRuntimeState.Running, cameraViewsReady = true)
        machine.onEvent(AssistRuntimeEvent.CloseCamera)

        val reopened = machine.onEvent(
            AssistRuntimeEvent.OpenCamera(hasCameraPermission = true, modelReady = true)
        )

        assertEquals(AssistRuntimeState.Starting, reopened.state)
        assertTrue(!reopened.effects.contains(AssistRuntimeEffect.StartCameraIfReady))

        val freshViews = machine.onEvent(AssistRuntimeEvent.CameraViewsReady)

        assertTrue(freshViews.effects.contains(AssistRuntimeEffect.StartCameraIfReady))
    }

    private fun AssistRuntimeTransition.hasRenderTarget(target: AssistRuntimeRenderTarget): Boolean {
        return effects.any { effect ->
            effect is AssistRuntimeEffect.Render && effect.target == target
        }
    }
}
